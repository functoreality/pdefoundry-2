#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of the 2D Wave equation with multiple components."""
import argparse
from typing import Tuple, List, Callable

import numpy as np
import dedalus.public as d3
from dedalus.core.field import Operand

from ..utils import coefs, terms, boundary
from ...common.basics import get_cli_args, gen_data
from .multi_component import MultiComponent2DPDE
try:
    from test_debug2d import dist, d3_dx, d3_dy
except ImportError:
    from ..utils.settings2d import dist, d3_dx, d3_dy


class MCompnWave2DPDE(MultiComponent2DPDE):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== Wave Equation with Multiple Components ========
    The PDE takes the form
        $$\partial_{tt}u_i + \mu_i(r)\partial_tu_i + L_iu_i + f_0(u)_i + s_i(r)
            + \partial_xf_1(u)_i + \partial_yf_2(u)_i = 0,$$
    $u_i(0,r)=g_i(r)$, $\partial_tu_i(0,r)=g_i(r)$, $t\in[0,1]$,
    $r=(x,y)\in[0,1]^2$, $0 \le i,j,k \le d_u-1$, $j \le k$.

    Here, each spatial second-order term $L_iu_i$ is randomly selected from
    the non-divergence form $L_iu_i=-a_i(r)\Delta u_i$, the factored form
    $L_iu_i=-\sqrt a_i(r)\nabla\cdot(\sqrt a_i(r)\nabla u_i)$, and the
    divergence form $L_iu_i=-\nabla\cdot(a_i(r)\nabla u_i)$ with equal
    probability, where $a_i(r)=a_i$ is taken to be a random scalar for
    simplicity, and $r=(x,y)$ denotes the spatial coordinates.

    We take $f_l(u)_i = \sum_ja_{lij}u_j + \sum_{j,k}b_{lijk}u_ju_k$.
    The coefficients $a,b$ are sparse arrays, with a total of at most $3d_u$
    non-zero entries.
    """
    PDE_TYPE_ID = 9
    TRIAL_N_SUB_STEPS = [1]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        coords = (self.coord_dict["x"], self.coord_dict["y"])
        for i in range(self.n_vars):
            self.term_obj_dict[f"ut_ic/{i}"] = coefs.RandomField(
                coords, self.periodic)
            self.term_obj_dict[f"mu/{i}"] = coefs.RandomConstOrField(
                coords, self.periodic,
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)
            self.term_obj_dict[f"bc/{i}"] = boundary.BoxDomainBCWithMur(
                coords, self.periodic,
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)
            self.term_obj_dict[f"bc/{i}"].assign_ic(
                self.term_obj_dict[f"u_ic/{i}"],
                self.term_obj_dict[f"ut_ic/{i}"])

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        diff_u_type = "_inhom" if args.inhom_diff_u else "_hom"
        return ("MCWave2D"
                + diff_u_type
                + boundary.BoxDomainBoundaryCondition.arg_str(args)
                + terms.MultiComponentMixedTerm.arg_str(args)
                + terms.HomSpatialOrder2Term.arg_str(args))

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--inhom_diff_u", action="store_true",
                            help="Whether the spatial differential term Lu has"
                            " spatial dependency.")
        boundary.BoxDomainBoundaryCondition.add_cli_args_(parser, ndim=2)
        terms.MultiComponentMixedTerm.add_cli_args_(parser)
        # same for InhomSpatialOrder2Term
        terms.HomSpatialOrder2Term.add_cli_args_(
            parser, kappa_min=1e-2, kappa_max=4)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        for key, term_obj in self.term_obj_dict.items():
            if not (key.startswith("bc") or key.startswith("Lu")):
                term_obj.reset(rng)
        for i in range(self.n_vars):
            if (self.term_obj_dict["f1"].self_coupling_wrt(i)
                or self.term_obj_dict["f2"].self_coupling_wrt(i)):
                # reasons for requiring non-zero wavespeed: The equation
                # u_{tt} = u_x is unstable, while u_{tt} = u_x + c^2u_{xx} is.
                self.term_obj_dict[f"Lu/{i}"].reset(rng, zero_prob=0.)
            else:
                self.term_obj_dict[f"Lu/{i}"].reset(rng, zero_prob=0.2)
            # BC depends on u_ic, ut_ic, and potentially Lu in the future
            # versions.
            self.term_obj_dict[f"bc/{i}"].reset(rng)

    def get_dedalus_problem(self) -> Tuple:
        u_list, lin_list, nonlin_list = self._get_fi_list()
        ut_list = [dist.Field(name=f"dt_u{i}", bases=self.bases)
                   for i in range(self.n_vars)]

        # Tau polynomials for non-periodic BCs
        tau_var_list = []
        for i in range(self.n_vars):
            tau_i_op, tau_i_var_list = boundary.tau_polynomial(
                dist, self.bases, self.periodic)
            lin_list[i] = lin_list[i] + tau_i_op
            tau_var_list.extend(tau_i_var_list)

        # Problem
        problem = d3.IVP(u_list + ut_list + tau_var_list)
        for i in range(self.n_vars):
            ui_op = u_list[i]
            dt_ui = ut_list[i]

            # Initial condition
            self.term_obj_dict[f"u_ic/{i}"].gen_dedalus_ops(ui_op)
            self.term_obj_dict[f"ut_ic/{i}"].gen_dedalus_ops(dt_ui) 

            # Source term
            si_op = dist.Field(name=f"s{i}", bases=self.bases)
            si_op = self.term_obj_dict[f"s/{i}"].gen_dedalus_ops(si_op)

            # Damping term
            mu_i = dist.Field(name=f"mu{i}", bases=self.bases)
            mu_i = self.term_obj_dict[f"mu/{i}"].gen_dedalus_ops(mu_i)

            lin_i = lin_list[i]
            nonlin_i = nonlin_list[i] + si_op
            if self.term_obj_dict[f"mu/{i}"].is_const:
                lin_i = lin_i + mu_i * dt_ui
            else:
                nonlin_i = nonlin_i + mu_i * dt_ui

            # Second-order term
            if self.inhom_diff_u:
                raise NotImplementedError  # dynamically decide lin or nonlin
                ai_op = dist.Field(name=f"a{i}", bases=self.bases)
                diff_ui = self.term_obj_dict[f"Lu/{i}"].gen_dedalus_ops(
                    ai_op, d3.grad(ui_op))
                nonlin_i = nonlin_i + diff_ui
            else:
                diff_ui = self.term_obj_dict[f"Lu/{i}"].gen_dedalus_ops(
                    d3.grad(ui_op))
                lin_i = lin_i + diff_ui

            problem.add_equation([d3.dt(ui_op) - dt_ui, 0])
            problem.add_equation([d3.dt(dt_ui) + lin_i, -nonlin_i])

            # Boundary conditions
            bc_i_ops_list = self.term_obj_dict[f"bc/{i}"].gen_dedalus_ops(
                ui_op, dt_ui, (d3_dx(ui_op), d3_dy(ui_op)),
                dist, self.bases)
            for bc_lhs, bc_rhs in bc_i_ops_list:
                problem.add_equation([bc_lhs, bc_rhs])

        var_dict = {f"u{i}": ui_op for i, ui_op in enumerate(u_list)}
        return var_dict, problem


if __name__ == "__main__":
    my_args = get_cli_args(MCompnWave2DPDE)
    pde_data_obj = MCompnWave2DPDE(my_args)
    gen_data(my_args, pde_data_obj)
