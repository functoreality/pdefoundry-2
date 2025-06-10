#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of the 2D PDE with multiple components."""
import argparse
from typing import Tuple, Dict, List, Callable
import numpy as np
from numpy.typing import NDArray
import dedalus.public as d3
from dedalus.core.field import Operand

from ..utils import coefs, terms, boundary
from ..utils.basics import DedalusPDEType
from ...common.basics import get_cli_args, gen_data
try:
    from test_debug2d import dist, d3_dx, d3_dy, get_bases
except ImportError:
    from ..utils.settings2d import dist, d3_dx, d3_dy, get_bases


class MultiComponent2DPDE(DedalusPDEType):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== PDE with Multiple Components ========
    The PDE takes the form
        $$\partial_tu_i + L_iu_i + f_0(u)_i + s_i(r) + \partial_xf_1(u)_i
            + \partial_yf_2(u)_i = 0,$$
    $u_i(0,r)=g_i(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$,
    $0 \le i,j,k \le d_u-1$, $j \le k$.
    Periodic boundary conditions are employed for simplicity.

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
    PREPROCESS_DAG = True
    PDE_TYPE_ID = 8
    TRIAL_N_SUB_STEPS = [2, 8, 32]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.n_vars = args.n_vars
        self.inhom_diff_u = args.inhom_diff_u
        self.periodic = args.periodic.tolist()

        # Dedalus Bases
        self.bases = get_bases(
            self.periodic, dealias=3/2)  # format (xbasis, ybasis)
        coords = dist.local_grids(*self.bases)
        self.coord_dict["x"], self.coord_dict["y"] = coords

        # PDE terms
        for i in range(self.n_vars):
            self.term_obj_dict[f"u_ic/{i}"] = coefs.RandomField(
                coords, self.periodic)
            self.term_obj_dict[f"s/{i}"] = coefs.RandomConstOrField(
                coords, self.periodic,
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)
            if self.inhom_diff_u:
                raise NotImplementedError
                self.term_obj_dict[f"Lu/{i}"] = terms.InhomSpatialOrder2Term(
                    coords, self.periodic,
                    min_val=args.kappa_min, max_val=args.kappa_max)
            else:
                self.term_obj_dict[f"Lu/{i}"] = terms.HomSpatialOrder2Term(
                    min_val=args.kappa_min, max_val=args.kappa_max)
            self.term_obj_dict[f"bc/{i}"] = boundary.BoxDomainBoundaryCondition(
                coords, self.periodic,
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)
            self.term_obj_dict[f"bc/{i}"].assign_ic(
                self.term_obj_dict[f"u_ic/{i}"])

        self.term_obj_dict["f0"] = terms.MultiComponentMixedTerm(
            self.n_vars,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.term_obj_dict["f1"] = terms.MultiComponentMixedTerm(
            self.n_vars,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.term_obj_dict["f2"] = terms.MultiComponentMixedTerm(
            self.n_vars,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        if getattr(args, "flux_num_nonlin", -1) >= 0:
            self.term_obj_dict["f1"].deg2_term.max_len = args.flux_num_nonlin
            self.term_obj_dict["f2"].deg2_term.max_len = args.flux_num_nonlin

    # @property
    # def sol_dict(self) -> Dict[str, NDArray[float]]:
    #     u_sol_list = [self.raw_sol_dict[f"u{i}"] for i in range(self.n_vars)]
    #     u_sol = np.stack(u_sol_list, axis=1)  # [n_t, n_vars, n_x, n_y]
    #     return {"u_all": u_sol}

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        diff_u_type = "_inhom" if args.inhom_diff_u else "_hom"
        if args.flux_num_nonlin >= 0:
            flux_type = f"_fNL{args.flux_num_nonlin}"
        else:
            flux_type = ""
        return ("MCompn2D"
                + diff_u_type
                + boundary.BoxDomainBoundaryCondition.arg_str(args)
                + flux_type
                + terms.MultiComponentMixedTerm.arg_str(args)
                + terms.HomSpatialOrder2Term.arg_str(args))

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--inhom_diff_u", action="store_true",
                            help="Whether the spatial differential term Lu has"
                            " spatial dependency.")
        parser.add_argument("--flux_num_nonlin", type=int, default=-1,
                            help="Number of non-linear terms in each flux function.")
        boundary.BoxDomainBoundaryCondition.add_cli_args_(parser, ndim=2)
        terms.MultiComponentMixedTerm.add_cli_args_(parser)
        # same for InhomSpatialOrder2Term
        terms.HomSpatialOrder2Term.add_cli_args_(parser, kappa_max=1e-2)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        for key, term_obj in self.term_obj_dict.items():
            if not (key.startswith("bc") or key.startswith("Lu")):
                term_obj.reset(rng)
        for i in range(self.n_vars):
            if (self.term_obj_dict["f1"].is_linear_wrt(i)
                and self.term_obj_dict["f2"].is_linear_wrt(i)):
                self.term_obj_dict[f"Lu/{i}"].reset(rng, zero_prob=0.2)
            else:
                self.term_obj_dict[f"Lu/{i}"].reset(rng, zero_prob=0.)
            # BC depends on u_ic, and potentially Lu in the future versions.
            self.term_obj_dict[f"bc/{i}"].reset(rng)

    def get_dedalus_problem(self) -> Tuple:
        u_list, lin_list, nonlin_list = self._get_fi_list()

        # Tau polynomials for non-periodic BCs
        tau_var_list = []
        for i in range(self.n_vars):
            tau_i_op, tau_i_var_list = boundary.tau_polynomial(
                dist, self.bases, self.periodic)
            lin_list[i] = lin_list[i] + tau_i_op
            tau_var_list.extend(tau_i_var_list)

        # Problem
        problem = d3.IVP(u_list + tau_var_list)
        for i, ui_op in enumerate(u_list):
            # Initial condition
            self.term_obj_dict[f"u_ic/{i}"].gen_dedalus_ops(ui_op)

            # Source term
            si_op = dist.Field(name=f"s{i}", bases=self.bases)
            si_op = self.term_obj_dict[f"s/{i}"].gen_dedalus_ops(si_op)
            lin_i = lin_list[i]
            nonlin_i = nonlin_list[i] + si_op

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

            problem.add_equation([d3.dt(ui_op) + lin_i, -nonlin_i])

            # Boundary conditions
            bc_i_ops_list = self.term_obj_dict[f"bc/{i}"].gen_dedalus_ops(
                    ui_op, (d3_dx(ui_op), d3_dy(ui_op)), dist, self.bases)
            for bc_lhs, bc_rhs in bc_i_ops_list:
                problem.add_equation([bc_lhs, bc_rhs])

        var_dict = {f"u{i}": ui_op for i, ui_op in enumerate(u_list)}
        return var_dict, problem

    def _get_fi_list(self) -> Tuple[List[Operand]]:
        r"""
        Generate 'u_list', 'lin_list', 'nonlin_list' describing the terms
        $f_i(u)$.
        """
        # Basic utilities
        lin_list = [0 for _ in range(self.n_vars)]
        nonlin_list = [0 for _ in range(self.n_vars)]

        # Fields
        u_list = [dist.Field(name=f"u{i}", bases=self.bases)
                  for i in range(self.n_vars)]

        # PDE Terms
        def add_vec_term(op_fn: Callable[[Operand], Operand],
                         ops_list: List[Tuple[Operand]]) -> None:
            for i in range(self.n_vars):
                term_lin, term_nonlin = ops_list[i]
                lin_list[i] = lin_list[i] + op_fn(term_lin)
                nonlin_list[i] = nonlin_list[i] + op_fn(term_nonlin)

        def id_fn(op_in):
            return op_in

        add_vec_term(id_fn, self.term_obj_dict["f0"].gen_dedalus_ops(u_list))
        add_vec_term(d3_dx, self.term_obj_dict["f1"].gen_dedalus_ops(u_list))
        add_vec_term(d3_dy, self.term_obj_dict["f2"].gen_dedalus_ops(u_list))
        return u_list, lin_list, nonlin_list


if __name__ == "__main__":
    my_args = get_cli_args(MultiComponent2DPDE)
    pde_data_obj = MultiComponent2DPDE(my_args)
    gen_data(my_args, pde_data_obj)
