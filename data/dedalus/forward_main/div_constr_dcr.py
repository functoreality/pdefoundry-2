#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of the 2D Wave equation with multiple components."""
import argparse
from typing import Tuple, List, Callable

import numpy as np
import dedalus.public as d3
from dedalus.core.field import Operand

from ..utils import coefs, terms, boundary
from ..utils.basics import DedalusPDEType
from ...common.basics import get_cli_args, gen_data
from ...common.utils_random import RandomValueSampler
from .multi_component import MultiComponent2DPDE
try:
    from test_debug2d import d3_coords, dist, d3_dx, d3_dy
except ImportError:
    from ..utils.settings2d import d3_coords, dist, d3_dx, d3_dy


class DivConstraintDCR2DPDE(MultiComponent2DPDE):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== Diffusion-Convection-Reaction PDE with Divergence-Constraint ========
    The PDE takes the form
        $$\partial_tu_i + L_iu_i + f_0(u)_i + s_i(r) + \partial_xf_1(u)_i
            + \partial_yf_2(u)_i + (-c_i)p + (\nabla p)_i = 0,$$
        $$\partial_xu_0 + \partial_yu_1 + c_0u_0 + c_1u_1 + c_2 = 0,$$
    $t\in[0,1]$, $r=(x,y)\in[0,1]^2$, $0 \le i,j,k \le 1$, $j \le k$.
    When the initial value is required to comply with the divergence
    constraint, the initial condition is taken as
    $u(0,r)=(c+\nabla)\times\psi(r)$.
    Periodic boundary conditions are employed for simplicity.

    Here, each spatial second-order term $L_iu_i$ is randomly selected from
    the non-divergence form $L_iu_i=-a_i(r)\Delta u_i$, the factored form
    $L_iu_i=-\sqrt a_i(r)\nabla\cdot(\sqrt a_i(r)\nabla u_i)$, and the
    divergence form $L_iu_i=-\nabla\cdot(a_i(r)\nabla u_i)$ with equal
    probability, where $a_i(r)=a_i$ is taken to be a random scalar for
    simplicity, and $r=(x,y)$ denotes the spatial coordinates.

    We take $f_l(u)_i = \sum_ja_{lij}u_j + \sum_{j,k}b_{lijk}u_ju_k$.
    The coefficients $a,b$ are sparse arrays, with a total of at most $6$
    non-zero entries.
    """
    PDE_TYPE_ID = 10

    def __init__(self, args: argparse.Namespace) -> None:
        args.n_vars = 2  # manually set it to be equal to the spatial dimension
        super().__init__(args)
        self.valid_ic = args.valid_ic
        self.term_obj_dict["c"] = coefs.RandomValue(
            3, args.coef_distribution, args.coef_magnitude)

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        diff_u_type = "_inhom" if args.inhom_diff_u else "_hom"
        ic_type = "_icV" if args.valid_ic else "_icA"
        return ("DivConstrDCR2D"
                + ic_type
                + diff_u_type
                + boundary.BoxDomainBoundaryCondition.arg_str(args)
                + RandomValueSampler.arg_str(args)
                + terms.HomSpatialOrder2Term.arg_str(args))

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--inhom_diff_u", action="store_true",
                            help="Make the spatial differential term Lu have"
                            " spatial dependency.")
        parser.add_argument("--valid_ic", action="store_true",
                            help="Make the initial condition to comply with "
                            "the divergence constraint.")
        boundary.BoxDomainBoundaryCondition.add_cli_args_(parser, ndim=2)
        RandomValueSampler.add_cli_args_(parser)
        # same for InhomSpatialOrder2Term
        terms.HomSpatialOrder2Term.add_cli_args_(parser, kappa_max=1e-2)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        for key, term_obj in self.term_obj_dict.items():
            if not (key.startswith("bc") or key.startswith("Lu")):
                term_obj.reset(rng)
        c_obj = self.term_obj_dict["c"]
        if c_obj.value[0] == 0. and c_obj.value[1] == 0.:
            c_obj.value[2] = 0.
        if self.valid_ic:
            self.term_obj_dict["u_ic/0"].div_constraint_2d_(
                rng, self.term_obj_dict["u_ic/1"], c_obj)
        for i in range(self.n_vars):
            if (self.term_obj_dict["f1"].is_linear_wrt(i)
                and self.term_obj_dict["f2"].is_linear_wrt(i)):
                self.term_obj_dict[f"Lu/{i}"].reset(rng, zero_prob=0.2)
            else:
                self.term_obj_dict[f"Lu/{i}"].reset(rng, zero_prob=0.)
            # BC depends on u_ic, and potentially Lu in the future versions.
            self.term_obj_dict[f"bc/{i}"].reset(rng)
            if self.periodic[i]:
                continue
            # specific for non-periodic BCs, which are found empirically to
            # improve probability of a successful solution
            alpha_obj = self.term_obj_dict[f"bc/{i}"].edge_term_dict[i][0].alpha
            if not alpha_obj.is_const:
                alpha_obj.reset(rng, field_prob=0.)
            alpha_obj = self.term_obj_dict[f"bc/{i}"].edge_term_dict[i][1].alpha
            if not alpha_obj.is_const:
                alpha_obj.reset(rng, field_prob=0.)

    def get_dedalus_problem(self) -> Tuple:
        u_list, lin_list, nonlin_list = self._get_fi_list()
        p_op = dist.Field(name="p", bases=self.bases)
        grad_list = [d3.grad(ui_op) for ui_op in u_list]
        unit_vecs = d3_coords.unit_vector_fields(dist)  # Tuple[Operand]

        # Gradient of p
        c_arr = self.term_obj_dict["c"].gen_dedalus_ops()  # NDArray[float]
        lin_list[0] = lin_list[0] + d3_dx(p_op) - c_arr[0] * p_op
        lin_list[1] = lin_list[1] + d3_dy(p_op) - c_arr[1] * p_op

        # Tau polynomials for non-periodic BCs
        tau_var_list = []
        for i in range(self.n_vars):
            tau_i_op, tau_i_var_list = boundary.tau_polynomial_raw(
                dist, self.bases, self.periodic)
            lin_list[i] = lin_list[i] + tau_i_op
            tau_var_list.extend(tau_i_var_list)
            tau_i_op, tau_i_var_list = boundary.tau_polynomial_raw(
                dist, self.bases, self.periodic, unit_vecs)
            grad_list[i] = grad_list[i] + tau_i_op
            tau_var_list.extend(tau_i_var_list)

        # Problem
        if np.allclose(c_arr, 0, atol=1e-3):  # divergence-free case
            # https://dedalus-project.readthedocs.io/en/latest/pages/gauge_conditions.html
            tau_p = dist.Field(name="tau_p")
            problem = d3.IVP(u_list + tau_var_list + [p_op, tau_p])
            problem.add_equation([unit_vecs[0] @ grad_list[0]
                                  + unit_vecs[1] @ grad_list[1]
                                  + tau_p, 0])
            problem.add_equation([d3.integ(p_op), 0])  # pressure gauge
        else:  # general case
            problem = d3.IVP(u_list + tau_var_list + [p_op])
            problem.add_equation([
                c_arr[0] * u_list[0] + unit_vecs[0] @ grad_list[0]
                + c_arr[1] * u_list[1] + unit_vecs[1] @ grad_list[1],
                -c_arr[2]])
        for i, ui_op in enumerate(u_list):
            # Initial condition
            self.term_obj_dict[f"u_ic/{i}"].gen_dedalus_ops(ui_op)

            # Source term
            si_op = self.term_obj_dict[f"s/{i}"].gen_dedalus_ops(
                dist.Field(name=f"s{i}", bases=self.bases))
            lin_i = lin_list[i]
            nonlin_i = nonlin_list[i] + si_op

            # Second-order term
            if self.inhom_diff_u:
                raise NotImplementedError  # dynamically decide lin or nonlin
                ai_op = dist.Field(name=f"a{i}", bases=self.bases)
                diff_ui = self.term_obj_dict[f"Lu/{i}"].gen_dedalus_ops(
                    ai_op, grad_list[i])
                nonlin_i = nonlin_i + diff_ui
            else:
                diff_ui = self.term_obj_dict[f"Lu/{i}"].gen_dedalus_ops(
                    grad_list[i])
                lin_i = lin_i + diff_ui

            problem.add_equation([d3.dt(ui_op) + lin_i, -nonlin_i])

            # Boundary conditions
            bc_i_ops_list = self.term_obj_dict[f"bc/{i}"].gen_dedalus_ops(
                ui_op,
                (unit_vecs[0] @ grad_list[i], unit_vecs[1] @ grad_list[i]),
                dist, self.bases)
            for bc_lhs, bc_rhs in bc_i_ops_list:
                problem.add_equation([bc_lhs, bc_rhs])

        var_dict = {"u": u_list[0], "v": u_list[1]}
        if all(self.periodic):
            # found to have large pressure values in non-periodic cases
            var_dict["p"] = p_op
        return var_dict, problem

    def gen_solution(self) -> None:
        try:
            super().gen_solution()
        except RuntimeError as err:
            if all(self.periodic):
                raise err
            print("RuntimeError detected, possibly due to a singular matrix "
                  "in the non-periodic case.")
            # reason for matrix being singular:
            # ut + Δu + px = 0, vt + Δv + py = 0, ux + vy = 0, ux|L,R=0,
            # if (u,v,p) is sol, then so is (u+ct,v,p-cx), or (u+c(t),v,p-c'(t)x)
            self.raw_sol_dict = {}


class DCDCRUnitTest(DivConstraintDCR2DPDE):
    TIMESTEPPER = d3.RK222
    # TRIAL_N_SUB_STEPS = [1]

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "DCDCR_debug"

    def reset_pde_bcalpha(self, rng: np.random.Generator) -> None:
        super().reset_pde(rng)
        # self.term_obj_dict["bc/0"].reset_debug()
        # self.term_obj_dict["bc/0"].edge_term_dict[0][0].reset(rng, bc_type=0)
        self.term_obj_dict["bc/0"].edge_term_dict[0][0].alpha.reset(rng, field_prob=0.)
        # self.term_obj_dict["bc/0"].edge_term_dict[0][0].beta.reset_debug()
        # self.term_obj_dict["bc/0"].edge_term_dict[0][1].reset(rng, bc_type=0)
        self.term_obj_dict["bc/0"].edge_term_dict[0][1].alpha.reset(rng, field_prob=0.)
        # self.term_obj_dict["bc/0"].edge_term_dict[0][1].beta.reset_debug()

    def reset_pde_ablation(self, rng: np.random.Generator) -> None:
        self.reset_debug()
        for key, term_obj in self.term_obj_dict.items():
            if not (key.startswith("bc") or key.startswith("Lu")):
                term_obj.reset(rng)
        c_obj = self.term_obj_dict["c"]
        if c_obj.value[0] == 0. and c_obj.value[1] == 0.:
            c_obj.value[2] = 0.
        self.term_obj_dict["bc/1"].reset(rng)
        # self.term_obj_dict["u_ic/0"].reset(rng)
        # self.term_obj_dict["u_ic/1"].reset(rng)
        # self.term_obj_dict["u_ic/0"].div_constraint_2d_(
        #     rng, self.term_obj_dict["u_ic/1"], self.term_obj_dict["c"])

    def reset_pde_err(self, rng: np.random.Generator) -> None:
        self.reset_debug()
        # self.term_obj_dict["bc/0"].reset(rng)
        self.term_obj_dict["bc/0"].edge_term_dict[0][0].reset(rng, bc_type=1)
        self.term_obj_dict["bc/0"].edge_term_dict[0][0].alpha.reset_debug()
        self.term_obj_dict["bc/0"].edge_term_dict[0][1].reset(rng, bc_type=1)
        self.term_obj_dict["bc/0"].edge_term_dict[0][1].alpha.reset_debug()
        # reason for matrix being singular:
        # ut + Δu + px = 0, vt + Δv + py = 0, ux + vy = 0, ux|L,R=0,
        # if (u,v,p) is sol, then so is (u+ct,v,p-cx), or (u+c(t),v,p-c'(t)x)

    def get_dedalus_problem_nop(self) -> Tuple:
        var_dict, problem = super().get_dedalus_problem()
        var_dict.pop("p")
        return var_dict, problem


if __name__ == "__main__":
    my_args = get_cli_args(DivConstraintDCR2DPDE)
    pde_data_obj = DivConstraintDCR2DPDE(my_args)
    gen_data(my_args, pde_data_obj)
