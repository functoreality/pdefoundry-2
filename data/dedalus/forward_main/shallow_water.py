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
from ...common.utils_random import RandomValueSampler
try:
    from test_debug2d import dist, d3_dx, d3_dy, get_bases
except ImportError:
    from ..utils.settings2d import dist, d3_dx, d3_dy, get_bases


class ShallowWater2DPDE(DedalusPDEType):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== Shallow Water Equation ========
    The PDE takes the form
        $$h_t + L_hh + f_h + s_h(r) + ((h+H(r))u)_x + ((h+H(r))v)_y = 0,$$
        $$u_t + L_uu + f_u + s_u(r) + uu_x + vu_y + g_1h_x = 0,$$
        $$v_t + L_vv + f_v + s_v(r) + uv_x + vv_y + g_2h_y = 0,$$
    $\eta(0,r)=g_\eta(r)$ for $\eta\in\{h,u,v\}$, $t\in[0,1]$,
    $r=(x,y)\in[0,1]^2$.
    Periodic boundary conditions are employed for simplicity.
    We take $[f_h;f_u;f_v] = f_0([h;u;v])$ with $f_0$ being the same as that of
    multi-component DCR/Wave equations.
    The initial water height $g_h(r)$ is taken to be a non-negative random
    field. The base height of the water $H(r)$ is also non-negative in the
    non-zero case.
    """
    PREPROCESS_DAG = True
    PDE_TYPE_ID = 12
    TRIAL_N_SUB_STEPS = [2, 8, 32]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.inhom_diff_u = args.inhom_diff_u
        self.periodic = args.periodic.tolist()

        # Dedalus Bases
        # 'self.bases' has format (xbasis, ybasis)
        self.bases = get_bases(self.periodic, dealias=3/2)
        coords = dist.local_grids(*self.bases)
        self.coord_dict["x"], self.coord_dict["y"] = coords

        # PDE terms
        for var_name in "huv":
            self.term_obj_dict[f"ic/{var_name}"] = coefs.RandomField(
                coords, self.periodic)
            self.term_obj_dict[f"s/{var_name}"] = coefs.RandomConstOrField(
                coords, self.periodic,
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)
            if self.inhom_diff_u:
                raise NotImplementedError
                self.term_obj_dict[f"L/{var_name}"] = terms.InhomSpatialOrder2Term(
                    coords, self.periodic,
                    min_val=args.kappa_min, max_val=args.kappa_max)
            else:
                self.term_obj_dict[f"L/{var_name}"] = terms.HomSpatialOrder2Term(
                    min_val=args.kappa_min, max_val=args.kappa_max)

        self.term_obj_dict["ic/h"] = coefs.NonNegField(  # override original
            coords, self.periodic, min_val=0.1, max_val=4)
        self.term_obj_dict["h_base"] = coefs.NonNegConstOrField(
            coords, self.periodic, min_val=1e-3, max_val=4)
        self.term_obj_dict["f0"] = terms.MultiComponentMixedTerm(
            3,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        # self.term_obj_dict["f1"] = terms.MultiComponentLinearTerm(
        #     3,
        #     coef_distribution=args.coef_distribution,
        #     coef_magnitude=args.coef_magnitude)
        # self.term_obj_dict["f2"] = terms.MultiComponentLinearTerm(
        #     3,
        #     coef_distribution=args.coef_distribution,
        #     coef_magnitude=args.coef_magnitude)
        self.term_obj_dict["g1"] = coefs.NonNegRandomValue(
            min_val=0.1, max_val=1)
        self.term_obj_dict["g2"] = coefs.NonNegRandomValue(
            min_val=0.1, max_val=1)

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        diff_u_type = "_inhom" if args.inhom_diff_u else "_hom"
        return ("SWE2D"
                + diff_u_type
                + boundary.BoxDomainBoundaryCondition.arg_str(args)
                + RandomValueSampler.arg_str(args)
                + terms.HomSpatialOrder2Term.arg_str(args))

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--inhom_diff_u", action="store_true",
                            help="Whether the spatial differential term Lu has"
                            " spatial dependency.")
        boundary.BoxDomainBoundaryCondition.add_cli_args_(parser, ndim=2)
        RandomValueSampler.add_cli_args_(parser)
        # same for InhomSpatialOrder2Term
        terms.HomSpatialOrder2Term.add_cli_args_(parser, kappa_max=1e-2)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        for key, term_obj in self.term_obj_dict.items():
            if not any(key.startswith(pat) for pat in ["bc", "L/", "h_base"]):
                term_obj.reset(rng)
        self.term_obj_dict["L/h"].reset(rng, zero_prob=0.2)
        self.term_obj_dict["L/u"].reset(rng, zero_prob=0.2)
        self.term_obj_dict["L/v"].reset(rng, zero_prob=0.2)
        self.term_obj_dict["h_base"].reset(rng, zero_prob=1, scalar_prob=0.2)

    def get_dedalus_problem(self) -> Tuple:
        # Fields
        h_op = dist.Field(name="h", bases=self.bases)
        u_op = dist.Field(name="u", bases=self.bases)
        v_op = dist.Field(name="v", bases=self.bases)
        var_list = [h_op, u_op, v_op]

        problem = d3.IVP(var_list)
        # lin_list = self._get_fi_list(var_list)
        f0_terms = self.term_obj_dict["f0"].gen_dedalus_ops(var_list)
        h_total = h_op + self.term_obj_dict["h_base"].gen_dedalus_ops(
            dist.Field(name="h_base", bases=self.bases))
        for i, var_name in enumerate("huv"):
            # variable and terms
            var_op = var_list[i]
            lin, nonlin = f0_terms[i]
            if var_name == "h":
                nonlin = nonlin + d3_dx(h_total * u_op) + d3_dy(h_total * v_op)
            elif var_name == "u":
                g_value = self.term_obj_dict["g1"].gen_dedalus_ops()
                lin = lin + g_value * d3_dx(h_op)
                nonlin = nonlin + u_op * d3_dx(u_op) + v_op * d3_dy(u_op)
            elif var_name == "v":
                g_value = self.term_obj_dict["g2"].gen_dedalus_ops()
                lin = lin + g_value * d3_dy(h_op)
                nonlin = nonlin + u_op * d3_dx(v_op) + v_op * d3_dy(v_op)

            # Initial condition
            self.term_obj_dict[f"ic/{var_name}"].gen_dedalus_ops(var_op)

            # Source term
            s_op = self.term_obj_dict[f"s/{var_name}"].gen_dedalus_ops(
                dist.Field(bases=self.bases))
            nonlin = nonlin + s_op

            # Second-order term
            if self.inhom_diff_u:
                raise NotImplementedError
            else:
                diff_var = self.term_obj_dict[f"L/{var_name}"].gen_dedalus_ops(
                    d3.grad(var_op))
                lin = lin + diff_var

            problem.add_equation([d3.dt(var_op) + lin, -nonlin])

        var_dict = {"h": h_op, "u": u_op, "v": v_op}
        return var_dict, problem

    '''
    def _get_fi_list(self, var_list: List[d3.Field]) -> List[Operand]:
        r"""Generate 'lin_list' describing the terms $f_i([h,u,v])$."""
        lin_list = self.term_obj_dict["f0"].gen_dedalus_ops(var_list)

        op_list = self.term_obj_dict["f1"].gen_dedalus_ops(var_list)
        for i, lin_op in enumerate(op_list):
            lin_list[i] = lin_list[i] + d3_dx(lin_op)

        op_list = self.term_obj_dict["f2"].gen_dedalus_ops(var_list)
        for i, lin_op in enumerate(op_list):
            lin_list[i] = lin_list[i] + d3_dy(lin_op)

        return lin_list
    '''


class SWEDebug(ShallowWater2DPDE):
    def reset_pde(self, rng: np.random.Generator) -> None:
        super().reset_pde(rng)
        # self.term_obj_dict["f0"].reset_debug()
        self.term_obj_dict["f1"].reset_debug()
        self.term_obj_dict["f2"].reset_debug()

    """
    def get_dedalus_problem(self) -> Tuple:
        var_dict, problem = super().get_dedalus_problem()
        h_ic = 1 + 1e-1 * (np.sin(2 * np.pi * self.coord_dict["x"])
                           + np.sin(2 * np.pi * self.coord_dict["y"]))
        var_dict["h"]["g"] = h_ic
        # var_dict["h"]["g"] = 1 + var_dict["h"]["g"]**2
        var_dict["u"]["g"] = np.zeros_like(h_ic)
        var_dict["v"]["g"] = np.zeros_like(h_ic)
        return var_dict, problem
    """

    def _get_fi_list(self, var_list: List[d3.Field]) -> List[Operand]:
        lin_list = super()._get_fi_list(var_list)
        h_op = var_list[0]
        lin_list[1] = lin_list[1] + 0.1 * d3_dx(h_op)
        lin_list[2] = lin_list[2] + 0.1 * d3_dy(h_op)
        return lin_list


if __name__ == "__main__":
    my_args = get_cli_args(ShallowWater2DPDE)
    pde_data_obj = ShallowWater2DPDE(my_args)
    gen_data(my_args, pde_data_obj)
