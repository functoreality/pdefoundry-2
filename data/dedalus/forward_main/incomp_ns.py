#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of the 2D incompressible Navier-Stokes equation."""
import argparse
from typing import Tuple
import numpy as np
import dedalus.public as d3

from ..utils import coefs, terms, boundary
from ..utils.basics import DedalusPDEType
from ...common.basics import get_cli_args, gen_data
try:
    from test_debug2d import d3_coords, dist, d3_dx, d3_dy, get_bases
except ImportError:
    from ..utils.settings2d import d3_coords, dist, d3_dx, d3_dy, get_bases


class IncompressibleNS2DPDE(DedalusPDEType):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== Incompressible Navier-Stokes Equation ========
    The PDE takes the form
        $$u_t-\nu(u_{xx}+u_{yy})+s^u(r)+(u^2)_x+(uv)_y+p_x=0$$
        $$v_t-\nu(v_{xx}+v_{yy})+s^v(r)+(uv)_x+(v^2)_y+p_y=0$$
        $$u_x+v_y=0$$
    $u(0,r)=g^u(r)$, $v(0,r)=g^v(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$.
    Periodic boundary conditions are imposed along the x-axis. For the y-axis,
    either the periodic boundary condition or the no-slip wall (Dirichlet)
    boundary condition is employed.
    """
    PREPROCESS_DAG = False
    TRIAL_N_SUB_STEPS = [2, 8, 32]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.valid_ic = args.valid_ic
        self.periodic = [True, not args.y_wall]
        self.ignore_p = args.ignore_p

        # Dedalus Bases
        # self.bases has format (xbasis, ybasis).
        self.bases = get_bases(self.periodic, dealias=3/2)
        coords = dist.local_grids(*self.bases)
        self.coord_dict["x"], self.coord_dict["y"] = coords

        # PDE terms; periodic by default
        self.term_obj_dict["u_ic"] = coefs.RandomField(coords, self.periodic)
        self.term_obj_dict["v_ic"] = coefs.RandomField(coords, self.periodic)
        self.term_obj_dict["s_u"] = coefs.RandomField(coords, self.periodic)
        self.term_obj_dict["s_v"] = coefs.RandomField(coords, self.periodic)
        self.term_obj_dict["visc"] = terms.HomSpatialOrder2Term(
            min_val=args.kappa_min, max_val=args.kappa_max)

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        prefix = "Baseline2D_INS"
        prefix += "_icV" if args.valid_ic else "_icA"
        prefix += "_npY" if args.y_wall else ""
        prefix += "_noP" if args.ignore_p else ""
        prefix += terms.HomSpatialOrder2Term.arg_str(args)
        return prefix

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--valid_ic", action="store_true",
                            help="Make the initial condition to comply with "
                            "the divergence constraint.")
        parser.add_argument("--y_wall", action="store_true",
                            help="Add a wall boundary at the end of y axis.")
        parser.add_argument("--ignore_p", action="store_true",
                            help="Do not save the pressure field.")
        # same for InhomSpatialOrder2Term
        terms.HomSpatialOrder2Term.add_cli_args_(parser, kappa_max=1e-2)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        super().reset_pde(rng)
        if self.valid_ic:
            c_obj = argparse.Namespace(size=3, value=[0,0,0])
            self.term_obj_dict["u_ic"].div_constraint_2d_(
                rng, self.term_obj_dict["v_ic"], c_obj)

    def get_dedalus_problem(self) -> Tuple:
        u_op = dist.Field(name="u", bases=self.bases)
        v_op = dist.Field(name="v", bases=self.bases)
        p_op = dist.Field(name="p", bases=self.bases)
        tau_p = dist.Field(name="tau_p")

        # boundary tau operands
        var_op_list = [u_op, v_op, p_op, tau_p]
        tau_args = (dist, self.bases, self.periodic)
        tau_u, tau_vars = boundary.tau_polynomial(*tau_args)
        var_op_list += tau_vars
        tau_v1, tau_vars = boundary.tau_polynomial_raw(*tau_args)
        var_op_list += tau_vars
        tau_v2, tau_vars = boundary.tau_polynomial_raw(*tau_args)
        var_op_list += tau_vars

        # Initial condition
        self.term_obj_dict["u_ic"].gen_dedalus_ops(u_op)
        self.term_obj_dict["v_ic"].gen_dedalus_ops(v_op)

        # PDE terms
        s_u = self.term_obj_dict["s_u"].gen_dedalus_ops(
            dist.Field(name="s_u", bases=self.bases))
        s_v = self.term_obj_dict["s_v"].gen_dedalus_ops(
            dist.Field(name="s_v", bases=self.bases))
        diff_u = self.term_obj_dict["visc"].gen_dedalus_ops(d3.grad(u_op))
        _, e_y = d3_coords.unit_vector_fields(dist)  # e_y: Operand
        diff_v = self.term_obj_dict["visc"].gen_dedalus_ops(
            d3.grad(v_op) + e_y * tau_v1)

        # Problem
        problem = d3.IVP(var_op_list)
        problem.add_equation([d3.dt(u_op) + diff_u + d3_dx(p_op) + tau_u,
                              -(s_u + d3_dx(u_op**2) + d3_dy(u_op * v_op))])
        problem.add_equation([d3.dt(v_op) + diff_v + d3_dy(p_op) + tau_v2,
                              -(s_v + d3_dx(u_op * v_op) + d3_dy(v_op**2))])
        problem.add_equation([d3_dx(u_op) + d3_dy(v_op) + tau_v1 + tau_p, 0])
        problem.add_equation([d3.integ(p_op), 0])  # pressure gauge
        if not self.periodic[1]:  # boundary conditions for y_wall
            problem.add_equation([u_op(y="left"), 0])
            problem.add_equation([u_op(y="right"), 0])
            problem.add_equation([v_op(y="left"), 0])
            problem.add_equation([v_op(y="right"), 0])

        if self.ignore_p:
            var_dict = {"u": u_op, "v": v_op}
        else:
            var_dict = {"u": u_op, "v": v_op, "p": p_op}
        return var_dict, problem


if __name__ == "__main__":
    my_args = get_cli_args(IncompressibleNS2DPDE)
    pde_data_obj = IncompressibleNS2DPDE(my_args)
    gen_data(my_args, pde_data_obj)
