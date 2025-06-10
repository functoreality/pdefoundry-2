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


class INSWithCoupledVar2DPDE(DedalusPDEType):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== Incompressible Navier-Stokes with Coupled Variable ========
    The PDE takes the form
        $$u_t-\nu(u_{xx}+u_{yy})+(u^2)_x+(uv)_y+p_x=0,$$
        $$v_t-\nu(v_{xx}+v_{yy})+(uv)_x+(v^2)_y+p_y+ab=0,$$
        $$b_t-D(b_{xx}+b_{yy})+(ub)_x+(vb)_y=0,$$
        $$u_x+v_y=0,$$
    $u(0,r)=g^u(r)$, $v(0,r)=g^v(r)$, $b(0,r)=g^b(r)$, $t\in[0,1]$,
    $r=(x,y)\in[0,1]^2$, $a\in\{0,-1\}$.
    The coupled variable $b$ represents a tracer field when $a=0$, and buoyancy
    field when $a=-1$.
    Periodic boundary conditions are imposed along the x-axis. The y-axis can
    also taken to be periodic. If not, we employ the no-slip wall (Dirichlet)
    boundary condition on the velocity vector $(u,v)$, and take
    $b|_{y=0}-\delta b_0=0,\ b|_{y=1}=0$ for the coupled variable.
    """
    PREPROCESS_DAG = False
    TRIAL_N_SUB_STEPS = [2, 8, 32]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.buoyancy = args.buoyancy
        self.valid_ic = args.valid_ic
        self.periodic = [True, not args.y_wall]
        self.ignore_p = args.ignore_p
        self.delta_b = args.delta_b
        self.viscosity = args.viscosity
        self.diffusivity = args.diffusivity

        # Dedalus Bases
        # self.bases has format (xbasis, ybasis).
        self.bases = get_bases(self.periodic, dealias=3/2)
        coords = dist.local_grids(*self.bases)
        self.coord_dict["x"], self.coord_dict["y"] = coords

        # PDE terms; periodic by default
        self.term_obj_dict["u_ic"] = coefs.RandomField(coords, self.periodic)
        self.term_obj_dict["v_ic"] = coefs.RandomField(coords, self.periodic)
        self.term_obj_dict["b_ic"] = coefs.RandomField(coords, self.periodic)

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        prefix = "Baseline2D_INS"
        prefix += "Buoyancy" if args.buoyancy else "Tracer"
        prefix += "_icV" if args.valid_ic else "_icA"
        prefix += "_npY" if args.y_wall else ""
        prefix += "_noP" if args.ignore_p else ""
        if args.y_wall:
            prefix += f"_db{args.delta_b:g}"
        prefix += f"_nu{args.viscosity:g}D{args.diffusivity:g}"
        return prefix

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--buoyancy", action="store_true",
                            help="Let the coupled variable work as buoyancy.")
        parser.add_argument("--valid_ic", action="store_true",
                            help="Make the initial condition to comply with "
                            "the divergence constraint.")
        parser.add_argument("--y_wall", action="store_true",
                            help="Add a wall boundary at the end of y axis.")
        parser.add_argument("--ignore_p", action="store_true",
                            help="Do not save the pressure field.")
        parser.add_argument("--delta_b", type=float, default=1.,
                            help="Value of the coupled variable b at the "
                            "bottom edge. Only works when y_wall is on.")
        parser.add_argument("--viscosity", type=float, default=1e-3,
                            help="Fluid viscosity.")
        parser.add_argument("--diffusivity", type=float, default=1e-2,
                            help="Coupled variable diffusivity.")
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
        b_op = dist.Field(name="b", bases=self.bases)
        p_op = dist.Field(name="p", bases=self.bases)
        tau_p = dist.Field(name="tau_p")

        # boundary tau operands
        var_op_list = [u_op, v_op, b_op, p_op, tau_p]
        tau_args = (dist, self.bases, self.periodic)
        tau_u, tau_vars = boundary.tau_polynomial(*tau_args)
        var_op_list += tau_vars
        tau_v1, tau_vars = boundary.tau_polynomial_raw(*tau_args)
        var_op_list += tau_vars
        tau_v2, tau_vars = boundary.tau_polynomial_raw(*tau_args)
        var_op_list += tau_vars
        tau_b, tau_vars = boundary.tau_polynomial(*tau_args)
        var_op_list += tau_vars

        # Initial condition
        self.term_obj_dict["u_ic"].gen_dedalus_ops(u_op)
        self.term_obj_dict["v_ic"].gen_dedalus_ops(v_op)
        self.term_obj_dict["b_ic"].gen_dedalus_ops(b_op)

        # Problem
        problem = d3.IVP(var_op_list)
        problem.add_equation([
            d3.dt(u_op) - self.viscosity * d3.lap(u_op) + d3_dx(p_op) + tau_u,
            -(d3_dx(u_op**2) + d3_dy(u_op * v_op))])
        diff_v = -self.viscosity * (d3.lap(v_op) + d3_dy(tau_v1))
        buoyancy_op = -b_op if self.buoyancy else 0
        problem.add_equation([
            d3.dt(v_op) + diff_v + d3_dy(p_op) + buoyancy_op + tau_v2,
            -(d3_dx(u_op * v_op) + d3_dy(v_op**2))])
        problem.add_equation([
            d3.dt(b_op) - self.diffusivity * d3.lap(b_op) + tau_b,
            -(d3_dx(u_op * b_op) + d3_dy(v_op * b_op))])
        problem.add_equation([d3_dx(u_op) + d3_dy(v_op) + tau_v1 + tau_p, 0])
        problem.add_equation([d3.integ(p_op), 0])  # pressure gauge
        if not self.periodic[1]:  # boundary conditions for y_wall
            problem.add_equation([u_op(y="left"), 0])
            problem.add_equation([u_op(y="right"), 0])
            problem.add_equation([v_op(y="left"), 0])
            problem.add_equation([v_op(y="right"), 0])
            problem.add_equation([b_op(y="left"), self.delta_b])
            problem.add_equation([b_op(y="right"), 0])

        if self.ignore_p:
            var_dict = {"u": u_op, "v": v_op, "b": b_op}
        else:
            var_dict = {"u": u_op, "v": v_op, "b": b_op, "p": p_op}
        return var_dict, problem


if __name__ == "__main__":
    my_args = get_cli_args(INSWithCoupledVar2DPDE)
    pde_data_obj = INSWithCoupledVar2DPDE(my_args)
    gen_data(my_args, pde_data_obj)
