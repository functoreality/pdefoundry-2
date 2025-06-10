#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Test PDE data generation on disk domain."""
import argparse
from typing import Tuple
import numpy as np
import dedalus.public as d3

from ..utils.basics import DedalusPDEType
from ...common.basics import get_cli_args, gen_data

# Parameters
N_PHI, N_R = 32, 32
RADIUS = 0.5
RADIUS_INNER = 0.5 * RADIUS
DEALIAS = 3 / 2
ANNULUS = False # True 

# Bases
d3_coords = d3.PolarCoordinates("phi", "r")
dist = d3.Distributor(d3_coords, dtype=np.float64)
if ANNULUS:
    bases = d3.AnnulusBasis(d3_coords, shape=(N_PHI, N_R),
                            radii=(RADIUS_INNER, RADIUS),
                            dealias=DEALIAS, dtype=np.float64)
else:
    bases = d3.DiskBasis(d3_coords, shape=(N_PHI, N_R), radius=RADIUS,
                         dealias=DEALIAS, dtype=np.float64)


class TestPDE(DedalusPDEType):
    # ~/documents/github/DedalusProject-dedalus-026decd/examples/ivp_disk_libration/libration.py
    T_SAVE_STEPS = [1]
    TIMESTEPPER = d3.RK222

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        # Dedalus Bases
        phi_coord, r_coord = dist.local_grids(bases)
        self.coord_dict["phi"] = phi_coord
        self.coord_dict["r"] = r_coord
        # x_coord = r_coord * np.cos(phi_coord)
        # y_coord = r_coord * np.sin(phi_coord)
        x_coord, y_coord = d3_coords.cartesian(phi_coord, r_coord)
        self.coord_dict["x"] = x_coord + 0.5
        self.coord_dict["y"] = y_coord + 0.5

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "test_disk"

    def get_dedalus_problem(self) -> Tuple:
        # Fields
        u_op = dist.Field(name="u", bases=bases)
        dt_u = dist.Field(name="dt_u", bases=bases)
        x_op = dist.Field(name="x", bases=bases)
        if ANNULUS:
            tau_u = dist.Field(name='tau_u', bases=bases.outer_edge)
            tau_ug = dist.Field(name='tau_u', bases=bases.outer_edge)
            tau_list = [tau_u, tau_ug]
            # First-order reduction
            rvec = dist.VectorField(d3_coords, bases=bases.radial_basis)
            rvec['g'][1] = self.coord_dict["r"]
            lift_basis = bases.derivative_basis(1)
            grad_u = d3.grad(u_op) + rvec * d3.Lift(tau_ug, lift_basis, -1)
            alpha = dist.Field(name='alpha', bases=bases.outer_edge)
        else:
            tau_u = dist.Field(name='tau_u', bases=bases.edge)
            tau_list = [tau_u]
            lift_basis = bases
            grad_u = d3.grad(u_op)
            alpha = dist.Field(name='alpha', bases=bases.edge)

        # Initial condition
        u_op['g'] = self.coord_dict["x"] + self.coord_dict["y"]
        x_op['g'] = self.coord_dict["x"]
        alpha['g'] = -np.sin(2 * self.coord_dict["phi"])  # both shape (N_PHI, 1)

        lin = -1e-1 * d3.div(grad_u) + d3.Lift(tau_u, lift_basis, -1)

        # PDE Terms
        problem = d3.IVP([u_op, dt_u] + tau_list)
        # problem.add_equation([d3.dt(u_op) + lin, d3.grad(x_op) @ grad_u])
        problem.add_equation([d3.dt(u_op) - dt_u, 0])
        problem.add_equation([d3.dt(dt_u) + lin, d3.grad(x_op) @ grad_u])

        problem.add_equation([u_op(r='right'), 0])
        # problem.add_equation([d3.radial(grad_u(r='right')), 0])
        # problem.add_equation([d3.radial(grad_u(r=RADIUS)), 0])
        # problem.add_equation([u_op(r=RADIUS), alpha + alpha * d3.radial(grad_u(r=RADIUS))])
        # problem.add_equation([d3.radial(grad_u(r=RADIUS)), alpha + alpha * u_op(r=RADIUS)])

        # # debug code
        # beta_op = dist.Field(name="beta", bases=bases.edge)
        # problem.add_equation([2 * u_op(r=RADIUS), 0.3 + beta_op])  # error code
        # problem.add_equation([2 * u_op(r=RADIUS), beta_op + 0.3])  # correct code

        if ANNULUS:
            problem.add_equation([u_op(r=RADIUS_INNER), 0])
            # negation required for u_n
            # problem.add_equation([-d3.radial(grad_u(r=RADIUS_INNER)), 0])
        var_dict = {"u": u_op}
        return var_dict, problem


if __name__ == "__main__":
    my_args = get_cli_args(TestPDE)
    pde_data_obj = TestPDE(my_args)
    gen_data(my_args, pde_data_obj)
