#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Test PDE data generation."""
import argparse
from typing import Tuple
import numpy as np
import dedalus.public as d3

from ..utils.basics import DedalusPDEType
from ...common.basics import get_cli_args, gen_data
try:
    from test_debug2d import dist, d3_dx, d3_dy, get_basis
except ImportError:
    from ..utils.settings2d import dist, d3_dx, d3_dy, get_basis


class TestPDE(DedalusPDEType):
    TRIAL_N_SUB_STEPS = [2, 8, 32]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        # Dedalus Bases
        xbasis = get_basis("x", True)
        ybasis = get_basis("y", True)
        self.bases = (xbasis, ybasis)
        self.coord_dict["x"] = dist.local_grid(xbasis)
        self.coord_dict["y"] = dist.local_grid(ybasis)

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "test"

    def _heat_eqn(self) -> Tuple:
        # Fields
        u_op = dist.Field(name="u", bases=self.bases)
        a_op = dist.Field(name="a", bases=self.bases)

        # Initial condition
        u_op['g'] = self.coord_dict["x"] + self.coord_dict["y"]
        a_op['g'] = self.coord_dict["x"] + self.coord_dict["y"] + 1

        # PDE Terms
        problem = d3.IVP([u_op])
        # problem.add_equation([d3.dt(u_op) - 1e-2 * d3_dy(d3_dy(u_op)), 0])
        problem.add_equation([d3.dt(u_op) - 1e-2 * d3.div(a_op * d3.grad(u_op)), 0])
        var_dict = {"u": u_op}
        return var_dict, problem

    def _swe(self) -> Tuple:
        # Fields
        h_op = dist.Field(name="h", bases=self.bases)
        hu_op = dist.Field(name="hu", bases=self.bases)
        hv_op = dist.Field(name="hv", bases=self.bases)

        # Initial condition
        h_ic = 1 + 1e-1 * (np.sin(2 * np.pi * self.coord_dict["x"])
                           + np.sin(2 * np.pi * self.coord_dict["y"]))
        h_op['g'] = h_ic
        hu_op['g'] = np.zeros_like(h_ic)
        hv_op['g'] = np.zeros_like(h_ic)

        # PDE Terms
        problem = d3.IVP([h_op, hu_op, hv_op])
        problem.add_equation([d3.dt(h_op) + d3_dx(hu_op) + d3_dy(hv_op), 0])
        grav = h_op**2
        u_nonlin = d3_dx(hu_op**2 / h_op + grav) + d3_dy(hu_op * hv_op / h_op)
        problem.add_equation([d3.dt(hu_op), -u_nonlin])
        v_nonlin = d3_dy(hv_op**2 / h_op + grav) + d3_dx(hu_op * hv_op / h_op)
        problem.add_equation([d3.dt(hv_op), -v_nonlin])
        var_dict = {"h": h_op, "hu": hu_op, "hv": hv_op}
        return var_dict, problem

    def _swe_aux_var(self) -> Tuple:  # matrix singular
        # Fields
        h_op = dist.Field(name="h", bases=self.bases)
        u_op = dist.Field(name="u", bases=self.bases)
        v_op = dist.Field(name="v", bases=self.bases)
        hu_op = dist.Field(name="hu", bases=self.bases)
        hv_op = dist.Field(name="hv", bases=self.bases)

        # Initial condition
        h_ic = 1 + 1e-1 * (np.sin(2 * np.pi * self.coord_dict["x"])
                           + np.sin(2 * np.pi * self.coord_dict["y"]))
        h_op['g'] = h_ic
        u_op['g'] = np.zeros_like(h_ic)
        v_op['g'] = np.zeros_like(h_ic)
        hu_op['g'] = h_op['g'] * u_op['g']
        hv_op['g'] = h_op['g'] * v_op['g']

        # PDE Terms
        problem = d3.IVP([h_op, u_op, v_op, hu_op, hv_op])
        problem.add_equation([hu_op, h_op * u_op])  # suspected source of singularity
        problem.add_equation([hv_op, h_op * v_op])
        problem.add_equation([d3.dt(h_op) + d3_dx(hu_op) + d3_dy(hv_op), 0])
        grav = h_op**2
        u_nonlin = d3_dx(hu_op * u_op + grav) + d3_dy(hu_op * v_op)
        problem.add_equation([d3.dt(hu_op), -u_nonlin])
        v_nonlin = d3_dy(hv_op * v_op + grav) + d3_dx(hu_op * v_op)
        problem.add_equation([d3.dt(hv_op), -v_nonlin])
        var_dict = {"h": h_op, "u": u_op, "v": v_op}
        return var_dict, problem

    def _swe_vec(self) -> Tuple:
        # Fields
        h_op = dist.Field(name="h", bases=self.bases)
        hu_op = dist.VectorField(dist.coordsystems[0], name="hu", bases=self.bases)

        # Initial condition
        h_ic = 1 + 1e-1 * (np.sin(2 * np.pi * self.coord_dict["x"])
                           + np.sin(2 * np.pi * self.coord_dict["y"]))
        h_op['g'] = h_ic
        hu_op['g'][:] = np.zeros_like(h_ic)

        # PDE Terms
        problem = d3.IVP([h_op, hu_op])
        problem.add_equation([d3.dt(h_op) + d3.div(hu_op), 0])
        nonlin = d3.div((hu_op * hu_op) / h_op) + d3.grad(h_op**2)
        # nonlin = d3.div(np.tensordot(hu_op, hu_op, 0) / h_op) + d3.grad(h_op**2)
        problem.add_equation([d3.dt(hu_op), -nonlin])
        var_dict = {"h": h_op}  # , "hu": hu_op}
        return var_dict, problem

    def _swe_convec(self) -> Tuple:
        # Fields
        h_op = dist.Field(name="h", bases=self.bases)
        u_op = dist.Field(name="u", bases=self.bases)
        v_op = dist.Field(name="v", bases=self.bases)

        # Initial condition
        h_ic = 1 + 1e-1 * (np.sin(2 * np.pi * self.coord_dict["x"])
                           + np.sin(2 * np.pi * self.coord_dict["y"]))
        h_op['g'] = h_ic
        u_op['g'] = np.zeros_like(h_ic)
        v_op['g'] = np.zeros_like(h_ic)

        # PDE Terms
        problem = d3.IVP([h_op, u_op, v_op])
        h_nonlin = d3_dx(h_op * u_op) + d3_dy(h_op * v_op)
        problem.add_equation([d3.dt(h_op), -h_nonlin])
        u_nonlin = u_op * d3_dx(u_op) + v_op * d3_dy(u_op)
        problem.add_equation([d3.dt(u_op) + d3_dx(h_op), -u_nonlin])
        v_nonlin = u_op * d3_dx(v_op) + v_op * d3_dy(v_op)
        problem.add_equation([d3.dt(v_op) + d3_dy(h_op), -v_nonlin])
        var_dict = {"h": h_op, "u": u_op, "v": v_op}
        return var_dict, problem

    def _swe_vec_convec(self) -> Tuple:
        # Fields
        h_op = dist.Field(name="h", bases=self.bases)
        u_op = dist.VectorField(dist.coordsystems[0], name="u", bases=self.bases)

        # Initial condition
        h_ic = 1 + 1e-1 * (np.sin(2 * np.pi * self.coord_dict["x"])
                           + np.sin(2 * np.pi * self.coord_dict["y"]))
        h_op['g'] = h_ic
        u_op['g'][:] = np.zeros_like(h_ic)

        # PDE Terms
        problem = d3.IVP([h_op, u_op])
        problem.add_equation([d3.dt(h_op), -d3.div(h_op * u_op)])
        nonlin = u_op @ d3.grad(u_op)
        problem.add_equation([d3.dt(u_op) + d3.grad(h_op), -nonlin])
        var_dict = {"h": h_op}
        return var_dict, problem

    get_dedalus_problem = _swe


class TestYBoundedPDE(DedalusPDEType):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        # Dedalus Bases
        xbasis = get_basis("x", True)
        ybasis = get_basis("y", False)
        self.bases = (xbasis, ybasis)
        self.coord_dict["x"] = dist.local_grid(xbasis)
        self.coord_dict["y"] = dist.local_grid(ybasis)

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "test"

    def _dcr(self) -> Tuple:
        (xbasis, ybasis) = self.bases

        # Fields
        u_op = dist.Field(name="u", bases=self.bases)
        a_op = dist.Field(name="a", bases=self.bases)
        bc_yr = dist.Field(name="bc_yr", bases=xbasis)

        # Initial condition
        u_op['g'] = self.coord_dict["x"] + self.coord_dict["y"]
        a_op['g'] = self.coord_dict["x"] + self.coord_dict["y"] + 1
        print(bc_yr['g'].shape)
        bc_yr['g'] = self.coord_dict["x"]

        # lin = -1e-1 * d3.div(a_op * d3.grad(u_op))
        lin = -1e-2 * d3.div(d3.grad(u_op))

        # Tau polynomials, following
        # https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_3.html
        tau1 = dist.Field(name='tau1', bases=xbasis)
        tau2 = dist.Field(name='tau2', bases=xbasis)
        tau_basis = ybasis.derivative_basis(2)
        p1_op = dist.Field(name='p_tau1', bases=tau_basis)
        p2_op = dist.Field(name='p_tau2', bases=tau_basis)
        p1_op['c'][:, -1] = 1
        p2_op['c'][:, -2] = 2
        lin = lin + tau1 * p1_op + tau2 * p2_op

        # PDE Terms
        problem = d3.IVP([u_op, tau1, tau2])
        # problem.add_equation([d3.dt(u_op) - 1e-2 * d3_dy(d3_dy(u_op)), 0])
        problem.add_equation([d3.dt(u_op) + lin, 0])
        problem.add_equation([u_op(y='left'), 0])
        # problem.add_equation([u_op(y='right'), 0])
        # problem.add_equation([d3_dy(u_op)(y='right'), bc_yr])
        # problem.add_equation([(bc_yr * u_op + d3_dy(u_op))(y='right'), bc_yr])
        problem.add_equation([d3_dy(u_op)(y='right'), bc_yr - bc_yr * u_op(y='right')])
        var_dict = {"u": u_op}
        return var_dict, problem

    def _wave(self) -> Tuple:
        (xbasis, ybasis) = self.bases

        # Fields
        u_op = dist.Field(name="u", bases=self.bases)
        ut_op = dist.Field(name="ut", bases=self.bases)
        bc_yr = dist.Field(name="bc_yr", bases=xbasis)

        # Initial condition
        # u_op['g'] = self.coord_dict["x"] + self.coord_dict["y"]
        u_op['g'] = 1 - self.coord_dict["y"]
        print(bc_yr['g'].shape)
        bc_yr['g'] = self.coord_dict["x"]

        lin = -4 * d3.div(d3.grad(u_op))

        # Tau polynomials, following
        # https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_3.html
        tau1 = dist.Field(name='tau1', bases=xbasis)
        tau2 = dist.Field(name='tau2', bases=xbasis)
        tau_basis = ybasis.derivative_basis(2)
        p1_op = dist.Field(name='p_tau1', bases=tau_basis)
        p2_op = dist.Field(name='p_tau2', bases=tau_basis)
        p1_op['c'][:, -1] = 1
        p2_op['c'][:, -2] = 2
        lin = lin + tau1 * p1_op + tau2 * p2_op

        # PDE Terms
        problem = d3.IVP([u_op, ut_op, tau1, tau2])
        problem.add_equation([d3.dt(u_op) - ut_op, 0])
        problem.add_equation([d3.dt(ut_op) + lin, 0])
        problem.add_equation([u_op(y='left'), 0])
        problem.add_equation([(ut_op + 2 * d3_dy(u_op))(y='right'), 0])
        # problem.add_equation([u_op(y='right'), 1])
        # problem.add_equation([d3_dy(u_op)(y='right'), bc_yr])
        # problem.add_equation([(bc_yr * u_op + d3_dy(u_op))(y='right'), bc_yr])
        # problem.add_equation([d3_dy(u_op)(y='right'), bc_yr - bc_yr * u_op(y='right')])
        var_dict = {"u": u_op}
        return var_dict, problem

    get_dedalus_problem = _wave


class TestXYBoundedPDE(DedalusPDEType):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        # Dedalus Bases
        xbasis = get_basis("x", False)
        ybasis = get_basis("y", False)
        self.bases = (xbasis, ybasis)
        self.coord_dict["x"] = dist.local_grid(xbasis)
        self.coord_dict["y"] = dist.local_grid(ybasis)

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "test_npXY"

    def get_dedalus_problem(self) -> Tuple:
        (xbasis, ybasis) = self.bases

        # Fields
        u_op = dist.Field(name="u", bases=self.bases)

        # Initial condition
        u_op['g'] = self.coord_dict["x"] + self.coord_dict["y"]

        lin = -1e-2 * d3.div(d3.grad(u_op))

        # Tau polynomials, following
        # https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_3.html
        tau_y1 = dist.Field(name='tau_y1', bases=xbasis)
        tau_y2 = dist.Field(name='tau_y2', bases=xbasis)
        tau_y_basis = ybasis.derivative_basis(2)
        p_y1_op = dist.Field(name='p_y_tau1', bases=tau_y_basis)
        p_y2_op = dist.Field(name='p_y_tau2', bases=tau_y_basis)
        p_y1_op['c'][:, -1] = 1
        p_y2_op['c'][:, -2] = 2
        lin = lin + tau_y1 * p_y1_op + tau_y2 * p_y2_op

        tau_x1 = dist.Field(name='tau_x1', bases=ybasis)
        tau_x2 = dist.Field(name='tau_x2', bases=ybasis)
        tau_x_basis = xbasis.derivative_basis(2)
        p_x1_op = dist.Field(name='p_x_tau1', bases=tau_x_basis)
        p_x2_op = dist.Field(name='p_x_tau2', bases=tau_x_basis)
        p_x1_op['c'][-1] = 1
        p_x2_op['c'][-2] = 2
        lin = lin + tau_x1 * p_x1_op + tau_x2 * p_x2_op

        # PDE Terms
        problem = d3.IVP([u_op, tau_y1, tau_y2, tau_x1, tau_x2])
        # problem.add_equation([d3.dt(u_op) - 1e-2 * d3_dy(d3_dy(u_op)), 0])
        problem.add_equation([d3.dt(u_op) + lin, 0])
        problem.add_equation([u_op(y='left'), 0])
        problem.add_equation([u_op(y='right'), 0])
        problem.add_equation([u_op(x='left'), 0])
        problem.add_equation([u_op(x='right'), 0])
        var_dict = {"u": u_op}
        return var_dict, problem


if __name__ == "__main__":
    my_args = get_cli_args(TestPDE)
    pde_data_obj = TestPDE(my_args)
    gen_data(my_args, pde_data_obj)
