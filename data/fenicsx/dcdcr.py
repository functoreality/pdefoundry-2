#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Solve the 2D diffusion-convection-reaction PDE by FEniCS-X. Read coefficients from HDF5 file."""
import sys
import argparse
from dolfinx import fem, mesh, log
import numpy as np
from mpi4py import MPI
from dolfinx.fem import (Function, FunctionSpace, Constant, form)
from ufl import (FacetNormal, TrialFunction, TestFunction, dx, ds, grad, dot, as_vector, Dx, lhs, rhs)
from dolfinx_mpc import MultiPointConstraint, LinearProblem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import h5py
from scipy.interpolate import RegularGridInterpolator
import os
import logging
from petsc4py import PETSc
from typing import Tuple, Optional, Callable
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gc
from .dcr import setup_logger, read_datasets, prepare_plot_2d_video, prepare_plot_2d_video_compare
from .utils.interp import get_2d_interp_function
class FenicsxDCDCR:
    r"""Solve the 2D div-constraineddiffusion-convection-reaction PDE by FEniCS-X. Read coefficients from HDF5 file."""
    N: int = 32
    MAXTIME: float = 1.0
    TIMESTEPS: int = 1000
    SAMPLESTEPS: int = 100
    FORWARD_EULER = 0
    BACKWARD_EULER = 1
    CRANK_NICOLSON = 2
    SSPRK22 = 3
    SSPRK33 = 4
    SSPRK104 = 5
    FEM_TYPE: Tuple = ("Lagrange", 1)

    def __init__(self,
                 datasets: dict, 
                 name_datasets: str,
                 data_idx: int,
                 n_grid: int = N,
                 time_steps: int = TIMESTEPS,
                 show_fenicsx_log: bool = False,
                 logger: Optional[logging.Logger] = None,
                 comm: Optional[MPI.Comm] = None):
                
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm

        if n_grid < 1:
            raise ValueError("Number of grids must be positive.")
        self.n_grid = n_grid
        if time_steps % self.SAMPLESTEPS != 0:
            raise ValueError(f"Mumber for time steps must be divisible by "
                             f"{self.SAMPLESTEPS}.")
        if time_steps < 1:
            raise ValueError("Number of time steps must be positive.")
        self.time_steps = time_steps
        self.time_sample_interval = max(int(time_steps / self.SAMPLESTEPS), 1)
        self.name_equ = f"{name_datasets}_N{self.n_grid}_equ{data_idx}_{time_integrator}_timesteps{time_steps}"
        if logger is None:
            os.makedirs("log", exist_ok=True)
            logger = setup_logger(f"log/{self.name_equ}.log")
        self.logger = logger
        self.show_fenicsx_log = show_fenicsx_log
        if self.show_fenicsx_log:
            log.set_log_level(log.LogLevel.INFO)
        else:
            log.set_log_level(log.LogLevel.WARNING)
        
        self.coef_f0_lin = {
            'coo_len': datasets['coef/f0/lin/coo_len'][data_idx],
            'coo_i': datasets['coef/f0/lin/coo_i'][data_idx],
            'coo_j': datasets['coef/f0/lin/coo_j'][data_idx],
            'coo_val': datasets['coef/f0/lin/coo_val'][data_idx]
        }
        self.coef_f0_deg2 = {
            'coo_len': datasets['coef/f0/deg2/coo_len'][data_idx],
            'coo_i': datasets['coef/f0/deg2/coo_i'][data_idx],
            'coo_j': datasets['coef/f0/deg2/coo_j'][data_idx],
            'coo_k': datasets['coef/f0/deg2/coo_k'][data_idx],
            'coo_val': datasets['coef/f0/deg2/coo_val'][data_idx]
        }

        self.coef_f1_lin = {
            'coo_len': datasets['coef/f1/lin/coo_len'][data_idx],
            'coo_i': datasets['coef/f1/lin/coo_i'][data_idx],
            'coo_j': datasets['coef/f1/lin/coo_j'][data_idx],
            'coo_val': datasets['coef/f1/lin/coo_val'][data_idx]
        }
        self.coef_f1_deg2 = {
            'coo_len': datasets['coef/f1/deg2/coo_len'][data_idx],
            'coo_i': datasets['coef/f1/deg2/coo_i'][data_idx],
            'coo_j': datasets['coef/f1/deg2/coo_j'][data_idx],
            'coo_k': datasets['coef/f1/deg2/coo_k'][data_idx],
            'coo_val': datasets['coef/f1/deg2/coo_val'][data_idx]
        }
        self.coef_f2_lin = {
            'coo_len': datasets['coef/f2/lin/coo_len'][data_idx],
            'coo_i': datasets['coef/f2/lin/coo_i'][data_idx],
            'coo_j': datasets['coef/f2/lin/coo_j'][data_idx],
            'coo_val': datasets['coef/f2/lin/coo_val'][data_idx]
        }
        self.coef_f2_deg2 = {
            'coo_len': datasets['coef/f2/deg2/coo_len'][data_idx],
            'coo_i': datasets['coef/f2/deg2/coo_i'][data_idx],
            'coo_j': datasets['coef/f2/deg2/coo_j'][data_idx],
            'coo_k': datasets['coef/f2/deg2/coo_k'][data_idx],
            'coo_val': datasets['coef/f2/deg2/coo_val'][data_idx]
        }
        self.s0 = datasets['coef/s/0/field'][data_idx].reshape((self.n_grid, self.n_grid,1))
        self.s1 = datasets['coef/s/1/field'][data_idx].reshape((self.n_grid, self.n_grid,1))
        self.s = np.stack([self.s0, self.s1], axis=-1)
        self.u0_ic = datasets['coef/u_ic/0'][data_idx].reshape((self.n_grid, self.n_grid,1))
        self.u1_ic = datasets['coef/u_ic/1'][data_idx].reshape((self.n_grid, self.n_grid,1))
        self.u_ic = np.stack([self.u0_ic, self.u1_ic], axis=-1)
        self.a_0 = datasets['coef/a/0'][data_idx][0,0]
        self.a_1 = datasets['coef/a/1'][data_idx][0,0]
        self.c = datasets['coef/c'][data_idx]
        self.x_coord = datasets['coord/x'].flatten()
        self.y_coord = datasets['coord/y'].flatten()

        self.domain = mesh.create_unit_square(self.comm, self.n_grid, self.n_grid)

        x_grid, y_grid = np.meshgrid(self.x_coord, self.y_coord, indexing='ij')
        points = np.stack([x_grid.flatten(), y_grid.flatten(),
                           np.zeros_like(x_grid.flatten())], axis=1)

        self.points, self.cells = self.locate_cells(points)
        v_cg2 = element("Lagrange", self.domain.topology.cell_name(), 2, shape=(self.domain.geometry.dim, ))
        s_cg1 = element("Lagrange", self.domain.topology.cell_name(), 1)
        self.V = FunctionSpace(self.domain, v_cg2)
        self.Q = Functionspace(self.domain, s_cg1)
        self.mpc_v = self.assign_periodic_boundary_conditions(self.V)
        self.mpc_q = self.assign_periodic_boundary_conditions(self.Q)
        self.s_func = get_2d_interp_function(self.s, self.x_coord, self.y_coord)
        self.u_ic_func = get_2d_interp_function(self.u_ic, self.x_coord, self.y_coord)
        
        self.c_0 = fem.Constant(self.domain, self.c[0])
        self.c_1 = fem.Constant(self.domain, self.c[1])
        self.c_2 = fem.Constant(self.domain, self.c[2])
        self.a_0 = fem.Constant(self.domain, self.a_0)
        self.a_1 = fem.Constant(self.domain, self.a_1)

    def assign_periodic_boundary_conditions(self, V) -> None:
        r"""Assign periodic boundary conditions to the function space."""
        mpc = MultiPointConstraint(V)

        # Create periodic constraints
        bcs = []
        mpc.create_periodic_constraint_geometrical(
            V, self._indicator, self._relation, bcs)
        mpc.finalize()
        self.logger.info(f"Number of slaves: {len(mpc.slaves)}")
        return mpc

    def locate_cells(self, points: NDArray) -> Tuple[NDArray, NDArray]:
        r"""Locate cells in the mesh for given points. Shape of points is (n_points, 3)."""
        tree = bb_tree(self.domain, self.domain.topology.dim)
        cells = []
        points_on_proc = []
        cell_candidates = compute_collisions_points(tree, points)
        colliding_cells = compute_colliding_cells(self.domain, cell_candidates, points)

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                cells.append(colliding_cells.links(i)[0])
                points_on_proc.append(point)
            else:
                self.logger.warning(f"Point {point} is outside the domain.")
        return np.array(points_on_proc), np.array(cells)

    def eval(self, f: fem.Function) -> NDArray:
        r"""Evaluate the fem Function at grid points."""
        return f.eval(self.points, self.cells).reshape(
            (len(self.x_coord), len(self.y_coord)))

    def _indicator(self, x):
        r"""
        Indicator function to determine if a point is on the
        top or right boundary
        """
        on_top_or_right = np.isclose(x[1], 1.0, atol=1e-6) | np.isclose(
            x[0], 1.0, atol=1e-6)
        return on_top_or_right

    def _relation(self, x):
        r"""
        Relation function to map points from the top/right boundary to
        the bottom/left boundary
        """
        out_x = np.zeros(x.shape)
        out_x[0] = np.where(np.isclose(x[0], 1.0, atol=1e-6), 0.0, x[0])
        out_x[1] = np.where(np.isclose(x[1], 1.0, atol=1e-6), 0.0, x[1])
        return out_x

    def _get_interp_function(self, values: NDArray) -> Callable:
        r"""Get the interpolation function for the given values."""
        def func(x):
            inner_func = RegularGridInterpolator((self.x_coord, self.y_coord), values)
            x_ = self._relation(x)[:2].T  # size is (n_points, 2)
            mins = np.array([min(self.x_coord), min(self.y_coord)])
            maxs = np.array([max(self.x_coord), max(self.y_coord)])
            x_ = np.clip(x_, mins, maxs)
            return inner_func(x_).T
        return func

    def _get_poly_term_lin(self, coef_dict: dict, u: fem.Function) -> NDArray:
        r"""Get the linear polynomial term f_i(u) of the PDE."""
        u_0 = u[0]
        u_1 = u[1]
        coo_len = coef_dict['coo_len']
        coo_i = coef_dict['coo_i'][:coo_len]
        coo_j = coef_dict['coo_j'][:coo_len]
        coo_val = coef_dict['coo_val'][:coo_len]
        f = [[fem.Constant(self.domain, 0) for _ in range(2)] for _ in range(2)]
        for i in range(coo_len):
            f[coo_i[i]][coo_j[i]] = fem.Constant(self.domain, coo_val[i])
        return [f[0][0] * u0 + f[0][1] * u1, f[1][0] * u0 + f[1][1] * u1]
    
    def _get_poly_term_deg2(self, coef_dict: dict, u: fem.Function) -> NDArray:
        r"""Get the quadratic polynomial term f_i(u) of the PDE."""
        u0 = u[0]
        u1 = u[1]
        coo_len = coef_dict['coo_len']
        coo_i = coef_dict['coo_i'][:coo_len]
        coo_j = coef_dict['coo_j'][:coo_len]
        coo_k = coef_dict['coo_k'][:coo_len]
        coo_val = coef_dict['coo_val'][:coo_len]
        f = [[[fem.Constant(self.domain, 0) for _ in range(2)] for _ in range(2)] for _ in range(2)]
        for i in range(coo_len):
            f[coo_i[i]][coo_j[i]][coo_k[i]] = fem.Constant(self.domain, coo_val[i])
        return [f[0][0][0] * u0 * u0 + f[0][0][1] * u0 * u1 + f[0][1][0] * u1 * u0 + f[0][1][1] * u1 * u1,
                f[1][0][0] * u0 * u0 + f[1][0][1] * u0 * u1 + f[1][1][0] * u1 * u0 + f[1][1][1] * u1 * u1]

    def solve(self) -> NDArray:
        r'''solve the PDE using the following timestep:
        u^{n + 1/2} = u^n + dt * A(u^n, v^n)
        p^{n+1} is the solution of the following linear problem:
        (-\Delta + \|c\|^2)p = - (c_2 + c_0 * u^{n+1/2}_0 + c_1 * u^{n+1/2}_0 + \nabla u^{n+1/2})
        u^{n+1} = u^{n+1/2} - dt * (\nabla p^{n+1} + p^{n+1} * c)
        '''
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        p = TrialFunction(self.Q)
        q = TestFunction(self.Q)
        s = Function(self.V)
        s.interpolate(self.s_func)
        u_n = Function(self.V)
        u_n.name = 'u_n'
        u_n.interpolate(self.u_ic_func)
        p_n = Function(self.Q)
        p_n.name = 'p_n'
        n = FacetNormal(self.domain)

        def get_lin_problem(a, L, mpc):
            problem = LinearProblem(a, L, mpc, petsc_options={
                "ksp_type": "cg",
                "pc_type": "gamg",
            })
            return problem

        dt = self.MAXTIME / self.time_steps
        k = Constant(self.domain, dt)
        kd = Constant(self.domain, 1 / dt)

        # get step 1
        F1 = dot((u - u_n)/k, v) * dx 
        # Lu
        F1 += as_vector([self.a_0 * (dot(grad(u_n[0]), n) * v[0] * ds - dot(grad(u_n[0]), grad(v[0])) * dx),
        self.a_1 * (dot(grad(u_n[1]), n) * v[1] * ds - dot(grad(u_n[1]), grad(v[1])) * dx)])
        # f_0
        F1 += dot(as_vector(self._get_poly_term_deg2(self.coef_f0_deg2, u_n)), v) * dx
        F1 += dot(as_vector(self._get_poly_term_lin(self.coef_f0_lin, u_n)), v) * dx
        # f_1
        F1 += dot(as_vector(self._get_poly_term_deg2(self.coef_f1_deg2, u_n)), v) * n[0] * ds -\
                dot(as_vector(self._get_poly_term_deg2(self.coef_f1_deg2, u_n)), Dx(v, 0)) * dx +\
                dot(as_vector(self._get_poly_term_lin(self.coef_f1_lin, u_n)), v) * n[0] * ds -\
                dot(as_vector(self._get_poly_term_lin(self.coef_f1_lin, u_n)), Dx(v, 0)) * dx
        # f_2
        F1 += dot(as_vector(self._get_poly_term_deg2(self.coef_f2_deg2, u_n)), v) * n[1] * ds -\
                dot(as_vector(self._get_poly_term_deg2(self.coef_f2_deg2, u_n)), Dx(v, 1)) * dx +\
                dot(as_vector(self._get_poly_term_lin(self.coef_f2_lin, u_n)), v) * n[1] * ds -\
                dot(as_vector(self._get_poly_term_lin(self.coef_f2_lin, u_n)), Dx(v, 1)) * dx
        
        F1 += dot(s, v) * dx

        a1 = form(lhs(F1))
        L1 = form(rhs(F1))
        problem1 = get_linear_problem(a1, L1, self.mpc_v)

        # step 2 (-\Delta + \|c\|^2)p = - (c_2 + c_0 * u^{n+1/2}_0 + c_1 * u^{n+1/2}_1 + \nabla u^{n+1/2})/dt
        u_ = Function(self.V)
        F2 = -dot(grad(p), n) * q * ds + dot(grad(p), grad(q)) * dx +\
             (self.c_0 * self.c_0 + self.c1 * self.c1) * p * q * dx
        F2 += kd * (c2 + dot(as_vector[self.c_0, self.c_1], u_)) * q * dx
        F2 += kd * (dot(u_, n) * q * ds - dot(u_, grad(q)) * dx)
        a2 = form(lhs(F2))
        L2 = form(rhs(F2))
        problem2 = get_linear_problem(a2, L2, self.mpc_q)

        # step 3
        p_ = Function(self.Q)
        F3 = dot((u - u_)/k, v) * dx
        F3 += grad(p_) *  v * dx
        F3 -= p_ * dot(as_vector([self.c_0, self.c_1]), grad(v)) * dx
        a3 = form(lhs(F3))
        L3 = form(rhs(F3))
        problem3 = get_linear_problem(a3, L3, self.mpc_v)

        ## initial condition
        u_n[0].interpolate(self.u0_ic_func)
        u_n[1].interpolate(self.u1_ic_func)
        u_n.x.scatter_forward()
        u_sol = [self.eval(u_n).reshape((self.n_grid, self.n_grid,-1))]
        p_sol = [self.eval(p_n).reshape((self.n_grid, self.n_grid,-1))]
        ## solve the PDE
        for i in range(self.time_steps):
            self.logger.info(f"Time step {i}")
            u_ = problem1.solve()
            u_.x.scatter_forward()
            p_ = problem2.solve()
            p_.x.scatter_forward()
            u_ = problem3.solve()
            u_.x.scatter_forward()

            u_n.x.array[:] = u_.x.array
            p_n.x.array[:] = p_.x.array
            if (i+1) % self.time_sample_interval == 0:
                self.logger.info(f"Time step {i+1}, time {dt*(i+1)}")
                u_sol.append(self.eval(u_n).reshape((self.n_grid, self.n_grid,-1)))
                p_sol.append(self.eval(p_n).reshape((self.n_grid, self.n_grid,-1)))
        return np.array(u_sol), np.array(p_sol)
    
    def record(self, u_sol: NDArray, save_dir: str = "results") -> None:
        r"""Record the solution."""
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{self.name_equ}.npy")
        np.save(filename, u_sol)
        self.logger.info(f"Solution saved to '{filename}'.")

    def plot(self, u_sol: NDArray, save_dir: str = "results",
             ref_sol: Optional[NDArray] = None) -> None:
        r"""Plot the solution."""
        os.makedirs(save_dir, exist_ok=True)
        if ref_sol is None:
            filename = os.path.join(save_dir, f"{self.name_equ}.gif")
            anim = prepare_plot_2d_video(u_sol, (self.x_coord, self.y_coord),
                                        ax_labels="xy", title="fenicsx solution",
                                        val_lim=(-3., 3.))
            anim.save(filename, writer="imagemagick")
            self.logger.info(f"Solution plot saved to '{filename}'.")
            plt.close()
        else:
            if u_sol.shape != ref_sol.shape:
                print(u_sol.shape)
                print(ref_sol.shape)
                raise ValueError("Solution and reference solution have different shapes.")
            filename = os.path.join(save_dir, f"{self.name_equ}_compare.gif")
            anim = prepare_plot_2d_video_compare(u_sol, ref_sol, (self.x_coord, self.y_coord),
                                                 ax_labels="xy", title1="fenicsx solution",
                                                 title2="reference solution",
                                                 val_lim=(-3., 3.))
            anim.save(filename, writer="imagemagick")
            self.logger.info(f"Solution plot saved to '{filename}'.")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the 2D diffusion-convection-reaction PDE by FEniCS-X.")
    parser.add_argument("--h5_file", type=str, help="Path to the HDF5 file.")
    parser.add_argument("--data_idx", type=int, help="Index of the dataset to use.")
    parser.add_argument("--n_grid", type=int, default=FenicsxDCDCR.N, help="Number of grids.")
    parser.add_argument("--time_steps", type=int, default=FenicsxDCDCR.TIMESTEPS, help="Number of time steps.")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the results.")
    parser.add_argument("--plot", action="store_true", help="Plot the solution.")
    parser.add_argument("--plot_ref", action="store_true", help="Plot the reference solution.")
    parser.add_argument("--show_fenicsx_log", action="store_true", help="Show fenicsx solver log.")
    args = parser.parse_args()

    datasets = h5py.File(args.h5_file, 'r')
    fenics_dcdcr = FenicsxDCDCR(datasets,
                            name_datasets=os.path.basename(args.h5_file),
                            data_idx=args.data_idx,
                            n_grid=args.n_grid,
                            time_steps=args.time_steps,
                            show_fenicsx_log=args.show_fenicsx_log)

    fenics_dcdcr.logger.info("Arguments:")
    fenics_dcdcr.logger.info(args)
    u_sol = fenics_dcdcr.solve()
    fenics_dcdcr.record(u_sol, save_dir=args.save_dir)
    if args.plot:
        ref_sol = None
        if args.plot_ref:
            # ref_sol = read_datasets(args.h5_file, ["sol/u"])["sol/u"][args.data_idx]
            with h5py.File(args.h5_file, 'r') as hdf:
                ref_sol = np.array(hdf["sol/u"][args.data_idx])
        fenics_dcdcr.plot(u_sol, save_dir=args.save_dir, ref_sol=ref_sol)



        


        

                


