#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Solve the 2D diffusion-convection-reaction PDE by FEniCS-X. Read coefficients from HDF5 file."""
import sys
import argparse
from dolfinx import fem, mesh, log
import numpy as np
from mpi4py import MPI
from ufl import (FacetNormal, TrialFunction, TestFunction, dx, ds, grad, dot, as_vector)
from dolfinx_mpc import MultiPointConstraint, LinearProblem
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import h5py
from scipy.interpolate import RegularGridInterpolator
import os
import logging
from petsc4py import PETSc
from nonlinear_assembly import NonlinearMPCProblem, NewtonSolverMPC
from typing import Tuple, Optional, Callable
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gc

# TODO: use logger from pdeformer2d/src/utils/record.py
def setup_logger(log_file):
    # 创建logger
    logger = logging.getLogger('dolfinx')
    logger.setLevel(logging.DEBUG)

    # 创建文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # 创建控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 创建格式器并添加到处理器中
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 添加处理器到logger中
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

# Read coefficients from .hdf5 file
def read_datasets(h5_file_path, dataset_names):
    datasets = {}
    with h5py.File(h5_file_path, 'r') as hdf:
        for name in dataset_names:
            if name in hdf:
                datasets[name] = np.array(hdf[name])
            else:
                raise ValueError(f"Dataset '{name}' not found in '{h5_file_path}'.")
    return datasets

# TODO: import from data.common.basics
def prepare_plot_2d_video(field_2d: NDArray[float],
                          coords: Optional[Tuple[NDArray[float]]] = None,
                          ax_labels: str = "xy",
                          title: str = "",
                          val_lim: Tuple[float] = (-1., 1.)) -> None:
    r""" Prepare a 2D video plot of the time-dependent field. """
    fig, ax = plt.subplots(figsize=(6, 5))
    field_2d = field_2d.transpose((0, 2, 1))  # Shape [n_t, n_y, n_x].
    if coords is None:
        my_im = plt.pcolormesh(field_2d[0], cmap="jet")
    else:
        (x_coord, y_coord) = coords
        my_im = plt.pcolormesh(
            x_coord.flat, y_coord.flat, field_2d[0], cmap="jet")
    plt.clim(max(np.min(field_2d), val_lim[0]), min(np.max(field_2d), val_lim[1]))

    def update(frame):
        my_im.set_array(field_2d[frame])
        ax.set_title(f"{title} (T={frame})")
    update(0)
    frame_step = 1 + (len(field_2d) - 1) // 25
    anim = FuncAnimation(
        fig, update, frames=range(0, len(field_2d), frame_step), interval=100)

    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.colorbar()
    plt.tight_layout()
    return anim

def prepare_plot_2d_video_compare(field_2d: NDArray[float],
                                  ref_field_2d: NDArray[float],
                                  coords: Optional[Tuple[NDArray[float]]] = None,
                                  ax_labels: str = "xy",
                                  title1: str = "", title2: str = "",
                                  val_lim: Tuple[float] = (-1., 1.)) -> None:
    r""" Prepare a 2D video plot of two time-dependent fields. """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    field_2d = field_2d.transpose((0, 2, 1))  # Shape [n_t, n_y, n_x].
    ref_field_2d = ref_field_2d.transpose((0, 2, 1))  # Shape [n_t, n_y, n_x].

    vmin = max(min(np.min(field_2d), np.min(ref_field_2d)), val_lim[0])
    vmax = min(max(np.max(field_2d), np.max(ref_field_2d)), val_lim[1])

    if coords is None:
        im1 = ax[0].pcolormesh(field_2d[0], cmap="jet", vmin=vmin, vmax=vmax)
        im2 = ax[1].pcolormesh(ref_field_2d[0], cmap="jet", vmin=vmin, vmax=vmax)
    else:
        (x_coord, y_coord) = coords
        im1 = ax[0].pcolormesh(x_coord.flat, y_coord.flat, field_2d[0],
                               cmap="jet", vmin=vmin, vmax=vmax)
        im2 = ax[1].pcolormesh(x_coord.flat, y_coord.flat, ref_field_2d[0],
                               cmap="jet", vmin=vmin, vmax=vmax)

    plt.colorbar(im1, ax=ax[0])
    plt.colorbar(im2, ax=ax[1])

    ax[0].set_title(title1)
    ax[1].set_title(title2)

    def update(frame):
        im1.set_array(field_2d[frame].flatten())
        im2.set_array(ref_field_2d[frame].flatten())
        ax[0].set_title(f"{title1} (T={frame})")
        ax[1].set_title(f"{title2} (T={frame})")

    update(0)
    frame_step = 1 + (len(field_2d) - 1) // 25
    anim = FuncAnimation(
        fig, update, frames=range(0, len(field_2d), frame_step), interval=100)

    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    plt.tight_layout()
    return anim

class FenicsxDCR:
    r"""Solve the 2D diffusion-convection-reaction PDE by FEniCS-X. Read coefficients from HDF5 file."""

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
                 time_integrator: str = "ssprk104",
                 show_fenicsx_log: bool = False,
                 logger: Optional[logging.Logger] = None,
                 comm: Optional[MPI.Comm] = None):

        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        time_integrator = time_integrator.lower()
        if time_integrator == "forward_euler":
            self.time_integrator = self.FORWARD_EULER
        elif time_integrator == "backward_euler":
            self.time_integrator = self.BACKWARD_EULER
        elif time_integrator == "crank_nicolson":
            self.time_integrator = self.CRANK_NICOLSON
        elif time_integrator == "ssprk22":
            self.time_integrator = self.SSPRK22
        elif time_integrator == "ssprk33":
            self.time_integrator = self.SSPRK33
        elif time_integrator == "ssprk104":
            self.time_integrator = self.SSPRK104
        else:
            raise ValueError(f"Time integrator '{time_integrator}' not supported.")

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

        # Read coefficients from HDF5 file
        self.coef_f0 = datasets["coef/f0"][data_idx][0]  # size is (4,)
        self.coef_f1 = datasets["coef/f1"][data_idx][0]  # size is (4,)
        self.coef_f2 = datasets["coef/f2"][data_idx][0]  # size is (4,)
        self.a = datasets["coef/Lu/value"][data_idx]  # scalar
        self.x_coord = datasets["coord/x"].flatten() # size is (128,)
        self.y_coord = datasets["coord/y"].flatten()  # size is (128,)
        self.s_field = datasets["coef/s/field"][data_idx]  # size is (128, 128)
        self.u_ic_field = datasets["coef/u_ic"][data_idx]  # size is (128, 128)

        self.domain = mesh.create_unit_square(self.comm, self.n_grid, self.n_grid)

        x_grid, y_grid = np.meshgrid(self.x_coord, self.y_coord, indexing='ij')
        points = np.stack([x_grid.flatten(), y_grid.flatten(),
                           np.zeros_like(x_grid.flatten())], axis=1)
        # record the points that are inside the domain and their corresponding cells
        # for interpolation of solution
        # size of points_on_proc is (n_points, 3), size of cells is (n_points,)
        self.points, self.cells = self.locate_cells(points)

        self.V = fem.functionspace(self.domain, self.FEM_TYPE)
        self.mpc = self.assign_periodic_boundary_conditions(self.V)
        self.s_func = self._get_interp_function(self.s_field)
        self.u_ic_func = self._get_interp_function(self.u_ic_field)

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

    def _get_poly_term(self, coef: NDArray, u: fem.Function) -> NDArray:
        r"""Get the polynomial term of the fem Function."""
        a0 = fem.Constant(self.domain, coef[0])
        a1 = fem.Constant(self.domain, coef[1])
        a2 = fem.Constant(self.domain, coef[2])
        a3 = fem.Constant(self.domain, coef[3])
        return a0 + a1*u + a2*u*u + a3*u*u*u

    def solve(self) -> NDArray:
        r"""Solve the diffusion-convection-reaction PDE."""
        # For implicit time schemes
        un = fem.Function(self.V)
        v = TestFunction(self.V)

        # apply initial condition
        un.interpolate(self.u_ic_func)

        # source term
        s = fem.Function(self.V)
        s.interpolate(self.s_func)

        def A(u, v, dt):
            r"""Spatial term times dt"""
            n = FacetNormal(self.domain)
            f0 = self._get_poly_term(self.coef_f0, u)
            f1 = self._get_poly_term(self.coef_f1, u)
            f2 = self._get_poly_term(self.coef_f2, u)
            a = self.a
            return dt * a * dot(grad(u), grad(v)) * dx - dt * a * dot(n, grad(u)) * v * ds + \
               dt * f0 * v * dx + dt * s * v * dx - dt * dot(as_vector([f1, f2]), grad(v)) * dx + \
               dt * dot(as_vector([f1, f2]), n) * v * ds

        def get_lin_problem(a, L):
            problem = LinearProblem(a, L, self.mpc, petsc_options={
                "ksp_type": "cg",
                "pc_type": "gamg",
            })
            return problem

        def solve_linear(a, L):
            problem = get_lin_problem(a, L)
            u = problem.solve()
            u.x.scatter_forward()
            return u

        def get_nonlin_problem_and_solver(F, u):
            # problem = NonlinearProblem(F, u)
            # solver = NewtonSolver(MPI.COMM_WORLD, problem)
            problem = NonlinearMPCProblem(F, u, self.mpc)
            solver = NewtonSolverMPC(self.comm, problem, self.mpc)
            solver.convergence_criterion = "incremental"  # [residual, incremental]
            solver.rtol = 1e-6
            solver.report = self.show_fenicsx_log
            ksp = solver.krylov_solver
            opts = PETSc.Options()
            option_prefix = ksp.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "cg"
            opts[f"{option_prefix}pc_type"] = "gamg"
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            ksp.setFromOptions()
            return problem, solver

        def solve_nonlinear(F, u):
            problem, solver = get_nonlin_problem_and_solver(F, u)
            n, converged = solver.solve(u)
            if self.show_fenicsx_log:
                self.logger.info(f"Number of Newton iterations: {n}")
            if not converged:
                self.logger.error("Newton solver did not converge.")
                raise RuntimeError("Newton solver did not converge.")
            u.x.scatter_forward()

        dt = self.MAXTIME / self.time_steps
        u_sol = [self.eval(un)]
        for i in range(self.time_steps):
            self.logger.info(f"Solving time step {i+1}/{self.time_steps}")
            # Update time (not needed for this case)
            t = i * dt
            # Solve variational problem
            if self.time_integrator == self.FORWARD_EULER:
                # u = fem.Function(self.V)
                # F = (u - un) * v * dx + A(un, v, dt)
                # solve_nonlinear(F, u)
                u = TrialFunction(self.V)
                a = u * v * dx
                L = un * v * dx - A(un, v, dt)
                u = solve_linear(a, L)
            elif self.time_integrator == self.BACKWARD_EULER:
                u = fem.Function(self.V)
                F = (u - un) * v * dx + A(u, v, dt)
                solve_nonlinear(F, u)
            elif self.time_integrator == self.CRANK_NICOLSON:
                u = fem.Function(self.V)
                F = (u - un) * v * dx + A(u, v, 1/2*dt) + A(un, v, 1/2*dt)
                solve_nonlinear(F, u)
            # Reference for SSPRK:
            # Numerical Methods for Conservation Laws: From Analysis to Algorithms
            # Chapter 9: Strong stability preserving time integration
            elif self.time_integrator == self.SSPRK22:
                # u1 = fem.Function(self.V)
                # F1 = (u1 - un) * v * dx + A(un, v, dt)
                # solve_nonlinear(F1, u1)

                # u2 = fem.Function(self.V)
                # F2 = (u2 - 1/2 * un - 1/2 * u1) * v * dx + A(u1, v, 1/2 * dt)
                # solve_nonlinear(F2, u2)

                u1 = TrialFunction(self.V)
                a = u1 * v * dx
                L = un * v * dx - A(un, v, dt)
                u1 = solve_linear(a, L)

                u2 = TrialFunction(self.V)
                a = u2 * v * dx
                L = 1/2 * un * v * dx + 1/2 * u1 * v * dx - A(u1, v, 1/2 * dt)
                u2 = solve_linear(a, L)

                u = u2
            elif self.time_integrator == self.SSPRK33:
                # u1 = fem.Function(self.V)
                # F1 = (u1 - un) * v * dx + A(un, v, dt)
                # solve_nonlinear(F1, u1)

                # u2 = fem.Function(self.V)
                # F2 = (u2 - 3/4 * un - 1/4 * u1) * v * dx + A(u1, v, 1/4 * dt)
                # solve_nonlinear(F2, u2)

                # u3 = fem.Function(self.V)
                # F3 = (u3 - 1/3 * un - 2/3 * u2) * v * dx + A(u2, v, 2/3 * dt)
                # solve_nonlinear(F3, u3)

                u1 = TrialFunction(self.V)
                a = u1 * v * dx
                L = un * v * dx - A(un, v, dt)
                u1 = solve_linear(a, L)

                u2 = TrialFunction(self.V)
                a = u2 * v * dx
                L = 3/4 * un * v * dx + 1/4 * u1 * v * dx - A(u1, v, 1/4 * dt)
                u2 = solve_linear(a, L)

                u3 = TrialFunction(self.V)
                a = u3 * v * dx
                L = 1/3 * un * v * dx + 2/3 * u2 * v * dx - A(u2, v, 2/3 * dt)
                u3 = solve_linear(a, L)

                u = u3
            elif self.time_integrator == self.SSPRK104:
                # u1 = fem.Function(self.V)
                # F1 = (u1 - un) * v * dx + A(un, v, 1/6 * dt)
                # solve_nonlinear(F1, u1)

                # u2 = fem.Function(self.V)
                # F2 = (u2 - u1) * v * dx + A(u1, v, 1/6 * dt)
                # solve_nonlinear(F2, u2)

                # u3 = fem.Function(self.V)
                # F3 = (u3 - u2) * v * dx + A(u2, v, 1/6 * dt)
                # solve_nonlinear(F3, u3)

                # u4 = fem.Function(self.V)
                # F4 = (u4 - u3) * v * dx + A(u3, v, 1/6 * dt)
                # solve_nonlinear(F4, u4)

                # u5 = fem.Function(self.V)
                # F5 = (u5 - 3/5 * un - 2/5 * u4) * v * dx + A(u4, v, 1/15 * dt)
                # solve_nonlinear(F5, u5)

                # u6 = fem.Function(self.V)
                # F6 = (u6 - u5) * v * dx + A(u5, v, 1/6 * dt)
                # solve_nonlinear(F6, u6)

                # u7 = fem.Function(self.V)
                # F7 = (u7 - u6) * v * dx + A(u6, v, 1/6 * dt)
                # solve_nonlinear(F7, u7)

                # u8 = fem.Function(self.V)
                # F8 = (u8 - u7) * v * dx + A(u7, v, 1/6 * dt)
                # solve_nonlinear(F8, u8)

                # u9 = fem.Function(self.V)
                # F9 = (u9 - u8) * v * dx + A(u8, v, 1/6 * dt)
                # solve_nonlinear(F9, u9)

                # u10 = fem.Function(self.V)
                # F10 = (u10 - 1/25 * un - 9/25 * u4 - 3/5 * u9) * v * dx + \
                #     A(u4, v, 3/50 * dt) + A(u9, v, 1/10 * dt)
                # solve_nonlinear(F10, u10)

                u1 = TrialFunction(self.V)
                a = u1 * v * dx
                L = un * v * dx - A(un, v, 1/6 * dt)
                u1 = solve_linear(a, L)

                u2 = TrialFunction(self.V)
                a = u2 * v * dx
                L = u1 * v * dx - A(u1, v, 1/6 * dt)
                u2 = solve_linear(a, L)

                u3 = TrialFunction(self.V)
                a = u3 * v * dx
                L = u2 * v * dx - A(u2, v, 1/6 * dt)
                u3 = solve_linear(a, L)

                u4 = TrialFunction(self.V)
                a = u4 * v * dx
                L = u3 * v * dx - A(u3, v, 1/6 * dt)
                u4 = solve_linear(a, L)

                u5 = TrialFunction(self.V)
                a = u5 * v * dx
                L = 3/5 * un * v * dx + 2/5 * u4 * v * dx - A(u4, v, 1/15 * dt)
                u5 = solve_linear(a, L)

                u6 = TrialFunction(self.V)
                a = u6 * v * dx
                L = u5 * v * dx - A(u5, v, 1/6 * dt)
                u6 = solve_linear(a, L)

                u7 = TrialFunction(self.V)
                a = u7 * v * dx
                L = u6 * v * dx - A(u6, v, 1/6 * dt)
                u7 = solve_linear(a, L)

                u8 = TrialFunction(self.V)
                a = u8 * v * dx
                L = u7 * v * dx - A(u7, v, 1/6 * dt)
                u8 = solve_linear(a, L)

                u9 = TrialFunction(self.V)
                a = u9 * v * dx
                L = u8 * v * dx - A(u8, v, 1/6 * dt)
                u9 = solve_linear(a, L)

                u10 = TrialFunction(self.V)
                a = u10 * v * dx
                L = 1/25 * un * v * dx + 9/25 * u4 * v * dx + 3/5 * u9 * v * dx - \
                    A(u4, v, 3/50 * dt) - A(u9, v, 1/10 * dt)
                u10 = solve_linear(a, L)

                u = u10
            else:
                self.logger.error(f"Unsupported time integrator: {self.time_integrator}")

            # Update previous solution
            un.x.array[:] = u.x.array
            if (i+1) % self.time_sample_interval == 0:
                u_sol.append(self.eval(u))

        return np.array(u_sol)  # (nt, 128, 128)

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
    parser.add_argument("--n_grid", type=int, default=FenicsxDCR.N, help="Number of grids.")
    parser.add_argument("--time_steps", type=int, default=FenicsxDCR.TIMESTEPS, help="Number of time steps.")
    parser.add_argument("--time_integrator", type=str, default="ssprk104",
                        help="Time integrator: forward_euler, backward_euler, crank_nicolson, ssprk22, ssprk33, ssprk104.")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the results.")
    parser.add_argument("--plot", action="store_true", help="Plot the solution.")
    parser.add_argument("--plot_ref", action="store_true", help="Plot the reference solution.")
    parser.add_argument("--show_fenicsx_log", action="store_true", help="Show fenicsx solver log.")
    args = parser.parse_args()

    dataset_names = ["coef/f0", "coef/f1", "coef/f2", "coef/Lu/value", "coord/x", "coord/y", "coef/s/field", "coef/u_ic"]
    datasets = read_datasets(args.h5_file, dataset_names)
    fenics_dcr = FenicsxDCR(datasets,
                            name_datasets=os.path.basename(args.h5_file),
                            data_idx=args.data_idx,
                            n_grid=args.n_grid,
                            time_steps=args.time_steps,
                            time_integrator=args.time_integrator,
                            show_fenicsx_log=args.show_fenicsx_log)

    fenics_dcr.logger.info("Arguments:")
    fenics_dcr.logger.info(args)
    u_sol = fenics_dcr.solve()
    fenics_dcr.record(u_sol, save_dir=args.save_dir)
    if args.plot:
        ref_sol = None
        if args.plot_ref:
            # ref_sol = read_datasets(args.h5_file, ["sol/u"])["sol/u"][args.data_idx]
            with h5py.File(args.h5_file, 'r') as hdf:
                ref_sol = np.array(hdf["sol/u"][args.data_idx])
        fenics_dcr.plot(u_sol, save_dir=args.save_dir, ref_sol=ref_sol)
