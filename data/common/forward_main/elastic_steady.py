#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of 2D elastic steady state equation."""
import argparse
import os
from datetime import datetime
from typing import List
from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from data.common import basics, coefs
from data.common.forward_main.elastic_wave import RectBoundaryCondition, TimeIndepForce


class ElasticSteadyEquation(basics.PDETypeBase):
    r"""
    ======== Elastic Steady State Equation ========
    The PDE takes the form
        $\sigma_{ji,j}+f_i(r)=0,$
    $r=(x,y)\in[0,1]^2$.

    Here, $\sigma_{ij} = \sigma_{ji}$ is the stress tensor, $f(r)$ is the external
    force. The stress (\sigma_{11}, \sigma_{22}, \sigma_{12})^T is determined by the
    strain (\epsilon_{11}, \epsilon_{22}, \epsilon_{12})^T through a 3x3 matrix $C$:
        $\sigma_{ij} = C_{ijkl}(r)\epsilon_{kl}.$
    The strain is given by
        $\epsilon_{ij}=\frac{1}{2}(\partial_i u_j+\partial_j u_i)$.

    ======== Detailed Description ========
    - Boundary condition on each side is randomly chosen from Dirichlet, Neumann,
      and Robin. The values of boundary conditions are randomly chosen from zero,
      random constant, and random 1D function.
    - The density $\rho(r)$ and elements of the stiffness tensor $C_{ijkl}(r)$ are
      randomly chosen from random positive constants and random 2D functions.
    - The external force $f_i(r)$ is randomly chosen from zero, random constant,
      random constant times $\rho(r)$ and random 2D functions.
    """
    VERSION: float = 2.1
    PREPROCESS_DAG: True
    PDE_TYPE_ID: 13
    N_VARS = 2
    CORNERS = (0, 1, 0, 1)  # corners of the domain, (x_min, x_max, y_min, y_max)

    type: int
    ISOTROPY = 0
    stiffness_dof: int  # number of independent stiffness tensor components

    n_x_grid: int
    n_y_grid: int
    n_t_grid: int

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._global_idx = -1  # only for plot
        bc = RectBoundaryCondition
        self.n_x_grid = args.n_x_grid
        self.n_y_grid = args.n_y_grid
        self.num_max_forces = args.n_forces
        self.num_max_force_types = args.n_types
        self.force_types = args.force_types

        if args.n_forces < 1:
            raise ValueError("n_forces should be greater than 0.")
        if args.n_types < 1:
            raise ValueError("n_types should be greater than 0.")
        if len(TimeIndepForce.SUPPORTED_TYPES) < args.n_types:
            raise ValueError(f"n_types should be less than or equal to"
                             f" {len(TimeIndepForce.SUPPORTED_TYPES)}.")
        if len(args.force_types) < args.n_types:
            raise ValueError("force_types should have at least n_types elements.")
        for force_type in args.force_types:
            if force_type not in TimeIndepForce.SUPPORTED_TYPES:
                raise ValueError(f"Unsupported force type: {force_type}")

        self.store_scatter = args.save_scatter
        x_coord = np.linspace(self.CORNERS[0], self.CORNERS[1], self.n_x_grid)
        y_coord = np.linspace(self.CORNERS[2], self.CORNERS[3], self.n_y_grid)
        self.coord_dict = {"x": x_coord, "y": y_coord}

        self.type = self.ISOTROPY  # only support isotropy now
        self.stiffness_dof = 2
        resolution = (args.n_x_grid, args.n_y_grid)
        self.plot_dir = os.path.join("plots", args.plot_dir,
                                     self.get_hdf5_file_prefix(args),
                                     datetime.now().strftime("%Y%m%d%H%M%S"))

        # PDE terms
        self.term_obj_dict["u_bc"] = bc(self.N_VARS,  # bc for time-dependent equation
                                        args.n_x_grid,
                                        args.n_y_grid,
                                        args.coef_distribution,
                                        args.coef_magnitude)
        self.term_obj_dict["rho"] = coefs.NonNegConstOrField(
            coords=(x_coord.reshape(-1, 1), y_coord.reshape(1, -1)),
            periodic=False, resolution=resolution,
            min_val=args.kappa_min, max_val=args.kappa_max)
        if self.type == self.ISOTROPY:
            self.term_obj_dict["C/0"] = coefs.NonNegConstOrField(
                coords=(x_coord.reshape(-1, 1), y_coord.reshape(1, -1)),
                periodic=False, resolution=resolution,
                min_val=args.kappa_min, max_val=args.kappa_max)  # Young's modulus
            self.term_obj_dict["C/1"] = coefs.NonNegConstOrField(
                coords=(x_coord.reshape(-1, 1), y_coord.reshape(1, -1)),
                periodic=False, resolution=resolution,
                min_val=0.01, max_val=1.)  # Poisson's ratio
        else:
            raise ValueError(f"Unsupported type: {self.type}")
        self.term_obj_dict["f"] = TimeIndepForce(
            args.n_x_grid, args.n_y_grid, self.CORNERS,
            args.coef_distribution, args.coef_magnitude)

        self.reset_debug()

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        stiff_type = "iso"
        return (f"ElasticSteady2D_{stiff_type}"
                f"_c{args.coef_distribution}{args.coef_magnitude:g}"
                f"_k{args.kappa_min:.0e}_{args.kappa_max:g}"
                f"_nf{args.n_forces}_ntf{args.n_types}"
                f"_ftype{''.join(map(str, args.force_types))}"
                f"_scat{args.save_scatter}")

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--n_x_grid", "-Nx", type=int, default=128,
                            help="number of grid points in x direction")
        parser.add_argument("--n_y_grid", "-Ny", type=int, default=128,
                            help="number of grid points in y direction")
        parser.add_argument("--n_forces", type=int, default=2,
                            help="maximum number of point forces")
        parser.add_argument("--n_types", type=int, default=2,
                            help="maximum number of force types in field force")
        parser.add_argument("--force_types", type=int, nargs="+",
                            default=TimeIndepForce.SUPPORTED_TYPES,
                            help="supported force types")
        parser.add_argument("--plot_dir", type=str, default="ElasticSteady2D",
                            help="directory for plots")
        parser.add_argument("--save_scatter", action="store_true",
                            help="save scatter point solution")
        coefs.RandomConstOrField.add_cli_args_(parser)
        coefs.NonNegConstOrField.add_cli_args_(parser)
        TimeIndepForce.add_cli_args_(parser)
        return parser

    @abstractmethod
    def solve_static(self) -> NDArray[float]:
        r'''
        Solve the static equation to get the solution. Return the interpolated
        solution, scatter solution, and scatter points. Shape of returned arrays
        should be [n_x_grid, n_y_grid, 2], [n_nodes, 2], and [n_nodes, 2],
        respectively.
        '''
        pass

    def reset_pde(self, rng: np.random.Generator) -> None:
        self.raw_sol_dict = None
        self.term_obj_dict["u_bc"].reset(rng)
        self.term_obj_dict["rho"].reset(rng)
        for i in range(self.stiffness_dof):
            self.term_obj_dict[f"C/{i}"].reset(rng)
        n_forces = rng.choice(self.num_max_forces) + 1
        n_types = rng.choice(self.num_max_force_types) + 1
        self.term_obj_dict["f"].reset(
            rng, rho=self.term_obj_dict["rho"].field,
            n_forces=n_forces, types=self._gen_force_types(rng, n_types))
        self._global_idx += 1  # only for plot

    def reset_debug(self) -> None:
        self.raw_sol_dict = None
        super().reset_debug()

    def gen_solution(self) -> None:
        r"""
        Generate the PDE solution corresponding to the current PDE parameters.
        """
        try:
            u, u_scat, pts = self.solve_static()
        except:
            self.raw_sol_dict = {}  # failed
            return

        self._solution = u  # [n_x_grid, n_y_grid, 2], for plot only.
        if self.store_scatter:
            solution = u_scat  # [n_nodes, 2]
            self.raw_sol_dict = {"solution": solution.astype(np.float32),
                                 "mesh_points": pts}
        else:
            solution = u
            self.raw_sol_dict = {"solution": solution.astype(np.float32)}

    def _gen_force_types(self, rng: np.random.Generator, n_types: int) -> List[int]:
        r"""
        Generate n_types force types in self.force_types.
        """
        return rng.choice(self.force_types, n_types, replace=False)

    def _plot2d(self,
                field: NDArray[float],  # [n_x_grid, n_y_grid, n_dim]
                filename: str,
                title: str = "",
                n_dim: int = 2) -> None:
        if n_dim == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            im1 = ax1.imshow(field[:, :, 0], cmap='jet', origin='lower')
            ax1.set_title(f"{title} x")
            fig.colorbar(im1, ax=ax1)
            ax1.set_xticks([])
            ax1.set_yticks([])
            im2 = ax2.imshow(field[:, :, 1], cmap='jet', origin='lower')
            ax2.set_title(f"{title} y")
            fig.colorbar(im2, ax=ax2)
            ax2.set_xticks([])
            ax2.set_yticks([])
        elif n_dim == 1:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.imshow(field, cmap='jet', origin='lower')
            ax.set_title(title)
            fig.colorbar(im, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            raise ValueError("n_dim should be 1 or 2.")
        plt.savefig(filename)
        plt.close()

    def plot(self, plot_coef: bool = True) -> None:
        solution = self._solution
        root_dir = os.path.join(os.getcwd(), self.plot_dir)
        idxstr = str(self._global_idx)
        save_dir = os.path.join(root_dir, idxstr)
        os.makedirs(save_dir, exist_ok=True)
        filename1 = os.path.join(save_dir, "solution.png")
        self._plot2d(solution, filename1, "Solution")

        if plot_coef:
            # plot rho, C/0, C/1, f
            name_lst = ["rho", "C/0", "C/1", "f"]
            os.makedirs(os.path.join(save_dir, "C"), exist_ok=True)
            for name in name_lst:
                field = self.term_obj_dict[name].field
                filename = os.path.join(save_dir, f"{name}.png")
                if name == "f":
                    n_dim = 2
                else:
                    n_dim = 1
                self._plot2d(field, filename, name, n_dim=n_dim)
