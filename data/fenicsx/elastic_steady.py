#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of 2D elastic steady state equation."""
import argparse
from typing import Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from ufl import (TrialFunction, TestFunction, grad, Measure, nabla_div,
                 Identity, inner, sym)
from data.common import basics
from data.common.forward_main import elastic_steady
from data.fenicsx.utils.interp import (get_1d_interp_function,
                                       get_2d_interp_function, locate_cells)
from data.fenicsx.utils.domain import extract_square_boundary

class ElasticSteadyEquation(elastic_steady.ElasticSteadyEquation):
    __doc__ = "Generate dataset of the 2 components 2D elastic steady state equation with FEniCSx." + \
              elastic_steady.ElasticSteadyEquation.__doc__

    SOLVER: str = "fenicsx"
    VERSION: float = 2.1

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.n_sol_grid = args.n_sol_grid
        self.comm = MPI.COMM_WORLD
        # transform (x_min, x_max, y_min, y_max) to [(x_min, y_min), (x_max, y_max)]
        corner = [np.array([self.CORNERS[0], self.CORNERS[2]]),
                  np.array([self.CORNERS[1], self.CORNERS[3]])]
        self.domain = mesh.create_rectangle(
            self.comm, corner, [self.n_sol_grid, self.n_sol_grid],
            cell_type=dolfinx.cpp.mesh.CellType.triangle)
        # V is vector function space
        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (2, )))
        # W is scalar function space
        self.W = fem.functionspace(self.domain, ("Lagrange", 1))
        self.bd_name, self.bd_flag_dict, self.bd_facets_dict = \
            extract_square_boundary(self.domain, self.CORNERS)
        self.var_name = ["0", "1"]
        # points to interpolate the solution
        x_coord = self.coord_dict["x"]
        y_coord = self.coord_dict["y"]
        X, Y = np.meshgrid(x_coord, y_coord, indexing="ij")
        self.pts = np.stack([X.flatten(), Y.flatten(),
                             np.zeros_like(X.flatten())], axis=1)
        self.cells = locate_cells(self.domain, self.pts)

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = elastic_steady.ElasticSteadyEquation.get_cli_args_parser()
        parser.add_argument("--n_sol_grid", type=int, default=64,
                            help="resolution of finite element space for solution")
        return parser

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        prefix = elastic_steady.ElasticSteadyEquation.get_hdf5_file_prefix(args)
        prefix += f"_N{args.n_sol_grid}"
        return prefix

    def solve_static(self) -> NDArray[float]:
        r'''
        Solve the static equation to get the solution. Return the interpolated
        solution, scatter solution, and scatter points. Shape of returned arrays
        should be [n_x_grid, n_y_grid, 2], [n_nodes, 2], and [n_nodes, 2],
        respectively.
        '''
        x_coord = self.coord_dict["x"]  # shape [n_x_grid, ]
        y_coord = self.coord_dict["y"]  # shape [n_y_grid, ]
        f_field = self.term_obj_dict["f"].field  # shape [n_x_grid, n_y_grid, 2]

        f = fem.Function(self.V)
        f.interpolate(get_2d_interp_function(f_field, x_coord, y_coord))

        property_func_dict = self.get_property_func_dict()
        G = property_func_dict["G"]
        lamb = property_func_dict["lamb"]

        bc_dict = self.term_obj_dict["u_bc"].get_dict_rep(
            self.var_name, self.bd_name)
        bcs, T = self.set_bc(bc_dict)

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        dx = Measure("dx", domain=self.domain)
        ds = Measure("ds", domain=self.domain)
        sigma = self.sigma
        epsilon = self.epsilon

        a = inner(sigma(u, lamb, G), epsilon(v)) * dx
        L = inner(f, v) * dx + inner(T, v) * ds
        petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=petsc_options)
        uh = problem.solve()
        uh.x.scatter_forward()
        u = self._eval(uh)  # shape [n_x_grid, n_y_grid, 2]
        u_scat = uh.x.array[:].reshape(-1, 2)  # shape [n_nodes, 2]
        pts = self.domain.geometry.x[:, :2]  # shape [n_nodes, 2]
        return u, u_scat, pts

    def set_bc(self, bc_dict: Dict[str, Dict[str, List]]
               ) -> Tuple[List, fem.Function]:
        r"""
        Set boundary conditions for the function space.

        Args:
            bc_dict: Dictionary of boundary conditions. The format is
            {var_name: {boundary_name: [bc_type, bc_val]}}.

        Returns:
            bcs: DirichletBC list.
            T: fem.Function cooresponding to the Neumann boundary conditions.
        """
        DIRICHLET = 0
        NEUMANN = 1
        SUPPORTED_BC_TYPES = [DIRICHLET, NEUMANN]
        x_coord = self.coord_dict["x"]
        y_coord = self.coord_dict["y"]
        coord_map = {"left": (1, y_coord), "right": (1, y_coord),
                     "bottom": (0, x_coord), "top": (0, x_coord)}
        fdim = self.domain.topology.dim - 1

        # check bc_dict
        for var in self.var_name:
            if var not in bc_dict:
                raise ValueError(f"Variable {var} not found in bc_dict.")
            for bd in self.bd_name:
                if bd not in bc_dict[var]:
                    raise ValueError(f"Boundary {bd} not found in bc_dict[{var}].")
                if bc_dict[var][bd][0] not in SUPPORTED_BC_TYPES:
                    raise ValueError(f"Unsupported bc_type {bc_dict[var][bd][0]}.")

        # dirichlet bcs
        bcs = []
        for var_idx, var in enumerate(self.var_name):
            Vi = self.V.sub(var_idx).collapse()[0]
            var_bc_dict = bc_dict[var]
            def dirichlet_bc_val_func(x):
                out = np.zeros(x.shape[1])
                for bd in var_bc_dict:
                    if var_bc_dict[bd][0] != DIRICHLET:
                        continue
                    bd_flag = self.bd_flag_dict[bd]
                    ax, coord = coord_map[bd]
                    bd_val_func = get_1d_interp_function(
                        var_bc_dict[bd][1], coord)
                    out[bd_flag(x)] = bd_val_func(x[ax, bd_flag(x)])
                return out
            all_bd_dofs = [[], []]
            for bd in var_bc_dict:
                if var_bc_dict[bd][0] != DIRICHLET:
                    continue
                facets = self.bd_facets_dict[bd]
                # bd_dofs is [NDArray[int], NDArray[int]]
                bd_dofs = fem.locate_dofs_topological(
                    (self.V.sub(var_idx), Vi), fdim, facets)
                all_bd_dofs[0].append(bd_dofs[0])
                all_bd_dofs[1].append(bd_dofs[1])
            # get unique dofs
            all_bd_dofs = [np.unique(np.concatenate(dofs))
                           for dofs in all_bd_dofs]
            if len(all_bd_dofs[0]) != len(all_bd_dofs[1]):
                raise ValueError(f"The number of dofs for subspace and function "
                                 f"space should be the same, but got "
                                 f"{len(all_bd_dofs[0])} and "
                                 f"{len(all_bd_dofs[1])}.")
            u_bci = fem.Function(Vi)
            u_bci.interpolate(dirichlet_bc_val_func)
            bci = fem.dirichletbc(u_bci, all_bd_dofs, self.V.sub(var_idx))
            bcs.append(bci)

        # neumann bcs
        T = fem.Function(self.V)
        def neumann_bc_val_func(x):
            x_ = np.clip(x[:2].T, self.CORNERS[:2], self.CORNERS[2:])
            out = np.zeros((2, x_.shape[0]))
            for var_idx, var in enumerate(self.var_name):
                var_bc_dict = bc_dict[var]
                for bd in var_bc_dict:
                    if var_bc_dict[bd][0] != NEUMANN:
                        continue
                    bd_flag = self.bd_flag_dict[bd]
                    ax, coord = coord_map[bd]
                    bd_val_func = get_1d_interp_function(
                        var_bc_dict[bd][1], coord)
                    flg = bd_flag(x)
                    out[var_idx, flg] = bd_val_func(x[ax, flg])
            return out
        T.interpolate(neumann_bc_val_func)
        return bcs, T

    @staticmethod
    def epsilon(u: fem.Function) -> fem.Function:
        r"""
        Compute the strain tensor from the displacement field.
        """
        return sym(grad(u))

    @staticmethod
    def sigma(u: fem.Function,
              lamb: fem.Function,
              G: fem.Function) -> fem.Function:
        r"""
        Compute the stress tensor from the displacement field and the
        material properties.
        """
        epsilon = ElasticSteadyEquation.epsilon
        return lamb * nabla_div(u) * Identity(len(u)) + 2 * G * epsilon(u)

    def get_property_func_dict(self) -> Dict[str, fem.Function]:
        r"""
        Get the dictionary of material properties in the form of fem.Functions.
        """
        x_coord = self.coord_dict["x"]
        y_coord = self.coord_dict["y"]
        E_field = self.term_obj_dict["C/0"].field  # Young's modulus
        nu_field = self.term_obj_dict["C/1"].field  # Poisson's ratio
        G_field = E_field / (2 * (1 + nu_field))  # 1st Lame parameter
        lamb_field = 2 * nu_field * G_field / (1 - nu_field)  # 2nd Lame parameter
        G = fem.Function(self.W)
        G.interpolate(get_2d_interp_function(G_field, x_coord, y_coord))
        lamb = fem.Function(self.W)
        lamb.interpolate(get_2d_interp_function(lamb_field, x_coord, y_coord))
        return {"G": G, "lamb": lamb}

    def _eval(self, f: fem.Function) -> NDArray[float]:
        r"""
        Evaluate the function f at the grid points.
        """
        return f.eval(self.pts, self.cells).reshape(self.n_x_grid, self.n_y_grid, -1)


if __name__ == "__main__":
    my_args = basics.get_cli_args(ElasticSteadyEquation)
    pde_data_obj = ElasticSteadyEquation(my_args)
    basics.gen_data(my_args, pde_data_obj)
