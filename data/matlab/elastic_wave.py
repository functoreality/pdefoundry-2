#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of 2D elastic wave equation."""
import argparse
from typing import Tuple
import os
import numpy as np
from numpy.typing import NDArray
import matlab.engine
from data.common import basics
from data.common.forward_main import elastic_wave


class ElasticWaveEquation(elastic_wave.ElasticWaveEquation):
    __doc__ = "Generate dataset of the 2 components 2D elastic wave equation with Matlab." + \
              elastic_wave.ElasticWaveEquation.__doc__

    SOLVER: str = "matlab"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.max_thread = args.max_thread
        self.hmax = args.hmax  # maximum grid size
        self.eng = matlab.engine.start_matlab()
        # add directory of this file to the matlab path
        self.eng.addpath(os.path.dirname(os.path.abspath(__file__)))
        self.eng.maxNumCompThreads(args.max_thread)

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = elastic_wave.ElasticWaveEquation.get_cli_args_parser()
        parser.add_argument("--hmax", type=float, default=0.05,
                            help="maximum grid size")
        parser.add_argument("--max_thread", type=int, default=1,
                            help="maximum number of threads for matlab")
        return parser

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        prefix = elastic_wave.ElasticWaveEquation.get_hdf5_file_prefix(args)
        prefix += f"_hmax{args.hmax:g}"
        return prefix

    def get_static_args(self) -> Tuple:
        r"""
        Get the input arguments for static solver (matlab engine).
        """
        domain_bounds = matlab.double(list(self.CORNERS))
        x_grid = self.coord_dict["x"]
        y_grid = self.coord_dict["y"]
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        Hmax = float(self.hmax)

        if self.motion_type == self.BC_CHANGE:
            boundary_conditions = self.term_obj_dict["u_ibc"].list_rep
        else:
            boundary_conditions = self.term_obj_dict["u_bc"].list_rep
        # use bc_type and bc_val only
        bc_list = []
        for bc in boundary_conditions:
            bc_type = matlab.int64(bc[0])
            bc_val = matlab.double(np.ascontiguousarray(bc[1]))
            bc_list.append([bc_type, bc_val])
        boundary_conditions = bc_list

        if self.motion_type == self.IC_CHANGE:
            external_force = self.term_obj_dict["f_ic"].field
        else:
            external_force = self.term_obj_dict["f"].field
        external_force = matlab.double(external_force)

        if self.type == self.ISOTROPY:
            # shape is [n_x_grid, n_y_grid, 3]
            # Young's modulus, Poisson's ratio, density
            material_properties = np.stack([self.term_obj_dict["C/0"].field,
                                            self.term_obj_dict["C/1"].field,
                                            self.term_obj_dict["rho"].field], axis=2)
        else:
            raise NotImplementedError
        material_properties = matlab.double(material_properties)
        X = matlab.double(X)
        Y = matlab.double(Y)

        input_args = (domain_bounds, X, Y, Hmax, boundary_conditions,
                      external_force, material_properties)
        return input_args

    def solve_static(self) -> NDArray[float]:
        r'''
        Solve the static equation to get the initial condition. Shape of
        returned array is [n_x_grid, n_y_grid, 2].
        '''
        input_args = self.get_static_args()
        ux, uy = self.eng.static(*input_args, nargout=2)
        # shape is [n_x_grid, n_y_grid, 2]
        ic = np.stack([np.array(ux), np.array(uy)], axis=2)
        return np.ascontiguousarray(ic)

    def get_dynamic_args(self) -> Tuple:
        r"""
        Get the input arguments for dynamic solver (matlab engine).
        """
        domain_bounds = matlab.double(list(self.CORNERS))
        x_grid = self.coord_dict["x"]
        y_grid = self.coord_dict["y"]
        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        Hmax = float(self.hmax)

        boundary_conditions = self.term_obj_dict["u_bc"].list_rep
        # use bc_type and bc_val only
        bc_list = []
        for bc in boundary_conditions:
            bc_type = matlab.int64(bc[0])
            bc_val = matlab.double(np.ascontiguousarray(bc[1]))
            bc_list.append([bc_type, bc_val])
        boundary_conditions = bc_list
        t_coord = np.linspace(0, 1, self.n_t_grid)

        external_force = []
        # time-independent external force
        external_force.append(matlab.double(self.term_obj_dict["f"].field))
        # time-dependent external force
        if self.motion_type == self.TIMEDEP_FORCE:
            fr = self.term_obj_dict["f_rt"].fr_field
            ft = self.term_obj_dict["f_rt"].ft_field
        else:
            fr = np.zeros((self.n_x_grid, self.n_y_grid, 2))
            ft = np.zeros(self.n_t_grid)

        external_force.append(matlab.double(np.ascontiguousarray(fr)))
        external_force.append(matlab.double(np.ascontiguousarray(ft)))

        if self.type == self.ISOTROPY:
            # shape is [n_x_grid, n_y_grid, 3]
            # Young's modulus, Poisson's ratio, density
            material_properties = np.stack([self.term_obj_dict["C/0"].field,
                                            self.term_obj_dict["C/1"].field,
                                            self.term_obj_dict["rho"].field], axis=2)
        else:
            raise NotImplementedError
        material_properties = matlab.double(material_properties)
        u0 = self.ic
        if self.motion_type == self.RANDOM_IVEL:
            v0 = self.term_obj_dict["ut_ic"].field
        else:
            v0 = np.zeros_like(u0)
        u0 = matlab.double(u0)
        v0 = matlab.double(v0)
        X = matlab.double(X)
        Y = matlab.double(Y)
        t_coord = matlab.double(t_coord)

        input_args = (domain_bounds, X, Y, t_coord, Hmax, boundary_conditions,
                      external_force, material_properties, u0, v0)
        return input_args

    def solve_dynamic(self) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
        r'''
        Solve the dynamic equation to get the solution. Return the interpolated
        solution, scatter solution, and scatter points. Shape of returned arrays
        should be [n_x_grid, n_y_grid, n_t_grid, 2], [n_nodes, n_t_grid, 2], and
        [n_nodes, 2], respectively.
        '''
        input_args = self.get_dynamic_args()
        ux, uy, ux_scat, uy_scat, pts = self.eng.dynamic(*input_args, nargout=5)
        # shape [n_x_grid, n_y_grid, n_t_grid, 1] ->
        # shape [n_x_grid, n_y_grid, n_t_grid, 2]
        u = np.stack([np.array(ux), np.array(uy)], axis=3)
        # shape [n_nodes, n_t_grid, 1] -> shape [n_nodes, n_t_grid, 2]
        u_scat = np.stack([np.array(ux_scat), np.array(uy_scat)], axis=2)
        # shape [n_nodes, 2]
        pts = np.array(pts)
        return u, u_scat, pts


if __name__ == "__main__":
    my_args = basics.get_cli_args(ElasticWaveEquation)
    pde_data_obj = ElasticWaveEquation(my_args)
    basics.gen_data(my_args, pde_data_obj)
