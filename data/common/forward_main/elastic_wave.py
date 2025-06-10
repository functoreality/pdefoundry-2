#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of 2D elastic wave equation."""
import argparse
import os
from datetime import datetime
from typing import Callable, Dict, List, Tuple, Optional, Union
from abc import abstractmethod
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data.common import basics, coefs
from data.common.utils_random import RandomValueSampler, GaussianRandomFieldSampler


class RandomBoundaryCondition(basics.PDETermBase):
    r"""
    Non-periodic boundary condition of a PDE, which is taken to be Dirichlet,
    Neumann with equal probability. The value of boundary condition can be zero,
    a real number(scalar) or a spatially-varying function.
    """
    DIRICHLET = 0
    NEUMANN = 1
    bc_type: int

    ZERO_VAL = 0
    UNIT_VAL = 1
    SCALAR_VAL = 2
    FIELD_VAL = 3
    bc_val_type: int
    bc_val: NDArray[float]
    const_sampler: RandomValueSampler
    field_sampler: GaussianRandomFieldSampler
    add_cli_args_ = RandomValueSampler.add_cli_args_

    def __init__(self,
                 resolution: Union[int, Tuple[int]],
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        if isinstance(resolution, int):
            resolution = (resolution,)
        self.resolution = resolution
        self.const_sampler = RandomValueSampler(coef_distribution, coef_magnitude)
        self.field_sampler = GaussianRandomFieldSampler(
            len(resolution), resolution, alpha=3, tau=5, sigma=50)

    def __str__(self) -> str:
        data = self.bc_val
        shape = self.resolution
        if self.bc_type == self.DIRICHLET:
            type_str = "D"
        elif self.bc_type == self.NEUMANN:
            type_str = "N"
        else:
            raise NotImplementedError
        val_str = f"shape {shape}, min {data.min():.4g}, max {data.max():.4g}"
        return f"({type_str}) {val_str}"

    def reset(self,
              rng: np.random.Generator,
              bc_type: Optional[int] = None,
              bc_val_type: Optional[int] = None,
              endpt_vals: Optional[Tuple[float, float]] = None,
              ) -> None:
        r"""
        Args:
            rng (np.random.Generator): Random number generator instance.
            bc_type (Optional[int]): Type of the boundary condition. Choices:
                0 (Dirichlet), 1 (Neumann)
            bc_val_type (Optional[int]): Type of the boundary value. Choices:
                0 (zero), 1 (unit), 2 (scalar), 3 (field)
            endpt_vals (Optional[Tuple[float, float]]): Values at the endpoints,
                only available when resolution is 1D.
        """
        if bc_type is None:
            self.bc_type = rng.choice(2)
        else:
            self.bc_type = bc_type

        if self.bc_type == self.DIRICHLET:
            self.u_coef = 1
            self.dn_coef = 0
        elif self.bc_type == self.NEUMANN:
            self.u_coef = 0
            self.dn_coef = 1

        if bc_val_type is None:
            zero_prob = 1.
            unit_prob = 0.2
            scalar_prob = 1.
            field_prob = 1.
            probs = np.array([zero_prob, unit_prob, scalar_prob, field_prob])
            self.bc_val_type = rng.choice(4, p=probs/probs.sum())
        else:
            self.bc_val_type = bc_val_type

        if self.bc_val_type == self.ZERO_VAL:
            self.bc_val = np.zeros(self.resolution)
        elif self.bc_val_type == self.UNIT_VAL:
            self.bc_val = np.ones(self.resolution)
        elif self.bc_val_type == self.SCALAR_VAL:
            value = self.const_sampler(rng)
            self.bc_val = np.full(self.resolution, value)
        elif self.bc_val_type == self.FIELD_VAL:
            self.bc_val = self.field_sampler(rng).reshape(self.resolution)

        if endpt_vals is not None:
            if len(self.resolution) != 1:
                raise ValueError("endpt_vals is only available when resolution is 1D.")
            # add linear function to bc_val to match the endpt_vals
            left_val = endpt_vals[0] - self.bc_val[0]
            right_val = endpt_vals[1] - self.bc_val[-1]
            self.bc_val += np.linspace(left_val, right_val, self.resolution[0])

    def reset_debug(self) -> None:
        self.bc_type = self.DIRICHLET
        self.u_coef = 1
        self.dn_coef = 0

        self.bc_val_type = self.ZERO_VAL
        self.bc_val = np.zeros(self.resolution)

    def gen_dedalus_ops(self):
        raise NotImplementedError

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {f"{prefix}/bc_type": self.bc_type,
                f"{prefix}/coef_type": self.bc_val_type,
                f"{prefix}/field": self.bc_val}

    def prepare_plot(self, title: str = "") -> None:
        raise NotImplementedError


class RectBoundaryCondition(basics.PDETermBase):
    r"""
    Boundary condition for a PDE on a rectangular domain. Boundary condition on
    each side for each component is randomly chosen from Dirichlet and Neumann.
    So the total number of boundary conditions is 4*n_vars.
    """
    n_vars: int
    n_x_grid: int
    n_y_grid: int
    bcs: List[List[RandomBoundaryCondition]]
    bc_types: List[List[int]]
    bc_val_types: List[List[int]]
    const_sampler: RandomValueSampler
    add_cli_args_ = RandomBoundaryCondition.add_cli_args_

    DIRICHLET = 0
    NEUMANN = 1

    ZERO_VAL = 0
    UNIT_VAL = 1
    SCALAR_VAL = 2
    FIELD_VAL = 3

    def __init__(self,
                 n_vars: int = 2,
                 n_x_grid: int = 128,
                 n_y_grid: int = 128,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.n_x_grid = n_x_grid
        self.n_y_grid = n_y_grid
        self.const_sampler = RandomValueSampler(coef_distribution, coef_magnitude)
        self.n_sides = 4
        self.res = [n_y_grid, n_y_grid, n_x_grid, n_x_grid]
        self.bcs = [[] for _ in range(self.n_vars)]
        rbc = RandomBoundaryCondition
        for i in range(n_vars):
            for j in range(self.n_sides):
                # left, right, bottom, top
                self.bcs[i].append(rbc((self.res[j],),
                                   coef_distribution, coef_magnitude))

    def reset(self,
              rng: np.random.Generator,
              bc_type: Optional[Union[int, List[List[int]]]] = None,
              bc_val_type: Optional[Union[int, List[List[int]]]] = None) -> None:
        if bc_type is None:
            # random generate types for each side, at least one Dirichlet
            bc_types = [rng.choice([self.DIRICHLET, self.NEUMANN], self.n_sides)
                        for _ in range(self.n_vars)]
            for i in range(self.n_vars):
                if self.DIRICHLET not in bc_types[i]:
                    bc_types[i][rng.choice(self.n_sides)] = self.DIRICHLET
        elif isinstance(bc_type, int):
            if bc_type == self.NEUMANN:
                # all Neumann boundary conditions are not supported now
                raise NotImplementedError
            bc_types = [[bc_type] * self.n_sides for _ in range(self.n_vars)]
        else:
            if len(bc_type) != self.n_vars:
                raise ValueError(f"bc_type should have length n_vars={self.n_vars}.")
            for i in range(self.n_vars):
                if len(bc_type[i]) != self.n_sides:
                    raise ValueError(f"each element in bc_type should have length"
                                     f" n_sides={self.n_sides}.")
            bc_types = bc_type

        # additional criterion
        if sum(bc_types[0] == self.DIRICHLET) == 1 and sum(bc_types[1] == self.DIRICHLET) == 1:
            idx1 = np.where(bc_types[0] == self.DIRICHLET)[0][0]
            idx2 = np.where(bc_types[1] == self.DIRICHLET)[0][0]
            if idx1 in [2, 3] and idx2 in [0, 1]:
                # random choose another side to be Dirichlet
                idx_comp = rng.choice([0, 1])
                if idx_comp == 0:
                    idx = rng.choice([0, 1])
                else:
                    idx = rng.choice([2, 3])
                bc_types[idx_comp][idx] = self.DIRICHLET

        if bc_val_type is None:
            bc_val_types = [rng.choice(4, self.n_sides) for _ in range(self.n_vars)]
        elif isinstance(bc_val_type, int):
            bc_val_types = [[bc_val_type] * self.n_sides for _ in range(self.n_vars)]
        else:
            if len(bc_val_type) != self.n_vars:
                raise ValueError(f"bc_val_type should have length n_vars={self.n_vars}.")
            for i in range(self.n_vars):
                if len(bc_val_type[i]) != self.n_sides:
                    raise ValueError(f"each element in bc_val_type should have length"
                                     f" n_sides={self.n_sides}.")
            bc_val_types = bc_val_type

        for i, bc_list in enumerate(self.bcs):

            # generate corner values for dirichlet boundary conditions
            corner_vals = [self.const_sampler(rng) for _ in range(self.n_sides)]
            for j, bc in enumerate(bc_list):
                if bc_types[i][j] == self.DIRICHLET:
                    endpt_vals = corner_vals[i], corner_vals[(i+1) % self.n_sides]
                else:
                    endpt_vals = None
                bc.reset(rng, bc_types[i][j], bc_val_types[i][j], endpt_vals)

        self.bc_types = bc_types
        self.bc_val_types = bc_val_types

    def reset_debug(self) -> None:
        for bc_list in self.bcs:
            for bc in bc_list:
                bc.reset_debug()

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        bd_names = ["left", "right", "bottom", "top"]
        data_dict = {}
        for i, bc_list in enumerate(self.bcs):
            for j, bc in enumerate(bc_list):
                data_dict.update(bc.get_data_dict(f"{prefix}/bd_{bd_names[j]}/{i}"))
        return data_dict

    # def prepare_plot(self, title: str = "") -> None:
    #     raise NotImplementedError

    def _get_list_rep(self) -> List:
        r"""Get the list representation of all n_vars * n_sides bcs, each is a
        list [type, val]."""
        list_rep = []
        for bc_list in self.bcs:
            for bc in bc_list:
                list_rep.append([bc.bc_type, bc.bc_val])
        return list_rep

    @property
    def list_rep(self) -> List:
        r"""List representation of all n_vars * n_sides bcs."""
        return self._get_list_rep()

    def get_dict_rep(self, var_name: List[str], bd_name: List[str]) -> Dict:
        r"""
        Get the dictionary representation of all n_vars * n_sides bcs.

        Args:
            var_name: List of variable names.
            bd_name: List of boundary names.

        Returns:
            dict_rep: Dictionary representation of all n_vars * n_sides bcs.
            Format: {var_name[i]: {bd_name[j]: [type, val]}}
        """
        if len(var_name) != self.n_vars:
            raise ValueError(f"var_name should have length {self.n_vars}.")
        if len(bd_name) != self.n_sides:
            raise ValueError(f"bd_name should have length {self.n_sides}.")
        dict_rep = {}
        for idx_var, var in enumerate(var_name):
            var_dict = {}
            for idx_bd, bd in enumerate(bd_name):
                idx = idx_var * self.n_sides + idx_bd
                var_dict[bd] = self.list_rep[idx]
            dict_rep[var] = var_dict
        return dict_rep


class TimeDepForce(basics.PDETermBase):
    r"""
    Time-dependent external force term. The term can be factorized into spatial
    and temporal components $fr_i(r)$ and $ft_i(t)$.
    """
    n_x_grid: int
    n_y_grid: int
    corners: Tuple[int, int, int, int]
    t_coord: NDArray[float]
    DEFAULT_SCALES = [1000., 100., 1000.]  # default scales for different types of forces

    def __init__(self,
                 n_x_grid: int = 128,
                 n_y_grid: int = 128,
                 corners: Tuple[int, int, int, int] = (0, 1, 0, 1),
                 t_coord: Optional[NDArray[float]] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.
                 ) -> None:
        super().__init__()
        self.n_x_grid = n_x_grid
        self.n_y_grid = n_y_grid
        self.corners = corners
        if t_coord is None:
            t_coord = np.linspace(0, 1, 100)
        self.t_coord = t_coord
        self.time_term = TimeFunc(t_coord)
        self.space_term = TimeIndepForce(n_x_grid,
                                         n_y_grid,
                                         corners,
                                         coef_distribution,
                                         coef_magnitude)

    def reset(self,
              rng: np.random.Generator,
              rho: NDArray[float],
              n_forces: int = 1,
              n_types: int = 1,
              types: Optional[List[int]] = None,
              scales: Optional[List[float]] = None) -> None:
        self.time_term.reset(rng)
        if scales is None:
            scales = self.DEFAULT_SCALES
        self.space_term.reset(rng, rho, n_forces, n_types, types, scales)

    def reset_debug(self) -> None:
        self.time_term.reset_debug()
        self.space_term.reset_debug()

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        data_dict = {}
        data_dict.update(self.time_term.get_data_dict(f"{prefix}/ft"))
        data_dict.update(self.space_term.get_data_dict(f"{prefix}/fr"))
        return data_dict

    # def prepare_plot(self, title: str = "") -> None:
    #     raise NotImplementedError

    @property
    def fr_field(self) -> NDArray[float]:
        r"""Spatial component field of the external force."""
        return self.space_term.field

    @property
    def ft_field(self) -> NDArray[float]:
        r"""Temporal component field of the external force."""
        return self.time_term.field


class TimeFunc(basics.PDETermBase):
    r"""
    Temporal component of the external force $ft_i(t)$. The function is now set
    to be a Ricker wavelet.
    """
    # TODO: add more kinds of time functions
    freq: float  # frequency of the Ricker wavelet
    delay: float  # delay of the Ricker wavelet

    def __init__(self,
                 t_coord: Optional[NDArray[float]] = None,
                 ) -> None:
        super().__init__()
        if t_coord is None:
            t_coord = np.linspace(0, 1, 100)  # default time list
        self.t_coord = t_coord

    def reset(self, rng: np.random.Generator) -> None:
        self.freq = rng.uniform(10., 30.)
        self.delay = rng.uniform(0., 0.2)

    def reset_debug(self) -> None:
        self.freq = 30.
        self.delay = 0.

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {f"{prefix}/field": self.field.copy()}

    # def prepare_plot(self, title: str = "") -> None:
    #     raise NotImplementedError

    def _get_func(self) -> Callable[[NDArray[float]], NDArray[float]]:
        r"""Get the temporal component function of the external force."""
        def func(t: NDArray[float]) -> NDArray[float]:
            r"""ricker wavelet."""
            t = t - self.delay
            val = (1. - 2. * (np.pi*self.freq*t)**2) \
                * np.exp(-(np.pi*self.freq*t)**2)
            return val
        return func

    def _get_field(self) -> NDArray[float]:
        r"""Get the temporal component field."""
        return self.func(self.t_coord)

    @property
    def field(self) -> NDArray[float]:
        r"""Temporal component field."""
        return self._get_field()

    @property
    def func(self) -> Callable[[NDArray[float]], NDArray[float]]:
        r"""Temporal component function."""
        return self._get_func()


class TimeIndepForce(basics.PDETermBase):
    r"""Time-independent external force term $f_i(r)$. The term is set to be a
    combination of MultiRadialForce, FieldForce, and MultiParallelForce.
    """
    n_x_grid: int
    n_y_grid: int
    corners: Tuple[int, int, int, int]
    n_forces: int
    types: List[int]

    MULTIRAIDALFORCE = 0
    FIELDFORCE = 1
    MULTIPARALLELFORCE = 2
    SUPPORTED_TYPES = [MULTIRAIDALFORCE, FIELDFORCE, MULTIPARALLELFORCE]
    DEFAULT_SCALES = [100., 10., 100.]

    def __init__(self,
                 n_x_grid: int = 128,
                 n_y_grid: int = 128,
                 corners: Tuple[int, int, int, int] = (0, 1, 0, 1),
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        self.n_x_grid = n_x_grid
        self.n_y_grid = n_y_grid
        self.corners = corners
        term1 = MultiRadialForce(n_x_grid, n_y_grid, corners, coef_distribution, coef_magnitude)
        term2 = FieldForce(n_x_grid, n_y_grid, corners, coef_distribution, coef_magnitude)
        term3 = MultiParallelForce(n_x_grid, n_y_grid, corners, coef_distribution, coef_magnitude)
        self.terms = [term1, term2, term3]
        self._field = None
        self.reset_debug()

    def reset(self,
              rng: np.random.Generator,
              rho: NDArray[float],
              n_forces: int = 1,
              n_types: int = 1,
              types: Optional[List[int]] = None,
              scales: Optional[List[float]] = None) -> None:
        self.n_forces = n_forces

        self._field = None
        if n_types < 1:
            raise ValueError("n_types should be greater than 0.")
        if n_types > len(self.SUPPORTED_TYPES):
            raise ValueError(
                f"n_types should be less than or equal to {len(self.SUPPORTED_TYPES)}.")
        if types is None:
            types = rng.choice(self.SUPPORTED_TYPES, n_types, replace=False)
        if scales is None:
            scales = self.DEFAULT_SCALES
        if len(scales) != len(self.SUPPORTED_TYPES):
            raise ValueError(f"len(scales) should be equal to {len(self.SUPPORTED_TYPES)}.")
        for i, term in enumerate(self.terms):
            if i in types:
                if i == self.MULTIRAIDALFORCE:
                    term.reset(rng, n_forces=n_forces, scale=scales[i])
                elif i == self.FIELDFORCE:
                    term.reset(rng, rho=rho, scale=scales[i])
                elif i == self.MULTIPARALLELFORCE:
                    term.reset(rng, n_forces=n_forces, scale=scales[i])
            if i not in types:
                term.reset_debug()

        self.types = types

    def reset_debug(self) -> None:
        self.n_forces = 1
        self._field = None
        self.types = [0]
        for term in self.terms:
            term.reset_debug()

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        # only return the field
        return {f"{prefix}/field": self.field.copy()}

    def _get_field(self) -> NDArray[float]:
        r"""Get the force field."""
        _field = np.zeros((self.n_x_grid, self.n_y_grid, 2))
        for i, term in enumerate(self.terms):
            if i in self.types:
                _field += term.field
        return _field

    @property
    def field(self) -> NDArray[float]:
        r"""Force field."""
        if self._field is None:
            self._field = self._get_field()
        return self._field


class MultiRadialForce(basics.PDETermBase):
    r"""Radially outward force field."""
    n_x_grid: int
    n_y_grid: int
    const_sampler: RandomValueSampler
    n_forces: int
    src_pts: NDArray[float]  # coordinates of the force points
    sigmas: NDArray[float]  # sizes (standard deviations) of the gaussian distributions
    amps: NDArray[float]  # amplitudes or normalizing constants for gaussian distributions

    def __init__(self,
                 n_x_grid: int = 128,
                 n_y_grid: int = 128,
                 corners: Tuple[int, int, int, int] = (0, 1, 0, 1),
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        self.n_x_grid = n_x_grid
        self.n_y_grid = n_y_grid
        self.corners = corners
        self.const_sampler = RandomValueSampler(coef_distribution, coef_magnitude)
        self._field = None

    def reset(self,
              rng: np.random.Generator,
              n_forces: int = 1,
              scale: float = 100.) -> None:
        self._field = None
        self.n_forces = n_forces
        corners = self.corners
        if n_forces < 1:
            raise ValueError("n_forces should be greater than 0.")
        # random source points in the domain
        self.src_pts = np.array([[rng.uniform(corners[0], corners[1]),
                                  rng.uniform(corners[2], corners[3])]
                                 for _ in range(n_forces)])
        self.sigmas = np.array([rng.uniform(0.01, 0.03) for _ in range(n_forces)])
        self.amps = 1 / (2*np.pi*self.sigmas) * scale * \
            np.power(10, rng.uniform(-0.25, 0.25, size=n_forces))

    def reset_debug(self) -> None:
        self._field = None
        self.n_forces = 1
        self.src_pts = np.array([[0.5, 0.5]])
        self.sigmas = np.array([0.02])
        self.amps = 1 / (2*np.pi*self.sigmas) * 100.

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {f"{prefix}/src_pts": self.src_pts,
                f"{prefix}/sigmas": self.sigmas,
                f"{prefix}/amps": self.amps,
                f"{prefix}/field": self.field.copy()}

    # def prepare_plot(self, title: str = "") -> None:
    #     raise NotImplementedError

    def _get_func(self) -> Callable[[NDArray[float]], NDArray[float]]:
        r"""Get the force field function."""
        def func(r: NDArray[float]) -> NDArray[float]:
            r"""gaussian function."""
            # shape of r: [n_pts, 2]
            if r.ndim != 2 or r.shape[1] != 2:
                raise ValueError("r should have shape [n_pts, 2].")
            val = np.zeros_like(r)
            for i in range(self.n_forces):
                rx = r[:, 0] - self.src_pts[i, 0]
                ry = r[:, 1] - self.src_pts[i, 1]
                # turn (rx, ry) into unit vector
                r_norm = np.sqrt(rx**2 + ry**2)
                unit_rx = rx / (1e-8 + r_norm)
                unit_ry = ry / (1e-8 + r_norm)
                w = self.amps[i] * np.exp(- (rx**2 + ry**2) / (2*self.sigmas[i]**2))
                # shape of val: [n_pts, 2]
                val += np.stack([w * unit_rx, w * unit_ry], axis=1)
            return val
        return func

    def _get_field(self) -> NDArray[float]:
        r"""Get the force field."""
        corners = self.corners
        X = np.linspace(corners[0], corners[1], self.n_x_grid)
        Y = np.linspace(corners[2], corners[3], self.n_y_grid)
        XX, YY = np.meshgrid(X, Y, indexing='ij')
        r = np.stack([XX.flatten(), YY.flatten()], axis=1)
        return self.func(r).reshape(self.n_x_grid, self.n_y_grid, 2)

    @property
    def func(self) -> Callable[[NDArray[float]], NDArray[float]]:
        r"""Force field function."""
        return self._get_func()

    @property
    def field(self) -> NDArray[float]:
        r"""Force field."""
        if self._field is None:
            self._field = self._get_field()
        return self._field


class FieldForce(basics.PDETermBase):
    r"""
    Field force term. $f_i(r) = a_i\rho(r) + b_i$.
    Here, $a$ and $b$ are randomly chosen from zero, random constant, and random
    2D functions.
    """
    n_x_grid: int
    n_y_grid: int
    _coefs: List[coefs.RandomConstOrField]  # [a0, a1, b0, b1]
    _rho: NDArray[float]  # [n_x_grid, n_y_grid]

    def __init__(self,
                 n_x_grid: int = 128,
                 n_y_grid: int = 128,
                 corners: Tuple[int, int, int, int] = (0, 1, 0, 1),
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        self.n_x_grid = n_x_grid
        self.n_y_grid = n_y_grid
        resolution = (n_x_grid, n_y_grid)
        x_coord = np.linspace(corners[0], corners[1], n_x_grid)
        y_coord = np.linspace(corners[2], corners[3], n_y_grid)
        coords = (x_coord.reshape(-1, 1), y_coord.reshape(1, -1))
        self._coefs = [coefs.RandomConstOrField(
            coords=coords, periodic=False, resolution=resolution,
            coef_distribution=coef_distribution, coef_magnitude=coef_magnitude)
            for _ in range(4)]
        self._field = None
        self._scale = 10.
        self.reset_debug()

    def reset(self, rng: np.random.Generator,
              rho: NDArray[float],
              scale: Optional[float] = 10.) -> None:
        self._field = None
        self._rho = rho
        self._scale = scale
        probs = np.array([1., 0.2, 1., 1.])
        coef_types = rng.choice(4, 2, p=probs/probs.sum())  # different for a and b
        one_hot_probs = np.eye(4)[coef_types]
        for i, coef in enumerate(self._coefs):
            coef.reset(rng=rng, zero_prob=one_hot_probs[i//2, 0],
                       unit_prob=one_hot_probs[i//2, 1], scalar_prob=one_hot_probs[i//2, 2],
                       field_prob=one_hot_probs[i//2, 3])  # a0, a1, b0, b1

    def reset_debug(self) -> None:
        self._field = None
        self._rho = np.ones((self.n_x_grid, self.n_y_grid))
        self._scale = 10.
        for coef in self._coefs:
            coef.reset_debug()

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {prefix + "/field": self.field.copy(),
                prefix + "/coef_types": np.array([coef.coef_type for coef in self._coefs])}

    # def prepare_plot(self, title: str = "") -> None:
    #     raise NotImplementedError

    def _get_field(self) -> NDArray[float]:  # [n_x_grid, n_y_grid, 2]
        r"""Get the force field."""
        a0, a1, b0, b1 = [coef.field for coef in self._coefs]
        field = np.stack([a0*self._rho + b0, a1*self._rho + b1], axis=2)
        return field * self._scale

    @property
    def field(self) -> NDArray[float]:
        r"""Force field."""
        if self._field is None:
            self._field = self._get_field()
        return self._field


class MultiParallelForce(basics.PDETermBase):
    r"""
    Multiple point parallel external forces. Approximate the point forces by gaussian
    distributions.
    """
    n_x_grid: int
    n_y_grid: int
    const_sampler: RandomValueSampler
    n_forces: int
    pts: NDArray[float]  # coordinates of the force points
    sigmas: NDArray[float]  # sizes (standard deviations) of the gaussian distributions
    amps: NDArray[float]  # amplitudes or normalizing constants for gaussian distributions
    force_vecs: NDArray[float]  # force vectors
    # field: NDArray[float]  # [n_x_grid, n_y_grid, 2]

    def __init__(self,
                 n_x_grid: int = 128,
                 n_y_grid: int = 128,
                 corners: Tuple[int, int, int, int] = (0, 1, 0, 1),
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        self.n_x_grid = n_x_grid
        self.n_y_grid = n_y_grid
        self.corners = corners
        # use a larger magnitude
        self.const_sampler = RandomValueSampler(coef_distribution, 25*coef_magnitude)
        self._field = None

    def reset(self,
              rng: np.random.Generator,
              n_forces: int = 1,
              scale: float = 100.) -> None:
        self.n_forces = n_forces
        corners = self.corners
        if n_forces < 1:
            raise ValueError("n_force should be greater than 0.")
        # random source points in the domain
        self.pts = np.array([[rng.uniform(corners[0], corners[1]),
                              rng.uniform(corners[2], corners[3])]
                             for _ in range(n_forces)])
        self.sigmas = np.array([rng.uniform(0.01, 0.03) for _ in range(n_forces)])
        self.amps = 1 / (2*np.pi*self.sigmas) * scale * \
            np.power(10, rng.uniform(-0.25, 0.25, size=n_forces))

        def sample_force():
            theta = rng.uniform(0, 2*np.pi)
            force = np.array([np.cos(theta), np.sin(theta)])
            return force
        self.force_vecs = np.array([sample_force() for _ in range(n_forces)])
        # # generate the force field at grid points
        # X = np.linspace(corners[0], corners[1], self.n_x_grid)
        # Y = np.linspace(corners[2], corners[3], self.n_y_grid)
        # XX, YY = np.meshgrid(X, Y, indexing='ij')
        # r = np.stack([XX.flatten(), YY.flatten()], axis=1)
        # self.field = self.force_func(r).reshape(self.n_x_grid, self.n_y_grid, 2)

    def reset_debug(self) -> None:
        self.n_forces = 1
        self.pts = np.array([[0.5, 0.5]])
        self.sigmas = np.array([0.02])
        self.amps = 1 / (2*np.pi*self.sigmas) * 100.
        self.force_vecs = np.array([[1., 0.]])
        # # generate the force field at grid points
        # corners = self.corners
        # X = np.linspace(corners[0], corners[1], self.n_x_grid)
        # Y = np.linspace(corners[2], corners[3], self.n_y_grid)
        # XX, YY = np.meshgrid(X, Y, indexing='ij')
        # r = np.stack([XX.flatten(), YY.flatten()], axis=1)
        # self.field = self.force_func(r).reshape(self.n_x_grid, self.n_y_grid, 2)

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {f"{prefix}/pts": self.pts,
                f"{prefix}/sigmas": self.sigmas,
                f"{prefix}/amps": self.amps,
                f"{prefix}/force_vecs": self.force_vecs,
                f"{prefix}/field": self.field.copy()}
        # return {f"{prefix}/field": self.field}

    # def prepare_plot(self, title: str = "") -> None:
    #     raise NotImplementedError

    def _get_func(self) -> Callable[[NDArray[float]], NDArray[float]]:
        r"""Get the force field function."""
        def func(r: NDArray[float]) -> NDArray[float]:
            r"""gaussian function."""
            # shape of r: [n_pts, 2]
            if r.ndim != 2 or r.shape[1] != 2:
                raise ValueError("r should have shape [n_pts, 2].")
            val = np.zeros_like(r)
            for i in range(self.n_forces):
                rx = r[:, 0] - self.pts[i, 0]
                ry = r[:, 1] - self.pts[i, 1]
                w = self.amps[i] * np.exp(- (rx**2 + ry**2) / (2*self.sigmas[i]**2))
                val += np.outer(w, self.force_vecs[i])
            return val
        return func

    def _get_field(self) -> NDArray[float]:
        r"""Get the force field."""
        corners = self.corners
        X = np.linspace(corners[0], corners[1], self.n_x_grid)
        Y = np.linspace(corners[2], corners[3], self.n_y_grid)
        XX, YY = np.meshgrid(X, Y, indexing='ij')
        r = np.stack([XX.flatten(), YY.flatten()], axis=1)
        return self.func(r).reshape(self.n_x_grid, self.n_y_grid, 2)

    @property
    def func(self) -> Callable[[NDArray[float]], NDArray[float]]:
        r"""Force field function."""
        return self._get_func()

    @property
    def field(self) -> NDArray[float]:
        r"""Force field."""
        if self._field is None:
            self._field = self._get_field()
        return self._field


class ElasticWaveEquation(basics.PDETypeBase):
    r"""
    ======== Elastic Wave Equation ========
    The PDE takes the form
        $\rho(r)\partial_{tt}u_i-\sigma_{ji,j}-fr_i(r)ft_i(t)-f_i(r)=0,$
    $u_i(0,r)=g_i(r)$, $\partial_t u_i(0,r)=h_i(r)$, $t\in[0,1]$,
    $r=(x,y)\in[0,1]^2$.

    Here, $\sigma_{ij} = \sigma_{ji}$ is the stress tensor, $\rho(r)$ is the
    density, $f(r)$ is the time-independent external force, and $fr_i(r)ft_i(t)$
    is the time-dependent external force factorized into spatial and temporal
    components. The stress (\sigma_{11}, \sigma_{22}, \sigma_{12})^T is
    determined by the strain (\epsilon_{11}, \epsilon_{22}, \epsilon_{12})^T
    through a 3x3 matrix $C$:
        $\sigma_{ij} = C_{ijkl}(r)\epsilon_{kl}.$
    The strain is given by
        $\epsilon_{ij}=\frac{1}{2}(\partial_i u_j+\partial_j u_i)$.

    ======== Detailed Description ========
    - Boundary condition on each side is randomly chosen from Dirichlet,
      Neumann, and Robin. The values of boundary conditions are randomly chosen
      from zero, random constant, and random 1D function.
    - The initial displacement $g_i(r)$ is set to be the numerical solution of
      the Steady-State equation $\sigma_{ji,j}+f_i(r)=0$ with specific boundary
      conditions. These boundary conditions are either the same as the ones used
      in the time-dependent equation, or chosen randomly again.
      When introduce motion by initial displacement, a gaussian function will
      be added to the initial displacement.
    - The initial velocity $h_i(r)$ is set to zero.
    - The density $\rho(r)$ and elements of the stiffness tensor $C_{ijkl}(r)$
      are randomly chosen from random positive constants and random 2D
      functions.
    - The time-independent external force $f_i(r)$ is randomly chosen from zero,
      random constant, random constant times $\rho(r)$ and random 2D functions.
    - The time-dependent external force can be factorized into spatial and
      temporal components $fr_i(r)$ and $ft_i(t)$. The spatial component is
      set to be a gaussian function, and the temporal component is set to be a
      Ricker wavelet.
    - The motion is either introduced by the time-dependent external force,
      change of boundary condition, change of initial displacement, randomly
      generated initial displacement, or randomly generated initial velocity,
      each with equal probability.
    """
    VERSION: float = 2.1
    PREPROCESS_DAG: True
    PDE_TYPE_ID: 9
    N_VARS = 2
    CORNERS = (0, 1, 0, 1)  # corners of the domain, (x_min, x_max, y_min, y_max)
    STOP_SIM_TIME = 1.  # stop simulation time

    _solution: NDArray[float]  # [n_x_grid, n_y_grid, n_t_grid, 2]
    type: int
    ISOTROPY = 0
    stiffness_dof: int  # number of independent stiffness tensor components

    motion_type: int
    TIMEDEP_FORCE = 0
    BC_CHANGE = 1
    IC_CHANGE = 2
    RANDOM_IC = 3
    RANDOM_IVEL = 4
    SUPPORTED_MOTION_TYPES = [TIMEDEP_FORCE, BC_CHANGE, IC_CHANGE, RANDOM_IC, RANDOM_IVEL]

    n_x_grid: int
    n_y_grid: int
    n_t_grid: int

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._global_idx = -1  # only for plot
        bc = RectBoundaryCondition
        self.n_x_grid = args.n_x_grid
        self.n_y_grid = args.n_y_grid
        self.n_t_grid = args.n_t_grid
        self.num_max_forces = args.n_forces

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
        self.force_types = args.force_types

        self.num_max_force_types = args.n_types
        self.store_scatter = args.save_scatter
        x_coord = np.linspace(self.CORNERS[0], self.CORNERS[1], self.n_x_grid)
        y_coord = np.linspace(self.CORNERS[2], self.CORNERS[3], self.n_y_grid)
        t_coord = np.linspace(0, self.STOP_SIM_TIME, self.n_t_grid)
        self.coord_dict = {"x": x_coord, "y": y_coord, "t": t_coord}

        self.supported_motion_types = args.motion_types
        for motion_type in self.supported_motion_types:
            if motion_type not in self.SUPPORTED_MOTION_TYPES:
                raise ValueError(f"Unsupported motion type: {motion_type}")

        self.type = self.ISOTROPY  # only support isotropy now
        self.stiffness_dof = 2
        resolution = (args.n_x_grid, args.n_y_grid)
        self.plot_dir = os.path.join("plots", args.plot_dir,
                                     self.get_hdf5_file_prefix(args),
                                     datetime.now().strftime("%Y%m%d%H%M%S"))

        # PDE terms
        self.term_obj_dict["u_ibc"] = bc(self.N_VARS,  # bc for initial condition
                                         args.n_x_grid,
                                         args.n_y_grid,
                                         args.coef_distribution,
                                         args.coef_magnitude)
        self._ic = None  # to be calculated
        self.raw_sol_dict = None  # to be calculated
        self._solution = None  # record for plot
        coords_comp2 = (x_coord.reshape(-1, 1, 1),
                        y_coord.reshape(1, -1, 1),
                        np.array([0, 1]).reshape(1, 1, -1))
        coords = (x_coord.reshape(-1, 1), y_coord.reshape(1, -1))
        # initial displacement, only used in motion_type RANDOM_IC
        self.term_obj_dict["u_ic"] = coefs.RandomConstOrField(
            coords=coords_comp2, periodic=False, resolution=resolution + (2,),
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        # initial velocity
        self.term_obj_dict["ut_ic"] = coefs.RandomConstOrField(
            coords=coords_comp2, periodic=False, resolution=resolution + (2,),
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.term_obj_dict["u_bc"] = bc(self.N_VARS,  # bc for time-dependent equation
                                        args.n_x_grid,
                                        args.n_y_grid,
                                        args.coef_distribution,
                                        args.coef_magnitude)
        self.term_obj_dict["rho"] = coefs.NonNegConstOrField(
            coords=coords, periodic=False, resolution=resolution,
            min_val=args.kappa_min, max_val=args.kappa_max)
        if self.type == self.ISOTROPY:
            self.term_obj_dict["C/0"] = coefs.NonNegConstOrField(
                coords=coords, periodic=False, resolution=resolution,
                min_val=args.kappa_min, max_val=args.kappa_max)  # Young's modulus
            self.term_obj_dict["C/1"] = coefs.NonNegConstOrField(
                coords=coords, periodic=False, resolution=resolution,
                min_val=0.01, max_val=1.)  # Poisson's ratio
        self.term_obj_dict["f"] = TimeIndepForce(
            args.n_x_grid, args.n_y_grid, self.CORNERS,
            args.coef_distribution, args.coef_magnitude)
        self.term_obj_dict["f_rt"] = TimeDepForce(
            args.n_x_grid, args.n_y_grid, self.CORNERS, t_coord,
            args.coef_distribution, args.coef_magnitude)
        self.term_obj_dict["f_ic"] = TimeIndepForce(
            args.n_x_grid, args.n_y_grid, self.CORNERS,
            args.coef_distribution, args.coef_magnitude)

        self.reset_debug()

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        stiff_type = "iso"
        return (f"ElasticWave2D_{stiff_type}"
                f"_c{args.coef_distribution}{args.coef_magnitude:g}"
                f"_k{args.kappa_min:.0e}_{args.kappa_max:g}"
                f"_nf{args.n_forces}_ntf{args.n_types}"
                f"_type{''.join(map(str, args.motion_types))}"
                f"_ftype{''.join(map(str, args.force_types))}"
                f"_scat{args.save_scatter}")

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--n_x_grid", "-Nx", type=int, default=128,
                            help="number of grid points in x direction")
        parser.add_argument("--n_y_grid", "-Ny", type=int, default=128,
                            help="number of grid points in y direction")
        parser.add_argument("--n_t_grid", "-Nt", type=int, default=101,
                            help="number of grid points in t direction")
        parser.add_argument("--n_forces", type=int, default=2,
                            help="maximum number of point forces")
        parser.add_argument("--n_types", type=int, default=2,
                            help="maximum number of force types in field force")
        parser.add_argument("--force_types", type=int, nargs="+",
                            default=TimeIndepForce.SUPPORTED_TYPES,
                            help="supported force types")
        parser.add_argument("--plot_dir", type=str, default="ElasticWave2D",
                            help="directory for plots")
        parser.add_argument("--save_scatter", action="store_true",
                            help="save scatter point solution")
        parser.add_argument("--motion_types", type=int, nargs="+", default=cls.SUPPORTED_MOTION_TYPES,
                            help="supported motion types")
        coefs.RandomConstOrField.add_cli_args_(parser)
        coefs.NonNegConstOrField.add_cli_args_(parser)
        TimeDepForce.add_cli_args_(parser)
        return parser

    def _get_ic(self) -> NDArray[float]:
        r'''
        Get initial condition.
        '''
        if self.motion_type == self.RANDOM_IC:
            ic = self.term_obj_dict["u_ic"].field
        else:
            ic = self.solve_static()
        return ic

    @abstractmethod
    def solve_static(self) -> NDArray[float]:
        r'''
        Solve the static equation to get the initial condition. Shape of
        returned array is [n_x_grid, n_y_grid, 2].
        '''
        pass

    @property
    def ic(self) -> NDArray[float]:
        r'''
        Initial condition. Shape is [n_x_grid, n_y_grid, 2].
        '''
        if self._ic is None:
            self._ic = self._get_ic()
        return self._ic

    def reset_pde(self, rng: np.random.Generator,
                  motion_type: Optional[int] = None) -> None:
        self.raw_sol_dict = None
        self._ic = None
        self.term_obj_dict["u_bc"].reset(rng)
        self.term_obj_dict["rho"].reset(rng)
        for i in range(self.stiffness_dof):
            self.term_obj_dict[f"C/{i}"].reset(rng)
        n_forces = rng.choice(self.num_max_forces) + 1
        n_types = rng.choice(self.num_max_force_types) + 1
        self.term_obj_dict["f"].reset(
            rng, rho=self.term_obj_dict["rho"].field,
            n_forces=n_forces, types=self._gen_force_types(rng, n_types))

        if motion_type is None:
            motion_type = rng.choice(self.supported_motion_types)
        self.motion_type = motion_type
        if motion_type == self.TIMEDEP_FORCE:
            n_forces = rng.choice(self.num_max_forces) + 1
            n_types = rng.choice(self.num_max_force_types) + 1
            self.term_obj_dict["f_rt"].reset(
                rng, rho=self.term_obj_dict["rho"].field,
                n_forces=n_forces, types=self._gen_force_types(rng, n_types))
        elif motion_type == self.BC_CHANGE:
            # keep the types of bc
            bc_type = self.term_obj_dict["u_bc"].bc_types
            self.term_obj_dict["u_ibc"].reset(rng, bc_type=bc_type)
        elif motion_type == self.IC_CHANGE:
            n_forces = rng.choice(self.num_max_forces) + 1
            n_types = rng.choice(self.num_max_force_types) + 1
            self.term_obj_dict["f_ic"].reset(
                rng, rho=self.term_obj_dict["rho"].field,
                n_forces=n_forces, types=self._gen_force_types(rng, n_types))
            # TODO: set the u_bc to be compatable with ic
        elif motion_type == self.RANDOM_IC:
            prob = {"zero_prob": 0, "unit_prob": 0.,
                    "scalar_prob": 1., "field_prob": 1.}
            self.term_obj_dict["u_ic"].reset(rng, **prob)
        elif motion_type == self.RANDOM_IVEL:
            prob = {"zero_prob": 0, "unit_prob": 0.,
                    "scalar_prob": 1., "field_prob": 1.}
            self.term_obj_dict["ut_ic"].reset(rng, **prob)
        else:
            raise NotImplementedError
        self._global_idx += 1  # only for plot

    def reset_debug(self) -> None:
        self._ic = None
        self.raw_sol_dict = None
        self.motion_type = self.TIMEDEP_FORCE
        super().reset_debug()

    def gen_solution(self) -> None:
        r"""
        Generate the PDE solution corresponding to the current PDE parameters.
        """
        try:
            u, u_scat, pts = self.solve_dynamic()
        except:
            self.raw_sol_dict = {}  # failed
            return

        # check if the solution is too stable
        norm_u = np.sqrt(np.sum(u**2))
        norm_u_diff = np.sqrt(np.sum(u - u[:, :, 0:1, :])**2)
        if norm_u_diff < 0.1 * norm_u:
            self.raw_sol_dict = {}
            return

        self._solution = u  # [n_x_grid, n_y_grid, n_t_grid, 2], for plot only.
        if self.store_scatter:
            solution = u_scat  # [n_nodes, n_t_grid, 2]
            self.raw_sol_dict = {"solution": solution.astype(np.float32),
                                 "mesh_points": pts, "ic": self.ic}
        else:
            solution = u  # [n_x_grid, n_y_grid, n_t_grid, 2]
            self.raw_sol_dict = {"solution": solution.astype(np.float32),
                                 "ic": self.ic}

    def _gen_force_types(self, rng: np.random.Generator, n_types: int) -> List[int]:
        r"""
        Generate n_types force types in self.force_types.
        """
        return rng.choice(self.force_types, n_types, replace=False)

    @abstractmethod
    def solve_dynamic(self) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
        r'''
        Solve the dynamic equation to get the solution. Return the interpolated
        solution, scatter solution, and scatter points. Shape of returned arrays
        should be [n_x_grid, n_y_grid, n_t_grid, 2], [n_nodes, n_t_grid, 2], and
        [n_nodes, 2], respectively.
        '''
        pass

    @property
    def coef_dict(self) -> Dict[str, Union[int, float, NDArray]]:
        r""" A dictionary containing the current PDE coefficients. """
        coef_dict = super().coef_dict
        coef_dict.update({"motion_type": self.motion_type,
                          "ic": self.ic})
        return coef_dict

    def _plot3d(self, solution: NDArray[float], filename: str) -> None:
        r"""
        Plot and save the 3D solution.
        """
        time_steps = self.coord_dict["t"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.set_title("Displacement x")
        max1 = np.max(solution[:, :, :, 0])
        min1 = np.min(solution[:, :, :, 0])
        im1 = ax1.imshow(solution[:, :, 0, 0], cmap='jet', origin='lower',
                         vmin=min1, vmax=max1)
        fig.colorbar(im1, ax=ax1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        max2 = np.max(solution[:, :, :, 1])
        min2 = np.min(solution[:, :, :, 1])
        im2 = ax2.imshow(solution[:, :, 0, 1], cmap='jet', origin='lower',
                         vmin=min2, vmax=max2)
        ax2.set_title('Displacement y')
        fig.colorbar(im2, ax=ax2)
        ax2.set_xticks([])
        ax2.set_yticks([])

        def update(frame):
            im1.set_data(solution[:, :, frame, 0])
            im2.set_data(solution[:, :, frame, 1])
            # refresh the title of the figure and colorbar of ax1 and ax2
            ax1.set_title('Displacement X at time {:.2f}'.format(time_steps[frame]))
            ax2.set_title('Displacement Y at time {:.2f}'.format(time_steps[frame]))
            return im1, im2

        ani = animation.FuncAnimation(fig, update, frames=self.n_t_grid, blit=True)
        ani.save(filename, writer='pillow', fps=10)
        plt.close()

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

    def _plot1d(self,
                field: NDArray[float],  # [n_t_grid]
                filename: str,
                title: str = "") -> None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(field)
        ax.set_title(title)
        plt.savefig(filename)
        plt.close()

    def plot(self, plot_coef: bool = True) -> None:
        # TODO: change the behavior
        solution = self._solution
        u0 = self.ic
        root_dir = os.path.join(os.getcwd(), self.plot_dir)
        # TODO: change the dir
        # generate filename based on the current time
        # timestr = datetime.now().strftime("%Y%m%d%H%M%S")
        # save_dir = os.path.join(root_dir, f"type_{self.motion_type}", timestr)
        idxstr = str(self._global_idx)
        save_dir = os.path.join(root_dir, f"type_{self.motion_type}", idxstr)
        os.makedirs(save_dir, exist_ok=True)
        filename1 = os.path.join(save_dir, "init_disp.png")
        self._plot2d(u0, filename1, "Initial Displacement")

        filename2 = os.path.join(save_dir, "solution.gif")
        filename3 = os.path.join(save_dir, "diff.gif")
        self._plot3d(solution, filename2)
        self._plot3d(solution - u0[:, :, np.newaxis, :], filename3)

        # always plot coefficients
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

        if self.motion_type == self.IC_CHANGE:
            # plot f_ic
            field = self.term_obj_dict["f_ic"].field
            filename = os.path.join(save_dir, "f_ic.png")
            self._plot2d(field, filename, "f_ic", n_dim=2)

        if self.motion_type == self.TIMEDEP_FORCE:
            # plot f_rt
            fr_field = self.term_obj_dict["f_rt"].fr_field
            ft_field = self.term_obj_dict["f_rt"].ft_field
            filename1 = os.path.join(save_dir, "f_r.png")
            self._plot2d(fr_field, filename1, "f_r", n_dim=2)
            filename2 = os.path.join(save_dir, "f_t.png")
            self._plot1d(ft_field, filename2, "f_t")

        if self.motion_type == self.RANDOM_IVEL:
            # plot ut_ic
            field = self.term_obj_dict["ut_ic"].field
            filename = os.path.join(save_dir, "ut_ic.png")
            self._plot2d(field, filename, "ut_ic", n_dim=2)
