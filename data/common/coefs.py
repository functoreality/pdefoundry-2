r"""Commonly used randomized PDE coefficients."""
import argparse
from typing import Dict, Tuple, Union, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from .basics import PDETermBase, prepare_plot_2d
from .utils_random import RandomValueSampler, GaussianRandomFieldSampler, real_split


class RandomValue(PDETermBase):
    r"""
    Random coefficient(s) in a PDE, each entry set to zero with certain
    probability.
    """
    ZERO_COEF = 0
    UNIT_COEF = 1
    SCALAR_COEF = 2
    sampler: RandomValueSampler
    size: Union[None, int, Tuple[int]]
    value: NDArray[float]

    def __init__(self,
                 size: Union[None, int, Tuple[int]] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        self.sampler = RandomValueSampler(coef_distribution, coef_magnitude)
        self.size = size

    add_cli_args_ = RandomValueSampler.add_cli_args_
    arg_str = RandomValueSampler.arg_str

    def reset(self, rng: np.random.Generator) -> None:
        entry_type = rng.choice(3, size=self.size)
        value = self.sampler(rng, size=self.size)
        value = np.where(entry_type == self.ZERO_COEF, 0., value)
        value = np.where(entry_type == self.UNIT_COEF, 1., value)
        self.value = value

    def reset_debug(self) -> None:
        self.value = np.zeros(self.size)

    def get_data_dict(self, prefix: str) -> Dict[str, NDArray[float]]:
        return {prefix: self.value}


class NonNegRandomValue(PDETermBase):
    r"""
    Random non-negative coefficient(s) in the PDE, which may appear in
    diffusion, damping or wave propagation, etc.
    """
    log_min: float
    log_max: float
    size: Union[None, int, Tuple[int]]
    value: NDArray[float]

    def __init__(self,
                 min_val: float = 1e-3,
                 max_val: float = 1.,
                 size: Union[None, int, Tuple[int]] = None,
                 ) -> None:
        super().__init__()
        self.log_min = np.log(min_val)
        self.log_max = np.log(max_val)
        self.size = size

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser,
                      kappa_min: float = 1e-3,
                      kappa_max: float = 1.) -> None:
        parser.add_argument("--kappa_min", type=float, default=kappa_min,
                            help="Lower bound for the non-negative "
                            "coefficient. Must be positive.")
        parser.add_argument("--kappa_max", type=float, default=kappa_max,
                            help="Similar to --kappa_min, but for the upper "
                            "bound.")

    @staticmethod
    def arg_str(args: argparse.Namespace) -> str:
        return f"_k{args.kappa_min:.0e}_{args.kappa_max:g}"

    def reset(self,
              rng: np.random.Generator,
              *,
              zero_prob: Union[float, NDArray[float]] = 0.,
              unit_prob: float = 0.1) -> None:
        value = rng.uniform(self.log_min, self.log_max, size=self.size)
        value = np.exp(value)
        if self.log_min <= 0 <= self.log_max:
            mask = rng.choice([False, True],
                              p=[unit_prob, 1 - unit_prob],
                              size=self.size)
            value = np.where(mask, value, 1.)
        mask = rng.random(self.size) >= zero_prob
        self.value = np.where(mask, value, 0.)

    def reset_debug(self, zero: bool = False) -> None:
        if zero:
            self.value = np.zeros(self.size)
        else:
            value = (self.log_min + self.log_max) / 2
            self.value = np.full(self.size, np.exp(value))

    def get_data_dict(self, prefix: str) -> Dict[str, NDArray[float]]:
        return {prefix: self.value}


def field_arg_tuples(coords: Tuple[NDArray[float]],
                     periodic: Union[bool, Tuple[bool]] = True,
                     resolution: Union[None, int, Tuple[int]] = None,
                     ) -> Tuple[Tuple]:
    r"""
    Reformulate the arguments for random field generators into tuples.
    """
    # coords
    if isinstance(coords, np.ndarray):
        coords = (coords,)
    elif isinstance(coords, list):
        coords = tuple(coords)
    """
    # eg. ([n_x, 1, 1], [1, n_y, 1]) -> [[n_x], [n_y]]
    coords = [coord.flat for coord in coords]
    # eg. [[n_x], [n_y]] -> [[n_x, 1], [1, n_y]]
    coords = np.meshgrid(*coords, copy=False, sparse=True, indexing="ij")
    coords = tuple(coords)  # list -> tuple
    """

    # resolution
    if resolution is None:
        resolution = tuple(coord.size for coord in coords)
    elif isinstance(resolution, int):
        resolution = (resolution,) * len(coords)
    elif len(resolution) != len(coords):
        raise ValueError(
            f"The length of 'resolution' ({len(resolution)}) "
            f"should be equal to that of 'coords' ({len(coords)})!")

    # periodic
    if isinstance(periodic, bool):
        periodic = (periodic,) * len(coords)
    elif len(periodic) != len(coords):
        raise ValueError(
            f"The length of 'periodic' ({len(periodic)}) "
            f"should be equal to that of 'coords' ({len(coords)})!")
    return coords, periodic, resolution


def diff_along_axis0(arr: NDArray[float]) -> NDArray[float]:
    r"""Compute numerical difference along axis 0 of an array."""
    # np.diff does not preserve array shape
    diff_low = arr[1:2] - arr[0:1]
    diff_body = (arr[2:] - arr[:-2]) / 2
    diff_high = arr[-1:] - arr[-2:-1]
    return np.concatenate((diff_low, diff_body, diff_high), axis=0)


class RandomField(PDETermBase):
    r"""
    Generating random field term of a PDE.

    Args:
        coords (Tuple[NDArray[float]]): Coordinate points along all axes. These
            arrays will be reshaped automatically, so there is no strict
            restriction on the shapes of these input arrays. For example, the
            shapes can be ([x_res],) for 1D, ([x_res, 1], [1, y_res]) for 2D,
            and ([x_res, 1, 1], [1, y_res, 1], [1, 1, z_res]) for 3D.
        periodic (Union[bool, Tuple[bool]])
        resolution (Union[None, int, Tuple[int]])
    """
    field: NDArray[float]
    coords: Tuple[NDArray[float]]
    # periodic: Tuple[bool]
    resolution: Tuple[int]
    sampler: GaussianRandomFieldSampler

    def __init__(self,
                 coords: Tuple[NDArray[float]],
                 periodic: Union[bool, Tuple[bool]] = True,
                 resolution: Union[None, int, Tuple[int]] = None) -> None:
        super().__init__()
        self.coords, _, self.resolution = field_arg_tuples(
            coords, periodic, resolution)
        self.sampler = GaussianRandomFieldSampler(
            len(self.resolution), self.resolution, alpha=3, tau=5, sigma=50)

    def __str__(self) -> str:
        data = self.field
        shape = self.field.shape  # or self.resolution
        return f"shape {shape}, range [{data.min():.4g}, {data.max():.4g}]"
        # return f"[{data.flat[0]:.4g} ... {data.flat[-1]:.4g}], shape {shape}"

    def reset(self, rng: np.random.Generator) -> None:
        field = self.sampler(rng)
        self.field, = field  # [1, ...] -> [...]

    def reset_debug(self) -> None:
        self.field = np.zeros(self.resolution)

    def get_data_dict(self, prefix: str) -> Dict[str, NDArray[float]]:
        return {prefix: self.field}

    def prepare_plot(self, title: str = "") -> None:
        if len(self.resolution) == 1:
            plt.figure()
            coord, = self.coords
            plt.plot(coord.flat, self.field)
            plt.title(title)
        elif len(self.resolution) == 2:
            prepare_plot_2d(self.field, title=title)

    def boundary_values(self, axis: int, from_end: bool) -> Tuple[NDArray[float]]:
        r"""
        Get the field values as well as its normal derivatives on one edge of
        the domain $[0,1]^d$. Typical usage case: When the random field
        specifies the initial value of some PDE variable, we compute its
        boundary values $u|\Gamma$ as well as $\partial u/\partial n$ to set a
        boundary condition that is compatible with this initial value.
        """
        if from_end:
            f_value = self.field.take(-1, axis=axis)
            f2_value = self.field.take(-2, axis=axis)
        else:
            f_value = self.field.take(0, axis=axis)
            f2_value = self.field.take(1, axis=axis)
        fn_value = self.field.shape[axis] * (f_value - f2_value)
        return f_value, fn_value

    def elim_div_2d_(self, field_y) -> None:
        r"""
        Regenerate the vector field with null divergence. The original field
        stored will be treated as $\psi$, and the new field would be set as the
        x-component of the divergence-free vector field $\nabla\times\psi$,
        with field_y set as the y-component. The new vector field will be
        rescaled according to the original field afterwards.

        Args:
            field_y (RandomField)
        """
        if self.field.ndim != 2:
            raise RuntimeError("This method works only for the 2D case, "
                               f"not {self.field.ndim}D.")

        psi_x = self.resolution[0] * diff_along_axis0(self.field)
        psi_y = self.resolution[1] * diff_along_axis0(self.field.T).T

        # shift to mean zero
        psi_x -= psi_x.mean()
        psi_y -= psi_y.mean()

        # rescale to the same magnitude
        sql2_old = np.mean((self.field - self.field.mean())**2
                           + (field_y.field - field_y.field.mean())**2)
        sql2_new = np.mean(psi_x**2 + psi_y**2)
        scale = np.sqrt(sql2_old / (1e-6 + sql2_new))
        psi_x = scale * psi_x + self.field.mean()
        psi_y = scale * psi_y + field_y.field.mean()

        # reset fields
        self.field = psi_y
        field_y.field = -psi_x

    def div_constraint_2d_(self,
                           rng: np.random.Generator,
                           field_y,
                           coef: RandomValue) -> None:
        r"""
        Regenerate the vector field with divergence constraint
        $\partial_xu_0+\partial_yu_1+c_0u_0+c_1u_1+c_2=0$.

        Args:
            rng (np.random.Generator)
            field_y (RandomField)
            coef (RandomValue)
        """
        if self.field.ndim != 2 or field_y.field.ndim != 2:
            raise RuntimeError("This method works only for the 2D case.")
        if coef.size != 3:
            raise ValueError(f"Size of 'coef' ({coef.size}) should be 3.")

        # Compute g = (c + \nabla)\times\psi, which satisfies
        # (c + \nabla)\cdot g = 0.
        psi = self.field.copy()
        psi_x = self.resolution[0] * diff_along_axis0(psi)
        psi_y = self.resolution[1] * diff_along_axis0(psi.T).T
        g_0 = coef.value[1] * psi + psi_y
        g_1 = -(coef.value[0] * psi + psi_x)

        # rescale to the same magnitude (like RMSNorm)
        sql2_old = np.mean(self.field**2 + field_y.field**2)
        sql2_new = np.mean(g_0**2 + g_1**2)
        scale = np.sqrt(sql2_old / (1e-8 + sql2_new))
        g_0 *= scale
        g_1 *= scale

        # reset constant term c_2
        if coef.value[0] == 0. and coef.value[1] == 0.:  # divergence-free case
            coef.value[2] = 0.
        elif coef.value[2] != 0.:  # inhomogeneous constraint case
            shift = rng.normal(size=2)
            g_0 += shift[0]
            g_1 += shift[1]
            coef.value[2] = -(coef.value[:2] @ shift)

        # reset fields
        self.field = g_0
        field_y.field = g_1

    def interpolated_field(self,
                           coords: Optional[Tuple[NDArray[float]]] = None,
                           ) -> NDArray[float]:
        r"""Interpolate the field values to the specified grid points."""
        # interpolate to non-uniform grids
        coords_old = tuple(np.linspace(0, 1, axis_resolution + 1)[:-1]
                           for axis_resolution in self.field.shape)
        interp = RegularGridInterpolator(
            coords_old, self.field, bounds_error=False, fill_value=None)
        if coords is None:
            coords = self.coords
        field = interp(coords)
        return field


class RandomConstOrField(RandomField):
    r"""
    Coefficient term involved in a PDE, which can be a zero, a real number
    (scalar) or a spatially-varying random field.
    """
    ZERO_COEF = 0
    UNIT_COEF = 1
    SCALAR_COEF = 2
    FIELD_COEF = 3
    coef_type: int
    const_sampler: RandomValueSampler

    def __init__(self,
                 coords: Tuple[NDArray[float]],
                 periodic: Union[bool, Tuple[bool]] = True,
                 resolution: Union[None, int, Tuple[int]] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(coords, periodic=periodic, resolution=resolution)
        self.const_sampler = RandomValueSampler(coef_distribution, coef_magnitude)

    def __str__(self) -> str:
        if self.coef_type == self.ZERO_COEF:
            return "0"
        if self.coef_type == self.UNIT_COEF:
            return "1"
        if self.coef_type == self.SCALAR_COEF:
            return f"{self.field.flat[0]:.4g}"
        return "field " + super().__str__()

    @property
    def is_const(self) -> bool:
        r""" Whether the current coefficient is a constant. """
        return self.coef_type != self.FIELD_COEF

    add_cli_args_ = RandomValueSampler.add_cli_args_
    arg_str = RandomValueSampler.arg_str

    def reset(self,
              rng: np.random.Generator,
              *,
              zero_prob: float = 1.,
              unit_prob: float = 0.2,
              scalar_prob: float = 1.,
              field_prob: float = 1.,
              coef_type: Optional[int] = None,
              ) -> None:
        if coef_type is None:
            probs = np.array([zero_prob, unit_prob, scalar_prob, field_prob])
            if probs.sum() == 0:
                raise ValueError("At least one of the probs should be positive.")
            self.coef_type = rng.choice(4, p=probs/probs.sum())
        else:
            self.coef_type = coef_type

        if self.coef_type == self.ZERO_COEF:
            self.field = np.zeros(self.resolution)
        elif self.coef_type == self.UNIT_COEF:
            self.field = np.ones(self.resolution)
        elif self.coef_type == self.SCALAR_COEF:
            value = self.const_sampler(rng)
            self.field = np.full(self.resolution, value)
        elif self.coef_type == self.FIELD_COEF:
            super().reset(rng)

    def reset_debug(self) -> None:
        self.coef_type = self.ZERO_COEF
        super().reset_debug()

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {prefix + "/coef_type": self.coef_type,
                prefix + "/field": self.field}

    def prepare_plot(self, title: str = "") -> None:
        if not self.is_const:
            super().prepare_plot(title=title)


def _get_positive_field(rng: np.random.Generator,
                        field: NDArray[float],
                        log_min: float,
                        log_max: float) -> NDArray[float]:
    r"""Produce a field with positive values based on a real-valued one."""
    # values normalized to [0,1]
    field -= field.min()
    field /= 1e-8 + field.max()

    # let the range of field span a random sub-interval of
    # the full interval [log_min,log_max]
    (margin_bottom, span, _) = real_split(rng, 3, log_max - log_min)
    # we have margin_bottom + span + _ == log_max - log_min
    field = log_min + margin_bottom + span * field

    field = field.clip(log_min, log_max)
    return np.exp(field)


class NonNegField(RandomField):
    r""" Generating a non-negative random field term of a PDE. """
    log_min: float
    log_max: float

    def __init__(self,
                 coords: Tuple[NDArray[float]],
                 periodic: Union[bool, Tuple[bool]] = True,
                 resolution: Union[None, int, Tuple[int]] = None,
                 min_val: float = 1e-3,
                 max_val: float = 1.) -> None:
        super().__init__(coords, periodic=periodic, resolution=resolution)
        self.log_min = np.log(min_val)
        self.log_max = np.log(max_val)

    add_cli_args_ = NonNegRandomValue.add_cli_args_
    arg_str = NonNegRandomValue.arg_str

    def reset(self, rng: np.random.Generator) -> None:
        field = self.sampler(rng)
        field, = field  # [1, ...] -> [...]
        self.field = _get_positive_field(rng, field, self.log_min, self.log_max)

    def reset_debug(self, zero: bool = False) -> None:
        if zero:
            self.field = np.zeros(self.resolution)
        else:
            value = (self.log_min + self.log_max) / 2
            self.field = np.full(self.resolution, np.exp(value))

    def prepare_plot(self, title: str = "") -> None:
        super().prepare_plot(title=title)
        if len(self.resolution) == 1:
            plt.yscale("log")
            plt.ylim(np.exp(self.log_min), np.exp(self.log_max))


class NonNegConstOrField(RandomConstOrField):
    r"""
    A non-negative coefficient in the PDE, which may appear in diffusion or
    wave propagation, etc. Can be a zero, a real number (scalar) or a
    spatially-varying random field.
    """
    log_min: float
    log_max: float

    def __init__(self,
                 coords: Tuple[NDArray[float]],
                 periodic: Union[bool, Tuple[bool]] = True,
                 resolution: Union[None, int, Tuple[int]] = None,
                 min_val: float = 1e-3,
                 max_val: float = 1.) -> None:
        super().__init__(coords, periodic=periodic, resolution=resolution)
        self.log_min = np.log(min_val)
        self.log_max = np.log(max_val)
        self.const_sampler = None

    add_cli_args_ = NonNegRandomValue.add_cli_args_
    arg_str = NonNegRandomValue.arg_str

    def reset(self,
              rng: np.random.Generator,
              *,
              zero_prob: float = 0.,
              unit_prob: float = 0.2,
              scalar_prob: float = 1.,
              field_prob: float = 1.,
              coef_type: Optional[int] = None) -> None:
        if coef_type is None:
            if self.log_min > 0 or self.log_max < 0:
                unit_prob = 0.
            probs = np.array([zero_prob, unit_prob, scalar_prob, field_prob])
            if probs.sum() == 0:
                raise ValueError("At least one of the probs should be positive.")
            self.coef_type = rng.choice(4, p=probs/probs.sum())
        else:
            self.coef_type = coef_type

        if self.coef_type == self.ZERO_COEF:
            self.field = np.zeros(self.resolution)
        elif self.coef_type == self.UNIT_COEF:
            self.field = np.ones(self.resolution)
        elif self.coef_type == self.SCALAR_COEF:
            value = rng.uniform(self.log_min, self.log_max)
            self.field = np.full(self.resolution, np.exp(value))
        elif self.coef_type == self.FIELD_COEF:
            field = self.sampler(rng)
            field, = field  # [1, ...] -> [...]
            self.field = _get_positive_field(rng, field, self.log_min, self.log_max)

    def reset_debug(self, zero: bool = False) -> None:
        if zero:
            self.coef_type = self.ZERO_COEF
            self.field = np.zeros(self.resolution)
        else:
            self.coef_type = self.SCALAR_COEF
            value = (self.log_min + self.log_max) / 2
            self.field = np.full(self.resolution, np.exp(value))

    def prepare_plot(self, title: str = "") -> None:
        super().prepare_plot(title=title)
        if not self.is_const and len(self.resolution) == 1:
            plt.yscale("log")
            plt.ylim(np.exp(self.log_min), np.exp(self.log_max))
