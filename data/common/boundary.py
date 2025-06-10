r"""Boundary conditions for PDEs."""
import argparse
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
from numpy.typing import NDArray

from .basics import PDETermBase
from . import coefs


class EdgeBoundaryCondition(PDETermBase):
    r"""
    Non-periodic boundary condition of a PDE, applied to one single edge of the
    square domain, taking the general form $Bu(r)=\beta(r)$ for $r\in\Gamma_i$.

    The linear boundary operator $Bu$ is taken to be Robin with (D) type
    $Bu=u+\alpha(r)\partial u/\partial n$, and Robin with (N) type
    $Bu=\alpha(r)u+\partial u/\partial n$, with equal probability.

    The coefficient field $\alpha(r)$ as well as the term $\beta(r)$ are taken
    to be zero, one, a random scalar, or a random field with certain
    probability. Note that when $\alpha(r)$ equals zero, the boundary
    condition would degenerate to the Dirichlet type or the Neumann type. We
    may also set $\beta(r)$ to meet the initial condition.
    """
    ROBIN_D = 0
    ROBIN_N = 1
    bc_type: int
    beta_from_ic: bool
    coef_cls: type = coefs.RandomConstOrField
    alpha: coef_cls
    beta: coef_cls

    def __init__(self,
                 coords: Tuple[NDArray[float]],
                 periodic: Union[bool, Tuple[bool]] = True,
                 resolution: Union[None, int, Tuple[int]] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        self.alpha = self.coef_cls(
            coords, periodic=periodic, resolution=resolution,
            coef_distribution=coef_distribution,
            coef_magnitude=coef_magnitude)
        self.beta = self.coef_cls(
            coords, periodic=periodic, resolution=resolution,
            coef_distribution=coef_distribution,
            coef_magnitude=coef_magnitude)

    def __str__(self) -> str:
        if self.bc_type == self.ROBIN_D:
            bc_str = r"R(D), u + \alpha(r)u_n = \beta(r)"
        elif self.bc_type == self.ROBIN_N:
            bc_str = r"R(N), u_n + \alpha(r)u = \beta(r)"
        bc_str += rf", \alpha(r): {self.alpha}, \beta(r): {self.beta}"
        if self.beta_from_ic:
            bc_str += " (from IC)"
        return bc_str

    def reset(self,
              rng: np.random.Generator,
              u_init: Optional[NDArray[float]] = None,
              un_init: Optional[NDArray[float]] = None,
              *,
              bc_type: Optional[int] = None) -> None:
        r"""
        Reset boundary condition for this edge.
        Args:
            rng (np.random.Generator): Random number generator instance.
            u_init (Optional[NDArray[float]]): Value of the initial condition
                g(x) at this edge, i.e. $g(x)|\Gamma_i$. Default: None, do not
                set $\beta$ according to the initial condition.
            un_init (Optional[NDArray[float]]): Value of $(\partial g/\partial
                n)|\Gamma_i$. Default: None, do not set $\beta$ according to
                the initial condition.

        Keyword Args:
            bc_type (Optional[int]): Type of the boundary condition. Choices:
                0 (Robin, Dirichlet-based), 1 (Robin, Neumann-based).
        """
        if bc_type is None:
            self.bc_type = rng.choice(2)
        else:
            self.bc_type = bc_type
        self.alpha.reset(rng)
        self.beta.reset(rng)

        # Default case
        self.beta_from_ic = False
        if u_init is None or un_init is None:
            return
        if self.beta.is_const or bool(rng.choice(2)):
            return

        # Make beta comply with the initial condition.
        self.beta_from_ic = True
        if self.bc_type == self.ROBIN_D:
            self.beta.field = u_init + self.alpha.field * un_init
        elif self.bc_type == self.ROBIN_N:
            self.beta.field = un_init + self.alpha.field * u_init

    def reset_debug(self, u_init: Optional[NDArray[float]] = None) -> None:
        self.bc_type = self.ROBIN_D
        self.alpha.reset_debug()
        if u_init is None:
            self.beta_from_ic = False
            self.beta.reset_debug()
        else:
            self.beta_from_ic = True
            self.beta.field = u_init
            self.beta.coef_type = self.beta.FIELD_COEF

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        data_dict = {prefix + "/bc_type": self.bc_type,
                     prefix + "/beta_from_ic": self.beta_from_ic}
        data_dict.update(self.alpha.get_data_dict(prefix + "/alpha"))
        data_dict.update(self.beta.get_data_dict(prefix + "/beta"))
        return data_dict

    def prepare_plot(self, title: str = "") -> None:
        self.alpha.prepare_plot(title=title + r" $\alpha(r)$")
        self.beta.prepare_plot(title=title + r" $\beta(r)$")


class EdgeBCWithMur(EdgeBoundaryCondition):
    r"""
    Non-periodic boundary condition of a wave equation, applied to one single
    edge of the square domain, taking the general form $Bu(r)=\beta(r)$ for
    $r\in\Gamma_i$.

    The linear boundary operator $Bu$ is taken to be Robin with (D) type
    $Bu=u+\alpha(r)\partial u/\partial n$, Robin with (N) type
    $Bu=\alpha(r)u+\partial u/\partial n$, and (generalized) Mur type
    $Bu=u_t+\alpha(r)u+\gamma(r)\partial u/\partial n$ with equal probability.
    """
    MUR_R = 2
    gamma: coefs.RandomConstOrField

    def __init__(self,
                 coords: Tuple[NDArray[float]],
                 periodic: Union[bool, Tuple[bool]] = True,
                 resolution: Union[None, int, Tuple[int]] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(coords, periodic, resolution,
                         coef_distribution=coef_distribution,
                         coef_magnitude=coef_magnitude)
        self.gamma = self.coef_cls(
            coords, periodic=periodic, resolution=resolution,
            coef_distribution=coef_distribution,
            coef_magnitude=coef_magnitude)

    def __str__(self) -> str:
        if self.bc_type != self.MUR_R:
            return super().__str__()
        bc_str = (r"Mur, u_t + \alpha(r)u + \gamma(r)u_n = \beta(r), "
                  rf"\alpha(r): {self.alpha}, \gamma(r): {self.gamma}, "
                  rf"\beta(r): {self.beta}")
        if self.beta_from_ic:
            bc_str += " (from IC)"
        return bc_str

    def reset(self,
              rng: np.random.Generator,
              u_init: Optional[NDArray[float]] = None,
              un_init: Optional[NDArray[float]] = None,
              ut_init: Optional[NDArray[float]] = None,
              *,
              bc_type: Optional[int] = None) -> None:
        r"""
        Reset boundary condition for this edge.
        Args:
            rng (np.random.Generator): Random number generator instance.
            u_init (Optional[NDArray[float]]): Value of the initial condition
                g(x) at this edge, i.e. $g(x)|\Gamma_i$. Default: None, do not
                set $\beta$ according to the initial condition.
            un_init (Optional[NDArray[float]]): Value of $(\partial g/\partial
                n)|\Gamma_i$. Default: None, do not set $\beta$ according to
                the initial condition.
            ut_init (NDArray[float]): Initial value of $u_t$, i.e. $h|\Gamma_i$.
                Default: None, do not set $\beta$ according to IC.

        Keyword Args:
            bc_type (Optional[int]): Type of the boundary condition. Choices:
                0 (Robin, Dirichlet-based), 1 (Robin, Neumann-based),
                2 (generalized Mur).
        """
        if bc_type is None:
            self.bc_type = rng.choice(3)
        else:
            self.bc_type = bc_type

        if self.bc_type != self.MUR_R:
            super().reset(rng, u_init, un_init, bc_type=self.bc_type)
            self.gamma.reset_debug()
            return

        # generalized Mur case
        self.alpha.reset(rng)
        self.beta.reset(rng)
        self.gamma.reset(rng)

        # Default case
        self.beta_from_ic = False
        if u_init is None or un_init is None or ut_init is None:
            return
        if self.beta.is_const or bool(rng.choice(2)):
            return

        # Make beta comply with the initial condition
        self.beta_from_ic = True
        self.beta.field = (ut_init + self.alpha.field * u_init
                           + self.gamma.field * un_init)

    def reset_debug(self, u_init: Optional[NDArray[float]] = None) -> None:
        super().reset_debug(u_init)
        self.gamma.reset_debug()

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        data_dict = super().get_data_dict(prefix)
        data_dict.update(self.gamma.get_data_dict(prefix + "/gamma"))
        return data_dict

    def prepare_plot(self, title: str = "") -> None:
        super().prepare_plot(title=title)
        self.gamma.prepare_plot(title=title + r" $\gamma(r)$")


def exclude_entry(source: Union[List, Tuple], idx: int) -> Union[List, Tuple]:
    r"""
    Exclude one certain entry from a tuple/list.

    Examples:
        >>> exclude_entry((0, -1, 3), 0)
        (-1, 3)
        >>> exclude_entry((0, -1, 3), 1)
        (0, 3)
        >>> exclude_entry([0, -1, 3], -1)
        [0, -1]
        >>> exclude_entry((0, -1, 3), 5)
        (0, -1, 3)
    """
    if idx == -1:
        return source[:-1]
    return source[:idx] + source[idx + 1:]


def exclude_coord_entry(source: Tuple[NDArray], idx: int) -> Tuple[NDArray]:
    r"""
    Exclude one certain entry from a tuple/list of coordinates.

    Examples:
        >>> a = np.array([[0, 1, 2]])
        >>> b = np.array([[4], [6]])
        >>> exclude_coord_entry((a, b), 0)
        ([4, 6],)
        >>> exclude_coord_entry((a, b), 1)
        ([0, 1, 2],)
    """
    source = tuple(np.take(array, 0, axis=idx) for array in source)
    return exclude_entry(source, idx)


class BoxDomainBoundaryCondition(PDETermBase):
    r"""
    Non-periodic boundary condition of a PDE, including all edges (surfaces) of
    a square (cubic) domain, unless the domain is periodic along this axis.

    For each edge or surface, the boundary condition takes the general form
    $Bu(r)=\beta(r)$ for $r\in\Gamma_i$.
    The linear boundary operator $Bu$ is taken to be Robin with (D) type
    $Bu=u+\alpha(r)\partial u/\partial n$, and Robin with (N) type
    $Bu=\alpha(r)u+\partial u/\partial n$, with equal probability.

    The coefficient field $\alpha(r)$ as well as the term $\beta(r)$ are taken
    to be zero, one, a random scalar, or a random field with certain
    probability. Note that when $\alpha(r)$ equals zero, the boundary
    condition would degenerate to the Dirichlet type or the Neumann type. We
    may also set $\beta(r)$ to meet the initial condition.
    """
    edge_bc_cls: type = EdgeBoundaryCondition
    edge_term_dict: Dict[int, Tuple[edge_bc_cls]]
    ic_obj: Optional[coefs.RandomField] = None

    def __init__(self,
                 coords: Tuple[NDArray[float]],
                 periodic: Union[bool, Tuple[bool]] = True,
                 resolution: Union[None, int, Tuple[int]] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        coords, periodic, resolution = coefs.field_arg_tuples(
            coords, periodic, resolution)
        self.edge_term_dict = {}
        for i_ax, periodic_ax in enumerate(periodic):
            if periodic_ax:
                continue
            bc_low = self.edge_bc_cls(
                exclude_coord_entry(coords, i_ax),
                exclude_entry(periodic, i_ax),
                exclude_entry(resolution, i_ax),
                coef_distribution=coef_distribution,
                coef_magnitude=coef_magnitude)
            bc_high = self.edge_bc_cls(
                exclude_coord_entry(coords, i_ax),
                exclude_entry(periodic, i_ax),
                exclude_entry(resolution, i_ax),
                coef_distribution=coef_distribution,
                coef_magnitude=coef_magnitude)
            self.edge_term_dict[i_ax] = (bc_low, bc_high)

    def __str__(self) -> str:
        bc_str_list = []
        for i_ax, (bc_low, bc_high) in self.edge_term_dict.items():
            bc_str_list.append("xyzw"[i_ax] + "_low: " + str(bc_low))
            bc_str_list.append("xyzw"[i_ax] + "_high: " + str(bc_high))
        return "\n    ".join(bc_str_list)

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser, ndim: int = 2) -> None:
        ax_name_all = "xyzw"[:ndim]

        def periodic_arr(periodic_axes: str) -> NDArray[bool]:
            r"""
            Examples for the case ndim==2:
                >>> periodic_arr("xy")
                array([True, True])
                >>> periodic_arr("x")
                array([True, False])
                >>> periodic_arr("/")
                array([False, False])

            Example for the case ndim==3:
                >>> periodic_arr("y")
                array([False, True, False])
            """
            periodic_axes = periodic_axes.lower()
            periodic = [ax_name in periodic_axes for ax_name in ax_name_all]
            # Convert list to NumPy array so that it can be saved into the HDF5
            # file. See the code in basics.PDEDataRecorder.save_hdf5 .
            return np.array(periodic)
        parser.add_argument(
            "--periodic", "-p", type=periodic_arr, default="xyzw",
            help="Periodic along which axes. Passing '/' means periodic over "
            "no axes. Default: 'xyzw', periodic over all axes.")

    @staticmethod
    def arg_str(args: argparse.Namespace) -> str:
        non_periodic_axes = ""
        for ax_name, flag in zip("xyzw", args.periodic):
            if not flag:
                non_periodic_axes += ax_name
        if len(non_periodic_axes) > 0:
            return "_np" + non_periodic_axes.upper()
        return ""

    def reset(self, rng: np.random.Generator) -> None:
        for i_ax, (bc_low, bc_high) in self.edge_term_dict.items():
            if self.ic_obj is None:
                bc_low.reset(rng)
                bc_high.reset(rng)
            else:
                u_init, un_init = self.ic_obj.boundary_values(i_ax, False)
                bc_low.reset(rng, u_init, un_init)
                u_init, un_init = self.ic_obj.boundary_values(i_ax, True)
                bc_high.reset(rng, u_init, un_init)

    def reset_debug(self, beta_from_ic: bool = True) -> None:
        def get_bv(axis: int, from_end: bool) -> Optional[NDArray[float]]:
            if self.ic_obj is None or not beta_from_ic:
                return None
            u_init, _ = self.ic_obj.boundary_values(axis, from_end)
            return u_init
        for i_ax, (bc_low, bc_high) in self.edge_term_dict.items():
            bc_low.reset_debug(get_bv(i_ax, False))
            bc_high.reset_debug(get_bv(i_ax, True))

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        data_dict = {}
        for i_ax, (bc_low, bc_high) in self.edge_term_dict.items():
            ax_name = "xyzw"[i_ax]
            data_dict.update(bc_low.get_data_dict(f"{prefix}/{ax_name}_low"))
            data_dict.update(bc_high.get_data_dict(f"{prefix}/{ax_name}_high"))
        return data_dict

    def prepare_plot(self, title: str = "") -> None:
        for i_ax, (bc_low, bc_high) in self.edge_term_dict.items():
            ax_name = "xyzw"[i_ax]
            bc_low.prepare_plot(title=f"{title}|${ax_name}_{{low}}$")
            bc_high.prepare_plot(title=f"{title}|${ax_name}_{{high}}$")

    def assign_ic(self, ic_obj: coefs.RandomField) -> None:
        r"""
        Assign the corresponding initial-condition generator object to allow
        setting $\beta$ accordingly with a certain probability.
        """
        self.ic_obj = ic_obj


class BoxDomainBCWithMur(BoxDomainBoundaryCondition):
    r"""
    Non-periodic boundary condition of a PDE, including all edges (surfaces) of
    a square (cubic) domain, unless the domain is periodic along this axis.

    For each edge or surface, the boundary condition takes the general form
    $Bu(r)=\beta(r)$ for $r\in\Gamma_i$.
    The linear boundary operator $Bu$ is taken to be Robin with (D) type
    $Bu=u+\alpha(r)\partial u/\partial n$, Robin with (N) type
    $Bu=\alpha(r)u+\partial u/\partial n$, and (generalized) Mur type
    $Bu=u_t+\alpha(r)u+\gamma(r)\partial u/\partial n$ with equal probability.

    The coefficient fields $\alpha(r),\gamma(r)$ as well as the term $\beta(r)$
    are taken to be zero, one, a random scalar, or a random field with certain
    probability. We may also set $\beta(r)$ to meet the initial condition.
    """
    edge_bc_cls: type = EdgeBCWithMur
    edge_term_dict: Dict[int, Tuple[edge_bc_cls]]  # update typing
    ut_ic_obj: Optional[coefs.RandomField] = None

    def reset(self, rng: np.random.Generator) -> None:
        for i_ax, (bc_low, bc_high) in self.edge_term_dict.items():
            if self.ic_obj is None or self.ut_ic_obj is None:
                bc_low.reset(rng)
                bc_high.reset(rng)
            else:
                u_init, un_init = self.ic_obj.boundary_values(i_ax, False)
                ut_init, _ = self.ut_ic_obj.boundary_values(i_ax, False)
                bc_low.reset(rng, u_init, un_init, ut_init)
                u_init, un_init = self.ic_obj.boundary_values(i_ax, True)
                ut_init, _ = self.ut_ic_obj.boundary_values(i_ax, True)
                bc_high.reset(rng, u_init, un_init, ut_init)

    def assign_ic(self,
                  ic_obj: coefs.RandomField,
                  ut_ic_obj: coefs.RandomField = None) -> None:
        if ut_ic_obj is None:
            raise RuntimeError("Please specify 'ut_ic_obj'.")
        self.ic_obj = ic_obj
        self.ut_ic_obj = ut_ic_obj


class DiskDomain(PDETermBase):
    r"""A random disk-shaped domain."""
    min_diameter: float
    radius: float
    center: NDArray[float]

    def __init__(self, min_diameter: float) -> None:
        super().__init__()
        self.min_diameter = min_diameter

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--min_diameter", type=float, default=0.6,
                            help="Minimal diameter of the disk.")

    @staticmethod
    def arg_str(args: argparse.Namespace) -> str:
        return "_disk"

    def reset(self, rng: np.random.Generator) -> None:
        self.radius = 0.5 * rng.uniform(self.min_diameter, 1.0)
        self.center = rng.uniform(self.radius, 1. - self.radius, size=2)

    def reset_debug(self) -> None:
        self.radius = 0.5
        self.center = np.array([0.5, 0.5])

    def get_data_dict(self, prefix: str) -> Dict[str, Union[float, NDArray[float]]]:
        return {prefix + "/radius": self.radius,
                prefix + "/center": self.center}
