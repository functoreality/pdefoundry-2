r"""Frequently used PDE terms with randomized coefficients."""
import argparse
from typing import Dict, Union, Optional

import numpy as np
from numpy.typing import NDArray

from .basics import PDETermBase
from .utils_random import RandomValueSampler, int_split
from . import coefs


class PolySinusNonlinTerm(coefs.RandomValue):
    r"""
    Generate coefficients for the non-linear term $f(u)$, in the form
    $f(u) = \sum_{k=1}^3c_{0k}u^k + \sum_{j=1}^Jc_{j0}h_j(c_{j1}u+c_{j2}u^2)$.
    Here, each sinusoidal function $h_j\in\{\sin,\cos\}$ is selected
        with equal probability.
    """

    def __init__(self,
                 num_sinusoid: int = 0,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        size = (1 + num_sinusoid, 4)
        super().__init__(size, coef_distribution, coef_magnitude)

    @property
    def is_zero(self) -> bool:
        r"""Check whether the coefficient has the form [[0, 0, 0., 0.]]."""
        return np.all(self.value == 0)

    @property
    def is_linear(self) -> bool:
        r"""Check whether the coefficient has the form [[0, *, 0., 0.]]."""
        return (self.value[0, 2] == 0 and self.value[0, 3] == 0
                and np.all(self.value[1:, 0] == 0))

    @classmethod
    def add_cli_args_(cls,
                      parser: argparse.ArgumentParser,
                      coef_distribution: str = "U",
                      coef_magnitude: float = 1.) -> None:
        parser.add_argument("--num_sinusoid", "-J", type=int, default=0,
                            help="total number of sinusoidal terms")
        super().add_cli_args_(parser, coef_distribution, coef_magnitude)

    @classmethod
    def arg_str(cls, args: argparse.Namespace) -> str:
        output = super().arg_str(args)
        if args.num_sinusoid > 0:
            output = f"_sJ{args.num_sinusoid}" + output
        return output

    def reset(self,
              rng: np.random.Generator,
              num_sinusoid: Optional[int] = None) -> None:
        super().reset(rng)
        self.value[0, 0] = 0  # do not keep the constant term
        if num_sinusoid is not None:
            self.value[1 + num_sinusoid:, :] = 0

        # fix invalid sinusoidal terms
        for j in range(1, self.size[0]):
            if self.value[j, 0] == 0:
                self.value[j, :] = 0
            elif self.value[j, 1] == 0 and self.value[j, 2] == 0:
                self.value[j, 1] = 1


class HomSpatialOrder2Term(coefs.NonNegRandomValue):
    r"""
    Spatial second-order term, whose form is randomly selected from
    the non-divergence form $Lu=-a\Delta u$, the factored form
    $Lu=-\sqrt a\nabla\cdot(\sqrt a\nabla u)$, and the divergence form
    $Lu=-\nabla\cdot(a\nabla u)$ with equal probability. Here $a$ is
    taken to be a random non-negative real number.
    Note that these three forms are mathematically equivalent since $a$ has no
    spatial dependency, and we distinct them only because they correspond to
    different DAG representations for PDEformer.
    """
    NON_DIV_FORM = 0
    FACTORED_FORM = 1
    DIV_FORM = 2
    diff_type: int

    def __str__(self) -> str:
        if self.diff_type == self.NON_DIV_FORM:
            diff_str = r"non-divergence $-a\Delta u$"
        if self.diff_type == self.FACTORED_FORM:
            diff_str = r"factored $-\sqrt a\nabla\cdot(\sqrt a\nabla u)$"
        if self.diff_type == self.DIV_FORM:
            diff_str = r"divergence $-\nabla\cdot(a\nabla u)$"
        return diff_str + f", a: {self.value}"

    def reset(self,
              rng: np.random.Generator,
              *,
              zero_prob: float = 0.,
              unit_prob: float = 0.1,
              diff_type: Optional[int] = None) -> None:
        super().reset(rng, zero_prob=zero_prob, unit_prob=unit_prob)
        if diff_type is None:
            self.diff_type = rng.choice(3)
        else:
            self.diff_type = diff_type

    def reset_debug(self, zero: bool = False, diff_type: int = 0) -> None:
        super().reset_debug(zero=zero)
        self.diff_type = diff_type

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, float]]:
        return {prefix + "/diff_type": self.diff_type,
                prefix + "/value": self.value}


class InhomSpatialOrder2Term(coefs.NonNegField):
    r"""
    Spatial second-order term, whose form is randomly selected from
    the non-divergence form $Lu=-a(r)\Delta u$, the factored form
    $Lu=-\sqrt a(r)\nabla\cdot(\sqrt a(r)\nabla u)$, and the divergence form
    $Lu=-\nabla\cdot(a(r)\nabla u)$ with equal probability. Here $a(r)$ is
    taken to be a random field, and $r=(x,y,z)$ denotes the spatial
    coordinates.

    Note: We call the second form factored since for 1D wave equation,
        $$u_{tt}-c(x)(c(x)u_x)_x =
        (\partial_t-c(x)\partial_x)(\partial_t+c(x)\partial_x)u,$$
    where $c(x) = \sqrt a(x)$.
    """
    NON_DIV_FORM = 0
    FACTORED_FORM = 1
    DIV_FORM = 2
    diff_type: int

    def __str__(self) -> str:
        if self.diff_type == self.NON_DIV_FORM:
            diff_str = r"non-divergence $-a(r)\Delta u$"
        if self.diff_type == self.FACTORED_FORM:
            diff_str = r"factored $-\sqrt a(r)\nabla\cdot(\sqrt a(r)\nabla u)$"
        if self.diff_type == self.DIV_FORM:
            diff_str = r"divergence $-\nabla\cdot(a(r)\nabla u)$"
        return diff_str + ", a(r): " + super().__str__()

    def reset(self,
              rng: np.random.Generator,
              *,
              diff_type: Optional[int] = None) -> None:
        super().reset(rng)
        if diff_type is None:
            self.diff_type = rng.choice(3)
        else:
            self.diff_type = diff_type

    def reset_debug(self, zero: bool = False, diff_type: int = 0) -> None:
        super().reset_debug(zero=zero)
        self.diff_type = diff_type

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {prefix + "/diff_type": self.diff_type,
                prefix + "/field": self.field}


class MultiComponentLinearTerm(PDETermBase):
    r"""
    Linear term (in vector form) for multi-component PDEs:
        $$f(u)_i = \sum_ja_{ij}u_j,$$
    where $0 \le i,j \le d_u-1$.
    Coefficient matrix $a_{ij}$ stored in the COOrdinate format.
    """
    n_vars: int
    max_len: int
    sampler: RandomValueSampler
    coo_len: int
    coo_vals: NDArray[float]
    coo_i: NDArray[int]
    coo_j: NDArray[int]

    def __init__(self,
                 n_vars: int = 2,
                 max_len: Optional[int] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        self.n_vars = n_vars
        if max_len is None:
            self.max_len = 2 * self.n_vars
        else:
            self.max_len = min(max_len, self.n_vars**2)
        self.sampler = RandomValueSampler(coef_distribution, coef_magnitude)

    def __str__(self) -> str:
        return str(self.todense())

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser,
                      coef_distribution: str = "U",
                      coef_magnitude: float = 1.) -> None:
        parser.add_argument("--n_vars", "-v", type=int, default=2,
                            help="number of variables (components) in the PDE")
        RandomValueSampler.add_cli_args_(
            parser, coef_distribution, coef_magnitude)

    @staticmethod
    def arg_str(args: argparse.Namespace) -> str:
        return f"_nv{args.n_vars}" + RandomValueSampler.arg_str(args)

    def reset(self,
              rng: np.random.Generator,
              n_terms: Optional[int] = None) -> None:
        self._reset_vals(rng, n_terms)
        coo_ij = rng.choice(self.n_vars**2, self.coo_len, replace=False)
        self.coo_i, self.coo_j = divmod(coo_ij, self.n_vars)

    def reset_debug(self) -> None:
        self.coo_len = 0
        self.coo_vals = np.zeros(0)
        self.coo_i = np.zeros(0, dtype=int)
        self.coo_j = np.zeros(0, dtype=int)

    def self_coupling_wrt(self, i: int) -> bool:
        r"""Check whether this term has self-coupling with respect to $u_i$."""
        return np.any((self.coo_i == i) & (self.coo_j == i))

    def todense(self) -> NDArray[float]:
        r""" Convert the sparse 2D/3D array to dense format. """
        dense_arr = np.zeros([self.n_vars] * 2)
        for (i, j, val) in zip(self.coo_i, self.coo_j, self.coo_vals):
            dense_arr[i, j] = val
        return dense_arr

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray]]:
        pad_len = self.max_len - self.coo_len
        data_dict = {prefix + "/coo_len": self.coo_len}

        def add_item(name, data):
            data_dict[prefix + name] = np.pad(data, (0, pad_len))

        add_item("/coo_vals", self.coo_vals)
        add_item("/coo_i", self.coo_i)
        add_item("/coo_j", self.coo_j)
        return data_dict

    def _reset_vals(self,
                    rng: np.random.Generator,
                    n_terms: Optional[int]) -> None:
        if n_terms is None:
            self.coo_len = rng.integers(self.max_len + 1)
        else:
            self.coo_len = min(n_terms, self.max_len)
        coo_vals = self.sampler(rng, size=self.coo_len)
        unit_mask = rng.choice([False, True], size=self.coo_len)
        self.coo_vals = np.where(unit_mask, 1., coo_vals)


class MultiComponentDegree2Term(MultiComponentLinearTerm):
    r"""
    Degree-two term (in vector form) for multi-component PDEs:
        $$f(u)_i = \sum_{j,k}b_{ijk}u_ju_k,$$
    where $0 \le i,j,k \le d_u-1$, $j \le k$.
    Coefficient tensor $b_{ijk}$ stored in the COOrdinate format.
    """
    coo_k: NDArray[int]

    def reset(self,
              rng: np.random.Generator,
              n_terms: Optional[int] = None) -> None:
        self._reset_vals(rng, n_terms)
        # only generate entries with j <= k
        coo_j_all, coo_k_all = np.triu_indices(self.n_vars)
        num_jk_all = coo_j_all.shape[0]
        coo_ijk = rng.choice(self.n_vars * num_jk_all,
                             self.coo_len, replace=False)
        self.coo_i, coo_jk_idx = divmod(coo_ijk, num_jk_all)
        self.coo_j = coo_j_all[coo_jk_idx]
        self.coo_k = coo_k_all[coo_jk_idx]

    def reset_debug(self) -> None:
        super().reset_debug()
        self.coo_k = np.zeros(0, dtype=int)

    def is_linear_wrt(self, i: int) -> bool:
        r""" Check whether this term is linear with respect to $u_i$. """
        # return i not in self.coo_i
        return np.all((self.coo_i != i) | (self.coo_j != i) | (self.coo_k != i))

    def self_coupling_wrt(self, i: int) -> bool:
        return np.any((self.coo_i == i) & ((self.coo_j == i) | (self.coo_k == i)))

    def todense(self) -> NDArray[float]:
        dense_arr = np.zeros([self.n_vars] * 3)
        for (i, j, k, val) in zip(self.coo_i, self.coo_j, self.coo_k, self.coo_vals):
            dense_arr[i, j, k] = val
        return dense_arr

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray]]:
        data_dict = super().get_data_dict(prefix)
        pad_len = self.max_len - self.coo_len
        data_dict[prefix + "/coo_k"] = np.pad(self.coo_k, (0, pad_len))
        return data_dict


class MultiComponentMixedTerm(PDETermBase):
    r"""
    Polynomial term $f(u)$ (in vector form) with degree up to two for
    multi-component PDEs:
        $$f(u)_i = \sum_ja_{ij}u_j + \sum_{j,k}b_{ijk}u_ju_k,$$
    where $0 \le i,j,k \le d_u-1$, $j \le k$. The coefficients $a,b$ are sparse
    arrays, with a total of at most $3d$ non-zero entries.
    """
    max_len: int
    lin_term: MultiComponentLinearTerm
    deg2_term: MultiComponentDegree2Term

    def __init__(self,
                 n_vars: int = 2,
                 max_len: Optional[int] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        if max_len is None:
            self.max_len = 3 * n_vars
        else:
            self.max_len = min(max_len, 2 * n_vars**2)
        self.lin_term = MultiComponentLinearTerm(
            n_vars, max_len, coef_distribution, coef_magnitude)
        self.deg2_term = MultiComponentDegree2Term(
            n_vars, max_len, coef_distribution, coef_magnitude)

    def __str__(self) -> str:
        return ("linear: " + str(self.lin_term)
                + "\n deg2: " + str(self.deg2_term))

    add_cli_args_ = MultiComponentLinearTerm.add_cli_args_
    arg_str = MultiComponentLinearTerm.arg_str

    def reset(self, rng: np.random.Generator) -> None:
        n_lin_terms, n_deg2_terms, _ = int_split(rng, 3, self.max_len)
        self.lin_term.reset(rng, n_lin_terms)
        self.deg2_term.reset(rng, n_deg2_terms)

    def reset_debug(self) -> None:
        self.lin_term.reset_debug()
        self.deg2_term.reset_debug()

    def is_linear_wrt(self, i: int) -> bool:
        r""" Check whether this term is linear with respect to $u_i$. """
        return self.deg2_term.is_linear_wrt(i)

    def self_coupling_wrt(self, i: int) -> bool:
        r"""Check whether this term has self-coupling with respect to $u_i$."""
        return (self.lin_term.self_coupling_wrt(i)
                or self.deg2_term.self_coupling_wrt(i))

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray]]:
        data_dict = self.lin_term.get_data_dict(prefix + "/lin")
        data_dict.update(self.deg2_term.get_data_dict(prefix + "/deg2"))
        return data_dict


class MCompnConvecDegree2Term(MultiComponentDegree2Term):
    r"""
    Degree-two term (in vector form) for multi-component convection PDEs:
        $$f(u)_i = \sum_{j,k}b_{ijk}u_jDu_k,$$
    where $0 \le i,j,k \le d_u-1$, $j \le k$, and
    $D\in\{\partial_x,\partial_y,\partial_z\}$ is the differential operator.
    Coefficient tensor $b_{ijk}$ stored in the sparse COOrdinate format.
    """

    def reset(self,
              rng: np.random.Generator,
              n_terms: Optional[int] = None) -> None:
        self._reset_vals(rng, n_terms)
        coo_ijk = rng.choice(self.n_vars**3, self.coo_len, replace=False)
        self.coo_i, coo_jk_idx = divmod(coo_ijk, self.n_vars**2)
        self.coo_j, self.coo_k = divmod(coo_jk_idx, self.n_vars)


class MCompnConvecMixedTerm(MultiComponentMixedTerm):
    r"""
    Polynomial term $f(u)$ (in vector form) with degree up to two for
    multi-component convection-form PDEs:
        $$f(u)_i = \sum_ja_{ij}u_j + \sum_{j,k}b_{ijk}u_jDu_k,$$
    where $0 \le i,j,k \le d_u-1$, $j \le k$, and
    $D\in\{\partial_x,\partial_y,\partial_z\}$ is the differential operator.
    The coefficients $a,b$ are sparse arrays, with a total of at most $3d_u$
    non-zero entries.
    """
    deg2_term: MCompnConvecDegree2Term

    def __init__(self,
                 n_vars: int = 2,
                 max_len: Optional[int] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(n_vars, max_len, coef_distribution, coef_magnitude)
        self.deg2_term = MCompnConvecDegree2Term(
            n_vars, max_len, coef_distribution, coef_magnitude)
