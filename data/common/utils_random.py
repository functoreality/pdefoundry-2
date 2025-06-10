r""" Utilities for generation of random object. """
import math
import argparse
from typing import Union, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def real_split(rng: np.random.Generator,
               k: int = 2,
               total: float = 1.) -> NDArray[float]:
    r"""
    Randomly split a real number `total` into the sum `k` non-negative real
    numbers. The returned value is an array containing `k` float elements.
    """
    return total * rng.dirichlet(np.ones(k))


def int_split(rng: np.random.Generator,
              k: int = 2,
              total: int = 1) -> NDArray[int]:
    r"""
    Randomly split an integer `total` into the sum `k` non-negative integers.
    The returned value is an array containing `k` int elements.
    """
    n = total
    # sample 0 <= t[0] < t[1] < ... < t[k-2] < n + k - 1
    t_arr = rng.choice(n + k - 1, k - 1, replace=False)
    t_arr = np.sort(t_arr)
    # s[i] = t[i] - i, 0 <= s[0] <= s[1] <= .. <= s[k-2] <= n
    s_arr = t_arr - np.arange(k - 1)
    # define s[-1] = 0, s[k-1] = n
    s_arr = np.concatenate(([0], s_arr, [n]))
    m_arr = np.diff(s_arr)  # m[i] = s[i] - s[i-1]
    return m_arr


class RandomValueSampler:
    r"""Generate random scalar coefficients for PDEs."""
    distribution: str
    magnitude: float

    def __init__(self,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        self.distribution = coef_distribution
        self.magnitude = coef_magnitude

    def __call__(self,
                 rng: np.random.Generator,
                 size: Union[None, int, Tuple[int]] = None,
                 ) -> NDArray[float]:
        r"""Generate random values."""
        if self.distribution == "U":
            array = rng.uniform(-1, 1, size=size)
        elif self.distribution == "N":
            array = rng.normal(size=size)
        elif self.distribution == "L":
            array = rng.laplace(size=size)
        elif self.distribution == "C":
            array = rng.standard_cauchy(size=size)
        else:
            raise NotImplementedError
        return self.magnitude * array

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser,
                      coef_distribution: str = "U",
                      coef_magnitude: float = 1.) -> None:
        r""" Add command-line arguments for the related PDE terms. """
        parser.add_argument("--coef_distribution", type=str,
                            default=coef_distribution,
                            choices=["U", "N", "L", "C"], help=r"""
                            distribution type of random PDE coefficients.
                            choices: 'U' (uniform U([-m, m]); default),
                                     'N' (normal N(0, m^2)),
                                     'L' (Laplace with mean 0 and scale m),
                                     'C' (Cauchy with mean 0 and scale m),
                            """)
        parser.add_argument("--coef_magnitude", "-m", type=float,
                            default=coef_magnitude,
                            help="magnitude of randomly generate PDE coefficients")

    @staticmethod
    def arg_str(args: argparse.Namespace) -> str:
        r"""
        Obtain string representation of the command-line arguments to be used
        in data file names.
        """
        return f"_c{args.coef_distribution}{args.coef_magnitude:g}"


class GaussianRandomFieldSampler:
    r"""$X ~ N(\mu, \sigma (-\Delta + \tau^2 I)^{-\alpha})$ with random number $\mu$"""

    def __init__(self,
                 ndim: int,
                 nxys: Union[List[int], int],
                 alpha: Optional[float] = None,
                 tau: Optional[float] = None,
                 sigma: Optional[float] = None) -> None:
        self.ndim = ndim
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma

        if isinstance(nxys, int):
            self.nxys = [nxys] * self.ndim
        elif isinstance(nxys, (tuple, list)) and len(nxys) == self.ndim:
            self.nxys = nxys
        else:
            raise ValueError(f"Unsupported 'nxys' {nxys}!")

        self.sqrt2 = math.sqrt(2.)
        self.pi2 = math.pi**2

    def __call__(self,
                 rng: np.random.Generator,
                 batch_size: int = 1,
                 alpha: Optional[float] = None,
                 tau: Optional[float] = None,
                 sigma: Optional[float] = None) -> NDArray[float]:
        if alpha is None:
            alpha = self.alpha

        if tau is None:
            tau = self.tau

        if sigma is None:
            sigma = self.sigma

        alpha, tau, sigma = self.get_default_paras(self.ndim, alpha, tau, sigma)

        ks = self.get_ks(nxy_list=self.nxys)

        k2_sum = 0.
        nx_mul = 1.

        for dim_idx in range(self.ndim):
            nx_mul *= self.nxys[dim_idx]
            k2_sum = k2_sum + ks[dim_idx]**2

        sqrt_eig = nx_mul * self.sqrt2 * sigma * (4 * self.pi2 * k2_sum + tau**2)**(-alpha / 2.)
        # sqrt_eig[(0,) * self.ndim] = 0.

        noise_h = rng.standard_normal((2, batch_size, *self.nxys))
        noise_h = noise_h[0] + 1.j * noise_h[1]  # [batch_size, *nxys]
        grf_h = sqrt_eig[np.newaxis, ...] * noise_h
        grf = np.fft.ifftn(grf_h, axes=tuple(range(1, self.ndim + 1)))
        grf = grf.real

        return grf
        # return self._normalize(grf)

    @staticmethod
    def get_default_paras(ndim=2, alpha: Optional[float] = None, tau: Optional[float] = None, sigma: Optional[float] = None):
        # by default: sigma = tau**((2 * alpha - ndim) / 2)
        if sigma is None:
            assert (alpha is not None) and (tau is not None)
            sigma = tau**(alpha - ndim / 2)

        if alpha is None:
            assert (tau is not None) and (sigma is not None)
            alpha = math.log(sigma) / math.log(tau) + ndim / 2

        if tau is None:
            assert (alpha is not None) and (sigma is not None)
            tau = sigma**(1 / (alpha - ndim / 2))

        return alpha, tau, sigma

    @staticmethod
    def get_ks(nxy_list: List[int]) -> List[NDArray[float]]:
        ndim = len(nxy_list)
        ks = []
        for dim_idx in range(ndim):
            nx = nxy_list[dim_idx]
            kmax = nx // 2
            kx = np.concatenate(
                (np.arange(0, kmax), np.arange(-kmax, 0)), axis=0)
            ks.append(kx[(None,) * dim_idx + (...,) +
                      (None,) * (ndim - 1 - dim_idx)])
        return ks

    @staticmethod
    def _normalize(data):
        data = (data - data.mean()) / data.std()
        return data


if __name__ == "__main__":  # unit test
    grf_generator = GaussianRandomFieldSampler(2, (256, 128), alpha=3, tau=5, sigma=100)
    rng = np.random.default_rng()
    res, = grf_generator(rng)
    print(res.shape)
    plt.imshow(res.T, cmap="jet", origin="lower")
    plt.colorbar()
    plt.show()
