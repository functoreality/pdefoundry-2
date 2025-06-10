#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of the 2D Wave Equation in a disk."""
import argparse
from typing import Tuple
import numpy as np
import dedalus.public as d3

from ..utils import coefs, terms, boundary
from ...common.basics import get_cli_args, gen_data
from ...common.utils_random import int_split
from .disk_dcr import DiffConvecReacDiskPDE


class WaveDiskPDE(DiffConvecReacDiskPDE):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== Wave Equation in a Disk ========
    The PDE takes the form
        $$u_{tt}+\mu(r)u_t+Lu+f_0(u)+s(r)+f_1(u)_x+f_2(u)_y=0,$$
    $u(0,r)=g(r)$, $u_t(0,r)=h(r)$, $t\in[0,1]$, $r=(x,y)\in\Omega$, with
    boundary condition $Bu|\partial\Omega=\beta(r)$.

    Here, the spatial second-order term $Lu$ is randomly selected from
    the non-divergence form $Lu=-a(r)\Delta u$, the factored form
    $Lu=-\sqrt a(r)\nabla\cdot(\sqrt a(r)\nabla u)$, and the divergence form
    $Lu=-\nabla\cdot(a(r)\nabla u)$ with equal probability, where $a(r)$ is
    taken to be a random scalar or a random field, and $r=(x,y,z)$ denotes the
    spatial coordinates.

    We take $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
                     + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$
    for $i=0,1,2$, where $J_0+J_1+J_2\le J$ are randomly generated.

    In terms of the boundary condition, we set
    $Bu=u+\alpha(r)\partial u/\partial n$,
    $Bu=\alpha(r)u+\partial u/\partial n$, and
    $Bu=u_t+\alpha(r)u+\gamma(r)\partial u/\partial n$ with equal
    probability. Each of the terms $\alpha(r),\beta(r)$, and $\gamma(r)$
    are randomly selected from zero, one, a random scalar, or a random field.

    The computational domain $\Omega\subset[0,1]^2$ is a disk with random
    radius and center location.
    """
    PDE_TYPE_ID = 7
    TRIAL_N_SUB_STEPS = [1]
    IS_WAVE = True


class WaveDiskPDEDebug(WaveDiskPDE):
    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "WaveDisk_debug"

    def reset_pde(self, rng: np.random.Generator) -> None:
        self.term_obj_dict["domain"].reset(rng)
        self._reset_coords(self.term_obj_dict["domain"])
        self.reset_debug()
        # self.term_obj_dict["bc/outer"].bc_type = 2
        # self.term_obj_dict["bc/outer"].gamma.reset(rng, coef_type=3)
        self.term_obj_dict["u_ic"].reset(rng)
        self.term_obj_dict["Lu"].reset(rng)


if __name__ == "__main__":
    my_args = get_cli_args(WaveDiskPDE)
    pde_data_obj = WaveDiskPDE(my_args)
    gen_data(my_args, pde_data_obj)
