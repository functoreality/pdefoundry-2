#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Generate dataset of the 2D diffusion-convection-reaction PDE on
randomly-located disk domains.
"""
import argparse
from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import dedalus.public as d3

from ..utils import coefs, terms, boundary
from ..utils.basics import DedalusPDEType
from ...common.basics import get_cli_args, gen_data, prepare_plot_2d_video
from ...common.utils_random import int_split

# Parameters
try:
    import test_debug2d
    N_PHI, N_R = 32, 32
except:
    N_PHI, N_R = 128, 128
DEALIAS = 2
PERIODIC = False
RESOLUTION = 128

# Bases
d3_coords = d3.PolarCoordinates("phi", "r")
dist = d3.Distributor(d3_coords, dtype=np.float64)


class DiffConvecReacDiskPDE(DedalusPDEType):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== Diffusion-Convection-Reaction Equation in a Disk ========
    The PDE takes the form
        $$u_t+Lu+f_0(u)+s(r)+f_1(u)_x+f_2(u)_y=0,$$
    $u(0,r)=g(r)$, $t\in[0,1]$, $r=(x,y)\in\Omega$, with boundary condition
    $Bu|\partial\Omega=\beta(r)$.

    Here, the spatial second-order term $Lu$ is randomly selected from
    the non-divergence form $Lu=-a(r)\Delta u$, the factored form
    $Lu=-\sqrt a(r)\nabla\cdot(\sqrt a(r)\nabla u)$, and the divergence form
    $Lu=-\nabla\cdot(a(r)\nabla u)$ with equal probability, where $a(r)$ is
    taken to be a random scalar or a random field, and $r=(x,y,z)$ denotes the
    spatial coordinates.

    We take $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
                     + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$
    for $i=0,1,2$, where $J_0+J_1+J_2\le J$ are randomly generated.

    In terms of the boundary condition, we set for each edge with non-periodic
    boundary $Bu=u+\alpha(r)\partial u/\partial n$ and
    $Bu=\alpha(r)u+\partial u/\partial n$ with equal probability. Each of
    the terms $\alpha(r)$ as well as $\beta(r)$ are randomly selected from
    zero, one, a random scalar, or a random field. Note that when $\alpha(r)$
    equals zero, the boundary condition would degenerate to the Dirichlet type
    or the Neumann type.

    The computational domain $\Omega\subset[0,1]^2$ is a disk with random
    radius and center location.
    """
    PREPROCESS_DAG = True
    PDE_TYPE_ID = 6
    TRIAL_N_SUB_STEPS = [4, 8, 16]
    IS_WAVE = False
    sol_coord_dict: Dict[str, NDArray[float]]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.inhom_diff_u = args.inhom_diff_u
        self.num_sinusoid = args.num_sinusoid
        self.sol_coord_dict = {}

        # Fake coordinates
        coords = (np.full((N_PHI, 1), np.nan), np.full((1, N_R), np.nan))

        # PDE terms
        self.term_obj_dict["u_ic"] = coefs.RandomField(coords, PERIODIC, RESOLUTION)
        if self.IS_WAVE:
            self.term_obj_dict["ut_ic"] = coefs.RandomField(
                coords, PERIODIC, RESOLUTION)
        self.term_obj_dict["f0"] = terms.PolySinusNonlinTerm(
            num_sinusoid=args.num_sinusoid,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.term_obj_dict["f1"] = terms.PolySinusNonlinTerm(
            num_sinusoid=args.num_sinusoid,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.term_obj_dict["f2"] = terms.PolySinusNonlinTerm(
            num_sinusoid=args.num_sinusoid,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        if self.IS_WAVE:
            self.term_obj_dict["mu"] = coefs.RandomConstOrField(
                coords, PERIODIC, RESOLUTION,
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)
        self.term_obj_dict["s"] = coefs.RandomConstOrField(
            coords, PERIODIC, RESOLUTION,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        if args.inhom_diff_u:
            self.term_obj_dict["Lu"] = terms.InhomSpatialOrder2Term(
                coords, PERIODIC, RESOLUTION,
                min_val=args.kappa_min, max_val=args.kappa_max)
        else:
            self.term_obj_dict["Lu"] = terms.HomSpatialOrder2Term(
                min_val=args.kappa_min, max_val=args.kappa_max)
        self.term_obj_dict["domain"] = boundary.DiskDomain(args.min_diameter)

        if self.IS_WAVE:
            bc_cls = boundary.EdgeBCWithMur
        else:
            bc_cls = boundary.EdgeBoundaryCondition
        self.term_obj_dict["bc/outer"] = bc_cls(
            coords, PERIODIC, RESOLUTION,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)

    @property
    def sol_dict(self) -> Dict[str, NDArray[float]]:
        if not self.raw_sol_dict:  # dict empty
            return {}
        sol_with_coords = self.raw_sol_dict.copy()
        for ax_name, coord in self.sol_coord_dict.items():
            sol_with_coords["coords/" + ax_name] = coord
        return sol_with_coords

    @classmethod
    def get_hdf5_file_prefix(cls, args: argparse.Namespace) -> str:
        diff_u_type = "_inhom" if args.inhom_diff_u else "_hom"
        prefix_base = "Wave2D" if cls.IS_WAVE else "DiffConvecReac2D"
        return (prefix_base
                + diff_u_type
                + boundary.DiskDomain.arg_str(args)
                + terms.PolySinusNonlinTerm.arg_str(args)
                + terms.HomSpatialOrder2Term.arg_str(args))

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--inhom_diff_u", action="store_true",
                            help="Whether the spatial differential term Lu has"
                            " spatial dependency.")
        boundary.DiskDomain.add_cli_args_(parser)
        terms.PolySinusNonlinTerm.add_cli_args_(parser)
        # same for InhomSpatialOrder2Term
        if cls.IS_WAVE:
            terms.HomSpatialOrder2Term.add_cli_args_(
                parser, kappa_min=1e-2, kappa_max=4)
        else:
            terms.HomSpatialOrder2Term.add_cli_args_(parser, kappa_max=1e-2)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        self.term_obj_dict["domain"].reset(rng)
        self._reset_coords(self.term_obj_dict["domain"])

        # coefficients of f_i(u)
        num_sinusoid = int_split(rng, 3, self.num_sinusoid)
        self.term_obj_dict["f0"].reset(rng, num_sinusoid[0])
        self.term_obj_dict["f1"].reset(rng, num_sinusoid[1])
        self.term_obj_dict["f2"].reset(rng, num_sinusoid[2])

        # random fields
        self.term_obj_dict["u_ic"].reset(rng)
        self.term_obj_dict["s"].reset(rng)
        if self.IS_WAVE:
            self.term_obj_dict["ut_ic"].reset(rng)
            self.term_obj_dict["mu"].reset(rng)

        if self.inhom_diff_u:
            self.term_obj_dict["Lu"].reset(rng)
        elif (self.term_obj_dict["f1"].is_zero
              and self.term_obj_dict["f2"].is_zero):
            self.term_obj_dict["Lu"].reset(rng, zero_prob=0.2)
        elif (self.term_obj_dict["f1"].is_linear
              and self.term_obj_dict["f2"].is_linear
              and not self.IS_WAVE):
            self.term_obj_dict["Lu"].reset(rng, zero_prob=0.2)
        else:
            self.term_obj_dict["Lu"].reset(rng, zero_prob=0.)

        # boundary conditions
        self.term_obj_dict["bc/outer"].reset(rng)

    def get_dedalus_problem(self) -> Tuple:
        # Fields
        u_op = dist.Field(name="u", bases=self.bases)
        if self.IS_WAVE:
            dt_u = dist.Field(name="dt_u", bases=self.bases)
            mu_op = dist.Field(name="mu", bases=self.bases)
        s_op = dist.Field(name="s", bases=self.bases)
        tau_u = dist.Field(name="tau_u", bases=self.bases.edge)
        x_op = dist.Field(name="x", bases=self.bases)
        y_op = dist.Field(name="y", bases=self.bases)

        coords = (self.sol_coord_dict["x"], self.sol_coord_dict["y"])

        # Initial condition
        self.term_obj_dict["u_ic"].gen_dedalus_ops(u_op, coords)
        if self.IS_WAVE:
            self.term_obj_dict["ut_ic"].gen_dedalus_ops(dt_u, coords)
        x_op["g"] = self.sol_coord_dict["x"]
        y_op["g"] = self.sol_coord_dict["y"]

        # PDE Terms
        f0_lin, f0_nonlin = self.term_obj_dict["f0"].gen_dedalus_ops(u_op)
        f1_lin, f1_nonlin = self.term_obj_dict["f1"].gen_dedalus_ops(u_op)
        f2_lin, f2_nonlin = self.term_obj_dict["f2"].gen_dedalus_ops(u_op)
        s_op = self.term_obj_dict["s"].gen_dedalus_ops(s_op, coords)
        lin = f0_lin + d3.Lift(tau_u, self.bases, -1)
        nonlin = f0_nonlin + s_op
        # "+ y_op" aims to convert float to dedalus operand.
        nonlin += d3.grad(x_op) @ d3.grad(f1_lin + f1_nonlin + y_op)
        nonlin += d3.grad(y_op) @ d3.grad(f2_lin + f2_nonlin + x_op)

        if self.IS_WAVE:
            mu_op = self.term_obj_dict["mu"].gen_dedalus_ops(mu_op, coords)
            if self.term_obj_dict["mu"].is_const:
                lin = lin + mu_op * dt_u
            else:
                nonlin = nonlin + mu_op * dt_u

        grad_u = d3.grad(u_op)
        if self.inhom_diff_u:
            diff_u = self.term_obj_dict["Lu"].gen_dedalus_ops(
                dist.Field(name="kappa", bases=self.bases),
                grad_u, coords)
            lin = lin + diff_u[0]
            nonlin = nonlin + diff_u[1]
        else:
            diff_u = self.term_obj_dict["Lu"].gen_dedalus_ops(grad_u)
            lin = lin + diff_u

        # Boundary conditions
        if self.IS_WAVE:
            bc_ops = self.term_obj_dict["bc/outer"].gen_dedalus_ops(
                u_op(r="right"),
                d3.radial(grad_u(r="right")),
                dt_u(r="right"),
                dist.Field(name="alpha", bases=self.bases.edge),
                dist.Field(name="gamma", bases=self.bases.edge),
                dist.Field(name="beta", bases=self.bases.edge),
                # coords at outer edge r="right"
                coords=tuple(coord[:, -1] for coord in coords))
        else:
            bc_ops = self.term_obj_dict["bc/outer"].gen_dedalus_ops(
                u_op(r="right"),
                d3.radial(grad_u(r="right")),
                dist.Field(name="alpha", bases=self.bases.edge),
                dist.Field(name="beta", bases=self.bases.edge),
                # coords at outer edge r="right"
                coords=tuple(coord[:, -1] for coord in coords))

        # Problem
        if self.IS_WAVE:
            problem = d3.IVP([u_op, dt_u, tau_u])
            problem.add_equation([d3.dt(u_op) - dt_u, 0])
            problem.add_equation([d3.dt(dt_u) + lin, -nonlin])
        else:
            problem = d3.IVP([u_op, tau_u])
            problem.add_equation([d3.dt(u_op) + lin, -nonlin])
        problem.add_equation([bc_ops[0], bc_ops[1]])
        return {"u": u_op}, problem

    def plot(self, plot_coef: bool = True) -> None:
        # plot coefficients
        if plot_coef:
            for prefix, term_obj in self.term_obj_dict.items():
                term_obj.prepare_plot(prefix)

        # plot current solution
        coords = (self.sol_coord_dict["x"], self.sol_coord_dict["y"])
        anim = prepare_plot_2d_video(
            self.raw_sol_dict["u"], coords=coords, title="u")

        plt.show()

    def _reset_coords(self, domain_obj: boundary.DiskDomain) -> None:
        r"""Reset the coordinate points according to the sampled domain."""
        self.bases = d3.DiskBasis(
            d3_coords, shape=(N_PHI, N_R), radius=domain_obj.radius,
            dealias=DEALIAS, dtype=np.float64)
        phi_coord, r_coord = dist.local_grids(self.bases)
        self.sol_coord_dict["phi"] = phi_coord
        self.sol_coord_dict["r"] = r_coord
        x_coord, y_coord = d3_coords.cartesian(phi_coord, r_coord)
        self.sol_coord_dict["x"] = x_coord + domain_obj.center[0]
        self.sol_coord_dict["y"] = y_coord + domain_obj.center[1]


class DCRUnitTest(DiffConvecReacDiskPDE):
    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "DCRDisk_debug"

    def reset_pde(self, rng: np.random.Generator) -> None:
        self.reset_debug()
        self.term_obj_dict["f0"].reset(rng)
        self.term_obj_dict["f1"].reset(rng)
        self.term_obj_dict["f2"].reset(rng)
        self.term_obj_dict["u_ic"].reset(rng)
        self.term_obj_dict["s"].reset(rng)
        self.term_obj_dict["Lu"].reset(rng)


if __name__ == "__main__":
    my_args = get_cli_args(DiffConvecReacDiskPDE)
    pde_data_obj = DiffConvecReacDiskPDE(my_args)
    gen_data(my_args, pde_data_obj)
