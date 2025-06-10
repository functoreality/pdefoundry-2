#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of the 2D wave equation."""
import argparse
from typing import Tuple
import numpy as np
import dedalus.public as d3

from ..utils import coefs, terms, boundary
from ..utils.basics import DedalusPDEType
from ...common.basics import get_cli_args, gen_data
from ...common.utils_random import int_split
try:
    from test_debug2d import dist, d3_dx, d3_dy, get_bases
except ImportError:
    from ..utils.settings2d import dist, d3_dx, d3_dy, get_bases


class Wave2DPDE(DedalusPDEType):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== Wave Equation ========
    The PDE takes the form
        $$u_{tt}+\mu(r)u_t+Lu+f_0(u)+s(r)+f_1(u)_x+f_2(u)_y=0,$$
    $u(0,r)=g(r)$, $u_t(0,r)=h(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$.
    For $i=1,\dots,4$, if periodic boundary condition is not employed on edge
    $\Gamma_i$, we impose an additional constraint $B_iu|\Gamma_i=\beta_i(r)$.

    Here, the spatial second-order term $Lu$ is randomly selected from
    the non-divergence form $Lu=-a(r)\Delta u$, the factored form
    $Lu=-\sqrt a(r)\nabla\cdot(\sqrt a(r)\nabla u)$, and the divergence form
    $Lu=-\nabla\cdot(a(r)\nabla u)$ with equal probability, where $a(r)$ is
    taken to be a random scalar or a random field, and $r=(x,y)$ denotes the
    spatial coordinates.

    We take $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
                     + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$
    for $i=0,1,2$, where $J_0+J_1+J_2\le J$ are randomly generated.

    In terms of the boundary condition, we set
    $B_iu=u+\alpha_i(r)\partial u/\partial n$,
    $B_iu=\alpha_i(r)u+\partial u/\partial n$, and
    $B_iu=u_t+\alpha_i(r)u+\gamma_i(r)\partial u/\partial n$ with equal
    probability. Each of the terms $\alpha_i(r),\beta_i(r)$, and $\gamma_i(r)$
    are randomly selected from zero, one, a random scalar, or a random field.
    We may also set $\beta_i(r)$ to meet the initial condition with certain
    probability.
    """
    PREPROCESS_DAG = True
    PDE_TYPE_ID = 7

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.inhom_diff_u = args.inhom_diff_u
        self.num_sinusoid = args.num_sinusoid
        self.periodic = args.periodic.tolist()

        # Dedalus Bases
        self.bases = get_bases(self.periodic)  # format (xbasis, ybasis)
        coords = dist.local_grids(*self.bases)
        self.coord_dict["x"], self.coord_dict["y"] = coords

        # PDE terms
        self.term_obj_dict["u_ic"] = coefs.RandomField(coords, self.periodic)
        self.term_obj_dict["ut_ic"] = coefs.RandomField(coords, self.periodic)
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
        self.term_obj_dict["mu"] = coefs.RandomConstOrField(
            coords, self.periodic,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.term_obj_dict["s"] = coefs.RandomConstOrField(
            coords, self.periodic,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        if args.inhom_diff_u:
            self.term_obj_dict["Lu"] = terms.InhomSpatialOrder2Term(
                coords, self.periodic,
                min_val=args.kappa_min, max_val=args.kappa_max)
        else:
            self.term_obj_dict["Lu"] = terms.HomSpatialOrder2Term(
                min_val=args.kappa_min, max_val=args.kappa_max)
        self.term_obj_dict["bc"] = boundary.BoxDomainBCWithMur(
            coords, self.periodic,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.term_obj_dict["bc"].assign_ic(
            self.term_obj_dict["u_ic"], self.term_obj_dict["ut_ic"])

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        diff_u_type = "_inhom" if args.inhom_diff_u else "_hom"
        return ("Wave2D"
                + diff_u_type
                + boundary.BoxDomainBCWithMur.arg_str(args)
                + terms.PolySinusNonlinTerm.arg_str(args)
                + terms.HomSpatialOrder2Term.arg_str(args))

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--inhom_diff_u", action="store_true",
                            help="Whether the spatial differential term Lu has"
                            " spatial dependency.")
        boundary.BoxDomainBCWithMur.add_cli_args_(parser, ndim=2)
        terms.PolySinusNonlinTerm.add_cli_args_(parser)
        # same for InhomSpatialOrder2Term
        terms.HomSpatialOrder2Term.add_cli_args_(
            parser, kappa_min=1e-2, kappa_max=4)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        # coefficients of f_i(u)
        num_sinusoid = int_split(rng, 3, self.num_sinusoid)
        self.term_obj_dict["f0"].reset(rng, num_sinusoid[0])
        self.term_obj_dict["f1"].reset(rng, num_sinusoid[1])
        self.term_obj_dict["f2"].reset(rng, num_sinusoid[2])

        # random fields
        self.term_obj_dict["u_ic"].reset(rng)
        self.term_obj_dict["ut_ic"].reset(rng)
        self.term_obj_dict["mu"].reset(rng)
        self.term_obj_dict["s"].reset(rng)
        if self.inhom_diff_u:
            self.term_obj_dict["Lu"].reset(rng)
        elif (self.term_obj_dict["f1"].is_zero
              and self.term_obj_dict["f2"].is_zero):
            self.term_obj_dict["Lu"].reset(rng, zero_prob=0.2)
        else:
            # reasons for requiring non-zero wavespeed: The equation
            # u_{tt} = u_x is unstable, while u_{tt} = u_x + c^2u_{xx} is.
            self.term_obj_dict["Lu"].reset(rng, zero_prob=0.)

        # boundary conditions
        self.term_obj_dict["bc"].reset(rng)

    def get_dedalus_problem(self) -> Tuple:
        # Fields
        u_op = dist.Field(name="u", bases=self.bases)
        dt_u = dist.Field(name="dt_u", bases=self.bases)
        mu_op = dist.Field(name="mu", bases=self.bases)
        s_op = dist.Field(name="s", bases=self.bases)

        # Initial condition
        self.term_obj_dict["u_ic"].gen_dedalus_ops(u_op)
        self.term_obj_dict["ut_ic"].gen_dedalus_ops(dt_u)

        # PDE Terms
        f0_lin, f0_nonlin = self.term_obj_dict["f0"].gen_dedalus_ops(u_op)
        f1_lin, f1_nonlin = self.term_obj_dict["f1"].gen_dedalus_ops(u_op)
        f2_lin, f2_nonlin = self.term_obj_dict["f2"].gen_dedalus_ops(u_op)
        s_op = self.term_obj_dict["s"].gen_dedalus_ops(s_op)
        lin = f0_lin + d3_dx(f1_lin) + d3_dy(f2_lin)
        nonlin = f0_nonlin + d3_dx(f1_nonlin) + d3_dy(f2_nonlin) + s_op

        mu_op = self.term_obj_dict["mu"].gen_dedalus_ops(mu_op)
        if self.term_obj_dict["mu"].is_const:
            lin = lin + mu_op * dt_u
        else:
            nonlin = nonlin + mu_op * dt_u

        if self.inhom_diff_u:
            kappa_op = dist.Field(name="kappa", bases=self.bases)
            diff_u = self.term_obj_dict["Lu"].gen_dedalus_ops(kappa_op, d3.grad(u_op))
            lin = lin + diff_u[0]
            nonlin = nonlin + diff_u[1]
        else:
            diff_u = self.term_obj_dict["Lu"].gen_dedalus_ops(d3.grad(u_op))
            lin = lin + diff_u

        # Boundary conditions
        bc_ops_list = self.term_obj_dict["bc"].gen_dedalus_ops(
            u_op, dt_u, (d3_dx(u_op), d3_dy(u_op)), dist, self.bases)
        tau_op, tau_var_list = boundary.tau_polynomial(
            dist, self.bases, self.periodic)

        # Problem
        problem = d3.IVP([u_op, dt_u] + tau_var_list)
        problem.add_equation([d3.dt(u_op) - dt_u, 0])
        problem.add_equation([d3.dt(dt_u) + lin + tau_op, -nonlin])
        for bc_lhs, bc_rhs in bc_ops_list:
            problem.add_equation([bc_lhs, bc_rhs])
        return {"u": u_op}, problem


class WaveUnitTest(Wave2DPDE):
    # T_SAVE_STEPS = 2  # success rate 4/12 for 1, 2

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "Wave_debug"

    def reset_pde(self, rng: np.random.Generator) -> None:
        # self.reset_debug()
        # self.term_obj_dict["f0"].reset(rng)
        # self.term_obj_dict["f1"].reset(rng)
        # self.term_obj_dict["f2"].reset(rng)
        # self.term_obj_dict["u_ic"].reset(rng)
        # self.term_obj_dict["s"].reset(rng)
        # self.term_obj_dict["Lu"].reset(rng)
        super().reset_pde(rng)
        self.term_obj_dict["mu"].reset(rng, field_prob=1e3)


if __name__ == "__main__":
    my_args = get_cli_args(Wave2DPDE)
    pde_data_obj = Wave2DPDE(my_args)
    gen_data(my_args, pde_data_obj)
