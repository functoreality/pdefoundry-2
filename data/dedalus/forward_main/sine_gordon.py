#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate dataset of the 2D Sine-Gordon equation."""
import argparse
from typing import Tuple
import numpy as np
import dedalus.public as d3

from ..utils import coefs, terms
from ..utils.basics import DedalusPDEType
from ...common.basics import get_cli_args, gen_data
try:
    from test_debug2d import dist, d3_dx, d3_dy, get_bases
except ImportError:
    from ..utils.settings2d import dist, d3_dx, d3_dy, get_bases


class SineGordon2DPDE(DedalusPDEType):
    r"""
    Generate dataset of 2D time-dependent PDE solutions with Dedalus-v3.
    ======== Sine-Gordon Equation ========
    The PDE takes the form
        $$u_{tt}-\nu\Delta u+(-1)sin(u)=0,$$
    $u(0,r)=g(r)$, $u_t(0,r)=h(r)$, $t\in[0,1]$, $r=(x,y)\in[0,1]^2$.
    """
    PREPROCESS_DAG = False

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        # Dedalus Bases
        self.bases = get_bases(periodic=[True, True])  # Tuple (xbasis, ybasis)
        coords = dist.local_grids(*self.bases)
        self.coord_dict["x"], self.coord_dict["y"] = coords

        # PDE terms; periodic by default
        self.term_obj_dict["u_ic"] = coefs.RandomField(coords)
        self.term_obj_dict["ut_ic"] = coefs.RandomField(coords)
        self.term_obj_dict["Lu"] = terms.HomSpatialOrder2Term(
            min_val=args.kappa_min, max_val=args.kappa_max)

    @staticmethod
    def get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return ("Baseline2D_SineGordon"
                + terms.HomSpatialOrder2Term.arg_str(args))

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        # same for InhomSpatialOrder2Term
        terms.HomSpatialOrder2Term.add_cli_args_(
            parser, kappa_min=1e-2, kappa_max=4)
        return parser

    def get_dedalus_problem(self) -> Tuple:
        # Fields
        u_op = dist.Field(name="u", bases=self.bases)
        dt_u = dist.Field(name="dt_u", bases=self.bases)

        # Initial condition
        self.term_obj_dict["u_ic"].gen_dedalus_ops(u_op)
        self.term_obj_dict["ut_ic"].gen_dedalus_ops(dt_u)

        # PDE Terms
        diff_u = self.term_obj_dict["Lu"].gen_dedalus_ops(d3.grad(u_op))

        # Problem
        problem = d3.IVP([u_op, dt_u])
        problem.add_equation([d3.dt(u_op) - dt_u, 0])
        # Note RHS has no negation.
        problem.add_equation([d3.dt(dt_u) + diff_u, np.sin(u_op)])
        return {"u": u_op}, problem


if __name__ == "__main__":
    my_args = get_cli_args(SineGordon2DPDE)
    pde_data_obj = SineGordon2DPDE(my_args)
    gen_data(my_args, pde_data_obj)
