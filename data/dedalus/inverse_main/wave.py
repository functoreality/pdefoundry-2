#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate inverse-problem dataset of the 2D wave PDE."""
import argparse
import numpy as np
from ...common import basics
from ..forward_main.wave import Wave2DPDE


class Wave2DInverse(Wave2DPDE):
    r"""
    Generate inverse-problem dataset of 2D time-dependent PDE solutions with
    Dedalus-v3.
    ======== Wave Equation ========
    """
    PREPROCESS_DAG = False
    IS_INVERSE = True

    def __init__(self, args):
        super().__init__(args)
        self.per_sample_src = args.per_sample_src

    @classmethod
    def get_hdf5_file_prefix(cls, args: argparse.Namespace) -> str:
        file_prefix = super().get_hdf5_file_prefix(args)
        if args.per_sample_src:
            file_prefix += "_psS"
        return file_prefix

    @classmethod
    def get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = super().get_cli_args_parser()
        parser.add_argument("--per_sample_src", action="store_true",
                            help="Generate the source term s(r) independently "
                            "for each sample, instead of using a shared "
                            "source. This is used in the FWI (full waveform inversion) task.")
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        super().reset_pde(rng)
        # Re-generate s to make sure it is a field.
        field_coef_type = self.term_obj_dict["s"].FIELD_COEF
        self.term_obj_dict["s"].reset(rng, coef_type=field_coef_type)

    def reset_sample(self, rng: np.random.Generator) -> None:
        self.term_obj_dict["u_ic"].reset(rng)
        self.term_obj_dict["ut_ic"].reset(rng)
        if self.per_sample_src:  # For FWI, we need to reset the source term
            field_coef_type = self.term_obj_dict["s"].FIELD_COEF
            self.term_obj_dict["s"].reset(rng, coef_type=field_coef_type)


if __name__ == "__main__":
    my_args = basics.get_cli_args(Wave2DInverse)
    pde_data_obj = Wave2DInverse(my_args)
    basics.gen_data(my_args, pde_data_obj)
