#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Generate inverse-problem dataset of the 2D wave PDE."""
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

    def reset_pde(self, rng: np.random.Generator) -> None:
        super().reset_pde(rng)
        field_coef_type = self.term_obj_dict["s"].FIELD_COEF
        self.term_obj_dict["s"].reset(rng, coef_type=field_coef_type)

    def reset_sample(self, rng: np.random.Generator) -> None:
        self.term_obj_dict["u_ic"].reset(rng)
        self.term_obj_dict["ut_ic"].reset(rng)


if __name__ == "__main__":
    my_args = basics.get_cli_args(Wave2DInverse)
    pde_data_obj = Wave2DInverse(my_args)
    basics.gen_data(my_args, pde_data_obj)
