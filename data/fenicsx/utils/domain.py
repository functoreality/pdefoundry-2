r"""Domain and boundary utilities for FEniCSx."""
from dolfinx import mesh
import numpy as np
from typing import Tuple, List, Dict

def extract_square_boundary(domain: mesh.Mesh,
                            corners: Tuple[float, float, float, float]
                            ) -> Tuple[List, Dict, Dict]:
    r"""
    Extract boundary facets of a square domain.

    Args:
        domain: Mesh object.
        corners: (x_min, x_max, y_min, y_max).

    Returns:
        bd_name: List of boundary names.
        bd_flag_dict: Dictionary of boundary flag functions.
        bd_facets_dict: Dictionary of boundary facets.
    """
    def l_bd(x):
        return np.isclose(x[0], 0)

    def r_bd(x):
        return np.isclose(x[0], 1)

    def b_bd(x):
        return np.isclose(x[1], 0)

    def t_bd(x):
        return np.isclose(x[1], 1)

    bd_name = ["left", "right", "bottom", "top"]
    bd_flag_dict = {bd_name[0]: l_bd, bd_name[1]: r_bd,
                    bd_name[2]: b_bd, bd_name[3]: t_bd}
    bd_facets_dict = {key: mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, bd_flag_dict[key]) for key in bd_name}
    out = (bd_name, bd_flag_dict, bd_facets_dict)
    return out
