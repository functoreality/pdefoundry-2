r"""Settings for generating 2D PDE solution."""
from typing import Optional, Callable, List, Tuple
import numpy as np
import dedalus.public as d3
from dedalus.core.field import Operand
from dedalus.core.basis import Basis

# Parameters
X_L = 0
X_R = 1
N_X_GRID = 128

# Substitutions
d3_coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(d3_coords, dtype=np.float64)


def d3_dx(operand: Operand) -> Operand:
    r""" Obtain the x-derivative of a Dedalus operator. """
    return d3.Differentiate(operand, d3_coords['x'])


def d3_dy(operand: Operand) -> Operand:
    r""" Obtain the y-derivative of a Dedalus operator. """
    return d3.Differentiate(operand, d3_coords['y'])


def get_basis(coord_name: str,
              periodic: bool = True,
              basis_type: Optional[Callable] = None,
              dealias: float = 2.) -> Basis:
    r""" Obtain Dedalus basis for one coordinate. """
    if basis_type is None:
        basis_type = d3.RealFourier if periodic else d3.Chebyshev
    return basis_type(d3_coords[coord_name], size=N_X_GRID,
                      bounds=(X_L, X_R), dealias=dealias)


def get_bases(periodic: List[bool], dealias: float = 2.) -> Tuple[Basis]:
    r""" Obtain the Dedalus bases for both coordinates. """
    return tuple(get_basis(coord_name, coord_perioidic, dealias=dealias)
                 for coord_name, coord_perioidic in zip("xy", periodic))
