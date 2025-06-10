r"""Interpolation functions and solutions for FEniCSx."""
from dolfinx import mesh
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import numpy as np
from typing import Callable
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator, CubicSpline

boundary_1d = np.array([0, 1])
boundary_2d = np.array([[0, 0], [1, 1]])

def get_1d_interp_function(field: NDArray, coords: NDArray) -> Callable:
    r"""
    Get 1D interpolated function from field.

    Args:
        field: Field values. Shape (n, ) for scalar field or
        (n, k) for vector field.
        coords: Coordinates of the field. Shape (n, ).

    Returns:
        func: Interpolated function. for x of shape (m, ), func(x) returns
        interpolated field values of shape (m, ) or (m, k).
    """
    if field.ndim == 1:
        def func(x):
            x_ = np.clip(x, boundary_1d[0], boundary_1d[1])
            inner_func = CubicSpline(coords, field)
            return inner_func(x_)
    elif field.ndim == 2:
        def func(x):
            nvar = field.shape[1]
            x_ = np.clip(x, boundary_1d[0], boundary_1d[1])
            out = np.zeros((nvar, len(x_)))
            for i in range(nvar):
                inner_func = CubicSpline(coords, field[:, i])
                out[i] = inner_func(x_)
            return out
    else:
        raise ValueError("Field should be 1D or 2D.")
    return func

def get_2d_interp_function(
        field: NDArray, x_coord: NDArray, y_coord: NDArray) -> Callable:
    r"""
    Get 2D interpolated function from field. The input field should be of size
    (n_x, n_y) for scalar field or (n_x, n_y, k) for vector field.

    Args:
        field: Field values. Shape (n_x, n_y) for scalar field or
        (n_x, n_y, k) for vector field.
        x_coord: x coordinates of the field. Shape (n_x, ).
        y_coord: y coordinates of the field. Shape (n_y, ).

    Returns:
        func: Interpolated function. for x of shape (2, m) or (3, m),
        func(x) returns interpolated field values of shape (m, ) or (m, k).
    """
    if field.ndim == 2:
        def func(x):
            x_ = np.clip(x[:2].T, boundary_2d[0], boundary_2d[1])
            inner_func = RegularGridInterpolator((x_coord, y_coord), field)
            return inner_func(x_)
    elif field.ndim == 3:
        n_var = field.shape[-1]
        def func(x):
            x_ = np.clip(x[:2].T, boundary_2d[0], boundary_2d[1])
            out = np.zeros((n_var, x_.shape[0]))
            for i in range(n_var):
                inner_func_i = RegularGridInterpolator(
                    (x_coord, y_coord), field[..., i])
                out[i] = inner_func_i(x_)
            return out
    else:
        raise ValueError("Field should be 2D or 3D.")
    return func

def locate_cells(domain: mesh.Mesh,
                 points: NDArray[float]) -> NDArray[int]:
    r"""
    Locate cells in the mesh given points.

    For fem Function f, domain and points, use
    >>> cells = locate_cells(domain, points)
    >>> f.eval(points, cells)
    to get the interpolated values of f at points.

    Args:
        domain: Mesh of the domain.
        points: Points to locate. Shape (n, 3). For 2D mesh, first two columns
        are used.

    Returns:
        cells: Cell indices of the mesh. Shape (n, ).
    """
    tree = bb_tree(domain, domain.topology.dim)
    cells = []
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(domain, cell_candidates, points)

    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
        else:
            raise ValueError(f"Point {point} is outside of the domain.")
    return np.array(cells)
