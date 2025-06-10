r"""Frequently used PDE terms with randomized coefficients."""
from typing import List, Tuple, Union, Callable, Optional

import numpy as np
from numpy.typing import NDArray
import dedalus.public as d3
from dedalus.core.field import Operand

from ...common import terms


class PolySinusNonlinTerm(terms.PolySinusNonlinTerm):
    __doc__ = terms.PolySinusNonlinTerm.__doc__

    def gen_dedalus_ops(self, u_op: d3.Field = None) -> Tuple[Operand]:
        if u_op is None:
            raise ValueError("Argument 'u_op' must be specified.")
        value = self.value

        # polynomial part
        lin_op = value[0, 1] * u_op
        u2_op = u_op**2
        nonlin_op = value[0, 2] * u2_op + value[0, 3] * u_op**3

        # sinusoidal part
        for j in range(1, value.shape[0]):
            if value[j, 0] == 0:
                continue
            op_j = value[j, 1] * u_op + value[j, 2] * u2_op
            if value[j, 3] > 0:  # UNIT_COEF or positive SCALAR_COEF
                gj_op = np.sin(op_j)
            else:
                gj_op = np.cos(op_j)  # ZERO_COEF or negative SCALAR_COEF
            nonlin_op = nonlin_op + value[j, 0] * gj_op

        return lin_op, nonlin_op


class HomSpatialOrder2Term(terms.HomSpatialOrder2Term):
    __doc__ = terms.HomSpatialOrder2Term.__doc__

    def gen_dedalus_ops(self, grad_u: Operand = None) -> Operand:
        c_or_c2 = self.value.item()
        if c_or_c2 == 0:
            return 0.

        if grad_u is None:
            raise ValueError("Input 'grad_u' must be specified.")
        if c_or_c2 == 1:
            return -d3.div(grad_u)

        if self.diff_type == self.FACTORED_FORM:
            c_or_c2 = np.sqrt(c_or_c2)  # c rather than c^2

        # compute the 2nd-order differential term
        if self.diff_type == self.NON_DIV_FORM:
            return -c_or_c2 * d3.div(grad_u)
        if self.diff_type == self.FACTORED_FORM:
            return -c_or_c2 * d3.div(c_or_c2 * grad_u)
        if self.diff_type == self.DIV_FORM:
            return -d3.div(c_or_c2 * grad_u)
        raise NotImplementedError


class InhomSpatialOrder2Term(terms.InhomSpatialOrder2Term):
    __doc__ = terms.InhomSpatialOrder2Term.__doc__

    def gen_dedalus_ops(self,
                        field_op: d3.Field = None,
                        grad_u: Operand = None,
                        coords: Optional[Tuple[NDArray[float]]] = None,
                        reduce_fn: Callable[[NDArray[float]], float] = np.mean,
                        ) -> Tuple[Operand]:
        if grad_u is None:
            raise ValueError("Input 'grad_u' must be specified.")
        if field_op is None:
            raise ValueError("Input 'field_op' must be specified.")

        field = self.interpolated_field(coords)
        laplace_u = d3.div(grad_u)

        # factored case
        if self.diff_type == self.FACTORED_FORM:
            field = np.sqrt(field)
            scalar_val = reduce_fn(field)
            lin_op = -(scalar_val**2) * laplace_u
            field_op["g"] = field
            nonlin_op = -field_op * d3.div(field_op * grad_u) - lin_op
            return lin_op, nonlin_op

        # other cases
        scalar_val = reduce_fn(field)
        lin_op = -scalar_val * laplace_u
        field_op["g"] = field - scalar_val
        if self.diff_type == self.NON_DIV_FORM:
            nonlin_op = -field_op * laplace_u
        elif self.diff_type == self.DIV_FORM:
            nonlin_op = -d3.div(field_op * grad_u)
        return lin_op, nonlin_op


class MultiComponentLinearTerm(terms.MultiComponentLinearTerm):
    __doc__ = terms.MultiComponentLinearTerm.__doc__

    def gen_dedalus_ops(self, u_list: List[d3.Field] = None) -> List[Operand]:
        if u_list is None:
            raise ValueError("Argument 'u_list' must be specified.")
        if self.n_vars != len(u_list):
            raise ValueError("Length of 'u_list' should be equal to 'n_vars'.")

        op_list = [0 for _ in u_list]
        for (i, j, val) in zip(self.coo_i, self.coo_j, self.coo_vals):
            op_list[i] = op_list[i] + val * u_list[j]
        return op_list


class MultiComponentDegree2Term(terms.MultiComponentDegree2Term):
    __doc__ = terms.MultiComponentDegree2Term.__doc__

    def gen_dedalus_ops(self, u_list: List[d3.Field] = None) -> List[Operand]:
        if u_list is None:
            raise ValueError("Argument 'u_list' must be specified.")
        if self.n_vars != len(u_list):
            raise ValueError("Length of 'u_list' should be equal to 'n_vars'.")

        op_list = [0 for _ in u_list]
        for (i, j, k, val) in zip(self.coo_i, self.coo_j, self.coo_k, self.coo_vals):
            op_list[i] = op_list[i] + val * u_list[j] * u_list[k]
        return op_list


class MultiComponentMixedTerm(terms.MultiComponentMixedTerm):
    __doc__ = terms.MultiComponentMixedTerm.__doc__
    lin_term: MultiComponentLinearTerm
    deg2_term: MultiComponentDegree2Term

    def __init__(self,
                 n_vars: int = 2,
                 max_len: Optional[int] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(n_vars, max_len, coef_distribution, coef_magnitude)
        self.lin_term = MultiComponentLinearTerm(
            n_vars, max_len, coef_distribution, coef_magnitude)
        self.deg2_term = MultiComponentDegree2Term(
            n_vars, max_len, coef_distribution, coef_magnitude)

    def gen_dedalus_ops(self, u_list: List[d3.Field] = None) -> List[Tuple[Operand]]:
        lin_op_list = self.lin_term.gen_dedalus_ops(u_list)
        deg2_op_list = self.deg2_term.gen_dedalus_ops(u_list)
        return list(zip(lin_op_list, deg2_op_list))


class MCompnConvecDegree2Term(MultiComponentDegree2Term):
    __doc__ = terms.MCompnConvecDegree2Term.__doc__

    def gen_dedalus_ops(self,
                        u_list: List[d3.Field] = None,
                        diff: Callable[[d3.Field], Operand] = None,
                        ) -> List[Operand]:
        if u_list is None or diff is None:
            raise ValueError("Argument 'u_list' and 'diff' must be specified.")
        if self.n_vars != len(u_list):
            raise ValueError("Length of 'u_list' should be equal to 'n_vars'.")

        op_list = [0 for _ in u_list]
        for (i, j, k, val) in zip(self.coo_i, self.coo_j, self.coo_k, self.coo_vals):
            op_list[i] = op_list[i] + val * u_list[j] * diff(u_list[k])
        return op_list


class MCompnConvecMixedTerm(MultiComponentMixedTerm):
    __doc__ = terms.MCompnConvecMixedTerm.__doc__
    deg2_term: MCompnConvecDegree2Term

    def __init__(self,
                 n_vars: int = 2,
                 max_len: Optional[int] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(n_vars, max_len, coef_distribution, coef_magnitude)
        self.deg2_term = MCompnConvecDegree2Term(
            n_vars, max_len, coef_distribution, coef_magnitude)

    def gen_dedalus_ops(self,
                        u_list: List[d3.Field] = None,
                        diff: Callable[[d3.Field], Operand] = None,
                        ) -> List[Tuple[Operand]]:
        lin_op_list = self.lin_term.gen_dedalus_ops(u_list)
        deg2_op_list = self.deg2_term.gen_dedalus_ops(u_list, diff)
        return list(zip(lin_op_list, deg2_op_list))
