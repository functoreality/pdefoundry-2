r"""Boundary conditions for PDEs."""
from typing import Dict, List, Tuple, Union, Optional

from numpy.typing import NDArray
import dedalus.public as d3
from dedalus.core.field import Operand
from dedalus.core.basis import Basis

from . import coefs
from ...common import boundary
from ...common.boundary import exclude_entry


class EdgeBoundaryCondition(boundary.EdgeBoundaryCondition):
    __doc__ = boundary.EdgeBoundaryCondition.__doc__
    coef_cls: type = coefs.RandomConstOrField
    alpha: coef_cls
    beta: coef_cls

    def gen_dedalus_ops(self,
                        u_op: Operand = None,
                        dn_u: Operand = None,
                        alpha_op: d3.Field = None,
                        beta_op: d3.Field = None,
                        coords: Optional[Tuple[NDArray[float]]] = None,
                        ) -> Tuple[Operand]:
        if None in [u_op, dn_u, alpha_op, beta_op]:
            raise ValueError("All inputs [u_op, dn_u, alpha_op, beta_op] "
                             "must be specified.")
        if self.bc_type == self.ROBIN_D:
            op1, op2 = u_op, dn_u
        elif self.bc_type == self.ROBIN_N:
            op1, op2 = dn_u, u_op
        op2 = self.alpha.gen_dedalus_ops(alpha_op, coords) * op2

        lin_lhs = op1
        nlin_rhs = self.beta.gen_dedalus_ops(beta_op, coords)
        if self.alpha.is_const:
            lin_lhs = lin_lhs + op2
        else:
            # For disk_dcr, 'scalar + operand' causes error, while
            # 'operand + scalar' works well.
            nlin_rhs = -op2 + nlin_rhs
        return (lin_lhs, nlin_rhs)


class EdgeBCWithMur(boundary.EdgeBCWithMur):
    __doc__ = boundary.EdgeBCWithMur.__doc__
    coef_cls: type = coefs.RandomConstOrField
    alpha: coef_cls
    beta: coef_cls
    gamma: coef_cls

    def gen_dedalus_ops(self,
                        u_op: Operand = None,
                        dn_u: Operand = None,
                        ut_op: Operand = None,
                        alpha_op: d3.Field = None,
                        gamma_op: d3.Field = None,
                        beta_op: d3.Field = None,
                        coords: Optional[Tuple[NDArray[float]]] = None,
                        ) -> Tuple[Operand]:
        if None in [u_op, dn_u, ut_op, alpha_op, gamma_op, beta_op]:
            raise ValueError("All inputs [u_op, dn_u, ut_op, alpha_op, "
                             "gamma_op, beta_op] must be specified.")
        if self.bc_type != self.MUR_R:
            return EdgeBoundaryCondition.gen_dedalus_ops(
                self, u_op, dn_u, alpha_op, beta_op, coords=coords)

        lin_lhs = ut_op
        nlin_rhs = self.beta.gen_dedalus_ops(beta_op, coords)

        # u term
        u_term = self.alpha.gen_dedalus_ops(alpha_op, coords) * u_op
        if self.alpha.is_const:
            lin_lhs = lin_lhs + u_term
        else:
            nlin_rhs = -u_term + nlin_rhs

        # u_n term
        un_term = self.gamma.gen_dedalus_ops(gamma_op, coords) * dn_u
        if self.gamma.is_const:
            lin_lhs = lin_lhs + un_term
        else:
            nlin_rhs = -un_term + nlin_rhs

        return (lin_lhs, nlin_rhs)


class BoxDomainBoundaryCondition(boundary.BoxDomainBoundaryCondition):
    __doc__ = boundary.BoxDomainBoundaryCondition.__doc__
    edge_bc_cls: type = EdgeBoundaryCondition
    edge_term_dict: Dict[int, Tuple[edge_bc_cls]]  # update typing
    ic_obj: Optional[coefs.RandomField] = None

    def gen_dedalus_ops(self,
                        u_op: Operand = None,
                        grad_u_tuple: Tuple[Operand] = None,
                        dist: d3.Distributor = None,
                        bases: Tuple[Basis] = None) -> List[Tuple[Operand]]:
        if None in [u_op, grad_u_tuple, dist, bases]:
            raise ValueError("All inputs [u_op, grad_u_tuple, dist, bases] "
                             "must be specified.")
        bc_ops_list = []
        for i_ax, (bc_low, bc_high) in self.edge_term_dict.items():
            def new_field_op() -> dist.Field:
                return dist.Field(bases=exclude_entry(bases, i_ax))

            slicer_kwarg = {"xyzw"[i_ax]: "left"}
            bc_ops_list.append(bc_low.gen_dedalus_ops(
                u_op(**slicer_kwarg),
                -grad_u_tuple[i_ax](**slicer_kwarg),  # negation required
                new_field_op(),
                new_field_op()))
            slicer_kwarg = {"xyzw"[i_ax]: "right"}
            bc_ops_list.append(bc_high.gen_dedalus_ops(
                u_op(**slicer_kwarg),
                grad_u_tuple[i_ax](**slicer_kwarg),
                new_field_op(),
                new_field_op()))
        return bc_ops_list


class BoxDomainBCWithMur(boundary.BoxDomainBCWithMur):
    __doc__ = boundary.BoxDomainBCWithMur.__doc__
    edge_bc_cls: type = EdgeBCWithMur
    edge_term_dict: Dict[int, Tuple[edge_bc_cls]]  # update typing
    ut_ic_obj: Optional[coefs.RandomField] = None

    def gen_dedalus_ops(self,
                        u_op: Operand = None,
                        ut_op: Operand = None,
                        grad_u_tuple: Tuple[Operand] = None,
                        dist: d3.Distributor = None,
                        bases: Tuple[Basis] = None) -> List[Tuple[Operand]]:
        if None in [u_op, ut_op, grad_u_tuple, dist, bases]:
            raise ValueError("All inputs [u_op, grad_u_tuple, dist, bases] "
                             "must be specified.")
        bc_ops_list = []
        for i_ax, (bc_low, bc_high) in self.edge_term_dict.items():
            def new_field_op() -> dist.Field:
                return dist.Field(bases=exclude_entry(bases, i_ax))

            slicer_kwarg = {"xyzw"[i_ax]: "left"}
            bc_ops_list.append(bc_low.gen_dedalus_ops(
                u_op(**slicer_kwarg),
                -grad_u_tuple[i_ax](**slicer_kwarg),  # negation required
                ut_op(**slicer_kwarg),
                new_field_op(),
                new_field_op(),
                new_field_op()))
            slicer_kwarg = {"xyzw"[i_ax]: "right"}
            bc_ops_list.append(bc_high.gen_dedalus_ops(
                u_op(**slicer_kwarg),
                grad_u_tuple[i_ax](**slicer_kwarg),
                ut_op(**slicer_kwarg),
                new_field_op(),
                new_field_op(),
                new_field_op()))
        return bc_ops_list


def tau_polynomial(dist: d3.Distributor,
                   bases: Tuple[Basis],
                   periodic: Union[bool, Tuple[bool]] = False,
                   vector_tau: bool = False) -> Tuple:
    r"""
    Tau polynomials for non-periodic axes as required in Dedalus V3, following
    https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_3.html

    Returns:
        result_op (Operand)
        var_op_list (List[d3.Field])
    """
    n_dim = len(bases)
    if isinstance(periodic, bool):
        periodic = (periodic,) * n_dim
    result_op = 0
    var_op_list = []
    for i_ax in range(n_dim):
        if periodic[i_ax]:
            continue
        if vector_tau:
            tau1 = dist.VectorField(bases=exclude_entry(bases, i_ax))
            tau2 = dist.VectorField(bases=exclude_entry(bases, i_ax))
        else:
            tau1 = dist.Field(bases=exclude_entry(bases, i_ax))
            tau2 = dist.Field(bases=exclude_entry(bases, i_ax))
        var_op_list.extend([tau1, tau2])
        tau_basis = bases[i_ax].derivative_basis(2)
        p1_op = dist.Field(bases=tau_basis)
        p2_op = dist.Field(bases=tau_basis)
        if i_ax == 0:
            p1_op['c'][-1] = 1
            p2_op['c'][-2] = 2
        elif i_ax == 1:
            p1_op['c'][:, -1] = 1
            p2_op['c'][:, -2] = 2
        elif i_ax == 2:
            p1_op['c'][:, :, -1] = 1
            p2_op['c'][:, :, -2] = 2
        else:  # generalizing the above cases, but a bit harder to understand
            p1_op['c'][(slice(None),) * i_ax + (-1,)] = 1
            p2_op['c'][(slice(None),) * i_ax + (-2,)] = 2
        result_op = result_op + tau1 * p1_op + tau2 * p2_op

    return result_op, var_op_list


def tau_polynomial_raw(dist: d3.Distributor,
                       bases: Tuple[Basis],
                       periodic: Union[bool, Tuple[bool]] = False,
                       sum_bases: Tuple[Operand] = (1, 1, 1, 1),
                       vector_tau: bool = False) -> Tuple:
    r"""
    Tau polynomial for non-periodic axes as required in Dedalus V3, following
    https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_3.html

    Returns:
        result_op (Operand)
        var_op_list (List[d3.Field])
    """
    n_dim = len(bases)
    if isinstance(periodic, bool):
        periodic = (periodic,) * n_dim
    result_op = 0
    var_op_list = []
    for i_ax in range(n_dim):
        if periodic[i_ax]:
            continue
        if vector_tau:
            tau = dist.VectorField(bases=exclude_entry(bases, i_ax))
        else:
            tau = dist.Field(bases=exclude_entry(bases, i_ax))
        var_op_list.append(tau)
        lift_basis = bases[i_ax].derivative_basis(1)
        lift_tau = d3.Lift(tau, lift_basis, -1)
        result_op = result_op + sum_bases[i_ax] * lift_tau

    return result_op, var_op_list


DiskDomain = boundary.DiskDomain
