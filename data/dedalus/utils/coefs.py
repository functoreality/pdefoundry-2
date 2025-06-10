r"""Commonly used randomized PDE coefficients."""
from typing import Union, Callable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
import dedalus.public as d3

from ...common import coefs


class RandomValue(coefs.RandomValue):
    __doc__ = coefs.RandomValue.__doc__

    def gen_dedalus_ops(self) -> Union[float, NDArray[float]]:
        if self.size is None:
            return self.value.item()
        return self.value


class NonNegRandomValue(coefs.NonNegRandomValue):
    __doc__ = coefs.NonNegRandomValue.__doc__

    def gen_dedalus_ops(self) -> Union[float, NDArray[float]]:
        if self.size is None:
            return self.value.item()
        return self.value


class RandomField(coefs.RandomField):
    __doc__ = coefs.RandomField.__doc__

    def gen_dedalus_ops(self,
                        field_op: d3.Field = None,
                        coords: Optional[Tuple[NDArray[float]]] = None,
                        ) -> d3.Field:
        if field_op is None:
            raise ValueError("Input 'field_op' must be specified.")
        field = self.interpolated_field(coords)
        field_op["g"] = field.reshape(field_op["g"].shape)
        return field_op


class RandomConstOrField(coefs.RandomConstOrField):
    __doc__ = coefs.RandomConstOrField.__doc__

    def gen_dedalus_ops(self,
                        field_op: d3.Field = None,
                        coords: Optional[Tuple[NDArray[float]]] = None,
                        ) -> Union[float, d3.Field]:
        if self.coef_type == self.ZERO_COEF:
            return 0.
        if self.coef_type == self.UNIT_COEF:
            return 1.
        if self.coef_type == self.SCALAR_COEF:
            return self.field.flat[0]
        return RandomField.gen_dedalus_ops(self, field_op, coords)


class NonNegField(coefs.NonNegField):
    __doc__ = coefs.NonNegField.__doc__
    gen_dedalus_ops = RandomField.gen_dedalus_ops

    """
    def gen_dedalus_ops(self,
                        field_op: d3.Field = None,
                        coords: Optional[Tuple[NDArray[float]]] = None,
                        reduce_fn: Callable[[NDArray[float]], float] = np.mean,
                        ) -> d3.Field:
        if field_op is None:
            raise ValueError("Input 'field_op' must be specified.")
        field = self.interpolated_field(coords)
        scalar_val = reduce_fn(field)
        field_op["g"] = field.reshape(field_op["g"].shape) - scalar_val
        return scalar_val, field_op
    """


class NonNegConstOrField(coefs.NonNegConstOrField):
    __doc__ = coefs.NonNegConstOrField.__doc__
    # Do not split into scalar part and mean-zero field part.
    gen_dedalus_ops = RandomConstOrField.gen_dedalus_ops
