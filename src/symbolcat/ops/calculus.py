"""
symbolcat.ops.calculus
~~~~~~~~~~~~~~~~~~~~~~

Differentiation, in forward (``diff``, ``jvp``) and reverse (``vjp``) mode,
Taylor series, and time derivatives of expressions with ``D``.
"""

from ._calculus._diff import diff, gradient, hessian, jacobian, jvp
from ._calculus._reverse import reverse_gradient, vjp
from ._calculus._series import series
from ._calculus._time import expand_derivatives, time_derivative

__all__ = [
    "diff",
    "expand_derivatives",
    "gradient",
    "hessian",
    "jacobian",
    "jvp",
    "reverse_gradient",
    "series",
    "time_derivative",
    "vjp",
]
