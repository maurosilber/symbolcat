"""
symbolcat.backends._std
~~~~~~~~~~~~~~~~~~~~~~~

Python's standard library as implementation.
"""

import builtins
import cmath
import math
import operator
from typing import Any

from .. import complex as cx
from .. import operators as op
from .. import real


def gammainc(a: float, x: float, /) -> float:
    """The regularized lower incomplete gamma function, ``P(a, x)``.

    A series for ``x < a + 1``, and a continued fraction for ``Q = 1 - P``
    otherwise (Numerical Recipes, 6.2).

    >>> round(gammainc(1, 2.0), 12) == round(1 - math.exp(-2.0), 12)
    True
    """
    if a <= 0 or x < 0:
        raise ValueError(f"gammainc needs a > 0 and x >= 0, got {a}, {x}")
    if x == 0:
        return 0.0
    log_prefactor = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1:
        term = total = 1 / a
        n = a
        while abs(term) > abs(total) * 1e-16:
            n += 1
            term *= x / n
            total += term
        return total * math.exp(log_prefactor)
    # modified Lentz's method
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return 1 - math.exp(log_prefactor) * h


OPERATORS: dict[Any, Any] = {
    op.eq: operator.eq,
    op.ne: operator.ne,
    op.lt: operator.lt,
    op.le: operator.le,
    op.gt: operator.gt,
    op.ge: operator.ge,
    op.or_: operator.or_,
    op.xor: operator.xor,
    op.and_: operator.and_,
    op.lshift: operator.lshift,
    op.rshift: operator.rshift,
    op.add: operator.add,
    op.sub: operator.sub,
    op.mul: operator.mul,
    op.matmul: operator.matmul,
    op.truediv: operator.truediv,
    op.floordiv: operator.floordiv,
    op.mod: operator.mod,
    op.pow: operator.pow,
    op.neg: operator.neg,
    op.pos: operator.pos,
    op.invert: operator.invert,
    op.getitem: operator.getitem,
}

IMPL: dict[Any, Any] = {
    **OPERATORS,
    real.abs: builtins.abs,
    real.ceil: math.ceil,
    real.floor: math.floor,
    real.sqrt: math.sqrt,
    real.exp: math.exp,
    real.log: math.log,
    real.log2: math.log2,
    real.log10: math.log10,
    real.sin: math.sin,
    real.cos: math.cos,
    real.tan: math.tan,
    real.asin: math.asin,
    real.acos: math.acos,
    real.atan: math.atan,
    real.atan2: math.atan2,
    real.sinh: math.sinh,
    real.cosh: math.cosh,
    real.tanh: math.tanh,
    real.erf: math.erf,
    real.gamma: math.gamma,
    real.gammainc: gammainc,
    real.factorial: math.factorial,
    real.hypot: math.hypot,
    real.ifelse: lambda condition, a, b: a if condition else b,
    real.constants.e: math.e,
    real.constants.pi: math.pi,
    real.constants.tau: math.tau,
    real.constants.inf: math.inf,
    real.constants.nan: math.nan,
    cx.re: lambda z: z.real,
    cx.im: lambda z: z.imag,
    cx.conj: lambda z: z.conjugate(),
    cx.abs: builtins.abs,
    cx.phase: cmath.phase,
    cx.rect: cmath.rect,
    cx.exp: cmath.exp,
    cx.log: cmath.log,
    cx.log10: cmath.log10,
    cx.sqrt: cmath.sqrt,
    cx.sin: cmath.sin,
    cx.cos: cmath.cos,
    cx.tan: cmath.tan,
    cx.asin: cmath.asin,
    cx.acos: cmath.acos,
    cx.atan: cmath.atan,
    cx.sinh: cmath.sinh,
    cx.cosh: cmath.cosh,
    cx.tanh: cmath.tanh,
    cx.constants.I: 1j,
}
