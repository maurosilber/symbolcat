"""
symbolcat.ops.egraph._assume
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assumptions about sub-expressions, passed as ``assume=[...]``.

By default, ``simplify`` and ``optimize`` only apply rules that are valid
wherever the expression is defined. Some identities need more:
``log(x * y) == log(x) + log(y)`` only when ``x`` and ``y`` are positive.
Assumptions enable those rules.

>>> from symbolcat import Symbol, real
>>> from symbolcat.ops.egraph import positive, simplify
>>> x, y = Symbol("x"), Symbol("y")
>>> print(simplify(real.log(x * y) - real.log(x)))
log(x * y) - log(x)
>>> print(simplify(real.log(x * y) - real.log(x), assume=[positive(x)]))
log(y)

Properties are also derived: literals, ``exp(a)`` and ``cosh(a)`` are
signed, sums and products of positives are positive, and so on.
Contradictory assumptions (``positive(x)`` and ``negative(x)``) make the
expression undefined, and it is returned unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..._core import Expr, as_expr

type Property = Literal["positive", "nonnegative", "negative", "nonzero", "integer"]


@dataclass(frozen=True)
class Assumption:
    property: Property
    expr: Expr


def positive(expr: Any) -> Assumption:
    """Assume ``expr > 0``."""
    return Assumption("positive", as_expr(expr))


def nonnegative(expr: Any) -> Assumption:
    """Assume ``expr >= 0``."""
    return Assumption("nonnegative", as_expr(expr))


def negative(expr: Any) -> Assumption:
    """Assume ``expr < 0``."""
    return Assumption("negative", as_expr(expr))


def nonzero(expr: Any) -> Assumption:
    """Assume ``expr != 0``."""
    return Assumption("nonzero", as_expr(expr))


def integer(expr: Any) -> Assumption:
    """Assume ``expr`` is an integer."""
    return Assumption("integer", as_expr(expr))
