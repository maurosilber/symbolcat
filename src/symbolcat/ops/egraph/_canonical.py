"""
symbolcat.ops.egraph._canonical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Put extracted products in a conventional order.

Extraction cannot tell apart equally cheap orderings of a product
(``3 * (x * y)``, ``y * x * 3``, ...). This algebra flattens chains of
products and sorts the factors: numbers, then named constants, then symbols
alphabetically, then everything else in their original order. A leading
fraction becomes a division (``pi / 6``, not ``1 / 6 * pi``).

Flattening rebuilds products, so it must not be used on DAG extractions:
a factor shared between products would be duplicated.
"""

from __future__ import annotations

from math import prod
from typing import Any

from ..._core import Algebra, Call, Constant, Expr, Literal, Symbol, cata
from ...lib import operators as op

canonical_algebra = Algebra[Expr]("canonical")


@canonical_algebra.register(Expr)
def _(node: Expr) -> Expr:
    return node


def factors(expr: Expr, /) -> list[Expr]:
    if isinstance(expr, Call) and expr.func is op.mul:
        return [f for arg in expr.args for f in factors(arg)]
    return [expr]


def is_number(expr: Expr, /) -> bool:
    """A literal, or a fraction of literals (how rationals come back)."""
    if isinstance(expr, Call) and expr.func is op.truediv:
        return all(isinstance(arg, Literal) for arg in expr.args)
    return isinstance(expr, Literal)


def rank(expr: Expr, /) -> tuple[int, str]:
    if is_number(expr):
        return 0, ""
    if isinstance(expr, Constant):
        return 1, expr.name
    if isinstance(expr, Symbol):
        return 2, expr.name
    return 3, ""


@canonical_algebra.register(op.mul)
def _(node: Call[Expr]) -> Expr:
    ordered = sorted((f for arg in node.args for f in factors(arg)), key=rank)
    # multiply literal factors: 2 * 2 * x -> 4 * x
    numbers = [f for f in ordered if isinstance(f, Literal)]
    if len(numbers) > 1 and all(isinstance(f.value, int | float) for f in numbers):
        product = Literal(prod(f.value for f in numbers))
        ordered = [product, *(f for f in ordered if not isinstance(f, Literal))]
    # a leading fraction is written as a division: pi / 6, not 1 / 6 * pi
    denominator = None
    first = ordered[0]
    if isinstance(first, Call) and first.func is op.truediv and len(ordered) > 1:
        numerator, denominator = first.args
        ordered = ordered[1:] if numerator == Literal(1) else [numerator, *ordered[1:]]
    result = ordered[0]
    for factor in ordered[1:]:
        result = op.mul(result, factor)
    return result if denominator is None else op.truediv(result, denominator)


def canonical(expr: Any, /) -> Expr:
    """Flatten and sort products.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(canonical(y * (x * 3) * real.constants.pi))
    3 * pi * x * y
    """
    return cata(canonical_algebra, expr)
