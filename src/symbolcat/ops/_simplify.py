"""
symbolcat.ops._simplify
~~~~~~~~~~~~~~~~~~~~~~~

Constant folding and algebraic identities (``x + 0``, ``x * 1``, ...).

Children arrive already simplified, so each rule only inspects one layer
and a single bottom-up pass is enough. Because it works on a single layer,
``simplify_algebra`` can also be used as a "smart constructor":
``simplify_algebra(x * 1)`` returns ``x``.
"""

from __future__ import annotations

from typing import Any

from .._core import Algebra, Call, Expr, Literal, cata
from ..lib import operators as op
from ..lib.std import IMPL

simplify_algebra = Algebra[Expr]("simplify")


def is_number(expr: Expr, value: Any, /) -> bool:
    return (
        isinstance(expr, Literal)
        and isinstance(expr.value, int | float | complex)
        and not isinstance(expr.value, bool)
        and expr.value == value
    )


@simplify_algebra.register(Expr)
def fold(node: Expr, /) -> Expr:
    """Evaluate calls whose arguments are all numeric literals."""
    if not isinstance(node, Call) or node.func not in IMPL:
        return node
    if not all(isinstance(arg, Literal) for arg in node.args):
        return node
    func = IMPL[node.func]
    if not callable(func):
        return node
    try:
        return Literal(func(*(arg.value for arg in node.args)))
    except (ArithmeticError, ValueError, TypeError):
        return node


@simplify_algebra.register(op.add)
def _(node: Call[Expr]) -> Expr:
    a, b = node.args
    if is_number(a, 0):
        return b
    if is_number(b, 0):
        return a
    return fold(node)


@simplify_algebra.register(op.sub)
def _(node: Call[Expr]) -> Expr:
    a, b = node.args
    if is_number(b, 0):
        return a
    if is_number(a, 0):
        return simplify_algebra(-b)
    if a == b:
        return Literal(0)
    return fold(node)


@simplify_algebra.register(op.mul)
def _(node: Call[Expr]) -> Expr:
    a, b = node.args
    if is_number(a, 0) or is_number(b, 0):
        return Literal(0)
    if is_number(a, 1):
        return b
    if is_number(b, 1):
        return a
    if is_number(a, -1):
        return simplify_algebra(-b)
    return fold(node)


@simplify_algebra.register(op.truediv)
def _(node: Call[Expr]) -> Expr:
    a, b = node.args
    if is_number(b, 1):
        return a
    if is_number(a, 0) and not is_number(b, 0):
        return Literal(0)
    if a == b and not isinstance(a, Literal):
        return Literal(1)
    return fold(node)


@simplify_algebra.register(op.pow)
def _(node: Call[Expr]) -> Expr:
    a, b = node.args
    if is_number(b, 0):
        return Literal(1)
    if is_number(b, 1) or is_number(a, 1):
        return a
    return fold(node)


@simplify_algebra.register(op.neg)
def _(node: Call[Expr]) -> Expr:
    (a,) = node.args
    if isinstance(a, Call) and a.func is op.neg:
        return a.args[0]
    return fold(node)


@simplify_algebra.register(op.pos)
def _(node: Call[Expr]) -> Expr:
    return node.args[0]


def simplify_naive(expr: Any, /) -> Expr:
    """Fold constants and remove trivial operations.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> print(simplify_naive(0 * x + 1 * (x ** (3 - 2)) + 2 * 3))
    x + 6
    """
    return cata(simplify_algebra, expr)
