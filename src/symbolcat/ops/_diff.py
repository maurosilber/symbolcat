"""
symbolcat.ops._diff
~~~~~~~~~~~~~~~~~~~

Symbolic differentiation.

Differentiating needs the value of each sub-expression, not only its
derivative (the product rule uses both). The algebra therefore folds every
node into a ``(value, derivative)`` pair: a catamorphism into a product
type, the same trick used by forward-mode automatic differentiation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .. import real
from .._core import Algebra, Call, Constant, Expr, Function, Literal, Symbol, cata
from ..lib import operators as op
from ._simplify import is_number, simplify_naive

type Dual = tuple[Expr, Expr]
type Rule = Callable[[tuple[Expr, ...], tuple[Expr, ...]], Expr]

ZERO = Literal(0)
ONE = Literal(1)

diff_algebra = Algebra[Dual]("diff")


@diff_algebra.register(Symbol)
def _(node: Symbol, *, var: Symbol) -> Dual:
    return node, ONE if node == var else ZERO


@diff_algebra.register(Constant, Literal)
def _(node: Constant | Literal, *, var: Symbol) -> Dual:
    return node, ZERO


def rule(*funcs: Function) -> Callable[[Rule], Rule]:
    """Register a derivative rule ``(args, dargs) -> derivative``."""

    def decorator(derivative: Rule) -> Rule:
        @diff_algebra.register(*funcs)
        def _(node: Call[Dual], *, var: Symbol) -> Dual:
            args = tuple(value for value, _ in node.args)
            dargs = tuple(d for _, d in node.args)
            return node.func(*args), derivative(args, dargs)

        return derivative

    return decorator


@rule(op.add)
def _(args, dargs):
    return dargs[0] + dargs[1]


@rule(op.sub)
def _(args, dargs):
    return dargs[0] - dargs[1]


@rule(op.neg)
def _(args, dargs):
    return -dargs[0]


@rule(op.pos)
def _(args, dargs):
    return dargs[0]


@rule(op.mul)
def _(args, dargs):
    (a, b), (da, db) = args, dargs
    return da * b + a * db


@rule(op.truediv)
def _(args, dargs):
    (a, b), (da, db) = args, dargs
    return (da * b - a * db) / b**2


@rule(op.pow)
def _(args, dargs):
    (a, b), (da, db) = args, dargs
    if is_number(simplify_naive(db), 0):
        return b * a ** (b - 1) * da
    return a**b * (db * real.log(a) + b * da / a)


def chain(outer: Callable[[Expr], Expr]) -> Rule:
    """Rule for f(a), given f'(a)."""
    return lambda args, dargs: outer(args[0]) * dargs[0]


rule(real.exp)(chain(real.exp))
rule(real.log)(chain(lambda a: 1 / a))
rule(real.sqrt)(chain(lambda a: 1 / (2 * real.sqrt(a))))
rule(real.sin)(chain(real.cos))
rule(real.cos)(chain(lambda a: -real.sin(a)))
rule(real.tan)(chain(lambda a: 1 / real.cos(a) ** 2))
rule(real.asin)(chain(lambda a: 1 / real.sqrt(1 - a**2)))
rule(real.acos)(chain(lambda a: -1 / real.sqrt(1 - a**2)))
rule(real.atan)(chain(lambda a: 1 / (1 + a**2)))
rule(real.sinh)(chain(real.cosh))
rule(real.cosh)(chain(real.sinh))
rule(real.tanh)(chain(lambda a: 1 - real.tanh(a) ** 2))


def diff(expr: Any, var: Symbol) -> Expr:
    """Derivative of ``expr`` with respect to ``var``.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(diff(x**3 + y * x, x))
    3 * x ** 2 + y
    >>> print(diff(real.sin(x) * x, x))
    cos(x) * x + sin(x)
    """
    _, derivative = cata(diff_algebra, expr, var=var)
    return simplify_naive(derivative)
