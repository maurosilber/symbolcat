"""
symbolcat.ops._calculus._time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The total time derivative.

The unknowns are functions of time, and their derivatives are written with
``D``. The total derivative is a directional derivative (``jvp``) whose seed
maps each unknown ``x`` to ``D(x)``, each ``D(x)`` to ``D(D(x))``, and time
to 1. Other symbols (parameters) are constant.

``expand_derivatives`` rewrites ``D`` of an expression, ``D(x * y)``, as the
time derivative of the expression, so that ``D`` is only applied to
variables. ``change_variables`` substitutes the unknowns of differential
equations by expressions of new unknowns, and expands the derivatives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..._core import (
    Call,
    Done,
    Expr,
    Literal,
    Symbol,
    args_of,
    as_expr,
    cata,
    func_of,
    hylo,
)
from ...functions.calculus import D, delay, is_variable
from .._analysis._inspect import union, variables
from .._transform._substitute import substitute
from ._diff import jvp


def delays(expr: Any, /) -> tuple[Call[Expr], ...]:
    """The delayed values ``delay(x, tau)`` in ``expr``, in order of first
    appearance.

    >>> from symbolcat import Symbol, delay
    >>> x, tau = Symbol("x"), Symbol("tau")
    >>> delays(x + delay(x, tau) * delay(x, 2 * tau))
    (Call(delay, (Symbol('x'), Symbol('tau'))), Call(delay, (Symbol('x'), Call(mul, (Literal(2), Symbol('tau'))))))
    """

    # A delay is kept, and its arguments are unfolded in turn: it goes first
    # in its own layer so that it comes before the delays inside it.
    def coalgebra(node: Expr, /) -> Expr:
        if isinstance(node, Call) and func_of(node) is delay:
            return Call(delay, (Done({node: None}), *args_of(node)))
        return node

    return tuple(hylo(union, coalgebra, as_expr(expr)))


def _base(var: Expr, /) -> Expr:
    """The symbol of a time derivative: ``x`` for ``D(D(x))``."""
    while isinstance(var, Call):
        (var,) = args_of(var)
    return var


def time_derivative(
    expr: Any, unknowns: Iterable[Symbol], /, t: Symbol | None = None
) -> Expr:
    """Total derivative of ``expr`` with respect to time.

    ``unknowns`` are the symbols that depend on time; their derivatives are
    written with ``D``. ``t`` is the time symbol, if ``expr`` depends on it
    explicitly. Every other symbol is constant.

    >>> from symbolcat import D, Symbol, real
    >>> x, y, k, t = Symbol("x"), Symbol("y"), Symbol("k"), Symbol("t")
    >>> print(time_derivative(x**2 + y**2 - k, [x, y]))
    2 * x * D(x) + 2 * y * D(y)
    >>> print(time_derivative(D(x) - k * real.sin(t), [x], t))
    D(D(x)) - k * cos(t)
    """
    unknowns = set(unknowns)
    seed: dict[Expr, Expr] = {
        var: D(var) for var in variables(expr) if _base(var) in unknowns
    }
    # d/dt x(t - tau) = D(x)(t - tau), for a constant delay
    for node in delays(expr):
        value, tau = args_of(node)
        if _base(value) in unknowns:
            seed[node] = delay(D(value), tau)
    if t is not None:
        seed[t] = Literal(1)
    return jvp(expr, seed)


def expand_derivatives(
    expr: Any, unknowns: Iterable[Symbol], /, t: Symbol | None = None
) -> Expr:
    """Rewrite ``D`` of expressions as their time derivatives, so that ``D``
    is only applied to variables (symbols, or derivatives of them).

    ``unknowns`` and ``t`` are as in ``time_derivative``.

    >>> from symbolcat import D, Symbol
    >>> x, y, k = Symbol("x"), Symbol("y"), Symbol("k")
    >>> print(expand_derivatives(D(k * x * y) + D(D(x)), [x, y]))
    k * D(x) * y + k * x * D(y) + D(D(x))
    """
    unknowns = tuple(unknowns)

    def algebra(layer: Expr, /) -> Expr:
        if not isinstance(layer, Call):
            return layer
        node = Call(func_of(layer), args_of(layer))
        if func_of(node) is D and not is_variable(node):
            (arg,) = args_of(node)
            return time_derivative(arg, unknowns, t)
        return node

    return cata(algebra, expr)


def change_variables[S: Symbol](
    equations: Iterable[Any],
    mapping: Mapping[S, Any],
    unknowns: Iterable[Symbol],
    /,
    t: Symbol | None = None,
) -> list[Expr]:
    """Substitute each old unknown in ``mapping`` by an expression of the
    new ``unknowns``, and expand the derivatives.

    >>> from symbolcat import D, Symbol, real
    >>> x, r, theta = Symbol("x"), Symbol("r"), Symbol("theta")
    >>> (eq,) = change_variables([D(x)], {x: r * real.cos(theta)}, [r, theta])
    >>> print(eq)
    D(r) * cos(theta) + r * (-sin(theta) * D(theta))
    """
    unknowns = tuple(unknowns)
    return [
        expand_derivatives(substitute(eq, mapping), unknowns, t) for eq in equations
    ]
