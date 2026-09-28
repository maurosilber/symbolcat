"""
symbolcat.ops._calculus._time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The total time derivative.

The unknowns are functions of time, and their derivatives are written with
``D``. The total derivative is a directional derivative (``jvp``) whose seed
maps each unknown ``x`` to ``D(x)``, each ``D(x)`` to ``D(D(x))``, and time
to 1. Other symbols (parameters) are constant.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..._core import Call, Expr, Literal, Symbol
from ..._derivative import D
from .._analysis._inspect import variables
from ._diff import jvp


def _base(var: Expr, /) -> Expr:
    """The symbol of a time derivative: ``x`` for ``D(D(x))``."""
    while isinstance(var, Call):
        (var,) = var.args
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
    if t is not None:
        seed[t] = Literal(1)
    return jvp(expr, seed)
