"""
symbolcat._derivative
~~~~~~~~~~~~~~~~~~~~~

The time derivative as an expression node.

``D(x)`` is the derivative of ``x`` with respect to time, and ``D(D(x))`` the
second derivative. It is a node, not an operation: ``D(x)`` stays as written,
so it can appear in equations (``D(x) - v``) as an unknown of its own.
``ops.calculus.time_derivative`` computes the derivative of an expression, and writes
the derivatives of the unknowns with ``D``.

>>> from symbolcat import D, Symbol
>>> x, v = Symbol("x"), Symbol("v")
>>> print(D(x) - v)
D(x) - v

Two more nodes build other kinds of differential equations:

- ``delay(x, tau)``: the value of ``x`` at time ``t - tau``, for delay
  differential equations. It is independent of ``x``.
- ``partial(u, x)``: the partial derivative of ``u`` with respect to a
  spatial coordinate ``x``, for partial differential equations.
"""

from __future__ import annotations

from typing import Any, TypeGuard

from ._core import Call, Expr, Symbol, args_of, func_of, name_of
from ._core.infer import kind_algebra
from ._core.kinds import Kind, Real, array_kind, join
from ._core.signatures import Delay, Derivative

D = Derivative("D", arity=1)
delay = Delay("delay", arity=2)
partial = Delay("partial", arity=2)


@kind_algebra.register(D, delay, partial)
def _(node: Call[Kind]) -> Kind:
    """The kind of the first argument; the time delay and the spatial
    coordinate are scalars."""
    value, *rest = args_of(node)
    for kind in rest:
        if kind.shape:
            raise ValueError(
                f"{name_of(func_of(node))} needs a scalar second argument, "
                f"got shape {kind.shape}"
            )
    return array_kind(value.shape, join(Real(), value.element))


type Variable = Symbol | Call[Expr]
"""A symbol or a time derivative of one (see ``is_variable``)."""


def is_variable(expr: Any, /) -> TypeGuard[Variable]:
    """Whether ``expr`` is a symbol or a time derivative of one, ``D(D(x))``.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> is_variable(D(D(x))), is_variable(D(2 * x))
    (True, False)
    """
    while isinstance(expr, Call) and func_of(expr) is D:
        (expr,) = args_of(expr)
    return isinstance(expr, Symbol)
