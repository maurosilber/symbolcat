"""
symbolcat.ops._transform._substitute
~~~~~~~~~~~~~~~~~~~~~~~~~

Replace symbols, sub-expressions or functions.

The layer handed to the algebra is the node with its children already
substituted, so replacing it is a single dictionary lookup.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..._core import Algebra, Call, Expr, args_of, as_expr, cata, func_of

substitute_algebra = Algebra[Expr]("substitute")


@substitute_algebra.register(Expr)
def _(node: Expr, *, mapping: Mapping[Any, Any]) -> Expr:
    return as_expr(mapping.get(node, node))


@substitute_algebra.register(Call)
def _(node: Call[Expr], *, mapping: Mapping[Any, Any]) -> Expr:
    func = mapping.get(func_of(node), func_of(node))
    if func is not func_of(node):
        return as_expr(func(*args_of(node)))
    return as_expr(mapping.get(node, node))


def substitute(expr: Any, mapping: Mapping[Any, Any], /) -> Expr:
    """Replace symbols, sub-expressions or functions.

    Replacement is simultaneous, so symbols can be swapped.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(substitute(x + 2 * y, {x: y, y: x}))
    y + 2 * x
    >>> print(substitute(real.sin(x) + 1, {real.sin: real.cos, 1: 2}))
    cos(x) + 2
    """
    mapping = {as_expr(k) if not callable(k) else k: v for k, v in mapping.items()}
    return cata(substitute_algebra, expr, mapping=mapping)


def substitute_variables(expr: Any, mapping: Mapping[Any, Any], /) -> Expr:
    """Replace variables (symbols, and time derivatives of them) or
    sub-expressions, matching the original nodes.

    Unlike ``substitute``, a variable is atomic: ``D(x)`` is independent of
    ``x``, so replacing ``x`` leaves ``D(x)`` alone, and replacing ``D(x)``
    leaves ``D(D(x))`` alone.

    >>> from symbolcat import D, Symbol
    >>> x, a, b = Symbol("x"), Symbol("a"), Symbol("b")
    >>> print(substitute_variables(D(D(x)) + D(x) + x, {D(x): a, x: b}))
    D(D(x)) + a + b
    """
    from ..._derivative import is_variable

    mapping = {as_expr(k): as_expr(v) for k, v in mapping.items()}

    def algebra(layer: Expr, /) -> tuple[Expr, Expr]:
        if not isinstance(layer, Call):
            return layer, mapping.get(layer, layer)
        original = Call(func_of(layer), tuple(o for o, _ in args_of(layer)))
        if original in mapping:
            return original, mapping[original]
        if is_variable(original):
            return original, original
        return original, Call(func_of(layer), tuple(r for _, r in args_of(layer)))

    _, replaced = cata(algebra, expr)
    return replaced
