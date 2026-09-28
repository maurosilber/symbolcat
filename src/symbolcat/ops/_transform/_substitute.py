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

from ..._core import Algebra, Call, Expr, as_expr, cata

substitute_algebra = Algebra[Expr]("substitute")


@substitute_algebra.register(Expr)
def _(node: Expr, *, mapping: Mapping[Any, Any]) -> Expr:
    return as_expr(mapping.get(node, node))


@substitute_algebra.register(Call)
def _(node: Call[Expr], *, mapping: Mapping[Any, Any]) -> Expr:
    func = mapping.get(node.func, node.func)
    if func is not node.func:
        return as_expr(func(*node.args))
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
