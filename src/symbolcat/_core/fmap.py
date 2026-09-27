"""
symbolcat._core.fmap
~~~~~~~~~~~~~~~~~~~~

The functor map: apply a function to the direct children of a node.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import singledispatch
from typing import Any

from .expr import Call, Expr


@singledispatch
def fmap(node: Any, f: Callable[[Any], Any], /) -> Any:
    """Apply ``f`` to the direct children, keeping the node shape.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> fmap(x + y, lambda child: child.name)
    Call(add, ('x', 'y'))
    """
    raise TypeError(f"{type(node).__name__} is not an expression node")


@fmap.register(Expr)
def _(node: Expr, f: Callable[[Any], Any]) -> Expr:
    return node


@fmap.register(Call)
def _[T, S](node: Call[T], f: Callable[[T], S]) -> Call[S]:
    return Call(node.func, tuple(map(f, node.args)))
