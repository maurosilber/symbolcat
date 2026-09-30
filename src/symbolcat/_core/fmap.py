"""
symbolcat._core.fmap
~~~~~~~~~~~~~~~~~~~~

The functor map: apply a function to the direct children of a node.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import singledispatch
from typing import Any

from .expr import Call, Expr, args_of, func_of


@singledispatch
def fmap(node: Any, f: Callable[[Any], Any], /) -> Any:
    """Apply ``f`` to the direct children, keeping the node shape.

    When the new children are expressions, the result is a new node, and its
    kind is inferred again. Otherwise, it is a layer, which keeps the kind of
    ``node``.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> from symbolcat.ops.algebras import name_of
    >>> fmap(x + y, name_of)
    Call(add, ('x', 'y'))
    """
    raise TypeError(f"{type(node).__name__} is not an expression node")


@fmap.register(Expr)
def _(node: Expr, f: Callable[[Any], Any]) -> Expr:
    return node


@fmap.register(Call)
def _[T, S](node: Call[T], f: Callable[[T], S]) -> Call[S]:
    args = tuple(map(f, args_of(node)))
    if all(isinstance(arg, Expr) for arg in args):
        return Call(func_of(node), args)
    return Call(func_of(node), args, node.kind)
