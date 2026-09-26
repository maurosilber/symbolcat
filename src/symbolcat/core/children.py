"""
symbolcat.core.children
~~~~~~~~~~~~~~~~~~~~~~~

The direct children of a node.
"""

from __future__ import annotations

from functools import singledispatch
from typing import Any

from .expr import Call, Expr


@singledispatch
def children(node: Any) -> tuple[Any, ...]:
    """The direct children of a node.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> children(x + 1)
    (Symbol('x'), Literal(1))
    >>> children(x)
    ()
    """
    raise TypeError(f"{type(node).__name__} is not an expression node")


@children.register(Expr)
def _(node: Expr) -> tuple[Any, ...]:
    return ()


@children.register(Call)
def _[T](node: Call[T]) -> tuple[T, ...]:
    return node.args
