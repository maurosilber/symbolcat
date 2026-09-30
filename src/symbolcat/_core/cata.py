"""
symbolcat._core.cata
~~~~~~~~~~~~~~~~~~~~

The catamorphism: fold an expression bottom-up with an algebra.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .expr import as_expr
from .hylo import hylo


def cata[A](algebra: Callable[..., A], expr: Any, /, **ctx: Any) -> A:
    """Fold ``expr`` bottom-up with ``algebra``.

    ``algebra`` receives one layer at a time: a node whose children have
    already been replaced by their folded values (``fmap(node, fold)``).
    Extra keyword arguments are forwarded to every algebra call.

    Shared sub-expressions (the same object reachable twice) are folded once.
    The traversal uses an explicit stack, so deep trees do not hit the
    recursion limit.

    >>> from symbolcat import Symbol
    >>> from symbolcat.ops.algebras import children
    >>> x = Symbol("x")
    >>> count = lambda node: 1 + sum(children(node))
    >>> cata(count, x * x + 1)
    5
    """
    # An expression is its own layer: the coalgebra only has to return it.
    return hylo(lambda layer: algebra(layer, **ctx), _layer, as_expr(expr))


def _layer(node: Any, /) -> Any:
    return node
