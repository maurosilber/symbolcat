"""
symbolcat._core.cata
~~~~~~~~~~~~~~~~~~~~

The catamorphism: fold an expression bottom-up with an algebra.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .children import children
from .expr import as_expr
from .fmap import fmap


def cata[A](algebra: Callable[..., A], expr: Any, /, **ctx: Any) -> A:
    """Fold ``expr`` bottom-up with ``algebra``.

    ``algebra`` receives one layer at a time: a node whose children have
    already been replaced by their folded values (``fmap(node, fold)``).
    Extra keyword arguments are forwarded to every algebra call.

    Shared sub-expressions (the same object reachable twice) are folded once.
    The traversal uses an explicit stack, so deep trees do not hit the
    recursion limit.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> count = lambda node: 1 + sum(children(node))
    >>> cata(count, x * x + 1)
    5
    """
    expr = as_expr(expr)
    done: dict[int, A] = {}
    stack = [expr]
    while stack:
        node = stack[-1]
        if id(node) in done:
            stack.pop()
            continue
        pending = [c for c in children(node) if id(c) not in done]
        if pending:
            stack.extend(pending)
            continue
        stack.pop()
        done[id(node)] = algebra(fmap(node, lambda c: done[id(c)]), **ctx)
    return done[id(expr)]
