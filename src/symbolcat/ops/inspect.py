"""
symbolcat.ops.inspect
~~~~~~~~~~~~~~~~~~~~~

Queries about the structure of an expression.
"""

from __future__ import annotations

from typing import Any

from ..core import Algebra, Call, Constant, Literal, Symbol, cata, children

# Folds into an insertion-ordered set (a dict with None values).
free_symbols_algebra = Algebra[dict[Symbol, None]]("free_symbols")


@free_symbols_algebra.register(Symbol)
def _(node: Symbol) -> dict[Symbol, None]:
    return {node: None}


@free_symbols_algebra.register(Constant, Literal)
def _(node: Constant | Literal) -> dict[Symbol, None]:
    return {}


@free_symbols_algebra.register(Call)
def _(node: Call[dict[Symbol, None]]) -> dict[Symbol, None]:
    out: dict[Symbol, None] = {}
    for arg in node.args:
        out |= arg
    return out


def free_symbols(expr: Any) -> tuple[Symbol, ...]:
    """User symbols in ``expr``, in order of first appearance.

    Library constants (``real.pi``) are not free.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> free_symbols(y * real.pi + x * y)
    (Symbol('y'), Symbol('x'))
    """
    return tuple(cata(free_symbols_algebra, expr))


def count_nodes(expr: Any) -> int:
    """Number of nodes in the tree. The algebra needs no registry.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> count_nodes(x**2 + 1)
    5
    """
    return cata(lambda node: 1 + sum(children(node)), expr)


def depth(expr: Any) -> int:
    """Length of the longest path from the root to a leaf.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> depth(x**2 + 1)
    3
    """
    return cata(lambda node: 1 + max(children(node), default=0), expr)
