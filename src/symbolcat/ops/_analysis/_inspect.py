"""
symbolcat.ops._analysis._inspect
~~~~~~~~~~~~~~~~~~~~~~

Queries about the structure of an expression.
"""

from __future__ import annotations

from typing import Any

from ... import complex as cx
from ..._core import (
    Algebra,
    Call,
    Constant,
    Done,
    Expr,
    Literal,
    Symbol,
    as_expr,
    cata,
    children,
    hylo,
)
from ..._core import Function as _Function
from ..._derivative import Variable, is_variable


def union[T](layer: Expr, /) -> dict[T, None]:
    """The children of ``layer``, insertion-ordered sets, merged in order."""
    out: dict[T, None] = {}
    for child in children(layer):
        out |= child
    return out


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


def free_symbols(expr: Any, /) -> tuple[Symbol, ...]:
    """User symbols in ``expr``, in order of first appearance.

    Library constants (``real.constants.pi``) are not free.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> free_symbols(y * real.constants.pi + x * y)
    (Symbol('y'), Symbol('x'))
    """
    return tuple(cata(free_symbols_algebra, expr))


def count_nodes(expr: Any, /) -> int:
    """Number of nodes in the tree. The algebra needs no registry.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> count_nodes(x**2 + 1)
    5
    """
    return cata(lambda node: 1 + sum(children(node)), expr)


def depth(expr: Any, /) -> int:
    """Length of the longest path from the root to a leaf.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> depth(x**2 + 1)
    3
    """
    return cata(lambda node: 1 + max(children(node), default=0), expr)


def variables(expr: Any, /) -> tuple[Variable, ...]:
    """Symbols and time derivatives of symbols in ``expr``, in order of first
    appearance. ``D(x)`` is a variable of its own: it does not add ``x``.

    >>> from symbolcat import D, Symbol
    >>> x, v = Symbol("x"), Symbol("v")
    >>> variables(D(D(x)) + v * D(x))
    (Call(D, (Call(D, (Symbol('x'),)),)), Symbol('v'), Call(D, (Symbol('x'),)))
    """

    # A cata would fold the ``x`` inside ``D(x)`` first; the coalgebra stops
    # at a variable instead of unfolding it.
    def coalgebra(node: Expr, /) -> Expr | Done[dict[Variable, None]]:
        return Done({node: None}) if is_variable(node) else node

    return tuple(hylo(union, coalgebra, as_expr(expr)))


# Whether a node is complex, given whether its arguments are.
complex_algebra = Algebra[bool]("complex")

# Complex functions with a real result.
REAL_VALUED = frozenset([cx.re, cx.im, cx.abs, cx.phase])


@complex_algebra.register(Symbol)
def _(node: Symbol) -> bool:
    return False


@complex_algebra.register(Constant)
def _(node: Constant) -> bool:
    return node is cx.constants.I


@complex_algebra.register(Literal)
def _(node: Literal) -> bool:
    return isinstance(node.value, complex)


@complex_algebra.register(Call)
def _(node: Call[bool]) -> bool:
    return any(node.args)


@complex_algebra.register(
    *(f for f in vars(cx).values() if isinstance(f, _Function) and f is not cx.conj)
)
def _(node: Call[bool]) -> bool:
    return node.func not in REAL_VALUED


def is_complex(expr: Any, /) -> bool:
    """Whether ``expr`` may have a complex value.

    Symbols are real, and so are the functions in ``real``, of real
    arguments. The functions in ``complex`` are complex, except ``re``,
    ``im``, ``abs`` and ``phase``.

    >>> from symbolcat import Symbol, complex
    >>> x = Symbol("x")
    >>> is_complex(complex.exp(x)), is_complex(complex.abs(complex.exp(x)))
    (True, False)
    """
    return cata(complex_algebra, as_expr(expr))
