"""
symbolcat.ops._transform._share
~~~~~~~~~~~~~~~~~~~~

Shared sub-expressions.

``cata`` folds each object once, so an expression whose repeated parts are
the same object is computed faster. ``share`` makes structurally equal
parts the same object, and ``cse`` names them.
"""

from __future__ import annotations

from collections import Counter
from itertools import count
from typing import Any

from ..._core import Call, Expr, Symbol, as_expr, cata, children, fmap
from .._analysis._inspect import free_symbols


def share(expr: Any, /) -> Expr:
    """The same expression, with equal sub-expressions as the same object.

    >>> from symbolcat import Symbol, real
    >>> x = Symbol("x")
    >>> expr = share(real.sin(x + 1) * real.cos(x + 1))
    >>> left, right = expr.args
    >>> left.args[0] is right.args[0]
    True
    """
    return _share(as_expr(expr), {})


def _share(expr: Expr, cache: dict[Expr, Expr], /) -> Expr:
    # Each layer arrives with its children already shared, so equal layers
    # are equal nodes: keep the first one.
    return cata(lambda node: cache.setdefault(node, node), expr)


def _uses(expr: Expr, /) -> Counter[int]:
    """How many parents each node has (by identity) in a shared expression.

    Not a fold: ``cata`` visits a shared node once, and this counts the
    edges into it.
    """
    uses: Counter[int] = Counter()
    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for child in children(node):
            uses[id(child)] += 1
            stack.append(child)
    return uses


def cse(expr: Any, /, *, prefix: str = "_") -> tuple[list[tuple[Symbol, Expr]], Expr]:
    """Common sub-expression elimination.

    Returns assignments ``(temporary, value)``, in the order they can be
    computed, and the expression written with the temporaries. Temporaries
    are named ``prefix`` plus a number, avoiding the names of free symbols.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> assignments, result = cse(real.sin(x + y) ** 2 + real.cos(x + y) * (x + y))
    >>> for temporary, value in assignments:
    ...     print(temporary, "=", value)
    _0 = x + y
    >>> print(result)
    sin(_0) ** 2 + cos(_0) * _0
    """
    cache: dict[Expr, Expr] = {}
    shared = _share(as_expr(expr), cache)
    uses = _uses(shared)
    taken = {symbol.name for symbol in free_symbols(shared)}
    names = (name for i in count() if (name := f"{prefix}{i}") not in taken)
    assignments: list[tuple[Symbol, Expr]] = []

    # A catamorphism into (original, reduced) pairs: the original identifies
    # the shared node (to look up its uses), the reduced one uses temporaries.
    def algebra(layer: Expr, /) -> tuple[Expr, Expr]:
        original = cache[fmap(layer, lambda pair: pair[0])]
        reduced = fmap(layer, lambda pair: pair[1])
        if isinstance(original, Call) and uses[id(original)] > 1:
            temporary = Symbol(next(names), kind=reduced.kind)
            assignments.append((temporary, reduced))
            reduced = temporary
        return original, reduced

    _, result = cata(algebra, shared)
    return assignments, result
