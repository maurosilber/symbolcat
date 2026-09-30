"""
symbolcat._core.hylo
~~~~~~~~~~~~~~~~~~~~

The hylomorphism: unfold a seed top-down with a coalgebra, and fold the
layers bottom-up with an algebra, without building the tree in between.
The anamorphism is the special case that only unfolds.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from .children import children
from .expr import Expr
from .fmap import fmap


@dataclass(frozen=True, slots=True)
class Done[A]:
    """A finished value in place of a seed: the unfolding stops there.

    A coalgebra can return one instead of a layer, or put one among the
    children of a layer. ``hylo`` uses its value as is, without unfolding
    or folding it.
    """

    value: A


def hylo[S, A](
    algebra: Callable[[Any], A],
    coalgebra: Callable[[S], Any],
    seed: S,
    /,
    *,
    key: Callable[[S], Hashable] = id,
) -> A:
    """Unfold ``seed`` with ``coalgebra`` and fold it with ``algebra``.

    ``coalgebra`` turns a seed into one layer: a node whose children are
    the next seeds, or ``Done`` values. ``algebra`` receives each layer with
    its children already folded, as in ``cata``.

    Seeds with the same ``key`` are unfolded and folded once. The default,
    ``id``, suits seeds that are expressions, since hashing one hashes the
    whole tree below it. The traversal uses an explicit stack, so deep
    trees do not hit the recursion limit.

    >>> from symbolcat.ops.algebras import Call, Literal
    >>> from symbolcat.lib import operators as op
    >>> def halves(n):  # n as a sum of halves, down to 1
    ...     return Literal(n) if n <= 1 else Call(op.add, (n // 2, n - n // 2))
    >>> hylo(lambda layer: 1 if isinstance(layer, Literal) else sum(layer.args),
    ...      halves, 5, key=lambda n: n)
    5
    """
    done: dict[Hashable, A] = {}
    # Keeping the layers also keeps their seeds alive, so ``id`` keys
    # cannot be reused by other objects during the traversal.
    layers: dict[Hashable, Any] = {}

    def value(child: Any) -> A:
        return child.value if isinstance(child, Done) else done[key(child)]

    stack = [seed]
    while stack:
        s = stack[-1]
        k = key(s)
        if k in done:
            stack.pop()
            continue
        if k not in layers:
            layers[k] = coalgebra(s)
        layer = layers[k]
        if isinstance(layer, Done):
            stack.pop()
            done[k] = layer.value
            continue
        pending = [
            c
            for c in children(layer)
            if not isinstance(c, Done) and key(c) not in done
        ]
        if pending:
            stack.extend(pending)
            continue
        stack.pop()
        done[k] = algebra(fmap(layer, value))
    return done[key(seed)]


def ana[S](
    coalgebra: Callable[[S], Any],
    seed: S,
    /,
    *,
    key: Callable[[S], Hashable] = id,
) -> Expr:
    """Unfold ``seed`` into an expression with ``coalgebra``.

    Each layer, with its children unfolded, is a node of the result, so
    seeds with the same ``key`` become the same (shared) sub-expression.
    ``Done`` holds a finished sub-expression.

    >>> from symbolcat.ops.algebras import Call, Literal
    >>> from symbolcat.lib import operators as op
    >>> def halves(n):
    ...     return Literal(n) if n <= 1 else Call(op.add, (n // 2, n - n // 2))
    >>> expr = ana(halves, 5, key=lambda n: n)
    >>> print(expr)
    1 + 1 + (1 + (1 + 1))
    >>> two, three = expr.args
    >>> three.args[1] is two  # both unfolded from the seed 2
    True
    """
    return hylo(_node, coalgebra, seed, key=key)


def _node(layer: Expr, /) -> Expr:
    return layer
