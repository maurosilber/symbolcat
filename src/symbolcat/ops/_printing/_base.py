"""
symbolcat.ops._printing._base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared pieces of the printers: every node folds into a ``Printed`` (text plus
precedence), so a parent can decide whether its children need parentheses
without looking back into the tree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
from typing import Literal as LiteralType

from ..._core import Expr, Operator
from ..._core.precedence import UNARY

ATOM = 100


@dataclass(frozen=True)
class Printed:
    """Printed text, with its precedence and its number of dimensions (for
    arrays, see ``with_ndim``)."""

    text: str
    precedence: int = ATOM
    ndim: int = 0


def with_ndim(algebra: Callable[..., Printed], /) -> Callable[..., Printed]:
    """``algebra``, with the number of dimensions of each printed node
    taken from its kind, so a rule can tell arrays apart (``\\odot`` for
    the entrywise product) without tracking them itself."""

    def fold(layer: Expr, /, **ctx: Any) -> Printed:
        return replace(algebra(layer, **ctx), ndim=len(layer.kind.shape))

    return fold


type Side = LiteralType["left", "right"]


def needs_parens(child: Printed, parent: Operator, side: Side, /) -> bool:
    if child.precedence != parent.precedence:
        return child.precedence < parent.precedence
    # Same precedence: only the side matching the associativity is free.
    return parent.assoc != side


def operand(child: Printed, parent: Operator, side: Side, /) -> str:
    if needs_parens(child, parent, side):
        return f"({child.text})"
    return child.text


def indices(key: Printed, /) -> tuple[int, ...] | None:
    """The integer indices of a printed literal key (``3``, ``3.0`` or
    ``(0, 1)``), or ``None`` for a symbolic key."""
    import ast

    try:
        value = ast.literal_eval(key.text)
    except (ValueError, SyntaxError):
        return None
    values = value if isinstance(value, tuple) else (value,)
    if not all(isinstance(v, (int, float)) and float(v).is_integer() for v in values):
        return None
    return tuple(int(v) for v in values)


def literal_precedence(value: Any, /) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
        return UNARY
    return ATOM
