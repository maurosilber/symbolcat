"""
symbolcat.ops._printing._base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared pieces of the printers: every node folds into a ``Printed`` (text plus
precedence), so a parent can decide whether its children need parentheses
without looking back into the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal as LiteralType

from ... import linalg
from ..._core import Function, Operator
from ..._core.precedence import UNARY
from ...lib import operators as op
from ...linalg import ENTRYWISE

ATOM = 100


@dataclass(frozen=True)
class Printed:
    """Printed text, with its precedence and, for explicit arrays and
    operations on them, the number of dimensions."""

    text: str
    precedence: int = ATOM
    ndim: int = 0


def result_ndim(func: Function, args: tuple[Printed, ...], /) -> int:
    """Dimensions of the result, when its arguments are explicit arrays."""
    if func is op.matmul:
        a, b = args
        return max(a.ndim + b.ndim - 2, 0) if a.ndim and b.ndim else 0
    if func in ENTRYWISE or func in (linalg.transpose, linalg.inv):
        return max((arg.ndim for arg in args), default=0)
    return 0


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
