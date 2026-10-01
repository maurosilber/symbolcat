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

from ..._core import Expr, Operator, Symbol, assoc_of, name_of, precedence_of
from ..._core.precedence import UNARY
from .._analysis._inspect import free_symbols
from .._transform._substitute import substitute

ATOM = 100


def disambiguate(expr: Any, /) -> Any:
    """``expr`` with the symbols that share a name renamed after their ids,
    ``x#3f2a``, so that they print differently. Symbols print by name, but
    different ones can share it: a ``Dummy``, or a symbol of another kind.

    >>> from symbolcat import Dummy
    >>> x, y = Symbol("x"), Dummy("x")
    >>> a, b = free_symbols(disambiguate(x + y))
    >>> name_of(a) != name_of(b), name_of(a).startswith("x#")
    (True, True)
    """
    groups: dict[str, list[Symbol]] = {}
    for symbol in free_symbols(expr):
        groups.setdefault(name_of(symbol), []).append(symbol)
    renames: dict[Symbol, Symbol] = {}
    for name, symbols in groups.items():
        if len(symbols) == 1:
            continue
        # objects are 16-byte aligned: the low bits of the id carry nothing
        tags = [f"{id(s) >> 4 & 0xFFFF:04x}" for s in symbols]
        if len(set(tags)) < len(tags):
            tags = [f"{id(s):x}" for s in symbols]
        for symbol, tag in zip(symbols, tags, strict=True):
            renames[symbol] = Symbol(f"{name}#{tag}", kind=symbol.kind)
    return substitute(expr, renames) if renames else expr


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
    if child.precedence != precedence_of(parent):
        return child.precedence < precedence_of(parent)
    # Same precedence: only the side matching the associativity is free.
    return assoc_of(parent) != side


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
