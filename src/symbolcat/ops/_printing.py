"""
symbolcat.ops._printing
~~~~~~~~~~~~~~~~~~~~~~~

Print an expression as Python source.

The algebra folds each node into a ``Printed`` (text plus precedence), so a
parent can decide whether its children need parentheses without looking
back into the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Literal as Side

from .._core import (
    Algebra,
    Call,
    Constant,
    Function,
    Literal,
    Operator,
    Symbol,
    cata,
    name_of,
)
from .._core.precedence import POSTFIX, UNARY
from ..lib import operators as op

ATOM = 100


@dataclass(frozen=True)
class Printed:
    text: str
    precedence: int = ATOM


def needs_parens(child: Printed, parent: Operator, side: Side["left", "right"]) -> bool:
    if child.precedence != parent.precedence:
        return child.precedence < parent.precedence
    # Same precedence: only the side matching the associativity is free.
    return parent.assoc != side


def operand(child: Printed, parent: Operator, side: Side["left", "right"]) -> str:
    if needs_parens(child, parent, side):
        return f"({child.text})"
    return child.text


def literal_precedence(value: Any) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
        return UNARY
    return ATOM


str_algebra = Algebra[Printed]("str")


@str_algebra.register(Symbol, Constant)
def _(node: Symbol | Constant) -> Printed:
    return Printed(name_of(node))


@str_algebra.register(Literal)
def _(node: Literal) -> Printed:
    return Printed(repr(node.value), literal_precedence(node.value))


@str_algebra.register(Function)
def _(node: Call[Printed]) -> Printed:
    args = ", ".join(arg.text for arg in node.args)
    return Printed(f"{name_of(node.func)}({args})")


@str_algebra.register(Operator)
def _(node: Call[Printed]) -> Printed:
    func: Operator = node.func  # type: ignore[assignment]
    if len(node.args) == 1:
        (arg,) = node.args
        text = func.fmt.format(operand(arg, func, "right"))
    else:
        left, right = node.args
        text = func.fmt.format(
            operand(left, func, "left"), operand(right, func, "right")
        )
    return Printed(text, func.precedence)


@str_algebra.register(op.getitem)
def _(node: Call[Printed]) -> Printed:
    obj, key = node.args
    return Printed(f"{operand(obj, op.getitem, 'left')}[{key.text}]", POSTFIX)


def to_str(expr: Any) -> str:
    """Python source for an expression.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> to_str((x + y) * 2 - x**-y)
    '(x + y) * 2 - x ** (-y)'
    """
    return cata(str_algebra, expr).text
