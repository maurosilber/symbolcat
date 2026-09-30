"""
symbolcat.ops._printing._str
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Print an expression as Python source.
"""

from __future__ import annotations

from typing import Any

from ..._core import (
    Algebra,
    Call,
    Constant,
    Function,
    Literal,
    Operator,
    Symbol,
    args_of,
    cata,
    fmt_of,
    func_of,
    name_of,
    precedence_of,
    value_of,
)
from ..._core.precedence import POSTFIX
from ...lib import operators as op
from ...linalg import Array, nest
from ._base import Printed, literal_precedence, operand

str_algebra = Algebra[Printed]("str")


@str_algebra.register(Symbol, Constant)
def _(node: Symbol | Constant) -> Printed:
    return Printed(name_of(node))


@str_algebra.register(Literal)
def _(node: Literal) -> Printed:
    return Printed(repr(value_of(node)), literal_precedence(value_of(node)))


@str_algebra.register(Function)
def _(node: Call[Printed]) -> Printed:
    args = ", ".join(arg.text for arg in args_of(node))
    return Printed(f"{name_of(func_of(node))}({args})")


@str_algebra.register(Array)
def _(node: Call[Printed]) -> Printed:
    assert isinstance(func_of(node), Array)
    rows = nest(node.kind.shape, [arg.text for arg in args_of(node)])
    if len(node.kind.shape) == 1:
        return Printed(f"vector([{', '.join(rows)}])")
    text = ", ".join(f"[{', '.join(row)}]" for row in rows)
    return Printed(f"matrix([{text}])")


@str_algebra.register(Operator)
def _(node: Call[Printed]) -> Printed:
    func = func_of(node)
    assert isinstance(func, Operator)
    if len(args_of(node)) == 1:
        (arg,) = args_of(node)
        text = fmt_of(func).format(operand(arg, func, "right"))
    else:
        left, right = args_of(node)
        text = fmt_of(func).format(
            operand(left, func, "left"), operand(right, func, "right")
        )
    return Printed(text, precedence_of(func))


@str_algebra.register(op.getitem)
def _(node: Call[Printed]) -> Printed:
    obj, key = args_of(node)
    return Printed(f"{operand(obj, op.getitem, 'left')}[{key.text}]", POSTFIX)


def to_str(expr: Any, /) -> str:
    """Python source for an expression.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> to_str((x + y) * 2 - x**-y)
    '(x + y) * 2 - x ** (-y)'
    """
    return cata(str_algebra, expr).text
