"""
symbolcat.ops.egraph._convert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translate between symbolcat expressions and egglog terms.

``to_egglog`` is one more algebra folded with ``cata``. ``from_egglog``
walks the extracted egglog term back, memoizing on the term so that shared
sub-terms (from DAG extraction) become shared Python objects, which
``cata`` then folds only once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, cast

from egglog import BigRat, String, Vec, f64, get_callable_args, get_callable_fn, i64

from ... import real
from ..._core import (
    Algebra,
    Call,
    Constant,
    Expr,
    Function,
    Literal,
    Symbol,
    cata,
    name_of,
)
from ...lib import operators as op
from . import _term as term
from ._term import Num

# symbolcat functions with a dedicated egglog constructor
FUNCTIONS: dict[Function, Any] = {
    op.add: Num.__add__,
    op.sub: Num.__sub__,
    op.mul: Num.__mul__,
    op.truediv: Num.__truediv__,
    op.pow: Num.__pow__,
    op.neg: Num.__neg__,
    real.abs: term.abs,
    real.exp: term.exp,
    real.log: term.log,
    real.sqrt: term.sqrt,
    real.sin: term.sin,
    real.cos: term.cos,
    real.tan: term.tan,
    real.asin: term.asin,
    real.acos: term.acos,
    real.atan: term.atan,
    real.sinh: term.sinh,
    real.cosh: term.cosh,
    real.tanh: term.tanh,
}
FUNCTIONS_BACK = {v: k for k, v in FUNCTIONS.items()}

# symbolcat constants with a dedicated egglog constant
CONSTANTS: dict[Constant, Num] = {
    real.constants.pi: Num.pi,
    real.constants.e: Num.e,
}
# An extracted constant is recognized by its callable, not by the term.
CONSTANTS_BACK = {get_callable_fn(v): k for k, v in CONSTANTS.items()}


@dataclass
class Table:
    """Objects that egglog cannot represent, referenced by index."""

    items: list[Any] = field(default_factory=list)
    _index: dict[int, int] = field(default_factory=dict)

    def index(self, obj: Any) -> int:
        key = id(obj)
        if key not in self._index:
            self._index[key] = len(self.items)
            self.items.append(obj)
        return self._index[key]

    def __getitem__(self, index: int) -> Any:
        return self.items[index]


to_egglog_algebra = Algebra[Num]("to_egglog")


@to_egglog_algebra.register(Symbol)
def _(node: Symbol, *, table: Table) -> Num:
    return Num.var(name_of(node))


@to_egglog_algebra.register(Constant)
def _(node: Constant, *, table: Table) -> Num:
    if (constant := CONSTANTS.get(node)) is not None:
        return constant
    return Num.opaque(table.index(node))


@to_egglog_algebra.register(Literal)
def _(node: Literal, *, table: Table) -> Num:
    value = node.value
    if isinstance(value, int) and not isinstance(value, bool):
        return Num(BigRat(value, 1))
    if isinstance(value, float):
        return Num.float(value)
    return Num.opaque(table.index(node))


@to_egglog_algebra.register(Call)
def _(node: Call[Num], *, table: Table) -> Num:
    if (func := FUNCTIONS.get(node.func)) is not None:
        return func(*node.args)
    return Num.apply(table.index(node.func), Vec(*node.args))


def to_egglog(expr: Any, table: Table) -> Num:
    """Translate an expression, storing untranslatable parts in ``table``."""
    return cata(to_egglog_algebra, expr, table=table)


def from_rational(value: Fraction) -> Expr:
    if value.denominator == 1:
        return Literal(value.numerator)
    return op.truediv(value.numerator, value.denominator)


def from_egglog(num: Num, table: Table) -> Expr:
    """Translate an extracted egglog term back into an expression."""
    memo: dict[Num, Expr] = {}

    # Typed accessors give the ``*Like`` argument types; extracted terms
    # always hold the egglog types, hence the casts.
    def convert(num: Num) -> Expr:
        if num in memo:
            return memo[num]
        fn = get_callable_fn(num)
        result: Expr
        if fn in CONSTANTS_BACK:
            result = CONSTANTS_BACK[fn]
        elif (rational := get_callable_args(num, Num)) is not None:
            result = from_rational(cast("BigRat", rational[0]).value)
        elif (floating := get_callable_args(num, Num.float)) is not None:
            result = Literal(cast("f64", floating[0]).value)
        elif (name := get_callable_args(num, Num.var)) is not None:
            result = Symbol(cast("String", name[0]).value)
        elif (index := get_callable_args(num, Num.opaque)) is not None:
            result = table[cast("i64", index[0]).value]
        elif (call := get_callable_args(num, Num.apply)) is not None:
            func = table[cast("i64", call[0]).value]
            result = func(*map(convert, call[1].value))
        else:
            args = cast("tuple[Num, ...]", get_callable_args(num))
            result = FUNCTIONS_BACK[fn](*map(convert, args))
        memo[num] = result
        return result

    return convert(num)
