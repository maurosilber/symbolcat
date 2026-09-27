"""
symbolcat.ops.egraph.convert
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translate between symbolcat expressions and egglog terms.

``to_egglog`` is one more algebra folded with ``cata``. ``from_egglog``
walks the extracted egglog term back, memoizing on the term so that shared
sub-terms (from DAG extraction) become shared Python objects, which
``cata`` then folds only once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from egglog import BigRat, Vec, get_callable_args, get_callable_fn, get_literal_value

from ... import operators as op
from ... import real
from ...core import Algebra, Call, Constant, Expr, Function, Literal, Symbol, cata
from . import term
from .term import Num

# symbolcat functions with a dedicated egglog constructor
FUNCTIONS: dict[Function, Any] = {
    op.add: Num.__add__,
    op.sub: Num.__sub__,
    op.mul: Num.__mul__,
    op.truediv: Num.__truediv__,
    op.pow: Num.__pow__,
    op.neg: Num.__neg__,
    real.exp: term.exp,
    real.log: term.log,
    real.sqrt: term.sqrt,
    real.sin: term.sin,
    real.cos: term.cos,
    real.tan: term.tan,
}
FUNCTIONS_BACK = {v: k for k, v in FUNCTIONS.items()}


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
    return Num.var(node.name)


@to_egglog_algebra.register(Constant)
def _(node: Constant, *, table: Table) -> Num:
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

    def convert(num: Num) -> Expr:
        if num in memo:
            return memo[num]
        fn = get_callable_fn(num)
        args = get_callable_args(num) or ()
        if fn == Num:
            (rational,) = args
            result = from_rational(rational.value)
        elif fn == Num.float:
            (value,) = args
            result = Literal(value.value)
        elif fn == Num.var:
            (name,) = args
            result = Symbol(get_literal_value(name))
        elif fn == Num.opaque:
            (index,) = args
            result = table[get_literal_value(index)]
        elif fn == Num.apply:
            index, vec = args
            func = table[get_literal_value(index)]
            result = func(*map(convert, get_callable_args(vec) or ()))
        else:
            result = FUNCTIONS_BACK[fn](*map(convert, args))
        memo[num] = result
        return result

    return convert(num)
