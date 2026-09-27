"""
symbolcat.ops.egraph.term
~~~~~~~~~~~~~~~~~~~~~~~~~

The egglog term language that expressions are translated into.

Only the operations with rewrite rules get a dedicated constructor.
Everything else is kept opaque so it survives the round trip:

- ``Num.opaque(i)``: a leaf (constant, non-numeric literal) stored at
  index ``i`` of the translation table;
- ``Num.apply(i, args)``: a call to the function stored at index ``i``,
  whose arguments are still rewritten.
"""

from __future__ import annotations

from egglog import (
    BigRatLike,
    Expr,
    StringLike,
    Vec,
    f64Like,
    function,
    i64Like,
)


class Num(Expr):
    """A real-valued term."""

    def __init__(self, value: BigRatLike) -> None:
        """An exact rational number."""

    @classmethod
    def float(cls, value: f64Like) -> Num:
        """A floating point number."""

    @classmethod
    def var(cls, name: StringLike) -> Num:
        """A free symbol."""

    @classmethod
    def opaque(cls, index: i64Like) -> Num:
        """A leaf without rewrite rules."""

    @classmethod
    def apply(cls, index: i64Like, args: Vec[Num]) -> Num:
        """A call to a function without rewrite rules."""

    def __add__(self, other: Num) -> Num: ...

    def __sub__(self, other: Num) -> Num: ...

    def __mul__(self, other: Num) -> Num: ...

    def __truediv__(self, other: Num) -> Num: ...

    def __pow__(self, other: Num) -> Num: ...

    def __neg__(self) -> Num: ...


@function
def exp(x: Num) -> Num: ...


@function
def log(x: Num) -> Num: ...


@function
def sqrt(x: Num) -> Num: ...


@function
def sin(x: Num) -> Num: ...


@function
def cos(x: Num) -> Num: ...


@function
def tan(x: Num) -> Num: ...
