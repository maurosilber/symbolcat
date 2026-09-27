"""
symbolcat.ops.egraph._term
~~~~~~~~~~~~~~~~~~~~~~~~~~

The egglog term language that expressions are translated into.

Only the operations with rewrite rules get a dedicated constructor.
Everything else is kept opaque so it survives the round trip:

- ``Num.opaque(i)``: a leaf (constant, non-numeric literal) stored at
  index ``i`` of the translation table;
- ``Num.apply(i, args)``: a call to the function stored at index ``i``,
  whose arguments are still rewritten.

Constructor costs define the size model used by ``simplify``, evaluated
natively by egglog: every node costs about ``NODE``, and small differences
break ties in favor of the conventional form (``x ** 2`` over ``x * x``,
``2 * x`` over ``x + x``, fewer divisions). See also ``_rules.literal_costs``.
"""

from __future__ import annotations

from typing import ClassVar

from egglog import (
    BigRatLike,
    Expr,
    StringLike,
    Vec,
    f64Like,
    function,
    i64Like,
    method,
)

NODE = 1000


class Num(Expr):
    """A real-valued term."""

    pi: ClassVar[Num]
    e: ClassVar[Num]

    @method(cost=NODE)
    def __init__(self, value: BigRatLike) -> None:
        """An exact rational number."""

    @method(cost=NODE)
    @classmethod
    def float(cls, value: f64Like) -> Num:
        """A floating point number."""

    @method(cost=NODE)
    @classmethod
    def var(cls, name: StringLike) -> Num:
        """A free symbol."""

    @method(cost=NODE)
    @classmethod
    def opaque(cls, index: i64Like) -> Num:
        """A leaf without rewrite rules."""

    @method(cost=NODE)
    @classmethod
    def apply(cls, index: i64Like, args: Vec[Num]) -> Num:
        """A call to a function without rewrite rules."""

    @method(cost=NODE + 20)
    def __add__(self, other: Num) -> Num: ...

    @method(cost=NODE + 20)
    def __sub__(self, other: Num) -> Num: ...

    @method(cost=NODE + 10)
    def __mul__(self, other: Num) -> Num: ...

    @method(cost=NODE + 30)
    def __truediv__(self, other: Num) -> Num: ...

    @method(cost=NODE)
    def __pow__(self, other: Num) -> Num: ...

    @method(cost=NODE + 20)
    def __neg__(self) -> Num: ...


@function(cost=NODE)
def abs(x: Num) -> Num: ...


@function(cost=NODE)
def exp(x: Num) -> Num: ...


@function(cost=NODE)
def log(x: Num) -> Num: ...


@function(cost=NODE)
def sqrt(x: Num) -> Num: ...


@function(cost=NODE)
def sin(x: Num) -> Num: ...


@function(cost=NODE)
def cos(x: Num) -> Num: ...


@function(cost=NODE)
def tan(x: Num) -> Num: ...


@function(cost=NODE)
def asin(x: Num) -> Num: ...


@function(cost=NODE)
def acos(x: Num) -> Num: ...


@function(cost=NODE)
def atan(x: Num) -> Num: ...


@function(cost=NODE)
def sinh(x: Num) -> Num: ...


@function(cost=NODE)
def cosh(x: Num) -> Num: ...


@function(cost=NODE)
def tanh(x: Num) -> Num: ...
