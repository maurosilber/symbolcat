"""
symbolcat.ops._analysis._dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~

Physical dimensions of an expression.

A dimension is a mapping from base units to exponents (``{"m": 1, "s": -2}``
for an acceleration). Every node folds into a pair: its value, needed to
read literal exponents (``x ** 2``), and its dimension.

Operations on explicit arrays are first computed entry by entry, so the
entries of an array can have different dimensions (a state vector of a
position and a velocity). A symbol with an array shape has one dimension
for all its entries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

from ..._core import (
    Algebra,
    Call,
    Constant,
    Expr,
    Function,
    Literal,
    Symbol,
    args_of,
    cata,
    func_of,
    name_of,
)
from ...functions import complex as cx
from ...functions import linalg, real
from ...functions import operators as op
from ...functions.calculus import D
from ...functions.linalg import Array, nest
from .._evaluate import evaluate
from .._transform._arrays import expand_arrays

type Dimension = dict[str, Fraction]
type Pair = tuple[Expr, Dimension]

dimensions_algebra = Algebra[Pair]("dimensions")

DIMENSIONLESS: Dimension = {}


def normalize(dim: Mapping[str, Any], /) -> Dimension:
    return {k: Fraction(v) for k, v in sorted(dim.items()) if v != 0}


def combine(a: Dimension, b: Dimension, /, sign: int = 1) -> Dimension:
    return normalize({k: a.get(k, 0) + sign * b.get(k, 0) for k in a.keys() | b.keys()})


def scale(a: Dimension, factor: Fraction, /) -> Dimension:
    return normalize({k: v * factor for k, v in a.items()})


def show(dim: Dimension, /) -> str:
    return " * ".join(f"{k}**{v}" if v != 1 else k for k, v in dim.items()) or "1"


def same(node: Call[Pair], /) -> Dimension:
    """All arguments must have the same dimension, which is returned."""
    dims = [dim for _, dim in args_of(node)]
    if any(dim != dims[0] for dim in dims):
        raise ValueError(
            f"{name_of(func_of(node))} of incompatible dimensions: "
            + ", ".join(show(dim) for dim in dims)
        )
    return dims[0]


def dimensionless(node: Call[Pair], /) -> None:
    for value, dim in args_of(node):
        if dim:
            raise ValueError(
                f"{name_of(func_of(node))} needs a dimensionless argument, "
                f"got {value} with {show(dim)}"
            )


@dimensions_algebra.register(Symbol)
def _(node: Symbol, *, units: Mapping[Symbol | str, Dimension], **ctx: Any) -> Pair:
    for key in (node, name_of(node)):  # given by symbol, or by name
        if key in units:
            return node, units[key]
    raise ValueError(f"No units given for {name_of(node)!r}")


@dimensions_algebra.register(Constant, Literal)
def _(node: Constant | Literal, **ctx: Any) -> Pair:
    return node, DIMENSIONLESS


@dimensions_algebra.register(Function)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    """By default, functions (exp, log, sin, ...) take and return pure numbers."""
    dimensionless(node)
    return func_of(node)(*(v for v, _ in args_of(node))), DIMENSIONLESS


type Rule = Callable[[Call[Pair]], Dimension]


def rule(*funcs: Function | type) -> Callable[[Rule], Rule]:
    """Register ``node -> dimension``; the value is rebuilt from the arguments."""

    def decorator(dimension: Rule) -> Rule:
        @dimensions_algebra.register(*funcs)
        def _(node: Call[Pair], **ctx: Any) -> Pair:
            return func_of(node)(*(v for v, _ in args_of(node))), dimension(node)

        return dimension

    return decorator


@dimensions_algebra.register(D)
def _(node: Call[Pair], *, time: Dimension | None, **ctx: Any) -> Pair:
    """The dimension of the argument, divided by time."""
    ((value, dim),) = args_of(node)
    if time is None:
        raise ValueError("dimensions: pass the dimension of `time` to use D")
    return D(value), combine(dim, time, -1)


@rule(
    op.add,
    op.sub,
    op.mod,
    op.pos,
    op.neg,
    real.abs,
    real.floor,
    real.ceil,
    real.hypot,
    cx.re,
    cx.im,
    cx.conj,
    cx.abs,
)
def _(node: Call[Pair]) -> Dimension:
    return same(node)


@rule(real.ifelse)
def _(node: Call[Pair]) -> Dimension:
    """Both branches have the same dimension."""
    condition, a, b = args_of(node)
    dimensionless(Call(func_of(node), (condition,)))
    return same(Call(func_of(node), (a, b)))


@rule(op.eq, op.ne, op.lt, op.le, op.gt, op.ge, real.atan2)
def _(node: Call[Pair]) -> Dimension:
    same(node)
    return DIMENSIONLESS


@rule(cx.phase)
def _(node: Call[Pair]) -> Dimension:
    return DIMENSIONLESS


@rule(cx.rect)
def _(node: Call[Pair]) -> Dimension:
    """The dimension of the modulus; the angle is dimensionless."""
    (_, r), phi = args_of(node)
    dimensionless(Call(func_of(node), (phi,)))
    return r


@rule(Array, linalg.transpose, linalg.trace, linalg.sum, linalg.norm)
def _(node: Call[Pair]) -> Dimension:
    """Arrays with unknown entries have the same dimension for all of them."""
    return same(node)


@rule(linalg.inv)
def _(node: Call[Pair]) -> Dimension:
    ((_, a),) = args_of(node)
    return scale(a, Fraction(-1))


@rule(op.mul, op.matmul, linalg.cross)
def _(node: Call[Pair]) -> Dimension:
    (_, a), (_, b) = args_of(node)
    return combine(a, b)


@rule(op.truediv, op.floordiv)
def _(node: Call[Pair]) -> Dimension:
    (_, a), (_, b) = args_of(node)
    return combine(a, b, sign=-1)


@rule(real.sqrt, cx.sqrt)
def _(node: Call[Pair]) -> Dimension:
    ((_, a),) = args_of(node)
    return scale(a, Fraction(1, 2))


@rule(op.getitem)
def _(node: Call[Pair]) -> Dimension:
    (_, container), (key, key_dim) = args_of(node)
    if key_dim:
        raise ValueError(f"Index {key} has dimension {show(key_dim)}")
    return container


@dimensions_algebra.register(linalg.det)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    """The product of ``n`` entries, for an ``n`` by ``n`` matrix."""
    ((value, dim),) = args_of(node)
    n, _ = value.kind.shape
    return linalg.det(value), scale(dim, Fraction(n))


@rule(op.pow)
def _(node: Call[Pair]) -> Dimension:
    (_, base), (exponent, exponent_dim) = args_of(node)
    if exponent_dim:
        raise ValueError(f"Exponent {exponent} has dimension {show(exponent_dim)}")
    if not base:
        return DIMENSIONLESS
    try:
        value = evaluate(exponent)
    except (NameError, NotImplementedError):
        raise ValueError(
            f"Exponent {exponent} of a dimensional base must be a number"
        ) from None
    try:
        # floats come from exponents like 1 / 2
        power = Fraction(value).limit_denominator(1000)
    except (TypeError, ValueError):
        raise ValueError(f"Exponent {exponent} must be a real number") from None
    return scale(base, power)


type Exponents = dict[str, int | Fraction]


def dimensions[S: Symbol](
    expr: Any,
    units: Mapping[S, Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    /,
    *,
    time: Mapping[str, Any] | None = None,
) -> Exponents | list[Any]:
    """Dimension of ``expr``, given the dimension of each symbol.

    ``time`` is the dimension of time, needed for time derivatives: ``D(x)``
    has the dimension of ``x`` divided by ``time``.

    For an explicit array, the dimension of each entry, as (nested) lists.

    Dimensions map base units to exponents (integers, or ``Fraction`` for
    roots); ``{}`` is dimensionless. Raises
    ``ValueError`` on inconsistent dimensions, like adding a length to a
    time, or taking the logarithm of a length.

    >>> from symbolcat import Symbol, real
    >>> x, t, v0 = Symbol("x"), Symbol("t"), Symbol("v0")
    >>> units = {x: {"m": 1}, t: {"s": 1}, v0: {"m": 1, "s": -1}}
    >>> dimensions(v0 * t + x, units)
    {'m': 1}
    >>> dimensions(real.sqrt(x / t**2), units)
    {'m': Fraction(1, 2), 's': -1}
    >>> dimensions(x + t, units)
    Traceback (most recent call last):
    ...
    ValueError: add of incompatible dimensions: m, s
    >>> from symbolcat.functions.linalg import vector
    >>> dimensions(vector([x, v0]) * t, units)
    [{'m': 1, 's': 1}, {'m': 1}]
    >>> from symbolcat import D
    >>> dimensions(D(D(x)), units, time={"s": 1})
    {'m': 1, 's': -2}
    """
    normalized = {k: normalize(v) for k, v in units.items()}
    expr = expand_arrays(expr)

    def dimension(expr: Expr, /) -> Exponents:
        _, dim = cata(
            dimensions_algebra,
            expr,
            units=normalized,
            time=None if time is None else normalize(time),
        )
        return {k: int(v) if v.denominator == 1 else v for k, v in dim.items()}

    if isinstance(expr, Call) and isinstance(func_of(expr), Array):
        return nest(expr.kind.shape, [dimension(e) for e in args_of(expr)])
    return dimension(expr)
