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

from ... import linalg, real
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
from ..._derivative import D
from ...lib import operators as op
from ...linalg import Array, nest
from .._evaluate import evaluate
from .._transform._arrays import expand_arrays, shape

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
    dims = [dim for _, dim in node.args]
    if any(dim != dims[0] for dim in dims):
        raise ValueError(
            f"{name_of(node.func)} of incompatible dimensions: "
            + ", ".join(show(dim) for dim in dims)
        )
    return dims[0]


def dimensionless(node: Call[Pair], /) -> None:
    for value, dim in node.args:
        if dim:
            raise ValueError(
                f"{name_of(node.func)} needs a dimensionless argument, "
                f"got {value} with {show(dim)}"
            )


@dimensions_algebra.register(Symbol)
def _(node: Symbol, *, units: Mapping[Symbol, Dimension], **ctx: Any) -> Pair:
    if node not in units:
        raise ValueError(f"No units given for {name_of(node)!r}")
    return node, units[node]


@dimensions_algebra.register(Constant, Literal)
def _(node: Constant | Literal, **ctx: Any) -> Pair:
    return node, DIMENSIONLESS


@dimensions_algebra.register(Function)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    """By default, functions (exp, log, sin, ...) take and return pure numbers."""
    dimensionless(node)
    return node.func(*(v for v, _ in node.args)), DIMENSIONLESS


type Rule = Callable[[Call[Pair]], Dimension]


def rule(*funcs: Function | type) -> Callable[[Rule], Rule]:
    """Register ``node -> dimension``; the value is rebuilt from the arguments."""

    def decorator(dimension: Rule) -> Rule:
        @dimensions_algebra.register(*funcs)
        def _(node: Call[Pair], **ctx: Any) -> Pair:
            return node.func(*(v for v, _ in node.args)), dimension(node)

        return dimension

    return decorator


@dimensions_algebra.register(D)
def _(node: Call[Pair], *, time: Dimension | None, **ctx: Any) -> Pair:
    """The dimension of the argument, divided by time."""
    ((value, dim),) = node.args
    if time is None:
        raise ValueError("dimensions: pass the dimension of `time` to use D")
    return D(value), combine(dim, time, -1)


@rule(
    op.add, op.sub, op.mod, op.pos, op.neg, real.abs, real.floor, real.ceil, real.hypot
)
def _(node: Call[Pair]) -> Dimension:
    return same(node)


@rule(real.ifelse)
def _(node: Call[Pair]) -> Dimension:
    """Both branches have the same dimension."""
    condition, a, b = node.args
    dimensionless(Call(node.func, (condition,)))
    return same(Call(node.func, (a, b)))


@rule(op.eq, op.ne, op.lt, op.le, op.gt, op.ge, real.atan2)
def _(node: Call[Pair]) -> Dimension:
    same(node)
    return DIMENSIONLESS


@rule(Array, linalg.transpose, linalg.trace, linalg.sum, linalg.norm)
def _(node: Call[Pair]) -> Dimension:
    """Arrays with unknown entries have the same dimension for all of them."""
    return same(node)


@rule(linalg.inv)
def _(node: Call[Pair]) -> Dimension:
    ((_, a),) = node.args
    return scale(a, Fraction(-1))


@rule(op.mul, op.matmul, linalg.cross)
def _(node: Call[Pair]) -> Dimension:
    (_, a), (_, b) = node.args
    return combine(a, b)


@rule(op.truediv, op.floordiv)
def _(node: Call[Pair]) -> Dimension:
    (_, a), (_, b) = node.args
    return combine(a, b, sign=-1)


@rule(real.sqrt)
def _(node: Call[Pair]) -> Dimension:
    ((_, a),) = node.args
    return scale(a, Fraction(1, 2))


@rule(op.getitem)
def _(node: Call[Pair]) -> Dimension:
    (_, container), (key, key_dim) = node.args
    if key_dim:
        raise ValueError(f"Index {key} has dimension {show(key_dim)}")
    return container


@dimensions_algebra.register(linalg.det)
def _(
    node: Call[Pair], *, shapes: Mapping[Symbol, tuple[int, ...]], **ctx: Any
) -> Pair:
    """The product of ``n`` entries, for an ``n`` by ``n`` matrix."""
    ((value, dim),) = node.args
    n, _ = shape(value, shapes)
    return linalg.det(value), scale(dim, Fraction(n))


@rule(op.pow)
def _(node: Call[Pair]) -> Dimension:
    (_, base), (exponent, exponent_dim) = node.args
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


def dimensions(
    expr: Any,
    units: Mapping[Symbol, Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    /,
    shapes: Mapping[Any, Any] | None = None,
    *,
    time: Mapping[str, Any] | None = None,
) -> Exponents | list[Any]:
    """Dimension of ``expr``, given the dimension of each symbol.

    ``time`` is the dimension of time, needed for time derivatives: ``D(x)``
    has the dimension of ``x`` divided by ``time``.

    For an array, the dimension of each entry, as (nested) lists. ``shapes``
    gives the shapes of symbols that are arrays (see ``expand_arrays``).

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
    >>> from symbolcat.linalg import vector
    >>> dimensions(vector([x, v0]) * t, units)
    [{'m': 1, 's': 1}, {'m': 1}]
    >>> from symbolcat import D
    >>> dimensions(D(D(x)), units, time={"s": 1})
    {'m': 1, 's': -2}
    """
    normalized = {
        Symbol(k) if isinstance(k, str) else k: normalize(v) for k, v in units.items()
    }
    shapes = {
        Symbol(k) if isinstance(k, str) else k: v for k, v in (shapes or {}).items()
    }
    expr = expand_arrays(expr, shapes)

    def dimension(expr: Expr, /) -> Exponents:
        _, dim = cata(
            dimensions_algebra,
            expr,
            units=normalized,
            shapes=shapes,
            time=None if time is None else normalize(time),
        )
        return {k: int(v) if v.denominator == 1 else v for k, v in dim.items()}

    if isinstance(expr, Call) and isinstance(expr.func, Array):
        return nest(expr.func.shape, [dimension(e) for e in expr.args])
    return dimension(expr)
