"""
symbolcat.ops._polynomial
~~~~~~~~~~~~~~~~~~~~~~~~~

Polynomial coefficients.

Every node folds into a pair: its value (rebuilt, to be used as a
coefficient) and its coefficients as a polynomial in the variable, or
``None`` if it is not one.
"""

from __future__ import annotations

from typing import Any

from .._core import Algebra, Call, Constant, Expr, Literal, Symbol, cata
from ..lib import operators as op
from ._simplify import is_number, simplify_exact

type Poly = dict[int, Expr]
type Pair = tuple[Expr, Poly | None]

coefficients_algebra = Algebra[Pair]("coefficients")


def is_constant(poly: Poly | None, /) -> bool:
    return poly is not None and set(poly) <= {0}


def add(a: Poly, b: Poly, /, *, subtract: bool = False) -> Poly:
    out = dict(a)
    for k, c in b.items():
        if k not in out:
            out[k] = -c if subtract else c
        else:
            out[k] = out[k] - c if subtract else out[k] + c
    return out


def mul(a: Poly, b: Poly, /) -> Poly:
    out: Poly = {}
    for i, ca in a.items():
        for j, cb in b.items():
            out[i + j] = out[i + j] + ca * cb if i + j in out else ca * cb
    return out


@coefficients_algebra.register(Symbol)
def _(node: Symbol, *, var: Symbol) -> Pair:
    return node, ({1: Literal(1)} if node == var else {0: node})


@coefficients_algebra.register(Constant, Literal)
def _(node: Constant | Literal, *, var: Symbol) -> Pair:
    return node, {0: node}


@coefficients_algebra.register(Call)
def _(node: Call[Pair], *, var: Symbol) -> Pair:
    """Any other function is a polynomial only if it does not depend on var."""
    value = node.func(*(v for v, _ in node.args))
    if all(is_constant(poly) for _, poly in node.args):
        return value, {0: value}
    return value, None


@coefficients_algebra.register(op.add, op.sub, op.mul)
def _(node: Call[Pair], *, var: Symbol) -> Pair:
    (a, pa), (b, pb) = node.args
    value = node.func(a, b)
    if pa is None or pb is None:
        return value, None
    if node.func is op.mul:
        return value, mul(pa, pb)
    return value, add(pa, pb, subtract=node.func is op.sub)


@coefficients_algebra.register(op.neg, op.pos)
def _(node: Call[Pair], *, var: Symbol) -> Pair:
    ((a, pa),) = node.args
    value = node.func(a)
    if pa is None or node.func is op.pos:
        return value, pa
    return value, {k: -c for k, c in pa.items()}


@coefficients_algebra.register(op.truediv)
def _(node: Call[Pair], *, var: Symbol) -> Pair:
    (a, pa), (b, pb) = node.args
    value = a / b
    if pa is None or not is_constant(pb):
        return value, None
    return value, {k: c / b for k, c in pa.items()}


@coefficients_algebra.register(op.pow)
def _(node: Call[Pair], *, var: Symbol) -> Pair:
    (a, pa), (b, pb) = node.args
    value = a**b
    if is_constant(pa) and is_constant(pb):
        return value, {0: value}
    exponent = simplify_exact(b)
    if pa is None or not isinstance(exponent, Literal):
        return value, None
    n = exponent.value
    if type(n) is not int or n < 0:
        return value, None
    result: Poly = {0: Literal(1)}
    for _ in range(n):
        result = mul(result, pa)
    return value, result


def coefficients(expr: Any, var: Symbol, /) -> dict[int, Expr]:
    """Coefficients of ``expr`` as a polynomial in ``var``, by degree.

    Coefficients may depend on other symbols. Raises ``ValueError`` if
    ``expr`` is not a polynomial in ``var``.

    >>> from symbolcat import Symbol, real
    >>> x, a = Symbol("x"), Symbol("a")
    >>> coeffs = coefficients((x + 1) ** 2 + a * x - real.sin(a), x)
    >>> {k: str(c) for k, c in coeffs.items()}
    {0: '1 - sin(a)', 1: '2 + a', 2: '1'}
    """
    _, poly = cata(coefficients_algebra, expr, var=var)
    if poly is None:
        raise ValueError(f"{expr} is not a polynomial in {var}")
    simplified = {k: simplify_exact(c) for k, c in sorted(poly.items())}
    return {k: c for k, c in simplified.items() if not is_number(c, 0)}
