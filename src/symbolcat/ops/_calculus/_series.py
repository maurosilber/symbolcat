"""
symbolcat.ops._calculus._series
~~~~~~~~~~~~~~~~~~~~~

Taylor series.

Every node folds into its truncated Taylor coefficients: ``diff``'s
``(value, derivative)`` pairs, extended to the first ``n`` coefficients.
Sums and products combine coefficients directly; any other function ``f``
is composed through its derivatives, which ``diff`` provides, so every
function ``diff`` knows gets a series for free.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import factorial
from typing import Any

from ... import linalg, real
from ... import operators as op
from ..._core import (
    Algebra,
    Call,
    Constant,
    Dummy,
    Expr,
    Function,
    Literal,
    Symbol,
    args_of,
    as_expr,
    cata,
    func_of,
    name_of,
    value_of,
)
from .._transform._arrays import entries, expand_arrays
from .._transform._simplify import as_fraction, is_number, simplify_exact
from .._transform._substitute import substitute
from ._diff import diff_algebra

type Series = list[Expr]

series_algebra = Algebra[Series]("series")

# Extra coefficients computed internally, consumed when a quotient has to
# cancel leading zeros (sin(x) / x at 0).
PADDING = 4
DUMMY = Dummy("t")


def zeros(n: int, /) -> Series:
    return [Literal(0)] * n


def constant(value: Expr, n: int, /) -> Series:
    return [value, *zeros(n - 1)]


def is_constant(s: Series, /) -> bool:
    return all(is_number(c, 0) for c in s[1:])


def clean(s: Series, /) -> Series:
    return [simplify_exact(c) for c in s]


def mul(a: Series, b: Series, /, *, product: Function = op.mul) -> Series:
    """The series of a product, or of any other bilinear ``product``."""
    n = len(a)
    out: Series = []
    for k in range(n):
        terms = [product(a[i], b[k - i]) for i in range(k + 1)]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        out.append(total)
    return clean(out)


def div(a: Series, b: Series, /) -> Series:
    """a / b, cancelling common leading zeros (which costs padding)."""
    while is_number(b[0], 0):
        if not is_number(a[0], 0):
            raise ValueError("The series has a pole: a negative power")
        a, b = [*a[1:], Literal(0)], [*b[1:], Literal(0)]
        if len(a) and all(is_number(c, 0) for c in b):
            raise ValueError("Division by a zero series")
    out: Series = []
    for k in range(len(a)):
        total = a[k]
        for j in range(1, k + 1):
            total = total - b[j] * out[k - j]
        out.append(simplify_exact(total / b[0]))
    return out


def derivative(expr: Expr, var: Symbol, /) -> Expr:
    # diff's own simplification folds 1 / 2 into 0.5; keep coefficients exact
    _, result = cata(diff_algebra, expr, seed={var: Literal(1)})
    return simplify_exact(result)


def compose(f: Callable[[Expr], Expr], u: Series, /) -> Series:
    """f(u) = sum_k f^(k)(u0) / k! * (u - u0) ** k, for any function diff knows.

    The fallback: repeated symbolic derivatives grow quickly, so the common
    functions use the recurrences below instead.
    """
    n = len(u)
    h = [Literal(0), *u[1:]]
    g = as_expr(f(DUMMY))
    out = zeros(n)
    power = constant(Literal(1), n)  # h ** k
    for k in range(n):
        value = simplify_exact(substitute(g, {DUMMY: u[0]}) / factorial(k))
        out = clean([o + value * p for o, p in zip(out, power, strict=True)])
        power = mul(power, h)
        g = derivative(g, DUMMY)
    return out


# Recurrences from each function's differential equation, e.g. exp(u)' =
# exp(u) * u': the coefficients of the result follow from the earlier ones,
# with O(n**2) operations.


def total(terms: Sequence[Expr], /) -> Expr:
    result = terms[0] if terms else Literal(0)
    for term in terms[1:]:
        result = result + term
    return result


def deriv(u: Series, /) -> Series:
    """Coefficients of du/dx (the last one is lost, and padded with 0)."""
    return clean([(k + 1) * u[k + 1] for k in range(len(u) - 1)] + zeros(1))


def integrate(d: Series, c0: Expr, /) -> Series:
    """The series with derivative ``d`` and value ``c0``."""
    return clean([c0] + [d[k - 1] / k for k in range(1, len(d))])


def exp_series(u: Series, /) -> Series:
    out = [simplify_exact(real.exp(u[0]))]
    for k in range(1, len(u)):
        out.append(
            simplify_exact(total([j * u[j] * out[k - j] for j in range(1, k + 1)]) / k)
        )
    return out


def sin_cos(u: Series, /, *, hyperbolic: bool = False) -> tuple[Series, Series]:
    """(sin(u), cos(u)), or (sinh(u), cosh(u)): s' = c u', c' = -+ s u'."""
    sin, cos = (real.sinh, real.cosh) if hyperbolic else (real.sin, real.cos)
    s, c = [simplify_exact(sin(u[0]))], [simplify_exact(cos(u[0]))]
    for k in range(1, len(u)):
        ds = total([j * u[j] * c[k - j] for j in range(1, k + 1)]) / k
        dc = total([j * u[j] * s[k - j] for j in range(1, k + 1)]) / k
        s.append(simplify_exact(ds))
        c.append(simplify_exact(dc if hyperbolic else -dc))
    return s, c


def power(u: Series, b: Expr, /) -> Series:
    """u ** b for a constant b: k u0 p_k = sum_j ((b + 1) j - k) u_j p_(k-j)."""
    if is_number(u[0], 0):
        raise ValueError("The series of a ** b with a -> 0 is not a power series")
    out = [simplify_exact(u[0] ** b)]
    for k in range(1, len(u)):
        terms = [((b + 1) * j - k) * u[j] * out[k - j] for j in range(1, k + 1)]
        out.append(simplify_exact(total(terms) / (k * u[0])))
    return out


def log_series(u: Series, /) -> Series:
    return integrate(div(deriv(u), u), simplify_exact(real.log(u[0])))


def atan_series(u: Series, /) -> Series:
    one_plus_u2 = clean([Literal(1) + u[0] ** 2, *mul(u, u)[1:]])
    return integrate(div(deriv(u), one_plus_u2), simplify_exact(real.atan(u[0])))


def asin_series(u: Series, /) -> Series:
    one_minus_u2 = clean([Literal(1) - c for c in mul(u, u)])
    one_minus_u2[0] = simplify_exact(Literal(1) - u[0] ** 2)
    root = power(one_minus_u2, HALF)
    return integrate(div(deriv(u), root), simplify_exact(real.asin(u[0])))


HALF = Literal(1) / 2
FUNCTIONS: dict[Function, Callable[[Series], Series]] = {
    real.exp: exp_series,
    real.log: log_series,
    real.sqrt: lambda u: power(u, HALF),
    real.sin: lambda u: sin_cos(u)[0],
    real.cos: lambda u: sin_cos(u)[1],
    real.tan: lambda u: div(*sin_cos(u)),
    real.sinh: lambda u: sin_cos(u, hyperbolic=True)[0],
    real.cosh: lambda u: sin_cos(u, hyperbolic=True)[1],
    real.tanh: lambda u: div(*sin_cos(u, hyperbolic=True)),
    real.atan: atan_series,
    real.asin: asin_series,
    # acos(u) = pi / 2 - asin(u)
    real.acos: lambda u: clean([real.acos(u[0]), *(-c for c in asin_series(u)[1:])]),
}


@series_algebra.register(Symbol)
def _(node: Symbol, *, var: Symbol, at: Expr, n: int) -> Series:
    if node == var:
        return [at, Literal(1), *zeros(n - 2)]
    return constant(node, n)


@series_algebra.register(Constant, Literal)
def _(node: Constant | Literal, *, var: Symbol, at: Expr, n: int) -> Series:
    return constant(node, n)


@series_algebra.register(Function)
def _(node: Call[Series], *, var: Symbol, at: Expr, n: int) -> Series:
    if func_of(node) in FUNCTIONS:
        return FUNCTIONS[func_of(node)](args_of(node)[0])
    if len(args_of(node)) == 1:
        return compose(func_of(node), args_of(node)[0])
    if all(is_constant(arg) for arg in args_of(node)):
        return constant(func_of(node)(*(arg[0] for arg in args_of(node))), n)
    raise NotImplementedError(
        f"series of {name_of(func_of(node))} with a varying argument"
    )


@series_algebra.register(op.add, op.sub)
def _(node: Call[Series], **ctx: Any) -> Series:
    a, b = args_of(node)
    return clean([func_of(node)(x, y) for x, y in zip(a, b, strict=True)])


@series_algebra.register(op.neg, op.pos)
def _(node: Call[Series], **ctx: Any) -> Series:
    (a,) = args_of(node)
    return clean([func_of(node)(x) for x in a])


@series_algebra.register(op.mul)
def _(node: Call[Series], **ctx: Any) -> Series:
    return mul(*args_of(node))


@series_algebra.register(op.matmul, linalg.cross)
def _(node: Call[Series], **ctx: Any) -> Series:
    a, b = args_of(node)
    return mul(a, b, product=func_of(node))


@series_algebra.register(linalg.transpose, linalg.trace, linalg.sum)
def _(node: Call[Series], **ctx: Any) -> Series:
    """Linear functions apply to each coefficient."""
    (a,) = args_of(node)
    return clean([func_of(node)(c) for c in a])


@series_algebra.register(linalg.det, linalg.inv, linalg.norm)
def _(node: Call[Series], *, n: int, **ctx: Any) -> Series:
    (a,) = args_of(node)
    if not is_constant(a):
        raise NotImplementedError(
            f"series of {name_of(func_of(node))} with a varying argument: "
            "give the entries of the array"
        )
    return constant(func_of(node)(a[0]), n)


@series_algebra.register(op.truediv)
def _(node: Call[Series], **ctx: Any) -> Series:
    return div(*args_of(node))


@series_algebra.register(op.pow)
def _(node: Call[Series], **ctx: Any) -> Series:
    base, exponent = args_of(node)
    if not is_constant(exponent):  # a ** b == exp(b * log(a))
        return exp_series(mul(exponent, log_series(base)))
    b = simplify_exact(exponent[0])
    if isinstance(b, Literal) and type(value_of(b)) is int and value_of(b) >= 0:
        result = constant(Literal(1), len(base))
        for _ in range(value_of(b)):
            result = mul(result, base)
        return result
    return power(base, b)


def term(coefficient: Expr, power: Expr | None, /) -> Expr:
    """coefficient * power, written as a person would: x ** 3 / 6, -x."""
    if power is None:
        return coefficient
    if is_number(coefficient, 1):
        return power
    if is_number(coefficient, -1):
        return -power
    if isinstance(coefficient, Call) and func_of(coefficient) is op.truediv:
        numerator, denominator = args_of(coefficient)
        if isinstance(denominator, Literal):
            return term(numerator, power) / denominator
    if isinstance(coefficient, Call) and func_of(coefficient) is op.mul:
        # a rational factor goes outside: a * x ** 2 / 2, not 1 / 2 * a * x ** 2
        for i, factor in enumerate(args_of(coefficient)):
            other = args_of(coefficient)[1 - i]
            if (value := as_fraction(factor)) is not None and value.denominator != 1:
                numerator = term(Literal(value.numerator), other)
                return term(numerator, power) / value.denominator
    return coefficient * power


def series(
    expr: Any,
    var: Symbol,
    /,
    *,
    at: Any = 0,
    order: int = 6,
) -> Expr:
    """Taylor polynomial of ``expr`` in ``var`` around ``at``.

    Terms of degree below ``order`` are kept. Operations on explicit arrays
    are first computed entry by entry (see ``expand_arrays``), and an array
    gets the series of each entry.

    >>> from symbolcat import Symbol, real
    >>> x = Symbol("x")
    >>> print(series(real.sin(x), x))
    x - x ** 3 / 6 + x ** 5 / 120
    >>> print(series(real.exp(x), x, at=1, order=3))
    exp(1) + exp(1) * (x - 1) + exp(1) * (x - 1) ** 2 / 2
    >>> print(series(real.sin(x) / x, x, order=5))
    1 - x ** 2 / 6 + x ** 4 / 120
    >>> from symbolcat.linalg import vector
    >>> print(series(real.exp(vector([x, -x])), x, order=3))
    vector([1 + x + x ** 2 / 2, 1 - x + x ** 2 / 2])
    """
    expr = expand_arrays(expr)
    if isinstance(expr, Call) and (items := entries(expr)) is not None:
        return func_of(expr)(*(series(e, var, at=at, order=order) for e in items))
    at = as_expr(at)
    n = order + PADDING
    coefficients = cata(series_algebra, expr, var=var, at=at, n=n)[:order]
    base = var if is_number(at, 0) else var - at
    result: Expr | None = None
    for k, c in enumerate(coefficients):
        if is_number(c, 0):
            continue
        power = None if k == 0 else base if k == 1 else base**k
        t = term(c, power)
        if result is None:
            result = t
        elif isinstance(t, Call) and func_of(t) is op.neg:
            result = result - args_of(t)[0]
        elif (
            isinstance(t, Call)
            and func_of(t) is op.truediv
            and isinstance(args_of(t)[0], Call)
            and func_of(args_of(t)[0]) is op.neg
        ):
            result = result - args_of(args_of(t)[0])[0] / args_of(t)[1]
        else:
            result = result + t
    return Literal(0) if result is None else result
