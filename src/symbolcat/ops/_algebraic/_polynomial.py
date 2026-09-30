"""
symbolcat.ops._algebraic._polynomial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Polynomials with rational coefficients: division, Gröbner bases, GCDs,
square-free factorization, rational roots, and the cancellation of rational
functions.

A polynomial is a dict from monomials (tuples of exponents, one per
generator) to nonzero ``Fraction`` coefficients. The generators are
symbols, or for ``cancel`` any sub-expression that is not a rational
operation (``sin(x)``, a float): an *atom*.

Monomial orders: ``"lex"`` compares the exponents of the generators in
order, ``"grevlex"`` compares the total degree first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from itertools import combinations
from typing import Any

from ..._core import (
    Call,
    Done,
    Expr,
    Literal,
    Symbol,
    args_of,
    as_expr,
    func_of,
    hylo,
    value_of,
)
from ...lib import operators as op
from .._analysis._inspect import free_symbols
from .._transform._simplify import from_fraction

type Monomial = tuple[int, ...]
type Poly = dict[Monomial, Fraction]
type Key = Callable[[Monomial], Any]

ORDERS: dict[str, Key] = {
    "lex": lambda m: m,
    "grevlex": lambda m: (sum(m), tuple(-e for e in reversed(m))),
}


class NotPolynomial(ValueError):
    """The expression is not a polynomial (or rational function) with
    rational coefficients in the generators."""


# Arithmetic


def _const(value: Fraction | int, n: int, /) -> Poly:
    return {(0,) * n: Fraction(value)} if value else {}


def _add(a: Poly, b: Poly, /, scale: Fraction = Fraction(1)) -> Poly:
    out = dict(a)
    for m, c in b.items():
        value = out.get(m, Fraction(0)) + scale * c
        if value:
            out[m] = value
        else:
            out.pop(m, None)
    return out


def _mul(a: Poly, b: Poly, /) -> Poly:
    out: Poly = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            m = tuple(x + y for x, y in zip(ma, mb, strict=True))
            value = out.get(m, Fraction(0)) + ca * cb
            if value:
                out[m] = value
            else:
                out.pop(m, None)
    return out


def _term(p: Poly, mono: Monomial, coeff: Fraction, /) -> Poly:
    """``p`` times ``coeff * mono``."""
    return {
        tuple(x + y for x, y in zip(m, mono, strict=True)): c * coeff
        for m, c in p.items()
    }


def _pow(p: Poly, n: int, size: int, /) -> Poly:
    out = _const(1, size)
    for _ in range(n):
        out = _mul(out, p)
    return out


def _leading(p: Poly, key: Key, /) -> tuple[Monomial, Fraction]:
    m = max(p, key=key)
    return m, p[m]


def _monic(p: Poly, key: Key, /) -> Poly:
    if not p:
        return p
    _, c = _leading(p, key)
    return {m: v / c for m, v in p.items()}


def _divides(a: Monomial, b: Monomial, /) -> bool:
    return all(x <= y for x, y in zip(a, b, strict=True))


def _quotient(b: Monomial, a: Monomial, /) -> Monomial:
    return tuple(y - x for x, y in zip(a, b, strict=True))


def _divide(f: Poly, divisors: Sequence[Poly], key: Key, /) -> tuple[list[Poly], Poly]:
    """Multivariate division: ``f = sum(q * g) + r``, where no term of ``r``
    is divisible by a leading term of the divisors."""
    quotients: list[Poly] = [{} for _ in divisors]
    leads = [_leading(g, key) for g in divisors]
    remainder: Poly = {}
    p = dict(f)
    while p:
        m, c = _leading(p, key)
        for i, (gm, gc) in enumerate(leads):
            if _divides(gm, m):
                mono, coeff = _quotient(m, gm), c / gc
                quotients[i] = _add(quotients[i], {mono: coeff})
                p = _add(p, _term(divisors[i], mono, coeff), -Fraction(1))
                break
        else:
            remainder[m] = c
            del p[m]
    return quotients, remainder


def _exact_divide(f: Poly, g: Poly, key: Key, /) -> Poly:
    (q,), r = _divide(f, [g], key)
    if r:
        raise ArithmeticError("the division is not exact")
    return q


def _groebner(polys: Iterable[Poly], key: Key, /) -> list[Poly]:
    """The reduced Gröbner basis, by Buchberger's algorithm (skipping pairs
    with coprime leading monomials)."""
    basis = [_monic(p, key) for p in polys if p]
    pairs = list(combinations(range(len(basis)), 2))
    while pairs:
        i, j = pairs.pop()
        (mi, _), (mj, _) = _leading(basis[i], key), _leading(basis[j], key)
        if all(not (x and y) for x, y in zip(mi, mj, strict=True)):
            continue
        lcm = tuple(max(x, y) for x, y in zip(mi, mj, strict=True))
        s = _add(
            _term(basis[i], _quotient(lcm, mi), Fraction(1)),
            _term(basis[j], _quotient(lcm, mj), Fraction(1)),
            -Fraction(1),
        )
        _, r = _divide(s, basis, key)
        if r:
            basis.append(_monic(r, key))
            pairs.extend((k, len(basis) - 1) for k in range(len(basis) - 1))
    # Minimize: drop elements whose leading monomial another one divides.
    minimal: list[Poly] = []
    for k, p in enumerate(basis):
        m = _leading(p, key)[0]
        if not any(
            _divides(_leading(q, key)[0], m) and (_leading(q, key)[0] != m or other < k)
            for other, q in enumerate(basis)
            if other != k
        ):
            minimal.append(p)
    # Reduce each element by the others.
    reduced = []
    for k, p in enumerate(minimal):
        others = minimal[:k] + minimal[k + 1 :]
        _, r = _divide(p, others, key) if others else ([], p)
        reduced.append(_monic(r, key))
    return sorted(reduced, key=lambda p: key(_leading(p, key)[0]))


def _variables_used(p: Poly, /) -> set[int]:
    return {i for m in p for i, e in enumerate(m) if e}


def _gcd(a: Poly, b: Poly, n: int, /) -> Poly:
    """The monic GCD (lex order). Euclid's algorithm for univariate
    polynomials; otherwise from the LCM, the generator of the intersection
    of the ideals, eliminating an extra variable ``t``: <t a, (1 - t) b>."""
    key = ORDERS["lex"]
    if not a or not b:
        return _monic(a or b, key)
    used = _variables_used(a) | _variables_used(b)
    if not used:
        return _const(1, n)
    if len(used) == 1:
        while b:
            _, r = _divide(a, [b], key)
            a, b = b, r
        return _monic(a, key)
    lift: Callable[[Poly], Poly] = lambda p: {(0, *m): c for m, c in p.items()}
    t: Poly = {(1,) + (0,) * n: Fraction(1)}
    one_minus_t = _add(_const(1, n + 1), t, -Fraction(1))
    basis = _groebner([_mul(t, lift(a)), _mul(one_minus_t, lift(b))], key)
    # The elimination ideal, without t, is principal: generated by the LCM.
    (lcm, *_) = [
        {m[1:]: c for m, c in g.items()} for g in basis if 0 not in _variables_used(g)
    ]
    return _monic(_exact_divide(_mul(a, b), lcm, key), key)


# Conversion


_RATIONAL_OPS = (op.add, op.sub, op.mul, op.truediv, op.neg, op.pos)


def _rational_layer(node: Expr, /) -> Call[Any] | None:
    """``node`` as one rational operation of its arguments, or ``None``.

    The exponent of a power is an ``int`` (as ``Done``), not an argument to
    unfold. Shared by the coalgebras of ``_rational`` and ``_atoms``, which
    must not open the generators (``sin(x)``), as a ``cata`` would.
    """
    if not isinstance(node, Call):
        return None
    if func_of(node) in _RATIONAL_OPS:
        return node
    if func_of(node) is op.pow:
        base, exponent = args_of(node)
        if isinstance(exponent, Literal) and type(value_of(exponent)) is int:
            return Call(op.pow, (base, Done(value_of(exponent))))
    return None


def _rational(
    expr: Expr, index: dict[Expr, int], /, *, polynomial: bool = False
) -> tuple[Poly, Poly]:
    """``expr`` as a quotient of polynomials in the generators of ``index``.

    With ``polynomial``, only division by constants is allowed.
    """
    n = len(index)

    def coalgebra(node: Expr, /) -> Any:
        if node in index:
            mono = tuple(1 if i == index[node] else 0 for i in range(n))
            return Done(({mono: Fraction(1)}, _const(1, n)))
        if isinstance(node, Literal) and type(value_of(node)) in (int, Fraction):
            return Done((_const(value_of(node), n), _const(1, n)))
        if (layer := _rational_layer(node)) is None:
            raise NotPolynomial(f"{node} is not a rational operation of the generators")
        # The node itself goes last, to name it in errors.
        return Call(func_of(layer), (*args_of(layer), Done(node)))

    def algebra(layer: Call[Any], /) -> tuple[Poly, Poly]:
        *args, node = args_of(layer)
        if func_of(layer) in (op.add, op.sub):
            (an, ad), (bn, bd) = args
            sign = Fraction(1) if func_of(layer) is op.add else -Fraction(1)
            return (_add(_mul(an, bd), _mul(bn, ad), sign), _mul(ad, bd))
        if func_of(layer) is op.mul:
            (an, ad), (bn, bd) = args
            return (_mul(an, bn), _mul(ad, bd))
        if func_of(layer) is op.truediv:
            (an, ad), (bn, bd) = args
            if not bn:
                raise ZeroDivisionError(f"division by zero in {node}")
            if polynomial and _variables_used(bn) | _variables_used(bd):
                raise NotPolynomial(f"{node} divides by a polynomial")
            return (_mul(an, bd), _mul(ad, bn))
        if func_of(layer) in (op.neg, op.pos):
            ((an, ad),) = args
            return (
                {m: -c for m, c in an.items()} if func_of(layer) is op.neg else an,
                ad,
            )
        (bn, bd), exponent = args  # op.pow
        if exponent < 0:
            if polynomial:
                raise NotPolynomial(f"{node} has a negative exponent")
            bn, bd, exponent = bd, bn, -exponent
        return (_pow(bn, exponent, n), _pow(bd, exponent, n))

    return hylo(algebra, coalgebra, expr)


def _atoms(expr: Expr, /) -> list[Expr]:
    """The maximal sub-expressions that are not rational operations."""

    def coalgebra(node: Expr, /) -> Any:
        if (layer := _rational_layer(node)) is not None:
            return layer
        if isinstance(node, Literal) and type(value_of(node)) in (int, Fraction):
            return Done({})
        return Done({node: None})

    # Folds into an insertion-ordered set (a dict with None values).
    def algebra(layer: Call[Any], /) -> dict[Expr, None]:
        out: dict[Expr, None] = {}
        for arg in args_of(layer):
            if isinstance(arg, dict):  # not the exponent of a power
                out |= arg
        return out

    return list(hylo(algebra, coalgebra, expr))


def _poly(expr: Any, gens: Sequence[Expr], /) -> Poly:
    expr = as_expr(expr)
    index = {g: i for i, g in enumerate(gens)}
    num, den = _rational(expr, index, polynomial=True)
    ((_, c),) = den.items()
    return {m: v / c for m, v in num.items()}


def _gens(exprs: Iterable[Any], gens: Sequence[Expr] | None, /) -> tuple[Expr, ...]:
    if gens is not None:
        return tuple(gens)
    return tuple(dict.fromkeys(s for e in exprs for s in free_symbols(e)))


def _to_expr(p: Poly, gens: Sequence[Expr], key: Key = ORDERS["lex"], /) -> Expr:
    """An expression for ``p``, with its terms in decreasing order."""
    # A loop over the sorted terms, not an ``ana``: a ``Poly`` is flat, and
    # the sum is built left-nested, as the operators would build it.
    total: Expr | None = None
    for m in sorted(p, key=key, reverse=True):
        c = p[m]
        factors = [g if e == 1 else g**e for g, e in zip(gens, m, strict=True) if e]
        monomial: Expr | None = None
        for f in factors:
            monomial = f if monomial is None else monomial * f
        magnitude = abs(c)
        if monomial is None:
            term = from_fraction(magnitude)
        elif magnitude == 1:
            term = monomial
        else:
            term = from_fraction(magnitude) * monomial
        if total is None:
            total = -term if c < 0 else term
        else:
            total = total - term if c < 0 else total + term
    return Literal(0) if total is None else total


# Public functions


def groebner(
    polynomials: Iterable[Any],
    gens: Sequence[Symbol] | None = None,
    /,
    *,
    order: str = "lex",
) -> tuple[Expr, ...]:
    """The reduced Gröbner basis of the ideal of ``polynomials`` (with
    rational coefficients), in the generators ``gens`` (by default, their
    symbols in order of appearance), for a monomial ``order``.

    With ``"lex"``, the basis is triangular: its last elements only depend
    on the last generators, so a polynomial system can be solved by back
    substitution.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> for p in groebner([x**2 + y**2 - 1, x - y], [x, y]):
    ...     print(p)
    y ** 2 - 1 / 2
    x - y
    """
    polynomials = list(polynomials)
    generators = _gens(polynomials, gens)
    key = ORDERS[order]
    basis = _groebner([_poly(p, generators) for p in polynomials], key)
    return tuple(_to_expr(p, generators, key) for p in basis)


def polynomial_divide(
    f: Any,
    divisors: Sequence[Any],
    gens: Sequence[Symbol] | None = None,
    /,
    *,
    order: str = "lex",
) -> tuple[list[Expr], Expr]:
    """Quotients and remainder of the multivariate division of ``f`` by
    ``divisors``: ``f = sum(q * g) + r``.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> (q,), r = polynomial_divide(x**3 - 1, [x - 1])
    >>> print(q, "|", r)
    x ** 2 + x + 1 | 0
    """
    generators = _gens([f, *divisors], gens)
    key = ORDERS[order]
    quotients, remainder = _divide(
        _poly(f, generators), [_poly(g, generators) for g in divisors], key
    )
    return [_to_expr(q, generators, key) for q in quotients], _to_expr(
        remainder, generators, key
    )


def polynomial_gcd(a: Any, b: Any, gens: Sequence[Symbol] | None = None, /) -> Expr:
    """The monic greatest common divisor of two polynomials.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(polynomial_gcd(x**2 - y**2, x**2 + 2 * x * y + y**2))
    x + y
    """
    generators = _gens([a, b], gens)
    return _to_expr(
        _gcd(_poly(a, generators), _poly(b, generators), len(generators)), generators
    )


def polynomial_lcm(a: Any, b: Any, gens: Sequence[Symbol] | None = None, /) -> Expr:
    """The monic least common multiple of two polynomials.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> print(polynomial_lcm(x**2 - 1, x + 1))
    x ** 2 - 1
    """
    generators = _gens([a, b], gens)
    key = ORDERS["lex"]
    pa, pb = _poly(a, generators), _poly(b, generators)
    gcd = _gcd(pa, pb, len(generators))
    return _to_expr(_monic(_exact_divide(_mul(pa, pb), gcd, key), key), generators)


def _univariate(expr: Any, var: Symbol, /) -> Poly:
    return _poly(expr, [var])


def _derivative(p: Poly, /) -> Poly:
    return {(m[0] - 1,): c * m[0] for m, c in p.items() if m[0]}


def square_free_factorization(expr: Any, var: Symbol, /) -> list[tuple[Expr, int]]:
    """Yun's algorithm: ``expr = c * prod(f ** k)`` for monic, square-free and
    pairwise coprime factors ``f``, returned with their multiplicities ``k``.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> for factor, k in square_free_factorization((x - 1) ** 2 * (x + 2), x):
    ...     print(factor, k)
    x + 2 1
    x - 1 2
    """
    key = ORDERS["lex"]
    f = _univariate(expr, var)
    df = _derivative(f)
    a = _gcd(f, df, 1)
    b = _exact_divide(f, a, key)
    c = _exact_divide(df, a, key)
    d = _add(c, _derivative(b), -Fraction(1))
    out = []
    k = 1
    while _variables_used(b):
        a = _gcd(b, d, 1)
        b = _exact_divide(b, a, key)
        c = _exact_divide(d, a, key)
        d = _add(c, _derivative(b), -Fraction(1))
        if _variables_used(a):
            out.append((_to_expr(_monic(a, key), [var]), k))
        k += 1
    return out


def _divisors(n: int, /) -> list[int]:
    n = abs(n)
    small = [d for d in range(1, int(n**0.5) + 1) if n % d == 0]
    return sorted({*small, *(n // d for d in small)})


def rational_roots(expr: Any, var: Symbol, /) -> list[Expr]:
    """The distinct rational roots of a polynomial with rational
    coefficients, in increasing order (the rational root theorem).

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> [str(r) for r in rational_roots(2 * x**3 - x**2 - 2 * x + 1, x)]
    ['-1', '1 / 2', '1']
    """
    p = _univariate(expr, var)
    if not p:
        raise ValueError("rational_roots: the zero polynomial")
    lowest = min(m[0] for m in p)
    roots: set[Fraction] = {Fraction(0)} if lowest else set()
    p = {(m[0] - lowest,): c for m, c in p.items()}
    scale = 1
    for c in p.values():
        scale = scale * c.denominator // _int_gcd(scale, c.denominator)
    coefficients = {m[0]: int(c * scale) for m, c in p.items()}
    degree = max(coefficients)
    for num in _divisors(coefficients.get(0, 0)) if degree else []:
        for den in _divisors(coefficients[degree]):
            for candidate in (Fraction(num, den), Fraction(-num, den)):
                if sum(c * candidate**e for e, c in coefficients.items()) == 0:
                    roots.add(candidate)
    return [from_fraction(r) for r in sorted(roots)]


def _int_gcd(a: int, b: int, /) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def cancel(expr: Any, /) -> Expr:
    """Write ``expr`` as a single fraction ``numerator / denominator`` of
    polynomials without common factors.

    Sub-expressions that are not rational operations (``sin(x)``, a float)
    are treated as generators.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(cancel((x**2 - 1) / (x + 1)))
    x - 1
    >>> print(cancel(1 / x + 1 / y))
    (x + y) / (x * y)
    >>> print(cancel(real.sin(x) / real.sin(x) ** 2))
    1 / sin(x)
    """
    expr = as_expr(expr)
    gens = _atoms(expr)
    index = {g: i for i, g in enumerate(gens)}
    num, den = _rational(expr, index)
    if not num:
        return Literal(0)
    key = ORDERS["lex"]
    gcd = _gcd(num, den, len(gens))
    num, den = _exact_divide(num, gcd, key), _exact_divide(den, gcd, key)
    _, lead = _leading(den, key)
    num = {m: c / lead for m, c in num.items()}
    den = {m: c / lead for m, c in den.items()}
    numerator = _to_expr(num, gens)
    if den == _const(1, len(gens)):
        return numerator
    return numerator / _to_expr(den, gens)
