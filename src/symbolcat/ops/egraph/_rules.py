"""
symbolcat.ops.egraph._rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rewrite rules over ``Num`` terms.

Each ruleset is a generator: its parameters are the pattern variables
(typed by their annotation) and it yields the rules.

Rules only add equalities to the e-graph; nothing is destroyed, so the
order in which they fire does not matter. Choosing the best equivalent
expression is left to extraction (see ``_cost``).

Rulesets come in two kinds:

- ``SHRINK``: every rule makes the expression smaller, so they can run
  until nothing changes.
- ``GROW``: rules that reshape an expression and can make the e-graph grow
  without bound, so they run for a limited number of iterations.

Like ``symbolcat.ops.simplify``, some rules assume the expression is
defined: denominators are non-zero (``a / a -> 1``), and arguments of
``log`` and bases of real powers are positive (``exp(log(a)) -> a``,
``log(a * b) -> log(a) + log(b)``).
"""

from __future__ import annotations

from collections.abc import Iterator

from egglog import (
    BigInt,
    BigRat,
    Fact,
    RewriteOrRule,
    birewrite,
    eq,
    f64,
    ne,
    rewrite,
    rule,
    ruleset,
    set_cost,
    union,
)
from egglog.egraph import _RewriteBuilder

from ._term import (
    NODE,
    Num,
    abs,
    acos,
    asin,
    atan,
    cos,
    cosh,
    exp,
    log,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)

type Rules = Iterator[RewriteOrRule]


def rat(numerator: int, denominator: int = 1) -> Num:
    return Num(BigRat(numerator, denominator))


ZERO, ONE, TWO, MINUS_ONE, HALF = rat(0), rat(1), rat(2), rat(-1), rat(1, 2)


def integer(r: BigRat) -> Fact:
    return eq(r.denom).to(BigInt(1))


def nonzero(r: BigRat) -> Fact:
    return ne(r).to(BigRat(0, 1))


def replace(lhs: Num) -> _RewriteBuilder[Num]:
    """Rewrite and hide the left-hand side from matching and extraction.

    For rules whose right-hand side is always better (constant folding,
    identities): otherwise the other rules keep rewriting the left-hand side
    (``0 * 3`` is commuted, distributed, ...) and the e-graph grows for
    nothing.
    """
    return rewrite(lhs, subsume=True)


def small(r: BigRat, bound: int = 16) -> tuple[Fact, Fact]:
    """Conditions for a literal exponent ``r`` to be within ``bound``."""
    return r <= BigRat(bound, 1), r >= BigRat(-bound, 1)


@ruleset
def contradictions(p: BigRat, q: BigRat, f: f64, g: f64) -> Rules:
    """Two different numbers proven equal mean the input is undefined.

    The rules assume non-zero denominators and positive bases, so a
    division by zero or a square root of a negative number can prove, say,
    ``-1 == 1``. Constant folding would then create new numbers forever.
    Any such merge is recorded as ``0 == 1``, which stops the search.
    """
    yield rule(eq(Num(p)).to(Num(q)), ne(p).to(q)).then(union(ZERO).with_(ONE))
    yield rule(eq(Num.float(f)).to(Num.float(g)), ne(f).to(g)).then(
        union(ZERO).with_(ONE)
    )


@ruleset
def literal_costs(a: Num, p: BigRat) -> Rules:
    """Negative and fractional literals cost a bit more: ``1 / x`` over
    ``x ** -1``, ``x - 1`` over ``x + -1``, and ``x / 6`` over
    ``x * (1 / 6)``. Named constants cost as much as any leaf."""
    yield rule(eq(a).to(Num(p)), p < BigRat(0, 1)).then(set_cost(Num(p), NODE + 50))
    yield rule(eq(a).to(Num(p)), p >= BigRat(0, 1), ne(p.denom).to(BigInt(1))).then(
        set_cost(Num(p), NODE + 40)
    )
    yield rule(eq(a).to(Num.pi)).then(set_cost(Num.pi, NODE))
    yield rule(eq(a).to(Num.e)).then(set_cost(Num.e, NODE))


@ruleset
def fold(p: BigRat, q: BigRat, f: f64, g: f64) -> Rules:
    """Arithmetic on literals: exact on rationals, IEEE on floats.

    Folded left-hand sides are hidden (see ``replace``), except for
    negations, products and powers of rationals, which other rules match on
    (``x + -2 -> x - 2`` through ``-(2)``, ``4`` as ``2 * 2`` or ``2 ** 2``).
    """
    yield replace(Num(p) + Num(q)).to(Num(p + q))
    yield replace(Num(p) - Num(q)).to(Num(p - q))
    yield rewrite(Num(p) * Num(q)).to(Num(p * q))
    yield replace(Num(p) / Num(q)).to(Num(p / q), nonzero(q))
    yield rewrite(-Num(p)).to(Num(-p))  # visible: x + -2 becomes x - 2 through -(2)
    # small integer powers only; 0 ** negative is left alone
    yield rewrite(Num(p) ** Num(q)).to(Num(p**q), integer(q), nonzero(p), *small(q, 64))
    yield replace(Num.float(f) + Num.float(g)).to(Num.float(f + g))
    yield replace(Num.float(f) - Num.float(g)).to(Num.float(f - g))
    yield replace(Num.float(f) * Num.float(g)).to(Num.float(f * g))
    yield replace(Num.float(f) / Num.float(g)).to(Num.float(f / g), ne(g).to(f64(0.0)))
    yield replace(-Num.float(f)).to(Num.float(-f))
    # floats absorb rationals
    yield replace(Num(p) + Num.float(f)).to(Num.float(p.to_f64() + f))
    yield replace(Num.float(f) + Num(p)).to(Num.float(f + p.to_f64()))
    yield replace(Num(p) * Num.float(f)).to(Num.float(p.to_f64() * f))
    yield replace(Num.float(f) * Num(p)).to(Num.float(f * p.to_f64()))
    yield replace(abs(Num(p))).to(Num(p.max(-p)))
    # square roots of perfect squares (BigRat.sqrt fails on the others)
    for k in range(2, 13):
        yield replace(sqrt(rat(k * k))).to(rat(k))
        yield replace(sqrt(rat(1, k * k))).to(rat(1, k))


@ruleset
def coefficients(a: Num, p: BigRat, q: BigRat, f: f64, g: f64) -> Rules:
    """Collect numeric coefficients, so they meet and fold.

    Associativity and commutativity could find these too, but they are the
    first rules the back-off scheduler bans. These do not terminate on their
    own (the e-class of ``0`` also contains ``k * 0`` for every ``k``), so
    they belong to ``GROW``.
    """
    yield rewrite(Num(p) * (Num(q) * a)).to(Num(p * q) * a)
    yield rewrite(Num(p) * a + Num(q) * a).to(Num(p + q) * a)
    yield rewrite(a + Num(p) * a).to(Num(p + BigRat(1, 1)) * a)
    yield rewrite(Num.float(f) * (Num.float(g) * a)).to(Num.float(f * g) * a)
    yield rewrite(Num.float(f) * a + Num.float(g) * a).to(Num.float(f + g) * a)


@ruleset
def identities(a: Num) -> Rules:
    """Identities that remove operations.

    Those whose result is a constant (``a * 0 -> 0``) hide their left-hand
    side: the e-class of ``0`` would otherwise fill up with rewritings of
    ``0 * a``.
    """
    # arithmetic
    yield rewrite(a + ZERO).to(a)
    yield rewrite(a - ZERO).to(a)
    yield rewrite(ZERO - a).to(-a)
    yield rewrite(a * ONE).to(a)
    yield rewrite(a * Num.float(1.0)).to(a)
    yield rewrite(a + Num.float(0.0)).to(a)
    yield replace(a * ZERO).to(ZERO)
    yield rewrite(a * MINUS_ONE).to(-a)
    yield rewrite(a / ONE).to(a)
    yield replace(ZERO / a).to(ZERO)
    yield replace(a - a).to(ZERO)
    yield replace(a / a).to(ONE)
    neg_a = -a
    yield rewrite(-neg_a).to(a)
    yield replace(a**ZERO).to(ONE)
    yield rewrite(a**ONE).to(a)
    yield replace(ONE**a).to(ONE)
    yield replace(sqrt(ZERO)).to(ZERO)
    yield rewrite(sqrt(a) ** TWO).to(a)
    # absolute value
    yield rewrite(abs(-a)).to(abs(a))
    yield rewrite(abs(abs(a))).to(abs(a))
    yield rewrite(abs(a) ** TWO).to(a**TWO)
    yield rewrite(sqrt(a**TWO)).to(abs(a))
    # abs of values that are never negative
    yield rewrite(abs(a**TWO)).to(a**TWO)
    yield rewrite(abs(sqrt(a))).to(sqrt(a))
    yield rewrite(abs(exp(a))).to(exp(a))
    yield rewrite(abs(cosh(a))).to(cosh(a))
    # exponential and logarithm
    yield replace(exp(ZERO)).to(ONE)
    yield replace(exp(ONE)).to(Num.e)
    yield replace(log(ONE)).to(ZERO)
    yield replace(log(Num.e)).to(ONE)
    yield rewrite(log(exp(a))).to(a)
    yield rewrite(exp(log(a))).to(a)
    # trigonometric
    yield replace(sin(ZERO)).to(ZERO)
    yield replace(cos(ZERO)).to(ONE)
    yield replace(tan(ZERO)).to(ZERO)
    yield replace(sin(Num.pi)).to(ZERO)
    yield replace(cos(Num.pi)).to(MINUS_ONE)
    yield replace(sin(Num.pi / TWO)).to(ONE)
    yield replace(cos(Num.pi / TWO)).to(ZERO)
    # angles with rational sine, cosine or tangent
    for angle, sine, cosine in [
        (Num.pi / rat(6), rat(1, 2), None),
        (Num.pi / rat(3), None, rat(1, 2)),
        (Num.pi * rat(3, 2), MINUS_ONE, ZERO),
        (TWO * Num.pi, ZERO, ONE),
    ]:
        if sine is not None:
            yield replace(sin(angle)).to(sine)
        if cosine is not None:
            yield replace(cos(angle)).to(cosine)
    yield replace(tan(Num.pi / rat(4))).to(ONE)
    yield replace(sin(a) ** TWO + cos(a) ** TWO).to(ONE)
    yield rewrite(sin(asin(a))).to(a)
    yield rewrite(cos(acos(a))).to(a)
    yield rewrite(tan(atan(a))).to(a)
    yield replace(asin(ZERO)).to(ZERO)
    yield replace(atan(ZERO)).to(ZERO)
    yield replace(acos(ONE)).to(ZERO)
    yield replace(asin(ONE)).to(Num.pi / TWO)
    yield replace(acos(ZERO)).to(Num.pi / TWO)
    yield replace(atan(ONE)).to(Num.pi / rat(4))
    # hyperbolic
    yield replace(sinh(ZERO)).to(ZERO)
    yield replace(cosh(ZERO)).to(ONE)
    yield replace(tanh(ZERO)).to(ZERO)
    yield replace(cosh(a) ** TWO - sinh(a) ** TWO).to(ONE)
    yield rewrite(cosh(a) + sinh(a)).to(exp(a))
    yield rewrite(cosh(a) - sinh(a)).to(exp(-a))
    yield rewrite(exp(a) - exp(-a)).to(TWO * sinh(a))
    yield rewrite(exp(a) + exp(-a)).to(TWO * cosh(a))


@ruleset
def algebra(a: Num, b: Num, c: Num, p: BigRat, q: BigRat) -> Rules:
    """Commutative ring, and the other operations in terms of + and *."""
    yield rewrite(a + b).to(b + a)
    yield rewrite(a * b).to(b * a)
    yield birewrite(a + (b + c)).to((a + b) + c)
    yield birewrite(a * (b * c)).to((a * b) * c)
    yield birewrite(a * (b + c)).to(a * b + a * c)
    yield rewrite(a * b + a).to(a * (b + ONE))
    yield birewrite(a + a).to(TWO * a)
    yield birewrite(a - b).to(a + -b)
    yield birewrite(-a).to(MINUS_ONE * a)
    yield birewrite(a / b).to(a * b**MINUS_ONE)
    yield rewrite((a / b) / c).to(a / (b * c))
    yield rewrite(a / (b / c)).to(a * c / b)
    # factoring
    yield rewrite(a**TWO - b**TWO).to((a + b) * (a - b))
    yield rewrite(a**TWO + TWO * a * b + b**TWO).to((a + b) ** TWO)
    yield rewrite(a**TWO - TWO * a * b + b**TWO).to((a - b) ** TWO)
    # x ** 2 + p * x + q == (x + p / 2) ** 2 when q == (p / 2) ** 2
    half_p = p / BigRat(2, 1)
    yield rewrite(a**TWO + Num(p) * a + Num(q)).to(
        (a + Num(half_p)) ** TWO, eq(q).to(half_p * half_p)
    )
    # x ** 2 - k ** 2 == (x + k) * (x - k); BigRat.sqrt fails on non-squares
    for k in range(1, 13):
        yield rewrite(a**TWO - rat(k * k)).to((a + rat(k)) * (a - rat(k)))
    # sums and differences of cubes
    three = rat(3)
    yield rewrite(a**three - b**three).to((a - b) * (a**TWO + a * b + b**TWO))
    yield rewrite(a**three + b**three).to((a + b) * (a**TWO - a * b + b**TWO))
    for k in range(1, 7):
        cube, k_ = rat(k**3), rat(k)
        yield rewrite(a**three - cube).to((a - k_) * (a**TWO + k_ * a + rat(k * k)))
        yield rewrite(a**three + cube).to((a + k_) * (a**TWO - k_ * a + rat(k * k)))


@ruleset
def powers(a: Num, b: Num, c: Num, p: BigRat, q: BigRat) -> Rules:
    yield birewrite(sqrt(a)).to(a**HALF)
    yield birewrite(a * a).to(a**TWO)
    # Exponents are only combined, never split, and literal exponents stay
    # small: otherwise x ** 2 * x ** 2 -> x ** 4 -> ... grows without bound.
    yield rewrite(a**b * a**c).to(a ** (b + c))
    yield rewrite(a * a**b).to(a ** (b + ONE))
    yield rewrite((a * b) ** Num(q)).to(
        a ** Num(q) * b ** Num(q), integer(q), *small(q)
    )
    yield rewrite(a ** Num(q) * b ** Num(q)).to(
        (a * b) ** Num(q), integer(q), *small(q)
    )
    yield rewrite((a ** Num(p)) ** Num(q)).to(
        a ** Num(p * q), integer(q), *small(p * q)
    )
    # small positive integer powers as products: x ** 3 == x * x ** 2, and by
    # squaring, x ** 8 == x ** 4 * x ** 4 (DAG extraction shares the halves)
    half_p = p / BigRat(2, 1)
    yield rewrite(a ** Num(p)).to(
        a ** Num(half_p) * a ** Num(half_p),
        integer(half_p),
        p > BigRat(2, 1),
        p <= BigRat(16, 1),
    )
    yield rewrite(a ** Num(p)).to(
        a * a ** Num(p - BigRat(1, 1)),
        integer(p),
        p > BigRat(1, 1),
        p <= BigRat(16, 1),
    )
    yield rewrite(abs(a * b)).to(abs(a) * abs(b))
    # literal factors out of roots and into squares: sqrt(4 * x) == 2 * sqrt(x)
    yield rewrite(sqrt(Num(p) * a)).to(sqrt(Num(p)) * sqrt(a), p > BigRat(0, 1))
    for k in range(2, 13):
        yield rewrite(rat(k * k) * a**TWO).to((rat(k) * a) ** TWO)


@ruleset
def exponential(a: Num, b: Num) -> Rules:
    yield birewrite(exp(a) * exp(b)).to(exp(a + b))
    yield rewrite(exp(a) ** b).to(exp(a * b))
    yield birewrite(Num.e**a).to(exp(a))
    yield birewrite(log(a * b)).to(log(a) + log(b))
    # one way: the reverse folds any product into a logarithm
    yield rewrite(log(a**b)).to(b * log(a))
    yield rewrite(exp(b * log(a))).to(a**b)


@ruleset
def trigonometric(a: Num, b: Num) -> Rules:
    yield rewrite(sin(-a)).to(-sin(a))
    yield rewrite(cos(-a)).to(cos(a))
    yield birewrite(tan(a)).to(sin(a) / cos(a))
    yield birewrite(sin(a + b)).to(sin(a) * cos(b) + cos(a) * sin(b))
    yield birewrite(cos(a + b)).to(cos(a) * cos(b) - sin(a) * sin(b))
    yield birewrite(sin(a) ** TWO).to(ONE - cos(a) ** TWO)
    yield rewrite(ONE + tan(a) ** TWO).to(cos(a) ** rat(-2))


@ruleset
def hyperbolic(a: Num, b: Num) -> Rules:
    yield rewrite(sinh(-a)).to(-sinh(a))
    yield rewrite(cosh(-a)).to(cosh(a))
    yield birewrite(tanh(a)).to(sinh(a) / cosh(a))
    yield birewrite(sinh(a)).to((exp(a) - exp(-a)) / TWO)
    yield birewrite(cosh(a)).to((exp(a) + exp(-a)) / TWO)
    yield birewrite(sinh(a + b)).to(sinh(a) * cosh(b) + cosh(a) * sinh(b))
    yield birewrite(cosh(a + b)).to(cosh(a) * cosh(b) + sinh(a) * sinh(b))
    yield rewrite(ONE - tanh(a) ** TWO).to(cosh(a) ** rat(-2))


SHRINK = (contradictions, fold, identities, literal_costs)
GROW = (coefficients, algebra, powers, exponential, trigonometric, hyperbolic)
