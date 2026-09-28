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

Terms can also be arrays, where operations apply entrywise: the scalar
rules still hold entry by entry. The matrix functions have their own
rules (``matrices``), valid for every shape.

Rules are valid wherever the expression is defined: ``a / a -> 1`` and
``exp(log(a)) -> a`` only extend the domain. Rules that could change the
value where the expression is defined, like ``log(a * b) -> log(a) +
log(b)`` (false for negative ``a`` and ``b``), only fire under assumptions:
properties of terms stored as relations (``is_positive``, ...), given by
the user (``_assume``) or derived by ``signs``.
"""

from __future__ import annotations

from collections.abc import Iterator

from egglog import (
    BigInt,
    BigRat,
    Fact,
    RewriteOrRule,
    String,
    Unit,
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
    det,
    exp,
    inv,
    is_integer,
    is_negative,
    is_nonnegative,
    is_nonzero,
    is_positive,
    log,
    matmul,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trace,
    transpose,
)

type Rules = Iterator[RewriteOrRule]


def rat(numerator: int, denominator: int = 1, /) -> Num:
    return Num(BigRat(numerator, denominator))


ZERO, ONE, TWO, MINUS_ONE, HALF = rat(0), rat(1), rat(2), rat(-1), rat(1, 2)


def integer(r: BigRat, /) -> Fact:
    return eq(r.denom).to(BigInt(1))


def nonzero(r: BigRat, /) -> Unit:
    return ne(r).to(BigRat(0, 1))


def replace(lhs: Num, /) -> _RewriteBuilder[Num]:
    """Rewrite and hide the left-hand side from matching and extraction.

    For rules whose right-hand side is always better (constant folding,
    identities): otherwise the other rules keep rewriting the left-hand side
    (``0 * 3`` is commuted, distributed, ...) and the e-graph grows for
    nothing.
    """
    return rewrite(lhs, subsume=True)


def small(r: BigRat, /, *, bound: int = 16) -> tuple[Unit, Unit]:
    """Conditions for a literal exponent ``r`` to be within ``bound``."""
    return r <= BigRat(bound, 1), r >= BigRat(-bound, 1)


@ruleset
def contradictions(p: BigRat, q: BigRat) -> Rules:
    """Two different numbers proven equal mean the input is undefined.

    The rules assume non-zero denominators and positive bases, so a
    division by zero or a square root of a negative number can prove, say,
    ``-1 == 1``. Constant folding would then create new numbers forever.
    Any such merge is recorded as ``0 == 1``, which stops the search.
    """
    yield rule(eq(Num(p)).to(Num(q)), ne(p).to(q)).then(union(ZERO).with_(ONE))
    # Not for floats: they are approximate, so equal values can fold to
    # different floats (rounding, and underflow: 0.5 ** 1100 == 0.0).


@ruleset
def signs(a: Num, b: Num, c: Num, p: BigRat) -> Rules:
    """Derive properties of terms. Only adds facts, so it terminates.

    Assumptions that contradict each other or a literal are recorded as
    ``0 == 1``, like other contradictions: the input is then undefined.
    """
    # literals and constants
    yield rule(eq(a).to(Num(p)), p > BigRat(0, 1)).then(is_positive(a))
    yield rule(eq(a).to(Num(p)), p < BigRat(0, 1)).then(is_negative(a))
    # no signs from floats: rounding residues (2.2e-16) end up merged with
    # exact zeros, which would look like a contradiction
    yield rule(eq(a).to(Num.pi)).then(is_positive(a))
    yield rule(eq(a).to(Num.e)).then(is_positive(a))
    # implications between properties
    yield rule(is_positive(a)).then(is_nonnegative(a), is_nonzero(a))
    yield rule(is_negative(a)).then(is_nonzero(a))
    # sums, products and quotients
    for op in (Num.__add__, Num.__mul__, Num.__truediv__):
        yield rule(eq(c).to(op(a, b)), is_positive(a), is_positive(b)).then(
            is_positive(c)
        )
    for op in (Num.__add__, Num.__mul__):
        yield rule(eq(c).to(op(a, b)), is_nonnegative(a), is_nonnegative(b)).then(
            is_nonnegative(c)
        )
    yield rule(eq(c).to(a + b), is_negative(a), is_negative(b)).then(is_negative(c))
    for op in (Num.__mul__, Num.__truediv__):
        yield rule(eq(c).to(op(a, b)), is_negative(a), is_negative(b)).then(
            is_positive(c)
        )
        yield rule(eq(c).to(op(a, b)), is_positive(a), is_negative(b)).then(
            is_negative(c)
        )
        yield rule(eq(c).to(op(a, b)), is_negative(a), is_positive(b)).then(
            is_negative(c)
        )
        yield rule(eq(c).to(op(a, b)), is_nonzero(a), is_nonzero(b)).then(is_nonzero(c))
    yield rule(eq(c).to(-a), is_positive(a)).then(is_negative(c))
    yield rule(eq(c).to(-a), is_negative(a)).then(is_positive(c))
    yield rule(eq(c).to(-a), is_nonzero(a)).then(is_nonzero(c))
    # functions and powers
    yield rule(eq(c).to(exp(a))).then(is_positive(c))
    yield rule(eq(c).to(cosh(a))).then(is_positive(c))
    yield rule(eq(c).to(abs(a))).then(is_nonnegative(c))
    yield rule(eq(c).to(abs(a)), is_nonzero(a)).then(is_positive(c))
    yield rule(eq(c).to(a**TWO)).then(is_nonnegative(c))
    yield rule(eq(c).to(a**TWO), is_nonzero(a)).then(is_positive(c))
    yield rule(eq(c).to(a**b), is_positive(a)).then(is_positive(c))
    yield rule(eq(c).to(sqrt(a)), is_nonnegative(a)).then(is_nonnegative(c))
    yield rule(eq(c).to(sqrt(a)), is_positive(a)).then(is_positive(c))
    # integers (literals are not marked as integers: that would enable the
    # integer rules in ``assumed`` for every literal exponent, duplicating
    # the bounded ones in ``powers``)
    for op in (Num.__add__, Num.__sub__, Num.__mul__):
        yield rule(eq(c).to(op(a, b)), is_integer(a), is_integer(b)).then(is_integer(c))
        yield rule(eq(c).to(op(a, Num(p))), is_integer(a), integer(p)).then(
            is_integer(c)
        )
        yield rule(eq(c).to(op(Num(p), a)), is_integer(a), integer(p)).then(
            is_integer(c)
        )
    yield rule(eq(c).to(-a), is_integer(a)).then(is_integer(c))
    # contradictions
    contradiction = union(ZERO).with_(ONE)
    yield rule(is_positive(a), is_negative(a)).then(contradiction)
    yield rule(is_nonnegative(a), is_negative(a)).then(contradiction)
    yield rule(is_positive(a), eq(a).to(ZERO)).then(contradiction)
    yield rule(is_nonzero(a), eq(a).to(ZERO)).then(contradiction)
    yield rule(is_integer(a), eq(a).to(Num(p)), ne(p.denom).to(BigInt(1))).then(
        contradiction
    )


@ruleset
def assumed(a: Num, b: Num, c: Num) -> Rules:
    """Rules that need properties of their terms."""
    # logarithms: log(a * b) is defined when a * b > 0, so a > 0 is enough
    yield rewrite(log(a * b)).to(log(a) + log(b), is_positive(a))
    yield rewrite(log(a * b)).to(log(a) + log(b), is_positive(b))
    yield rewrite(log(a**b)).to(b * log(a), is_positive(a))
    # absolute values and roots
    yield rewrite(abs(a)).to(a, is_nonnegative(a))
    yield rewrite(abs(a)).to(-a, is_negative(a))
    yield rewrite(sqrt(a * b)).to(sqrt(a) * sqrt(b), is_positive(a))
    yield rewrite(sqrt(a * b)).to(sqrt(a) * sqrt(b), is_positive(b))
    # powers with any exponent
    # (literal integer exponents are handled by ``powers``, with a bound)
    # (k * y) ** c == k ** c * y ** c for k > 0 wherever it is defined
    yield rewrite((a * b) ** c).to(a**c * b**c, is_positive(a))
    yield rewrite((a * b) ** c).to(a**c * b**c, is_positive(b))
    yield rewrite((a * b) ** c).to(a**c * b**c, is_integer(c))
    yield rewrite(a**c * b**c).to((a * b) ** c, is_nonnegative(a), is_nonnegative(b))
    yield rewrite((a**b) ** c).to(a ** (b * c), is_nonnegative(a))
    yield rewrite((a**b) ** c).to(a ** (b * c), is_integer(c))
    yield rewrite(a**b).to(exp(b * log(a)), is_positive(a))
    yield rewrite(ZERO**a).to(ZERO, is_positive(a))
    # integer multiples of pi
    yield rewrite(sin(Num.pi * a)).to(ZERO, is_integer(a))
    yield rewrite(tan(Num.pi * a)).to(ZERO, is_integer(a))
    yield rewrite(cos(Num.pi * a)).to(MINUS_ONE**a, is_integer(a))
    yield rewrite(MINUS_ONE ** (TWO * a)).to(ONE, is_integer(a))
    yield rewrite(sin(Num.pi * (a + HALF))).to(MINUS_ONE**a, is_integer(a))
    yield rewrite(cos(Num.pi * (a + HALF))).to(ZERO, is_integer(a))
    yield rewrite(atan(a) + atan(ONE / a)).to(Num.pi / TWO, is_positive(a))
    yield rewrite(atan(a) + atan(ONE / a)).to(-Num.pi / TWO, is_negative(a))


@ruleset
def literal_costs(a: Num, p: BigRat, q: BigRat) -> Rules:
    """Negative and fractional literals cost a bit more: ``1 / x`` over
    ``x ** -1``, ``x - 1`` over ``x + -1``, and ``x / 6`` over
    ``x * (1 / 6)``. Named constants cost as much as any leaf."""
    yield rule(eq(a).to(Num(p)), p < BigRat(0, 1)).then(set_cost(Num(p), NODE + 50))
    yield rule(eq(a).to(Num(p)), p >= BigRat(0, 1), ne(p.denom).to(BigInt(1))).then(
        set_cost(Num(p), NODE + 40)
    )
    # roots of fractions cost more than roots of integers: sqrt(3) / 2 over
    # sqrt(3/4), (3/4) ** (1/2) and 2 ** (-1/2)
    fraction = ne(p.denom).to(BigInt(1))
    yield rule(eq(a).to(sqrt(Num(p))), fraction).then(set_cost(sqrt(Num(p)), 5 * NODE))
    yield rule(eq(a).to(Num(p) ** Num(q)), ne(q.denom).to(BigInt(1))).then(
        set_cost(Num(p) ** Num(q), 5 * NODE)
    )
    # and roots with a square factor a bit more: sqrt(3) / 2 over sqrt(12) / 4
    for k in range(2, 11):
        yield rule(
            eq(a).to(sqrt(Num(p))), p > BigRat(0, 1), integer(p / BigRat(k * k, 1))
        ).then(set_cost(sqrt(Num(p)), NODE + 20))
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
    yield rewrite(Num(p) ** Num(q)).to(
        Num(p**q), integer(q), nonzero(p), *small(q, bound=64)
    )
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
    # arithmetic, with the neutral element on either side (commutativity is
    # a growing rule, so it may not get there)
    for zero, one in [(ZERO, ONE), (Num.float(0.0), Num.float(1.0))]:
        yield rewrite(a + zero).to(a)
        yield rewrite(zero + a).to(a)
        yield rewrite(a * one).to(a)
        yield rewrite(one * a).to(a)
    yield rewrite(a - ZERO).to(a)
    yield rewrite(ZERO - a).to(-a)
    yield replace(a * ZERO).to(ZERO)
    yield replace(ZERO * a).to(ZERO)
    yield rewrite(a * MINUS_ONE).to(-a)
    yield rewrite(MINUS_ONE * a).to(-a)
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
    yield replace(cos(a) ** TWO + sin(a) ** TWO).to(ONE)
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
def algebra(a: Num, b: Num, c: Num, d: Num, p: BigRat, q: BigRat) -> Rules:
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
    # from Metatheory.jl's CAS theory
    yield rewrite(a + b * a).to((b + ONE) * a)
    yield rewrite(a / (a / b)).to(b)
    yield rewrite(a * (b / a)).to(b)
    yield birewrite(a * (b / c)).to((a * b) / c)
    # adding fractions, from Herbie and SymbolicUtils.jl: no rule introduces
    # d / d, so these are not derivable
    yield rewrite(a / b + c / d).to((a * d + b * c) / (b * d))
    yield rewrite(a / b - c / d).to((a * d - b * c) / (b * d))
    yield rewrite(c + a / b).to((c * b + a) / b)
    yield rewrite(c - a / b).to((c * b - a) / b)
    # rationalized roots (these extend the domain where the denominator is 0)
    yield rewrite((a - b) / (sqrt(a) + sqrt(b))).to(sqrt(a) - sqrt(b))
    yield rewrite((a**TWO - b) / (a + sqrt(b))).to(a - sqrt(b))
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
def horner(b: Num, c: Num, d: Num, p: BigRat, name: String) -> Rules:
    """Steps towards Horner's form in a symbol, c0 + x * (c1 + x * (c2 + ...)).

    Factor the symbol out of each monomial, then out of the sum, and repeat
    inside. They run on their own before the search (``PREPARE``): mixed
    with the other growing rules, commutativity and distributivity would
    rewrite every intermediate sum, and the search would stop at its node
    budget long before a polynomial of high degree reaches Horner's form.
    """
    x = Num.var(name)
    # factor out of a left-nested sum without reassociating it
    yield rewrite((c + x * b) + x * d).to(c + x * (b + d))
    yield rewrite((c + x) + x * d).to(c + x * (ONE + d))
    yield rewrite((c + x * b) + x).to(c + x * (b + ONE))
    yield rewrite(x * b + x * d).to(x * (b + d))
    # factor out of a monomial: c * x ** 3 == x * (c * x ** 2)
    below = x ** Num(p - BigRat(1, 1))
    in_range = (integer(p), p > BigRat(1, 1), p <= BigRat(64, 1))
    yield rewrite(b * x ** Num(p)).to(x * (b * below), *in_range)
    yield rewrite(x ** Num(p)).to(x * below, *in_range)
    # symbol first in products, so the steps above match: 3 * x -> x * 3
    yield rewrite(b * x).to(x * b)


@ruleset
def powers(a: Num, b: Num, c: Num, p: BigRat, q: BigRat) -> Rules:
    yield birewrite(sqrt(a)).to(a**HALF)
    yield birewrite(a * a).to(a**TWO)
    # Exponents are only combined, never split, and literal exponents stay
    # small: otherwise x ** 2 * x ** 2 -> x ** 4 -> ... grows without bound.
    yield rewrite(a**b * a**c).to(a ** (b + c))
    yield rewrite(a * a**b).to(a ** (b + ONE))
    yield rewrite(a**b * a).to(a ** (b + ONE))
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
    # absolute values and roots, after Herbie, SymPy and SymbolicUtils.jl
    yield birewrite(abs(a * b)).to(abs(a) * abs(b))
    yield birewrite(abs(a / b)).to(abs(a) / abs(b))
    yield rewrite(abs(a ** Num(q))).to(abs(a) ** Num(q), integer(q))
    yield rewrite(abs(a) ** Num(q)).to(a ** Num(q), integer(q / BigRat(2, 1)))
    yield rewrite((a**TWO) ** b).to(abs(a) ** (TWO * b))
    yield rewrite(sqrt(a) * sqrt(b)).to(sqrt(a * b))
    yield rewrite(sqrt(a) / sqrt(b)).to(sqrt(a / b))
    yield rewrite(sqrt(a * b)).to(sqrt(abs(a)) * sqrt(abs(b)))
    yield rewrite(sqrt(a / b)).to(sqrt(abs(a)) / sqrt(abs(b)))
    yield rewrite(sqrt(a) ** b).to(a ** (b / TWO))  # sqrt(a) needs a >= 0
    yield rewrite(ONE / sqrt(a)).to(sqrt(a) / a)
    yield rewrite(cos(abs(a))).to(cos(a))
    yield rewrite(cosh(abs(a))).to(cosh(a))
    # square roots of literals: sqrt(12) == 2 * sqrt(3), sqrt(3/4) == sqrt(12) / 4
    for k in range(2, 11):
        k2 = BigRat(k * k, 1)
        yield rewrite(sqrt(Num(q))).to(
            rat(k) * sqrt(Num(q / k2)), integer(q / k2), q > k2
        )
    d = BigRat(q.denom, BigInt(1))
    yield rewrite(sqrt(Num(q))).to(
        sqrt(Num(q * d * d)) / Num(d), q > BigRat(0, 1), ne(q.denom).to(BigInt(1))
    )
    # literal factors out of roots and into squares: sqrt(4 * x) == 2 * sqrt(x)
    yield rewrite(sqrt(Num(p) * a)).to(sqrt(Num(p)) * sqrt(a), p > BigRat(0, 1))
    for k in range(2, 13):
        yield rewrite(rat(k * k) * a**TWO).to((rat(k) * a) ** TWO)


@ruleset
def exponential(a: Num, b: Num, p: BigRat) -> Rules:
    yield birewrite(exp(a) * exp(b)).to(exp(a + b))
    yield rewrite(exp(a) ** b).to(exp(a * b))
    yield birewrite(Num.e**a).to(exp(a))
    # only literal factors: exp(3 * x) -> exp(3) ** x would tie with exp(x) ** 3
    yield rewrite(exp(Num(p) * a)).to(exp(a) ** Num(p))
    yield rewrite(exp(a - b)).to(exp(a) / exp(b))
    # valid wherever the left-hand side is defined (a, b > 0)
    yield rewrite(log(a) + log(b)).to(log(a * b))
    yield rewrite(log(a) - log(b)).to(log(a / b))
    yield rewrite(-log(a)).to(log(ONE / a))
    yield rewrite(Num(p) * log(a)).to(log(a ** Num(p)))
    yield rewrite(exp(b * log(a))).to(a**b)
    # log(a * b) is defined for a * b > 0, where it is log|a| + log|b| (with
    # assumptions, ``abs`` goes away); likewise log(a ** b) == b * log|a|,
    # as a ** b is only defined for negative a if b is an integer
    yield rewrite(log(a * b)).to(log(abs(a)) + log(abs(b)))
    yield rewrite(log(a / b)).to(log(abs(a)) - log(abs(b)))
    yield rewrite(log(a**b)).to(b * log(abs(a)))
    # log(a ** p) is defined only for a ** p > 0: a > 0 unless p is even
    half_p = p / BigRat(2, 1)
    yield rewrite(log(a ** Num(p))).to(Num(p) * log(abs(a)), integer(half_p))
    yield rewrite(log(a ** Num(p))).to(Num(p) * log(a), ne(half_p.denom).to(BigInt(1)))


def _root(n: int, /) -> Num:
    return sqrt(rat(n))


# sin(p * pi) for rational p in [0, 1/2], from SymPy and Herbie
SINES = {
    (1, 12): (_root(6) - _root(2)) / rat(4),
    (1, 10): (_root(5) - ONE) / rat(4),
    (1, 8): sqrt(TWO - _root(2)) / TWO,
    (1, 6): HALF,
    (1, 4): _root(2) / TWO,
    (3, 10): (_root(5) + ONE) / rat(4),
    (1, 3): _root(3) / TWO,
    (3, 8): sqrt(TWO + _root(2)) / TWO,
    (5, 12): (_root(6) + _root(2)) / rat(4),
    (1, 2): ONE,
}
# tan(p * pi) for rational p in [0, 1/2)
TANGENTS = {
    (1, 12): TWO - _root(3),
    (1, 8): _root(2) - ONE,
    (1, 6): _root(3) / rat(3),
    (1, 4): ONE,
    (1, 3): _root(3),
    (3, 8): _root(2) + ONE,
    (5, 12): TWO + _root(3),
}


@ruleset
def angles(p: BigRat) -> Rules:
    """Special values at rational multiples of pi.

    Angles are first reduced into [0, pi/2], and cosines written as sines:
    coefficients are only ever combined, never split, so sin(2 * pi / 3)
    would otherwise never reach the tables.
    """
    zero, one, half = BigRat(0, 1), BigRat(1, 1), BigRat(1, 2)
    for angle in (Num(p) * Num.pi, Num.pi * Num(p)):
        yield rewrite(sin(angle)).to(-sin(Num(-p) * Num.pi), p < zero)
        yield rewrite(sin(angle)).to(-sin(Num(p - one) * Num.pi), p > one)
        yield rewrite(sin(angle)).to(sin(Num(one - p) * Num.pi), p > half, p <= one)
        yield rewrite(cos(angle)).to(cos(Num(-p) * Num.pi), p < zero)
        yield rewrite(cos(angle)).to(-cos(Num(p - one) * Num.pi), p > one)
        yield rewrite(cos(angle)).to(-cos(Num(one - p) * Num.pi), p > half, p <= one)
        yield rewrite(cos(angle)).to(sin(Num(half - p) * Num.pi), p >= zero, p <= half)
        yield rewrite(tan(angle)).to(-tan(Num(-p) * Num.pi), p < zero)
        yield rewrite(tan(angle)).to(tan(Num(p - one) * Num.pi), p > half)
    for (n, d), value in SINES.items():
        yield replace(sin(rat(n, d) * Num.pi)).to(value)
        yield replace(sin(Num.pi * rat(n, d))).to(value)
    for (n, d), value in TANGENTS.items():
        yield replace(tan(rat(n, d) * Num.pi)).to(value)
        yield replace(tan(Num.pi * rat(n, d))).to(value)


@ruleset
def inverse_trigonometric(a: Num) -> Rules:
    """From Herbie and SymPy."""
    yield rewrite(asin(-a)).to(-asin(a))
    yield rewrite(atan(-a)).to(-atan(a))
    yield rewrite(acos(-a)).to(Num.pi - acos(a))
    yield birewrite(acos(a)).to(Num.pi / TWO - asin(a))
    yield rewrite(cos(asin(a))).to(sqrt(ONE - a**TWO))
    yield rewrite(sin(acos(a))).to(sqrt(ONE - a**TWO))
    yield rewrite(tan(asin(a))).to(a / sqrt(ONE - a**TWO))
    yield rewrite(tan(acos(a))).to(sqrt(ONE - a**TWO) / a)
    yield rewrite(sin(atan(a))).to(a / sqrt(ONE + a**TWO))
    yield rewrite(cos(atan(a))).to(ONE / sqrt(ONE + a**TWO))
    # special values (the others follow from symmetry and acos <-> asin)
    for (n, d), value in SINES.items():
        yield replace(asin(value)).to(Num.pi * rat(n, d))
    for (n, d), value in TANGENTS.items():
        yield replace(atan(value)).to(Num.pi * rat(n, d))


@ruleset
def trigonometric(a: Num, b: Num) -> Rules:
    yield rewrite(sin(-a)).to(-sin(a))
    yield rewrite(cos(-a)).to(cos(a))
    yield birewrite(tan(a)).to(sin(a) / cos(a))
    yield birewrite(sin(a + b)).to(sin(a) * cos(b) + cos(a) * sin(b))
    yield birewrite(cos(a + b)).to(cos(a) * cos(b) - sin(a) * sin(b))
    yield birewrite(sin(a) ** TWO).to(ONE - cos(a) ** TWO)
    yield rewrite(ONE + tan(a) ** TWO).to(cos(a) ** rat(-2))
    # Pythagorean identities, after Metatheory.jl (with its signs corrected),
    # writing sec, csc and cot as powers of cos, sin and tan
    yield rewrite(sin(a) ** TWO - ONE).to(-(cos(a) ** TWO))
    yield rewrite(cos(a) ** TWO - ONE).to(-(sin(a) ** TWO))
    yield rewrite(tan(a) ** TWO - cos(a) ** rat(-2)).to(MINUS_ONE)
    yield rewrite(cos(a) ** rat(-2) - ONE).to(tan(a) ** TWO)
    yield rewrite(ONE + tan(a) ** rat(-2)).to(sin(a) ** rat(-2))
    yield rewrite(sin(a) ** rat(-2) - ONE).to(tan(a) ** rat(-2))
    yield rewrite(cos(a) ** rat(-2) - tan(a) ** TWO).to(ONE)
    yield rewrite(tan(a) ** rat(-2) - sin(a) ** rat(-2)).to(MINUS_ONE)
    # from Herbie, SymPy (fu.py) and SymbolicUtils.jl
    half_sum, half_diff = (a + b) / TWO, (a - b) / TWO
    # double angle, as shortcuts
    yield rewrite(sin(a) * cos(a)).to(sin(TWO * a) / TWO)
    yield rewrite(cos(a) ** TWO - sin(a) ** TWO).to(cos(TWO * a))
    yield rewrite(sin(a) ** TWO - cos(a) ** TWO).to(-cos(TWO * a))
    # half angle
    yield birewrite(sin(a) / (ONE + cos(a))).to(tan(a / TWO))
    yield rewrite((ONE - cos(a)) / sin(a)).to(tan(a / TWO))
    yield rewrite((sin(a) + sin(b)) / (cos(a) + cos(b))).to(tan(half_sum))
    yield rewrite((sin(a) - sin(b)) / (cos(a) + cos(b))).to(tan(half_diff))
    yield rewrite(sqrt((ONE - cos(a)) / TWO)).to(abs(sin(a / TWO)))
    yield rewrite(sqrt((ONE + cos(a)) / TWO)).to(abs(cos(a / TWO)))
    # tangent addition, only compressing: expanding tan(a + b) is undefined
    # where tan(a) is, even if tan(a + b) is not
    yield rewrite((tan(a) + tan(b)) / (ONE - tan(a) * tan(b))).to(tan(a + b))
    yield rewrite(TWO * tan(a) / (ONE - tan(a) ** TWO)).to(tan(TWO * a))
    yield rewrite(-ONE / tan(a)).to(tan(a + Num.pi / TWO))


@ruleset
def trigonometric_expansions(a: Num, b: Num) -> Rules:
    """Identities that expand an expression, from Herbie and SymPy (fu.py).

    They match almost every trigonometric expression, so the e-graph grows
    quickly: ``max_nodes`` keeps the search short.
    """
    half_sum, half_diff = (a + b) / TWO, (a - b) / TWO
    # triple angle
    three = rat(3)
    yield birewrite(sin(three * a)).to(three * sin(a) - rat(4) * sin(a) ** three)
    yield birewrite(cos(three * a)).to(rat(4) * cos(a) ** three - three * cos(a))
    # power reduction
    yield rewrite(sin(a) ** TWO).to(HALF - cos(TWO * a) / TWO)
    yield rewrite(cos(a) ** TWO).to(HALF + cos(TWO * a) / TWO)
    # product to sum
    yield rewrite(sin(a) * cos(b)).to((sin(a + b) + sin(a - b)) / TWO)
    yield rewrite(cos(a) * cos(b)).to((cos(a - b) + cos(a + b)) / TWO)
    yield rewrite(sin(a) * sin(b)).to((cos(a - b) - cos(a + b)) / TWO)
    # sum to product
    yield rewrite(sin(a) + sin(b)).to(TWO * sin(half_sum) * cos(half_diff))
    yield rewrite(sin(a) - sin(b)).to(TWO * cos(half_sum) * sin(half_diff))
    yield rewrite(cos(a) + cos(b)).to(TWO * cos(half_sum) * cos(half_diff))
    yield rewrite(cos(a) - cos(b)).to(-TWO * sin(half_sum) * sin(half_diff))
    yield rewrite(sin(a) + cos(a)).to(_root(2) * sin(a + Num.pi / rat(4)))
    yield rewrite(_root(3) * sin(a) + cos(a)).to(TWO * sin(a + Num.pi / rat(6)))
    yield rewrite(sin(a) + _root(3) * cos(a)).to(TWO * sin(a + Num.pi / rat(3)))


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
    # from Herbie, SymPy and SymbolicUtils.jl
    half_sum, half_diff = (a + b) / TWO, (a - b) / TWO
    yield rewrite(cosh(a) ** TWO - ONE).to(sinh(a) ** TWO)
    yield rewrite(sinh(a) ** TWO + ONE).to(cosh(a) ** TWO)
    yield rewrite(sinh(a) * cosh(a)).to(sinh(TWO * a) / TWO)
    yield rewrite(cosh(a) ** TWO + sinh(a) ** TWO).to(cosh(TWO * a))
    # the denominators are positive, so both directions are valid
    yield birewrite((tanh(a) + tanh(b)) / (ONE + tanh(a) * tanh(b))).to(tanh(a + b))
    yield birewrite(TWO * tanh(a) / (ONE + tanh(a) ** TWO)).to(tanh(TWO * a))
    yield birewrite(tanh(a)).to((exp(TWO * a) - ONE) / (exp(TWO * a) + ONE))
    # half angle
    yield birewrite(sinh(a) / (cosh(a) + ONE)).to(tanh(a / TWO))
    yield rewrite((cosh(a) - ONE) / sinh(a)).to(tanh(a / TWO))
    yield birewrite(sqrt((cosh(a) + ONE) / TWO)).to(cosh(a / TWO))
    yield rewrite(sinh(a / TWO)).to(sinh(a) / sqrt(TWO * (cosh(a) + ONE)))
    # sum to product
    yield rewrite(sinh(a) + sinh(b)).to(TWO * sinh(half_sum) * cosh(half_diff))
    yield rewrite(sinh(a) - sinh(b)).to(TWO * cosh(half_sum) * sinh(half_diff))
    yield rewrite(cosh(a) + cosh(b)).to(TWO * cosh(half_sum) * cosh(half_diff))
    yield rewrite(cosh(a) - cosh(b)).to(TWO * sinh(half_sum) * sinh(half_diff))


@ruleset
def matrices(a: Num, b: Num) -> Rules:
    """Matrix identities that hold for any shape where both sides are
    defined. ``det(a @ b) == det(a) * det(b)`` is left out: it needs square
    factors, and ``a @ b`` can be square when ``a`` and ``b`` are not.

    A literal 0 stands for a zero array of any shape.
    """
    yield rewrite(transpose(transpose(a))).to(a)
    yield rewrite(inv(inv(a))).to(a)
    yield rewrite(det(transpose(a))).to(det(a))
    yield rewrite(trace(transpose(a))).to(trace(a))
    yield replace(matmul(a, ZERO)).to(ZERO)
    yield replace(matmul(ZERO, a)).to(ZERO)
    yield replace(transpose(ZERO)).to(ZERO)
    yield replace(trace(ZERO)).to(ZERO)


@ruleset
def matrix_algebra(a: Num, b: Num) -> Rules:
    """Reshaping matrix expressions.

    Without shapes, some identities only hold in one direction, and others
    not at all: ``a @ b + a @ c -> a @ (b + c)`` fails when ``a @ b`` is a
    scalar broadcast over the vector ``a @ c``.
    """
    # not backwards: transpose(v) @ transpose(w) is a dot product of vectors,
    # but transpose(w @ v) the transpose of a scalar
    yield rewrite(transpose(matmul(a, b))).to(matmul(transpose(b), transpose(a)))
    yield birewrite(inv(transpose(a))).to(transpose(inv(a)))
    yield rewrite(trace(matmul(a, b))).to(trace(matmul(b, a)))
    yield rewrite(det(inv(a))).to(ONE / det(a))
    yield rewrite(transpose(a) + transpose(b)).to(transpose(a + b))
    yield rewrite(transpose(a) - transpose(b)).to(transpose(a - b))


SHRINK = (contradictions, fold, identities, matrices)
#: Costs for extraction, set once at the end: on an undefined input (two
#: literals merged) they would conflict.
COSTS = (literal_costs,)
#: Rules deriving facts, run to saturation once per iteration (not in every
#: shrinking pass: they only feed ``assumed`` and the contradiction check).
ANALYSIS = (signs,)
#: Growing rules that run on their own before the search.
PREPARE = (horner,)
GROW = (
    angles,
    inverse_trigonometric,
    coefficients,
    algebra,
    powers,
    exponential,
    trigonometric,
    hyperbolic,
    trigonometric_expansions,
    matrix_algebra,
    assumed,
)
