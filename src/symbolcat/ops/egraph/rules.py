"""
symbolcat.ops.egraph.rules
~~~~~~~~~~~~~~~~~~~~~~~~~~

Rewrite rules over ``Num`` terms.

Rules only add equalities to the e-graph; nothing is destroyed, so the
order in which they fire does not matter. Choosing the best equivalent
expression is left to extraction (see ``cost``).

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

from egglog import BigInt, BigRat, birewrite, eq, f64, ne, rewrite, ruleset, vars_

from .term import (
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

a, b, c = vars_("a b c", Num)
p, q = vars_("p q", BigRat)
f, g = vars_("f g", f64)


def rat(numerator: int, denominator: int = 1) -> Num:
    return Num(BigRat(numerator, denominator))


ZERO, ONE, TWO, MINUS_ONE, HALF = rat(0), rat(1), rat(2), rat(-1), rat(1, 2)
neg_a = -a
is_integer = eq(q.denom).to(BigInt(1))


def small(r: BigRat, bound: int = 16) -> tuple:
    """Conditions for a literal exponent ``r`` to be within ``bound``."""
    return r <= BigRat(bound, 1), r >= BigRat(-bound, 1)


# Arithmetic on literals: exact on rationals, IEEE on floats, and floats
# absorb rationals.
fold = ruleset(
    rewrite(Num(p) + Num(q)).to(Num(p + q)),
    rewrite(Num(p) - Num(q)).to(Num(p - q)),
    rewrite(Num(p) * Num(q)).to(Num(p * q)),
    rewrite(Num(p) / Num(q)).to(Num(p / q), ne(q).to(BigRat(0, 1))),
    rewrite(-Num(p)).to(Num(-p)),
    # small integer powers only; 0 ** negative is left alone
    rewrite(Num(p) ** Num(q)).to(
        Num(p**q),
        is_integer,
        ne(p).to(BigRat(0, 1)),
        *small(q, 64),
    ),
    rewrite(Num.float(f) + Num.float(g)).to(Num.float(f + g)),
    rewrite(Num.float(f) - Num.float(g)).to(Num.float(f - g)),
    rewrite(Num.float(f) * Num.float(g)).to(Num.float(f * g)),
    rewrite(Num.float(f) / Num.float(g)).to(Num.float(f / g), ne(g).to(f64(0.0))),
    rewrite(-Num.float(f)).to(Num.float(-f)),
    rewrite(Num(p) + Num.float(f)).to(Num.float(p.to_f64() + f)),
    rewrite(Num.float(f) + Num(p)).to(Num.float(f + p.to_f64())),
    rewrite(Num(p) * Num.float(f)).to(Num.float(p.to_f64() * f)),
    rewrite(Num.float(f) * Num(p)).to(Num.float(f * p.to_f64())),
    name="fold",
)

# Identities that remove operations.
identities = ruleset(
    # arithmetic
    rewrite(a + ZERO).to(a),
    rewrite(a - ZERO).to(a),
    rewrite(ZERO - a).to(-a),
    rewrite(a * ONE).to(a),
    rewrite(a * ZERO).to(ZERO),
    rewrite(a * MINUS_ONE).to(-a),
    rewrite(a / ONE).to(a),
    rewrite(ZERO / a).to(ZERO),
    rewrite(a - a).to(ZERO),
    rewrite(a / a).to(ONE),
    rewrite(-neg_a).to(a),
    rewrite(a**ZERO).to(ONE),
    rewrite(a**ONE).to(a),
    rewrite(ONE**a).to(ONE),
    rewrite(sqrt(ZERO)).to(ZERO),
    rewrite(sqrt(a) ** TWO).to(a),
    # absolute value
    rewrite(abs(-a)).to(abs(a)),
    rewrite(abs(abs(a))).to(abs(a)),
    rewrite(abs(a) ** TWO).to(a**TWO),
    rewrite(sqrt(a**TWO)).to(abs(a)),
    # exponential and logarithm
    rewrite(exp(ZERO)).to(ONE),
    rewrite(exp(ONE)).to(Num.e),
    rewrite(log(ONE)).to(ZERO),
    rewrite(log(Num.e)).to(ONE),
    rewrite(log(exp(a))).to(a),
    rewrite(exp(log(a))).to(a),
    # trigonometric
    rewrite(sin(ZERO)).to(ZERO),
    rewrite(cos(ZERO)).to(ONE),
    rewrite(tan(ZERO)).to(ZERO),
    rewrite(sin(Num.pi)).to(ZERO),
    rewrite(cos(Num.pi)).to(MINUS_ONE),
    rewrite(sin(Num.pi / TWO)).to(ONE),
    rewrite(cos(Num.pi / TWO)).to(ZERO),
    rewrite(sin(a) ** TWO + cos(a) ** TWO).to(ONE),
    rewrite(sin(asin(a))).to(a),
    rewrite(cos(acos(a))).to(a),
    rewrite(tan(atan(a))).to(a),
    # hyperbolic
    rewrite(sinh(ZERO)).to(ZERO),
    rewrite(cosh(ZERO)).to(ONE),
    rewrite(tanh(ZERO)).to(ZERO),
    rewrite(cosh(a) ** TWO - sinh(a) ** TWO).to(ONE),
    rewrite(cosh(a) + sinh(a)).to(exp(a)),
    rewrite(cosh(a) - sinh(a)).to(exp(-a)),
    name="identities",
)

# Commutative ring, and the other operations in terms of + and *.
algebra = ruleset(
    rewrite(a + b).to(b + a),
    rewrite(a * b).to(b * a),
    birewrite(a + (b + c)).to((a + b) + c),
    birewrite(a * (b * c)).to((a * b) * c),
    birewrite(a * (b + c)).to(a * b + a * c),
    rewrite(a * b + a).to(a * (b + ONE)),
    birewrite(a + a).to(TWO * a),
    birewrite(a - b).to(a + -b),
    birewrite(-a).to(MINUS_ONE * a),
    birewrite(a / b).to(a * b**MINUS_ONE),
    name="algebra",
)

powers = ruleset(
    birewrite(sqrt(a)).to(a**HALF),
    birewrite(a * a).to(a**TWO),
    # Exponents are only combined, never split, and literal exponents stay
    # small: otherwise x ** 2 * x ** 2 -> x ** 4 -> ... grows without bound.
    rewrite(a**b * a**c).to(a ** (b + c)),
    rewrite(a * a**b).to(a ** (b + ONE)),
    rewrite((a * b) ** Num(q)).to(a ** Num(q) * b ** Num(q), is_integer, *small(q)),
    rewrite((a ** Num(p)) ** Num(q)).to(a ** Num(p * q), is_integer, *small(p * q)),
    # small positive integer powers as products: x ** 3 == x * x ** 2
    rewrite(a ** Num(p)).to(
        a * a ** Num(p - BigRat(1, 1)),
        eq(p.denom).to(BigInt(1)),
        p > BigRat(1, 1),
        p <= BigRat(16, 1),
    ),
    rewrite(abs(a * b)).to(abs(a) * abs(b)),
    name="powers",
)

exponential = ruleset(
    birewrite(exp(a) * exp(b)).to(exp(a + b)),
    rewrite(exp(a) ** b).to(exp(a * b)),
    birewrite(Num.e**a).to(exp(a)),
    birewrite(log(a * b)).to(log(a) + log(b)),
    birewrite(log(a**b)).to(b * log(a)),
    name="exponential",
)

trigonometric = ruleset(
    rewrite(sin(-a)).to(-sin(a)),
    rewrite(cos(-a)).to(cos(a)),
    birewrite(tan(a)).to(sin(a) / cos(a)),
    birewrite(sin(a + b)).to(sin(a) * cos(b) + cos(a) * sin(b)),
    birewrite(cos(a + b)).to(cos(a) * cos(b) - sin(a) * sin(b)),
    birewrite(sin(a) ** TWO).to(ONE - cos(a) ** TWO),
    name="trigonometric",
)

hyperbolic = ruleset(
    rewrite(sinh(-a)).to(-sinh(a)),
    rewrite(cosh(-a)).to(cosh(a)),
    birewrite(tanh(a)).to(sinh(a) / cosh(a)),
    birewrite(sinh(a)).to((exp(a) - exp(-a)) / TWO),
    birewrite(cosh(a)).to((exp(a) + exp(-a)) / TWO),
    name="hyperbolic",
)

SHRINK = (fold, identities)
GROW = (algebra, powers, exponential, trigonometric, hyperbolic)
