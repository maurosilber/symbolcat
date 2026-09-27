"""
symbolcat.ops.egraph.rules
~~~~~~~~~~~~~~~~~~~~~~~~~~

Rewrite rules over ``Num`` terms.

Rules only add equalities to the e-graph; nothing is destroyed, so the
order in which they fire does not matter. Choosing the best equivalent
expression is left to extraction (see ``cost``).

As in ``symbolcat.ops.simplify``, a few rules assume a non-zero
denominator (``a / a -> 1``).
"""

from __future__ import annotations

from egglog import BigInt, BigRat, birewrite, eq, f64, ne, rewrite, ruleset, vars_

from .term import Num, cos, exp, log, sin, sqrt, tan

a, b, c = vars_("a b c", Num)
p, q = vars_("p q", BigRat)
f, g = vars_("f g", f64)


def rat(numerator: int, denominator: int = 1) -> Num:
    return Num(BigRat(numerator, denominator))


ZERO, ONE, TWO, MINUS_ONE, HALF = rat(0), rat(1), rat(2), rat(-1), rat(1, 2)
neg_a = -a

# Exact arithmetic on rational literals. Terminating: always shrinks.
fold = ruleset(
    rewrite(Num(p) + Num(q)).to(Num(p + q)),
    rewrite(Num(p) - Num(q)).to(Num(p - q)),
    rewrite(Num(p) * Num(q)).to(Num(p * q)),
    rewrite(Num(p) / Num(q)).to(Num(p / q), ne(q).to(BigRat(0, 1))),
    rewrite(-Num(p)).to(Num(-p)),
    # integer powers only; 0 ** negative is left alone
    rewrite(Num(p) ** Num(q)).to(
        Num(p**q),
        eq(q.denom).to(BigInt(1)),
        ne(p).to(BigRat(0, 1)),
    ),
    rewrite(Num.float(f) + Num.float(g)).to(Num.float(f + g)),
    rewrite(Num.float(f) * Num.float(g)).to(Num.float(f * g)),
    name="fold",
)

# Identities that remove an operation. Terminating: always shrinks.
identities = ruleset(
    rewrite(a + ZERO).to(a),
    rewrite(a * ONE).to(a),
    rewrite(a * ZERO).to(ZERO),
    rewrite(a - a).to(ZERO),
    rewrite(a / a).to(ONE),
    rewrite(a**ZERO).to(ONE),
    rewrite(a**ONE).to(a),
    rewrite(-neg_a).to(a),
    rewrite(exp(ZERO)).to(ONE),
    rewrite(log(ONE)).to(ZERO),
    rewrite(sin(ZERO)).to(ZERO),
    rewrite(cos(ZERO)).to(ONE),
    rewrite(log(exp(a))).to(a),
    rewrite(sqrt(a) ** TWO).to(a),
    rewrite(sin(a) ** TWO + cos(a) ** TWO).to(ONE),
    name="identities",
)

# Equalities that reshape an expression. Non-terminating: the e-graph can
# grow without bound, so they run for a limited number of iterations.
algebra = ruleset(
    # commutative ring
    rewrite(a + b).to(b + a),
    rewrite(a * b).to(b * a),
    birewrite(a + (b + c)).to((a + b) + c),
    birewrite(a * (b * c)).to((a * b) * c),
    birewrite(a * (b + c)).to(a * b + a * c),
    # other operations in terms of + and *
    birewrite(a - b).to(a + -b),
    birewrite(-a).to(MINUS_ONE * a),
    birewrite(a / b).to(a * b**MINUS_ONE),
    birewrite(sqrt(a)).to(a**HALF),
    # powers
    birewrite(a * a).to(a**TWO),
    birewrite(a**b * a**c).to(a ** (b + c)),
    rewrite(a * a**b).to(a ** (b + ONE)),
    # small positive integer powers as products: x ** 3 == x * x ** 2
    rewrite(a ** Num(p)).to(
        a * a ** Num(p - BigRat(1, 1)),
        eq(p.denom).to(BigInt(1)),
        p > BigRat(1, 1),
        p <= BigRat(16, 1),
    ),
    rewrite(a + a).to(TWO * a),
    # exponentials and trigonometry
    birewrite(exp(a) * exp(b)).to(exp(a + b)),
    rewrite(sin(-a)).to(-sin(a)),
    rewrite(cos(-a)).to(cos(a)),
    birewrite(tan(a)).to(sin(a) / cos(a)),
    name="algebra",
)
