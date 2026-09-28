import pytest

pytest.importorskip("egglog")

from egglog import EGraph

from symbolcat import Symbol, complex, real
from symbolcat.lib import Constant, Function
from symbolcat.lib.std import IMPL
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Call, Literal, as_expr
from symbolcat.ops.egraph import (
    integer,
    negative,
    nonnegative,
    optimize,
    positive,
    simplify,
)
from symbolcat.ops.egraph._convert import Table, from_egglog, to_egglog

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
f = Function("f", arity=1)


@pytest.mark.parametrize(
    "expr",
    [
        x + 2 * y - x / 3,
        real.sin(x) ** -1 + real.sqrt(real.exp(x)),
        f(x + 1) * real.constants.pi,
        Literal(1) / 3 + 0.25,
        real.atan2(x, y)[0],
    ],
)
def test_roundtrip(expr):
    table = Table()
    egraph = EGraph()
    term = egraph.extract(egraph.let("root", to_egglog(expr, table)))
    assert from_egglog(term, table) == expr


def test_opaque_parts_keep_identity():
    """Functions and constants egglog does not know come back as the same objects."""
    g = Function("f", arity=1)  # same name as f, different function
    k = Constant("k")
    expr = simplify(f(x) + g(x) + k * 1)
    assert isinstance(expr, Call)
    calls, constant = expr.args
    assert isinstance(calls, Call)
    assert {id(arg.func) for arg in calls.args} == {id(f), id(g)}
    assert constant is k


def test_known_constants_come_back():
    pi, e = real.constants.pi, real.constants.e
    assert simplify(pi + e * 1) == pi + e


@pytest.mark.parametrize(
    "expr, expected",
    [
        (x * y + x * z, x * (y + z)),
        (real.sin(x) ** 2 + real.cos(x) ** 2 + 1, Literal(2)),
        (x / x + 0 * y, Literal(1)),
        ((x + y) - y, x),
        (real.exp(x) * real.exp(y), real.exp(x + y)),
        (Literal(1) / 2 + Literal(1) / 3, Literal(5) / 6),
        (Literal(2) ** 10 - 1000, Literal(24)),
        (real.log(real.exp(x)), x),
        (real.tan(x) * real.cos(x), real.sin(x)),
        (x - x + f(x * 1), f(x)),
        (x + 0.5 + 0.25, x + 0.75),
        (x + 0.5 + Literal(1) / 4, x + 0.75),
        ((x + 1) ** 2 - x**2 - 2 * x, Literal(1)),
        # constants
        (real.sin(real.constants.pi) + real.cos(real.constants.pi), Literal(-1)),
        (real.log(real.constants.e), Literal(1)),
        (real.constants.e**x / real.exp(x), Literal(1)),
        # powers and absolute value
        ((x * y) ** 2 / y**2, x**2),
        (real.sqrt(x**2), real.abs(x)),
        (real.abs(-x) - real.abs(x), Literal(0)),
        # exponential and logarithm
        (real.log(x**3) / real.log(x), Literal(3)),
        (real.exp(real.log(x)), x),
        # trigonometric
        (real.sin(x) * real.cos(y) + real.cos(x) * real.sin(y), real.sin(y + x)),
        (1 - real.cos(x) ** 2, real.sin(x) ** 2),
        (real.sin(real.asin(x)), x),
        # hyperbolic
        (real.cosh(x) + real.sinh(x), real.exp(x)),
        (real.cosh(x) ** 2 - real.sinh(x) ** 2, Literal(1)),
        (real.tanh(x) * real.cosh(x), real.sinh(x)),
        (2 * real.sinh(x) * real.cosh(x), real.sinh(2 * x)),
        # coefficients and floats
        (2 * x + 3 * x, 5 * x),
        (x + x + x, 3 * x),
        (2.0 * x * 0.5, x),
        # tie-breaking: conventional forms
        (x * x, x**2),
        (x / y / z, x / (y * z)),
        (real.exp(2 * real.log(x)), x**2),
        (real.cos(x) ** 2 - real.sin(x) ** 2, real.cos(2 * x)),
        (real.abs(real.exp(x)), real.exp(x)),
        (real.abs(x) * real.abs(x), x**2),
        # factoring and cancellation
        ((x**2 - 1) / (x - 1), x + 1),
        ((x**2 - y**2) / (x + y), x - y),
        ((x**2 + 2 * x + 1) / (x + 1), x + 1),
        ((x * y) / (x * z), y / z),
        # literals
        (real.sqrt(Literal(16)) + real.sqrt(Literal(1) / 4), Literal(9) / 2),
        (real.sqrt(4 * x**2), 2 * real.abs(x)),
        (
            real.asin(Literal(0)) + real.atan(Literal(0)) + real.acos(Literal(1)),
            Literal(0),
        ),
        # trigonometric and hyperbolic
        (real.sin(x + real.constants.pi), -real.sin(x)),
        (real.sin(x) / real.cos(x), real.tan(x)),
        (real.exp(x) - real.exp(-x), 2 * real.sinh(x)),
        (real.sin(x + 2 * real.constants.pi), real.sin(x)),
        (real.sin(real.constants.pi / 6) + real.cos(real.constants.pi / 3), Literal(1)),
        (real.cos(x) ** 4 - real.sin(x) ** 4, real.cos(2 * x)),
        (real.sin(2 * x) / real.sin(x), 2 * real.cos(x)),
        # more factoring
        ((x**3 - 1) / (x - 1), x**2 + x + 1),
        (x / (2 * x), Literal(1) / 2),
        (real.exp(real.log(x) + real.log(y)), x * y),
        # fractions are written as divisions
        (real.constants.pi * x / 6, real.constants.pi * x / 6),
        # like terms, and products in canonical order
        (x * y + 2 * x * y, 3 * x * y),
        ((x + y) ** 2 - x**2 - y**2, 2 * x * y),
        (y * (x * 3) * real.constants.pi, 3 * real.constants.pi * x * y),
        ((x + 1) * (x - 1) - x**2, Literal(-1)),
        # roots, logarithms and hyperbolic functions
        (real.exp(real.log(x) / 2), real.sqrt(x)),
        (real.log(1 / x), -real.log(x)),
        (real.cosh(x) ** 2 + real.sinh(x) ** 2, real.cosh(2 * x)),
        (4 * real.sin(x) ** 2 * real.cos(x) ** 2, real.sin(2 * x) ** 2),
    ],
)
def test_simplify(expr, expected):
    assert simplify(expr) == expected


def test_deep_identity_with_larger_limits():
    expr = real.sin(x) ** 4 + real.cos(x) ** 4 + real.sin(2 * x) ** 2 / 2
    assert simplify(expr, max_nodes=10_000, match_limit=10_000) == Literal(1)


@pytest.mark.parametrize(
    "expr, needs_assumption",
    [
        (real.sqrt(x**2) - x, Literal(0)),  # 0 only for nonnegative x
        (real.log(x**2), 2 * real.log(x)),  # undefined for negative x
        (real.abs(x) - x, Literal(0)),
        (real.sin(real.constants.pi * y), Literal(0)),  # needs integer y
    ],
)
def test_strict_by_default(expr, needs_assumption):
    assert simplify(expr) != needs_assumption


@pytest.mark.parametrize(
    "expr, expected",
    [
        # valid wherever the input is defined, so no assumption needed
        (real.log(x * y) - real.log(x), real.log(y)),  # needs x, y > 0
        (real.log(x) + real.log(y), real.log(x * y)),
        (real.log(x / y) + real.log(y), real.log(x)),
        (real.log(x**3) / real.log(x), Literal(3)),  # odd power: x > 0
        (real.sqrt(x**2), real.abs(x)),
    ],
)
def test_rules_valid_where_defined(expr, expected):
    assert simplify(expr) == expected


@pytest.mark.parametrize(
    "expr, assume, expected",
    [
        (real.log(x * y) - real.log(x), [positive(x)], real.log(y)),
        (real.log(x**2 * y) - 2 * real.log(x), [positive(x)], real.log(y)),
        (real.sqrt(x**2), [nonnegative(x)], x),
        (real.abs(x) + x, [negative(x)], Literal(0)),
        (real.abs(real.exp(y) * x), [positive(x)], x * real.exp(y)),
        (real.sin(real.constants.pi * y), [integer(y)], Literal(0)),
        (real.cos(real.constants.pi * y) ** 2, [integer(y)], Literal(1)),
        (Literal(0) ** (x + 1), [positive(x)], Literal(0)),
    ],
)
def test_assumptions(expr, assume, expected):
    assert simplify(expr, assume=assume) == expected


def test_contradictory_assumptions_return_input():
    expr = real.log(x * y)
    assert simplify(expr, assume=[positive(x), negative(x)]) == expr


a, b, c, d = Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d")


@pytest.mark.parametrize(
    "expr, expected",
    [
        # the tests of Metatheory.jl's CAS theory
        (2 * a + a + a, 4 * a),
        (a * c * b, a * b * c),
        (1 * x * 2, 2 * x),
        ((a**2 * b**2) ** 3, (a * b) ** 6),
        (a + b + 0 * c + d, a + b + d),
        (a + b + c * 0 + d - d, a + b),
        (a * x**y * b * x**d, a * b * x ** (y + d)),
        (real.cos(x) ** 2 + 1 + real.sin(x) ** 2, Literal(2)),
        # and rules from its theories
        (x / (x / y), y),
        (x * (y / x), y),
        (real.sin(x) ** 2 - 1, -(real.cos(x) ** 2)),
        (real.cos(x) ** 2 - 1, -(real.sin(x) ** 2)),
        (real.cos(x) ** -2 - 1, real.tan(x) ** 2),
        (1 + real.tan(x) ** -2, real.sin(x) ** -2),
    ],
)
def test_metatheory(expr, expected):
    assert simplify(expr) == expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        # from Herbie, SymPy, SymbolicUtils.jl and egglog's Herbie port
        (
            (real.sin(x) + real.sin(y)) / (real.cos(x) + real.cos(y)),
            real.tan((x + y) / 2),
        ),
        (3 * real.sin(x) - 4 * real.sin(x) ** 3, real.sin(3 * x)),
        (2 * real.sin(x) ** 2 - 1, -real.cos(2 * x)),
        (real.sin(x) / (1 + real.cos(x)), real.tan(x / 2)),
        (real.cosh(x) ** 2 - 1, real.sinh(x) ** 2),
        (real.sinh(x) ** 2 + 1, real.cosh(x) ** 2),
        ((real.exp(2 * x) - 1) / (real.exp(2 * x) + 1), real.tanh(x)),
        # special values and inverse functions
        (real.sin(2 * real.constants.pi / 3), real.sqrt(Literal(3)) / 2),
        (real.asin(Literal(1) / 2), real.constants.pi / 6),
        (real.asin(x) + real.acos(x), real.constants.pi / 2),
        (real.acos(-x) + real.acos(x), real.constants.pi),
        # roots, absolute values, logarithms and fractions
        (real.sqrt(Literal(3) / 4), real.sqrt(Literal(3)) / 2),
        (real.sqrt(Literal(18)) / 3, real.sqrt(Literal(2))),
        (real.sqrt(x) * real.sqrt(y), real.sqrt(x * y)),
        (real.abs(x) * real.abs(y), real.abs(x * y)),
        (real.log(x) - real.log(y), real.log(x / y)),
        (1 / x - 1 / (x + 1), 1 / (x + x**2)),
    ],
)
def test_rules_from_other_libraries(expr, expected):
    assert simplify(expr) == expected


def _horner(coefficients, constant_first=True):
    result = Literal(coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        if constant_first:
            result = coefficient + x * result
        else:
            result = x * result + coefficient
    return result


@pytest.mark.parametrize("degree", [3, 6, 9])
@pytest.mark.parametrize("ascending", [True, False])
def test_horner_is_reachable(degree, ascending):
    """The rules prove a polynomial equal to its Horner form."""
    from symbolcat.ops.egraph._saturate import search

    terms = [c * x**k for k, c in enumerate(range(1, degree + 2))]
    if not ascending:
        terms.reverse()
    poly = sum(terms[1:], start=terms[0])
    found = search(poly, iterations=8, max_nodes=20_000)
    assert found is not None
    coefficients = list(range(1, degree + 2))
    assert found.equivalent(_horner(coefficients)) or found.equivalent(
        _horner(coefficients, constant_first=False)
    )


def test_optimize_evaluates_polynomials_by_horner():
    """One multiplication per degree, and no powers."""
    from symbolcat.lib import operators as op

    result = optimize(sum(k * x**k for k in range(1, 6)))
    funcs = [getattr(n, "func", None) for n in _walk(result)]
    assert op.pow not in funcs
    assert funcs.count(op.mul) == 5


def test_max_nodes_stops_early():
    expr = (x + y + z) ** 4 - real.sin(x + y) ** 2
    result = simplify(expr, max_nodes=1)
    env = {x: 0.3, y: -1.7, z: 0.9}
    assert evaluate(result, env, lib=IMPL) == pytest.approx(
        evaluate(expr, env, lib=IMPL)
    )


@pytest.mark.parametrize(
    "expr",
    [Literal(1) / 0, x / 0 + 1],  # pyrefly: ignore[division-by-zero]
)
def test_division_by_zero_is_returned_unchanged(expr):
    """Rules assuming non-zero denominators prove two different numbers
    equal on these; that must neither hang nor produce a wrong result."""
    assert simplify(expr) == expr


def test_square_root_of_negative_terminates():
    """Rules assume positive bases, so the result follows sqrt(a) ** 2 == a;
    what matters is that the search terminates."""
    simplify((-real.sqrt(Literal(-1))) ** 2)


def test_optimize_shares_subexpressions():
    expr = optimize(real.sin(x) ** 2 + real.sin(x) * y)
    sins = [node for node in _walk(expr) if getattr(node, "func", None) is real.sin]
    assert len({id(node) for node in sins}) == 1


@pytest.mark.parametrize(
    "expr, multiplications",
    [
        (x**8, 3),  # x2 = x * x; x4 = x2 * x2; x8 = x4 * x4
        (x**3 * y**3, 3),  # xy = x * y; xy * (xy * xy)
    ],
)
def test_optimize_uses_repeated_squaring(expr, multiplications):
    from symbolcat.lib import operators as op

    result = optimize(expr)
    muls = {id(n) for n in _walk(result) if getattr(n, "func", None) is op.mul}
    assert len(muls) == multiplications


def test_optimize_avoids_expensive_operations():
    expr = optimize(x**3 + x**2)
    assert "**" not in str(expr)


@pytest.mark.parametrize(
    "expr",
    [
        x**3 + x**2 + x,
        (x + y) ** 2 - (x - y) ** 2,
        real.sin(x) ** 2 + real.sin(x) * real.cos(x) + x / x,
        real.sqrt(x**2) + real.abs(x * y),
        real.sin(x + y) - real.cos(x - y) + real.tan(-x),
        real.sinh(x) ** 2 + real.tanh(-y) * real.cosh(x + y),
        (x * y) ** 3 / (x**2 * y) + real.atan(x) - real.acos(x / 2),
    ],
)
@pytest.mark.parametrize("transform", [simplify, optimize])
@pytest.mark.parametrize("env", [{x: 0.3, y: -1.7}, {x: -0.8, y: 0.4}])
def test_equivalent_values(expr, transform, env):
    """Rules must not change values, including at negative points."""
    assert evaluate(transform(expr), env, lib=IMPL) == pytest.approx(
        evaluate(expr, env, lib=IMPL)
    )


def _walk(expr):
    stack, seen = [expr], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(getattr(node, "args", ()))


def _random_expr(rng, depth):
    """A random expression over x, y, small integers and pi."""
    unary = [
        real.sin,
        real.cos,
        real.tan,
        real.exp,
        real.log,
        real.sqrt,
        real.abs,
        real.sinh,
        real.cosh,
        real.tanh,
        real.asin,
        real.atan,
        lambda a: -a,
    ]
    binary = [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
        lambda a, b: a ** Literal(rng.choice([2, 3, -1])),
    ]
    if depth == 0 or rng.random() < 0.2:
        r = rng.random()
        if r < 0.6:
            return rng.choice([x, y])
        if r < 0.9:
            return Literal(rng.choice([1, 2, 3, -1, 0]))
        return real.constants.pi
    if rng.random() < 0.35:
        return rng.choice(unary)(_random_expr(rng, depth - 1))
    return rng.choice(binary)(
        _random_expr(rng, depth - 1), _random_expr(rng, depth - 1)
    )


def _value(expr, env):
    import math

    try:
        v = evaluate(expr, env, lib=IMPL)
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None
    return v if isinstance(v, float | int) and math.isfinite(v) else None


@pytest.mark.parametrize("seed", range(4))
def test_random_expressions_keep_their_value(seed):
    """Fuzzing: wherever the input is defined, the result is defined and has
    the same value (the result may be defined on a larger domain)."""
    import random

    rng = random.Random(seed)
    for _ in range(10):
        expr = _random_expr(rng, 4)
        result = simplify(expr)
        for _ in range(4):
            env = {x: rng.uniform(-3, 3), y: rng.uniform(-3, 3)}
            a, b = _value(expr, env), _value(result, env)
            if a is not None:
                assert b is not None, (expr, result, env)
                assert b == pytest.approx(a, rel=1e-4, abs=1e-9), (expr, result, env)


# complex expressions


I = complex.constants.I


@pytest.mark.parametrize(
    "expr, expected",
    [
        (I * I, -1),
        (complex.re(complex.exp(I * x)), real.cos(x)),
        (complex.abs(complex.exp(I * x)), 1),
        (complex.exp(I * x) * complex.exp(-I * x), 1),
        ((x + I * y) * (x - I * y), x**2 + y**2),
        (complex.re(complex.exp(I * x)) ** 2 + complex.im(complex.exp(I * x)) ** 2, 1),
        (complex.exp(I * (x + x)), complex.exp(I * (2 * x))),
    ],
)
def test_complex(expr, expected):
    assert simplify(expr) == as_expr(expected)


def test_complex_is_not_real():
    """I ** 2 is not nonnegative: abs(I * I) is 1, not I * I."""
    assert simplify(real.abs(I * I)) == Literal(1)
    assert evaluate(simplify(real.sqrt(-(I**2)))) == 1


def test_complex_is_hidden_from_real_rules():
    f = Function("f", arity=1)
    expr = f(I * x) * f(I * x)
    assert evaluate(simplify(expr), {x: 1.0}, lib={**IMPL, f: lambda v: v}) == -1


def test_complex_assumptions():
    assert simplify(complex.abs(complex.rect(x, y)), assume=[positive(x)]) == x
