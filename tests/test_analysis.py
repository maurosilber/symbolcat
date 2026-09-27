from fractions import Fraction

import pytest

from symbolcat import Symbol, real
from symbolcat.linalg import matrix, vector
from symbolcat.ops import (
    coefficients,
    dimensions,
    evaluate,
    gradient,
    hessian,
    jacobian,
    series,
    solve_dependencies,
)
from symbolcat.ops.algebras import Literal

x, y, a = Symbol("x"), Symbol("y"), Symbol("a")


# coefficients


@pytest.mark.parametrize(
    "expr, expected",
    [
        (3 * x**2 - x + 5, {0: "5", 1: "-1", 2: "3"}),
        ((x + a) ** 2, {0: "a * a", 1: "a + a", 2: "1"}),
        (x * (x - 1) / 2, {1: "-1 / 2", 2: "1 / 2"}),
        (real.sin(a) * x + real.exp(a), {0: "exp(a)", 1: "sin(a)"}),
        (Literal(7), {0: "7"}),
        (x - x, {}),
    ],
)
def test_coefficients(expr, expected):
    assert {k: str(c) for k, c in coefficients(expr, x).items()} == expected


@pytest.mark.parametrize("expr", [1 / x, real.sin(x), x**a, x**-1, x ** Literal(1 / 2)])
def test_coefficients_of_non_polynomials(expr):
    with pytest.raises(ValueError, match="not a polynomial"):
        coefficients(expr, x)


# series


@pytest.mark.parametrize(
    "expr, kwargs, expected",
    [
        (real.sin(x), {}, "x - x ** 3 / 6 + x ** 5 / 120"),
        (real.cos(x), {}, "1 - x ** 2 / 2 + x ** 4 / 24"),
        (real.exp(x), {"order": 4}, "1 + x + x ** 2 / 2 + x ** 3 / 6"),
        (real.log(1 + x), {"order": 4}, "x - x ** 2 / 2 + x ** 3 / 3"),
        (1 / (1 - x), {"order": 4}, "1 + x + x ** 2 + x ** 3"),
        (real.sqrt(1 + x), {"order": 3}, "1 + x / 2 - x ** 2 / 8"),
        (real.sin(x) / x, {"order": 5}, "1 - x ** 2 / 6 + x ** 4 / 120"),
        (x**3 + 2 * x, {"at": 1}, "3 + 5 * (x - 1) + 3 * (x - 1) ** 2 + (x - 1) ** 3"),
        (x**x, {"at": 1, "order": 3}, "1 + (x - 1) + (x - 1) ** 2"),
    ],
)
def test_series(expr, kwargs, expected):
    assert str(series(expr, x, **kwargs)) == expected


@pytest.mark.parametrize(
    "expr, at",
    [
        (real.tan(x) + real.atan(x), 0),
        (real.exp(real.sin(x)), 0),
        (real.cosh(x) / (1 + x**2), 0.5),
        (real.log(x) * x**2, 2),
    ],
)
def test_series_approximates_the_function(expr, at):
    approx = series(expr, x, at=at, order=8)
    point = {x: at + 0.05}
    assert evaluate(approx, point) == pytest.approx(evaluate(expr, point), rel=1e-9)


def test_series_with_other_symbols():
    assert str(series(real.exp(a * x), x, order=3)) == "1 + a * x + a * a * x ** 2 / 2"


def test_series_of_a_pole():
    with pytest.raises(ValueError, match="pole"):
        series(1 / x, x)


# gradient, jacobian, hessian


def test_gradient_default_variables():
    assert gradient(x**2 * y) == vector([2 * x * y, x**2])


def test_gradient_explicit_variables():
    assert gradient(x * y, (y,)) == vector([x])


def test_jacobian():
    j = jacobian([x * y, real.sin(x)], (x, y))
    assert j == matrix([[y, x], [real.cos(x), Literal(0)]])


def test_jacobian_of_a_vector():
    assert jacobian(vector([x * y, y]), (x, y)) == matrix([[y, x], [0, 1]])


def test_jacobian_of_a_matrix():
    with pytest.raises(TypeError, match="sequence or a vector"):
        jacobian(matrix([[x]]))


def test_hessian_is_symmetric():
    h = hessian(x**3 * y + real.exp(x * y))
    (_, h01), (h10, _) = evaluate(h, {x: 0.3, y: -1.2})
    assert h01 == pytest.approx(h10)


# dimensions


t, v, g, m = Symbol("t"), Symbol("v"), Symbol("g"), Symbol("m")
UNITS = {
    x: {"m": 1},
    t: {"s": 1},
    v: {"m": 1, "s": -1},
    g: {"m": 1, "s": -2},
    m: {"kg": 1},
}


@pytest.mark.parametrize(
    "expr, expected",
    [
        (x + v * t - g * t**2 / 2, {"m": 1}),
        (m * g * x, {"kg": 1, "m": 2, "s": -2}),
        (real.sqrt(2 * x / g), {"s": 1}),
        (real.exp(-t * v / x), {}),
        (x ** Literal(1) / 2, {"m": 1}),
        (x ** (Literal(1) / 2), {"m": Fraction(1, 2)}),
        (real.abs(v) < v, {}),
        (real.atan2(x, v * t), {}),
    ],
)
def test_dimensions(expr, expected):
    assert dimensions(expr, UNITS) == expected


@pytest.mark.parametrize(
    "expr, message",
    [
        (x + t, "incompatible dimensions: m, s"),
        (real.log(x), "dimensionless argument"),
        (x**t, "Exponent t has dimension s"),
        (x ** Symbol("n"), "No units given for 'n'"),
        (x < t, "incompatible"),
    ],
)
def test_dimension_errors(expr, message):
    units = UNITS if "n" in message else {**UNITS, Symbol("n"): {}}
    with pytest.raises(ValueError, match=message):
        dimensions(expr, units)


def test_dimensions_with_string_keys():
    assert dimensions(x / t, {"x": {"m": 1}, "t": {"s": 1}}) == {"m": 1, "s": -1}


def test_dimensionless_symbolic_exponent():
    n = Symbol("n")
    assert dimensions(Symbol("k") ** n, {Symbol("k"): {}, n: {}}) == {}


def test_dimensional_base_needs_numeric_exponent():
    n = Symbol("n")
    with pytest.raises(ValueError, match="must be a number"):
        dimensions(x**n, {**UNITS, n: {}})


# solve_dependencies


def test_solve_dependencies_layers():
    z, w = Symbol("z"), Symbol("w")
    system = {z: x + y, w: 2 * a, x: a**2, y: x + w}
    assert solve_dependencies(system) == [(w, x), (y,), (z,)]


def test_solve_dependencies_inputs():
    assert solve_dependencies({x: x, y: 2 * x}) == [(x,), (y,)]


def test_solve_dependencies_cycles():
    with pytest.raises(ValueError, match="Cyclic dependencies among x, y"):
        solve_dependencies({x: y + 1, y: 2 * x, a: Literal(1)})
    with pytest.raises(ValueError, match="Cyclic"):
        solve_dependencies({x: x + 1})
