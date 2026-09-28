import math
from typing import Any

import pytest

from symbolcat import D, Symbol, real
from symbolcat.ops import (
    change_variables,
    diff,
    dimensions,
    euler_lagrange,
    evaluate,
    expand_derivatives,
    jacobian,
    jvp,
    simplify_naive,
    substitute_variables,
    time_derivative,
    to_latex,
    to_str,
    variables,
)
from symbolcat.ops.algebras import Literal

x, y, v, k, t = Symbol("x"), Symbol("y"), Symbol("v"), Symbol("k"), Symbol("t")


def test_derivative_is_a_structural_node():
    assert D(x) == D(x)
    assert D(x) != D(y)
    assert {D(x): 1}[D(x)] == 1
    assert to_str(D(D(x)) - v) == "D(D(x)) - v"


@pytest.mark.parametrize(
    "expr, expected",
    [
        (D(x), "x'"),
        (D(D(x)), "x''"),
        (D(x) ** 2, "x'^{2}"),
        (D(x + y), r"\left(x + y\right)'"),
    ],
)
def test_latex(expr, expected):
    assert to_latex(expr) == expected


def test_variables():
    assert variables(D(D(x)) + v * D(x) + x) == (D(D(x)), v, D(x), x)
    assert variables(real.sin(k) * 2) == (k,)


def test_diff_treats_derivatives_as_independent():
    expr = D(x) ** 2 + x * D(x)
    assert diff(expr, x) == D(x)
    assert diff(expr, D(x)) == simplify_naive(2 * D(x) + x)
    assert diff(D(D(x)), D(x)) == Literal(0)


def test_diff_of_derivative_of_expression():
    with pytest.raises(NotImplementedError, match="time_derivative"):
        diff(D(x + y), x)


def test_diff_with_respect_to_derivative_of_expression():
    var: Any = D(2 * x)
    with pytest.raises(TypeError, match="with respect to a Symbol"):
        diff(x, var)


def test_jacobian_with_respect_to_derivatives():
    eqs = [D(x) - v, D(v) + k * x]
    assert jacobian(eqs, [D(x), D(v)]) == jacobian([x, v], [x, v])


def test_jvp():
    dx = Symbol("dx")
    assert jvp(x * y, {x: dx}) == dx * y
    assert jvp(real.sin(x), {x: dx}) == real.cos(x) * dx


def test_time_derivative_parameters_are_constant():
    assert time_derivative(k * x, [x]) == k * D(x)
    assert time_derivative(k * x, []) == Literal(0)


def test_time_derivative_explicit_time():
    assert time_derivative(real.sin(t), [x], t) == real.cos(t)
    assert time_derivative(real.sin(t), [x]) == Literal(0)


def test_time_derivative_of_derivatives():
    assert time_derivative(D(x) * x, [x]) == D(D(x)) * x + D(x) * D(x)


def test_pendulum_constraint():
    """The hidden constraints of the Cartesian pendulum."""
    L = Symbol("L")
    constraint = x**2 + y**2 - L**2
    velocity = time_derivative(constraint, [x, y])
    assert velocity == 2 * x * D(x) + 2 * y * D(y)
    acceleration = time_derivative(velocity, [x, y])
    assert diff(acceleration, D(D(x))) == 2 * x
    assert diff(acceleration, D(x)) == simplify_naive(2 * D(x) + 2 * D(x))


def test_expand_derivatives():
    assert expand_derivatives(D(k * x), [x]) == k * D(x)
    assert expand_derivatives(D(D(x * x)), [x]) == time_derivative(
        time_derivative(x * x, [x]), [x]
    )
    assert expand_derivatives(D(x) + D(D(x)), [x]) == D(x) + D(D(x))


def test_diff_after_expand_derivatives():
    assert diff(expand_derivatives(D(x + y), [x, y]), D(x)) == Literal(1)


def test_change_variables_polar():
    r, theta = Symbol("r"), Symbol("theta")
    mapping = {x: r * real.cos(theta), y: r * real.sin(theta)}
    # the squared speed, in polar coordinates, is D(r)² + r² D(theta)²
    (speed,) = change_variables([D(x) ** 2 + D(y) ** 2], mapping, [r, theta])
    point = {r: 1.3, theta: 0.4, D(r): 0.2, D(theta): -0.7}
    expected = 0.2**2 + 1.3**2 * 0.7**2
    assert evaluate(substitute_variables(speed, point)) == pytest.approx(expected)


def test_euler_lagrange_pendulum():
    """L = m l² D(θ)² / 2 + m g l cos θ gives m l² D(D(θ)) + m g l sin θ."""
    theta, m, g, l = Symbol("theta"), Symbol("m"), Symbol("g"), Symbol("l")
    lagrangian = m * l**2 * D(theta) ** 2 / 2 + m * g * l * real.cos(theta)
    (eq,) = euler_lagrange(lagrangian, [theta])
    point = {theta: 0.3, D(theta): 0.5, D(D(theta)): -1.1, m: 2.0, g: 9.8, l: 1.5}
    expected = 2.0 * 1.5**2 * -1.1 + 2.0 * 9.8 * 1.5 * math.sin(0.3)
    assert evaluate(substitute_variables(eq, point)) == pytest.approx(expected)


def test_euler_lagrange_two_coordinates():
    m = Symbol("m")
    lagrangian = m * (D(x) ** 2 + D(y) ** 2) / 2 - k * x * y
    eqs = euler_lagrange(lagrangian, [x, y])
    point = {D(D(x)): 0.3, D(D(y)): -0.2, x: 1.0, y: 2.0, m: 3.0, k: 0.5}
    values = [evaluate(substitute_variables(eq, point)) for eq in eqs]
    assert values == pytest.approx([3.0 * 0.3 + 0.5 * 2.0, 3.0 * -0.2 + 0.5 * 1.0])


def test_dimensions_of_derivatives():
    units = {x: {"m": 1}, v: {"m": 1, "s": -1}}
    assert dimensions(D(x) - v, units, time={"s": 1}) == {"m": 1, "s": -1}
    with pytest.raises(ValueError, match="incompatible"):
        dimensions(D(D(x)) - v, units, time={"s": 1})
    with pytest.raises(ValueError, match="dimension of `time`"):
        dimensions(D(x), units)
