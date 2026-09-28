from typing import Any

import pytest

from symbolcat import D, Symbol, real
from symbolcat.ops import (
    diff,
    jacobian,
    jvp,
    simplify_naive,
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
    dx, dy = Symbol("dx"), Symbol("dy")
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
