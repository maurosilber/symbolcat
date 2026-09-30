import math
import random
from typing import Any

import pytest

from symbolcat import D, Symbol, real
from symbolcat.linalg import to_list
from symbolcat.models.dynamics import (
    lie_derivative,
    linearize,
    observability_matrix,
    sensitivities,
    steady_state,
)
from symbolcat.models.ode import ODE, taylor_solution, to_first_order
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Literal
from symbolcat.ops.transform import substitute_variables

x, v, k, g, u, t = (Symbol(n) for n in ("x", "v", "k", "g", "u", "t"))


def value(expr: Any, point: dict[Any, float]) -> float:
    return evaluate(substitute_variables(expr, point))


def random_point(*symbols: Any, seed: int = 0) -> dict[Any, float]:
    rng = random.Random(seed)
    return {s: rng.uniform(0.5, 1.5) for s in symbols}


oscillator = ODE({x: v, v: -k * x})


def test_ode():
    assert oscillator.states == (x, v)
    assert oscillator.parameters == (k,)
    assert oscillator.equations == (D(x) - v, D(v) - -k * x)


def test_ode_parameters_exclude_time_and_bases():
    ode = ODE({x: D(x), D(x): -k * x + real.sin(t)}, t)
    assert ode.parameters == (k,)


def test_ode_checks_states():
    with pytest.raises(TypeError, match="symbols or D"):
        ODE({2 * x: v})


def test_to_first_order():
    ode = to_first_order(ODE({x: D(x), D(x): D(D(x)), D(D(x)): -x}))
    _, x_t, x_tt = ode.states
    assert (str(x_t), str(x_tt)) == ("x_t", "x_tt")
    assert ode.rhs[x] == x_t
    assert ode.rhs[x_t] == x_tt


def test_to_first_order_name_collision():
    with pytest.raises(ValueError, match="x_t is already used"):
        to_first_order(ODE({x: D(x), D(x): Symbol("x_t")}))


def test_lie_derivative_conserved_energy():
    energy = v**2 + k * x**2
    point = random_point(x, v, k)
    assert value(lie_derivative(energy, oscillator), point) == pytest.approx(0)


def test_lie_derivative_higher_order():
    # the second derivative of x is the acceleration
    point = random_point(x, v, k)
    assert value(lie_derivative(x, oscillator, 2), point) == pytest.approx(
        -point[k] * point[x]
    )


def test_lie_derivative_explicit_time():
    ode = ODE({x: real.sin(t)}, t)
    assert value(lie_derivative(x * t, ode), {x: 2.0, t: 1.0}) == pytest.approx(
        math.sin(1.0) + 2.0
    )


def test_steady_state():
    system = steady_state(ODE({x: k - x, v: x - v}))
    assert system.unknowns == (x, v)
    assert system.equations == (k - x, x - v)


def test_linearize_pendulum():
    ode = ODE({x: v, v: -g * real.sin(x) + u})
    lin = linearize(ode, [u], [x], at={x: 0})
    assert to_list(lin.A) == [[Literal(0), Literal(1)], [-g, Literal(0)]]
    assert lin.B is not None and to_list(lin.B) == [[Literal(0)], [Literal(1)]]
    assert lin.C is not None and to_list(lin.C) == [[Literal(1), Literal(0)]]
    assert lin.D is not None and to_list(lin.D) == [[Literal(0)]]


def test_linearize_without_inputs():
    lin = linearize(oscillator)
    assert lin.B is None and lin.C is None and lin.D is None


def test_observability():
    rows = to_list(observability_matrix(oscillator, [x]))
    point = random_point(x, v, k)
    assert [[value(e, point) for e in row] for row in rows] == [[1, 0], [0, 1]]
    # the velocity alone determines the position (through D(v) = -k x) ...
    rows = to_list(observability_matrix(oscillator, [v]))
    assert [[value(e, point) for e in row] for row in rows] == [
        [0, 1],
        [-point[k], 0],
    ]
    # ... but not in a free particle
    rows = to_list(observability_matrix(ODE({x: v, v: Literal(0)}), [v]))
    assert [[value(e, point) for e in row] for row in rows] == [[0, 1], [0, 0]]


def test_sensitivities():
    """For D(x) = -k x, x = x0 exp(-k t), so dx/dk = -t x."""
    ode = sensitivities(ODE({x: -k * x}), [k])
    _, s = ode.states
    assert ode.states == (x, s) and str(s) == "dx_dk"
    point = random_point(x, k, s)
    assert value(ode.rhs[s], point) == pytest.approx(-point[k] * point[s] - point[x])


def test_sensitivities_need_symbol_states():
    with pytest.raises(TypeError, match="to_first_order"):
        sensitivities(ODE({x: D(x), D(x): -k * x}), [k])


def test_taylor_solution_exponential():
    (solution,) = taylor_solution(ODE({x: k * x}), 4, t).values()
    point = {x: 2.0, k: 0.3, t: 0.1}
    expected = 2.0 * sum((0.3 * 0.1) ** n / math.factorial(n) for n in range(5))
    assert value(solution, point) == pytest.approx(expected)


def test_taylor_solution_oscillator_around_t0():
    ode = ODE({x: v, v: -x})
    solution = taylor_solution(ode, 3, t, 1)
    point = {x: 1.0, v: 0.0, t: 1.5}
    h = 0.5
    assert value(solution[x], point) == pytest.approx(1 - h**2 / 2)
    assert value(solution[v], point) == pytest.approx(-h + h**3 / 6)


def test_taylor_solution_explicit_time():
    # D(x) = t, x(t0) = x0: x = x0 + (t² - t0²) / 2
    solution = taylor_solution(ODE({x: t}, t), 3, t0=1)
    assert value(solution[x], {x: 0.0, t: 3.0}) == pytest.approx(4.0)


def test_taylor_solution_needs_time():
    with pytest.raises(ValueError, match="time symbol"):
        taylor_solution(oscillator, 2)
