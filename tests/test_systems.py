import math
from typing import Any

import pytest

from symbolcat import D, Symbol, delay, partial, real
from symbolcat.lib import operators as op
from symbolcat.linalg import to_list
from symbolcat.ops import (
    ODE,
    SDE,
    ContinuousEvent,
    DiscreteEvent,
    DiscreteSystem,
    IndexOneDAE,
    characteristic_matrix,
    delays,
    diff,
    evaluate,
    fixed_points,
    grid_symbol,
    hessian_sparsity,
    identifiability,
    ito_lemma,
    ito_to_stratonovich,
    iterate,
    jacobian_sparsity,
    linearize,
    method_of_lines,
    sensitivities,
    solve,
    stratonovich_to_ito,
    structural_simplify,
    substitute_variables,
    time_derivative,
    to_first_order,
    transfer_function,
)
from symbolcat.ops.algebras import Literal

x, y, v, w, k, g, t, tau, s = (Symbol(n) for n in "x y v w k g t tau s".split())


def value(expr: Any, point: dict[Any, Any]) -> Any:
    return evaluate(substitute_variables(expr, point))


# events


def test_continuous_event():
    e = Symbol("e")
    event = ContinuousEvent(op.eq(y, 1), {v: -e * v}, direction=-1)
    assert event.condition == y - 1
    with pytest.raises(ValueError, match="direction"):
        ContinuousEvent(y, {v: 0}, direction=2)
    with pytest.raises(TypeError, match="variables"):
        ContinuousEvent(y, {2 * v: 0})


def test_events_follow_to_first_order():
    bounce = ContinuousEvent(x, {D(x): -D(x)}, direction=-1)
    ode = to_first_order(ODE({x: D(x), D(x): -g}, events=[bounce]))
    (event,) = ode.events
    x_t = Symbol("x_t")
    assert event.affect == {x_t: -x_t}


def test_events_follow_aliases():
    # v is an alias of w: an event on v assigns to w
    equations = [D(x) - v, v - w, D(w) + k * x]
    event = DiscreteEvent(op.eq(t, 1), {v: 2 * v})
    dae = structural_simplify(equations, [x, v, w], t, events=[event])
    (mapped,) = dae.events
    assert mapped.affect == {w: 2 * w}


def test_event_on_eliminated_expression():
    equations = [D(x) - v, v + 2 * w, D(w) + k * x]
    event = DiscreteEvent(op.eq(t, 1), {v: 0})
    with pytest.raises(ValueError, match="eliminated"):
        structural_simplify(equations, [x, v, w], t, events=[event])


def test_event_must_assign_states():
    event = DiscreteEvent(op.eq(t, 1), {v: 0})
    with pytest.raises(ValueError, match="not a state"):
        IndexOneDAE((D(x) - v, v - 1), (x,), (D(x), v), (), t, events=(event,))


def test_sensitivities_reject_events():
    ode = ODE({x: -k * x}, events=[DiscreteEvent(op.eq(t, 1), {x: 0})])
    with pytest.raises(NotImplementedError, match="events"):
        sensitivities(ode, [k])


# delays


def test_delays_in_derivatives():
    expr = x * delay(x, tau)
    assert delays(expr) == (delay(x, tau),)
    assert diff(expr, x) == delay(x, tau)
    assert time_derivative(expr, [x]) == D(x) * delay(x, tau) + x * delay(D(x), tau)


def test_delay_is_a_parameter_of_the_ode():
    ode = ODE({x: -k * delay(x, tau)})
    assert set(ode.parameters) == {k, tau}


def test_characteristic_matrix():
    ode = ODE({x: -k * delay(x, tau) - g * x})
    ((entry,),) = to_list(characteristic_matrix(ode, s))
    point = {s: 0.3, k: 2.0, tau: 0.5, g: 1.5}
    assert value(entry, point) == pytest.approx(0.3 + 1.5 + 2.0 * math.exp(-0.15))


def test_characteristic_matrix_without_delays():
    ode = ODE({x: v, v: -k * x})
    rows = to_list(characteristic_matrix(ode, s))
    assert rows == [[s, Literal(-1)], [k, s]]


# discrete


def test_discrete_system():
    r = Symbol("r")
    logistic = DiscreteSystem({x: r * x * (1 - x)})
    (eq,) = fixed_points(logistic).equations
    assert sorted(value(root, {r: 3.0}) for root in solve(eq, x)) == pytest.approx(
        [0.0, 2 / 3]
    )
    lin = linearize(logistic, at={x: 0})
    assert to_list(lin.A) == [[r]]


def test_iterate():
    twice = iterate(DiscreteSystem({x: y, y: x + y}), 2)
    assert value(twice.update[x], {x: 1, y: 1}) == 2
    assert value(twice.update[y], {x: 1, y: 1}) == 3


# SDE


def test_sde_diffusion_shapes():
    sde = SDE({x: -x, y: -y}, {x: [1, 0], y: [0, k]})
    assert sde.noise == 2
    assert sde.diffusion[y] == (Literal(0), k)
    with pytest.raises(ValueError, match="one coefficient per Wiener"):
        SDE({x: -x, y: -y}, {x: [1, 0], y: [k]})


def test_ito_stratonovich_roundtrip():
    mu, sigma = Symbol("mu"), Symbol("sigma")
    ito = SDE({x: mu * x}, {x: sigma * real.sqrt(x)})
    back = stratonovich_to_ito(ito_to_stratonovich(ito))
    point = {x: 1.7, mu: 0.3, sigma: 0.4}
    assert value(back.drift[x], point) == pytest.approx(0.3 * 1.7)
    with pytest.raises(ValueError, match="Itô"):
        ito_to_stratonovich(ito_to_stratonovich(ito))


def test_ito_lemma_geometric_brownian_motion():
    mu, sigma = Symbol("mu"), Symbol("sigma")
    gbm = SDE({x: mu * x}, {x: sigma * x})
    drift, (diffusion,) = ito_lemma(real.log(x), gbm)
    point = {x: 2.3, mu: 0.3, sigma: 0.4}
    assert value(drift, point) == pytest.approx(0.3 - 0.4**2 / 2)
    assert value(diffusion, point) == pytest.approx(0.4)
    # Stratonovich: the ordinary chain rule
    drift, _ = ito_lemma(real.log(x), ito_to_stratonovich(gbm))
    assert value(drift, point) == pytest.approx(0.3 - 0.4**2 / 2)


# PDE


def test_method_of_lines_heat():
    u, xs = Symbol("u"), Symbol("x")
    heat = D(u) - k * partial(partial(u, xs), xs)
    system = method_of_lines([heat], [u], xs, (0, 1), 5, left=[u], right=[u - 1])
    assert len(system.equations) == 5
    assert system.unknowns == tuple(grid_symbol(u, i) for i in range(5))
    dae = structural_simplify(system.equations, [grid_symbol(u, i) for i in range(5)])
    # the Dirichlet conditions are aliases of constants
    assert set(dae.observed) == {grid_symbol(u, 0), grid_symbol(u, 4)}
    assert dae.states == tuple(grid_symbol(u, i) for i in (1, 2, 3))


def test_method_of_lines_neumann_and_coordinate():
    u, xs = Symbol("u"), Symbol("x")
    # D(u) = u_x + x, with u_x = 0 on the left
    system = method_of_lines(
        [D(u) - partial(u, xs) - xs], [u], xs, (0, 2), 3, left=[partial(u, xs)], right=[u]
    )
    left, middle, right = system.equations
    u0, u1, u2 = (grid_symbol(u, i) for i in range(3))
    assert value(left, {u0: 1.0, u1: 3.0}) == 2.0
    assert value(middle, {D(u1): 5.0, u0: 1.0, u2: 3.0}) == 5.0 - 1.0 - 1.0


# analysis


def test_sparsity():
    assert jacobian_sparsity([x * D(x), y], [x, D(x), y]) == (
        (True, True, False),
        (False, False, True),
    )
    assert hessian_sparsity(x**2 + y, [x, y]) == ((True, False), (False, False))


def test_identifiability_two_compartments():
    """x1 feeds x2 at rate a, both decay; only x1 is measured."""
    x1, x2, a, b = (Symbol(n) for n in "x1 x2 a b".split())
    ode = ODE({x1: -(a + b) * x1, x2: a * x1 - b * x2})
    result = identifiability(ode, [x1])
    assert result[x1]
    assert not result[x2]  # x2 does not affect x1
    assert not result[a] and not result[b]  # only a + b is


def test_identifiability_observed_product():
    x1, a, b = Symbol("x1"), Symbol("a"), Symbol("b")
    result = identifiability(ODE({x1: -a * x1}), [b * x1], [a, b])
    assert result == {x1: False, a: True, b: False}


def test_transfer_function():
    u, c = Symbol("u"), Symbol("c")
    ode = ODE({x: v, v: -k * x - c * v + u})
    ((g_,),) = to_list(transfer_function(linearize(ode, [u], [x]), s))
    point = {s: 0.7, k: 2.0, c: 0.5}
    assert value(g_, point) == pytest.approx(1 / (0.7**2 + 0.5 * 0.7 + 2.0))


def test_transfer_function_needs_io():
    with pytest.raises(ValueError, match="inputs and outputs"):
        transfer_function(linearize(ODE({x: -x})), s)
