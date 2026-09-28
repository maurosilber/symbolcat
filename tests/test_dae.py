import math
from typing import Any

import pytest

from symbolcat import D, Symbol, real
from symbolcat.lib import operators as op
from symbolcat.ops import (
    IndexOneDAE,
    Pantelides,
    dummy_derivatives,
    lower_index,
    pantelides,
    time_derivative,
    to_ode,
)
from symbolcat.ops._dae._system import rename

x, y, w, z, T = Symbol("x"), Symbol("y"), Symbol("w"), Symbol("z"), Symbol("T")
g, L, k, t = Symbol("g"), Symbol("L"), Symbol("k"), Symbol("t")


def check_matching(result: Pantelides):
    """Each highest derivative is matched to a distinct highest equation."""
    highest = {
        i for i, d in enumerate(result.equation_derivative) if d is None
    }
    assert set(result.matching) == set(result.highest_variables)
    assert set(result.matching.values()) == highest


def test_ode_is_unchanged():
    result = pantelides([D(x) - w, D(w) + k * x], [x, w])
    assert result.differentiations == (0, 0)
    assert result.equations == (D(x) - w, D(w) + k * x)
    assert result.highest_variables == (D(x), D(w))
    check_matching(result)


def test_index_one():
    """An algebraic variable defined by an algebraic equation."""
    result = pantelides([D(x) - y, y - x**2], [x, y])
    assert result.differentiations == (0, 0)
    check_matching(result)


def test_index_two():
    result = pantelides([D(x) - y, x - real.sin(t)], [x, y], t)
    assert result.differentiations == (0, 1)
    assert result.equations[2] == D(x) - real.cos(t)
    check_matching(result)


def test_pendulum():
    """The Cartesian pendulum is index 3: the constraint is differentiated
    twice, and the velocity equations once."""
    equations = [
        op.eq(D(x), w),
        op.eq(D(y), z),
        op.eq(D(w), T * x),
        op.eq(D(z), T * y - g),
        op.eq(0, x**2 + y**2 - L**2),
    ]
    result = pantelides(equations, [x, y, w, z, T])
    assert result.differentiations == (1, 1, 0, 0, 2)
    assert set(result.highest_variables) == {T, D(w), D(z), D(D(x)), D(D(y))}
    once = result.equation_derivative[4]
    assert once is not None
    twice = result.equation_derivative[once]
    assert twice is not None
    constraint = result.equations[4]
    expected = time_derivative(time_derivative(constraint, [x, y]), [x, y])
    assert result.equations[twice] == expected
    check_matching(result)


def test_equation_as_eq():
    result = pantelides([op.eq(D(x), -k * x)], [x])
    assert result.equations == (D(x) - -k * x,)


def test_parameters_are_not_unknowns():
    result = pantelides([D(x) - k], [x])
    assert result.variables == (x, D(x))


def test_count_mismatch():
    with pytest.raises(ValueError, match="2 equations for 1 unknowns"):
        pantelides([D(x), x - 1], [x])


def test_structurally_singular():
    with pytest.raises(ValueError, match="structurally singular"):
        pantelides([D(x) - y, D(x) + y * x], [x, z])


# index-1 systems


def pendulum() -> Pantelides:
    equations = [
        op.eq(D(x), w),
        op.eq(D(y), z),
        op.eq(D(w), T * x),
        op.eq(D(z), T * y - g),
        op.eq(0, x**2 + y**2 - L**2),
    ]
    return pantelides(equations, [x, y, w, z, T])


def test_lower_index_pendulum():
    dae = lower_index(pendulum())
    assert set(dae.states) == {x, y, w, z, D(x), D(y)}
    assert set(dae.unknowns) == {T, D(w), D(z), D(D(x)), D(D(y))}
    assert dae.parameters == (g,)  # L only appears in the dropped constraint


def test_dummy_derivatives_pendulum():
    theta = 0.5
    at: dict[Any, float] = {x: math.sin(theta), y: -math.cos(theta), L: 1.0}
    dae = dummy_derivatives(pendulum(), at=at)
    assert len(dae.equations) == 9  # every equation is kept
    # Near the bottom, the horizontal position is the one to integrate.
    assert dae.states == (x, D(x))
    assert set(dae.parameters) == {g, L}


def test_dummy_derivatives_index_two():
    result = pantelides([D(x) - y, x - real.sin(t)], [x, y], t)
    dae = dummy_derivatives(result)
    assert dae.states == ()
    assert set(dae.unknowns) == {x, y, D(x)}


def test_dummy_derivatives_ode_is_unchanged():
    dae = dummy_derivatives(pantelides([D(x) - w, D(w) + k * x], [x, w]))
    assert dae.states == (x, w)
    assert dae.unknowns == (D(x), D(w))


def test_index_one_dae_checks_counts():
    with pytest.raises(ValueError, match="1 equations for 2 unknowns"):
        IndexOneDAE((x - 1,), (), (x, y), ())
    with pytest.raises(ValueError, match="derivative of the state x"):
        IndexOneDAE((y - 1,), (x,), (y,), ())


def test_rename_matches_original_nodes():
    a, b = Symbol("a"), Symbol("b")
    assert rename(D(D(x)) * D(x), {D(x): a, D(D(x)): b}) == b * a


# numeric


def rk4(f, u, t1: float, n: int):
    h, t_ = t1 / n, 0.0
    for _ in range(n):
        k1 = f(t_, u)
        k2 = f(t_ + h / 2, u + h / 2 * k1)
        k3 = f(t_ + h / 2, u + h / 2 * k2)
        k4 = f(t_ + h, u + h * k3)
        u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t_ += h
    return u


def test_to_ode_linear():
    pytest.importorskip("numpy")
    dae = lower_index(pantelides([D(x) - w, D(w) + k * x], [x, w]))
    f = to_ode(dae, {k: 1.0})
    u = rk4(f, f.state_vector({x: 1.0, w: 0.0}), math.pi, 200)
    assert u == pytest.approx([-1.0, 0.0], abs=1e-6)


def test_to_ode_missing_parameter():
    pytest.importorskip("numpy")
    dae = lower_index(pantelides([D(x) + k * x], [x]))
    with pytest.raises(ValueError, match="no value for k"):
        to_ode(dae)


def test_to_ode_explicit_time():
    pytest.importorskip("numpy")
    dae = dummy_derivatives(pantelides([D(x) - y, x - real.sin(t)], [x, y], t))
    f = to_ode(dae)
    unknowns = dict(zip(dae.unknowns, f.solve(1.0, []), strict=True))
    assert unknowns[y] == pytest.approx(math.cos(1.0))


@pytest.mark.parametrize("reduce", ["lower_index", "dummy_derivatives"])
def test_pendulum_matches_angle(reduce):
    """The Cartesian pendulum follows theta'' = -g/L sin(theta)."""
    np = pytest.importorskip("numpy")
    theta, gravity, t1, n = 0.5, 9.81, 2.0, 400
    initial: dict[Any, float] = {x: math.sin(theta), y: -math.cos(theta)}
    initial |= {w: 0.0, z: 0.0}
    initial |= {D(x): 0.0, D(y): 0.0}
    parameters: dict[Any, float] = {g: gravity, L: 1.0}
    if reduce == "lower_index":
        dae = lower_index(pendulum())
    else:
        dae = dummy_derivatives(pendulum(), at=initial | parameters)
    f = to_ode(
        dae,
        {p: parameters[p] for p in dae.parameters},
        guess={v: initial.get(v, 0.0) for v in dae.unknowns},
    )
    u = rk4(f, f.state_vector(initial), t1, n)
    values = dict(zip(dae.states, u, strict=True))
    values |= dict(zip(dae.unknowns, f.solve(t1, u), strict=True))

    angle = rk4(
        lambda _, s: np.array([s[1], -gravity * np.sin(s[0])]),
        np.array([theta, 0.0]),
        t1,
        n,
    )
    assert values[x] == pytest.approx(math.sin(angle[0]), abs=1e-6)
    assert values[y] == pytest.approx(-math.cos(angle[0]), abs=1e-6)
    constraint = values[x] ** 2 + values[y] ** 2 - 1
    if reduce == "dummy_derivatives":
        assert constraint == pytest.approx(0, abs=1e-12)
