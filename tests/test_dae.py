import random
from typing import Any

import pytest

from symbolcat import D, Symbol, real
from symbolcat.lib import operators as op
from symbolcat.ops import (
    IndexOneDAE,
    Pantelides,
    dummy_derivatives,
    eliminate_aliases,
    evaluate,
    initialization_system,
    lower_index,
    pantelides,
    solve_for_derivatives,
    structural_simplify,
    substitute_variables,
    time_derivative,
    variables,
)

x, y, w, z, T = Symbol("x"), Symbol("y"), Symbol("w"), Symbol("z"), Symbol("T")
g, L, k, t = Symbol("g"), Symbol("L"), Symbol("k"), Symbol("t")


def check_matching(result: Pantelides):
    """Each highest derivative is matched to a distinct highest equation."""
    highest = {i for i, d in enumerate(result.equation_derivative) if d is None}
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


def value(expr: Any, point: dict[Any, float]) -> float:
    """Evaluate, with the variables (including ``D(x)``) at ``point``."""
    return evaluate(substitute_variables(expr, point))


def test_lower_index_pendulum():
    dae = lower_index(pendulum())
    assert set(dae.states) == {x, y, w, z, D(x), D(y)}
    assert set(dae.unknowns) == {T, D(w), D(z), D(D(x)), D(D(y))}
    assert dae.parameters == (g,)  # L only appears in the dropped constraint


def test_dummy_derivatives_pendulum():
    dae = dummy_derivatives(pendulum(), prefer=[x])
    assert len(dae.equations) == 9  # every equation is kept
    assert dae.states == (x, D(x))
    assert set(dae.parameters) == {g, L}


def test_dummy_derivatives_prefer():
    dae = dummy_derivatives(pendulum(), prefer=[y])
    assert dae.states == (y, D(y))


def test_dummy_derivatives_index_two():
    result = pantelides([D(x) - y, x - real.sin(t)], [x, y], t)
    dae = dummy_derivatives(result)
    assert dae.states == ()
    assert set(dae.unknowns) == {x, y, D(x)}


def test_dummy_derivatives_ode_is_unchanged():
    dae = dummy_derivatives(pantelides([D(x) - w, D(w) + k * x], [x, w]))
    assert dae.states == (x, w)
    assert dae.unknowns == (D(x), D(w))


def test_index_one_dae_checks():
    with pytest.raises(ValueError, match="1 equations for 2 unknowns"):
        IndexOneDAE((x - 1,), (), (x, y), ())
    with pytest.raises(ValueError, match="derivative of the state x"):
        IndexOneDAE((y - 1,), (x,), (y,), ())
    with pytest.raises(ValueError, match="structurally singular"):
        IndexOneDAE((x - 1, x + 1), (), (x, y), ())


def test_pendulum_explicit_ode():
    """Solved for the highest derivatives, the tension of the pendulum is
    T = (g y - D(x)² - D(y)²) / (x² + y²)."""
    ode = solve_for_derivatives(lower_index(pendulum()))
    assert set(ode.states) == {x, y, w, z, D(x), D(y)}
    rng = random.Random(0)
    point = {v: rng.uniform(0.5, 1.5) for v in (x, y, w, z, D(x), D(y), g)}
    tension = (g * y - D(x) ** 2 - D(y) ** 2) / (x**2 + y**2)
    assert value(ode.rhs[w], point) == pytest.approx(value(tension * x, point))
    assert value(ode.rhs[z], point) == pytest.approx(value(tension * y - g, point))
    assert value(ode.rhs[D(x)], point) == pytest.approx(value(tension * x, point))


# initialization


def test_initialization_pendulum():
    system = initialization_system(pendulum())
    assert len(system.equations) == 9
    assert system.degrees_of_freedom == 2  # an angle and an angular velocity
    fixed = initialization_system(pendulum(), {x: 0.6, D(x): 0.0})
    assert fixed.degrees_of_freedom == 0
    assert x not in fixed.unknowns
    assert all(x not in variables(eq) for eq in fixed.equations)


def test_initialization_fixes_time():
    result = pantelides([D(x) - y, x - real.sin(t)], [x, y], t)
    system = initialization_system(result, {t: 0})
    assert system.equations[1] == x - real.sin(0)


# structural simplification


def test_alias_of_parameter():
    system, aliases = eliminate_aliases([D(x) - y, y - 2 * k], [x, y])
    assert aliases == {y: 2 * k}
    assert system.equations == (D(x) - 2 * k,)
    assert system.unknowns == (x,)


def test_alias_keeps_differentiated_unknown():
    system, aliases = eliminate_aliases([D(x) + 2 * y, x + 2 * y], [x, y])
    assert set(aliases) == {y}
    assert system.unknowns == (x,)


def test_alias_prefers_eliminating_undifferentiated():
    system, aliases = eliminate_aliases([D(y) - x, x + y], [x, y])
    assert set(aliases) == {x}
    assert system.unknowns == (y,)


def test_alias_substitutes_inside_derivatives():
    # both are differentiated: x = 2 * y is eliminated, and D(x) = 2 * D(y)
    system, aliases = eliminate_aliases([D(x) + D(y) - k, x - 2 * y], [x, y])
    assert aliases == {x: 2 * y}
    (eq,) = system.equations
    point = {D(y): 0.7, k: 0.3}
    assert value(eq, point) == pytest.approx(3 * 0.7 - 0.3)


def test_alias_chain():
    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    system, aliases = eliminate_aliases([a - b, b - c, D(c) + k * c], [a, b, c])
    assert system.unknowns == (c,)
    assert aliases == {a: c, b: c}


def test_nonlinear_is_not_an_alias():
    system, aliases = eliminate_aliases([D(x) - y, y - x**2], [x, y])
    assert aliases == {}
    assert len(system.equations) == 2


def test_structural_simplify_pendulum():
    theta = Symbol("theta")
    equations = [
        op.eq(D(x), w),
        op.eq(D(y), z),
        op.eq(D(w), T * x),
        op.eq(D(z), T * y - g),
        op.eq(0, x**2 + y**2 - L**2),
        op.eq(theta, x),  # an alias
    ]
    dae = structural_simplify(equations, [x, y, w, z, T, theta], prefer=[x])
    assert dae.observed == {theta: x}
    assert dae.states == (x, D(x))
    assert len(dae.equations) == 9
