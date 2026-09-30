import random
from typing import Any

import pytest

from symbolcat import D, Symbol, real
from symbolcat import operators as op
from symbolcat.ops import evaluate
from symbolcat.ops._algebraic._matching import perfect_matching
from symbolcat.ops.algebras import Literal
from symbolcat.ops.equations import EquationSystem, block_triangular, solve_linear, tear
from symbolcat.ops.transform import substitute_variables

x, y, z, a, b = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("a"), Symbol("b")


def value(expr: Any, point: dict[Any, float]) -> float:
    return evaluate(substitute_variables(expr, point))


# EquationSystem


def test_equation_system():
    system = EquationSystem([op.eq(x, y), x + y - a], [x, y])
    assert system.equations == (x - y, x + y - a)
    assert system.incidence == ((0, 1), (0, 1))
    assert system.degrees_of_freedom == 0


def test_equation_system_derivative_unknowns():
    system = EquationSystem([D(x) - y], [D(x), y])
    assert system.incidence == ((0, 1),)


def test_equation_system_checks_unknowns():
    with pytest.raises(TypeError, match="symbols or D"):
        EquationSystem([x], [2 * x])
    with pytest.raises(ValueError, match="repeated"):
        EquationSystem([x], [x, x])


def test_perfect_matching():
    assert perfect_matching([[0, 1], [0]], 2) == {0: 1, 1: 0}
    assert perfect_matching([[0], [0]], 2) is None
    assert perfect_matching([[0]], 2) is None


# substitute_variables


def test_substitute_variables_is_atomic():
    assert substitute_variables(D(x) + x, {x: y}) == D(x) + y
    assert substitute_variables(D(D(x)) + D(x), {D(x): y}) == D(D(x)) + y
    assert substitute_variables(real.sin(x), {real.sin(x): y}) == y


# block triangular form


def test_block_triangular_order():
    system = EquationSystem([z - x * y, x + y - 3, x - y - 1], [x, y, z])
    blocks = block_triangular(system)
    assert [set(block.unknowns) for block in blocks] == [{x, y}, {z}]


def test_block_triangular_chain():
    system = EquationSystem([z - y, y - x, x - 1], [x, y, z])
    blocks = block_triangular(system)
    assert [block.unknowns for block in blocks] == [(x,), (y,), (z,)]


def test_block_triangular_singular():
    with pytest.raises(ValueError, match="structurally singular"):
        block_triangular(EquationSystem([x - 1, x + 1], [x, y]))


# tearing


def test_tear_explicit_chain():
    torn = tear(EquationSystem([y - 2 * x, x - a], [x, y]))
    assert torn.tears == ()
    assert torn.residuals == ()
    assert dict(torn.solved) == {x: a, y: 2 * x}


def test_tear_nonlinear():
    torn = tear(EquationSystem([x**2 + y**2 - 1, y - x], [x, y]))
    assert len(torn.tears) == 1
    assert len(torn.residuals) == 1


def test_tear_skips_zero_coefficient():
    torn = tear(EquationSystem([x * y - 1, y - 2], [x, y]))
    assert dict(torn.solved)[y] == Literal(2)
    # x * y - 1 is linear in x, with a coefficient that is known once y is
    assert [u for u, _ in torn.solved] == [y, x]


# linear systems


def test_solve_linear():
    rng = random.Random(0)
    point = {a: rng.random(), b: rng.random()}
    solution = solve_linear([x + y - a, x - y - b], [x, y])
    assert value(solution[x], point) == pytest.approx((point[a] + point[b]) / 2)
    assert value(solution[y], point) == pytest.approx((point[a] - point[b]) / 2)


def test_solve_linear_symbolic_coefficients():
    rng = random.Random(1)
    point = {a: rng.random(), b: rng.random()}
    solution = solve_linear([a * x + y - 1, x + b * y], [x, y])
    det = point[a] * point[b] - 1
    assert value(solution[x], point) == pytest.approx(point[b] / det)
    assert value(solution[y], point) == pytest.approx(-1 / det)


def test_solve_linear_needs_pivoting():
    solution = solve_linear([y - 1, x + y - 3], [x, y])
    assert value(solution[x], {}) == 2
    assert value(solution[y], {}) == 1


def test_solve_linear_derivatives():
    solution = solve_linear([D(x) - y, 2 * D(y) + x], [D(x), D(y)])
    assert solution[D(x)] == y
    assert value(solution[D(y)], {x: 1.0}) == -0.5


def test_solve_linear_nonlinear():
    with pytest.raises(ValueError, match="not linear"):
        solve_linear([x * y - 1, x - y], [x, y])


def test_solve_linear_singular():
    with pytest.raises(ValueError, match="singular"):
        solve_linear([x + y, 2 * x + 2 * y - 1], [x, y])


def test_solve_linear_count():
    with pytest.raises(ValueError, match="1 equations for 2 unknowns"):
        solve_linear([x + y], [x, y])
