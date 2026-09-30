from typing import Any

import pytest

from symbolcat import Symbol, linalg, real
from symbolcat.kinds import Matrix
from symbolcat.linalg import cross, det, inv, matrix, norm, trace, transpose, vector
from symbolcat.ops import evaluate
from symbolcat.ops._calculus._reverse import _postorder
from symbolcat.ops.algebras import Literal
from symbolcat.ops.calculus import gradient, reverse_gradient, vjp

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
A, B = Symbol("A", kind=Matrix(3, 3)), Symbol("B", kind=Matrix(3, 3))


@pytest.mark.parametrize(
    "expr",
    [
        x**2 * y,
        real.sin(x * y) + real.exp(y / x),
        x**x + 2**y,
        real.sqrt(x) * real.log(y) - real.tanh(x * y),
        real.tan(x) - real.atan(y) + real.asin(x / 2) + real.acos(y / 3),
        real.cosh(x) * real.sinh(-y) + (+x),
        (x + y) * (x + y) - x / y,
    ],
)
def test_matches_forward_mode(expr):
    env = {x: 0.7, y: 1.3}
    forward = evaluate(gradient(expr, (x, y)), env)
    reverse = evaluate(reverse_gradient(expr, (x, y)), env)
    assert reverse == pytest.approx(forward)


def test_exact():
    assert reverse_gradient(x**2 * y) == vector([2 * x * y, x**2])
    assert vjp(x * y, (x, y, z)) == (y, x, Literal(0))
    assert vjp(Literal(3), (x,)) == (Literal(0),)


def test_constant_exponent_has_no_log():
    (d,) = vjp(x**3, (x,))
    assert d == 3 * x**2


def test_cotangent():
    assert vjp(x * y, (x,), z) == (y * z,)


def test_explicit_arrays():
    v = vector([x, y])
    m = matrix([[x, 1], [y, x * y]])
    expr = linalg.sum(real.sin(v) * 2) + v @ m @ v + det(m) + trace(transpose(m))
    expr = expr + norm(m[1]) + v[0] * m[1][1]
    env = {x: 0.7, y: 1.3}
    forward = evaluate(gradient(expr, (x, y)), env)
    reverse = evaluate(reverse_gradient(expr, (x, y)), env)
    assert reverse == pytest.approx(forward)


def test_array_cotangent():
    v = vector([x * y, x])
    assert vjp(v, (x,), vector([1, 2])) == (2 + y,)


@pytest.mark.parametrize(
    "func",
    [
        lambda M: det(M),
        lambda M: trace(inv(M) @ M @ M),
        lambda M: norm(M),
        lambda M: linalg.sum(transpose(M) * M),
        lambda M: norm(cross(M[0], M[1])),
        lambda M: M[0] @ M @ M[1] + trace(M * 2),
    ],
)
def test_matrix_variable_matches_finite_differences(func):
    np = pytest.importorskip("numpy")
    from symbolcat.lib.numpy import IMPL

    rng = np.random.default_rng(0)
    a0, b0 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    expr = func(A @ B)
    (grad,) = vjp(expr, (A,))
    symbolic = evaluate(grad, {A: a0, B: b0}, lib=IMPL)

    h = 1e-6
    numeric = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            step = np.zeros((3, 3))
            step[i, j] = h
            plus = evaluate(expr, {A: a0 + step, B: b0}, lib=IMPL)
            minus = evaluate(expr, {A: a0 - step, B: b0}, lib=IMPL)
            numeric[i, j] = (plus - minus) / (2 * h)
    assert symbolic == pytest.approx(numeric, rel=1e-5, abs=1e-8)


def test_shared_nodes_are_visited_once():
    expr = x
    for _ in range(12):
        expr = expr * expr  # x ** 4096: a tree of 4096 leaves, 13 distinct nodes
    (d,) = vjp(expr, (x,))
    assert evaluate(d, {x: 1.0}) == 4096
    assert len(_postorder(d)) < 100  # distinct nodes


def test_unknown_function():
    with pytest.raises(NotImplementedError, match="reverse: no rule for gamma"):
        vjp(real.gamma(x), (x,))


def test_unknown_function_of_constants_is_fine():
    assert vjp(real.erf(y) * x, (x,)) == (real.erf(y),)


def test_with_respect_to_expression():
    var: Any = 2 * x
    with pytest.raises(TypeError, match="with respect to a Symbol"):
        vjp(x, (var,))
