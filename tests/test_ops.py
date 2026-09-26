import math

import pytest

from symbolcat import (
    Function,
    Literal,
    Symbol,
    count_nodes,
    depth,
    diff,
    evaluate,
    free_symbols,
    real,
    simplify,
    substitute,
)
from symbolcat import operators as op
from symbolcat.lib.std import IMPL

x, y = Symbol("x"), Symbol("y")


# evaluate


def test_evaluate_std():
    assert evaluate(x + 2 * y, {x: 1, y: 3}) == 7
    assert evaluate(real.cos(real.pi)) == -1.0
    assert evaluate(x[1], {"x": [10, 20]}) == 20


def test_evaluate_missing_symbol():
    with pytest.raises(NameError, match="'x'"):
        evaluate(x + 1)


def test_user_symbol_is_not_a_constant():
    pi = Symbol("pi")
    with pytest.raises(NameError, match="'pi'"):
        evaluate(pi)
    assert evaluate(pi + real.pi, {pi: 1}) == 1 + math.pi
    assert substitute(pi * real.pi, {pi: 2}) == 2 * real.pi
    assert free_symbols(pi * real.pi) == (pi,)


def test_evaluate_missing_implementation():
    f = Function("f", arity=1)
    with pytest.raises(NotImplementedError, match="'f'"):
        evaluate(f(1))
    assert evaluate(f(x), {x: 2}, lib={**IMPL, f: lambda v: v * 10}) == 20


def test_evaluate_numpy():
    np = pytest.importorskip("numpy")
    from symbolcat.lib.numpy import IMPL as NP

    values = np.linspace(0, 1, 5)
    result = evaluate(real.exp(x) * real.pi, {x: values}, lib=NP)
    np.testing.assert_allclose(result, np.exp(values) * np.pi)


def test_evaluate_sympy():
    sympy = pytest.importorskip("sympy")
    from symbolcat.lib.sympy import IMPL as SP

    result = evaluate(real.sin(x) ** 2 + real.cos(x) ** 2, lib=SP)
    assert sympy.simplify(result) == 1


# substitute


def test_substitute_is_simultaneous():
    assert substitute(x - y, {x: y, y: x}) == y - x


def test_substitute_subexpressions_functions_and_literals():
    assert substitute(real.sin(x + y) * 2, {x + y: x}) == real.sin(x) * 2
    assert substitute(real.sin(x), {real.sin: real.cos}) == real.cos(x)
    assert substitute(x + 1, {1: 2}) == x + 2
    assert substitute(x + 1.0, {1: 2}) == x + 1.0


def test_substitute_with_values():
    assert simplify(substitute(x * y, {x: 2, y: 3})) == Literal(6)


# inspect


def test_free_symbols():
    assert free_symbols(y * real.pi + x * y) == (y, x)
    assert free_symbols(Literal(1)) == ()


def test_count_and_depth():
    assert count_nodes(x) == 1
    assert count_nodes(real.sin(x) + 1) == 4
    assert depth(real.sin(x) + 1) == 3


# simplify


@pytest.mark.parametrize(
    "expr, expected",
    [
        (x + 0, x),
        (0 + x, x),
        (x - 0, x),
        (0 - x, -x),
        (x - x, Literal(0)),
        (x * 1, x),
        (1 * x, x),
        (x * 0, Literal(0)),
        (-1 * x, -x),
        (x / 1, x),
        (x / x, Literal(1)),
        (x**0, Literal(1)),
        (x**1, x),
        (op.neg(-x), x),
        (+x, x),
        (Literal(2) * 3 + x, 6 + x),
        (x + real.sqrt(Literal(4)), x + 2.0),
    ],
)
def test_simplify(expr, expected):
    assert simplify(expr) == expected


def test_simplify_does_not_fold_errors():
    assert simplify(Literal(1) / 0) == Literal(1) / 0


# diff


@pytest.mark.parametrize(
    "expr, expected",
    [
        (Literal(3), Literal(0)),
        (x, Literal(1)),
        (y, Literal(0)),
        (x * y, y),
        (x**3, 3 * x**2),
        (real.sin(x), real.cos(x)),
        (real.exp(2 * x), real.exp(2 * x) * 2),
    ],
)
def test_diff_exact(expr, expected):
    assert diff(expr, x) == expected


@pytest.mark.parametrize(
    "expr",
    [
        x**2 * real.sin(x) / (1 + x),
        real.sqrt(x) * real.log(x),
        x**x,
        real.tan(x) - real.atan(x) + real.tanh(x),
        real.asin(x / 2) + real.acos(x / 3),
        real.cosh(x) * real.sinh(-x),
        2**x,
    ],
)
def test_diff_numeric(expr):
    """Compare against a central finite difference."""
    x0, h = 0.7, 1e-6
    numeric = (evaluate(expr, {x: x0 + h}) - evaluate(expr, {x: x0 - h})) / (2 * h)
    assert evaluate(diff(expr, x), {x: x0}) == pytest.approx(numeric, rel=1e-6)


def test_diff_higher_order():
    assert diff(diff(x**3, x), x) == 3 * (2 * x)


def test_diff_unknown_function():
    with pytest.raises(NotImplementedError, match="diff: no rule for erf"):
        diff(real.erf(x), x)


def test_math_consistency():
    assert evaluate(real.tau / 2) == math.pi
