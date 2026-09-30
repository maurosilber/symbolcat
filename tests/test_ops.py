import math
from typing import Any

import pytest

from symbolcat import Function, Symbol, real
from symbolcat import operators as op
from symbolcat.backends.std import IMPL
from symbolcat.linalg import matrix, to_list, vector
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Literal
from symbolcat.ops.calculus import diff
from symbolcat.ops.inspect import count_nodes, depth, free_symbols
from symbolcat.ops.transform import simplify_naive, substitute

x, y = Symbol("x"), Symbol("y")


# evaluate


def test_evaluate_std():
    assert evaluate(x + 2 * y, {x: 1, y: 3}) == 7
    assert evaluate(real.cos(real.constants.pi)) == -1.0
    assert evaluate(x[1], {"x": [10, 20]}) == 20


def test_evaluate_missing_symbol():
    with pytest.raises(NameError, match="'x'"):
        evaluate(x + 1)


def test_user_symbol_is_not_a_constant():
    pi = Symbol("pi")
    with pytest.raises(NameError, match="'pi'"):
        evaluate(pi)
    assert evaluate(pi + real.constants.pi, {pi: 1}) == 1 + math.pi
    assert substitute(pi * real.constants.pi, {pi: 2}) == 2 * real.constants.pi
    assert free_symbols(pi * real.constants.pi) == (pi,)


def test_evaluate_missing_implementation():
    f = Function("f", arity=1)
    with pytest.raises(NotImplementedError, match="'f'"):
        evaluate(f(1))
    assert evaluate(f(x), {x: 2}, backend={**IMPL, f: lambda v: v * 10}) == 20


def test_evaluate_numpy():
    np = pytest.importorskip("numpy")
    from symbolcat.backends.numpy import IMPL as NP

    values = np.linspace(0, 1, 5)
    result = evaluate(real.exp(x) * real.constants.pi, {x: values}, backend=NP)
    np.testing.assert_allclose(result, np.exp(values) * np.pi)


def test_evaluate_sympy():
    sympy = pytest.importorskip("sympy")
    from symbolcat.backends.sympy import IMPL as SP

    result = evaluate(real.sin(x) ** 2 + real.cos(x) ** 2, backend=SP)
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
    assert simplify_naive(substitute(x * y, {x: 2, y: 3})) == Literal(6)


# inspect


def test_free_symbols():
    assert free_symbols(y * real.constants.pi + x * y) == (y, x)
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
    assert simplify_naive(expr) == expected


def test_simplify_does_not_fold_errors():
    undefined = Literal(1) / 0  # pyrefly: ignore[division-by-zero]
    assert simplify_naive(undefined) == undefined


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
    with pytest.raises(NotImplementedError, match="diff: no rule for gamma"):
        diff(real.gamma(x), x)


def test_diff_with_respect_to_expression():
    var: Any = 2 * x
    with pytest.raises(TypeError, match="with respect to a Symbol"):
        diff(var, var)


# linalg


def test_arrays_compare_by_shape_and_entries():
    assert vector([x, y]) == vector([x, y])
    assert vector([x, y]) != vector([y, x])
    assert vector([x, y]) != matrix([[x, y]])
    assert to_list(matrix([[x, y]])) == [[x, y]]


def test_matrix_rows_must_have_the_same_length():
    with pytest.raises(ValueError, match="same length"):
        matrix([[x, y], [x]])


def test_evaluate_arrays():
    m = matrix([[x, 1], [0, 2 * x]])
    assert evaluate(m, {x: 3}) == [[3, 1], [0, 6]]
    np = pytest.importorskip("numpy")
    from symbolcat.backends.numpy import IMPL as NUMPY

    assert (evaluate(m, {x: 3}, backend=NUMPY) == np.array([[3, 1], [0, 6]])).all()


def test_diff_array_entrywise():
    assert diff(vector([x**2, y]), x) == vector([2 * x, Literal(0)])


def test_math_consistency():
    assert evaluate(real.constants.tau / 2) == math.pi


def test_ifelse_only_evaluates_the_branch_taken():
    expr = real.ifelse(x > 0, real.log(x), 0)
    assert evaluate(expr, {x: -1}) == 0
    assert evaluate(expr, {x: math.e}) == 1


def test_ifelse_with_array_conditions_needs_both_branches():
    import numpy as np

    from symbolcat.backends import numpy as np_lib

    values = np.array([-1.0, 4.0])
    expr = real.ifelse(x > 0, real.sqrt(x), 0)
    with np.errstate(invalid="ignore"):
        result = evaluate(expr, {x: values}, backend=np_lib.IMPL)
    assert list(result) == [0, 2]


@pytest.mark.parametrize("a", [0.3, 1.0, 2.5, 30.0])
@pytest.mark.parametrize("value", [0.0, 0.1, 1.0, 3.0, 40.0])
def test_gammainc(a, value):
    import sympy

    lower: Any = sympy.lowergamma(a, value)
    expected = float((lower / sympy.gamma(a)).evalf(30))
    assert evaluate(real.gammainc(a, x), {x: value}) == pytest.approx(
        expected, abs=1e-14
    )


def test_gammainc_derivative():
    a = Symbol("a")
    expr = real.gammainc(2.5, x**2)
    h, point = 1e-6, 0.8
    numeric = (evaluate(expr, {x: point + h}) - evaluate(expr, {x: point - h})) / (
        2 * h
    )
    assert evaluate(diff(expr, x), {x: point}) == pytest.approx(numeric)
    with pytest.raises(NotImplementedError, match="with respect to a"):
        diff(real.gammainc(a, x), a)
