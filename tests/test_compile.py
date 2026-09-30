import inspect
import math

import pytest

from symbolcat import Function, Symbol, linalg, real
from symbolcat import complex as cx
from symbolcat import operators as op
from symbolcat.backends.std import IMPL
from symbolcat.ops import compile, evaluate

x, y = Symbol("x"), Symbol("y")

EXPRESSIONS = [
    x + 2 * y,
    real.sin(x) ** 2 + real.sin(x) * y,
    real.atan2(y, x) - real.constants.pi,
    real.ifelse(x < y, x, y),
    (x > 0) & (y > 0),
    -(x**-y),
    cx.re(cx.exp(cx.constants.I * x)) + y,
]


@pytest.mark.parametrize("expr", EXPRESSIONS)
@pytest.mark.parametrize("backend", ["std", "numpy", "sympy"])
def test_compile_is_evaluate(expr, backend):
    pytest.importorskip(backend if backend != "std" else "math")
    module = __import__(f"symbolcat.backends.{backend}", fromlist=["IMPL"])
    f = compile(expr, [x, y], backend=module.IMPL)
    expected = evaluate(expr, {x: 1.5, y: 0.5}, backend=module.IMPL)
    assert f(1.5, 0.5) == pytest.approx(expected)


def test_compile_arrays():
    np = pytest.importorskip("numpy")
    from symbolcat.backends.numpy import IMPL as NP

    m = linalg.matrix([[x, 1], [0, y]])
    assert compile(m @ m, [x, y])(2, 3) == [[4, 5], [0, 9]]
    result = compile(m @ m, [x, y], backend=NP)(2, 3)
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [[4, 5], [0, 9]]


def test_compile_names():
    sin = Symbol("sin")
    f = compile(real.sin(x) + cx.sin(x) + sin, [x, sin])
    assert f(1.0, 2.0) == pytest.approx(math.sin(1) + math.sin(1) + 2)
    source = inspect.getsource(f)
    assert source.startswith("def f(x, sin):")
    assert "sin_1(x)" in source and "sin_2(x)" in source


def test_compile_arguments():
    f = compile(x - y, [y, x], name="g")
    assert f.__name__ == "g"
    assert f(1, 3) == 2
    with pytest.raises(ValueError, match="Free symbols not in args: y"):
        compile(x - y, [x])


def test_compile_literals():
    assert compile(x + math.inf)(1.0) == math.inf
    assert compile(x * -2)(3) == -6


def test_compile_ifelse_is_lazy():
    assert compile(real.ifelse(x > 0, real.log(x), 0))(-1.0) == 0


def test_compile_implementations():
    f = Function("f", arity=1)
    with pytest.raises(NotImplementedError, match="'f'"):
        compile(f(x))
    assert compile(f(x) + 1, backend={**IMPL, f: lambda v: v * 10})(2) == 21


def test_compile_non_python_operators():
    sympy = pytest.importorskip("sympy")
    from symbolcat.backends.sympy import IMPL as SP

    f = compile(op.eq(x, y), backend=SP)
    assert "eq(x, y)" in inspect.getsource(f)
    a, b = sympy.symbols("a b")
    assert f(a, b) == sympy.Eq(a, b)
