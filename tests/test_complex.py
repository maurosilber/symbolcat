import builtins
import cmath
import math

import pytest

from symbolcat import Symbol, complex, linalg, real
from symbolcat.lib import Function
from symbolcat.lib import numpy as np_lib
from symbolcat.lib import sympy as sympy_lib
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Call, Literal, func_of
from symbolcat.ops.calculus import diff, series, vjp
from symbolcat.ops.inspect import dimensions, is_complex, shape
from symbolcat.ops.printing import to_c, to_julia, to_latex, to_mathml, to_python
from symbolcat.ops.transform import real_imag, simplify_naive

x, y = Symbol("x"), Symbol("y")
I = complex.constants.I
z = x + I * y
POINT = {x: 0.3, y: -0.7}

UNARY = [
    complex.re,
    complex.im,
    complex.conj,
    complex.abs,
    complex.phase,
    complex.exp,
    complex.log,
    complex.log10,
    complex.sqrt,
    complex.sin,
    complex.cos,
    complex.tan,
    complex.asin,
    complex.acos,
    complex.atan,
    complex.sinh,
    complex.cosh,
    complex.tanh,
]


# evaluation


def test_principal_branch():
    assert evaluate(complex.sqrt(-1)) == 1j
    assert evaluate(complex.log(-1)) == cmath.log(-1)
    assert evaluate(complex.rect(2, math.pi / 2)) == pytest.approx(2j)
    assert evaluate(I * I) == -1


@pytest.mark.parametrize("func", UNARY)
def test_backends_agree(func):
    expr = func(z)
    expected = evaluate(expr, POINT)
    assert builtins.complex(evaluate(expr, POINT, lib=np_lib.IMPL)) == pytest.approx(
        expected
    )
    value = evaluate(expr, POINT, lib=sympy_lib.IMPL)
    assert builtins.complex(value.evalf()) == pytest.approx(expected)


@pytest.mark.parametrize("func", [complex.sqrt, complex.log, complex.asin])
def test_numpy_outside_the_real_domain(func):
    """NumPy's real functions give nan for sqrt(-1): these give complex."""
    expected = evaluate(func(x), {x: -2.0})
    assert builtins.complex(evaluate(func(x), {x: -2.0}, lib=np_lib.IMPL)) == expected


def test_entrywise():
    v = linalg.vector([x, y])
    assert shape(complex.exp(I * v)) == (2,)
    assert evaluate(complex.re(complex.exp(I * v)), POINT) == pytest.approx(
        [math.cos(0.3), math.cos(-0.7)]
    )


# derivatives


EXPRESSIONS = [
    complex.re(complex.exp(I * x) * y),
    complex.im(complex.log(z) ** 2),
    complex.abs(complex.sin(z)),
    complex.phase(complex.sqrt(z) + 2),
    complex.re(complex.conj(z) * complex.tanh(z)),
    complex.abs(complex.rect(x**2 + 1, y * x)),
    complex.re(complex.rect(y, x) * complex.atan(z)),
    # a real node built from complex ones, inside a complex function
    complex.re(complex.exp(I * (z * complex.conj(z)))),
    complex.im(complex.acos(z / 2) * complex.asin(z / 3) + complex.log10(z)),
    real.sin(complex.abs(z)) * complex.re(complex.cosh(z) / complex.tan(z)),
]


def finite_difference(expr, var, h=1e-6):
    def at(value):
        return evaluate(expr, {**POINT, var: value})

    return (at(POINT[var] + h) - at(POINT[var] - h)) / (2 * h)


@pytest.mark.parametrize("expr", EXPRESSIONS)
@pytest.mark.parametrize("var", [x, y])
def test_diff(expr, var):
    assert evaluate(diff(expr, var), POINT) == pytest.approx(
        finite_difference(expr, var), rel=1e-5, abs=1e-8
    )


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_reverse_is_real(expr):
    """Real-valued expressions have real gradients."""
    for d, var in zip(vjp(expr, (x, y)), (x, y)):
        assert not is_complex(d)
        assert evaluate(d, POINT) == pytest.approx(
            evaluate(diff(expr, var), POINT), rel=1e-9, abs=1e-12
        )


def test_reverse_of_a_complex_output():
    """The cotangent pairs as re(g * df): 1 gives the gradient of re(f),
    and -I that of im(f)."""
    f = complex.exp(I * x * y)
    for g, part in [(1, complex.re), (-I, complex.im)]:
        (d,) = vjp(f, (x,), g)
        assert evaluate(d, POINT) == pytest.approx(evaluate(diff(part(f), x), POINT))


def test_derivative_of_real_arguments():
    assert diff(complex.re(x**2), x) == 2 * x
    assert vjp(complex.im(x) + complex.phase(x), (x,)) == (Literal(0),)
    assert diff(complex.exp(I * x), x) == complex.exp(I * x) * I


def test_series():
    assert str(series(complex.exp(I * x), x, order=3)) == "1 + I * x - x ** 2 / 2"


# simplification


def test_simplify():
    assert simplify_naive(complex.re(x)) == x
    assert simplify_naive(complex.conj(x * y)) == x * y
    assert simplify_naive(complex.im(x)) == Literal(0)
    assert simplify_naive(complex.conj(complex.conj(z))) == z
    assert simplify_naive(complex.re(z)) == complex.re(z)
    assert simplify_naive(I * I) == Literal(-1)
    assert simplify_naive(complex.exp(0)) == Literal(1.0)
    assert simplify_naive(complex.sqrt(-4)) == Literal(2j)


def test_real_functions_stay_real():
    """real.sqrt(-1) is not a number, so it is kept."""
    assert simplify_naive(real.sqrt(-1)) == real.sqrt(-1)
    with pytest.raises(ValueError):
        evaluate(real.sqrt(-1))


def test_is_complex():
    assert not is_complex(x * real.sin(y))
    assert is_complex(z)
    assert is_complex(Literal(1j))
    assert is_complex(complex.conj(z))
    assert not is_complex(complex.conj(x))
    assert not is_complex(complex.abs(z) + complex.re(z))


def test_dimensions():
    units = {x: {"m": 1}, y: {"s": 1}}
    assert dimensions(complex.abs(complex.rect(x, y / y)), units) == {"m": 1}
    assert dimensions(complex.sqrt(x * x), units) == {"m": 1}
    assert dimensions(complex.phase(x), units) == {}
    with pytest.raises(ValueError):
        dimensions(complex.exp(x), units)


# printing


@pytest.mark.parametrize(
    "expr, expected",
    [
        (
            complex.re(complex.exp(I * x)),
            r"\operatorname{Re}\left(\exp\left(i \cdot x\right)\right)",
        ),
        (complex.conj(x + y), r"\overline{x + y}"),
        (complex.abs(z), r"\left|x + i \cdot y\right|"),
        (complex.phase(z), r"\arg\left(x + i \cdot y\right)"),
        (complex.rect(2, x) * y, r"2 e^{i x} \cdot y"),
        (Literal(1 + 2j) * x, r"\left(1 + 2 i\right) \cdot x"),
        (Literal(-1j) ** x, r"\left(-i\right)^{x}"),
    ],
)
def test_latex(expr, expected):
    assert to_latex(expr) == expected


def test_mathml():
    assert "<mi>Re</mi>" in to_mathml(complex.re(z))
    assert "<mi>i</mi>" in to_mathml(I)
    assert "¯" in to_mathml(complex.conj(x))


def test_python():
    expr = complex.re(complex.exp(I * x) * complex.conj(z)) + complex.phase(z)
    source = to_python(expr)
    assert ".real" in source and ".conjugate()" in source and "1j" in source
    namespace = dict(vars(cmath))
    exec(source, namespace)  # noqa: S102
    assert namespace["f"](0.3, -0.7) == pytest.approx(evaluate(expr, POINT))


def test_julia():
    expr = complex.re(complex.sqrt(x) * complex.rect(1, y)) + complex.phase(I * x)
    assert to_julia(expr) == (
        "function f(x, y)\n"
        "    return real(sqrt(complex(x)) * (1 * cis(y))) + angle(im * x)\n"
        "end"
    )


@pytest.mark.parametrize("expr", [complex.re(x), I * x, Literal(1j) * x])
def test_c_rejects_complex(expr):
    with pytest.raises(NotImplementedError):
        to_c(expr)


def test_diff_of_a_complex_power():
    f = complex.re(z**x)
    assert evaluate(diff(f, x), POINT) == pytest.approx(finite_difference(f, x))
    (d,) = vjp(f, (x,))
    assert evaluate(d, POINT) == pytest.approx(finite_difference(f, x))


# real and imaginary parts

PARTS = [
    *(func(z) for func in UNARY),
    *(func(x) for func in UNARY),
    z**3,
    z**-2,
    z**x,
    z ** (x + I * y),
    z / (x - 2 * I),
    complex.rect(x, y) * z,
    real.ifelse(x < 0, z, complex.conj(z)),
    complex.exp(complex.sqrt(z) * complex.log(z)),
    linalg.vector([z, x]) * I,
]
GRID = [(u / 2, v / 2) for u in range(-5, 6) for v in range(-5, 6)]


@pytest.mark.parametrize("expr", PARTS, ids=str)
def test_real_imag(expr):
    re, im = real_imag(expr)
    assert not is_complex(re) and not is_complex(im)
    for u, v in GRID:
        point = {x: u, y: v}
        on_cut = (func_of(expr) if isinstance(expr, Call) else None) in (
            complex.asin,
            complex.acos,
        )
        if on_cut and v == 0 and abs(u) > 1:
            continue  # asin and acos on their branch cut: cmath uses the sign of 0
        try:
            expected = evaluate(expr, point)
            got_re, got_im = evaluate(re, point), evaluate(im, point)
        except (ValueError, ZeroDivisionError):
            continue  # singular points, like log(0) in z ** x at z = 0
        if isinstance(expected, list):
            got = [builtins.complex(a, b) for a, b in zip(got_re, got_im)]
            assert got == pytest.approx(expected)
        else:
            assert builtins.complex(got_re, got_im) == pytest.approx(expected)


def test_real_imag_of_real_input_on_the_branch_cut():
    for func in (complex.asin, complex.acos, complex.sqrt, complex.log):
        re, im = real_imag(func(x))
        for value in (-2.5, 2.5):
            expected = evaluate(func(x), {x: value})
            got = builtins.complex(evaluate(re, {x: value}), evaluate(im, {x: value}))
            assert got == pytest.approx(expected)


def test_real_imag_keeps_unknown_functions():
    f = Function("f", arity=1)
    re, im = real_imag(f(z))
    assert re == complex.re(f(z)) and im == complex.im(f(z))
    assert real_imag(f(x)) == (f(x), Literal(0))
