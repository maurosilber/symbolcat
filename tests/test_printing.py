import pytest

from symbolcat import Symbol, real, to_latex, to_str
from symbolcat import operators as op

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")


@pytest.mark.parametrize(
    "expr, expected",
    [
        (x + y * z, "x + y * z"),
        ((x + y) * z, "(x + y) * z"),
        (x - (y - z), "x - (y - z)"),
        ((x - y) - z, "x - y - z"),
        (x / (y * z), "x / (y * z)"),
        (x**y**z, "x ** y ** z"),
        ((x**y) ** z, "(x ** y) ** z"),
        (-(x**2), "-x ** 2"),
        ((-x) ** 2, "(-x) ** 2"),
        (x + -1, "x + -1"),
        (x * -1, "x * -1"),
        (op.eq(x < y, z < x), "(x < y) == (z < x)"),
        (x[y + 1] * 2, "x[y + 1] * 2"),
        ((x + y)[0], "(x + y)[0]"),
        (real.atan2(x, y + 1), "real.atan2(x, y + 1)"),
        (real.pi * x, "real.pi * x"),
        (x & y | z, "x & y | z"),
        (x & (y | z), "x & (y | z)"),
    ],
)
def test_to_str(expr, expected):
    assert to_str(expr) == expected
    assert str(expr) == expected


def test_to_str_roundtrip():
    """Printed expressions are valid Python that rebuilds the same tree."""
    expr = (x + 2) ** -y / real.sin(x - y) % 3
    assert eval(to_str(expr), {"x": x, "y": y, "real": real}) == expr


@pytest.mark.parametrize(
    "expr, expected",
    [
        (x + y, "x + y"),
        (x * y, r"x \cdot y"),
        ((x + y) * z, r"\left(x + y\right) \cdot z"),
        (x / (y + z), r"\frac{x}{y + z}"),
        ((x + y) ** 2, r"\left(x + y\right)^{2}"),
        (x ** (y + z), "x^{y + z}"),
        ((x**y) ** z, r"\left(x^{y}\right)^{z}"),
        (-(x**2), "-x^{2}"),
        (x // y, r"\left\lfloor \frac{x}{y} \right\rfloor"),
        (x[0], "x_{0}"),
        (op.eq(x, y), "x = y"),
        (x <= y, r"x \leq y"),
        (real.sin(x), r"\sin\left(x\right)"),
        (real.sqrt(x + 1), r"\sqrt{x + 1}"),
        (real.abs(x), r"\left|x\right|"),
        (real.factorial(x + 1), r"\left(x + 1\right)!"),
        (real.log10(x), r"\log_{10}\left(x\right)"),
        (real.hypot(x, y), r"\operatorname{hypot}\left(x, y\right)"),
        (real.pi * real.e**x, r"\pi \cdot e^{x}"),
        (real.inf, r"\infty"),
        (x * 1e-5, r"x \cdot 10^{-5}"),
        (x * 2.5e20, r"x \cdot 2.5 \times 10^{20}"),
    ],
)
def test_to_latex(expr, expected):
    assert to_latex(expr) == expected


@pytest.mark.parametrize(
    "name, expected",
    [
        ("x", "x"),
        ("alpha", r"\alpha"),
        ("Omega", r"\Omega"),
        ("x_1", "x_{1}"),
        ("alpha_i_j", r"\alpha_{i_{j}}"),
        ("k_on", r"k_{\mathrm{on}}"),
        ("rate", r"\mathrm{rate}"),
    ],
)
def test_latex_symbol_names(name, expected):
    assert to_latex(Symbol(name)) == expected


def test_repr_latex():
    assert (x**2)._repr_latex_() == "$x^{2}$"
