import math
import xml.dom.minidom

import pytest

from symbolcat import Symbol, real
from symbolcat.lib import Function
from symbolcat.ops import (
    cse,
    evaluate,
    share,
    to_graphviz,
    to_mathml,
    to_python,
    tree_view,
)
from symbolcat.ops.algebras import Call, Literal, cata

x, y = Symbol("x"), Symbol("y")


def _distinct_calls(expr):
    visited = []
    cata(visited.append, expr)
    return sum(isinstance(node, Call) for node in visited)


# share and cse


def test_share_merges_equal_subexpressions():
    expr = real.sin(x + y) + real.cos(x + y) * (x + y)
    shared = share(expr)
    assert shared == expr
    assert _distinct_calls(shared) < _distinct_calls(expr)


def test_share_keeps_different_literals_apart():
    expr = share((x + 1) * (x + 1.0))
    assert isinstance(expr, Call)
    left, right = expr.args
    assert left is not right


def test_cse_names_repeated_subexpressions():
    s = real.sin(x)
    assignments, result = cse(s**2 + s * y + real.sin(x + y))
    assert [(str(t), str(v)) for t, v in assignments] == [("_0", "sin(x)")]
    assert str(result) == "_0 ** 2 + _0 * y + sin(x + y)"


def test_cse_orders_nested_temporaries():
    inner = x + y
    outer = real.exp(inner)
    assignments, result = cse(outer * outer + inner)
    assert [str(v) for _, v in assignments] == ["x + y", "exp(_0)"]
    assert str(result) == "_1 * _1 + _0"


def test_cse_avoids_existing_names():
    t = Symbol("_0")
    assignments, _ = cse(real.sin(t) * real.sin(t))
    assert [str(temp) for temp, _ in assignments] == ["_1"]


# to_python


def _compile(source, namespace):
    namespace = dict(namespace)
    exec(source, namespace)  # noqa: S102
    return namespace["f"]


@pytest.mark.parametrize(
    "expr",
    [
        real.sin(x) ** 2 + real.sin(x) * y,
        real.exp(-(x**2) / 2) / real.sqrt(2 * real.constants.pi),
        real.atan2(y, x) + real.hypot(x, y) - real.abs(x),
        (x + y) ** 3 / (1 + (x + y) ** 2),
    ],
)
def test_to_python_matches_evaluate(expr):
    f = _compile(to_python(expr, args=(x, y)), vars(math))
    for values in [(0.3, 1.7), (-1.2, 0.4)]:
        env = dict(zip((x, y), values, strict=True))
        assert f(*values) == pytest.approx(evaluate(expr, env))


def test_to_python_with_numpy():
    np = pytest.importorskip("numpy")
    f = _compile(to_python(real.sin(x) * real.exp(x)), vars(np))
    values = np.linspace(0, 1, 5)
    np.testing.assert_allclose(f(values), np.sin(values) * np.exp(values))


def test_to_python_checks_names():
    with pytest.raises(ValueError, match="not in args"):
        to_python(x + y, args=(x,))
    with pytest.raises(ValueError, match="identifier"):
        to_python(Symbol("a b") + 1)
    with pytest.raises(ValueError, match="identifier"):
        to_python(x, name="my function")


# tree_view and to_graphviz


def test_tree_view():
    f = Function("f", arity=2)
    expected = """\
+
├── f
│   ├── x
│   └── -
│       └── y
└── 1"""
    assert tree_view(f(x, -y) + 1) == expected


def test_to_graphviz_draws_shared_nodes_once():
    s = real.sin(x)
    dot = to_graphviz(s * s)
    assert dot.count('[label="sin"]') == 1
    assert dot.count("-> n1;") == 2  # both factors point to the sin node


def test_to_graphviz_escapes_labels():
    # the label of a string literal is its repr: '"a"'
    assert r'''label="'\"a\"'"''' in to_graphviz(Literal('"a"'))


# to_mathml


@pytest.mark.parametrize(
    "expr, fragment",
    [
        (x / y, "<mfrac><mi>x</mi><mi>y</mi></mfrac>"),
        (x**2, "<msup><mi>x</mi><mn>2</mn></msup>"),
        (
            (x + y) ** 2,
            "<mo>(</mo><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow><mo>)</mo>",
        ),
        (real.sqrt(x), "<msqrt><mi>x</mi></msqrt>"),
        (Symbol("alpha_1"), "<msub><mi>α</mi><mn>1</mn></msub>"),
        (real.constants.pi, "<mi>π</mi>"),
        (x <= y, "<mo>≤</mo>"),
        (x < y, "<mo>&lt;</mo>"),
        (real.abs(x), "<mo>|</mo><mi>x</mi><mo>|</mo>"),
        (x[0], "<msub><mi>x</mi><mn>0</mn></msub>"),
    ],
)
def test_to_mathml(expr, fragment):
    markup = to_mathml(expr)
    xml.dom.minidom.parseString(markup)  # well formed
    assert fragment in markup


def test_to_mathml_display():
    assert 'display="block"' in to_mathml(x, display=True)
