import pytest

pytest.importorskip("egglog")

from egglog import EGraph

from symbolcat import Symbol, real
from symbolcat.lib import Constant, Function
from symbolcat.lib.std import IMPL
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Literal
from symbolcat.ops.egraph import optimize, simplify
from symbolcat.ops.egraph._convert import Table, from_egglog, to_egglog

x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
f = Function("f", arity=1)


@pytest.mark.parametrize(
    "expr",
    [
        x + 2 * y - x / 3,
        real.sin(x) ** -1 + real.sqrt(real.exp(x)),
        f(x + 1) * real.constants.pi,
        Literal(1) / 3 + 0.25,
        real.atan2(x, y)[0],
    ],
)
def test_roundtrip(expr):
    table = Table()
    egraph = EGraph()
    term = egraph.extract(egraph.let("root", to_egglog(expr, table)))
    assert from_egglog(term, table) == expr


def test_opaque_parts_keep_identity():
    """Functions and constants egglog does not know come back as the same objects."""
    g = Function("f", arity=1)  # same name as f, different function
    k = Constant("k")
    expr = simplify(f(x) + g(x) + k * 1)
    funcs = {id(arg.func) for arg in expr.args[0].args}
    assert funcs == {id(f), id(g)}
    assert expr.args[1] is k


def test_known_constants_come_back():
    pi, e = real.constants.pi, real.constants.e
    assert simplify(pi + e * 1) == pi + e


@pytest.mark.parametrize(
    "expr, expected",
    [
        (x * y + x * z, x * (y + z)),
        (real.sin(x) ** 2 + real.cos(x) ** 2 + 1, Literal(2)),
        (x / x + 0 * y, Literal(1)),
        ((x + y) - y, x),
        (real.exp(x) * real.exp(y), real.exp(x + y)),
        (Literal(1) / 2 + Literal(1) / 3, Literal(5) / 6),
        (Literal(2) ** 10 - 1000, Literal(24)),
        (real.log(real.exp(x)), x),
        (real.tan(x) * real.cos(x), real.sin(x)),
        (x - x + f(x * 1), f(x)),
        (x + 0.5 + 0.25, x + 0.75),
        (x + 0.5 + Literal(1) / 4, x + 0.75),
        ((x + 1) ** 2 - x**2 - 2 * x, Literal(1)),
        # constants
        (real.sin(real.constants.pi) + real.cos(real.constants.pi), Literal(-1)),
        (real.log(real.constants.e), Literal(1)),
        (real.constants.e**x / real.exp(x), Literal(1)),
        # powers and absolute value
        ((x * y) ** 2 / y**2, x**2),
        (real.sqrt(x**2), real.abs(x)),
        (real.abs(-x) - real.abs(x), Literal(0)),
        # exponential and logarithm
        (real.log(x * y) - real.log(x), real.log(y)),
        (real.log(x**3) / real.log(x), Literal(3)),
        (real.exp(real.log(x)), x),
        # trigonometric
        (real.sin(x) * real.cos(y) + real.cos(x) * real.sin(y), real.sin(y + x)),
        (1 - real.cos(x) ** 2, real.sin(x) ** 2),
        (real.sin(real.asin(x)), x),
        # hyperbolic
        (real.cosh(x) + real.sinh(x), real.exp(x)),
        (real.cosh(x) ** 2 - real.sinh(x) ** 2, Literal(1)),
        (real.tanh(x) * real.cosh(x), real.sinh(x)),
    ],
)
def test_simplify(expr, expected):
    assert simplify(expr) == expected


def test_simplify_does_not_divide_by_zero():
    assert simplify(Literal(1) / 0) == Literal(1) / 0


def test_optimize_shares_subexpressions():
    expr = optimize(real.sin(x) ** 2 + real.sin(x) * y)
    sins = [node for node in _walk(expr) if getattr(node, "func", None) is real.sin]
    assert len({id(node) for node in sins}) == 1


def test_optimize_avoids_expensive_operations():
    expr = optimize(x**3 + x**2)
    assert "**" not in str(expr)


@pytest.mark.parametrize(
    "expr",
    [
        x**3 + x**2 + x,
        (x + y) ** 2 - (x - y) ** 2,
        real.sin(x) ** 2 + real.sin(x) * real.cos(x) + x / x,
        real.sqrt(x**2) + real.abs(x * y),
        real.sin(x + y) - real.cos(x - y) + real.tan(-x),
        real.sinh(x) ** 2 + real.tanh(-y) * real.cosh(x + y),
        (x * y) ** 3 / (x**2 * y) + real.atan(x) - real.acos(x / 2),
    ],
)
@pytest.mark.parametrize("transform", [simplify, optimize])
@pytest.mark.parametrize("env", [{x: 0.3, y: -1.7}, {x: -0.8, y: 0.4}])
def test_equivalent_values(expr, transform, env):
    """Rules must not change values, including at negative points."""
    assert evaluate(transform(expr), env, IMPL) == pytest.approx(
        evaluate(expr, env, IMPL)
    )


def _walk(expr):
    stack, seen = [expr], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(getattr(node, "args", ()))
