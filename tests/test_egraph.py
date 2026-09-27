import pytest

pytest.importorskip("egglog")

from egglog import EGraph

from symbolcat import Function, Literal, Symbol, evaluate, real
from symbolcat.lib.std import IMPL
from symbolcat.ops.egraph import optimize, simplify
from symbolcat.ops.egraph.convert import Table, from_egglog, to_egglog

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
    expr = simplify(f(x) + g(x) + real.constants.pi * 1)
    funcs = {id(arg.func) for arg in expr.args[0].args}
    assert funcs == {id(f), id(g)}
    assert expr.args[1] is real.constants.pi


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
    ],
)
@pytest.mark.parametrize("transform", [simplify, optimize])
def test_equivalent_values(expr, transform):
    env = {x: 0.3, y: -1.7}
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
