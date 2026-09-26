import sys

import pytest

from symbolcat import (
    Algebra,
    Call,
    Constant,
    Function,
    Literal,
    Operator,
    Symbol,
    cata,
    children,
    fmap,
    real,
)
from symbolcat import operators as op

x, y = Symbol("x"), Symbol("y")


def test_operators_build_calls():
    assert x + 1 == Call(op.add, (x, Literal(1)))
    assert 1 + x == Call(op.add, (Literal(1), x))
    assert -x == Call(op.neg, (x,))
    assert x[0] == Call(op.getitem, (x, Literal(0)))
    assert (x < y) == Call(op.lt, (x, y))
    assert op.eq(x, y) == Call(op.eq, (x, y))
    assert op.eq(1, x) == Call(op.eq, (Literal(1), x))
    assert op.ne(x, y) == Call(op.ne, (x, y))


def test_structural_equality_and_hash():
    assert x + y == Symbol("x") + Symbol("y")
    assert x + y != y + x
    assert {x + 1: "a"}[x + 1] == "a"


def test_literal_equality_includes_type():
    assert Literal(1) != Literal(1.0)
    assert Literal(1) != Literal(True)
    assert len({Literal(1), Literal(1.0), Literal(True)}) == 3


def test_constants_compare_by_identity():
    assert real.constants.pi == real.constants.pi
    assert real.constants.pi != Constant("pi")
    assert real.constants.pi != Symbol("pi")


def test_functions_compare_by_identity():
    assert Function("f") != Function("f")
    f = Function("f")
    assert f(x) == f(x)


def test_arity_is_checked():
    with pytest.raises(TypeError):
        real.sin(x, y)


def test_fmap_replaces_children():
    layer = fmap(x + y, lambda child: child.name)
    assert layer == Call(op.add, ("x", "y"))
    assert fmap(x, str) is x


def test_children():
    assert children(x + 1) == (x, Literal(1))
    assert children(x) == ()
    assert children(Literal(1)) == ()


def test_functor_rejects_non_nodes():
    with pytest.raises(TypeError):
        children(1)
    with pytest.raises(TypeError):
        fmap(1, str)


def test_cata_passes_folded_children():
    seen = []

    def algebra(node):
        seen.append(node)
        return len(seen)

    cata(algebra, x + 1)
    # children are folded first, and replaced by their results
    assert seen[-1] == Call(op.add, (seen.index(x) + 1, seen.index(Literal(1)) + 1))


def test_cata_folds_shared_subexpressions_once():
    calls = 0

    def algebra(node):
        nonlocal calls
        calls += 1
        return 0

    shared = x + y
    cata(algebra, shared * shared)
    assert calls == 4  # x, y, shared, mul


def test_cata_forwards_context():
    def algebra(node, *, scale):
        return scale * (1 + sum(children(node)))

    assert cata(algebra, x, scale=2) == 2


def test_cata_handles_deep_trees():
    expr = x
    for _ in range(10 * sys.getrecursionlimit()):
        expr = expr + 1
    assert cata(lambda node: 1 + max(children(node), default=0), expr) > 10_000


def test_algebra_dispatch_priority():
    alg = Algebra("test")
    alg.register(Symbol, Constant, Literal)(lambda node: "leaf")
    alg.register(Call)(lambda node: "call")
    alg.register(Operator)(lambda node: "operator")
    alg.register(op.add)(lambda node: "add")

    assert cata(alg, x) == "leaf"
    assert cata(alg, real.sin(x)) == "call"
    assert cata(alg, x * y) == "operator"
    assert cata(alg, x + y) == "add"


def test_algebra_missing_rule():
    alg = Algebra("partial")
    alg.register(Symbol)(lambda node: node)
    with pytest.raises(NotImplementedError, match="partial: no rule for sin"):
        cata(alg, real.sin(x))


def test_algebra_copy_is_independent():
    base = Algebra("base")
    base.register(Symbol)(lambda node: 1)
    child = base.copy("child")
    child.register(Symbol)(lambda node: 2)
    assert cata(base, x) == 1
    assert cata(child, x) == 2
