import sys

import pytest

from symbolcat import Constant, Function, Symbol, real
from symbolcat import operators as op
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import (
    Algebra,
    Call,
    Done,
    Literal,
    Operator,
    ana,
    args_of,
    cata,
    children,
    fmap,
    hylo,
    name_of,
)
from symbolcat.ops.printing import to_latex

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
    assert x + y == x + y
    assert x + y != Symbol("x") + Symbol("y")
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


def test_names_from_class_body():
    class Model:
        x = Symbol()
        k = Constant()
        f = Function(arity=1)
        g = Function("gee")
        y = f(x) + k  # built before the names are assigned

    assert (name_of(Model.x), name_of(Model.k), name_of(Model.f)) == ("x", "k", "f")
    assert name_of(Model.g) == "gee"  # explicit names are kept
    assert str(Model.y) == "f(x) + k"


@pytest.mark.parametrize(
    "use",
    [
        lambda: str(Symbol()),
        lambda: str(Function(arity=1)(Symbol("x"))),
        lambda: to_latex(Constant()),
        lambda: evaluate(Symbol(), {}),
    ],
)
def test_unnamed_objects_fail_clearly(use):
    with pytest.raises(ValueError, match="has no name"):
        use()


def test_unnamed_objects_have_a_repr():
    assert repr(Symbol()) == "Symbol()"
    assert repr(Function()(Symbol("x"))) == "Call(<unnamed>, (Symbol('x'),))"


def test_functions_compare_by_identity():
    assert Function("f") != Function("f")
    f = Function("f")
    assert f(x) == f(x)


def test_arity_is_checked():
    with pytest.raises(TypeError):
        real.sin(x, y)  # type: ignore  # invalid on purpose


def test_fmap_replaces_children():
    layer = fmap(x + y, lambda child: name_of(child))
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


def halves(n):
    return Literal(n) if n <= 1 else Call(op.add, (n // 2, n - n // 2))


def test_ana_shares_equal_seeds():
    calls = []

    def coalgebra(n):
        calls.append(n)
        return halves(n)

    expr = ana(coalgebra, 5, key=lambda n: n)
    assert isinstance(expr, Call)
    two, three = args_of(expr)
    assert isinstance(three, Call)
    assert args_of(three)[1] is two
    assert sorted(calls) == [1, 2, 3, 5]


def test_hylo_done_stops_unfolding():
    def coalgebra(n):
        if n == 3:
            return Done(100)  # in place of a layer
        return Call(op.add, (n - 1, Done(n)))  # among the children

    assert hylo(lambda layer: sum(args_of(layer)), coalgebra, 5, key=lambda n: n) == 109


def test_ana_handles_deep_trees():
    def coalgebra(n):
        return Literal(0) if n == 0 else Call(op.add, (n - 1, Done(Literal(1))))

    n = 10 * sys.getrecursionlimit()
    expr = ana(coalgebra, n, key=lambda n: n)
    assert cata(lambda node: 1 + max(children(node), default=0), expr) == n + 1


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
