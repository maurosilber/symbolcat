import pickle
from typing import Any

import pytest

from symbolcat import D, Symbol, complex, linalg, real
from symbolcat.kinds import Boolean, Complex, Kind, Matrix, Real, Vector
from symbolcat.lib import Constant, Function
from symbolcat.lib import operators as op
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Algebra, Call, Expr, Literal, fmap
from symbolcat.ops.transform import cse, simplify_naive, substitute

x, y = Symbol("x"), Symbol("y")
z = Symbol("z", kind=Complex())
v = Symbol("v", kind=Vector(3))
A = Symbol("A", kind=Matrix(2, 3))
S = Symbol("S", kind=Matrix(2, 2))
I = complex.constants.I


# leaves


def test_leaves():
    assert x.kind == Real()
    assert A.kind.shape == (2, 3)
    assert I.kind == Complex()
    assert Constant("c").kind == Real()
    assert Literal(True).kind == Boolean()
    assert Literal(2.0).kind == Real()
    assert Literal(1j).kind == Complex()


def test_symbols_compare_by_name_and_kind():
    assert Symbol("v", kind=Vector(3)) == v
    assert Symbol("v") != v
    assert Symbol("v", kind=Vector(2)) != v
    assert hash(Symbol("v", kind=Vector(3))) == hash(v)


def test_repr():
    assert repr(v) == "Symbol('v', kind=Vector(3))"
    assert repr(Symbol("w", kind=Vector(2, Complex()))) == (
        "Symbol('w', kind=Vector(2, Complex()))"
    )
    assert repr(x) == "Symbol('x')"


# inferred kinds


@pytest.mark.parametrize(
    "expr, kind",
    [
        (x + 1, Real()),
        (x < y, Boolean()),
        (op.eq(z, 1), Boolean()),
        ((x < 1) & (y > 0), Boolean()),
        (~(x < 1), Boolean()),
        (Literal(True) + True, Real()),
        (x & y, Real()),
        (x * z, Complex()),
        (-z, Complex()),
        (complex.exp(x), Complex()),
        (complex.abs(z), Real()),
        (complex.conj(x), Real()),
        (complex.conj(z), Complex()),
        (complex.rect(x, y), Complex()),
        (real.ifelse(x < 0, x, z), Complex()),
        (2 * v + 1, Vector(3)),
        (real.sin(v), Vector(3)),
        (complex.exp(I * v), Vector(3, Complex())),
        (A @ v, Vector(2)),
        (v @ v, Real()),
        (linalg.transpose(A), Matrix(3, 2)),
        (A[0], Vector(3)),
        (A[0][1], Real()),
        (linalg.det(S) + linalg.trace(S), Real()),
        (linalg.inv(S), Matrix(2, 2)),
        (linalg.norm(A), Real()),
        (linalg.cross(v, v), Vector(3)),
        (linalg.vector([x, z]), Vector(2, Complex())),
        (linalg.vector([x < 1, True]), Vector(2, Boolean())),
        (D(v), Vector(3)),
        (D(D(x)), Real()),
    ],
)
def test_inferred(expr, kind):
    assert expr.kind == kind


@pytest.mark.parametrize(
    "build, error, match",
    [
        (lambda: real.sin(z), TypeError, "sin needs real arguments"),  # type: ignore  # invalid on purpose
        (lambda: z < 1, TypeError, "lt needs real arguments"),  # type: ignore  # invalid on purpose
        (lambda: z // 2, TypeError, "floordiv needs real arguments"),  # type: ignore  # invalid on purpose
        (lambda: complex.rect(z, 1), TypeError, "rect needs real arguments"),  # type: ignore  # invalid on purpose
        (lambda: real.ifelse(z, 1, 2), TypeError, "ifelse needs real arguments"),  # type: ignore  # invalid on purpose
        (lambda: v + S[0], ValueError, "add of incompatible shapes"),
        (lambda: v[v], ValueError, "index must be a scalar"),  # type: ignore  # invalid on purpose
        (lambda: D(D(v) + S[0]), ValueError, "add of incompatible shapes"),
        (lambda: Function("f")(v), ValueError, "f needs scalar arguments"),
    ],
)
def test_invalid_calls_raise_when_built(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_new_function_kind():
    from symbolcat.ops.algebras import entrywise, kind_algebra

    f = Function("f", arity=1)
    kind_algebra.register(f)(entrywise(lambda func, arg: Complex()))
    assert f(v).kind == Vector(3, Complex())


# layers


def test_layers_keep_the_kind_of_their_node():
    layer = fmap(A @ v, lambda child: "text")
    assert layer.kind == Vector(2)


def test_rebuilt_nodes_infer_their_kind_again():
    assert fmap(x + 1, lambda child: v if child == x else child).kind == Vector(3)
    assert substitute(real.sin(x) + y, {x: v}).kind == Vector(3)
    with pytest.raises(ValueError, match="incompatible shapes"):
        substitute(x + y, {x: v, y: S[0]})


def test_unfolding_layers_have_an_unspecified_kind():
    assert Call(op.add, ("seed", "seed")).kind == Kind()


def test_pickle_keeps_kinds():
    expr = linalg.transpose(A) @ A
    loaded = pickle.loads(pickle.dumps(expr))
    assert loaded.kind == Matrix(3, 3)
    assert loaded.args[1] == A


# algebras


def test_algebra_kind_qualifier():
    describe = Algebra[str]("describe")
    describe.register(Expr)(lambda node: "leaf")
    describe.register(Expr, kind=Vector)(lambda node: "vector leaf")
    describe.register(Call)(lambda node: "call")
    describe.register(op.add)(lambda node: "add")
    describe.register(op.add, kind=Matrix)(lambda node: "matrix add")

    assert describe(x) == "leaf"
    assert describe(v) == "vector leaf"
    assert describe(x + 1) == "add"
    assert describe(S + S) == "matrix add"
    assert describe(real.sin(v)) == "call"


def test_algebra_kind_qualifier_must_be_a_kind():
    not_a_kind: Any = int
    with pytest.raises(TypeError, match="not a Kind"):
        Algebra("bad").register(Expr, kind=not_a_kind)


# operations keep kinds


def test_simplify_keeps_the_kind():
    assert simplify_naive(S * 0) == linalg.zeros((2, 2))
    assert simplify_naive(linalg.sum(S * 0) + x) == x
    assert simplify_naive(S - S + S) == S
    assert simplify_naive((A * 0) @ v) == linalg.zeros(2)


def test_cse_temporaries_keep_the_kind():
    assignments, result = cse(linalg.det(S @ S) + linalg.trace(S @ S))
    ((temporary, value),) = assignments
    assert temporary.kind == Matrix(2, 2)
    assert value == S @ S
    assert result == linalg.det(temporary) + linalg.trace(temporary)


def test_evaluate_by_name():
    np = pytest.importorskip("numpy")
    from symbolcat.lib.numpy import IMPL

    s = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert evaluate(linalg.det(S), {"S": s}, lib=IMPL) == pytest.approx(-2.0)
