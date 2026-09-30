import pytest

from symbolcat import Symbol, linalg, real
from symbolcat.kinds import Matrix, Vector
from symbolcat.linalg import (
    cross,
    det,
    identity,
    inv,
    matrix,
    norm,
    to_list,
    trace,
    transpose,
    vector,
    zeros,
)
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Literal, as_expr
from symbolcat.ops.calculus import diff
from symbolcat.ops.inspect import shape
from symbolcat.ops.printing import to_latex, to_mathml
from symbolcat.ops.transform import expand_arrays, simplify_naive

x, y = Symbol("x"), Symbol("y")
A, B = Symbol("A", kind=Matrix(2, 2)), Symbol("B", kind=Matrix(2, 2))
A3, B3 = Symbol("A", kind=Matrix(3, 3)), Symbol("B", kind=Matrix(3, 3))
v = vector([x, y])
m = matrix([[x, 1], [0, y]])
rect = matrix([[1, 2, 3], [4, 5, 6]])


def test_constructors():
    assert identity(2) == matrix([[1, 0], [0, 1]])
    assert zeros(2) == vector([0, 0])
    assert zeros((1, 2)) == matrix([[0, 0]])


# shape


@pytest.mark.parametrize(
    "expr, expected",
    [
        (x, ()),
        (v, (2,)),
        (2 * m + 1, (2, 2)),
        (real.sin(v), (2,)),
        (m @ v, (2,)),
        (v @ m, (2,)),
        (v @ v, ()),
        (rect @ transpose(rect), (2, 2)),
        (transpose(rect), (3, 2)),
        (m[0], (2,)),
        (m[0][1], ()),
        (det(m) + trace(m) + norm(v) + linalg.sum(m), ()),
        (inv(m), (2, 2)),
        (cross(vector([x, y, 1]), vector([1, 2, 3])), (3,)),
    ],
)
def test_shape(expr, expected):
    assert shape(expr) == expected


def test_shape_of_symbols():
    P, Q = Symbol("P", kind=Matrix(3, 2)), Symbol("Q", kind=Matrix(2, 4))
    assert shape(P @ Q) == (3, 4)
    assert shape(transpose(P) * 2) == (2, 3)
    assert shape(Symbol("u", kind=Vector(3)) @ P) == (2,)


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: v + vector([x, y, 1]), "add of incompatible shapes"),
        (lambda: m * v, "mul of incompatible shapes"),  # type: ignore  # invalid on purpose
        (lambda: x @ v, "matmul needs arrays"),  # type: ignore  # invalid on purpose
        (lambda: rect @ rect, "matmul of incompatible shapes"),
        (lambda: det(rect), "det needs a square matrix"),
        (lambda: det(v), "det needs a square matrix"),  # type: ignore  # invalid on purpose
        (lambda: trace(x), "trace needs a square matrix"),  # type: ignore  # invalid on purpose
        (lambda: norm(x), "norm needs a vector or a matrix"),  # type: ignore  # invalid on purpose
        (lambda: transpose(x), "transpose needs a vector or a matrix"),  # type: ignore  # invalid on purpose
        (lambda: cross(v, v), "cross needs two vectors of length 3"),
        (lambda: vector([v, x]), "array entries must be scalars"),
        (lambda: real.atan2(v, 1) + real.hypot(m, 1), "incompatible shapes"),  # type: ignore  # invalid on purpose
        (lambda: m[v], "must be a scalar"),  # type: ignore  # invalid on purpose
        (lambda: A + A3, "add of incompatible shapes"),
    ],
)
def test_shape_errors_when_built(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_functions_take_scalars_by_default():
    from symbolcat.lib import Function

    f = Function("f", arity=1)
    with pytest.raises(ValueError, match="f needs scalar arguments"):
        f(v)


# expand_arrays


@pytest.mark.parametrize(
    "expr, expected",
    [
        (2 * v, vector([2 * x, 2 * y])),
        (v * v, vector([x * x, y * y])),
        (-real.exp(v), vector([-real.exp(x), -real.exp(y)])),
        (v @ v, x * x + y * y),
        (m @ v, vector([x * x + 1 * y, 0 * x + y * y])),
        (v @ m, vector([x * x + y * 0, x * 1 + y * y])),
        (m[1], vector([0, y])),
        (m[1][1], y),
        (transpose(m), matrix([[x, 0], [1, y]])),
        (transpose(v), v),
        (trace(m), x + y),
        (linalg.sum(v), x + y),
        (norm(v), real.sqrt(x**2 + y**2)),
        (det(m), x * y - 1 * Literal(0)),
        (det(matrix([[x]])), x),
    ],
)
def test_expand(expr, expected):
    assert expand_arrays(expr) == expected


def test_expand_keeps_arrays_with_unknown_entries():
    assert expand_arrays(A @ m) == A @ m
    assert expand_arrays(det(A) * 2) == det(A) * 2
    # A scalar symbol is broadcast
    a = Symbol("a")
    assert expand_arrays(a * v) == vector([a * x, a * y])


def test_expand_det_and_inv():
    n = matrix([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
    assert evaluate(det(n)) == 18
    product = evaluate(inv(n) @ n)
    assert [e for row in product for e in row] == pytest.approx(
        [1, 0, 0, 0, 1, 0, 0, 0, 1]
    )
    assert evaluate(inv(matrix([[4]]))) == [[0.25]]


def test_expand_cross():
    e0, e1 = vector([1, 0, 0]), vector([0, 1, 0])
    assert evaluate(cross(e0, e1)) == [0, 0, 1]


# evaluate


BACKENDS = ["std", "numpy", "sympy"]


def lib(name):
    pytest.importorskip(name if name != "std" else "math")
    module = __import__(f"symbolcat.lib.{name}", fromlist=["IMPL"])
    return module.IMPL


def as_lists(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "expr, expected",
    [
        (m * 2, [[6, 2], [0, 4]]),
        (m * m, [[9, 1], [0, 4]]),
        (m @ m, [[9, 5], [0, 4]]),
        (m @ v, [11, 4]),
        (v @ v, 13),
        (det(m) + trace(m), 11),
        (transpose(m) - m, [[0, -1], [1, 0]]),
    ],
)
def test_evaluate_is_backend_independent(backend, expr, expected):
    result = as_lists(evaluate(expr, {x: 3, y: 2}, lib=lib(backend)))
    vector_result = isinstance(expected, list) and not isinstance(expected[0], list)
    if backend == "sympy" and vector_result:  # vectors are sympy columns
        expected = [[e] for e in expected]
    assert result == expected


def test_evaluate_numpy_values():
    np = pytest.importorskip("numpy")
    from symbolcat.lib.numpy import IMPL

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    env = {A: a, x: 2.0, y: 0.0}
    assert evaluate(A * m, env, lib=IMPL) == pytest.approx(a * [[2, 1], [0, 0]])
    assert evaluate(A @ A, env, lib=IMPL) == pytest.approx(a @ a)
    assert evaluate(det(A), env, lib=IMPL) == pytest.approx(-2.0)
    assert evaluate(inv(A) @ A, env, lib=IMPL) == pytest.approx(np.eye(2))


def test_evaluate_sympy_values_are_entrywise():
    sympy = pytest.importorskip("sympy")
    from symbolcat.lib.sympy import IMPL

    a = sympy.Matrix([[1, 2], [3, 4]])
    assert evaluate(A * A, {A: a}, lib=IMPL) == sympy.Matrix([[1, 4], [9, 16]])
    assert evaluate(A @ A, {A: a}, lib=IMPL) == a @ a
    assert evaluate(real.exp(A), {A: a}, lib=IMPL) == a.applyfunc(sympy.exp)


# diff


def test_diff_of_arrays_is_entrywise():
    assert diff(m @ v, x) == matrix([[1, 0], [0, 0]]) @ v + m @ vector([1, 0])
    assert diff(A @ A, x) == Literal(0)
    assert diff(det(A) + trace(transpose(A)), x) == Literal(0)


@pytest.mark.parametrize(
    "func",
    [
        lambda M: det(M),
        lambda M: trace(inv(M) @ M @ M),
        lambda M: norm(M),
        lambda M: linalg.sum(transpose(M) * M),
        lambda M: norm(cross(M[0], M[1])),
    ],
)
def test_diff_matches_finite_differences(func):
    """Differentiate a matrix symbol A + x B, and evaluate with NumPy."""
    np = pytest.importorskip("numpy")
    from symbolcat.lib.numpy import IMPL

    rng = np.random.default_rng(0)
    env = {A3: rng.normal(size=(3, 3)), B3: rng.normal(size=(3, 3))}
    expr = func(A3 + x * B3)
    x0, h = 0.3, 1e-6

    def f(value):
        return evaluate(expr, {**env, x: value}, lib=IMPL)

    numeric = (f(x0 + h) - f(x0 - h)) / (2 * h)
    symbolic = evaluate(diff(expr, x), {**env, x: x0}, lib=IMPL)
    assert symbolic == pytest.approx(numeric, rel=1e-5)


def test_derivative_of_a_constant_array_is_zero():
    u = Symbol("u", kind=Vector(3))
    assert diff(transpose(A) @ A + inv(A) + A * x, x) == A
    assert diff(linalg.sum(A) + trace(A @ B) + norm(cross(u, u)), x) == Literal(0)
    assert simplify_naive(Literal(0)[x] + A[0]) == A[0]


# printing

V = r"\begin{pmatrix} x \\ y \end{pmatrix}"
u, w = Symbol("u", kind=Vector(3)), Symbol("w", kind=Vector(3))


@pytest.mark.parametrize(
    "expr, expected",
    [
        (v * v, rf"{V} \odot {V}"),
        (2 * v, rf"2 \cdot {V}"),
        (A * B, r"A \odot B"),
        (2 * A, r"2 \cdot A"),
        (A @ B, r"A \cdot B"),
        (transpose(A + B), r"\left(A + B\right)^{\mathsf{T}}"),
        (inv(A), "A^{-1}"),
        (det(A), r"\det\left(A\right)"),
        (trace(A), r"\operatorname{tr}\left(A\right)"),
        (norm(A), r"\left\|A\right\|"),
        (cross(u, w + u), r"u \times \left(w + u\right)"),
    ],
)  # fmt: skip
def test_latex(expr, expected):
    r"""The entrywise product of arrays is written ``\odot``, also for symbols
    of an array kind."""
    assert to_latex(expr) == expected


def test_mathml():
    assert "<mo>⊙</mo>" in to_mathml(v * v)
    assert "<mo>⋅</mo>" in to_mathml(v @ v)
    assert '<mi mathvariant="sans-serif">T</mi>' in to_mathml(transpose(A))


def test_to_list():
    assert to_list(m) == [[x, Literal(1)], [Literal(0), y]]
    with pytest.raises(TypeError, match="not an array"):
        to_list(A + B)  # pyrefly: ignore[bad-argument-type]


# to_python


def test_to_python_writes_arrays_as_lists():
    import math

    from symbolcat.ops.printing import to_python

    expr = m @ v + 1
    source = to_python(expr)
    namespace = dict(vars(math))
    exec(source, namespace)  # noqa: S102
    assert namespace["f"](3, 2) == evaluate(expr, {x: 3, y: 2})


def test_to_python_shares_the_determinant_of_inv():
    from symbolcat.ops.printing import to_python

    source = to_python(inv(m))
    assert source.count("x * y") == 1


def test_to_python_with_numpy():
    np = pytest.importorskip("numpy")
    from symbolcat.ops.printing import to_python

    source = to_python(det(A) * (A @ m), array="array")
    assert "array([[x, 1], [0, y]])" in source
    namespace = {"array": np.array, "det": np.linalg.det}
    exec(source, namespace)  # noqa: S102
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.linalg.det(a) * (a @ np.array([[3, 1], [0, 2]]))
    assert namespace["f"](a, 3, 2) == pytest.approx(expected)


# series


def test_series_of_an_array():
    from symbolcat.ops.calculus import series

    assert series(vector([real.sin(x), x]) * 2, x, order=4) == vector(
        [2 * x - x**3 / 3, 2 * x]
    )


def test_series_of_matrix_products():
    np = pytest.importorskip("numpy")
    from symbolcat.lib.numpy import IMPL
    from symbolcat.ops.calculus import series

    expr = transpose(A + x * B) @ (A + x * B)
    polynomial = series(expr, x, order=3)
    rng = np.random.default_rng(1)
    env = {A: rng.normal(size=(2, 2)), B: rng.normal(size=(2, 2)), x: 0.7}
    # the product is quadratic in x, so the series is exact
    assert evaluate(polynomial, env, lib=IMPL) == pytest.approx(
        evaluate(expr, env, lib=IMPL)
    )


def test_series_of_det_needs_entries():
    from symbolcat.ops.calculus import series

    with pytest.raises(NotImplementedError, match="series of det"):
        series(det(A + x * B), x)
    assert series(det(A) * x, x) == det(A) * x


# dimensions


def test_dimensions_of_entries():
    from symbolcat.ops.inspect import dimensions

    t = Symbol("t")
    units = {x: {"m": 1}, t: {"s": 1}}
    state = vector([x, x / t])
    assert dimensions(state, units) == [{"m": 1}, {"m": 1, "s": -1}]
    assert dimensions(matrix([[x, t], [2 * x, t]]) @ vector([t, x]), units) == [
        {"m": 1, "s": 1},
        {"m": 1, "s": 1},
    ]
    with pytest.raises(ValueError, match="add of incompatible dimensions"):
        dimensions(vector([x, t]) @ vector([x, x]), units)


def test_dimensions_of_matrix_functions():
    from symbolcat.ops.inspect import dimensions

    units = {A3: {"m": 1}, x: {"m": 1}}
    assert dimensions(det(A3), units) == {"m": 3}
    assert dimensions(inv(A3) @ A3, units) == {}
    assert dimensions(trace(transpose(A3)) + x, units) == {"m": 1}
    assert dimensions(A3 @ vector([x, x, x]), units) == {"m": 2}
    u = Symbol("u", kind=Vector(3))
    with pytest.raises(ValueError, match="vector of incompatible dimensions"):
        dimensions(u + vector([x, x, 1]), {u: {"m": 1}, x: {"m": 1}})


# e-graph


def test_egraph_simplifies_each_entry():
    egraph = pytest.importorskip("symbolcat.ops.egraph")
    assert egraph.simplify(vector([x + x, x * y - y * x])) == vector([2 * x, 0])
    assert egraph.simplify(det(m)) == x * y
    assert egraph.simplify(inv(matrix([[x, 0], [0, y]]))) == matrix(
        [[1 / x, 0], [0, 1 / y]]
    )


@pytest.mark.parametrize(
    "expr, expected",
    [
        (transpose(transpose(A)), A),
        (inv(inv(A)) - A, 0),
        (trace(A @ B) - trace(B @ A), 0),
        (det(inv(A)) * det(A), 1),
        (det(transpose(A)) - det(A), 0),
        (transpose(transpose(A) @ B) - transpose(B) @ A, 0),
        (A @ B - A @ B, 0),
    ],
)
def test_egraph_matrix_identities(expr, expected):
    egraph = pytest.importorskip("symbolcat.ops.egraph")
    assert egraph.simplify(expr) == as_expr(expected)


def test_egraph_keeps_matrix_products_in_order():
    egraph = pytest.importorskip("symbolcat.ops.egraph")
    assert egraph.simplify(A @ B - B @ A) != Literal(0)


def test_egraph_does_not_transpose_scalars():
    egraph = pytest.importorskip("symbolcat.ops.egraph")
    u, w = Symbol("u", kind=Vector(3)), Symbol("w", kind=Vector(3))
    result = egraph.simplify(transpose(u) @ transpose(w))
    assert shape(result) == ()


def test_egraph_keeps_values_of_matrix_expressions():
    np = pytest.importorskip("numpy")
    egraph = pytest.importorskip("symbolcat.ops.egraph")
    from symbolcat.lib.numpy import IMPL

    rng = np.random.default_rng(2)
    env = {A3: rng.normal(size=(3, 3)), B3: rng.normal(size=(3, 3)), x: 0.4}
    for expr in [
        transpose(A3 @ B3) + transpose(B3) * x,
        inv(transpose(A3)) @ (A3 @ B3) - x * B3,
        det(inv(A3) @ B3) * trace(A3 @ B3 @ A3),
    ]:
        for simplified in [egraph.simplify(expr), egraph.optimize(expr)]:
            assert evaluate(simplified, env, lib=IMPL) == pytest.approx(
                evaluate(expr, env, lib=IMPL)
            )
