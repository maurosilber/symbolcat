"""Static types of expressions, checked by the type checkers (``pixi run ty
check``, ``pyrefly check`` and ``zuban check``), not by pytest: ``assert_type``
does nothing at runtime.

``invalid`` is never called. Each of its lines must be a static error: ty
reports an unused ``type: ignore`` in this file (see pyproject.toml).
"""

from typing import Any, assert_type

from symbolcat import D, Symbol, complex, delay, linalg, real
from symbolcat.kinds import Boolean, Complex, Matrix, Real, Vector
from symbolcat.lib import Constant, Function
from symbolcat.ops.algebras import Call, Expr, Literal, args_of, kind_algebra

x = Symbol("x")
z = Symbol("z", kind=Complex())
v = Symbol("v", kind=Vector(3))
A = Symbol("A", kind=Matrix(2, 3))
I = complex.constants.I


def test_leaves() -> None:
    assert_type(x, Symbol[Real])
    assert_type(v, Symbol[Vector])
    assert_type(v.kind, Vector)
    assert_type(A.kind.rows, int)
    assert_type(I, Constant[Complex])
    assert_type(Literal(True), Literal[Boolean])
    assert_type(Literal(2), Literal[Real])
    assert_type(Literal(1j), Literal[Complex])


def test_operators() -> None:
    assert_type(x + 1, Call[Expr, Real])
    assert_type(2 * x, Call[Expr, Real])
    assert_type(x * z, Call[Expr, Complex])
    assert_type(1j * x, Call[Expr, Complex])
    assert_type(x < 1, Call[Expr, Boolean])
    assert_type((x < 1) & (x > 0), Call[Expr, Boolean])
    assert_type(~(x < 1), Call[Expr, Boolean])
    assert_type((x < 1) + 1, Call[Expr, Real])
    assert_type(x // 2, Call[Expr, Real])
    assert_type(-z, Call[Expr, Complex])
    assert_type(x * v, Call[Expr, Vector])
    assert_type(v / 2, Call[Expr, Vector])
    assert_type(v < x, Call[Expr, Vector])
    assert_type(A @ v, Call[Expr, Vector])
    assert_type(v @ v, Call[Expr, Real])
    assert_type(A[0], Call[Expr, Vector])
    assert_type(A[0][1], Call[Expr, Real])


def test_functions() -> None:
    assert_type(real.sin(x), Call[Expr, Real])
    assert_type(real.sin(2), Call[Expr, Real])
    assert_type(real.exp(v), Call[Expr, Vector])
    assert_type(real.atan2(x, 1), Call[Expr, Real])
    assert_type(real.hypot(v, x), Call[Expr, Vector])
    assert_type(real.ifelse(x < 0, -x, x), Call[Expr, Real])
    assert_type(real.ifelse(x < 0, z, 1), Call[Expr, Complex])
    assert_type(complex.exp(x), Call[Expr, Complex])
    assert_type(complex.abs(z), Call[Expr, Real])
    assert_type(complex.conj(x), Call[Expr, Real])
    assert_type(complex.conj(z), Call[Expr, Complex])
    assert_type(complex.rect(x, 1), Call[Expr, Complex])
    assert_type(D(v), Call[Expr, Vector])
    assert_type(delay(x, 2), Call[Expr, Real])


def test_linalg() -> None:
    m = linalg.matrix([[x, 1], [0, x]])
    assert_type(m, Call[Expr, Matrix])
    assert_type(linalg.vector([x, 1]), Call[Expr, Vector])
    assert_type(linalg.zeros((2, 2)), Call[Expr, Matrix])
    assert_type(linalg.transpose(A), Call[Expr, Matrix])
    assert_type(linalg.det(m) + linalg.trace(m), Call[Expr, Real])
    assert_type(linalg.inv(m), Call[Expr, Matrix])
    assert_type(linalg.norm(v), Call[Expr, Real])
    assert_type(linalg.cross(v, v), Call[Expr, Vector])


def test_user_functions() -> None:
    f = Function("f", arity=1)
    assert_type(f(x), Call[Expr, Any])  # unchecked, unless annotated
    outer: Function[[Expr[Vector], Expr[Vector]], Matrix] = Function("outer")
    kind_algebra.register(outer)(
        lambda node: Matrix(args_of(node)[0].n, args_of(node)[1].n)
    )
    assert_type(outer(v, v), Call[Expr, Matrix])
    assert outer(v, v).kind == Matrix(3, 3)


def invalid() -> None:
    _ = real.sin(z)  # type: ignore
    _ = real.log(I)  # type: ignore
    _ = z < 1  # type: ignore
    _ = z // 2  # type: ignore
    _ = x @ v  # type: ignore
    _ = 2 @ v  # type: ignore
    _ = v[v]  # type: ignore
    _ = A < z  # type: ignore
    _ = complex.rect(z, 1)  # type: ignore
    _ = real.ifelse(z, 1, 2)  # type: ignore
    _ = linalg.det(x)  # type: ignore
    _ = linalg.transpose(x)  # type: ignore
    _ = linalg.norm(x)  # type: ignore
    _ = delay(x, v)  # type: ignore
    _ = A + v * z  # type: ignore
    outer: Function[[Expr[Vector], Expr[Vector]], Matrix] = Function("outer")
    _ = outer(x, v)  # type: ignore
