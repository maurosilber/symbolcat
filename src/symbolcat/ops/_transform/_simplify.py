"""
symbolcat.ops._transform._simplify
~~~~~~~~~~~~~~~~~~~~~~~

Constant folding and algebraic identities (``x + 0``, ``x * 1``, ...).

Children arrive already simplified, so each rule only inspects one layer
and a single bottom-up pass is enough. Because it works on a single layer,
``simplify_algebra`` can also be used as a "smart constructor":
``simplify_algebra(x * 1)`` returns ``x``.

Simplification keeps the kind of every node: a rule may return a scalar for
an array (``A * 0 -> 0``), which is then broadcast to its shape, so the
parents stay valid (``sum(A * 0)``).
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Any

from ... import complex as cx
from ... import linalg, real
from ..._core import Algebra, Call, Expr, Function, Literal, cata
from ...lib import operators as op
from ...lib.std import IMPL
from ...linalg import Array
from .._analysis._inspect import is_complex


class KindPreserving(Algebra[Expr]):
    """An algebra into expressions, whose results keep the kind of the
    node: a scalar result for an array is broadcast to its shape."""

    def __call__(self, node: Expr, **ctx: Any) -> Expr:
        result = super().__call__(node, **ctx)
        shape = node.kind.shape
        if result.kind.shape == shape:
            return result
        if result.kind.shape:
            raise AssertionError(f"{self.name}: {node} changed shape to {result}")
        return linalg.full(shape, result)


simplify_algebra = KindPreserving("simplify")


def is_number(expr: Expr, value: Any, /) -> bool:
    return (
        isinstance(expr, Literal)
        and isinstance(expr.value, int | float | complex)
        and not isinstance(expr.value, bool)
        and expr.value == value
    )


def is_zero(expr: Expr, /) -> bool:
    """A literal 0, or an explicit array of them."""
    if isinstance(expr, Call) and isinstance(expr.func, Array):
        return all(is_number(entry, 0) for entry in expr.args)
    return is_number(expr, 0)


def as_fraction(expr: Expr, /) -> Fraction | None:
    """An integer literal, or a quotient of them, as a ``Fraction``."""
    if isinstance(expr, Literal):
        return Fraction(expr.value) if type(expr.value) is int else None
    if isinstance(expr, Call) and expr.func is op.neg:
        (arg,) = expr.args
        value = as_fraction(arg)
        return None if value is None else -value
    if isinstance(expr, Call) and expr.func is op.truediv:
        num, den = expr.args
        if (
            isinstance(num, Literal)
            and isinstance(den, Literal)
            and type(num.value) is int
            and type(den.value) is int
            and den.value
        ):
            return Fraction(num.value, den.value)
    return None


def from_fraction(value: Fraction, /) -> Expr:
    """An integer literal, or a quotient of them in lowest terms."""
    if value.denominator == 1:
        return Literal(value.numerator)
    return op.truediv(value.numerator, value.denominator)


RATIONAL: dict[Function, Callable[..., Fraction | None]] = {
    op.add: lambda a, b: a + b,
    op.sub: lambda a, b: a - b,
    op.mul: lambda a, b: a * b,
    op.truediv: lambda a, b: a / b if b else None,
    op.pow: lambda a, b: a**b if b.denominator == 1 and (a or b >= 0) else None,
    op.neg: lambda a: -a,
}


def fold_rational(node: Call[Expr], /) -> Expr | None:
    """Exact arithmetic on rationals: 1 / 2 + 1 / 3 -> 5 / 6."""
    if node.func not in RATIONAL:
        return None
    values = [as_fraction(arg) for arg in node.args]
    if any(v is None for v in values):
        return None
    result = RATIONAL[node.func](*values)
    return None if result is None else from_fraction(result)


@simplify_algebra.register(Expr)
def fold(node: Expr, /, *, exact: bool = False) -> Expr:
    """Evaluate calls whose arguments are all numeric literals.

    With ``exact``, rationals are folded exactly (``1 / 2 + 1 / 3 -> 5 / 6``),
    and functions only into integer results: ``cos(0) -> 1``, but ``sin(1)``
    is kept.
    """
    if exact and isinstance(node, Call) and (rational := fold_rational(node)):
        return rational
    if not isinstance(node, Call) or node.func not in IMPL:
        return node
    if not all(isinstance(arg, Literal) for arg in node.args):
        return node
    func = IMPL[node.func]
    if not callable(func):
        return node
    values = [arg.value for arg in node.args]
    if exact and not all(type(v) is int for v in values):
        return node
    try:
        result = func(*values)
    except (ArithmeticError, ValueError, TypeError):
        return node
    real_arguments = not any(isinstance(v, complex) for v in values)
    if isinstance(result, complex) and not result.imag and real_arguments:
        result = result.real  # cmath.exp(0) is 1
    if exact:
        if isinstance(result, float) and result.is_integer() and abs(result) < 2**53:
            result = int(result)
        if type(result) is not int:
            return node
    return Literal(result)


@simplify_algebra.register(op.add)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    a, b = node.args
    if is_zero(a):
        return b
    if is_zero(b):
        return a
    return fold(node, exact=exact)


@simplify_algebra.register(op.sub)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    a, b = node.args
    if is_zero(b):
        return a
    if is_zero(a):
        return simplify_algebra(-b, exact=exact)
    if a == b:
        return Literal(0)
    return fold(node, exact=exact)


@simplify_algebra.register(op.mul)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    a, b = node.args
    if is_zero(a) or is_zero(b):
        return Literal(0)
    if is_number(a, 1):
        return b
    if is_number(b, 1):
        return a
    if is_number(a, -1):
        return simplify_algebra(-b, exact=exact)
    if a is cx.constants.I and b is cx.constants.I:
        return Literal(-1)
    return fold(node, exact=exact)


@simplify_algebra.register(op.truediv)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    a, b = node.args
    if is_number(b, 1):
        return a
    if is_zero(a) and not is_zero(b):
        return Literal(0)
    if a == b and not isinstance(a, Literal):
        return Literal(1)
    return fold(node, exact=exact)


@simplify_algebra.register(op.pow)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    a, b = node.args
    if is_number(b, 0):
        return Literal(1)
    if is_number(b, 1) or is_number(a, 1):
        return a
    return fold(node, exact=exact)


@simplify_algebra.register(op.neg)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    (a,) = node.args
    if isinstance(a, Call) and a.func is op.neg:
        return a.args[0]
    return fold(node, exact=exact)


@simplify_algebra.register(real.ifelse)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    condition, a, b = node.args
    if isinstance(condition, Literal) and type(condition.value) is bool:
        return a if condition.value else b
    if a == b:
        return a
    return node


@simplify_algebra.register(cx.re, cx.conj)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    """Symbols are real, so ``re(x)`` and ``conj(x)`` are ``x``."""
    (z,) = node.args
    if node.func is cx.conj and isinstance(z, Call) and z.func is cx.conj:
        return z.args[0]
    if not is_complex(z):
        return z
    return fold(node, exact=exact)


@simplify_algebra.register(cx.im)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    (z,) = node.args
    if not is_complex(z):
        return Literal(0)
    return fold(node, exact=exact)


@simplify_algebra.register(op.pos)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    return node.args[0]


@simplify_algebra.register(
    op.getitem, op.matmul, linalg.transpose, linalg.trace, linalg.sum, linalg.cross
)
def _(node: Call[Expr], *, exact: bool = False) -> Expr:
    """Linear in the first argument (and in the second for the products):
    zero for a zero argument."""
    linear = node.args[:1] if node.func is op.getitem else node.args
    return Literal(0) if any(is_zero(arg) for arg in linear) else node


def simplify_naive(expr: Any, /) -> Expr:
    """Fold constants and remove trivial operations.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> print(simplify_naive(0 * x + 1 * (x ** (3 - 2)) + 2 * 3))
    x + 6
    """
    return cata(simplify_algebra, expr)


def simplify_exact(expr: Any, /) -> Expr:
    """Like ``simplify_naive``, but keeps results exact: no floats.

    >>> from symbolcat import Symbol, real
    >>> from symbolcat.ops.algebras import Literal
    >>> x = Symbol("x")
    >>> print(simplify_exact(Literal(2) * 3 * x + Literal(1) / 2 + Literal(1) / 3))
    6 * x + 1 / 2 + 1 / 3
    >>> print(simplify_exact(Literal(1) / 2 + Literal(1) / 3 + real.cos(Literal(0))))
    11 / 6
    """
    return cata(simplify_algebra, expr, exact=True)
