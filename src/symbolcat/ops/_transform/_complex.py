"""
symbolcat.ops._transform._complex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real and imaginary parts of complex expressions.

Every node folds into the pair ``(re, im)`` of real expressions, like
``diff`` folds into ``(value, derivative)``: ``(a + b i)(c + d i)`` is
``(a c - b d) + (a d + b c) i``, ``exp(a + b i)`` is ``exp(a) (cos(b) +
i sin(b))``, and so on, with the principal branches of ``cmath``. Symbols
are real, so their imaginary part is 0.

With both parts real, the rules for real expressions apply to them: the
e-graph simplifies complex expressions this way.

On a branch cut, ``cmath`` picks the side from the sign of a zero
imaginary part (``asin(2 + 0j)`` and ``asin(2 - 0j)`` differ), which a real
symbol does not have: there, the parts of ``asin(x + I * y)`` at ``y = 0``
can be the conjugate of ``cmath``'s value. For real arguments, like
``asin(x)``, they agree.
"""

from __future__ import annotations

from typing import Any

from ... import complex as cx
from ... import real
from ..._core import Algebra, Call, Constant, Expr, Function, Literal, Symbol, cata
from ...lib import operators as op
from ...linalg import Array
from .._analysis._inspect import is_complex
from ._simplify import is_number, simplify_algebra

type Pair = tuple[Expr, Expr]

ZERO, ONE = Literal(0), Literal(1)
PI = real.constants.pi

complex_parts_algebra = Algebra[Pair]("complex_parts")


def smart(expr: Expr, /) -> Expr:
    """Simplify one layer, exactly, so zeros vanish as the pairs are built."""
    return simplify_algebra(expr, exact=True)


def call(func: Function, /, *args: Expr) -> Expr:
    return smart(func(*args))


def add(z: Pair, w: Pair, /) -> Pair:
    return smart(z[0] + w[0]), smart(z[1] + w[1])


def neg(z: Pair, /) -> Pair:
    return smart(-z[0]), smart(-z[1])


def mul(z: Pair, w: Pair, /) -> Pair:
    (a, b), (c, d) = z, w
    return (
        smart(smart(a * c) - smart(b * d)),
        smart(smart(a * d) + smart(b * c)),
    )


def div(z: Pair, w: Pair, /) -> Pair:
    (a, b), (c, d) = z, w
    if is_real(w):
        return smart(a / c), smart(b / c)
    denominator = smart(c**2 + d**2)
    return (
        smart(smart(smart(a * c) + smart(b * d)) / denominator),
        smart(smart(smart(b * c) - smart(a * d)) / denominator),
    )


def is_real(z: Pair, /) -> bool:
    return is_number(z[1], 0)


def modulus(z: Pair, /) -> Expr:
    a, b = z
    return real.abs(a) if is_number(b, 0) else real.sqrt(a**2 + b**2)


def exp(z: Pair, /) -> Pair:
    a, b = z
    if is_real(z):
        return call(real.exp, a), ZERO
    r = call(real.exp, a)
    return smart(r * call(real.cos, b)), smart(r * call(real.sin, b))


def log(z: Pair, /) -> Pair:
    """``log|z| + i arg(z)``: for a negative real, ``log(-a) + i pi``."""
    a, b = z
    if is_real(z):
        return real.log(real.abs(a)), real.ifelse(a < 0, PI, 0)
    return real.log(a**2 + b**2) / 2, real.atan2(b, a)


def sqrt(z: Pair, /) -> Pair:
    """The principal root, with a nonnegative real part."""
    a, b = z
    if is_real(z):
        return real.ifelse(a < 0, 0, real.sqrt(a)), real.ifelse(a < 0, real.sqrt(-a), 0)
    r = modulus(z)
    sign = real.ifelse(b < 0, -1, 1)
    return real.sqrt((r + a) / 2), sign * real.sqrt((r - a) / 2)


def sin(z: Pair, /) -> Pair:
    a, b = z
    if is_real(z):
        return call(real.sin, a), ZERO
    return (
        smart(call(real.sin, a) * call(real.cosh, b)),
        smart(call(real.cos, a) * call(real.sinh, b)),
    )


def cos(z: Pair, /) -> Pair:
    a, b = z
    if is_real(z):
        return call(real.cos, a), ZERO
    return (
        smart(call(real.cos, a) * call(real.cosh, b)),
        smart(-smart(call(real.sin, a) * call(real.sinh, b))),
    )


def sinh(z: Pair, /) -> Pair:
    a, b = z
    if is_real(z):
        return call(real.sinh, a), ZERO
    return (
        smart(call(real.sinh, a) * call(real.cos, b)),
        smart(call(real.cosh, a) * call(real.sin, b)),
    )


def cosh(z: Pair, /) -> Pair:
    a, b = z
    if is_real(z):
        return call(real.cosh, a), ZERO
    return (
        smart(call(real.cosh, a) * call(real.cos, b)),
        smart(call(real.sinh, a) * call(real.sin, b)),
    )


I: Pair = (ZERO, ONE)


def times_i(z: Pair, /) -> Pair:
    return mul(I, z)


def asin(z: Pair, /) -> Pair:
    """``-i log(i z + sqrt(1 - z ** 2))``.

    On the branch cut, a real ``z > 1``, the formula gives the conjugate of
    ``cmath``'s value: ``cmath`` takes the side of the cut from the sign of
    the (zero) imaginary part, which a real ``z`` does not have.
    """
    root = sqrt(add((ONE, ZERO), neg(mul(z, z))))
    re, im = mul((ZERO, Literal(-1)), log(add(times_i(z), root)))
    if is_real(z):
        im = smart(real.ifelse(z[0] > 1, smart(-im), im))
    return re, im


def acos(z: Pair, /) -> Pair:
    """``pi / 2 - asin(z)``."""
    return add((PI / 2, ZERO), neg(asin(z)))


def atan(z: Pair, /) -> Pair:
    """``i / 2 (log(1 - i z) - log(1 + i z))``."""
    iz = times_i(z)
    one = (ONE, ZERO)
    difference = add(log(add(one, neg(iz))), neg(log(add(one, iz))))
    return mul((ZERO, Literal(1) / 2), difference)


def power(z: Pair, w: Pair, /) -> Pair:
    """Integer powers by repeated products; others as ``exp(w log(z))``."""
    (_, _), (c, _) = z, w
    if is_real(w) and isinstance(c, Literal) and type(c.value) is int:
        n = c.value
        if abs(n) <= 16:
            result, base = (ONE, ZERO), z
            for _ in range(abs(n)):
                result = mul(result, base)
            return result if n >= 0 else div((ONE, ZERO), result)
    return exp(mul(w, log(z)))


HOLOMORPHIC = {
    cx.exp: exp,
    cx.log: log,
    cx.log10: lambda z: div(log(z), (real.log(10), ZERO)),
    cx.sqrt: sqrt,
    cx.sin: sin,
    cx.cos: cos,
    cx.tan: lambda z: div(sin(z), cos(z)),
    cx.asin: asin,
    cx.acos: acos,
    cx.atan: atan,
    cx.sinh: sinh,
    cx.cosh: cosh,
    cx.tanh: lambda z: div(sinh(z), cosh(z)),
}


@complex_parts_algebra.register(Symbol)
def _(node: Symbol) -> Pair:
    return node, ZERO


@complex_parts_algebra.register(Constant)
def _(node: Constant) -> Pair:
    return I if node is cx.constants.I else (node, ZERO)


@complex_parts_algebra.register(Literal)
def _(node: Literal) -> Pair:
    if isinstance(node.value, complex):
        return Literal(node.value.real), Literal(node.value.imag)
    return node, ZERO


@complex_parts_algebra.register(Function)
def _(node: Call[Pair]) -> Pair:
    """A real function of real arguments is real. Otherwise its parts are
    kept as ``re`` and ``im`` of the call."""
    if all(is_real(arg) for arg in node.args):
        return node.func(*(a for a, _ in node.args)), ZERO
    value = node.func(*(join(arg) for arg in node.args))
    return cx.re(value), cx.im(value)


@complex_parts_algebra.register(op.add)
def _(node: Call[Pair]) -> Pair:
    return add(*node.args)


@complex_parts_algebra.register(op.sub)
def _(node: Call[Pair]) -> Pair:
    z, w = node.args
    return add(z, neg(w))


@complex_parts_algebra.register(op.neg)
def _(node: Call[Pair]) -> Pair:
    return neg(node.args[0])


@complex_parts_algebra.register(op.pos)
def _(node: Call[Pair]) -> Pair:
    return node.args[0]


@complex_parts_algebra.register(op.mul)
def _(node: Call[Pair]) -> Pair:
    return mul(*node.args)


@complex_parts_algebra.register(op.truediv)
def _(node: Call[Pair]) -> Pair:
    return div(*node.args)


@complex_parts_algebra.register(op.pow)
def _(node: Call[Pair]) -> Pair:
    z, w = node.args
    if is_real(z) and is_real(w):
        return z[0] ** w[0], ZERO
    return power(z, w)


@complex_parts_algebra.register(*HOLOMORPHIC)
def _(node: Call[Pair]) -> Pair:
    return HOLOMORPHIC[node.func](node.args[0])


@complex_parts_algebra.register(cx.re)
def _(node: Call[Pair]) -> Pair:
    return node.args[0][0], ZERO


@complex_parts_algebra.register(cx.im)
def _(node: Call[Pair]) -> Pair:
    return node.args[0][1], ZERO


@complex_parts_algebra.register(cx.conj)
def _(node: Call[Pair]) -> Pair:
    a, b = node.args[0]
    return a, smart(-b)


@complex_parts_algebra.register(cx.abs)
def _(node: Call[Pair]) -> Pair:
    return modulus(node.args[0]), ZERO


@complex_parts_algebra.register(cx.phase)
def _(node: Call[Pair]) -> Pair:
    a, b = node.args[0]
    return real.atan2(b, a), ZERO


@complex_parts_algebra.register(cx.rect)
def _(node: Call[Pair]) -> Pair:
    (r, _), (phi, _) = node.args
    return smart(r * real.cos(phi)), smart(r * real.sin(phi))


@complex_parts_algebra.register(real.ifelse)
def _(node: Call[Pair]) -> Pair:
    (condition, _), (a, b), (c, d) = node.args
    return real.ifelse(condition, a, c), smart(real.ifelse(condition, b, d))


@complex_parts_algebra.register(Array)
def _(node: Call[Pair]) -> Pair:
    return (
        node.func(*(a for a, _ in node.args)),
        smart(node.func(*(b for _, b in node.args))),
    )


def join(z: Pair, /) -> Expr:
    """``re + I * im``, leaving out the parts that are 0."""
    a, b = z
    if is_number(b, 0):
        return a
    imaginary = cx.constants.I if is_number(b, 1) else smart(cx.constants.I * b)
    return imaginary if is_number(a, 0) else a + imaginary


def real_imag(expr: Any, /) -> tuple[Expr, Expr]:
    """The real and imaginary parts of ``expr``, as real expressions.

    Symbols are real. The parts of functions that are not in ``complex`` or
    ``real`` are kept as ``re(f(z))`` and ``im(f(z))``.

    >>> from symbolcat import Symbol, complex
    >>> x, y = Symbol("x"), Symbol("y")
    >>> z = x + complex.constants.I * y
    >>> re, im = real_imag(z * z)
    >>> print(re, "|", im)
    x * x - y * y | x * y + y * x
    >>> for part in real_imag(complex.exp(complex.constants.I * x)):
    ...     print(part)
    cos(x)
    sin(x)
    """
    return cata(complex_parts_algebra, expr)


COMPLEX_FUNCTIONS = frozenset(f for f in vars(cx).values() if isinstance(f, Function))


def _uses_complex(node: Any, /) -> bool:
    if isinstance(node, Call):
        return any(node.args) or node.func in COMPLEX_FUNCTIONS
    return is_complex(node)


def uses_complex(expr: Any, /) -> bool:
    """Whether ``expr`` has a complex function, constant or number
    anywhere, even if its value is real (``re(z)``)."""
    return cata(_uses_complex, expr)
