"""
symbolcat.ops._algebraic._solve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic solution of one equation for one unknown.

Three strategies, in order:

1. Polynomials in the unknown (with any coefficients): degrees 1 and 2 by
   formula; higher degrees with rational coefficients by splitting off the
   rational roots, until what is left has degree 2 at most.
2. Rational functions: the zeros of the numerator (``cancel``) that are not
   zeros of the denominator.
3. Isolation: when the unknown appears once, invert the functions around it
   (``exp(x) = a`` gives ``x = log(a)``). Inverse trigonometric functions
   give their principal values only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ... import real
from ..._core import Call, Expr, Literal, Symbol, args_of, func_of, name_of
from ..._derivative import Variable, is_variable
from ...lib import operators as op
from .._analysis._polynomial import coefficients
from .._transform._simplify import is_number, simplify_exact
from .._transform._substitute import substitute_variables
from ._polynomial import (
    NotPolynomial,
    _univariate,
    cancel,
    polynomial_divide,
    rational_roots,
)
from ._system import residual

_SOLVE = Symbol("__solve")


def solve(equation: Any, unknown: Variable, /) -> list[Expr]:
    """The solutions of ``equation`` (a residual, or ``op.eq``) for
    ``unknown``, a symbol or ``D(x)``.

    Raises ``NotImplementedError`` if no strategy applies.

    >>> from symbolcat import Symbol, real
    >>> from symbolcat.lib import operators as op
    >>> x, a = Symbol("x"), Symbol("a")
    >>> [str(s) for s in solve(x**2 - a, x)]
    ['-sqrt(a)', 'sqrt(a)']
    >>> [str(s) for s in solve(op.eq(real.exp(2 * x), a), x)]
    ['log(a) / 2']
    >>> [str(s) for s in solve(x**3 - 6 * x**2 + 11 * x - 6, x)]
    ['1', '2', '3']
    """
    if not is_variable(unknown):
        raise TypeError(f"solve: the unknown must be a symbol or D(x), not {unknown}")
    expr = substitute_variables(residual(equation), {unknown: _SOLVE})
    solutions = _solve(expr)
    back = {_SOLVE: unknown}
    return [simplify_exact(substitute_variables(s, back)) for s in solutions]


def _solve(expr: Expr, /) -> list[Expr]:
    try:
        coeffs = coefficients(expr, _SOLVE)
    except ValueError:
        coeffs = None
    if coeffs is not None:
        return _polynomial(coeffs, expr)
    fraction = cancel(expr) if _is_rational(expr) else expr
    if isinstance(fraction, Call) and func_of(fraction) is op.truediv:
        numerator, denominator = args_of(fraction)
        return [
            s
            for s in _solve(numerator)
            if not is_number(
                simplify_exact(substitute_variables(denominator, {_SOLVE: s})), 0
            )
        ]
    if fraction != expr:  # cancelled into a polynomial, without common roots
        return _solve(fraction)
    if _count(expr) == 1:
        return _isolate(expr, Literal(0))
    raise NotImplementedError(f"solve: cannot solve {expr} = 0")


def _is_rational(expr: Expr, /) -> bool:
    try:
        cancel(expr)
    except (NotPolynomial, ZeroDivisionError):
        return False
    return True


def _polynomial(coeffs: dict[int, Expr], expr: Expr, /) -> list[Expr]:
    degree = max(coeffs, default=0)
    c = lambda k: coeffs.get(k, Literal(0))
    if degree == 0:
        return []
    if degree == 1:
        return [simplify_exact(-c(0) / c(1))]
    if degree == 2:
        a, b, c0 = c(2), c(1), c(0)
        if is_number(b, 0):
            root = real.sqrt(simplify_exact(-c0 / a))
            return [simplify_exact(-root), root]
        discriminant = simplify_exact(b**2 - 4 * a * c0)
        if is_number(discriminant, 0):
            return [simplify_exact(-b / (2 * a))]
        root = real.sqrt(discriminant)
        return [
            simplify_exact((-b - root) / (2 * a)),
            simplify_exact((-b + root) / (2 * a)),
        ]
    try:
        _univariate(expr, _SOLVE)
    except NotPolynomial:
        raise NotImplementedError(
            f"solve: degree {degree} with non-rational coefficients"
        ) from None
    roots = rational_roots(expr, _SOLVE)
    rest = expr
    for r in roots:
        while True:
            (q,), remainder = polynomial_divide(rest, [_SOLVE - r], [_SOLVE])
            if not is_number(remainder, 0):
                break
            rest = q
    remaining = coefficients(rest, _SOLVE)
    if max(remaining, default=0) > 2:
        raise NotImplementedError(
            f"solve: {rest} has degree {max(remaining)} and no rational roots"
        )
    return [*roots, *_polynomial(remaining, rest)]


def _count(expr: Expr, /) -> int:
    if expr == _SOLVE:
        return 1
    if isinstance(expr, Call):
        return sum(_count(arg) for arg in args_of(expr))
    return 0


# For f(u) = rhs, the values of u.
type Inverse = Callable[[Expr], list[Expr]]

INVERSES: dict[Any, Inverse] = {
    real.exp: lambda r: [real.log(r)],
    real.log: lambda r: [real.exp(r)],
    real.sqrt: lambda r: [r**2],
    real.sin: lambda r: [real.asin(r)],
    real.cos: lambda r: [real.acos(r)],
    real.tan: lambda r: [real.atan(r)],
    real.asin: lambda r: [real.sin(r)],
    real.acos: lambda r: [real.cos(r)],
    real.atan: lambda r: [real.tan(r)],
    op.neg: lambda r: [-r],
    op.pos: lambda r: [r],
}


def _isolate(expr: Expr, rhs: Expr, /) -> list[Expr]:
    """The values of the unknown where ``expr = rhs``, when it appears once."""
    if expr == _SOLVE:
        return [rhs]
    assert isinstance(expr, Call)
    (position,) = [i for i, arg in enumerate(args_of(expr)) if _count(arg)]
    inner = args_of(expr)[position]
    func = func_of(expr)
    if func in INVERSES:
        values = INVERSES[func](rhs)
    elif func in (op.add, op.sub, op.mul, op.truediv, op.pow):
        other = args_of(expr)[1 - position]
        values = _invert_binary(func, position, other, rhs)
    else:
        raise NotImplementedError(f"solve: cannot invert {name_of(func)}")
    return [s for value in values for s in _isolate(inner, value)]


def _invert_binary(func: Any, position: int, other: Expr, rhs: Expr, /) -> list[Expr]:
    left = position == 0
    if func is op.add:
        return [rhs - other]
    if func is op.sub:
        return [rhs + other] if left else [other - rhs]
    if func is op.mul:
        return [rhs / other]
    if func is op.truediv:
        return [rhs * other] if left else [other / rhs]
    # pow
    if not left:
        return [real.log(rhs) / real.log(other)]
    if is_number(other, 2):
        root = real.sqrt(rhs)
        return [-root, root]
    return [rhs ** (1 / other)]
