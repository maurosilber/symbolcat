"""
symbolcat.ops._printing._julia
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate Julia source for a function computing an expression.

Powers are ``^``, indices are 1-based, and explicit arrays are Julia array
literals (``[a, b]``, ``[a b; c d]``). ``erf``, ``gamma`` and
``gammainc`` come from SpecialFunctions.jl.

Complex functions whose real version throws outside its domain (``sqrt``,
``log``, ...) are applied to ``complex(z)``, so they take the principal
branch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ... import complex as cx
from ... import operators as op
from ... import real
from ..._core import (
    Call,
    Constant,
    Expr,
    Function,
    Literal,
    Operator,
    Symbol,
    args_of,
    cata,
    func_of,
    name_of,
    precedence_of,
    value_of,
)
from ..._core.precedence import POSTFIX, POW
from ...linalg import Array, nest
from .._analysis._inspect import free_symbols
from .._transform._arrays import expand_arrays
from .._transform._share import cse
from ._base import Printed, indices, literal_precedence, operand
from ._str import str_algebra

FUNCTIONS: dict[Function, str] = {
    real.atan2: "atan",
    cx.re: "real",
    cx.im: "imag",
    cx.phase: "angle",
}

# Complex functions that throw a DomainError for some real arguments.
PROMOTED = frozenset([cx.log, cx.log10, cx.sqrt, cx.asin, cx.acos])

CONSTANTS = {
    real.constants.e: "ℯ",
    real.constants.pi: "pi",
    real.constants.tau: "(2pi)",
    real.constants.inf: "Inf",
    real.constants.nan: "NaN",
    cx.constants.I: "im",
}

julia_algebra = str_algebra.copy("julia")


@julia_algebra.register(Literal)
def _(node: Literal) -> Printed:
    value = value_of(node)
    if isinstance(value, bool):
        return Printed("true" if value else "false")
    if isinstance(value, float) and math.isnan(value):
        return Printed("NaN")
    if isinstance(value, complex):
        return Printed(f"({value.real!r} + {value.imag!r}im)")
    if value in (float("inf"), float("-inf")):
        return Printed("Inf" if value > 0 else "-Inf", literal_precedence(value))
    return Printed(repr(value), literal_precedence(value))


@julia_algebra.register(Constant)
def _(node: Constant) -> Printed:
    return Printed(CONSTANTS.get(node) or name_of(node))


@julia_algebra.register(*FUNCTIONS)
def _(node: Call[Printed]) -> Printed:
    return Printed(
        f"{FUNCTIONS[func_of(node)]}({', '.join(a.text for a in args_of(node))})"
    )


@julia_algebra.register(*PROMOTED)
def _(node: Call[Printed]) -> Printed:
    (arg,) = args_of(node)
    return Printed(f"{name_of(func_of(node))}(complex({arg.text}))")


@julia_algebra.register(real.gammainc)
def _(node: Call[Printed]) -> Printed:
    a, x = args_of(node)
    return Printed(f"gamma_inc({a.text}, {x.text})[1]", POSTFIX)


@julia_algebra.register(cx.rect)
def _(node: Call[Printed]) -> Printed:
    r, phi = args_of(node)
    return Printed(
        f"{operand(r, op.mul, 'left')} * cis({phi.text})", precedence_of(op.mul)
    )


@julia_algebra.register(op.pow)
def _(node: Call[Printed]) -> Printed:
    base, exponent = args_of(node)
    text = f"{operand(base, op.pow, 'left')} ^ {operand(exponent, op.pow, 'right')}"
    return Printed(text, POW)


@julia_algebra.register(op.floordiv, op.mod)
def _(node: Call[Printed]) -> Printed:
    a, b = args_of(node)
    name = "fld" if func_of(node) is op.floordiv else "mod"
    return Printed(f"{name}({a.text}, {b.text})")


@julia_algebra.register(op.xor)
def _(node: Call[Printed]) -> Printed:
    a, b = args_of(node)
    func = op.xor
    assert isinstance(func, Operator)
    text = f"{operand(a, func, 'left')} ⊻ {operand(b, func, 'right')}"
    return Printed(text, precedence_of(func))


@julia_algebra.register(op.getitem)
def _(node: Call[Printed]) -> Printed:
    obj, key = args_of(node)
    found = indices(key)
    index = f"{key.text} + 1" if found is None else ", ".join(str(i + 1) for i in found)
    return Printed(f"{operand(obj, op.getitem, 'left')}[{index}]", POSTFIX)


@julia_algebra.register(Array)
def _(node: Call[Printed]) -> Printed:
    assert isinstance(func_of(node), Array)
    rows = nest(node.kind.shape, [arg.text for arg in args_of(node)])
    if len(node.kind.shape) == 1:
        return Printed(f"[{', '.join(rows)}]")
    return Printed("[" + "; ".join(" ".join(row) for row in rows) + "]")


def to_julia(
    expr: Any,
    /,
    *,
    args: Sequence[Symbol] | None = None,
    name: str = "f",
) -> str:
    """Source of a Julia function computing ``expr``.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(to_julia(real.sin(x) ** 2 + real.sin(x) * y))
    function f(x, y)
        _0 = sin(x)
        return _0 ^ 2 + _0 * y
    end
    >>> v = Symbol("v")
    >>> print(to_julia(v[0] * x**-1, name="g"))
    function g(v, x)
        return v[1] * x ^ (-1)
    end
    """
    expr = expand_arrays(expr)
    if args is None:
        args = free_symbols(expr)
    params = [name_of(arg) for arg in args]
    assignments, result = cse(expr, prefix="_")

    def source(value: Expr, /) -> str:
        return cata(julia_algebra, value).text

    lines = [f"function {name}({', '.join(params)})"]
    lines += [f"    {name_of(t)} = {source(v)}" for t, v in assignments]
    lines += [f"    return {source(result)}", "end"]
    return "\n".join(lines)
