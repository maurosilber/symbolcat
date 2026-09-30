"""
symbolcat.ops._printing._c
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate C source for a function computing an expression.

Numbers are doubles, so integer literals are written with a decimal point
(``1.0 / 2.0``, not the integer division ``1 / 2``). Powers use ``pow``, and
``ifelse`` the conditional operator. The functions come from ``<math.h>``.
Complex functions and numbers are not supported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ... import complex as cx
from ... import real
from ..._core import (
    Call,
    Constant,
    Done,
    Expr,
    Function,
    Literal,
    Symbol,
    cata,
    hylo,
    name_of,
)
from ..._core.precedence import POSTFIX
from ...lib import operators as op
from ...linalg import Array
from .._analysis._inspect import free_symbols, union
from .._transform._arrays import expand_arrays
from .._transform._share import cse
from ._base import Printed, indices, literal_precedence, operand
from ._str import str_algebra

FUNCTIONS: dict[Function, str] = {
    real.abs: "fabs",
    real.gamma: "tgamma",
    real.atan2: "atan2",
}

CONSTANTS = {
    real.constants.e: "M_E",
    real.constants.pi: "M_PI",
    real.constants.tau: "(2.0 * M_PI)",
    real.constants.inf: "INFINITY",
    real.constants.nan: "NAN",
}

c_algebra = str_algebra.copy("c")


COMPLEX = [f for f in vars(cx).values() if isinstance(f, Function)]


def _no_complex(node: Any, /) -> Printed:
    raise NotImplementedError("to_c: C output uses doubles, not complex numbers")


@c_algebra.register(Literal)
def _(node: Literal) -> Printed:
    value = node.value
    if isinstance(value, complex):
        _no_complex(node)
    if isinstance(value, bool):
        return Printed("1" if value else "0")
    if isinstance(value, int):
        return Printed(f"{value}.0", literal_precedence(value))
    if isinstance(value, float) and math.isnan(value):
        return Printed("NAN")
    if value in (float("inf"), float("-inf")):
        return Printed("INFINITY" if value > 0 else "(-INFINITY)")
    return Printed(repr(value), literal_precedence(value))


@c_algebra.register(Constant)
def _(node: Constant) -> Printed:
    if node is cx.constants.I:
        _no_complex(node)
    return Printed(CONSTANTS.get(node) or name_of(node))


c_algebra.register(*COMPLEX)(_no_complex)


@c_algebra.register(Function)
def _(node: Call[Printed]) -> Printed:
    if node.func is real.factorial:
        (arg,) = node.args
        return Printed(f"tgamma({arg.text} + 1.0)")
    name = FUNCTIONS.get(node.func, name_of(node.func))
    return Printed(f"{name}({', '.join(arg.text for arg in node.args)})")


@c_algebra.register(op.pow)
def _(node: Call[Printed]) -> Printed:
    base, exponent = node.args
    return Printed(f"pow({base.text}, {exponent.text})")


@c_algebra.register(op.floordiv)
def _(node: Call[Printed]) -> Printed:
    a, b = node.args
    return Printed(f"floor({a.text} / {b.text})")


@c_algebra.register(op.mod)
def _(node: Call[Printed]) -> Printed:
    a, b = node.args
    return Printed(f"fmod({a.text}, {b.text})")


@c_algebra.register(real.ifelse)
def _(node: Call[Printed]) -> Printed:
    condition, a, b = node.args
    return Printed(f"({condition.text} ? {a.text} : {b.text})")


@c_algebra.register(op.getitem)
def _(node: Call[Printed]) -> Printed:
    obj, key = node.args
    found = indices(key)
    index = (
        f"[(int)({key.text})]" if found is None else "".join(f"[{i}]" for i in found)
    )
    return Printed(f"{operand(obj, op.getitem, 'left')}{index}", POSTFIX)


def _arrays(expr: Expr, /) -> dict[str, None]:
    """The names of the symbols indexed as arrays, ``x[i]``."""

    def coalgebra(node: Expr, /) -> Expr:
        if (
            isinstance(node, Call)
            and node.func is op.getitem
            and isinstance(array := node.args[0], Symbol)
        ):
            return Call(op.getitem, (Done({name_of(array): None}), node.args[1]))
        return node

    return hylo(union, coalgebra, expr)


def _source(expr: Expr, /) -> str:
    return cata(c_algebra, expr).text


def to_c(
    expr: Any,
    /,
    *,
    args: Sequence[Symbol] | None = None,
    name: str = "f",
) -> str:
    """Source of a C function computing ``expr`` from double arguments.

    A scalar is returned; an array is written, flattened in row-major
    order, to an extra ``double *out`` argument. Indexed arguments are
    pointers. Repeated sub-expressions
    are computed once (``cse``).

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(to_c(real.sin(x) ** 2 / 2 + real.sin(x) * y))
    double f(double x, double y) {
        const double _0 = sin(x);
        return pow(_0, 2.0) / 2.0 + _0 * y;
    }
    >>> from symbolcat.linalg import vector
    >>> print(to_c(vector([x * y, real.ifelse(x < y, x, y)]), name="g"))
    void g(double x, double y, double *out) {
        out[0] = x * y;
        out[1] = (x < y ? x : y);
    }
    >>> v = Symbol("v")
    >>> print(to_c(v[1] * x))
    double f(const double *v, double x) {
        return v[1] * x;
    }
    """
    expr = expand_arrays(expr)
    if args is None:
        args = free_symbols(expr)
    params = [name_of(arg) for arg in args]
    assignments, result = cse(expr, prefix="_")
    body = [f"    const double {name_of(t)} = {_source(v)};" for t, v in assignments]
    arrays = _arrays(expr)
    signature = ", ".join(
        f"const double *{p}" if p in arrays else f"double {p}" for p in params
    )
    if isinstance(result, Call) and isinstance(result.func, Array):
        out = ", " if params else ""
        lines = [f"void {name}({signature}{out}double *out) {{", *body]
        lines += [f"    out[{i}] = {_source(e)};" for i, e in enumerate(result.args)]
    else:
        lines = [
            f"double {name}({signature}) {{",
            *body,
            f"    return {_source(result)};",
        ]
    return "\n".join([*lines, "}"])
