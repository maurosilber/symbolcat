"""
symbolcat.ops._printing._python
~~~~~~~~~~~~~~~~~~~~~

Generate Python source for a function computing an expression.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ... import complex as cx
from ... import real
from ..._core import Algebra, Call, Constant, Expr, Symbol, cata, name_of
from ..._core.precedence import POSTFIX
from ...lib import operators as op
from ...linalg import Array, nest
from .._analysis._inspect import free_symbols
from .._transform._arrays import expand_arrays
from .._transform._share import cse
from ._base import Printed, operand
from ._str import str_algebra


def python_algebra(array: str | None, /) -> Algebra[Printed]:
    """Like ``str``, with explicit arrays as (nested) list literals, wrapped
    in a call to ``array`` if given."""
    algebra = str_algebra.copy("python")

    @algebra.register(Array)
    def _(node: Call[Printed]) -> Printed:
        assert isinstance(node.func, Array)
        rows = nest(node.func.shape, [arg.text for arg in node.args])
        if len(node.func.shape) == 2:
            rows = ["[" + ", ".join(row) + "]" for row in rows]
        text = "[" + ", ".join(rows) + "]"
        return Printed(text if array is None else f"{array}({text})")

    @algebra.register(real.ifelse)
    def _(node: Call[Printed]) -> Printed:
        condition, a, b = (arg.text for arg in node.args)
        return Printed(f"({a} if {condition} else {b})")

    @algebra.register(cx.re, cx.im, cx.conj)
    def _(node: Call[Printed]) -> Printed:
        (arg,) = node.args
        method = {cx.re: "real", cx.im: "imag", cx.conj: "conjugate()"}[node.func]
        text = operand(arg, op.getitem, "left")
        if text[0].isdigit():  # 2.real is a syntax error
            text = f"({text})"
        return Printed(f"{text}.{method}", POSTFIX)

    @algebra.register(Constant)
    def _(node: Constant) -> Printed:
        return Printed("1j" if node is cx.constants.I else name_of(node))

    return algebra


def to_python(
    expr: Any,
    /,
    *,
    args: Sequence[Symbol] | None = None,
    name: str = "f",
    array: str | None = None,
) -> str:
    """Source of a Python function ``name(*args)`` computing ``expr``.

    Repeated sub-expressions are computed once, into temporaries (see
    ``cse``). ``args`` defaults to the free symbols of ``expr``. Functions
    and constants are written by name (``sin``, ``pi``), so the source runs
    in a namespace that defines them, like ``vars(math)`` or ``vars(numpy)``,
    or ``vars(cmath)`` for complex functions. ``complex.re``, ``im`` and
    ``conj`` are written as attributes (``z.real``), and the imaginary unit
    as ``1j``.

    Operations on explicit arrays are computed entry by entry (see
    ``expand_arrays``). Explicit arrays are written as nested lists, or
    as a call to ``array`` if given (``"array"``, for NumPy).

    >>> import math
    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> source = to_python(real.sin(x) ** 2 + real.sin(x) * y + real.constants.pi)
    >>> print(source)
    def f(x, y):
        _0 = sin(x)
        return _0 ** 2 + _0 * y + pi
    >>> namespace = dict(vars(math))
    >>> exec(source, namespace)
    >>> namespace["f"](0.0, 2.0) == math.pi
    True
    >>> from symbolcat.linalg import matrix
    >>> print(to_python(matrix([[x, 1], [0, y]]) @ matrix([[y], [x]]), array="array"))
    def f(x, y):
        return array([[x * y + 1 * x], [0 * y + y * x]])
    """
    expr = expand_arrays(expr)
    if args is None:
        args = free_symbols(expr)
    params = [name_of(arg) for arg in args]
    for identifier in (name, *params):
        if not identifier.isidentifier():
            raise ValueError(f"{identifier!r} is not a valid Python identifier")
    missing = [s.name for s in free_symbols(expr) if s.name not in params]
    if missing:
        raise ValueError(f"Free symbols not in args: {', '.join(missing)}")
    assignments, result = cse(expr, prefix="_")
    lines = [f"def {name}({', '.join(params)}):"]
    algebra = python_algebra(array)

    def source(value: Expr, /) -> str:
        return cata(algebra, value).text

    lines += [f"    {name_of(temp)} = {source(value)}" for temp, value in assignments]
    lines.append(f"    return {source(result)}")
    return "\n".join(lines)
