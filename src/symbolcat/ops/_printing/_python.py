"""
symbolcat.ops._printing._python
~~~~~~~~~~~~~~~~~~~~~

Generate Python source for a function computing an expression.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..._core import Algebra, Call, Expr, Symbol, cata, name_of
from ...linalg import Array, nest
from .._analysis._inspect import free_symbols
from .._transform._arrays import expand_arrays
from .._transform._share import cse
from ._base import Printed
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

    return algebra


def to_python(
    expr: Any,
    /,
    *,
    args: Sequence[Symbol] | None = None,
    name: str = "f",
    shapes: Mapping[Any, Any] | None = None,
    array: str | None = None,
) -> str:
    """Source of a Python function ``name(*args)`` computing ``expr``.

    Repeated sub-expressions are computed once, into temporaries (see
    ``cse``). ``args`` defaults to the free symbols of ``expr``. Functions
    and constants are written by name (``sin``, ``pi``), so the source runs
    in a namespace that defines them, like ``vars(math)`` or ``vars(numpy)``.

    Operations on arrays are computed entry by entry (see ``expand_arrays``,
    which takes ``shapes``). Explicit arrays are written as nested lists, or
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
    expr = expand_arrays(expr, shapes)
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
