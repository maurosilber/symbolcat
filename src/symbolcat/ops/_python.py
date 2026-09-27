"""
symbolcat.ops._python
~~~~~~~~~~~~~~~~~~~~~

Generate Python source for a function computing an expression.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._core import Symbol, as_expr, name_of
from ._inspect import free_symbols
from ._printing import to_str
from ._share import cse


def to_python(
    expr: Any, /, *, args: Sequence[Symbol] | None = None, name: str = "f"
) -> str:
    """Source of a Python function ``name(*args)`` computing ``expr``.

    Repeated sub-expressions are computed once, into temporaries (see
    ``cse``). ``args`` defaults to the free symbols of ``expr``. Functions
    and constants are written by name (``sin``, ``pi``), so the source runs
    in a namespace that defines them, like ``vars(math)`` or ``vars(numpy)``.

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
    """
    expr = as_expr(expr)
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
    lines += [f"    {name_of(temp)} = {to_str(value)}" for temp, value in assignments]
    lines.append(f"    return {to_str(result)}")
    return "\n".join(lines)
