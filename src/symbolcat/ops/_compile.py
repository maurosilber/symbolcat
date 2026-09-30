"""
symbolcat.ops._compile
~~~~~~~~~~~~~~~~~~~~~~

Compile an expression into a Python function with an implementation library.

The source is the one of ``to_python``, except that functions and constants
are names bound to their implementation in the backend: ``real.sin`` and
``complex.sin`` are different names, bound to ``math.sin`` and
``cmath.sin`` by ``backends.std``. Operators are written with Python syntax
when the backend implements them as Python does (``operator.add``), and as
calls otherwise.
"""

from __future__ import annotations

import builtins
import cmath
import linecache
from collections.abc import Sequence
from itertools import count
from types import FunctionType
from typing import Any

from .. import operators as op
from .. import real
from .._core import (
    Call,
    Constant,
    Function,
    Literal,
    Operator,
    Symbol,
    args_of,
    as_expr,
    func_of,
    name_of,
    value_of,
)
from ..backends._resolve import Backend, as_mapping
from ..backends._std import IMPL as STD
from ..backends._std import OPERATORS
from ..linalg import Array, nest
from ._printing._base import Printed
from ._printing._python import function_source, parameters
from ._printing._str import str_algebra
from ._transform._arrays import expand_arrays
from ._transform._share import cse


def _is_source(value: Any, /) -> bool:
    """Whether ``repr(value)`` is Python source for ``value``."""
    if type(value) in (bool, int):
        return True
    return type(value) in (float, complex) and cmath.isfinite(value)


_filenames = count()  # tells apart the sources in ``linecache``


def compile[S: Symbol](
    expr: Any,
    args: Sequence[S] | None = None,
    /,
    *,
    backend: Backend | None = None,
    name: str = "f",
) -> FunctionType:
    """A Python function ``name(*args)`` computing ``expr`` with ``backend``.

    It computes what ``evaluate`` does, once the source is compiled:
    ``compile(expr, [x, y])(1, 2)`` is ``evaluate(expr, {x: 1, y: 2})``.
    ``args`` defaults to the free symbols of ``expr``. ``backend`` is a
    module of ``symbolcat.backends`` (the default is ``std``), or a mapping
    like their ``IMPL``. Repeated sub-expressions are computed once (see
    ``cse``), and ``inspect.getsource`` gives the source.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> f = compile(real.sin(x) ** 2 + real.sin(x) * y + real.constants.pi)
    >>> f(0.0, 2.0)
    3.141592653589793
    >>> import inspect
    >>> print(inspect.getsource(f))
    def f(x, y):
        _0 = sin(x)
        return _0 ** 2 + _0 * y + pi

    ``ifelse`` only computes the branch taken when the backend's is the one
    of ``std``; other backends (``numpy.where``) compute both.

    >>> compile(real.ifelse(x > 0, real.log(x), 0))(-1.0)
    0
    """
    backend = as_mapping(backend)
    expr = expand_arrays(as_expr(expr))
    params = parameters(expr, args, name)
    assignments, result = cse(expr, prefix="_")

    taken = {name, *params, *(name_of(temp) for temp, _ in assignments)}
    names: dict[Any, str] = {}  # the identifier bound to each implementation
    namespace: dict[str, Any] = {"__builtins__": builtins}

    def fresh(label: str, value: Any, /) -> str:
        """A new identifier, named after ``label``, bound to ``value``."""
        base = label if label.isidentifier() and not label.startswith("_") else "f"
        identifier = next(
            n for i in count() if (n := base if i == 0 else f"{base}_{i}") not in taken
        )
        taken.add(identifier)
        namespace[identifier] = value
        return identifier

    def bind(key: Any, label: str, /) -> str:
        """An identifier bound to ``backend[key]``, named after ``label``."""
        if key not in names:
            try:
                names[key] = fresh(label, backend[key])
            except KeyError:
                raise NotImplementedError(f"No implementation for {label!r}") from None
        return names[key]

    def native(func: Function, /) -> bool:
        """Whether ``backend`` implements ``func`` as Python syntax does."""
        return func in OPERATORS and backend.get(func) is OPERATORS[func]

    algebra = str_algebra.copy("compile")

    @algebra.register(Literal)
    def _(node: Literal) -> Printed:
        value = value_of(node)
        if _is_source(value):
            return str_algebra(node)
        return Printed(fresh("literal", value))

    @algebra.register(Constant)
    def _(node: Constant) -> Printed:
        return Printed(bind(node, name_of(node)))

    @algebra.register(Function)
    def call(node: Call[Printed]) -> Printed:
        func = func_of(node)
        args = ", ".join(arg.text for arg in args_of(node))
        return Printed(f"{bind(func, name_of(func))}({args})")

    @algebra.register(Operator, op.getitem)
    def _(node: Call[Printed]) -> Printed:
        return str_algebra(node) if native(func_of(node)) else call(node)

    @algebra.register(Array)
    def _(node: Call[Printed]) -> Printed:
        rows = nest(node.kind.shape, [arg.text for arg in args_of(node)])
        if len(node.kind.shape) == 2:
            rows = ["[" + ", ".join(row) + "]" for row in rows]
        text = "[" + ", ".join(rows) + "]"
        if Array not in backend:
            return Printed(text)
        return Printed(f"{bind(Array, 'array')}({text})")

    @algebra.register(real.ifelse)
    def _(node: Call[Printed]) -> Printed:
        if backend.get(real.ifelse) is not STD[real.ifelse]:
            return call(node)
        condition, a, b = (arg.text for arg in args_of(node))
        return Printed(f"({a} if {condition} else {b})")

    source = function_source(name, params, assignments, result, algebra)
    filename = f"<symbolcat.compile {name} {next(_filenames)}>"
    code = builtins.compile(source, filename, "exec")
    # so inspect.getsource and tracebacks show the source
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(code, namespace)  # noqa: S102 (the source is generated here)
    return namespace[name]
