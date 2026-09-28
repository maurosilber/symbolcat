"""
symbolcat.ops._evaluate
~~~~~~~~~~~~~~~~~~~~~~~

Evaluate an expression with an implementation library.

The algebra is the same for every backend; the library (a mapping from
abstract functions to concrete ones) is passed as context to ``cata``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .._core import Algebra, Call, Constant, Literal, Symbol, cata, name_of
from ..linalg import Array, nest
from ._transform._arrays import expand_arrays

evaluate_algebra = Algebra[Any]("evaluate")


@evaluate_algebra.register(Symbol)
def _(node: Symbol, *, env: Mapping[Symbol, Any], lib: Mapping[Any, Any]) -> Any:
    name_of(node)  # unnamed symbols would all compare equal
    if node in env:
        return env[node]
    if Symbol in lib:
        return lib[Symbol](node)
    raise NameError(f"No value for symbol {node.name!r}")


@evaluate_algebra.register(Constant)
def _(node: Constant, *, env: Mapping[Symbol, Any], lib: Mapping[Any, Any]) -> Any:
    try:
        return lib[node]
    except KeyError:
        raise NotImplementedError(f"No value for {name_of(node)!r}") from None


@evaluate_algebra.register(Literal)
def _(node: Literal, **ctx: Any) -> Any:
    return node.value


@evaluate_algebra.register(Call)
def _(node: Call[Any], *, env: Mapping[Symbol, Any], lib: Mapping[Any, Any]) -> Any:
    try:
        func = lib[node.func]
    except KeyError:
        raise NotImplementedError(
            f"No implementation for {name_of(node.func)!r}"
        ) from None
    return func(*node.args)


@evaluate_algebra.register(Array)
def _(node: Call[Any], *, env: Mapping[Symbol, Any], lib: Mapping[Any, Any]) -> Any:
    """Nested lists of the entries, or ``lib[Array]`` applied to them."""
    assert isinstance(node.func, Array)
    rows = nest(node.func.shape, node.args)
    return lib[Array](rows) if Array in lib else rows


def evaluate(
    expr: Any,
    env: Mapping[Symbol, Any] | Mapping[str, Any] | None = None,
    /,
    *,
    lib: Mapping[Any, Any] | None = None,
) -> Any:
    """Evaluate ``expr`` with values for its symbols and a library.

    ``env`` maps symbols (or their names) to values.
    ``lib`` defaults to ``symbolcat.lib.std.IMPL``.
    Operations on arrays are first computed entry by entry
    (``expand_arrays``), so every backend gives the same result. Arrays
    evaluate to nested lists, unless ``lib`` maps ``Array`` to a constructor
    taking them. Values in ``env`` with a ``shape`` attribute (NumPy arrays,
    SymPy matrices) are arrays; operations on them are left to ``lib``.

    >>> from symbolcat import Symbol, real
    >>> x = Symbol("x")
    >>> evaluate(2 * real.cos(x) ** 2, {x: 0})
    2.0
    >>> evaluate(x + 1, {"x": 1})
    2
    """
    if lib is None:
        from ..lib.std import IMPL as lib
    env = {Symbol(k) if isinstance(k, str) else k: v for k, v in (env or {}).items()}
    shapes = {k: s for k, v in env.items() if (s := tuple(getattr(v, "shape", ())))}
    return cata(evaluate_algebra, expand_arrays(expr, shapes), env=env, lib=lib)
