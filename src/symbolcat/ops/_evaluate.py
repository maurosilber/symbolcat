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


def evaluate(
    expr: Any,
    env: Mapping[Symbol | str, Any] | None = None,
    lib: Mapping[Any, Any] | None = None,
) -> Any:
    """Evaluate ``expr`` with values for its symbols and a library.

    ``env`` maps symbols (or their names) to values.
    ``lib`` defaults to ``symbolcat.lib.std.IMPL``.

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
    return cata(evaluate_algebra, expr, env=env, lib=lib)
