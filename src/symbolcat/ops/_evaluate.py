"""
symbolcat.ops._evaluate
~~~~~~~~~~~~~~~~~~~~~~~

Evaluate an expression with an implementation library.

The algebra is the same for every backend; the library (a mapping from
abstract functions to concrete ones) is passed as context to the fold.

The fold is ``cata``'s, except that ``ifelse`` evaluates its condition
first and, when it is a single boolean, only the branch taken: in
``ifelse(x > 0, log(x), 0)``, ``log(x)`` is not evaluated for negative
``x``. Array conditions (NumPy) need both branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import real
from .._core import (
    Algebra,
    Call,
    Constant,
    Expr,
    Literal,
    Symbol,
    as_expr,
    children,
    fmap,
    name_of,
)
from ..linalg import Array, nest
from ._transform._arrays import expand_arrays

evaluate_algebra = Algebra[Any]("evaluate")


@evaluate_algebra.register(Symbol)
def _(node: Symbol, *, env: Mapping[Symbol | str, Any], lib: Mapping[Any, Any]) -> Any:
    name_of(node)  # unnamed symbols would all compare equal
    if node in env:
        return env[node]
    if node.name in env:  # given by name
        return env[node.name]
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


def is_scalar_bool(value: Any, /) -> bool:
    """A Python or NumPy boolean, not an array or a symbolic condition."""
    return isinstance(value, bool) or (
        getattr(value, "shape", None) == () and getattr(value, "dtype", None) == bool
    )


def fold(expr: Expr, /, **ctx: Any) -> Any:
    """Like ``cata(evaluate_algebra, expr)``, with a lazy ``ifelse``."""
    # Written out rather than as a ``hylo``: which branch of an ``ifelse``
    # to visit depends on the value of its condition, and a coalgebra only
    # sees seeds, not folded values.
    done: dict[int, Any] = {}
    stack = [expr]
    while stack:
        node = stack[-1]
        if id(node) in done:
            stack.pop()
            continue
        if isinstance(node, Call) and node.func is real.ifelse:
            condition, a, b = node.args
            if id(condition) not in done:
                stack.append(condition)
                continue
            if is_scalar_bool(value := done[id(condition)]):
                taken = a if value else b
                if id(taken) not in done:
                    stack.append(taken)
                    continue
                stack.pop()
                done[id(node)] = done[id(taken)]
                continue
        pending = [c for c in children(node) if id(c) not in done]
        if pending:
            stack.extend(pending)
            continue
        stack.pop()
        done[id(node)] = evaluate_algebra(fmap(node, lambda c: done[id(c)]), **ctx)
    return done[id(expr)]


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
    Operations on explicit arrays are first computed entry by entry
    (``expand_arrays``), so every backend gives the same result. Arrays
    evaluate to nested lists, unless ``lib`` maps ``Array`` to a constructor
    taking them. Operations on symbols of an array kind are left to ``lib``.

    >>> from symbolcat import Symbol, real
    >>> x = Symbol("x")
    >>> evaluate(2 * real.cos(x) ** 2, {x: 0})
    2.0
    >>> evaluate(x + 1, {"x": 1})
    2
    >>> evaluate(real.ifelse(x > 0, real.log(x), 0), {x: -1})
    0
    """
    if lib is None:
        from ..lib.std import IMPL as lib
    return fold(as_expr(expand_arrays(expr)), env=env or {}, lib=lib)
