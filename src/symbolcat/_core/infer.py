"""
symbolcat._core.infer
~~~~~~~~~~~~~~~~~~~~~

The kind of a call, from the kinds of its arguments.

Inferring kinds is one more algebra, ``kind_algebra``: it collapses a layer
whose children are already kinds. A call applies it once, when it is built,
and stores the result, so ``expr.kind`` is ``cata(kind_algebra, expr)``
computed incrementally. An invalid call (``real.sin`` of a vector, the sum of
vectors of different lengths) is an error when it is built.

Functions declare their rule by registering it, like for any algebra:

>>> from symbolcat import Symbol
>>> from symbolcat import Function
>>> from symbolcat.ops.algebras import args_of, kind_algebra
>>> from symbolcat.kinds import Vector
>>> double = Function("double", arity=1)
>>> @kind_algebra.register(double)
... def _(node):
...     (kind,) = args_of(node)
...     return kind
>>> double(Symbol("v", kind=Vector(3))).kind
Vector(3)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .algebra import Algebra
from .expr import Call, Expr, Function, _label, args_of, func_of
from .kinds import Complex, Kind, Real, Scalar, array_kind

kind_algebra = Algebra[Kind]("kind")

type ElementRule = Callable[..., Scalar]


def infer(func: Function, args: tuple[Expr, ...], /) -> Kind:
    """The kind of ``func(*args)``."""
    return kind_algebra(Call(func, tuple(arg.kind for arg in args), Kind()))


@kind_algebra.register(Expr)
def _(node: Expr) -> Kind:
    """Leaves declare their kind."""
    return node.kind


@kind_algebra.register(Function)
def _(node: Call[Kind]) -> Kind:
    """By default, functions take scalars and return a real."""
    for kind in args_of(node):
        if kind.shape:
            raise ValueError(
                f"{_label(func_of(node))} needs scalar arguments, got shape {kind.shape}"
            )
    return Real()


def entrywise(element: ElementRule, /) -> Callable[[Call[Kind]], Kind]:
    """A rule for a function applied entry by entry, broadcasting scalars.

    ``element(func, *kinds)`` gives the kind of an entry from the kinds of
    the arguments' entries.
    """

    def rule(node: Call[Kind], /, **ctx: Any) -> Kind:
        if all(isinstance(kind, Scalar) for kind in args_of(node)):  # a fast path
            return element(func_of(node), *args_of(node))
        shapes = [kind.shape for kind in args_of(node)]
        arrays = list(dict.fromkeys(shape for shape in shapes if shape))
        if len(arrays) > 1:
            raise ValueError(
                f"{_label(func_of(node))} of incompatible shapes: "
                + ", ".join(map(str, shapes))
            )
        kind = element(func_of(node), *(kind.element for kind in args_of(node)))
        return array_kind(arrays[0] if arrays else (), kind)

    return rule


def real_arguments(func: Function, /, *kinds: Scalar) -> None:
    """Raise for a complex argument, where only real ones are defined."""
    if any(type(kind) is Complex for kind in kinds):
        raise TypeError(f"{_label(func)} needs real arguments, got a complex one")
