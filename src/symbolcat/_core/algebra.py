"""
symbolcat._core.algebra
~~~~~~~~~~~~~~~~~~~~~~~

Algebras assembled from rules registered per node type or function.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

from ..types._kinds import Kind
from .opaque import _init, _Opaque


class _AlgebraData(NamedTuple):
    name: str
    by_func: dict[Key[Function], Callable[..., Any]]
    by_kind: dict[Key[type], Callable[..., Any]]
    by_node: dict[Key[type], Callable[..., Any]]


class Algebra[A](_Opaque):
    """An algebra assembled from handlers registered per node or function.

    A handler receives a single layer (children already folded) plus any
    context passed to ``cata``. Keys can be:

    - a ``Function`` instance (e.g. ``operators.add``): most specific;
    - a ``Function`` subclass (e.g. ``Operator``): default for that kind;
    - an ``Expr`` subclass (``Symbol``, ``Constant``, ``Literal``, ``Call``).

    >>> depth = Algebra("depth")
    >>> @depth.register(Expr)  # any leaf
    ... def _(node):
    ...     return 1
    >>> @depth.register(Call)
    ... def _(node):
    ...     return 1 + max(args_of(node))
    >>> from symbolcat import Symbol
    >>> from symbolcat.ops.algebras import args_of, cata
    >>> x = Symbol("x")
    >>> cata(depth, (x + 1) * 2)
    3

    ``kind`` restricts a handler to nodes of that kind (``symbolcat.types``),
    including its subclasses (``kind=Real`` also matches booleans). For the
    same key, it takes precedence over a handler without ``kind``:

    >>> from symbolcat.types import Vector
    >>> describe = Algebra("describe")
    >>> @describe.register(Expr)
    ... def _(node):
    ...     return "scalar"
    >>> @describe.register(Expr, kind=Vector)
    ... def _(node):
    ...     return f"vector of {node.kind.n}"
    >>> describe(x), describe(Symbol("v", kind=Vector(3)))
    ('scalar', 'vector of 3')
    """

    __slots__ = ()

    def __init__(self, name: str, /) -> None:
        _init(self, _AlgebraData(name, {}, {}, {}))

    def __repr__(self) -> str:
        return f"Algebra({self.__symbolcat__.name!r})"

    def register(
        self, *keys: Function | type, kind: type[Kind] | None = None
    ) -> Callable[[Callable[..., A]], Any]:
        if kind is not None and not issubclass(kind, Kind):
            raise TypeError(f"Cannot restrict to {kind!r}, which is not a Kind")

        def decorator(handler: Callable[..., A]) -> Callable[..., A]:
            # a local import: rules are registered while ``expr`` loads
            from .expr import Expr, Function

            data = self.__symbolcat__
            for key in keys:
                if isinstance(key, Function):
                    data.by_func[key, kind] = handler
                elif issubclass(key, Function):
                    data.by_kind[key, kind] = handler
                elif issubclass(key, Expr):
                    data.by_node[key, kind] = handler
                else:
                    raise TypeError(f"Cannot register {key!r}")
            return handler

        return decorator

    def copy(self, name: str, /) -> Algebra[A]:
        """A new algebra that starts with the same handlers."""
        new = Algebra[A](name)
        data, new_data = self.__symbolcat__, new.__symbolcat__
        new_data.by_func.update(data.by_func)
        new_data.by_kind.update(data.by_kind)
        new_data.by_node.update(data.by_node)
        return new

    def handler_for(self, node: Expr, /) -> Callable[..., A]:
        data = self.__symbolcat__
        kinds = (*type(node.kind).__mro__[:-1], None)  # without object
        if isinstance(node, Call):
            func = func_of(node)
            for kind in kinds:
                if (handler := data.by_func.get((func, kind))) is not None:
                    return handler
            if (handler := _lookup(data.by_kind, type(func), kinds)) is not None:
                return handler
        if (handler := _lookup(data.by_node, type(node), kinds)) is not None:
            return handler
        what = _label(func_of(node)) if isinstance(node, Call) else type(node).__name__
        raise NotImplementedError(f"{data.name}: no rule for {what}")

    def __call__(self, node: Expr, **ctx: Any) -> A:
        return self.handler_for(node)(node, **ctx)


# A registry key: a function or class, restricted to a kind of node or not.
type Key[K] = tuple[K, type[Kind] | None]


def _lookup[V](
    registry: dict[Key[type], V], cls: type, kinds: tuple[type | None, ...], /
) -> V | None:
    """The handler for the most specific class in the MRO of ``cls`` and,
    for the same class, the most specific kind."""
    for base in cls.__mro__:
        for kind in kinds:
            if (handler := registry.get((base, kind))) is not None:
                return handler
    return None


# The handlers dispatch on the node classes, whose module needs ``Algebra``
# (for kind inference).
from .expr import Call, Expr, Function, _label, func_of
