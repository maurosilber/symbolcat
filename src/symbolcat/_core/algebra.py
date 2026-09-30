"""
symbolcat._core.algebra
~~~~~~~~~~~~~~~~~~~~~~~

Algebras assembled from rules registered per node type or function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .kinds import Kind


@dataclass
class Algebra[A]:
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
    ...     return 1 + max(node.args)
    >>> from symbolcat import Symbol
    >>> from symbolcat.ops.algebras import cata
    >>> x = Symbol("x")
    >>> cata(depth, (x + 1) * 2)
    3

    ``kind`` restricts a handler to nodes of that kind (``symbolcat.kinds``).
    For the same key, it takes precedence over a handler without ``kind``:

    >>> from symbolcat.kinds import Vector
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

    name: str
    _by_func: dict[Key[Function], Callable[..., A]] = field(
        default_factory=dict, init=False
    )
    _by_kind: dict[Key[type], Callable[..., A]] = field(
        default_factory=dict, init=False
    )
    _by_node: dict[Key[type], Callable[..., A]] = field(
        default_factory=dict, init=False
    )

    def register(
        self, *keys: Function | type, kind: type[Kind] | None = None
    ) -> Callable[[Callable[..., A]], Any]:
        if kind is not None and not issubclass(kind, Kind):
            raise TypeError(f"Cannot restrict to {kind!r}, which is not a Kind")

        def decorator(handler: Callable[..., A]) -> Callable[..., A]:
            # a local import: rules are registered while ``expr`` loads
            from .expr import Expr, Function

            for key in keys:
                if isinstance(key, Function):
                    self._by_func[key, kind] = handler
                elif issubclass(key, Function):
                    self._by_kind[key, kind] = handler
                elif issubclass(key, Expr):
                    self._by_node[key, kind] = handler
                else:
                    raise TypeError(f"Cannot register {key!r}")
            return handler

        return decorator

    def copy(self, name: str, /) -> Algebra[A]:
        """A new algebra that starts with the same handlers."""
        new = Algebra[A](name)
        new._by_func.update(self._by_func)
        new._by_kind.update(self._by_kind)
        new._by_node.update(self._by_node)
        return new

    def handler_for(self, node: Expr, /) -> Callable[..., A]:
        kinds = (*type(node.kind).__mro__[:-1], None)  # without object
        if isinstance(node, Call):
            for kind in kinds:
                if (handler := self._by_func.get((node.func, kind))) is not None:
                    return handler
            if (handler := _lookup(self._by_kind, type(node.func), kinds)) is not None:
                return handler
        if (handler := _lookup(self._by_node, type(node), kinds)) is not None:
            return handler
        what = _label(node.func) if isinstance(node, Call) else type(node).__name__
        raise NotImplementedError(f"{self.name}: no rule for {what}")

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
from .expr import Call, Expr, Function, _label
