"""
symbolcat._core.algebra
~~~~~~~~~~~~~~~~~~~~~~~

Algebras assembled from rules registered per node type or function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .expr import Call, Expr, Function, _label


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
    """

    name: str
    _by_func: dict[Function, Callable[..., A]] = field(default_factory=dict)
    _by_kind: dict[type, Callable[..., A]] = field(default_factory=dict)
    _by_node: dict[type, Callable[..., A]] = field(default_factory=dict)

    def register(self, *keys: Function | type) -> Callable[[Callable[..., A]], Any]:
        def decorator(handler: Callable[..., A]) -> Callable[..., A]:
            for key in keys:
                if isinstance(key, Function):
                    self._by_func[key] = handler
                elif issubclass(key, Function):
                    self._by_kind[key] = handler
                elif issubclass(key, Expr):
                    self._by_node[key] = handler
                else:
                    raise TypeError(f"Cannot register {key!r}")
            return handler

        return decorator

    def copy(self, name: str) -> Algebra[A]:
        """A new algebra that starts with the same handlers."""
        return Algebra(
            name, dict(self._by_func), dict(self._by_kind), dict(self._by_node)
        )

    def handler_for(self, node: Expr) -> Callable[..., A]:
        if isinstance(node, Call):
            if (handler := self._by_func.get(node.func)) is not None:
                return handler
            if (handler := _lookup_mro(self._by_kind, type(node.func))) is not None:
                return handler
        if (handler := _lookup_mro(self._by_node, type(node))) is not None:
            return handler
        what = _label(node.func) if isinstance(node, Call) else type(node).__name__
        raise NotImplementedError(f"{self.name}: no rule for {what}")

    def __call__(self, node: Expr, **ctx: Any) -> A:
        return self.handler_for(node)(node, **ctx)


def _lookup_mro[V](registry: dict[type, V], cls: type) -> V | None:
    for base in cls.__mro__:
        if base in registry:
            return registry[base]
    return None
