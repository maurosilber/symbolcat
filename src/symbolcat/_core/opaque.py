"""
symbolcat._core.opaque
~~~~~~~~~~~~~~~~~~~~~~

Objects whose data is hidden from their subclasses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


class _Opaque:
    """Base class of objects that keep all their data in a single slot,
    ``__symbolcat__``, out of the namespace of their subclasses: a downstream
    library can subclass them, and define any attribute, without colliding
    with it. The data is read with accessor functions (``args_of``,
    ``name_of``, ...), and cannot be changed.
    """

    __slots__ = ("__symbolcat__",)

    if TYPE_CHECKING:
        __symbolcat__: Any

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "__symbolcat__":
            raise AttributeError(f"{type(self).__name__} objects are immutable")
        super().__setattr__(name, value)  # attributes of subclasses

    def __delattr__(self, name: str) -> None:
        if name == "__symbolcat__":
            raise AttributeError(f"{type(self).__name__} objects are immutable")
        super().__delattr__(name)

    def __reduce__(self) -> tuple[Any, ...]:
        state = getattr(self, "__dict__", None) or None  # of subclasses
        return _restore, (type(self), self.__symbolcat__), state


def _init(obj: _Opaque, data: Any, /) -> None:
    """Set the data of ``obj``, once, when it is created."""
    object.__setattr__(obj, "__symbolcat__", data)


def _restore[T: _Opaque](cls: type[T], data: Any, /) -> T:
    obj = object.__new__(cls)
    _init(obj, data)
    return obj
