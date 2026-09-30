"""
symbolcat.backends._resolve
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mapping of a backend given as a module (``backends.numpy``), a mapping,
or ``None`` (``backends.std``).
"""

from collections.abc import Mapping
from types import ModuleType
from typing import Any

type Backend = Mapping[Any, Any] | ModuleType


def as_mapping(backend: Backend | None, /) -> Mapping[Any, Any]:
    """The ``IMPL`` of a backend module, or the mapping itself.

    >>> from symbolcat.backends import std
    >>> as_mapping(std) is as_mapping(None) is std.IMPL
    True
    """
    if backend is None:
        from ._std import IMPL

        return IMPL
    if isinstance(backend, ModuleType):
        try:
            return backend.IMPL
        except AttributeError:
            raise TypeError(f"{backend.__name__} is not a backend: no IMPL") from None
    return backend
