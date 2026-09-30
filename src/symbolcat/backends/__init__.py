"""
symbolcat.backends
~~~~~~~~~~~~~~~~~~

Implementation libraries for ``symbolcat.ops.evaluate`` and
``symbolcat.ops.compile``, which take a module or its ``IMPL``:

>>> from symbolcat import Symbol, backends, real
>>> from symbolcat.ops import evaluate
>>> x = Symbol("x")
>>> evaluate(real.cos(x), {x: 0}, backend=backends.std)
1.0

``std``, ``numpy`` and ``sympy`` each expose ``IMPL``, a mapping from
abstract functions and constants (``operators.add``, ``real.sin``,
``real.constants.pi``) to concrete objects. Optionally, the ``Symbol`` class
maps to a factory used for free symbols, which lets backends like SymPy keep
them symbolic. ``numpy`` and ``sympy`` require those packages, and are
imported when first used.
"""

from importlib import import_module
from types import ModuleType

__all__ = ["numpy", "std", "sympy"]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
