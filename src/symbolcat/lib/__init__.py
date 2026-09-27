"""
symbolcat.lib
~~~~~~~~~~~~~

Libraries of symbolic functions and their implementations.

- ``Function``, ``Constant``: define new symbolic functions and constants.
- ``operators``: the operators produced by Python syntax (``x + y``, ...).
- ``std``, ``numpy``, ``sympy``: implementation libraries for
  ``symbolcat.ops.evaluate``. Each exposes ``IMPL``, a mapping from abstract
  functions and constants (``operators.add``, ``real.sin``,
  ``real.constants.pi``) to concrete objects. Optionally, the ``Symbol`` class
  maps to a factory used for free symbols, which lets backends like SymPy
  keep them symbolic. ``numpy`` and ``sympy`` require those packages.
"""

# Imported from the module (not the package) so this also works while
# ``symbolcat._core`` is still initializing (it imports ``operators``).
from .._core.expr import Constant, Function

__all__ = ["Constant", "Function"]
