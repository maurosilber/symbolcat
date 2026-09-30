"""
symbolcat.backends.sympy
~~~~~~~~~~~~~~~~~~~~~~~~

SymPy as implementation; free symbols become ``sympy.Symbol``. Requires SymPy.

``IMPL`` maps abstract functions and constants to their implementation.
Pass this module, or ``IMPL``, to ``symbolcat.ops.evaluate`` or
``symbolcat.ops.compile`` as ``backend``.
"""

from ._sympy import IMPL

__all__ = ["IMPL"]
