"""
symbolcat.lib.sympy
~~~~~~~~~~~~~~~~~~~

SymPy as implementation; free symbols become ``sympy.Symbol``. Requires SymPy.

``IMPL`` maps abstract functions and constants to their implementation;
pass it to ``symbolcat.ops.evaluate`` as ``lib``.
"""

from ._sympy import IMPL

__all__ = ["IMPL"]
