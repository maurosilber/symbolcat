"""
symbolcat.backends.numpy
~~~~~~~~~~~~~~~~~~~~~~~~

NumPy as implementation. Requires NumPy.

``IMPL`` maps abstract functions and constants to their implementation;
pass it to ``symbolcat.ops.evaluate`` as ``backend``.
"""

from ._numpy import IMPL

__all__ = ["IMPL"]
