"""
symbolcat.backends.numpy
~~~~~~~~~~~~~~~~~~~~~~~~

NumPy as implementation. Requires NumPy.

``IMPL`` maps abstract functions and constants to their implementation.
Pass this module, or ``IMPL``, to ``symbolcat.ops.evaluate`` or
``symbolcat.ops.compile`` as ``backend``.
"""

from ._numpy import IMPL

__all__ = ["IMPL"]
