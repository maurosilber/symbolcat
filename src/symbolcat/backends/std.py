"""
symbolcat.backends.std
~~~~~~~~~~~~~~~~~~~~~~

Python's standard library (``operator`` and ``math``) as implementation.

``IMPL`` maps abstract functions and constants to their implementation.
Pass this module, or ``IMPL``, to ``symbolcat.ops.evaluate`` or
``symbolcat.ops.compile`` as ``backend``.
"""

from ._std import IMPL

__all__ = ["IMPL"]
