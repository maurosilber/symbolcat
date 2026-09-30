"""
symbolcat.backends.std
~~~~~~~~~~~~~~~~~~~~~~

Python's standard library (``operator`` and ``math``) as implementation.

``IMPL`` maps abstract functions and constants to their implementation;
pass it to ``symbolcat.ops.evaluate`` as ``backend``.
"""

from ._std import IMPL

__all__ = ["IMPL"]
