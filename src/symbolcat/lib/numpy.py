"""
symbolcat.lib.numpy
~~~~~~~~~~~~~~~~~~~

NumPy as implementation. Requires NumPy.

``IMPL`` maps abstract functions and constants to their implementation;
pass it to ``symbolcat.ops.evaluate`` as ``lib``.
"""

from ._numpy import IMPL

__all__ = ["IMPL"]
