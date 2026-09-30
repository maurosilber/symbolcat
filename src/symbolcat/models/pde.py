"""
symbolcat.models.pde
~~~~~~~~~~~~~~~~~~~~

Partial differential equations, written with ``partial(u, x)``:
discretization by the method of lines.
"""

from ._pde._lines import method_of_lines

__all__ = [
    "method_of_lines",
]
