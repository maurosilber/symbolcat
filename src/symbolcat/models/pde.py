"""
symbolcat.models.pde
~~~~~~~~~~~~~~~~~~~~

Partial differential equations, written with ``partial(u, x)``:
discretization by the method of lines.
"""

from ._pde._lines import grid_symbol, method_of_lines

__all__ = [
    "grid_symbol",
    "method_of_lines",
]
