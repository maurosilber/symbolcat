"""
symbolcat.models.dae
~~~~~~~~~~~~~~~~~~~~

Differential-algebraic equations: index reduction (Pantelides' algorithm,
dummy derivatives), consistent initialization and structural
simplification.
"""

from ._dae._initialization import initialization_system
from ._dae._pantelides import Pantelides, pantelides
from ._dae._structure import eliminate_aliases, structural_simplify
from ._dae._system import IndexOneDAE, dummy_derivatives, lower_index

__all__ = [
    "IndexOneDAE",
    "Pantelides",
    "dummy_derivatives",
    "eliminate_aliases",
    "initialization_system",
    "lower_index",
    "pantelides",
    "structural_simplify",
]
