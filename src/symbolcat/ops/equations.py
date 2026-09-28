"""
symbolcat.ops.equations
~~~~~~~~~~~~~~~~~~~~~~~

Algebraic equations: solving one equation, or a linear system; systems of
equations and their structure (block triangular form, tearing); and exact
ranks.
"""

from ._algebraic._linear import solve_linear
from ._algebraic._rank import generic_rank, rational_nullspace, rational_rank
from ._algebraic._solve import solve
from ._algebraic._structure import TornSystem, block_triangular, tear
from ._algebraic._system import EquationSystem
from ._analysis._dependencies import solve_dependencies

__all__ = [
    "EquationSystem",
    "TornSystem",
    "block_triangular",
    "generic_rank",
    "rational_nullspace",
    "rational_rank",
    "solve",
    "solve_dependencies",
    "solve_linear",
    "tear",
]
