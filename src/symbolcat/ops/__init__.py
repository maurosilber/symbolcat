"""
symbolcat.ops
~~~~~~~~~~~~~

Operations on expressions. Each one is an algebra folded with a catamorphism.

- ``evaluate``: compute an expression with an implementation library.
- ``transform``: substitution, simplification, sharing, rewriting rules.
- ``inspect``: symbols, variables, size, shape, dimensions, sparsity.
- ``printing``: text, LaTeX, MathML, and Python, C and Julia source.
- ``calculus``: derivatives (forward and reverse), series, time derivatives.
- ``polynomials``: Gröbner bases, GCDs, factorization, rational functions.
- ``equations``: solving equations, systems of equations and their structure.

Two more subpackages:

- ``symbolcat.ops.algebras``: the tools to write new operations (``cata``,
  ``Algebra``, node types), and in ``algebras.builtin``, the algebras behind
  these operations, to add rules for new functions.
- ``symbolcat.ops.egraph``: simplification by equality saturation, imported
  on demand. Requires the optional ``egglog`` package.
"""

from . import algebras, calculus, equations, inspect, polynomials, printing, transform
from ._evaluate import evaluate

__all__ = [
    "algebras",
    "calculus",
    "equations",
    "evaluate",
    "inspect",
    "polynomials",
    "printing",
    "transform",
]
