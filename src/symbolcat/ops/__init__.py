"""
symbolcat.ops
~~~~~~~~~~~~~

Operations on expressions. Each one is an algebra folded with a catamorphism.

Two subpackages are imported on demand:

- ``symbolcat.ops.algebras``: the algebras behind these operations and the
  tools to write new ones (``cata``, ``Algebra``, node types), to add rules
  for new functions or define new operations.
- ``symbolcat.ops.egraph``: simplification by equality saturation. Requires
  the optional ``egglog`` package.
"""

from ._diff import diff
from ._evaluate import evaluate
from ._inspect import count_nodes, depth, free_symbols
from ._latex import to_latex
from ._printing import to_str
from ._simplify import simplify_naive
from ._substitute import substitute

__all__ = [
    "count_nodes",
    "depth",
    "diff",
    "evaluate",
    "free_symbols",
    "simplify_naive",
    "substitute",
    "to_latex",
    "to_str",
]
