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

from ._arrays import expand_arrays, shape
from ._dependencies import solve_dependencies
from ._diff import diff, gradient, hessian, jacobian
from ._dimensions import dimensions
from ._evaluate import evaluate
from ._inspect import count_nodes, depth, free_symbols
from ._latex import to_latex
from ._mathml import to_mathml
from ._polynomial import coefficients
from ._printing import to_str
from ._python import to_python
from ._series import series
from ._share import cse, share
from ._simplify import simplify_naive
from ._substitute import substitute
from ._views import to_graphviz, tree_view

__all__ = [
    "coefficients",
    "count_nodes",
    "cse",
    "depth",
    "diff",
    "dimensions",
    "evaluate",
    "expand_arrays",
    "free_symbols",
    "gradient",
    "hessian",
    "jacobian",
    "series",
    "shape",
    "share",
    "simplify_naive",
    "solve_dependencies",
    "substitute",
    "to_graphviz",
    "to_latex",
    "to_mathml",
    "to_python",
    "to_str",
    "tree_view",
]
