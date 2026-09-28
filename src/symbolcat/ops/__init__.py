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

from ._analysis._dependencies import solve_dependencies
from ._analysis._dimensions import dimensions
from ._analysis._inspect import count_nodes, depth, free_symbols, variables
from ._analysis._polynomial import coefficients
from ._calculus._diff import diff, gradient, hessian, jacobian, jvp
from ._calculus._reverse import reverse_gradient, vjp
from ._calculus._series import series
from ._calculus._time import time_derivative
from ._dae._numeric import ODEFunction, to_ode
from ._dae._pantelides import Pantelides, pantelides
from ._dae._system import IndexOneDAE, dummy_derivatives, lower_index
from ._evaluate import evaluate
from ._printing._latex import to_latex
from ._printing._mathml import to_mathml
from ._printing._python import to_python
from ._printing._str import to_str
from ._printing._views import to_graphviz, tree_view
from ._transform._arrays import expand_arrays, shape
from ._transform._share import cse, share
from ._transform._simplify import simplify_naive
from ._transform._substitute import substitute

__all__ = [
    "IndexOneDAE",
    "ODEFunction",
    "Pantelides",
    "coefficients",
    "count_nodes",
    "cse",
    "depth",
    "diff",
    "dimensions",
    "dummy_derivatives",
    "evaluate",
    "expand_arrays",
    "free_symbols",
    "gradient",
    "hessian",
    "jacobian",
    "jvp",
    "lower_index",
    "pantelides",
    "reverse_gradient",
    "series",
    "shape",
    "share",
    "simplify_naive",
    "solve_dependencies",
    "substitute",
    "time_derivative",
    "to_graphviz",
    "to_latex",
    "to_mathml",
    "to_ode",
    "to_python",
    "to_str",
    "tree_view",
    "variables",
    "vjp",
]
