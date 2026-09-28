"""
symbolcat.ops
~~~~~~~~~~~~~

Operations on expressions. Each one is an algebra folded with a catamorphism.

Besides operations on single expressions, there are operations on systems
of equations: their structure (``EquationSystem``, ``block_triangular``,
``tear``, ``solve_linear``), differential-algebraic equations
(``pantelides``, ``dummy_derivatives``, ``structural_simplify``, ...) and
explicit ODEs (``ODE``, ``lie_derivative``, ``linearize``, ...). Time
derivatives are written with ``symbolcat.D``.

Two subpackages are imported on demand:

- ``symbolcat.ops.algebras``: the algebras behind these operations and the
  tools to write new ones (``cata``, ``Algebra``, node types), to add rules
  for new functions or define new operations.
- ``symbolcat.ops.egraph``: simplification by equality saturation. Requires
  the optional ``egglog`` package.
"""

from ._algebraic._linear import solve_linear
from ._algebraic._structure import TornSystem, block_triangular, tear
from ._algebraic._system import EquationSystem
from ._analysis._dependencies import solve_dependencies
from ._analysis._dimensions import dimensions
from ._analysis._inspect import count_nodes, depth, free_symbols, variables
from ._analysis._polynomial import coefficients
from ._calculus._diff import diff, gradient, hessian, jacobian, jvp
from ._calculus._reverse import reverse_gradient, vjp
from ._calculus._series import series
from ._calculus._time import change_variables, expand_derivatives, time_derivative
from ._calculus._variational import euler_lagrange
from ._dae._initialization import initialization_system
from ._dae._pantelides import Pantelides, pantelides
from ._dae._structure import eliminate_aliases, structural_simplify
from ._dae._system import IndexOneDAE, dummy_derivatives, lower_index
from ._evaluate import evaluate
from ._ode._analysis import (
    Linearization,
    lie_derivative,
    linearize,
    observability_matrix,
    steady_state,
)
from ._ode._sensitivity import sensitivities, sensitivity_symbol
from ._ode._series import taylor_solution
from ._ode._system import ODE, solve_for_derivatives, to_first_order
from ._printing._latex import to_latex
from ._printing._mathml import to_mathml
from ._printing._python import to_python
from ._printing._str import to_str
from ._printing._views import to_graphviz, tree_view
from ._transform._arrays import expand_arrays, shape
from ._transform._share import cse, share
from ._transform._simplify import simplify_naive
from ._transform._substitute import substitute, substitute_variables

__all__ = [
    "ODE",
    "EquationSystem",
    "IndexOneDAE",
    "Linearization",
    "Pantelides",
    "TornSystem",
    "block_triangular",
    "change_variables",
    "coefficients",
    "count_nodes",
    "cse",
    "depth",
    "diff",
    "dimensions",
    "dummy_derivatives",
    "eliminate_aliases",
    "euler_lagrange",
    "evaluate",
    "expand_arrays",
    "expand_derivatives",
    "free_symbols",
    "gradient",
    "hessian",
    "initialization_system",
    "jacobian",
    "jvp",
    "lie_derivative",
    "linearize",
    "lower_index",
    "observability_matrix",
    "pantelides",
    "reverse_gradient",
    "sensitivities",
    "sensitivity_symbol",
    "series",
    "shape",
    "share",
    "simplify_naive",
    "solve_dependencies",
    "solve_for_derivatives",
    "solve_linear",
    "steady_state",
    "structural_simplify",
    "substitute",
    "substitute_variables",
    "taylor_solution",
    "tear",
    "time_derivative",
    "to_first_order",
    "to_graphviz",
    "to_latex",
    "to_mathml",
    "to_python",
    "to_str",
    "tree_view",
    "variables",
    "vjp",
]
