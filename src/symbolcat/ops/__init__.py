"""
symbolcat.ops
~~~~~~~~~~~~~

Operations on expressions. Each one is an algebra folded with a catamorphism.

Besides operations on single expressions, there are:

- algebra: polynomials (``groebner``, ``polynomial_gcd``, ``cancel``, ...),
  ``solve``, ``solve_linear``, ``generic_rank``, and rewriting with user
  rules (``Wild``, ``Rule``, ``rewrite``);
- systems of equations: ``EquationSystem``, ``block_triangular``, ``tear``;
- differential-algebraic equations: ``pantelides``, ``dummy_derivatives``,
  ``structural_simplify``, ...;
- differential equations: ``ODE`` (with delays and events), ``SDE``,
  ``DiscreteSystem``, and PDEs through ``method_of_lines``; their analysis
  (``lie_derivative``, ``linearize``, ``identifiability``, ...);
- reaction networks: ``Reaction``, ``ReactionSystem``,
  ``reaction_rate_equations``, ``conservation_laws``, ...;
- code generation: ``to_python``, ``to_c``, ``to_julia``.

Time derivatives are written with ``symbolcat.D``.

Two subpackages are imported on demand:

- ``symbolcat.ops.algebras``: the algebras behind these operations and the
  tools to write new ones (``cata``, ``Algebra``, node types), to add rules
  for new functions or define new operations.
- ``symbolcat.ops.egraph``: simplification by equality saturation. Requires
  the optional ``egglog`` package.
"""

from ._algebraic._linear import solve_linear
from ._algebraic._polynomial import (
    cancel,
    groebner,
    polynomial_divide,
    polynomial_gcd,
    polynomial_lcm,
    rational_roots,
    square_free_factorization,
)
from ._algebraic._rank import generic_rank, rational_nullspace, rational_rank
from ._algebraic._solve import solve
from ._algebraic._structure import TornSystem, block_triangular, tear
from ._algebraic._system import EquationSystem
from ._analysis._dependencies import solve_dependencies
from ._analysis._dimensions import dimensions
from ._analysis._inspect import count_nodes, depth, free_symbols, variables
from ._analysis._polynomial import coefficients
from ._analysis._sparsity import hessian_sparsity, jacobian_sparsity
from ._calculus._diff import diff, gradient, hessian, jacobian, jvp
from ._calculus._reverse import reverse_gradient, vjp
from ._calculus._series import series
from ._calculus._time import (
    change_variables,
    delays,
    expand_derivatives,
    time_derivative,
)
from ._calculus._variational import euler_lagrange
from ._dae._initialization import initialization_system
from ._dae._pantelides import Pantelides, pantelides
from ._dae._structure import eliminate_aliases, structural_simplify
from ._dae._system import IndexOneDAE, dummy_derivatives, lower_index
from ._evaluate import evaluate
from ._events import ContinuousEvent, DiscreteEvent
from ._ode._analysis import (
    Linearization,
    lie_derivative,
    linearize,
    observability_matrix,
    steady_state,
    transfer_function,
)
from ._ode._delay import characteristic_matrix
from ._ode._discrete import DiscreteSystem, fixed_points, iterate
from ._ode._identifiability import identifiability
from ._ode._sensitivity import sensitivities, sensitivity_symbol
from ._ode._series import taylor_solution
from ._ode._system import ODE, solve_for_derivatives, to_first_order
from ._pde._lines import grid_symbol, method_of_lines
from ._printing._c import to_c
from ._printing._julia import to_julia
from ._printing._latex import to_latex
from ._printing._mathml import to_mathml
from ._printing._python import to_python
from ._printing._str import to_str
from ._printing._views import to_graphviz, tree_view
from ._reactions._network import (
    Reaction,
    ReactionSystem,
    chemical_langevin,
    complexes,
    conservation_laws,
    deficiency,
    is_weakly_reversible,
    linkage_classes,
    propensities,
    propensity,
    rate_law,
    reaction_rate_equations,
    stoichiometry_matrix,
)
from ._sde._system import SDE, ito_lemma, ito_to_stratonovich, stratonovich_to_ito
from ._transform._arrays import expand_arrays, shape
from ._transform._rewrite import Rule, Wild, match, rewrite
from ._transform._share import cse, share
from ._transform._simplify import simplify_naive
from ._transform._substitute import substitute, substitute_variables

__all__ = [
    "ODE",
    "SDE",
    "ContinuousEvent",
    "DiscreteEvent",
    "DiscreteSystem",
    "EquationSystem",
    "IndexOneDAE",
    "Linearization",
    "Pantelides",
    "Reaction",
    "ReactionSystem",
    "Rule",
    "TornSystem",
    "Wild",
    "block_triangular",
    "cancel",
    "change_variables",
    "characteristic_matrix",
    "chemical_langevin",
    "coefficients",
    "complexes",
    "conservation_laws",
    "count_nodes",
    "cse",
    "deficiency",
    "delays",
    "depth",
    "diff",
    "dimensions",
    "dummy_derivatives",
    "eliminate_aliases",
    "euler_lagrange",
    "evaluate",
    "expand_arrays",
    "expand_derivatives",
    "fixed_points",
    "free_symbols",
    "generic_rank",
    "gradient",
    "grid_symbol",
    "groebner",
    "hessian",
    "hessian_sparsity",
    "identifiability",
    "initialization_system",
    "is_weakly_reversible",
    "iterate",
    "ito_lemma",
    "ito_to_stratonovich",
    "jacobian",
    "jacobian_sparsity",
    "jvp",
    "lie_derivative",
    "linearize",
    "linkage_classes",
    "lower_index",
    "match",
    "method_of_lines",
    "observability_matrix",
    "pantelides",
    "polynomial_divide",
    "polynomial_gcd",
    "polynomial_lcm",
    "propensities",
    "propensity",
    "rate_law",
    "rational_nullspace",
    "rational_rank",
    "rational_roots",
    "reaction_rate_equations",
    "reverse_gradient",
    "rewrite",
    "sensitivities",
    "sensitivity_symbol",
    "series",
    "shape",
    "share",
    "simplify_naive",
    "solve",
    "solve_dependencies",
    "solve_for_derivatives",
    "solve_linear",
    "square_free_factorization",
    "steady_state",
    "stoichiometry_matrix",
    "stratonovich_to_ito",
    "structural_simplify",
    "substitute",
    "substitute_variables",
    "taylor_solution",
    "tear",
    "time_derivative",
    "to_c",
    "to_first_order",
    "to_graphviz",
    "to_julia",
    "to_latex",
    "to_mathml",
    "to_python",
    "to_str",
    "transfer_function",
    "tree_view",
    "variables",
    "vjp",
]
