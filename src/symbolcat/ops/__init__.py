"""
symbolcat.ops
~~~~~~~~~~~~~

Operations on expressions. Each one is an algebra folded with ``cata``.

The algebras are exported too, so they can be extended with new rules
(``latex_algebra.register(my_function)``) or copied and customized
(``latex_algebra.copy("my_latex")``).
"""

from .diff import diff, diff_algebra
from .evaluate import evaluate, evaluate_algebra
from .inspect import count_nodes, depth, free_symbols, free_symbols_algebra
from .latex import latex_algebra, to_latex
from .printing import Printed, str_algebra, to_str
from .simplify import simplify, simplify_algebra
from .substitute import substitute, substitute_algebra

__all__ = [
    "Printed",
    "count_nodes",
    "depth",
    "diff",
    "diff_algebra",
    "evaluate",
    "evaluate_algebra",
    "free_symbols",
    "free_symbols_algebra",
    "latex_algebra",
    "simplify",
    "simplify_algebra",
    "str_algebra",
    "substitute",
    "substitute_algebra",
    "to_latex",
    "to_str",
]
