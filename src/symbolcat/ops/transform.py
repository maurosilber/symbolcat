"""
symbolcat.ops.transform
~~~~~~~~~~~~~~~~~~~~~~~

Transformations of expressions: substitution, simplification, sharing of
repeated sub-expressions, rewriting with user rules (``Wild``, ``Rule``,
``rewrite``), and the real and imaginary parts of complex expressions.
"""

from ._transform._arrays import expand_arrays
from ._transform._complex import real_imag
from ._transform._rewrite import Rule, Wild, match, rewrite
from ._transform._share import cse, share
from ._transform._simplify import simplify_naive
from ._transform._substitute import substitute, substitute_variables

__all__ = [
    "Rule",
    "Wild",
    "cse",
    "expand_arrays",
    "match",
    "real_imag",
    "rewrite",
    "share",
    "simplify_naive",
    "substitute",
    "substitute_variables",
]
