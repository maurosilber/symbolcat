"""
symbolcat.ops.transform
~~~~~~~~~~~~~~~~~~~~~~~

Transformations of expressions: substitution, simplification, sharing of
repeated sub-expressions, and rewriting with user rules (``Wild``,
``Rule``, ``rewrite``).
"""

from ._transform._arrays import expand_arrays
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
    "rewrite",
    "share",
    "simplify_naive",
    "substitute",
    "substitute_variables",
]
