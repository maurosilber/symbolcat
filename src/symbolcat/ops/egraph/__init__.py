"""
symbolcat.ops.egraph
~~~~~~~~~~~~~~~~~~~~

Simplification and optimization with e-graphs, using egglog.

Unlike ``symbolcat.ops.simplify``, which applies local rules in a single
bottom-up pass, equality saturation explores many equivalent forms at once
(commuting, factoring, expanding, ...) and then extracts the best one
according to a cost model.

- ``simplify``: the smallest equivalent expression.
- ``optimize``: the cheapest to evaluate, sharing repeated sub-expressions.
- ``positive``, ``nonnegative``, ``negative``, ``nonzero``, ``integer``:
  assumptions for ``assume=[...]``, enabling rules that are not valid for
  every input (``log(x * y) -> log(x) + log(y)`` needs ``positive(x)``).

The translation to egglog is one more algebra (``_convert.to_egglog``).
Rules live in ``_rules`` (as generators, one ruleset per topic), the cost
models in ``_term`` (``simplify``, native to egglog) and ``_cost``
(``optimize``), and the search loop in ``_saturate``.

Requires the ``egglog`` package.
"""

from ._assume import integer, negative, nonnegative, nonzero, positive
from ._saturate import optimize, simplify

__all__ = [
    "integer",
    "negative",
    "nonnegative",
    "nonzero",
    "optimize",
    "positive",
    "simplify",
]
