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

The translation to egglog is one more algebra (``to_egglog``); rules live
in ``rules`` and cost models in ``cost``.

Requires the ``egglog`` package.
"""

from .saturate import optimize, simplify

__all__ = ["optimize", "simplify"]
