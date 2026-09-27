"""
symbolcat.ops.egraph.cost
~~~~~~~~~~~~~~~~~~~~~~~~~

Cost models deciding which equivalent expression is extracted.

Both count only ``Num`` nodes: the egglog primitives inside a literal
(``BigRat``, ``BigInt``, ``String``, ...) are free, so a number costs the
same as a symbol.
"""

from __future__ import annotations

from typing import Any

from egglog import DagCostModel, EGraph, get_callable_fn

from .convert import FUNCTIONS
from .term import Num, cos, exp, log, sin, sqrt, tan

LEAVES = (Num, Num.float, Num.var, Num.opaque)
NUM_FUNCTIONS = frozenset((*LEAVES, Num.apply, *FUNCTIONS.values()))


def node_cost(costs: dict[Any, int], default: int) -> DagCostModel[int]:
    """Additive cost model with a cost per ``Num`` constructor."""

    def marginal_cost(egraph: EGraph, expr: Any) -> int:
        fn = get_callable_fn(expr)
        if fn is None or fn not in NUM_FUNCTIONS:
            return 0
        return costs.get(fn, default)

    return DagCostModel(marginal_cost, 0)


#: Number of nodes: the smallest expression wins.
size = node_cost({}, default=1)

#: Rough evaluation cost: prefers cheap operations and, with DAG
#: extraction, reusing sub-expressions.
evaluation = node_cost(
    {
        **dict.fromkeys(LEAVES, 0),
        Num.__add__: 1,
        Num.__sub__: 1,
        Num.__neg__: 1,
        Num.__mul__: 2,
        Num.__truediv__: 8,
        sqrt: 10,
        Num.__pow__: 20,
        exp: 20,
        log: 20,
        sin: 20,
        cos: 20,
        tan: 25,
    },
    default=20,
)
