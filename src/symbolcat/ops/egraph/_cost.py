"""
symbolcat.ops.egraph._cost
~~~~~~~~~~~~~~~~~~~~~~~~~~

Cost models deciding which equivalent expression is extracted.

``evaluation`` counts only ``Num`` nodes: the egglog primitives inside a
literal (``BigRat``, ``BigInt``, ``String``, ...) are free, so a number costs
the same as a symbol.
"""

from __future__ import annotations

from typing import Any

from egglog import DagCostModel, EGraph, get_callable_fn

from ._convert import CONSTANTS_BACK, FUNCTIONS
from ._term import Num, abs, cos, exp, log, sin, sqrt, tan

LEAVES = (Num, Num.float, Num.var, Num.opaque, *CONSTANTS_BACK)
NUM_FUNCTIONS = frozenset((*LEAVES, Num.apply, *FUNCTIONS.values()))


def node_cost(costs: dict[Any, int], /, *, default: int) -> DagCostModel[Any]:
    """Additive cost model with a cost per ``Num`` constructor."""

    def marginal_cost(egraph: EGraph, expr: Any, /) -> int:
        fn = get_callable_fn(expr)
        if fn is None or fn not in NUM_FUNCTIONS:
            return 0
        return costs.get(fn, default)

    # (``int`` does not satisfy egglog's ``ComparableAdd`` protocol for type
    # checkers: typeshed's ``int.__add__`` has a positional-only parameter)
    return DagCostModel[Any](marginal_cost, 0)


#: Number of nodes, with ties broken towards conventional forms. ``None``
#: selects egglog's native extraction with the costs declared on the
#: constructors in ``_term``, which is much faster than a Python callback.
size = None

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
        abs: 2,
        sqrt: 10,
        Num.__pow__: 25,
        exp: 20,
        log: 20,
        sin: 20,
        cos: 20,
        tan: 25,
    },
    default=20,
)
