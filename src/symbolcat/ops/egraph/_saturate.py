"""
symbolcat.ops.egraph._saturate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simplify and optimize expressions by equality saturation.

The expression is added to an e-graph, the rewrite rules grow it with
equivalent forms, and the cheapest form according to a cost model is
extracted back.
"""

from __future__ import annotations

from typing import Any, Literal

from egglog import DagCostModel, EGraph, Schedule, back_off, run, seq

from ..._core import Expr
from . import _cost as cost
from . import _rules as rules
from ._convert import Table, from_egglog, to_egglog


def schedule(iterations: int, match_limit: int = 1000) -> Schedule:
    """Alternate the growing rules with the shrinking ones to saturation.

    Growing rules run under a back-off scheduler: a rule that matches more
    than ``match_limit`` times in an iteration is banned for a while, so a
    few explosive rules (commutativity, associativity) do not starve the
    rest.
    """
    scheduler = back_off(match_limit=match_limit)
    shrink = seq(*map(run, rules.SHRINK)).saturate()
    grow = seq(*(run(r, scheduler=scheduler) for r in rules.GROW))
    return seq(shrink, seq(grow, shrink) * iterations)


def saturate(
    expr: Any,
    *,
    iterations: int,
    cost_model: DagCostModel[Any],
    extractor: Literal["tree", "greedy-dag"],
) -> Expr:
    table = Table()
    egraph = EGraph()
    root = egraph.let("root", to_egglog(expr, table))
    egraph.run(schedule(iterations))
    best = egraph.extract(root, cost_model=cost_model, extractor=extractor)
    return from_egglog(best, table)


def simplify(expr: Any, iterations: int = 8) -> Expr:
    """The smallest equivalent expression found.

    >>> from symbolcat import Symbol, real
    >>> x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    >>> print(simplify(x * y + x * z))
    x * (y + z)
    >>> print(simplify(real.sin(x) ** 2 + real.cos(x) ** 2 + 1))
    2
    >>> print(simplify((x + 1) ** 2 - x**2 - 2 * x))
    1
    """
    return saturate(expr, iterations=iterations, cost_model=cost.size, extractor="tree")


def optimize(expr: Any, iterations: int = 8) -> Expr:
    """The cheapest equivalent expression to evaluate that was found.

    Uses DAG extraction: repeated sub-expressions are returned as the same
    object, so ``cata`` (and therefore ``evaluate``) computes them once.

    >>> from symbolcat import Symbol, real
    >>> x = Symbol("x")
    >>> expr = optimize(real.sin(x) ** 2 + real.sin(x))
    >>> print(expr)
    sin(x) + sin(x) * sin(x)
    >>> single, product = expr.args
    >>> product.args[0] is product.args[1] is single
    True
    """
    return saturate(
        expr,
        iterations=iterations,
        cost_model=cost.evaluation,
        extractor="greedy-dag",
    )
