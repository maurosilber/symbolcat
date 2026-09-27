"""
symbolcat.ops.egraph.saturate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simplify and optimize expressions by equality saturation.

The expression is added to an e-graph, the rewrite rules grow it with
equivalent forms, and the cheapest form according to a cost model is
extracted back.
"""

from __future__ import annotations

from typing import Any, Literal

from egglog import DagCostModel, EGraph, Schedule, run, seq

from ...core import Expr
from . import cost, rules
from .convert import Table, from_egglog, to_egglog


def schedule(iterations: int) -> Schedule:
    """Alternate the growing rules with the shrinking ones to saturation."""
    shrink = seq(run(rules.fold), run(rules.identities)).saturate()
    return seq(shrink, seq(run(rules.algebra), shrink) * iterations)


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


def simplify(expr: Any, iterations: int = 5) -> Expr:
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


def optimize(expr: Any, iterations: int = 5) -> Expr:
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
