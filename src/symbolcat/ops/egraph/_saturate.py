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

from egglog import DagCostModel, EGraph, Schedule, back_off, eq, run, seq

from ..._core import Expr, as_expr
from . import _cost as cost
from . import _rules as rules
from ._canonical import canonical
from ._convert import Table, from_egglog, to_egglog


def shrink(passes: int = 50) -> Schedule:
    """The shrinking rules, until nothing changes or for ``passes`` passes.

    When the input is undefined, the rules can prove two different numbers
    equal (``_rules.contradictions`` records that as ``0 == 1``); constant
    folding would then keep creating new numbers forever, so each run also
    stops as soon as that holds.
    """
    stop = eq(rules.ZERO).to(rules.ONE)
    return seq(*(run(r, stop) for r in rules.SHRINK)) * passes


def grow(match_limit: int = 1000) -> Schedule:
    """One pass of the growing rules.

    Growing rules run under a back-off scheduler: a rule that matches more
    than ``match_limit`` times in an iteration is banned for a while, so a
    few explosive rules (commutativity, associativity) do not starve the
    rest. The scheduler is persistent, so bans carry over between passes.

    This is run separately from ``shrink``: egglog does not support the
    ``0 == 1`` stop condition in a schedule that uses a custom scheduler.
    """
    scheduler = back_off(match_limit=match_limit).persistent()
    return seq(*(run(r, scheduler=scheduler) for r in rules.GROW))


def size(egraph: EGraph) -> int:
    return sum(n for _, n in egraph.all_function_sizes())


def contradiction(egraph: EGraph) -> bool:
    """Whether the rules proved two different numbers equal, which means
    the input is undefined (a division by zero, a square root of a negative
    number, ...)."""
    return egraph.check_bool(eq(rules.ZERO).to(rules.ONE))


def saturate(
    expr: Any,
    *,
    iterations: int,
    max_nodes: int,
    cost_model: DagCostModel[Any] | None,
    extractor: Literal["tree", "greedy-dag"],
) -> Expr:
    """Grow the e-graph for ``iterations`` passes, or until it has more than
    ``max_nodes`` e-nodes, and extract the cheapest expression.

    An undefined input (such as a division by zero, or the square root of a
    negative number) is returned unchanged.
    """
    expr = as_expr(expr)
    table = Table()
    egraph = EGraph()
    root = egraph.let("root", to_egglog(expr, table))
    growing, shrinking = grow(), shrink()
    egraph.run(shrinking)
    for _ in range(iterations):
        if contradiction(egraph):
            return expr
        egraph.run(growing)
        egraph.run(shrinking)
        if size(egraph) > max_nodes:
            break
    if contradiction(egraph):
        return expr
    if cost_model is None:
        best = egraph.extract(root, extractor=extractor)
    else:
        best = egraph.extract(root, cost_model=cost_model, extractor=extractor)
    result = from_egglog(best, table)
    # Ordering products flattens them, which would undo the sharing that DAG
    # extraction found, so it is only applied to trees.
    return canonical(result) if extractor == "tree" else result


def simplify(expr: Any, iterations: int = 8, max_nodes: int = 20_000) -> Expr:
    """The smallest equivalent expression found.

    The e-graph grows for ``iterations`` passes of the rewrite rules, or
    until it holds more than ``max_nodes`` e-nodes.

    >>> from symbolcat import Symbol, real
    >>> x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    >>> print(simplify(x * y + x * z))
    x * (y + z)
    >>> print(simplify(real.sin(x) ** 2 + real.cos(x) ** 2 + 1))
    2
    >>> print(simplify((x + 1) ** 2 - x**2 - 2 * x))
    1
    """
    return saturate(
        expr,
        iterations=iterations,
        max_nodes=max_nodes,
        cost_model=cost.size,
        extractor="tree",
    )


def optimize(expr: Any, iterations: int = 8, max_nodes: int = 20_000) -> Expr:
    """The cheapest equivalent expression to evaluate that was found.

    ``iterations`` and ``max_nodes`` limit the search as in ``simplify``.

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
        max_nodes=max_nodes,
        cost_model=cost.evaluation,
        extractor="greedy-dag",
    )
