"""
symbolcat.ops.egraph._saturate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simplify and optimize expressions by equality saturation.

The expression is added to an e-graph, the rewrite rules grow it with
equivalent forms, and the cheapest form according to a cost model is
extracted back.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from itertools import count
from threading import Lock
from typing import Any, Literal

from egglog import DagCostModel, EGraph, Schedule, back_off, eq, run, seq

from ..._core import Expr, as_expr
from . import _cost as cost
from . import _rules as rules
from ._assume import Assumption
from ._canonical import canonical
from ._convert import Table, from_egglog, to_egglog
from ._term import Num, is_integer, is_negative, is_nonnegative, is_nonzero, is_positive


def shrink(*, passes: int = 50) -> Schedule:
    """The shrinking rules, until nothing changes or for ``passes`` passes.

    When the input is undefined, the rules can prove two different numbers
    equal (``_rules.contradictions`` records that as ``0 == 1``); constant
    folding would then keep creating new numbers forever, so each run also
    stops as soon as that holds.
    """
    stop = eq(rules.ZERO).to(rules.ONE)
    return seq(*(run(r, stop) for r in rules.SHRINK)) * passes


def grow(*, match_limit: int = 1000) -> Schedule:
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


def size(egraph: EGraph, /) -> int:
    return sum(n for _, n in egraph.all_function_sizes())


def contradiction(egraph: EGraph, /) -> bool:
    """Whether the rules proved two different numbers equal, which means
    the input is undefined (a division by zero, a square root of a negative
    number, ...)."""
    return egraph.check_bool(eq(rules.ZERO).to(rules.ONE))


PROPERTIES = {
    "positive": is_positive,
    "nonnegative": is_nonnegative,
    "negative": is_negative,
    "nonzero": is_nonzero,
    "integer": is_integer,
}


@dataclass
class Search:
    """A saturated e-graph holding the expression being simplified."""

    egraph: EGraph
    root: Num
    table: Table

    def equivalent(self, other: Any, /) -> bool:
        """Whether the search proved ``other`` equal to the expression.

        Only the shrinking rules run on ``other`` itself (so that ``2 * x ** 0``
        matches ``2``), not the rules that search.
        """
        term = self.egraph.let(f"other{next(_names)}", to_egglog(other, self.table))
        self.egraph.run(shrink())
        return self.egraph.check_bool(eq(self.root).to(term))

    def extract(
        self,
        *,
        cost_model: DagCostModel[Any] | None,
        extractor: Literal["tree", "greedy-dag"],
    ) -> Expr:
        if cost_model is None:
            best = self.egraph.extract(self.root, extractor=extractor)
        else:
            best = self.egraph.extract(
                self.root, cost_model=cost_model, extractor=extractor
            )
        result = from_egglog(best, self.table)
        # Ordering products flattens them, which would undo the sharing that
        # DAG extraction found, so it is only applied to trees.
        return canonical(result) if extractor == "tree" else result


_names = count()


def search(
    expr: Any,
    /,
    *,
    assume: Iterable[Assumption] = (),
    iterations: int,
    max_nodes: int,
    match_limit: int = 1000,
    prepare_passes: int = 64,
    egraph: EGraph | None = None,
) -> Search | None:
    """Saturate an e-graph with ``expr``.

    First the ``PREPARE`` rules run alone, until nothing changes or for
    ``prepare_passes`` passes. Then the growing rules run for ``iterations``
    passes, or until the e-graph has more than ``max_nodes`` e-nodes.

    Returns ``None`` if the input is undefined: the rules then prove two
    different numbers equal (as for a division by zero, or contradictory
    assumptions).
    """
    table = Table()
    if egraph is None:
        egraph = EGraph()
    root = egraph.let(f"root{next(_names)}", to_egglog(expr, table))
    for assumption in assume:
        term = to_egglog(assumption.expr, table)
        egraph.register(PROPERTIES[assumption.property](term))
    shrinking = shrink()
    analysis = seq(*map(run, rules.ANALYSIS)).saturate()
    egraph.run(shrinking)
    egraph.run(analysis)
    preparing = seq(*map(run, rules.PREPARE))
    for _ in range(prepare_passes):
        report = egraph.run(preparing)
        egraph.run(shrinking)
        if not report.updated or size(egraph) > max_nodes:
            break
    growing = grow(match_limit=match_limit)
    for _ in range(iterations):
        if contradiction(egraph):
            return None
        egraph.run(growing)
        egraph.run(shrinking)
        egraph.run(analysis)
        if size(egraph) > max_nodes:
            break
    if contradiction(egraph):
        return None
    egraph.run(seq(*map(run, rules.COSTS)).saturate())
    return Search(egraph, root, table)


def saturate(
    expr: Any,
    /,
    *,
    assume: Iterable[Assumption],
    iterations: int,
    max_nodes: int,
    match_limit: int,
    cost_model: DagCostModel[Any] | None,
    extractor: Literal["tree", "greedy-dag"],
) -> Expr:
    """Search, and extract the cheapest expression.

    An undefined input is returned unchanged.
    """
    expr = as_expr(expr)
    with _lock:
        egraph = _shared()
        egraph.push()
        try:
            found = search(
                expr,
                assume=assume,
                iterations=iterations,
                max_nodes=max_nodes,
                match_limit=match_limit,
                egraph=egraph,
            )
            if found is None:
                return expr
            return found.extract(cost_model=cost_model, extractor=extractor)
        finally:
            egraph.pop()


_lock = Lock()


@cache
def _shared() -> EGraph:
    """An e-graph with every ruleset registered.

    Registering the rules costs more than most searches, so each search runs
    on this e-graph between ``push`` and ``pop`` instead of on a new one.
    """
    egraph = EGraph()
    others = rules.ANALYSIS + rules.PREPARE + rules.COSTS
    for schedule in (shrink(), grow(), seq(*map(run, others))):
        egraph.run(schedule)
    return egraph


def simplify(
    expr: Any,
    /,
    *,
    assume: Iterable[Assumption] = (),
    iterations: int = 8,
    max_nodes: int = 5_000,
    match_limit: int = 1_000,
) -> Expr:
    """The smallest equivalent expression found.

    Only rules valid wherever ``expr`` is defined are applied, unless
    ``assume`` enables more (see ``positive``, ``integer``, ...).

    The e-graph grows for ``iterations`` passes of the rewrite rules, or
    until it holds more than ``max_nodes`` e-nodes. A rule matching more than
    ``match_limit`` times in a pass is banned for a while (back-off). Deeper
    identities may need larger limits (see the last example).

    >>> from symbolcat import Symbol, real
    >>> x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    >>> print(simplify(x * y + x * z))
    x * (y + z)
    >>> print(simplify(real.sin(x) ** 2 + real.cos(x) ** 2 + 1))
    2
    >>> print(simplify((x + 1) ** 2 - x**2 - 2 * x))
    1
    >>> e = real.sin(x) ** 4 + real.cos(x) ** 4 + real.sin(2 * x) ** 2 / 2
    >>> print(simplify(e, max_nodes=10_000, match_limit=10_000))
    1
    """
    return saturate(
        expr,
        assume=assume,
        iterations=iterations,
        max_nodes=max_nodes,
        match_limit=match_limit,
        cost_model=cost.size,
        extractor="tree",
    )


def optimize(
    expr: Any,
    /,
    *,
    assume: Iterable[Assumption] = (),
    iterations: int = 8,
    max_nodes: int = 5_000,
    match_limit: int = 1_000,
) -> Expr:
    """The cheapest equivalent expression to evaluate that was found.

    ``assume``, ``iterations``, ``max_nodes`` and ``match_limit`` are as in
    ``simplify``.

    Uses DAG extraction: repeated sub-expressions are returned as the same
    object, so ``cata`` (and therefore ``evaluate``) computes them once.

    >>> from symbolcat import Symbol, real
    >>> x = Symbol("x")
    >>> expr = optimize(real.sin(x) ** 2 + real.sin(x))
    >>> from symbolcat.ops.algebras import cata
    >>> visited = []
    >>> _ = cata(visited.append, expr)
    >>> sum(getattr(node, "func", None) is real.sin for node in visited)
    1
    """
    return saturate(
        expr,
        assume=assume,
        iterations=iterations,
        max_nodes=max_nodes,
        match_limit=match_limit,
        cost_model=cost.evaluation,
        extractor="greedy-dag",
    )
