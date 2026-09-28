"""
symbolcat.ops._algebraic._matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matching equations to unknowns in the bipartite incidence graph, by
augmenting paths.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

type Graph = Sequence[Sequence[int]]  # equation -> incident unknowns


def augment(
    eq: int,
    graph: Graph,
    assign: dict[int, int],
    allowed: Callable[[int], bool],
    vcolor: set[int],
    ecolor: set[int],
) -> bool:
    """Search an augmenting path from ``eq``, through ``allowed`` unknowns.

    On success, ``assign`` (unknown -> equation) is updated along the path.
    The equations and unknowns visited are colored. Unknowns are tried in
    the order of ``graph[eq]``, so earlier ones are preferred.
    """
    ecolor.add(eq)
    candidates = [v for v in graph[eq] if allowed(v)]
    for v in candidates:
        if v not in assign:
            assign[v] = eq
            return True
    for v in candidates:
        if v not in vcolor:
            vcolor.add(v)
            if augment(assign[v], graph, assign, allowed, vcolor, ecolor):
                assign[v] = eq
                return True
    return False


def perfect_matching(graph: Graph, n: int, /) -> dict[int, int] | None:
    """A matching of every equation to a distinct one of the ``n`` unknowns,
    as ``unknown -> equation``, or ``None`` if there is none.

    >>> perfect_matching([[0, 1], [0]], 2)
    {0: 1, 1: 0}
    >>> perfect_matching([[0], [0]], 2) is None
    True
    """
    if len(graph) != n:
        return None
    assign: dict[int, int] = {}
    for eq in range(len(graph)):
        if not augment(eq, graph, assign, lambda _: True, set(), set()):
            return None
    return assign
