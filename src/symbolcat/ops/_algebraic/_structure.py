"""
symbolcat.ops._algebraic._structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The structure of a system of equations.

- ``block_triangular``: split a system into the smallest blocks that can be
  solved one after the other (the block lower triangular, or BLT, form).
  Matching each equation to an unknown orients the incidence graph: an
  equation depends on the equations that compute its other unknowns. The
  blocks are its strongly connected components, in dependency order.
- ``tear``: within a block, solve as many equations as possible explicitly,
  one unknown each, given a few *tearing* unknowns. What is left is a
  smaller system: the residual equations in the tearing unknowns.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..._core import Expr, Literal
from ...functions.calculus import Variable
from .._analysis._inspect import variables
from .._calculus._diff import diff
from .._transform._simplify import is_number, simplify_exact
from .._transform._substitute import substitute_variables
from ._matching import perfect_matching
from ._system import EquationSystem


def block_triangular(system: EquationSystem, /) -> tuple[EquationSystem, ...]:
    """The blocks of ``system``, in the order they can be solved: each block
    only needs the unknowns of the blocks before it.

    Raises ``ValueError`` if the system is not square, or is structurally
    singular.

    >>> from symbolcat import Symbol
    >>> x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    >>> system = EquationSystem([z - x * y, x + y - 3, x - y - 1], [x, y, z])
    >>> for block in block_triangular(system):
    ...     print([str(eq) for eq in block.equations], block.unknowns)
    ['x + y - 3', 'x - y - 1'] (Symbol('x'), Symbol('y'))
    ['z - x * y'] (Symbol('z'),)
    """
    incidence = system.incidence
    matching = perfect_matching(incidence, len(system.unknowns))
    if matching is None:
        raise ValueError(
            f"block_triangular: {len(system.equations)} equations "
            f"for {len(system.unknowns)} unknowns, or structurally singular"
        )
    unknown_of = {eq: v for v, eq in matching.items()}
    depends = [
        sorted({matching[v] for v in incidence[eq]} - {eq})
        for eq in range(len(system.equations))
    ]
    return tuple(
        EquationSystem(
            [system.equations[eq] for eq in component],
            [system.unknowns[unknown_of[eq]] for eq in component],
        )
        for component in _strongly_connected(depends)
    )


def _strongly_connected(successors: list[list[int]], /) -> list[list[int]]:
    """Tarjan's algorithm, iterative. A component is emitted after every
    component it reaches: dependencies first."""
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    stack: list[int] = []
    on_stack: set[int] = set()
    components: list[list[int]] = []

    def visit(node: int, /) -> None:
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)

    for root in range(len(successors)):
        if root in index:
            continue
        visit(root)
        work = [(root, iter(successors[root]))]
        while work:
            node, pending = work[-1]
            for successor in pending:
                if successor not in index:
                    visit(successor)
                    work.append((successor, iter(successors[successor])))
                    break
                if successor in on_stack:
                    low[node] = min(low[node], index[successor])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component: list[int] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    components.append(sorted(component))
    return components


@dataclass(frozen=True)
class TornSystem:
    """The result of ``tear``.

    Given values of the ``tears``, each of ``solved`` (an unknown and its
    value) is computed in order, from the tears and the previous ones.
    The ``residuals`` are then equations in the tears alone.
    """

    solved: tuple[tuple[Variable, Expr], ...]
    residuals: tuple[Expr, ...]
    tears: tuple[Variable, ...]


def solve_for(equation: Expr, unknown: Variable, /) -> Expr | None:
    """The value of ``unknown`` if ``equation`` is linear in it, with a
    coefficient free of it, else ``None``.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(solve_for(2 * x - y, x))
    y / 2
    >>> solve_for(x**2 - y, x) is None
    True
    """
    coefficient = simplify_exact(diff(equation, unknown))
    if is_number(coefficient, 0) or unknown in variables(coefficient):
        return None
    rest = substitute_variables(equation, {unknown: Literal(0)})
    return simplify_exact(-rest / coefficient)


def tear(system: EquationSystem, /) -> TornSystem:
    """Solve the equations of ``system`` explicitly, one unknown at a time,
    choosing tearing unknowns where no equation can be solved.

    An equation is solved for an unknown when it is linear in it (with a
    coefficient free of it), and every other unknown it contains is known
    (a tear, or solved before). When no equation can be solved, the unknown
    appearing in the most remaining equations becomes a tear.

    >>> from symbolcat import Symbol
    >>> x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    >>> system = EquationSystem([y - 2 * x, z - x - y, x * z - 1], [x, y, z])
    >>> torn = tear(system)
    >>> torn.tears
    (Symbol('x'),)
    >>> for unknown, value in torn.solved:
    ...     print(unknown, "=", value)
    y = 2 * x
    z = -(-x - y)
    >>> print(torn.residuals[0])
    x * z - 1
    """
    unknowns = set(system.unknowns)
    remaining = list(system.equations)
    known: set[Variable] = set()
    solved: list[tuple[Variable, Expr]] = []
    tears: list[Variable] = []

    def unknown_in(eq: Expr, /) -> list[Variable]:
        return [v for v in variables(eq) if v in unknowns and v not in known]

    while len(known) < len(unknowns):
        for eq in remaining:
            found = unknown_in(eq)
            if len(found) != 1:
                continue
            (unknown,) = found
            value = solve_for(eq, unknown)
            if value is not None:
                solved.append((unknown, value))
                known.add(unknown)
                remaining.remove(eq)
                break
        else:
            counts = {v: 0 for v in system.unknowns if v not in known}
            for eq in remaining:
                for v in unknown_in(eq):
                    counts[v] += 1
            tear_ = max(counts, key=lambda v: counts[v])
            tears.append(tear_)
            known.add(tear_)
    return TornSystem(tuple(solved), tuple(remaining), tuple(tears))
