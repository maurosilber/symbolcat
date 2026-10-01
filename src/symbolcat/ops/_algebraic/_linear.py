"""
symbolcat.ops._algebraic._linear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic solution of linear systems, by Gauss-Jordan elimination.

Pivots are taken among the entries that are not literally zero, preferring
numbers. An entry is taken as nonzero unless ``simplify_exact`` folds it
to zero, so the solution is valid where the pivots do not vanish.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..._core import Expr, Literal
from ...functions.calculus import Variable
from .._analysis._inspect import variables
from .._calculus._diff import diff
from .._transform._simplify import as_fraction, is_number, simplify_exact
from .._transform._substitute import substitute_variables
from ._system import EquationSystem


def solve_linear(
    equations: Iterable[Any], unknowns: Iterable[Variable], /
) -> dict[Variable, Expr]:
    """The solution of linear ``equations`` (residuals, or ``op.eq``) for
    the ``unknowns``, which can be symbols or ``D(x)``.

    Raises ``ValueError`` if the equations are not linear in the unknowns,
    or the system is singular.

    >>> from symbolcat import Symbol
    >>> x, y, a = Symbol("x"), Symbol("y"), Symbol("a")
    >>> solution = solve_linear([2 * x - a, y - x], [x, y])
    >>> print(solution[x], "|", solution[y])
    a / 2 | -(-1 / 2 * a)
    """
    system = EquationSystem(equations, unknowns)
    n = len(system.unknowns)
    if len(system.equations) != n:
        raise ValueError(
            f"solve_linear: {len(system.equations)} equations for {n} unknowns"
        )
    zero = {u: Literal(0) for u in system.unknowns}
    rows: list[list[Expr]] = []
    for eq in system.equations:
        row = [simplify_exact(diff(eq, u)) for u in system.unknowns]
        for entry in row:
            if any(v in zero for v in variables(entry)):
                raise ValueError(f"solve_linear: {eq} is not linear in the unknowns")
        rows.append([*row, simplify_exact(-substitute_variables(eq, zero))])

    for col in range(n):
        candidates = [r for r in range(col, n) if not is_number(rows[r][col], 0)]
        if not candidates:
            raise ValueError("solve_linear: the system is singular")
        pivot_row = min(
            candidates,
            key=lambda r: (
                as_fraction(rows[r][col]) is None,
                sum(not is_number(e, 0) for e in rows[r][:n]),
            ),
        )
        rows[col], rows[pivot_row] = rows[pivot_row], rows[col]
        pivot = rows[col]
        for r in range(n):
            if r == col or is_number(rows[r][col], 0):
                continue
            factor = simplify_exact(rows[r][col] / pivot[col])
            # The pivot row is zero before ``col``: those entries are kept.
            rows[r] = [
                a
                if k < col
                else Literal(0)
                if k == col
                else simplify_exact(a - factor * b)
                for k, (a, b) in enumerate(zip(rows[r], pivot, strict=True))
            ]
    return {
        u: simplify_exact(rows[i][n] / rows[i][i])
        for i, u in enumerate(system.unknowns)
    }
