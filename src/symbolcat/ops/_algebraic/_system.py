"""
symbolcat.ops._algebraic._system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A system of equations in some unknowns.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ..._core import Call, Expr, args_of, as_expr, func_of
from ..._derivative import Variable, is_variable
from ...lib import operators as op
from .._analysis._inspect import variables


def residual(equation: Any, /) -> Expr:
    """``lhs - rhs`` for ``op.eq(lhs, rhs)``; any other expression is
    already a residual, equal to zero.

    >>> from symbolcat import Symbol
    >>> from symbolcat.lib import operators as op
    >>> x = Symbol("x")
    >>> print(residual(op.eq(x, 1)))
    x - 1
    """
    equation = as_expr(equation)
    if isinstance(equation, Call) and func_of(equation) is op.eq:
        lhs, rhs = args_of(equation)
        return lhs - rhs
    return equation


@dataclass(frozen=True, init=False)
class EquationSystem:
    """``equations`` (residuals, equal to zero) in the ``unknowns``.

    Unknowns are symbols or time derivatives of them, ``D(x)``. Every other
    symbol is a parameter. Equations can be given as ``op.eq(lhs, rhs)``.

    >>> from symbolcat import Symbol
    >>> x, y, k = Symbol("x"), Symbol("y"), Symbol("k")
    >>> system = EquationSystem([x + y - k, x - y], [x, y])
    >>> system.incidence
    ((0, 1), (0, 1))
    """

    equations: tuple[Expr, ...]
    unknowns: tuple[Variable, ...]

    def __init__(self, equations: Iterable[Any], unknowns: Iterable[Any], /):
        unknowns = tuple(unknowns)
        for u in unknowns:
            if not is_variable(u):
                raise TypeError(f"unknowns must be symbols or D(...) of them: {u}")
        if len(set(unknowns)) != len(unknowns):
            raise ValueError("repeated unknowns")
        object.__setattr__(self, "equations", tuple(map(residual, equations)))
        object.__setattr__(self, "unknowns", unknowns)

    @property
    def degrees_of_freedom(self) -> int:
        """How many more unknowns than equations."""
        return len(self.unknowns) - len(self.equations)

    @property
    def incidence(self) -> tuple[tuple[int, ...], ...]:
        """For each equation, the indices of the unknowns it contains."""
        index = {u: i for i, u in enumerate(self.unknowns)}
        return tuple(
            tuple(index[v] for v in variables(eq) if v in index)
            for eq in self.equations
        )
