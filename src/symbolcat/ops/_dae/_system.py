"""
symbolcat.ops._dae._system
~~~~~~~~~~~~~~~~~~~~~~~~~~

Index-1 systems built from the result of Pantelides' algorithm.

An ``IndexOneDAE`` is ``F(t, states, unknowns) = 0``, where the Jacobian of
``F`` with respect to the unknowns is nonsingular: given the states, the
equations determine the unknowns, and with them the derivatives of the
states. Two constructions:

- ``lower_index``: the equations that were differentiated the most, solved
  for the highest derivatives. The other equations (the constraints) are
  dropped, so a numerical solution drifts away from them.
- ``dummy_derivatives``: every equation is kept. For each differentiated
  equation, one derivative becomes an algebraic unknown, a *dummy
  derivative*, and stops being the derivative of a state (Mattsson and
  Söderlind, "Index reduction in differential-algebraic equations using
  dummy derivatives", SIAM J. Sci. Comput. 14 (1993)).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..._core import Call, Expr, Literal, Symbol, cata
from ..._derivative import D, Variable
from .._analysis._inspect import free_symbols, variables
from .._calculus._diff import diff
from .._evaluate import evaluate
from ._pantelides import Pantelides


@dataclass(frozen=True)
class IndexOneDAE:
    """``equations`` (residuals, equal to zero) in the ``states``, which are
    integrated, and the ``unknowns``, which the equations determine given
    the states. The derivative of each state, ``D(state)``, is a state or an
    unknown. ``parameters`` are the other symbols, besides the time ``t``.
    """

    equations: tuple[Expr, ...]
    states: tuple[Variable, ...]
    unknowns: tuple[Variable, ...]
    parameters: tuple[Symbol, ...]
    t: Symbol | None = None

    def __post_init__(self) -> None:
        if len(self.equations) != len(self.unknowns):
            raise ValueError(
                f"{len(self.equations)} equations for {len(self.unknowns)} unknowns"
            )
        known = set(self.states) | set(self.unknowns)
        for state in self.states:
            if D(state) not in known:
                raise ValueError(f"the derivative of the state {state} is missing")


def _system(
    result: Pantelides,
    equations: Sequence[Expr],
    is_state: Sequence[bool],
) -> IndexOneDAE:
    states = tuple(v for v, s in zip(result.variables, is_state, strict=True) if s)
    unknowns = tuple(
        v for v, s in zip(result.variables, is_state, strict=True) if not s
    )
    known = set(result.variables)
    parameters = tuple(
        dict.fromkeys(
            s
            for eq in equations
            for s in free_symbols(eq)
            if s != result.t and s not in known
        )
    )
    return IndexOneDAE(tuple(equations), states, unknowns, parameters, result.t)


def lower_index(result: Pantelides, /) -> IndexOneDAE:
    """The highest equations, to solve for the highest derivatives.

    Every lower derivative is a state, so the constraints that were
    differentiated (the lower equations) are not enforced, and a numerical
    solution drifts away from them.

    >>> from symbolcat import D, Symbol
    >>> from symbolcat.ops import pantelides
    >>> x, y, k = Symbol("x"), Symbol("y"), Symbol("k")
    >>> dae = lower_index(pantelides([D(x) - y, x - k], [x, y]))
    >>> dae.states, dae.unknowns
    ((Symbol('x'),), (Symbol('y'), Call(D, (Symbol('x'),))))
    >>> for eq in dae.equations:
    ...     print(eq)
    D(x) - y
    D(x)
    """
    is_state = [d is not None for d in result.variable_derivative]
    return _system(result, result.highest_equations, is_state)


def dummy_derivatives(
    result: Pantelides,
    /,
    *,
    at: Mapping[Any, float] | None = None,
) -> IndexOneDAE:
    """Keep every equation, and make one derivative per differentiated
    equation a dummy: an algebraic unknown, instead of the derivative of a
    state.

    Level by level, from the most differentiated equations down, the
    dummies are chosen among the derivatives of the unknowns chosen at the
    level above, so that the Jacobian of the differentiated equations with
    respect to them is nonsingular. The choice is made by pivoting on that
    Jacobian evaluated ``at`` a point (a mapping of variables to values,
    such as the initial conditions). Variables not given take random
    values. The choice is only valid near that point.

    >>> from symbolcat import D, Symbol
    >>> from symbolcat.ops import pantelides
    >>> x, y, k = Symbol("x"), Symbol("y"), Symbol("k")
    >>> dae = dummy_derivatives(pantelides([D(x) - y, x - k], [x, y]))
    >>> dae.states, dae.unknowns
    ((), (Symbol('x'), Symbol('y'), Call(D, (Symbol('x'),))))
    """
    point = _Point(at or {})
    eq_below = _inverse(result.equation_derivative)
    var_below = _inverse(result.variable_derivative)
    rows = [
        e
        for e, d in enumerate(result.equation_derivative)
        if d is None and eq_below[e] is not None
    ]
    columns = [
        v
        for v, d in enumerate(result.variable_derivative)
        if d is None and var_below[v] is not None
    ]
    dummies: set[int] = set()
    while rows:
        matrix = [
            [
                point.value(diff(result.equations[e], result.variables[v]))
                for v in columns
            ]
            for e in rows
        ]
        chosen = [columns[c] for c in _pivot_columns(matrix)]
        dummies.update(chosen)
        rows = [
            below
            for e in rows
            if (below := eq_below[e]) is not None and eq_below[below] is not None
        ]
        columns = [
            below
            for v in chosen
            if (below := var_below[v]) is not None and var_below[below] is not None
        ]
    is_state = [
        d is not None and d not in dummies for d in result.variable_derivative
    ]
    return _system(result, result.equations, is_state)


def _inverse(derivative: Sequence[int | None], /) -> list[int | None]:
    """For each index, the index it is the derivative of, or ``None``."""
    below: list[int | None] = [None] * len(derivative)
    for i, d in enumerate(derivative):
        if d is not None:
            below[d] = i
    return below


class _Point:
    """Values of the variables, random where not given."""

    def __init__(self, values: Mapping[Any, float], /) -> None:
        self.values = dict(values)
        self.random = random.Random(0)

    def value(self, expr: Expr, /) -> float:
        mapping: dict[Expr, Expr] = {}
        for var in (*variables(expr), *free_symbols(expr)):
            if var not in self.values:
                self.values[var] = self.random.uniform(0.5, 1.5)
            mapping[var] = Literal(float(self.values[var]))
        return float(evaluate(rename(expr, mapping)))


def _pivot_columns(matrix: list[list[float]], /) -> list[int]:
    """Columns forming a nonsingular square submatrix, one per row: Gaussian
    elimination, pivoting on the largest entry of each row."""
    rows = [row[:] for row in matrix]
    scale = max((abs(a) for row in rows for a in row), default=0.0) or 1.0
    chosen: list[int] = []
    for r, row in enumerate(rows):
        candidates = [c for c in range(len(row)) if c not in chosen]
        c = max(candidates, key=lambda c: abs(row[c]), default=None)
        if c is None or abs(row[c]) <= 1e-12 * scale:
            raise ValueError(
                "dummy_derivatives: singular Jacobian; try another point `at`"
            )
        chosen.append(c)
        for other in rows[r + 1 :]:
            factor = other[c] / row[c]
            for j in range(len(other)):
                other[j] -= factor * row[j]
    return chosen


def rename(expr: Any, mapping: Mapping[Expr, Expr], /) -> Expr:
    """Replace sub-expressions, matching the original nodes: unlike
    ``substitute``, ``D(x)`` is not replaced inside ``D(D(x))``.

    >>> from symbolcat import D, Symbol
    >>> x, a, b = Symbol("x"), Symbol("a"), Symbol("b")
    >>> print(rename(D(D(x)) + D(x), {D(x): a, D(D(x)): b}))
    b + a
    """

    def algebra(layer: Expr, /) -> tuple[Expr, Expr]:
        if not isinstance(layer, Call):
            return layer, mapping.get(layer, layer)
        original = Call(layer.func, tuple(o for o, _ in layer.args))
        if original in mapping:
            return original, mapping[original]
        return original, Call(layer.func, tuple(r for _, r in layer.args))

    _, renamed = cata(algebra, expr)
    return renamed
