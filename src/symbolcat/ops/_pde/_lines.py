"""
symbolcat.ops._pde._lines
~~~~~~~~~~~~~~~~~~~~~~~~~

The method of lines: discretize the spatial coordinate of a partial
differential equation on a uniform grid, leaving a system of differential
equations in time, one set of unknowns per grid point.

Spatial derivatives, ``partial(u, x)``, become central differences at the
interior points, of second order in the spacing. The boundary conditions
are equations at the ends of the interval, where the first derivatives
become one-sided differences. The result is a DAE (the boundary equations
are algebraic): ``structural_simplify`` eliminates Dirichlet conditions as
aliases.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ..._core import Expr, Literal, Symbol, as_expr, name_of
from ..._derivative import D, partial
from .._algebraic._system import EquationSystem, residual
from .._transform._simplify import simplify_exact
from .._transform._substitute import substitute_variables


def grid_symbol(u: Symbol, i: int, /) -> Symbol:
    """The unknown ``u`` at the grid point ``i``: ``u_3``."""
    return Symbol(f"{name_of(u)}_{i}")


def method_of_lines(
    equations: Iterable[Any],
    unknowns: Sequence[Symbol],
    x: Symbol,
    domain: tuple[Any, Any],
    n: int,
    /,
    *,
    left: Iterable[Any] = (),
    right: Iterable[Any] = (),
) -> EquationSystem:
    """Discretize ``x`` in ``domain`` with ``n`` grid points, ends included.

    ``equations`` hold at the interior points. ``left`` and ``right`` are
    the boundary conditions, equations in the unknowns and their first
    derivatives in ``x`` at each end, one per unknown and end.

    >>> from symbolcat import D, Symbol, partial
    >>> u, x, k = Symbol("u"), Symbol("x"), Symbol("k")
    >>> heat = D(u) - k * partial(partial(u, x), x)
    >>> system = method_of_lines([heat], [u], x, (0, 1), 4, left=[u], right=[u - 1])
    >>> for eq in system.equations:
    ...     print(eq)
    u_0
    D(u_1) - k * ((u_2 - 2 * u_1 + u_0) * 9)
    D(u_2) - k * ((u_3 - 2 * u_2 + u_1) * 9)
    u_3 - 1
    """
    if n < 3:
        raise ValueError("method_of_lines: at least 3 grid points")
    equations = [residual(eq) for eq in equations]
    left, right = [residual(eq) for eq in left], [residual(eq) for eq in right]
    a, b = as_expr(domain[0]), as_expr(domain[1])
    h = simplify_exact((b - a) / (n - 1))
    inv_h = simplify_exact(1 / h)
    inv_h2 = simplify_exact(inv_h**2)
    grid = {u: [grid_symbol(u, i) for i in range(n)] for u in unknowns}

    def at(i: int, first: str, /) -> dict[Expr, Expr]:
        """The replacements at grid point ``i``, with ``first`` differences
        (``"central"``, ``"forward"`` or ``"backward"``)."""
        mapping: dict[Expr, Expr] = {x: simplify_exact(a + i * h)}
        for u, us in grid.items():
            mapping[u] = us[i]
            mapping[D(u)] = D(us[i])
            if first == "central":
                mapping[partial(u, x)] = (us[i + 1] - us[i - 1]) * (inv_h / 2)
                mapping[partial(partial(u, x), x)] = (
                    us[i + 1] - 2 * us[i] + us[i - 1]
                ) * inv_h2
            elif first == "forward":
                mapping[partial(u, x)] = (us[i + 1] - us[i]) * inv_h
            else:
                mapping[partial(u, x)] = (us[i] - us[i - 1]) * inv_h
        return mapping

    def discretize(eq: Expr, mapping: dict[Expr, Expr], /) -> Expr:
        return simplify_exact(substitute_variables(eq, mapping))

    out = [discretize(eq, at(0, "forward")) for eq in left]
    for i in range(1, n - 1):
        mapping = at(i, "central")
        out += [discretize(eq, mapping) for eq in equations]
    out += [discretize(eq, at(n - 1, "backward")) for eq in right]
    return EquationSystem(out, [s for us in grid.values() for s in us])
