"""
symbolcat.ops._dae._pantelides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pantelides' algorithm: find the equations to differentiate, and how many
times, so that a DAE can be solved for its highest derivatives.

The algorithm is structural: it only looks at which variables appear in
which equation (the incidence graph), and matches each equation to one of
its highest derivatives by augmenting paths. When an equation cannot be
matched, every equation and variable visited by the failed search is
differentiated, and the search starts again from the derivative.

Reference: C. C. Pantelides, "The consistent initialization of
differential-algebraic systems", SIAM J. Sci. Stat. Comput. 9 (1988).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..._core import Call, Expr, Symbol, args_of
from ..._derivative import D, Variable, is_variable
from ...ops._algebraic._matching import Graph, augment, perfect_matching
from ...ops._algebraic._system import residual
from ...ops._analysis._inspect import variables
from ...ops._calculus._time import _base, time_derivative


@dataclass(frozen=True)
class Pantelides:
    """The result of ``pantelides``.

    ``equations`` holds the original equations, followed by the derivatives
    that were added; ``equation_derivative[i]`` is the index of the
    derivative of equation ``i``, or ``None``. Likewise for ``variables``,
    the unknowns and their derivatives, and ``variable_derivative``.
    ``matching`` maps each highest derivative to the index of its equation,
    and ``t`` is the time symbol, if the equations depend on it explicitly.
    """

    equations: tuple[Expr, ...]
    equation_derivative: tuple[int | None, ...]
    variables: tuple[Variable, ...]
    variable_derivative: tuple[int | None, ...]
    matching: Mapping[Variable, int]
    t: Symbol | None = None

    @property
    def highest_equations(self) -> tuple[Expr, ...]:
        """The equations that were not differentiated further: the system
        to solve for the highest derivatives."""
        return tuple(
            eq
            for eq, d in zip(self.equations, self.equation_derivative, strict=True)
            if d is None
        )

    @property
    def highest_variables(self) -> tuple[Variable, ...]:
        """The highest derivative of each unknown."""
        return tuple(
            var
            for var, d in zip(self.variables, self.variable_derivative, strict=True)
            if d is None
        )

    @property
    def differentiations(self) -> tuple[int, ...]:
        """How many times each original equation was differentiated."""
        n = len(self.equations) - sum(d is not None for d in self.equation_derivative)
        out = []
        for i in range(n):
            count = 0
            while (i := self.equation_derivative[i]) is not None:
                count += 1
            out.append(count)
        return tuple(out)


def _check_nonsingular(graph: Graph, n: int, /) -> None:
    """Pantelides terminates if the equations can be matched to the unknowns,
    identifying each unknown with its derivatives."""
    if len(graph) != n:
        raise ValueError(f"pantelides: {len(graph)} equations for {n} unknowns")
    if perfect_matching(graph, n) is None:
        raise ValueError("pantelides: the system is structurally singular")


def pantelides(
    equations: Iterable[Any], unknowns: Iterable[Symbol], /, t: Symbol | None = None
) -> Pantelides:
    """Differentiate the equations of a DAE until it can be solved for the
    highest derivatives of its unknowns.

    ``equations`` are residuals (equal to zero) or ``op.eq(lhs, rhs)``.
    ``unknowns`` are the symbols that depend on time, and ``t`` is the time
    symbol, if the equations depend on it explicitly. Derivatives are
    written with ``D``.

    >>> from symbolcat import D, Symbol, real
    >>> x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
    >>> result = pantelides([D(x) - y, x - real.sin(t)], [x, y], t)
    >>> result.differentiations
    (0, 1)
    >>> for eq in result.highest_equations:
    ...     print(eq)
    D(x) - y
    D(x) - cos(t)
    """
    unknowns = tuple(unknowns)
    bases = set(unknowns)
    if len(bases) != len(unknowns):
        raise ValueError("pantelides: repeated unknowns")

    vars: list[Variable] = []
    index: dict[Variable, int] = {}
    var_diff: list[int | None] = []

    def add_var(var: Variable) -> int:
        if var in index:
            return index[var]
        below = None
        if isinstance(var, Call):
            (inner,) = args_of(var)
            assert is_variable(inner)
            below = add_var(inner)
        index[var] = len(vars)
        vars.append(var)
        var_diff.append(None)
        if below is not None:
            var_diff[below] = index[var]
        return index[var]

    eqs: list[Expr] = []
    graph: list[list[int]] = []
    eq_diff: list[int | None] = []

    def add_eq(eq: Expr) -> int:
        incident = [add_var(v) for v in variables(eq) if _base(v) in bases]
        graph.append(incident)
        eqs.append(eq)
        eq_diff.append(None)
        return len(eqs) - 1

    for u in unknowns:
        add_var(u)
    for eq in equations:
        add_eq(residual(eq))
    n = len(eqs)

    base_index: dict[Expr, int] = {u: i for i, u in enumerate(unknowns)}
    _check_nonsingular(
        [sorted({base_index[_base(vars[v])] for v in vs}) for vs in graph],
        len(unknowns),
    )

    assign: dict[int, int] = {}  # variable -> equation
    for k in range(n):
        i = k
        while True:
            vcolor: set[int] = set()
            ecolor: set[int] = set()

            def highest(v: int, /) -> bool:
                return var_diff[v] is None

            if augment(i, graph, assign, highest, vcolor, ecolor):
                break
            for j in sorted(vcolor):
                add_var(D(vars[j]))
            for e in sorted(ecolor):
                eq_diff[e] = add_eq(time_derivative(eqs[e], unknowns, t))
            for j in vcolor:
                derivative, source = var_diff[j], eq_diff[assign[j]]
                assert derivative is not None and source is not None
                assign[derivative] = source
            next_eq = eq_diff[i]
            assert next_eq is not None
            i = next_eq

    matching = {vars[v]: e for v, e in sorted(assign.items()) if var_diff[v] is None}
    return Pantelides(
        equations=tuple(eqs),
        equation_derivative=tuple(eq_diff),
        variables=tuple(vars),
        variable_derivative=tuple(var_diff),
        matching=matching,
        t=t,
    )
