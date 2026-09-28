"""
symbolcat.ops._dae._system
~~~~~~~~~~~~~~~~~~~~~~~~~~

Index-1 systems built from the result of Pantelides' algorithm.

An ``IndexOneDAE`` is ``F(t, states, unknowns) = 0``, where the Jacobian of
``F`` with respect to the unknowns is (structurally) nonsingular: given the
states, the equations determine the unknowns, and with them the derivatives
of the states. Two constructions:

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

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..._core import Expr, Symbol
from ..._derivative import D, Variable
from .._algebraic._matching import augment, perfect_matching
from .._algebraic._system import EquationSystem
from .._analysis._inspect import free_symbols, variables
from .._calculus._time import _base
from .._events import Event
from ._pantelides import Pantelides


@dataclass(frozen=True)
class IndexOneDAE:
    """``equations`` (residuals, equal to zero) in the ``states``, which are
    integrated, and the ``unknowns``, which the equations determine given
    the states. The derivative of each state, ``D(state)``, is a state or an
    unknown. ``parameters`` are the other symbols, besides the time ``t``.
    ``observed`` holds unknowns that were eliminated, with their values.
    ``events`` can only assign to states and parameters.

    Raises ``ValueError`` if the equations cannot be matched to the unknowns.
    """

    equations: tuple[Expr, ...]
    states: tuple[Variable, ...]
    unknowns: tuple[Variable, ...]
    parameters: tuple[Symbol, ...]
    t: Symbol | None = None
    observed: Mapping[Symbol, Expr] = field(default_factory=dict)
    events: tuple[Event, ...] = ()

    def __post_init__(self) -> None:
        assignable = set(self.states) | set(self.parameters)
        for event in self.events:
            for var in event.affect:
                if var not in assignable:
                    raise ValueError(
                        f"an event assigns to {var}, which is not a state or a "
                        "parameter (prefer it as a state in dummy_derivatives)"
                    )
        known = set(self.states) | set(self.unknowns)
        for state in self.states:
            if D(state) not in known:
                raise ValueError(f"the derivative of the state {state} is missing")
        if len(self.equations) != len(self.unknowns):
            raise ValueError(
                f"{len(self.equations)} equations for {len(self.unknowns)} unknowns"
            )
        system = EquationSystem(self.equations, self.unknowns)
        if perfect_matching(system.incidence, len(self.unknowns)) is None:
            raise ValueError("the equations are structurally singular in the unknowns")


def _system(
    result: Pantelides,
    equations: Sequence[Expr],
    is_state: Sequence[bool],
    events: Iterable[Event],
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
    return IndexOneDAE(
        tuple(equations), states, unknowns, parameters, result.t, events=tuple(events)
    )


def lower_index(result: Pantelides, /, *, events: Iterable[Event] = ()) -> IndexOneDAE:
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
    return _system(result, result.highest_equations, is_state, events)


def dummy_derivatives(
    result: Pantelides,
    /,
    *,
    prefer: Iterable[Symbol] = (),
    events: Iterable[Event] = (),
) -> IndexOneDAE:
    """Keep every equation, and make one derivative per differentiated
    equation a dummy: an algebraic unknown, instead of the derivative of a
    state.

    Level by level, from the most differentiated equations down, the
    dummies are chosen among the derivatives of the unknowns chosen at the
    level above, by matching each differentiated equation to one of them.
    The matching is structural, so the Jacobian of those equations with
    respect to the dummies is nonsingular for generic values, but it may be
    singular somewhere: the choice is only valid away from there.

    The derivatives of the unknowns in ``prefer`` are chosen last, so those
    unknowns are kept as states when possible.

    >>> from symbolcat import D, Symbol
    >>> from symbolcat.ops import pantelides
    >>> x, y, k = Symbol("x"), Symbol("y"), Symbol("k")
    >>> dae = dummy_derivatives(pantelides([D(x) - y, x - k], [x, y]))
    >>> dae.states, dae.unknowns
    ((), (Symbol('x'), Symbol('y'), Call(D, (Symbol('x'),))))
    """
    prefer = set(prefer)
    incidence = [set(variables(eq)) for eq in result.equations]
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
        columns.sort(key=lambda v: _base(result.variables[v]) in prefer)
        graph = [
            [c for c, v in enumerate(columns) if result.variables[v] in incidence[e]]
            for e in rows
        ]
        assign: dict[int, int] = {}
        for r in range(len(rows)):
            if not augment(r, graph, assign, lambda _: True, set(), set()):
                raise ValueError("dummy_derivatives: structurally singular")
        chosen = [columns[c] for c in sorted(assign)]
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
    is_state = [d is not None and d not in dummies for d in result.variable_derivative]
    return _system(result, result.equations, is_state, events)


def _inverse(derivative: Sequence[int | None], /) -> list[int | None]:
    """For each index, the index it is the derivative of, or ``None``."""
    below: list[int | None] = [None] * len(derivative)
    for i, d in enumerate(derivative):
        if d is not None:
            below[d] = i
    return below
