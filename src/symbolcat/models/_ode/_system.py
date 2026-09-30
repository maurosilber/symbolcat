"""
symbolcat.ops._ode._system
~~~~~~~~~~~~~~~~~~~~~~~~~~

Explicit ODEs, ``D(state) = rhs[state]``.

``solve_for_derivatives`` builds one from an index-1 DAE whose equations are
linear in the unknowns (like those of ``lower_index``), and
``to_first_order`` renames the states that are derivatives, ``D(x)``, into
new symbols.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..._core import Call, Expr, Symbol, as_expr, name_of
from ..._derivative import D, Variable, is_variable
from ...ops._algebraic._linear import solve_linear
from ...ops._analysis._inspect import free_symbols
from ...ops._calculus._time import _base
from ...ops._transform._substitute import substitute_variables
from .._dae._system import IndexOneDAE
from .._events import Event


@dataclass(frozen=True, init=False)
class ODE:
    """``D(state) = rhs[state]`` for each state, in the order of ``rhs``.

    States are symbols, or derivatives of them (``D(x)``, from second-order
    equations). ``t`` is the time symbol, if ``rhs`` depends on it.
    ``events`` change the states (see ``ContinuousEvent``). The right-hand
    side can contain delayed states, ``delay(x, tau)``.

    >>> from symbolcat import Symbol
    >>> x, v, k = Symbol("x"), Symbol("v"), Symbol("k")
    >>> ode = ODE({x: v, v: -k * x})
    >>> ode.parameters
    (Symbol('k'),)
    >>> for eq in ode.equations:
    ...     print(eq)
    D(x) - v
    D(v) - -k * x
    """

    rhs: Mapping[Variable, Expr]
    t: Symbol | None
    events: tuple[Event, ...]

    def __init__(
        self,
        rhs: Mapping[Any, Any],
        /,
        t: Symbol | None = None,
        *,
        events: Iterable[Event] = (),
    ) -> None:
        for state in rhs:
            if not is_variable(state):
                raise TypeError(f"states must be symbols or D(...) of them: {state}")
        object.__setattr__(self, "rhs", {s: as_expr(f) for s, f in rhs.items()})
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "events", tuple(events))

    @property
    def states(self) -> tuple[Variable, ...]:
        return tuple(self.rhs)

    @property
    def parameters(self) -> tuple[Symbol, ...]:
        """The symbols in ``rhs`` that are not states, nor the time."""
        excluded = {_base(s) for s in self.rhs} | {self.t}
        return tuple(
            dict.fromkeys(
                s
                for f in self.rhs.values()
                for s in free_symbols(f)
                if s not in excluded
            )
        )

    @property
    def equations(self) -> tuple[Expr, ...]:
        """The residuals ``D(state) - rhs[state]``."""
        return tuple(D(s) - f for s, f in self.rhs.items())


def solve_for_derivatives(dae: IndexOneDAE, /) -> ODE:
    """The explicit ODE of an index-1 DAE whose equations are linear in the
    unknowns: solve them (``solve_linear``), and read the derivative of
    each state.

    >>> from symbolcat import D, Symbol
    >>> from symbolcat.models.dae import lower_index, pantelides
    >>> x, v, m, k = Symbol("x"), Symbol("v"), Symbol("m"), Symbol("k")
    >>> dae = lower_index(pantelides([D(x) - v, m * D(v) + k * x], [x, v]))
    >>> print(solve_for_derivatives(dae).rhs[v])
    -(k * x) / m
    """
    solution = solve_linear(dae.equations, dae.unknowns)
    states = set(dae.states)
    return ODE(
        {s: D(s) if D(s) in states else solution[D(s)] for s in dae.states},
        dae.t,
        events=dae.events,
    )


def to_first_order(ode: ODE, /) -> ODE:
    """Rename the states that are derivatives into new symbols: ``D(x)``
    into ``x_t``, ``D(D(x))`` into ``x_tt``.

    >>> from symbolcat import D, Symbol
    >>> x, k = Symbol("x"), Symbol("k")
    >>> ode = to_first_order(ODE({x: D(x), D(x): -k * x}))
    >>> for state, rhs in ode.rhs.items():
    ...     print(f"D({state}) = {rhs}")
    D(x) = x_t
    D(x_t) = -k * x
    """
    taken = {s for f in ode.rhs.values() for s in free_symbols(f)} | {
        s for s in ode.rhs if isinstance(s, Symbol)
    }
    names: dict[Expr, Expr] = {}
    for state in ode.rhs:
        if not isinstance(state, Call):
            continue
        order, base = 0, state
        while isinstance(base, Call):
            (base,) = base.args
            order += 1
        assert isinstance(base, Symbol)
        symbol = Symbol(f"{name_of(base)}_{'t' * order}", kind=state.kind)
        if symbol in taken:
            raise ValueError(f"to_first_order: {symbol.name} is already used")
        names[state] = symbol
    return ODE(
        {names.get(s, s): substitute_variables(f, names) for s, f in ode.rhs.items()},
        ode.t,
        events=[
            event.map(lambda e: substitute_variables(e, names), names)
            for event in ode.events
        ],
    )
