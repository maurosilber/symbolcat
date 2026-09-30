"""
symbolcat.ops._ode._discrete
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Discrete-time systems, ``x[k + 1] = update[x](x[k])``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..._core import Expr, Symbol, as_expr
from ..._derivative import Variable, is_variable
from ...ops._algebraic._system import EquationSystem
from ...ops._analysis._inspect import free_symbols
from ...ops._transform._simplify import simplify_exact
from ...ops._transform._substitute import substitute_variables
from .._events import Event


@dataclass(frozen=True, init=False)
class DiscreteSystem:
    """``x[k + 1] = update[x]``, an expression of the current values.

    ``linearize`` applies, as for an ``ODE``: its ``A`` is the Jacobian of
    the update, whose eigenvalues inside the unit circle mean stability.

    >>> from symbolcat import Symbol
    >>> x, r = Symbol("x"), Symbol("r")
    >>> logistic = DiscreteSystem({x: r * x * (1 - x)})
    >>> logistic.parameters
    (Symbol('r'),)
    """

    update: Mapping[Variable, Expr]
    events: tuple[Event, ...]

    def __init__(
        self, update: Mapping[Any, Any], /, *, events: Iterable[Event] = ()
    ) -> None:
        for state in update:
            if not is_variable(state):
                raise TypeError(f"states must be symbols: {state}")
        object.__setattr__(self, "update", {s: as_expr(f) for s, f in update.items()})
        object.__setattr__(self, "events", tuple(events))

    @property
    def rhs(self) -> Mapping[Variable, Expr]:
        """The update, under the name ``linearize`` expects."""
        return self.update

    @property
    def states(self) -> tuple[Variable, ...]:
        return tuple(self.update)

    @property
    def parameters(self) -> tuple[Symbol, ...]:
        states = set(self.update)
        return tuple(
            dict.fromkeys(
                s
                for f in self.update.values()
                for s in free_symbols(f)
                if s not in states
            )
        )


def fixed_points(system: DiscreteSystem, /) -> EquationSystem:
    """The equations of the fixed points, ``update[x] = x``.

    >>> from symbolcat import Symbol
    >>> x, r = Symbol("x"), Symbol("r")
    >>> [str(eq) for eq in fixed_points(DiscreteSystem({x: r * x})).equations]
    ['r * x - x']
    """
    return EquationSystem([f - s for s, f in system.update.items()], system.states)


def iterate(system: DiscreteSystem, n: int, /) -> DiscreteSystem:
    """The update applied ``n`` times: ``x[k + n]`` from ``x[k]``.

    >>> from symbolcat import Symbol
    >>> x, a = Symbol("x"), Symbol("a")
    >>> print(iterate(DiscreteSystem({x: a * x + 1}), 2).update[x])
    a * (a * x + 1) + 1
    """
    if n < 1:
        raise ValueError("iterate: n must be at least 1")
    update = dict(system.update)
    # Iterated, not unfolded: only the last composition is kept.
    for _ in range(n - 1):
        update = {
            s: simplify_exact(substitute_variables(f, update))
            for s, f in system.update.items()
        }
    return DiscreteSystem(update, events=system.events)
