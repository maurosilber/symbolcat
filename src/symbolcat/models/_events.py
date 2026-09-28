"""
symbolcat.ops._events
~~~~~~~~~~~~~~~~~~~~~

Events: changes to the state of a system, at times set by a condition.

- ``ContinuousEvent``: when ``condition`` crosses zero (a bouncing ball's
  height, ``y``), apply ``affect``.
- ``DiscreteEvent``: when the boolean ``condition`` holds at a check (at a
  time, ``t == 2``), apply ``affect``.

An ``affect`` maps variables (states or parameters) to their new values,
expressions of the values before the event. Events are carried through the
transformations of their system: when an unknown is eliminated or renamed,
it is substituted in the conditions and the new values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .._core import Expr, as_expr
from .._derivative import Variable, is_variable
from ..ops._algebraic._system import residual


def _affect(affect: Mapping[Any, Any], /) -> dict[Variable, Expr]:
    out: dict[Variable, Expr] = {}
    for var, value in affect.items():
        if not is_variable(var):
            raise TypeError(f"events can only assign to variables, not {var}")
        out[var] = as_expr(value)
    return out


def _map_affect(
    affect: Mapping[Variable, Expr],
    func: Callable[[Expr], Expr],
    rename: Mapping[Any, Expr],
    /,
) -> dict[Variable, Expr]:
    out: dict[Variable, Expr] = {}
    for var, value in affect.items():
        target = rename.get(var, var)
        if not is_variable(target):
            raise ValueError(
                f"an event assigns to {var}, which was eliminated as {target}"
            )
        out[target] = func(value)
    return out


@dataclass(frozen=True, init=False)
class ContinuousEvent:
    """When ``condition`` crosses zero, set the variables in ``affect`` to
    their new values. ``direction`` restricts the crossings: ``1`` upward,
    ``-1`` downward, ``0`` both. The condition can be ``op.eq(a, b)``.

    >>> from symbolcat import Symbol
    >>> y, v, e = Symbol("y"), Symbol("v"), Symbol("e")
    >>> bounce = ContinuousEvent(y, {v: -e * v}, direction=-1)
    >>> print(bounce.condition, bounce.affect[v])
    y -e * v
    """

    condition: Expr
    affect: Mapping[Variable, Expr]
    direction: int

    def __init__(
        self, condition: Any, affect: Mapping[Any, Any], /, direction: int = 0
    ) -> None:
        if direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0 or 1, not {direction}")
        object.__setattr__(self, "condition", residual(condition))
        object.__setattr__(self, "affect", _affect(affect))
        object.__setattr__(self, "direction", direction)

    def map(
        self, func: Callable[[Expr], Expr], rename: Mapping[Any, Expr] = {}, /
    ) -> ContinuousEvent:
        """Apply ``func`` to the condition and the new values, and ``rename``
        the assigned variables."""
        return ContinuousEvent(
            func(self.condition),
            _map_affect(self.affect, func, rename),
            self.direction,
        )


@dataclass(frozen=True, init=False)
class DiscreteEvent:
    """When the boolean ``condition`` holds at a check, set the variables
    in ``affect`` to their new values.

    >>> from symbolcat import Symbol
    >>> from symbolcat.lib import operators as op
    >>> t, k = Symbol("t"), Symbol("k")
    >>> event = DiscreteEvent(op.eq(t, 2), {k: 2 * k})
    >>> print(event.condition)
    t == 2
    """

    condition: Expr
    affect: Mapping[Variable, Expr]

    def __init__(self, condition: Any, affect: Mapping[Any, Any], /) -> None:
        object.__setattr__(self, "condition", as_expr(condition))
        object.__setattr__(self, "affect", _affect(affect))

    def map(
        self, func: Callable[[Expr], Expr], rename: Mapping[Any, Expr] = {}, /
    ) -> DiscreteEvent:
        """Apply ``func`` to the condition and the new values, and ``rename``
        the assigned variables."""
        return DiscreteEvent(
            func(self.condition), _map_affect(self.affect, func, rename)
        )


type Event = ContinuousEvent | DiscreteEvent
