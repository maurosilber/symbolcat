"""
symbolcat.models.ode
~~~~~~~~~~~~~~~~~~~~

Explicit differential equations, ``D(x) = f(x)`` (with delays and events),
and discrete-time systems: building and transforming them.
"""

from ..ops._calculus._time import change_variables
from ..ops._calculus._variational import euler_lagrange
from ._events import ContinuousEvent, DiscreteEvent
from ._ode._discrete import DiscreteSystem, iterate
from ._ode._series import taylor_solution
from ._ode._system import ODE, solve_for_derivatives, to_first_order

__all__ = [
    "ODE",
    "ContinuousEvent",
    "DiscreteEvent",
    "DiscreteSystem",
    "change_variables",
    "euler_lagrange",
    "iterate",
    "solve_for_derivatives",
    "taylor_solution",
    "to_first_order",
]
