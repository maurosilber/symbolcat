"""
symbolcat.ops._ode._delay
~~~~~~~~~~~~~~~~~~~~~~~~~

Delay differential equations: an ``ODE`` whose right-hand side contains
delayed states, ``delay(x, tau)``.

Linearized around an equilibrium, ``D(x) = A0 x + sum_k A_k x(t - tau_k)``,
and exponential solutions ``exp(s t)`` exist where the characteristic matrix
``s I - A0 - sum_k A_k exp(-s tau_k)`` is singular: its determinant is the
characteristic equation, whose roots decide the stability.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ... import real
from ..._core import Call, Expr, Literal, Symbol
from ...linalg import matrix
from .._calculus._diff import jvp
from .._calculus._time import delays
from .._transform._simplify import simplify_exact
from .._transform._substitute import substitute_variables
from ._system import ODE


def characteristic_matrix(
    ode: ODE, s: Symbol, /, *, at: Mapping[Any, Any] | None = None
) -> Call[Expr]:
    """``s I - A0 - sum_k A_k exp(-s tau_k)``, from the Jacobians of the
    right-hand side with respect to the states (``A0``) and to the delayed
    states (``A_k``), evaluated ``at`` a point (an equilibrium) if given.

    Without delays, it is ``s I - A``.

    >>> from symbolcat import Symbol, delay
    >>> x, k, tau, s = Symbol("x"), Symbol("k"), Symbol("tau"), Symbol("s")
    >>> print(characteristic_matrix(ODE({x: -k * delay(x, tau)}), s))
    matrix([[s - -k * exp(-s * tau)]])
    """
    states = list(ode.states)
    rhs = list(ode.rhs.values())
    point = dict(at or {})
    rows = []
    for i, f in enumerate(rhs):
        row = []
        for j, x in enumerate(states):
            entry: Expr = s if i == j else Literal(0)
            entry = entry - jvp(f, {x: Literal(1)})
            for node in delays(f):
                value, tau = node.args
                if value == x:
                    entry = entry - jvp(f, {node: Literal(1)}) * real.exp(-s * tau)
            if point:
                entry = substitute_variables(entry, point)
            row.append(simplify_exact(entry))
        rows.append(row)
    return matrix(rows)
