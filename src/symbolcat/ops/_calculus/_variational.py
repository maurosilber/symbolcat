"""
symbolcat.ops._calculus._variational
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equations of motion from a Lagrangian.

The Lagrangian is an expression of the coordinates ``q`` and their
velocities ``D(q)``, which ``diff`` treats as independent variables. The
Euler-Lagrange equation of each coordinate is then

    d/dt (∂L/∂D(q)) - ∂L/∂q = 0.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..._core import Expr, Symbol
from ..._derivative import D
from .._transform._simplify import simplify_exact
from ._diff import diff
from ._time import time_derivative


def euler_lagrange(
    lagrangian: Any, coordinates: Iterable[Symbol], /, t: Symbol | None = None
) -> list[Expr]:
    """The Euler-Lagrange equations (residuals) of ``lagrangian``, one per
    coordinate. ``t`` is the time symbol, if the Lagrangian depends on it.

    >>> from symbolcat import D, Symbol
    >>> x, m, k = Symbol("x"), Symbol("m"), Symbol("k")
    >>> (eq,) = euler_lagrange(m * D(x) ** 2 / 2 - k * x**2 / 2, [x])
    >>> print(eq)
    m * (2 * D(D(x))) / 2 - -(k * (2 * x) / 2)
    """
    coordinates = tuple(coordinates)
    return [
        simplify_exact(
            time_derivative(diff(lagrangian, D(q)), coordinates, t)
            - diff(lagrangian, q)
        )
        for q in coordinates
    ]
