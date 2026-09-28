"""
symbolcat.ops._dae._initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consistent initialization.

The initial values of a DAE must satisfy every equation, and also the
hidden constraints: the derivatives of the equations that Pantelides'
algorithm differentiated. The initialization system has them all, in every
variable (the unknowns and their derivatives). Its degrees of freedom are
the initial values that can be chosen freely; fixing that many gives a
square system to solve.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..._core import as_expr
from .._algebraic._system import EquationSystem
from .._transform._substitute import substitute_variables
from ._pantelides import Pantelides


def initialization_system(
    result: Pantelides, fixed: Mapping[Any, Any] | None = None, /
) -> EquationSystem:
    """The equations the initial values must satisfy, given the ``fixed``
    ones (a mapping of variables, or the time, to values).

    >>> from symbolcat import D, Symbol
    >>> from symbolcat.ops import pantelides
    >>> x, y, L = Symbol("x"), Symbol("y"), Symbol("L")
    >>> result = pantelides([D(x) - y, x**2 - L**2], [x, y])
    >>> system = initialization_system(result)
    >>> system.degrees_of_freedom
    0
    >>> for eq in system.equations:
    ...     print(eq)
    D(x) - y
    x ** 2 - L ** 2
    2 * x * D(x)
    """
    fixed = {as_expr(k): v for k, v in (fixed or {}).items()}
    return EquationSystem(
        [substitute_variables(eq, fixed) for eq in result.equations],
        [v for v in result.variables if v not in fixed],
    )
