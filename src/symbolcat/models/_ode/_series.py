"""
symbolcat.ops._ode._series
~~~~~~~~~~~~~~~~~~~~~~~~~~

Taylor series solutions of ODEs.

The ``k``-th derivative of a state along the solutions is its ``k``-th Lie
derivative, so ``x(t) = sum_k L^k(x)(t0) (t - t0)^k / k!``, with the states
in the coefficients standing for their initial values.
"""

from __future__ import annotations

from math import factorial
from typing import Any

from ..._core import Expr, Literal, Symbol, as_expr, value_of
from ..._derivative import Variable
from ...ops._transform._simplify import simplify_exact
from ...ops._transform._substitute import substitute_variables
from ._analysis import lie_derivative
from ._system import ODE


def taylor_solution(
    ode: ODE, order: int, /, t: Symbol | None = None, t0: Any = 0
) -> dict[Variable, Expr]:
    """The Taylor polynomial of degree ``order`` of each state around
    ``t0``, in the time ``t`` (defaults to ``ode.t``). In the coefficients,
    the states stand for their values at ``t0``.

    >>> from symbolcat import Symbol
    >>> x, v, t = Symbol("x"), Symbol("v"), Symbol("t")
    >>> solution = taylor_solution(ODE({x: v, v: -x}), 3, t)
    >>> print(solution[x])
    x + v * t + -x * t ** 2 / 2 + -v * t ** 3 / 6
    """
    time = t if t is not None else ode.t
    if time is None:
        raise ValueError("taylor_solution: pass the time symbol t")
    start = as_expr(t0)
    at_start: dict[Expr, Expr] = {} if ode.t is None else {ode.t: start}
    step = (
        time - start
        if not (isinstance(start, Literal) and value_of(start) == 0)
        else time
    )
    solution: dict[Variable, Expr] = {}
    for state in ode.states:
        term: Expr = state
        total: Expr = state
        for k in range(1, order + 1):
            term = lie_derivative(term, ode)
            coefficient = substitute_variables(term, at_start)
            total = total + coefficient * step**k / Literal(factorial(k))
        solution[state] = simplify_exact(total)
    return solution
