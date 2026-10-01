"""
symbolcat.ops._ode._identifiability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local structural identifiability: which parameters (and initial states)
the outputs of a model determine, for generic values, locally.

The parameters are appended as constant states, ``D(p) = 0``, and a
quantity is identifiable when its column of the observability matrix of
the extended system is independent of the others: removing it lowers the
rank. The rank is generic (``generic_rank``), so the answer holds with high
probability. It needs rational right-hand sides and outputs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..._core import Literal, Symbol
from ...functions.calculus import Variable
from ...functions.linalg import to_list
from ...ops._algebraic._rank import generic_rank
from ._analysis import observability_matrix
from ._system import ODE


def identifiability(
    ode: ODE, outputs: Sequence[Any], parameters: Sequence[Symbol] | None = None, /
) -> dict[Variable, bool]:
    """Whether each state (its initial value) and each of the
    ``parameters`` (by default, all of them) is locally identifiable from
    the ``outputs``.

    >>> from symbolcat import Symbol
    >>> x, a, b = Symbol("x"), Symbol("a"), Symbol("b")
    >>> identifiability(ODE({x: -a * b * x}), [x])
    {Symbol('x'): True, Symbol('a'): False, Symbol('b'): False}
    >>> identifiability(ODE({x: -a * x}), [b * x], [a, b])
    {Symbol('x'): False, Symbol('a'): True, Symbol('b'): False}
    """
    if parameters is None:
        parameters = ode.parameters
    extended = ODE({**ode.rhs, **{p: Literal(0) for p in parameters}}, ode.t)
    rows = to_list(observability_matrix(extended, outputs))
    full = generic_rank(rows)
    result: dict[Variable, bool] = {}
    for j, var in enumerate(extended.states):
        without = [[e for k, e in enumerate(row) if k != j] for row in rows]
        result[var] = generic_rank(without) < full
    return result
