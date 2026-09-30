"""
symbolcat.ops._ode._analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analysis of explicit ODEs.

- ``lie_derivative``: the derivative of an expression along the solutions,
  a directional derivative (``jvp``) in the direction of the right-hand
  side. It is zero for a conserved quantity.
- ``steady_state``: the equations of the equilibria, ``rhs = 0``.
- ``linearize``: the Jacobians of the state-space form around a point.
- ``observability_matrix``: the gradients of the outputs and of their Lie
  derivatives, whose rank tells which states the outputs determine.
- ``transfer_function``: ``C (s I - A)^-1 B + D``, from a linearization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..._core import Call, Expr, Literal, Symbol, args_of
from ..._derivative import Variable
from ...linalg import matrix, to_list
from ...ops._algebraic._linear import solve_linear
from ...ops._algebraic._polynomial import cancel
from ...ops._algebraic._system import EquationSystem
from ...ops._calculus._diff import gradient, jacobian, jvp
from ...ops._transform._simplify import simplify_naive
from ...ops._transform._substitute import substitute_variables
from ._discrete import DiscreteSystem
from ._system import ODE


def lie_derivative(expr: Any, ode: ODE, /, order: int = 1) -> Expr:
    """The ``order``-th derivative of ``expr`` along the solutions of ``ode``.

    >>> from symbolcat import Symbol
    >>> x, v, k = Symbol("x"), Symbol("v"), Symbol("k")
    >>> ode = ODE({x: v, v: -k * x})
    >>> print(lie_derivative(v**2 + k * x**2, ode))
    2 * v * (-k * x) + k * (2 * x * v)
    """
    seed: dict[Expr, Expr] = {s: f for s, f in ode.rhs.items()}
    if ode.t is not None:
        seed[ode.t] = Literal(1)
    # Iterated, not unfolded: only the last derivative is kept.
    for _ in range(order):
        expr = jvp(expr, seed)
    return expr


def steady_state(ode: ODE, /) -> EquationSystem:
    """The equations of the equilibria, ``rhs = 0``, in the states.

    >>> from symbolcat import Symbol
    >>> x, a, b = Symbol("x"), Symbol("a"), Symbol("b")
    >>> [str(eq) for eq in steady_state(ODE({x: a - b * x})).equations]
    ['a - b * x']
    """
    return EquationSystem(ode.rhs.values(), ode.states)


@dataclass(frozen=True)
class Linearization:
    """``D(dx) = A dx + B du`` and ``dy = C dx + D du``, for deviations of the
    states ``dx``, inputs ``du`` and outputs ``dy``. ``B``, ``C`` and ``D``
    are ``None`` without inputs or outputs."""

    A: Call[Expr]
    B: Call[Expr] | None
    C: Call[Expr] | None
    D: Call[Expr] | None


def linearize(
    ode: ODE | DiscreteSystem,
    /,
    inputs: Sequence[Variable] = (),
    outputs: Sequence[Any] = (),
    *,
    at: Mapping[Any, Any] | None = None,
) -> Linearization:
    """The Jacobians of the right-hand side and of the ``outputs`` with
    respect to the states and the ``inputs`` (some of the parameters),
    evaluated ``at`` a point (a mapping of symbols to values), if given.

    For a ``DiscreteSystem``, ``A`` is the Jacobian of the update.

    >>> from symbolcat import Symbol, real
    >>> x, v, u, g = Symbol("x"), Symbol("v"), Symbol("u"), Symbol("g")
    >>> ode = ODE({x: v, v: -g * real.sin(x) + u})
    >>> lin = linearize(ode, [u], [x], at={x: 0})
    >>> print(lin.A, lin.B, lin.C, lin.D)
    matrix([[0, 1], [-g, 0]]) matrix([[0], [1]]) matrix([[1, 0]]) matrix([[0]])
    """

    def evaluate(m: Call[Expr], /) -> Call[Expr]:
        result = simplify_naive(m if at is None else substitute_variables(m, at))
        assert isinstance(result, Call)
        return result

    rhs = list(ode.rhs.values())
    states = list(ode.states)
    return Linearization(
        evaluate(jacobian(rhs, states)),
        evaluate(jacobian(rhs, list(inputs))) if inputs else None,
        evaluate(jacobian(list(outputs), states)) if outputs else None,
        evaluate(jacobian(list(outputs), list(inputs))) if inputs and outputs else None,
    )


def observability_matrix(ode: ODE, outputs: Sequence[Any], /) -> Call[Expr]:
    """The gradients, with respect to the states, of each output and of its
    Lie derivatives up to the number of states minus one.

    The states are locally observable from the outputs where it has full
    rank (the observability rank condition).

    >>> from symbolcat import Symbol
    >>> x, v, k = Symbol("x"), Symbol("v"), Symbol("k")
    >>> print(observability_matrix(ODE({x: v, v: -k * x}), [x]))
    matrix([[1, 0], [0, 1]])
    """
    rows = []
    for k in range(len(ode.states)):
        for h in outputs:
            derivative = lie_derivative(h, ode, order=k)
            rows.append(args_of(gradient(derivative, ode.states)))
    return matrix(rows)


def transfer_function(lin: Linearization, s: Symbol, /) -> Call[Expr]:
    """The transfer matrix ``C (s I - A)^-1 B + D`` from the inputs to the
    outputs, as rational functions of ``s`` without common factors.

    >>> from symbolcat import Symbol
    >>> x, v, u, k, c, s = (Symbol(n) for n in "x v u k c s".split())
    >>> oscillator = ODE({x: v, v: -k * x - c * v + u})
    >>> print(transfer_function(linearize(oscillator, [u], [x]), s))
    matrix([[1 / (s ** 2 + s * c + k)]])
    """
    if lin.B is None or lin.C is None or lin.D is None:
        raise ValueError(
            "transfer_function: the linearization needs inputs and outputs"
        )
    A, B, C, D = map(to_list, (lin.A, lin.B, lin.C, lin.D))
    n, m = len(A), len(B[0])
    unknowns = [Symbol(f"__tf_{i}") for i in range(n)]
    # (s I - A) X = B, one column of B at a time
    columns = []
    for j in range(m):
        equations = [
            sum(
                ((s if i == k else Literal(0)) - A[i][k]) * unknowns[k]
                for k in range(n)
            )
            - B[i][j]
            for i in range(n)
        ]
        solution = solve_linear(equations, unknowns)
        columns.append([solution[u] for u in unknowns])
    rows = [
        [
            cancel(sum((C[r][k] * columns[j][k] for k in range(n)), D[r][j]))
            for j in range(m)
        ]
        for r in range(len(C))
    ]
    return matrix(rows)
