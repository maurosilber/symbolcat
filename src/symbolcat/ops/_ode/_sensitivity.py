"""
symbolcat.ops._ode._sensitivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forward sensitivity equations.

The sensitivity ``S = ∂x/∂p`` of the states with respect to a parameter
follows ``D(S) = ∂f/∂x S + ∂f/∂p``: the directional derivative (``jvp``) of
the right-hand side, with seed ``S`` for the states and 1 for ``p``.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..._core import Expr, Literal, Symbol, name_of
from .._calculus._diff import jvp
from ._system import ODE


def sensitivity_symbol(state: Symbol, parameter: Symbol, /) -> Symbol:
    """The symbol of the sensitivity of ``state`` to ``parameter``,
    ``dx_dk``."""
    return Symbol(f"d{name_of(state)}_d{name_of(parameter)}")


def sensitivities(ode: ODE, parameters: Iterable[Symbol], /) -> ODE:
    """``ode`` augmented with the sensitivities of each state to each of the
    ``parameters``, named by ``sensitivity_symbol`` (``dx_dk``).

    States must be symbols (see ``to_first_order``).

    >>> from symbolcat import Symbol
    >>> x, k = Symbol("x"), Symbol("k")
    >>> ode = sensitivities(ODE({x: -k * x}), [k])
    >>> print(ode.rhs[Symbol("dx_dk")])
    -x + -k * dx_dk
    """
    if ode.events:
        raise NotImplementedError(
            "sensitivities: events make the sensitivities jump; not supported"
        )
    states = []
    for state in ode.states:
        if not isinstance(state, Symbol):
            raise TypeError(
                f"sensitivities: the state {state} is not a symbol; "
                "use to_first_order first"
            )
        states.append(state)
    rhs: dict[Expr, Expr] = {s: f for s, f in ode.rhs.items()}
    for p in parameters:
        seed: dict[Expr, Expr] = {s: sensitivity_symbol(s, p) for s in states}
        seed[p] = Literal(1)
        for s in states:
            rhs[sensitivity_symbol(s, p)] = jvp(ode.rhs[s], seed)
    return ODE(rhs, ode.t)
