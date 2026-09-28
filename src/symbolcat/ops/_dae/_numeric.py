"""
symbolcat.ops._dae._numeric
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compile an index-1 DAE into the right-hand side of an ODE, ``du/dt = f(t, u)``,
for integrators such as ``scipy.integrate.solve_ivp``. Requires NumPy.

The residuals and their Jacobian with respect to the unknowns are compiled
to Python functions (``to_python``). Each evaluation of ``f`` solves the
equations for the unknowns by Newton's method, starting from the previous
solution, and reads the derivatives of the states from the states and the
unknowns. When the equations are linear in the unknowns (``lower_index``),
Newton's method takes a single step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..._core import Expr, Symbol
from ..._derivative import D
from ...linalg import vector
from .._calculus._diff import jacobian
from .._printing._python import to_python
from ._system import IndexOneDAE, rename


@dataclass
class ODEFunction:
    """The right-hand side ``f(t, u)`` of the ODE of an index-1 DAE.

    ``u`` holds the values of ``dae.states``, in order. ``solve(t, u)``
    gives the values of ``dae.unknowns``. The last solution is kept in
    ``guess``, as the starting point of the next Newton iteration.
    """

    dae: IndexOneDAE
    parameters: tuple[float, ...]
    guess: Any
    residual: Callable[..., Any] = field(repr=False)
    jacobian: Callable[..., Any] = field(repr=False)
    tol: float = 1e-10
    maxiter: int = 50
    # For each state, where its derivative is: (True, i) for states[i],
    # (False, i) for unknowns[i].
    _derivatives: list[tuple[bool, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        states = {s: i for i, s in enumerate(self.dae.states)}
        unknowns = {z: i for i, z in enumerate(self.dae.unknowns)}
        self._derivatives = [
            (True, states[D(s)]) if D(s) in states else (False, unknowns[D(s)])
            for s in self.dae.states
        ]

    def solve(self, t: float, u: Any, /) -> Any:
        """The unknowns, given the states ``u`` at time ``t``."""
        import numpy as np

        z = np.array(self.guess, dtype=float)
        for _ in range(self.maxiter):
            r = np.asarray(self.residual(t, *u, *z, *self.parameters), dtype=float)
            if np.max(np.abs(r), initial=0.0) <= self.tol:
                break
            J = np.asarray(self.jacobian(t, *u, *z, *self.parameters), dtype=float)
            z = z - np.linalg.solve(J, r)
        else:
            raise RuntimeError(f"Newton's method did not converge at t={t}")
        self.guess = z
        return z

    def __call__(self, t: float, u: Any, /) -> Any:
        import numpy as np

        z = self.solve(t, u)
        return np.array(
            [u[i] if is_state else z[i] for is_state, i in self._derivatives]
        )

    def state_vector(self, values: Mapping[Any, float], /) -> Any:
        """The values of the states, in order, from a mapping."""
        import numpy as np

        return np.array([values[s] for s in self.dae.states], dtype=float)


def to_ode(
    dae: IndexOneDAE,
    parameters: Mapping[Any, float] | None = None,
    /,
    *,
    guess: Mapping[Any, float] | None = None,
    tol: float = 1e-10,
    maxiter: int = 50,
) -> ODEFunction:
    """Compile ``dae`` into the right-hand side ``f(t, u)`` of an ODE.

    ``parameters`` gives the value of each of ``dae.parameters``. ``guess``
    gives starting values for the unknowns (zero if not given), for the
    first Newton iteration: for nonlinear equations, it selects the
    solution (the branch) to follow.

    >>> from symbolcat import D, Symbol
    >>> from symbolcat.ops import lower_index, pantelides
    >>> x, v, k = Symbol("x"), Symbol("v"), Symbol("k")
    >>> dae = lower_index(pantelides([D(x) - v, D(v) + k * x], [x, v]))
    >>> f = to_ode(dae, {k: 4.0})
    >>> f(0.0, [1.0, 0.0])
    array([ 0., -4.])
    """
    import numpy as np

    parameters = parameters or {}
    missing = [p.name for p in dae.parameters if p not in parameters]
    if missing:
        raise ValueError(f"to_ode: no value for {', '.join(missing)}")

    t = Symbol("_t")
    states = [Symbol(f"_s{i}") for i in range(len(dae.states))]
    unknowns = [Symbol(f"_z{i}") for i in range(len(dae.unknowns))]
    params = [Symbol(f"_p{i}") for i in range(len(dae.parameters))]
    mapping: dict[Expr, Expr] = {
        **dict(zip(dae.states, states, strict=True)),
        **dict(zip(dae.unknowns, unknowns, strict=True)),
        **dict(zip(dae.parameters, params, strict=True)),
    }
    if dae.t is not None:
        mapping[dae.t] = t
    equations = [rename(eq, mapping) for eq in dae.equations]
    args = [t, *states, *unknowns, *params]

    namespace: dict[str, Any] = dict(vars(np))
    exec(to_python(vector(equations), args=args, name="residual"), namespace)
    exec(
        to_python(jacobian(equations, unknowns), args=args, name="jacobian"),
        namespace,
    )
    return ODEFunction(
        dae,
        tuple(float(parameters[p]) for p in dae.parameters),
        np.array([(guess or {}).get(z, 0.0) for z in dae.unknowns], dtype=float),
        namespace["residual"],
        namespace["jacobian"],
        tol=tol,
        maxiter=maxiter,
    )
