"""
symbolcat.ops._sde._system
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stochastic differential equations,
``dx_i = f_i dt + sum_j g_ij dW_j``, driven by independent Wiener processes.

In Itô's convention, functions of the state follow Itô's lemma, with a
second-order term; in Stratonovich's, the ordinary chain rule. The two
conventions describe the same process when the drifts differ by
``1/2 sum_jk g_kj dg_ij/dx_k``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..._core import Expr, Literal, Symbol, as_expr
from ...functions.calculus import Variable, is_variable
from ...ops._analysis._inspect import free_symbols
from ...ops._calculus._diff import diff
from ...ops._transform._simplify import simplify_exact

CONVENTIONS = ("ito", "stratonovich")


@dataclass(frozen=True, init=False)
class SDE:
    """``dx = drift[x] dt + sum_j diffusion[x][j] dW_j`` for each state.

    ``diffusion`` gives, for each state, one coefficient per Wiener
    process; a single expression means a single process. ``convention`` is
    ``"ito"`` or ``"stratonovich"``.

    >>> from symbolcat import Symbol
    >>> x, mu, sigma = Symbol("x"), Symbol("mu"), Symbol("sigma")
    >>> gbm = SDE({x: mu * x}, {x: sigma * x})
    >>> gbm.noise, gbm.parameters
    (1, (Symbol('mu'), Symbol('sigma')))
    """

    drift: Mapping[Variable, Expr]
    diffusion: Mapping[Variable, tuple[Expr, ...]]
    t: Symbol | None
    convention: str

    def __init__(
        self,
        drift: Mapping[Any, Any],
        diffusion: Mapping[Any, Any],
        /,
        t: Symbol | None = None,
        *,
        convention: str = "ito",
    ) -> None:
        if convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}")
        for state in drift:
            if not is_variable(state):
                raise TypeError(f"states must be symbols: {state}")
        if set(diffusion) - set(drift):
            raise ValueError("diffusion for a state without drift")
        columns = {
            s: tuple(map(as_expr, g)) if isinstance(g, Sequence) else (as_expr(g),)
            for s, g in diffusion.items()
        }
        noise = max((len(g) for g in columns.values()), default=0)
        if any(len(g) != noise for g in columns.values()):
            raise ValueError("every state needs one coefficient per Wiener process")
        zeros = (Literal(0),) * noise
        object.__setattr__(self, "drift", {s: as_expr(f) for s, f in drift.items()})
        object.__setattr__(self, "diffusion", {s: columns.get(s, zeros) for s in drift})
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "convention", convention)

    @property
    def states(self) -> tuple[Variable, ...]:
        return tuple(self.drift)

    @property
    def noise(self) -> int:
        """The number of Wiener processes."""
        return len(next(iter(self.diffusion.values()), ()))

    @property
    def parameters(self) -> tuple[Symbol, ...]:
        excluded = set(self.drift) | {self.t}
        exprs = [
            *self.drift.values(),
            *(g for gs in self.diffusion.values() for g in gs),
        ]
        return tuple(
            dict.fromkeys(
                s for e in exprs for s in free_symbols(e) if s not in excluded
            )
        )


def _correction(sde: SDE, /) -> dict[Variable, Expr]:
    """``1/2 sum_jk g_kj dg_ij/dx_k`` for each state ``i``."""
    out: dict[Variable, Expr] = {}
    for i in sde.states:
        total: Expr = Literal(0)
        for j in range(sde.noise):
            for k in sde.states:
                total = total + sde.diffusion[k][j] * diff(sde.diffusion[i][j], k)
        out[i] = total / 2
    return out


def ito_to_stratonovich(sde: SDE, /) -> SDE:
    """The same process, in Stratonovich's convention.

    >>> from symbolcat import Symbol
    >>> x, mu, sigma = Symbol("x"), Symbol("mu"), Symbol("sigma")
    >>> print(ito_to_stratonovich(SDE({x: mu * x}, {x: sigma * x})).drift[x])
    mu * x - sigma * x * sigma / 2
    """
    if sde.convention != "ito":
        raise ValueError("ito_to_stratonovich: the SDE is not in Itô's convention")
    correction = _correction(sde)
    return SDE(
        {s: simplify_exact(f - correction[s]) for s, f in sde.drift.items()},
        sde.diffusion,
        sde.t,
        convention="stratonovich",
    )


def stratonovich_to_ito(sde: SDE, /) -> SDE:
    """The same process, in Itô's convention."""
    if sde.convention != "stratonovich":
        raise ValueError(
            "stratonovich_to_ito: the SDE is not in Stratonovich's convention"
        )
    correction = _correction(sde)
    return SDE(
        {s: simplify_exact(f + correction[s]) for s, f in sde.drift.items()},
        sde.diffusion,
        sde.t,
        convention="ito",
    )


def ito_lemma(expr: Any, sde: SDE, /) -> tuple[Expr, tuple[Expr, ...]]:
    """The drift and diffusion coefficients of ``h = expr``, a function of
    the states (and time): ``dh = drift dt + sum_j diffusion[j] dW_j``.

    In Itô's convention, the drift has the second-order term
    ``1/2 sum_ik (g g^T)_ik d²h/dx_i dx_k``.

    >>> from symbolcat import Symbol, real
    >>> x, mu, sigma = Symbol("x"), Symbol("mu"), Symbol("sigma")
    >>> drift, (diffusion,) = ito_lemma(real.log(x), SDE({x: mu * x}, {x: sigma * x}))
    >>> print(drift, "|", diffusion)
    mu * x * (1 / x) + sigma * x * (sigma * x) * (-1 / x ** 2) / 2 | 1 / x * (sigma * x)
    """
    h = as_expr(expr)
    gradient = {i: diff(h, i) for i in sde.states}
    drift: Expr = diff(h, sde.t) if sde.t is not None else Literal(0)
    for i in sde.states:
        drift = drift + sde.drift[i] * gradient[i]
    if sde.convention == "ito":
        second: Expr = Literal(0)
        for i in sde.states:
            for k in sde.states:
                covariance: Expr = Literal(0)
                for j in range(sde.noise):
                    covariance = covariance + sde.diffusion[i][j] * sde.diffusion[k][j]
                second = second + covariance * diff(gradient[i], k)
        drift = drift + second / 2
    diffusion = tuple(
        simplify_exact(
            sum((gradient[i] * sde.diffusion[i][j] for i in sde.states), Literal(0))
        )
        for j in range(sde.noise)
    )
    return simplify_exact(drift), diffusion
