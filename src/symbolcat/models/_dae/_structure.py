"""
symbolcat.ops._dae._structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structural simplification of a DAE.

``eliminate_aliases`` removes the trivial equations: an unknown equal to a
multiple of another (``x = y``, ``x = -2 * y``) or to an expression of the
parameters (``x = k``). ``structural_simplify`` then reduces the index, with
Pantelides' algorithm and dummy derivatives. The index-1 system can be split
further with ``block_triangular`` and ``tear``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from ..._core import Expr, Literal, Symbol, value_of
from ...ops._algebraic._structure import solve_for
from ...ops._algebraic._system import EquationSystem, residual
from ...ops._analysis._inspect import variables
from ...ops._calculus._diff import diff
from ...ops._calculus._time import _base, expand_derivatives
from ...ops._transform._simplify import as_fraction, is_number, simplify_exact
from ...ops._transform._substitute import substitute, substitute_variables
from .._events import Event
from ._pantelides import pantelides
from ._system import IndexOneDAE, dummy_derivatives


def _is_nonzero_number(expr: Expr, /) -> bool:
    value = as_fraction(expr)
    if value is not None:
        return value != 0
    return (
        isinstance(expr, Literal)
        and type(value_of(expr)) is float
        and value_of(expr) != 0
    )


def _alias(
    eq: Expr, unknowns: set[Symbol], differentiated: set[Expr], /
) -> tuple[Symbol, Expr] | None:
    """The unknown that ``eq`` defines as an alias, and its value."""
    found = [v for v in variables(eq) if _base(v) in unknowns]
    if not found or len(found) > 2 or not all(isinstance(v, Symbol) for v in found):
        return None
    coefficients = [simplify_exact(diff(eq, v)) for v in found]
    if not all(_is_nonzero_number(c) for c in coefficients):
        return None
    rest = simplify_exact(substitute_variables(eq, dict.fromkeys(found, Literal(0))))
    if len(found) == 2 and not is_number(rest, 0):
        return None
    # Keep differentiated unknowns: they are candidate states.
    found.sort(key=lambda v: v in differentiated)
    u = found[0]
    assert isinstance(u, Symbol)
    value = solve_for(eq, u)
    assert value is not None
    return u, value


def eliminate_aliases(
    equations: Iterable[Any], unknowns: Iterable[Symbol], /, t: Symbol | None = None
) -> tuple[EquationSystem, dict[Symbol, Expr]]:
    """Remove the equations that make an unknown a numeric multiple of
    another one, or an expression of the parameters, and substitute that
    unknown everywhere.

    Returns the remaining system and the eliminated unknowns, with their
    values in terms of the remaining ones. Between two aliases, an unknown
    that appears differentiated is kept.

    >>> from symbolcat import D, Symbol
    >>> x, v, w, k = Symbol("x"), Symbol("v"), Symbol("w"), Symbol("k")
    >>> system, aliases = eliminate_aliases([D(x) - v, v - w, D(w) + k * x], [x, v, w])
    >>> [str(eq) for eq in system.equations], aliases
    (['D(x) - w', 'D(w) + k * x'], {Symbol('v'): Symbol('w')})
    """
    equations = [residual(eq) for eq in equations]
    unknowns = list(unknowns)
    aliases: dict[Symbol, Expr] = {}
    while True:
        remaining = set(unknowns)
        differentiated = {
            _base(v) for eq in equations for v in variables(eq) if v not in remaining
        }
        for i, eq in enumerate(equations):
            if (alias := _alias(eq, remaining, differentiated)) is not None:
                break
        else:
            break
        u, value = alias
        del equations[i]
        unknowns.remove(u)
        equations = [
            simplify_exact(expand_derivatives(substitute(e, {u: value}), unknowns, t))
            for e in equations
        ]
        aliases = {
            k: simplify_exact(substitute(a, {u: value})) for k, a in aliases.items()
        }
        aliases[u] = value
    return EquationSystem(equations, unknowns), aliases


def structural_simplify(
    equations: Iterable[Any],
    unknowns: Iterable[Symbol],
    /,
    t: Symbol | None = None,
    *,
    prefer: Iterable[Symbol] = (),
    events: Iterable[Event] = (),
) -> IndexOneDAE:
    """Eliminate the aliases, and reduce the index with Pantelides' algorithm
    and dummy derivatives. The eliminated unknowns are ``observed``.

    ``prefer`` lists unknowns to keep as states (see ``dummy_derivatives``).
    The aliases are substituted in the ``events``: an event can assign to
    an eliminated unknown only if it is an alias of another one (``x = y``).

    >>> from symbolcat import D, Symbol
    >>> x, v, w, k = Symbol("x"), Symbol("v"), Symbol("w"), Symbol("k")
    >>> dae = structural_simplify([D(x) - v, v - w, D(w) + k * x], [x, v, w])
    >>> dae.states, dae.observed
    ((Symbol('x'), Symbol('w')), {Symbol('v'): Symbol('w')})
    """
    system, aliases = eliminate_aliases(equations, unknowns, t)
    bases = [u for u in system.unknowns if isinstance(u, Symbol)]

    def eliminate(expr: Expr, /) -> Expr:
        return simplify_exact(expand_derivatives(substitute(expr, aliases), bases, t))

    events = [event.map(eliminate, aliases) for event in events]
    result = pantelides(system.equations, bases, t)
    dae = dummy_derivatives(result, prefer=prefer, events=events)
    return replace(dae, observed=aliases)
