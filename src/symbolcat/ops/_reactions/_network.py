"""
symbolcat.ops._reactions._network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chemical reaction networks, as in Catalyst.jl.

A ``Reaction`` turns substrates into products at a rate. By mass action, the
rate law is the rate constant times the product of the substrate
concentrations, each raised to its stoichiometry; with ``combinatoric``
rate laws (the default, as in Catalyst), also divided by the factorial of
the stoichiometry, counting the distinct ways the molecules can meet.

From a network:

- ``reaction_rate_equations``: the deterministic ODE,
  ``D(s) = sum_r N[s, r] v_r``, with the stoichiometry matrix ``N``.
- ``chemical_langevin``: the chemical Langevin SDE, one Wiener process per
  reaction, with diffusion ``N[s, r] sqrt(v_r)``.
- ``propensities``: the jump rates of the stochastic (Gillespie) model,
  with falling factorials of the molecule counts.
- ``conservation_laws``: linear combinations of the species that are
  constant, from the left null space of ``N``.
- ``deficiency``, ``linkage_classes``, ``is_weakly_reversible``: the
  structure of the graph of complexes (Feinberg's network theory).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import factorial
from typing import Any

from ... import real
from ..._core import Call, Expr, Literal, Symbol, as_expr
from ...linalg import matrix
from .._algebraic._rank import rational_nullspace, rational_rank
from .._algebraic._structure import _strongly_connected
from .._analysis._inspect import free_symbols
from .._transform._simplify import simplify_exact
from .._ode._system import ODE
from .._sde._system import SDE

type Complex = frozenset[tuple[Symbol, int]]


def _stoichiometry(species: Mapping[Symbol, int] | Iterable[Symbol], /) -> dict[Symbol, int]:
    if isinstance(species, Mapping):
        out = {s: int(n) for s, n in species.items() if n}
    else:
        out = {}
        for s in species:
            out[s] = out.get(s, 0) + 1
    for s, n in out.items():
        if not isinstance(s, Symbol) or n < 0:
            raise ValueError(f"invalid species or stoichiometry: {s}, {n}")
    return out


@dataclass(frozen=True, init=False)
class Reaction:
    """``substrates -> products`` at ``rate``.

    Substrates and products are mappings from species to stoichiometries,
    or sequences of species (repeated for higher stoichiometries); empty for
    creation from, or degradation to, nothing. With ``mass_action=False``,
    ``rate`` is the whole rate law.

    >>> from symbolcat import Symbol
    >>> A, B, k = Symbol("A"), Symbol("B"), Symbol("k")
    >>> dimerization = Reaction(k, [A, A], [B])
    >>> print(dimerization)
    2 A -> B, k
    """

    rate: Expr
    substrates: Mapping[Symbol, int]
    products: Mapping[Symbol, int]
    mass_action: bool

    def __init__(
        self,
        rate: Any,
        substrates: Mapping[Symbol, int] | Iterable[Symbol],
        products: Mapping[Symbol, int] | Iterable[Symbol],
        /,
        *,
        mass_action: bool = True,
    ) -> None:
        object.__setattr__(self, "rate", as_expr(rate))
        object.__setattr__(self, "substrates", _stoichiometry(substrates))
        object.__setattr__(self, "products", _stoichiometry(products))
        object.__setattr__(self, "mass_action", mass_action)

    def net(self, species: Symbol, /) -> int:
        """The net change of ``species`` when the reaction occurs once."""
        return self.products.get(species, 0) - self.substrates.get(species, 0)

    def __str__(self) -> str:
        def side(complex_: Mapping[Symbol, int], /) -> str:
            terms = [f"{n} {s}" if n != 1 else str(s) for s, n in complex_.items()]
            return " + ".join(terms) or "∅"

        return f"{side(self.substrates)} -> {side(self.products)}, {self.rate}"


@dataclass(frozen=True, init=False)
class ReactionSystem:
    """A network of ``reactions`` among ``species`` (by default, in order of
    appearance). ``t`` is the time symbol, if the rates depend on it.
    """

    reactions: tuple[Reaction, ...]
    species: tuple[Symbol, ...]
    t: Symbol | None

    def __init__(
        self,
        reactions: Iterable[Reaction],
        species: Iterable[Symbol] | None = None,
        /,
        t: Symbol | None = None,
    ) -> None:
        reactions = tuple(reactions)
        if species is None:
            species = dict.fromkeys(
                s for r in reactions for s in (*r.substrates, *r.products)
            )
        object.__setattr__(self, "reactions", reactions)
        object.__setattr__(self, "species", tuple(species))
        object.__setattr__(self, "t", t)

    @property
    def parameters(self) -> tuple[Symbol, ...]:
        excluded = set(self.species) | {self.t}
        return tuple(
            dict.fromkeys(
                s for r in self.reactions for s in free_symbols(r.rate) if s not in excluded
            )
        )


def rate_law(reaction: Reaction, /, *, combinatoric: bool = True) -> Expr:
    """The rate of ``reaction``, in concentrations.

    >>> from symbolcat import Symbol
    >>> A, k = Symbol("A"), Symbol("k")
    >>> print(rate_law(Reaction(k, [A, A], [])))
    k * A ** 2 / 2
    """
    if not reaction.mass_action:
        return reaction.rate
    law: Expr = reaction.rate
    for species, n in reaction.substrates.items():
        law = law * (species if n == 1 else species**n)
    denominator = 1
    if combinatoric:
        for n in reaction.substrates.values():
            denominator *= factorial(n)
    return law / denominator if denominator != 1 else law


def propensity(reaction: Reaction, /, *, combinatoric: bool = True) -> Expr:
    """The jump rate of ``reaction``, in molecule counts: falling factorials
    ``A (A - 1) ... (A - n + 1)`` for a stoichiometry ``n``.

    >>> from symbolcat import Symbol
    >>> A, k = Symbol("A"), Symbol("k")
    >>> print(propensity(Reaction(k, [A, A], [])))
    k * (A * (A - 1)) / 2
    """
    if not reaction.mass_action:
        return reaction.rate
    law: Expr = reaction.rate
    denominator = 1
    for species, n in reaction.substrates.items():
        falling: Expr = species
        for i in range(1, n):
            falling = falling * (species - i)
        law = law * falling
        if combinatoric:
            denominator *= factorial(n)
    return law / denominator if denominator != 1 else law


def stoichiometry_matrix(system: ReactionSystem, /) -> Call[Expr]:
    """The net stoichiometries: species (rows) by reactions (columns).

    >>> from symbolcat import Symbol
    >>> A, B, k = Symbol("A"), Symbol("B"), Symbol("k")
    >>> print(stoichiometry_matrix(ReactionSystem([Reaction(k, [A], [B])])))
    matrix([[-1], [1]])
    """
    return matrix(
        [[r.net(s) for r in system.reactions] for s in system.species]
    )


def reaction_rate_equations(system: ReactionSystem, /, *, combinatoric: bool = True) -> ODE:
    """The deterministic mass-action ODE.

    >>> from symbolcat import Symbol
    >>> A, B, k = Symbol("A"), Symbol("B"), Symbol("k")
    >>> ode = reaction_rate_equations(ReactionSystem([Reaction(k, [A], [B])]))
    >>> print(ode.rhs[A], "|", ode.rhs[B])
    -(k * A) | k * A
    """
    laws = [rate_law(r, combinatoric=combinatoric) for r in system.reactions]
    return ODE(
        {s: _combination(system, s, laws) for s in system.species},
        system.t,
    )


def _combination(system: ReactionSystem, species: Symbol, terms: list[Expr], /) -> Expr:
    total: Expr = Literal(0)
    for r, term in zip(system.reactions, terms, strict=True):
        n = r.net(species)
        if n:
            total = total + (term if n == 1 else n * term)
    return simplify_exact(total)


def chemical_langevin(system: ReactionSystem, /, *, combinatoric: bool = True) -> SDE:
    """The chemical Langevin equation (in Itô's convention): the rate
    equations, plus one Wiener process per reaction, with diffusion
    ``N[s, r] sqrt(v_r)``.

    >>> from symbolcat import Symbol
    >>> A, k = Symbol("A"), Symbol("k")
    >>> sde = chemical_langevin(ReactionSystem([Reaction(k, [A], [])]))
    >>> print(sde.diffusion[A][0])
    -sqrt(k * A)
    """
    laws = [rate_law(r, combinatoric=combinatoric) for r in system.reactions]
    drift = reaction_rate_equations(system, combinatoric=combinatoric).rhs
    diffusion = {
        s: [
            simplify_exact(r.net(s) * real.sqrt(law))
            for r, law in zip(system.reactions, laws, strict=True)
        ]
        for s in system.species
    }
    return SDE(drift, diffusion, system.t)


def propensities(system: ReactionSystem, /, *, combinatoric: bool = True) -> tuple[Expr, ...]:
    """The jump rate of each reaction (see ``propensity``)."""
    return tuple(propensity(r, combinatoric=combinatoric) for r in system.reactions)


def conservation_laws(system: ReactionSystem, /) -> tuple[Expr, ...]:
    """Linear combinations of the species that the reactions keep
    constant, with integer coefficients.

    >>> from symbolcat import Symbol
    >>> E, S, C, P, k = (Symbol(n) for n in "E S C P k".split())
    >>> network = ReactionSystem(
    ...     [Reaction(k, [E, S], [C]), Reaction(k, [C], [E, S]), Reaction(k, [C], [E, P])]
    ... )
    >>> for law in conservation_laws(network):
    ...     print(law)
    E + C
    S + C + P
    """
    n = [[r.net(s) for s in system.species] for r in system.reactions]
    # y N = 0 for the rows y of the left null space: the null space of N^T
    basis = rational_nullspace(n)
    laws = []
    for i, vector in enumerate(basis):
        # an elementary row operation, on the current basis: still a basis
        vector = basis[i] = _nonnegative(vector, basis[:i] + basis[i + 1 :])
        total: Expr | None = None
        for s, c in zip(system.species, vector, strict=True):
            if not c:
                continue
            term: Expr = s if abs(c) == 1 else int(abs(c)) * s
            if total is None:
                total = -term if c < 0 else term
            else:
                total = total - term if c < 0 else total + term
        assert total is not None
        laws.append(total)
    return tuple(laws)


def _nonnegative(vector: list[Any], others: list[list[Any]], /) -> list[Any]:
    """Among ``±vector``, plus or minus one of ``others``, the one with the
    fewest negative (then nonzero) entries."""
    candidates = [vector, [-v for v in vector]]
    for other in others:
        for sign in (1, -1):
            for base in (vector, [-v for v in vector]):
                candidates.append([a + sign * b for a, b in zip(base, other, strict=True)])
    candidates = [c for c in candidates if any(c)]
    return min(
        candidates,
        key=lambda c: (sum(v < 0 for v in c), sum(v != 0 for v in c)),
    )


def complexes(system: ReactionSystem, /) -> tuple[Complex, ...]:
    """The distinct complexes (substrate and product sides), in order of
    appearance. The empty complex stands for nothing."""
    out: dict[Complex, None] = {}
    for r in system.reactions:
        out[frozenset(r.substrates.items())] = None
        out[frozenset(r.products.items())] = None
    return tuple(out)


def _edges(system: ReactionSystem, /) -> tuple[tuple[Complex, ...], list[tuple[int, int]]]:
    nodes = complexes(system)
    index = {c: i for i, c in enumerate(nodes)}
    edges = [
        (index[frozenset(r.substrates.items())], index[frozenset(r.products.items())])
        for r in system.reactions
    ]
    return nodes, edges


def linkage_classes(system: ReactionSystem, /) -> tuple[tuple[int, ...], ...]:
    """The connected components of the graph of complexes (reactions as
    undirected edges), as indices into ``complexes``."""
    nodes, edges = _edges(system)
    parent = list(range(len(nodes)))

    def root(i: int, /) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        parent[root(a)] = root(b)
    classes: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        classes.setdefault(root(i), []).append(i)
    return tuple(tuple(c) for c in classes.values())


def deficiency(system: ReactionSystem, /) -> int:
    """Complexes, minus linkage classes, minus the rank of the
    stoichiometry matrix. The deficiency zero theorem applies when it is 0.

    >>> from symbolcat import Symbol
    >>> A, B, k = Symbol("A"), Symbol("B"), Symbol("k")
    >>> deficiency(ReactionSystem([Reaction(k, [A], [B]), Reaction(k, [B], [A])]))
    0
    """
    n = [[r.net(s) for r in system.reactions] for s in system.species]
    return len(complexes(system)) - len(linkage_classes(system)) - rational_rank(n)


def is_weakly_reversible(system: ReactionSystem, /) -> bool:
    """Whether every reaction is part of a cycle of reactions: each linkage
    class is strongly connected.

    >>> from symbolcat import Symbol
    >>> A, B, C, k = Symbol("A"), Symbol("B"), Symbol("C"), Symbol("k")
    >>> cycle = [Reaction(k, [A], [B]), Reaction(k, [B], [C]), Reaction(k, [C], [A])]
    >>> is_weakly_reversible(ReactionSystem(cycle)), is_weakly_reversible(ReactionSystem(cycle[:2]))
    (True, False)
    """
    nodes, edges = _edges(system)
    successors: list[list[int]] = [[] for _ in nodes]
    for a, b in edges:
        successors[a].append(b)
    components = _strongly_connected(successors)
    return len(components) == len(linkage_classes(system))
