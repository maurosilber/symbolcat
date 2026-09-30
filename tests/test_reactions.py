from typing import Any

import pytest

from symbolcat import Symbol
from symbolcat.linalg import to_list
from symbolcat.models.dynamics import lie_derivative
from symbolcat.models.reactions import (
    Reaction,
    ReactionSystem,
    chemical_langevin,
    conservation_laws,
    deficiency,
    is_weakly_reversible,
    linkage_classes,
    propensity,
    rate_law,
    reaction_rate_equations,
    stoichiometry_matrix,
)
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Literal, value_of
from symbolcat.ops.transform import substitute_variables

A, B, C, E, S, P = (Symbol(n) for n in ["A", "B", "C", "E", "S", "P"])
k1, k2, k3 = Symbol("k1"), Symbol("k2"), Symbol("k3")


def value(expr: Any, point: dict[Any, Any]) -> Any:
    return evaluate(substitute_variables(expr, point))


def michaelis_menten() -> ReactionSystem:
    return ReactionSystem(
        [
            Reaction(k1, [E, S], [C]),
            Reaction(k2, [C], [E, S]),
            Reaction(k3, [C], [E, P]),
        ]
    )


def test_reaction():
    r = Reaction(k1, {A: 2, B: 1}, [C, C])
    assert r.substrates == {A: 2, B: 1}
    assert r.products == {C: 2}
    assert r.net(A) == -2 and r.net(C) == 2
    assert str(Reaction(k1, [], [A])) == "∅ -> A, k1"
    with pytest.raises(ValueError, match="stoichiometry"):
        Reaction(k1, {A: -1}, [])


def test_rate_laws():
    point = {A: 3.0, B: 2.0, k1: 0.5}
    r = Reaction(k1, {A: 2, B: 1}, [C])
    assert value(rate_law(r), point) == pytest.approx(0.5 * 9 * 2 / 2)
    assert value(rate_law(r, combinatoric=False), point) == pytest.approx(0.5 * 9 * 2)
    hill = Reaction(k1 * A / (1 + A), [], [B], mass_action=False)
    assert rate_law(hill) == k1 * A / (1 + A)


def test_species_and_parameters():
    network = michaelis_menten()
    assert network.species == (E, S, C, P)
    assert network.parameters == (k1, k2, k3)


def test_reaction_rate_equations():
    ode = reaction_rate_equations(michaelis_menten())
    point = {E: 1.0, S: 2.0, C: 0.5, P: 0.0, k1: 3.0, k2: 0.7, k3: 0.2}
    binding = 3.0 * 1.0 * 2.0
    assert value(ode.rhs[S], point) == pytest.approx(-binding + 0.7 * 0.5)
    assert value(ode.rhs[C], point) == pytest.approx(binding - 0.9 * 0.5)
    assert value(ode.rhs[P], point) == pytest.approx(0.2 * 0.5)


def test_stoichiometry_matrix():
    rows = to_list(stoichiometry_matrix(michaelis_menten()))
    assert [[value_of(e) for e in row] for row in rows] == [
        [-1, 1, 1],
        [-1, 1, 0],
        [1, -1, -1],
        [0, 0, 1],
    ]


def test_conservation_laws_are_conserved():
    network = michaelis_menten()
    ode = reaction_rate_equations(network)
    laws = conservation_laws(network)
    assert len(laws) == 2
    point = {E: 1.0, S: 2.0, C: 0.5, P: 0.3, k1: 3.0, k2: 0.7, k3: 0.2}
    for law in laws:
        assert value(lie_derivative(law, ode), point) == pytest.approx(0)


def test_no_conservation_laws():
    network = ReactionSystem([Reaction(k1, [], [A]), Reaction(k2, [A], [])])
    assert conservation_laws(network) == ()


def test_propensities():
    a = propensity(Reaction(k1, {A: 3}, []))
    assert value(a, {A: 5, k1: 1}) == pytest.approx(5 * 4 * 3 / 6)


def test_chemical_langevin():
    sde = chemical_langevin(michaelis_menten())
    assert sde.noise == 3
    point = {E: 1.0, S: 2.0, C: 0.5, P: 0.0, k1: 3.0, k2: 0.7, k3: 0.2}
    # P only changes in the third reaction
    diffusion = [value(d, point) for d in sde.diffusion[P]]
    assert diffusion == pytest.approx([0, 0, (0.2 * 0.5) ** 0.5])
    assert sde.diffusion[P][0] == Literal(0)


def test_network_structure():
    network = michaelis_menten()
    assert linkage_classes(network) == (({E: 1, S: 1}, {C: 1}, {E: 1, P: 1}),)
    assert deficiency(network) == 0
    assert not is_weakly_reversible(network)


def test_deficiency_one():
    # A + B <-> 2B, B -> A (Feinberg's classic example has deficiency 1)
    network = ReactionSystem(
        [
            Reaction(k1, [A, B], [B, B]),
            Reaction(k2, [B, B], [A, B]),
            Reaction(k3, [B], [A]),
        ]
    )
    classes = linkage_classes(network)
    assert [len(c) for c in classes] == [2, 2]  # A + B, 2 B | B, A
    assert deficiency(network) == 1
