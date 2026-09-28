"""
symbolcat.models.stochastic
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stochastic differential equations: Itô and Stratonovich conventions, and
Itô's lemma.
"""

from ._sde._system import SDE, ito_lemma, ito_to_stratonovich, stratonovich_to_ito

__all__ = [
    "SDE",
    "ito_lemma",
    "ito_to_stratonovich",
    "stratonovich_to_ito",
]
