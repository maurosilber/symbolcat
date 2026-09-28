"""
symbolcat.models
~~~~~~~~~~~~~~~~

Models of dynamical systems, built from expressions with ``symbolcat.D``
(time derivatives), ``delay`` and ``partial``, and their symbolic analysis.

- ``dae``: differential-algebraic equations: index reduction, consistent
  initialization, structural simplification.
- ``ode``: explicit ODEs (with delays and events) and discrete-time systems.
- ``dynamics``: Lie derivatives, equilibria, linearization, transfer
  functions, observability, identifiability, sensitivities.
- ``stochastic``: stochastic differential equations.
- ``pde``: partial differential equations, by the method of lines.
- ``reactions``: chemical reaction networks.
"""

from . import dae, dynamics, ode, pde, reactions, stochastic

__all__ = ["dae", "dynamics", "ode", "pde", "reactions", "stochastic"]
