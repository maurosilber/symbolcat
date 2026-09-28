"""
symbolcat.ops._analysis._dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Order a system of definitions by their dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..._core import Symbol, as_expr
from ._inspect import free_symbols


def solve_dependencies(system: Mapping[Symbol, Any], /) -> list[tuple[Symbol, ...]]:
    """Layers of ``system`` that can be computed in order.

    ``system`` maps symbols to their definitions. Each layer holds the
    symbols whose definitions depend only on earlier layers, and on symbols
    not defined in ``system`` (inputs). A definition that is the symbol
    itself (``{x: x}``) marks an input. Raises ``ValueError`` on cycles.

    >>> from symbolcat import Symbol
    >>> x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")
    >>> solve_dependencies({z: x + y, x: 2 * t, y: x**2})
    [(Symbol('x'),), (Symbol('y'),), (Symbol('z'),)]
    >>> solve_dependencies({x: y, y: x})
    Traceback (most recent call last):
    ...
    ValueError: Cyclic dependencies among x, y
    """
    pending = {
        symbol: (set() if as_expr(value) == symbol else set(free_symbols(value)))
        & system.keys()
        for symbol, value in system.items()
    }
    layers: list[tuple[Symbol, ...]] = []
    while pending:
        layer = tuple(symbol for symbol, deps in pending.items() if not deps)
        if not layer:
            names = ", ".join(symbol.name for symbol in pending)
            raise ValueError(f"Cyclic dependencies among {names}")
        layers.append(layer)
        pending = {s: deps - set(layer) for s, deps in pending.items() if deps}
    return layers
