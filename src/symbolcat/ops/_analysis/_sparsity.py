"""
symbolcat.ops._analysis._sparsity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sparsity patterns: which entries of a Jacobian or Hessian can be nonzero.

The Jacobian pattern is structural: an entry can be nonzero where the
expression contains the variable. The Hessian pattern is that of the
Jacobian of the gradient, so it only marks the variables that interact
nonlinearly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..._core import as_expr
from ...functions.calculus import Variable
from ._inspect import free_symbols, variables

type Pattern = tuple[tuple[bool, ...], ...]


def jacobian_sparsity(exprs: Sequence[Any], vars: Sequence[Variable], /) -> Pattern:
    """For each expression (rows) and variable (columns), whether the
    derivative can be nonzero.

    >>> from symbolcat import Symbol
    >>> x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    >>> jacobian_sparsity([x * y, z + 1], [x, y, z])
    ((True, True, False), (False, False, True))
    """
    rows = []
    for expr in exprs:
        expr = as_expr(expr)
        present = {*variables(expr), *free_symbols(expr)}
        rows.append(tuple(v in present for v in vars))
    return tuple(rows)


def hessian_sparsity(expr: Any, vars: Sequence[Variable], /) -> Pattern:
    """For each pair of variables, whether the second derivative can be
    nonzero.

    >>> from symbolcat import Symbol, real
    >>> x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    >>> hessian_sparsity(x * y + real.sin(z) + x, [x, y, z])
    ((False, True, False), (True, False, False), (False, False, True))
    """
    from .._calculus._diff import diff

    return jacobian_sparsity([diff(expr, v) for v in vars], vars)
