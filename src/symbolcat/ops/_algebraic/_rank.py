"""
symbolcat.ops._algebraic._rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exact linear algebra over the rationals, and the generic rank of symbolic
matrices.

The generic rank is the rank for almost all values of the symbols. It is
computed exactly, with rational arithmetic, at random rational values: the
rank there is the generic rank, except on a set of measure zero
(Schwartz-Zippel), so the answer is correct with high probability.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from ..._core import Call, Expr, func_of
from ...lib import operators as op
from ...lib.std import IMPL
from ...linalg import Array, to_list
from .._analysis._inspect import free_symbols, variables
from .._evaluate import evaluate
from .._transform._substitute import substitute_variables

type Rows = list[list[Fraction]]


def _echelon(rows: Rows, /) -> tuple[Rows, list[int]]:
    """Reduced row echelon form, and the pivot columns."""
    rows = [row[:] for row in rows]
    pivots: list[int] = []
    r = 0
    for col in range(len(rows[0]) if rows else 0):
        pivot = next((i for i in range(r, len(rows)) if rows[i][col]), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        lead = rows[r][col]
        rows[r] = [v / lead for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col]:
                factor = rows[i][col]
                rows[i] = [
                    a - factor * b for a, b in zip(rows[i], rows[r], strict=True)
                ]
        pivots.append(col)
        r += 1
    return rows, pivots


def rational_rank(rows: Sequence[Sequence[Any]], /) -> int:
    """The rank of a matrix of rationals.

    >>> rational_rank([[1, 2], [2, 4]])
    1
    """
    return len(_echelon([[Fraction(v) for v in row] for row in rows])[1])


def rational_nullspace(rows: Sequence[Sequence[Any]], /) -> list[list[Fraction]]:
    """A basis of the (right) null space of a matrix of rationals, each
    vector scaled to integers without a common factor.

    >>> rational_nullspace([[1, 1, 0], [0, 1, 1]])
    [[Fraction(1, 1), Fraction(-1, 1), Fraction(1, 1)]]
    """
    matrix = [[Fraction(v) for v in row] for row in rows]
    n = len(matrix[0]) if matrix else 0
    reduced, pivots = _echelon(matrix)
    basis = []
    for free in (c for c in range(n) if c not in pivots):
        vector = [Fraction(0)] * n
        vector[free] = Fraction(1)
        for row, col in zip(reduced, pivots, strict=False):
            vector[col] = -row[free]
        basis.append(_integer(vector))
    return basis


def _integer(vector: list[Fraction], /) -> list[Fraction]:
    from math import gcd, lcm

    scale = lcm(*(v.denominator for v in vector))
    ints = [int(v * scale) for v in vector]
    common = gcd(*ints) or 1
    return [Fraction(v // common) for v in ints]


def _exact_div(a: Any, b: Any, /) -> Fraction:
    return Fraction(a) / Fraction(b)


def _exact_pow(a: Any, b: Any, /) -> Any:
    if type(b) is not int and not (isinstance(b, Fraction) and b.denominator == 1):
        raise TypeError("generic_rank: entries must be rational functions")
    return Fraction(a) ** int(b)


EXACT = {**IMPL, op.truediv: _exact_div, op.pow: _exact_pow}


def generic_rank(matrix: Any, /, *, seed: int = 0) -> int:
    """The rank of a matrix of rational functions, for generic values of
    its symbols (and variables, like ``D(x)``).

    ``matrix`` is a symbolic matrix or a list of rows.

    >>> from symbolcat import Symbol
    >>> from symbolcat.linalg import matrix
    >>> x, y = Symbol("x"), Symbol("y")
    >>> generic_rank(matrix([[x, y], [2 * x, 2 * y]]))
    1
    >>> generic_rank(matrix([[x, y], [y, x]]))
    2
    """
    if isinstance(matrix, Call) and isinstance(func_of(matrix), Array):
        rows = to_list(matrix)
    else:
        rows = [list(row) for row in matrix]
    rng = random.Random(seed)
    point: dict[Expr, Any] = {}
    values = []
    for row in rows:
        out = []
        for entry in row:
            for var in (*variables(entry), *free_symbols(entry)):
                if var not in point:
                    point[var] = Fraction(rng.randint(1, 10**9), rng.randint(1, 10**9))
            value = evaluate(substitute_variables(entry, point), lib=EXACT)
            if not isinstance(value, (int, Fraction)) or isinstance(value, bool):
                raise TypeError("generic_rank: entries must be rational functions")
            out.append(Fraction(value))
        values.append(out)
    return rational_rank(values)
