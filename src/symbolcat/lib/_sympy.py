"""
symbolcat.lib._sympy
~~~~~~~~~~~~~~~~~~~~

SymPy as implementation. Free symbols become ``sympy.Symbol``.

Arrays become ``sympy.Matrix``, and vectors are columns. Operators and
functions apply entrywise to matrices, as they do in NumPy.
"""

from collections.abc import Callable
from typing import Any

import sympy

from .. import linalg, real
from .._core import Symbol, name_of
from ..linalg import ENTRYWISE, Array
from . import operators as op
from ._std import OPERATORS


def entrywise(func: Callable[..., Any], /) -> Callable[..., Any]:
    """Apply ``func`` entry by entry to matrices, broadcasting scalars."""

    def apply(*args: Any) -> Any:
        shapes = {a.shape for a in args if isinstance(a, sympy.MatrixBase)}
        if not shapes:
            return func(*args)
        if len(shapes) > 1:
            raise ValueError(f"incompatible shapes: {', '.join(map(str, shapes))}")
        ((rows, cols),) = shapes
        return sympy.Matrix(
            rows,
            cols,
            lambda i, j: func(
                *(a[i, j] if isinstance(a, sympy.MatrixBase) else a for a in args)
            ),
        )

    return apply


IMPL: dict[Any, Any] = {
    **OPERATORS,
    Array: sympy.Matrix,
    Symbol: lambda symbol: sympy.Symbol(name_of(symbol)),
    op.eq: sympy.Eq,
    op.ne: sympy.Ne,
    real.abs: sympy.Abs,
    real.ceil: sympy.ceiling,
    real.floor: sympy.floor,
    real.sqrt: sympy.sqrt,
    real.exp: sympy.exp,
    real.log: sympy.log,
    real.log2: lambda x: sympy.log(x, 2),
    real.log10: lambda x: sympy.log(x, 10),
    real.sin: sympy.sin,
    real.cos: sympy.cos,
    real.tan: sympy.tan,
    real.asin: sympy.asin,
    real.acos: sympy.acos,
    real.atan: sympy.atan,
    real.atan2: sympy.atan2,
    real.sinh: sympy.sinh,
    real.cosh: sympy.cosh,
    real.tanh: sympy.tanh,
    real.erf: sympy.erf,
    real.gamma: sympy.gamma,
    real.factorial: sympy.factorial,
    real.hypot: lambda x, y: sympy.sqrt(x**2 + y**2),
    real.constants.e: sympy.E,
    real.constants.pi: sympy.pi,
    real.constants.tau: 2 * sympy.pi,
    real.constants.inf: sympy.oo,
    real.constants.nan: sympy.nan,
    linalg.transpose: lambda a: a.T,
    linalg.trace: lambda a: a.trace(),
    linalg.det: lambda a: a.det(),
    linalg.inv: lambda a: a.inv(),
    linalg.norm: lambda a: a.norm(),
    linalg.sum: lambda a: sum(a),
    linalg.cross: lambda a, b: a.cross(b),
    op.matmul: lambda a, b: a.dot(b) if a.shape == b.shape == (a.rows, 1) else a @ b,
}
IMPL.update({f: entrywise(IMPL[f]) for f in ENTRYWISE if f in IMPL})
