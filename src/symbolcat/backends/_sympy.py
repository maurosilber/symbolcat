"""
symbolcat.backends._sympy
~~~~~~~~~~~~~~~~~~~~~~~~~

SymPy as implementation. Free symbols become ``sympy.Symbol``.

Arrays become ``sympy.Matrix``, and vectors are columns. Operators and
functions apply entrywise to matrices, as they do in NumPy.
"""

from collections.abc import Callable
from typing import Any

import sympy

from .._core import Symbol, name_of
from ..functions import complex as cx
from ..functions import linalg, real
from ..functions import operators as op
from ..functions.linalg import ENTRYWISE, Array
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


def _gammainc(a: Any, x: Any, /) -> Any:
    lower: Any = sympy.lowergamma(a, x)
    return lower / sympy.gamma(a)


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
    real.gammainc: _gammainc,
    real.factorial: sympy.factorial,
    real.hypot: lambda x, y: sympy.sqrt(x**2 + y**2),
    real.ifelse: lambda c, a, b: sympy.Piecewise((a, c), (b, True)),
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
    cx.re: sympy.re,
    cx.im: sympy.im,
    cx.conj: sympy.conjugate,
    cx.abs: sympy.Abs,
    cx.phase: sympy.arg,
    cx.rect: lambda r, phi: r * sympy.exp(sympy.I * phi),
    cx.exp: sympy.exp,
    cx.log: sympy.log,
    cx.log10: lambda z: sympy.log(z, 10),
    cx.sqrt: sympy.sqrt,
    cx.sin: sympy.sin,
    cx.cos: sympy.cos,
    cx.tan: sympy.tan,
    cx.asin: sympy.asin,
    cx.acos: sympy.acos,
    cx.atan: sympy.atan,
    cx.sinh: sympy.sinh,
    cx.cosh: sympy.cosh,
    cx.tanh: sympy.tanh,
    cx.constants.I: sympy.I,
    op.matmul: lambda a, b: a.dot(b) if a.shape == b.shape == (a.rows, 1) else a @ b,
}
IMPL.update({f: entrywise(IMPL[f]) for f in ENTRYWISE if f in IMPL})
