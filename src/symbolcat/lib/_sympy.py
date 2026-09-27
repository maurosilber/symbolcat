"""
symbolcat.lib._sympy
~~~~~~~~~~~~~~~~~~~~

SymPy as implementation. Free symbols become ``sympy.Symbol``.
"""

import sympy

from .. import real
from .._core import Symbol, name_of
from ..linalg import Array
from . import operators as op
from ._std import OPERATORS

IMPL = {
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
}
