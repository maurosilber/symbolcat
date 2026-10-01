"""
symbolcat.backends._numpy
~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy as implementation. Operators are shared with ``std``, as NumPy
arrays already overload them.
"""

import math
from typing import Any

import numpy as np

from ..functions import complex as cx
from ..functions import linalg, real
from ..functions.linalg import Array
from ._std import OPERATORS, gammainc

IMPL: dict[Any, Any] = {
    **OPERATORS,
    Array: np.array,
    real.abs: np.abs,
    real.ceil: np.ceil,
    real.floor: np.floor,
    real.sqrt: np.sqrt,
    real.exp: np.exp,
    real.log: np.log,
    real.log2: np.log2,
    real.log10: np.log10,
    real.sin: np.sin,
    real.cos: np.cos,
    real.tan: np.tan,
    real.asin: np.arcsin,
    real.acos: np.arccos,
    real.atan: np.arctan,
    real.atan2: np.arctan2,
    real.sinh: np.sinh,
    real.cosh: np.cosh,
    real.tanh: np.tanh,
    real.erf: np.vectorize(math.erf),
    real.gamma: np.vectorize(math.gamma),
    real.gammainc: np.vectorize(gammainc),
    real.factorial: np.vectorize(math.factorial),
    real.hypot: np.hypot,
    real.ifelse: np.where,
    real.constants.e: np.e,
    real.constants.pi: np.pi,
    real.constants.tau: 2 * np.pi,
    real.constants.inf: np.inf,
    real.constants.nan: np.nan,
    linalg.transpose: np.transpose,
    linalg.trace: np.trace,
    linalg.det: np.linalg.det,
    linalg.inv: np.linalg.inv,
    linalg.norm: np.linalg.norm,
    linalg.sum: np.sum,
    linalg.cross: np.cross,
    # np.emath returns complex results outside the real domain: sqrt(-1) = 1j
    cx.re: np.real,
    cx.im: np.imag,
    cx.conj: np.conj,
    cx.abs: np.abs,
    cx.phase: np.angle,
    cx.rect: lambda r, phi: r * np.exp(1j * np.asarray(phi)),
    cx.exp: np.exp,
    cx.log: np.emath.log,
    cx.log10: np.emath.log10,
    cx.sqrt: np.emath.sqrt,
    cx.sin: np.sin,
    cx.cos: np.cos,
    cx.tan: np.tan,
    cx.asin: np.emath.arcsin,
    cx.acos: np.emath.arccos,
    cx.atan: lambda z: np.arctan(np.asarray(z, dtype=complex)),
    cx.sinh: np.sinh,
    cx.cosh: np.cosh,
    cx.tanh: np.tanh,
    cx.constants.I: 1j,
}
