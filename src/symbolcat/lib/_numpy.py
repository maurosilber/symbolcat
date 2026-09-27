"""
symbolcat.lib._numpy
~~~~~~~~~~~~~~~~~~~~

NumPy as implementation. Operators are shared with ``std``, as NumPy
arrays already overload them.
"""

import math

import numpy as np

from .. import linalg, real
from ..linalg import Array
from ._std import OPERATORS

IMPL = {
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
    real.factorial: np.vectorize(math.factorial),
    real.hypot: np.hypot,
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
}
