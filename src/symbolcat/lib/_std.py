"""
symbolcat.lib._std
~~~~~~~~~~~~~~~~~~

Python's standard library as implementation.
"""

import builtins
import cmath
import math
import operator

from .. import complex as cx
from .. import real
from . import operators as op

OPERATORS = {
    op.eq: operator.eq,
    op.ne: operator.ne,
    op.lt: operator.lt,
    op.le: operator.le,
    op.gt: operator.gt,
    op.ge: operator.ge,
    op.or_: operator.or_,
    op.xor: operator.xor,
    op.and_: operator.and_,
    op.lshift: operator.lshift,
    op.rshift: operator.rshift,
    op.add: operator.add,
    op.sub: operator.sub,
    op.mul: operator.mul,
    op.matmul: operator.matmul,
    op.truediv: operator.truediv,
    op.floordiv: operator.floordiv,
    op.mod: operator.mod,
    op.pow: operator.pow,
    op.neg: operator.neg,
    op.pos: operator.pos,
    op.invert: operator.invert,
    op.getitem: operator.getitem,
}

IMPL = {
    **OPERATORS,
    real.abs: builtins.abs,
    real.ceil: math.ceil,
    real.floor: math.floor,
    real.sqrt: math.sqrt,
    real.exp: math.exp,
    real.log: math.log,
    real.log2: math.log2,
    real.log10: math.log10,
    real.sin: math.sin,
    real.cos: math.cos,
    real.tan: math.tan,
    real.asin: math.asin,
    real.acos: math.acos,
    real.atan: math.atan,
    real.atan2: math.atan2,
    real.sinh: math.sinh,
    real.cosh: math.cosh,
    real.tanh: math.tanh,
    real.erf: math.erf,
    real.gamma: math.gamma,
    real.factorial: math.factorial,
    real.hypot: math.hypot,
    real.ifelse: lambda condition, a, b: a if condition else b,
    real.constants.e: math.e,
    real.constants.pi: math.pi,
    real.constants.tau: math.tau,
    real.constants.inf: math.inf,
    real.constants.nan: math.nan,
    cx.re: lambda z: z.real,
    cx.im: lambda z: z.imag,
    cx.conj: lambda z: z.conjugate(),
    cx.abs: builtins.abs,
    cx.phase: cmath.phase,
    cx.rect: cmath.rect,
    cx.exp: cmath.exp,
    cx.log: cmath.log,
    cx.log10: cmath.log10,
    cx.sqrt: cmath.sqrt,
    cx.sin: cmath.sin,
    cx.cos: cmath.cos,
    cx.tan: cmath.tan,
    cx.asin: cmath.asin,
    cx.acos: cmath.acos,
    cx.atan: cmath.atan,
    cx.sinh: cmath.sinh,
    cx.cosh: cmath.cosh,
    cx.tanh: cmath.tanh,
    cx.constants.I: 1j,
}
