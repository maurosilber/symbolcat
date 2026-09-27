"""
symbolcat.real
~~~~~~~~~~~~~~

Mathematical functions and constants, named after Python's ``math`` module.

>>> from symbolcat import Symbol, real
>>> x = Symbol("x")
>>> print(2 * real.cos(x) + real.constants.pi)
2 * cos(x) + pi
"""

from ._core import Constant as _Constant
from ._core import Function as _Function


def _fn(name: str, arity: int = 1) -> _Function:
    return _Function(name, arity=arity)


abs = _fn("abs")
ceil = _fn("ceil")
floor = _fn("floor")
sqrt = _fn("sqrt")
exp = _fn("exp")
log = _fn("log")
log2 = _fn("log2")
log10 = _fn("log10")
sin = _fn("sin")
cos = _fn("cos")
tan = _fn("tan")
asin = _fn("asin")
acos = _fn("acos")
atan = _fn("atan")
atan2 = _fn("atan2", 2)
sinh = _fn("sinh")
cosh = _fn("cosh")
tanh = _fn("tanh")
erf = _fn("erf")
gamma = _fn("gamma")
factorial = _fn("factorial")
hypot = _fn("hypot", 2)


class constants:
    e = _Constant()
    pi = _Constant()
    tau = _Constant()
    inf = _Constant()
    nan = _Constant()
