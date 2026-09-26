"""
symbolcat.real
~~~~~~~~~~~~~~

Mathematical functions and constants, named after Python's ``math`` module.

>>> from symbolcat import Symbol, real
>>> x = Symbol("x")
>>> print(2 * real.cos(x) + real.pi)
2 * cos(x) + pi
"""

from .core import Constant, Function


def _fn(name: str, arity: int = 1) -> Function:
    return Function(name, arity=arity)


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

e = Constant("e")
pi = Constant("pi")
tau = Constant("tau")
inf = Constant("inf")
nan = Constant("nan")
