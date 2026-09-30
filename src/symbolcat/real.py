"""
symbolcat.real
~~~~~~~~~~~~~~

Mathematical functions and constants, named after Python's ``math`` module.

>>> from symbolcat import Symbol, real
>>> x = Symbol("x")
>>> print(2 * real.cos(x) + real.constants.pi)
2 * cos(x) + pi

``ifelse(condition, a, b)`` is ``a`` where ``condition`` holds, else ``b``:
piecewise functions, like ``ifelse(x < 0, -x, x)``.

Functions are applied entry by entry to vectors and matrices, and are only
defined for real arguments.

``gammainc(a, x)`` is the regularized lower incomplete gamma function,
``P(a, x) = 1 / gamma(a) * integral(t ** (a - 1) * exp(-t), t, 0, x)``, as
in SciPy.
"""

from ._core import Constant as _Constant
from ._core import Function as _Function
from ._core.infer import entrywise as _entrywise
from ._core.infer import kind_algebra as _kind_algebra
from ._core.infer import real_arguments as _real_arguments
from ._core.kinds import Real as _Real
from ._core.kinds import Scalar as _Scalar
from ._core.kinds import join as _join


def _fn(name: str, /, *, arity: int = 1) -> _Function:
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
atan2 = _fn("atan2", arity=2)
sinh = _fn("sinh")
cosh = _fn("cosh")
tanh = _fn("tanh")
erf = _fn("erf")
gamma = _fn("gamma")
gammainc = _fn("gammainc", arity=2)
factorial = _fn("factorial")
hypot = _fn("hypot", arity=2)
ifelse = _fn("ifelse", arity=3)


class constants:
    e = _Constant()
    pi = _Constant()
    tau = _Constant()
    inf = _Constant()
    nan = _Constant()


def _real_valued(func: _Function, /, *kinds: _Scalar) -> _Scalar:
    _real_arguments(func, *kinds)
    return _Real()


def _ifelse(func: _Function, condition: _Scalar, a: _Scalar, b: _Scalar, /) -> _Scalar:
    _real_arguments(func, condition)
    return _join(a, b)


_kind_algebra.register(
    *(f for f in list(globals().values()) if isinstance(f, _Function))
)(_entrywise(_real_valued))
_kind_algebra.register(ifelse)(_entrywise(_ifelse))
