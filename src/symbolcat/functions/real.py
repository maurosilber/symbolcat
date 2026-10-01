"""
symbolcat.functions.real
~~~~~~~~~~~~~~~~~~~~~~~~

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

from .._core import Constant as _Constant
from .._core import Function as _Function
from .._core.infer import entrywise as _entrywise
from .._core.infer import kind_algebra as _kind_algebra
from .._core.infer import real_arguments as _real_arguments
from .._core.signatures import IfElse as _IfElse
from .._core.signatures import RealFunction as _RealFunction
from .._core.signatures import RealFunction2 as _RealFunction2
from ..types._kinds import Real as _Real
from ..types._kinds import Scalar as _Scalar
from ..types._kinds import join as _join

abs = _RealFunction("abs", arity=1)
ceil = _RealFunction("ceil", arity=1)
floor = _RealFunction("floor", arity=1)
sqrt = _RealFunction("sqrt", arity=1)
exp = _RealFunction("exp", arity=1)
log = _RealFunction("log", arity=1)
log2 = _RealFunction("log2", arity=1)
log10 = _RealFunction("log10", arity=1)
sin = _RealFunction("sin", arity=1)
cos = _RealFunction("cos", arity=1)
tan = _RealFunction("tan", arity=1)
asin = _RealFunction("asin", arity=1)
acos = _RealFunction("acos", arity=1)
atan = _RealFunction("atan", arity=1)
atan2 = _RealFunction2("atan2", arity=2)
sinh = _RealFunction("sinh", arity=1)
cosh = _RealFunction("cosh", arity=1)
tanh = _RealFunction("tanh", arity=1)
erf = _RealFunction("erf", arity=1)
gamma = _RealFunction("gamma", arity=1)
gammainc = _RealFunction2("gammainc", arity=2)
factorial = _RealFunction("factorial", arity=1)
hypot = _RealFunction2("hypot", arity=2)
ifelse = _IfElse("ifelse", arity=3)


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
