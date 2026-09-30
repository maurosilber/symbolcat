"""
symbolcat.complex
~~~~~~~~~~~~~~~~~

Complex functions and constants, named after Python's ``cmath`` module.

>>> from symbolcat import Symbol, complex
>>> t, omega = Symbol("t"), Symbol("omega")
>>> z = complex.exp(complex.constants.I * omega * t)
>>> print(complex.re(z))
re(exp(I * omega * t))

These are functions of their own, not the ones in ``real``: the identities
that only hold for real numbers (``log(exp(x)) = x``, ``sqrt(x ** 2) =
abs(x)``) are never applied to them, and they take the principal branch, as
``cmath`` does: ``complex.sqrt(-1)`` is ``1j``, while ``real.sqrt(-1)`` is
undefined.

- ``re``, ``im``, ``conj``, ``abs`` and ``phase`` (the argument, in
  ``(-pi, pi]``) split a complex number, and ``rect(r, phi)`` builds one
  from polar coordinates.
- ``exp``, ``log``, ``sqrt``, the trigonometric and hyperbolic functions
  are holomorphic.

Functions are applied entry by entry to vectors and matrices. ``re``,
``im``, ``abs`` and ``phase`` are real, ``conj`` has the kind of its
argument, and the others are complex.

Symbols are real. Derivatives are with respect to real variables; for
holomorphic functions, they are the complex derivative.
"""

from ._core import Constant as _Constant
from ._core import Function as _Function
from ._core.infer import entrywise as _entrywise
from ._core.infer import kind_algebra as _kind_algebra
from ._core.infer import real_arguments as _real_arguments
from ._core.kinds import Complex as _Complex
from ._core.kinds import Real as _Real
from ._core.kinds import Scalar as _Scalar
from ._core.kinds import join as _join
from ._core.signatures import ComplexFunction as _ComplexFunction
from ._core.signatures import ComplexToReal as _ComplexToReal
from ._core.signatures import Conjugate as _Conjugate
from ._core.signatures import Polar as _Polar

re = _ComplexToReal("re", arity=1)
im = _ComplexToReal("im", arity=1)
conj = _Conjugate("conj", arity=1)
abs = _ComplexToReal("abs", arity=1)
phase = _ComplexToReal("phase", arity=1)
rect = _Polar("rect", arity=2)
exp = _ComplexFunction("exp", arity=1)
log = _ComplexFunction("log", arity=1)
log10 = _ComplexFunction("log10", arity=1)
sqrt = _ComplexFunction("sqrt", arity=1)
sin = _ComplexFunction("sin", arity=1)
cos = _ComplexFunction("cos", arity=1)
tan = _ComplexFunction("tan", arity=1)
asin = _ComplexFunction("asin", arity=1)
acos = _ComplexFunction("acos", arity=1)
atan = _ComplexFunction("atan", arity=1)
sinh = _ComplexFunction("sinh", arity=1)
cosh = _ComplexFunction("cosh", arity=1)
tanh = _ComplexFunction("tanh", arity=1)


class constants:
    I = _Constant(kind=_Complex())  # the imaginary unit


def _complex_valued(func: _Function, /, *kinds: _Scalar) -> _Scalar:
    return _Complex()


def _real_valued(func: _Function, /, *kinds: _Scalar) -> _Scalar:
    return _Real()


def _conjugate(func: _Function, kind: _Scalar, /) -> _Scalar:
    return _join(_Real(), kind)  # real for a real argument


def _polar(func: _Function, /, *kinds: _Scalar) -> _Scalar:
    _real_arguments(func, *kinds)
    return _Complex()


_kind_algebra.register(exp, log, log10, sqrt, sin, cos, tan)(
    _entrywise(_complex_valued)
)
_kind_algebra.register(asin, acos, atan, sinh, cosh, tanh)(_entrywise(_complex_valued))
_kind_algebra.register(re, im, abs, phase)(_entrywise(_real_valued))
_kind_algebra.register(rect)(_entrywise(_polar))
_kind_algebra.register(conj)(_entrywise(_conjugate))
