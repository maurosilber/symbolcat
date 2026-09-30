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


def _fn(name: str, /, *, arity: int = 1) -> _Function:
    return _Function(name, arity=arity)


re = _fn("re")
im = _fn("im")
conj = _fn("conj")
abs = _fn("abs")
phase = _fn("phase")
rect = _fn("rect", arity=2)
exp = _fn("exp")
log = _fn("log")
log10 = _fn("log10")
sqrt = _fn("sqrt")
sin = _fn("sin")
cos = _fn("cos")
tan = _fn("tan")
asin = _fn("asin")
acos = _fn("acos")
atan = _fn("atan")
sinh = _fn("sinh")
cosh = _fn("cosh")
tanh = _fn("tanh")


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
