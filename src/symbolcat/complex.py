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

Symbols are real. Derivatives are with respect to real variables; for
holomorphic functions, they are the complex derivative.
"""

from ._core import Constant as _Constant
from ._core import Function as _Function


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
    I = _Constant()  # the imaginary unit
