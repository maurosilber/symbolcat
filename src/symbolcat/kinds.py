"""
symbolcat.kinds
~~~~~~~~~~~~~~~

Kinds: what the value of an expression is.

Every expression has a ``kind``. Symbols are real unless declared otherwise,
and the kind of a call follows from its function and the kinds of its
arguments, when it is built:

>>> from symbolcat import Symbol, linalg, real
>>> from symbolcat.kinds import Matrix, Vector
>>> x = Symbol("x")
>>> A = Symbol("A", kind=Matrix(2, 3))
>>> v = Symbol("v", kind=Vector(3))
>>> (x < 1).kind, (A @ v).kind, linalg.norm(v).kind
(Boolean(), Vector(2), Real())

An invalid expression is an error where it is written:

>>> real.sin(v) + A @ v
Traceback (most recent call last):
...
ValueError: add of incompatible shapes: (3,), (2,)

- ``Boolean``, ``Real``, ``Complex``: scalars, each holding the values of the
  previous ones. Arithmetic promotes booleans to reals; ``real`` functions
  need real arguments.
- ``Vector(n)``, ``Matrix(rows, cols)``: arrays, of real entries unless an
  ``element`` kind is given (``Vector(2, Complex())``).
"""

from ._core.kinds import Boolean, Complex, Kind, Matrix, Real, Scalar, Vector

__all__ = ["Boolean", "Complex", "Kind", "Matrix", "Real", "Scalar", "Vector"]
