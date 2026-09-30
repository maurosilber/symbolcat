"""
symbolcat._core.kinds
~~~~~~~~~~~~~~~~~~~~~

Kinds: what the value of an expression is.

Every expression has a kind. The scalar kinds are ``Boolean``, ``Real`` and
``Complex``, and the array kinds ``Vector`` and ``Matrix`` have a shape and
the kind of their entries. Symbols and constants declare theirs (``Real``
by default), a literal takes it from its value, and a call gets it from a
rule of its function when it is built (see ``kind_algebra``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Kind:
    """Base class of kinds. An instance is an unspecified kind."""

    @property
    def shape(self) -> tuple[int, ...]:
        """``()`` for a scalar, ``(n,)`` for a vector, ``(n, m)`` for a matrix."""
        return ()

    @property
    def element(self) -> Scalar:
        """The kind of the entries (or the kind itself, for a scalar)."""
        raise TypeError("an unspecified kind has no element kind")


@dataclass(frozen=True, repr=False)
class Scalar(Kind):
    """Base class of the scalar kinds, ordered ``Boolean < Real < Complex``:
    each one holds the values of the previous ones."""

    rank: ClassVar[int]

    @property
    def element(self) -> Scalar:
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, repr=False)
class Boolean(Scalar):
    """A truth value, as produced by comparisons."""

    rank = 0


@dataclass(frozen=True, repr=False)
class Real(Scalar):
    """A real number."""

    rank = 1


@dataclass(frozen=True, repr=False)
class Complex(Scalar):
    """A complex number."""

    rank = 2


@dataclass(frozen=True, repr=False)
class Vector(Kind):
    """A vector of ``n`` entries of kind ``element``.

    >>> Vector(3)
    Vector(3)
    >>> Vector(2, Complex()).shape
    (2,)
    """

    n: int
    element: Scalar = Real()

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.n,)

    def __repr__(self) -> str:
        return _repr(self, self.n)


@dataclass(frozen=True, repr=False)
class Matrix(Kind):
    """A matrix of ``rows`` by ``cols`` entries of kind ``element``.

    >>> Matrix(3, 2)
    Matrix(3, 2)
    >>> Matrix(2, 2, Complex()).shape
    (2, 2)
    """

    rows: int
    cols: int
    element: Scalar = Real()

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.rows, self.cols)

    def __repr__(self) -> str:
        return _repr(self, self.rows, self.cols)


def _repr(kind: Vector | Matrix, /, *sizes: int) -> str:
    args = [str(size) for size in sizes]
    if kind.element != Real():
        args.append(repr(kind.element))
    return f"{type(kind).__name__}({', '.join(args)})"


def array_kind(shape: tuple[int, ...], element: Scalar, /) -> Kind:
    """The kind of the given shape and entries.

    >>> array_kind((), Real())
    Real()
    >>> array_kind((2, 3), Complex())
    Matrix(2, 3, Complex())
    """
    match shape:
        case ():
            return element
        case (n,):
            return Vector(n, element)
        case (rows, cols):
            return Matrix(rows, cols, element)
    raise ValueError(f"arrays must have 1 or 2 dimensions, got shape {shape}")


def join(*kinds: Scalar) -> Scalar:
    """The smallest scalar kind holding all of ``kinds``.

    >>> join(Boolean(), Real())
    Real()
    """
    return max(kinds, key=lambda kind: kind.rank)
