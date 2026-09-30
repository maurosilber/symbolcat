"""
symbolcat.linalg
~~~~~~~~~~~~~~~~

Vectors, matrices and linear algebra functions.

An array is a ``Call`` of an ``Array`` function, whose arguments are the
entries in row-major order. There is one ``Array`` per shape, so arrays of
the same shape and entries compare equal, and every operation that handles
``Call`` nodes (substitution, simplification, ...) works on the entries.

>>> from symbolcat import Symbol, linalg
>>> x, y = Symbol("x"), Symbol("y")
>>> m = linalg.matrix([[x, 1], [0, y]])
>>> print(m)
matrix([[x, 1], [0, y]])
>>> linalg.to_list(m)
[[Symbol('x'), Literal(1)], [Literal(0), Symbol('y')]]

A symbol can also be an array, with unknown entries, by declaring its kind:

>>> from symbolcat.kinds import Matrix
>>> A = Symbol("A", kind=Matrix(2, 2))
>>> (A @ m).kind
Matrix(2, 2)

Arrays follow NumPy: operators (except ``@``) and ``real`` functions apply
entrywise, broadcasting scalars, and ``@`` is the matrix product, which is
the dot product for two vectors. ``symbolcat.ops.transform.expand_arrays`` computes
these entry by entry:

>>> from symbolcat import ops
>>> print(ops.transform.expand_arrays(m @ linalg.vector([1, 2])))
vector([x * 1 + 1 * 2, 0 * 1 + y * 2])
>>> print(ops.transform.expand_arrays(linalg.det(m)))
x * y - 1 * 0
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cache
from typing import Any, overload

from . import complex as _complex
from . import real as _real
from ._core import Call, Expr, Function, Operator, args_of, func_of, name_of
from ._core.infer import kind_algebra
from ._core.kinds import Kind, Matrix, Real, Scalar, Vector, array_kind, join
from ._core.signatures import Transpose
from .lib import operators as _op


@dataclass(frozen=True, repr=False, eq=False)
class Array(Function):
    """The function building an array of a given ``shape`` from its entries.

    Only vectors (one dimension) and matrices (two dimensions) are supported.
    """

    _: KW_ONLY
    shape: tuple[int, ...] = ()


def array_shape(func: Array, /) -> tuple[int, ...]:
    """The shape of the arrays that ``func`` builds.

    >>> array_shape(array((2, 3)))
    (2, 3)
    """
    return func.shape


@cache
def array(shape: tuple[int, ...], /) -> Array:
    """The ``Array`` function of a given shape."""
    if len(shape) not in (1, 2):
        raise ValueError(f"arrays must have 1 or 2 dimensions, got shape {shape}")
    name = "vector" if len(shape) == 1 else "matrix"
    return Array(name, arity=math.prod(shape), shape=shape)


def vector(entries: Iterable[Any], /) -> Call[Expr, Vector]:
    """A vector of expressions.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> print(vector([x, 2 * x]))
    vector([x, 2 * x])
    """
    entries = tuple(entries)
    return array((len(entries),))(*entries)


def matrix(rows: Iterable[Iterable[Any]], /) -> Call[Expr, Matrix]:
    """A matrix of expressions, given as a sequence of rows.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> print(matrix([[x, 0], [0, x]]))
    matrix([[x, 0], [0, x]])
    """
    table = [tuple(row) for row in rows]
    ncols = len(table[0]) if table else 0
    if any(len(row) != ncols for row in table):
        raise ValueError("matrix rows must have the same length")
    return array((len(table), ncols))(*(entry for row in table for entry in row))


def identity(n: int, /) -> Call[Expr, Matrix]:
    """The ``n`` by ``n`` identity matrix.

    >>> print(identity(2))
    matrix([[1, 0], [0, 1]])
    """
    return matrix([int(i == j) for j in range(n)] for i in range(n))


@overload
def zeros(shape: int | tuple[int], /) -> Call[Expr, Vector]: ...
@overload
def zeros(shape: tuple[int, int], /) -> Call[Expr, Matrix]: ...
@overload
def zeros(shape: tuple[int, ...], /) -> Call[Expr]: ...
def zeros(shape: int | tuple[int, ...], /) -> Call[Expr]:
    """An array of zeros.

    >>> print(zeros((2, 1)))
    matrix([[0], [0]])
    """
    return full(shape, 0)


@overload
def full(shape: int | tuple[int], value: Any, /) -> Call[Expr, Vector]: ...
@overload
def full(shape: tuple[int, int], value: Any, /) -> Call[Expr, Matrix]: ...
@overload
def full(shape: tuple[int, ...], value: Any, /) -> Call[Expr]: ...
def full(shape: int | tuple[int, ...], value: Any, /) -> Call[Expr]:
    """An array with every entry equal to ``value``.

    >>> from symbolcat import Symbol
    >>> print(full(2, Symbol("x")))
    vector([x, x])
    """
    shape = (shape,) if isinstance(shape, int) else shape
    return array(shape)(*[value] * math.prod(shape))


def nest[T](shape: tuple[int, ...], entries: Sequence[T], /) -> list[Any]:
    """Row-major ``entries`` as nested lists of the given shape."""
    if len(shape) == 1:
        return list(entries)
    _, ncols = shape
    return [list(entries[i : i + ncols]) for i in range(0, len(entries), ncols)]


def to_list(expr: Call[Any], /) -> list[Any]:
    """The entries of an array, as (nested) lists.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> to_list(vector([x, 1]))
    [Symbol('x'), Literal(1)]
    """
    if not isinstance(func_of(expr), Array):
        raise TypeError(f"not an array: {expr}")
    return nest(expr.kind.shape, args_of(expr))


type _Array = Expr[Vector] | Expr[Matrix]

transpose = Transpose("transpose", arity=1)
trace: Function[[Expr[Matrix]], Real] = Function("trace", arity=1)
det: Function[[Expr[Matrix]], Real] = Function("det", arity=1)
inv: Function[[Expr[Matrix]], Matrix] = Function("inv", arity=1)
# Euclidean for vectors, Frobenius for matrices
norm: Function[[_Array], Real] = Function("norm", arity=1)
sum: Function[[_Array], Real] = Function("sum", arity=1)  # of all the entries
cross: Function[[Expr[Vector], Expr[Vector]], Vector] = Function("cross", arity=2)

# Functions applied entry by entry, broadcasting scalars.
ENTRYWISE: frozenset[Function] = frozenset(
    [
        *(
            f
            for f in vars(_op).values()
            if isinstance(f, Operator) and f not in (_op.matmul, _op.getitem)
        ),
        *(f for f in vars(_real).values() if isinstance(f, Function)),
        *(f for f in vars(_complex).values() if isinstance(f, Function)),
    ]
)


# Kinds


@kind_algebra.register(Array)
def _(node: Call[Kind]) -> Kind:
    func = func_of(node)
    assert isinstance(func, Array)
    for kind in args_of(node):
        if kind.shape:
            raise ValueError(f"array entries must be scalars, got shape {kind.shape}")
    elements = [kind.element for kind in args_of(node)]
    return array_kind(array_shape(func), join(*elements) if elements else Real())


def _array(node: Call[Kind], /) -> Kind:
    """The single argument, which must be a vector or a matrix."""
    (kind,) = args_of(node)
    if not kind.shape:
        raise ValueError(
            f"{name_of(func_of(node))} needs a vector or a matrix, got a scalar"
        )
    return kind


def _square(node: Call[Kind], /) -> Matrix:
    """The single argument, which must be a square matrix."""
    (kind,) = args_of(node)
    match kind.shape:
        case (rows, cols) if rows == cols:
            assert isinstance(kind, Matrix)
            return kind
    raise ValueError(
        f"{name_of(func_of(node))} needs a square matrix, got shape {kind.shape}"
    )


def _numeric(kind: Scalar, /) -> Scalar:
    return join(Real(), kind)


@kind_algebra.register(transpose)
def _(node: Call[Kind]) -> Kind:
    """Reverses the axes: a vector is its own transpose, as in NumPy."""
    kind = _array(node)
    return array_kind(tuple(reversed(kind.shape)), kind.element)


@kind_algebra.register(trace, det)
def _(node: Call[Kind]) -> Kind:
    return _numeric(_square(node).element)


@kind_algebra.register(inv)
def _(node: Call[Kind]) -> Kind:
    kind = _square(node)
    return Matrix(kind.rows, kind.cols, _numeric(kind.element))


@kind_algebra.register(norm)
def _(node: Call[Kind]) -> Kind:
    _array(node)
    return Real()


@kind_algebra.register(sum)
def _(node: Call[Kind]) -> Kind:
    return _numeric(_array(node).element)


@kind_algebra.register(cross)
def _(node: Call[Kind]) -> Kind:
    a, b = args_of(node)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError(
            f"cross needs two vectors of length 3, got {a.shape} and {b.shape}"
        )
    return Vector(3, join(Real(), a.element, b.element))
