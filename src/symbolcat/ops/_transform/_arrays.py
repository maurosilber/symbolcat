"""
symbolcat.ops._transform._arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shapes of expressions, and array operations computed entry by entry.

A shape is ``()`` for a scalar, ``(n,)`` for a vector and ``(n, m)`` for a
matrix. It is part of the kind of every expression (``symbolcat.kinds``),
checked when the expression is built. ``expand_arrays`` folds each node
into the expression with the operations on explicit arrays
(``linalg.vector``, ``linalg.matrix``) computed entry by entry. Operations
on arrays with unknown entries (a symbol of a matrix kind) are kept, as only
a backend can compute them.
"""

from __future__ import annotations

from functools import reduce
from itertools import repeat
from typing import Any

from ... import linalg, real
from ..._core import (
    Algebra,
    Call,
    Expr,
    Literal,
    args_of,
    as_expr,
    cata,
    func_of,
    value_of,
)
from ...lib import operators as op
from ...linalg import Array, array, matrix, nest, vector

type Shape = tuple[int, ...]
type Rows = list[list[Expr]]

SCALAR: Shape = ()

arrays_algebra = Algebra[Expr]("arrays")


def entries(expr: Expr, /) -> tuple[Expr, ...] | None:
    """The entries of an explicit array, or ``None``."""
    if isinstance(expr, Call) and isinstance(func_of(expr), Array):
        return args_of(expr)
    return None


def total(terms: list[Expr], /) -> Expr:
    return reduce(op.add, terms) if terms else Literal(0)


def dot(a: list[Expr], b: list[Expr], /) -> Expr:
    return total([x * y for x, y in zip(a, b)])


def minor(rows: Rows, i: int, j: int, /) -> Rows:
    return [row[:j] + row[j + 1 :] for k, row in enumerate(rows) if k != i]


def determinant(rows: Rows, /) -> Expr:
    """Laplace expansion along the first row."""
    if not rows:
        return Literal(1)
    if len(rows) == 1:
        return rows[0][0]
    result = rows[0][0] * determinant(minor(rows, 0, 0))
    for j in range(1, len(rows)):
        term = rows[0][j] * determinant(minor(rows, 0, j))
        result = result - term if j % 2 else result + term
    return result


def rows_of(expr: Expr, /) -> Rows | None:
    """The rows of an explicit matrix, or ``None``."""
    items = entries(expr)
    return None if items is None else nest(expr.kind.shape, items)


@arrays_algebra.register(Expr)
def _(node: Expr) -> Expr:
    """Leaves, explicit arrays, and functions of scalars stay as they are:
    the layer is already the node with its arguments expanded."""
    return node


@arrays_algebra.register(*linalg.ENTRYWISE)
def _(node: Call[Expr]) -> Expr:
    """Applied entry by entry, broadcasting scalars."""
    shape = node.kind.shape
    if not shape:
        return node
    columns = []
    for arg in args_of(node):
        items = entries(arg) if arg.kind.shape else repeat(arg)
        if items is None:
            return node
        columns.append(items)
    return array(shape)(*map(func_of(node), *columns))


@arrays_algebra.register(op.matmul)
def _(node: Call[Expr]) -> Expr:
    """Matrix product, as NumPy: a vector is a row on the left, a column on
    the right, and the product of two vectors is their dot product."""
    a, b = args_of(node)
    sa, sb = a.kind.shape, b.kind.shape
    ea, eb = entries(a), entries(b)
    if ea is None or eb is None:
        return node
    rows = nest(sa, ea) if len(sa) == 2 else [list(ea)]
    columns = [list(c) for c in zip(*nest(sb, eb))] if len(sb) == 2 else [list(eb)]
    products = [dot(row, column) for row in rows for column in columns]
    shape = node.kind.shape
    return array(shape)(*products) if shape else products[0]


@arrays_algebra.register(op.getitem)
def _(node: Call[Expr]) -> Expr:
    """An entry of a vector, or a row of a matrix."""
    obj, key = args_of(node)
    items = entries(obj)
    if items is None or not (isinstance(key, Literal) and type(value_of(key)) is int):
        return node
    item = nest(obj.kind.shape, items)[value_of(key)]
    return vector(item) if node.kind.shape else item


@arrays_algebra.register(linalg.transpose)
def _(node: Call[Expr]) -> Expr:
    """Reverses the axes: a vector is its own transpose, as in NumPy."""
    (value,) = args_of(node)
    if len(value.kind.shape) == 1:
        return value
    rows = rows_of(value)
    return node if rows is None else matrix(zip(*rows))


@arrays_algebra.register(linalg.trace)
def _(node: Call[Expr]) -> Expr:
    rows = rows_of(args_of(node)[0])
    return node if rows is None else total([row[i] for i, row in enumerate(rows)])


@arrays_algebra.register(linalg.det)
def _(node: Call[Expr]) -> Expr:
    rows = rows_of(args_of(node)[0])
    return node if rows is None else determinant(rows)


@arrays_algebra.register(linalg.inv)
def _(node: Call[Expr]) -> Expr:
    """The adjugate divided by the determinant."""
    rows = rows_of(args_of(node)[0])
    if rows is None:
        return node
    d = determinant(rows)  # a shared node, computed once by ``cata``
    n = len(rows)
    if n == 1:
        return matrix([[1 / d]])

    def cofactor(i: int, j: int) -> Expr:
        c = determinant(minor(rows, i, j))
        return -c if (i + j) % 2 else c

    return matrix([cofactor(j, i) / d for j in range(n)] for i in range(n))


@arrays_algebra.register(linalg.norm)
def _(node: Call[Expr]) -> Expr:
    items = entries(args_of(node)[0])
    return node if items is None else real.sqrt(total([x**2 for x in items]))


@arrays_algebra.register(linalg.sum)
def _(node: Call[Expr]) -> Expr:
    items = entries(args_of(node)[0])
    return node if items is None else total(list(items))


@arrays_algebra.register(linalg.cross)
def _(node: Call[Expr]) -> Expr:
    ea, eb = map(entries, args_of(node))
    if ea is None or eb is None:
        return node
    (a0, a1, a2), (b0, b1, b2) = ea, eb
    return vector([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def shape(expr: Any, /) -> Shape:
    """Shape of ``expr``: ``()`` for a scalar, ``(n,)`` for a vector, and
    ``(n, m)`` for a matrix. It is part of the kind of ``expr``.

    Symbols are scalars, unless declared with an array kind. Incompatible
    shapes are an error when the expression is built.

    >>> from symbolcat import Symbol
    >>> from symbolcat.kinds import Matrix, Vector
    >>> from symbolcat.linalg import vector
    >>> A = Symbol("A", kind=Matrix(3, 2))
    >>> shape(A @ vector([1, 2]))
    (3,)
    >>> Symbol("v", kind=Vector(3)) + vector([1, 2])
    Traceback (most recent call last):
    ...
    ValueError: add of incompatible shapes: (3,), (2,)
    """
    return as_expr(expr).kind.shape


def expand_arrays(expr: Any, /) -> Expr:
    """Compute the operations on explicit arrays entry by entry.

    Operations on symbols of an array kind are kept.

    >>> from symbolcat import Symbol, real
    >>> from symbolcat.linalg import vector
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(expand_arrays(2 * real.sin(vector([x, y]))))
    vector([2 * sin(x), 2 * sin(y)])
    >>> print(expand_arrays(vector([x, y]) @ vector([y, x])))
    x * y + y * x
    """
    return cata(arrays_algebra, expr)
