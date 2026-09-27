"""
symbolcat.ops._arrays
~~~~~~~~~~~~~~~~~~~~~

Shapes of expressions, and array operations computed entry by entry.

A shape is ``()`` for a scalar, ``(n,)`` for a vector and ``(n, m)`` for a
matrix. Every node folds into a pair: the expression, with the operations
on explicit arrays (``linalg.vector``, ``linalg.matrix``) computed entry by
entry, and its shape. Operations on arrays with unknown entries (a symbol
with a matrix shape) are kept, as only a backend can compute them.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import reduce
from itertools import repeat
from typing import Any

from .. import linalg, real
from .._core import (
    Algebra,
    Call,
    Constant,
    Expr,
    Function,
    Literal,
    Symbol,
    cata,
    name_of,
)
from ..lib import operators as op
from ..linalg import Array, array, matrix, nest, vector

type Shape = tuple[int, ...]
type Pair = tuple[Expr, Shape]
type Rows = list[list[Expr]]

SCALAR: Shape = ()

arrays_algebra = Algebra[Pair]("arrays")


def entries(expr: Expr, /) -> tuple[Expr, ...] | None:
    """The entries of an explicit array, or ``None``."""
    if isinstance(expr, Call) and isinstance(expr.func, Array):
        return expr.args
    return None


def rebuild(node: Call[Pair], shape: Shape, /) -> Pair:
    return node.func(*(value for value, _ in node.args)), shape


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


def argument(node: Call[Pair], /, *, ndim: tuple[int, ...] = (1, 2)) -> Pair:
    """The single argument, which must be an array of the given dimensions."""
    ((value, shape),) = node.args
    if len(shape) not in ndim:
        kinds: dict[tuple[int, ...], str] = {(1,): "a vector", (2,): "a matrix"}
        kind = kinds.get(ndim, "an array")
        raise ValueError(f"{name_of(node.func)} needs {kind}, got {value}")
    return value, shape


def square(node: Call[Pair], /) -> tuple[Expr, Shape, Rows | None]:
    """The single argument, which must be a square matrix, and its rows."""
    value, shape = argument(node, ndim=(2,))
    if shape[0] != shape[1]:
        raise ValueError(
            f"{name_of(node.func)} needs a square matrix, got shape {shape}"
        )
    items = entries(value)
    return value, shape, None if items is None else nest(shape, items)


@arrays_algebra.register(Symbol)
def _(node: Symbol, *, shapes: Mapping[Symbol, Shape]) -> Pair:
    return node, shapes.get(node, SCALAR)


@arrays_algebra.register(Constant, Literal)
def _(node: Constant | Literal, **ctx: Any) -> Pair:
    return node, SCALAR


@arrays_algebra.register(Array)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    assert isinstance(node.func, Array)
    for value, shape in node.args:
        if shape:
            raise ValueError(f"array entries must be scalars, got {value}")
    return rebuild(node, node.func.shape)


@arrays_algebra.register(Function)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    """By default, functions take and return scalars."""
    for value, shape in node.args:
        if shape:
            raise ValueError(
                f"{name_of(node.func)} needs scalar arguments, got {value} "
                f"with shape {shape}"
            )
    return rebuild(node, SCALAR)


@arrays_algebra.register(*linalg.ENTRYWISE)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    """Applied entry by entry, broadcasting scalars."""
    shapes = [shape for _, shape in node.args]
    arrays = list(dict.fromkeys(shape for shape in shapes if shape))
    if len(arrays) > 1:
        raise ValueError(
            f"{name_of(node.func)} of incompatible shapes: "
            + ", ".join(map(str, shapes))
        )
    if not arrays:
        return rebuild(node, SCALAR)
    (shape,) = arrays
    columns = []
    for value, s in node.args:
        items = entries(value) if s else repeat(value)
        if items is None:
            return rebuild(node, shape)
        columns.append(items)
    return array(shape)(*map(node.func, *columns)), shape


@arrays_algebra.register(op.matmul)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    """Matrix product, as NumPy: a vector is a row on the left, a column on
    the right, and the product of two vectors is their dot product."""
    (a, sa), (b, sb) = node.args
    if not sa or not sb:
        raise ValueError(f"matmul needs arrays, got shapes {sa} and {sb}")
    if sa[-1] != sb[0]:
        raise ValueError(f"matmul of incompatible shapes: {sa} and {sb}")
    shape = (*list(sa)[:-1], *list(sb)[1:])  # the inner axis is contracted
    ea, eb = entries(a), entries(b)
    if ea is None or eb is None:
        return rebuild(node, shape)
    rows = nest(sa, ea) if len(sa) == 2 else [list(ea)]
    columns = [list(c) for c in zip(*nest(sb, eb))] if len(sb) == 2 else [list(eb)]
    products = [dot(row, column) for row in rows for column in columns]
    if not shape:
        return products[0], shape
    return array(shape)(*products), shape


@arrays_algebra.register(op.getitem)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    """An entry of a vector, or a row of a matrix. Indexing a scalar is
    allowed, as its value may be a container."""
    (obj, s), (key, key_shape) = node.args
    if key_shape:
        raise ValueError(f"index {key} must be a scalar")
    shape = s[1:]
    items = entries(obj)
    if items is None or not (isinstance(key, Literal) and type(key.value) is int):
        return rebuild(node, shape)
    item = nest(s, items)[key.value]
    return (vector(item) if shape else item), shape


@arrays_algebra.register(linalg.transpose)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    """Reverses the axes: a vector is its own transpose, as in NumPy."""
    value, s = argument(node)
    items = entries(value)
    if items is None:
        return rebuild(node, tuple(reversed(s)))
    if len(s) == 1:
        return value, s
    return matrix(zip(*nest(s, items))), tuple(reversed(s))


@arrays_algebra.register(linalg.trace)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    _, _, rows = square(node)
    if rows is None:
        return rebuild(node, SCALAR)
    return total([row[i] for i, row in enumerate(rows)]), SCALAR


@arrays_algebra.register(linalg.det)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    _, _, rows = square(node)
    if rows is None:
        return rebuild(node, SCALAR)
    return determinant(rows), SCALAR


@arrays_algebra.register(linalg.inv)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    """The adjugate divided by the determinant."""
    _, shape, rows = square(node)
    if rows is None:
        return rebuild(node, shape)
    d = determinant(rows)  # a shared node, computed once by ``cata``
    n = len(rows)
    if n == 1:
        return matrix([[1 / d]]), shape

    def cofactor(i: int, j: int) -> Expr:
        c = determinant(minor(rows, i, j))
        return -c if (i + j) % 2 else c

    return matrix([cofactor(j, i) / d for j in range(n)] for i in range(n)), shape


@arrays_algebra.register(linalg.norm)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    value, _ = argument(node)
    items = entries(value)
    if items is None:
        return rebuild(node, SCALAR)
    return real.sqrt(total([x**2 for x in items])), SCALAR


@arrays_algebra.register(linalg.sum)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    value, _ = argument(node)
    items = entries(value)
    if items is None:
        return rebuild(node, SCALAR)
    return total(list(items)), SCALAR


@arrays_algebra.register(linalg.cross)
def _(node: Call[Pair], **ctx: Any) -> Pair:
    (a, sa), (b, sb) = node.args
    if sa != (3,) or sb != (3,):
        raise ValueError(f"cross needs two vectors of length 3, got {sa} and {sb}")
    ea, eb = entries(a), entries(b)
    if ea is None or eb is None:
        return rebuild(node, sa)
    (a0, a1, a2), (b0, b1, b2) = ea, eb
    return vector([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0]), sa


def _shapes(shapes: Mapping[Any, Any] | None, /) -> dict[Symbol, Shape]:
    return {
        Symbol(k) if isinstance(k, str) else k: tuple(v)
        for k, v in (shapes or {}).items()
    }


def shape(expr: Any, shapes: Mapping[Any, Any] | None = None, /) -> Shape:
    """Shape of ``expr``: ``()`` for a scalar, ``(n,)`` for a vector, and
    ``(n, m)`` for a matrix.

    Symbols are scalars, unless ``shapes`` maps them (or their names) to a
    shape. Raises ``ValueError`` on incompatible shapes.

    >>> from symbolcat import Symbol
    >>> from symbolcat.linalg import vector
    >>> A, x = Symbol("A"), Symbol("x")
    >>> shape(A @ vector([x, 1]), {A: (3, 2)})
    (3,)
    >>> shape(A + vector([x, 1]), {A: (3,)})
    Traceback (most recent call last):
    ...
    ValueError: add of incompatible shapes: (3,), (2,)
    """
    _, result = cata(arrays_algebra, expr, shapes=_shapes(shapes))
    return result


def expand_arrays(expr: Any, shapes: Mapping[Any, Any] | None = None, /) -> Expr:
    """Compute the operations on explicit arrays entry by entry.

    Symbols are scalars, unless ``shapes`` maps them (or their names) to a
    shape; operations on those are kept. Raises ``ValueError`` on
    incompatible shapes.

    >>> from symbolcat import Symbol, real
    >>> from symbolcat.linalg import matrix, vector
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(expand_arrays(2 * real.sin(vector([x, y]))))
    vector([2 * sin(x), 2 * sin(y)])
    >>> print(expand_arrays(vector([x, y]) @ vector([y, x])))
    x * y + y * x
    """
    result, _ = cata(arrays_algebra, expr, shapes=_shapes(shapes))
    return result
