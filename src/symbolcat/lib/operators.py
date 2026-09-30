"""
symbolcat.lib.operators
~~~~~~~~~~~~~~~~~~~~~~~

Operators produced by Python syntax on expressions (``x + y``, ``x[0]``, ...).

These are declarations, with the kind of their result (see
``symbolcat.kinds``): what each operator *means* is given by the algebras in
``symbolcat.ops`` and the implementation libraries in ``symbolcat.lib``.

Precedences follow Python (higher binds tighter). Comparisons are non-associative, as
``a < b < c`` means something else in Python.
"""

from __future__ import annotations

from .._core import precedence as _p
from .._core.expr import Call as _Call
from .._core.expr import Function as _Function
from .._core.expr import Operator as _Operator
from .._core.expr import args_of as _args_of
from .._core.infer import entrywise as _entrywise
from .._core.infer import kind_algebra as _kind_algebra
from .._core.infer import real_arguments as _real_arguments
from .._core.kinds import Boolean as _Boolean
from .._core.kinds import Kind as _Kind
from .._core.kinds import Real as _Real
from .._core.kinds import Scalar as _Scalar
from .._core.kinds import array_kind as _array_kind
from .._core.kinds import join as _join


def _binary(
    name: str, symbol: str, precedence: int, /, *, assoc: str = "left"
) -> _Operator:
    return _Operator(
        name,
        arity=2,
        fmt=f"{{}} {symbol} {{}}",
        precedence=precedence,
        assoc=assoc,
    )


def _unary(name: str, symbol: str, /) -> _Operator:
    return _Operator(name, arity=1, fmt=f"{symbol}{{}}", precedence=_p.UNARY)


eq = _binary("eq", "==", _p.CMP, assoc="none")
ne = _binary("ne", "!=", _p.CMP, assoc="none")
lt = _binary("lt", "<", _p.CMP, assoc="none")
le = _binary("le", "<=", _p.CMP, assoc="none")
gt = _binary("gt", ">", _p.CMP, assoc="none")
ge = _binary("ge", ">=", _p.CMP, assoc="none")

or_ = _binary("or_", "|", _p.OR)
xor = _binary("xor", "^", _p.XOR)
and_ = _binary("and_", "&", _p.AND)
lshift = _binary("lshift", "<<", _p.SHIFT)
rshift = _binary("rshift", ">>", _p.SHIFT)

add = _binary("add", "+", _p.ADD)
sub = _binary("sub", "-", _p.ADD)
mul = _binary("mul", "*", _p.MUL)
matmul = _binary("matmul", "@", _p.MUL)
truediv = _binary("truediv", "/", _p.MUL)
floordiv = _binary("floordiv", "//", _p.MUL)
mod = _binary("mod", "%", _p.MUL)
pow = _binary("pow", "**", _p.POW, assoc="right")

neg = _unary("neg", "-")
pos = _unary("pos", "+")
invert = _unary("invert", "~")

getitem = _Operator("getitem", arity=2, fmt="{}[{}]", precedence=_p.POSTFIX)


# Kinds. Arithmetic promotes booleans to reals, as Python does, and only
# comparisons and logical operators on booleans give booleans.


def _numeric(func: _Function, /, *kinds: _Scalar) -> _Scalar:
    return _join(_Real(), *kinds)


def _real(func: _Function, /, *kinds: _Scalar) -> _Scalar:
    _real_arguments(func, *kinds)
    return _Real()


def _equality(func: _Function, /, *kinds: _Scalar) -> _Scalar:
    return _Boolean()


def _order(func: _Function, /, *kinds: _Scalar) -> _Scalar:
    _real_arguments(func, *kinds)
    return _Boolean()


def _logical(func: _Function, /, *kinds: _Scalar) -> _Scalar:
    _real_arguments(func, *kinds)
    return _Boolean() if all(k == _Boolean() for k in kinds) else _Real()


for _funcs, _element in [
    ((eq, ne), _equality),
    ((lt, le, gt, ge), _order),
    ((or_, xor, and_, invert), _logical),
    ((lshift, rshift, floordiv, mod), _real),
    ((add, sub, mul, truediv, pow, neg, pos), _numeric),
]:
    _kind_algebra.register(*_funcs)(_entrywise(_element))
del _funcs, _element


@_kind_algebra.register(matmul)
def _(node: _Call[_Kind]) -> _Kind:
    """Matrix product, as NumPy: a vector is a row on the left, a column on
    the right, and the product of two vectors is their dot product."""
    a, b = _args_of(node)
    sa, sb = a.shape, b.shape
    if not sa or not sb:
        raise ValueError(f"matmul needs arrays, got shapes {sa} and {sb}")
    if sa[-1] != sb[0]:
        raise ValueError(f"matmul of incompatible shapes: {sa} and {sb}")
    shape = (*list(sa)[:-1], *list(sb)[1:])  # the inner axis is contracted
    return _array_kind(shape, _join(_Real(), a.element, b.element))


@_kind_algebra.register(getitem)
def _(node: _Call[_Kind]) -> _Kind:
    """An entry of a vector, or a row of a matrix. Indexing a scalar is
    allowed, as its value may be a container."""
    obj, key = _args_of(node)
    if key.shape:
        raise ValueError(f"index must be a scalar, got shape {key.shape}")
    return _array_kind(obj.shape[1:], obj.element)
