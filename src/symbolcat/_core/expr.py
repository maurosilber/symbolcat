"""
symbolcat._core.expr
~~~~~~~~~~~~~~~~~~~~

Expression nodes and symbolic functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Expr:
    """Base class of expression nodes. Provides the operator overloads.

    ``==`` and ``hash`` are structural, so expressions can be used as
    dictionary keys. Use ``operators.eq`` and ``operators.ne`` to build
    symbolic comparisons: ``operators.eq(x, 1)``.
    """

    __slots__ = ()

    def __str__(self) -> str:
        from ..ops import to_str

        return to_str(self)

    def _repr_latex_(self) -> str:
        from ..ops import to_latex

        return f"${to_latex(self)}$"

    # Comparisons
    def __lt__(self, other: Any) -> Call[Expr]:
        return _op.lt(self, other)

    def __le__(self, other: Any) -> Call[Expr]:
        return _op.le(self, other)

    def __gt__(self, other: Any) -> Call[Expr]:
        return _op.gt(self, other)

    def __ge__(self, other: Any) -> Call[Expr]:
        return _op.ge(self, other)

    # Containers
    def __getitem__(self, key: Any) -> Call[Expr]:
        return _op.getitem(self, key)

    # Arithmetic
    def __add__(self, other: Any) -> Call[Expr]:
        return _op.add(self, other)

    def __radd__(self, other: Any) -> Call[Expr]:
        return _op.add(other, self)

    def __sub__(self, other: Any) -> Call[Expr]:
        return _op.sub(self, other)

    def __rsub__(self, other: Any) -> Call[Expr]:
        return _op.sub(other, self)

    def __mul__(self, other: Any) -> Call[Expr]:
        return _op.mul(self, other)

    def __rmul__(self, other: Any) -> Call[Expr]:
        return _op.mul(other, self)

    def __matmul__(self, other: Any) -> Call[Expr]:
        return _op.matmul(self, other)

    def __rmatmul__(self, other: Any) -> Call[Expr]:
        return _op.matmul(other, self)

    def __truediv__(self, other: Any) -> Call[Expr]:
        return _op.truediv(self, other)

    def __rtruediv__(self, other: Any) -> Call[Expr]:
        return _op.truediv(other, self)

    def __floordiv__(self, other: Any) -> Call[Expr]:
        return _op.floordiv(self, other)

    def __rfloordiv__(self, other: Any) -> Call[Expr]:
        return _op.floordiv(other, self)

    def __mod__(self, other: Any) -> Call[Expr]:
        return _op.mod(self, other)

    def __rmod__(self, other: Any) -> Call[Expr]:
        return _op.mod(other, self)

    def __pow__(self, other: Any) -> Call[Expr]:
        return _op.pow(self, other)

    def __rpow__(self, other: Any) -> Call[Expr]:
        return _op.pow(other, self)

    # Bitwise
    def __and__(self, other: Any) -> Call[Expr]:
        return _op.and_(self, other)

    def __rand__(self, other: Any) -> Call[Expr]:
        return _op.and_(other, self)

    def __or__(self, other: Any) -> Call[Expr]:
        return _op.or_(self, other)

    def __ror__(self, other: Any) -> Call[Expr]:
        return _op.or_(other, self)

    def __xor__(self, other: Any) -> Call[Expr]:
        return _op.xor(self, other)

    def __rxor__(self, other: Any) -> Call[Expr]:
        return _op.xor(other, self)

    def __lshift__(self, other: Any) -> Call[Expr]:
        return _op.lshift(self, other)

    def __rlshift__(self, other: Any) -> Call[Expr]:
        return _op.lshift(other, self)

    def __rshift__(self, other: Any) -> Call[Expr]:
        return _op.rshift(self, other)

    def __rrshift__(self, other: Any) -> Call[Expr]:
        return _op.rshift(other, self)

    # Unary
    def __neg__(self) -> Call[Expr]:
        return _op.neg(self)

    def __pos__(self) -> Call[Expr]:
        return _op.pos(self)

    def __invert__(self) -> Call[Expr]:
        return _op.invert(self)


@dataclass(frozen=True, repr=False)
class Symbol(Expr):
    """A user-defined named leaf. Compares by name.

    The name can be omitted when the symbol is assigned in a class body,
    where it takes the attribute name:

    >>> class Model:
    ...     x = Symbol()
    ...     y = Symbol("why")
    >>> Model.x, Model.y
    (Symbol('x'), Symbol('why'))
    """

    name: str = ""

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"

    def __set_name__(self, owner: type, name: str) -> None:
        if not self.name:
            object.__setattr__(self, "name", name)


@dataclass(frozen=True, repr=False, eq=False)
class Constant(Expr):
    """A library-defined named leaf, such as ``real.constants.pi``.

    Like ``Function``, constants compare by identity, so they never collide
    with a user ``Symbol`` of the same name. As with ``Symbol``, the name can
    be omitted when assigned in a class body.
    """

    name: str = ""

    def __repr__(self) -> str:
        return f"Constant({self.name!r})"

    def __set_name__(self, owner: type, name: str) -> None:
        if not self.name:
            object.__setattr__(self, "name", name)


@dataclass(frozen=True, repr=False, eq=False)
class Literal(Expr):
    """A leaf wrapping a plain Python value.

    Equality also compares the value type, so ``1``, ``1.0`` and ``True``
    are different literals.
    """

    value: Any

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Literal)
            and type(self.value) is type(other.value)
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((type(self.value), self.value))


@dataclass(frozen=True, repr=False)
class Call[T](Expr):
    """A function applied to arguments of type ``T``."""

    func: Function
    args: tuple[T, ...]

    def __repr__(self) -> str:
        return f"Call({self.func.name}, {self.args!r})"


def as_expr(value: Any) -> Expr:
    """Wrap plain Python values into a ``Literal``."""
    if isinstance(value, Expr):
        return value
    return Literal(value)


@dataclass(frozen=True, eq=False)
class Function:
    """A symbolic function. Calling it builds a ``Call`` node.

    Functions compare by identity, so they can be used as registry keys
    by algebras and implementation libraries.
    """

    name: str
    arity: int | None = None

    def __call__(self, *args: Any) -> Call[Expr]:
        if self.arity is not None and len(args) != self.arity:
            raise TypeError(
                f"{self.name} expected {self.arity} arguments, got {len(args)}"
            )
        return Call(self, tuple(map(as_expr, args)))


@dataclass(frozen=True, eq=False)
class Operator(Function):
    """A function written in infix/prefix notation.

    ``fmt`` is the Python spelling, ``precedence`` follows Python's rules
    (higher binds tighter) and ``assoc`` is ``"left"`` or ``"right"``.
    """

    fmt: str = ""
    precedence: int = 0
    assoc: str = "left"


# Expr's operator overloads need the operator definitions, which in turn
# need the classes above.
from ..lib import operators as _op
