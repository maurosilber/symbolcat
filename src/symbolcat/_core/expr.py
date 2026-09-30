"""
symbolcat._core.expr
~~~~~~~~~~~~~~~~~~~~

Expression nodes and symbolic functions.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, cast, overload

from .kinds import Boolean, Complex, Kind, Real
from .opaque import _init, _Opaque
from .overloads import Overloads

if TYPE_CHECKING:
    from typing_extensions import TypeVar

    # unparameterized, a Call has any kind (Python 3.13 would write
    # ``class Call[T, K: Kind = Any]``)
    _T_co = TypeVar("_T_co", covariant=True)
    _K_co = TypeVar("_K_co", bound=Kind, covariant=True, default=Any)
else:
    from typing import TypeVar

    _T_co = TypeVar("_T_co", covariant=True)
    _K_co = TypeVar("_K_co", bound=Kind, covariant=True)


class Expr[K: Kind](Overloads[K], _Opaque):
    """Base class of expression nodes. Provides the operator overloads.

    ``==`` and ``hash`` are structural, so expressions can be used as
    dictionary keys. Use ``operators.eq`` and ``operators.ne`` to build
    symbolic comparisons: ``operators.eq(x, 1)``.

    Every expression has a ``kind`` (see ``symbolcat.kinds``), which is also
    its type parameter for static type checkers: ``Symbol("x")`` is an
    ``Expr[Real]``, and ``x < 1`` an ``Expr[Boolean]``. Unparameterized,
    ``Expr`` is an expression of any kind.
    """

    __slots__ = ()

    if TYPE_CHECKING:

        @property
        def kind(self) -> K: ...

    def __str__(self) -> str:
        from ..ops._printing._str import to_str

        return to_str(self)

    def _repr_latex_(self) -> str:
        from ..ops._printing._latex import to_latex

        return f"${to_latex(self)}$"

    if not TYPE_CHECKING:  # typed in overloads.pyi
        # Comparisons
        def __lt__(self, other: Any) -> Any:
            return _op.lt(self, other)

        def __le__(self, other: Any) -> Any:
            return _op.le(self, other)

        def __gt__(self, other: Any) -> Any:
            return _op.gt(self, other)

        def __ge__(self, other: Any) -> Any:
            return _op.ge(self, other)

        # Containers
        def __getitem__(self, key: Any) -> Any:
            return _op.getitem(self, key)

        # Arithmetic
        def __add__(self, other: Any) -> Any:
            return _op.add(self, other)

        def __radd__(self, other: Any) -> Any:
            return _op.add(other, self)

        def __sub__(self, other: Any) -> Any:
            return _op.sub(self, other)

        def __rsub__(self, other: Any) -> Any:
            return _op.sub(other, self)

        def __mul__(self, other: Any) -> Any:
            return _op.mul(self, other)

        def __rmul__(self, other: Any) -> Any:
            return _op.mul(other, self)

        def __matmul__(self, other: Any) -> Any:
            return _op.matmul(self, other)

        def __rmatmul__(self, other: Any) -> Any:
            return _op.matmul(other, self)

        def __truediv__(self, other: Any) -> Any:
            return _op.truediv(self, other)

        def __rtruediv__(self, other: Any) -> Any:
            return _op.truediv(other, self)

        def __floordiv__(self, other: Any) -> Any:
            return _op.floordiv(self, other)

        def __rfloordiv__(self, other: Any) -> Any:
            return _op.floordiv(other, self)

        def __mod__(self, other: Any) -> Any:
            return _op.mod(self, other)

        def __rmod__(self, other: Any) -> Any:
            return _op.mod(other, self)

        def __pow__(self, other: Any) -> Any:
            return _op.pow(self, other)

        def __rpow__(self, other: Any) -> Any:
            return _op.pow(other, self)

        # Bitwise
        def __and__(self, other: Any) -> Any:
            return _op.and_(self, other)

        def __rand__(self, other: Any) -> Any:
            return _op.and_(other, self)

        def __or__(self, other: Any) -> Any:
            return _op.or_(self, other)

        def __ror__(self, other: Any) -> Any:
            return _op.or_(other, self)

        def __xor__(self, other: Any) -> Any:
            return _op.xor(self, other)

        def __rxor__(self, other: Any) -> Any:
            return _op.xor(other, self)

        def __lshift__(self, other: Any) -> Any:
            return _op.lshift(self, other)

        def __rlshift__(self, other: Any) -> Any:
            return _op.lshift(other, self)

        def __rshift__(self, other: Any) -> Any:
            return _op.rshift(self, other)

        def __rrshift__(self, other: Any) -> Any:
            return _op.rshift(other, self)

        # Unary
        def __neg__(self) -> Any:
            return _op.neg(self)

        def __pos__(self) -> Any:
            return _op.pos(self)

        def __invert__(self) -> Any:
            return _op.invert(self)


class _Named:
    """Mixin for objects whose name can be omitted in the constructor.

    Assigned in a class body, the object takes the attribute name. The name
    is only required where it is used (printing, evaluation, ...), so an
    expression can refer to it before the class body finishes:

    >>> class Model:
    ...     x = Symbol()
    ...     y = x + 1
    >>> print(Model.y)
    x + 1
    """

    __slots__ = ()

    if TYPE_CHECKING:
        __symbolcat__: Any

    def __set_name__(self, owner: type, name: str) -> None:
        data = self.__symbolcat__
        if not data.name:
            object.__setattr__(self, "__symbolcat__", data._replace(name=name))

    def __repr__(self) -> str:
        name = self.__symbolcat__.name
        return (
            f"{type(self).__name__}({name!r})" if name else f"{type(self).__name__}()"
        )


def name_of(obj: Symbol | Constant | Function | Algebra, /) -> str:
    """The name of ``obj``, or a clear error if it never got one.

    >>> name_of(Symbol("x"))
    'x'
    >>> name_of(Symbol())
    Traceback (most recent call last):
    ...
    ValueError: Symbol has no name: pass one to the constructor, or assign it in a class body
    """
    name = obj.__symbolcat__.name
    if not name:
        raise ValueError(
            f"{type(obj).__name__} has no name: pass one to the constructor, "
            "or assign it in a class body"
        )
    return name


def _label(obj: Symbol | Constant | Function, /) -> str:
    """The name for messages, which must not fail on unnamed objects."""
    return obj.__symbolcat__.name or "<unnamed>"


class _LeafData(NamedTuple):
    name: str
    kind: Kind


_REAL: Any = Real()


class Symbol[K: Kind](_Named, Expr[K]):
    """A user-defined named leaf. Compares by name and kind.

    Symbols are real, unless another ``kind`` is given:

    >>> from symbolcat.kinds import Matrix
    >>> Symbol("A", kind=Matrix(3, 2))
    Symbol('A', kind=Matrix(3, 2))

    The name can be omitted when the symbol is assigned in a class body,
    where it takes the attribute name:

    >>> class Model:
    ...     x = Symbol()
    ...     y = Symbol("why")
    >>> Model.x, Model.y
    (Symbol('x'), Symbol('why'))
    """

    __slots__ = ()

    @overload
    def __init__(self: Symbol[Real], name: str = "") -> None: ...
    @overload
    def __init__(self, name: str = "", *, kind: K) -> None: ...
    def __init__(self, name: str = "", *, kind: Any = _REAL) -> None:
        _init(self, _LeafData(name, kind))

    @property
    def kind(self) -> K:
        return self.__symbolcat__.kind

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self.__symbolcat__[:2] == cast(Any, other).__symbolcat__[:2]

    def __hash__(self) -> int:
        return hash(self.__symbolcat__[:2])

    def __repr__(self) -> str:
        name, kind = self.__symbolcat__[:2]
        if kind == Real():
            return super().__repr__()
        name = repr(name) if name else ""
        return f"Symbol({name}{', ' if name else ''}kind={kind!r})"


class Constant[K: Kind](_Named, Expr[K]):
    """A library-defined named leaf, such as ``real.constants.pi``.

    Like ``Function``, constants compare by identity, so they never collide
    with a user ``Symbol`` of the same name. As with ``Symbol``, the name can
    be omitted when assigned in a class body, and the ``kind`` is real
    unless given.
    """

    __slots__ = ()

    @overload
    def __init__(self: Constant[Real], name: str = "") -> None: ...
    @overload
    def __init__(self, name: str = "", *, kind: K) -> None: ...
    def __init__(self, name: str = "", *, kind: Any = _REAL) -> None:
        _init(self, _LeafData(name, kind))

    @property
    def kind(self) -> K:
        return self.__symbolcat__.kind


class Literal[K: Kind](Expr[K]):
    """A leaf wrapping a plain Python value.

    Equality also compares the value type, so ``1``, ``1.0`` and ``True``
    are different literals.

    The kind follows the value: ``Boolean`` for a ``bool``, ``Complex`` for
    a complex number, and ``Real`` otherwise.

    >>> Literal(True).kind, Literal(2).kind, Literal(1j).kind
    (Boolean(), Real(), Complex())
    """

    __slots__ = ()

    @overload
    def __init__(self: Literal[Boolean], value: bool) -> None: ...
    @overload
    def __init__(self: Literal[Real], value: float | numbers.Real) -> None: ...
    @overload
    def __init__(self: Literal[Complex], value: complex | numbers.Complex) -> None: ...
    @overload
    def __init__(self: Literal[Any], value: Any) -> None: ...
    def __init__(self, value: Any) -> None:
        _init(self, value)

    def __repr__(self) -> str:
        return f"Literal({self.__symbolcat__!r})"

    @property
    def kind(self) -> K:
        value = self.__symbolcat__
        kind: Kind
        if isinstance(value, bool):
            kind = Boolean()
        elif isinstance(value, numbers.Complex) and not isinstance(value, numbers.Real):
            kind = Complex()
        else:
            kind = Real()
        return cast(K, kind)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Literal):
            return False
        a, b = self.__symbolcat__, other.__symbolcat__
        return type(a) is type(b) and a == b

    def __hash__(self) -> int:
        value = self.__symbolcat__
        return hash((type(value), value))


class _CallData(NamedTuple):
    func: Function
    args: tuple[Any, ...]
    kind: Kind


# The default kind of a Call: to be inferred from its arguments.
_INFER: Any = Kind()


class Call(Expr[_K_co], Generic[_T_co, _K_co]):  # noqa: UP046 (for the default)
    """A function applied to arguments of type ``T``, with a result of kind
    ``K`` (any, if not given).

    ``kind`` is the kind of the result. When the arguments are expressions,
    it is inferred from their kinds (see ``kind_algebra``), and an invalid
    call raises. A layer, whose arguments are already folded, keeps the kind
    of the node it comes from (see ``fmap``), or is unspecified (``Kind()``)
    if not given. The kind is derived, so it is not compared.
    """

    __slots__ = ()

    def __init__(
        self,
        func: Function,
        args: tuple[_T_co, ...],
        kind: _K_co = _INFER,
    ) -> None:
        inferred: Kind = kind
        if kind is _INFER:
            if all(isinstance(arg, Expr) for arg in args):
                inferred = _infer(func, cast(tuple[Any, ...], args))
            else:
                inferred = Kind()  # a layer, like the seeds of an unfold
        _init(self, _CallData(func, args, inferred))

    @property
    def kind(self) -> _K_co:
        return self.__symbolcat__.kind

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self.__symbolcat__[:2] == cast(Any, other).__symbolcat__[:2]

    def __hash__(self) -> int:
        return hash(self.__symbolcat__[:2])

    def __repr__(self) -> str:
        func, args, _ = self.__symbolcat__
        return f"Call({_label(func)}, {args!r})"


# Accessors. Nodes and functions keep their data out of their namespace, so
# that subclasses (in downstream libraries) cannot collide with it; these
# functions read it.


def func_of(node: Call[Any, Any], /) -> Function:
    """The function of a call.

    >>> from symbolcat import Symbol, real
    >>> func_of(real.sin(Symbol("x"))) is real.sin
    True
    """
    return node.__symbolcat__.func


def args_of[T](node: Call[T, Any], /) -> tuple[T, ...]:
    """The arguments of a call.

    >>> from symbolcat import Symbol
    >>> args_of(Symbol("x") + 1)
    (Symbol('x'), Literal(1))
    """
    return node.__symbolcat__.args


def value_of(node: Literal[Any], /) -> Any:
    """The Python value of a literal.

    >>> value_of(Literal(2))
    2
    """
    return node.__symbolcat__


def arity_of(func: Function, /) -> int | None:
    """The number of arguments of a function, or ``None`` if any."""
    return func.__symbolcat__.arity


def fmt_of(op: Operator, /) -> str:
    """The Python spelling of an operator, as a format string (``"{} + {}"``)."""
    return op.__symbolcat__.fmt


def precedence_of(op: Operator, /) -> int:
    """The precedence of an operator, following Python (higher binds tighter)."""
    return op.__symbolcat__.precedence


def assoc_of(op: Operator, /) -> str:
    """The associativity of an operator: ``"left"``, ``"right"`` or ``"none"``."""
    return op.__symbolcat__.assoc


def as_expr(value: Any, /) -> Expr:
    """Wrap plain Python values into a ``Literal``."""
    if isinstance(value, Expr):
        return value
    return Literal(value)


class _FunctionData(NamedTuple):
    name: str
    arity: int | None


class Function[**P, R: Kind](_Named, _Opaque):
    """A symbolic function. Calling it builds a ``Call`` node.

    Functions compare by identity, so they can be used as registry keys
    by algebras and implementation libraries. As with ``Symbol``, the name
    can be omitted when assigned in a class body:

    >>> class Model:
    ...     f = Function(arity=1)
    >>> Model.f
    Function('f')

    For static type checkers, ``Function[P, R]`` takes the arguments ``P``
    and returns an expression of kind ``R``. Unparameterized, it takes any
    arguments, and returns an expression of any kind:

    >>> from symbolcat.kinds import Matrix, Vector
    >>> outer: Function[[Expr[Vector], Expr[Vector]], Matrix] = Function("outer")
    """

    __slots__ = ()

    def __init__(
        self: Function[..., Any], name: str = "", *, arity: int | None = None
    ) -> None:
        _init(self, _FunctionData(name, arity))

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Call[Expr, R]:
        if kwargs:
            raise TypeError(f"{_label(self)} takes no keyword arguments")
        arity = self.__symbolcat__.arity
        if arity is not None and len(args) != arity:
            raise TypeError(
                f"{_label(self)} expected {arity} arguments, got {len(args)}"
            )
        return Call(self, tuple(map(as_expr, args)))


class _OperatorData(NamedTuple):
    name: str
    arity: int | None
    fmt: str
    precedence: int
    assoc: str


class Operator(Function):
    """A function written in infix/prefix notation.

    ``fmt`` is the Python spelling, ``precedence`` follows Python's rules
    (higher binds tighter) and ``assoc`` is ``"left"`` or ``"right"``.
    """

    __slots__ = ()

    def __init__(
        self,
        name: str = "",
        *,
        arity: int | None = None,
        fmt: str = "",
        precedence: int = 0,
        assoc: str = "left",
    ) -> None:
        _init(self, _OperatorData(name, arity, fmt, precedence, assoc))


# Kind inference and Expr's operator overloads need the algebra and the
# operator definitions, which in turn need the classes above.
from .. import operators as _op
from .infer import infer as _infer

if TYPE_CHECKING:
    from .algebra import Algebra
