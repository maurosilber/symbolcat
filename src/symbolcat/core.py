"""
symbolcat.core
~~~~~~~~~~~~~~

The expression tree, its functor structure and the catamorphism.

An expression is a tree built from three kinds of nodes:

- ``Symbol``: a named leaf (``x``, or ``real.pi`` for library constants).
- ``Literal``: a leaf wrapping a Python value (``2``, ``0.5``).
- ``Call``: a ``Function`` applied to child expressions.

``Call`` is generic in the type of its children. A ``Call[Expr]`` is part of
an expression tree, while a ``Call[str]`` is a *layer* whose children have
already been folded into strings. ``fmap`` goes from one to the other, and
``cata`` uses it to fold a whole tree bottom-up with an *algebra*: a function
that knows how to collapse a single layer. The traversal lives here, once;
every operation (printing, LaTeX, evaluation, differentiation, ...) is only
an algebra.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any


class Expr:
    """Base class of expression nodes. Provides the operator overloads.

    ``==`` and ``hash`` are structural, so expressions can be used as
    dictionary keys. Use ``operators.eq`` and ``operators.ne`` to build
    symbolic comparisons: ``operators.eq(x, 1)``.
    """

    __slots__ = ()

    def __str__(self) -> str:
        from .ops import to_str

        return to_str(self)

    def _repr_latex_(self) -> str:
        from .ops import to_latex

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
    """A named leaf. Library constants live in a namespace (``real.pi``)."""

    name: str
    namespace: str = ""

    def __repr__(self) -> str:
        return f"Symbol({self.qualname!r})"

    @property
    def qualname(self) -> str:
        return f"{self.namespace}.{self.name}" if self.namespace else self.name


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
        return f"Call({self.func.qualname}, {self.args!r})"


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
    namespace: str = ""
    arity: int | None = None

    @property
    def qualname(self) -> str:
        return f"{self.namespace}.{self.name}" if self.namespace else self.name

    def __call__(self, *args: Any) -> Call[Expr]:
        if self.arity is not None and len(args) != self.arity:
            raise TypeError(
                f"{self.qualname} expected {self.arity} arguments, got {len(args)}"
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


# ---------------------------------------------------------------------------
# Functor structure
# ---------------------------------------------------------------------------


@singledispatch
def children(node: Any) -> tuple[Any, ...]:
    """The direct children of a node.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> children(x + 1)
    (Symbol('x'), Literal(1))
    >>> children(x)
    ()
    """
    raise TypeError(f"{type(node).__name__} is not an expression node")


@children.register(Expr)
def _(node: Expr) -> tuple[Any, ...]:
    return ()


@children.register(Call)
def _[T](node: Call[T]) -> tuple[T, ...]:
    return node.args


@singledispatch
def fmap(node: Any, f: Callable[[Any], Any]) -> Any:
    """Apply ``f`` to the direct children, keeping the node shape.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> fmap(x + y, lambda child: child.name)
    Call(op.add, ('x', 'y'))
    """
    raise TypeError(f"{type(node).__name__} is not an expression node")


@fmap.register(Expr)
def _(node: Expr, f: Callable[[Any], Any]) -> Expr:
    return node


@fmap.register(Call)
def _[T, S](node: Call[T], f: Callable[[T], S]) -> Call[S]:
    return Call(node.func, tuple(map(f, node.args)))


# ---------------------------------------------------------------------------
# Catamorphism
# ---------------------------------------------------------------------------


def cata[A](algebra: Callable[..., A], expr: Any, **ctx: Any) -> A:
    """Fold ``expr`` bottom-up with ``algebra``.

    ``algebra`` receives one layer at a time: a node whose children have
    already been replaced by their folded values (``fmap(node, fold)``).
    Extra keyword arguments are forwarded to every algebra call.

    Shared sub-expressions (the same object reachable twice) are folded once.
    The traversal uses an explicit stack, so deep trees do not hit the
    recursion limit.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> count = lambda node: 1 + sum(children(node))
    >>> cata(count, x * x + 1)
    5
    """
    expr = as_expr(expr)
    done: dict[int, A] = {}
    stack = [expr]
    while stack:
        node = stack[-1]
        if id(node) in done:
            stack.pop()
            continue
        pending = [c for c in children(node) if id(c) not in done]
        if pending:
            stack.extend(pending)
            continue
        stack.pop()
        done[id(node)] = algebra(fmap(node, lambda c: done[id(c)]), **ctx)
    return done[id(expr)]


@dataclass
class Algebra[A]:
    """An algebra assembled from handlers registered per node or function.

    A handler receives a single layer (children already folded) plus any
    context passed to ``cata``. Keys can be:

    - a ``Function`` instance (e.g. ``operators.add``): most specific;
    - a ``Function`` subclass (e.g. ``Operator``): default for that kind;
    - an ``Expr`` subclass (``Symbol``, ``Literal``, ``Call``).

    >>> depth = Algebra("depth")
    >>> @depth.register(Symbol, Literal)
    ... def _(node):
    ...     return 1
    >>> @depth.register(Call)
    ... def _(node):
    ...     return 1 + max(node.args)
    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> cata(depth, (x + 1) * 2)
    3
    """

    name: str
    _by_func: dict[Function, Callable[..., A]] = field(default_factory=dict)
    _by_kind: dict[type, Callable[..., A]] = field(default_factory=dict)
    _by_node: dict[type, Callable[..., A]] = field(default_factory=dict)

    def register(self, *keys: Function | type) -> Callable[[Callable[..., A]], Any]:
        def decorator(handler: Callable[..., A]) -> Callable[..., A]:
            for key in keys:
                if isinstance(key, Function):
                    self._by_func[key] = handler
                elif issubclass(key, Function):
                    self._by_kind[key] = handler
                elif issubclass(key, Expr):
                    self._by_node[key] = handler
                else:
                    raise TypeError(f"Cannot register {key!r}")
            return handler

        return decorator

    def copy(self, name: str) -> Algebra[A]:
        """A new algebra that starts with the same handlers."""
        return Algebra(
            name, dict(self._by_func), dict(self._by_kind), dict(self._by_node)
        )

    def handler_for(self, node: Expr) -> Callable[..., A]:
        if isinstance(node, Call):
            if (handler := self._by_func.get(node.func)) is not None:
                return handler
            if (handler := _lookup_mro(self._by_kind, type(node.func))) is not None:
                return handler
        if (handler := _lookup_mro(self._by_node, type(node))) is not None:
            return handler
        what = node.func.qualname if isinstance(node, Call) else type(node).__name__
        raise NotImplementedError(f"{self.name}: no rule for {what}")

    def __call__(self, node: Expr, **ctx: Any) -> A:
        return self.handler_for(node)(node, **ctx)


def _lookup_mro[V](registry: dict[type, V], cls: type) -> V | None:
    for base in cls.__mro__:
        if base in registry:
            return registry[base]
    return None


# Expr's operator overloads need the operator definitions, which in turn
# need the classes above.
from . import operators as _op
