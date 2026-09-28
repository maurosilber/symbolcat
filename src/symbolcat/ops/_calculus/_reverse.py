"""
symbolcat.ops._calculus._reverse
~~~~~~~~~~~~~~~~~~~~~~

Reverse-mode differentiation.

Forward mode (``diff``) folds each node into ``(value, derivative)``, one
pass per variable. Reverse mode computes the derivatives with respect to
all variables in one backward pass: starting from the output, each node
receives its cotangent (the derivative of the output with respect to the
node) and passes a share of it to its arguments.

The backward pass goes top-down and a node with several parents adds up
their contributions, so it is not a catamorphism. Instead, the expression
is shared (``share``), sorted children-first, and walked in reverse. Each
node is visited once, and the derivatives share sub-expressions.

A rule receives the node, its cotangent ``g`` and the shapes of its
arguments and result, and returns the cotangent of each argument.

Complex nodes (``symbolcat.complex``) have complex cotangents, paired with
tangents as ``re(g * dz)``. For a holomorphic function, the rule is then the
same as for a real one: ``f'(a) * g``. Symbols are real, and so is the
cotangent of every real node: where a complex node passes a cotangent to a
real argument, only its real part is kept.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from typing import Any

from ... import complex as cx
from ... import linalg, real
from ..._core import (
    Algebra,
    Call,
    Expr,
    Function,
    Literal,
    Symbol,
    as_expr,
    children,
    fmap,
    name_of,
)
from ...lib import operators as op
from ...linalg import Array, array, identity, matrix, transpose, vector
from .._analysis._inspect import complex_algebra, free_symbols, is_complex
from .._transform._arrays import Shape, _shapes, arrays_algebra, entries
from .._transform._share import share
from .._transform._simplify import simplify_naive

type Cotangents = tuple[Expr, ...]

# The cotangent of an argument without a derivative: an error if it is needed.
NO_DERIVATIVE = Literal(None)
type Rule = Callable[..., Cotangents]

reverse_algebra = Algebra[Cotangents]("reverse")


def rule(*funcs: Function | type) -> Callable[[Rule], Rule]:
    """Register a rule ``(node, g, shapes, shape) -> argument cotangents``."""

    def decorator(pullback: Rule) -> Rule:
        @reverse_algebra.register(*funcs)
        def _(node: Call[Expr], *, g: Expr, shapes: tuple[Shape, ...], shape: Shape):
            return pullback(node, g, shapes, shape)

        return pullback

    return decorator


# Entrywise functions. A scalar broadcast against an array is handled by
# ``unbroadcast``, after the rule.


@rule(op.add)
def _(node, g, shapes, shape):
    return g, g


@rule(op.sub)
def _(node, g, shapes, shape):
    return g, -g


@rule(op.neg)
def _(node, g, shapes, shape):
    return (-g,)


@rule(op.pos)
def _(node, g, shapes, shape):
    return (g,)


@rule(op.mul)
def _(node, g, shapes, shape):
    a, b = node.args
    return b * g, a * g


@rule(op.truediv)
def _(node, g, shapes, shape):
    a, b = node.args
    return g / b, -a / b**2 * g


@rule(op.pow)
def _(node, g, shapes, shape):
    # The exponent's cotangent is only used if the exponent depends on a
    # variable, so log(a) does not appear for constant exponents.
    a, b = node.args
    log = cx.log if is_complex(a) else real.log
    return b * a ** (b - 1) * g, node * log(a) * g


def chain(outer: Callable[[Expr], Expr], /) -> Rule:
    """Rule for f(a), given f'(a)."""
    return lambda node, g, shapes, shape: (outer(node.args[0]) * g,)


rule(real.exp)(lambda node, g, shapes, shape: (node * g,))
rule(real.log)(chain(lambda a: 1 / a))
rule(real.sqrt)(lambda node, g, shapes, shape: (g / (2 * node),))
rule(real.sin)(chain(real.cos))
rule(real.cos)(chain(lambda a: -real.sin(a)))
rule(real.tan)(chain(lambda a: 1 / real.cos(a) ** 2))
rule(real.asin)(chain(lambda a: 1 / real.sqrt(1 - a**2)))
rule(real.acos)(chain(lambda a: -1 / real.sqrt(1 - a**2)))
rule(real.atan)(chain(lambda a: 1 / (1 + a**2)))
rule(real.sinh)(chain(real.cosh))
rule(real.cosh)(chain(real.sinh))
rule(real.tanh)(lambda node, g, shapes, shape: ((1 - node**2) * g,))
rule(real.erf)(chain(lambda a: 2 / real.sqrt(real.constants.pi) * real.exp(-(a**2))))


@rule(real.gammainc)
def _(node, g, shapes, shape):
    a, x = node.args
    return NO_DERIVATIVE, x ** (a - 1) * real.exp(-x) / real.gamma(a) * g


rule(cx.exp)(lambda node, g, shapes, shape: (node * g,))
rule(cx.log)(chain(lambda a: 1 / a))
rule(cx.log10)(chain(lambda a: 1 / (a * cx.log(10))))
rule(cx.sqrt)(lambda node, g, shapes, shape: (g / (2 * node),))
rule(cx.sin)(chain(cx.cos))
rule(cx.cos)(chain(lambda a: -cx.sin(a)))
rule(cx.tan)(chain(lambda a: 1 / cx.cos(a) ** 2))
rule(cx.asin)(chain(lambda a: 1 / cx.sqrt(1 - a**2)))
rule(cx.acos)(chain(lambda a: -1 / cx.sqrt(1 - a**2)))
rule(cx.atan)(chain(lambda a: 1 / (1 + a**2)))
rule(cx.sinh)(chain(cx.cosh))
rule(cx.cosh)(chain(cx.sinh))
rule(cx.tanh)(lambda node, g, shapes, shape: ((1 - node**2) * g,))

# Not holomorphic. re, im, abs and phase are real, so their cotangent is.
rule(cx.re)(lambda node, g, shapes, shape: (g,))
rule(cx.im)(lambda node, g, shapes, shape: (-cx.constants.I * g,))
rule(cx.conj)(lambda node, g, shapes, shape: (cx.conj(g),))


@rule(cx.abs)
def _(node, g, shapes, shape):
    (z,) = node.args
    return (g * cx.conj(z) / node,)


@rule(cx.phase)
def _(node, g, shapes, shape):
    (z,) = node.args
    return (-cx.constants.I * g * cx.conj(z) / cx.abs(z) ** 2,)


@rule(cx.rect)
def _(node, g, shapes, shape):
    _, phi = node.args
    return cx.re(g * cx.rect(1, phi)), cx.re(cx.constants.I * g * node)


@rule(real.ifelse)
def _(node, g, shapes, shape):
    condition, _, _ = node.args
    return Literal(0), real.ifelse(condition, g, 0), real.ifelse(condition, 0, g)


@rule(op.eq, op.ne, op.lt, op.le, op.gt, op.ge, op.and_, op.or_, op.invert)
def _(node, g, shapes, shape):
    return tuple(Literal(0) for _ in node.args)


# Arrays


def entry(expr: Expr, shape: Shape, index: tuple[int, ...], /) -> Expr:
    """An entry of an array: taken directly from an explicit array, or
    indexed otherwise."""
    if (items := entries(expr)) is not None:
        return items[_flat(shape, index)]
    for i in index:
        expr = expr[i]
    return expr


def _flat(shape: Shape, index: tuple[int, ...], /) -> int:
    return index[0] if len(shape) == 1 else index[0] * shape[1] + index[1]


def outer(u: Expr, n: int, v: Expr, m: int, /) -> Call[Expr]:
    """The outer product of two vectors of lengths ``n`` and ``m``."""
    return matrix(
        [entry(u, (n,), (i,)) * entry(v, (m,), (j,)) for j in range(m)]
        for i in range(n)
    )


@rule(Array)
def _(node, g, shapes, shape):
    return tuple(entry(g, shape, index) for index in product(*map(range, shape)))


@rule(op.getitem)
def _(node, g, shapes, shape):
    """An entry of a vector, or a row of a matrix, scattered into zeros."""
    _, key = node.args
    (s, _) = shapes
    if not s:
        raise NotImplementedError("reverse: getitem of a scalar-shaped container")
    if not (isinstance(key, Literal) and type(key.value) is int):
        raise NotImplementedError(f"reverse: getitem with a non-integer key {key}")
    k = key.value % s[0]
    if len(s) == 1:
        cotangent = vector(g if i == k else 0 for i in range(s[0]))
    else:
        cotangent = matrix(
            [entry(g, shape, (j,)) if i == k else 0 for j in range(s[1])]
            for i in range(s[0])
        )
    return cotangent, Literal(0)


@rule(op.matmul)
def _(node, g, shapes, shape):
    a, b = node.args
    sa, sb = shapes
    match len(sa), len(sb):
        case 2, 2:
            return g @ transpose(b), transpose(a) @ g
        case 1, 2:
            return b @ g, outer(a, sa[0], g, sb[1])
        case 2, 1:
            return outer(g, sa[0], b, sb[0]), transpose(a) @ g
        case _:
            return g * b, g * a


@rule(linalg.transpose)
def _(node, g, shapes, shape):
    return (transpose(g),)


@rule(linalg.trace)
def _(node, g, shapes, shape):
    ((n, _),) = shapes
    return (g * identity(n),)


@rule(linalg.sum)
def _(node, g, shapes, shape):
    (s,) = shapes
    return (array(s)(*[g] * math.prod(s)),)


@rule(linalg.det)
def _(node, g, shapes, shape):
    (a,) = node.args
    return (g * node * transpose(linalg.inv(a)),)


@rule(linalg.inv)
def _(node, g, shapes, shape):
    return (-(transpose(node) @ g @ transpose(node)),)


@rule(linalg.norm)
def _(node, g, shapes, shape):
    (a,) = node.args
    return (g * a / node,)


@rule(linalg.cross)
def _(node, g, shapes, shape):
    a, b = node.args
    return linalg.cross(b, g), linalg.cross(g, a)


def unbroadcast(cotangent: Expr, arg: Shape, out: Shape, /) -> Expr:
    """The cotangent of a scalar broadcast against an array sums over it."""
    if not arg and out:
        return linalg.sum(cotangent)
    return cotangent


def _postorder(expr: Expr, /) -> list[Expr]:
    """The distinct nodes (by identity), each after its children."""
    order: list[Expr] = []
    done: set[int] = set()
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
        done.add(id(node))
        order.append(node)
    return order


_pack = Function("pack")  # groups the results, to simplify them together


def vjp(
    expr: Any,
    vars: Sequence[Symbol],
    cotangent: Any = 1,
    /,
    *,
    shapes: Mapping[Any, Any] | None = None,
) -> tuple[Expr, ...]:
    """Vector-Jacobian product: the derivative of ``expr`` with respect to
    each of ``vars``, weighted by ``cotangent``, in one backward pass.

    ``cotangent`` has the shape of ``expr``. Symbols are scalars, unless
    ``shapes`` maps them (or their names) to a shape. A variable can be an
    array, and its derivative then has its shape.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> for d in vjp(real.sin(x * y), (x, y)):
    ...     print(d)
    y * cos(x * y)
    x * cos(x * y)
    """
    for var in vars:
        if not isinstance(var, Symbol):
            raise TypeError(
                f"vjp: can only differentiate with respect to a Symbol, not {var}"
            )
    symbol_shapes = _shapes(shapes)
    expr = share(expr)
    order = _postorder(expr)

    # Forward pass: shapes, and the nodes that depend on a variable.
    targets = set(vars)
    shape_of: dict[int, Shape] = {}
    is_complex: dict[int, bool] = {}
    active: set[int] = set()
    for node in order:
        layer = fmap(node, lambda c: (c, shape_of[id(c)]))
        _, shape_of[id(node)] = arrays_algebra(layer, shapes=symbol_shapes)
        is_complex[id(node)] = complex_algebra(fmap(node, lambda c: is_complex[id(c)]))
        if (isinstance(node, Symbol) and node in targets) or any(
            id(c) in active for c in children(node)
        ):
            active.add(id(node))

    # Backward pass: push the cotangents from the output to the variables.
    adjoint: dict[int, Expr] = {}
    if id(expr) in active:
        adjoint[id(expr)] = as_expr(cotangent)
    for node in reversed(order):
        g = adjoint.get(id(node))
        if g is None or not isinstance(node, Call):
            continue
        arg_shapes = tuple(shape_of[id(arg)] for arg in node.args)
        shape = shape_of[id(node)]
        pullback = reverse_algebra.handler_for(node)
        cotangents = pullback(node, g=g, shapes=arg_shapes, shape=shape)
        for i, (arg, s, c) in enumerate(
            zip(node.args, arg_shapes, cotangents, strict=True)
        ):
            if id(arg) not in active:
                continue
            if c is NO_DERIVATIVE:
                raise NotImplementedError(
                    f"reverse: no derivative of {name_of(node.func)} "
                    f"with respect to argument {i}"
                )
            if node.func in linalg.ENTRYWISE:
                c = unbroadcast(c, s, shape)
            if not is_complex[id(arg)]:
                if node.func is cx.im or node.func is cx.phase:
                    continue  # constant on the reals
                if is_complex[id(node)]:
                    c = cx.re(c)
            previous = adjoint.get(id(arg))
            adjoint[id(arg)] = c if previous is None else previous + c

    nodes = {node: node for node in order if isinstance(node, Symbol)}
    results = [
        adjoint.get(id(nodes[var]), Literal(0)) if var in nodes else Literal(0)
        for var in vars
    ]
    packed = simplify_naive(_pack(*results))
    assert isinstance(packed, Call)
    return packed.args


def reverse_gradient(
    expr: Any,
    vars: Sequence[Symbol] | None = None,
    /,
    *,
    shapes: Mapping[Any, Any] | None = None,
) -> Call[Expr]:
    """Gradient of a scalar ``expr`` with respect to each of the scalar
    ``vars``, in one backward pass. Equal to ``gradient``.

    ``vars`` defaults to the free symbols of ``expr``.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(reverse_gradient(x**2 * y))
    vector([2 * x * y, x ** 2])
    """
    if vars is None:
        vars = free_symbols(expr)
    return vector(vjp(expr, vars, shapes=shapes))
