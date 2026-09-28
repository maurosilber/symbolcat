"""
symbolcat.ops._calculus._diff
~~~~~~~~~~~~~~~~~~~

Symbolic differentiation.

Differentiating needs the value of each sub-expression, not only its
derivative (the product rule uses both). The algebra therefore folds every
node into a ``(value, derivative)`` pair: a catamorphism into a product
type, the same trick used by forward-mode automatic differentiation.

The derivatives of the variables are the ``seed``, a mapping from each
variable to its derivative (its tangent); the rest have none. The seed
``{x: 1}`` gives the partial derivative with respect to ``x``, and seeding
every unknown with its time derivative gives the total time derivative.
A variable is a symbol or a time derivative of one, ``D(x)``, which is an
unknown of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ... import linalg, real
from ..._core import Algebra, Call, Constant, Expr, Function, Literal, Symbol, cata
from ..._derivative import D, Variable, is_variable
from ...lib import operators as op
from ...linalg import Array, matrix, vector
from .._analysis._inspect import free_symbols
from .._transform._simplify import is_number, simplify_naive

type Dual = tuple[Expr, Expr]
type Seed = Mapping[Expr, Expr]
type Rule = Callable[[tuple[Expr, ...], tuple[Expr, ...]], Expr]

ZERO = Literal(0)
ONE = Literal(1)

diff_algebra = Algebra[Dual]("diff")


@diff_algebra.register(Symbol)
def _(node: Symbol, *, seed: Seed) -> Dual:
    return node, seed.get(node, ZERO)


@diff_algebra.register(Constant, Literal)
def _(node: Constant | Literal, *, seed: Seed) -> Dual:
    return node, ZERO


@diff_algebra.register(D)
def _(node: Call[Dual], *, seed: Seed) -> Dual:
    """``D(x)`` is a variable, independent of ``x``."""
    ((value, _),) = node.args
    variable = D(value)
    if not is_variable(variable):
        raise NotImplementedError(
            f"diff: D of an expression, {variable}; expand it with time_derivative"
        )
    return variable, seed.get(variable, ZERO)


def rule(*funcs: Function) -> Callable[[Rule], Rule]:
    """Register a derivative rule ``(args, dargs) -> derivative``."""

    def decorator(derivative: Rule) -> Rule:
        @diff_algebra.register(*funcs)
        def _(node: Call[Dual], *, seed: Seed) -> Dual:
            args = tuple(value for value, _ in node.args)
            dargs = tuple(d for _, d in node.args)
            return node.func(*args), derivative(args, dargs)

        return derivative

    return decorator


@rule(op.add)
def _(args, dargs):
    return dargs[0] + dargs[1]


@rule(op.sub)
def _(args, dargs):
    return dargs[0] - dargs[1]


@rule(op.neg)
def _(args, dargs):
    return -dargs[0]


@rule(op.pos)
def _(args, dargs):
    return dargs[0]


@rule(op.mul)
def _(args, dargs):
    (a, b), (da, db) = args, dargs
    return da * b + a * db


@rule(op.truediv)
def _(args, dargs):
    (a, b), (da, db) = args, dargs
    return (da * b - a * db) / b**2


@rule(op.pow)
def _(args, dargs):
    (a, b), (da, db) = args, dargs
    if is_number(simplify_naive(db), 0):
        return b * a ** (b - 1) * da
    return a**b * (db * real.log(a) + b * da / a)


@diff_algebra.register(Array, linalg.transpose, linalg.trace, linalg.sum)
def _(node: Call[Dual], *, seed: Seed) -> Dual:
    """Linear functions: the derivative is the function of the derivatives.
    Arrays are differentiated entrywise."""
    values = tuple(value for value, _ in node.args)
    derivatives = tuple(d for _, d in node.args)
    return node.func(*values), node.func(*derivatives)


@rule(op.getitem)
def _(args, dargs):
    (_, key), (da, _) = args, dargs
    return da[key]


@rule(op.matmul)
def _(args, dargs):
    (a, b), (da, db) = args, dargs
    return da @ b + a @ db


@rule(linalg.det)
def _(args, dargs):
    ((a,), (da,)) = args, dargs
    return linalg.det(a) * linalg.trace(linalg.inv(a) @ da)


@rule(linalg.inv)
def _(args, dargs):
    ((a,), (da,)) = args, dargs
    return -(linalg.inv(a) @ da @ linalg.inv(a))


@rule(linalg.norm)
def _(args, dargs):
    ((a,), (da,)) = args, dargs
    return linalg.sum(a * da) / linalg.norm(a)


@rule(linalg.cross)
def _(args, dargs):
    (a, b), (da, db) = args, dargs
    return linalg.cross(da, b) + linalg.cross(a, db)


def chain(outer: Callable[[Expr], Expr], /) -> Rule:
    """Rule for f(a), given f'(a)."""
    return lambda args, dargs: outer(args[0]) * dargs[0]


rule(real.exp)(chain(real.exp))
rule(real.log)(chain(lambda a: 1 / a))
rule(real.sqrt)(chain(lambda a: 1 / (2 * real.sqrt(a))))
rule(real.sin)(chain(real.cos))
rule(real.cos)(chain(lambda a: -real.sin(a)))
rule(real.tan)(chain(lambda a: 1 / real.cos(a) ** 2))
rule(real.asin)(chain(lambda a: 1 / real.sqrt(1 - a**2)))
rule(real.acos)(chain(lambda a: -1 / real.sqrt(1 - a**2)))
rule(real.atan)(chain(lambda a: 1 / (1 + a**2)))
rule(real.sinh)(chain(real.cosh))
rule(real.cosh)(chain(real.sinh))
rule(real.tanh)(chain(lambda a: 1 - real.tanh(a) ** 2))


def diff(expr: Any, var: Variable, /) -> Expr:
    """Derivative of ``expr`` with respect to ``var``.

    ``var`` is a symbol or a time derivative of one, ``D(x)``, which is
    independent of ``x``.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(diff(x**3 + y * x, x))
    3 * x ** 2 + y
    >>> print(diff(real.sin(x) * x, x))
    cos(x) * x + sin(x)
    >>> from symbolcat import D
    >>> print(diff(D(x) ** 2 + x, D(x)))
    2 * D(x)
    """
    if not is_variable(var):
        raise TypeError(
            "diff: can only differentiate with respect to a Symbol "
            f"or its time derivative, not {var}"
        )
    return jvp(expr, {var: ONE})


def jvp(expr: Any, seed: Seed, /) -> Expr:
    """Jacobian-vector product: the directional derivative of ``expr``, where
    ``seed`` maps each variable to its derivative (the direction).

    Variables are symbols or time derivatives of them, ``D(x)``.

    >>> from symbolcat import Symbol
    >>> x, y, dx, dy = Symbol("x"), Symbol("y"), Symbol("dx"), Symbol("dy")
    >>> print(jvp(x * y, {x: dx, y: dy}))
    dx * y + x * dy
    """
    _, derivative = cata(diff_algebra, expr, seed=seed)
    return simplify_naive(derivative)


def gradient(expr: Any, vars: Sequence[Variable] | None = None, /) -> Call[Expr]:
    """Vector of derivatives of ``expr`` with respect to each of ``vars``.

    ``vars`` defaults to the free symbols of ``expr``.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(gradient(x**2 * y))
    vector([2 * x * y, x ** 2])
    """
    if vars is None:
        vars = free_symbols(expr)
    return vector(diff(expr, var) for var in vars)


def jacobian(
    exprs: Sequence[Any] | Call[Expr], vars: Sequence[Variable] | None = None, /
) -> Call[Expr]:
    """Matrix of derivatives of each of ``exprs`` (rows) with respect to
    ``vars`` (columns).

    ``exprs`` is a sequence or a vector. ``vars`` defaults to the free
    symbols of ``exprs``, in order of appearance.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(jacobian([x * y, x + y]))
    matrix([[y, x], [1, 1]])
    """
    if isinstance(exprs, Call):
        if not (isinstance(exprs.func, Array) and len(exprs.func.shape) == 1):
            raise TypeError(f"jacobian: expected a sequence or a vector, got {exprs}")
        exprs = exprs.args
    if vars is None:
        vars = tuple(dict.fromkeys(s for e in exprs for s in free_symbols(e)))
    return matrix([diff(expr, var) for var in vars] for expr in exprs)


def hessian(expr: Any, vars: Sequence[Variable] | None = None, /) -> Call[Expr]:
    """Matrix of second derivatives of ``expr`` with respect to each pair of
    ``vars``.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(hessian(x**2 * y))
    matrix([[2 * y, 2 * x], [2 * x, 0]])
    """
    if vars is None:
        vars = free_symbols(expr)
    return jacobian(gradient(expr, vars), vars)
