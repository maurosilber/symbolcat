"""
symbolcat
~~~~~~~~~

A minimalistic symbolic package, organized around catamorphisms.

>>> from symbolcat import Symbol, real, evaluate, to_latex
>>> x, y = Symbol("x"), Symbol("y")
>>> expr = x + 3 * real.sin(y)
>>> print(expr)
x + 3 * sin(y)
>>> print(to_latex(expr))
x + 3 \\cdot \\sin\\left(y\\right)
>>> evaluate(expr, {x: 1, y: 0})
1.0
"""

from . import operators, real
from .core import (
    Algebra,
    Call,
    Constant,
    Expr,
    Function,
    Literal,
    Operator,
    Symbol,
    cata,
    children,
    fmap,
)
from .ops import (
    count_nodes,
    depth,
    diff,
    evaluate,
    free_symbols,
    simplify,
    substitute,
    to_latex,
    to_str,
)


def symbols(names: str) -> tuple[Symbol, ...]:
    """Create symbols from a space separated string.

    >>> x, y = symbols("x y")
    >>> x
    Symbol('x')
    """
    return tuple(map(Symbol, names.split()))


__all__ = [
    "Algebra",
    "Call",
    "Constant",
    "Expr",
    "Function",
    "Literal",
    "Operator",
    "Symbol",
    "cata",
    "children",
    "count_nodes",
    "depth",
    "diff",
    "evaluate",
    "fmap",
    "free_symbols",
    "operators",
    "real",
    "simplify",
    "substitute",
    "symbols",
    "to_latex",
    "to_str",
]
