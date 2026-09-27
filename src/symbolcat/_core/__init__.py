"""
symbolcat._core
~~~~~~~~~~~~~~~

The expression tree, its functor structure and the catamorphism.

An expression is a tree built from these kinds of nodes:

- ``Symbol``: a user-defined named leaf (``x``).
- ``Constant``: a library-defined named leaf (``pi``).
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

from .algebra import Algebra
from .cata import cata
from .children import children
from .expr import (
    Call,
    Constant,
    Expr,
    Function,
    Literal,
    Operator,
    Symbol,
    as_expr,
    name_of,
)
from .fmap import fmap

__all__ = [
    "Algebra",
    "Call",
    "Constant",
    "Expr",
    "Function",
    "Literal",
    "Operator",
    "Symbol",
    "as_expr",
    "cata",
    "children",
    "fmap",
    "name_of",
]
