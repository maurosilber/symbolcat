"""
symbolcat.ops.algebras
~~~~~~~~~~~~~~~~~~~~~~

Extension API: the pieces needed to write new operations, as algebras
folded with ``cata``. The algebras behind ``symbolcat.ops`` are in
``symbolcat.ops.algebras.builtin``, to add rules for new functions.

Define a new operation as an algebra and fold it with ``cata``:

>>> from symbolcat import Symbol
>>> from symbolcat.ops.algebras import Algebra, Call, cata, Expr
>>> depth = Algebra("depth")
>>> @depth.register(Expr)  # any leaf
... def _(node):
...     return 1
>>> @depth.register(Call)
... def _(node):
...     return 1 + max(node.args)
>>> cata(depth, Symbol("x") + 1)
2

The other node types are ``symbolcat.Symbol``, and ``symbolcat.lib.Function``
and ``symbolcat.lib.Constant``.
"""

from ..._core import (
    Algebra,
    Call,
    Expr,
    Literal,
    Operator,
    as_expr,
    cata,
    children,
    fmap,
    name_of,
)
from .._printing._base import Printed
from . import builtin

__all__ = [
    "Algebra",
    "Call",
    "Expr",
    "Literal",
    "Operator",
    "Printed",
    "as_expr",
    "builtin",
    "cata",
    "children",
    "fmap",
    "name_of",
]
