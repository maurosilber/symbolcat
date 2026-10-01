"""
symbolcat.ops.algebras
~~~~~~~~~~~~~~~~~~~~~~

Extension API: the pieces needed to write new operations, as algebras
folded with ``cata``. The algebras behind ``symbolcat.ops`` are in
``symbolcat.ops.algebras.builtin``, to add rules for new functions.

Define a new operation as an algebra and fold it with ``cata``:

>>> from symbolcat import Symbol
>>> from symbolcat.ops.algebras import Algebra, Call, args_of, cata, Expr
>>> depth = Algebra("depth")
>>> @depth.register(Expr)  # any leaf
... def _(node):
...     return 1
>>> @depth.register(Call)
... def _(node):
...     return 1 + max(args_of(node))
>>> cata(depth, Symbol("x") + 1)
2

Top-down traversals unfold with a coalgebra instead: ``ana`` builds an
expression from a seed, and ``hylo`` folds the layers as they are unfolded.
``Done`` stops the unfolding with a finished value.

The other node types are ``symbolcat.Symbol``, and ``symbolcat.Function``
and ``symbolcat.Constant``.

``kind_algebra`` gives the kind of a call (``symbolcat.types``) from the
kinds of its arguments. ``entrywise(element)`` builds a rule for a function
applied entry by entry, broadcasting scalars.
"""

from ..._core import (
    Algebra,
    Call,
    Done,
    Expr,
    Literal,
    Operator,
    ana,
    args_of,
    arity_of,
    as_expr,
    assoc_of,
    cata,
    children,
    fmap,
    fmt_of,
    func_of,
    hylo,
    name_of,
    precedence_of,
    value_of,
)
from ..._core.infer import entrywise, kind_algebra
from .._printing._base import Printed
from . import builtin

__all__ = [
    "Algebra",
    "Call",
    "Done",
    "Expr",
    "Literal",
    "Operator",
    "Printed",
    "ana",
    "args_of",
    "arity_of",
    "as_expr",
    "assoc_of",
    "builtin",
    "cata",
    "children",
    "entrywise",
    "fmap",
    "fmt_of",
    "func_of",
    "hylo",
    "kind_algebra",
    "name_of",
    "precedence_of",
    "value_of",
]
