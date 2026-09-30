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

Top-down traversals unfold with a coalgebra instead: ``ana`` builds an
expression from a seed, and ``hylo`` folds the layers as they are unfolded.
``Done`` stops the unfolding with a finished value.

The other node types are ``symbolcat.Symbol``, and ``symbolcat.lib.Function``
and ``symbolcat.lib.Constant``.

``kind_algebra`` gives the kind of a call (``symbolcat.kinds``) from the
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
    as_expr,
    cata,
    children,
    fmap,
    hylo,
    name_of,
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
    "as_expr",
    "builtin",
    "cata",
    "children",
    "entrywise",
    "fmap",
    "hylo",
    "kind_algebra",
    "name_of",
]
