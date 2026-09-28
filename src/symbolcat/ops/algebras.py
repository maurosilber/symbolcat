"""
symbolcat.ops.algebras
~~~~~~~~~~~~~~~~~~~~~~

Extension API: the algebras behind ``symbolcat.ops`` and the pieces needed to
write new ones.

Add a rule for a new function to an existing operation:

>>> from symbolcat import Symbol, ops
>>> from symbolcat.lib import Function
>>> from symbolcat.ops.algebras import Printed, latex_algebra
>>> sinc = Function("sinc", arity=1)
>>> @latex_algebra.register(sinc)
... def _(node):
...     return Printed(rf"\\operatorname{{sinc}}\\left({node.args[0].text}\\right)")
>>> print(ops.to_latex(sinc(Symbol("x"))))
\\operatorname{sinc}\\left(x\\right)

Or define a new operation as an algebra and fold it with ``cata``:

>>> from symbolcat.ops.algebras import Algebra, Call, Expr, cata
>>> depth = Algebra("depth")
>>> @depth.register(Expr)  # any leaf
... def _(node):
...     return 1
>>> @depth.register(Call)
... def _(node):
...     return 1 + max(node.args)
>>> cata(depth, Symbol("x") + 1)
2
"""

from .._core import (
    Algebra,
    Call,
    Constant,
    Expr,
    Function,
    Literal,
    Operator,
    Symbol,
    as_expr,
    cata,
    children,
    fmap,
    name_of,
)
from ._arrays import arrays_algebra
from ._diff import diff_algebra
from ._dimensions import dimensions_algebra
from ._evaluate import evaluate_algebra
from ._inspect import free_symbols_algebra
from ._latex import latex_algebra
from ._mathml import mathml_algebra
from ._polynomial import coefficients_algebra
from ._printing import Printed, str_algebra
from ._reverse import reverse_algebra
from ._series import series_algebra
from ._simplify import simplify_algebra
from ._substitute import substitute_algebra

__all__ = [
    "Algebra",
    "Call",
    "Constant",
    "Expr",
    "Function",
    "Literal",
    "Operator",
    "Printed",
    "Symbol",
    "arrays_algebra",
    "as_expr",
    "cata",
    "children",
    "coefficients_algebra",
    "diff_algebra",
    "dimensions_algebra",
    "evaluate_algebra",
    "fmap",
    "free_symbols_algebra",
    "latex_algebra",
    "mathml_algebra",
    "name_of",
    "reverse_algebra",
    "series_algebra",
    "simplify_algebra",
    "str_algebra",
    "substitute_algebra",
]
