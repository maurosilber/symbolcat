"""
symbolcat.ops.algebras.builtin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algebras behind ``symbolcat.ops``. Register a rule on one of them to
teach an operation about a new function:

>>> from symbolcat import Symbol
>>> from symbolcat.lib import Function
>>> from symbolcat.ops.algebras import Printed
>>> from symbolcat.ops.algebras.builtin import latex_algebra
>>> from symbolcat.ops.printing import to_latex
>>> sinc = Function("sinc", arity=1)
>>> @latex_algebra.register(sinc)
... def _(node):
...     return Printed(rf"\\operatorname{{sinc}}\\left({node.args[0].text}\\right)")
>>> print(to_latex(sinc(Symbol("x"))))
\\operatorname{sinc}\\left(x\\right)
"""

from .._analysis._dimensions import dimensions_algebra
from .._analysis._inspect import free_symbols_algebra
from .._analysis._polynomial import coefficients_algebra
from .._calculus._diff import diff_algebra
from .._calculus._reverse import reverse_algebra
from .._calculus._series import series_algebra
from .._evaluate import evaluate_algebra
from .._printing._latex import latex_algebra
from .._printing._mathml import mathml_algebra
from .._printing._str import str_algebra
from .._transform._arrays import arrays_algebra
from .._transform._simplify import simplify_algebra
from .._transform._substitute import substitute_algebra

__all__ = [
    "arrays_algebra",
    "coefficients_algebra",
    "diff_algebra",
    "dimensions_algebra",
    "evaluate_algebra",
    "free_symbols_algebra",
    "latex_algebra",
    "mathml_algebra",
    "reverse_algebra",
    "series_algebra",
    "simplify_algebra",
    "str_algebra",
    "substitute_algebra",
]
