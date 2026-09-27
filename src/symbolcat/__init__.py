"""
symbolcat
~~~~~~~~~

A minimalistic symbolic package, organized around catamorphisms.

- ``Symbol``: create symbols and combine them with Python operators.
- ``real``: mathematical functions and constants.
- ``linalg``: vectors and matrices of expressions.
- ``ops``: operations on expressions (printing, LaTeX, evaluation, ...).
- ``lib``: defining new functions, and implementation libraries.

>>> from symbolcat import Symbol, ops, real
>>> x, y = Symbol("x"), Symbol("y")
>>> expr = x + 3 * real.sin(y)
>>> print(expr)
x + 3 * sin(y)
>>> print(ops.to_latex(expr))
x + 3 \\cdot \\sin\\left(y\\right)
>>> ops.evaluate(expr, {x: 1, y: 0})
1.0
"""

from . import lib, linalg, ops, real
from ._core import Symbol

__all__ = ["Symbol", "lib", "linalg", "ops", "real"]
