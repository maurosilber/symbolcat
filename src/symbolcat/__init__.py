"""
symbolcat
~~~~~~~~~

A minimalistic symbolic package, organized around catamorphisms.

- ``Symbol``: create symbols and combine them with Python operators.
  Symbols compare by name and kind; a ``Dummy`` compares by identity.
- ``Function``, ``Constant``: define new symbolic functions and constants.
- ``D``: the time derivative of a symbol, ``D(x)``, as an unknown of its own.
  ``delay(x, tau)`` and ``partial(u, x)`` build delay and partial
  differential equations.
- ``real``: mathematical functions and constants.
- ``complex``: complex functions and constants.
- ``linalg``: vectors and matrices of expressions.
- ``stats``: probability distributions, and propagation of uncertainty.
- ``ops``: operations on expressions (printing, LaTeX, evaluation, ...).
- ``models``: models of dynamical systems (DAEs, ODEs, SDEs, PDEs, reaction
  networks) and their analysis.
- ``operators``: the operators produced by Python syntax (``x + y``, ...).
- ``backends``: implementation libraries for ``ops.evaluate`` and
  ``ops.compile`` (``std``, ``numpy``, ``sympy``).

>>> from symbolcat import Symbol, ops, real
>>> x, y = Symbol("x"), Symbol("y")
>>> expr = x + 3 * real.sin(y)
>>> print(expr)
x + 3 * sin(y)
>>> print(ops.printing.to_latex(expr))
x + 3 \\cdot \\sin\\left(y\\right)
>>> ops.evaluate(expr, {x: 1, y: 0})
1.0
"""

from . import backends, complex, linalg, models, operators, ops, real, stats
from ._core import Constant, Dummy, Function, Symbol
from ._derivative import D, delay, partial

__all__ = [
    "Constant",
    "D",
    "Dummy",
    "Function",
    "Symbol",
    "backends",
    "complex",
    "delay",
    "linalg",
    "models",
    "operators",
    "ops",
    "partial",
    "real",
    "stats",
]
