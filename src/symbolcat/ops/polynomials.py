"""
symbolcat.ops.polynomials
~~~~~~~~~~~~~~~~~~~~~~~~~

Polynomials with rational coefficients: coefficients, division, Gröbner
bases, GCDs, factorization, roots, and the cancellation of rational
functions.
"""

from ._algebraic._polynomial import (
    cancel,
    groebner,
    polynomial_divide,
    polynomial_gcd,
    polynomial_lcm,
    rational_roots,
    square_free_factorization,
)
from ._analysis._polynomial import coefficients

__all__ = [
    "cancel",
    "coefficients",
    "groebner",
    "polynomial_divide",
    "polynomial_gcd",
    "polynomial_lcm",
    "rational_roots",
    "square_free_factorization",
]
