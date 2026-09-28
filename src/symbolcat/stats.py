"""
symbolcat.stats
~~~~~~~~~~~~~~~

Probability distributions whose densities, means and variances are
expressions.

A distribution holds its parameters, which are expressions, and gives its
density (``pdf``, ``logpdf``), cumulative distribution function (``cdf``),
``mean`` and ``variance``, written with the functions in ``real``. So every
operation works on them: printing, evaluation, derivatives, code
generation, ...

>>> from symbolcat import Symbol, ops, stats
>>> x, mu, sigma = Symbol("x"), Symbol("mu"), Symbol("sigma")
>>> noise = stats.Normal(mu, sigma)
>>> print(noise.logpdf(x))
-((x - mu) / sigma) ** 2 / 2 - log(sigma) - log(2 * pi) / 2
>>> score = ops.calculus.diff(noise.logpdf(x), mu)
>>> print(ops.polynomials.cancel(score))
(x - mu) / sigma ** 2

A log-likelihood is a sum of log-densities, one per observation:

>>> k, y1, y2 = Symbol("k"), Symbol("y1"), Symbol("y2")
>>> model = stats.Exponential(k)
>>> print(model.logpdf(y1) + model.logpdf(y2))
log(k) - k * y1 + (log(k) - k * y2)

The formulas hold on the ``support`` of the distribution, where they are
smooth; outside, the density is zero, which they do not check. For discrete
distributions, ``pdf`` is the probability mass function.

``propagate`` gives the mean and variance of an expression of independent
random variables, to first order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from . import real
from ._core import Expr, Literal, Symbol, as_expr
from .ops._calculus._diff import diff
from .ops._transform._simplify import simplify_naive
from .ops._transform._substitute import substitute

__all__ = [
    "Distribution",
    "Exponential",
    "Gamma",
    "LogNormal",
    "Normal",
    "Poisson",
    "Uniform",
    "propagate",
]

INF = real.constants.inf
PI = real.constants.pi


@dataclass(frozen=True, init=False)
class Distribution:
    """A probability distribution, with expressions as parameters.

    Subclasses define ``logpdf`` or ``pdf`` (each defaults to the other),
    ``mean``, ``variance`` and ``support``, and ``cdf`` when it has a closed
    form.
    """

    def _set(self, **params: Any) -> None:
        for name, value in params.items():
            object.__setattr__(self, name, as_expr(value))

    @property
    def params(self) -> tuple[Expr, ...]:
        return tuple(getattr(self, f.name) for f in fields(self))

    def pdf(self, x: Any, /) -> Expr:
        """The probability density (or mass) at ``x``."""
        return real.exp(self.logpdf(x))

    def logpdf(self, x: Any, /) -> Expr:
        """The logarithm of the probability density (or mass) at ``x``."""
        return real.log(self.pdf(x))

    def cdf(self, x: Any, /) -> Expr:
        """The probability of a value less than or equal to ``x``."""
        raise NotImplementedError(f"{type(self).__name__}: no closed-form cdf")

    @property
    def mean(self) -> Expr:
        raise NotImplementedError

    @property
    def variance(self) -> Expr:
        raise NotImplementedError

    @property
    def support(self) -> tuple[Expr, Expr]:
        """The lower and upper bounds of the values."""
        return as_expr(-INF), as_expr(INF)

    def __str__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(str, self.params))})"


@dataclass(frozen=True, init=False)
class Normal(Distribution):
    """The normal distribution, of mean ``mu`` and standard deviation
    ``sigma``.

    >>> from symbolcat import Symbol
    >>> x = Symbol("x")
    >>> print(Normal().cdf(x))
    (1 + erf(x / sqrt(2))) / 2
    """

    mu: Expr
    sigma: Expr

    def __init__(self, mu: Any = 0, sigma: Any = 1) -> None:
        self._set(mu=mu, sigma=sigma)

    def _z(self, x: Any, /) -> Expr:
        return simplify_naive((x - self.mu) / self.sigma)

    def pdf(self, x: Any, /) -> Expr:
        return real.exp(-(self._z(x) ** 2) / 2) / (self.sigma * real.sqrt(2 * PI))

    def logpdf(self, x: Any, /) -> Expr:
        return -(self._z(x) ** 2) / 2 - real.log(self.sigma) - real.log(2 * PI) / 2

    def cdf(self, x: Any, /) -> Expr:
        return (1 + real.erf(self._z(x) / real.sqrt(2))) / 2

    @property
    def mean(self) -> Expr:
        return self.mu

    @property
    def variance(self) -> Expr:
        return self.sigma**2


@dataclass(frozen=True, init=False)
class LogNormal(Distribution):
    """The distribution of ``exp(X)``, for ``X`` normal of mean ``mu`` and
    standard deviation ``sigma``."""

    mu: Expr
    sigma: Expr

    def __init__(self, mu: Any = 0, sigma: Any = 1) -> None:
        self._set(mu=mu, sigma=sigma)

    @property
    def _normal(self) -> Normal:
        return Normal(self.mu, self.sigma)

    def pdf(self, x: Any, /) -> Expr:
        return self._normal.pdf(real.log(x)) / x

    def logpdf(self, x: Any, /) -> Expr:
        return self._normal.logpdf(real.log(x)) - real.log(x)

    def cdf(self, x: Any, /) -> Expr:
        return self._normal.cdf(real.log(x))

    @property
    def mean(self) -> Expr:
        return real.exp(self.mu + self.sigma**2 / 2)

    @property
    def variance(self) -> Expr:
        return (real.exp(self.sigma**2) - 1) * real.exp(2 * self.mu + self.sigma**2)

    @property
    def support(self) -> tuple[Expr, Expr]:
        return Literal(0), as_expr(INF)


@dataclass(frozen=True, init=False)
class Uniform(Distribution):
    """The uniform distribution between ``low`` and ``high``."""

    low: Expr
    high: Expr

    def __init__(self, low: Any = 0, high: Any = 1) -> None:
        self._set(low=low, high=high)

    def pdf(self, x: Any, /) -> Expr:
        return 1 / (self.high - self.low)

    def logpdf(self, x: Any, /) -> Expr:
        return -real.log(self.high - self.low)

    def cdf(self, x: Any, /) -> Expr:
        return (x - self.low) / (self.high - self.low)

    @property
    def mean(self) -> Expr:
        return (self.low + self.high) / 2

    @property
    def variance(self) -> Expr:
        return (self.high - self.low) ** 2 / 12

    @property
    def support(self) -> tuple[Expr, Expr]:
        return self.low, self.high


@dataclass(frozen=True, init=False)
class Exponential(Distribution):
    """The exponential distribution, of ``rate`` (the inverse of the mean)."""

    rate: Expr

    def __init__(self, rate: Any = 1) -> None:
        self._set(rate=rate)

    def pdf(self, x: Any, /) -> Expr:
        return self.rate * real.exp(-self.rate * x)

    def logpdf(self, x: Any, /) -> Expr:
        return real.log(self.rate) - self.rate * x

    def cdf(self, x: Any, /) -> Expr:
        return 1 - real.exp(-self.rate * x)

    @property
    def mean(self) -> Expr:
        return 1 / self.rate

    @property
    def variance(self) -> Expr:
        return 1 / self.rate**2

    @property
    def support(self) -> tuple[Expr, Expr]:
        return Literal(0), as_expr(INF)


@dataclass(frozen=True, init=False)
class Gamma(Distribution):
    """The gamma distribution, of ``shape`` and ``rate``: the waiting time
    for ``shape`` events of a Poisson process of ``rate``.

    Its cdf needs the incomplete gamma function, which ``real`` does not
    have.
    """

    shape: Expr
    rate: Expr

    def __init__(self, shape: Any, rate: Any = 1) -> None:
        self._set(shape=shape, rate=rate)

    def pdf(self, x: Any, /) -> Expr:
        k, rate = self.shape, self.rate
        return rate**k * x ** (k - 1) * real.exp(-rate * x) / real.gamma(k)

    def logpdf(self, x: Any, /) -> Expr:
        k, rate = self.shape, self.rate
        return (
            k * real.log(rate)
            + (k - 1) * real.log(x)
            - rate * x
            - real.log(real.gamma(k))
        )

    @property
    def mean(self) -> Expr:
        return self.shape / self.rate

    @property
    def variance(self) -> Expr:
        return self.shape / self.rate**2

    @property
    def support(self) -> tuple[Expr, Expr]:
        return Literal(0), as_expr(INF)


@dataclass(frozen=True, init=False)
class Poisson(Distribution):
    """The Poisson distribution, of mean ``rate``: the number of events in a
    unit of time. Its values are the integers from 0.

    Its cdf is a sum up to the value, with no closed form.
    """

    rate: Expr

    def __init__(self, rate: Any) -> None:
        self._set(rate=rate)

    def pdf(self, k: Any, /) -> Expr:
        return self.rate**k * real.exp(-self.rate) / real.factorial(k)

    def logpdf(self, k: Any, /) -> Expr:
        return k * real.log(self.rate) - self.rate - real.log(real.factorial(k))

    @property
    def mean(self) -> Expr:
        return self.rate

    @property
    def variance(self) -> Expr:
        return self.rate

    @property
    def support(self) -> tuple[Expr, Expr]:
        return Literal(0), as_expr(INF)


def propagate(expr: Any, inputs: Mapping[Symbol, Distribution], /) -> tuple[Expr, Expr]:
    """The mean and variance of ``expr``, where the symbols in ``inputs``
    are independent random variables with those distributions.

    To first order in the deviations from the means (the delta method, or
    linear error propagation): the mean is ``expr`` at the means, and the
    variance is ``sum_i (d expr / d x_i) ** 2 * variance_i``, with the
    derivatives at the means. Both are exact when ``expr`` is linear.

    >>> from symbolcat import Symbol
    >>> x, y, s = Symbol("x"), Symbol("y"), Symbol("s")
    >>> mean, variance = propagate(x * y, {x: Normal(2, s), y: Normal(3, 1)})
    >>> print(mean)
    6
    >>> print(variance)
    9 * s ** 2 + 4
    """
    means = {symbol: dist.mean for symbol, dist in inputs.items()}
    mean = simplify_naive(substitute(expr, means))
    terms = [
        substitute(diff(expr, symbol), means) ** 2 * dist.variance
        for symbol, dist in inputs.items()
    ]
    variance = sum(terms[1:], terms[0]) if terms else Literal(0)
    return mean, simplify_naive(variance)
