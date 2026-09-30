"""
symbolcat.stats
~~~~~~~~~~~~~~~

Probability distributions whose densities, means and variances are
expressions.

A distribution holds its parameters, which are expressions, and gives its
density (``pdf``, ``logpdf``), cumulative distribution function (``cdf``),
``mean``, ``variance`` and ``central_moment``, written with the functions in
``real``. So every operation works on them: printing, evaluation,
derivatives, code generation, ...

>>> from symbolcat import Symbol, ops, stats
>>> x, mu, sigma = Symbol("x"), Symbol("mu"), Symbol("sigma")
>>> noise = stats.Normal(mu, sigma)
>>> print(noise.logpdf(x))
-((x - mu) / sigma) ** 2 / 2 - log(sigma) - log(2 * pi) / 2
>>> score = ops.calculus.diff(noise.logpdf(x), mu)
>>> print(ops.polynomials.cancel(score))
(x - mu) / sigma ** 2

Outside the ``support``, the density is 0 (and its logarithm ``-inf``):

>>> rate = Symbol("rate")
>>> print(stats.Exponential(rate).pdf(x))
ifelse(x < 0, 0, rate * exp(-rate * x))
>>> ops.evaluate(stats.Exponential(1).logpdf(x), {x: -1})
-inf

``assume_support=True`` gives the formula alone, which is smoother to
differentiate. A log-likelihood is a sum of them, one per observation:

>>> y1, y2 = Symbol("y1"), Symbol("y2")
>>> model = stats.Exponential(rate)
>>> print(model.logpdf(y1, assume_support=True) + model.logpdf(y2, assume_support=True))
log(rate) - rate * y1 + (log(rate) - rate * y2)

For discrete distributions, ``pdf`` is the probability mass function.

``MultivariateNormal`` has a mean vector and a covariance matrix
(``linalg``). ``propagate`` gives the mean and variance of an expression of
random variables, to first or second order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from . import real
from ._core import Call, Expr, Literal, Symbol, args_of, as_expr, func_of
from .lib import operators as op
from .linalg import Array, det, inv, matrix, to_list, vector
from .ops._algebraic._polynomial import cancel
from .ops._calculus._diff import diff
from .ops._transform._simplify import is_number, simplify_exact
from .ops._transform._substitute import substitute

__all__ = [
    "Distribution",
    "Exponential",
    "Gamma",
    "LogNormal",
    "MultivariateNormal",
    "Normal",
    "Poisson",
    "Uniform",
    "propagate",
]

INF = real.constants.inf
PI = real.constants.pi
ZERO = Literal(0)


@dataclass(frozen=True, init=False)
class Distribution:
    """A probability distribution, with expressions as parameters.

    Subclasses give the formulas on the support: ``_logpdf`` or ``_pdf``
    (each defaults to the other), and ``_cdf`` when it has a closed form;
    and ``mean``, ``variance``, ``support`` and ``_central_moment``.
    """

    #: Whether the lower bound is excluded from the support.
    _open_below: ClassVar[bool] = False

    def _set(self, **params: Any) -> None:
        for name, value in params.items():
            object.__setattr__(self, name, as_expr(value))

    @property
    def params(self) -> tuple[Expr, ...]:
        return tuple(getattr(self, f.name) for f in fields(self))

    def pdf(self, x: Any, /, *, assume_support: bool = False) -> Expr:
        """The probability density (or mass) at ``x``: 0 outside the
        support, unless ``assume_support``."""
        return self._outside(self._pdf(x), x, ZERO, assume_support)

    def logpdf(self, x: Any, /, *, assume_support: bool = False) -> Expr:
        """The logarithm of the probability density (or mass) at ``x``:
        ``-inf`` outside the support, unless ``assume_support``."""
        return self._outside(self._logpdf(x), x, -INF, assume_support)

    def cdf(self, x: Any, /, *, assume_support: bool = False) -> Expr:
        """The probability of a value less than or equal to ``x``: 0 below
        the support and 1 above it, unless ``assume_support``."""
        value = self._cdf(x)
        if assume_support:
            return value
        low, high = self.support
        if high != as_expr(INF):
            value = real.ifelse(x > high, 1, value)
        if low != as_expr(-INF):
            value = real.ifelse(self._below(x), 0, value)
        return value

    def _outside(self, value: Expr, x: Any, fill: Any, assume: bool, /) -> Expr:
        if assume:
            return value
        low, high = self.support
        conditions = []
        if low != as_expr(-INF):
            conditions.append(self._below(x))
        if high != as_expr(INF):
            conditions.append(x > high)
        if not conditions:
            return value
        condition = conditions[0] if len(conditions) == 1 else op.or_(*conditions)
        return real.ifelse(condition, fill, value)

    def _below(self, x: Any, /) -> Expr:
        low, _ = self.support
        return x <= low if self._open_below else x < low

    def _pdf(self, x: Any, /) -> Expr:
        return real.exp(self._logpdf(x))

    def _logpdf(self, x: Any, /) -> Expr:
        return real.log(self._pdf(x))

    def _cdf(self, x: Any, /) -> Expr:
        raise NotImplementedError(f"{type(self).__name__}: no closed-form cdf")

    @property
    def mean(self) -> Expr:
        raise NotImplementedError

    @property
    def variance(self) -> Expr:
        raise NotImplementedError

    def central_moment(self, n: int, /) -> Expr:
        """``E[(X - mean) ** n]``, for ``n`` up to 4."""
        if n == 1:
            return ZERO
        if n == 2:
            return self.variance
        return self._central_moment(n)

    def _central_moment(self, n: int, /) -> Expr:
        raise NotImplementedError(f"{type(self).__name__}: central moment {n}")

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
        return simplify_exact((x - self.mu) / self.sigma)

    def _pdf(self, x: Any, /) -> Expr:
        return real.exp(-(self._z(x) ** 2) / 2) / (self.sigma * real.sqrt(2 * PI))

    def _logpdf(self, x: Any, /) -> Expr:
        return -(self._z(x) ** 2) / 2 - real.log(self.sigma) - real.log(2 * PI) / 2

    def _cdf(self, x: Any, /) -> Expr:
        return (1 + real.erf(self._z(x) / real.sqrt(2))) / 2

    @property
    def mean(self) -> Expr:
        return self.mu

    @property
    def variance(self) -> Expr:
        return self.sigma**2

    def _central_moment(self, n: int, /) -> Expr:
        return {3: ZERO, 4: 3 * self.sigma**4}[n]


@dataclass(frozen=True, init=False)
class LogNormal(Distribution):
    """The distribution of ``exp(X)``, for ``X`` normal of mean ``mu`` and
    standard deviation ``sigma``."""

    mu: Expr
    sigma: Expr
    _open_below = True

    def __init__(self, mu: Any = 0, sigma: Any = 1) -> None:
        self._set(mu=mu, sigma=sigma)

    @property
    def _normal(self) -> Normal:
        return Normal(self.mu, self.sigma)

    def _pdf(self, x: Any, /) -> Expr:
        return self._normal._pdf(real.log(x)) / x

    def _logpdf(self, x: Any, /) -> Expr:
        return self._normal._logpdf(real.log(x)) - real.log(x)

    def _cdf(self, x: Any, /) -> Expr:
        return self._normal._cdf(real.log(x))

    def _raw_moment(self, n: int, /) -> Expr:
        return real.exp(n * self.mu + n**2 * self.sigma**2 / 2)

    @property
    def mean(self) -> Expr:
        return self._raw_moment(1)

    @property
    def variance(self) -> Expr:
        return (real.exp(self.sigma**2) - 1) * real.exp(2 * self.mu + self.sigma**2)

    def _central_moment(self, n: int, /) -> Expr:
        m1, m2, m3, m4 = map(self._raw_moment, (1, 2, 3, 4))
        if n == 3:
            return m3 - 3 * m1 * m2 + 2 * m1**3
        return {4: m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4}[n]

    @property
    def support(self) -> tuple[Expr, Expr]:
        return ZERO, as_expr(INF)


@dataclass(frozen=True, init=False)
class Uniform(Distribution):
    """The uniform distribution between ``low`` and ``high``."""

    low: Expr
    high: Expr

    def __init__(self, low: Any = 0, high: Any = 1) -> None:
        self._set(low=low, high=high)

    def _pdf(self, x: Any, /) -> Expr:
        return 1 / (self.high - self.low)

    def _logpdf(self, x: Any, /) -> Expr:
        return -real.log(self.high - self.low)

    def _cdf(self, x: Any, /) -> Expr:
        return (x - self.low) / (self.high - self.low)

    @property
    def mean(self) -> Expr:
        return (self.low + self.high) / 2

    @property
    def variance(self) -> Expr:
        return (self.high - self.low) ** 2 / 12

    def _central_moment(self, n: int, /) -> Expr:
        return {3: ZERO, 4: (self.high - self.low) ** 4 / 80}[n]

    @property
    def support(self) -> tuple[Expr, Expr]:
        return self.low, self.high


@dataclass(frozen=True, init=False)
class Exponential(Distribution):
    """The exponential distribution, of ``rate`` (the inverse of the mean)."""

    rate: Expr

    def __init__(self, rate: Any = 1) -> None:
        self._set(rate=rate)

    def _pdf(self, x: Any, /) -> Expr:
        return self.rate * real.exp(-self.rate * x)

    def _logpdf(self, x: Any, /) -> Expr:
        return real.log(self.rate) - self.rate * x

    def _cdf(self, x: Any, /) -> Expr:
        return 1 - real.exp(-self.rate * x)

    @property
    def mean(self) -> Expr:
        return 1 / self.rate

    @property
    def variance(self) -> Expr:
        return 1 / self.rate**2

    def _central_moment(self, n: int, /) -> Expr:
        return {3: 2 / self.rate**3, 4: 9 / self.rate**4}[n]

    @property
    def support(self) -> tuple[Expr, Expr]:
        return ZERO, as_expr(INF)


@dataclass(frozen=True, init=False)
class Gamma(Distribution):
    """The gamma distribution, of ``shape`` and ``rate``: the waiting time
    for ``shape`` events of a Poisson process of ``rate``."""

    shape: Expr
    rate: Expr

    def __init__(self, shape: Any, rate: Any = 1) -> None:
        self._set(shape=shape, rate=rate)

    def _pdf(self, x: Any, /) -> Expr:
        k, rate = self.shape, self.rate
        return rate**k * x ** (k - 1) * real.exp(-rate * x) / real.gamma(k)

    def _logpdf(self, x: Any, /) -> Expr:
        k, rate = self.shape, self.rate
        return (
            k * real.log(rate)
            + (k - 1) * real.log(x)
            - rate * x
            - real.log(real.gamma(k))
        )

    def _cdf(self, x: Any, /) -> Expr:
        return real.gammainc(self.shape, self.rate * x)

    @property
    def mean(self) -> Expr:
        return self.shape / self.rate

    @property
    def variance(self) -> Expr:
        return self.shape / self.rate**2

    def _central_moment(self, n: int, /) -> Expr:
        k, rate = self.shape, self.rate
        return {3: 2 * k / rate**3, 4: 3 * k * (k + 2) / rate**4}[n]

    @property
    def support(self) -> tuple[Expr, Expr]:
        return ZERO, as_expr(INF)


@dataclass(frozen=True, init=False)
class Poisson(Distribution):
    """The Poisson distribution, of mean ``rate``: the number of events in a
    unit of time. Its values are the integers from 0."""

    rate: Expr

    def __init__(self, rate: Any) -> None:
        self._set(rate=rate)

    def _pdf(self, k: Any, /) -> Expr:
        return self.rate**k * real.exp(-self.rate) / real.factorial(k)

    def _logpdf(self, k: Any, /) -> Expr:
        return k * real.log(self.rate) - self.rate - real.log(real.factorial(k))

    def _cdf(self, k: Any, /) -> Expr:
        """The upper regularized incomplete gamma function."""
        return 1 - real.gammainc(real.floor(k) + 1, self.rate)

    @property
    def mean(self) -> Expr:
        return self.rate

    @property
    def variance(self) -> Expr:
        return self.rate

    def _central_moment(self, n: int, /) -> Expr:
        return {3: self.rate, 4: self.rate * (1 + 3 * self.rate)}[n]

    @property
    def support(self) -> tuple[Expr, Expr]:
        return ZERO, as_expr(INF)


def _array(value: Any, ndim: int, /) -> Call[Expr]:
    if isinstance(value, Call) and isinstance(func_of(value), Array):
        if len(value.kind.shape) != ndim:
            raise ValueError(f"expected an array with {ndim} dimensions: {value}")
        return value
    return vector(value) if ndim == 1 else matrix(value)


@dataclass(frozen=True, init=False)
class MultivariateNormal(Distribution):
    """The normal distribution of a vector, of mean ``mu`` and covariance
    matrix ``cov``, given as sequences or as ``linalg`` arrays.

    Independent entries have the product of their densities:

    >>> from symbolcat import Symbol, ops
    >>> x, y = Symbol("x"), Symbol("y")
    >>> dist = MultivariateNormal([0, 1], [[4, 0], [0, 9]])
    >>> joint = ops.evaluate(dist.pdf([x, y]), {x: 0.5, y: 2.0})
    >>> product = Normal(0, 2).pdf(x) * Normal(1, 3).pdf(y)
    >>> abs(joint - ops.evaluate(product, {x: 0.5, y: 2.0})) < 1e-15
    True
    """

    mu: Call[Expr]
    cov: Call[Expr]

    def __init__(self, mu: Any, cov: Any) -> None:
        mu, cov = _array(mu, 1), _array(cov, 2)
        assert isinstance(func_of(mu), Array) and isinstance(func_of(cov), Array)
        (n,) = mu.kind.shape
        if cov.kind.shape != (n, n):
            raise ValueError(f"the covariance of {n} variables must be {n} by {n}")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "cov", cov)

    @property
    def size(self) -> int:
        return len(args_of(self.mu))

    def _logpdf(self, x: Any, /) -> Expr:
        d = _array(x, 1) - self.mu
        return (
            -(d @ inv(self.cov) @ d) / 2
            - real.log(det(self.cov)) / 2
            - self.size * real.log(2 * PI) / 2
        )

    @property
    def mean(self) -> Expr:
        return self.mu

    @property
    def variance(self) -> Expr:
        """The variances of the entries: the diagonal of ``cov``."""
        rows = to_list(self.cov)
        return vector(rows[i][i] for i in range(self.size))

    def central_moment(self, n: int, /) -> Expr:
        raise NotImplementedError("MultivariateNormal: use cov")


def propagate(
    expr: Any,
    inputs: Mapping[Any, Distribution],
    /,
    *,
    order: int = 1,
) -> tuple[Expr, Expr]:
    """The mean and variance of ``expr``, where the symbols in ``inputs``
    are random variables with those distributions.

    Each key of ``inputs`` is a symbol, or a tuple of symbols for a
    ``MultivariateNormal`` (correlated variables). Different keys are
    independent.

    ``expr`` is expanded to ``order`` (1 or 2) around the means, and the
    mean and variance of the expansion are returned. To first order (the
    delta method, or linear error propagation), the mean is ``expr`` at the
    means, and the variance ``g @ cov @ g``, for the gradient ``g`` at the
    means. To second order, the Hessian adds ``tr(H @ cov) / 2`` to the
    mean, and terms with the third and fourth central moments to the
    variance. Both are exact for polynomials of that degree.

    >>> from symbolcat import Symbol
    >>> x, y, s = Symbol("x"), Symbol("y"), Symbol("s")
    >>> inputs = {x: Normal(2, s), y: Normal(3, 1)}
    >>> mean, variance = propagate(x * y, inputs)
    >>> print(mean, "|", variance)
    6 | 9 * s ** 2 + 4
    >>> mean, variance = propagate(x * y, inputs, order=2)
    >>> print(mean, "|", variance)
    6 | 10 * s ** 2 + 4
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, not {order}")
    variables: list[Symbol] = []
    means: dict[Symbol, Expr] = {}
    cov: dict[tuple[int, int], Expr] = {}
    third: dict[int, Expr] = {}
    fourth: dict[int, Expr] = {}
    for key, dist in inputs.items():
        start = len(variables)
        if isinstance(key, Symbol):
            variables.append(key)
            means[key] = dist.mean
            cov[start, start] = dist.variance
            if order == 2 and not isinstance(dist, Normal):
                third[start] = dist.central_moment(3)
                fourth[start] = dist.central_moment(4)
            continue
        if not isinstance(dist, MultivariateNormal) or dist.size != len(key):
            raise TypeError(
                f"{key}: a tuple of symbols needs a MultivariateNormal of its size"
            )
        rows = to_list(dist.cov)
        for a, symbol in enumerate(key):
            variables.append(symbol)
            means[symbol] = args_of(dist.mu)[a]
            for b in range(len(key)):
                cov[start + a, start + b] = rows[a][b]

    def at_means(value: Expr, /) -> Expr:
        return simplify_exact(substitute(value, means))

    def c(i: int, j: int, /) -> Expr:
        return cov.get((i, j), ZERO)

    n = range(len(variables))
    first = [diff(expr, v) for v in variables]
    g = [at_means(d) for d in first]
    mean = at_means(expr)
    terms: list[Expr] = [g[i] * g[j] * c(i, j) for i in n for j in n if (i, j) in cov]

    if order == 2:
        hessian = {
            (i, j): h
            for i in n
            for j in n
            if not is_number(h := at_means(diff(first[i], variables[j])), 0)
        }
        mean = mean + sum(
            (h * c(j, k) / 2 for (j, k), h in hessian.items() if (j, k) in cov), ZERO
        )

        def isserlis(j: int, k: int, l: int, m: int, /) -> Expr:
            """``E[d_j d_k d_l d_m]``, for independent groups of jointly
            normal variables, or a single variable with its moments."""
            if j == k == l == m and j in fourth:
                return fourth[j]
            return c(j, k) * c(l, m) + c(j, l) * c(k, m) + c(j, m) * c(k, l)

        for (j, k), h in hessian.items():
            if j == k and j in third:
                terms.append(g[j] * h * third[j])
            for (l, m), h2 in hessian.items():
                deviation = isserlis(j, k, l, m) - c(j, k) * c(l, m)
                terms.append(h * h2 * simplify_exact(deviation) / 4)

    terms = [t for t in map(simplify_exact, terms) if not is_number(t, 0)]
    variance = sum(terms[1:], terms[0]) if terms else ZERO
    return cancel(mean), cancel(variance)
