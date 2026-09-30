import math
from typing import Any

import pytest
import sympy
import sympy.stats as ss

from symbolcat import Symbol, linalg, real, stats
from symbolcat.backends import numpy as np_lib
from symbolcat.ops import evaluate
from symbolcat.ops.calculus import diff
from symbolcat.ops.printing import to_latex
from symbolcat.ops.transform import substitute

x, a, b = Symbol("x"), Symbol("a"), Symbol("b")

# (distribution, the same in sympy.stats, a point in the support)
CASES = [
    (stats.Normal(0.5, 2.0), ss.Normal("X", 0.5, 2.0), 1.3),
    (stats.LogNormal(0.2, 0.7), ss.LogNormal("X", 0.2, 0.7), 1.8),
    (stats.Uniform(-1.0, 3.0), ss.Uniform("X", -1.0, 3.0), 0.4),
    (stats.Exponential(1.5), ss.Exponential("X", 1.5), 0.8),
    (stats.Gamma(2.5, 4.0), ss.Gamma("X", 2.5, 1 / 4.0), 0.9),
    (stats.Poisson(3.5), ss.Poisson("X", 3.5), 2),
]
IDS = [type(case[0]).__name__ for case in CASES]


@pytest.mark.parametrize("dist, reference, point", CASES, ids=IDS)
def test_against_sympy(dist, reference, point):
    density: Any = ss.density(reference)
    assert evaluate(dist.pdf(x), {x: point}) == pytest.approx(float(density(point)))
    if isinstance(dist, stats.LogNormal):
        return  # sympy has no moments for it; see test_lognormal_moments
    mean: Any = ss.E(reference)
    variance: Any = ss.variance(reference)
    assert evaluate(dist.mean) == pytest.approx(float(mean))
    assert evaluate(dist.variance) == pytest.approx(float(variance))


def test_lognormal_moments():
    mu, sigma = 0.2, 0.7
    dist = stats.LogNormal(mu, sigma)
    mean = math.exp(mu + sigma**2 / 2)
    assert evaluate(dist.mean) == pytest.approx(mean)
    assert evaluate(dist.variance) == pytest.approx(mean**2 * (math.exp(sigma**2) - 1))


@pytest.mark.parametrize("dist, reference, point", CASES, ids=IDS)
def test_logpdf(dist, reference, point):
    assert evaluate(dist.logpdf(x), {x: point}) == pytest.approx(
        math.log(evaluate(dist.pdf(x), {x: point}))
    )


@pytest.mark.parametrize("dist, reference, point", CASES, ids=IDS)
def test_cdf(dist, reference, point):
    cdf: Any = ss.cdf(reference)
    assert evaluate(dist.cdf(x), {x: point}) == pytest.approx(float(cdf(point)))


@pytest.mark.parametrize(
    "dist, reference, point",
    [case for case in CASES if not isinstance(case[0], stats.Poisson)],
    ids=IDS[:-1],
)
def test_pdf_is_the_derivative_of_the_cdf(dist, reference, point):
    assert evaluate(diff(dist.cdf(x), x), {x: point}) == pytest.approx(
        evaluate(dist.pdf(x), {x: point})
    )


def test_poisson_cdf_is_a_sum():
    dist = stats.Poisson(2.5)
    for k in range(6):
        expected = sum(evaluate(dist.pdf(i)) for i in range(k + 1))
        assert evaluate(dist.cdf(x), {x: k}) == pytest.approx(expected)
        assert evaluate(dist.cdf(x), {x: k + 0.5}) == pytest.approx(expected)


@pytest.mark.parametrize("dist, reference, point", CASES, ids=IDS)
@pytest.mark.parametrize("n", [3, 4])
def test_central_moments(dist, reference, point, n):
    if isinstance(dist, stats.LogNormal):
        m, s2 = evaluate(dist.mean), 0.7**2
        raw = [math.exp(k * 0.2 + k**2 * s2 / 2) for k in range(5)]
        expected = sum(math.comb(n, k) * raw[k] * (-m) ** (n - k) for k in range(n + 1))
    else:
        moment: Any = ss.cmoment(reference, n)
        expected = float(moment)
    assert evaluate(dist.central_moment(n)) == pytest.approx(expected)


# (distribution, points below and above the support; None if unbounded)
BOUNDED = [
    (stats.LogNormal(), 0.0, None),
    (stats.Uniform(-1, 3), -1.5, 3.5),
    (stats.Exponential(2), -0.1, None),
    (stats.Gamma(2.5, 4), -0.1, None),
    (stats.Poisson(3), -1, None),
]


@pytest.mark.parametrize("dist, below, above", BOUNDED)
def test_outside_the_support(dist, below, above):
    for point, cdf in [(below, 0), (above, 1)]:
        if point is None:
            continue
        assert evaluate(dist.pdf(x), {x: point}) == 0
        assert evaluate(dist.logpdf(x), {x: point}) == -math.inf
        assert evaluate(dist.cdf(x), {x: point}) == cdf


def test_assume_support():
    dist = stats.Exponential(a)
    assert dist.pdf(x, assume_support=True) == a * real.exp(-a * x)
    assert dist.pdf(x) == real.ifelse(x < 0, 0, a * real.exp(-a * x))
    assert stats.Normal().pdf(x) == stats.Normal().pdf(x, assume_support=True)
    assert str(stats.Uniform(a, b).cdf(x)) == (
        "ifelse(x < a, 0, ifelse(x > b, 1, (x - a) / (b - a)))"
    )


def test_symbolic_parameters():
    dist = stats.Normal(a, b)
    assert dist.params == (a, b)
    assert dist.mean == a
    assert str(dist) == "Normal(a, b)"
    assert dist == stats.Normal(a, b)
    assert substitute(dist.pdf(x), {a: 0.5, b: 2.0}) == stats.Normal(0.5, 2.0).pdf(x)


def test_defaults():
    assert stats.Normal() == stats.Normal(0, 1)
    assert stats.Uniform().support == stats.Uniform(0, 1).support
    assert str(stats.Exponential()) == "Exponential(1)"


def test_support():
    inf = real.constants.inf
    assert stats.Normal().support == (-inf, inf)
    assert stats.Uniform(a, b).support == (a, b)
    assert stats.Exponential(a).support[0] == stats.Gamma(a).support[0]


def test_maximum_likelihood():
    """The score of an exponential sample vanishes at 1 / mean."""
    rate = Symbol("rate")
    data = [0.5, 1.5, 1.0]
    log_likelihood = sum(stats.Exponential(rate).logpdf(y) for y in data)
    score = diff(log_likelihood, rate)
    assert evaluate(score, {rate: len(data) / sum(data)}) == pytest.approx(0)


def test_numpy_backend():
    import numpy as np

    values = np.array([-1.0, 0.0, 2.0])
    result = evaluate(
        stats.Normal(a, b).pdf(x), {x: values, a: 0.0, b: 1.0}, backend=np_lib.IMPL
    )
    expected = np.exp(-(values**2) / 2) / np.sqrt(2 * np.pi)
    assert result == pytest.approx(expected)


def test_latex():
    assert to_latex(stats.Exponential(a).pdf(x, assume_support=True)) == (
        r"a \cdot \exp\left(-a \cdot x\right)"
    )


# propagate


def test_propagate_linear_is_exact():
    mean, variance = stats.propagate(
        2 * x + 3 * a + 1, {x: stats.Normal(1, 2), a: stats.Uniform(0, 6)}
    )
    assert evaluate(mean) == 12
    assert evaluate(variance) == pytest.approx(4 * 4 + 9 * 3)


def test_propagate_first_order():
    mu, sigma = Symbol("mu"), Symbol("sigma")
    mean, variance = stats.propagate(real.exp(x), {x: stats.Normal(mu, sigma)})
    assert mean == real.exp(mu)
    assert evaluate(variance, {mu: 0.3, sigma: 0.1}) == pytest.approx(
        math.exp(0.3) ** 2 * 0.01
    )


def test_propagate_keeps_other_symbols():
    mean, variance = stats.propagate(a * x, {x: stats.Exponential(2)})
    assert evaluate(mean, {a: 3}) == 1.5
    assert evaluate(variance, {a: 3}) == 9 / 4


@pytest.mark.parametrize(
    "expr, inputs, mean, variance",
    [
        # exact to second order for quadratic expressions
        (
            x * a,
            {x: stats.Normal(2, 3), a: stats.Normal(-1, 0.5)},
            -2,
            4 * 0.25 + 1 * 9 + 9 * 0.25,
        ),
        (x**2, {x: stats.Exponential(2)}, 2 / 4, 20 / 16),
        (x**2 + x, {x: stats.Uniform(0, 1)}, 1 / 3 + 1 / 2, None),
        (x**2 * 3, {x: stats.Poisson(4)}, 3 * 20, None),
    ],
)
def test_propagate_second_order(expr, inputs, mean, variance):
    got_mean, got_variance = stats.propagate(expr, inputs, order=2)
    assert evaluate(got_mean) == pytest.approx(mean)
    if variance is None:  # compare with the exact variance, from sympy
        ((_, dist),) = inputs.items()
        reference = {
            stats.Uniform: lambda: ss.Uniform("X", 0, 1),
            stats.Poisson: lambda: ss.Poisson("X", 4),
        }[type(dist)]()
        polynomial = sympy.lambdify(sympy.Symbol("x"), sympy.sympify(str(expr)))(
            reference
        )
        exact: Any = ss.variance(polynomial)
        variance = float(exact)
    assert evaluate(got_variance) == pytest.approx(variance)


def test_propagate_correlated():
    s, c = Symbol("s"), Symbol("c")
    inputs = {(x, a): stats.MultivariateNormal([1, 2], [[s, c], [c, 1]])}
    mean, variance = stats.propagate(x * a, inputs, order=2)
    values = {s: 2.0, c: 0.5}
    # E[x a] = 2 + c, Var = mu_x^2 s_a + mu_a^2 s_x + 2 mu_x mu_a c + s_x s_a + c^2
    assert evaluate(mean, values) == pytest.approx(2.5)
    assert evaluate(variance, values) == pytest.approx(1 + 8 + 2 + 2 + 0.25)
    mean, variance = stats.propagate(x - a, inputs)
    assert evaluate(variance, values) == pytest.approx(2 + 1 - 2 * 0.5)


def test_propagate_mixed():
    """A correlated pair, independent of a third variable."""
    inputs = {
        (x, a): stats.MultivariateNormal([0, 0], [[1, 0.5], [0.5, 1]]),
        b: stats.Exponential(1),
    }
    _, variance = stats.propagate(x * b + a, inputs, order=2)
    # Var(x b) + Var(a) + 2 Cov(x b, a) = E[x^2] E[b^2] + 1 + 2 * 0.5 * E[b]
    assert evaluate(variance) == pytest.approx(2 + 1 + 1)


def test_propagate_errors():
    with pytest.raises(ValueError):
        stats.propagate(x, {x: stats.Normal()}, order=3)
    with pytest.raises(TypeError):
        stats.propagate(x, {(x, a): stats.Normal()})


def test_multivariate_normal():
    import numpy as np

    mu, cov = [1.0, -1.0], [[2.0, 0.3], [0.3, 1.0]]
    dist = stats.MultivariateNormal(mu, cov)
    point = np.array([0.5, 0.2])
    d = point - np.array(mu)
    expected = np.exp(-d @ np.linalg.inv(cov) @ d / 2) / np.sqrt(
        np.linalg.det(2 * np.pi * np.array(cov))
    )
    assert evaluate(dist.pdf([x, a]), {x: 0.5, a: 0.2}) == pytest.approx(expected)
    assert dist.variance == linalg.vector([2.0, 1.0])
    with pytest.raises(ValueError):
        stats.MultivariateNormal([0, 0], [[1, 0, 0]])


def test_propagate_nothing():
    mean, variance = stats.propagate(a, {})
    assert mean == a
    assert evaluate(variance) == 0


def test_sympy_backend():
    from symbolcat.backends import sympy as sympy_lib

    value = evaluate(stats.Normal().pdf(x), backend=sympy_lib.IMPL)
    assert float(value.subs(sympy.Symbol("x"), 0.7)) == pytest.approx(
        math.exp(-(0.7**2) / 2) / math.sqrt(2 * math.pi)
    )
