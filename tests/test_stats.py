import math
from typing import Any

import pytest
import sympy
import sympy.stats as ss

from symbolcat import Symbol, real, stats
from symbolcat.lib import numpy as np_lib
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


@pytest.mark.parametrize(
    "dist, reference, point",
    [case for case in CASES if not isinstance(case[0], stats.Gamma | stats.Poisson)],
    ids=["Normal", "LogNormal", "Uniform", "Exponential"],
)
def test_cdf(dist, reference, point):
    cdf: Any = ss.cdf(reference)
    assert evaluate(dist.cdf(x), {x: point}) == pytest.approx(float(cdf(point)))
    # the density is the derivative of the cdf
    assert evaluate(diff(dist.cdf(x), x), {x: point}) == pytest.approx(
        evaluate(dist.pdf(x), {x: point})
    )


@pytest.mark.parametrize("dist", [stats.Gamma(2), stats.Poisson(1)])
def test_no_closed_form_cdf(dist):
    with pytest.raises(NotImplementedError):
        dist.cdf(x)


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
        stats.Normal(a, b).pdf(x), {x: values, a: 0.0, b: 1.0}, lib=np_lib.IMPL
    )
    expected = np.exp(-(values**2) / 2) / np.sqrt(2 * np.pi)
    assert result == pytest.approx(expected)


def test_latex():
    assert to_latex(stats.Exponential(a).pdf(x)) == (
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


def test_propagate_nothing():
    mean, variance = stats.propagate(a, {})
    assert mean == a
    assert evaluate(variance) == 0


def test_sympy_backend():
    from symbolcat.lib import sympy as sympy_lib

    value = evaluate(stats.Normal().pdf(x), lib=sympy_lib.IMPL)
    assert float(value.subs(sympy.Symbol("x"), 0.7)) == pytest.approx(
        math.exp(-(0.7**2) / 2) / math.sqrt(2 * math.pi)
    )
