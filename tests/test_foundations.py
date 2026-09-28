import math
import random
import shutil
import subprocess
from fractions import Fraction
from typing import Any

import pytest

from symbolcat import D, Symbol, real
from symbolcat.lib import operators as op
from symbolcat.linalg import vector
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Literal
from symbolcat.ops.calculus import diff, vjp
from symbolcat.ops.equations import (
    generic_rank,
    rational_nullspace,
    rational_rank,
    solve,
)
from symbolcat.ops.inspect import dimensions
from symbolcat.ops.polynomials import (
    cancel,
    groebner,
    polynomial_divide,
    polynomial_gcd,
    polynomial_lcm,
    rational_roots,
    square_free_factorization,
)
from symbolcat.ops.printing import to_c, to_julia, to_latex, to_python
from symbolcat.ops.transform import (
    Rule,
    Wild,
    match,
    rewrite,
    simplify_naive,
    substitute_variables,
)

x, y, z, a, b = (Symbol(n) for n in ["x", "y", "z", "a", "b"])


def value(expr: Any, point: dict[Any, Any]) -> Any:
    return evaluate(substitute_variables(expr, point))


# conditionals


def test_ifelse_evaluate_and_diff():
    f = real.ifelse(x < 0, -x, x**2)
    assert evaluate(f, {x: -2}) == 2
    assert evaluate(f, {x: 3}) == 9
    df = diff(f, x)
    assert evaluate(df, {x: -2}) == -1
    assert evaluate(df, {x: 3}) == 6


def test_ifelse_reverse_mode():
    f = real.ifelse(x < y, x * y, x + y)
    for point in ({x: 1.0, y: 2.0}, {x: 3.0, y: 2.0}):
        forward = [evaluate(diff(f, v), point) for v in (x, y)]
        reverse = [evaluate(d, point) for d in vjp(f, [x, y])]
        assert forward == pytest.approx(reverse)


def test_ifelse_simplify():
    assert simplify_naive(real.ifelse(Literal(True), x, y)) == x
    assert simplify_naive(real.ifelse(x < 0, y, y)) == y


def test_ifelse_numpy():
    np = pytest.importorskip("numpy")
    from symbolcat.lib.numpy import IMPL

    result = evaluate(real.ifelse(x < 0, -x, x), {x: np.array([-1.0, 2.0])}, lib=IMPL)
    assert result.tolist() == [1.0, 2.0]


def test_ifelse_printing():
    f = real.ifelse(x < 0, -x, x)
    assert "cases" in to_latex(f)
    namespace: dict[str, Any] = {}
    exec(to_python(f), namespace)  # noqa: S102
    assert namespace["f"](-3) == 3


def test_ifelse_dimensions():
    units = {x: {"m": 1}, y: {"m": 1}, z: {"s": 1}}
    assert dimensions(real.ifelse(x < y, x, y), units) == {"m": 1}
    with pytest.raises(ValueError, match="incompatible"):
        dimensions(real.ifelse(x < y, x, z), units)


# polynomials


def test_groebner_triangular():
    basis = groebner([x**2 + y**2 - 1, x - y], [x, y])
    # the last element only depends on y
    assert basis[0] == y**2 - Literal(1) / 2


def test_groebner_inconsistent():
    assert groebner([x - 1, x - 2], [x]) == (Literal(1),)


def test_groebner_grevlex():
    basis = groebner([x**2 - y, x * y - 1], [x, y], order="grevlex")
    point = {x: 1, y: 1}
    assert all(value(p, point) == 0 for p in basis)


def test_gcd_lcm():
    f, g = (x - 1) * (x + y) ** 2, (x + y) * (x - 2)
    assert polynomial_gcd(f, g) == x + y
    lcm = polynomial_lcm(f, g)
    for point in ({x: 3, y: 5}, {x: -2, y: 7}):
        assert value(lcm, point) == value(f * (x - 2), point)


def test_gcd_univariate_coprime():
    assert polynomial_gcd(x**2 + 1, x - 1) == Literal(1)


def test_polynomial_divide():
    (q1, q2), r = polynomial_divide(x**2 * y + x * y**2 + y**2, [x * y - 1, y**2 - 1])
    point = {x: 3, y: 5}
    total = value(q1 * (x * y - 1) + q2 * (y**2 - 1) + r, point)
    assert total == value(x**2 * y + x * y**2 + y**2, point)


def test_polynomial_rejects_non_polynomial():
    with pytest.raises(ValueError, match="rational operation"):
        groebner([real.sin(x)], [x])
    with pytest.raises(ValueError, match="divides by a polynomial"):
        groebner([1 / x], [x])


def test_square_free_factorization():
    factors = dict(square_free_factorization((x - 1) ** 3 * (x + 1) ** 2 * (x - 5), x))
    assert factors == {x - 5: 1, x + 1: 2, x - 1: 3}


def test_rational_roots():
    roots = rational_roots(6 * x**3 - 11 * x**2 + 6 * x - 1, x)
    assert [evaluate(r) for r in roots] == pytest.approx([1 / 3, 1 / 2, 1])
    assert rational_roots(x**2 + 1, x) == []
    assert rational_roots(x**2 * (x - 3), x) == [Literal(0), Literal(3)]


def test_cancel():
    assert cancel((x**2 - y**2) / (x - y)) == x + y
    assert cancel(x / x) == Literal(1)
    assert cancel(x - x) == Literal(0)
    expr = 1 / (x + 1) - 1 / (x - 1)
    point = {x: Fraction(5, 3)}
    assert value(cancel(expr), point) == value(expr, point)


def test_cancel_with_atoms():
    s = real.sin(x)
    assert cancel((s**2 - 1) / (s - 1)) == s + 1


# solve


@pytest.mark.parametrize(
    "equation, expected",
    [
        (2 * x - 3, [1.5]),
        (x**2 - 5 * x + 6, [2, 3]),
        (x**3 - 7 * x - 6, [-2, -1, 3]),
        ((x**2 - 4) / (x - 2), [-2]),
        (real.exp(x) - 3, [math.log(3)]),
        (real.sqrt(x + 1) - 2, [3]),
        (op.eq(1 / x, 4), [0.25]),
        (2**x - 8, [3]),
    ],
)
def test_solve_numeric(equation, expected):
    solutions = sorted(evaluate(s) for s in solve(equation, x))
    assert solutions == pytest.approx(expected)


def test_solve_symbolic_quadratic():
    solutions = solve(a * x**2 + b * x - 1, x)
    point = {a: 2.0, b: 3.0}
    for s in solutions:
        root = value(s, point)
        assert 2 * root**2 + 3 * root - 1 == pytest.approx(0)


def test_solve_derivative():
    (solution,) = solve(2 * D(x) - y, D(x))
    assert value(solution, {y: 4}) == 2


def test_solve_unsupported():
    with pytest.raises(NotImplementedError):
        solve(x + real.sin(x), x)
    with pytest.raises(NotImplementedError):
        solve(x**5 - x - 1, x)


# rank


def test_rational_rank_and_nullspace():
    rows = [[1, 2, 3], [2, 4, 6], [1, 0, 1]]
    assert rational_rank(rows) == 2
    (vector_,) = rational_nullspace(rows)
    assert all(sum(r * v for r, v in zip(row, vector_)) == 0 for row in rows)


def test_generic_rank():
    assert generic_rank([[x, x * y], [1, y]]) == 1
    assert generic_rank([[x, 1 / y], [y, x]]) == 2
    with pytest.raises(TypeError, match="rational functions"):
        generic_rank([[real.sin(x)]])


# rewriting


def test_rewrite_postwalk_fixpoint():
    u = Wild("u")
    rules = [Rule(real.sin(u) ** 2 + real.cos(u) ** 2, 1), Rule(u * 1, u)]
    expr = (real.sin(x * 1) ** 2 + real.cos(x) ** 2) * y
    assert rewrite(expr, rules) == Literal(1) * y


def test_rewrite_repeated_wildcard():
    u = Wild("u")
    rule = Rule(u - u, 0)
    assert rewrite(x - x, [rule]) == Literal(0)
    assert rewrite(x - y, [rule]) == x - y


def test_rewrite_predicate_and_function():
    n = Wild("n", predicate=lambda e: isinstance(e, Literal) and type(e.value) is int)
    u = Wild("u")
    # x ** n -> x * x ** (n - 1), for integers
    rule = Rule(u**n, lambda u, n: u * u ** (n.value - 1) if n.value > 1 else None)
    assert rewrite(x**3, [rule]) == x * (x * x**1)
    assert rewrite(x**y, [rule]) == x**y


def test_rewrite_prewalk():
    u, v = Wild("u"), Wild("v")
    swap = Rule(real.sin(u + v), real.sin(u) * real.cos(v) + real.cos(u) * real.sin(v))
    result = rewrite(real.sin(x + y), [swap], strategy="prewalk")
    point = {x: 0.3, y: 0.5}
    assert value(result, point) == pytest.approx(math.sin(0.8))


def test_match_is_syntactic():
    u, v = Wild("u"), Wild("v")
    assert match(u + v, x + y) == {"u": x, "v": y}
    assert match(u * 2, 2 * x) is None


# code generation


def run_c(source: str, call: str, tmp_path) -> float:
    main = f"""#include <math.h>
#include <stdio.h>
{source}
int main(void) {{ printf("%.17g\\n", {call}); return 0; }}
"""
    (tmp_path / "f.c").write_text(main)
    subprocess.run(
        ["cc", "-o", str(tmp_path / "f"), str(tmp_path / "f.c"), "-lm"], check=True
    )
    out = subprocess.run(
        [str(tmp_path / "f")], capture_output=True, text=True, check=True
    )
    return float(out.stdout)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_compiles_and_matches(tmp_path):
    expr = real.sin(x) ** 2 / 2 + real.ifelse(x < y, y, x) + real.constants.pi * x**-1
    result = run_c(to_c(expr, args=[x, y]), "f(0.7, 1.3)", tmp_path)
    assert result == pytest.approx(evaluate(expr, {x: 0.7, y: 1.3}))


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
def test_c_vector_output(tmp_path):
    source = to_c(vector([x * y, x - y]), args=[x, y], name="g")
    call = "(g(2.0, 3.0, out), out[0] * 10 + out[1])"
    result = run_c("double out[2];\n" + source, call, tmp_path)
    assert result == 6 * 10 - 1


def julia_works() -> bool:
    if shutil.which("julia") is None:
        return False
    try:
        out = subprocess.run(
            ["julia", "--startup-file=no", "-e", "print(1)"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.stdout == "1"


@pytest.mark.skipif(not julia_works(), reason="no working Julia")
def test_julia_runs_and_matches():
    expr = real.sin(x) ** 2 / 2 + real.ifelse(x < y, y, x) + real.atan2(y, x) * x**-1
    source = to_julia(expr, args=[x, y]) + "\nprintln(f(0.7, 1.3))"
    out = subprocess.run(
        ["julia", "--startup-file=no", "-e", source],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert float(out.stdout) == pytest.approx(evaluate(expr, {x: 0.7, y: 1.3}))


def test_julia_indices_are_one_based():
    v = Symbol("v")
    assert "v[1]" in to_julia(v[0])
    assert "v[1] ^ 2" in to_julia(v[0] ** 2)


def test_random_polynomial_gcd_roundtrip():
    rng = random.Random(0)
    for _ in range(5):
        common = x + rng.randint(1, 5) * y
        f = common * (x - rng.randint(1, 5))
        g = common * (y + rng.randint(1, 5))
        assert polynomial_gcd(f, g) == polynomial_gcd(common, common)
