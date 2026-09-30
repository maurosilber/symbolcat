"""
symbolcat.ops._transform._rewrite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Term rewriting with user rules.

A rule is a pattern and a replacement. Patterns are expressions with
wildcards, ``Wild("a")``, that match any sub-expression (optionally, only
those satisfying a predicate); a wildcard that appears twice must match
equal sub-expressions. Matching is syntactic: ``a + b`` does not match
``b + a`` (for rewriting modulo commutativity, see ``ops.egraph``).

``rewrite`` applies the first matching rule at each node, bottom-up
(``"postwalk"``) or top-down (``"prewalk"``), repeating the traversal until
nothing changes (``fixpoint``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..._core import Call, Expr, Symbol, ana, as_expr, cata
from ._substitute import substitute

type Bindings = dict[str, Expr]


@dataclass(frozen=True, repr=False)
class Wild(Symbol):
    """A pattern variable. ``predicate``, if given, restricts what it
    matches.

    >>> from symbolcat.ops.algebras import Literal
    >>> n = Wild("n", predicate=lambda e: isinstance(e, Literal))
    >>> match(n, Literal(2)), match(n, Symbol("x"))
    ({'n': Literal(2)}, None)
    """

    predicate: Callable[[Expr], bool] | None = field(default=None, compare=False)


def match(
    pattern: Any, expr: Any, /, bindings: Bindings | None = None
) -> Bindings | None:
    """The bindings of the wildcards of ``pattern`` that make it equal to
    ``expr``, or ``None``.

    >>> from symbolcat import real
    >>> a, x = Wild("a"), Symbol("x")
    >>> print(match(real.sin(a) ** 2, real.sin(x + 1) ** 2)["a"])
    x + 1
    """
    pattern, expr = as_expr(pattern), as_expr(expr)
    bindings = {} if bindings is None else bindings
    if isinstance(pattern, Wild):
        if pattern.predicate is not None and not pattern.predicate(expr):
            return None
        if pattern.name in bindings:
            return bindings if bindings[pattern.name] == expr else None
        return {**bindings, pattern.name: expr}
    if isinstance(pattern, Call):
        if not isinstance(expr, Call) or expr.func is not pattern.func:
            return None
        if len(expr.args) != len(pattern.args):
            return None
        for p, e in zip(pattern.args, expr.args, strict=True):
            result = match(p, e, bindings)
            if result is None:
                return None
            bindings = result
        return bindings
    return bindings if pattern == expr else None


@dataclass(frozen=True)
class Rule:
    """Replace what matches ``pattern`` by ``replacement``: an expression
    with the same wildcards, or a function of the bindings (by name)
    returning an expression, or ``None`` to not apply the rule.

    >>> from symbolcat import real
    >>> a = Wild("a")
    >>> rule = Rule(real.sin(a) ** 2 + real.cos(a) ** 2, 1)
    >>> print(rule(real.sin(Symbol("x")) ** 2 + real.cos(Symbol("x")) ** 2))
    1
    """

    pattern: Any
    replacement: Any

    def __call__(self, expr: Expr, /) -> Expr | None:
        """The rewritten ``expr``, or ``None`` if the rule does not apply."""
        bindings = match(self.pattern, expr)
        if bindings is None:
            return None
        if callable(self.replacement) and not isinstance(self.replacement, Expr):
            result = self.replacement(**bindings)
            return None if result is None else as_expr(result)
        return substitute(
            self.replacement, {Wild(name): value for name, value in bindings.items()}
        )


def _apply(rules: Iterable[Rule], node: Expr, /) -> Expr | None:
    for rule in rules:
        result = rule(node)
        if result is not None and result != node:
            return result
    return None


def _postwalk(expr: Expr, rules: list[Rule], /) -> Expr:
    def algebra(layer: Expr, /) -> Expr:
        node = Call(layer.func, layer.args) if isinstance(layer, Call) else layer
        return _apply(rules, node) or node

    return cata(algebra, expr)


def _prewalk(expr: Expr, rules: list[Rule], /) -> Expr:
    # The dual of ``_postwalk``: rewrite each node before unfolding its
    # arguments, which are then rewritten in turn.
    def coalgebra(node: Expr, /) -> Expr:
        for _ in range(1000):
            result = _apply(rules, node)
            if result is None:
                break
            node = result
        return node

    return ana(coalgebra, expr)


STRATEGIES: Mapping[str, Callable[[Expr, list[Rule]], Expr]] = {
    "postwalk": _postwalk,
    "prewalk": _prewalk,
}


def rewrite(
    expr: Any,
    rules: Iterable[Rule],
    /,
    *,
    strategy: str = "postwalk",
    fixpoint: bool = True,
    max_iterations: int = 100,
) -> Expr:
    """Apply ``rules`` to every node of ``expr``.

    At each node, the first rule that changes it is applied. With
    ``fixpoint``, the traversal repeats until nothing changes (at most
    ``max_iterations`` times).

    >>> from symbolcat import real
    >>> a, b, c = Wild("a"), Wild("b"), Wild("c")
    >>> expand = Rule(a * (b + c), a * b + a * c)
    >>> x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    >>> print(rewrite(2 * x * (y + z * (x + 1)), [expand]))
    2 * x * y + (2 * x * (z * x) + 2 * x * (z * 1))
    """
    rules = list(rules)
    walk = STRATEGIES[strategy]
    expr = as_expr(expr)
    for _ in range(max_iterations if fixpoint else 1):
        result = walk(expr, rules)
        if result == expr:
            break
        expr = result
    return expr
