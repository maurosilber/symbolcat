r"""
symbolcat.ops._latex
~~~~~~~~~~~~~~~~~~~~

Compile an expression to LaTeX.

Like printing, every node folds into a ``Printed`` so parentheses are only
added where needed. Constructs that delimit their operands by themselves
(``\frac``, exponents, ``\sqrt``, ``\left|\cdot\right|``) never add them.

>>> from symbolcat import Symbol, real
>>> x, alpha_1 = Symbol("x"), Symbol("alpha_1")
>>> print(to_latex(real.sqrt(x ** 2 + 1) / (alpha_1 * real.sin(x))))
\frac{\sqrt{x^{2} + 1}}{\alpha_{1} \cdot \sin\left(x\right)}
"""

from __future__ import annotations

from typing import Any

from .. import real
from .._core import (
    Algebra,
    Call,
    Constant,
    Function,
    Literal,
    Operator,
    Symbol,
    cata,
    name_of,
)
from .._core.precedence import POSTFIX, POW
from ..lib import operators as op
from ._printing import Printed, Side, literal_precedence, needs_parens

GREEK = frozenset(
    [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "varepsilon",
        "zeta",
        "eta",
        "theta",
        "vartheta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "pi",
        "varpi",
        "rho",
        "varrho",
        "sigma",
        "varsigma",
        "tau",
        "upsilon",
        "phi",
        "varphi",
        "chi",
        "psi",
        "omega",
        "Gamma",
        "Delta",
        "Theta",
        "Lambda",
        "Xi",
        "Pi",
        "Sigma",
        "Upsilon",
        "Phi",
        "Psi",
        "Omega",
    ]
)

INFIX = {
    op.eq: "=",
    op.ne: r"\neq",
    op.lt: "<",
    op.le: r"\leq",
    op.gt: ">",
    op.ge: r"\geq",
    op.or_: r"\lor",
    op.xor: r"\oplus",
    op.and_: r"\land",
    op.lshift: r"\ll",
    op.rshift: r"\gg",
    op.add: "+",
    op.sub: "-",
    op.mul: r"\cdot",
    op.matmul: r"\cdot",
    op.mod: r"\bmod",
}

PREFIX = {
    op.neg: "-",
    op.pos: "+",
    op.invert: r"\lnot ",
}

FUNCTION_NAMES = {
    real.exp: r"\exp",
    real.log: r"\ln",
    real.log2: r"\log_{2}",
    real.log10: r"\log_{10}",
    real.sin: r"\sin",
    real.cos: r"\cos",
    real.tan: r"\tan",
    real.asin: r"\arcsin",
    real.acos: r"\arccos",
    real.atan: r"\arctan",
    real.sinh: r"\sinh",
    real.cosh: r"\cosh",
    real.tanh: r"\tanh",
    real.gamma: r"\Gamma",
}

# Functions rendered as delimiters around their single argument.
ENCLOSING = {
    real.abs: (r"\left|", r"\right|"),
    real.floor: (r"\left\lfloor ", r"\right\rfloor"),
    real.ceil: (r"\left\lceil ", r"\right\rceil"),
    real.sqrt: (r"\sqrt{", "}"),
}

CONSTANTS = {
    real.constants.e: "e",
    real.constants.pi: r"\pi",
    real.constants.tau: r"\tau",
    real.constants.inf: r"\infty",
    real.constants.nan: r"\mathrm{NaN}",
}


def paren(text: str) -> str:
    return rf"\left({text}\right)"


def operand(child: Printed, parent: Operator, side: Side) -> str:
    return paren(child.text) if needs_parens(child, parent, side) else child.text


def latex_name(name: str) -> str:
    r"""Format a symbol name, with greek letters and nested subscripts.

    >>> print(latex_name("alpha_i_j"))
    \alpha_{i_{j}}
    >>> print(latex_name("rate"))
    \mathrm{rate}
    """
    base, _, sub = name.partition("_")
    if base in GREEK:
        base = "\\" + base
    elif len(base) > 1:
        base = rf"\mathrm{{{base}}}"
    if sub:
        return f"{base}_{{{latex_name(sub)}}}"
    return base


def latex_number(value: Any) -> str:
    if isinstance(value, bool):
        return rf"\mathrm{{{value}}}"
    if isinstance(value, float):
        text = repr(value)
        if "e" in text:
            mantissa, exponent = text.split("e")
            exponent = str(int(exponent))
            if mantissa in ("1", "1.0"):
                return f"10^{{{exponent}}}"
            return rf"{mantissa} \times 10^{{{exponent}}}"
        if text in ("inf", "-inf"):
            return text.replace("inf", r"\infty")
        return text
    return str(value)


latex_algebra = Algebra[Printed]("latex")


@latex_algebra.register(Symbol)
def _(node: Symbol) -> Printed:
    return Printed(latex_name(name_of(node)))


@latex_algebra.register(Constant)
def _(node: Constant) -> Printed:
    return Printed(CONSTANTS.get(node) or latex_name(name_of(node)))


@latex_algebra.register(Literal)
def _(node: Literal) -> Printed:
    return Printed(latex_number(node.value), literal_precedence(node.value))


@latex_algebra.register(Function)
def _(node: Call[Printed]) -> Printed:
    func = node.func
    name = FUNCTION_NAMES.get(func, rf"\operatorname{{{name_of(func)}}}")
    return Printed(name + paren(", ".join(arg.text for arg in node.args)))


@latex_algebra.register(*ENCLOSING)
def _(node: Call[Printed]) -> Printed:
    left, right = ENCLOSING[node.func]
    (arg,) = node.args
    return Printed(f"{left}{arg.text}{right}")


@latex_algebra.register(real.factorial)
def _(node: Call[Printed]) -> Printed:
    (arg,) = node.args
    text = arg.text if arg.precedence >= POSTFIX else paren(arg.text)
    return Printed(f"{text}!", POSTFIX)


@latex_algebra.register(Operator)
def _(node: Call[Printed]) -> Printed:
    func = node.func
    assert isinstance(func, Operator)
    if func in PREFIX:
        (arg,) = node.args
        text = PREFIX[func] + operand(arg, func, "right")
    else:
        left, right = node.args
        symbol = INFIX.get(func, func.fmt.format("", "").strip())
        text = f"{operand(left, func, 'left')} {symbol} {operand(right, func, 'right')}"
    return Printed(text, func.precedence)


@latex_algebra.register(op.truediv)
def _(node: Call[Printed]) -> Printed:
    num, den = node.args
    return Printed(rf"\frac{{{num.text}}}{{{den.text}}}")


@latex_algebra.register(op.floordiv)
def _(node: Call[Printed]) -> Printed:
    num, den = node.args
    return Printed(rf"\left\lfloor \frac{{{num.text}}}{{{den.text}}} \right\rfloor")


@latex_algebra.register(op.pow)
def _(node: Call[Printed]) -> Printed:
    base, exponent = node.args
    return Printed(f"{operand(base, op.pow, 'left')}^{{{exponent.text}}}", POW)


@latex_algebra.register(op.getitem)
def _(node: Call[Printed]) -> Printed:
    obj, key = node.args
    return Printed(f"{operand(obj, op.getitem, 'left')}_{{{key.text}}}", POSTFIX)


def to_latex(expr: Any) -> str:
    r"""LaTeX source for an expression.

    >>> from symbolcat import Symbol
    >>> x, y = Symbol("x"), Symbol("y")
    >>> to_latex((x + y) ** 2 / 3)
    '\\frac{\\left(x + y\\right)^{2}}{3}'
    """
    return cata(latex_algebra, expr).text
