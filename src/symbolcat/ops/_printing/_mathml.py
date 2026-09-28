"""
symbolcat.ops._printing._mathml
~~~~~~~~~~~~~~~~~~~~~

Compile an expression to presentation MathML.

Like the other printers, every node folds into a ``Printed`` (markup plus
precedence), so parentheses are only added where needed. Each piece of
markup is a single element, as ``<msup>`` and ``<mfrac>`` require.
"""

from __future__ import annotations

from typing import Any

from ... import linalg, real
from ..._core import (
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
from ..._core.precedence import POSTFIX, POW
from ...lib import operators as op
from ...linalg import Array, nest
from ._base import Printed, Side, literal_precedence, needs_parens, result_ndim

GREEK = {
    name: chr(code)
    for name, code in [
        ("alpha", 0x3B1), ("beta", 0x3B2), ("gamma", 0x3B3), ("delta", 0x3B4),
        ("epsilon", 0x3B5), ("zeta", 0x3B6), ("eta", 0x3B7), ("theta", 0x3B8),
        ("iota", 0x3B9), ("kappa", 0x3BA), ("lambda", 0x3BB), ("mu", 0x3BC),
        ("nu", 0x3BD), ("xi", 0x3BE), ("pi", 0x3C0), ("rho", 0x3C1),
        ("sigma", 0x3C3), ("tau", 0x3C4), ("upsilon", 0x3C5), ("phi", 0x3C6),
        ("chi", 0x3C7), ("psi", 0x3C8), ("omega", 0x3C9), ("Gamma", 0x393),
        ("Delta", 0x394), ("Theta", 0x398), ("Lambda", 0x39B), ("Xi", 0x39E),
        ("Pi", 0x3A0), ("Sigma", 0x3A3), ("Phi", 0x3A6), ("Psi", 0x3A8),
        ("Omega", 0x3A9),
    ]
}  # fmt: skip

INFIX = {
    op.eq: "=",
    op.ne: "≠",
    op.lt: "&lt;",
    op.le: "≤",
    op.gt: "&gt;",
    op.ge: "≥",
    op.or_: "∨",
    op.xor: "⊕",
    op.and_: "∧",
    op.lshift: "≪",
    op.rshift: "≫",
    op.add: "+",
    op.sub: "−",
    op.mul: "⋅",
    op.matmul: "⋅",
    op.mod: "mod",
}

PREFIX = {op.neg: "−", op.pos: "+", op.invert: "¬"}

FUNCTION_NAMES = {
    real.log: "ln",
    real.asin: "arcsin",
    real.acos: "arccos",
    real.atan: "arctan",
    real.gamma: "Γ",
    linalg.trace: "tr",
}

ENCLOSING = {
    real.abs: ("|", "|"),
    real.floor: ("⌊", "⌋"),
    real.ceil: ("⌈", "⌉"),
    linalg.norm: ("‖", "‖"),
}

# Functions rendered as a superscript on their single argument.
SUPERSCRIPT = {
    linalg.transpose: '<mi mathvariant="sans-serif">T</mi>',
    linalg.inv: "<mrow><mo>−</mo><mn>1</mn></mrow>",
}

CONSTANTS = {
    real.constants.e: "<mi>e</mi>",
    real.constants.pi: "<mi>π</mi>",
    real.constants.tau: "<mi>τ</mi>",
    real.constants.inf: "<mi>∞</mi>",
    real.constants.nan: "<mi>NaN</mi>",
}


def mo(text: str, /) -> str:
    return f"<mo>{text}</mo>"


def mrow(*parts: str) -> str:
    return f"<mrow>{''.join(parts)}</mrow>"


def paren(markup: str, /, left: str = "(", right: str = ")") -> str:
    return mrow(mo(left), markup, mo(right))


def operand(child: Printed, parent: Operator, side: Side, /) -> str:
    return paren(child.text) if needs_parens(child, parent, side) else child.text


def mathml_name(name: str, /) -> str:
    """A symbol name, with greek letters and nested subscripts.

    >>> print(mathml_name("alpha_1"))
    <msub><mi>α</mi><mn>1</mn></msub>
    """
    base, _, sub = name.partition("_")
    base_markup = f"<mi>{GREEK.get(base, base)}</mi>"
    if not sub:
        return base_markup
    sub_markup = f"<mn>{sub}</mn>" if sub.isdigit() else mathml_name(sub)
    return f"<msub>{base_markup}{sub_markup}</msub>"


def mathml_number(value: Any, /) -> str:
    if isinstance(value, bool):
        return f"<mi>{value}</mi>"
    if isinstance(value, int | float) and value < 0:
        return mrow(mo("−"), mathml_number(-value))
    if isinstance(value, float):
        text = repr(value)
        if text == "inf":
            return "<mi>∞</mi>"
        if "e" in text:
            mantissa, exponent = text.split("e")
            power = f"<msup><mn>10</mn><mn>{int(exponent)}</mn></msup>"
            if mantissa in ("1", "1.0"):
                return power
            return mrow(f"<mn>{mantissa}</mn>", mo("×"), power)
        return f"<mn>{text}</mn>"
    return f"<mn>{value}</mn>"


mathml_algebra = Algebra[Printed]("mathml")


@mathml_algebra.register(Symbol)
def _(node: Symbol) -> Printed:
    return Printed(mathml_name(name_of(node)))


@mathml_algebra.register(Constant)
def _(node: Constant) -> Printed:
    return Printed(CONSTANTS.get(node) or mathml_name(name_of(node)))


@mathml_algebra.register(Literal)
def _(node: Literal) -> Printed:
    return Printed(mathml_number(node.value), literal_precedence(node.value))


@mathml_algebra.register(Function)
def _(node: Call[Printed]) -> Printed:
    name = FUNCTION_NAMES.get(node.func, name_of(node.func))
    args = mo(",").join(arg.text for arg in node.args)
    # U+2061 is the invisible "function application" operator
    text = mrow(f"<mi>{name}</mi>", mo("⁡"), paren(mrow(args)))
    return Printed(text, ndim=result_ndim(node.func, node.args))


@mathml_algebra.register(Array)
def _(node: Call[Printed]) -> Printed:
    assert isinstance(node.func, Array)
    rows = nest(node.func.shape, [arg.text for arg in node.args])
    if len(node.func.shape) == 1:  # a column vector
        rows = [[entry] for entry in rows]
    table = "".join(
        "<mtr>" + "".join(f"<mtd>{entry}</mtd>" for entry in row) + "</mtr>"
        for row in rows
    )
    return Printed(paren(f"<mtable>{table}</mtable>"), ndim=len(node.func.shape))


@mathml_algebra.register(*ENCLOSING)
def _(node: Call[Printed]) -> Printed:
    (arg,) = node.args
    text = paren(arg.text, *ENCLOSING[node.func])
    return Printed(text, ndim=result_ndim(node.func, node.args))


@mathml_algebra.register(*SUPERSCRIPT)
def _(node: Call[Printed]) -> Printed:
    (arg,) = node.args
    text = f"<msup>{operand(arg, op.pow, 'left')}{SUPERSCRIPT[node.func]}</msup>"
    return Printed(text, POW, arg.ndim)


@mathml_algebra.register(linalg.cross)
def _(node: Call[Printed]) -> Printed:
    a, b = node.args
    text = mrow(operand(a, op.mul, "left"), mo("×"), operand(b, op.mul, "right"))
    return Printed(text, op.mul.precedence, ndim=1)


@mathml_algebra.register(real.sqrt)
def _(node: Call[Printed]) -> Printed:
    (arg,) = node.args
    return Printed(f"<msqrt>{arg.text}</msqrt>")


@mathml_algebra.register(real.log10, real.log2)
def _(node: Call[Printed]) -> Printed:
    (arg,) = node.args
    base = "10" if node.func is real.log10 else "2"
    log = f"<msub><mi>log</mi><mn>{base}</mn></msub>"
    return Printed(mrow(log, mo("⁡"), paren(arg.text)))


@mathml_algebra.register(real.factorial)
def _(node: Call[Printed]) -> Printed:
    (arg,) = node.args
    text = arg.text if arg.precedence >= POSTFIX else paren(arg.text)
    return Printed(mrow(text, mo("!")), POSTFIX)


@mathml_algebra.register(Operator)
def _(node: Call[Printed]) -> Printed:
    func = node.func
    assert isinstance(func, Operator)
    if func in PREFIX:
        (arg,) = node.args
        return Printed(
            mrow(mo(PREFIX[func]), operand(arg, func, "right")), func.precedence
        )
    left, right = node.args
    symbol = INFIX.get(func, func.fmt.format("", "").strip())
    if func is op.mul and left.ndim and right.ndim:
        symbol = "⊙"  # entrywise product of arrays
    text = mrow(operand(left, func, "left"), mo(symbol), operand(right, func, "right"))
    return Printed(text, func.precedence, result_ndim(func, node.args))


@mathml_algebra.register(op.truediv)
def _(node: Call[Printed]) -> Printed:
    num, den = node.args
    text = f"<mfrac>{num.text}{den.text}</mfrac>"
    return Printed(text, ndim=result_ndim(op.truediv, node.args))


@mathml_algebra.register(op.floordiv)
def _(node: Call[Printed]) -> Printed:
    num, den = node.args
    return Printed(paren(f"<mfrac>{num.text}{den.text}</mfrac>", "⌊", "⌋"))


@mathml_algebra.register(op.pow)
def _(node: Call[Printed]) -> Printed:
    base, exponent = node.args
    text = f"<msup>{operand(base, op.pow, 'left')}{exponent.text}</msup>"
    return Printed(text, POW, result_ndim(op.pow, node.args))


@mathml_algebra.register(op.getitem)
def _(node: Call[Printed]) -> Printed:
    obj, key = node.args
    return Printed(
        f"<msub>{operand(obj, op.getitem, 'left')}{key.text}</msub>", POSTFIX
    )


def to_mathml(expr: Any, /, *, display: bool = False) -> str:
    """Presentation MathML for an expression.

    With ``display``, the formula is set as a block instead of inline.

    >>> from symbolcat import Symbol, real
    >>> x = Symbol("x")
    >>> print(to_mathml(real.sqrt(x) / 2))
    <math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><msqrt><mi>x</mi></msqrt><mn>2</mn></mfrac></math>
    """
    attributes = ' display="block"' if display else ""
    body = cata(mathml_algebra, expr).text
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML"{attributes}>{body}</math>'
