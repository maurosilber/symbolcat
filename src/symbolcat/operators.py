"""
symbolcat.operators
~~~~~~~~~~~~~~~~~~~

Operators produced by Python syntax on expressions (``x + y``, ``x[0]``, ...).

These are only declarations: what each operator *means* is given by the
algebras in ``symbolcat.ops`` and the implementation libraries in
``symbolcat.lib``.

Precedences follow Python (higher binds tighter). Comparisons are non-associative, as
``a < b < c`` means something else in Python.
"""

from .core import Operator

CMP, OR, XOR, AND, SHIFT, ADD, MUL, UNARY, POW, POSTFIX = range(10)


def _binary(name: str, symbol: str, precedence: int, assoc: str = "left") -> Operator:
    return Operator(
        name,
        arity=2,
        fmt=f"{{}} {symbol} {{}}",
        precedence=precedence,
        assoc=assoc,
    )


def _unary(name: str, symbol: str) -> Operator:
    return Operator(name, arity=1, fmt=f"{symbol}{{}}", precedence=UNARY)


eq = _binary("eq", "==", CMP, assoc="none")
ne = _binary("ne", "!=", CMP, assoc="none")
lt = _binary("lt", "<", CMP, assoc="none")
le = _binary("le", "<=", CMP, assoc="none")
gt = _binary("gt", ">", CMP, assoc="none")
ge = _binary("ge", ">=", CMP, assoc="none")

or_ = _binary("or_", "|", OR)
xor = _binary("xor", "^", XOR)
and_ = _binary("and_", "&", AND)
lshift = _binary("lshift", "<<", SHIFT)
rshift = _binary("rshift", ">>", SHIFT)

add = _binary("add", "+", ADD)
sub = _binary("sub", "-", ADD)
mul = _binary("mul", "*", MUL)
matmul = _binary("matmul", "@", MUL)
truediv = _binary("truediv", "/", MUL)
floordiv = _binary("floordiv", "//", MUL)
mod = _binary("mod", "%", MUL)
pow = _binary("pow", "**", POW, assoc="right")

neg = _unary("neg", "-")
pos = _unary("pos", "+")
invert = _unary("invert", "~")

getitem = Operator("getitem", arity=2, fmt="{}[{}]", precedence=POSTFIX)
