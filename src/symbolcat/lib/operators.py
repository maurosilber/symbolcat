"""
symbolcat.lib.operators
~~~~~~~~~~~~~~~~~~~~~~~

Operators produced by Python syntax on expressions (``x + y``, ``x[0]``, ...).

These are only declarations: what each operator *means* is given by the
algebras in ``symbolcat.ops`` and the implementation libraries in
``symbolcat.lib``.

Precedences follow Python (higher binds tighter). Comparisons are non-associative, as
``a < b < c`` means something else in Python.
"""

from .._core import precedence as _p
from .._core.expr import Operator as _Operator


def _binary(
    name: str, symbol: str, precedence: int, /, *, assoc: str = "left"
) -> _Operator:
    return _Operator(
        name,
        arity=2,
        fmt=f"{{}} {symbol} {{}}",
        precedence=precedence,
        assoc=assoc,
    )


def _unary(name: str, symbol: str, /) -> _Operator:
    return _Operator(name, arity=1, fmt=f"{symbol}{{}}", precedence=_p.UNARY)


eq = _binary("eq", "==", _p.CMP, assoc="none")
ne = _binary("ne", "!=", _p.CMP, assoc="none")
lt = _binary("lt", "<", _p.CMP, assoc="none")
le = _binary("le", "<=", _p.CMP, assoc="none")
gt = _binary("gt", ">", _p.CMP, assoc="none")
ge = _binary("ge", ">=", _p.CMP, assoc="none")

or_ = _binary("or_", "|", _p.OR)
xor = _binary("xor", "^", _p.XOR)
and_ = _binary("and_", "&", _p.AND)
lshift = _binary("lshift", "<<", _p.SHIFT)
rshift = _binary("rshift", ">>", _p.SHIFT)

add = _binary("add", "+", _p.ADD)
sub = _binary("sub", "-", _p.ADD)
mul = _binary("mul", "*", _p.MUL)
matmul = _binary("matmul", "@", _p.MUL)
truediv = _binary("truediv", "/", _p.MUL)
floordiv = _binary("floordiv", "//", _p.MUL)
mod = _binary("mod", "%", _p.MUL)
pow = _binary("pow", "**", _p.POW, assoc="right")

neg = _unary("neg", "-")
pos = _unary("pos", "+")
invert = _unary("invert", "~")

getitem = _Operator("getitem", arity=2, fmt="{}[{}]", precedence=_p.POSTFIX)
