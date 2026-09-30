"""
symbolcat._core.overloads
~~~~~~~~~~~~~~~~~~~~~~~~~

The static signatures of the operator overloads of expressions
(``x + y``, ``x[0]``, ...), in ``overloads.pyi``: they give the kind of the
result from the kinds of the operands. At runtime, ``Expr`` defines the
overloads itself, and ``Overloads`` is an empty base class.
"""


class Overloads[K]:
    """Base class of ``Expr``, which declares its operator overloads for
    static type checkers."""

    __slots__ = ()
