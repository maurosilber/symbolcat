"""
symbolcat._core.signatures
~~~~~~~~~~~~~~~~~~~~~~~~~~

Classes of library functions that share a static signature, such as the
real functions applied entry by entry (``real.sin``). They behave as any
``Function``; their signatures, which give the kind of the result from the
kinds of the arguments, are in ``signatures.pyi``. At runtime, the kind is
given by the rules of ``kind_algebra``.
"""

from .expr import Function


class RealFunction(Function):
    """A real function of a real argument, applied entry by entry."""

    __slots__ = ()


class RealFunction2(Function):
    """A real function of two real arguments, applied entry by entry."""

    __slots__ = ()


class IfElse(Function):
    """``ifelse(condition, a, b)``, entry by entry."""

    __slots__ = ()


class ComplexFunction(Function):
    """A complex function of a complex argument, applied entry by entry."""

    __slots__ = ()


class ComplexToReal(Function):
    """A real function of a complex argument, applied entry by entry."""

    __slots__ = ()


class Conjugate(Function):
    """The complex conjugate: real for a real argument."""

    __slots__ = ()


class Polar(Function):
    """A complex number from two real ones (polar form), entry by entry."""

    __slots__ = ()


class Transpose(Function):
    """The transpose of a vector (itself) or of a matrix."""

    __slots__ = ()


class Derivative(Function):
    """A function whose result has the kind of its argument (``D(x)``)."""

    __slots__ = ()


class Delay(Function):
    """A function whose result has the kind of its first argument, and whose
    second one is real (``delay(x, tau)``, ``partial(u, x)``)."""

    __slots__ = ()
