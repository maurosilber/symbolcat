# Static signatures of the classes of library functions (see signatures.py).
#
# The overloads are tried in order: the scalar ones first, so a signature
# accepting a scalar or an array only matches when an array is involved.
# The shapes of arrays and the kinds of their entries are only checked at
# runtime.
#
# Each class subclasses ``Function[..., Kind]``: a function of any arguments,
# which its overloads narrow.

from numbers import Complex as _ComplexNumber
from numbers import Real as _RealNumber
from typing import overload

from .expr import Call, Expr, Function
from .kinds import Complex, Kind, Matrix, Real, Vector

type R = Expr[Real] | float | _RealNumber
type C = Expr[Complex] | complex | _ComplexNumber
type V = Expr[Vector]
type M = Expr[Matrix]

class RealFunction(Function[..., Kind]):
    @overload
    def __call__(self, x: R, /) -> Call[Expr, Real]: ...
    @overload
    def __call__(self, x: V, /) -> Call[Expr, Vector]: ...
    @overload
    def __call__(self, x: M, /) -> Call[Expr, Matrix]: ...

class RealFunction2(Function[..., Kind]):
    @overload
    def __call__(self, x: R, y: R, /) -> Call[Expr, Real]: ...
    @overload
    def __call__(self, x: V | R, y: V | R, /) -> Call[Expr, Vector]: ...
    @overload
    def __call__(self, x: M | R, y: M | R, /) -> Call[Expr, Matrix]: ...

class IfElse(Function[..., Kind]):
    @overload
    def __call__(self, condition: R, a: R, b: R, /) -> Call[Expr, Real]: ...
    @overload
    def __call__(self, condition: R, a: C, b: C, /) -> Call[Expr, Complex]: ...
    @overload
    def __call__(
        self, condition: V | R, a: V | C, b: V | C, /
    ) -> Call[Expr, Vector]: ...
    @overload
    def __call__(
        self, condition: M | R, a: M | C, b: M | C, /
    ) -> Call[Expr, Matrix]: ...

class ComplexFunction(Function[..., Kind]):
    @overload
    def __call__(self, z: C, /) -> Call[Expr, Complex]: ...
    @overload
    def __call__(self, z: V, /) -> Call[Expr, Vector]: ...
    @overload
    def __call__(self, z: M, /) -> Call[Expr, Matrix]: ...

class ComplexToReal(Function[..., Kind]):
    @overload
    def __call__(self, z: C, /) -> Call[Expr, Real]: ...
    @overload
    def __call__(self, z: V, /) -> Call[Expr, Vector]: ...
    @overload
    def __call__(self, z: M, /) -> Call[Expr, Matrix]: ...

class Conjugate(Function[..., Kind]):
    @overload
    def __call__(self, z: R, /) -> Call[Expr, Real]: ...
    @overload
    def __call__(self, z: C, /) -> Call[Expr, Complex]: ...
    @overload
    def __call__(self, z: V, /) -> Call[Expr, Vector]: ...
    @overload
    def __call__(self, z: M, /) -> Call[Expr, Matrix]: ...

class Polar(Function[..., Kind]):
    @overload
    def __call__(self, r: R, phi: R, /) -> Call[Expr, Complex]: ...
    @overload
    def __call__(self, r: V | R, phi: V | R, /) -> Call[Expr, Vector]: ...
    @overload
    def __call__(self, r: M | R, phi: M | R, /) -> Call[Expr, Matrix]: ...

class Transpose(Function[..., Kind]):
    @overload
    def __call__(self, a: V, /) -> Call[Expr, Vector]: ...
    @overload
    def __call__(self, a: M, /) -> Call[Expr, Matrix]: ...

class Derivative(Function[..., Kind]):
    def __call__[K: Kind](self, x: Expr[K], /) -> Call[Expr, K]: ...

class Delay(Function[..., Kind]):
    def __call__[K: Kind](self, x: Expr[K], y: R, /) -> Call[Expr, K]: ...
