"""Nodes, functions and algebras keep their data in a single hidden slot, so
subclasses (in downstream libraries) can define any attribute."""

import pickle

import pytest

from symbolcat import Symbol, linalg, real
from symbolcat.lib import Constant, Function
from symbolcat.lib import operators as op
from symbolcat.ops import evaluate
from symbolcat.ops.algebras import Algebra, Literal, args_of, name_of

OPERATORS = {
    f"__{prefix}{name}__"
    for name in [
        "add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "pow",
        "and", "or", "xor", "lshift", "rshift",
    ]
    for prefix in ("", "r")
} | {f"__{name}__" for name in ["neg", "pos", "invert", "lt", "le", "gt", "ge", "getitem"]}  # fmt: skip
# the machinery of generic classes and of __slots__
TYPING = {
    "__annotations__",
    "__class_getitem__",
    "__module__",
    "__orig_bases__",
    "__parameters__",
    "__slots__",
    "__type_params__",
}
x = Symbol("x")


@pytest.mark.parametrize(
    "obj, public",
    [
        (x, {"kind", "_repr_latex_", "__set_name__"}),
        (x + 1, {"kind", "_repr_latex_"}),
        (Literal(2), {"kind", "_repr_latex_"}),
        (real.constants.pi, {"kind", "_repr_latex_", "__set_name__"}),
        (Function("f"), {"__call__", "__set_name__"}),
        (real.sin, {"__call__", "__set_name__"}),
        (op.add, {"__call__", "__set_name__"}),
        (linalg.array((2,)), {"__call__", "__set_name__"}),
        (Algebra("a"), {"__call__", "register", "copy", "handler_for"}),
    ],
    ids=lambda value: repr(value) if not isinstance(value, set) else "",
)
def test_only_the_api_is_in_the_namespace(obj, public):
    names = set(dir(obj)) - set(dir(object())) - OPERATORS - TYPING
    assert names == public | {"__symbolcat__"}


class Species(Symbol):
    """A downstream subclass, with attributes named like the hidden data."""

    def __init__(self, name: str, *, value: float) -> None:
        super().__init__(name)
        self.name = f"species {name}"
        self.value = value
        self.args = ("anything",)


def test_subclasses_can_define_any_attribute():
    a = Species("A", value=2.0)
    assert (a.name, a.value, a.args) == ("species A", 2.0, ("anything",))
    assert name_of(a) == "A"
    assert str(a + x) == "A + x"
    assert args_of(a + x) == (a, x)
    assert evaluate(2 * a, {a: 3}) == 6


def test_data_is_immutable():
    with pytest.raises(AttributeError, match="immutable"):
        x.__symbolcat__ = ("y", x.kind)
    with pytest.raises(AttributeError):
        x.name = "y"
    with pytest.raises(AttributeError):
        del x.__symbolcat__


def test_constants_and_functions_compare_by_identity():
    assert Constant("c") != Constant("c")
    assert Function("f") != Function("f")
    assert Symbol("s") == Symbol("s")


@pytest.mark.parametrize(
    "obj", [x, x + 1, Literal(2), linalg.matrix([[x, 1], [0, x]])], ids=str
)
def test_pickle(obj):
    loaded = pickle.loads(pickle.dumps(obj))
    assert str(loaded) == str(obj)
    assert loaded.kind == obj.kind


def test_pickle_keeps_the_attributes_of_subclasses():
    loaded = pickle.loads(pickle.dumps(Species("A", value=2.0)))
    assert (name_of(loaded), loaded.name, loaded.value) == ("A", "species A", 2.0)
