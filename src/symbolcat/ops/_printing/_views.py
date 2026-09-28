"""
symbolcat.ops._printing._views
~~~~~~~~~~~~~~~~~~~~

Views of the structure of an expression: an indented tree and a Graphviz
graph.
"""

from __future__ import annotations

from itertools import count
from typing import Any

from ..._core import Call, Constant, Expr, Literal, Operator, Symbol, cata, children
from ..._core import name_of as _name_of


def label(node: Expr, /) -> str:
    """A short label for a node: a name, a value, or an operator symbol."""
    if isinstance(node, Symbol | Constant):
        return _name_of(node)
    if isinstance(node, Literal):
        return repr(node.value)
    if isinstance(node, Call):
        func = node.func
        if isinstance(func, Operator) and func.fmt:
            return func.fmt.replace("{}", "").strip() or _name_of(func)
        return _name_of(func)
    raise TypeError(f"Not an expression node: {node!r}")


def _tree(layer: Expr, /) -> list[str]:
    if not isinstance(layer, Call):
        return [label(layer)]
    lines = [label(layer)]
    *init, last = layer.args
    for child in init:
        first, *rest = child
        lines += [f"├── {first}", *(f"│   {line}" for line in rest)]
    first, *rest = last
    lines += [f"└── {first}", *(f"    {line}" for line in rest)]
    return lines


def tree_view(expr: Any, /) -> str:
    """The expression as an indented tree.

    >>> from symbolcat import Symbol, real
    >>> x, y = Symbol("x"), Symbol("y")
    >>> print(tree_view(2 * real.sin(x) + y))
    +
    ├── *
    │   ├── 2
    │   └── sin
    │       └── x
    └── y
    """
    return "\n".join(cata(_tree, expr))


def to_graphviz(expr: Any, /) -> str:
    """The expression as a Graphviz (DOT) graph.

    Each object is one node, so shared sub-expressions (as returned by
    ``share`` or ``egraph.optimize``) are drawn once.

    >>> from symbolcat import Symbol, real
    >>> from symbolcat.ops import share
    >>> x = Symbol("x")
    >>> print(to_graphviz(share(real.sin(x) * real.sin(x))))
    digraph {
      ordering=out;
      n0 [label="x"];
      n1 [label="sin"];
      n1 -> n0;
      n2 [label="*"];
      n2 -> n1;
      n2 -> n1;
    }
    """
    ids = count()
    lines = ["digraph {", "  ordering=out;"]

    def algebra(layer: Expr, /) -> str:
        node = f"n{next(ids)}"
        text = label(layer).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'  {node} [label="{text}"];')
        lines.extend(f"  {node} -> {child};" for child in children(layer))
        return node

    cata(algebra, expr)
    lines.append("}")
    return "\n".join(lines)
