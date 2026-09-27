# symbolcat: a minimalistic symbolic python package, built on catamorphisms

symbolcat is a small clone of [symbolite](https://github.com/dyscolab/symbolite).
You create symbols and combine them with ordinary Python operators to build
expressions. You can then print them, compile them to LaTeX, evaluate them with
different backends, substitute values into them, simplify them, or
differentiate them.

```python
>>> from symbolcat import Symbol, real
>>> x = Symbol("x")
>>> y = Symbol("y")
>>> expr = x + 3 * real.sin(y)
>>> print(expr)
x + 3 * sin(y)

```

## Package layout

The top level only has what you need to build expressions and act on them:

- `Symbol`: a symbol; combine symbols with Python operators.
- `real`: mathematical functions (`real.sin`) and constants
  (`real.constants.pi`).
- `ops`: operations on expressions (`ops.to_latex`, `ops.evaluate`, ...).
  - `ops.egraph`: simplification with e-graphs (needs `egglog`).
  - `ops.algebras`: the extension API (algebras, `cata`, node types).
- `lib`: `Function` and `Constant` to define new ones, the Python-syntax
  `operators`, and the backends for `ops.evaluate` (`lib.std`, `lib.numpy`,
  `lib.sympy`).

Everything else lives in private (underscore) modules, so autocompletion
only shows the public names.

## Organization: one traversal, many algebras

All of these operations are written as
[catamorphisms](https://lab.abilian.com/Tech/Python/OOP/The%20Zen%20of%20Polymorphism%20-%204%20Ways%20to%20Write%20Cleaner%20Python/#approach-4-the-ultimate-decoupling-with-catamorphism),
also called folds. The code is split into three parts:

1. **The data** (the private `symbolcat._core`). There are four node types: the leaves
   `Symbol` (user-defined, `x`), `Constant` (library-defined, `real.constants.pi`) and
   `Literal` (`2`), plus `Call(func, args)`. Two `singledispatch` functions
   describe their shape from the outside: `children` lists a node's direct
   children, and `fmap` applies a function to them, keeping the node shape.
1. **The traversal** (`cata`). It is written once and folds the tree bottom-up.
   At each node it first folds the children, and then passes the node, with its
   children *already replaced by their results*, to an **algebra**.
1. **The algebras** (`symbolcat.ops`). Each operation is just a rule for
   collapsing a single layer. None of them contain a recursive call.

Everything needed to write algebras is in `symbolcat.ops.algebras`.

```python
>>> from symbolcat.ops.algebras import cata, children
>>> count = lambda node: 1 + sum(children(node))
>>> cata(count, x * y + 1)
5

```

Most operations need one rule per node type or function. The `Algebra` class
collects those rules in a registry:

```python
>>> from symbolcat.ops.algebras import Algebra, Call, Expr
>>> depth = Algebra("depth")
>>> @depth.register(Expr)  # any leaf
... def _(node):
...     return 1
>>> @depth.register(Call)
... def _(node):
...     return 1 + max(node.args)  # args are already depths
>>> cata(depth, (x + 1) * 2)
3

```

A handler can be registered for three kinds of keys. When several match, the
most specific one wins:

- a specific function, such as `operators.add` or `real.sin`;
- a kind of function, such as `Function` or `Operator`;
- a node type, such as `Symbol`, `Constant`, `Literal` or `Call`.

## Operations

| function | algebra folds each node into |
| -------------- | --------------------------------------------- |
| `to_str` | `Printed(text, precedence)` |
| `to_latex` | `Printed(text, precedence)` |
| `evaluate` | a backend value (Python, NumPy, SymPy) |
| `substitute` | the rebuilt expression, with replacements |
| `simplify` | the rebuilt expression, constant-folded |
| `diff` | a `(value, derivative)` pair |
| `free_symbols` | the ordered set of user symbols |

### LaTeX

```python
>>> from symbolcat.ops import to_latex
>>> alpha_1 = Symbol("alpha_1")
>>> print(to_latex(real.sqrt(x**2 + 1) / (alpha_1 * real.sin(x))))
\frac{\sqrt{x^{2} + 1}}{\alpha_{1} \cdot \sin\left(x\right)}
>>> print(to_latex(-(x + y) ** 2))
-\left(x + y\right)^{2}

```

Parentheses are added only where they are needed. This works because every
child arrives as text *plus its precedence*. Expressions also define
`_repr_latex_`, so Jupyter renders them as formulas.

### Evaluation with different backends

```python
>>> from symbolcat.ops import evaluate
>>> evaluate(expr, {x: 1, y: 0})
1.0
>>> from symbolcat.lib import sympy as libsympy  # doctest: +SKIP
>>> evaluate(3 * real.cos(x), lib=libsympy.IMPL)  # doctest: +SKIP
3*cos(x)

```

A backend is a plain mapping from abstract functions (`real.cos`) to concrete
ones (`math.cos`). The algebra is the same for every backend, and the backend
is passed to it as context. User-defined functions just need an entry in the
mapping:

```python
>>> from symbolcat.lib import Function
>>> from symbolcat.lib.std import IMPL
>>> f = Function("f", arity=1)
>>> evaluate(f(x) + 1, {x: -3}, lib={**IMPL, f: lambda v: 2 * abs(v)})
7

```

### Substitution, simplification and differentiation

```python
>>> from symbolcat.ops import diff, simplify, substitute
>>> print(substitute(x + 2 * y, {x: y, y: x}))
y + 2 * x
>>> print(simplify(0 * x + 1 * x ** (3 - 2)))
x
>>> print(diff(x**3 + y * x, x))
3 * x ** 2 + y

```

`diff` needs both the value and the derivative of each child (think of the
product rule). Its algebra therefore folds every node into a
`(value, derivative)` pair, the same idea as forward-mode automatic
differentiation.

### E-graphs

`symbolcat.ops.egraph` simplifies and optimizes expressions with
[egglog](https://github.com/egraphs-good/egglog-python) (install the `egraph`
extra). Unlike `simplify`, which makes one bottom-up pass of local rules,
equality saturation keeps many equivalent forms at once (commuted, factored,
expanded, ...) and extracts the best one according to a cost model.

```python
>>> from symbolcat.ops import egraph
>>> z = Symbol("z")
>>> print(egraph.simplify(x * y + x * z))
x * (y + z)
>>> print(egraph.simplify((x + 1) ** 2 - x**2 - 2 * x))
1

```

`egraph.optimize` picks the cheapest expression to evaluate instead of the
smallest one, and returns repeated sub-expressions as the same object, so
`cata`, and therefore `evaluate`, computes them only once:

```python
>>> print(egraph.optimize(real.sin(x) ** 2 + real.sin(x)))
sin(x) + sin(x) * sin(x)

```

The translation into egglog is one more algebra. Functions
without rewrite rules are kept as opaque calls, so any expression survives
the round trip.

## Extending

A new function only needs rules in the algebras that should support it:

```python
>>> from symbolcat.ops.algebras import Printed, latex_algebra
>>> sinc = Function("sinc", arity=1)
>>> @latex_algebra.register(sinc)
... def _(node):
...     return Printed(rf"\operatorname{{sinc}}\left({node.args[0].text}\right)")
>>> print(to_latex(sinc(x) ** 2))
\operatorname{sinc}\left(x\right)^{2}

```

A new operation is a new algebra. Nothing in the core has to change.
