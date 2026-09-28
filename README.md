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
- `D`: the time derivative of a symbol, `D(x)`, an unknown of its own.
  `delay(x, tau)` and `partial(u, x)` build delay and partial differential
  equations.
- `real`: mathematical functions (`real.sin`, `real.ifelse`) and constants
  (`real.constants.pi`).
- `complex`: complex functions (`complex.exp`, `complex.re`, `complex.conj`)
  and the imaginary unit (`complex.constants.I`), named after `cmath`.
- `linalg`: vectors, matrices and linear algebra (`det`, `inv`, ...).
- `stats`: probability distributions (`stats.Normal(mu, sigma).logpdf(x)`),
  and propagation of uncertainty (`stats.propagate`).
- `ops`: operations on expressions, `ops.evaluate`, and one namespace per
  topic:
  - `ops.transform`: substitution, simplification, rewriting rules.
  - `ops.inspect`: symbols, size, shape, dimensions, sparsity patterns.
  - `ops.printing`: text, LaTeX, MathML, and Python, C and Julia source.
  - `ops.calculus`: derivatives, series, time derivatives.
  - `ops.polynomials`: Gröbner bases, GCDs, rational functions.
  - `ops.equations`: solving equations, systems of equations.
  - `ops.egraph`: simplification with e-graphs (needs `egglog`).
  - `ops.algebras`: the extension API (algebras, `cata`, node types).
- `models`: models of dynamical systems and their symbolic analysis:
  `models.dae` (index reduction, structural simplification), `models.ode`,
  `models.dynamics` (linearization, identifiability, ...),
  `models.stochastic`, `models.pde` and `models.reactions`.
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
| `simplify_naive` | the rebuilt expression, constant-folded |
| `diff` | a `(value, derivative)` pair |
| `free_symbols` | the ordered set of user symbols |
| `to_mathml` | `Printed(markup, precedence)` |
| `tree_view`, `to_graphviz` | lines of an indented tree, or a graph node id |
| `share` | the rebuilt node, reusing an equal earlier one |
| `cse`, `to_python` | an `(original, reduced)` pair, naming repeated nodes |
| `coefficients` | a `(value, polynomial)` pair |
| `series` | the first Taylor coefficients |
| `dimensions` | a `(value, dimension)` pair |
| `shape`, `expand_arrays` | an `(expanded, shape)` pair |

`gradient`, `jacobian` and `hessian` build on `diff`, and `solve_dependencies`
orders a system `{symbol: expression}` using `free_symbols`.

### LaTeX

```python
>>> from symbolcat.ops.printing import to_latex
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
>>> from symbolcat.ops.calculus import diff
>>> from symbolcat.ops.transform import simplify_naive, substitute
>>> print(substitute(x + 2 * y, {x: y, y: x}))
y + 2 * x
>>> print(simplify_naive(0 * x + 1 * x ** (3 - 2)))
x
>>> print(diff(x**3 + y * x, x))
3 * x ** 2 + y

```

`diff` needs both the value and the derivative of each child (think of the
product rule). Its algebra therefore folds every node into a
`(value, derivative)` pair, the same idea as forward-mode automatic
differentiation.

`gradient` returns a vector, and `jacobian` and `hessian` return matrices.
These come from `linalg`, which renders them as matrices in LaTeX and
evaluates them to nested lists (NumPy arrays, SymPy matrices):

```python
>>> from symbolcat.ops.calculus import hessian
>>> print(to_latex(hessian(x**2 * y)))
\begin{pmatrix} 2 \cdot y & 2 \cdot x \\ 2 \cdot x & 0 \end{pmatrix}
>>> evaluate(hessian(x**2 * y), {x: 1, y: 2})
[[4, 2], [2, 0]]

```

### Vectors and matrices

Arrays follow NumPy. Operators and `real` functions apply entry by entry,
broadcasting scalars, while `@` is the matrix product (the dot product for
two vectors). `linalg` also has `transpose`, `trace`, `det`, `inv`, `norm`,
`sum` and `cross`. `expand_arrays` computes these entry by entry:

```python
>>> from symbolcat import linalg
>>> from symbolcat.ops.inspect import shape
>>> from symbolcat.ops.transform import expand_arrays
>>> m = linalg.matrix([[x, 1], [0, y]])
>>> print(expand_arrays(linalg.det(m)))
x * y - 1 * 0
>>> print(to_latex(m * m + linalg.inv(m)))
\begin{pmatrix} x & 1 \\ 0 & y \end{pmatrix} \odot \begin{pmatrix} x & 1 \\ 0 & y \end{pmatrix} + \begin{pmatrix} x & 1 \\ 0 & y \end{pmatrix}^{-1}

```

`evaluate` expands arrays first, so every backend gives the same result.
So do `to_python` (arrays become lists, or `array=` calls), `series` and
`dimensions` (per entry), and the e-graph, which also knows matrix identities
like `transpose(a @ b) == transpose(b) @ transpose(a)`.
Symbols are scalars, unless given a shape. Operations on symbols with an
array shape are kept, and computed by the backend:

```python
>>> A = Symbol("A")
>>> shape(linalg.transpose(A) @ A, {A: (3, 2)})
(2, 2)
>>> B = Symbol("B")
>>> print(diff(linalg.det(A + x * B), x))
det(A + x * B) * trace(inv(A + x * B) @ B)

```

### Code generation, series and dimensions

`to_python` writes a function that computes an expression, with repeated
sub-expressions computed once:

```python
>>> from symbolcat.ops.printing import to_python
>>> print(to_python(real.sin(x) ** 2 + real.sin(x) * y))
def f(x, y):
    _0 = sin(x)
    return _0 ** 2 + _0 * y

```

`series` gives Taylor polynomials, extending the `(value, derivative)` idea of
`diff` to the first coefficients, and `dimensions` checks physical units:

```python
>>> from symbolcat.ops.calculus import series
>>> from symbolcat.ops.inspect import dimensions
>>> print(series(real.sin(x) / x, x, order=5))
1 - x ** 2 / 6 + x ** 4 / 120
>>> t, v = Symbol("t"), Symbol("v")
>>> dimensions(x + v * t, {x: {"m": 1}, t: {"s": 1}, v: {"m": 1, "s": -1}})
{'m': 1}

```

### E-graphs

`symbolcat.ops.egraph` simplifies and optimizes expressions with
[egglog](https://github.com/egraphs-good/egglog-python) (install the `egraph`
extra). Unlike `simplify_naive`, which makes one bottom-up pass of local rules,
equality saturation keeps many equivalent forms at once (commuted, factored,
expanded, ...) and extracts the best one according to a cost model.

```python
>>> from symbolcat.ops import egraph
>>> z = Symbol("z")
>>> print(egraph.simplify(x * y + x * z))
x * (y + z)
>>> print(egraph.simplify((x + 1) ** 2 - x**2 - 2 * x))
1
>>> print(egraph.simplify((x**2 - 1) / (x - 1)))
x + 1
>>> quartic = real.sin(x) ** 4 + real.cos(x) ** 4 + real.sin(2 * x) ** 2 / 2
>>> print(egraph.simplify(quartic, max_nodes=10_000, match_limit=10_000))
1

```

The rules cover arithmetic (exact on rationals), factoring, powers, `abs`,
`exp` and `log`, and trigonometric and hyperbolic identities, including the
math theories of [Metatheory.jl](https://github.com/JuliaSymbolics/Metatheory.jl).

By default, only rules that are valid wherever the expression is defined are
applied (the result may be defined on a larger domain: `x / x` becomes `1`).
So `log(x * y) - log(x)` becomes `log(y)` (both sides need `x, y > 0`),
but `sqrt(x**2) - x` only becomes `0` for nonnegative `x`, which is an
assumption:

```python
>>> from symbolcat.ops.egraph import integer, nonnegative
>>> print(egraph.simplify(real.log(x * y) - real.log(x)))
log(y)
>>> print(egraph.simplify(real.sqrt(x**2) - x))
abs(x) - x
>>> print(egraph.simplify(real.sqrt(x**2) - x, assume=[nonnegative(x)]))
0
>>> n = Symbol("n")
>>> print(egraph.simplify(real.cos(real.constants.pi * n) ** 2, assume=[integer(n)]))
1

```

The assumptions are `positive`, `nonnegative`, `negative`, `nonzero` and
`integer`, for any sub-expression. More properties are derived (`exp(x) > 0`,
a product of positives is positive, ...). An undefined input (a division by
zero, contradictory assumptions) is returned unchanged.

The search is bounded by `iterations` (passes of the rewrite rules),
`max_nodes` (size of the e-graph) and `match_limit` (matches of a rule in a
pass before it is banned for a while), so it stays fast on large expressions
at the cost of possibly missing a simplification, like the one above with
the default limits.

`egraph.optimize` picks the cheapest expression to evaluate instead of the
smallest one, and returns repeated sub-expressions as the same object, so
`cata`, and therefore `evaluate`, computes them only once:

```python
>>> print(egraph.optimize(x**8))  # three shared squarings
x * x * (x * x) * (x * x * (x * x))
>>> poly = 6 * x**5 + 5 * x**4 + 4 * x**3 + 3 * x**2 + 2 * x + 1
>>> print(egraph.optimize(poly))  # Horner's scheme
1 + ((((6 * x + 5) * x + 4) * x + 3) * x + 2) * x

```

The translation into egglog is one more algebra. Functions
without rewrite rules are kept as opaque calls, so any expression survives
the round trip.

## Extending

A new function only needs rules in the algebras that should support it:

```python
>>> from symbolcat.ops.algebras import Printed
>>> from symbolcat.ops.algebras.builtin import latex_algebra
>>> sinc = Function("sinc", arity=1)
>>> @latex_algebra.register(sinc)
... def _(node):
...     return Printed(rf"\operatorname{{sinc}}\left({node.args[0].text}\right)")
>>> print(to_latex(sinc(x) ** 2))
\operatorname{sinc}\left(x\right)^{2}

```

A new operation is a new algebra. Nothing in the core has to change.
