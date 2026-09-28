"""
symbolcat.ops.inspect
~~~~~~~~~~~~~~~~~~~~~

Queries about expressions: their symbols and variables, size, shape,
physical dimensions and sparsity patterns.
"""

from ._analysis._dimensions import dimensions
from ._analysis._inspect import count_nodes, depth, free_symbols, variables
from ._analysis._sparsity import hessian_sparsity, jacobian_sparsity
from ._calculus._time import delays
from ._transform._arrays import shape

__all__ = [
    "count_nodes",
    "delays",
    "depth",
    "dimensions",
    "free_symbols",
    "hessian_sparsity",
    "jacobian_sparsity",
    "shape",
    "variables",
]
