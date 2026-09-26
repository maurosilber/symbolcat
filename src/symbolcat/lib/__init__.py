"""
symbolcat.lib
~~~~~~~~~~~~~

Implementation libraries used by ``evaluate``.

A library is a plain mapping from abstract functions and constants
(``operators.add``, ``real.sin``, ``real.pi``) to concrete objects.
Optionally, the ``Symbol`` class maps to a factory used for free symbols,
which lets backends like SymPy keep them symbolic.

- ``std``: Python's ``operator`` and ``math`` modules.
- ``numpy``: NumPy ufuncs (import requires NumPy).
- ``sympy``: SymPy functions (import requires SymPy).
"""
