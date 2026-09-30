"""
symbolcat.backends
~~~~~~~~~~~~~~~~~~

Implementation libraries for ``symbolcat.ops.evaluate``.

``std``, ``numpy`` and ``sympy`` each expose ``IMPL``, a mapping from
abstract functions and constants (``operators.add``, ``real.sin``,
``real.constants.pi``) to concrete objects. Optionally, the ``Symbol`` class
maps to a factory used for free symbols, which lets backends like SymPy keep
them symbolic. ``numpy`` and ``sympy`` require those packages.
"""
