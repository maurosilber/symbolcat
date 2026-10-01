"""
symbolcat.functions
~~~~~~~~~~~~~~~~~~~

The functions expressions are built from. Each one declares the kind of its
result (see ``symbolcat.types``); what it means is given by the algebras in
``symbolcat.ops``, and how it is computed by ``symbolcat.backends``.

- ``operators``: the operators produced by Python syntax (``x + y``, ...).
- ``real``: mathematical functions and constants, like Python's ``math``.
- ``complex``: complex functions and constants, like Python's ``cmath``.
- ``linalg``: vectors, matrices and linear algebra functions.
- ``calculus``: the time derivative ``D``, ``delay`` and ``partial``.

``real``, ``complex``, ``linalg`` and ``operators`` are also available from
``symbolcat``.
"""
