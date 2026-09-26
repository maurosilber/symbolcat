import importlib.util

# Optional backends: skip collecting their modules when not installed.
collect_ignore = [
    f"src/symbolcat/lib/{name}.py"
    for name in ("numpy", "sympy")
    if importlib.util.find_spec(name) is None
]
