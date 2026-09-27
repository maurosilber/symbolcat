import importlib.util

# Optional dependencies: skip collecting their modules when not installed.
collect_ignore = [
    f"src/symbolcat/lib/{name}.py"
    for name in ("numpy", "sympy")
    if importlib.util.find_spec(name) is None
]
if importlib.util.find_spec("egglog") is None:
    collect_ignore.append("src/symbolcat/ops/egraph")
