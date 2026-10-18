"""Enforces the Google-style rule: import modules, never classes or functions.

`from pkg import name` is allowed only when `pkg.name` is itself a module.
Imports from `typing`-like modules are exempt, as in the Google style guide.

Usage: python scripts/check_imports.py FILE [FILE ...]
"""

import ast
import importlib.util
import pathlib
import sys

_EXEMPT_MODULES = frozenset(
    {"__future__", "typing", "typing_extensions", "collections.abc"}
)


def _is_module(qualified_name: str) -> bool:
    try:
        return importlib.util.find_spec(qualified_name) is not None
    except (ImportError, ValueError):
        return False


def find_violations(path: pathlib.Path) -> list[str]:
    """Lists `from ... import ...` statements that import non-modules.

    Args:
        path: Python source file to inspect.

    Returns:
        One human-readable message per offending imported name.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level != 0:
            continue
        if node.module is None or node.module in _EXEMPT_MODULES:
            continue
        for alias in node.names:
            if not _is_module(f"{node.module}.{alias.name}"):
                violations.append(
                    f"{path}:{node.lineno}: import the module, not "
                    f"`{alias.name}` from `{node.module}`"
                )
    return violations


def main(argv: list[str]) -> int:
    """Checks every file in `argv`; returns a process exit code."""
    violations = [v for arg in argv for v in find_violations(pathlib.Path(arg))]
    for violation in violations:
        print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
