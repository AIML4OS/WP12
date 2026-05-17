"""
Acceptance test 0 — graduation-readiness check.

Asserts that document_load.py and endpoints.py do not import (directly or
transitively among themselves) from summarizer.py, bench.py, or app_summarizer.py.

Run from the v1/ directory:
    python tools/check_shareable.py
"""
import ast
import sys
from pathlib import Path

V1 = Path(__file__).resolve().parent.parent
CONSUMER_MODULES = {"summarizer", "bench", "app_summarizer"}
SHAREABLE_FILES = ["document_load.py", "endpoints.py"]


def imports_in_file(path: Path) -> set[str]:
    """Return the set of top-level module names imported by a Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        print(f"  SYNTAX ERROR in {path.name}: {exc}")
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def main() -> int:
    failures: list[str] = []
    for filename in SHAREABLE_FILES:
        path = V1 / filename
        if not path.exists():
            print(f"  MISSING: {filename}")
            failures.append(f"{filename} not found")
            continue
        imported = imports_in_file(path)
        bad = imported & CONSUMER_MODULES
        if bad:
            failures.append(f"{filename} imports consumer module(s): {sorted(bad)}")
            print(f"  FAIL: {filename} imports {sorted(bad)}")
        else:
            print(f"  ok:   {filename}")

    if failures:
        print(f"\nGraduation-readiness check FAILED ({len(failures)} issue(s)).")
        print("The would-be-shared files import consumer-specific code.")
        print("Move consumer-specific logic to summarizer.py, bench.py, or app_summarizer.py.")
        return 1

    print("\nGraduation-readiness check PASSED — document_load and endpoints are clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
