#!/usr/bin/env python
"""Exit 1 when a Colab notebook cell has a SyntaxError under column-0 magics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jspace.notebook_lint import lint_paths  # noqa: E402


def main() -> int:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not notebooks:
        print("no notebooks", file=sys.stderr)
        return 1
    errors = lint_paths(notebooks)
    if errors:
        print(f"{len(errors)} notebook syntax error(s):")
        for error in errors:
            print(error)
        return 1
    print(f"ok: {len(notebooks)} notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
