"""Syntax-check Colab notebooks the way a column-0-only magic transform does.

Google Colab has raised SyntaxError on ``!`` and ``%`` line magics that are
not at the first character of the line (colabtools issue 3507). IPython on
this machine rewrites those indented magics and they parse. This linter does
not: a magic is stripped only when it starts at column 0. An indented magic
is left as Python, which is a SyntaxError. That is the failure mode the
notebooks hit in Colab.

Top-level ``%pip`` / ``!python`` lines, including backslash continuations,
are replaced with ``pass`` before ``ast.parse``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterable


def _logical_lines(lines: list[str]) -> list[tuple[int, list[str]]]:
    """Group physical lines joined by a trailing backslash."""
    groups: list[tuple[int, list[str]]] = []
    buffer: list[str] = []
    start = 0
    for index, line in enumerate(lines):
        if not buffer:
            start = index
        buffer.append(line)
        if line.rstrip().endswith("\\"):
            continue
        groups.append((start, buffer))
        buffer = []
    if buffer:
        groups.append((start, buffer))
    return groups


def _is_line_magic(text: str) -> bool:
    stripped = text.lstrip(" ")
    if stripped.startswith("\t"):
        return False
    return stripped.startswith("%") or stripped.startswith("!")


def normalize_colab_source(source: str) -> str:
    """Strip column-0 magics. Leave indented magics in place."""
    lines = source.splitlines()
    out: list[str] = []
    for _start, group in _logical_lines(lines):
        first = group[0]
        indent = len(first) - len(first.lstrip(" "))
        if _is_line_magic(first) and indent == 0 and not first.startswith("\t"):
            out.extend("pass" for _ in group)
            continue
        out.extend(group)
    return "\n".join(out)


def syntax_errors(source: str) -> list[str]:
    """Parse one cell. Empty cells are clean."""
    if not source.strip():
        return []
    normalized = normalize_colab_source(source)
    try:
        ast.parse(normalized)
    except SyntaxError as exc:
        return [f"line {exc.lineno}: {exc.msg}"]
    return []


def lint_notebook(path: Path | str) -> list[str]:
    """Return ``path:cell N: ...`` errors for one notebook."""
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        for message in syntax_errors(source):
            errors.append(f"{path.as_posix()}:cell {index}: {message}")
    return errors


def lint_paths(paths: Iterable[Path | str]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        errors.extend(lint_notebook(path))
    return errors
