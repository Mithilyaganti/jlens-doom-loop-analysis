"""CPU tests for the Colab notebook syntax lint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jspace.notebook_lint import lint_paths, syntax_errors  # noqa: E402


def test_indented_pip_magic_is_a_syntax_error() -> None:
    source = "try:\n    %pip install -q vllm\n    print('ok')\nexcept Exception as exc:\n    print(exc)\n"
    errors = syntax_errors(source)
    assert errors
    assert "invalid syntax" in errors[0]


def test_indented_shell_magic_is_a_syntax_error() -> None:
    source = "if ready:\n    !python scripts/02_baseline_pass.py\n"
    errors = syntax_errors(source)
    assert errors
    assert any("invalid" in error or "decimal" in error for error in errors)


def test_column_zero_magic_with_continuation_parses() -> None:
    source = (
        "%pip install -q transformers \\\n"
        "    accelerate datasets\n"
        "import os\n"
        "print(os.getcwd())\n"
    )
    assert syntax_errors(source) == []


def test_plain_python_parses() -> None:
    assert syntax_errors("import subprocess\nsubprocess.run(['python', '-V'], check=True)\n") == []


def test_repo_notebooks_pass() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks
    assert lint_paths(notebooks) == []
