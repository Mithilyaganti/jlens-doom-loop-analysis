"""Paths inside the package. The repo gitignore ignores any ``data/`` directory."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CORPUS_DIR = PACKAGE_ROOT / "corpus"
MGSM_DIR = CORPUS_DIR / "mgsm"
