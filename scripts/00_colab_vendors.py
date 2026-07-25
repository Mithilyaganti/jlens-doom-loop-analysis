#!/usr/bin/env python
"""Ensure vendored jlens is present (Colab clones omit git submodule content)."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("colab_vendors")

OPEN_JLENS_REPO = "https://github.com/eliebak/open-jlens-data.git"
JLENS_PKG = ROOT / "vendor" / "open-jlens-data" / "code" / "jacobian-lens"


def ensure_open_jlens() -> Path:
    marker = JLENS_PKG / "jlens" / "__init__.py"
    if marker.is_file():
        logger.info("open-jlens vendor present: %s", JLENS_PKG)
        return JLENS_PKG

    dest = ROOT / "vendor" / "open-jlens-data"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir() and not marker.is_file():
        import shutil

        logger.warning("Removing incomplete vendor dir %s", dest)
        shutil.rmtree(dest, ignore_errors=True)

    logger.info("Cloning open-jlens-data → %s", dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", OPEN_JLENS_REPO, str(dest)],
        check=True,
    )
    if not marker.is_file():
        raise RuntimeError(f"Clone succeeded but jlens package missing at {marker}")
    return JLENS_PKG


def pip_install_jlens(pkg_dir: Path) -> None:
    logger.info("Installing jlens editable from %s", pkg_dir)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(pkg_dir)],
        check=True,
    )


def main() -> int:
    pkg = ensure_open_jlens()
    try:
        import jlens  # noqa: F401
    except ImportError:
        pip_install_jlens(pkg)
        import jlens  # noqa: F401
    logger.info("jlens import OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
