"""Bootstrap open-jlens vendor when GitHub clone omits submodule content."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OPEN_JLENS_REPO = "https://github.com/eliebak/open-jlens-data.git"
JLENS_PKG = ROOT / "vendor" / "open-jlens-data" / "code" / "jacobian-lens"
JLENS_MARKER = JLENS_PKG / "jlens" / "__init__.py"


def ensure_open_jlens() -> Path:
    if JLENS_MARKER.is_file():
        return JLENS_PKG

    dest = ROOT / "vendor" / "open-jlens-data"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir() and not JLENS_MARKER.is_file():
        import shutil

        logger.warning("Removing incomplete open-jlens vendor at %s", dest)
        shutil.rmtree(dest, ignore_errors=True)

    logger.info("Cloning open-jlens-data (required for jlens) → %s", dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", OPEN_JLENS_REPO, str(dest)],
        check=True,
    )
    if not JLENS_MARKER.is_file():
        raise RuntimeError(f"Clone OK but jlens missing at {JLENS_MARKER}")
    return JLENS_PKG


def pip_install_jlens(pkg_dir: Path | None = None) -> None:
    pkg_dir = pkg_dir or JLENS_PKG
    logger.info("pip install -e %s", pkg_dir)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(pkg_dir)],
        check=True,
    )


def ensure_jlens_importable() -> None:
    """Clone vendor if needed, add to sys.path, pip install if import fails."""
    pkg = ensure_open_jlens()
    pkg_str = str(pkg)
    if pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)
    try:
        import jlens  # noqa: F401
        return
    except ImportError:
        pip_install_jlens(pkg)
        import jlens  # noqa: F401
