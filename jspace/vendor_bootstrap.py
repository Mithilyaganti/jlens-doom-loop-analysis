"""Bootstrap open-jlens vendor when GitHub clone omits submodule content."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OPEN_JLENS_REPO = "https://github.com/eliebak/open-jlens-data.git"

# Prefer /content/open-jlens-data on Colab so vendor never dirties the git clone.
_COLAB_JLENS = Path("/content/open-jlens-data")
_VENDOR_JLENS = ROOT / "vendor" / "open-jlens-data"


def _jlens_pkg(root: Path) -> Path:
    return root / "code" / "jacobian-lens"


def _jlens_marker(root: Path) -> Path:
    return _jlens_pkg(root) / "jlens" / "__init__.py"


def ensure_open_jlens() -> Path:
    for root in (_COLAB_JLENS, _VENDOR_JLENS):
        if _jlens_marker(root).is_file():
            return _jlens_pkg(root)

    # On Colab, clone outside the repo tree; elsewhere use vendor/
    dest = _COLAB_JLENS if Path("/content").is_dir() else _VENDOR_JLENS
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir() and not _jlens_marker(dest).is_file():
        import shutil

        logger.warning("Removing incomplete open-jlens vendor at %s", dest)
        shutil.rmtree(dest, ignore_errors=True)

    logger.info("Cloning open-jlens-data (required for jlens) → %s", dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", OPEN_JLENS_REPO, str(dest)],
        check=True,
    )
    marker = _jlens_marker(dest)
    if not marker.is_file():
        raise RuntimeError(f"Clone OK but jlens missing at {marker}")
    return _jlens_pkg(dest)


def pip_install_jlens(pkg_dir: Path | None = None) -> None:
    pkg_dir = pkg_dir or ensure_open_jlens()
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
