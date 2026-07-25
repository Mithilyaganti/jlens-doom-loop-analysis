#!/usr/bin/env python
"""Ensure vendored jlens is present (Colab clones omit git submodule content)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("colab_vendors")


def main() -> int:
    from jspace.vendor_bootstrap import ensure_jlens_importable

    ensure_jlens_importable()
    logger.info("jlens import OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
