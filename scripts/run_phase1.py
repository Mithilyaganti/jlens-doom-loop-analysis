#!/usr/bin/env python
"""Run full Phase 1 pipeline for Qwen3.5-4B end-to-end."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "00_setup.py",
    "01_workspace_band.py",
    "02_trigger_tokens.py",
    "03_exp1_static_geometry.py",
    "04_exp2_dynamic.py",
    "05_exp3_causal.py",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("phase1")


def run_one(name: str) -> int:
    path = ROOT / "scripts" / name
    logger.info("=" * 60)
    logger.info("RUNNING %s", name)
    logger.info("=" * 60)
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT) + (__import__("os").pathsep + env.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
        env=env,
    )
    logger.info("%s exited %s", name, proc.returncode)
    return proc.returncode


def main() -> int:
    # Optional: only run from a specific step
    start = 0
    if len(sys.argv) > 1:
        try:
            start = int(sys.argv[1])
        except ValueError:
            # match by name prefix
            for i, s in enumerate(SCRIPTS):
                if sys.argv[1] in s:
                    start = i
                    break

    codes = []
    for s in SCRIPTS[start:]:
        rc = run_one(s)
        codes.append((s, rc))
        # Hard stop on setup / workspace-band failures (later stages depend on them).
        # Trigger extraction may return 2 if zero loops (honest empty table) — continue
        # only if the CSV was written; other non-zero codes abort.
        if rc != 0 and s.startswith("00_"):
            logger.error("Setup failed — aborting pipeline")
            return rc
        if rc != 0 and s.startswith("01_"):
            logger.error("Workspace band failed — aborting (no invented boundaries)")
            return rc
        if rc not in (0, 2) and s.startswith("02_"):
            logger.error("Trigger extraction failed hard — aborting")
            return rc
        if rc != 0 and s.startswith("03_"):
            logger.error("Exp1 failed — aborting")
            return rc
    logger.info("Phase 1 complete: %s", codes)
    return 0 if all(c in (0, 2) for _, c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
