#!/usr/bin/env python
"""Unattended production runner: resume Exp3 until complete, then finalize deliverables."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "results" / "production_run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("production")

CKPT = ROOT / "results" / "checkpoints" / "exp3_main.json"
DONE = ROOT / "results" / "exp3" / "stats.json"
CONDITIONS = [
    "baseline",
    "ablate_trigger",
    "ablate_random",
    "ablate_control",
    "ablate_trigger_sensory",
    "ablate_trigger_motor",
]


def exp3_progress() -> tuple[int, int, dict]:
    max_prompts = int(os.environ.get("JLENS_EXP3_PROMPTS", "20"))
    total = max_prompts * len(CONDITIONS)
    if not CKPT.is_file():
        return 0, total, {}
    d = json.loads(CKPT.read_text(encoding="utf-8"))
    pp = d.get("per_prompt", {})
    done = sum(len(v) for v in pp.values())
    by_cond = {}
    for c in CONDITIONS:
        n = sum(1 for v in pp.values() if c in v)
        loops = sum(1 for v in pp.values() if c in v and v[c].get("is_loop"))
        by_cond[c] = {"done": n, "loops": loops}
    return done, total, by_cond


def run_script(name: str) -> int:
    # Skip launching if another exp3 GPU worker is already active
    if name == "05_exp3_causal.py":
        lock = ROOT / "results" / ".exp3.lock"
        if lock.is_file():
            try:
                pid = int(lock.read_text().strip().split()[0])
                import subprocess as sp

                r = sp.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
                if str(pid) in r.stdout and "python" in r.stdout.lower():
                    logger.info("Exp3 already running (PID %s) — waiting, not spawning duplicate", pid)
                    return 0
            except (ValueError, OSError):
                pass
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))
    env.setdefault("JLENS_EXP3_PROMPTS", "20")
    env.setdefault("JLENS_MAX_NEW_TOKENS", "1536")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    logger.info("Launching %s", name)
    proc = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts" / name)],
        cwd=str(ROOT),
        env=env,
    )
    logger.info("%s exited %s", name, proc.returncode)
    return proc.returncode


def finalize() -> None:
    from jspace.checkpoint import git_checkpoint

    run_script("merge_trigger_tables.py")
    git_checkpoint("production: phase1 complete")


def main() -> int:
    logger.info("=== Production run started %s ===", datetime.now(timezone.utc).isoformat())
    if DONE.is_file():
        logger.info("Exp3 already complete")
        finalize()
        return 0

    attempts = 0
    max_attempts = 50
    while not DONE.is_file() and attempts < max_attempts:
        attempts += 1
        done, total, by_cond = exp3_progress()
        logger.info(
            "Attempt %d — exp3 progress %d/%d — %s",
            attempts,
            done,
            total,
            json.dumps(by_cond),
        )
        rc = run_script("05_exp3_causal.py")
        if DONE.is_file():
            break
        if rc != 0:
            logger.warning("Exp3 returned %s; will retry after 30s", rc)
        time.sleep(30)

    if not DONE.is_file():
        logger.error("Exp3 did not complete after %d attempts", max_attempts)
        return 1

    finalize()
    logger.info("=== Production run complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
