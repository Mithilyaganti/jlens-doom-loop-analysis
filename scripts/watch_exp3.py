#!/usr/bin/env python
"""Poll Exp3 checkpoint until complete, then run final analysis if needed."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "results" / "checkpoints" / "exp3_main.json"
DONE_MARKER = ROOT / "results" / "exp3" / "stats.json"
CONDITIONS = [
    "baseline",
    "ablate_trigger",
    "ablate_random",
    "ablate_control",
    "ablate_trigger_sensory",
    "ablate_trigger_motor",
]
MAX_PROMPTS = int(__import__("os").environ.get("JLENS_EXP3_PROMPTS", "20"))


def progress() -> tuple[int, int]:
    if not CKPT.is_file():
        return 0, MAX_PROMPTS * len(CONDITIONS)
    d = json.loads(CKPT.read_text(encoding="utf-8"))
    pp = d.get("per_prompt", {})
    done = 0
    for _pid, conds in pp.items():
        done += len(conds)
    total = MAX_PROMPTS * len(CONDITIONS)
    return done, total


def main() -> int:
    if DONE_MARKER.is_file():
        print("Exp3 already complete:", DONE_MARKER)
        return 0

    while True:
        done, total = progress()
        print(f"exp3 progress {done}/{total}", flush=True)
        if DONE_MARKER.is_file():
            print("complete")
            return 0
        if done >= total:
            print("checkpoint full but stats missing — re-running exp3 for analysis")
            rc = subprocess.call(
                [sys.executable, str(ROOT / "scripts" / "05_exp3_causal.py")],
                cwd=str(ROOT),
                env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
            )
            return rc
        time.sleep(120)


if __name__ == "__main__":
    raise SystemExit(main())
