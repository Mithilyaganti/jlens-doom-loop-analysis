#!/usr/bin/env python
"""Verify Phase 1 deliverables exist and are non-empty (production gate)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ok(path: Path, min_bytes: int = 1) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"MISSING: {path.relative_to(ROOT)}"
    if path.stat().st_size < min_bytes:
        return False, f"EMPTY: {path.relative_to(ROOT)}"
    return True, f"OK: {path.relative_to(ROOT)}"


def main() -> int:
    required = [
        (ROOT / "README.md", 500),
        (ROOT / "results" / "trigger_tokens_qwen3.5-4b.csv", 1),
        (ROOT / "results" / "workspace_band_qwen3.5-4b.json", 50),
        (ROOT / "results" / "RESULTS_SUMMARY.md", 200),
        (ROOT / "results" / "exp1" / "norm_by_layer.png", 1000),
        (ROOT / "results" / "exp1" / "findings.json", 10),
        (ROOT / "results" / "exp2" / "summary.json", 10),
        (ROOT / "results" / "exp2" / "workspace_occupancy_pre_onset.png", 1000),
        (ROOT / "results" / "exp3" / "stats.json", 10),
        (ROOT / "results" / "exp3" / "loop_rate_by_condition.png", 1000),
        (ROOT / "results" / "exp3" / "per_prompt_loop_rate_change.csv", 10),
        (ROOT / "results" / "exp3" / "analysis_plan.md", 50),
        (ROOT / "results" / "status" / "00_setup.md", 50),
        (ROOT / "results" / "status" / "01_workspace_band.md", 50),
        (ROOT / "results" / "status" / "02_trigger_tokens.md", 50),
        (ROOT / "results" / "status" / "03_exp1.md", 50),
        (ROOT / "results" / "status" / "04_exp2.md", 50),
        (ROOT / "results" / "status" / "05_exp3.md", 50),
    ]
    fails = []
    for p, mb in required:
        good, msg = ok(p, mb)
        print(msg)
        if not good:
            fails.append(msg)

    # sanity: exp3 checkpoint completeness
    ckpt = ROOT / "results" / "checkpoints" / "exp3_main.json"
    if ckpt.is_file():
        d = json.loads(ckpt.read_text(encoding="utf-8"))
        pp = d.get("per_prompt", {})
        n = sum(len(v) for v in pp.values())
        print(f"EXP3 checkpoint entries: {n}/120")
        if n < 120:
            fails.append(f"INCOMPLETE checkpoint: {n}/120")

    exp2_loop = ROOT / "results" / "exp2" / "looping"
    exp2_non = ROOT / "results" / "exp2" / "nonlooping"
    if exp2_loop.is_dir() and exp2_non.is_dir():
        nl = len(list(exp2_loop.glob("p*")))
        nn = len(list(exp2_non.glob("p*")))
        print(f"OK: exp2 traces looping={nl} nonlooping={nn}")
        if nl + nn < 10:
            fails.append(f"Too few exp2 traces: {nl + nn}")

    if fails:
        print("\nGATE FAILED:")
        for f in fails:
            print(" ", f)
        return 1
    print("\nGATE PASSED — Phase 1 deliverables verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
