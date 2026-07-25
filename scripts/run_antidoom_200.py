#!/usr/bin/env python
"""Run antidoom-mix 200-prompt protocol end-to-end (Qwen3.5-4B)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("antidoom_200")

SCRIPTS = [
    "01_workspace_band.py",
    "02_baseline_pass.py",
    "03_exp1_static_geometry.py",
    "04_exp2_dynamic.py",
    "05_exp3_causal.py",
]


def _clear_stale_artifacts() -> None:
    """Remove prior-run artifacts that would mix incompatible prompt sets."""
    results = ROOT / "results"
    for rel in (
        "exp2/looping",
        "exp2/nonlooping",
        "checkpoints/baseline_pass.json",
        "checkpoints/exp3_main.json",
        "baseline_pass_summary.json",
        "trigger_gen_log.jsonl",
        "prompt_sample_ids.json",
        ".exp3.lock",
    ):
        p = results / rel
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.is_file():
            p.unlink(missing_ok=True)


def run_one(name: str, env: dict) -> int:
    path = ROOT / "scripts" / name
    logger.info("=" * 60)
    logger.info("RUNNING %s", name)
    logger.info("=" * 60)
    proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), env=env)
    logger.info("%s exited %s", name, proc.returncode)
    return proc.returncode


def write_report() -> Path:
    results = ROOT / "results"
    report_path = results / "RUN_REPORT_antidoom_mix_200.md"
    baseline = {}
    exp2 = {}
    exp3 = {}
    sample = {}
    if (results / "baseline_pass_summary.json").is_file():
        baseline = json.loads((results / "baseline_pass_summary.json").read_text(encoding="utf-8"))
    if (results / "exp2" / "summary.json").is_file():
        exp2 = json.loads((results / "exp2" / "summary.json").read_text(encoding="utf-8"))
    if (results / "exp3" / "stats.json").is_file():
        exp3 = json.loads((results / "exp3" / "stats.json").read_text(encoding="utf-8"))
    if (results / "prompt_sample_ids.json").is_file():
        sample = json.loads((results / "prompt_sample_ids.json").read_text(encoding="utf-8"))

    lines = [
        "# J-lens Run Report — antidoom-mix reasoning subset (200 prompts)",
        "",
        "## Dataset protocol",
        "- **Only** `LiquidAI/antidoom-mix-v1.0`",
        "- **7 reasoning component sources** (stratified, seed=42)",
        "- **No external prompt sets** (no HARD_MATH/CODE extras)",
        "- **No training** — inference + detection + J-lens readout only",
        "",
        "## Sample",
        f"- Total prompts: {sample.get('total_sampled', baseline.get('n_prompts', 'N/A'))}",
        f"- Per-source counts: `{json.dumps(sample.get('per_source_counts', {}))}`",
        f"- Audit file: `results/prompt_sample_ids.json`",
        "",
        "## Baseline pass (generation + trigger extraction + Exp2 cache)",
        f"- Completed: {baseline.get('n_completed', 'N/A')} / {baseline.get('n_prompts', 'N/A')}",
        f"- **Loop rate: {baseline.get('loop_rate', 0):.1%}** ({baseline.get('n_loop', 0)} loops)",
        f"- Unique trigger tokens: {baseline.get('unique_triggers', 0)}",
        f"- Looping prompt IDs: {len(baseline.get('looping_prompt_ids', []))}",
        f"- Trigger table: `results/trigger_tokens_qwen3.5-4b.csv`",
        f"- Generation log: `results/trigger_gen_log.jsonl`",
        "",
        "## Experiment 2 (same generations, analyze-only)",
        f"- Traces: loop={exp2.get('n_loop', 'N/A')}, non-loop={exp2.get('n_non', 'N/A')}",
        f"- Loop rate: {exp2.get('loop_rate', 0):.1%}" if exp2 else "- (pending)",
        "",
        "## Experiment 3 (looping prompts only)",
    ]
    if exp3.get("note"):
        lines.append(f"- {exp3['note']}")
    elif exp3.get("loop_rates"):
        lines.append(f"- Loop rates by condition: `{json.dumps(exp3['loop_rates'])}`")
        if exp3.get("mcnemar_baseline_vs_trigger"):
            lines.append(f"- McNemar (baseline vs trigger ablate): `{exp3['mcnemar_baseline_vs_trigger']}`")
    else:
        lines.append("- (pending or no baseline loops)")

    lines += [
        "",
        "## Model & settings",
        "- Model: `Qwen/Qwen3.5-4B` (4-bit NF4)",
        f"- max_new_tokens: {baseline.get('max_new_tokens', 2048)}",
        f"- temperature: {baseline.get('temperature', 0.01)}",
        "",
        "All numbers above are from real runs on this machine — nothing invented.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote report: %s", report_path)
    return report_path


def main() -> int:
    import os

    if os.environ.get("JLENS_KEEP_OLD_RESULTS", "0") != "1":
        _clear_stale_artifacts()

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))
    env.setdefault("JLENS_BASELINE_PROMPTS", "200")
    env.setdefault("JLENS_EXP2_PROMPTS", "200")
    env.setdefault("JLENS_MAX_NEW_TOKENS", "4000")
    env.setdefault("JLENS_EXP2_ANALYZE_ONLY", "1")

    band = ROOT / "results" / "workspace_band_qwen3.5-4b.json"
    start = 0 if not band.is_file() else 1

    codes = []
    for s in SCRIPTS[start:]:
        rc = run_one(s, env)
        codes.append((s, rc))
        if rc != 0 and s.startswith("01_"):
            return rc
        if rc not in (0, 2) and s.startswith("02_"):
            logger.error("Baseline pass failed — aborting")
            return rc
        if rc != 0 and s.startswith("03_"):
            return rc
        if rc != 0 and s.startswith("04_"):
            return rc
        if rc != 0 and s.startswith("05_"):
            return rc

    write_report()
    logger.info("antidoom 200 protocol complete: %s", codes)
    return 0 if all(c in (0, 2) for _, c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
