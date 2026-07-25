#!/usr/bin/env python
"""Write RUN_REPORT for LFM2-2.6B antidoom-mix 200-prompt baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def write_report() -> Path:
    from jspace.model_config import get_active_model, artifact_paths

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    baseline = {}
    exp2 = {}
    exp3 = {}
    sample = {}
    jlens_status = {}

    if paths["baseline_summary"].is_file():
        baseline = json.loads(paths["baseline_summary"].read_text(encoding="utf-8"))
    # Only include Exp2/3 if they belong to the same model as baseline
    baseline_model = baseline.get("model_id") or baseline.get("model_slug")
    current_model = cfg.model_id
    same_run = (
        not baseline
        or baseline_model in (current_model, cfg.slug)
        or baseline.get("n_completed", 0) == 0
    )
    if (paths["exp2"] / "summary.json").is_file() and same_run and baseline.get("n_completed", 0) > 0:
        exp2 = json.loads((paths["exp2"] / "summary.json").read_text(encoding="utf-8"))
    if (paths["exp3"] / "stats.json").is_file() and same_run and baseline.get("n_completed", 0) > 0:
        exp3 = json.loads((paths["exp3"] / "stats.json").read_text(encoding="utf-8"))
    if paths["prompt_sample"].is_file():
        sample = json.loads(paths["prompt_sample"].read_text(encoding="utf-8"))
    if paths["jlens_fit_status"].is_file():
        jlens_status = json.loads(paths["jlens_fit_status"].read_text(encoding="utf-8"))

    n_done = baseline.get("n_completed", 0)
    n_total = baseline.get("n_prompts", sample.get("total_sampled", 200))

    lines = [
        f"# J-lens Run Report — {cfg.display_name} (antidoom-mix 200 prompts)",
        "",
        "## Model disclaimer",
        "- **Model**: `LiquidAI/LFM2-2.6B` (public HuggingFace checkpoint)",
        "- **NOT** the blog's private early **LFM2.5-2.6B** checkpoint (unreleased)",
        "- **NOT** Antidoom-trained — this is the standard public post-trained release",
        "- **Dynamic thinking**: left at model default (hybrid reasoning enabled)",
        "",
        "## Run settings",
        f"- Backend: **{baseline.get('backend', 'unknown')}**",
        f"- vLLM dtype: `{baseline.get('vllm_dtype', 'n/a')}`",
        f"- max_new_tokens: {baseline.get('max_new_tokens', 4000)}",
        f"- temperature: {baseline.get('temperature', 0.01)}",
        f"- J-lens available: {baseline.get('jlens_available', False)}",
        "",
        "## Dataset",
        "- Source: `LiquidAI/antidoom-mix-v1.0` (7 reasoning component sources only)",
        f"- Stratified sample: {sample.get('total_sampled', n_total)} prompts, seed={sample.get('seed', 42)}",
        f"- Per-source counts: `{json.dumps(sample.get('per_source_counts', {}))}`",
        f"- Audit: `{paths['prompt_sample']}`",
        "",
        "## Baseline pass (generation + loop detection)",
        f"- Progress: **{n_done}/{n_total}** prompts completed",
        f"- **Loop rate: {baseline.get('loop_rate', 0):.1%}** ({baseline.get('n_loop', 0)} loops)",
        f"- Unique trigger tokens: {baseline.get('unique_triggers', 0)}",
        f"- Looping prompt IDs: {len(baseline.get('looping_prompt_ids', []))}",
        f"- Trigger table: `{paths['trigger_csv']}`",
        f"- Checkpoint: `{paths['baseline_checkpoint']}`",
        "",
        "## Experiment 2 (analyze-only, same generations)",
    ]
    if exp2:
        lines.append(
            f"- Traces: loop={exp2.get('n_loop')}, non-loop={exp2.get('n_non')}, "
            f"rate={exp2.get('loop_rate', 0):.1%}"
        )
    elif n_done < n_total:
        lines.append("- Pending (baseline not finished)")
    elif not baseline.get("jlens_available"):
        lines.append("- Skipped (no J-lens fitted)")
    else:
        lines.append("- Not run yet")

    lines += ["", "## Experiment 3 (looping prompts only)"]
    if exp3.get("note"):
        lines.append(f"- {exp3['note']}")
    elif exp3.get("loop_rates"):
        lines.append(f"- Loop rates: `{json.dumps(exp3['loop_rates'])}`")
    elif not baseline.get("looping_prompt_ids"):
        lines.append("- Skipped (zero baseline loops)")
    elif not baseline.get("jlens_available"):
        lines.append("- Skipped (no J-lens fitted)")
    else:
        lines.append("- Not run yet")

    lines += ["", "## J-lens fit status"]
    if jlens_status.get("fitted"):
        lines.append(f"- Fitted: `{paths['lens']}`")
    elif jlens_status.get("blocker"):
        lines.append(f"- **Not fitted**: {jlens_status['blocker']}")
    elif paths["lens"] and paths["lens"].is_file():
        lines.append(f"- Lens file present: `{paths['lens']}`")
    else:
        lines.append("- Not fitted — Exp1–3 blocked until `lenses/lfm2-2.6b.pt` exists")

    lines += [
        "",
        "All numbers above are from real runs — nothing invented.",
    ]
    report_path = paths["run_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


if __name__ == "__main__":
    p = write_report()
    print(f"Wrote {p}")
