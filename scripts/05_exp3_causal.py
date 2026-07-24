#!/usr/bin/env python
"""Experiment 3 — Causal ablation of trigger J-lens directions."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp3")

MAX_PROMPTS = int(os.environ.get("JLENS_EXP3_PROMPTS", "40"))
MAX_NEW_TOKENS = int(os.environ.get("JLENS_MAX_NEW_TOKENS", "1536"))
EVAL_N = int(os.environ.get("JLENS_EVAL_N", "30"))


def load_prompts(max_n: int) -> list[dict]:
    """Load real antidoom-mix prompts only. Fail hard if unavailable."""
    from jspace.prompts import load_antidoom_mix

    return load_antidoom_mix(max_n)


def main() -> int:
    from jspace.loading import load_stack, clear_cuda
    from jspace.generation import apply_chat_template, generate_greedy
    from jspace.detection import detect_loop
    from jspace.token_sets import build_trigger_set, build_frequency_matched_content, estimate_unigram_from_wikitext
    from jspace.intervention import InterventionContext, build_condition_directions
    from jspace.analysis import (
        ensure_dir,
        load_json,
        save_json,
        plot_bar_with_ci,
        proportion_ci,
        mcnemar_test,
        cohens_h,
        write_status,
        append_results_summary,
    )
    from jspace.checkpoint import save_checkpoint, load_checkpoint, git_checkpoint

    out = ensure_dir(ROOT / "results" / "exp3")
    ckpt_name = "exp3_main"

    # Pre-register analysis plan
    plan = """# Exp 3 Analysis Plan (pre-registered)

1. Primary outcome: loop rate = fraction of prompts with find_inner_repetition hit.
2. Conditions (paired per prompt):
   - baseline
   - ablate_trigger (workspace band)
   - ablate_random (workspace band, norm-matched unit vectors)
   - ablate_control (frequency-matched content token directions)
   - ablate_trigger_sensory
   - ablate_trigger_motor
3. Statistical tests: McNemar on paired loop/no-loop baseline vs each intervention.
4. Effect size: Cohen's h on loop-rate proportions.
5. Secondary: mean generation length; small GSM8K-style eval accuracy (sanity that ablation is surgical).
6. Dose-response: ablate top-1 / top-3 / top-5 trigger directions (workspace).
"""
    (out / "analysis_plan.md").write_text(plan, encoding="utf-8")

    stack = load_stack()
    band_path = ROOT / "results" / "workspace_band_qwen3.5-4b.json"
    if not band_path.is_file():
        raise FileNotFoundError(
            f"Missing {band_path}. Run scripts/01_workspace_band.py first."
        )
    band = load_json(band_path)
    for key in (
        "workspace_start",
        "workspace_end",
        "key_workspace_layers",
        "sensory_layers",
        "motor_layers",
    ):
        if key not in band:
            raise ValueError(f"workspace band file missing required key: {key}")

    trigger_csv = ROOT / "results" / "trigger_tokens_qwen3.5-4b.csv"
    if not trigger_csv.is_file():
        raise FileNotFoundError(
            f"Missing {trigger_csv}. Run scripts/02_trigger_tokens.py first."
        )
    # Prefer empirical triggers; operational RESTART_WORDS only if labeled in set builder
    triggers = build_trigger_set(stack.tokenizer, trigger_csv, top_n=10, target_size=10)
    trigger_ids = [t["token_id"] for t in triggers]
    if not trigger_ids:
        raise RuntimeError("No trigger token IDs available for ablation.")
    unigram = estimate_unigram_from_wikitext(stack.tokenizer, n_docs=200)
    if not unigram:
        raise RuntimeError(
            "Could not build wikitext unigram counts for frequency-matched controls. "
            "Check Salesforce/wikitext access — refusing to invent control tokens."
        )
    controls = build_frequency_matched_content(stack.tokenizer, trigger_ids, unigram, n=5)
    control_ids = [c["token_id"] for c in controls]
    if not control_ids:
        raise RuntimeError(
            "Frequency-matched control set is empty — refusing to reuse trigger IDs as controls."
        )

    ws_layers = list(band["key_workspace_layers"])
    sensory = list(band["sensory_layers"])
    motor = list(band["motor_layers"])
    # Limit sensory/motor to a few layers for speed (subset of real bands only)
    sensory = sensory[:5] if len(sensory) > 5 else sensory
    motor = motor[-5:] if len(motor) > 5 else motor
    if not ws_layers:
        raise RuntimeError("key_workspace_layers is empty")

    conditions = [
        ("baseline", [], "baseline"),
        ("ablate_trigger", ws_layers, "ablate_trigger"),
        ("ablate_random", ws_layers, "ablate_random"),
        ("ablate_control", ws_layers, "ablate_control"),
        ("ablate_trigger_sensory", sensory, "ablate_trigger_sensory"),
        ("ablate_trigger_motor", motor, "ablate_trigger_motor"),
    ]

    prompts = load_prompts(MAX_PROMPTS)
    logger.info("Exp3: %d prompts, %d conditions", len(prompts), len(conditions))

    # per_prompt[prompt_id][condition] = {is_loop, n_tokens}
    per_prompt: dict[int, dict[str, dict]] = {}
    ckpt = load_checkpoint(ckpt_name)
    if ckpt and ckpt.get("per_prompt"):
        per_prompt = {int(k): v for k, v in ckpt["per_prompt"].items()}
        logger.info("Resumed Exp3 checkpoint with %d prompts partially done", len(per_prompt))

    def persist_ckpt() -> None:
        save_checkpoint(
            ckpt_name,
            {
                "per_prompt": per_prompt,
                "conditions": [c[0] for c in conditions],
                "max_prompts": MAX_PROMPTS,
            },
        )

    for ci, (cname, layers, dir_key) in enumerate(conditions):
        logger.info("=== Condition %s (%d layers) ===", cname, len(layers))
        dirs = {}
        if cname != "baseline" and layers:
            dirs = build_condition_directions(
                stack,
                dir_key if cname != "ablate_trigger" else "ablate_trigger",
                layers,
                trigger_token_ids=trigger_ids[:1],
                control_token_ids=control_ids[:1],
                n_directions=1,
                seed=0,
            )
            if "sensory" in cname or "motor" in cname:
                dirs = build_condition_directions(
                    stack,
                    "ablate_trigger",
                    layers,
                    trigger_token_ids=trigger_ids[:1],
                    n_directions=1,
                    seed=0,
                )

        for i, p in enumerate(prompts):
            pid = p["prompt_id"]
            if pid in per_prompt and cname in per_prompt[pid]:
                continue
            chat = apply_chat_template(stack.tokenizer, p["text"])
            try:
                if cname == "baseline" or not layers:
                    gen = generate_greedy(stack, chat, max_new_tokens=MAX_NEW_TOKENS, temperature=0.01)
                else:
                    with InterventionContext(stack, layers, dirs, position=None, strength=1.0):
                        gen = generate_greedy(stack, chat, max_new_tokens=MAX_NEW_TOKENS, temperature=0.01)
                loop = detect_loop(
                    gen["generated_text"],
                    token_ids=gen["generated_ids"],
                    tokenizer=stack.tokenizer,
                )
            except Exception as e:
                logger.exception("fail %s p%s: %s", cname, pid, e)
                clear_cuda()
                continue

            per_prompt.setdefault(pid, {})[cname] = {
                "is_loop": bool(loop.is_loop),
                "n_tokens": len(gen["generated_ids"]),
                "trigger_decoded": loop.trigger_decoded,
            }
            persist_ckpt()  # save after every prompt for crash safety
            if (i + 1) % 5 == 0:
                logger.info("  %s %d/%d", cname, i + 1, len(prompts))
            clear_cuda()

    git_checkpoint("exp3 main conditions complete")

    # Aggregate
    cond_names = [c[0] for c in conditions]
    rows = []
    for pid, d in per_prompt.items():
        row = {"prompt_id": pid}
        for cn in cond_names:
            if cn in d:
                row[f"{cn}_loop"] = int(d[cn]["is_loop"])
                row[f"{cn}_tokens"] = d[cn]["n_tokens"]
            else:
                row[f"{cn}_loop"] = None
                row[f"{cn}_tokens"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_prompt_loop_rate_change.csv", index=False)

    rates = {}
    cis = {}
    lengths = {}
    for cn in cond_names:
        col = f"{cn}_loop"
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        k = int(valid.sum())
        n = int(valid.shape[0])
        rates[cn] = k / n if n else 0.0
        cis[cn] = proportion_ci(k, n)
        tcol = f"{cn}_tokens"
        lengths[cn] = float(df[tcol].dropna().mean()) if tcol in df.columns else 0.0

    plot_bar_with_ci(
        list(rates.keys()),
        list(rates.values()),
        [cis[k] for k in rates],
        ylabel="Loop rate",
        title="Doom-loop rate by intervention condition",
        out_path=out / "loop_rate_by_condition.png",
    )

    # Layer band comparison for trigger ablation
    band_labels = ["sensory", "workspace", "motor"]
    band_rates = [
        rates.get("ablate_trigger_sensory", 0),
        rates.get("ablate_trigger", 0),
        rates.get("ablate_trigger_motor", 0),
    ]
    band_cis = [
        cis.get("ablate_trigger_sensory", 0),
        cis.get("ablate_trigger", 0),
        cis.get("ablate_trigger_motor", 0),
    ]
    plot_bar_with_ci(
        band_labels,
        band_rates,
        band_cis,
        ylabel="Loop rate",
        title="Trigger ablation by layer band",
        out_path=out / "loop_rate_by_layer_band.png",
        color="coral",
    )

    # Dose-response: top-1,3,5 on subset of prompts
    dose_prompts = prompts[: min(20, len(prompts))]
    dose_rates = {}
    for n_dir, label in [(1, "top-1"), (3, "top-3"), (5, "top-5")]:
        loops = 0
        total = 0
        dirs = build_condition_directions(
            stack,
            "ablate_trigger",
            ws_layers,
            trigger_token_ids=trigger_ids[:n_dir],
            n_directions=n_dir,
            seed=0,
        )
        for p in dose_prompts:
            chat = apply_chat_template(stack.tokenizer, p["text"])
            try:
                with InterventionContext(stack, ws_layers, dirs, position=None):
                    gen = generate_greedy(stack, chat, max_new_tokens=MAX_NEW_TOKENS, temperature=0.01)
                loop = detect_loop(gen["generated_text"], token_ids=gen["generated_ids"], tokenizer=stack.tokenizer)
                total += 1
                loops += int(loop.is_loop)
            except Exception as e:
                logger.warning("dose fail: %s", e)
            clear_cuda()
        dose_rates[label] = loops / total if total else 0.0
        logger.info("dose %s: %.2f (%d/%d)", label, dose_rates[label], loops, total)

    # include baseline dose point
    dose_labels = ["baseline"] + list(dose_rates.keys())
    dose_vals = [rates.get("baseline", 0)] + list(dose_rates.values())
    plot_bar_with_ci(
        dose_labels,
        dose_vals,
        [proportion_ci(int(v * len(dose_prompts)), len(dose_prompts)) for v in dose_vals],
        ylabel="Loop rate",
        title="Loop rate vs ablation dose (# trigger directions)",
        out_path=out / "loop_rate_vs_ablation_strength.png",
        color="seagreen",
    )

    # Lightweight eval quality: few arithmetic questions under each condition
    eval_qs = [
        ("What is 17 + 28? Answer with just the number.", "45"),
        ("What is 9 * 8? Answer with just the number.", "72"),
        ("What is 100 - 37? Answer with just the number.", "63"),
        ("What is 144 / 12? Answer with just the number.", "12"),
        ("What is 2^5? Answer with just the number.", "32"),
    ]
    eval_acc = {}
    for cname, layers, dir_key in conditions[:4]:  # subset of conditions
        correct = 0
        dirs = {}
        if cname != "baseline" and layers:
            dirs = build_condition_directions(
                stack,
                "ablate_trigger" if "trigger" in cname else dir_key,
                layers,
                trigger_token_ids=trigger_ids[:1],
                control_token_ids=control_ids[:1],
                n_directions=1,
            )
        for q, ans in eval_qs:
            chat = apply_chat_template(stack.tokenizer, q)
            try:
                if cname == "baseline" or not layers:
                    gen = generate_greedy(stack, chat, max_new_tokens=64, temperature=0.01)
                else:
                    with InterventionContext(stack, layers, dirs):
                        gen = generate_greedy(stack, chat, max_new_tokens=64, temperature=0.01)
                text = gen["generated_text"]
                if ans in text:
                    correct += 1
            except Exception:
                pass
            clear_cuda()
        eval_acc[cname] = correct / len(eval_qs)

    plot_bar_with_ci(
        list(eval_acc.keys()),
        list(eval_acc.values()),
        [0] * len(eval_acc),
        ylabel="Eval accuracy (tiny arithmetic set)",
        title="Eval quality by condition (surgical check)",
        out_path=out / "eval_quality_by_condition.png",
        color="slategray",
    )

    # McNemar baseline vs ablate_trigger
    stats = {}
    if "baseline_loop" in df.columns and "ablate_trigger_loop" in df.columns:
        sub = df.dropna(subset=["baseline_loop", "ablate_trigger_loop"])
        b = int(((sub["baseline_loop"] == 1) & (sub["ablate_trigger_loop"] == 0)).sum())
        c = int(((sub["baseline_loop"] == 0) & (sub["ablate_trigger_loop"] == 1)).sum())
        stats["mcnemar_baseline_vs_trigger"] = mcnemar_test(b, c)
        stats["cohens_h_baseline_vs_trigger"] = cohens_h(
            rates.get("baseline", 0), rates.get("ablate_trigger", 0)
        )

    stats["loop_rates"] = rates
    stats["mean_gen_lengths"] = lengths
    stats["eval_acc"] = eval_acc
    stats["dose_rates"] = dose_rates
    save_json(stats, out / "stats.json")

    # Interpret success
    r_base = rates.get("baseline", 0)
    r_trig = rates.get("ablate_trigger", 0)
    r_rand = rates.get("ablate_random", 0)
    r_ctrl = rates.get("ablate_control", 0)
    r_sens = rates.get("ablate_trigger_sensory", 0)
    positive = (r_trig < r_base) and (r_trig < r_rand) and (r_trig < r_ctrl) and (r_trig <= r_sens)
    negative = abs(r_trig - r_rand) < 0.05 and abs(r_trig - r_ctrl) < 0.05
    result_type = "positive" if positive else ("negative" if negative else "mixed")

    status = f"""# Experiment 3 — Causal Interventions

## Loop rates
{json.dumps(rates, indent=2)}

## Interpretation
- Result type: **{result_type}**
- Baseline loop rate: {r_base:.3f}
- Ablate trigger (workspace): {r_trig:.3f}
- Ablate random: {r_rand:.3f}
- Ablate control token: {r_ctrl:.3f}
- Ablate trigger sensory: {r_sens:.3f}
- Ablate trigger motor: {rates.get('ablate_trigger_motor', 0):.3f}

## Stats
{json.dumps(stats.get('mcnemar_baseline_vs_trigger', {}), indent=2)}
Cohen's h (baseline vs trigger ablate): {stats.get('cohens_h_baseline_vs_trigger')}

## Dose-response
{json.dumps(dose_rates, indent=2)}

## Eval quality (tiny set)
{json.dumps(eval_acc, indent=2)}

Both positive and negative results are publishable per PROJECT_SPEC §11.6.
"""
    write_status("05_exp3", status)
    append_results_summary("Experiment 3 — Causal Interventions", status)
    clear_cuda()
    return 0


if __name__ == "__main__":
    from jspace.run_lock import acquire_lock, release_lock

    if not acquire_lock("exp3"):
        logger.error(
            "Another Exp3 GPU process is already running (results/.exp3.lock). "
            "Kill duplicates before starting a new run."
        )
        raise SystemExit(3)
    try:
        raise SystemExit(main())
    finally:
        release_lock()
