#!/usr/bin/env python
"""Experiment 3 — Causal intervention test (fixed pre-registered protocol v2).

Two-pass design per baseline-looping prompt:
  Pass 1 regenerates the baseline (greedy, seeded) to find the loop onset
  fresh (trigger token + position via jspace.detection on the generated ids).
  Pass 2 runs each condition with interventions keyed to THAT trace's onset.
"""

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

MAX_NEW_TOKENS = int(os.environ.get("JLENS_MAX_NEW_TOKENS", "4000"))
TIMING_N = int(os.environ.get("JLENS_EXP3_TIMING_N", "20"))
DOSE_N = int(os.environ.get("JLENS_EXP3_DOSE_N", "20"))
GSM8K_MAX_NEW_TOKENS = 512
GSM8K_START_POS = 50  # interventions on eval questions start mid-generation (documented neutral choice)
TIMING_OFFSETS = (-3, -1)  # @0 is the main ablate_trigger_ws run
DOSE_NS = (1, 3, 5)

MAIN_CONDITIONS = [
    "baseline_rerun",
    "ablate_trigger_ws",
    "ablate_random",
    "ablate_control_token",
    "ablate_trigger_sensory",
    "ablate_trigger_motor",
    "logit_suppress_trigger",
]
GSM8K_CONDITIONS = ["baseline_rerun", "ablate_trigger_ws", "logit_suppress_trigger"]


def active_main_conditions() -> list[str]:
    """Locked v2 conditions, plus opt-in HF decoding baselines.

    ``JLENS_EXP3_DECODING_BASELINES=1`` appends ``repetition_penalty`` and
    ``no_repeat_ngram``. The default is off, so a rerun of the locked
    protocol keeps the same condition list. Those arms are not in
    ``ANALYSIS_PLAN`` and have no measured loop rate in this checkout.
    """
    from jspace.decoding_baselines import extend_condition_names

    return extend_condition_names(list(MAIN_CONDITIONS))

ANALYSIS_PLAN = """# Exp 3 Analysis Plan (pre-registered, fixed protocol v2)

1. Design: TWO-PASS. Pass 1 regenerates each baseline-looping prompt
   (greedy, seeded) to find the loop onset fresh (trigger token + position
   via jspace.detection on the generated ids). Prompts that do not re-loop
   are marked `stable_nonloop` and keep their pass-1 label for McNemar
   pairing; their interventions are keyed to the original exp2 trace onset
   (trace meta.json / re-detected from tokens.pt), else to the global top-1
   empirical trigger from position 0 (`global_fallback`).
2. Main conditions (paired per prompt; ablations start at onset_pos, workspace
   band = ALL band layers, directions are per-trace):
   - baseline_rerun (pass 1)
   - ablate_trigger_ws (trace trigger J-direction)
   - ablate_random (norm-matched unit random directions, seeded per prompt+layer)
   - ablate_control_token (frequency-matched content token direction)
   - ablate_trigger_sensory / ablate_trigger_motor (same direction, sensory/motor layers)
   - logit_suppress_trigger (positive control: -50 logit bias on the trace
     trigger token for JLENS_SUPPRESS_K (default 25) steps from onset, then off)
3. Timing sweep: ablate_trigger_ws at onset-3 and onset-1 (onset+0 = the main
   run) on the first JLENS_EXP3_TIMING_N (default 20) pass-1 loopers by prompt id.
4. Dose-response: top-1 (= main run) / top-3 / top-5 trigger directions
   (trace trigger first, then global empirical triggers, deduped) on the
   first JLENS_EXP3_DOSE_N (default 20) loopers by prompt id.
5. Surgical eval: 50-problem GSM8K sample (data/gsm8k_eval.json, HF
   openai/gsm8k test split, seed=42), greedy, max_new_tokens=512, conditions
   baseline_rerun / ablate_trigger_ws / logit_suppress_trigger. Interventions
   apply from generation step 50 (fixed neutral choice — eval questions have
   no trace onset; trigger = global top-1 empirical trigger).
   Score = exact match of the LAST number in the output vs gold answer
   (jspace.eval_sets.score_gsm8k_output; no number -> wrong).
6. Primary outcome: loop rate (find_inner_repetition hit). Stats per condition
   vs baseline_rerun: exact McNemar (jspace.stats.mcnemar_exact) + odds ratio,
   Wilson CIs on all rates (jspace.stats.wilson_ci), Cohen's h. Secondary:
   mean generated length with bootstrap CIs (jspace.stats.bootstrap_ci).
   Power note: jspace.stats.mcnemar_sample_size for HALVING the observed
   baseline loop rate at 80% power vs the realized number of pairs.

Env knobs: JLENS_MAX_NEW_TOKENS, JLENS_EXP3_TIMING_N, JLENS_EXP3_DOSE_N,
JLENS_GSM8K_N, JLENS_SUPPRESS_K, JLENS_EXP3_PROMPTS (cap on looping prompts,
for smoke runs).
"""


def load_looping_prompts() -> list[dict]:
    """Load only prompts that looped in the baseline pass (condition 1).

    JLENS_EXP3_PROMPTS caps the number of looping prompts (smoke runs).
    """
    from jspace.model_config import artifact_paths
    from jspace.prompts import load_prompt_sample

    paths = artifact_paths()
    summary_path = paths["baseline_summary"]
    # Legacy unscoped fallback
    if not summary_path.is_file():
        alt = paths["results"] / "baseline_pass_summary.json"
        if alt.is_file():
            summary_path = alt
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing {paths['baseline_summary']} (or legacy baseline_pass_summary.json) "
            "— run scripts/02_baseline_pass.py first."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    loop_ids = {int(x) for x in summary.get("looping_prompt_ids", [])}
    if not loop_ids:
        logger.warning("Baseline pass recorded zero looping prompts — Exp3 will be empty.")
        return []
    all_prompts = load_prompt_sample(paths["prompt_sample"])
    prompts = [p for p in all_prompts if int(p["prompt_id"]) in loop_ids]
    cap = os.environ.get("JLENS_EXP3_PROMPTS")
    if cap:
        try:
            cap_n = int(cap)
        except ValueError:
            cap_n = 0
        if cap_n > 0:
            prompts = prompts[:cap_n]
            logger.info("JLENS_EXP3_PROMPTS=%d cap applied: %d looping prompts", cap_n, len(prompts))
    logger.info("Exp3 looping-only subset: %d prompts", len(prompts))
    return prompts


def onset_from_original_trace(paths: dict, pid: int, tokenizer) -> dict | None:
    """Defensively recover the ORIGINAL baseline-pass loop onset for a prompt.

    Looks in results/exp2_{slug}/looping/p{pid}/ (canonical) and the legacy
    results/exp2/looping/p{pid}/. Reads meta.json — supporting both the new
    loop_trigger_pos / loop_trigger_token_id keys and the legacy
    trigger_token_index / trigger_token_id keys — and falls back to re-running
    the detector on tokens.pt. Returns None when nothing is recoverable.
    """
    import torch

    from jspace.detection import detect_loop

    for trace_dir in (paths["exp2"] / "looping" / f"p{pid}", paths["results"] / "exp2" / "looping" / f"p{pid}"):
        meta_path = trace_dir / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            pos = meta.get("loop_trigger_pos")
            tid = meta.get("loop_trigger_token_id")
            tstr = meta.get("loop_trigger_token_str", meta.get("trigger_decoded"))
            if pos is None or tid is None:
                pos = meta.get("trigger_token_index")
                tid = meta.get("trigger_token_id")
            if isinstance(pos, bool) or not isinstance(pos, int) or not isinstance(tid, int):
                pos, tid = None, None
            if pos is not None and tid is not None:
                return {
                    "onset_pos": int(pos),
                    "trigger_token_id": int(tid),
                    "trigger_token_str": str(tstr) if tstr else None,
                    "source": "trace_meta",
                }
        tokens_path = trace_dir / "tokens.pt"
        if tokens_path.is_file():
            try:
                payload = torch.load(tokens_path, map_location="cpu", weights_only=True)
            except TypeError:  # older torch without weights_only
                payload = torch.load(tokens_path, map_location="cpu")
            gen_ids = payload.get("generated_ids") if isinstance(payload, dict) else None
            if gen_ids:
                text = tokenizer.decode(list(gen_ids), skip_special_tokens=True)
                loop = detect_loop(text, token_ids=list(gen_ids), tokenizer=tokenizer)
                if loop.is_loop and loop.trigger_token_index is not None and loop.trigger_token_id is not None:
                    return {
                        "onset_pos": int(loop.trigger_token_index),
                        "trigger_token_id": int(loop.trigger_token_id),
                        "trigger_token_str": loop.trigger_token_str or loop.trigger_decoded,
                        "source": "trace_redetect",
                    }
    return None


def generate_greedy_with_processors(stack, prompt_text, *, max_new_tokens, temperature, logits_processor):
    """Delegate to jspace.generation.generate_greedy (seeded, greedy-capable)
    with the logits_processor list for the logit-suppression positive control.
    """
    from jspace.generation import generate_greedy as _gen

    return _gen(
        stack,
        prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        logits_processor=logits_processor,
    )


def tokenized_len(tokenizer, chat: str) -> int:
    """Prompt length exactly as generate_greedy tokenizes it (onset bookkeeping)."""
    return int(len(tokenizer(chat, truncation=True, max_length=2048)["input_ids"]))


def random_seed_for(pid: int, layer: int) -> int:
    """Deterministic per-(prompt, layer) seed for ablate_random directions."""
    return (int(pid) * 1_000_003 + int(layer) * 131 + 17) % (2 ** 31)


def main() -> int:
    from jspace.loading import load_stack, clear_cuda
    from jspace.model_config import get_active_model, artifact_paths
    from jspace.generation import apply_chat_template, generate_greedy
    from jspace.detection import detect_loop
    from jspace.token_sets import build_trigger_set, build_frequency_matched_content, estimate_unigram_from_wikitext
    from jspace.intervention import (
        InterventionContext,
        build_trigger_directions,
        random_direction,
        TriggerSuppressLogitsProcessor,
        suppress_k_from_env,
    )
    from jspace.eval_sets import load_gsm8k_eval, score_gsm8k_output, GSM8K_EVAL_PATH, gsm8k_n_from_env
    from jspace.stats import mcnemar_exact, wilson_ci, cohens_h, bootstrap_ci, mcnemar_power, mcnemar_sample_size
    from jspace.analysis import (
        ensure_dir,
        load_json,
        save_json,
        plot_bar_with_ci,
        write_status,
        append_results_summary,
    )
    from jspace.checkpoint import save_checkpoint, load_checkpoint

    suppress_k = suppress_k_from_env()
    conditions = active_main_conditions()
    gsm8k_n = gsm8k_n_from_env()
    logger.info(
        "Exp3 knobs: max_new_tokens=%d timing_n=%d dose_n=%d gsm8k_n=%d suppress_k=%d gsm8k_start_pos=%d",
        MAX_NEW_TOKENS, TIMING_N, DOSE_N, gsm8k_n, suppress_k, GSM8K_START_POS,
    )

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    out = ensure_dir(paths["exp3"])
    ckpt_name = "exp3_main"
    (out / "analysis_plan.md").write_text(ANALYSIS_PLAN, encoding="utf-8")

    stack = load_stack(require_lens=True)  # directions need the J-lens
    if stack.lens is None:
        raise RuntimeError("J-lens missing — Exp3 fixed protocol requires trigger J-directions.")
    band_path = paths["workspace_band"]
    if not band_path.is_file():
        raise FileNotFoundError(
            f"Missing {band_path}. Run scripts/01_workspace_band.py first."
        )
    band = load_json(band_path)
    for key in (
        "workspace_start",
        "workspace_end",
        "workspace_layers",
        "key_workspace_layers",
        "sensory_layers",
        "motor_layers",
    ):
        if key not in band:
            raise ValueError(f"workspace band file missing required key: {key}")

    trigger_csv = paths["trigger_csv"]
    if not trigger_csv.is_file():
        raise FileNotFoundError(
            f"Missing {trigger_csv}. Run scripts/02_trigger_tokens.py first."
        )
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
    control_token_id = int(control_ids[0])  # fixed pre-registered choice
    gsm8k_trigger_id = int(trigger_ids[0])  # eval questions have no trace trigger

    ws_layers = list(band["workspace_layers"])  # ALL band layers
    sensory = list(band["sensory_layers"])
    motor = list(band["motor_layers"])
    sensory = sensory[:5] if len(sensory) > 5 else sensory
    motor = motor[-5:] if len(motor) > 5 else motor
    if not ws_layers:
        raise RuntimeError("workspace_layers is empty")
    if not sensory or not motor:
        raise RuntimeError("sensory/motor bands empty — band comparison conditions undefined")

    prompts = load_looping_prompts()
    logger.info("Exp3: %d looping prompts, %d main conditions", len(prompts), len(conditions))
    if not prompts:
        stats = {
            "protocol": "exp3_fixed_protocol_v2",
            "loop_rates": {c: 0.0 for c in conditions},
            "note": "No looping prompts in baseline pass — causal interventions skipped (null by design).",
        }
        save_json(stats, out / "stats.json")
        status = "# Experiment 3 — Causal Interventions (fixed protocol v2)\n\nNo looping prompts in baseline pass; Exp3 skipped.\n"
        write_status("05_exp3", status)
        append_results_summary("Experiment 3 — Causal Interventions (fixed protocol v2)", status)
        clear_cuda()
        return 0

    # ---- checkpoint / resume (schema v2; legacy entries preserved as-is) ----
    per_prompt: dict[int, dict] = {}
    timing: dict[str, dict[str, dict]] = {}
    dose: dict[str, dict[str, dict]] = {}
    gsm8k: dict[str, dict] = {}
    legacy_conditions: list[str] = []
    ckpt = load_checkpoint(ckpt_name)
    if ckpt:
        if ckpt.get("per_prompt"):
            per_prompt = {int(k): v for k, v in ckpt["per_prompt"].items()}
        timing = {k: {str(p): r for p, r in v.items()} for k, v in (ckpt.get("timing") or {}).items()}
        dose = {k: {str(p): r for p, r in v.items()} for k, v in (ckpt.get("dose") or {}).items()}
        gsm8k = dict(ckpt.get("gsm8k") or {})
        legacy_conditions = list(ckpt.get("legacy_conditions") or ckpt.get("conditions") or [])
        logger.info(
            "Resumed Exp3 checkpoint (status=%s, %d prompts with entries, %d legacy conditions preserved)",
            ckpt.get("status"), len(per_prompt), len(legacy_conditions),
        )

    def n_done_pairs() -> int:
        return sum(sum(1 for c in v if c in conditions) for v in per_prompt.values())

    def persist_ckpt(status: str = "running") -> None:
        save_checkpoint(
            ckpt_name,
            {
                "status": status,
                "protocol": "exp3_fixed_protocol_v2",
                "model_id": cfg.model_id,
                "max_new_tokens": MAX_NEW_TOKENS,
                "per_prompt": per_prompt,
                "conditions": list(conditions),
                "legacy_conditions": legacy_conditions,
                "n_looping_prompts": len(prompts),
                "n_done_pairs": n_done_pairs(),
                "n_total_pairs": len(prompts) * len(conditions),
                "workspace_layers": ws_layers,
                "sensory_layers": sensory,
                "motor_layers": motor,
                "trigger_ids_used": trigger_ids[:1],
                "control_token_id": control_token_id,
                "timing_offsets": list(TIMING_OFFSETS),
                "timing_n": TIMING_N,
                "dose_ns": list(DOSE_NS),
                "dose_n": DOSE_N,
                "suppress_k": suppress_k,
                "gsm8k_start_pos": GSM8K_START_POS,
                "gsm8k_max_new_tokens": GSM8K_MAX_NEW_TOKENS,
                "timing": timing,
                "dose": dose,
                "gsm8k": gsm8k,
            },
        )

    # ------------------------- pass 1 + pass 2 -------------------------
    def run_intervention_condition(
        cond: str,
        pid: int,
        chat: str,
        plen: int,
        onset: dict,
        *,
        start_pos_override: int | None = None,
        n_dirs: int = 1,
    ) -> dict:
        """Run one pass-2 condition keyed to the trace onset. Returns the
        per-condition outcome dict (also stored in the checkpoint)."""
        offset = 0
        if start_pos_override is not None:
            start = max(0, int(start_pos_override))
        else:
            start = max(0, int(onset["onset_pos"]) + offset)
        trig_tid = int(onset["trigger_token_id"])

        from jspace.decoding_baselines import hf_overrides_for_condition

        decoding_overrides = hf_overrides_for_condition(cond)
        if decoding_overrides is not None:
            penalty = decoding_overrides.get("repetition_penalty")
            ngram = decoding_overrides.get("no_repeat_ngram_size")
            gen = generate_greedy(
                stack,
                chat,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                repetition_penalty=None if penalty is None else float(penalty),
                no_repeat_ngram_size=None if ngram is None else int(ngram),
            )
        elif cond == "logit_suppress_trigger":
            proc = TriggerSuppressLogitsProcessor(
                trig_tid, start_pos=start, k_steps=suppress_k, prompt_len=plen
            )
            gen = generate_greedy_with_processors(
                stack,
                chat,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                logits_processor=[proc],
            )
        else:
            if cond == "ablate_trigger_ws":
                layers = ws_layers
                dirs = build_trigger_directions(stack, trig_tid, layers)
                if n_dirs > 1:
                    extras = [t for t in trigger_ids if int(t) != trig_tid][: n_dirs - 1]
                    dirs = build_trigger_directions(stack, trig_tid, layers, extra_token_ids=extras)
            elif cond == "ablate_trigger_sensory":
                layers = sensory
                dirs = build_trigger_directions(stack, trig_tid, layers)
            elif cond == "ablate_trigger_motor":
                layers = motor
                dirs = build_trigger_directions(stack, trig_tid, layers)
            elif cond == "ablate_random":
                layers = ws_layers
                # norm-matched: trigger directions are unit-normalized, so unit randoms
                dirs = {l: [random_direction(stack.d_model, seed=random_seed_for(pid, l))] for l in layers}
            elif cond == "ablate_control_token":
                layers = ws_layers
                dirs = build_trigger_directions(stack, control_token_id, layers)
            else:
                raise ValueError(f"unknown condition {cond}")
            with InterventionContext(stack, layers, dirs, prompt_len=plen, start_pos=start):
                gen = generate_greedy(stack, chat, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)

        loop = detect_loop(
            gen["generated_text"],
            token_ids=gen["generated_ids"],
            tokenizer=stack.tokenizer,
        )
        result = {
            "is_loop": bool(loop.is_loop),
            "n_tokens": len(gen["generated_ids"]),
            "trigger_decoded": loop.trigger_decoded,
            "start_pos": start,
        }
        if cond == "logit_suppress_trigger":
            result["suppressed_steps"] = int(proc.n_suppressed_steps)
            result["suppress_window"] = list(proc.window())
        if decoding_overrides is not None:
            result["decoding_overrides"] = dict(decoding_overrides)
            result["decoding_backend"] = "hf"
        return result

    for i, p in enumerate(prompts):
        pid = int(p["prompt_id"])
        entry = per_prompt.setdefault(pid, {})

        # ---- pass 1: fresh baseline regeneration + onset detection ----
        if "_onset" not in entry:
            chat = apply_chat_template(stack.tokenizer, p["text"])
            try:
                gen = generate_greedy(stack, chat, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
                loop = detect_loop(
                    gen["generated_text"],
                    token_ids=gen["generated_ids"],
                    tokenizer=stack.tokenizer,
                )
            except Exception as e:
                logger.exception("pass-1 fail pid=%s: %s", pid, e)
                clear_cuda()
                continue
            onset = {
                "pass1_loop": bool(loop.is_loop),
                "onset_pos": loop.trigger_token_index if loop.is_loop else None,
                "trigger_token_id": loop.trigger_token_id if loop.is_loop else None,
                "trigger_token_str": loop.trigger_token_str or loop.trigger_decoded if loop.is_loop else None,
                "n_tokens": len(gen["generated_ids"]),
                "source": "pass1" if (loop.is_loop and loop.trigger_token_index is not None and loop.trigger_token_id is not None) else None,
            }
            if onset["source"] is None:
                # stable_nonloop (or unmappable hit): key interventions to the
                # ORIGINAL baseline trace onset; global fallback as last resort.
                trace_onset = onset_from_original_trace(paths, pid, stack.tokenizer)
                if trace_onset is not None:
                    onset.update(trace_onset)  # onset_pos / trigger_token_id / str / source
                else:
                    onset.update(
                        {
                            "onset_pos": 0,
                            "trigger_token_id": gsm8k_trigger_id,
                            "trigger_token_str": None,
                            "source": "global_fallback",
                        }
                    )
            entry["_onset"] = onset
            entry["baseline_rerun"] = {
                "is_loop": onset["pass1_loop"],
                "n_tokens": onset["n_tokens"],
                "trigger_decoded": loop.trigger_decoded,
            }
            persist_ckpt("running")
            logger.info(
                "  pass1 %d/%d pid=%s reloop=%s onset=%s@%s (%s) tokens=%d",
                i + 1, len(prompts), pid, onset["pass1_loop"],
                onset["trigger_token_str"], onset["onset_pos"], onset["source"],
                onset["n_tokens"],
            )
            clear_cuda()

        onset = entry["_onset"]

        # ---- pass 2: main conditions ----
        chat = apply_chat_template(stack.tokenizer, p["text"])
        plen = tokenized_len(stack.tokenizer, chat)
        for cond in conditions:
            if cond in ("baseline_rerun",) or cond in entry:
                continue
            try:
                entry[cond] = run_intervention_condition(cond, pid, chat, plen, onset)
            except Exception as e:
                logger.exception("fail %s pid=%s: %s", cond, pid, e)
                clear_cuda()
                continue
            persist_ckpt("running")
            logger.info(
                "  %s pid=%s loop=%s tokens=%d (pairs %d/%d)",
                cond, pid, entry[cond]["is_loop"], entry[cond]["n_tokens"],
                n_done_pairs(), len(prompts) * len(conditions),
            )
            clear_cuda()

    persist_ckpt("main_conditions_done")

    if os.environ.get("JLENS_GIT_CHECKPOINT", "0") == "1":
        from jspace.checkpoint import git_checkpoint

        git_checkpoint("exp3 main conditions complete")

    # ------------------------- subsets: loopers only -------------------------
    in_run_pids = {int(p["prompt_id"]) for p in prompts}
    looper_pids = sorted(
        [
            pid
            for pid in per_prompt
            if pid in in_run_pids and "_onset" in per_prompt[pid] and per_prompt[pid]["_onset"].get("pass1_loop")
        ],
        key=int,
    )
    timing_pids = looper_pids[:TIMING_N]
    dose_pids = looper_pids[:DOSE_N]
    logger.info("Subsets: %d loopers in run; timing_n=%d dose_n=%d", len(looper_pids), len(timing_pids), len(dose_pids))

    def onset_of(pid: int) -> dict:
        return per_prompt[pid]["_onset"]

    # Timing sweep: ablate_trigger_ws at onset-3 / onset-1 (onset+0 = main run)
    for off in TIMING_OFFSETS:
        cname = f"ablate_trigger_ws@{off}"
        bucket = timing.setdefault(cname, {})
        for pid in timing_pids:
            if str(pid) in bucket:
                continue
            p = next(x for x in prompts if int(x["prompt_id"]) == pid)
            chat = apply_chat_template(stack.tokenizer, p["text"])
            plen = tokenized_len(stack.tokenizer, chat)
            try:
                bucket[str(pid)] = run_intervention_condition(
                    "ablate_trigger_ws", pid, chat, plen, onset_of(pid),
                    start_pos_override=int(onset_of(pid)["onset_pos"]) + off,
                )
            except Exception as e:
                logger.exception("timing %s pid=%s fail: %s", cname, pid, e)
                clear_cuda()
                continue
            persist_ckpt("running")
            logger.info("  %s pid=%s loop=%s", cname, pid, bucket[str(pid)]["is_loop"])
            clear_cuda()

    # Dose-response: top-1 (reuse the main run) / top-3 / top-5 trigger directions
    for nd in DOSE_NS:
        cname = f"dose_top-{nd}"
        bucket = dose.setdefault(cname, {})
        if nd == 1:
            # top-1 IS the main per-trace trigger ablation — reuse, no recompute
            for pid in dose_pids:
                main = per_prompt.get(pid, {}).get("ablate_trigger_ws")
                if main is not None:
                    bucket[str(pid)] = dict(main)
            persist_ckpt("running")
            continue
        for pid in dose_pids:
            if str(pid) in bucket:
                continue
            p = next(x for x in prompts if int(x["prompt_id"]) == pid)
            chat = apply_chat_template(stack.tokenizer, p["text"])
            plen = tokenized_len(stack.tokenizer, chat)
            try:
                bucket[str(pid)] = run_intervention_condition(
                    "ablate_trigger_ws", pid, chat, plen, onset_of(pid), n_dirs=nd
                )
            except Exception as e:
                logger.exception("dose %s pid=%s fail: %s", cname, pid, e)
                clear_cuda()
                continue
            persist_ckpt("running")
            logger.info("  %s pid=%s loop=%s", cname, pid, bucket[str(pid)]["is_loop"])
            clear_cuda()

    # ------------------------- GSM8K surgical eval -------------------------
    gsm8k_problems: list[dict] = []
    if GSM8K_EVAL_PATH.is_file():
        gsm8k_problems = load_gsm8k_eval(GSM8K_EVAL_PATH, n=gsm8k_n)
    else:
        logger.error(
            "MISSING %s — GSM8K surgical eval cannot run. Create it with "
            "jspace.eval_sets.sample_gsm8k_eval(50, seed=42) (needs `datasets` + HF access). "
            "Reporting gsm8k as unavailable rather than substituting an eval.",
            GSM8K_EVAL_PATH,
        )
    for cond in GSM8K_CONDITIONS:
        # JLENS_GSM8K_N may have changed between runs — keep stored results in sync.
        g = gsm8k.get(cond)
        if g and isinstance(g.get("correct"), list):
            g["correct"] = g["correct"][: len(gsm8k_problems)]
            g["n_tokens"] = (g.get("n_tokens") or [])[: len(g["correct"])]
        gstate = gsm8k.setdefault(cond, {"correct": [], "n_tokens": []})
        for qi, prob in enumerate(gsm8k_problems):
            if qi < len(gstate["correct"]):
                continue  # resume
            chat = apply_chat_template(stack.tokenizer, prob["question"])
            plen = tokenized_len(stack.tokenizer, chat)
            try:
                if cond == "baseline_rerun":
                    gen = generate_greedy(stack, chat, max_new_tokens=GSM8K_MAX_NEW_TOKENS, temperature=0.0)
                elif cond == "ablate_trigger_ws":
                    dirs = build_trigger_directions(stack, gsm8k_trigger_id, ws_layers)
                    with InterventionContext(
                        stack, ws_layers, dirs, prompt_len=plen, start_pos=GSM8K_START_POS
                    ):
                        gen = generate_greedy(stack, chat, max_new_tokens=GSM8K_MAX_NEW_TOKENS, temperature=0.0)
                else:  # logit_suppress_trigger
                    proc = TriggerSuppressLogitsProcessor(
                        gsm8k_trigger_id, start_pos=GSM8K_START_POS, k_steps=suppress_k, prompt_len=plen
                    )
                    gen = generate_greedy_with_processors(
                        stack, chat, max_new_tokens=GSM8K_MAX_NEW_TOKENS,
                        temperature=0.0, logits_processor=[proc],
                    )
                ok = score_gsm8k_output(gen["generated_text"], prob["answer"])
            except Exception as e:
                logger.exception("gsm8k %s q%d fail: %s", cond, qi, e)
                clear_cuda()
                continue
            gstate["correct"].append(int(bool(ok)))
            gstate["n_tokens"].append(len(gen["generated_ids"]))
            persist_ckpt("running")
            clear_cuda()
        logger.info(
            "gsm8k %s: %d/%d correct", cond,
            sum(gstate["correct"]) if gstate["correct"] else 0, len(gstate["correct"]),
        )

    persist_ckpt("gsm8k_done")

    # ------------------------- aggregation -------------------------
    def cond_results(source: dict, cname: str, pids=None) -> dict[int, dict]:
        """pid -> outcome for a main condition (per_prompt) or subset dict."""
        if source is per_prompt:
            return {
                pid: entry[cname]
                for pid, entry in per_prompt.items()
                if cname in entry and (pids is None or pid in pids)
            }
        return {int(k): v for k, v in source.get(cname, {}).items()}

    timing_cond_names = [f"ablate_trigger_ws@{off}" for off in TIMING_OFFSETS]
    dose_cond_names = [f"dose_top-{nd}" for nd in DOSE_NS]

    # per-prompt CSV
    csv_cols = ["prompt_id", "pass1_loop", "onset_source", "onset_pos", "trigger_token_id", "trigger_token_str"]
    for cname in conditions + timing_cond_names + dose_cond_names:
        csv_cols += [f"{cname}_loop", f"{cname}_gen_length"]
    rows = []
    for pid in sorted({int(p["prompt_id"]) for p in prompts} | set(per_prompt)):
        entry = per_prompt.get(pid, {})
        onset = entry.get("_onset", {})
        row = {
            "prompt_id": pid,
            "pass1_loop": onset.get("pass1_loop"),
            "onset_source": onset.get("source"),
            "onset_pos": onset.get("onset_pos"),
            "trigger_token_id": onset.get("trigger_token_id"),
            "trigger_token_str": onset.get("trigger_token_str"),
        }
        for cname in conditions:
            r = entry.get(cname)
            row[f"{cname}_loop"] = int(r["is_loop"]) if r else None
            row[f"{cname}_gen_length"] = r["n_tokens"] if r else None
        for cname, src in [(f"ablate_trigger_ws@{off}", timing) for off in TIMING_OFFSETS] + [
            (f"dose_top-{nd}", dose) for nd in DOSE_NS
        ]:
            r = src.get(cname, {}).get(str(pid))
            row[f"{cname}_loop"] = int(r["is_loop"]) if r else None
            row[f"{cname}_gen_length"] = r["n_tokens"] if r else None
        rows.append(row)
    df = pd.DataFrame(rows, columns=csv_cols)
    df.to_csv(out / "per_prompt_loop_rate_change.csv", index=False)

    # rates / CIs / lengths for every reported condition
    reported: list[tuple[str, dict[int, dict]]] = [
        (c, cond_results(per_prompt, c)) for c in conditions
    ] + [
        (c, cond_results(timing, c)) for c in timing_cond_names
    ] + [
        (c, cond_results(dose, c)) for c in dose_cond_names
    ]
    rates: dict[str, float] = {}
    rate_ci: dict[str, list[float]] = {}
    lengths: dict[str, float] = {}
    length_ci: dict[str, list[float]] = {}
    for cname, res in reported:
        if not res:
            continue
        k = sum(1 for r in res.values() if r["is_loop"])
        n = len(res)
        rates[cname] = k / n
        lo, hi = wilson_ci(k, n)
        rate_ci[cname] = [float(lo), float(hi)]
        lens = [r["n_tokens"] for r in res.values()]
        point, blo, bhi = bootstrap_ci(lens, stat="mean", n_boot=2000, seed=0)
        lengths[cname] = float(point) if lens else 0.0
        length_ci[cname] = [float(blo), float(bhi)]

    # contrasts vs baseline_rerun (exact McNemar + effect sizes)
    contrasts: dict[str, dict] = {}
    base = cond_results(per_prompt, "baseline_rerun")
    for cname, res in reported:
        if cname == "baseline_rerun":
            continue
        paired = {pid: r for pid, r in res.items() if pid in base}
        if not paired:
            continue
        b = sum(1 for pid, r in paired.items() if base[pid]["is_loop"] and not r["is_loop"])
        c = sum(1 for pid, r in paired.items() if not base[pid]["is_loop"] and r["is_loop"])
        mc = mcnemar_exact(b, c)
        n_pairs = len(paired)
        r0 = sum(1 for pid in paired if base[pid]["is_loop"]) / n_pairs
        r1 = sum(1 for r in paired.values() if r["is_loop"]) / n_pairs
        lo0, hi0 = wilson_ci(sum(1 for pid in paired if base[pid]["is_loop"]), n_pairs)
        lo1, hi1 = wilson_ci(sum(1 for r in paired.values() if r["is_loop"]), n_pairs)
        contrasts[cname] = {
            "n_pairs": n_pairs,
            "b_baseline_only": int(b),
            "c_condition_only": int(c),
            "p_exact": mc["p_exact"],
            "odds_ratio_b_over_c": mc["odds_ratio_b_over_c"],
            "rate_baseline": r0,
            "rate_condition": r1,
            "wilson_ci_baseline": [float(lo0), float(hi0)],
            "wilson_ci_condition": [float(lo1), float(hi1)],
            "cohens_h": cohens_h(r0, r1),
        }

    # power note: halving the observed baseline loop rate at 80% power
    trig_pairs = contrasts.get("ablate_trigger_ws")
    power_note: dict = {"target_power": 0.8, "smallest_effect": "halve the baseline loop rate"}
    if trig_pairs and trig_pairs["n_pairs"] > 0 and trig_pairs["rate_baseline"] > 0:
        p10 = trig_pairs["rate_baseline"] / 2.0
        p01 = max(trig_pairs["c_condition_only"] / trig_pairs["n_pairs"], 0.01)
        n_have = trig_pairs["n_pairs"]
        required = mcnemar_sample_size(p10, p01, power_target=0.8, n_max=2000)
        power_note.update(
            {
                "contrast": "ablate_trigger_ws vs baseline_rerun",
                "observed_baseline_rate": trig_pairs["rate_baseline"],
                "assumed_p10": p10,
                "assumed_p01": p01,
                "n_pairs_realized": n_have,
                "n_pairs_required": required,
                "achieved_power": mcnemar_power(p10, p01, n_have),
                "adequately_powered": bool(required is not None and n_have >= required),
            }
        )
    else:
        power_note["note"] = "baseline loop rate is 0 (or no pairs) — power note undefined"
    print("POWER NOTE:", json.dumps(power_note, indent=2, default=str))
    logger.info("POWER NOTE: %s", json.dumps(power_note, default=str))

    # gsm8k summary
    gsm8k_summary: dict = {
        "n_problems": len(gsm8k_problems),
        "start_pos": GSM8K_START_POS,
        "trigger_token_id": gsm8k_trigger_id,
        "max_new_tokens": GSM8K_MAX_NEW_TOKENS,
        "source": "HF openai/gsm8k test split, seed=42, data/gsm8k_eval.json"
        if gsm8k_problems
        else "UNAVAILABLE (data/gsm8k_eval.json missing)",
        "accuracy": {},
        "wilson_ci": {},
        "mean_gen_length": {},
    }
    for cond in GSM8K_CONDITIONS:
        gstate = gsm8k.get(cond, {})
        correct = gstate.get("correct", [])
        if correct:
            k = int(sum(correct))
            n = len(correct)
            lo, hi = wilson_ci(k, n)
            gsm8k_summary["accuracy"][cond] = k / n
            gsm8k_summary["wilson_ci"][cond] = [float(lo), float(hi)]
            gsm8k_summary["mean_gen_length"][cond] = float(np.mean(gstate.get("n_tokens", [])))

    # ---- plots ----
    main_names = [c for c in conditions if c in rates]
    plot_bar_with_ci(
        main_names,
        [rates[c] for c in main_names],
        [(rate_ci[c][1] - rate_ci[c][0]) / 2 for c in main_names],
        ylabel="Loop rate",
        title="Doom-loop rate by intervention condition (fixed protocol)",
        out_path=out / "loop_rate_by_condition.png",
    )
    band_labels = ["sensory", "workspace", "motor"]
    band_keys = ["ablate_trigger_sensory", "ablate_trigger_ws", "ablate_trigger_motor"]
    plot_bar_with_ci(
        band_labels,
        [rates.get(k, 0.0) for k in band_keys],
        [(rate_ci.get(k, [0, 0])[1] - rate_ci.get(k, [0, 0])[0]) / 2 for k in band_keys],
        ylabel="Loop rate",
        title="Trigger ablation by layer band (onset-timed)",
        out_path=out / "loop_rate_by_layer_band.png",
        color="coral",
    )
    timing_plot = ["baseline_rerun", "ablate_trigger_ws@-3", "ablate_trigger_ws@-1", "ablate_trigger_ws"]
    timing_plot = [c for c in timing_plot if c in rates]
    plot_bar_with_ci(
        timing_plot,
        [rates[c] for c in timing_plot],
        [(rate_ci[c][1] - rate_ci[c][0]) / 2 for c in timing_plot],
        ylabel="Loop rate",
        title="Timing sweep: trigger ablation onset offset",
        out_path=out / "loop_rate_timing_sweep.png",
        color="darkorange",
    )
    dose_plot = [c for c in ["baseline_rerun"] + dose_cond_names if c in rates]
    plot_bar_with_ci(
        dose_plot,
        [rates[c] for c in dose_plot],
        [(rate_ci[c][1] - rate_ci[c][0]) / 2 for c in dose_plot],
        ylabel="Loop rate",
        title="Loop rate vs ablation dose (# trigger directions)",
        out_path=out / "loop_rate_vs_ablation_strength.png",
        color="seagreen",
    )
    len_names = [c for c in conditions if c in lengths]
    plot_bar_with_ci(
        len_names,
        [lengths[c] for c in len_names],
        [(length_ci[c][1] - length_ci[c][0]) / 2 for c in len_names],
        ylabel="Mean generated tokens",
        title="Generation length by condition (bootstrap 95% CI)",
        out_path=out / "gen_length_by_condition.png",
        color="slategray",
    )
    gsm8k_conds = list(gsm8k_summary["accuracy"].keys())
    if gsm8k_conds:
        plot_bar_with_ci(
            gsm8k_conds,
            [gsm8k_summary["accuracy"][c] for c in gsm8k_conds],
            [(gsm8k_summary["wilson_ci"][c][1] - gsm8k_summary["wilson_ci"][c][0]) / 2 for c in gsm8k_conds],
            ylabel="GSM8K exact-match accuracy",
            title=f"GSM8K surgical eval (n={gsm8k_summary['n_problems']}, interventions from step {GSM8K_START_POS})",
            out_path=out / "gsm8k_accuracy_by_condition.png",
            color="mediumseagreen",
        )

    # ---- stats.json / findings.json ----
    onset_sources: dict[str, int] = {}
    n_loopers = 0
    for pid in {int(p["prompt_id"]) for p in prompts}:
        onset = per_prompt.get(pid, {}).get("_onset")
        if not onset:
            continue
        onset_sources[onset["source"]] = onset_sources.get(onset["source"], 0) + 1
        n_loopers += int(bool(onset["pass1_loop"]))

    stats = {
        "protocol": "exp3_fixed_protocol_v2",
        "n_prompts": len(prompts),
        "n_pass1_loopers": n_loopers,
        "n_stable_nonloop": len(prompts) - n_loopers,
        "onset_sources": onset_sources,
        "loop_rates": rates,
        "loop_rate_wilson_ci": rate_ci,
        "mean_gen_lengths": lengths,
        "gen_length_bootstrap_ci": length_ci,
        "contrasts_vs_baseline_rerun": contrasts,
        "timing_offsets": list(TIMING_OFFSETS),
        "dose_ns": list(DOSE_NS),
        "suppress_k": suppress_k,
        "gsm8k": gsm8k_summary,
        "power_note": power_note,
    }
    save_json(stats, out / "stats.json")

    r_base = rates.get("baseline_rerun", 0.0)
    r_trig = rates.get("ablate_trigger_ws", 0.0)
    r_rand = rates.get("ablate_random", 0.0)
    r_ctrl = rates.get("ablate_control_token", 0.0)
    r_sens = rates.get("ablate_trigger_sensory", 0.0)
    r_sup = rates.get("logit_suppress_trigger", 0.0)
    positive = (r_trig < r_base) and (r_trig < r_rand) and (r_trig < r_ctrl)
    negative = abs(r_trig - r_rand) < 0.05 and abs(r_trig - r_ctrl) < 0.05
    result_type = "positive" if positive else ("negative" if negative else "mixed")
    acc = gsm8k_summary["accuracy"]
    if "baseline_rerun" in acc and "ablate_trigger_ws" in acc:
        gsm8k_surgical = bool((acc["baseline_rerun"] - acc["ablate_trigger_ws"]) <= 0.10)
    else:
        gsm8k_surgical = None

    findings = {
        "experiment": "exp3_causal_intervention_fixed_protocol",
        "protocol_version": 2,
        "model_id": cfg.model_id,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_prompts": len(prompts),
        "n_pass1_loopers": n_loopers,
        "n_stable_nonloop": len(prompts) - n_loopers,
        "onset_sources": onset_sources,
        "prompt_source": "baseline looping_prompt_ids (two-pass, pass-1 fresh onset)",
        "workspace_layers": ws_layers,
        "sensory_layers": sensory,
        "motor_layers": motor,
        "conditions": list(conditions),
        "timing_offsets": list(TIMING_OFFSETS),
        "timing_n": len(timing_pids),
        "dose_ns": list(DOSE_NS),
        "dose_n": len(dose_pids),
        "suppress_k": suppress_k,
        "loop_rates": rates,
        "loop_rate_wilson_ci": rate_ci,
        "mean_gen_lengths": lengths,
        "gen_length_bootstrap_ci": length_ci,
        "contrasts_vs_baseline_rerun": contrasts,
        "timing_sweep_rates": {c: rates.get(c) for c in timing_cond_names},
        "dose_response_rates": {c: rates.get(c) for c in dose_cond_names},
        "gsm8k": gsm8k_summary,
        "power_note": power_note,
        "result_type": result_type,
        "gsm8k_surgical_within_10pct": gsm8k_surgical,
        "interpretation": (
            f"Primary contrast (onset-timed per-trace trigger ablation, all workspace layers): "
            f"loop rate {r_trig:.3f} vs baseline_rerun {r_base:.3f} "
            f"(random {r_rand:.3f}, control-token {r_ctrl:.3f}, sensory {r_sens:.3f}, "
            f"logit-suppression positive control {r_sup:.3f}). Result type: {result_type}. "
            "Both positive and negative results are publishable per PROJECT_SPEC §11.6."
        ),
        "caveats": [
            "Pass-1 labels pair all McNemar contrasts; stable_nonloop prompts contribute "
            "only c-side discordance (condition loops where pass-1 did not).",
            "Interventions for stable_nonloop prompts are keyed to the original exp2 trace "
            "onset when available, else the global top-1 trigger from position 0.",
            f"GSM8K interventions start at generation step {GSM8K_START_POS} (fixed neutral "
            "choice; no trace onset exists for eval questions).",
            f"Logit suppression window is JLENS_SUPPRESS_K={suppress_k} steps from onset.",
        ],
    }
    save_json(findings, out / "findings.json")

    status = f"""# Experiment 3 — Causal Interventions (fixed protocol v2)

## Design
Two-pass, per-trace onset-timed interventions. {len(prompts)} looping prompts;
{n_loopers} re-looped in pass 1, {len(prompts) - n_loopers} stable_nonloop.
Onset sources: {json.dumps(onset_sources)}

## Loop rates (with Wilson 95% CI)
{json.dumps({k: [round(rates[k], 4), [round(x, 4) for x in rate_ci[k]]] for k in rates}, indent=2)}

## Contrasts vs baseline_rerun (exact McNemar)
{json.dumps(contrasts, indent=2, default=str)}

## Mean generation lengths (bootstrap 95% CI)
{json.dumps({k: [round(lengths[k], 1), [round(x, 1) for x in length_ci[k]]] for k in lengths}, indent=2)}

## Timing sweep (subset n={len(timing_pids)})
{json.dumps({c: rates.get(c) for c in timing_cond_names}, indent=2)}

## Dose-response (subset n={len(dose_pids)})
{json.dumps({c: rates.get(c) for c in dose_cond_names}, indent=2)}

## GSM8K surgical eval
{json.dumps(gsm8k_summary, indent=2)}

## Power note
{json.dumps(power_note, indent=2, default=str)}

## Interpretation
- Result type: **{result_type}**
- {findings['interpretation']}

Both positive and negative results are publishable per PROJECT_SPEC §11.6.
"""
    write_status("05_exp3", status)
    append_results_summary("Experiment 3 — Causal Interventions (fixed protocol v2)", status)
    save_checkpoint(
        ckpt_name,
        {
            "status": "complete",
            "protocol": "exp3_fixed_protocol_v2",
            "model_id": cfg.model_id,
            "max_new_tokens": MAX_NEW_TOKENS,
            "per_prompt": per_prompt,
            "conditions": list(conditions),
            "legacy_conditions": legacy_conditions,
            "n_looping_prompts": len(prompts),
            "n_done_pairs": n_done_pairs(),
            "n_total_pairs": len(prompts) * len(conditions),
            "workspace_layers": ws_layers,
            "sensory_layers": sensory,
            "motor_layers": motor,
            "trigger_ids_used": trigger_ids[:1],
            "control_token_id": control_token_id,
            "timing_offsets": list(TIMING_OFFSETS),
            "dose_ns": list(DOSE_NS),
            "suppress_k": suppress_k,
            "loop_rates": rates,
            "result_type": result_type,
            "timing": timing,
            "dose": dose,
            "gsm8k": gsm8k,
        },
    )
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

