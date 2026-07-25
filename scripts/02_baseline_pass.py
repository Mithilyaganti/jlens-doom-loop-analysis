#!/usr/bin/env python
"""Unified 200-prompt baseline pass: generation + loop detection + Tier-1/2 cache.

Feeds trigger-token extraction and Experiment 2 from the same generations.
Saves audit file results/prompt_sample_ids.json and per-trace Exp2 caches.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("baseline_pass")

TOTAL_PROMPTS = int(os.environ.get("JLENS_BASELINE_PROMPTS", "200"))
SEED = int(os.environ.get("JLENS_SAMPLE_SEED", "42"))
MAX_NEW_TOKENS = int(os.environ.get("JLENS_MAX_NEW_TOKENS", "4000"))
TEMPERATURE = float(os.environ.get("JLENS_TEMPERATURE", "0.01"))
TIER3_MAX = int(os.environ.get("JLENS_TIER3_MAX", "5"))


def main() -> int:
    from jspace.vendor_bootstrap import ensure_jlens_importable

    ensure_jlens_importable()

    from jspace.loading import load_stack, clear_cuda
    from jspace.model_config import get_active_model, artifact_paths
    from jspace.prompts import get_or_create_prompt_sample, DEFAULT_SAMPLE_PATH
    from jspace.generation import (
        apply_chat_template,
        generate_with_tiered_cache,
        generate_greedy,
        loop_result_to_meta,
        save_trace,
    )
    from jspace.detection import detect_loop
    from jspace.analysis import ensure_dir, load_json, save_json, write_status
    from jspace.vllm_backend import make_generator
    import pandas as pd

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    out = ensure_dir(paths["results"])
    exp2_out = ensure_dir(paths["exp2"])
    looping_dir = ensure_dir(exp2_out / "looping")
    nonloop_dir = ensure_dir(exp2_out / "nonlooping")
    ckpt_path = out / "checkpoints" / f"baseline_pass_{cfg.slug}.json"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    band_path = paths["workspace_band"]
    ws_layers: list[int] = []
    if band_path.is_file():
        band = load_json(band_path)
        ws_layers = list(band.get("key_workspace_layers") or [])
    else:
        logger.warning(
            "No workspace band at %s — running generation+detection only "
            "(Tier-1/2 J-lens cache skipped until lens+band exist).",
            band_path,
        )

    sample_path = paths["prompt_sample"]
    prompts = get_or_create_prompt_sample(sample_path, total=TOTAL_PROMPTS, seed=SEED)
    logger.info(
        "Baseline pass model=%s backend_env=%s prompts=%d max_new=%d temp=%s",
        cfg.model_id,
        os.environ.get("JLENS_BACKEND", "auto"),
        len(prompts),
        MAX_NEW_TOKENS,
        TEMPERATURE,
    )

    done_ids: set[int] = set()
    n_loop = 0
    looping_prompt_ids: list[int] = []
    trigger_counter: Counter = Counter()
    trigger_examples: dict[int, str] = {}

    def _ingest_saved_trace(pid: int) -> None:
        nonlocal n_loop
        for d in (looping_dir, nonloop_dir):
            mp = d / f"p{pid}" / "meta.json"
            if mp.is_file():
                meta = json.loads(mp.read_text(encoding="utf-8"))
                if meta.get("is_loop"):
                    n_loop += 1
                    looping_prompt_ids.append(pid)
                    tid = meta.get("trigger_token_id")
                    if tid is not None:
                        trigger_counter[int(tid)] += 1
                        trigger_examples.setdefault(int(tid), meta.get("trigger_decoded") or "")
                return

    if ckpt_path.is_file():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        done_ids = {int(x) for x in ckpt.get("completed_prompt_ids", [])}
        for pid in sorted(done_ids):
            _ingest_saved_trace(pid)
        logger.info("Resuming baseline pass: %d/%d already done", len(done_ids), len(prompts))

    # vLLM first — then tokenizer-only stack (avoids loading HF LFM2 for layout)
    vllm_gen = make_generator(cfg.model_id)
    stack = load_stack(require_lens=False)
    if vllm_gen is not None:
        logger.info("Using vLLM backend dtype=%s", getattr(vllm_gen, "dtype", "?"))
        stack.generation_backend = "vllm"
    else:
        logger.info("Using HuggingFace generate backend (quantize=%s)", cfg.hf_quantize)

    gen_log_path = paths["trigger_log"]
    log_mode = "a" if done_ids else "w"
    n_attempt = len(done_ids)

    if not done_ids and gen_log_path.is_file():
        gen_log_path.unlink()
    with gen_log_path.open(log_mode, encoding="utf-8") as logf:
        for i, p in enumerate(prompts):
            pid = int(p["prompt_id"])
            if pid in done_ids:
                continue

            n_attempt += 1
            chat = apply_chat_template(stack.tokenizer, p["text"])
            try:
                if stack.lens is None or not ws_layers:
                    # Generation + detection only (no J-lens tier cache yet)
                    gen = generate_greedy(
                        stack,
                        chat,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        vllm_generator=vllm_gen,
                    )
                    loop = detect_loop(
                        gen["generated_text"],
                        token_ids=gen["generated_ids"],
                        tokenizer=stack.tokenizer,
                    )
                    trace = {**gen, "loop": loop, "readouts": None, "residuals_ws": None}
                else:
                    trace = generate_with_tiered_cache(
                        stack,
                        chat,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        workspace_layers=ws_layers,
                        top_k=10,
                        cache_tier3=False,
                        vllm_generator=vllm_gen,
                    )
            except Exception as e:
                logger.exception("gen failed prompt %s: %s", pid, e)
                clear_cuda()
                continue

            loop = trace["loop"]
            meta = loop_result_to_meta(loop, trace, prompt_id=pid)
            meta["source"] = p.get("source")
            meta["hf_source"] = p.get("hf_source")
            meta["antidoom_id"] = p.get("id")
            meta["model_id"] = cfg.model_id
            meta["backend"] = trace.get("backend", "hf" if vllm_gen is None else "vllm")
            dest = looping_dir / f"p{pid}" if loop.is_loop else nonloop_dir / f"p{pid}"
            save_trace(dest, trace, meta)

            row = {
                "prompt_id": pid,
                "row_index": p.get("row_index"),
                "antidoom_id": p.get("id"),
                "source": p.get("source"),
                "is_loop": loop.is_loop,
                "trigger_token_id": loop.trigger_token_id,
                "trigger_decoded": loop.trigger_decoded,
                "trigger_token_str": loop.trigger_token_str,
                "n_tokens": len(trace.get("generated_ids", [])),
                "snippet": loop.hit.snippet if loop.hit else None,
                "generated_preview": trace.get("generated_text", "")[:500],
            }
            logf.write(json.dumps(row, ensure_ascii=False) + "\n")
            logf.flush()

            if loop.is_loop:
                n_loop += 1
                looping_prompt_ids.append(pid)
                if loop.trigger_token_id is not None:
                    trigger_counter[loop.trigger_token_id] += 1
                    trigger_examples.setdefault(
                        loop.trigger_token_id, loop.trigger_decoded or ""
                    )

            done_ids.add(pid)
            ckpt_path.write_text(
                json.dumps({"completed_prompt_ids": sorted(done_ids)}, indent=2),
                encoding="utf-8",
            )

            if (i + 1) % 5 == 0:
                logger.info(
                    "baseline %d/%d loops=%d unique_triggers=%d",
                    len(done_ids),
                    len(prompts),
                    n_loop,
                    len(trigger_counter),
                )
            clear_cuda()

    # Tier-3 full residuals for up to 5 looping traces (HTML visualization)
    tier3_done = 0
    if looping_prompt_ids and TIER3_MAX > 0:
        by_snippet = []
        for pid in looping_prompt_ids:
            mp = looping_dir / f"p{pid}" / "meta.json"
            if mp.is_file():
                meta = json.loads(mp.read_text(encoding="utf-8"))
                by_snippet.append((len(meta.get("snippet") or ""), pid))
        by_snippet.sort(reverse=True)
        for _, pid in by_snippet[:TIER3_MAX]:
            p = next(x for x in prompts if int(x["prompt_id"]) == pid)
            chat = apply_chat_template(stack.tokenizer, p["text"])
            try:
                trace3 = generate_with_tiered_cache(
                    stack,
                    chat,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    workspace_layers=ws_layers,
                    top_k=10,
                    cache_tier3=True,
                    vllm_generator=vllm_gen,
                )
                dest = looping_dir / f"p{pid}"
                meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
                save_trace(dest, trace3, meta)
                tier3_done += 1
            except Exception as e:
                logger.warning("Tier-3 cache failed for p%s: %s", pid, e)
            clear_cuda()

    total_triggers = sum(trigger_counter.values())
    rows = []
    for tid, count in trigger_counter.most_common():
        dec = trigger_examples.get(tid) or stack.tokenizer.decode([tid])
        share = count / total_triggers if total_triggers else 0.0
        rows.append(
            {
                "count": count,
                "share": round(share, 6),
                "token": repr(dec),
                "token_id": tid,
                "decoded": dec,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = paths["trigger_csv"]
    if df.empty:
        df = pd.DataFrame(columns=["count", "share", "token", "token_id", "decoded"])
    df.to_csv(csv_path, index=False)

    rate = n_loop / len(prompts) if prompts else 0.0
    summary = {
        "dataset": "LiquidAI/antidoom-mix-v1.0",
        "model_id": cfg.model_id,
        "model_slug": cfg.slug,
        "backend": "vllm" if vllm_gen is not None else "hf",
        "vllm_dtype": getattr(vllm_gen, "dtype", None),
        "sample_file": str(sample_path),
        "n_prompts": len(prompts),
        "n_completed": len(done_ids),
        "n_loop": n_loop,
        "loop_rate": rate,
        "unique_triggers": len(trigger_counter),
        "looping_prompt_ids": sorted(looping_prompt_ids),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "tier3_traces_saved": tier3_done,
        "empirical_table_empty": bool(df.empty),
        "jlens_available": stack.lens is not None,
        "note": (
            "Public LiquidAI/LFM2-2.6B is NOT the blog's private early LFM2.5-2.6B. "
            "It is also NOT Antidoom-trained. Dynamic reasoning left enabled."
        ),
    }
    save_json(summary, paths["baseline_summary"])

    status = f"""# Baseline Pass — {cfg.display_name} (antidoom-mix reasoning subset)

- Model: `{cfg.model_id}` (slug={cfg.slug})
- Backend: {"vllm" if vllm_gen else "hf"}
- Dataset: LiquidAI/antidoom-mix-v1.0 (7 reasoning component sources only)
- Prompts: {len(prompts)} (seed={SEED}), completed: {len(done_ids)}
- Loops detected: {n_loop} ({rate:.1%})
- Unique trigger token IDs: {len(trigger_counter)}
- max_new_tokens: {MAX_NEW_TOKENS}
- J-lens available: {stack.lens is not None}
- Audit: `{sample_path}`
- Trigger table: `{csv_path}`
"""
    write_status("02_baseline_pass", status)
    clear_cuda()
    return 0 if n_loop > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
