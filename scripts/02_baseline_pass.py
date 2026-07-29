#!/usr/bin/env python
"""Unified 200-prompt baseline pass: generation + loop detection (+ Tier-1/2 if band+lens).

Feeds trigger-token extraction. Saves audit file results/prompt_sample_ids.json
and per-trace caches under results/exp2_{slug}/.

Resume-safe: checkpoint JSON + on-disk meta.json. Optional Drive sync via
JLENS_DRIVE_SYNC_DIR (see jspace.drive_sync).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("baseline_pass")

TOTAL_PROMPTS = int(os.environ.get("JLENS_BASELINE_PROMPTS", "200"))
SEED = int(os.environ.get("JLENS_SAMPLE_SEED", "42"))
MAX_NEW_TOKENS = int(os.environ.get("JLENS_MAX_NEW_TOKENS", "4000"))
TEMPERATURE = float(os.environ.get("JLENS_TEMPERATURE", "0.01"))
TIER3_MAX = int(os.environ.get("JLENS_TIER3_MAX", "0"))  # 0 = skip Tier-3 in baseline (A4 later)
SMOKE_ONLY = os.environ.get("JLENS_BASELINE_SMOKE", "0") == "1"
SMOKE_N = int(os.environ.get("JLENS_BASELINE_SMOKE_N", "1"))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    from jspace.vendor_bootstrap import ensure_jlens_importable

    ensure_jlens_importable()

    from jspace.loading import load_stack, clear_cuda
    from jspace.model_config import get_active_model, artifact_paths
    from jspace.prompts import get_or_create_prompt_sample
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
    from jspace.drive_sync import (
        sync_baseline_artifacts,
        restore_from_drive,
        sync_every,
        drive_sync_dir,
    )
    import pandas as pd

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    out = ensure_dir(paths["results"])
    exp2_out = ensure_dir(paths["exp2"])
    looping_dir = ensure_dir(exp2_out / "looping")
    nonloop_dir = ensure_dir(exp2_out / "nonlooping")
    ckpt_path = paths["baseline_checkpoint"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Restore Drive backup before reading local checkpoints (Colab resume)
    if drive_sync_dir() is not None and os.environ.get("JLENS_DRIVE_RESTORE", "1") == "1":
        restore_from_drive(cfg.slug)

    band_path = paths["workspace_band"]
    ws_layers: list[int] = []
    if band_path.is_file():
        band = load_json(band_path)
        ws_layers = list(band.get("key_workspace_layers") or [])
    else:
        logger.warning(
            "No workspace band at %s — generation+detection only "
            "(Tier-1/2 J-lens cache skipped until lens+band exist).",
            band_path,
        )

    sample_path = paths["prompt_sample"]
    prompts = get_or_create_prompt_sample(sample_path, total=TOTAL_PROMPTS, seed=SEED)
    if SMOKE_ONLY:
        prompts = prompts[: max(1, SMOKE_N)]
        logger.warning("SMOKE MODE: only %d prompt(s)", len(prompts))

    logger.info(
        "Baseline pass model=%s backend_env=%s prompts=%d max_new=%d temp=%s drive=%s",
        cfg.model_id,
        os.environ.get("JLENS_BACKEND", "auto"),
        len(prompts),
        MAX_NEW_TOKENS,
        TEMPERATURE,
        drive_sync_dir(),
    )

    done_ids: set[int] = set()
    failed_ids: set[int] = set()
    n_loop = 0
    looping_prompt_ids: list[int] = []
    trigger_counter: Counter = Counter()
    trigger_examples: dict[int, str] = {}

    def _ingest_meta(pid: int, meta: dict) -> None:
        nonlocal n_loop
        if meta.get("is_loop"):
            if pid not in looping_prompt_ids:
                n_loop += 1
                looping_prompt_ids.append(pid)
                tid = meta.get("trigger_token_id")
                if tid is not None:
                    trigger_counter[int(tid)] += 1
                    trigger_examples.setdefault(int(tid), meta.get("trigger_decoded") or "")

    def _scan_disk_completed() -> None:
        """Treat any on-disk meta.json as completed (resume if ckpt lost)."""
        for d in (looping_dir, nonloop_dir):
            if not d.is_dir():
                continue
            for meta_path in d.glob("p*/meta.json"):
                try:
                    pid = int(meta_path.parent.name.lstrip("p"))
                except ValueError:
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                done_ids.add(pid)
                _ingest_meta(pid, meta)

    def persist_ckpt(*, last_id: int | None = None, status: str = "running") -> None:
        payload = {
            "saved_at": _utc(),
            "status": status,
            "model_id": cfg.model_id,
            "model_slug": cfg.slug,
            "backend_env": os.environ.get("JLENS_BACKEND", "auto"),
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "n_total": len(prompts),
            "n_completed": len(done_ids),
            "n_failed": len(failed_ids),
            "n_loop": n_loop,
            "loop_rate_so_far": n_loop / max(len(done_ids), 1),
            "completed_prompt_ids": sorted(done_ids),
            "failed_prompt_ids": sorted(failed_ids),
            "looping_prompt_ids": sorted(looping_prompt_ids),
            "last_prompt_id": last_id,
            "vllm_dtype": os.environ.get("JLENS_VLLM_DTYPE", ""),
        }
        tmp = ckpt_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(ckpt_path)

    # Resume: disk first, then merge checkpoint ids
    _scan_disk_completed()
    if ckpt_path.is_file():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        for x in ckpt.get("completed_prompt_ids", []):
            done_ids.add(int(x))
        for x in ckpt.get("failed_prompt_ids", []):
            failed_ids.add(int(x))
        # Re-ingest from disk for accurate loop counts
        n_loop = 0
        looping_prompt_ids = []
        trigger_counter = Counter()
        trigger_examples = {}
        _scan_disk_completed()
        logger.info(
            "Resuming baseline: %d/%d done, loops=%d, failed=%d",
            len(done_ids),
            len(prompts),
            n_loop,
            len(failed_ids),
        )

    persist_ckpt(status="starting")
    if drive_sync_dir() is not None:
        sync_baseline_artifacts(cfg.slug, reason="start")

    vllm_gen = make_generator(cfg.model_id)
    if vllm_gen is None and os.environ.get("JLENS_BACKEND", "").lower() == "vllm":
        # vLLM requested but init failed (e.g. tokenizer API mismatch) → full HF load
        logger.warning(
            "vLLM unavailable after init — switching to HF bf16/fp16 "
            "(JLENS_HF_QUANTIZE=%s)",
            os.environ.get("JLENS_HF_QUANTIZE", "0"),
        )
        os.environ["JLENS_BACKEND"] = "hf"
        if "JLENS_HF_QUANTIZE" not in os.environ:
            os.environ["JLENS_HF_QUANTIZE"] = "0"
    stack = load_stack(require_lens=False)
    if vllm_gen is not None:
        logger.info("Using vLLM backend dtype=%s", getattr(vllm_gen, "dtype", "?"))
        stack.generation_backend = "vllm"
    else:
        logger.info("Using HuggingFace generate backend (quantize=%s)", cfg.hf_quantize)

    gen_log_path = paths["trigger_log"]
    # Append if resuming; rewrite only on fresh run
    log_mode = "a" if done_ids else "w"
    if not done_ids and gen_log_path.is_file():
        gen_log_path.unlink()

    since_sync = 0
    with gen_log_path.open(log_mode, encoding="utf-8") as logf:
        for i, p in enumerate(prompts):
            pid = int(p["prompt_id"])
            if pid in done_ids:
                continue

            chat = apply_chat_template(stack.tokenizer, p["text"])
            try:
                # A1: generation+detection. Tier cache only if band+lens on HF stack.
                use_tier = (
                    stack.lens is not None
                    and bool(ws_layers)
                    and vllm_gen is None
                    and stack.model is not None
                )
                if not use_tier:
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
                        vllm_generator=None,
                    )
            except Exception as e:
                logger.exception("gen failed prompt %s: %s", pid, e)
                failed_ids.add(pid)
                persist_ckpt(last_id=pid, status="running")
                if drive_sync_dir() is not None:
                    sync_baseline_artifacts(cfg.slug, reason=f"fail_p{pid}")
                clear_cuda()
                continue

            loop = trace["loop"]
            meta = loop_result_to_meta(loop, trace, prompt_id=pid)
            meta["source"] = p.get("source")
            meta["hf_source"] = p.get("hf_source")
            meta["antidoom_id"] = p.get("id")
            meta["model_id"] = cfg.model_id
            meta["backend"] = trace.get("backend", "hf" if vllm_gen is None else "vllm")
            meta["vllm_dtype"] = getattr(vllm_gen, "dtype", None)
            meta["max_new_tokens"] = MAX_NEW_TOKENS
            meta["temperature"] = TEMPERATURE
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
                "backend": meta["backend"],
                "vllm_dtype": meta.get("vllm_dtype"),
            }
            logf.write(json.dumps(row, ensure_ascii=False) + "\n")
            logf.flush()
            os.fsync(logf.fileno())

            if loop.is_loop:
                n_loop += 1
                looping_prompt_ids.append(pid)
                if loop.trigger_token_id is not None:
                    trigger_counter[loop.trigger_token_id] += 1
                    trigger_examples.setdefault(
                        loop.trigger_token_id, loop.trigger_decoded or ""
                    )

            done_ids.add(pid)
            failed_ids.discard(pid)
            persist_ckpt(last_id=pid, status="running")
            since_sync += 1

            logger.info(
                "baseline %d/%d pid=%s loop=%s tokens=%d loops=%d",
                len(done_ids),
                len(prompts),
                pid,
                bool(loop.is_loop),
                len(trace.get("generated_ids", [])),
                n_loop,
            )

            if drive_sync_dir() is not None and since_sync >= sync_every():
                sync_baseline_artifacts(
                    cfg.slug, reason=f"after_p{pid}_done{len(done_ids)}"
                )
                since_sync = 0

            clear_cuda()

    # Optional Tier-3 (default off for A1)
    tier3_done = 0
    if looping_prompt_ids and TIER3_MAX > 0 and stack.lens is not None and ws_layers:
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
                    vllm_generator=None,
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
        dec = trigger_examples.get(tid) or (
            stack.tokenizer.decode([tid]) if stack.tokenizer is not None else str(tid)
        )
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

    rate = n_loop / len(done_ids) if done_ids else 0.0
    summary = {
        "dataset": "LiquidAI/antidoom-mix-v1.0",
        "model_id": cfg.model_id,
        "model_slug": cfg.slug,
        "backend": "vllm" if vllm_gen is not None else "hf",
        "vllm_dtype": getattr(vllm_gen, "dtype", None),
        "sample_file": str(sample_path),
        "seed": SEED,
        "n_prompts": len(prompts),
        "n_completed": len(done_ids),
        "n_failed": len(failed_ids),
        "n_loop": n_loop,
        "loop_rate": rate,
        "unique_triggers": len(trigger_counter),
        "looping_prompt_ids": sorted(looping_prompt_ids),
        "failed_prompt_ids": sorted(failed_ids),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "tier3_traces_saved": tier3_done,
        "empirical_table_empty": bool(df.empty),
        "jlens_available": stack.lens is not None,
        "complete": len(done_ids) >= len(prompts),
        "saved_at": _utc(),
        "note": (
            f"Baseline A1 for {cfg.model_id}. Eval set = seed={SEED} stratified 200. "
            "Not nerfed: max_new_tokens and temperature as configured."
        ),
    }
    save_json(summary, paths["baseline_summary"])
    # Also write legacy unscoped name for older scripts (Qwen slug-scoped is canonical)
    if cfg.slug == "qwen3.5-4b":
        save_json(summary, out / "baseline_pass_summary.json")

    persist_ckpt(status="complete" if summary["complete"] else "partial")
    if drive_sync_dir() is not None:
        sync_baseline_artifacts(cfg.slug, reason="final")

    status = f"""# Baseline Pass — {cfg.display_name} (antidoom-mix reasoning subset)

- Model: `{cfg.model_id}` (slug={cfg.slug})
- Backend: {"vllm " + str(getattr(vllm_gen, "dtype", "")) if vllm_gen else "hf"}
- Dataset: LiquidAI/antidoom-mix-v1.0 (7 reasoning component sources only)
- Prompts: {len(prompts)} (seed={SEED}), completed: {len(done_ids)}, failed: {len(failed_ids)}
- Loops detected: {n_loop} ({rate:.1%} of completed)
- Unique trigger token IDs: {len(trigger_counter)}
- max_new_tokens: {MAX_NEW_TOKENS} (not reduced)
- J-lens available: {stack.lens is not None}
- Audit: `{sample_path}`
- Trigger table: `{csv_path}`
- Checkpoint: `{ckpt_path}`
- Drive sync: `{drive_sync_dir()}`
"""
    write_status("02_baseline_pass", status)
    clear_cuda()
    if not done_ids:
        return 1
    return 0 if summary["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
