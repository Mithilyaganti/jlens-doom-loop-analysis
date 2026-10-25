#!/usr/bin/env python
"""Experiment 2 — Dynamic J-space monitoring before loop onset."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp2")

MAX_PROMPTS = int(os.environ.get("JLENS_EXP2_PROMPTS", "200"))
MAX_NEW_TOKENS = int(os.environ.get("JLENS_MAX_NEW_TOKENS", "4000"))
# Default: analyze traces from unified baseline pass (no regeneration).
ANALYZE_ONLY = os.environ.get("JLENS_EXP2_ANALYZE_ONLY", "1") == "1"
# Generation-only: save hooked traces + checkpoint; skip plots/summary analysis.
GEN_ONLY = os.environ.get("JLENS_EXP2_GEN_ONLY", "0") == "1"
CKPT_NAME = os.environ.get("JLENS_EXP2_CKPT", "exp2_gen_lfm2-2.6b")
LOCK_PATH = ROOT / "results" / ".exp2_gen.lock"
N_BOOT = int(os.environ.get("JLENS_EXP2_N_BOOT", "2000"))
N_PERM = int(os.environ.get("JLENS_EXP2_N_PERM", "10000"))

# Pre-onset window (positions relative to the anchor).
POSITIONS = list(range(-5, 1))
# Window head-room an anchor needs so rel -5..0 exist.
MIN_ANCHOR_POS = 5


def load_exp2_prompts(max_n: int) -> list[dict]:
    """Load stratified antidoom-mix reasoning subset (same as baseline pass)."""
    from jspace.prompts import get_or_create_prompt_sample

    prompts = get_or_create_prompt_sample(total=max_n)
    logger.info("Exp2 prompt set size: %d", len(prompts))
    return prompts


def _trace_complete(dest: Path) -> bool:
    """A hooked Exp2 trace is complete when meta + readouts exist."""
    return (dest / "meta.json").is_file() and (dest / "readouts.pt").is_file()


def _find_existing_trace(prompt_id: Any, looping_dir: Path, nonloop_dir: Path) -> Path | None:
    for base in (looping_dir, nonloop_dir):
        dest = base / f"p{prompt_id}"
        if _trace_complete(dest):
            return dest
    return None


def _prioritize_prompts(prompts: list[dict], priority_ids: list[int]) -> list[dict]:
    """Run known looping IDs first, then the rest (stable order)."""
    pri = {int(x) for x in priority_ids}
    first = [p for p in prompts if int(p["prompt_id"]) in pri]
    rest = [p for p in prompts if int(p["prompt_id"]) not in pri]
    # Preserve priority_ids order for the first block.
    by_id = {int(p["prompt_id"]): p for p in first}
    ordered_first = [by_id[i] for i in priority_ids if i in by_id]
    return ordered_first + rest


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import psutil

        return psutil.pid_exists(pid)
    except Exception:
        pass
    # Fallback: Windows tasklist / POSIX kill(0)
    if os.name == "nt":
        import subprocess

        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock() -> None:
    if LOCK_PATH.is_file():
        try:
            old = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            old_pid = int(old.get("pid", -1))
        except Exception:
            old_pid = -1
        if _pid_alive(old_pid):
            raise RuntimeError(
                f"Exp2 generation already running (lock {LOCK_PATH}, pid={old_pid}). "
                "Stop the other process or delete the lock if stale."
            )
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _release_lock() -> None:
    try:
        if LOCK_PATH.is_file():
            LOCK_PATH.unlink()
    except OSError:
        pass


def load_saved_trace(trace_dir: Path, *, with_tensors: bool = True) -> dict | None:
    """Load a previously saved Exp2 trace from disk.

    ``with_tensors=False`` reads only meta.json (light pass); readouts /
    residuals / tokens are loaded lazily when needed.
    """
    meta_path = trace_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    trace: dict = {"meta": meta}
    if not with_tensors:
        return trace
    readouts_path = trace_dir / "readouts.pt"
    if readouts_path.is_file():
        trace["readouts"] = torch.load(readouts_path, map_location="cpu", weights_only=True)
    residuals_path = trace_dir / "residuals_workspace_band.pt"
    if residuals_path.is_file():
        trace["residuals_ws"] = torch.load(residuals_path, map_location="cpu", weights_only=True)
    tokens_path = trace_dir / "tokens.pt"
    if tokens_path.is_file():
        tok = torch.load(tokens_path, map_location="cpu", weights_only=True)
        trace["generated_ids"] = tok.get("generated_ids", [])
        trace["prompt_len"] = tok.get("prompt_len")
    else:
        trace["generated_ids"] = []
        trace["prompt_len"] = None
    trace["loop"] = _loopresult_from_meta(meta)
    # Prefer the readout-local index (after cache-window mapping). Fall back to
    # raw trigger index only when the sequence was not truncated.
    local = meta.get("trigger_local_in_readouts")
    if local is None and not (meta.get("trunc") or {}).get("truncated"):
        local = meta.get("trigger_token_index")
    trace["trigger_local_in_readouts"] = local
    return trace


def _loopresult_from_meta(meta: dict):
    from jspace.detection import LoopResult

    return LoopResult(
        is_loop=bool(meta.get("is_loop")),
        hit=None,
        trigger_token_index=meta.get("trigger_token_index"),
        trigger_token_id=meta.get("trigger_token_id"),
        trigger_token_str=meta.get("trigger_token_str"),
        trigger_decoded=meta.get("trigger_decoded"),
        start_char=meta.get("start_char"),
        repeat_start_char=meta.get("repeat_start_char"),
        end_char=meta.get("end_char"),
    )


def iter_saved_traces(looping_dir: Path, nonloop_dir: Path, *, with_tensors: bool = True):
    """Yield (trace, dir, is_loop) for every saved trace, one at a time.

    Generator on purpose: loading all readouts/residuals of 200 traces at once
    costs ~1 GB RAM; sequential yielding keeps the analysis pass memory-safe.
    """
    for d in sorted(looping_dir.glob("p*")):
        t = load_saved_trace(d, with_tensors=with_tensors)
        if t:
            yield t, d, True
    for d in sorted(nonloop_dir.glob("p*")):
        t = load_saved_trace(d, with_tensors=with_tensors)
        if t:
            yield t, d, False


def workspace_occupancy_from_readouts(indices: torch.Tensor, values: torch.Tensor, thresh_ratio: float = 0.1) -> float:
    """Proxy occupancy: count of top-k entries with score >= thresh_ratio * top1."""
    # indices/values: [top_k]
    if values.numel() == 0:
        return 0.0
    v = values.float()
    # convert logits-ish to relative
    top1 = float(v[0])
    # Use gap from min in topk
    thr = top1 - abs(top1) * (1 - thresh_ratio) if top1 != 0 else v.mean()
    # Simpler: softmax entropy of top-k as inverse occupancy collapse
    p = torch.softmax(v, dim=-1)
    ent = float(-(p * (p + 1e-12).log()).sum())
    max_ent = float(np.log(len(v)))
    # occupancy proxy: entropy / max_ent (1 = spread, 0 = collapsed)
    return ent / max_ent if max_ent > 0 else 0.0


def _has_saved_traces(base: Path) -> bool:
    for sub in ("looping", "nonlooping"):
        d = base / sub
        if d.is_dir() and any((x / "meta.json").is_file() for x in d.glob("p*")):
            return True
    return False


def _resolve_exp2_dir(paths: dict[str, Path]) -> Path:
    """Prefer the slug-scoped dir; fall back to the legacy results/exp2 tree
    that already holds the LFM2 hooked traces."""
    primary = Path(paths["exp2"])
    legacy = ROOT / "results" / "exp2"
    if _has_saved_traces(primary):
        return primary
    if primary != legacy and _has_saved_traces(legacy):
        logger.info("Using legacy Exp2 trace tree: %s", legacy)
        return legacy
    return primary


def _lens_only_stack(cfg, paths, out: Path):
    """Lens + tokenizer + lazy unembedding rows — NO model load (CPU-safe).

    Trigger-direction computations only need J_mid (from the lens file) and a
    few W_U rows, which are read lazily from the locally cached safetensors
    shards (or a previously saved row cache under the results tree).
    """
    from types import SimpleNamespace

    from jspace.loading import load_jacobian_lens, load_tokenizer_only
    from jspace.geometry import SafetensorsUnembedRows, hf_safetensors_unembed

    lens_path = Path(cfg.lens_path or paths["lens"])
    if not lens_path.is_file():
        raise FileNotFoundError(
            f"ANALYZE_ONLY needs the lens at {lens_path} (no model will be loaded)."
        )
    lens = load_jacobian_lens(lens_path)

    rows = hf_safetensors_unembed(cfg.model_id)
    row_cache_path = out / "unembed_rows_cache.pt"
    if rows is None:
        if row_cache_path.is_file():
            cached = torch.load(row_cache_path, map_location="cpu", weights_only=True)
            rows = SafetensorsUnembedRows.from_row_cache(cached)
            logger.info("Unembedding rows from cache file: %s", row_cache_path)
        else:
            raise RuntimeError(
                f"Cannot resolve W_U rows for {cfg.model_id} without the model: "
                "no local safetensors and no row cache. Run once with the model "
                "or place the safetensors in the HF cache."
            )
    try:
        tokenizer = load_tokenizer_only(cfg.model_id)
    except Exception:
        # Offline machines: everything needed is already in the local cache.
        os.environ["HF_HUB_OFFLINE"] = "1"
        tokenizer = load_tokenizer_only(cfg.model_id)
    stack = SimpleNamespace(
        lens=lens,
        lm_head=SimpleNamespace(weight=rows),
        tokenizer=tokenizer,
        model=None,
        device=torch.device("cpu"),
    )
    stack.unembed_rows = rows  # type: ignore[attr-defined]
    stack.unembed_row_cache_path = row_cache_path  # type: ignore[attr-defined]
    return stack


def _persist_unembed_row_cache(stack) -> None:
    rows = getattr(stack, "unembed_rows", None)
    path = getattr(stack, "unembed_row_cache_path", None)
    if rows is None or path is None:
        return
    try:
        payload = {int(k): v for k, v in rows.export_rows().items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        logger.info("Saved %d unembedding rows to %s", len(payload), path)
    except Exception as e:  # pragma: no cover - cache write is best-effort
        logger.warning("Could not persist unembed row cache: %s", e)


def _fallback_trigger(trace: dict, tokenizer) -> tuple[int | None, int | None]:
    """(trigger_token_id, trigger_token_index) — from meta, else re-run the
    CPU detector over the saved generated text/tokens."""
    loop = trace.get("loop")
    if loop is not None and loop.trigger_token_id is not None:
        return loop.trigger_token_id, loop.trigger_token_index
    meta = trace.get("meta", {})
    gen_ids = trace.get("generated_ids") or []
    text = meta.get("generated_text") or ""
    if not text or not gen_ids or tokenizer is None:
        return None, None
    from jspace.detection import detect_loop

    res = detect_loop(text, token_ids=list(gen_ids), tokenizer=tokenizer)
    if res.is_loop:
        return res.trigger_token_id, res.trigger_token_index
    return None, None


def _group_stat(vals: list[float], *, wilson: bool = False) -> dict[str, Any]:
    """Bootstrap 95% CI (mean over observations); Wilson CI for 0/1 metrics."""
    clean = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not clean:
        return {"n": 0, "mean": float("nan")}
    from jspace.stats import bootstrap_ci

    point, lo, hi = bootstrap_ci(clean, n_boot=N_BOOT, seed=7)
    out: dict[str, Any] = {
        "n": len(clean),
        "mean": float(point),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
    }
    if wilson:
        from jspace.stats import wilson_ci

        k = int(sum(1.0 for v in clean if v >= 0.5))
        wlo, whi = wilson_ci(k, len(clean))
        out["wilson95_lo"] = float(wlo)
        out["wilson95_hi"] = float(whi)
    return out


def _perm(a: list[float], b: list[float], seed: int) -> float:
    from jspace.stats import permutation_test

    if not a or not b:
        return float("nan")
    return float(permutation_test(a, b, n_perm=N_PERM, seed=seed))


def main() -> int:
    from jspace.model_config import get_active_model, artifact_paths
    from jspace.analysis import (
        ensure_dir,
        load_json,
        save_json,
        write_status,
        append_results_summary,
        first_occurrence_anchor,
        plot_bar_with_ci,
        position_matched_anchor,
        readout_gen_offset,
    )
    from jspace.geometry import jlens_direction

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    out = _resolve_exp2_dir(paths)
    out = ensure_dir(out)
    looping_dir = ensure_dir(out / "looping")
    nonloop_dir = ensure_dir(out / "nonlooping")

    if ANALYZE_ONLY and GEN_ONLY:
        raise ValueError("Set only one of JLENS_EXP2_ANALYZE_ONLY / JLENS_EXP2_GEN_ONLY")

    band_path = paths["workspace_band"]
    if not band_path.is_file():
        raise FileNotFoundError(
            f"Missing {band_path}. Run scripts/01_workspace_band.py first."
        )
    band = load_json(band_path)
    if not band.get("key_workspace_layers"):
        raise ValueError("workspace band file missing key_workspace_layers")
    ws_layers = list(band["key_workspace_layers"])
    mid = int(band["mid_workspace_layer"])

    prompts = load_exp2_prompts(MAX_PROMPTS) if not ANALYZE_ONLY else []
    logger.info(
        "Exp2: %d prompts (analyze_only=%s gen_only=%s), ws_layers=%s, max_new=%d",
        len(prompts),
        ANALYZE_ONLY,
        GEN_ONLY,
        ws_layers,
        MAX_NEW_TOKENS,
    )

    stack = None
    if ANALYZE_ONLY:
        # Analysis needs only lens + tokenizer + a handful of W_U rows.
        stack = _lens_only_stack(cfg, paths, out)
        saved_meta = list(iter_saved_traces(looping_dir, nonloop_dir, with_tensors=False))
        if not saved_meta:
            raise RuntimeError("JLENS_EXP2_ANALYZE_ONLY=1 but no saved traces found.")
        logger.info("Analyzing %d saved traces (model NOT loaded)", len(saved_meta))
    else:
        from jspace.loading import load_stack

        stack = load_stack()

    # ==================================================================
    # Generation pass (skipped entirely in ANALYZE_ONLY)
    # ==================================================================
    if not ANALYZE_ONLY:
        from jspace.checkpoint import save_checkpoint, load_checkpoint
        from jspace.loading import clear_cuda
        from jspace.generation import (
            apply_chat_template,
            generate_with_tiered_cache,
            loop_result_to_meta,
            save_trace,
        )

        if stack.lens is None:
            raise RuntimeError(
                "Exp2 hooked generation requires lenses/lfm2-2.6b.pt — lens not loaded."
            )

        _acquire_lock()
        try:
            baseline_summary = {}
            bs_path = paths.get("baseline_summary")
            if bs_path and Path(bs_path).is_file():
                baseline_summary = load_json(bs_path)
            priority_ids = [int(x) for x in baseline_summary.get("looping_prompt_ids") or []]
            prompts = _prioritize_prompts(prompts, priority_ids)
            logger.info(
                "Priority looping IDs first (%d): %s",
                len(priority_ids),
                priority_ids,
            )

            ckpt = load_checkpoint(CKPT_NAME) or {
                "completed_ids": [],
                "failed_ids": [],
                "n_loop": 0,
                "n_non": 0,
                "model_id": cfg.model_id,
                "max_new_tokens": MAX_NEW_TOKENS,
                "workspace_layers": ws_layers,
            }
            completed = {int(x) for x in ckpt.get("completed_ids") or []}
            failed = {int(x) for x in ckpt.get("failed_ids") or []}

            # Also treat on-disk complete traces as done (resume-safe).
            for p in prompts:
                pid = int(p["prompt_id"])
                existing = _find_existing_trace(pid, looping_dir, nonloop_dir)
                if existing is not None:
                    completed.add(pid)

            def persist_ckpt(*, last_id: int | None = None, status: str = "running") -> None:
                save_checkpoint(
                    CKPT_NAME,
                    {
                        "status": status,
                        "model_id": cfg.model_id,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "workspace_layers": ws_layers,
                        "n_total": len(prompts),
                        "n_completed": len(completed),
                        "n_failed": len(failed),
                        "n_loop": int(ckpt.get("n_loop", 0)),
                        "n_non": int(ckpt.get("n_non", 0)),
                        "completed_ids": sorted(completed),
                        "failed_ids": sorted(failed),
                        "last_prompt_id": last_id,
                        "priority_ids": priority_ids,
                    },
                )

            persist_ckpt(status="running")
            todo = [p for p in prompts if int(p["prompt_id"]) not in completed]
            logger.info(
                "Resume: %d already complete, %d remaining of %d",
                len(completed),
                len(todo),
                len(prompts),
            )

            for i, p in enumerate(todo):
                pid = int(p["prompt_id"])
                chat = apply_chat_template(stack.tokenizer, p["text"])
                logger.info(
                    "Generating hooked trace %d/%d prompt_id=%s (done=%d loops=%s non=%s)",
                    i + 1,
                    len(todo),
                    pid,
                    len(completed),
                    ckpt.get("n_loop"),
                    ckpt.get("n_non"),
                )
                try:
                    trace = generate_with_tiered_cache(
                        stack,
                        chat,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=0.01,
                        workspace_layers=ws_layers,
                        top_k=10,
                        cache_tier3=False,
                    )
                except Exception as e:
                    logger.exception("prompt %s failed: %s", pid, e)
                    failed.add(pid)
                    persist_ckpt(last_id=pid, status="running")
                    clear_cuda()
                    continue

                loop = trace["loop"]
                meta = loop_result_to_meta(loop, trace, prompt_id=pid)
                meta["source"] = p.get("source")
                meta["hf_source"] = p.get("hf_source")
                meta["antidoom_id"] = p.get("antidoom_id")
                meta["model_id"] = cfg.model_id
                meta["backend"] = os.environ.get("JLENS_BACKEND", "hf")
                meta["jlens_hooks"] = True
                meta["workspace_layers"] = ws_layers
                if trace.get("readouts") is None:
                    logger.error("prompt %s produced no readouts — not marking complete", pid)
                    failed.add(pid)
                    persist_ckpt(last_id=pid, status="running")
                    clear_cuda()
                    continue

                dest = looping_dir / f"p{pid}" if loop.is_loop else nonloop_dir / f"p{pid}"
                # Remove stale counterpart folder if loop label flipped on re-run.
                other = nonloop_dir / f"p{pid}" if loop.is_loop else looping_dir / f"p{pid}"
                if other.exists() and other != dest:
                    import shutil

                    shutil.rmtree(other, ignore_errors=True)
                save_trace(dest, trace, meta)

                completed.add(pid)
                failed.discard(pid)
                if loop.is_loop:
                    ckpt["n_loop"] = int(ckpt.get("n_loop", 0)) + 1
                else:
                    ckpt["n_non"] = int(ckpt.get("n_non", 0)) + 1
                persist_ckpt(last_id=pid, status="running")
                clear_cuda()

            persist_ckpt(status="complete" if len(completed) >= len(prompts) else "partial")
            logger.info(
                "Generation pass finished: completed=%d/%d failed=%d",
                len(completed),
                len(prompts),
                len(failed),
            )
            # Free the generative model before the disk-based analysis pass.
            del stack
            from jspace.loading import clear_cuda

            clear_cuda()
            stack = _lens_only_stack(cfg, paths, out)
        finally:
            _release_lock()

    if GEN_ONLY:
        # Generation-only night run: skip analysis/plots; write a short status.
        ckpt = None
        try:
            from jspace.checkpoint import load_checkpoint

            ckpt = load_checkpoint(CKPT_NAME)
        except Exception:
            ckpt = None
        n_done = len((ckpt or {}).get("completed_ids") or [])
        status = f"""# Experiment 2 — Generation (in progress / checkpointed)

- Mode: **GEN_ONLY** (analysis deferred)
- Checkpoint: `results/checkpoints/{CKPT_NAME}.json`
- Completed hooked traces: **{n_done}** / {MAX_PROMPTS}
- max_new_tokens: {MAX_NEW_TOKENS}
- Workspace layers (Tier-2): {ws_layers}
- Status: {(ckpt or {}).get('status')}
- Loops so far: {(ckpt or {}).get('n_loop')} | Non-loops: {(ckpt or {}).get('n_non')}

Resume later with the same env flags; completed prompt IDs are skipped.
"""
        write_status("04_exp2", status)
        from jspace.loading import clear_cuda

        clear_cuda()
        return 0

    # ==================================================================
    # Disk-based analysis pass (ANALYZE_ONLY default path; also runs after
    # a generation pass). Uses lens + tokenizer + lazy W_U rows only.
    # ==================================================================
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Pass 1: light scan of every trace --------------------------
    loop_infos: list[dict[str, Any]] = []
    non_infos: list[dict[str, Any]] = []
    # prompt_id -> mid-layer topk rows for the whole cached window.
    mid_top: dict[Any, dict[str, np.ndarray]] = {}
    n_traces_seen = 0

    for i, (trace, dest, is_loop) in enumerate(iter_saved_traces(looping_dir, nonloop_dir)):
        n_traces_seen += 1
        meta = trace["meta"]
        readouts = trace.get("readouts")
        gen_ids = trace.get("generated_ids") or []
        prompt_len = trace.get("prompt_len")
        pid = meta.get("prompt_id")
        info: dict[str, Any] = {"prompt_id": pid, "dest": dest, "meta": meta}
        n_gp = int(readouts["indices"].shape[1]) if readouts is not None else 0
        info["n_gp"] = n_gp
        info["gen_len"] = len(gen_ids)
        info["gen_ids"] = np.asarray(gen_ids, dtype=np.int64) if gen_ids else np.zeros(0, dtype=np.int64)
        info["offset"] = (
            readout_gen_offset(meta, prompt_len, len(gen_ids), n_gp) if n_gp else None
        )

        if readouts is not None and mid < readouts["indices"].shape[0] and n_gp:
            mid_top[pid] = {
                "idx": readouts["indices"][mid].numpy(),
                "vals": readouts["values"][mid].numpy(),
            }

        if is_loop:
            loop = trace["loop"]
            trig_id, trig_idx_gen = _fallback_trigger(trace, stack.tokenizer)
            local = trace.get("trigger_local_in_readouts")
            if local is None:
                local = trig_idx_gen
            info["trigger_id"] = trig_id if trig_id is not None else loop.trigger_token_id
            info["trigger_decoded"] = meta.get("trigger_decoded")
            info["anchor"] = int(local) if local is not None else None
            # Normalized generation depth of the trigger (0=first, 1=last token).
            if trig_idx_gen is not None and info["gen_len"] > 1:
                info["depth"] = float(trig_idx_gen) / float(info["gen_len"] - 1)
            elif info["anchor"] is not None and n_gp > 1:
                info["depth"] = float(info["anchor"]) / float(n_gp - 1)
            else:
                info["depth"] = None
            loop_infos.append(info)
        else:
            # Legacy mid-sequence anchor (kept verbatim for legacy numbers).
            info["anchor_legacy"] = min(n_gp - 1, max(5, n_gp // 2)) if n_gp else None
            non_infos.append(info)

        del trace, readouts
        if (i + 1) % 25 == 0:
            logger.info("scan %d traces: loops=%d non=%d", i + 1, len(loop_infos), len(non_infos))

    n_loop = len(loop_infos)
    n_non = len(non_infos)
    logger.info("Scanned %d traces: %d looping, %d non-looping", n_traces_seen, n_loop, n_non)
    if not loop_infos:
        logger.warning("No looping traces on disk — matched-control analysis will be empty.")

    # ---- Aggregates --------------------------------------------------
    def occ_at(pid: Any, pos: int) -> float | None:
        rows = mid_top.get(pid)
        if rows is None or not (0 <= pos < rows["idx"].shape[0]):
            return None
        return workspace_occupancy_from_readouts(
            torch.from_numpy(rows["idx"][pos]), torch.from_numpy(rows["vals"][pos])
        )

    def top_ids_at(pid: Any, pos: int, k: int = 10) -> list[int] | None:
        rows = mid_top.get(pid)
        if rows is None or not (0 <= pos < rows["idx"].shape[0]):
            return None
        return [int(x) for x in rows["idx"][pos][:k]]

    # Legacy series (exact reproduction of the previous analysis).
    loop_occ = {p: [] for p in POSITIONS}
    non_occ = {p: [] for p in POSITIONS}
    loop_top1 = {p: [] for p in POSITIONS}
    loop_top3 = {p: [] for p in POSITIONS}
    loop_top10 = {p: [] for p in POSITIONS}
    non_top1 = {p: [] for p in POSITIONS}
    non_top3 = {p: [] for p in POSITIONS}
    non_top10 = {p: [] for p in POSITIONS}
    loop_align: list[float] = []
    non_align: list[float] = []

    # Matched-control accumulators.
    same_occ: list[float] = []          # occupancy at rel -2
    same_top1: list[float] = []         # top1 == trigger at rel -1
    same_align: list[float] = []        # cos(h_{-1}, dir(trigger)) at rel -1
    posm_occ: list[float] = []
    posm_top1: list[float] = []
    posm_align: list[float] = []
    per_trigger: dict[str, dict[str, Any]] = {}
    n_valid_mapping = sum(1 for nj in non_infos if nj["offset"] is not None)

    # Alignment tasks resolved in pass 2 (residual loads are the heavy part).
    # dest -> list of {"group", "pos", "token_id", "bucket"}
    align_tasks: dict[Path, list[dict[str, Any]]] = {}

    def add_align_task(dest: Path, group: str, pos: int, token_id: int, bucket: list[float]) -> None:
        if token_id is None or pos is None:
            return
        align_tasks.setdefault(dest, []).append(
            {"group": group, "pos": int(pos), "token_id": int(token_id), "bucket": bucket}
        )

    # Loop-trace legacy metrics + alignment task.
    for li in loop_infos:
        anchor = li["anchor"]
        n_gp = li["n_gp"]
        if anchor is None or not (0 <= anchor < n_gp):
            continue
        for rel in POSITIONS:
            pos = anchor + rel
            occ = occ_at(li["prompt_id"], pos)
            if occ is None:
                continue
            loop_occ[rel].append(occ)
            ids = top_ids_at(li["prompt_id"], pos)
            if ids is not None and li["trigger_id"] is not None:
                loop_top1[rel].append(1.0 if li["trigger_id"] == ids[0] else 0.0)
                loop_top3[rel].append(1.0 if li["trigger_id"] in ids[:3] else 0.0)
                loop_top10[rel].append(1.0 if li["trigger_id"] in ids[:10] else 0.0)
        if li["trigger_id"] is not None:
            add_align_task(li["dest"], "loop", anchor - 1, li["trigger_id"], loop_align)

    # Non-loop legacy metrics + alignment task (legacy mid anchor).
    for nj in non_infos:
        anchor = nj["anchor_legacy"]
        if anchor is None:
            continue
        gen_ids = nj["gen_ids"]
        for rel in POSITIONS:
            pos = anchor + rel
            occ = occ_at(nj["prompt_id"], pos)
            if occ is None:
                continue
            non_occ[rel].append(occ)
            ids = top_ids_at(nj["prompt_id"], pos)
            next_id = int(gen_ids[min(pos, len(gen_ids) - 1)]) if len(gen_ids) else None
            if ids is not None and next_id is not None:
                non_top1[rel].append(1.0 if next_id == ids[0] else 0.0)
                non_top3[rel].append(1.0 if next_id in ids[:3] else 0.0)
                non_top10[rel].append(1.0 if next_id in ids[:10] else 0.0)
        if len(gen_ids) and nj["n_gp"] > 1:
            nid = int(gen_ids[min(anchor, len(gen_ids) - 1)])
            add_align_task(nj["dest"], "non_legacy", anchor - 1, nid, non_align)

    # Matched controls: per loop trace k, over all non-loop traces j.
    for k, li in enumerate(loop_infos):
        t_k = li["trigger_id"]
        d_k = li["depth"]
        key = f"{li['prompt_id']}|tid={t_k}"
        entry: dict[str, Any] = {
            "prompt_id": li["prompt_id"],
            "trigger_token_id": t_k,
            "trigger_decoded": li["trigger_decoded"],
            "anchor": li["anchor"],
            "normalized_depth": d_k,
            "same_token_anchor_n": 0,
            "same_token_traces_with_token": 0,
            "position_matched_n": 0,
        }
        if t_k is not None:
            for nj in non_infos:
                if nj["offset"] is None or not nj["n_gp"]:
                    continue
                a = first_occurrence_anchor(nj["gen_ids"], t_k, nj["offset"], nj["n_gp"])
                if a is None:
                    continue
                entry["same_token_traces_with_token"] += 1
                if a < MIN_ANCHOR_POS:
                    continue
                occ = occ_at(nj["prompt_id"], a - 2)
                ids = top_ids_at(nj["prompt_id"], a - 1)
                if occ is not None:
                    same_occ.append(occ)
                    entry["same_token_anchor_n"] += 1
                if ids is not None:
                    same_top1.append(1.0 if t_k == ids[0] else 0.0)
                add_align_task(nj["dest"], "same_token", a - 1, t_k, same_align)
        if d_k is not None:
            for nj in non_infos:
                if nj["offset"] is None or not nj["n_gp"]:
                    continue
                a = position_matched_anchor(d_k, nj["n_gp"], min_pos=MIN_ANCHOR_POS)
                if a is None:
                    continue
                emitted = nj["gen_ids"][a + nj["offset"]] if a + nj["offset"] < len(nj["gen_ids"]) else None
                if emitted is None:
                    continue
                occ = occ_at(nj["prompt_id"], a - 2)
                ids = top_ids_at(nj["prompt_id"], a - 1)
                if occ is not None:
                    posm_occ.append(occ)
                if ids is not None:
                    posm_top1.append(1.0 if int(emitted) == ids[0] else 0.0)
                    entry["position_matched_n"] += 1
                add_align_task(nj["dest"], "pos_matched", a - 1, int(emitted), posm_align)
        per_trigger[key] = entry

    # ---- Pass 2: resolve alignment tasks (load residuals per trace) ----
    dir_cache: dict[int, torch.Tensor] = {}

    def direction(tid: int) -> torch.Tensor:
        v = dir_cache.get(tid)
        if v is None:
            v = jlens_direction(stack, int(tid), mid, normalize=True)
            dir_cache[tid] = v
        return v

    n_align_resolved = 0
    for dest, tasks in align_tasks.items():
        res_path = dest / "residuals_workspace_band.pt"
        if not res_path.is_file():
            continue
        try:
            residuals = torch.load(res_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.warning("Could not read %s: %s", res_path, e)
            continue
        if mid not in residuals:
            continue
        R = residuals[mid].float()
        for task in tasks:
            pos = task["pos"]
            if not (0 <= pos < R.shape[0]):
                continue
            h = R[pos]
            v = direction(task["token_id"])
            cos = float(torch.nn.functional.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item())
            task["bucket"].append(cos)
            n_align_resolved += 1
        del residuals, R

    logger.info(
        "Resolved %d alignment values (%d directions, %d traces with tasks)",
        n_align_resolved, len(dir_cache), len(align_tasks),
    )
    _persist_unembed_row_cache(stack)

    # ---- jspace.stats comparisons ------------------------------------
    metrics: dict[str, dict[str, Any]] = {}

    def metric_entry(
        name: str,
        loop_vals: list[float],
        same_vals: list[float],
        posm_vals: list[float],
        legacy_vals: list[float],
        *,
        binary: bool = False,
    ) -> None:
        metrics[name] = {
            "loop": _group_stat(loop_vals, wilson=binary),
            "same_token_anchor": _group_stat(same_vals, wilson=binary),
            "position_matched_anchor": _group_stat(posm_vals, wilson=binary),
            "legacy_mid_anchor_nonloop": _group_stat(legacy_vals, wilson=binary),
            "perm_p_loop_vs_same_token": _perm(loop_vals, same_vals, seed=11),
            "perm_p_loop_vs_position_matched": _perm(loop_vals, posm_vals, seed=12),
            "perm_p_loop_vs_legacy_mid": _perm(loop_vals, legacy_vals, seed=13),
        }

    metric_entry("occupancy_at_m2", loop_occ[-2], same_occ, posm_occ, non_occ[-2])
    metric_entry("top1_trigger_at_m1", loop_top1[-1], same_top1, posm_top1, non_top1[-1], binary=True)
    metric_entry("align_trigger_dir_at_m1", loop_align, same_align, posm_align, non_align)

    matched = {
        "description": (
            "Matched-control comparisons for the three headline pre-onset metrics. "
            "same_token_anchor: non-looping traces anchored at the first emitted "
            "occurrence of each looping trace's trigger token (window [a-5..a]). "
            "position_matched_anchor: non-looping traces anchored at the looping "
            "traces' mean-normalized trigger depth. legacy_mid_anchor_nonloop: the "
            "original arbitrary mid-sequence anchor (kept for continuity)."
        ),
        "loop_triggers": [
            {
                "prompt_id": li["prompt_id"],
                "trigger_token_id": li["trigger_id"],
                "trigger_decoded": li["trigger_decoded"],
                "anchor_in_readouts": li["anchor"],
                "normalized_depth": li["depth"],
            }
            for li in loop_infos
        ],
        "n_loop": n_loop,
        "n_nonloop": n_non,
        "n_nonloop_with_valid_gen_mapping": n_valid_mapping,
        "min_anchor_pos": MIN_ANCHOR_POS,
        "metrics": metrics,
        "per_trigger": per_trigger,
        "notes": [
            "Loop set is small (n=%d for LFM2) — permutation p-values are "
            "resolution-limited; treat as descriptive until more loops are collected." % n_loop,
            "same-token/position-matched anchors require a valid generated-token <-> "
            "readout-position mapping; traces with ambiguous legacy caches are skipped "
            "(n_valid=%d of %d)." % (n_valid_mapping, n_non),
            "Trigger directions use exact bf16 W_U rows read lazily from local "
            "safetensors (ANALYZE_ONLY does not load the quantized model; the "
            "generation-time analysis dequantized NF4 weights, so legacy alignment "
            "numbers may drift in the 3rd decimal).",
        ],
        "stats_settings": {"bootstrap_n": N_BOOT, "permutation_n": N_PERM},
    }

    # ---- Plots --------------------------------------------------------
    def means_cis(d, seed_offset=0):
        from jspace.stats import bootstrap_ci

        xs, ms, lo, hi = [], [], [], []
        for p in POSITIONS:
            arr = np.asarray(d[p], dtype=float)
            xs.append(p)
            if len(arr) == 0:
                ms.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
            else:
                m, l, h = bootstrap_ci(arr, n_boot=N_BOOT, seed=21 + seed_offset + p)
                ms.append(m)
                lo.append(l)
                hi.append(h)
        return xs, ms, lo, hi

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, d, color in [("looping", loop_occ, "C3"), ("non-looping", non_occ, "C0")]:
        xs, ms, lo, hi = means_cis(d)
        ax.plot(xs, ms, marker="o", label=label, color=color)
        ax.fill_between(xs, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("Position relative to trigger/anchor")
    ax.set_ylabel("Top-k entropy occupancy (1=spread)")
    ax.set_title("Workspace occupancy pre-onset (bootstrap 95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "workspace_occupancy_pre_onset.png", dpi=150)
    fig.savefig(out / "workspace_occupancy_pre_onset.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, d, style in [
        ("loop top-1", loop_top1, "-"),
        ("loop top-3", loop_top3, "--"),
        ("loop top-10", loop_top10, ":"),
        ("non top-1", non_top1, "-"),
        ("non top-3", non_top3, "--"),
    ]:
        xs, ms, _, _ = means_cis(d)
        ax.plot(xs, ms, marker="o", label=name, linestyle=style)
    ax.set_xlabel("Position relative to trigger/anchor")
    ax.set_ylabel("Fraction trigger/next in top-k readout")
    ax.set_title("Top-k J-lens readout convergence pre-onset")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "top1_readout_convergence.png", dpi=150)
    fig.savefig(out / "top1_readout_convergence.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    data = [loop_align, non_align]
    ax.boxplot(data, tick_labels=["looping", "non-looping"])
    ax.set_ylabel("cos(h_{ℓ,-1}, trigger_dir)")
    ax.set_title("Trigger-direction alignment at position −1")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "trigger_alignment_pre_onset.png", dpi=150)
    fig.savefig(out / "trigger_alignment_pre_onset.svg")
    plt.close(fig)

    # Matched-control headline bars.
    bar_labels = [
        "loop\nocc@−2", "same-token\nocc@−2", "pos-matched\nocc@−2", "legacy\nocc@−2",
        "loop\ntop1@−1", "same-token\ntop1@−1", "pos-matched\ntop1@−1", "legacy\ntop1@−1",
        "loop\nalign@−1", "same-token\nalign@−1", "pos-matched\nalign@−1", "legacy\nalign@−1",
    ]
    bar_means, bar_err = [], []
    for mname in ("occupancy_at_m2", "top1_trigger_at_m1", "align_trigger_dir_at_m1"):
        for gname in ("loop", "same_token_anchor", "position_matched_anchor", "legacy_mid_anchor_nonloop"):
            g = metrics[mname][gname]
            mean = g.get("mean", float("nan"))
            err = (
                max(g.get("ci95_hi", mean) - mean, mean - g.get("ci95_lo", mean))
                if g.get("n")
                else 0.0
            )
            if not math.isnan(mean) and not math.isfinite(err):
                err = 0.0
            bar_means.append(0.0 if math.isnan(mean) else mean)
            bar_err.append(err)
    plot_bar_with_ci(
        bar_labels,
        bar_means,
        bar_err,
        ylabel="Metric value (bootstrap 95% CI)",
        title="Headline pre-onset metrics: loop vs matched controls",
        out_path=out / "matched_controls_headline.png",
    )

    # HTML example for best looping trace
    html_path = out / "example_looping_trace.html"
    interesting_loops = [(li["meta"], li["dest"]) for li in loop_infos]
    if interesting_loops:
        interesting_loops.sort(key=lambda x: len(x[0].get("snippet") or ""), reverse=True)
        meta0, dest0 = interesting_loops[0]
        gen_text = meta0.get("generated_text", "")
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Looping trace example</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; }}
.loop {{ background: #ffe0e0; }}
.meta {{ background: #f4f4f4; padding: 1rem; border-radius: 8px; }}
pre {{ white-space: pre-wrap; word-break: break-word; }}
</style></head><body>
<h1>Example looping trace</h1>
<div class="meta">
<p><b>prompt_id</b>: {meta0.get('prompt_id')}</p>
<p><b>trigger</b>: {meta0.get('trigger_decoded')!r} (id={meta0.get('trigger_token_id')})</p>
<p><b>period</b>: {meta0.get('period')} chars, repeats={meta0.get('repeats')}</p>
<p><b>snippet</b>: {meta0.get('snippet')!r}</p>
</div>
<h2>Generated text</h2>
<pre>{_html_escape(gen_text)}</pre>
<p>Full residual Tier-3 visualization can be regenerated with JLENS_TIER3=1 for selected traces.</p>
</body></html>"""
        html_path.write_text(html, encoding="utf-8")
    else:
        html_path.write_text("<html><body><p>No looping traces collected.</p></body></html>", encoding="utf-8")

    n_traces = n_loop + n_non

    # ---- findings.json (merge: keep legacy keys, add new ones) --------
    def mean_or_none(x):
        return float(np.mean(x)) if x else None

    mean_loop_align = mean_or_none(loop_align)
    mean_non_align = mean_or_none(non_align)
    loop_top1_m1 = mean_or_none(loop_top1[-1])
    non_top1_m1 = mean_or_none(non_top1[-1])
    loop_occ_m2 = mean_or_none(loop_occ[-2])
    non_occ_m2 = mean_or_none(non_occ[-2])

    findings_new: dict[str, Any] = {
        "experiment": "exp2_dynamic_monitoring",
        "model_id": cfg.model_id,
        "n_prompts": len(prompts) if prompts else n_traces,
        "n_loop": n_loop,
        "n_non": n_non,
        "loop_rate": n_loop / max(n_traces, 1),
        "workspace_layers": ws_layers,
        "mean_loop_align": mean_loop_align,
        "mean_non_align": mean_non_align,
        "align_delta_loop_minus_non": (
            mean_loop_align - mean_non_align
            if mean_loop_align is not None and mean_non_align is not None
            else None
        ),
        "loop_top1_at_m1": loop_top1_m1,
        "non_top1_at_m1": non_top1_m1,
        "loop_occ_at_m2": loop_occ_m2,
        "non_occ_at_m2": non_occ_m2,
        "success_criterion_met": bool(
            (loop_top1_m1 or 0.0) > (non_top1_m1 or 0.0)
            or (mean_loop_align or 0.0) > (mean_non_align or 0.0)
        ),
        "headline_stats": {
            "occupancy_at_m2": metrics["occupancy_at_m2"],
            "top1_trigger_at_m1": metrics["top1_trigger_at_m1"],
            "align_trigger_dir_at_m1": metrics["align_trigger_dir_at_m1"],
        },
        "matched_control_analysis": matched,
        "analyze_without_model": True,
    }

    findings_path = out / "findings.json"
    findings: dict[str, Any] = dict(findings_new)
    if findings_path.is_file():
        try:
            prev = load_json(findings_path)
            if isinstance(prev, dict):
                findings = {**prev, **findings_new}
        except Exception as e:
            logger.warning("Could not merge previous findings.json: %s", e)
    save_json(findings, findings_path)

    # ---- summary.json + pre_onset_series.csv (schemas kept, CIs added) ----
    summary = {
        "n_prompts": len(prompts) if prompts else n_traces,
        "n_loop": n_loop,
        "n_non": n_non,
        "loop_rate": n_loop / max(n_traces, 1),
        "mean_loop_align": mean_loop_align,
        "mean_non_align": mean_non_align,
    }
    save_json(summary, out / "summary.json")

    rows = []
    for rel in POSITIONS:
        rows.append(
            {
                "rel_pos": rel,
                "loop_occ_mean": mean_or_none(loop_occ[rel]),
                "non_occ_mean": mean_or_none(non_occ[rel]),
                "loop_top1": mean_or_none(loop_top1[rel]),
                "non_top1": mean_or_none(non_top1[rel]),
                "loop_occ_ci_lo": _group_stat(loop_occ[rel]).get("ci95_lo"),
                "loop_occ_ci_hi": _group_stat(loop_occ[rel]).get("ci95_hi"),
                "non_occ_ci_lo": _group_stat(non_occ[rel]).get("ci95_lo"),
                "non_occ_ci_hi": _group_stat(non_occ[rel]).get("ci95_hi"),
                "loop_top1_wilson_lo": _group_stat(loop_top1[rel], wilson=True).get("wilson95_lo"),
                "loop_top1_wilson_hi": _group_stat(loop_top1[rel], wilson=True).get("wilson95_hi"),
                "non_top1_wilson_lo": _group_stat(non_top1[rel], wilson=True).get("wilson95_lo"),
                "non_top1_wilson_hi": _group_stat(non_top1[rel], wilson=True).get("wilson95_hi"),
            }
        )
    pd.DataFrame(rows).to_csv(out / "pre_onset_series.csv", index=False)

    # Human-readable matched-control summary lines.
    mc_lines = []
    for mname in ("occupancy_at_m2", "top1_trigger_at_m1", "align_trigger_dir_at_m1"):
        m = metrics[mname]
        mc_lines.append(
            "- {name}: loop {lm:.3f} [n={ln}] vs same-token {sm:.3f} [n={sn}] "
            "(p={ps:.4g}), pos-matched {pm:.3f} [n={pn}] (p={pp:.4g}), "
            "legacy {gm:.3f} [n={gn}] (p={pg:.4g})".format(
                name=mname,
                lm=m["loop"]["mean"], ln=m["loop"]["n"],
                sm=m["same_token_anchor"]["mean"], sn=m["same_token_anchor"]["n"],
                ps=m["perm_p_loop_vs_same_token"],
                pm=m["position_matched_anchor"]["mean"], pn=m["position_matched_anchor"]["n"],
                pp=m["perm_p_loop_vs_position_matched"],
                gm=m["legacy_mid_anchor_nonloop"]["mean"], gn=m["legacy_mid_anchor_nonloop"]["n"],
                pg=m["perm_p_loop_vs_legacy_mid"],
            )
        )
    status = f"""# Experiment 2 — Dynamic Monitoring

- Prompts: {n_traces}, loops: {n_loop}, non-loops: {n_non}, rate: {summary['loop_rate']:.1%}
- Workspace layers cached (Tier-2): {ws_layers}
- Mean trigger alignment (−1): loop={summary['mean_loop_align']}, non={summary['mean_non_align']}

**Matched-control headline metrics** (bootstrap 95% CIs + permutation p-values in
`findings.json` → `matched_control_analysis`):
{chr(10).join(mc_lines)}

Plots: `workspace_occupancy_pre_onset.png`, `top1_readout_convergence.png`,
`trigger_alignment_pre_onset.png`, `matched_controls_headline.png`,
`example_looping_trace.html`.
"""
    write_status("04_exp2", status)
    append_results_summary("Experiment 2 — Dynamic Monitoring", status)
    from jspace.loading import clear_cuda

    clear_cuda()
    return 0


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
