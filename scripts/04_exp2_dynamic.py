#!/usr/bin/env python
"""Experiment 2 — Dynamic J-space monitoring before loop onset."""

from __future__ import annotations

import json
import logging
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


def load_saved_trace(trace_dir: Path) -> dict | None:
    """Load a previously saved Exp2 trace from disk."""
    meta_path = trace_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    trace: dict = {"meta": meta}
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
    else:
        trace["generated_ids"] = []
    from jspace.detection import LoopResult

    trace["loop"] = LoopResult(
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
    # Prefer the readout-local index (after cache-window mapping). Fall back to
    # raw trigger index only when the sequence was not truncated.
    local = meta.get("trigger_local_in_readouts")
    if local is None and not (meta.get("trunc") or {}).get("truncated"):
        local = meta.get("trigger_token_index")
    trace["trigger_local_in_readouts"] = local
    return trace


def iter_saved_traces(looping_dir: Path, nonloop_dir: Path) -> list[tuple[dict, Path, bool]]:
    out: list[tuple[dict, Path, bool]] = []
    for d in sorted(looping_dir.glob("p*")):
        t = load_saved_trace(d)
        if t:
            out.append((t, d, True))
    for d in sorted(nonloop_dir.glob("p*")):
        t = load_saved_trace(d)
        if t:
            out.append((t, d, False))
    return out


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


def main() -> int:
    from jspace.loading import load_stack, clear_cuda
    from jspace.model_config import get_active_model, artifact_paths
    from jspace.generation import (
        apply_chat_template,
        generate_with_tiered_cache,
        loop_result_to_meta,
        save_trace,
    )
    from jspace.geometry import jlens_direction
    from jspace.analysis import (
        ensure_dir,
        load_json,
        save_json,
        plot_lines_by_layer,
        write_status,
        append_results_summary,
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    out = ensure_dir(paths["exp2"])
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

    # Load model when generating, or when analyzing (alignment needs lens).
    stack = load_stack()

    prompts = load_exp2_prompts(MAX_PROMPTS) if not ANALYZE_ONLY else []
    logger.info(
        "Exp2: %d prompts (analyze_only=%s gen_only=%s), ws_layers=%s, max_new=%d",
        len(prompts),
        ANALYZE_ONLY,
        GEN_ONLY,
        ws_layers,
        MAX_NEW_TOKENS,
    )

    # Aggregates for pre-onset positions -5..0
    positions = list(range(-5, 1))
    loop_occ = {p: [] for p in positions}
    non_occ = {p: [] for p in positions}
    loop_top1 = {p: [] for p in positions}  # 1 if trigger in top1
    loop_top3 = {p: [] for p in positions}
    loop_top10 = {p: [] for p in positions}
    non_top1 = {p: [] for p in positions}
    non_top3 = {p: [] for p in positions}
    non_top10 = {p: [] for p in positions}
    loop_align = []
    non_align = []

    n_loop = 0
    n_non = 0
    interesting_loops = []

    def process_trace(trace: dict, meta: dict, loop_is_loop: bool, dest: Path, i: int, total: int) -> None:
        nonlocal n_loop, n_non
        loop = trace["loop"]
        readouts = trace.get("readouts")
        residuals = trace.get("residuals_ws") or {}
        trig_idx = trace.get("trigger_local_in_readouts")
        if trig_idx is None:
            trig_idx = loop.trigger_token_index

        if loop_is_loop:
            n_loop += 1
            interesting_loops.append((meta, dest))
        else:
            n_non += 1

        if readouts is None:
            return

        idx_t = readouts["indices"]
        val_t = readouts["values"]
        n_gp = idx_t.shape[1]
        trigger_id = loop.trigger_token_id

        if not loop_is_loop:
            anchor = min(n_gp - 1, max(5, n_gp // 2))
        else:
            if trig_idx is None:
                return
            anchor = int(trig_idx)
            if anchor < 0 or anchor >= n_gp:
                return

        for rel in positions:
            pos = anchor + rel
            if pos < 0 or pos >= n_gp:
                continue
            if mid < idx_t.shape[0]:
                top_ids = idx_t[mid, pos].tolist()
                top_vals = val_t[mid, pos]
                occ = workspace_occupancy_from_readouts(idx_t[mid, pos], top_vals)
                if loop_is_loop:
                    loop_occ[rel].append(occ)
                    if trigger_id is not None:
                        loop_top1[rel].append(1.0 if trigger_id == top_ids[0] else 0.0)
                        loop_top3[rel].append(1.0 if trigger_id in top_ids[:3] else 0.0)
                        loop_top10[rel].append(1.0 if trigger_id in top_ids[:10] else 0.0)
                else:
                    gen_ids = trace.get("generated_ids") or []
                    next_id = gen_ids[min(pos, len(gen_ids) - 1)] if gen_ids else None
                    non_occ[rel].append(occ)
                    if next_id is not None:
                        non_top1[rel].append(1.0 if next_id == top_ids[0] else 0.0)
                        non_top3[rel].append(1.0 if next_id in top_ids[:3] else 0.0)
                        non_top10[rel].append(1.0 if next_id in top_ids[:10] else 0.0)

        if mid in residuals and loop_is_loop and trigger_id is not None:
            pos_m1 = anchor - 1
            if 0 <= pos_m1 < residuals[mid].shape[0]:
                h = residuals[mid][pos_m1].float()
                v = jlens_direction(stack, trigger_id, mid, normalize=True)
                cos = float(torch.nn.functional.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item())
                loop_align.append(cos)
        elif mid in residuals and not loop_is_loop and n_gp > 1:
            pos_m1 = anchor - 1
            if 0 <= pos_m1 < residuals[mid].shape[0]:
                h = residuals[mid][pos_m1].float()
                gen_ids = trace.get("generated_ids") or []
                nid = gen_ids[min(anchor, len(gen_ids) - 1)] if gen_ids else None
                if nid is not None:
                    v = jlens_direction(stack, nid, mid, normalize=True)
                    cos = float(torch.nn.functional.cosine_similarity(h.unsqueeze(0), v.unsqueeze(0)).item())
                    non_align.append(cos)

        if (i + 1) % 5 == 0:
            logger.info("exp2 %d/%d loops=%d non=%d", i + 1, total, n_loop, n_non)

    if ANALYZE_ONLY:
        saved = iter_saved_traces(looping_dir, nonloop_dir)
        if not saved:
            raise RuntimeError("JLENS_EXP2_ANALYZE_ONLY=1 but no saved traces found.")
        logger.info("Analyzing %d saved traces", len(saved))
        for i, (trace, dest, is_loop) in enumerate(saved):
            meta = trace["meta"]
            process_trace(trace, meta, is_loop, dest, i, len(saved))
    else:
        from jspace.checkpoint import save_checkpoint, load_checkpoint

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
                if not GEN_ONLY:
                    process_trace(trace, meta, loop.is_loop, dest, i, len(todo))
                else:
                    if loop.is_loop:
                        n_loop += 1
                    else:
                        n_non += 1

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
                "Generation pass finished: completed=%d/%d failed=%d loops=%s non=%s",
                len(completed),
                len(prompts),
                len(failed),
                ckpt.get("n_loop"),
                ckpt.get("n_non"),
            )
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
        clear_cuda()
        return 0

    # Plots
    def means_cis(d):
        xs, ms, lo, hi = [], [], [], []
        for p in positions:
            arr = np.array(d[p], dtype=float)
            xs.append(p)
            if len(arr) == 0:
                ms.append(np.nan)
                lo.append(np.nan)
                hi.append(np.nan)
            else:
                m = arr.mean()
                se = arr.std() / max(np.sqrt(len(arr)), 1)
                ms.append(m)
                lo.append(m - 1.96 * se)
                hi.append(m + 1.96 * se)
        return xs, ms, lo, hi

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, d, color in [("looping", loop_occ, "C3"), ("non-looping", non_occ, "C0")]:
        xs, ms, lo, hi = means_cis(d)
        ax.plot(xs, ms, marker="o", label=label, color=color)
        ax.fill_between(xs, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("Position relative to trigger/anchor")
    ax.set_ylabel("Top-k entropy occupancy (1=spread)")
    ax.set_title("Workspace occupancy pre-onset")
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

    # HTML example for best looping trace (Tier-3 re-run if we have loops)
    html_path = out / "example_looping_trace.html"
    if interesting_loops:
        meta0, dest0 = interesting_loops[0]
        # Prefer longest snippet
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
    summary = {
        "n_prompts": len(prompts) if prompts else n_traces,
        "n_loop": n_loop,
        "n_non": n_non,
        "loop_rate": n_loop / max(n_loop + n_non, 1),
        "mean_loop_align": float(np.mean(loop_align)) if loop_align else None,
        "mean_non_align": float(np.mean(non_align)) if non_align else None,
    }
    save_json(summary, out / "summary.json")

    # Save numeric series
    rows = []
    for rel in positions:
        rows.append(
            {
                "rel_pos": rel,
                "loop_occ_mean": float(np.mean(loop_occ[rel])) if loop_occ[rel] else None,
                "non_occ_mean": float(np.mean(non_occ[rel])) if non_occ[rel] else None,
                "loop_top1": float(np.mean(loop_top1[rel])) if loop_top1[rel] else None,
                "non_top1": float(np.mean(non_top1[rel])) if non_top1[rel] else None,
            }
        )
    pd.DataFrame(rows).to_csv(out / "pre_onset_series.csv", index=False)

    status = f"""# Experiment 2 — Dynamic Monitoring

- Prompts: {n_traces}, loops: {n_loop}, non-loops: {n_non}, rate: {summary['loop_rate']:.1%}
- Workspace layers cached (Tier-2): {ws_layers}
- Mean trigger alignment (−1): loop={summary['mean_loop_align']}, non={summary['mean_non_align']}

Plots: `workspace_occupancy_pre_onset.png`, `top1_readout_convergence.png`,
`trigger_alignment_pre_onset.png`, `example_looping_trace.html`.
"""
    write_status("04_exp2", status)
    append_results_summary("Experiment 2 — Dynamic Monitoring", status)
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
