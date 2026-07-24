#!/usr/bin/env python
"""Experiment 2 — Dynamic J-space monitoring before loop onset."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp2")

MAX_PROMPTS = int(os.environ.get("JLENS_EXP2_PROMPTS", "80"))
MAX_NEW_TOKENS = int(os.environ.get("JLENS_MAX_NEW_TOKENS", "2048"))
ANALYZE_ONLY = os.environ.get("JLENS_EXP2_ANALYZE_ONLY", "0") == "1"


def load_exp2_prompts(max_n: int) -> list[dict]:
    """Load real antidoom-mix + hard math/coding (PROJECT_SPEC §10.2)."""
    from jspace.prompts import load_antidoom_mix, with_hard_extras

    prompts = with_hard_extras(load_antidoom_mix(max_n))
    logger.info("Exp2 prompt set size: %d", len(prompts))
    return prompts


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
    trace["trigger_local_in_readouts"] = meta.get("trigger_token_index")
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

    out = ensure_dir(ROOT / "results" / "exp2")
    looping_dir = ensure_dir(out / "looping")
    nonloop_dir = ensure_dir(out / "nonlooping")

    stack = load_stack()
    band_path = ROOT / "results" / "workspace_band_qwen3.5-4b.json"
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
        "Exp2: %d prompts (analyze_only=%s), ws_layers=%s",
        len(prompts),
        ANALYZE_ONLY,
        ws_layers,
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
        for i, p in enumerate(prompts):
            chat = apply_chat_template(stack.tokenizer, p["text"])
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
                logger.exception("prompt %s failed: %s", p["prompt_id"], e)
                clear_cuda()
                continue

            loop = trace["loop"]
            meta = loop_result_to_meta(loop, trace, prompt_id=p["prompt_id"])
            meta["source"] = p.get("source")
            dest = looping_dir / f"p{p['prompt_id']}" if loop.is_loop else nonloop_dir / f"p{p['prompt_id']}"
            save_trace(dest, trace, meta)
            process_trace(trace, meta, loop.is_loop, dest, i, len(prompts))
            clear_cuda()

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
