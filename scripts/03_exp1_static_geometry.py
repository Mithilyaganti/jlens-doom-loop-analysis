#!/usr/bin/env python
"""Experiment 1 — Static geometry of trigger tokens in J-space."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp1")


def main() -> int:
    from jspace.loading import load_stack, clear_cuda
    from jspace.geometry import (
        jlens_direction_batch,
        pairwise_cosine,
        pairwise_cosine_matrix,
        direction_norms,
        workspace_alignment,
    )
    from jspace.token_sets import build_all_token_sets
    from jspace.analysis import (
        ensure_dir,
        load_json,
        save_json,
        plot_lines_by_layer,
        plot_pca_scatter,
        plot_heatmap,
        write_status,
        append_results_summary,
    )

    out = ensure_dir(ROOT / "results" / "exp1")
    stack = load_stack()

    band_path = ROOT / "results" / "workspace_band_qwen3.5-4b.json"
    if not band_path.is_file():
        raise FileNotFoundError(
            f"Missing {band_path}. Run scripts/01_workspace_band.py first — "
            "do not invent layer boundaries."
        )
    band = load_json(band_path)
    for key in ("workspace_start", "workspace_end", "mid_workspace_layer"):
        if key not in band:
            raise ValueError(f"workspace band file missing required key: {key}")

    trigger_csv = ROOT / "results" / "trigger_tokens_qwen3.5-4b.csv"
    sets = build_all_token_sets(stack.tokenizer, trigger_csv if trigger_csv.is_file() else None, size=30)

    # Save token sets
    for name, items in sets.items():
        pd.DataFrame(items).to_csv(out / f"token_set_{name}.csv", index=False)
    # Publishable trigger table copy
    if trigger_csv.is_file():
        pd.read_csv(trigger_csv).head(30).to_csv(out / "trigger_token_table.csv", index=False)
    else:
        pd.DataFrame(sets["trigger"]).to_csv(out / "trigger_token_table.csv", index=False)

    layers = sorted(stack.lens.source_layers)
    set_names = ["trigger", "freq_content", "discourse", "random"]
    mean_norms = {n: [] for n in set_names}
    mean_sims = {n: [] for n in set_names}
    mean_align = {n: [] for n in set_names}

    mid = int(band.get("mid_workspace_layer", layers[len(layers) // 2]))
    if mid not in stack.lens.jacobians:
        mid = min(stack.lens.jacobians.keys(), key=lambda x: abs(x - mid))

    mid_dirs = {}
    mid_labels = []
    mid_points = []

    for l in layers:
        for name in set_names:
            ids = [x["token_id"] for x in sets[name]]
            norms = direction_norms(stack, ids, l)
            dirs = jlens_direction_batch(stack, ids, l, normalize=True)
            mean_norms[name].append(float(norms.mean()))
            mean_sims[name].append(float(pairwise_cosine(dirs)))
            mean_align[name].append(workspace_alignment(dirs, stack, l, k=64))
            if l == mid:
                mid_dirs[name] = dirs
        if (l + 1) % 6 == 0:
            logger.info("geometry layer %d/%d", l + 1, len(layers))

    # Plots
    ws = (band["workspace_start"], band["workspace_end"])
    plot_lines_by_layer(
        layers,
        mean_norms,
        ylabel="Mean direction norm (pre-normalize)",
        title="J-lens direction strength by layer",
        out_path=out / "norm_by_layer.png",
        workspace=ws,
    )
    plot_lines_by_layer(
        layers,
        mean_sims,
        ylabel="Mean pairwise cosine similarity",
        title="Within-set pairwise similarity by layer",
        out_path=out / "pairwise_sim_by_layer.png",
        workspace=ws,
    )
    plot_lines_by_layer(
        layers,
        mean_align,
        ylabel="Fraction variance in top-64 subspace",
        title="Workspace subspace alignment by layer",
        out_path=out / "workspace_alignment_by_layer.png",
        workspace=ws,
    )

    # PCA of mid-layer directions
    all_dirs = []
    labels = []
    for name in set_names:
        d = mid_dirs[name].numpy()
        all_dirs.append(d)
        labels.extend([name] * d.shape[0])
    X = np.concatenate(all_dirs, axis=0)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        coords = U[:, :2] * S[:2]
    except Exception:
        coords = X[:, :2]
    plot_pca_scatter(
        coords,
        labels,
        title=f"PCA of token J-directions (layer {mid})",
        out_path=out / "pca_workspace_band.png",
        label_order=set_names,
    )

    # Heatmap: trigger vs each control at mid layer
    t_dirs = mid_dirs["trigger"]
    mats = []
    titles = []
    for name in set_names:
        if name == "trigger":
            M = pairwise_cosine_matrix(t_dirs).numpy()
        else:
            a = t_dirs
            b = mid_dirs[name]
            a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            M = (a_n @ b_n.T).numpy()
        mats.append(M)
        titles.append(name)
        plot_heatmap(
            M,
            title=f"Trigger vs {name} cosine (layer {mid})",
            out_path=out / f"heatmap_trigger_vs_{name}.png",
            vmin=-0.5,
            vmax=1.0,
        )

    # Combined figure-like summary heatmap: mean cross-sim
    summary = np.zeros((len(set_names), len(set_names)))
    for i, ni in enumerate(set_names):
        for j, nj in enumerate(set_names):
            a = mid_dirs[ni]
            b = mid_dirs[nj]
            a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            summary[i, j] = float((a_n @ b_n.T).mean())
    plot_heatmap(
        summary,
        title=f"Mean cross-set cosine (layer {mid})",
        out_path=out / "trigger_vs_controls_heatmap.png",
        xticklabels=set_names,
        yticklabels=set_names,
        vmin=-0.2,
        vmax=1.0,
    )

    # CSV of layer curves
    df = pd.DataFrame({"layer": layers})
    for name in set_names:
        df[f"norm_{name}"] = mean_norms[name]
        df[f"sim_{name}"] = mean_sims[name]
        df[f"align_{name}"] = mean_align[name]
    df.to_csv(out / "geometry_by_layer.csv", index=False)

    # Success criterion check in workspace band
    ws_layers = [l for l in layers if band["workspace_start"] <= l <= band["workspace_end"]]
    def band_mean(series_dict, name):
        idxs = [layers.index(l) for l in ws_layers if l in layers]
        return float(np.mean([series_dict[name][i] for i in idxs])) if idxs else float("nan")

    trig_sim = band_mean(mean_sims, "trigger")
    ctrl_sim = np.mean([band_mean(mean_sims, n) for n in ("freq_content", "discourse", "random")])
    trig_al = band_mean(mean_align, "trigger")
    ctrl_al = np.mean([band_mean(mean_align, n) for n in ("freq_content", "discourse", "random")])

    success = (trig_sim > ctrl_sim) or (trig_al > ctrl_al)
    findings = {
        "mid_layer": mid,
        "workspace_mean_pairwise_sim_trigger": trig_sim,
        "workspace_mean_pairwise_sim_controls": ctrl_sim,
        "workspace_mean_align_trigger": trig_al,
        "workspace_mean_align_controls": ctrl_al,
        "success_criterion_met": bool(success),
    }
    save_json(findings, out / "findings.json")

    status = f"""# Experiment 1 — Static Geometry

Compared trigger tokens vs frequency-matched content, discourse controls, and random tokens
across all layers.

**Workspace-band means (L{band['workspace_start']}–L{band['workspace_end']}):**
- Pairwise sim: trigger={trig_sim:.4f} vs controls={ctrl_sim:.4f}
- Workspace align: trigger={trig_al:.4f} vs controls={ctrl_al:.4f}
- Success criterion (higher clustering and/or alignment): **{success}**

Plots in `results/exp1/`. Mid-layer PCA at L{mid}.
"""
    write_status("03_exp1", status)
    append_results_summary("Experiment 1 — Static Geometry", status)
    clear_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
