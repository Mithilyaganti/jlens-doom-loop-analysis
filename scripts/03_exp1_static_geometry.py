#!/usr/bin/env python
"""Experiment 1 — Static geometry of trigger tokens in J-space."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp1")

# Geometry/stats-only rerun against saved token sets + lens + local safetensors
# unembedding rows — no model load, CPU-safe (primary integration path).
ANALYZE_ONLY = os.environ.get("JLENS_EXP1_ANALYZE_ONLY", "0") == "1"
N_BOOT = int(os.environ.get("JLENS_EXP1_N_BOOT", "2000"))
N_PERM = int(os.environ.get("JLENS_EXP1_N_PERM", "5000"))

SET_NAMES = ["trigger", "freq_content", "discourse", "random"]
CONTROLS = ["freq_content", "discourse", "random"]


def _load_token_sets_from_csvs(out: Path) -> dict[str, list[dict]]:
    """Reload token sets saved by a previous full run (analyze-only path)."""
    sets: dict[str, list[dict]] = {}
    for name in SET_NAMES:
        p = out / f"token_set_{name}.csv"
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} missing — run the full script once (without "
                "JLENS_EXP1_ANALYZE_ONLY) to build token sets."
            )
        df = pd.read_csv(p)
        rows = df.to_dict("records")
        for r in rows:
            r["token_id"] = int(r["token_id"])
        sets[name] = rows
    return sets


def _lens_only_stack(cfg, paths) -> SimpleNamespace:
    """Lens + lazy unembedding rows, no model (CPU-only analyze path)."""
    from jspace.loading import load_jacobian_lens
    from jspace.geometry import hf_safetensors_unembed

    lens_path = Path(cfg.lens_path or paths["lens"])
    if not lens_path.is_file():
        raise FileNotFoundError(f"Lens not found at {lens_path} — cannot run analyze-only.")
    lens = load_jacobian_lens(lens_path)
    rows = hf_safetensors_unembed(cfg.model_id)
    if rows is None:
        raise RuntimeError(
            f"Could not resolve local safetensors unembedding rows for {cfg.model_id}. "
            "Run the full script (loads the model) instead."
        )
    logger.info(
        "Analyze-only stack: lens %s (%d layers), lazy W_U rows via %s",
        lens_path.name, len(lens.source_layers), rows.tensor_name,
    )
    return SimpleNamespace(
        lens=lens,
        lm_head=SimpleNamespace(weight=rows),
        tokenizer=None,
        model=None,
        device=torch.device("cpu"),
    )


def main() -> int:
    from jspace.loading import load_stack, clear_cuda
    from jspace.model_config import get_active_model, artifact_paths
    from jspace.geometry import (
        jlens_direction_batch,
        per_token_mean_cosine,
        per_token_alignment,
        pairwise_cosine,
        pairwise_cosine_matrix,
        top_principal_basis,
    )
    from jspace.stats import bootstrap_ci, permutation_test
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

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    out = ensure_dir(ROOT / "results" / "exp1")

    if ANALYZE_ONLY:
        stack = _lens_only_stack(cfg, paths)
        sets = _load_token_sets_from_csvs(out)
        logger.info("ANALYZE_ONLY: %d token sets from cached CSVs", len(sets))
    else:
        stack = load_stack()
        from jspace.token_sets import build_all_token_sets

        trigger_csv = paths["trigger_csv"]
        sets = build_all_token_sets(stack.tokenizer, trigger_csv if trigger_csv.is_file() else None, size=30)
        # Save token sets
        for name, items in sets.items():
            pd.DataFrame(items).to_csv(out / f"token_set_{name}.csv", index=False)
        # Publishable trigger table copy
        if trigger_csv.is_file():
            pd.read_csv(trigger_csv).head(30).to_csv(out / "trigger_token_table.csv", index=False)
        else:
            pd.DataFrame(sets["trigger"]).to_csv(out / "trigger_token_table.csv", index=False)

    band_path = paths["workspace_band"]
    if not band_path.is_file():
        raise FileNotFoundError(
            f"Missing {band_path}. Run scripts/01_workspace_band.py first — "
            "do not invent layer boundaries."
        )
    band = load_json(band_path)
    for key in ("workspace_start", "workspace_end", "mid_workspace_layer"):
        if key not in band:
            raise ValueError(f"workspace band file missing required key: {key}")

    layers = sorted(stack.lens.source_layers)
    mean_norms = {n: [] for n in SET_NAMES}
    mean_sims = {n: [] for n in SET_NAMES}
    mean_align = {n: [] for n in SET_NAMES}

    # Per-token observations for bootstrap CIs / permutation tests.
    tok_sim = {n: {} for n in SET_NAMES}   # {set: {layer: np.ndarray[n_tokens]}}
    tok_align = {n: {} for n in SET_NAMES}

    mid = int(band.get("mid_workspace_layer", layers[len(layers) // 2]))
    if mid not in stack.lens.jacobians:
        mid = min(stack.lens.jacobians.keys(), key=lambda x: abs(x - mid))

    mid_dirs = {}

    for l in layers:
        # J does not depend on the token set: one SVD per layer, reused 4x.
        basis = top_principal_basis(stack, l)
        for name in SET_NAMES:
            ids = [x["token_id"] for x in sets[name]]
            raw = jlens_direction_batch(stack, ids, l, normalize=False)
            norms = raw.norm(dim=-1)
            dirs = raw / norms.clamp_min(1e-12).unsqueeze(-1)
            mean_norms[name].append(float(norms.mean()))
            sims_tok = per_token_mean_cosine(dirs)
            align_tok = per_token_alignment(dirs, basis[:64])
            mean_sims[name].append(float(sims_tok.mean()))
            mean_align[name].append(float(align_tok.mean()))
            tok_sim[name][l] = sims_tok.numpy().astype(float)
            tok_align[name][l] = align_tok.numpy().astype(float)
            if l == mid:
                mid_dirs[name] = dirs
        if (l + 1) % 6 == 0:
            logger.info("geometry layer %d/%d", l + 1, len(layers))

    # ------------------------------------------------------------------
    # Statistics: bootstrap 95% CIs over tokens + permutation tests
    # (trigger vs each control, permuting set membership over pooled
    # per-token values), per layer and for both metrics.
    # ------------------------------------------------------------------
    ci_lo = {n: [] for n in SET_NAMES}
    ci_hi = {n: [] for n in SET_NAMES}
    al_ci_lo = {n: [] for n in SET_NAMES}
    al_ci_hi = {n: [] for n in SET_NAMES}
    perm_sim = {c: [] for c in CONTROLS}
    perm_align = {c: [] for c in CONTROLS}

    for l in layers:
        for name in SET_NAMES:
            _, lo, hi = bootstrap_ci(tok_sim[name][l], n_boot=N_BOOT, seed=1000 + l)
            ci_lo[name].append(lo)
            ci_hi[name].append(hi)
            _, lo2, hi2 = bootstrap_ci(tok_align[name][l], n_boot=N_BOOT, seed=2000 + l)
            al_ci_lo[name].append(lo2)
            al_ci_hi[name].append(hi2)
        for c in CONTROLS:
            perm_sim[c].append(
                permutation_test(tok_sim["trigger"][l], tok_sim[c][l], n_perm=N_PERM, seed=3000 + l)
            )
            perm_align[c].append(
                permutation_test(tok_align["trigger"][l], tok_align[c][l], n_perm=N_PERM, seed=4000 + l)
            )

    # Plots (with bootstrap CI half-widths as error bars)
    ws = (band["workspace_start"], band["workspace_end"])
    plot_lines_by_layer(
        layers,
        mean_norms,
        ylabel="Mean direction norm (pre-normalize)",
        title="J-lens direction strength by layer",
        out_path=out / "norm_by_layer.png",
        workspace=ws,
    )
    sim_yerr = {n: (np.array(ci_hi[n]) - np.array(ci_lo[n])) / 2 for n in SET_NAMES}
    plot_lines_by_layer(
        layers,
        mean_sims,
        ylabel="Mean pairwise cosine similarity",
        title="Within-set pairwise similarity by layer",
        out_path=out / "pairwise_sim_by_layer.png",
        workspace=ws,
        yerr=sim_yerr,
    )
    align_yerr = {n: (np.array(al_ci_hi[n]) - np.array(al_ci_lo[n])) / 2 for n in SET_NAMES}
    plot_lines_by_layer(
        layers,
        mean_align,
        ylabel="Fraction variance in top-64 subspace",
        title="Workspace subspace alignment by layer",
        out_path=out / "workspace_alignment_by_layer.png",
        workspace=ws,
        yerr=align_yerr,
    )

    # PCA of mid-layer directions
    all_dirs = []
    labels = []
    for name in SET_NAMES:
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
        label_order=SET_NAMES,
    )

    # Heatmap: trigger vs each control at mid layer
    t_dirs = mid_dirs["trigger"]
    for name in SET_NAMES:
        if name == "trigger":
            M = pairwise_cosine_matrix(t_dirs).numpy()
        else:
            a = t_dirs
            b = mid_dirs[name]
            a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            M = (a_n @ b_n.T).numpy()
        plot_heatmap(
            M,
            title=f"Trigger vs {name} cosine (layer {mid})",
            out_path=out / f"heatmap_trigger_vs_{name}.png",
            vmin=-0.5,
            vmax=1.0,
        )

    # Combined figure-like summary heatmap: mean cross-sim
    summary = np.zeros((len(SET_NAMES), len(SET_NAMES)))
    for i, ni in enumerate(SET_NAMES):
        for j, nj in enumerate(SET_NAMES):
            a = mid_dirs[ni]
            b = mid_dirs[nj]
            a_n = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            b_n = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            summary[i, j] = float((a_n @ b_n.T).mean())
    plot_heatmap(
        summary,
        title=f"Mean cross-set cosine (layer {mid})",
        out_path=out / "trigger_vs_controls_heatmap.png",
        xticklabels=SET_NAMES,
        yticklabels=SET_NAMES,
        vmin=-0.2,
        vmax=1.0,
    )

    # CSV of layer curves (old columns preserved; CI + permutation columns added)
    df = pd.DataFrame({"layer": layers})
    for name in SET_NAMES:
        df[f"norm_{name}"] = mean_norms[name]
        df[f"sim_{name}"] = mean_sims[name]
        df[f"sim_{name}_ci_lo"] = ci_lo[name]
        df[f"sim_{name}_ci_hi"] = ci_hi[name]
        df[f"align_{name}"] = mean_align[name]
        df[f"align_{name}_ci_lo"] = al_ci_lo[name]
        df[f"align_{name}_ci_hi"] = al_ci_hi[name]
    for c in CONTROLS:
        df[f"perm_p_sim_trigger_vs_{c}"] = perm_sim[c]
        df[f"perm_p_align_trigger_vs_{c}"] = perm_align[c]
    df.to_csv(out / "geometry_by_layer.csv", index=False)

    # ------------------------------------------------------------------
    # Workspace-band statistics: pool per-token values across band layers
    # and test trigger vs each control on the pooled observations.
    # ------------------------------------------------------------------
    ws_layers = [l for l in layers if band["workspace_start"] <= l <= band["workspace_end"]]

    def pooled(store, name):
        return np.concatenate([store[name][l] for l in ws_layers if l in store[name]]) \
            if any(l in store[name] for l in ws_layers) else np.array([])

    def band_mean(series_dict, name):
        idxs = [layers.index(l) for l in ws_layers if l in layers]
        return float(np.mean([series_dict[name][i] for i in idxs])) if idxs else float("nan")

    trig_sim = band_mean(mean_sims, "trigger")
    ctrl_sim = np.mean([band_mean(mean_sims, n) for n in CONTROLS])
    trig_al = band_mean(mean_align, "trigger")
    ctrl_al = np.mean([band_mean(mean_align, n) for n in CONTROLS])

    band_stats: dict[str, Any] = {}
    success_statements: list[str] = []
    success_p: dict[str, float] = {}
    min_p_sim = float("nan")
    min_p_al = float("nan")

    for metric, store, trig_mean, ctrl_mean in (
        ("pairwise_sim", tok_sim, trig_sim, ctrl_sim),
        ("alignment", tok_align, trig_al, ctrl_al),
    ):
        entry: dict[str, Any] = {}
        for name in SET_NAMES:
            vals = pooled(store, name)
            if vals.size == 0:
                entry[name] = {"n": 0, "mean": float("nan")}
                continue
            point, lo, hi = bootstrap_ci(vals, n_boot=N_BOOT, seed=5000)
            entry[name] = {
                "n": int(vals.size),
                "mean": float(point),
                "ci95_lo": float(lo),
                "ci95_hi": float(hi),
            }
        for c in CONTROLS:
            a = pooled(store, "trigger")
            b = pooled(store, c)
            p = permutation_test(a, b, n_perm=N_PERM, seed=6000) if a.size and b.size else float("nan")
            entry[f"perm_p_trigger_vs_{c}"] = float(p)
        band_stats[metric] = entry

    ctrl_sim_means = [band_stats["pairwise_sim"][c]["mean"] for c in CONTROLS]
    ctrl_al_means = [band_stats["alignment"][c]["mean"] for c in CONTROLS]
    ps = [band_stats["pairwise_sim"].get(f"perm_p_trigger_vs_{c}") for c in CONTROLS]
    pa = [band_stats["alignment"].get(f"perm_p_trigger_vs_{c}") for c in CONTROLS]
    min_p_sim = float(np.nanmin(ps)) if ps else float("nan")
    min_p_al = float(np.nanmin(pa)) if pa else float("nan")
    success_p = {"pairwise_sim_min_p": min_p_sim, "alignment_min_p": min_p_al}

    t_e = band_stats["pairwise_sim"]["trigger"]
    a_e = band_stats["alignment"]["trigger"]
    sim_met = bool(trig_sim > ctrl_sim)
    al_met = bool(trig_al > ctrl_al)
    success = sim_met or al_met
    if sim_met:
        stmt = (
            f"Pairwise-sim criterion met: trigger {trig_sim:.4f} "
            f"[{t_e.get('ci95_lo', float('nan')):.4f}, {t_e.get('ci95_hi', float('nan')):.4f}] > "
            f"controls {ctrl_sim_means} with min permutation p={min_p_sim:.4g} "
            f"(n={t_e.get('n', 0)} trigger-token observations pooled over band layers {ws_layers})"
        )
    else:
        stmt = f"Pairwise-sim criterion NOT met (trigger {trig_sim:.4f} vs controls {ctrl_sim_means}; min p={min_p_sim:.4g})"
    success_statements.append(stmt)
    if al_met:
        stmt = (
            f"Workspace-alignment criterion met: trigger {trig_al:.4f} "
            f"[{a_e.get('ci95_lo', float('nan')):.4f}, {a_e.get('ci95_hi', float('nan')):.4f}] > "
            f"controls {ctrl_al_means} with min permutation p={min_p_al:.4g} "
            f"(n={a_e.get('n', 0)} trigger-token observations pooled over band layers {ws_layers})"
        )
    else:
        stmt = f"Workspace-alignment criterion NOT met (trigger {trig_al:.4f} vs controls {ctrl_al_means}; min p={min_p_al:.4g})"
    success_statements.append(stmt)

    findings = {
        "mid_layer": mid,
        "workspace_mean_pairwise_sim_trigger": trig_sim,
        "workspace_mean_pairwise_sim_controls": ctrl_sim,
        "workspace_mean_align_trigger": trig_al,
        "workspace_mean_align_controls": ctrl_al,
        "success_criterion_met": bool(success),
        "workspace_band_layers": [int(l) for l in ws_layers],
        "workspace_band_stats": band_stats,
        "success_p_values": success_p,
        "success_statements": success_statements,
        "stats_settings": {
            "bootstrap_n": N_BOOT,
            "permutation_n": N_PERM,
            "unit": "per-token values pooled over workspace-band layers",
            "analyze_only": ANALYZE_ONLY,
        },
    }
    # Keep any keys a previous run may have added.
    findings_path = out / "findings.json"
    if findings_path.is_file():
        try:
            prev = load_json(findings_path)
            if isinstance(prev, dict):
                merged = {**prev, **findings}
                findings = merged
        except Exception:
            pass
    save_json(findings, findings_path)

    for s in success_statements:
        logger.info("EXP1 %s", s)

    status = f"""# Experiment 1 — Static Geometry

Compared trigger tokens vs frequency-matched content, discourse controls, and random tokens
across all layers.

**Workspace-band means (L{band['workspace_start']}–L{band['workspace_end']}):**
- Pairwise sim: trigger={trig_sim:.4f} vs controls={ctrl_sim:.4f} (min perm p={min_p_sim:.4g})
- Workspace align: trigger={trig_al:.4f} vs controls={ctrl_al:.4f} (min perm p={min_p_al:.4g})
- Success criterion (higher clustering and/or alignment): **{success}**

**Significance statements**
{chr(10).join('- ' + s for s in success_statements)}

Bootstrap 95% CIs over tokens and trigger-vs-control permutation tests per layer are in
`geometry_by_layer.csv` (`*_ci_lo/hi`, `perm_p_*` columns); pooled band statistics under
`workspace_band_stats` in `findings.json`.

Plots in `results/exp1/`. Mid-layer PCA at L{mid}.
"""
    write_status("03_exp1", status)
    append_results_summary("Experiment 1 — Static Geometry", status)
    clear_cuda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
