"""Plotting, stats, and result I/O helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path | str) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_status(name: str, text: str) -> Path:
    path = ensure_dir(RESULTS / "status") / f"{name}.md"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    logger.info("Status written: %s", path)
    return path


def shade_workspace(ax, start: int, end: int, label: str = "workspace") -> None:
    ax.axvspan(start, end, alpha=0.15, color="steelblue", label=label)


def plot_lines_by_layer(
    layers: Sequence[int],
    series: dict[str, Sequence[float]],
    *,
    ylabel: str,
    title: str,
    out_path: Path,
    workspace: tuple[int, int] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, vals in series.items():
        ax.plot(layers, vals, marker="o", markersize=3, label=name, linewidth=1.5)
    if workspace:
        shade_workspace(ax, workspace[0], workspace[1])
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_workspace_band_metrics(
    metrics: dict[str, Any],
    band: dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir = ensure_dir(out_dir)
    layers = metrics["layers"]
    series = {
        "excess_kurtosis (norm)": _norm_list([metrics["kurtosis"][l] for l in layers]),
        "NTP top-1": [metrics["ntp_top1"].get(l, 0.0) for l in layers],
        "eff_dim (norm)": _norm_list(
            [metrics["effective_dimensionality"][l] for l in layers]
        ),
    }
    plot_lines_by_layer(
        layers,
        series,
        ylabel="Metric (normalized / accuracy)",
        title="Workspace band identification",
        out_path=out_dir / "workspace_band_metrics.png",
        workspace=(band["workspace_start"], band["workspace_end"]),
    )
    # Also raw curves separately
    plot_lines_by_layer(
        layers,
        {"excess_kurtosis": [metrics["kurtosis"][l] for l in layers]},
        ylabel="Excess kurtosis",
        title="J-lens excess kurtosis by layer",
        out_path=out_dir / "kurtosis_by_layer.png",
        workspace=(band["workspace_start"], band["workspace_end"]),
    )
    plot_lines_by_layer(
        layers,
        {
            "top-1": [metrics["ntp_top1"].get(l, 0.0) for l in layers],
            "top-5": [metrics["ntp_topk"].get(l, 0.0) for l in layers],
        },
        ylabel="Accuracy vs model next-token",
        title="J-lens NTP accuracy by layer",
        out_path=out_dir / "ntp_accuracy_by_layer.png",
        workspace=(band["workspace_start"], band["workspace_end"]),
        ylim=(0, 1.05),
    )
    plot_lines_by_layer(
        layers,
        {
            "participation_ratio": [
                metrics["effective_dimensionality"][l] for l in layers
            ]
        },
        ylabel="Effective dimensionality",
        title="J-space effective dimensionality by layer",
        out_path=out_dir / "effdim_by_layer.png",
        workspace=(band["workspace_start"], band["workspace_end"]),
    )


def _norm_list(vals: Sequence[float]) -> list[float]:
    a = np.asarray(vals, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    if hi - lo < 1e-12:
        return [0.0] * len(a)
    return list((a - lo) / (hi - lo))


def plot_bar_with_ci(
    labels: Sequence[str],
    means: Sequence[float],
    cis: Sequence[float] | None,
    *,
    ylabel: str,
    title: str,
    out_path: Path,
    color: str = "steelblue",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, color=color, alpha=0.85, yerr=cis, capsize=4, ecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_pca_scatter(
    points: np.ndarray,
    labels: Sequence[str],
    *,
    title: str,
    out_path: Path,
    label_order: Sequence[str] | None = None,
) -> None:
    """points: [n, 2] PCA coords; labels: set name per point."""
    fig, ax = plt.subplots(figsize=(8, 7))
    order = list(label_order) if label_order else sorted(set(labels))
    cmap = plt.cm.tab10
    for i, name in enumerate(order):
        mask = np.array([lb == name for lb in labels])
        if not mask.any():
            continue
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=40,
            alpha=0.75,
            label=name,
            color=cmap(i % 10),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def plot_heatmap(
    matrix: np.ndarray,
    *,
    title: str,
    out_path: Path,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    vmin: float = -1,
    vmax: float = 1,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=6)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(yticklabels, fontsize=6)
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def proportion_ci(k: int, n: int, z: float = 1.96) -> float:
    """Wilson-ish simple normal approx half-width for a proportion."""
    if n <= 0:
        return 0.0
    p = k / n
    return float(z * np.sqrt(max(p * (1 - p), 1e-12) / n))


def mcnemar_test(b: int, c: int) -> dict[str, float]:
    """McNemar test on discordant pairs (b, c). Returns chi2 and p-value."""
    from scipy.stats import chi2

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c}
    # continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(1 - chi2.cdf(chi2_stat, df=1))
    return {"chi2": float(chi2_stat), "p_value": p, "b": b, "c": c}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))))


def append_results_summary(section: str, body: str) -> None:
    path = RESULTS / "RESULTS_SUMMARY.md"
    header = "# Results Summary — J-Space Analysis of Doom Loops\n\n"
    if not path.exists():
        path.write_text(header, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {section}\n\n")
        f.write(body.strip() + "\n")
