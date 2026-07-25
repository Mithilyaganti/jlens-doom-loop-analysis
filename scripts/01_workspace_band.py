#!/usr/bin/env python
"""Empirically identify the J-space workspace band for the active model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("workspace_band")


def main() -> int:
    from jspace.loading import load_stack, clear_cuda
    from jspace.model_config import get_active_model, artifact_paths
    from jspace.geometry import compute_workspace_band_metrics, identify_workspace_band
    from jspace.analysis import (
        plot_workspace_band_metrics,
        save_json,
        write_status,
        append_results_summary,
        ensure_dir,
    )

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    out = ensure_dir(paths["results"])
    stack = load_stack()

    # Extra prompts from wikitext if available
    prompts = None
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        prompts = []
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) > 80:
                prompts.append(t[:400])
            if len(prompts) >= 40:
                break
        logger.info("Loaded %d wikitext prompts for NTP", len(prompts))
    except Exception as e:
        logger.warning("wikitext unavailable (%s); using defaults", e)

    metrics = compute_workspace_band_metrics(stack, prompts=prompts)
    band = identify_workspace_band(metrics)

    # Serialize metrics (keys as str for JSON)
    metrics_ser = {
        "layers": metrics["layers"],
        "kurtosis": {str(k): v for k, v in metrics["kurtosis"].items()},
        "effective_dimensionality": {
            str(k): v for k, v in metrics["effective_dimensionality"].items()
        },
        "entropy_dimensionality": {
            str(k): v for k, v in metrics["entropy_dimensionality"].items()
        },
        "ntp_top1": {str(k): v for k, v in metrics["ntp_top1"].items()},
        "ntp_topk": {str(k): v for k, v in metrics["ntp_topk"].items()},
        "n_prompts_ntp": metrics["n_prompts_ntp"],
        "n_layers": metrics["n_layers"],
    }
    save_json(metrics_ser, paths["workspace_band_metrics"])
    save_json(band, paths["workspace_band"])
    plot_workspace_band_metrics(metrics, band, out)

    status = f"""# Workspace Band — {cfg.display_name}

Empirically identified workspace band:

- **Model**: `{cfg.model_id}`
- **Start layer**: {band['workspace_start']}
- **End layer**: {band['workspace_end']}
- **Mid layer**: {band['mid_workspace_layer']}
- **Key layers (Tier-2 cache)**: {band['key_workspace_layers']}
- Sensory layers: {len(band['sensory_layers'])} layers before band
- Motor layers: {len(band['motor_layers'])} layers after band
- NTP prompts used: {metrics['n_prompts_ntp']}

Metrics and plots: `results/workspace_band_*.png`, `{paths['workspace_band']}`.
"""
    write_status("01_workspace_band", status)
    append_results_summary(
        f"Workspace Band ({cfg.display_name})",
        status,
    )
    clear_cuda()
    logger.info("Workspace band: L%d–L%d", band["workspace_start"], band["workspace_end"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
