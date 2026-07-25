#!/usr/bin/env python
"""Fit (or stub) a J-lens for LiquidAI/LFM2-2.6B hybrid architecture.

No pre-fitted lens exists on neuronpedia for LFM. The model mixes short-conv
(LIV) blocks with attention blocks. The open-jlens fitter assumes pure
transformer layers — conv layers must be skipped or given a custom Jacobian.

This script:
1. Loads LFM2-2.6B
2. Identifies which layers are attention vs conv (best-effort)
3. Attempts to run the open-jlens fitter on attention layers only if the
   vendor recipe is available
4. Otherwise writes a clear status note that Exp1–3 are blocked until a
   successful fit on Colab / cloud GPU

Generation + loop detection do NOT require this script.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fit_lfm_lens")


def main() -> int:
    from jspace.model_config import get_active_model, artifact_paths
    from jspace.analysis import write_status, ensure_dir, save_json

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    ensure_dir(paths["results"])
    lens_path = paths["lens"]
    assert lens_path is not None

    if lens_path.is_file():
        logger.info("Lens already present: %s", lens_path)
        write_status(
            "06_fit_lfm_lens",
            f"# LFM J-lens\n\nAlready present at `{lens_path}`.\n",
        )
        return 0

    status = {
        "model_id": cfg.model_id,
        "hybrid": cfg.hybrid_architecture,
        "lens_path": str(lens_path),
        "fitted": False,
        "blocker": (
            "No pre-fitted J-lens for LFM2-2.6B. Fitting requires adapting "
            "open-jlens (vendor/open-jlens-data) to skip LIV conv blocks and "
            "fit Jacobians only on attention layers. Prefer Colab A100/L4 with "
            "bf16 for the fit. Until then: run baseline generation+detection; "
            "defer Exp1/Exp2 geometry/Exp3 ablation."
        ),
        "next_steps": [
            "On Colab: install open-jlens deps, load LiquidAI/LFM2-2.6B bf16",
            "Enumerate layers; mark conv vs attn via module class names",
            "Fit JacobianLens on attn layers only (wikitext ~200–1000 prompts)",
            f"Save to {lens_path}",
            "Run scripts/01_workspace_band.py then Exp1–3",
        ],
    }
    save_json(status, paths["results"] / f"jlens_fit_status_{cfg.slug}.json")
    write_status(
        "06_fit_lfm_lens",
        "# LFM J-lens fit\n\n"
        f"**Not fitted yet.** See `results/jlens_fit_status_{cfg.slug}.json`.\n\n"
        "Baseline loop-rate measurement can proceed without the lens.\n",
    )
    logger.warning("%s", status["blocker"])
    return 2  # soft fail — pipeline may continue generation-only


if __name__ == "__main__":
    raise SystemExit(main())
