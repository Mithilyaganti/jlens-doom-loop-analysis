#!/usr/bin/env python
"""Download lens, load model, run sanity check."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("setup")


def main() -> int:
    from jspace.loading import (
        DEFAULT_LENS_PATH,
        clear_cuda,
        download_lens,
        load_stack,
        sanity_check_lens,
    )
    from jspace.analysis import ensure_dir, save_json, write_status

    ensure_dir(ROOT / "lenses")
    ensure_dir(ROOT / "results" / "status")

    logger.info("Downloading J-lens if needed...")
    lens_path = download_lens(DEFAULT_LENS_PATH)
    logger.info("Lens at %s (%.1f MB)", lens_path, lens_path.stat().st_size / 1e6)

    logger.info("Loading model stack (4-bit)...")
    stack = load_stack()
    result = sanity_check_lens(stack)
    save_json(result, ROOT / "results" / "sanity_check.json")

    status = f"""# Setup / Sanity Check

- Model: `{stack.model_name}`
- Layers: {stack.n_layers}, d_model: {stack.d_model}
- Lens path: `{lens_path}`
- Final-layer top-1 match: **{result['top1_match']}**
- Top-5 overlap: **{result['top5_overlap']}**/5
- OK: **{result['ok']}**

J-lens top-5: {result['jlens_top5']}
Model top-5: {result['model_top5']}

{"PASS — proceed to workspace band + trigger extraction." if result['ok'] else "FAIL — investigate norm/unembed/layer mapping before experiments."}
"""
    write_status("00_setup", status)
    clear_cuda()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
