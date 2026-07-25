#!/usr/bin/env python
"""Run LFM2-2.6B antidoom-mix 200-prompt protocol (generation-first)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_lfm")


def _archive_stale_exp2_if_needed(paths: dict) -> None:
    """Move prior-model Exp2 traces aside before a fresh LFM baseline."""
    import shutil
    from datetime import datetime

    ckpt = paths["baseline_checkpoint"]
    if ckpt.is_file():
        return
    exp2 = paths["exp2"]
    looping = exp2 / "looping"
    nonloop = exp2 / "nonlooping"
    if not looping.is_dir() and not nonloop.is_dir():
        return
    if not any(looping.glob("p*")) and not any(nonloop.glob("p*")):
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dest = paths["results"] / f"archive_pre_lfm_{stamp}" / "exp2"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exp2), str(dest))
    logger.info("Archived prior exp2 traces to %s", dest)


def main() -> int:
    os.environ.setdefault("JLENS_MODEL", "LiquidAI/LFM2-2.6B")
    os.environ.setdefault("JLENS_BASELINE_PROMPTS", "200")
    os.environ.setdefault("JLENS_MAX_NEW_TOKENS", "4000")
    os.environ.setdefault("JLENS_TEMPERATURE", "0.01")
    # Windows laptop: hf+nf4. Colab: set JLENS_BACKEND=vllm and JLENS_VLLM_DTYPE=fp8 or bfloat16
    if sys.platform == "win32" and os.environ.get("JLENS_BACKEND") is None:
        os.environ["JLENS_BACKEND"] = "hf"

    from jspace.model_config import get_active_model, artifact_paths

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    lens_path = paths["lens"]
    _archive_stale_exp2_if_needed(paths)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env.get("PYTHONPATH", ""))

    scripts: list[str] = [
        "00_colab_vendors.py",
        "06_fit_lfm_lens.py",
        "02_baseline_pass.py",
    ]

    # Exp1–3 only if a real lens file exists
    if lens_path and lens_path.is_file():
        scripts.extend(
            [
                "01_workspace_band.py",
                "03_exp1_static_geometry.py",
                "04_exp2_dynamic.py",
                "05_exp3_causal.py",
            ]
        )
        env["JLENS_EXP2_ANALYZE_ONLY"] = "1"
    else:
        logger.warning(
            "No lens at %s — running generation+detection only. "
            "Exp1–3 deferred until J-lens is fitted.",
            lens_path,
        )

    codes: list[tuple[str, int]] = []
    for name in scripts:
        path = ROOT / "scripts" / name
        logger.info("RUNNING %s", name)
        proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), env=env)
        codes.append((name, proc.returncode))
        if name.startswith("00_") and proc.returncode != 0:
            logger.error("Vendor setup failed — aborting")
            break
        if name.startswith("02_") and proc.returncode not in (0, 2):
            logger.error("Baseline pass failed hard")
            break
        if name.startswith("06_") and proc.returncode not in (0, 2):
            logger.info("Lens fit not available — continuing generation-only")

    # Always write report (even partial progress)
    report_proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "write_lfm_report.py")],
        cwd=str(ROOT),
        env=env,
    )
    if report_proc.returncode != 0:
        logger.warning("Report writer exited %s", report_proc.returncode)

    logger.info("done: %s", codes)
    return 0 if all(c in (0, 2) for _, c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
