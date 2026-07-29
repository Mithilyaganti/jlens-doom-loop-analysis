"""Periodic sync of results → Google Drive (or any backup directory).

Used on Colab so a disconnect does not lose checkpointed baseline/Exp2 work.

Env
---
JLENS_DRIVE_SYNC_DIR   Absolute path to Drive folder (required to enable sync)
JLENS_DRIVE_SYNC_EVERY How many completed prompts between full syncs (default 1)
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def drive_sync_dir() -> Path | None:
    raw = os.environ.get("JLENS_DRIVE_SYNC_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def sync_every() -> int:
    return max(1, int(os.environ.get("JLENS_DRIVE_SYNC_EVERY", "1")))


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy directory tree; skip huge residual tensors if env asks."""
    skip_heavy = os.environ.get("JLENS_DRIVE_SKIP_HEAVY", "0") == "1"
    heavy_names = {"residuals_ws.pt", "residuals_full.pt", "readouts.pt"}
    if not src.is_dir():
        return
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        if skip_heavy and path.name in heavy_names:
            continue
        rel = path.relative_to(src)
        _copy_file(path, dst / rel)


def sync_baseline_artifacts(slug: str, *, reason: str = "") -> Path | None:
    """Copy checkpoints, logs, metas, and summaries for ``slug`` to Drive.

    Always syncs lightweight JSON/CSV/jsonl and per-prompt ``meta.json``.
    Heavy ``.pt`` caches optional via JLENS_DRIVE_SKIP_HEAVY=0 (default copy all).
    """
    dest_root = drive_sync_dir()
    if dest_root is None:
        return None
    t0 = time.time()
    dest_root.mkdir(parents=True, exist_ok=True)
    results = ROOT / "results"

    # Flat critical files
    names = [
        f"checkpoints/baseline_pass_{slug}.json",
        f"trigger_gen_log_{slug}.jsonl",
        f"baseline_pass_summary_{slug}.json",
        f"trigger_tokens_{slug}.csv",
        "prompt_sample_ids.json",
        f"RUN_REPORT_{slug}_antidoom_mix_200.md",
        f"status/02_baseline_pass.md",
    ]
    for rel in names:
        src = results / rel
        if src.is_file():
            _copy_file(src, dest_root / "results" / rel)

    # Legacy unscoped names (if present)
    for rel in (
        "baseline_pass_summary.json",
        "trigger_gen_log.jsonl",
    ):
        src = results / rel
        if src.is_file():
            _copy_file(src, dest_root / "results" / rel)

    # Exp2 trees (slug-scoped + legacy)
    for exp2_name in (f"exp2_{slug}", "exp2"):
        src = results / exp2_name
        if src.is_dir():
            _copy_tree(src, dest_root / "results" / exp2_name)

    # Marker
    marker = dest_root / "LAST_SYNC.txt"
    marker.write_text(
        f"slug={slug}\nreason={reason}\nutc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n",
        encoding="utf-8",
    )
    logger.info(
        "Drive sync OK → %s (%.1fs) reason=%s",
        dest_root,
        time.time() - t0,
        reason or "periodic",
    )
    return dest_root


def restore_from_drive(slug: str) -> bool:
    """Copy Drive backup back into local results/ before a resume."""
    src_root = drive_sync_dir()
    if src_root is None or not src_root.is_dir():
        return False
    src_results = src_root / "results"
    if not src_results.is_dir():
        logger.warning("Drive sync dir has no results/: %s", src_root)
        return False
    dest = ROOT / "results"
    dest.mkdir(parents=True, exist_ok=True)

    # Prefer slug-scoped checkpoint
    copied = 0
    for path in src_results.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src_results)
        # Only restore files that look like this run (slug) or shared sample
        name = path.name
        if (
            slug in str(rel)
            or name == "prompt_sample_ids.json"
            or name.startswith(f"baseline_pass_{slug}")
            or name.startswith(f"trigger_gen_log_{slug}")
            or name.startswith(f"baseline_pass_summary_{slug}")
            or name.startswith(f"trigger_tokens_{slug}")
            or str(rel).startswith(f"exp2_{slug}")
        ):
            _copy_file(path, dest / rel)
            copied += 1
    logger.info("Restored %d files from Drive %s → local results/", copied, src_root)
    return copied > 0
