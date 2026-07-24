"""Checkpoint save/load for long-running experiment scripts."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = ROOT / "results" / "checkpoints"


def checkpoint_path(name: str) -> Path:
    return CHECKPOINTS / f"{name}.json"


def save_checkpoint(name: str, data: dict[str, Any]) -> Path:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    path = checkpoint_path(name)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)
    logger.info("Checkpoint saved: %s", path)
    return path


def load_checkpoint(name: str) -> dict[str, Any] | None:
    path = checkpoint_path(name)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def git_checkpoint(message: str) -> bool:
    """Create a git commit checkpoint if repo is initialized."""
    import subprocess

    root = ROOT
    if not (root / ".git").is_dir():
        return False
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(root), check=True, capture_output=True)
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        if not status.stdout.strip():
            logger.info("Git checkpoint skipped (clean tree): %s", message)
            return True
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(root),
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Git checkpoint: %s", message)
        return True
    except subprocess.CalledProcessError as e:
        logger.warning("Git checkpoint failed: %s", e.stderr or e)
        return False


def init_git_repo() -> None:
    import subprocess

    if (ROOT / ".git").is_dir():
        return
    subprocess.run(["git", "init"], cwd=str(ROOT), check=True, capture_output=True)
    logger.info("Initialized git repository at %s", ROOT)
