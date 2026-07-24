"""Single-instance lock to prevent duplicate GPU experiment processes."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "results" / ".exp3.lock"


def acquire_lock(name: str = "exp3") -> bool:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.is_file():
        try:
            old_pid = int(LOCK.read_text().strip().split()[0])
            # Check if process still alive (Windows)
            import subprocess

            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {old_pid}", "/NH"],
                capture_output=True,
                text=True,
            )
            if str(old_pid) in r.stdout and "python" in r.stdout.lower():
                return False
        except (ValueError, OSError):
            pass
    LOCK.write_text(f"{os.getpid()} {name}\n", encoding="utf-8")
    return True


def release_lock() -> None:
    if LOCK.is_file():
        try:
            if LOCK.read_text().strip().startswith(str(os.getpid())):
                LOCK.unlink(missing_ok=True)
        except OSError:
            pass
