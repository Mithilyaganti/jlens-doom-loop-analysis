#!/usr/bin/env python
"""Monitor Exp3 until complete; finalize and verify. Safe to run unattended."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "results" / "monitor.log"
DONE = ROOT / "results" / "exp3" / "stats.json"
CKPT = ROOT / "results" / "checkpoints" / "exp3_main.json"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def progress() -> int:
    if not CKPT.is_file():
        return 0
    d = json.loads(CKPT.read_text(encoding="utf-8"))
    return sum(len(v) for v in d.get("per_prompt", {}).values())


def ensure_production_running() -> None:
    """Start production runner only if no j-lens GPU exp3 worker exists."""
    import subprocess as sp

    r = sp.run(
        ['powershell', '-NoProfile', '-Command',
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*05_exp3_causal*' -and $_.ExecutablePath -like '*pythoncore*' } | Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if r.stdout.strip():
        log(f"GPU exp3 worker already running PIDs={r.stdout.strip()}")
        return
    lock = ROOT / "results" / ".exp3.lock"
    if lock.is_file():
        try:
            pid = int(lock.read_text().strip().split()[0])
            r2 = sp.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
            if str(pid) in r2.stdout:
                log(f"Lock held by PID {pid}; not spawning duplicate")
                return
        except (ValueError, OSError):
            pass
    log("No GPU worker found — launching run_production.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    sp.Popen(
        [sys.executable, "-u", str(ROOT / "scripts" / "run_production.py")],
        cwd=str(ROOT), env=env,
        stdout=open(ROOT / "results" / "production_run_console.log", "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    log("=== monitor_until_done started ===")
    last = -1
    stall = 0
    while not DONE.is_file():
        ensure_production_running()
        p = progress()
        if p == last:
            stall += 1
        else:
            stall = 0
            last = p
        log(f"progress {p}/120 stall_polls={stall}")
        if stall >= 120:  # ~60 min no progress at 30s polls
            log("ERROR: no checkpoint progress for 60 min")
            return 1
        time.sleep(30)

    log("Exp3 stats.json found — finalizing")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    for script in ("merge_trigger_tables.py", "verify_deliverables.py"):
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=str(ROOT), env=env).returncode
        log(f"{script} exit {rc}")
        if rc != 0:
            return rc
    from jspace.checkpoint import git_checkpoint
    git_checkpoint("production: phase1 verified complete")
    log("=== monitor complete — all gates passed ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
