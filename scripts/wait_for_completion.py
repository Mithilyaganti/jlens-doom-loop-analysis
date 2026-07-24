#!/usr/bin/env python
"""Wait for Exp3 completion; resume ONE worker if crashed; finalize when done.

Does NOT spawn duplicate GPU jobs. Read-only polling when worker alive.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DONE = ROOT / "results" / "exp3" / "stats.json"
CKPT = ROOT / "results" / "checkpoints" / "exp3_main.json"
LOG = ROOT / "results" / "wait_for_completion.log"
LOCK = ROOT / "results" / ".exp3.lock"


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


def gpu_worker_pid() -> int | None:
    r = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -like '*05_exp3_causal*' -and $_.ExecutablePath -like '*pythoncore*' } | "
            "Select-Object -ExpandProperty ProcessId -First 1",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    s = r.stdout.strip()
    if s.isdigit():
        return int(s)
    return None


def production_blocked() -> bool:
    r = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object { $_.CommandLine -like '*run_production*' -and $_.ExecutablePath -like '*pythoncore*' } | "
            "Measure-Object | Select-Object -ExpandProperty Count",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    try:
        return int(r.stdout.strip()) > 0
    except ValueError:
        return False


def start_single_exp3() -> None:
    log("Starting ONE Exp3 worker (resume from checkpoint)")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("JLENS_EXP3_PROMPTS", "20")
    env.setdefault("JLENS_MAX_NEW_TOKENS", "1536")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "scripts" / "05_exp3_causal.py")],
        cwd=str(ROOT),
        env=env,
        stdout=open(ROOT / "results" / "exp3_run.log", "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )


def finalize() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    for script in ("merge_trigger_tables.py", "verify_deliverables.py"):
        log(f"Running {script}")
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=str(ROOT), env=env).returncode
        if rc != 0:
            log(f"{script} failed with {rc}")
            return rc
    sys.path.insert(0, str(ROOT))
    from jspace.checkpoint import git_checkpoint

    git_checkpoint("production: phase1 verified complete")
    log("Finalization complete")
    return 0


def main() -> int:
    log("wait_for_completion started")
    last = -1
    stall = 0
    while not DONE.is_file():
        p = progress()
        worker = gpu_worker_pid()
        prod = production_blocked()
        log(f"progress={p}/120 worker={worker} production={prod} stall={stall}")

        if worker is None and not prod:
            if LOCK.is_file():
                try:
                    pid = int(LOCK.read_text().strip().split()[0])
                    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
                    if str(pid) in r.stdout and "python" in r.stdout.lower():
                        log(f"lock pid {pid} alive but not gpu worker — waiting")
                    else:
                        LOCK.unlink(missing_ok=True)
                        start_single_exp3()
                except (ValueError, OSError):
                    start_single_exp3()
            else:
                start_single_exp3()

        if p == last:
            stall += 1
        else:
            stall = 0
            last = p

        if stall >= 40 and worker is None:  # ~20 min no progress, no worker
            log("ERROR: stalled with no worker")
            return 1

        time.sleep(30)

    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
