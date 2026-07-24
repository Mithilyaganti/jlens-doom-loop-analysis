#!/usr/bin/env python
"""Initialize git repo and create a checkpoint commit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jspace.checkpoint import init_git_repo, git_checkpoint


def main() -> int:
    init_git_repo()
    ok = git_checkpoint("checkpoint: pipeline state save")
    print("checkpoint ok" if ok else "checkpoint skipped or failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
