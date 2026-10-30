#!/usr/bin/env python
"""Build LeanScope records JSONL from the hand-written scenario seed.

Reads data/leanscope/seed_scenarios.jsonl, expands five paraphrases and both
option orders, assigns the seeded coin-flip lean and the hash-locked split,
and writes one Record per condition. Generation is a separate step
(scripts/lean_01_generate.py). This script does not load a model and does not
call DeepSeek.

The split lock is created on first use and then treated as fixed. Pass
--update-lock only to append scenario ids that are not in the file yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leanscope.pipeline import build_records, manifest, write_records  # noqa: E402
from leanscope.scenarios import DEFAULT_SEED_PATH, load_seeds  # noqa: E402
from leanscope.splits import DEFAULT_LOCK_PATH, resolve_lock  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LeanScope item records.")
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "leanscope" / "records.jsonl",
    )
    parser.add_argument("--lean-seed", type=int, default=42)
    parser.add_argument(
        "--update-lock",
        action="store_true",
        help="Append scenario ids that are missing from the split lock. Does not move existing ids.",
    )
    args = parser.parse_args(argv)

    seeds = load_seeds(args.seed)
    lock = resolve_lock(
        args.lock,
        [seed.scenario_id for seed in seeds],
        update=args.update_lock,
    )
    records = build_records(seeds, lock["assignments"], lean_seed=args.lean_seed)
    n = write_records(args.out, records)
    summary = manifest(records, lean_seed=args.lean_seed)
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {n} records to {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
