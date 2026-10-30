#!/usr/bin/env python
"""Build LeanScope records JSONL from the hand-written scenario seed.

Reads data/leanscope/seed_scenarios.jsonl, expands five paraphrases and both
option orders, assigns the seeded coin-flip lean and the hash-locked split,
and writes one Record per condition. Generation is a separate step
(scripts/lean_01_generate.py). This script does not load a model and does not
call DeepSeek.

The split lock is created on first use and then treated as fixed. Pass
--update-lock only to append scenario ids that are not in the file yet.
``--preset`` only drops records after that assignment. It does not rewrite
the lock and it does not change a scenario's coin-flip lean. ``full`` is the
entire expansion. ``pilot`` and ``main`` are budget subsets. The script prints
the record count and the estimated L4-hours for the preset it writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leanscope.pipeline import build_records, manifest, write_records  # noqa: E402
from leanscope.presets import (  # noqa: E402
    PRESET_NAMES,
    apply_preset,
    estimate_run,
    format_estimate,
    get_preset,
    iter_presets,
    optional_full_natural_line,
)
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
        "--preset",
        choices=PRESET_NAMES,
        default="full",
        help="full keeps every record. pilot and main are subsets of that expansion.",
    )
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
    preset = get_preset(args.preset)
    full = build_records(seeds, lock["assignments"], lean_seed=args.lean_seed)
    for option in iter_presets():
        print(format_estimate(estimate_run(apply_preset(full, option), option)))
    records = apply_preset(full, preset)
    estimate = estimate_run(records, preset)
    n = write_records(args.out, records)
    summary = manifest(records, lean_seed=args.lean_seed)
    summary["preset"] = preset.name
    summary["estimate"] = {
        key: estimate[key]
        for key in (
            "n_records",
            "n_scenarios",
            "natural_samples",
            "natural_generations",
            "greedy_hours",
            "natural_hours",
            "total_hours",
        )
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if preset.name == "full":
        print(optional_full_natural_line(full))
    print(f"Wrote {n} records to {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
