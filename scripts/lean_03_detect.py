#!/usr/bin/env python
"""Score LeanScope detectors D1–D7 and write the results table.

CPU smoke, after ``scripts/lean_02_record.py --fake``:

    python scripts/lean_03_detect.py --fake \\
        --records /tmp/lean_records.jsonl \\
        --out /tmp/lean_detection.csv \\
        --samples 3 --n-boot 40

``--samples`` defaults to 0. A real natural-lean run uses ``--samples 20``
(twenty extra generations per item). No paid judge is called; D5 uses the
offline fake.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lean_03_detect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect leans and write a results table")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--reliability", type=Path)
    parser.add_argument("--fake", action="store_true", help="Rebuild the CPU fake stack. No download.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-shuffle", type=int, default=40)
    parser.add_argument("--n-random", type=int, default=24)
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="D6 generations per item. 0 skips sampling. The study default is 20.",
    )
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scores-out",
        type=Path,
        help="JSON map of D1 P0 band scores (item_id -> A-minus-B). Input to lean_05_panel_sim.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from leanscope.detectors import collect_scores
    from leanscope.metrics import build_results_table, write_reliability, write_results_csv, write_results_markdown
    from leanscope.schema import read_records

    records = read_records(args.records)
    if not records:
        logger.error("No records in %s", args.records)
        return 2
    if args.fake:
        from leanscope.fake_stack import FakeLeanStack, build_unigram, install_cpu_jlens_stub

        install_cpu_jlens_stub()
        stack = FakeLeanStack()
        unigram = build_unigram(stack.tokenizer)
        temperature = 0.0
    else:
        from jspace.loading import load_stack

        stack = load_stack()
        unigram = None
        temperature = args.sample_temperature
    logger.info("Scoring %d records (samples=%d)", len(records), args.samples)
    scores = collect_scores(
        stack,
        records,
        unigram=unigram,
        n_samples=args.samples,
        sample_temperature=temperature,
        n_random=args.n_random,
        seed=args.seed,
    )
    rows, extras = build_results_table(
        records,
        scores,
        n_boot=args.n_boot,
        n_shuffle=args.n_shuffle,
        seed=args.seed,
    )
    write_results_csv(args.out, rows)
    if args.scores_out:
        from leanscope.detectors import write_panel_scores

        n_scores = len(write_panel_scores(args.scores_out, scores))
        logger.info("Wrote %d D1 scores to %s", n_scores, args.scores_out)
    markdown = args.markdown or args.out.with_suffix(".md")
    reliability = args.reliability or args.out.with_suffix(".reliability.json")
    write_results_markdown(markdown, rows, csv_name=args.out.name)
    write_reliability(reliability, extras.get("reliability") or {})
    logger.info("Wrote %d rows to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
