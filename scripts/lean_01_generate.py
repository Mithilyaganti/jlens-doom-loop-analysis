#!/usr/bin/env python
"""Generate LeanScope replies with the repo's model stack.

Uses jspace.loading.load_stack and jspace.generation (chat template, greedy
generation, and the tiered activation cache when a lens is loaded).

CPU tests do not run this script. They call leanscope.pipeline.generate_records
with a fake module. This entry point is the GPU run.

Example (Qwen3.5-4B, bf16, after the records file exists):

    JLENS_MODEL=Qwen/Qwen3.5-4B JLENS_HF_QUANTIZE=0 \\
      python scripts/lean_01_generate.py \\
        --records data/leanscope/records.jsonl \\
        --out results/leanscope/records_generated.jsonl \\
        --activations-dir results/leanscope/activations \\
        --max-new-tokens 300

The seed expansion is 12 scenarios x 5 paraphrases x 2 orders x 14 conditions
= 1680 greedy generations. A rough estimate, not a measurement: at about 50
tokens/s of decode on an L4, 300 new tokens is about 3 hours of decode, and
recording activations adds another forward, so budget about 4 to 6 L4-hours.
Add ``--natural-samples 20`` only on a second pass. That draws 20 samples for
each none and trait row (4800 generations) and is several more L4-hours at the
same rate. The script resumes from --out and from the natural-sample JSONL.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leanscope.labels import agreement_report  # noqa: E402
from leanscope.pipeline import (  # noqa: E402
    GenerateConfig,
    generate_records,
    merge_existing,
)
from leanscope.schema import read_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lean_generate")


def _load_sample_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            item_id = row.get("item_id")
            if item_id:
                ids.add(str(item_id))
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate LeanScope replies.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--activations-dir", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--natural-samples", type=int, default=0)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args(argv)

    from jspace.loading import load_stack

    fresh = read_jsonl(args.records)
    if args.out.is_file() and args.out.resolve() != args.records.resolve():
        fresh = merge_existing(fresh, read_jsonl(args.out))
    elif args.out.is_file():
        fresh = read_jsonl(args.out)

    natural_path = args.out.with_suffix(".natural.jsonl")
    report_path = args.out.with_suffix(".natural_report.json")
    already = _load_sample_ids(natural_path)

    stack = load_stack()
    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        natural_samples=args.natural_samples,
        sample_temperature=args.sample_temperature,
        activations_dir=args.activations_dir,
        limit=args.limit,
    )
    records, natural_results, natural_rows = generate_records(
        fresh,
        stack,
        config,
        already_sampled=already,
        checkpoint_path=args.out,
        checkpoint_every=args.checkpoint_every,
    )
    if natural_rows:
        natural_path.parent.mkdir(parents=True, exist_ok=True)
        with natural_path.open("a", encoding="utf-8") as handle:
            for row in natural_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if natural_results:
        report = agreement_report(natural_results)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Natural-lean report: %s", json.dumps(report))

    n_filled = sum(1 for row in records if row.output_text.strip())
    n_parsed = sum(1 for row in records if row.recommended is not None)
    logger.info("Records with text: %d / %d; parsed recommendations: %d", n_filled, len(records), n_parsed)
    print(f"Wrote {args.out} ({n_filled} filled, {n_parsed} parsed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
