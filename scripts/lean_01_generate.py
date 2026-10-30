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

``--preset`` filters the records file down to a budget subset. ``full`` keeps
every row in the file. ``pilot`` and ``main`` drop rows only; splits and
coin-flip leans on the rows that remain are whatever the file already stored.
The script prints the record count and the estimated L4-hours before it loads
a model.

The full expansion is 100 scenarios x 5 paraphrases x 2 orders x 14 conditions
= 14000 greedy generations. A rough estimate, not a measurement: at about 50
tokens/s of decode on an L4, 300 new tokens is about 23 hours of decode, and
recording activations adds another forward, so budget about 35 L4-hours (the
30-to-40 band). Add ``--natural-samples 20`` only on a second pass over the
full file. That draws 20 samples for each none and trait row (40000
generations, about 67 hours of decode) and brings the total to about 102
L4-hours. ``--preset pilot`` is about 5 L4-hours and ``--preset main`` is
about 18, including their natural-lean samples. The script resumes from --out
and from the natural-sample JSONL.
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
from leanscope.presets import (  # noqa: E402
    PRESET_NAMES,
    apply_preset,
    estimate_run,
    format_estimate,
    get_preset,
    natural_sample_ids,
    optional_full_natural_line,
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
    parser.add_argument(
        "--preset",
        choices=PRESET_NAMES,
        default="full",
        help="full keeps every record in the file. pilot and main subset it.",
    )
    parser.add_argument(
        "--natural-samples",
        type=int,
        default=None,
        help="Override the preset's natural-lean sample count. The default follows the preset.",
    )
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args(argv)

    preset = get_preset(args.preset)
    fresh = apply_preset(read_jsonl(args.records), preset)
    if args.out.is_file() and args.out.resolve() != args.records.resolve():
        fresh = merge_existing(fresh, read_jsonl(args.out))
    elif args.out.is_file() and args.out.resolve() == args.records.resolve():
        fresh = apply_preset(read_jsonl(args.out), preset)

    natural_samples = preset.natural_samples if args.natural_samples is None else args.natural_samples
    sample_ids = set(natural_sample_ids(fresh, preset)) if natural_samples else set()
    estimate = estimate_run(fresh, preset, natural_samples=natural_samples)
    print(format_estimate(estimate))
    if preset.name == "full" and natural_samples == 0:
        print(optional_full_natural_line(fresh))

    natural_path = args.out.with_suffix(".natural.jsonl")
    report_path = args.out.with_suffix(".natural_report.json")
    already = _load_sample_ids(natural_path)

    from jspace.loading import load_stack

    stack = load_stack()
    config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        natural_samples=natural_samples,
        natural_item_ids=None if natural_samples == 0 else sample_ids,
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
