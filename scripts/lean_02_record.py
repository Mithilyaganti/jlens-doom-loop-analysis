#!/usr/bin/env python
"""Record LeanScope answers and residuals at P0, P1, and optional P2.

CPU smoke (no download, no GPU):

    python scripts/lean_02_record.py --fake --out /tmp/lean_records.jsonl --activations-dir /tmp/lean_act

A real run loads the active model. It does not download a judge and it does
not call a paid API. Pass ``--record-p2`` to also store the end-of-thinking
position.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("lean_02_record")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record lean-position residuals")
    parser.add_argument("--records", type=Path, help="Input JSONL. Omit with --fake to build a synthetic bank.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--activations-dir", type=Path, required=True)
    parser.add_argument("--fake", action="store_true", help="CPU fake model with a planted lean. No download.")
    parser.add_argument("--record-p2", action="store_true", help="Also record the end-of-thinking position.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--band", type=Path, help="workspace_band JSON. Default: the active model's band file.")
    parser.add_argument("--all-layers", action="store_true", help="Record every layer, not only the workspace band.")
    parser.add_argument(
        "--hash-splits",
        action="store_true",
        help="Overwrite split from the scenario_id hash. The synthetic --fake bank keeps its balanced splits unless this is set.",
    )
    parser.add_argument(
        "--strength-scaled",
        action="store_true",
        help="Fake only. Plant a full residual for explicit leans, a weak one for indirect, and none for neutral.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from leanscope.metrics import assign_split
    from leanscope.positions import record_item, resolve_workspace_layers
    from leanscope.schema import read_records, write_records

    if args.fake:
        from leanscope.fake_stack import (
            FakeLeanStack,
            build_synthetic_records,
            install_cpu_jlens_stub,
            record_corpus,
        )

        install_cpu_jlens_stub()
        stack = FakeLeanStack()
        records = read_records(args.records) if args.records else build_synthetic_records()
        if args.hash_splits:
            records = [
                replace(record, split=assign_split(record.scenario.scenario_id)) for record in records
            ]
        if args.limit:
            records = records[: args.limit]
        # The fake stack is four layers; record all of them so a per-layer
        # AUROC can be checked. ``--all-layers`` is what a real model uses.
        recorded = record_corpus(
            stack,
            records,
            args.activations_dir,
            record_p2=args.record_p2,
            layers=list(range(stack.n_layers)),
            max_new_tokens=args.max_new_tokens,
            strength_scaled=args.strength_scaled,
        )
        write_records(args.out, recorded)
        logger.info("Wrote %d fake records to %s", len(recorded), args.out)
        return 0

    from jspace.loading import load_stack
    from jspace.model_config import artifact_paths

    records = read_records(args.records) if args.records else []
    if not records:
        logger.error("A real run needs --records. Use --fake for the CPU bank.")
        return 2
    if args.limit:
        records = records[: args.limit]
    records = [replace(record, split=assign_split(record.scenario.scenario_id)) for record in records]
    stack = load_stack()
    band = args.band
    if band is None:
        candidate = artifact_paths()["workspace_band"]
        if candidate.is_file():
            band = candidate
    workspace = resolve_workspace_layers(stack, band)
    layers = list(range(int(stack.n_layers))) if args.all_layers else workspace
    logger.info("Recording %d items at layers %s", len(records), layers)
    recorded = []
    for record in records:
        recorded.append(
            record_item(
                stack,
                record,
                activations_dir=args.activations_dir,
                workspace_layers=workspace,
                layers=layers,
                record_p2=args.record_p2,
                max_new_tokens=args.max_new_tokens,
            )
        )
    write_records(args.out, recorded)
    logger.info("Wrote %d records to %s", len(recorded), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
