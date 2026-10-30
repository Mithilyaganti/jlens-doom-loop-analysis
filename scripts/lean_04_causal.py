#!/usr/bin/env python
"""LeanScope causal test: steer and ablate the option direction.

Paired conditions, mirroring scripts/05_exp3_causal.py:

- baseline: lean prompt, no hook
- steer_option / ablate_option at P0 and at P1
- steer_random / ablate_random: norm-matched random direction
- remove_lean_prompt: positive control (regenerate with the lean prompt gone)

The direction defaults to the difference of means (lean A minus lean B) on
the train split. A detector plugs in by calling leanscope.causal.run_causal
with its own direction_fn(records, *, layer, position).

CPU smoke (no GPU, no download):

    python scripts/lean_04_causal.py --smoke-fake --out /tmp/leanscope-causal

GPU run (not executed here). Needs Qwen3.5-4B, a JSONL of Records, and
activation files on the train split (torch dicts with P0 and P1 residuals
shaped [n_layers, d_model]):

    python scripts/lean_04_causal.py \\
        --records data/leanscope/records.jsonl \\
        --split test \\
        --layers 18,20,22 \\
        --positions P0,P1 \\
        --alpha 1.0 \\
        --max-new-tokens 256 \\
        --out results/leanscope/causal

Expected runtime, not measured: a test split of about 100 items and ten
greedy continuations each (baseline, positive control, and four edits at
two positions) is on the order of a thousand short generations. On
Qwen3.5-4B at 256 new tokens that is a few L4-hours, the same kind of pass
as one Exp3 condition sweep. It fits the plan's 2–4 L4-hour envelope for
the whole model side only if the test split stays near a hundred items.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, help="JSONL of leanscope Record rows")
    parser.add_argument("--out", type=Path, default=Path("results/leanscope/causal"))
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--layers", default="0", help="Comma-separated residual layers to edit")
    parser.add_argument("--positions", default="P0,P1")
    parser.add_argument("--alpha", type=float, default=1.0, help="Steer magnitude on the unit direction")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--neutral-system",
        default="",
        help="System prompt for the positive control. Empty removes the lean prompt.",
    )
    parser.add_argument(
        "--smoke-fake",
        action="store_true",
        help="CPU smoke on TinyFlipModel. Not a result about Qwen.",
    )
    return parser.parse_args(argv)


def _layers(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _positions(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def write_outputs(report: dict, out_dir: Path) -> None:
    from leanscope.causal import json_ready

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json_ready(report)
    (out_dir / "stats.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "per_item.jsonl").open("w", encoding="utf-8") as handle:
        for row in payload["outcomes"]:
            handle.write(json.dumps(row) + "\n")
    lines = [
        "# LeanScope causal",
        "",
        f"Protocol: {report['protocol']}. Direction: {report['direction']}.",
        f"Norm matching: {report['norm_matching']}.",
        f"Positive control: {report['positive_control']}.",
        "",
        "## Flip rate, exact McNemar, Wilson interval, Cohen's h",
        "",
    ]
    for key, stats in report["contrasts"].items():
        lines.append(
            f"- `{key}`: flip rate {stats['flip_rate']} "
            f"(Wilson {stats['flip_rate_wilson_ci']}), "
            f"McNemar p={stats['p_exact']} "
            f"(b={stats['b_baseline_follows_only']}, c={stats['c_condition_follows_only']}), "
            f"Cohen's h={stats['cohens_h']}, "
            f"follow {stats['rate_baseline_follows']} -> {stats['rate_condition_follows']}"
        )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_smoke(out_dir: Path) -> dict:
    """Four leaned items on the planted-direction test double."""
    import torch

    from leanscope.causal import run_causal
    from leanscope.schema import LeanCondition, Record, Scenario
    from leanscope.tiny_model import TinyFlipModel, option_token_ids, tiny_tokenize

    planted = torch.zeros(32)
    planted[0] = 1.0
    model = TinyFlipModel(planted, decision_at="P0")

    class _Stack:
        tokenizer = None

        def __init__(self, module):
            self.model = module
            self.d_model = int(module.planted.numel())
            self.n_layers = 1
            self.device = torch.device("cpu")

        @property
        def layers(self):
            return self.model.layers

    records = []
    for index in range(4):
        records.append(
            Record(
                item_id=f"smoke-{index}",
                scenario=Scenario(
                    scenario_id=f"s{index}",
                    family="advice",
                    question=f"Which plan should be adopted in case {index}?",
                    option_a="alpha plan",
                    option_b="beta plan",
                    paraphrase_id="0",
                    order="AB",
                ),
                condition=LeanCondition(
                    kind="assigned",
                    strength="explicit",
                    lean="A",
                    system_prompt="LEAN_A",
                ),
                output_text="Answer: A",
                recommended="A",
                voiced=True,
                activations_path=None,
                split="test",
            )
        )

    def direction_fn(rows, *, layer: int, position: str):
        del rows, layer, position
        return planted.clone()

    report = run_causal(
        _Stack(model),
        records,
        direction_fn=direction_fn,
        layers=[0],
        positions=["P0"],
        alpha=1.2,
        max_new_tokens=2,
        backend="logits",
        tokenize=tiny_tokenize,
        option_token_ids=option_token_ids(),
    )
    report["note"] = "CPU smoke with TinyFlipModel. Not a Qwen result."
    write_outputs(report, out_dir)
    return report


def run_gpu(args: argparse.Namespace) -> dict:
    from jspace.loading import load_stack

    from leanscope.causal import run_causal
    from leanscope.schema import read_jsonl

    if args.records is None:
        raise SystemExit("pass --records, or --smoke-fake for the CPU test double")
    records = read_jsonl(args.records)
    train = [row for row in records if row.split == "train"]
    evaluation = [row for row in records if row.split == args.split]
    if not train:
        raise SystemExit("difference of means needs train-split records with activations")
    if not evaluation:
        raise SystemExit(f"no records in split {args.split}")
    stack = load_stack(require_lens=False, download_if_missing=False)
    report = run_causal(
        stack,
        evaluation,
        direction_records=train,
        layers=_layers(args.layers),
        positions=_positions(args.positions),
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        backend="hf",
        neutral_system_prompt=args.neutral_system,
    )
    write_outputs(report, args.out)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_fake:
        report = run_smoke(args.out)
    else:
        report = run_gpu(args)
    steer = report["contrasts"].get("steer_option@P0", {})
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_items": report["n_items"],
                "steer_option_P0_flip_rate": steer.get("flip_rate"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
