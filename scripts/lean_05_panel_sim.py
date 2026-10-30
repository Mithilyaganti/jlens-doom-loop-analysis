#!/usr/bin/env python
"""LeanScope alternatives panel and Tier A simulated-user pre-test.

Conditions: C0 plain, C1 always both sides, C2 lean-triggered, C3 disclosure
only. The other side's case is a rerun under the opposite-lean prompt. The
C2 threshold is chosen on the dev split. Analyst agents sit behind LLMClient;
the default is the offline fake, which does not call a paid API.

Every output file is labelled: pre-test, not evidence about people.

CPU smoke:

    python scripts/lean_05_panel_sim.py --smoke-fake --out /tmp/leanscope-panel

GPU run (not executed here). ``--scores`` is a JSON map of item_id to the
detector's A-minus-B score, produced by the detector package. ``--analyst
fake`` stays offline. ``--analyst deepseek`` refuses instead of sending a
request.

    python scripts/lean_05_panel_sim.py \\
        --records data/leanscope/records.jsonl \\
        --scores data/leanscope/detector_scores.json \\
        --priors data/leanscope/priors.json \\
        --out results/leanscope/panel \\
        --max-new-tokens 256

``--prior-opposite`` sets each simulated user's prior to the option the
stored recommendation did not name. That is the disagreement subset, not a
measured preference.

Expected runtime, not measured: one opposite-lean generation per eval item
that shows a case (every item in C1, plus C2 triggers), 256 tokens, on
Qwen3.5-4B. For a few hundred items that is well under an L4-hour and is
smaller than the causal script. The simulation itself is CPU-only. No GPU
result is produced by this script's tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRETEST_LABEL = "pre-test, not evidence about people"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--scores", type=Path, help="JSON map of item_id to detector score")
    parser.add_argument("--priors", type=Path, help="JSON map of item_id to 'A' or 'B'")
    parser.add_argument(
        "--prior-opposite",
        action="store_true",
        help="Prior is the option the stored recommendation did not name",
    )
    parser.add_argument("--out", type=Path, default=Path("results/leanscope/panel"))
    parser.add_argument("--dev-split", default="dev")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--analyst", default="fake", choices=("fake", "deepseek"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--smoke-fake", action="store_true")
    return parser.parse_args(argv)


def write_outputs(pick, views, report, out_dir: Path) -> None:
    from leanscope.panel import views_to_dicts

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["threshold_pick"] = {
        "threshold": pick.threshold,
        "youden_j": pick.youden_j,
        "true_positive_rate": pick.true_positive_rate,
        "false_positive_rate": pick.false_positive_rate,
        "n_dev": pick.n_dev,
        "n_lean": pick.n_lean,
        "n_no_lean": pick.n_no_lean,
    }
    (out_dir / "simulation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "views.jsonl").open("w", encoding="utf-8") as handle:
        for row in views_to_dicts(views):
            handle.write(json.dumps(row) + "\n")
    lines = [
        PRETEST_LABEL,
        "",
        "# LeanScope Tier A panel simulation",
        "",
        "These numbers are a pre-test of the pipeline. They are not evidence about people.",
        "",
        f"Dev threshold |score| >= {pick.threshold} (Youden J={pick.youden_j}).",
        "",
        "## Follow rate and drop vs C0",
        "",
    ]
    for condition, rate in report.follow_rate.items():
        lines.append(
            f"- {condition}: follow {rate} (Wilson {report.follow_rate_wilson_ci[condition]}), "
            f"drop vs C0 {report.drop_vs_c0[condition]}, "
            f"extra tokens {report.extra_tokens[condition]}"
        )
    lines.append("")
    lines.append(
        f"False-trigger rate on no-lean items: {report.false_trigger_rate} "
        f"({report.false_trigger_count}/{report.n_no_lean}), "
        f"Wilson {report.false_trigger_wilson_ci}."
    )
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _priors_from_flag(records, args: argparse.Namespace) -> dict[str, str]:
    if args.priors is not None:
        payload = json.loads(args.priors.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in payload.items()}
    if args.prior_opposite:
        priors = {}
        for record in records:
            if record.split != args.eval_split or record.recommended not in ("A", "B"):
                continue
            priors[record.item_id] = "B" if record.recommended == "A" else "A"
        return priors
    raise SystemExit("pass --priors or --prior-opposite")


def run_smoke(out_dir: Path) -> dict:
    from leanscope.schema import LeanCondition, Record, Scenario
    from leanscope.simulate import make_client, run_tier_a, template_case

    def item(item_id, split, kind, lean, recommended="A"):
        system = "LEAN_A" if lean == "A" else ""
        return Record(
            item_id=item_id,
            scenario=Scenario(
                scenario_id=item_id,
                family="analysis",
                question=f"Which measure should {item_id} use?",
                option_a="outcome alpha",
                option_b="outcome beta",
                paraphrase_id="0",
                order="AB",
            ),
            condition=LeanCondition(
                kind=kind,
                strength="explicit" if lean else None,
                lean=lean,
                system_prompt=system,
            ),
            output_text=f"Answer: {recommended}",
            recommended=recommended,
            voiced=True if lean else False,
            activations_path=None,
            split=split,
        )

    records = []
    scores = {}
    # Dev: three no-lean, three explicit leans. Threshold should land at 0.7.
    for index, score in enumerate((0.05, 0.1, 0.2)):
        row = item(f"dev-none-{index}", "dev", "none", None)
        records.append(row)
        scores[row.item_id] = score
    for index, score in enumerate((0.7, 0.8, 0.9)):
        row = item(f"dev-lean-{index}", "dev", "assigned", "A")
        records.append(row)
        scores[row.item_id] = score
    for index in range(4):
        row = item(f"test-lean-{index}", "test", "assigned", "A")
        records.append(row)
        scores[row.item_id] = 0.95
    for index, score in enumerate((0.95, 0.95, 0.1, 0.1)):
        row = item(f"test-none-{index}", "test", "none", None, recommended="A")
        records.append(row)
        scores[row.item_id] = score
    priors = {f"test-lean-{index}": "B" for index in range(4)}
    pick, views, report = run_tier_a(
        records,
        scores,
        priors,
        case_fn=template_case,
        client=make_client("fake"),
    )
    write_outputs(pick, views, report, out_dir)
    return {"pick": pick, "report": report}


def model_case_fn(max_new_tokens: int):
    """Rerun the loaded study model under the opposite-lean prompt.

    The stack is loaded on the first call. This is the GPU path.
    """

    def case_fn(scenario, target: str):
        from jspace.generation import generate_greedy
        from jspace.loading import load_stack

        from leanscope.prompts import opposite_lean_system, render_prompt

        stack = getattr(case_fn, "_stack", None)
        if stack is None:
            stack = load_stack(require_lens=False, download_if_missing=False)
            case_fn._stack = stack
        prompt = render_prompt(scenario, opposite_lean_system(scenario, target))
        generated = generate_greedy(stack, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        return generated["generated_text"], len(generated["generated_ids"])

    return case_fn


def run_records(args: argparse.Namespace):
    from leanscope.panel import load_score_map
    from leanscope.schema import read_jsonl
    from leanscope.simulate import make_client, run_tier_a

    if args.records is None or args.scores is None:
        raise SystemExit("pass --records and --scores, or --smoke-fake")
    if args.analyst == "deepseek":
        raise SystemExit(
            "DeepSeek V4.1 Flash is disabled. The analyst stays on --analyst fake, "
            "which does not call a paid API."
        )
    records = read_jsonl(args.records)
    scores = load_score_map(json.loads(args.scores.read_text(encoding="utf-8")))
    priors = _priors_from_flag(records, args)
    pick, views, report = run_tier_a(
        records,
        scores,
        priors,
        case_fn=model_case_fn(args.max_new_tokens),
        client=make_client("fake"),
        dev_split=args.dev_split,
        eval_split=args.eval_split,
    )
    write_outputs(pick, views, report, args.out)
    return pick, report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_fake:
        result = run_smoke(args.out)
        report = result["report"]
        pick = result["pick"]
    else:
        pick, report = run_records(args)
    print(PRETEST_LABEL)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "threshold": pick.threshold,
                "follow_rate": report.follow_rate,
                "drop_vs_c0": report.drop_vs_c0,
                "false_trigger_rate": report.false_trigger_rate,
                "extra_tokens": report.extra_tokens,
                "label": report.label,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
