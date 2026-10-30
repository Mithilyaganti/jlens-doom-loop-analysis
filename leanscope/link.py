"""Connect the LeanScope stages on one record file.

Order: data records -> residual recording -> D1 scores -> causal steering
along D1's option direction -> alternatives panel triggered by that score.

The CPU path uses :class:`leanscope.fake_stack.FakeLeanStack`. A real Qwen
run uses the scripts in ``scripts/lean_run_all.sh`` and does not import this
module's fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from leanscope.schema import Record


def _auroc_for(records: Sequence[Record], scores: dict[str, float], strength: str) -> float:
    from leanscope.metrics import safe_auroc

    rows = [
        row
        for row in records
        if row.condition.kind == "assigned" and row.condition.strength == strength and row.item_id in scores
    ]
    labels = [1.0 if row.condition.lean == "A" else 0.0 for row in rows]
    values = [scores[row.item_id] for row in rows]
    return float(safe_auroc(labels, values))


def _mean_abs(records: Sequence[Record], scores: dict[str, float], *, kind: str, strength: str | None) -> float:
    chosen = []
    for row in records:
        if row.item_id not in scores or row.condition.kind != kind:
            continue
        if strength is not None and row.condition.strength != strength:
            continue
        chosen.append(abs(scores[row.item_id]))
    if not chosen:
        return float("nan")
    return float(sum(chosen) / len(chosen))


def run_cpu_chain(work: Path | str, *, noise_std: float = 0.40) -> dict[str, Any]:
    """Run data, record, detect, causal, and the panel on CPU.

    The fake plants a full residual for an explicit lean, a weak residual for
    an indirect lean, and nothing for a structure-matched neutral prompt.
    Causal steering uses D1's J-lens option direction. The panel threshold
    is fit on the dev split of those D1 scores.
    """
    from leanscope.fake_stack import (
        OPTION_A,
        OPTION_B,
        QUESTIONS,
        FakeLeanStack,
        build_unigram,
        install_cpu_jlens_stub,
        record_corpus,
        system_prompt_for,
    )

    install_cpu_jlens_stub()

    from leanscope.causal import run_causal
    from leanscope.detectors import collect_scores, item_scores, make_jlens_direction_fn, write_panel_scores
    from leanscope.metrics import build_results_table
    from leanscope.pipeline import build_records
    from leanscope.scenarios import load_seeds
    from leanscope.schema import Record, Scenario, read_jsonl, write_jsonl
    from leanscope.simulate import make_client, run_tier_a, template_case
    from leanscope.splits import load_lock

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    activations = work / "activations"
    stack = FakeLeanStack(noise_std=noise_std)

    seeds = load_seeds()
    lock = load_lock()
    seed = seeds[0]
    assignment = {seed.scenario_id: lock["assignments"][seed.scenario_id]}
    built = build_records([seed], assignment)
    data_slice = [
        row
        for row in built
        if row.scenario.paraphrase_id == "p0"
        and row.scenario.order == "AB"
        and (
            (row.condition.kind == "assigned" and row.condition.strength in ("explicit", "indirect"))
            or (row.condition.kind == "neutral_matched" and row.condition.strength == "explicit")
        )
    ]
    data_path = work / "data_slice.jsonl"
    write_jsonl(data_path, data_slice)
    data_slice = read_jsonl(data_path)

    bank = _scaled_bank(OPTION_A, OPTION_B, QUESTIONS, system_prompt_for)
    bank_path = work / "bank.jsonl"
    write_jsonl(bank_path, bank)
    bank = read_jsonl(bank_path)

    recorded_bank = record_corpus(
        stack,
        bank,
        activations,
        layers=list(range(stack.n_layers)),
        max_new_tokens=8,
        strength_scaled=True,
    )
    recorded_data = record_corpus(
        stack,
        data_slice,
        activations,
        layers=list(range(stack.n_layers)),
        max_new_tokens=8,
        strength_scaled=True,
    )
    write_jsonl(work / "recorded.jsonl", recorded_bank)

    scores_rows = collect_scores(
        stack,
        recorded_bank,
        unigram=build_unigram(stack.tokenizer),
        n_samples=0,
        n_random=4,
        seed=0,
    )
    table, _extras = build_results_table(recorded_bank, scores_rows, n_boot=20, n_shuffle=8, seed=0)
    d1 = item_scores(scores_rows)
    write_panel_scores(work / "d1_scores.json", scores_rows)

    data_rows = collect_scores(
        stack,
        recorded_data,
        unigram=build_unigram(stack.tokenizer),
        n_samples=0,
        n_random=0,
        seed=0,
    )
    data_scores = item_scores(data_rows)

    for layer in stack.model.layers:
        layer.noise_std = 0.0
    causal_items = [
        row
        for row in recorded_bank
        if row.split == "test" and row.condition.kind == "assigned" and row.condition.strength == "explicit"
    ]
    causal = run_causal(
        stack,
        causal_items,
        direction_fn=make_jlens_direction_fn(stack),
        per_item_direction=True,
        direction_name="D1_jlens",
        layers=[1],
        positions=["P0"],
        alpha=4.0,
        max_new_tokens=6,
        backend="hf",
    )

    priors = {
        row.item_id: ("B" if row.condition.lean == "A" else "A")
        for row in recorded_bank
        if row.split == "test" and row.condition.kind == "assigned"
    }
    panel_records = [row for row in recorded_bank if row.split in ("dev", "test")]
    pick, views, report = run_tier_a(
        panel_records,
        d1,
        priors,
        case_fn=template_case,
        client=make_client("fake"),
    )
    triggered = [
        views[row.item_id]["C2"].triggered
        for row in recorded_bank
        if row.split == "test" and row.condition.kind == "assigned" and row.condition.strength == "explicit"
    ]
    return {
        "n_data": len(recorded_data),
        "data_activations": [row.activations_path for row in recorded_data],
        "data_scores": data_scores,
        "explicit_auroc": _auroc_for(recorded_bank, d1, "explicit"),
        "indirect_auroc": _auroc_for(recorded_bank, d1, "indirect"),
        "mean_abs_explicit": _mean_abs(recorded_bank, d1, kind="assigned", strength="explicit"),
        "mean_abs_indirect": _mean_abs(recorded_bank, d1, kind="assigned", strength="indirect"),
        "mean_abs_neutral": _mean_abs(recorded_bank, d1, kind="neutral_matched", strength="explicit"),
        "n_table_rows": len(table),
        "causal": causal,
        "threshold": pick.threshold,
        "follow_rate": report.follow_rate,
        "drop_vs_c0": report.drop_vs_c0,
        "false_trigger_rate": report.false_trigger_rate,
        "explicit_test_triggered": triggered,
        "label": report.label,
        "scores_path": str(work / "d1_scores.json"),
    }


def _scaled_bank(option_a: str, option_b: str, questions: dict[str, str], system_prompt_for) -> list[Record]:
    """Both leans, with distinct prompt tokens so the fake's noise is not shared."""
    from leanscope.schema import Scenario

    markers = ("cat", "dog", "elk", "fox", "hen", "ram", "owl", "yak")
    families = ("analysis", "product", "advice")
    rows: list[Record] = []
    cursor = 0

    def add(kind: str, strength: str | None, lean: str | None, split: str, family: str) -> None:
        nonlocal cursor
        scenario_id = f"e2e-{kind}-{strength or 'na'}-{lean or 'na'}-{cursor}"
        question = f"{questions[family]} {markers[cursor % 8]} {markers[(cursor * 3) % 8]}"
        cursor += 1
        rows.append(
            Record(
                item_id=f"{scenario_id}-AB",
                scenario=Scenario(
                    scenario_id=scenario_id,
                    family=family,  # type: ignore[arg-type]
                    question=question,
                    option_a=option_a,
                    option_b=option_b,
                    paraphrase_id="p0",
                    order="AB",
                ),
                condition=_condition(kind, strength, lean, system_prompt_for),
                output_text="",
                recommended=None,
                voiced=None,
                activations_path=None,
                split=split,  # type: ignore[arg-type]
            )
        )

    for lean in ("A", "B"):
        for index in range(8):
            if index < 3:
                split = "dev"
            elif index < 6:
                split = "test"
            else:
                split = "train"
            add("assigned", "explicit", lean, split, families[index % 3])
        for index in range(12):
            add("assigned", "indirect", lean, "train", families[index % 3])
    for index in range(8):
        add("neutral_matched", "explicit", None, "dev" if index < 4 else "test", families[index % 3])
    return rows


def _condition(kind: str, strength: str | None, lean: str | None, system_prompt_for):
    from leanscope.schema import LeanCondition

    return LeanCondition(
        kind=kind,  # type: ignore[arg-type]
        strength=strength,  # type: ignore[arg-type]
        lean=lean,  # type: ignore[arg-type]
        system_prompt=system_prompt_for(kind, strength, lean),
    )
