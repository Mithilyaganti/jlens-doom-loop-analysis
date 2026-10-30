"""CPU tests for the alternatives panel and the Tier A pre-test.

The analyst is the offline fake. Reports must carry the label
'pre-test, not evidence about people'. No network and no model download.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace.stats import wilson_ci  # noqa: E402
from leanscope.panel import (  # noqa: E402
    DISCLOSURE_LINE,
    OTHER_CASE_HEADER,
    build_views_for_item,
    pick_threshold,
)
from leanscope.schema import LeanCondition, Record, Scenario  # noqa: E402
from leanscope.simulate import (  # noqa: E402
    PRETEST_LABEL,
    DeepSeekFlashClient,
    OfflineFakeAnalyst,
    PaidAPIDisabled,
    make_client,
    run_tier_a,
    template_case,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _condition(kind: str, lean: str | None) -> LeanCondition:
    """Match the shared schema: neutral carries a strength and a non-empty prompt."""
    if kind == "neutral_matched":
        return LeanCondition(
            kind="neutral_matched",
            strength="explicit",
            lean=None,
            system_prompt="Consider both options equally.",
        )
    if kind in ("none", "trait"):
        return LeanCondition(kind=kind, strength=None, lean=None, system_prompt="")
    return LeanCondition(
        kind=kind,
        strength="explicit",
        lean=lean,
        system_prompt="favour A" if lean == "A" else "favour B",
    )


def _record(
    item_id: str,
    *,
    split: str,
    kind: str,
    lean: str | None,
    recommended: str | None = "A",
    order: str = "AB",
) -> Record:
    return Record(
        item_id=item_id,
        scenario=Scenario(
            scenario_id=item_id,
            family="analysis",
            question=f"Which measure fits {item_id}?",
            option_a="outcome alpha",
            option_b="outcome beta",
            paraphrase_id="0",
            order=order,
        ),
        condition=_condition(kind, lean),
        output_text="Answer: A" if recommended == "A" else "Answer: B",
        recommended=recommended,
        voiced=bool(lean),
        activations_path=None,
        split=split,
    )


def test_pick_threshold_youden_and_tie_break():
    scores = [0.1, 0.2, 0.05, 0.9, 0.8, 0.7]
    labels = [False, False, False, True, True, True]
    pick = pick_threshold(scores, labels)
    assert pick.threshold == 0.7
    assert pick.youden_j == 1.0
    assert pick.true_positive_rate == 1.0
    assert pick.false_positive_rate == 0.0
    assert pick.n_lean == 3
    assert pick.n_no_lean == 3

    # Equal Youden: |score| >= 0.4 has J=0.5, |score| >= 0.9 has J=0.5 and a lower FPR.
    tied = pick_threshold([0.4, 0.9, 0.4, 0.1], [True, True, False, False])
    assert tied.threshold == 0.9
    assert abs(tied.youden_j - 0.5) < 1e-12
    assert tied.false_positive_rate == 0.0


def test_conditions_show_the_other_case_only_when_they_should():
    record = _record("item", split="test", kind="assigned", lean="A")
    calls: list[str] = []

    def case_fn(scenario, target: str) -> str:
        calls.append(target)
        assert scenario.scenario_id == "item"
        return "beta matters"

    quiet = build_views_for_item(
        record, score=0.1, threshold=0.7, case_fn=case_fn, conditions=("C0", "C2", "C3")
    )
    assert calls == []
    assert quiet["C0"].other_case is None
    assert OTHER_CASE_HEADER not in quiet["C0"].display_text
    assert quiet["C2"].triggered is False
    assert quiet["C2"].other_case is None
    assert quiet["C3"].disclosure == DISCLOSURE_LINE
    assert quiet["C3"].other_case is None
    assert DISCLOSURE_LINE in quiet["C3"].display_text
    assert quiet["C0"].n_tokens < quiet["C3"].n_tokens

    fired = build_views_for_item(
        record, score=-0.7, threshold=0.7, case_fn=case_fn, conditions=("C1", "C2")
    )
    assert calls == ["B"]
    assert fired["C1"].other_case == "beta matters"
    assert fired["C2"].triggered is True
    assert fired["C2"].other_case == "beta matters"
    assert OTHER_CASE_HEADER in fired["C1"].display_text
    assert fired["C1"].n_tokens > quiet["C0"].n_tokens


def test_tier_a_metrics_and_pretest_label():
    leaned = [_record(f"lean-{i}", split="test", kind="assigned", lean="A", order="BA" if i == 0 else "AB") for i in range(4)]
    no_lean = [_record(f"none-{i}", split="test", kind="none", lean=None) for i in range(4)]
    dev = [_record(f"dev-none-{i}", split="dev", kind="neutral_matched", lean=None) for i in range(3)]
    dev += [_record(f"dev-lean-{i}", split="dev", kind="assigned", lean="A") for i in range(3)]
    records = dev + leaned + no_lean
    scores = {}
    for index, score in enumerate((0.05, 0.1, 0.2)):
        scores[f"dev-none-{index}"] = score
    for index, score in enumerate((0.7, 0.8, 0.9)):
        scores[f"dev-lean-{index}"] = score
    for index in range(4):
        scores[f"lean-{index}"] = 0.95
    # 0.6 is below the dev threshold 0.7, so fitting on the test split would be visible.
    for index, score in enumerate((0.95, 0.6, 0.1, 0.1)):
        scores[f"none-{index}"] = score
    priors = {f"lean-{index}": "B" for index in range(4)}
    calls: list[str] = []

    def case_fn(scenario, target: str) -> str:
        calls.append(target)
        return "beta matters"

    pick, views, report = run_tier_a(records, scores, priors, case_fn=case_fn, client=OfflineFakeAnalyst())
    assert pick.threshold == 0.7
    assert views["none-1"]["C2"].triggered is False
    assert views["none-0"]["C2"].triggered is True
    assert views["lean-0"]["C2"].triggered is True
    # One generation per eval item that C1 (always) or a triggered C2 will show.
    assert len(calls) == 8

    assert report.label == PRETEST_LABEL
    payload = report.to_dict()
    assert payload["label"] == PRETEST_LABEL
    assert payload["warning"] == PRETEST_LABEL
    assert report.follow_rate == {"C0": 1.0, "C1": 0.0, "C2": 0.0, "C3": 1.0}
    assert report.drop_vs_c0 == {"C0": 0.0, "C1": 1.0, "C2": 1.0, "C3": 0.0}
    assert report.follow_rate_wilson_ci["C0"] == [float(x) for x in wilson_ci(4, 4)]
    assert report.follow_rate_wilson_ci["C1"] == [float(x) for x in wilson_ci(0, 4)]
    assert report.false_trigger_count == 1
    assert report.n_no_lean == 4
    assert report.false_trigger_rate == 0.25
    assert report.false_trigger_wilson_ci == [float(x) for x in wilson_ci(1, 4)]
    assert report.extra_tokens["C0"] == 0.0
    assert report.extra_tokens["C1"] > report.extra_tokens["C3"] > 0.0
    # Untriggered C2 rows add no tokens; the mean is still above C0.
    assert report.extra_tokens["C2"] > 0.0
    assert report.extra_tokens["C2"] < report.extra_tokens["C1"]
    assert report.n_unparsed == 0
    assert all(row["follows_lean"] for row in report.per_item if row["condition"] == "C0" and not row["no_lean"])
    assert all(not row["follows_lean"] for row in report.per_item if row["condition"] == "C1" and not row["no_lean"])


def test_offline_fake_keeps_prior_only_when_the_case_is_shown():
    record = _record("item", split="test", kind="assigned", lean="A")
    views = build_views_for_item(
        record,
        score=1.0,
        threshold=0.5,
        case_fn=lambda scenario, target: "beta matters",
    )
    agent = OfflineFakeAnalyst()
    from leanscope.simulate import analyst_messages

    plain = agent.complete(analyst_messages(record, "B", views["C0"]))
    both = agent.complete(analyst_messages(record, "B", views["C1"]))
    disclosed = agent.complete(analyst_messages(record, "B", views["C3"]))
    assert plain.text == "Answer: A"
    assert both.text == "Answer: B"
    assert disclosed.text == "Answer: A"


def test_paid_client_refuses_without_a_network_call():
    source = inspect.getsource(DeepSeekFlashClient)
    for token in ("requests", "urllib", "httpx", "openai", "socket"):
        assert token not in source
    client = make_client("deepseek")
    try:
        client.complete([{"role": "user", "content": "hi"}])
    except PaidAPIDisabled as exc:
        assert "OfflineFakeAnalyst" in str(exc)
    else:
        raise AssertionError("the paid client must refuse")
    assert isinstance(make_client("fake"), OfflineFakeAnalyst)


def test_template_case_names_the_unpushed_option():
    scenario = _record("item", split="test", kind="assigned", lean="A").scenario
    text = template_case(scenario, "B")
    assert "outcome beta" in text
    assert "outcome alpha" not in text


def test_panel_script_smoke_is_labelled_pretest(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "lean_05_panel_sim", os.path.join(ROOT, "scripts", "lean_05_panel_sim.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "panel"
    assert module.main(["--smoke-fake", "--out", str(out)]) == 0
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith(PRETEST_LABEL)
    payload = json.loads((out / "simulation.json").read_text(encoding="utf-8"))
    assert payload["label"] == PRETEST_LABEL
    assert payload["warning"] == PRETEST_LABEL
    assert payload["follow_rate"]["C0"] == 1.0
    assert payload["follow_rate"]["C1"] == 0.0
    assert payload["drop_vs_c0"]["C1"] == 1.0
    assert payload["false_trigger_rate"] == 0.5
    assert payload["threshold_pick"]["threshold"] == 0.7

    records = tmp_path / "records.jsonl"
    scores = tmp_path / "scores.json"
    records.write_text("{}\n", encoding="utf-8")
    scores.write_text("{}", encoding="utf-8")
    try:
        module.main(
            [
                "--records",
                str(records),
                "--scores",
                str(scores),
                "--prior-opposite",
                "--analyst",
                "deepseek",
            ]
        )
    except SystemExit as exc:
        assert "disabled" in str(exc)
    else:
        raise AssertionError("deepseek must be refused before any request")
