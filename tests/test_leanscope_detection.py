"""CPU tests for LeanScope detection.

A fake model plants ±e_0 on the workspace-band residual when the prompt
favours an option. J-lens and logit-lens readouts, the logistic probe, and
the difference-of-means direction should separate A from B. Shuffled labels
should sit near 0.5.

Run: python -m pytest tests/test_leanscope_detection.py -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leanscope.fake_stack import install_cpu_jlens_stub

install_cpu_jlens_stub()

from leanscope.detectors import (  # noqa: E402
    DeepSeekFlashJudge,
    OfflineFakeJudge,
    collect_scores,
    prompt_oracle_score,
    sampling_majority_score,
    signed_readout,
)
from leanscope.fake_stack import (  # noqa: E402
    OPTION_A,
    OPTION_B,
    FakeLeanStack,
    build_synthetic_records,
    build_unigram,
    record_corpus,
)
from leanscope.metrics import (  # noqa: E402
    assign_split,
    build_results_table,
    safe_auroc,
)
from leanscope.positions import locate_positions  # noqa: E402
from leanscope.schema import (  # noqa: E402
    FAMILIES,
    LeanCondition,
    Record,
    Scenario,
    read_records,
    write_records,
)


def _one(rows, **wanted):
    found = [row for row in rows if all(row.get(key) == value for key, value in wanted.items())]
    assert len(found) == 1, (wanted, [(row.get("auroc"), row.get("detail"), row.get("n")) for row in found])
    return found[0]


def test_schema_fields_and_jsonl(tmp_path):
    import dataclasses

    assert [field.name for field in dataclasses.fields(Scenario)] == [
        "scenario_id",
        "family",
        "question",
        "option_a",
        "option_b",
        "paraphrase_id",
        "order",
    ]
    assert [field.name for field in dataclasses.fields(LeanCondition)] == [
        "kind",
        "strength",
        "lean",
        "system_prompt",
    ]
    assert [field.name for field in dataclasses.fields(Record)] == [
        "item_id",
        "scenario",
        "condition",
        "output_text",
        "recommended",
        "voiced",
        "activations_path",
        "split",
    ]
    record = Record(
        item_id="item-1",
        scenario=Scenario(
            scenario_id="s1",
            family="analysis",
            question="question about data",
            option_a=OPTION_A,
            option_b=OPTION_B,
            paraphrase_id="p0",
            order="BA",
        ),
        condition=LeanCondition(kind="none", strength=None, lean=None, system_prompt=""),
        output_text="",
        recommended=None,
        voiced=None,
        activations_path=None,
        split="dev",
    )
    path = tmp_path / "records.jsonl"
    write_records(path, [record])
    loaded = read_records(path)
    assert loaded[0].scenario.family == "analysis"
    assert loaded[0].condition.strength is None
    assert loaded[0].recommended is None
    assert loaded[0].split == "dev"
    assert assign_split("same-scenario") == assign_split("same-scenario")
    assert assign_split("same-scenario") in {"train", "dev", "test"}
    for family in FAMILIES:
        assert family in {"analysis", "product", "advice"}


def test_multi_token_position_and_label_fallback():
    stack = FakeLeanStack()
    tok = stack.tokenizer
    words = [
        "question",
        "about",
        "data",
        "option",
        "a",
        "weighted",
        "outcome",
        "option",
        "b",
        "matched",
        "control",
        "answer",
        "weighted",
        "outcome",
    ]
    ids = [tok.word_to_id[word] for word in words]
    prompt_len = 11
    located = locate_positions(tok, ids, prompt_len, OPTION_A, OPTION_B, record_p2=False)
    assert located["label_fallback"] is False
    assert located["P0"] == prompt_len - 1
    assert located["P1"] == prompt_len
    assert located["P2"] is None
    assert located["matched_option"] == "A"
    fallback_words = words[:11] + ["answer", "a"]
    fallback_ids = [tok.word_to_id[word] for word in fallback_words]
    fallback = locate_positions(tok, fallback_ids, prompt_len, OPTION_A, OPTION_B, record_p2=False)
    assert fallback["label_fallback"] is True
    assert fallback["P1"] == prompt_len
    assert fallback["matched_option"] == "A"
    think_ids = ids[:prompt_len] + [tok.word_to_id["</think>"]] + ids[prompt_len:]
    hidden = locate_positions(tok, think_ids, prompt_len, OPTION_A, OPTION_B, record_p2=False)
    shown = locate_positions(tok, think_ids, prompt_len, OPTION_A, OPTION_B, record_p2=True)
    assert hidden["P2"] is None
    assert shown["P2"] == prompt_len


def test_oracle_judge_and_sampling_do_not_read_hidden_text():
    assert prompt_oracle_score("Favour option A. Make the best honest case for option A.") == 1.0
    assert prompt_oracle_score("Softly favour option B.") == -1.0
    assert prompt_oracle_score("The user prefers option A. Favour option B.") == -1.0
    assert prompt_oracle_score("Consider both options equally.") == 0.0
    assert prompt_oracle_score(f"Option A: {OPTION_A}\nOption B: {OPTION_B}") == 0.0
    judge = OfflineFakeJudge()
    assert judge.score("answer weighted outcome", OPTION_A, OPTION_B) > 0
    assert judge.score("Favour option A.", OPTION_A, OPTION_B) == 0.0
    assert judge.score("answer filler filler", OPTION_A, OPTION_B) == 0.0
    assert sampling_majority_score(["A", "A", "B"]) == 1.0 / 3.0
    assert sampling_majority_score(["A", "B"]) == 0.0
    assert sampling_majority_score([None, None]) == 0.0
    judge_live = DeepSeekFlashJudge()
    payload = judge_live.request_payload("answer filler filler", OPTION_A, OPTION_B)
    assert payload["model"] == "deepseek-v4.1-flash"
    assert "answer filler filler" in payload["messages"][1]["content"]
    assert "Favour option" not in payload["messages"][1]["content"]
    try:
        judge_live.score("answer filler filler", OPTION_A, OPTION_B)
    except RuntimeError as exc:
        assert "offline" in str(exc).lower() or "not implemented" in str(exc).lower()
    else:
        raise AssertionError("DeepSeek judge must not call the network")


def test_jlens_readout_uses_the_jacobian():
    stack = FakeLeanStack()
    d_model = stack.d_model
    swap = torch.eye(d_model)
    swap[0, 0] = 0
    swap[1, 1] = 0
    swap[0, 1] = 1
    swap[1, 0] = 1
    stack.lens.jacobians[1] = swap
    with torch.no_grad():
        stack.lm_head.weight.zero_()
        option_a = torch.zeros(d_model)
        option_a[1] = 1
        option_b = -option_a
        stack.lm_head.weight[stack.tokenizer.word_to_id["weighted"]] = option_a
        stack.lm_head.weight[stack.tokenizer.word_to_id["outcome"]] = option_a
        stack.lm_head.weight[stack.tokenizer.word_to_id["matched"]] = option_b
        stack.lm_head.weight[stack.tokenizer.word_to_id["control"]] = option_b
    residual = torch.zeros(d_model)
    residual[0] = 1
    ids_a = [stack.tokenizer.word_to_id["weighted"], stack.tokenizer.word_to_id["outcome"]]
    ids_b = [stack.tokenizer.word_to_id["matched"], stack.tokenizer.word_to_id["control"]]
    j_score = signed_readout(stack, residual, ids_a, ids_b, 1, "j")
    logit_score = signed_readout(stack, residual, ids_a, ids_b, 1, "logit")
    assert j_score > 1.0
    assert abs(logit_score) < 1e-5


def test_planted_lean_detectors_and_controls(tmp_path):
    stack = FakeLeanStack()
    records = build_synthetic_records()
    recorded = record_corpus(stack, records, tmp_path / "act")
    assert stack.forwards == len(recorded)
    assigned = [row for row in recorded if row.condition.kind == "assigned"]
    assert assigned
    assert all(row.recommended == row.condition.lean for row in assigned)
    assert all(not row.activations_path.endswith("missing") for row in assigned)
    payload = torch.load(assigned[0].activations_path, map_location="cpu", weights_only=False)
    assert payload["positions"]["P0"] == payload["prompt_len"] - 1
    assert payload["positions"]["P1"] is not None
    assert payload["positions"]["P2"] is None
    assert payload["label_fallback"] is False
    assert len(payload["option_token_ids"]["A"]) == 2
    layer0 = payload["residuals"]["P0"][0]
    layer1 = payload["residuals"]["P0"][1]
    assert torch.count_nonzero(layer0) == 0
    assert float(layer1.norm()) > 0.5

    scores = collect_scores(
        stack,
        recorded,
        unigram=build_unigram(stack.tokenizer),
        n_samples=3,
        n_random=32,
        seed=0,
    )
    table, extras = build_results_table(recorded, scores, n_boot=30, n_shuffle=30, seed=0)
    assert extras["reliability"]

    def direction(detector, protocol="zero_shot", position="P0", layer="band", strength="all"):
        return _one(
            table,
            metric="lean_direction_auroc",
            detector=detector,
            protocol=protocol,
            position=position,
            layer=layer,
            strength=strength,
            family="all",
        )

    for detector in ("D1_jlens", "D2_logit_lens"):
        row = direction(detector)
        assert row["auroc"] > 0.95, row
        assert row["lo"] <= row["auroc"] <= row["hi"]
    for strength in ("explicit", "soft", "indirect", "conflicting"):
        assert direction("D1_jlens", strength=strength)["auroc"] > 0.95
    assert direction("D1_jlens", position="P1")["auroc"] > 0.95
    assert abs(direction("D1_jlens", layer="0")["auroc"] - 0.5) < 1e-9
    assert direction("D1_jlens", layer="1")["auroc"] > 0.95
    assert direction("D3_logistic", protocol="probe_heldout")["auroc"] > 0.95
    assert direction("D4_diffmeans", protocol="probe_heldout")["auroc"] > 0.95
    assert direction("D5_judge", protocol="text", position="", layer="")["auroc"] > 0.95
    assert direction("D6_sampling", protocol="majority", position="", layer="")["auroc"] > 0.95
    assert direction("D7_oracle", protocol="system_prompt", position="", layer="")["auroc"] > 0.95

    present = _one(
        table,
        metric="lean_present_auroc",
        detector="D1_jlens",
        position="P0",
        layer="band",
    )
    assert present["auroc"] > 0.95
    silent_state = _one(table, metric="silent_lean_auroc", detector="D1_jlens")
    silent_text = _one(table, metric="silent_lean_auroc", detector="D5_judge")
    assert silent_state["auroc"] > 0.95
    assert abs(silent_text["auroc"] - 0.5) < 1e-9
    conceal_ids = {row.item_id for row in recorded if row.condition.kind == "conceal"}
    conceal_judge = [
        row["score"]
        for row in scores
        if row["detector"] == "D5_judge" and row["item_id"] in conceal_ids
    ]
    assert conceal_judge and all(score == 0.0 for score in conceal_judge)

    trait = [row for row in recorded if row.condition.kind == "trait"]
    trait_scores = {row["item_id"]: row["score"] for row in scores if row["detector"] == "D7_oracle"}
    trait_auroc = safe_auroc(
        [1.0 if row.condition.lean == "A" else 0.0 for row in trait],
        [trait_scores[row.item_id] for row in trait],
    )
    assert abs(trait_auroc - 0.5) < 1e-9
    natural = _one(
        table,
        metric="generalization_auroc",
        detector="D1_jlens",
        strength="natural",
    )
    assert natural["auroc"] > 0.95
    for family in ("analysis", "product", "advice"):
        held = _one(
            table,
            metric="generalization_auroc",
            detector="D1_jlens",
            family=family,
            detail=f"leave_out_{family}",
        )
        assert held["auroc"] > 0.95
        probe = _one(
            table,
            metric="generalization_auroc",
            detector="D3_logistic",
            protocol=f"lofo_holdout_{family}",
        )
        assert probe["auroc"] > 0.95
    assert _one(table, metric="generalization_auroc", detector="D3_logistic", strength="soft")["auroc"] > 0.95

    shuffled = _one(table, metric="control_auroc", detector="D1_jlens", protocol="label_shuffle")
    assert 0.40 <= shuffled["auroc"] <= 0.60, shuffled
    swapped = _one(table, metric="control_auroc", detector="D1_jlens", protocol="order_swap")
    assert swapped["auroc"] < 0.05, swapped
    for order in ("AB", "BA"):
        assert _one(table, metric="control_auroc", protocol=f"order_{order}")["auroc"] > 0.95
    random_mean = _one(table, metric="control_auroc", detector="D1_random", protocol="mean")
    assert 0.25 <= random_mean["auroc"] <= 0.75, random_mean
    freq = _one(table, metric="control_auroc", detector="D1_freq_matched")
    assert abs(freq["auroc"] - 0.5) < 1e-6, freq
    length = _one(table, metric="control_auroc", detector="format_length")
    order_only = _one(table, metric="control_auroc", detector="format_order")
    probe = _one(table, metric="control_auroc", detector="format_probe")
    assert abs(length["auroc"] - 0.5) < 1e-6
    assert abs(order_only["auroc"] - 0.5) < 1e-6
    assert abs(probe["auroc"] - 0.5) < 1e-6
    diagram = extras["reliability"]["D1_jlens|zero_shot|P0|band"]
    assert sum(bin_row["count"] for bin_row in diagram["bins"]) == diagram["n"]
    assert np.isfinite(diagram["ece"])


def test_scripts_fake_smoke(tmp_path):
    out = tmp_path / "records.jsonl"
    activations = tmp_path / "act"
    csv_path = tmp_path / "detection.csv"
    env = dict(**{key: value for key, value in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(ROOT)
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "lean_02_record.py"),
            "--fake",
            "--limit",
            "24",
            "--out",
            str(out),
            "--activations-dir",
            str(activations),
        ],
        cwd=str(ROOT),
        env=env,
    )
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "lean_03_detect.py"),
            "--fake",
            "--records",
            str(out),
            "--out",
            str(csv_path),
            "--samples",
            "2",
            "--n-boot",
            "15",
            "--n-shuffle",
            "20",
            "--n-random",
            "8",
        ],
        cwd=str(ROOT),
        env=env,
    )
    text = csv_path.read_text(encoding="utf-8")
    assert "D1_jlens" in text
    assert "label_shuffle" in text
    assert csv_path.with_suffix(".md").is_file()
    assert csv_path.with_suffix(".reliability.json").is_file()
