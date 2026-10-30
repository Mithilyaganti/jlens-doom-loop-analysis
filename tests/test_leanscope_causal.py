"""CPU tests for LeanScope steering, ablation, and the flip-rate statistics.

The model is TinyFlipModel: a one-layer module whose answer flips when a
planted direction is steered or ablated, and does not flip for a norm-matched
random direction. No GPU and no downloaded weights.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace.stats import cohens_h, mcnemar_exact, wilson_ci  # noqa: E402
from leanscope.causal import (  # noqa: E402
    _as_unit,
    absolute_position,
    contrast_stats,
    control_seed,
    difference_of_means_direction,
    locate_p1_from_text,
    make_full_forward_steer_hook,
    make_tracked_position_hook,
    matched_random,
    run_causal,
    steer_alpha_against_lean,
)
from leanscope.schema import (  # noqa: E402
    LeanCondition,
    Record,
    Scenario,
    read_jsonl,
    write_jsonl,
)
from leanscope.tiny_model import TinyFlipModel, option_token_ids, tiny_tokenize  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ALPHA = 1.2


class _Stack:
    tokenizer = None

    def __init__(self, model: TinyFlipModel):
        self.model = model
        self.d_model = int(model.planted.numel())
        self.n_layers = 1
        self.device = torch.device("cpu")

    @property
    def layers(self):
        return self.model.layers


def _scenario(index: int) -> Scenario:
    return Scenario(
        scenario_id=f"s{index}",
        family="advice" if index % 3 == 0 else "analysis" if index % 3 == 1 else "product",
        question=f"Which option should case {index} adopt?",
        option_a="alpha plan",
        option_b="beta plan",
        paraphrase_id=str(index % 2),
        order="AB" if index % 2 == 0 else "BA",
    )


def _record(index: int, *, lean: str = "A", split: str = "test", activations: str | None = None, system: str = "LEAN_A") -> Record:
    return Record(
        item_id=f"item-{index}",
        scenario=_scenario(index),
        condition=LeanCondition(
            kind="assigned",
            strength="explicit",
            lean=lean,
            system_prompt=system if lean == "A" else "LEAN_B",
        ),
        output_text="Answer: A",
        recommended="A" if lean == "A" else "B",
        voiced=True,
        activations_path=activations,
        split=split,
    )


def _planted(width: int = 32) -> torch.Tensor:
    vector = torch.zeros(width)
    vector[0] = 1.0
    return vector


def _assert_controls_are_not_aligned(item_ids, planted: torch.Tensor, position: str) -> None:
    """The random control must not be so parallel that alpha would flip the margin."""
    unit = _as_unit(planted)
    for item_id in item_ids:
        draw = matched_random(unit, seed=control_seed(item_id, 0, position))
        dot = abs(float(torch.dot(_as_unit(draw), unit)))
        assert dot < 0.45, f"{item_id} {position} random alignment {dot}"
        assert abs(float(draw.norm()) - float(unit.norm())) < 1e-5


def _run(model: TinyFlipModel, records, direction_fn, positions, **kwargs):
    return run_causal(
        _Stack(model),
        records,
        direction_fn=direction_fn,
        layers=[0],
        positions=positions,
        alpha=ALPHA,
        max_new_tokens=4,
        backend="logits",
        tokenize=tiny_tokenize,
        option_token_ids=option_token_ids(),
        **kwargs,
    )


def test_schema_jsonl_roundtrip(tmp_path):
    original = Record(
        item_id="row-1",
        scenario=Scenario(
            scenario_id="sc",
            family="product",
            question="Pick one",
            option_a="left",
            option_b="right",
            paraphrase_id=0,
            order="BA",
        ),
        condition=LeanCondition(kind="none", strength=None, lean=None, system_prompt=""),
        output_text="",
        recommended=None,
        voiced=None,
        activations_path=None,
        split="dev",
    )
    path = tmp_path / "nested" / "records.jsonl"
    write_jsonl(path, [original])
    loaded = read_jsonl(path)
    assert len(loaded) == 1
    assert loaded[0] == original
    assert loaded[0].scenario.paraphrase_id == "0"
    try:
        Scenario(
            scenario_id="x",
            family="other",
            question="q",
            option_a="a",
            option_b="b",
            paraphrase_id="1",
            order="AB",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("bad family must be rejected")


def test_contrast_stats_matches_mcnemar_wilson_and_h():
    baseline = ["A", "A", "A", "A", "B"]
    condition = ["B", "B", "A", "B", "A"]
    leans = ["A", "A", "A", "A", "A"]
    stats = contrast_stats(baseline, condition, leans)
    assert stats["n_pairs"] == 5
    assert stats["n_flips"] == 4
    assert stats["flip_rate"] == 0.8
    assert stats["b_baseline_follows_only"] == 3
    assert stats["c_condition_follows_only"] == 1
    mc = mcnemar_exact(3, 1)
    assert stats["p_exact"] == mc["p_exact"]
    assert stats["odds_ratio_b_over_c"] == mc["odds_ratio_b_over_c"]
    assert stats["flip_rate_wilson_ci"] == [float(x) for x in wilson_ci(4, 5)]
    assert stats["wilson_ci_baseline_follows"] == [float(x) for x in wilson_ci(4, 5)]
    assert stats["wilson_ci_condition_follows"] == [float(x) for x in wilson_ci(2, 5)]
    assert stats["cohens_h"] == cohens_h(4 / 5, 2 / 5)


def test_position_index_is_the_token_before_the_option_word():
    assert absolute_position(4, [5], "P0") == 3
    assert absolute_position(4, [5], "P1", {5, 6}) == 3
    assert absolute_position(4, [3, 5], "P1", {5, 6}) == 4

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            words = {1: "hello", 2: "alpha"}
            return " ".join(words[int(i)] for i in ids)

    found = locate_p1_from_text(5, [1, 2], "hello alpha", _Tok(), ["alpha"])
    assert found == 5


def test_steer_alpha_points_against_the_assigned_lean():
    assert steer_alpha_against_lean("A", 1.2) == -1.2
    assert steer_alpha_against_lean("B", -3.0) == 3.0


def test_tracked_hook_edits_only_the_target_step():
    layer = nn.Identity()
    direction = torch.zeros(4)
    direction[0] = 1
    handle = layer.register_forward_hook(
        make_tracked_position_hook(direction, abs_position=5, mode="steer", alpha=-1.0)
    )
    try:
        prefill = torch.ones(1, 4, 4)
        assert torch.equal(layer(prefill), prefill)
        step4 = torch.ones(1, 1, 4)
        assert torch.equal(layer(step4), step4)
        step5 = torch.ones(1, 1, 4)
        edited = layer(step5)
        assert abs(float(edited[0, 0, 0])) < 1e-5
        assert torch.allclose(edited[0, 0, 1:], step5[0, 0, 1:])
    finally:
        handle.remove()


def test_tracked_ablation_removes_only_the_planted_component():
    layer = nn.Identity()
    direction = torch.tensor([2.0, 0.0, 0.0, 0.0])
    handle = layer.register_forward_hook(
        make_tracked_position_hook(direction, abs_position=0, mode="ablate", alpha=0.0)
    )
    try:
        incoming = torch.tensor([[[2.0, 3.0, 0.0, 0.0]]])
        edited = layer(incoming)
    finally:
        handle.remove()
    assert abs(float(edited[0, 0, 0])) < 1e-5
    assert abs(float(edited[0, 0, 1]) - 3.0) < 1e-5


def test_steer_hook_keeps_tuple_outputs():
    class TupleLayer(nn.Module):
        def forward(self, hidden):
            return (hidden, "cache")

    layer = TupleLayer()
    direction = torch.zeros(4)
    direction[1] = 4  # non-unit; the hook must normalize
    handle = layer.register_forward_hook(
        make_full_forward_steer_hook(direction, position=1, alpha=2.0)
    )
    try:
        incoming = torch.zeros(1, 3, 4)
        hidden, cache = layer(incoming)
    finally:
        handle.remove()
    assert cache == "cache"
    assert torch.equal(hidden[0, 0], incoming[0, 0])
    assert abs(float(hidden[0, 1, 1]) - 2.0) < 1e-5
    assert float(hidden[0, 1, 0]) == 0.0


def test_planted_steer_flips_and_random_control_does_not():
    planted = _planted()
    records = [_record(i) for i in range(8)]
    _assert_controls_are_not_aligned([row.item_id for row in records], planted, "P0")

    def direction_fn(rows, *, layer, position):
        del rows, layer, position
        return planted.clone()

    report = _run(TinyFlipModel(planted, decision_at="P0"), records, direction_fn, ["P0"])
    assert report["n_items"] == 8
    assert report["direction"] == "pluggable"
    steer = report["contrasts"]["steer_option@P0"]
    assert steer["flip_rate"] == 1.0
    assert steer["n_flips"] == 8
    assert steer["b_baseline_follows_only"] == 8
    assert steer["c_condition_follows_only"] == 0
    assert abs(steer["p_exact"] - 2.0 * 0.5**8) < 1e-12
    assert steer["flip_rate_wilson_ci"] == [float(x) for x in wilson_ci(8, 8)]
    assert abs(steer["cohens_h"] - math.pi) < 1e-12
    assert steer["rate_baseline_follows"] == 1.0
    assert steer["rate_condition_follows"] == 0.0
    assert steer["flip_rate_wilson_ci"][0] > 0.6

    ablate = report["contrasts"]["ablate_option@P0"]
    assert ablate["flip_rate"] == 1.0
    assert ablate["b_baseline_follows_only"] == 8

    for key in ("steer_random@P0", "ablate_random@P0"):
        control = report["contrasts"][key]
        assert control["flip_rate"] == 0.0, key
        assert control["n_flips"] == 0
        assert math.isnan(control["p_exact"])
        assert control["cohens_h"] == 0.0
        assert control["rate_condition_follows"] == 1.0
        assert control["flip_rate_wilson_ci"] == [float(x) for x in wilson_ci(0, 8)]

    positive = report["contrasts"]["remove_lean_prompt"]
    assert positive["flip_rate"] == 1.0
    assert positive["rate_condition_follows"] == 0.0
    assert positive["b_baseline_follows_only"] == 8


def test_p1_steer_flips_and_p0_on_the_same_model_does_not():
    planted = _planted()
    records = [_record(i) for i in range(4)]
    _assert_controls_are_not_aligned([row.item_id for row in records], planted, "P0")
    _assert_controls_are_not_aligned([row.item_id for row in records], planted, "P1")

    def direction_fn(rows, *, layer, position):
        del rows, layer, position
        return planted.clone()

    report = _run(TinyFlipModel(planted, decision_at="P1"), records, direction_fn, ["P0", "P1"])
    assert report["contrasts"]["steer_option@P1"]["flip_rate"] == 1.0
    assert report["contrasts"]["ablate_option@P1"]["flip_rate"] == 1.0
    assert report["contrasts"]["steer_option@P0"]["flip_rate"] == 0.0
    assert report["contrasts"]["ablate_option@P0"]["flip_rate"] == 0.0
    assert report["contrasts"]["steer_random@P1"]["flip_rate"] == 0.0
    assert report["contrasts"]["ablate_random@P1"]["flip_rate"] == 0.0
    assert report["contrasts"]["remove_lean_prompt"]["flip_rate"] == 1.0
    # The option word is the second generated token, so P1 is not P0.
    positions = {
        (row["item_id"], row["position"]): row["abs_position"]
        for row in report["outcomes"]
        if row["position"] is not None
    }
    assert positions[("item-0", "P0")] == 3
    assert positions[("item-0", "P1")] == 4


def test_orthogonal_plugin_does_not_flip_and_difference_of_means_does(tmp_path):
    planted = _planted()
    records = [_record(i) for i in range(4)]

    def orthogonal(rows, *, layer, position):
        del rows, layer, position
        vector = torch.zeros_like(planted)
        vector[1] = 3.0
        return vector

    missed = _run(TinyFlipModel(planted, decision_at="P0"), records, orthogonal, ["P0"])
    assert missed["contrasts"]["steer_option@P0"]["flip_rate"] == 0.0

    train = []
    for index, lean, sign in ((0, "A", 1.0), (1, "A", 1.0), (2, "B", -1.0), (3, "B", -1.0)):
        vector = sign * planted
        path = tmp_path / f"{index}.pt"
        torch.save({"P0": vector.unsqueeze(0), "P1": {0: vector}}, path)
        train.append(_record(index, lean=lean, split="train", activations=str(path), system="LEAN_A"))
    recovered = difference_of_means_direction(train, layer=0, position="P0")
    cosine = float(torch.dot(_as_unit(recovered), _as_unit(planted)))
    assert cosine > 0.999
    # Dict-keyed P1 residuals load the same axis.
    recovered_p1 = difference_of_means_direction(train, layer=0, position="P1")
    assert float(torch.dot(_as_unit(recovered_p1), _as_unit(planted))) > 0.999

    used_default = run_causal(
        _Stack(TinyFlipModel(planted, decision_at="P0")),
        records,
        direction_records=train,
        layers=[0],
        positions=["P0"],
        alpha=ALPHA,
        max_new_tokens=2,
        backend="logits",
        tokenize=tiny_tokenize,
        option_token_ids=option_token_ids(),
    )
    assert used_default["direction"] == "difference_of_means"
    assert used_default["contrasts"]["steer_option@P0"]["flip_rate"] == 1.0
    assert used_default["contrasts"]["steer_random@P0"]["flip_rate"] == 0.0


def test_causal_script_smoke(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "lean_04_causal", os.path.join(ROOT, "scripts", "lean_04_causal.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "causal"
    assert module.main(["--smoke-fake", "--out", str(out)]) == 0
    stats = __import__("json").loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["contrasts"]["steer_option@P0"]["flip_rate"] == 1.0
    assert stats["contrasts"]["steer_random@P0"]["flip_rate"] == 0.0
    assert "CPU smoke" in stats["note"]
    try:
        module.main([])
    except SystemExit:
        pass
    else:
        raise AssertionError("a real run without records must stop")
