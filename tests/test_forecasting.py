"""CPU tests for pre-onset labelling, metrics, and trace-only detectors.

Run: python tests/test_forecasting.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace.detection import find_inner_repetition  # noqa: E402
from jspace.forecasting.data import (  # noqa: E402
    TraceRecord,
    load_baseline_traces,
    load_trace_dir,
)
from jspace.forecasting.detectors import (  # noqa: E402
    antidoom_confirmation_char,
    cusum_statistic,
    fit_cusum_in_control,
    online_ngram_repeat_scores,
)
from jspace.forecasting.features import (  # noqa: E402
    SURFACE_FEATURES,
    evaluate_feature_table,
    feature_matrix,
    load_feature_table,
    save_feature_table,
)
from jspace.forecasting.labels import (  # noqa: E402
    OnsetLabel,
    PreparedTrace,
    build_horizon_examples,
    label_trace,
)
from jspace.forecasting.metrics import (  # noqa: E402
    SPLIT_SALT,
    assign_split,
    clustered_bootstrap_ci,
    expected_calibration_error,
    lead_time_summary,
    safe_auroc,
    threshold_for_far,
)
from jspace.forecasting.run import analyze_traces, render_report, summarize_results  # noqa: E402


def _label(**kwargs) -> OnsetLabel:
    base = dict(
        prompt_id=1,
        is_loop=False,
        onset_token_index=None,
        onset_source="not_a_loop",
        n_tokens=100,
        split="train",
    )
    base.update(kwargs)
    return OnsetLabel(**base)


def _ids_for(n_by_split: dict[str, int]) -> dict[str, list[int]]:
    buckets = {name: [] for name in ("train", "dev", "test")}
    need = sum(n_by_split.values())
    i = 0
    got = 0
    while got < need:
        split = assign_split(i)
        if len(buckets[split]) < n_by_split.get(split, 0):
            buckets[split].append(i)
            got += 1
        i += 1
    return buckets


def test_antidoom_rule_thresholds():
    # Period is at least the detector's 16-character sample, so the sampler can see it.
    assert find_inner_repetition("abcdefghijklmnop" * 3)[0] is False
    found, hit = find_inner_repetition("abcdefghijklmnop" * 4)
    assert found and hit is not None
    assert hit.repeats >= 4
    assert hit.repeats * hit.period >= 60
    assert find_inner_repetition("abc" * 4)[0] is False


def test_split_is_hash_locked_to_the_prompt_sample():
    assert SPLIT_SALT == "jspace-forecast-v1"
    digest = hashlib.sha256(f"{SPLIT_SALT}:86352".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    expect = "train" if unit < 0.60 else "dev" if unit < 0.80 else "test"
    assert assign_split(86352) == expect
    assert assign_split(86352) == assign_split(86352)
    root = Path(__file__).resolve().parents[1]
    sample = json.loads((root / "results" / "prompt_sample_ids.json").read_text(encoding="utf-8"))
    ids = [int(p["prompt_id"]) for p in sample["prompts"]]
    assert len(ids) == 200
    counts = {"train": 0, "dev": 0, "test": 0}
    for pid in ids:
        counts[assign_split(pid)] += 1
    assert counts == {"train": 121, "dev": 39, "test": 40}
    loops = [86352, 96637, 120702, 123613, 133187, 179962, 199303, 290449, 340846, 346805, 422988]
    loop_counts = {"train": 0, "dev": 0, "test": 0}
    for pid in loops:
        loop_counts[assign_split(pid)] += 1
    assert loop_counts == {"train": 8, "dev": 1, "test": 2}


def test_onset_label_maps_the_second_copy_and_confirmation_is_after_onset():
    text = "abcdefghijklmnop" * 4
    strings = list(text)
    trace = PreparedTrace(
        prompt_id=7,
        token_ids=list(range(len(strings))),
        token_strings=strings,
        text=text,
        aligned=True,
        meta={"is_loop": True, "trigger_token_index": 16},
        source="unit",
    )
    label = label_trace(trace)
    assert label.is_loop
    assert label.onset_token_index == 16
    assert label.onset_source == "detector_token_map"
    assert label.confirmation_char == 64
    assert label.confirmation_token_index == 63
    assert label.confirmation_token_index > label.onset_token_index
    assert not label.disagrees_with_stored


def test_stored_onset_used_only_when_text_agrees():
    text = "abcdefghijklmnop" * 4
    trace = PreparedTrace(
        prompt_id=8,
        token_ids=list(range(64)),
        token_strings=None,
        text=text,
        aligned=False,
        meta={"is_loop": True, "trigger_token_index": 16, "n_tokens": 64},
        source=None,
    )
    label = label_trace(trace)
    assert label.is_loop
    assert label.onset_token_index == 16
    assert label.onset_source == "stored_trigger_index"

    clean = PreparedTrace(
        prompt_id=9,
        token_ids=list(range(20)),
        token_strings=None,
        text="A normal answer with no repeated block at all.",
        aligned=False,
        meta={"is_loop": True, "trigger_token_index": 3},
        source=None,
    )
    negative = label_trace(clean)
    assert not negative.is_loop
    assert negative.onset_token_index is None
    assert negative.disagrees_with_stored


def test_horizon_windows_match_length_and_position_inside_the_split():
    labels = [
        _label(prompt_id=1, is_loop=True, onset_token_index=200, n_tokens=400, split="train", onset_source="detector_token_map"),
        _label(prompt_id=2, is_loop=False, n_tokens=400, split="train"),
        _label(prompt_id=3, is_loop=False, n_tokens=40, split="train"),
        _label(prompt_id=4, is_loop=False, n_tokens=400, split="test"),
    ]
    examples, info = build_horizon_examples(labels, horizons=(32,), window_len=16)
    pos = [ex for ex in examples if ex.y == 1]
    neg = [ex for ex in examples if ex.y == 0]
    assert len(pos) == 1
    assert pos[0].decision_index == 200 - 32
    assert pos[0].decision_index + 32 == 200
    assert pos[0].length == 16
    assert neg[0].prompt_id == 2
    assert neg[0].decision_index == pos[0].decision_index
    assert neg[0].length == pos[0].length
    assert neg[0].match_kind == "absolute_position"
    assert neg[0].matched_prompt_id == 1
    assert all(ex.prompt_id != 4 for ex in neg)
    assert info["unmatched_positives"] == 0

    short = [
        _label(prompt_id=10, is_loop=True, onset_token_index=10, n_tokens=30, split="train", onset_source="detector_token_map"),
    ]
    _examples, info_short = build_horizon_examples(short, horizons=(32,), window_len=8)
    assert info_short["ineligible_by_horizon"]["32"] == 1
    assert _examples == []


def test_depth_match_and_reuse_are_deterministic():
    labels = [
        _label(prompt_id=1, is_loop=True, onset_token_index=180, n_tokens=200, split="dev", onset_source="detector_token_map"),
        _label(prompt_id=2, is_loop=False, n_tokens=50, split="dev"),
    ]
    examples, _info = build_horizon_examples(labels, horizons=(32,), window_len=16)
    neg = [ex for ex in examples if ex.y == 0]
    assert len(neg) == 1
    assert neg[0].match_kind == "normalized_depth"
    assert neg[0].length == 16
    assert 15 <= neg[0].decision_index <= 49
    again, _ = build_horizon_examples(labels, horizons=(32,), window_len=16)
    assert [ex.decision_index for ex in again] == [ex.decision_index for ex in examples]

    crowded = [
        _label(prompt_id=pid, is_loop=True, onset_token_index=200, n_tokens=400, split="train", onset_source="detector_token_map")
        for pid in (1, 2, 3)
    ]
    crowded.append(_label(prompt_id=9, is_loop=False, n_tokens=400, split="train"))
    matched, _ = build_horizon_examples(crowded, horizons=(32,), window_len=16)
    kinds = [ex.match_kind for ex in matched if ex.y == 0]
    assert kinds.count("absolute_position") == 1
    assert any(kind == "absolute_position_reused" for kind in kinds)


def test_metrics_auroc_ece_threshold_and_lead():
    assert safe_auroc([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0]) == 1.0
    assert expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]) == 0.0
    assert abs(expected_calibration_error([0, 0, 0, 1, 1, 1], [0.9] * 6) - 0.4) < 1e-9

    maxima = np.arange(100, dtype=float)
    thr = threshold_for_far(maxima, 0.05)
    assert int(np.sum(maxima >= thr)) == 5
    tied = np.ones(20)
    thr_tied = threshold_for_far(tied, 0.05)
    assert int(np.sum(tied >= thr_tied)) == 0

    scores = [np.zeros(80) for _ in range(4)]
    for row in scores:
        row[50] = 1.0
    summary = lead_time_summary(scores, [60, 60, 60, 60], threshold=0.5)
    assert summary["early_rate"] == 1.0
    assert summary["median_lead_tokens_early"] == 10
    late = lead_time_summary([np.array([0.0, 0.0, 1.0])], [1], threshold=0.5)
    assert late["n_early"] == 0
    assert late["median_signed_lead_tokens"] == -1

    def _mean(idx):
        return float(np.mean(np.arange(10)[idx]))

    ci = clustered_bootstrap_ci(np.arange(10), _mean, n_boot=200, seed=1)
    ci2 = clustered_bootstrap_ci(np.arange(10), _mean, n_boot=200, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci == ci2


def test_ngram_is_causal_and_cusum_accepts_any_score():
    ids = np.array([0, 1, 2] * 6, dtype=np.int64)
    scores = online_ngram_repeat_scores(ids)
    assert scores[11] == 1.0
    assert scores[10] < 1.0
    assert np.all(scores[:3] == 0.0)
    prefix = online_ngram_repeat_scores(ids[:8])
    assert np.allclose(prefix, scores[:8])

    short = online_ngram_repeat_scores([1, 1, 1, 1], token_char_lens=[1, 1, 1, 1])
    assert short[-1] < 1.0
    assert online_ngram_repeat_scores([1, 1, 1, 1])[-1] == 1.0

    text = "abcdefghijklmnop" * 4
    assert antidoom_confirmation_char(text) == 64
    assert antidoom_confirmation_char(text[:63]) is None

    got = cusum_statistic([0, 0, 1, 1, 1], mu=0.0, allowance=0.5)
    assert np.allclose(got, [0.0, 0.0, 0.5, 1.0, 1.5])
    controls = [np.zeros(30) for _ in range(40)]
    ref = fit_cusum_in_control(controls)
    assert ref["mu"] == 0.0
    probe = np.zeros(80)
    probe[50:] = 5.0
    series = cusum_statistic(probe, mu=ref["mu"], allowance=ref["allowance"])
    summary = lead_time_summary([series], [60], threshold_for_far([0.0] * 40, 0.05))
    assert summary["median_lead_tokens_early"] == 10


def test_surface_features_do_not_read_future_tokens():
    ids = np.arange(40)
    ids[10:20] = 7
    ngram = online_ngram_repeat_scores(ids)
    t = 25
    full = feature_matrix(ids, [t], window=8, ngram=ngram)
    trunc = feature_matrix(ids[: t + 1], [t], window=8, ngram=online_ngram_repeat_scores(ids[: t + 1]))
    assert np.allclose(full, trunc)
    assert full.shape == (1, len(SURFACE_FEATURES))


def test_feature_table_roundtrip_recomputes_the_split():
    buckets = _ids_for({"train": 12, "dev": 8, "test": 8})
    ids = []
    y = []
    for split, n_pos in (("train", 6), ("dev", 4), ("test", 4)):
        chosen = buckets[split]
        for i, pid in enumerate(chosen):
            ids.append(pid)
            y.append(1 if i < n_pos else 0)
    y = np.asarray(y)
    X = np.column_stack([y.astype(float), np.zeros(len(y))])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.npz"
        save_feature_table(path, name="probe", prompt_ids=ids, y=y, X=X, feature_names=["signal", "zero"], horizon=np.full(len(y), 32))
        loaded = load_feature_table(path)
    assert loaded["name"] == "probe"
    metrics = evaluate_feature_table(loaded["prompt_ids"], loaded["y"], loaded["X"], kind="logreg", horizons=loaded["horizon"], n_boot=20, seed=0)
    assert metrics["per_horizon"]["32"]["test"]["auroc"] == 1.0


def _loop_trace(pid: int) -> TraceRecord:
    strings = ["common" if i % 3 == 0 else f"u{i}z" for i in range(340)]
    strings.extend(["abcdefghij", "klmnop"] * 8)
    text = "".join(strings)
    ids = []
    for i, tok in enumerate(strings):
        if tok == "common":
            ids.append(1)
        elif tok == "abcdefghij":
            ids.append(7)
        elif tok == "klmnop":
            ids.append(8)
        else:
            ids.append(1000 + i)
    return TraceRecord(
        prompt_id=pid,
        generated_text=text,
        token_ids=ids,
        token_strings=strings,
        source="unit",
        meta={},
        text_complete=True,
        origin="synthetic",
    )


def _control_trace(pid: int, n_tokens: int = 400) -> TraceRecord:
    strings = [f"c{pid}n{i}q" for i in range(n_tokens)]
    return TraceRecord(
        prompt_id=pid,
        generated_text="".join(strings),
        token_ids=[10_000 + i for i in range(n_tokens)],
        token_strings=strings,
        source="unit",
        meta={"is_loop": False},
        text_complete=True,
        origin="synthetic",
    )


def test_analyze_traces_on_a_planted_signal():
    buckets = _ids_for({"train": 12, "dev": 8, "test": 8})
    records = []
    for split in ("train", "dev", "test"):
        half = len(buckets[split]) // 2
        for pid in buckets[split][:half]:
            records.append(_loop_trace(pid))
        for pid in buckets[split][half:]:
            records.append(_control_trace(pid))
    summary = analyze_traces(records, window_len=32, n_boot=30, seed=0)
    assert summary["data_status"] == "underpowered"
    assert summary["metrics_computed"] is True
    assert summary["labelling"]["n_loop_with_onset"] == 14
    assert summary["labelling"]["n_control"] == 14
    from jspace.forecasting.run import prepare_traces

    prepared, skipped = prepare_traces(records)
    assert skipped == []
    onsets = []
    for tr in prepared:
        label = label_trace(tr)
        if label.is_loop:
            onsets.append(label.onset_token_index)
    assert onsets == [342] * 14
    ngram = summary["detectors"]["ngram_repeat"]
    h32 = ngram["per_horizon"]["32"]["test"]
    assert abs(h32["auroc"] - 0.5) < 1e-9
    assert ngram["lead_time"]["early_rate"] == 0.0
    assert ngram["lead_time"]["median_signed_lead_tokens"] < 0
    assert ngram["fixed_rule_on_test"]["median_lag_tokens"] > 0
    assert ngram["fixed_rule_on_test"]["n_before_onset"] == 0
    surface = summary["detectors"]["surface_logreg"]["per_horizon"]["32"]["test"]
    assert surface["auroc"] > 0.9
    assert surface["ece"] < 0.35
    assert summary["detectors"]["surface_logreg"]["lead_time"]["early_rate"] >= 0.75
    assert summary["detectors"]["surface_lightgbm"]["per_horizon"]["64"]["test"]["auroc"] > 0.9
    assert summary["feature_sets"]["surface_repetition_logreg"]["per_horizon"]["128"]["test"]["auroc"] > 0.9
    cusum = summary["detectors"]["cusum_ngram"]
    assert cusum["lead_time"]["early_rate"] == 0.0
    assert cusum["reference"]["n_tokens"] > 0


def test_loader_ignores_previews_and_reads_meta(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        looping = root / "exp2_demo" / "looping" / "p15"
        looping.mkdir(parents=True)
        (looping / "meta.json").write_text(json.dumps({
            "prompt_id": 15,
            "is_loop": True,
            "generated_text": "abcdefghijklmnop" * 4,
            "trigger_token_index": 16,
            "n_tokens": 64,
            "source": "unit",
        }), encoding="utf-8")
        with patch("jspace.forecasting.data._read_tokens_pt", return_value=(list(range(64)), None)):
            (looping / "tokens.pt").write_bytes(b"placeholder")
            rec = load_trace_dir(looping)
            traces, info = load_baseline_traces(root, "demo")
        assert rec is not None and rec.token_ids == list(range(64))
        assert rec.text_complete
        assert info["n_trace_dirs"] == 1
        log = root / "trigger_gen_log_demo.jsonl"
        log.write_text(json.dumps({"prompt_id": 15, "generated_preview": "abc", "n_tokens": 4000, "is_loop": True}) + "\n", encoding="utf-8")
        _traces, info2 = load_baseline_traces(root, "demo")
        assert info2["n_preview_rows"] == 1
        assert info2["n_trace_dirs"] == 1
        assert all(tr.text_complete and tr.generated_text.startswith("abcdefghijklmnop") for tr in _traces)


def test_lfm_checkout_does_not_invent_forecasts():
    root = Path(__file__).resolve().parents[1]
    summary = summarize_results(root / "results", "lfm2-2.6b", n_boot=5)
    assert summary["data_status"] == "insufficient"
    assert summary["metrics_computed"] is False
    assert summary["detectors"] is None
    assert summary["labelling"]["n_usable"] == 0
    recorded = summary["recorded"]["baseline_summary_legacy"]
    assert recorded["n_loop"] == 11
    assert recorded["n_completed"] == 200
    assert summary["recorded"]["exp2_checkpoint"]["n_loop"] == 5
    assert summary["prompt_sample"]["recorded_baseline_loops_by_split"] == {"train": 8, "dev": 1, "test": 2}
    text = render_report(summary)
    assert "No forecasting metrics were computed." in text
    assert "11" in text
    assert "0.99" not in text
    assert summary["load"]["n_trace_dirs"] == 0


def _run_all() -> int:
    fns = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception:
            failed += 1
            import traceback
            traceback.print_exc()
            print("FAIL", fn.__name__)
        else:
            print("OK", fn.__name__)
    if failed:
        print(f"{failed} failed")
        return 1
    print(f"ALL {len(fns)} FORECAST TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
