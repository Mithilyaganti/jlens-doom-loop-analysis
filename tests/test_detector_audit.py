"""CPU tests for the loop-detector audit. No model and no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jspace.detector_audit import (  # noqa: E402
    HUMAN_AUDIT_COLUMNS,
    audit_report,
    exhaustive_repetition,
    human_audit_header,
    load_fixtures,
    recall_for_kind,
    recall_from_labels,
    sampler_hit,
)

FIXTURES = ROOT / "tests" / "fixtures" / "detector_loops.jsonl"
EXAMPLE_HTML = ROOT / "results" / "exp2" / "example_looping_trace.html"


def test_exhaustive_matches_numeric_rule_on_short_period() -> None:
    text = "abcdefghijklmno" * 4
    assert len(text) == 60
    found, hit = exhaustive_repetition(text)
    assert found and hit is not None
    assert hit.period == 15
    assert hit.repeats == 4
    assert sampler_hit(text) is False


def test_recall_not_measured_without_positives() -> None:
    out = recall_from_labels([False, False], [False, True])
    assert out["status"] == "not_measured"
    assert out["recall"] is None


def test_recall_is_hits_over_positives() -> None:
    out = recall_from_labels([True, True, False], [True, False, True])
    assert out["status"] == "measured"
    assert out["hits"] == 1
    assert out["misses"] == 1
    assert out["recall"] == 0.5


def test_fixture_recall_slices_are_derived() -> None:
    fixtures = load_fixtures(FIXTURES)
    short = recall_for_kind(fixtures, "short")
    tail = recall_for_kind(fixtures, "tail")
    assert short["status"] == "measured"
    assert tail["status"] == "measured"
    assert short["n_positive"] == short["hits"] + short["misses"]
    assert tail["n_positive"] == tail["hits"] + tail["misses"]
    assert short["recall"] == short["hits"] / short["n_positive"]
    assert tail["recall"] == tail["hits"] / tail["n_positive"]
    # The 15-char x4 fixture is an exhaustive positive the sampler misses.
    assert short["misses"] >= 1
    # The tail slice includes one catch (unit x5) so recall is not zero.
    assert tail["hits"] >= 1
    assert tail["misses"] >= 1


def test_report_does_not_invent_corpus_recall() -> None:
    fixtures = load_fixtures(FIXTURES)
    report = audit_report(fixtures, example_html=EXAMPLE_HTML)
    assert report["corpus_recall"]["status"] == "not_measured"
    example = report["checked_in_example_trace"]
    assert example["status"] in {"measured_one_example", "not_measured"}
    if example["status"] == "measured_one_example":
        assert isinstance(example["sampler"], bool)
        assert isinstance(example["exhaustive"], bool)
        assert "recall" not in example
    published = [row for row in report["threshold_sweep"] if row["published_rule"]]
    assert len(published) == 1
    assert published[0]["min_repeats"] == 4
    assert published[0]["min_total_repeated"] == 60


def test_human_sheet_is_header_only() -> None:
    header = human_audit_header()
    rows = [line for line in header.splitlines() if line.strip()]
    assert rows == [",".join(HUMAN_AUDIT_COLUMNS)]
    assert header.count("\n") == 1
    committed = (ROOT / "audit" / "human_loop_audit_sheet.csv").read_text(encoding="utf-8")
    assert committed == header


def test_negative_fixtures_are_not_exhaustive_positives() -> None:
    fixtures = load_fixtures(FIXTURES)
    negatives = [fx for fx in fixtures if fx.kind == "negative"]
    assert negatives
    for fx in negatives:
        found, _hit = exhaustive_repetition(fx.text)
        assert found is False
        assert sampler_hit(fx.text) is False
