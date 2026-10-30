"""CPU audit of the vendored Antidoom loop sampler.

The study labels loops with ``jspace.detection.find_inner_repetition``: a
16-character fingerprint every 128 characters, then a verify step. That
sampler is not an exhaustive search over periods. This module compares it
to an exhaustive period scan on caller-supplied text. It does not relabel
paper-1 traces and it does not invent recall on generations that are not
in the checkout.

Recall is ``sampler hits / exhaustive positives``. If a slice has no
exhaustive positive, recall is not measured.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from jspace.detection import find_inner_repetition

# Published Antidoom defaults. Kept here so a sweep can name the study rule
# without treating any other grid point as a tuned replacement.
PUBLISHED_MIN_REPEATS: int = 4
PUBLISHED_MIN_TOTAL_REPEATED: int = 60
PUBLISHED_MIN_PERIOD: int = 1
PUBLISHED_MAX_PERIOD: int = 1024
PUBLISHED_SAMPLE_LEN: int = 16
PUBLISHED_SAMPLE_INTERVAL: int = 128

HUMAN_AUDIT_COLUMNS: tuple[str, ...] = (
    "auditor",
    "trace_id",
    "source_path",
    "char_len",
    "sampler_hit",
    "exhaustive_hit",
    "human_is_loop",
    "human_notes",
    "disagreement_reason",
)


@dataclass(frozen=True)
class RepetitionHit:
    """One periodic span that meets the numeric rule."""

    start: int
    end: int
    period: int
    repeats: int

    def as_dict(self) -> dict[str, int]:
        return {
            "start": self.start,
            "end": self.end,
            "period": self.period,
            "repeats": self.repeats,
        }


def exhaustive_repetition(
    text: str,
    *,
    min_repeats: int = PUBLISHED_MIN_REPEATS,
    max_period: int = PUBLISHED_MAX_PERIOD,
    min_period: int = PUBLISHED_MIN_PERIOD,
    min_total_repeated: int = PUBLISHED_MIN_TOTAL_REPEATED,
) -> tuple[bool, RepetitionHit | None]:
    """Return the first period that has a run meeting the numeric rule.

    For each period ``p``, a run of positions where ``text[i] == text[i+p]``
    is a periodic span. Repeats are ``span // p`` and the span must be at
    least ``min_total_repeated`` characters. Periods are tried from
    ``min_period`` upward, so the hit is the shortest qualifying period,
    not a claim about which period the fingerprint sampler would report.
    """
    if min_repeats < 2:
        raise ValueError("min_repeats must be >= 2")
    if min_period < 1:
        raise ValueError("min_period must be >= 1")
    if not text or len(text) < min_total_repeated:
        return False, None

    n = len(text)
    period_limit = min(max_period, n // min_repeats)
    if period_limit < min_period:
        return False, None

    for period in range(min_period, period_limit + 1):
        span = n - period
        i = 0
        while i < span:
            if text[i] != text[i + period]:
                i += 1
                continue
            j = i + 1
            while j < span and text[j] == text[j + period]:
                j += 1
            # Equality on [i, j) plus one period is the periodic span.
            total = (j - i) + period
            repeats = total // period
            whole = repeats * period
            if repeats >= min_repeats and whole >= min_total_repeated:
                return True, RepetitionHit(
                    start=i,
                    end=i + whole,
                    period=period,
                    repeats=repeats,
                )
            i = j if j > i else i + 1
    return False, None


def sampler_hit(
    text: str,
    *,
    min_repeats: int = PUBLISHED_MIN_REPEATS,
    max_period: int = PUBLISHED_MAX_PERIOD,
    min_period: int = PUBLISHED_MIN_PERIOD,
    min_total_repeated: int = PUBLISHED_MIN_TOTAL_REPEATED,
    sample_len: int = PUBLISHED_SAMPLE_LEN,
    sample_interval: int = PUBLISHED_SAMPLE_INTERVAL,
) -> bool:
    """Published fingerprint sampler. Returns only the boolean label."""
    found, _hit = find_inner_repetition(
        text,
        min_repeats=min_repeats,
        max_period=max_period,
        min_period=min_period,
        min_total_repeated=min_total_repeated,
        sample_len=sample_len,
        sample_interval=sample_interval,
    )
    return bool(found)


def recall_from_labels(gold: Sequence[bool], predicted: Sequence[bool]) -> dict[str, object]:
    """Sampler recall against exhaustive positives.

    ``gold`` is the exhaustive label. Recall is not measured when there is
    no positive gold case.
    """
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    positives = 0
    hits = 0
    misses = 0
    for g, p in zip(gold, predicted):
        if not g:
            continue
        positives += 1
        if p:
            hits += 1
        else:
            misses += 1
    if positives == 0:
        return {
            "status": "not_measured",
            "reason": "no exhaustive positive in this slice",
            "n_positive": 0,
            "hits": 0,
            "misses": 0,
            "recall": None,
        }
    return {
        "status": "measured",
        "n_positive": positives,
        "hits": hits,
        "misses": misses,
        "recall": hits / positives,
    }


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    kind: str
    text: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "Fixture":
        return cls(
            fixture_id=str(row["id"]),
            kind=str(row["kind"]),
            text=str(row["text"]),
        )


def load_fixtures(path: Path | str) -> list[Fixture]:
    path = Path(path)
    fixtures: list[Fixture] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "id" not in row or "kind" not in row or "text" not in row:
            raise ValueError(f"{path}:{line_no} missing id, kind, or text")
        fixtures.append(Fixture.from_mapping(row))
    if not fixtures:
        raise ValueError(f"no fixtures in {path}")
    return fixtures


def score_texts(
    texts: Sequence[str],
    *,
    min_repeats: int = PUBLISHED_MIN_REPEATS,
    min_total_repeated: int = PUBLISHED_MIN_TOTAL_REPEATED,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for text in texts:
        exhaustive, hit = exhaustive_repetition(
            text,
            min_repeats=min_repeats,
            min_total_repeated=min_total_repeated,
        )
        sampled = sampler_hit(
            text,
            min_repeats=min_repeats,
            min_total_repeated=min_total_repeated,
        )
        rows.append(
            {
                "n_chars": len(text),
                "exhaustive": exhaustive,
                "sampler": sampled,
                "exhaustive_period": None if hit is None else hit.period,
                "exhaustive_repeats": None if hit is None else hit.repeats,
            }
        )
    return rows


def recall_for_kind(fixtures: Sequence[Fixture], kind: str) -> dict[str, object]:
    chosen = [fx for fx in fixtures if fx.kind == kind]
    if not chosen:
        return {
            "status": "not_measured",
            "reason": f"no fixtures with kind={kind!r}",
            "n_fixtures": 0,
            "n_positive": 0,
            "hits": 0,
            "misses": 0,
            "recall": None,
        }
    scored = score_texts([fx.text for fx in chosen])
    out = recall_from_labels(
        [bool(row["exhaustive"]) for row in scored],
        [bool(row["sampler"]) for row in scored],
    )
    out["kind"] = kind
    out["n_fixtures"] = len(chosen)
    return out


def threshold_sweep(
    fixtures: Sequence[Fixture],
    *,
    min_repeats_grid: Iterable[int] = (2, 3, 4, 5),
    min_total_grid: Iterable[int] = (40, 60, 80),
) -> list[dict[str, object]]:
    """Agreement of the sampler and the exhaustive scan on a small grid.

    The published (4, 60) cell is marked. The sweep does not pick a winner.
    """
    rows: list[dict[str, object]] = []
    texts = [fx.text for fx in fixtures]
    for min_repeats in min_repeats_grid:
        for min_total in min_total_grid:
            scored = score_texts(
                texts,
                min_repeats=int(min_repeats),
                min_total_repeated=int(min_total),
            )
            agree = sum(1 for row in scored if row["exhaustive"] == row["sampler"])
            recall = recall_from_labels(
                [bool(row["exhaustive"]) for row in scored],
                [bool(row["sampler"]) for row in scored],
            )
            rows.append(
                {
                    "min_repeats": int(min_repeats),
                    "min_total_repeated": int(min_total),
                    "published_rule": (
                        int(min_repeats) == PUBLISHED_MIN_REPEATS
                        and int(min_total) == PUBLISHED_MIN_TOTAL_REPEATED
                    ),
                    "n_fixtures": len(texts),
                    "n_exhaustive_positive": sum(1 for row in scored if row["exhaustive"]),
                    "n_sampler_positive": sum(1 for row in scored if row["sampler"]),
                    "n_agree": agree,
                    "sampler_recall_vs_exhaustive": recall["recall"],
                    "recall_status": recall["status"],
                }
            )
    return rows


def extract_html_pre(html: str) -> str | None:
    """Return the first ``<pre>`` body, or None if the page has none."""
    match = re.search(r"<pre>(.*?)</pre>", html, flags=re.DOTALL)
    if match is None:
        return None
    return match.group(1)


def score_checked_in_html(path: Path | str) -> dict[str, object]:
    """Score one checked-in HTML trace. This is a single example, not a rate."""
    path = Path(path)
    if not path.is_file():
        return {
            "status": "not_measured",
            "reason": f"missing {path}",
            "path": str(path),
        }
    text = extract_html_pre(path.read_text(encoding="utf-8"))
    if text is None:
        return {
            "status": "not_measured",
            "reason": f"no <pre> block in {path}",
            "path": str(path),
        }
    scored = score_texts([text])[0]
    return {
        "status": "measured_one_example",
        "path": str(path),
        "n_chars": scored["n_chars"],
        "exhaustive": scored["exhaustive"],
        "sampler": scored["sampler"],
        "note": (
            "One checked-in example trace. Not a recall for the study corpus. "
            "Full per-prompt generations are not in this checkout."
        ),
    }


def corpus_recall_status() -> dict[str, object]:
    """The generation logs in-repo are previews, so corpus recall is not measured."""
    return {
        "status": "not_measured",
        "reason": (
            "results/trigger_gen_log.jsonl and results/trigger_gen_log_partial_easy.jsonl "
            "store truncated generated_preview strings plus n_tokens. They are not "
            "full generations, so sampler recall on the study corpus was not measured."
        ),
    }


def audit_report(
    fixtures: Sequence[Fixture],
    *,
    example_html: Path | str | None = None,
) -> dict[str, object]:
    per_fixture: list[dict[str, object]] = []
    for fx in fixtures:
        row = score_texts([fx.text])[0]
        per_fixture.append(
            {
                "id": fx.fixture_id,
                "kind": fx.kind,
                **row,
            }
        )
    report: dict[str, object] = {
        "published_rule": {
            "min_repeats": PUBLISHED_MIN_REPEATS,
            "min_total_repeated": PUBLISHED_MIN_TOTAL_REPEATED,
            "sample_len": PUBLISHED_SAMPLE_LEN,
            "sample_interval": PUBLISHED_SAMPLE_INTERVAL,
        },
        "fixtures": per_fixture,
        "recall_short": recall_for_kind(fixtures, "short"),
        "recall_tail": recall_for_kind(fixtures, "tail"),
        "threshold_sweep": threshold_sweep(fixtures),
        "corpus_recall": corpus_recall_status(),
    }
    if example_html is not None:
        report["checked_in_example_trace"] = score_checked_in_html(example_html)
    return report


def human_audit_header() -> str:
    """CSV header only. No judgment rows."""
    return ",".join(HUMAN_AUDIT_COLUMNS) + "\n"
