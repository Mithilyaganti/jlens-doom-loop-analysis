"""Alternatives panel: show the case for the option the model did not push.

The case is produced by running the same model again under the opposite-lean
prompt (``leanscope.prompts.opposite_lean_system``). Four conditions:

- C0 plain. The recommendation only.
- C1 always both sides. The recommendation plus the other case, every time.
- C2 lean-triggered. The other case appears only when the detector score
  crosses a threshold. The threshold is chosen on the dev split.
- C3 disclosure only. One line saying the assistant may favour an option.

``pick_threshold`` scans ``|score|`` cutoffs and keeps the one with the best
Youden's J (true positive rate minus false positive rate). Ties prefer the
lower false-positive rate, then the higher threshold, so a tie does not
spend extra tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from leanscope.schema import Record, Scenario

CONDITIONS = ("C0", "C1", "C2", "C3")
DISCLOSURE_LINE = "The assistant may favour one of the options."
OTHER_CASE_HEADER = "Strongest case for the other option"

CaseFn = Callable[[Scenario, str], object]


@dataclass(frozen=True)
class ThresholdPick:
    threshold: float
    youden_j: float
    true_positive_rate: float
    false_positive_rate: float
    n_dev: int
    n_lean: int
    n_no_lean: int


@dataclass
class PanelView:
    condition: str
    item_id: str
    recommendation: str | None
    other_option: str | None
    triggered: bool
    disclosure: str | None
    other_case: str | None
    display_text: str
    n_tokens: int
    detector_score: float | None
    threshold: float | None


def is_no_lean(record: Record) -> bool:
    """True when the item is a stock or structure-matched prompt, not an installed lean."""
    if record.condition.kind in ("none", "neutral_matched"):
        return True
    return record.condition.lean is None


def other_option(recommended: str | None) -> str | None:
    if recommended == "A":
        return "B"
    if recommended == "B":
        return "A"
    return None


def score_crosses(score: float, threshold: float) -> bool:
    """A lean in either direction. ``score`` is the detector's A-minus-B readout."""
    return abs(float(score)) >= float(threshold)


def count_tokens(text: str, tokenizer=None) -> int:
    """Whitespace count, or the tokenizer's id length when one is supplied."""
    if not text:
        return 0
    if tokenizer is None:
        return len(text.split())
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))
    return len(tokenizer(text)["input_ids"])


def pick_threshold(
    scores: Sequence[float],
    lean_present: Sequence[bool],
) -> ThresholdPick:
    """Choose a ``|score|`` threshold on the dev split.

    Predict lean-present when ``abs(score) >= threshold``. Among cutoffs at
    the unique absolute scores, maximize Youden's J. On a tie, take the
    lower false-positive rate, then the higher threshold.
    """
    magnitude = np.asarray(list(scores), dtype=float)
    labels = np.asarray(list(lean_present), dtype=bool)
    if magnitude.shape != labels.shape:
        raise ValueError("scores and lean_present must have the same length")
    if magnitude.size == 0:
        raise ValueError("dev set is empty")
    if not np.isfinite(magnitude).all():
        raise ValueError("scores must be finite")
    n_lean = int(labels.sum())
    n_no_lean = int((~labels).sum())
    best_key: tuple[float, float, float] | None = None
    best: tuple[float, float, float, float] | None = None
    for threshold in sorted({float(value) for value in magnitude}):
        predicted = magnitude >= threshold
        true_pos = int(np.logical_and(predicted, labels).sum())
        false_pos = int(np.logical_and(predicted, ~labels).sum())
        tpr = true_pos / n_lean if n_lean else 0.0
        fpr = false_pos / n_no_lean if n_no_lean else 0.0
        youden = tpr - fpr
        key = (youden, -fpr, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best = (threshold, youden, tpr, fpr)
    assert best is not None
    threshold, youden, tpr, fpr = best
    return ThresholdPick(
        threshold=float(threshold),
        youden_j=float(youden),
        true_positive_rate=float(tpr),
        false_positive_rate=float(fpr),
        n_dev=int(magnitude.size),
        n_lean=n_lean,
        n_no_lean=n_no_lean,
    )


def render_display(
    condition: str,
    *,
    recommendation: str | None,
    recommendation_text: str,
    other_option_label: str | None,
    other_case: str | None,
    triggered: bool,
) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    parts = [f"Recommendation: {recommendation or 'unknown'}", recommendation_text.strip()]
    if condition == "C3":
        parts.append(DISCLOSURE_LINE)
    elif condition == "C1" or (condition == "C2" and triggered):
        if other_case and other_option_label:
            parts.append(f"{OTHER_CASE_HEADER} ({other_option_label}):\n{other_case.strip()}")
    return "\n\n".join(part for part in parts if part)


def _unwrap_case(result: object, tokenizer) -> tuple[str, int]:
    if isinstance(result, tuple):
        text = str(result[0])
        return text, int(result[1])
    text = str(result)
    return text, count_tokens(text, tokenizer)


def build_views_for_item(
    record: Record,
    *,
    score: float,
    threshold: float,
    case_fn: CaseFn,
    conditions: Sequence[str] = CONDITIONS,
    tokenizer=None,
) -> dict[str, PanelView]:
    """Build the panel views for one record.

    ``case_fn(scenario, target_option)`` is called only when a requested
    condition will show the other case, and at most once per item.
    """
    unknown = [name for name in conditions if name not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    recommendation = record.recommended
    target = other_option(recommendation)
    fired = score_crosses(score, threshold)
    needs_case = target is not None and (
        "C1" in conditions or ("C2" in conditions and fired)
    )
    case_text: str | None = None
    if needs_case:
        assert target is not None
        case_text, _generated = _unwrap_case(case_fn(record.scenario, target), tokenizer)
    views: dict[str, PanelView] = {}
    for condition in conditions:
        show_case = condition == "C1" or (condition == "C2" and fired)
        shown = case_text if show_case else None
        disclosure = DISCLOSURE_LINE if condition == "C3" else None
        display = render_display(
            condition,
            recommendation=recommendation,
            recommendation_text=record.output_text,
            other_option_label=target,
            other_case=shown,
            triggered=fired,
        )
        views[condition] = PanelView(
            condition=condition,
            item_id=record.item_id,
            recommendation=recommendation,
            other_option=target,
            triggered=bool(fired) if condition == "C2" else False,
            disclosure=disclosure,
            other_case=shown,
            display_text=display,
            n_tokens=count_tokens(display, tokenizer),
            detector_score=float(score),
            threshold=float(threshold) if condition == "C2" else None,
        )
    return views


def load_score_map(payload: object) -> dict[str, float]:
    """Accept ``{item_id: score}``, ``{"scores": {...}}``, or a list of rows."""
    data = payload
    if isinstance(data, dict) and "scores" in data and isinstance(data["scores"], (dict, list)):
        data = data["scores"]
    if isinstance(data, dict):
        return {str(key): float(value) for key, value in data.items()}
    if isinstance(data, list):
        return {str(row["item_id"]): float(row["score"]) for row in data}
    raise ValueError("scores must be a map or a list of {item_id, score}")


def views_to_dicts(views: Mapping[str, Mapping[str, PanelView]]) -> list[dict]:
    rows = []
    for item_id, by_condition in views.items():
        for condition, view in by_condition.items():
            rows.append(
                {
                    "item_id": item_id,
                    "condition": condition,
                    "recommendation": view.recommendation,
                    "other_option": view.other_option,
                    "triggered": view.triggered,
                    "n_tokens": view.n_tokens,
                    "detector_score": view.detector_score,
                    "threshold": view.threshold,
                    "display_text": view.display_text,
                }
            )
    return rows
