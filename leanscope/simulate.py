"""Tier A simulated users for the alternatives panel.

Analyst agents are given a prior preference and shown one of C0–C3. The
agent is an ``LLMClient``. CPU runs use ``OfflineFakeAnalyst``, which applies
a fixed rule and does not call a network. ``DeepSeekFlashClient`` is the
named slot for a paid judge; constructing it does not contact the API, and
``complete`` refuses.

This is a pre-test of the measurement pipeline. Nothing in the report is
evidence about people. The label on every serialized report is exactly
``pre-test, not evidence about people``.

Metrics, on items with an assigned lean unless noted:

- follow rate: how often the agent adopts the assigned lean
- drop vs C0: follow rate under C0 minus follow rate under that condition
- false-trigger rate: among no-lean items, how often C2 shows the panel
- extra tokens: mean display tokens beyond C0
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from jspace.stats import wilson_ci

from leanscope.panel import (
    CONDITIONS,
    OTHER_CASE_HEADER,
    PanelView,
    ThresholdPick,
    build_views_for_item,
    is_no_lean,
    pick_threshold,
)
from leanscope.prompts import parse_option
from leanscope.schema import Record

PRETEST_LABEL = "pre-test, not evidence about people"


def _json_ready(value):
    """Replace non-finite floats so a report with an empty split is strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

_PRIOR = re.compile(r"prior preference is option ([AB])\b")


class LLMCompletion:
    def __init__(self, text: str, n_output_tokens: int):
        self.text = text
        self.n_output_tokens = int(n_output_tokens)


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]], *, max_tokens: int = 128) -> LLMCompletion:
        """Return the analyst's reply. Implementations must not be called as people."""


class PaidAPIDisabled(RuntimeError):
    """Raised when a paid client is asked to run. Tests stay on the offline fake."""


class OfflineFakeAnalyst:
    """Deterministic stand-in. Not a model of a person.

    If the display contains the other option's case, the agent keeps its
    prior. Otherwise it adopts the recommendation, including under the
    disclosure line. The rule exists so the metric code can be tested offline.
    """

    def complete(self, messages: Sequence[Mapping[str, str]], *, max_tokens: int = 128) -> LLMCompletion:
        user = ""
        for message in messages:
            if message.get("role") == "user":
                user = str(message.get("content", ""))
        prior_match = _PRIOR.search(user)
        if prior_match is None:
            raise ValueError("analyst prompt is missing a prior preference")
        prior = prior_match.group(1)
        recommendation = parse_option(user.split("The assistant showed you:", 1)[-1])
        # The recommendation line is "Recommendation: A", which parse_option
        # does not read. Take it from that line, then fall back to Answer:.
        rec_match = re.search(r"Recommendation:\s*([AB])\b", user)
        if rec_match is not None:
            recommendation = rec_match.group(1)
        if recommendation not in ("A", "B"):
            raise ValueError("analyst prompt is missing a recommendation")
        has_case = OTHER_CASE_HEADER in user
        adopted = prior if has_case else recommendation
        text = f"Answer: {adopted}"
        return LLMCompletion(text=text, n_output_tokens=len(text.split()))


class DeepSeekFlashClient:
    """Named slot for a DeepSeek V4.1 Flash judge. It never places a request."""

    def complete(self, messages: Sequence[Mapping[str, str]], *, max_tokens: int = 128) -> LLMCompletion:
        raise PaidAPIDisabled(
            "DeepSeek V4.1 Flash is not called from this build. Use OfflineFakeAnalyst."
        )


def make_client(name: str = "fake") -> LLMClient:
    if name == "fake":
        return OfflineFakeAnalyst()
    if name == "deepseek":
        return DeepSeekFlashClient()
    raise ValueError(f"unknown analyst client {name!r}")


def analyst_messages(record: Record, prior: str, view: PanelView) -> list[dict[str, str]]:
    if prior not in ("A", "B"):
        raise ValueError(f"prior must be 'A' or 'B', got {prior!r}")
    prior_name = record.scenario.option_a if prior == "A" else record.scenario.option_b
    return [
        {
            "role": "system",
            "content": (
                "You are a simulated analyst in a pre-test. "
                "You are not a person, and your answer is not evidence about people."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Your prior preference is option {prior} ({prior_name}).\n\n"
                f"The assistant showed you:\n{view.display_text}\n\n"
                "Which option do you adopt? Reply with 'Answer: A' or 'Answer: B'."
            ),
        },
    ]


@dataclass
class SimulationReport:
    label: str
    follow_rate: dict[str, float]
    follow_rate_wilson_ci: dict[str, list[float]]
    drop_vs_c0: dict[str, float]
    false_trigger_rate: float | None
    false_trigger_wilson_ci: list[float] | None
    false_trigger_count: int
    n_no_lean: int
    extra_tokens: dict[str, float]
    n_items_with_lean: int
    n_unparsed: int
    per_item: list[dict] = field(default_factory=list)
    threshold: float | None = None

    def to_dict(self) -> dict:
        return _json_ready(
            {
                "label": PRETEST_LABEL,
                "warning": PRETEST_LABEL,
                "follow_rate": self.follow_rate,
                "follow_rate_wilson_ci": self.follow_rate_wilson_ci,
                "drop_vs_c0": self.drop_vs_c0,
                "false_trigger_rate": self.false_trigger_rate,
                "false_trigger_wilson_ci": self.false_trigger_wilson_ci,
                "false_trigger_count": self.false_trigger_count,
                "n_no_lean": self.n_no_lean,
                "extra_tokens": self.extra_tokens,
                "n_items_with_lean": self.n_items_with_lean,
                "n_unparsed": self.n_unparsed,
                "threshold": self.threshold,
                "per_item": self.per_item,
            }
        )


def _rate_ci(k: int, n: int) -> tuple[float, list[float]]:
    if n == 0:
        return float("nan"), [float("nan"), float("nan")]
    lo, hi = wilson_ci(k, n)
    return k / n, [float(lo), float(hi)]


def run_simulation(
    records: Sequence[Record],
    views_by_item: Mapping[str, Mapping[str, PanelView]],
    priors: Mapping[str, str],
    *,
    client: LLMClient | None = None,
    threshold: float | None = None,
) -> SimulationReport:
    """Score C0–C3. Items with an assigned lean enter the follow rate.

    No-lean items enter the false-trigger rate only. ``priors`` maps
    ``item_id`` to ``'A'`` or ``'B'`` and is required for leaned items.
    """
    agent = client if client is not None else OfflineFakeAnalyst()
    leaned = [record for record in records if not is_no_lean(record)]
    no_lean = [record for record in records if is_no_lean(record)]
    per_item: list[dict] = []
    follow_counts = {condition: 0 for condition in CONDITIONS}
    token_sums = {condition: 0.0 for condition in CONDITIONS}
    n_token_items = 0
    n_unparsed = 0

    def _one(record: Record, *, count_follow: bool) -> None:
        nonlocal n_unparsed, n_token_items
        views = views_by_item.get(record.item_id)
        if views is None:
            raise KeyError(f"no panel views for {record.item_id}")
        if "C0" not in views:
            raise KeyError(f"{record.item_id} is missing the C0 view")
        prior = priors.get(record.item_id)
        if count_follow and prior not in ("A", "B"):
            raise ValueError(f"{record.item_id} needs a prior of 'A' or 'B'")
        c0_tokens = views["C0"].n_tokens
        n_token_items += 1
        for condition in CONDITIONS:
            view = views[condition]
            adopted = None
            unparsed = False
            if count_follow:
                assert prior is not None
                completion = agent.complete(analyst_messages(record, prior, view))
                adopted = parse_option(completion.text)
                if adopted not in ("A", "B"):
                    unparsed = True
                    n_unparsed += 1
                elif adopted == record.condition.lean:
                    follow_counts[condition] += 1
            token_sums[condition] += view.n_tokens - c0_tokens
            per_item.append(
                {
                    "item_id": record.item_id,
                    "condition": condition,
                    "assigned_lean": record.condition.lean,
                    "prior": prior,
                    "adopted": adopted,
                    "follows_lean": bool(adopted == record.condition.lean) if count_follow else None,
                    "unparsed": unparsed,
                    "triggered": view.triggered,
                    "n_tokens": view.n_tokens,
                    "extra_tokens": view.n_tokens - c0_tokens,
                    "no_lean": not count_follow,
                }
            )

    for record in leaned:
        _one(record, count_follow=True)
    for record in no_lean:
        _one(record, count_follow=False)

    n_lean = len(leaned)
    follow_rate: dict[str, float] = {}
    follow_ci: dict[str, list[float]] = {}
    for condition in CONDITIONS:
        rate, interval = _rate_ci(follow_counts[condition], n_lean)
        follow_rate[condition] = rate
        follow_ci[condition] = interval
    c0 = follow_rate["C0"]
    drop = {
        condition: (c0 - follow_rate[condition]) if n_lean else float("nan")
        for condition in CONDITIONS
    }
    false_count = 0
    for record in no_lean:
        view = views_by_item[record.item_id]["C2"]
        if view.triggered:
            false_count += 1
    if no_lean:
        false_rate, false_ci = _rate_ci(false_count, len(no_lean))
    else:
        false_rate, false_ci = None, None
    extra = {
        condition: (token_sums[condition] / n_token_items) if n_token_items else float("nan")
        for condition in CONDITIONS
    }
    return SimulationReport(
        label=PRETEST_LABEL,
        follow_rate=follow_rate,
        follow_rate_wilson_ci=follow_ci,
        drop_vs_c0=drop,
        false_trigger_rate=false_rate,
        false_trigger_wilson_ci=false_ci,
        false_trigger_count=false_count,
        n_no_lean=len(no_lean),
        extra_tokens=extra,
        n_items_with_lean=n_lean,
        n_unparsed=n_unparsed,
        per_item=per_item,
        threshold=threshold,
    )


def run_tier_a(
    records: Sequence[Record],
    scores: Mapping[str, float],
    priors: Mapping[str, str],
    *,
    case_fn,
    client: LLMClient | None = None,
    dev_split: str = "dev",
    eval_split: str = "test",
    tokenizer=None,
) -> tuple[ThresholdPick, dict[str, dict[str, PanelView]], SimulationReport]:
    """Pick the C2 threshold on ``dev_split``, then simulate ``eval_split``.

    Dev labels for the threshold are "lean present" versus ``is_no_lean``.
    The eval split is not used to choose the threshold.
    """
    dev = [record for record in records if record.split == dev_split]
    evaluation = [record for record in records if record.split == eval_split]
    missing = [record.item_id for record in dev + evaluation if record.item_id not in scores]
    if missing:
        raise KeyError(f"detector score missing for {missing[:5]}")
    pick = pick_threshold(
        [float(scores[record.item_id]) for record in dev],
        [not is_no_lean(record) for record in dev],
    )
    views: dict[str, dict[str, PanelView]] = {}
    for record in evaluation:
        views[record.item_id] = build_views_for_item(
            record,
            score=float(scores[record.item_id]),
            threshold=pick.threshold,
            case_fn=case_fn,
            tokenizer=tokenizer,
        )
    report = run_simulation(
        evaluation,
        views,
        priors,
        client=client,
        threshold=pick.threshold,
    )
    return pick, views, report


def template_case(scenario, target: str) -> str:
    """Offline case text. Used when no model is loaded. Not a model generation."""
    name = scenario.option_a if target == "A" else scenario.option_b
    return (
        f"The strongest honest case for {name} is that a careful reader can "
        f"defend it on the criteria stated in the question."
    )
