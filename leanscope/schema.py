"""Shared LeanScope record contract.

Three agents write code against these dataclasses. Field names and the allowed
values are the contract: do not add, rename, or drop fields.

Coordinates. ``option_a`` / ``option_b`` are the canonical contents of a
scenario. ``LeanCondition.lean`` and ``Record.recommended`` use the same
coordinates: ``"A"`` always means ``option_a``, ``"B"`` always means
``option_b``. ``Scenario.order`` only changes which option block is printed
first. The letters in the user message stay attached to those contents, so a
system prompt that says "option A" is asking for ``option_a`` under both orders.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

Family = Literal["analysis", "product", "advice"]
Order = Literal["AB", "BA"]
Kind = Literal["none", "assigned", "neutral_matched", "trait", "conceal"]
Strength = Literal["explicit", "soft", "indirect", "conflicting"]
Side = Literal["A", "B"]
SplitName = Literal["train", "dev", "test"]

FAMILIES: tuple[Family, ...] = ("analysis", "product", "advice")
ORDERS: tuple[Order, ...] = ("AB", "BA")
KINDS: tuple[Kind, ...] = ("none", "assigned", "neutral_matched", "trait", "conceal")
STRENGTHS: tuple[Strength, ...] = ("explicit", "soft", "indirect", "conflicting")
SIDES: tuple[Side, ...] = ("A", "B")
SPLITS: tuple[SplitName, ...] = ("train", "dev", "test")

_SCENARIO_FIELDS = (
    "scenario_id",
    "family",
    "question",
    "option_a",
    "option_b",
    "paraphrase_id",
    "order",
)
_CONDITION_FIELDS = ("kind", "strength", "lean", "system_prompt")
_RECORD_FIELDS = (
    "item_id",
    "scenario",
    "condition",
    "output_text",
    "recommended",
    "voiced",
    "activations_path",
    "split",
)


def _reject_unknown(payload: dict[str, Any], fields: tuple[str, ...], what: str) -> None:
    unknown = sorted(set(payload) - set(fields))
    if unknown:
        raise ValueError(f"Unknown {what} fields: {unknown}")
    missing = [name for name in fields if name not in payload]
    if missing:
        raise ValueError(f"Missing {what} fields: {missing}")


def _one_of(value: Any, allowed: tuple[str, ...], what: str) -> str:
    if value not in allowed:
        raise ValueError(f"{what} must be one of {allowed}, got {value!r}")
    return str(value)


@dataclass(frozen=True)
class Scenario:
    """One surface form of a two-option judgment call.

    ``paraphrase_id`` and ``order`` vary inside a ``scenario_id``. Splits and
    the assigned lean are properties of ``scenario_id`` alone.
    """

    scenario_id: str
    family: Family
    question: str
    option_a: str
    option_b: str
    paraphrase_id: str
    order: Order

    def __post_init__(self) -> None:
        if not str(self.scenario_id).strip():
            raise ValueError("scenario_id is empty")
        _one_of(self.family, FAMILIES, "family")
        _one_of(self.order, ORDERS, "order")
        if not str(self.paraphrase_id).strip():
            raise ValueError("paraphrase_id is empty")
        if not str(self.question).strip():
            raise ValueError(f"{self.scenario_id}: question is empty")
        if not str(self.option_a).strip() or not str(self.option_b).strip():
            raise ValueError(f"{self.scenario_id}: options must be non-empty")
        if self.option_a.strip() == self.option_b.strip():
            raise ValueError(f"{self.scenario_id}: option_a and option_b are the same")


@dataclass(frozen=True)
class LeanCondition:
    """One lean installation, including the exact system prompt that carries it.

    ``strength`` and ``lean`` are None for ``none`` and ``trait``.
    ``neutral_matched`` keeps the strength it is matched to and has ``lean`` None.
    ``assigned`` and ``conceal`` have both a strength and a canonical lean.
    """

    kind: Kind
    strength: Strength | None
    lean: Side | None
    system_prompt: str

    def __post_init__(self) -> None:
        _one_of(self.kind, KINDS, "kind")
        if self.strength is not None:
            _one_of(self.strength, STRENGTHS, "strength")
        if self.lean is not None:
            _one_of(self.lean, SIDES, "lean")
        if not isinstance(self.system_prompt, str):
            raise ValueError("system_prompt must be a string")
        if self.kind in ("none", "trait"):
            if self.strength is not None or self.lean is not None:
                raise ValueError(f"{self.kind} conditions have no strength and no lean")
            if self.system_prompt != "":
                raise ValueError(f"{self.kind} is the no-prompt condition")
        elif self.kind == "neutral_matched":
            if self.strength is None or self.lean is not None:
                raise ValueError("neutral_matched needs a strength and lean=None")
            if not self.system_prompt:
                raise ValueError("neutral_matched prompt is empty")
        elif self.kind in ("assigned", "conceal"):
            if self.strength is None or self.lean is None:
                raise ValueError(f"{self.kind} needs a strength and a lean")
            if not self.system_prompt:
                raise ValueError(f"{self.kind} prompt is empty")
        else:
            raise ValueError(self.kind)


@dataclass
class Record:
    """One model call: a scenario surface under one condition, plus its labels.

    ``output_text`` is empty and the label fields are None until generation.
    ``recommended`` is the canonical option in the greedy answer.
    ``voiced`` is whether a judge says the reply advocates ``condition.lean``.
    """

    item_id: str
    scenario: Scenario
    condition: LeanCondition
    output_text: str
    recommended: Side | None
    voiced: bool | None
    activations_path: str | None
    split: SplitName

    def __post_init__(self) -> None:
        if not str(self.item_id).strip():
            raise ValueError("item_id is empty")
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if not isinstance(self.condition, LeanCondition):
            raise TypeError("condition must be a LeanCondition")
        if not isinstance(self.output_text, str):
            raise ValueError("output_text must be a string")
        if self.recommended is not None:
            _one_of(self.recommended, SIDES, "recommended")
        if self.voiced is not None and not isinstance(self.voiced, bool):
            raise ValueError("voiced must be a bool or None")
        if self.activations_path is not None and not isinstance(self.activations_path, str):
            raise ValueError("activations_path must be a string or None")
        _one_of(self.split, SPLITS, "split")


def scenario_from_dict(payload: dict[str, Any]) -> Scenario:
    _reject_unknown(payload, _SCENARIO_FIELDS, "scenario")
    return Scenario(
        scenario_id=str(payload["scenario_id"]),
        family=payload["family"],
        question=str(payload["question"]),
        option_a=str(payload["option_a"]),
        option_b=str(payload["option_b"]),
        paraphrase_id=str(payload["paraphrase_id"]),
        order=payload["order"],
    )


def condition_from_dict(payload: dict[str, Any]) -> LeanCondition:
    _reject_unknown(payload, _CONDITION_FIELDS, "condition")
    return LeanCondition(
        kind=payload["kind"],
        strength=payload["strength"],
        lean=payload["lean"],
        system_prompt=str(payload["system_prompt"]),
    )


def record_from_dict(payload: dict[str, Any]) -> Record:
    _reject_unknown(payload, _RECORD_FIELDS, "record")
    scenario_payload = payload["scenario"]
    condition_payload = payload["condition"]
    if not isinstance(scenario_payload, dict) or not isinstance(condition_payload, dict):
        raise ValueError("record scenario and condition must be objects")
    return Record(
        item_id=str(payload["item_id"]),
        scenario=scenario_from_dict(scenario_payload),
        condition=condition_from_dict(condition_payload),
        output_text="" if payload["output_text"] is None else str(payload["output_text"]),
        recommended=payload["recommended"],
        voiced=payload["voiced"],
        activations_path=payload["activations_path"],
        split=payload["split"],
    )


def record_to_dict(record: Record) -> dict[str, Any]:
    return asdict(record)


def write_jsonl(path: Path | str, records: Iterable[Record]) -> int:
    """Write records as UTF-8 JSONL. Returns the number of rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_dict(record), ensure_ascii=False))
            handle.write("\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> list[Record]:
    """Load a records JSONL file. Blank lines are skipped."""
    return list(iter_jsonl(path))


def iter_jsonl(path: Path | str) -> Iterator[Record]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            try:
                yield record_from_dict(payload)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
