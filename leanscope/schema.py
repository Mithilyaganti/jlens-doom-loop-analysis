"""Shared LeanScope records.

The three paper-2 agents share this file. Field names are the contract:
extra keys in a JSON object are ignored on load, and a missing field is an
error, so a writer that uses exactly these fields round-trips.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TextIO

FAMILIES = ("analysis", "product", "advice")
ORDERS = ("AB", "BA")
KINDS = ("none", "assigned", "neutral_matched", "trait", "conceal")
STRENGTHS = ("explicit", "soft", "indirect", "conflicting")
LEANS = ("A", "B")
SPLITS = ("train", "dev", "test")
RECOMMENDED = ("A", "B")


def _require(value: Any, allowed: Sequence[str], field: str) -> str:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {list(allowed)}, got {value!r}")
    return str(value)


def _optional(value: Any, allowed: Sequence[str], field: str) -> str | None:
    if value is None:
        return None
    return _require(value, allowed, field)


@dataclass
class Scenario:
    scenario_id: str
    family: str
    question: str
    option_a: str
    option_b: str
    paraphrase_id: str
    order: str

    def __post_init__(self) -> None:
        self.scenario_id = str(self.scenario_id)
        self.family = _require(self.family, FAMILIES, "family")
        self.question = str(self.question)
        self.option_a = str(self.option_a)
        self.option_b = str(self.option_b)
        self.paraphrase_id = str(self.paraphrase_id)
        self.order = _require(self.order, ORDERS, "order")


@dataclass
class LeanCondition:
    kind: str
    strength: str | None
    lean: str | None
    system_prompt: str

    def __post_init__(self) -> None:
        self.kind = _require(self.kind, KINDS, "kind")
        self.strength = _optional(self.strength, STRENGTHS, "strength")
        self.lean = _optional(self.lean, LEANS, "lean")
        self.system_prompt = "" if self.system_prompt is None else str(self.system_prompt)


@dataclass
class Record:
    item_id: str
    scenario: Scenario
    condition: LeanCondition
    output_text: str
    recommended: str | None
    voiced: bool | None
    activations_path: str | None
    split: str

    def __post_init__(self) -> None:
        self.item_id = str(self.item_id)
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if not isinstance(self.condition, LeanCondition):
            raise TypeError("condition must be a LeanCondition")
        self.output_text = "" if self.output_text is None else str(self.output_text)
        self.recommended = _optional(self.recommended, RECOMMENDED, "recommended")
        if self.voiced is not None and not isinstance(self.voiced, bool):
            raise TypeError("voiced must be a bool or None")
        if self.activations_path is not None:
            self.activations_path = str(self.activations_path)
        self.split = _require(self.split, SPLITS, "split")


def scenario_from_dict(data: dict) -> Scenario:
    return Scenario(
        scenario_id=data["scenario_id"],
        family=data["family"],
        question=data["question"],
        option_a=data["option_a"],
        option_b=data["option_b"],
        paraphrase_id=data["paraphrase_id"],
        order=data["order"],
    )


def condition_from_dict(data: dict) -> LeanCondition:
    return LeanCondition(
        kind=data["kind"],
        strength=data.get("strength"),
        lean=data.get("lean"),
        system_prompt=data.get("system_prompt", ""),
    )


def record_to_dict(record: Record) -> dict:
    return asdict(record)


def record_from_dict(data: dict) -> Record:
    return Record(
        item_id=data["item_id"],
        scenario=scenario_from_dict(data["scenario"]),
        condition=condition_from_dict(data["condition"]),
        output_text=data.get("output_text", ""),
        recommended=data.get("recommended"),
        voiced=data.get("voiced"),
        activations_path=data.get("activations_path"),
        split=data["split"],
    )


def write_jsonl(path: str | Path, records: Iterable[Record]) -> None:
    """Write records as UTF-8 JSONL. One Record per line."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        _write_records(handle, records)


def _write_records(handle: TextIO, records: Iterable[Record]) -> None:
    for record in records:
        handle.write(json.dumps(record_to_dict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Record]:
    """Read a JSONL file of Records. Blank lines are skipped."""
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            try:
                yield record_from_dict(payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
