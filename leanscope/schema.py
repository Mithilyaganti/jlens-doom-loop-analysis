"""Shared LeanScope records.

The three paper-2 agents share this file. Field names are the contract:
do not rename them. JSONL helpers round-trip the nested dataclasses.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

FAMILIES = ("analysis", "product", "advice")
ORDERS = ("AB", "BA")
KINDS = ("none", "assigned", "neutral_matched", "trait", "conceal")
STRENGTHS = ("explicit", "soft", "indirect", "conflicting")
LEANS = ("A", "B")
SPLITS = ("train", "dev", "test")
RECOMMENDED = ("A", "B")


def _check(name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


def _optional(name: str, value, allowed: tuple[str, ...]):
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        value = str(value)
    return _check(name, value, allowed)


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
        self.family = _check("family", str(self.family), FAMILIES)
        self.question = str(self.question)
        self.option_a = str(self.option_a)
        self.option_b = str(self.option_b)
        self.paraphrase_id = str(self.paraphrase_id)
        self.order = _check("order", str(self.order), ORDERS)


@dataclass
class LeanCondition:
    kind: str
    strength: str | None
    lean: str | None
    system_prompt: str

    def __post_init__(self) -> None:
        self.kind = _check("kind", str(self.kind), KINDS)
        self.strength = _optional("strength", self.strength, STRENGTHS)
        self.lean = _optional("lean", self.lean, LEANS)
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
            self.scenario = Scenario(**self.scenario)
        if not isinstance(self.condition, LeanCondition):
            self.condition = LeanCondition(**self.condition)
        self.output_text = "" if self.output_text is None else str(self.output_text)
        self.recommended = _optional("recommended", self.recommended, RECOMMENDED)
        if self.voiced is not None and not isinstance(self.voiced, bool):
            raise ValueError(f"voiced must be bool or None, got {self.voiced!r}")
        if self.activations_path is not None:
            self.activations_path = str(self.activations_path)
        self.split = _check("split", str(self.split), SPLITS)


def record_to_dict(record: Record) -> dict:
    return asdict(record)


def record_from_dict(payload: dict) -> Record:
    return Record(
        item_id=payload["item_id"],
        scenario=payload["scenario"],
        condition=payload["condition"],
        output_text=payload.get("output_text", ""),
        recommended=payload.get("recommended"),
        voiced=payload.get("voiced"),
        activations_path=payload.get("activations_path"),
        split=payload["split"],
    )


def write_records(path: Path | str, records: Iterable[Record]) -> None:
    """Write one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_dict(record), ensure_ascii=False))
            handle.write("\n")


def read_records(path: Path | str) -> list[Record]:
    """Read a JSONL file of records. Blank lines are skipped."""
    return list(_iter_records(Path(path).open("r", encoding="utf-8")))


def _iter_records(handle: TextIO) -> Iterator[Record]:
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield record_from_dict(json.loads(line))
