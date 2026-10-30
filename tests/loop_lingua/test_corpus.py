"""Locks the committed prompt corpus to the sampler."""

from __future__ import annotations

import json
from pathlib import Path

from loop_lingua.layout import CORPUS_DIR
from loop_lingua.prompts import (
    EN_INSTRUCTION,
    HI_INSTRUCTION,
    TE_INSTRUCTION,
    EligibleItem,
    pilot_item_ids,
    romanized_instruction,
    select_item_ids,
)


def _rows(name: str) -> list[dict[str, object]]:
    path = CORPUS_DIR / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_matched_set_is_the_stratified_sample() -> None:
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    pool = _rows("pool_index.jsonl")
    items = [
        EligibleItem(
            str(row["item_id"]),
            str(row["source"]),
            str(row["category"]),
            frozenset(row["languages"]),  # type: ignore[arg-type]
        )
        for row in pool
    ]
    chosen, allocation = select_item_ids(items, 300, seed=int(manifest["seed"]))
    assert chosen == manifest["item_ids"]
    assert allocation == manifest["category_allocation"]
    assert sum(allocation.values()) == 300
    assert manifest["dropped_sources"]["mgsm"] == 250
    assert manifest["selection_fields"] == ["item_id", "category", "languages"]
    assert "answer" not in manifest["selection_fields"]


def test_prompts_are_parallel_and_in_script() -> None:
    rows = _rows("prompts_matched.jsonl")
    by_lang: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_lang.setdefault(str(row["language"]), []).append(row)
    assert {len(group) for group in by_lang.values()} == {300}
    ids = [{str(row["item_id"]) for row in group} for group in by_lang.values()]
    assert ids[0] == ids[1] == ids[2] == ids[3]
    prefixes = {
        "en": EN_INSTRUCTION,
        "hi": HI_INSTRUCTION,
        "te": TE_INSTRUCTION,
        "te_roman": romanized_instruction(),
    }
    telugu_block = range(0x0C00, 0x0C80)
    for language, prefix in prefixes.items():
        for row in by_lang[language]:
            prompt = str(row["prompt"])
            assert prompt.startswith(prefix)
            assert "cot_content" not in row
            assert "answer" not in row
            if language == "te_roman":
                assert not any(ord(char) in telugu_block for char in prompt)
            if language == "te":
                assert any(ord(char) in telugu_block for char in prompt)
            if language == "hi":
                assert any(0x0900 <= ord(char) <= 0x097F for char in prompt)


def test_tenglish_slots_are_empty_and_pilot_is_96() -> None:
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    slots = json.loads((CORPUS_DIR / "tenglish_slots.json").read_text(encoding="utf-8"))
    assert slots["status"] == "placeholder"
    assert slots["hand_edited"] == 0
    assert len(slots["slots"]) == 100
    assert all(slot["prompt"] is None and slot["edited"] is False for slot in slots["slots"])
    categories = {}
    for row in _rows("prompts_matched.jsonl"):
        if row["language"] == "en":
            categories[str(row["item_id"])] = str(row["category"])
    assert pilot_item_ids(manifest["item_ids"], categories) == manifest["pilot_ids"]
    pilot = _rows("pilot_96.jsonl")
    assert len(pilot) == 96
    assert len({row["item_id"] for row in pilot}) == 24
    assert {row["language"] for row in pilot} == {"en", "hi", "te", "te_roman"}


def test_gold_is_sidecar_only() -> None:
    gold = _rows("gold.jsonl")
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert {str(row["item_id"]) for row in gold} == set(manifest["item_ids"])
    assert all(str(row["answer"]).isalpha() for row in gold)


def test_recorded_offset_bug_counts() -> None:
    payload = json.loads((CORPUS_DIR / "offset_bug_mgsm.json").read_text(encoding="utf-8"))
    assert payload["languages"]["en"]["n_length_mismatches"] == 0
    assert payload["languages"]["te"]["n_length_mismatches"] == 149
    assert payload["languages"]["en"]["n_texts"] == 250
    assert payload["languages"]["te"]["n_texts"] == 250
    assert "tokenizer_json" not in payload
    notices = (CORPUS_DIR / "DATA_NOTICES.md").read_text(encoding="utf-8")
    assert "MIT" in notices and "CC-BY-4.0" in notices


def test_live_tokenizer_still_mismatches_149_telugu() -> None:
    import pytest

    from loop_lingua.reproduce_bug import run, tokenizer_json_path

    if tokenizer_json_path() is None:
        pytest.skip("Qwen tokenizer.json is not cached")
    payload = run()
    languages = payload["languages"]
    assert isinstance(languages, dict)
    assert languages["en"]["n_length_mismatches"] == 0
    assert languages["te"]["n_length_mismatches"] == 149
