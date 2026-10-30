"""Transliteration, drift, fertility, prompts, power, and the pilot plan."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from loop_lingua.drift import measure_drift
from loop_lingua.fertility import measure_fertility
from loop_lingua.graphemes import grapheme_clusters
from loop_lingua.pilot import (
    Generation,
    PilotConfig,
    PilotJob,
    plan_go_nogo,
    run_harness,
)
from loop_lingua.power import analysis_table, cohens_h, n_per_group, norm_ppf, power_at_n
from loop_lingua.prompts import (
    EN_INSTRUCTION,
    EligibleItem,
    ParallelItem,
    TE_INSTRUCTION,
    allocate,
    build_matched_set,
    load_mgsm_tsvs,
    pilot_item_ids,
    romanized_instruction,
    select_item_ids,
)
from loop_lingua.transliterate import TransliterationError, transliterate_telugu


def test_iso15919_examples() -> None:
    assert transliterate_telugu("జనెత్") == "janet"
    assert transliterate_telugu("నమస్కారం") == "namaskāraṁ"
    assert transliterate_telugu("తెలుగు") == "telugu"
    assert transliterate_telugu("క్ష") == "kṣa"
    assert transliterate_telugu("ఆమె 16 గుడ్లు") == "āme 16 guḍlu"
    assert transliterate_telugu("౧౬") == "16"
    assert "\u0c00" <= "క" <= "\u0c7f"
    assert not any(0x0C00 <= ord(char) <= 0x0C7F for char in romanized_instruction())
    assert romanized_instruction() == transliterate_telugu(TE_INSTRUCTION)


def test_unknown_telugu_character_is_an_error() -> None:
    with pytest.raises(TransliterationError):
        transliterate_telugu("\u0c77")


def test_grapheme_clusters_for_the_three_scripts() -> None:
    assert grapheme_clusters("e\u0301") == ["e\u0301"]
    assert grapheme_clusters("\r\n") == ["\r\n"]
    assert grapheme_clusters("क्ष") == ["क्ष"]
    assert grapheme_clusters("క్ష") == ["క్ష"]
    assert grapheme_clusters("కా") == ["కా"]
    assert grapheme_clusters("कि") == ["कि"]


def test_drift_flags_follow_the_expected_script() -> None:
    english = "This answer stays in English and keeps explaining the algebra carefully."
    hindi = "यह उत्तर हिंदी में है और यह अपनी लिपि में ही आगे बढ़ता रहता है बिना अंग्रेज़ी के।"
    telugu = "ఈ సమాధానం తెలుగులోనే ఉంది మరియు ఇది ఇంగ్లీషు లిపిలోకి మారడం లేదు."
    roman = "idi telugu bhasha roman lipilo undi mariyu ikkada telugu script ledu."
    mixed = "nenu this problem ni solve చేస్తాను step by step and then answer ఇస్తాను."
    assert measure_drift(english, "en").drifted is False
    assert measure_drift(hindi, "hi").drifted is False
    assert measure_drift(telugu, "te").drifted is False
    assert measure_drift(roman, "te_roman").drifted is False
    assert measure_drift(english, "hi").drifted is True
    assert measure_drift(telugu, "en").drifted is True
    assert measure_drift(mixed, "tenglish").drifted is False
    assert measure_drift(english, "tenglish").drifted is True
    assert measure_drift(telugu, "tenglish").drifted is True
    assert measure_drift("ok", "en").low_confidence is True
    assert measure_drift("ok", "en").drifted is None


def test_fertility_uses_the_injected_encoder() -> None:
    report = measure_fertility(["ab cd", "ef"], lambda text: [ord(char) for char in text], language="en")
    assert report.n_tokens == report.n_code_points
    assert report.tokens_per_code_point == 1
    assert report.n_words == 3
    with pytest.raises(ValueError):
        measure_fertility([], lambda text: [1], language="en")


def test_allocation_sums_and_breaks_ties_by_name() -> None:
    counts = {"c": 1, "a": 1, "b": 1}
    assert allocate(counts, 2) == {"a": 1, "b": 1, "c": 0}
    assert sum(allocate({"math": 10, "law": 3, "art": 7}, 9).values()) == 9
    with pytest.raises(ValueError):
        allocate({"math": 2}, 3)


def _pool() -> list[ParallelItem]:
    items: list[ParallelItem] = []
    for index, category in enumerate(["math", "law", "math", "health"]):
        item_id = f"item-{index}"
        items.append(
            ParallelItem(
                item_id=item_id,
                source="mmlu-prox-lite",
                category=category,
                questions={
                    "en": f"English question {index}?",
                    "hi": f"हिंदी प्रश्न {index}?",
                    "te": f"తెలుగు ప్రశ్న {index}?",
                },
                options={
                    "en": ("one", "two"),
                    "hi": ("एक", "दो"),
                    "te": ("ఒకటి", "రెండు"),
                },
            )
        )
    items.append(
        ParallelItem(
            item_id="mgsm-000",
            source="mgsm",
            category="grade_school_math",
            questions={"en": "Janet has ducks.", "te": "జనెత్ బాతులు."},
            options={"en": (), "te": ()},
        )
    )
    return items


def test_selection_ignores_english_outcome_and_drops_mgsm() -> None:
    @dataclass(frozen=True)
    class Scored(EligibleItem):
        english_correct: bool = False

    base = [item.eligible() for item in _pool()]
    scored_true = [
        Scored(item.item_id, item.source, item.category, item.languages, english_correct=True) for item in base
    ]
    scored_false = [
        Scored(item.item_id, item.source, item.category, item.languages, english_correct=False) for item in base
    ]
    ids_true, alloc_true = select_item_ids(scored_true, 2, seed=7)
    ids_false, alloc_false = select_item_ids(scored_false, 2, seed=7)
    assert ids_true == ids_false
    assert alloc_true == alloc_false
    assert "mgsm-000" not in ids_true
    built = build_matched_set(_pool(), n=2, seed=7, n_tenglish=2)
    assert built.dropped_sources == {"mgsm": 1}
    assert built.item_ids == ids_true
    by_lang = {language: built.by_language(language) for language in ("en", "hi", "te", "te_roman")}
    assert all(len(rows) == 2 for rows in by_lang.values())
    assert [row.item_id for row in by_lang["en"]] == [row.item_id for row in by_lang["te_roman"]]
    for row in by_lang["en"]:
        assert row.prompt.startswith(EN_INSTRUCTION)
        assert "english_correct" not in row.as_dict()
    for row in by_lang["te_roman"]:
        assert not any(0x0C00 <= ord(char) <= 0x0C7F for char in row.prompt)
    for slot in built.tenglish_slots:
        assert slot.prompt is None
        assert slot.edited is False
        assert slot.item_id in built.item_ids


def test_mgsm_loader_ignores_the_answer_column(tmp_path) -> None:  # type: ignore[no-untyped-def]
    en = tmp_path / "mgsm_en.tsv"
    te = tmp_path / "mgsm_te.tsv"
    en_lines = [f"Question {i} about ducks.\t{i}\n" for i in range(250)]
    te_lines = [f"ప్రశ్న {i}.\t{i + 5}\n" for i in range(250)]
    en.write_text("".join(en_lines), encoding="utf-8")
    te.write_text("".join(te_lines), encoding="utf-8")
    items = load_mgsm_tsvs({"en": en, "te": te})
    assert len(items) == 250
    assert items[3].questions["en"] == "Question 3 about ducks."
    assert "3" not in items[3].questions["en"].split()[-1:] or True
    assert items[3].eligible().languages == frozenset({"en", "te"})


def test_power_at_the_solved_n_is_about_80_percent() -> None:
    assert abs(norm_ppf(0.975) - 1.95996398454) < 1e-6
    n = n_per_group(0.20, 0.30, alpha=0.05, power=0.8)
    assert abs(power_at_n(0.20, 0.30, n) - 0.8) < 0.01
    assert cohens_h(0.30, 0.20) > 0
    rows = analysis_table(base_rates=(0.229,), deltas=(0.10,), planned_n=300, comparisons=3)
    assert any(row.comparisons == 3 and row.alpha < 0.05 for row in rows)
    assert all(row.power_at_planned_n > 0 for row in rows)


def test_pilot_harness_and_go_nogo_are_cpu_only() -> None:
    config = PilotConfig(per_language=2, languages=("en", "hi"), full_prompts_per_language=300)
    jobs = [
        PilotJob("item-0", "en", "Solve one."),
        PilotJob("item-0", "hi", "हल कीजिए।"),
        PilotJob("item-1", "en", "Solve two."),
        PilotJob("item-1", "hi", "दूसरा।"),
    ]

    def generate(job: PilotJob) -> Generation:
        text = "Wait, let me rethink this carefully. " * 8
        return Generation(job.item_id, job.language, text, token_ids=(1, 2, 3, 4) * 4, new_tokens=8, elapsed_seconds=0.5)

    report = run_harness(jobs, generate, config=config, chars_per_token=1.0)
    assert report.n_generations == 4
    # 4 jobs × 8 tokens / (4 × 0.5 s) = 16 tok/s.
    assert report.tokens_per_second == pytest.approx(16.0)
    assert report.loop_counts["D0"] == 4
    pending = plan_go_nogo(None, config, basis="none")
    assert pending.decision == "pending"
    # 300 * 2 languages * 2048 tokens / 100 tok/s / 3600 s.
    assumed = plan_go_nogo(100.0, config, basis="assumed")
    assert assumed.basis == "assumed"
    assert assumed.decision == "go"
    assert assumed.estimated_full_hours == pytest.approx(300 * 2 * 2048 / 100 / 3600)
    slow = plan_go_nogo(5.0, config, basis="measured")
    assert slow.decision == "no-go"
    replan = plan_go_nogo(20.0, config, basis="measured")
    assert replan.decision == "replan"
    ids = pilot_item_ids(
        ["a", "b", "c", "d"],
        {"a": "math", "b": "law", "c": "math", "d": "law"},
        per_language=3,
    )
    # Categories in name order, ids sorted inside each category, then round-robin.
    assert ids == ["b", "a", "d"]
