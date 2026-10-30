"""Matched prompt sets for EN, HI, TE, and romanized Telugu.

Selection sees only an item id, a category, and which language strings
exist. It does not see the question text, the gold letter, or any score
on the English item. Romanized Telugu is the Telugu prompt passed through
ISO 15919. The Tenglish arm is a placeholder list for hand edits.

The matched 300 are drawn from MMLU-ProX-Lite test items that exist in
English, Hindi, and Telugu. MGSM is loaded and license-checked, then
dropped by that language requirement because it has no Hindi. The 250
English–Telugu pairs are still written out as a supplement.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from loop_lingua.licenses import require_known
from loop_lingua.transliterate import transliterate_telugu

MATCHED_LANGUAGES: tuple[str, ...] = ("en", "hi", "te", "te_roman")
SOURCE_LANGUAGES: tuple[str, ...] = ("en", "hi", "te")
DEFAULT_N = 300
DEFAULT_SEED = 20261030
TENGISH_SLOTS = 100
PILOT_PER_LANGUAGE = 24
OPTION_LETTERS = "ABCDEFGHIJ"

EN_INSTRUCTION = (
    "Solve the following multiple-choice problem. "
    "Think step by step, then give the final answer."
)
HI_INSTRUCTION = (
    "निम्नलिखित बहुविकल्पीय प्रश्न को हल कीजिए। "
    "चरण-दर-चरण सोचिए और अंतिम उत्तर दीजिए।"
)
TE_INSTRUCTION = (
    "కింది బహుళైచ్ఛిక ప్రశ్నను సాధించండి. "
    "దశలవారీగా ఆలోచించి, చివరి సమాధానం ఇవ్వండి."
)


def romanized_instruction() -> str:
    """Latin instruction, derived from the Telugu line so the two cannot drift."""
    return transliterate_telugu(TE_INSTRUCTION)


@dataclass(frozen=True)
class EligibleItem:
    """The only fields the sampler is allowed to read."""

    item_id: str
    source: str
    category: str
    languages: frozenset[str]

    def has_languages(self, required: Iterable[str]) -> bool:
        return all(lang in self.languages for lang in required)


@dataclass(frozen=True)
class ParallelItem:
    """Full parallel record. Gold stays out of this type on purpose."""

    item_id: str
    source: str
    category: str
    questions: dict[str, str]
    options: dict[str, tuple[str, ...]]

    def eligible(self) -> EligibleItem:
        return EligibleItem(
            item_id=self.item_id,
            source=self.source,
            category=self.category,
            languages=frozenset(lang for lang, text in self.questions.items() if text.strip()),
        )


@dataclass(frozen=True)
class PromptRecord:
    item_id: str
    source: str
    category: str
    language: str
    license_id: str
    question: str
    options: tuple[str, ...]
    prompt: str

    def as_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "source": self.source,
            "category": self.category,
            "language": self.language,
            "license_id": self.license_id,
            "question": self.question,
            "options": list(self.options),
            "prompt": self.prompt,
        }


@dataclass
class TenglishSlot:
    slot: int
    item_id: str
    category: str
    prompt: str | None = None
    edited: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "item_id": self.item_id,
            "category": self.category,
            "prompt": self.prompt,
            "edited": self.edited,
        }


@dataclass
class BuiltSet:
    records: list[PromptRecord]
    item_ids: list[str]
    category_allocation: dict[str, int]
    tenglish_slots: list[TenglishSlot]
    dropped_sources: dict[str, int] = field(default_factory=dict)

    def by_language(self, language: str) -> list[PromptRecord]:
        return [row for row in self.records if row.language == language]


def allocate(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation. Ties break by category name."""
    if n < 0:
        raise ValueError("n must be non-negative")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("no items to allocate")
    if n > total:
        raise ValueError(f"requested {n} items but only {total} are eligible")
    if n == 0:
        return {key: 0 for key in counts}
    raw = {key: n * counts[key] / total for key in counts}
    base = {key: int(raw[key]) for key in counts}
    left = n - sum(base.values())
    order = sorted(counts, key=lambda key: (-(raw[key] - int(raw[key])), key))
    for key in order:
        if left == 0:
            break
        if base[key] >= counts[key]:
            continue
        base[key] += 1
        left -= 1
    if left != 0:
        raise RuntimeError("largest-remainder allocation could not place every item")
    if sum(base.values()) != n:
        raise RuntimeError("allocation does not sum to n")
    return base


def select_item_ids(
    items: Sequence[EligibleItem],
    n: int,
    *,
    seed: int = DEFAULT_SEED,
    required: Sequence[str] = SOURCE_LANGUAGES,
) -> tuple[list[str], dict[str, int]]:
    """Stratified ids. Reads only id, category, and the language set."""
    seen: set[str] = set()
    eligible: list[EligibleItem] = []
    for item in items:
        if not item.has_languages(required):
            continue
        if item.item_id in seen:
            raise ValueError(f"duplicate item id {item.item_id}")
        seen.add(item.item_id)
        eligible.append(item)
    by_category: dict[str, list[str]] = defaultdict(list)
    for item in eligible:
        by_category[item.category].append(item.item_id)
    for category in by_category:
        by_category[category] = sorted(set(by_category[category]))
    allocation = allocate({key: len(value) for key, value in by_category.items()}, n)
    rng = random.Random(seed)
    chosen: list[str] = []
    for category in sorted(allocation):
        ids = list(by_category[category])
        rng.shuffle(ids)
        take = allocation[category]
        chosen.extend(ids[:take])
    if len(chosen) != n:
        raise RuntimeError(f"selected {len(chosen)} ids, expected {n}")
    return sorted(chosen), allocation


def _clean_options(options: Sequence[str | None]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for option in options:
        if option is None:
            continue
        text = str(option).strip()
        if text == "":
            continue
        cleaned.append(text)
    if len(cleaned) > len(OPTION_LETTERS):
        raise ValueError("more than 10 options")
    return tuple(cleaned)


def render_prompt(instruction: str, question: str, options: Sequence[str]) -> str:
    lines = [instruction.strip(), "", question.strip(), ""]
    for index, option in enumerate(options):
        lines.append(f"{OPTION_LETTERS[index]}. {option}")
    return "\n".join(lines).rstrip() + "\n"


def render_record(item: ParallelItem, language: str) -> PromptRecord:
    if language == "te_roman":
        telugu = render_record(item, "te")
        prompt = transliterate_telugu(telugu.prompt)
        return PromptRecord(
            item_id=item.item_id,
            source=item.source,
            category=item.category,
            language="te_roman",
            license_id=item.source,
            question=transliterate_telugu(telugu.question),
            options=tuple(transliterate_telugu(option) for option in telugu.options),
            prompt=prompt,
        )
    if language not in item.questions:
        raise KeyError(f"{item.item_id} has no {language} text")
    instruction = {"en": EN_INSTRUCTION, "hi": HI_INSTRUCTION, "te": TE_INSTRUCTION}[language]
    question = item.questions[language].strip()
    options = item.options.get(language, ())
    require_known(item.source)
    return PromptRecord(
        item_id=item.item_id,
        source=item.source,
        category=item.category,
        language=language,
        license_id=item.source,
        question=question,
        options=options,
        prompt=render_prompt(instruction, question, options),
    )


def build_matched_set(
    items: Sequence[ParallelItem],
    *,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    n_tenglish: int = TENGISH_SLOTS,
) -> BuiltSet:
    """Build the matched prompts. Gold and outcome fields are not accepted."""
    eligible = [item.eligible() for item in items]
    present = {item.item_id: item for item in items}
    if len(present) != len(items):
        raise ValueError("duplicate parallel item ids")
    dropped: dict[str, int] = defaultdict(int)
    for item in eligible:
        if not item.has_languages(SOURCE_LANGUAGES):
            dropped[item.source] += 1
    chosen, allocation = select_item_ids(eligible, n, seed=seed)
    records: list[PromptRecord] = []
    for item_id in chosen:
        item = present[item_id]
        for language in MATCHED_LANGUAGES:
            records.append(render_record(item, language))
    slots = tenglish_placeholder_slots(chosen, present, n_slots=n_tenglish)
    return BuiltSet(
        records=records,
        item_ids=chosen,
        category_allocation=allocation,
        tenglish_slots=slots,
        dropped_sources=dict(dropped),
    )


def _round_robin(item_ids: Sequence[str], categories: Mapping[str, str], n: int) -> list[str]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for item_id in item_ids:
        by_category[categories[item_id]].append(item_id)
    for category in by_category:
        by_category[category] = sorted(by_category[category])
    picked: list[str] = []
    names = sorted(by_category)
    cursor = {name: 0 for name in names}
    while len(picked) < n:
        progressed = False
        for name in names:
            index = cursor[name]
            bucket = by_category[name]
            if index < len(bucket):
                picked.append(bucket[index])
                cursor[name] = index + 1
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
    if len(picked) != n:
        raise ValueError(f"could only pick {len(picked)} of {n} items")
    return picked


def tenglish_placeholder_slots(
    item_ids: Sequence[str],
    items_by_id: Mapping[str, ParallelItem],
    *,
    n_slots: int = TENGISH_SLOTS,
) -> list[TenglishSlot]:
    """Empty slots. Prompts stay null until a person edits them."""
    categories = {item_id: items_by_id[item_id].category for item_id in item_ids}
    queue = _round_robin(item_ids, categories, n_slots)
    return [
        TenglishSlot(slot=index, item_id=item_id, category=categories[item_id], prompt=None, edited=False)
        for index, item_id in enumerate(queue)
    ]


def pilot_item_ids(
    item_ids: Sequence[str],
    categories: Mapping[str, str],
    *,
    per_language: int = PILOT_PER_LANGUAGE,
) -> list[str]:
    """Stratified pilot ids. Same ids are used in every language."""
    return _round_robin(item_ids, categories, per_language)


def load_mmlu_prox_lite_parquets(paths: Mapping[str, Path]) -> tuple[list[ParallelItem], dict[str, str]]:
    """Load EN/HI/TE parquet shards. Returns items and a gold map.

    The gold map is for a later scoring script. ``build_matched_set`` never
    receives it.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("reading MMLU-ProX-Lite parquet needs pyarrow") from exc

    tables: dict[str, dict[int, dict[str, object]]] = {}
    for language, path in paths.items():
        table = pq.read_table(path)
        rows: dict[int, dict[str, object]] = {}
        question_ids = table.column("question_id").to_pylist()
        categories = table.column("category").to_pylist()
        questions = table.column("question").to_pylist()
        answers = table.column("answer").to_pylist()
        option_columns = [table.column(f"option_{i}").to_pylist() for i in range(10)]
        for index, question_id in enumerate(question_ids):
            options = _clean_options([option_columns[opt][index] for opt in range(10)])
            rows[int(question_id)] = {
                "category": str(categories[index]),
                "question": str(questions[index]),
                "answer": str(answers[index]),
                "options": options,
            }
        tables[language] = rows

    languages = list(paths)
    shared = set(tables[languages[0]])
    for language in languages[1:]:
        shared &= set(tables[language])
    items: list[ParallelItem] = []
    gold: dict[str, str] = {}
    for question_id in sorted(shared):
        categories = {tables[language][question_id]["category"] for language in languages}
        if len(categories) != 1:
            raise ValueError(f"category mismatch for question {question_id}")
        answers = {tables[language][question_id]["answer"] for language in languages}
        if len(answers) != 1:
            raise ValueError(f"gold mismatch for question {question_id}")
        item_id = f"mmlu-prox-lite-{question_id}"
        items.append(
            ParallelItem(
                item_id=item_id,
                source="mmlu-prox-lite",
                category=str(tables[languages[0]][question_id]["category"]),
                questions={language: str(tables[language][question_id]["question"]) for language in languages},
                options={
                    language: tuple(tables[language][question_id]["options"])  # type: ignore[arg-type]
                    for language in languages
                },
            )
        )
        gold[item_id] = str(tables[languages[0]][question_id]["answer"])
    return items, gold


def load_mgsm_tsvs(paths: Mapping[str, Path]) -> list[ParallelItem]:
    """Load MGSM question TSVs. No gold is returned; the answer column is ignored."""
    columns: dict[str, list[str]] = {}
    for language, path in paths.items():
        questions: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            question, _sep, _answer = line.partition("\t")
            questions.append(question)
        columns[language] = questions
    lengths = {len(rows) for rows in columns.values()}
    if lengths != {250}:
        raise ValueError(f"MGSM files are not the expected 250 rows: { {k: len(v) for k, v in columns.items()} }")
    items: list[ParallelItem] = []
    n = 250
    for index in range(n):
        items.append(
            ParallelItem(
                item_id=f"mgsm-{index:03d}",
                source="mgsm",
                category="grade_school_math",
                questions={language: columns[language][index] for language in columns},
                options={language: () for language in columns},
            )
        )
    return items


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"jsonl row in {path} is not an object")
                rows.append(payload)
    return rows
