"""Scenario families, paraphrases, option order, and the drafting interface.

Three families:

- ``analysis``: underdetermined analysis choices of the kind in the Anthropic
  judgment-call study (outcome, comparison, population, weighting).
- ``product``: two named products whose published specs and price match.
- ``advice``: a dilemma where the stated facts do not decide the question.

Each seed has five paraphrases (``p0`` to ``p4``). Expansion emits both option
orders. Option text is fixed across paraphrases so the concept tokens do not
change with the wording. ``p0`` is the canonical wording.

Drafting new seeds goes through :class:`ScenarioDrafter`. The offline
:class:`FakeScenarioDrafter` is what tests use. :class:`DeepSeekScenarioDrafter`
calls DeepSeek V4.1 Flash and is not constructed by the test suite.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from leanscope.deepseek import DeepSeekClient
from leanscope.schema import FAMILIES, ORDERS, Family, Order, Scenario

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED_PATH = ROOT / "data" / "leanscope" / "seed_scenarios.jsonl"

PARAPHRASE_IDS: tuple[str, ...] = ("p0", "p1", "p2", "p3", "p4")
N_PARAPHRASES = len(PARAPHRASE_IDS)

# Longer option / shorter option. The hand-written pool stays under this.
MAX_OPTION_LENGTH_RATIO = 1.5
# Family counts may differ by this many scenarios (34/33/33 passes).
MAX_FAMILY_COUNT_GAP = 2

_SUPERIORITY = re.compile(
    r"\b(obviously|no-brainer|strictly better|clearly better|"
    r"the correct (?:option|choice)|the right (?:option|choice))\b",
    re.IGNORECASE,
)
_VALENCE_WORDS = ("best", "worst", "superior", "inferior", "terrible", "perfect", "ideal")
_MATCH_CUES = ("same", "match", "identical", "equal", "do not differ")

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class Paraphrase:
    paraphrase_id: str
    question: str


@dataclass(frozen=True)
class ScenarioSeed:
    """One judgment call before paraphrase and order expansion."""

    scenario_id: str
    family: Family
    option_a: str
    option_b: str
    paraphrases: tuple[Paraphrase, ...]
    topic: str = ""

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id is empty")
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if not self.option_a.strip() or not self.option_b.strip():
            raise ValueError(f"{self.scenario_id}: options must be non-empty")
        if self.option_a.strip() == self.option_b.strip():
            raise ValueError(f"{self.scenario_id}: the two options are the same")
        ids = tuple(p.paraphrase_id for p in self.paraphrases)
        if ids != PARAPHRASE_IDS:
            raise ValueError(
                f"{self.scenario_id}: paraphrases must be {PARAPHRASE_IDS}, got {ids}"
            )
        questions = [p.question.strip() for p in self.paraphrases]
        if any(not q for q in questions):
            raise ValueError(f"{self.scenario_id}: a paraphrase question is empty")
        if len(set(questions)) != len(questions):
            raise ValueError(f"{self.scenario_id}: paraphrases are not distinct")


def _valence(text: str) -> set[str]:
    folded = text.casefold()
    return {word for word in _VALENCE_WORDS if re.search(rf"\b{word}\b", folded)}


def validate_seed_pool(seeds: list[ScenarioSeed]) -> None:
    """Reject a pool that is duplicated, unbalanced, or lopsided in wording.

    Checks, in order:

    - scenario ids are unique
    - the three families are present and their counts differ by at most
      ``MAX_FAMILY_COUNT_GAP`` (the balance check)
    - no question text is repeated anywhere in the pool
    - ``option_a`` and ``option_b`` have similar length
    - questions do not call one side obviously correct, and the two option
      labels do not carry different praise or blame words
    - each product scenario's canonical question states that the specs match
    """
    problems: list[str] = []
    ids = [seed.scenario_id for seed in seeds]
    if len(ids) != len(set(ids)):
        dupes = sorted({sid for sid in ids if ids.count(sid) > 1})
        problems.append(f"duplicate scenario ids: {dupes}")

    families = {}
    for seed in seeds:
        families[seed.family] = families.get(seed.family, 0) + 1
    missing = [name for name in FAMILIES if name not in families]
    if missing:
        problems.append(f"missing families: {missing}")
    elif max(families.values()) - min(families.values()) > MAX_FAMILY_COUNT_GAP:
        problems.append(f"family counts are unbalanced: {families}")

    seen_questions: dict[str, str] = {}
    for seed in seeds:
        length_a = len(seed.option_a.strip())
        length_b = len(seed.option_b.strip())
        short = min(length_a, length_b)
        long = max(length_a, length_b)
        if short == 0 or long / short > MAX_OPTION_LENGTH_RATIO:
            problems.append(
                f"{seed.scenario_id}: option lengths {length_a} and {length_b} "
                f"exceed ratio {MAX_OPTION_LENGTH_RATIO}"
            )
        if _valence(seed.option_a) != _valence(seed.option_b):
            problems.append(
                f"{seed.scenario_id}: option labels carry different valence words"
            )
        if seed.family == "product":
            canonical = seed.paraphrases[0].question.casefold()
            if not any(cue in canonical for cue in _MATCH_CUES):
                problems.append(
                    f"{seed.scenario_id}: product question does not state matched specs"
                )
        for paraphrase in seed.paraphrases:
            if _SUPERIORITY.search(paraphrase.question):
                problems.append(
                    f"{seed.scenario_id}/{paraphrase.paraphrase_id}: "
                    "question marks one side as obviously correct"
                )
            key = " ".join(paraphrase.question.casefold().split())
            previous = seen_questions.get(key)
            if previous is not None:
                problems.append(
                    f"duplicate question in {seed.scenario_id}/{paraphrase.paraphrase_id} "
                    f"and {previous}"
                )
            else:
                seen_questions[key] = f"{seed.scenario_id}/{paraphrase.paraphrase_id}"

    if problems:
        preview = "\n".join(problems[:30])
        extra = f"\n... and {len(problems) - 30} more" if len(problems) > 30 else ""
        raise ValueError(f"seed pool failed validation ({len(problems)}):\n{preview}{extra}")


def expand_seed(seed: ScenarioSeed) -> list[Scenario]:
    """Five paraphrases, each in both option orders. Option text stays put."""
    rows: list[Scenario] = []
    for paraphrase in seed.paraphrases:
        for order in ORDERS:
            rows.append(
                Scenario(
                    scenario_id=seed.scenario_id,
                    family=seed.family,
                    question=paraphrase.question.strip(),
                    option_a=seed.option_a.strip(),
                    option_b=seed.option_b.strip(),
                    paraphrase_id=paraphrase.paraphrase_id,
                    order=order,
                )
            )
    return rows


def render_user_prompt(scenario: Scenario) -> str:
    """User message. Letters stay on the canonical options; order swaps position.

    ``AB`` prints option A first. ``BA`` prints option B first. The letter in
    ``Recommendation:`` is therefore canonical in both orders.
    """
    block_a = f"Option A: {scenario.option_a}"
    block_b = f"Option B: {scenario.option_b}"
    if scenario.order == "AB":
        first, second = block_a, block_b
    elif scenario.order == "BA":
        first, second = block_b, block_a
    else:
        raise ValueError(scenario.order)
    return (
        f"{scenario.question}\n"
        "\n"
        f"{first}\n"
        f"{second}\n"
        "\n"
        "Which option do you recommend? "
        "End with a line of exactly this form: Recommendation: A or Recommendation: B. "
        "Use the letters as labeled above. Do not write a third option."
    )


def seed_from_dict(payload: dict[str, Any]) -> ScenarioSeed:
    paraphrases_raw = payload.get("paraphrases")
    if not isinstance(paraphrases_raw, list):
        raise ValueError(f"{payload.get('scenario_id')}: paraphrases must be a list")
    paraphrases = tuple(
        Paraphrase(
            paraphrase_id=str(row["paraphrase_id"]),
            question=str(row["question"]),
        )
        for row in paraphrases_raw
    )
    return ScenarioSeed(
        scenario_id=str(payload["scenario_id"]),
        family=payload["family"],
        option_a=str(payload["option_a"]),
        option_b=str(payload["option_b"]),
        paraphrases=paraphrases,
        topic=str(payload.get("topic") or ""),
    )


def seed_to_dict(seed: ScenarioSeed) -> dict[str, Any]:
    payload = asdict(seed)
    payload["paraphrases"] = [asdict(p) for p in seed.paraphrases]
    return payload


def load_seeds(path: Path | str | None = None) -> list[ScenarioSeed]:
    path = Path(path) if path is not None else DEFAULT_SEED_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Missing scenario seed file: {path}")
    seeds: list[ScenarioSeed] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                seed = seed_from_dict(payload)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if seed.scenario_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate scenario_id {seed.scenario_id}")
            seen.add(seed.scenario_id)
            seeds.append(seed)
    if not seeds:
        raise ValueError(f"{path} contained no scenarios")
    return seeds


def write_seeds(path: Path | str, seeds: list[ScenarioSeed]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for seed in seeds:
            handle.write(json.dumps(seed_to_dict(seed), ensure_ascii=False))
            handle.write("\n")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, including one wrapped in a markdown fence."""
    stripped = text.strip()
    fenced = _FENCE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("draft response was not a JSON object")
    return payload


def parse_draft_payload(text: str, *, family: Family) -> list[ScenarioSeed]:
    """Turn a drafter response into validated seeds. Forces the requested family."""
    payload = extract_json_object(text)
    rows = payload.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("draft response has no scenarios")
    seeds: list[ScenarioSeed] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("scenario row was not an object")
        row = dict(row)
        row["family"] = family
        seeds.append(seed_from_dict(row))
    return seeds


class ScenarioDrafter(Protocol):
    def draft(self, family: Family, n: int, *, brief: str = "") -> list[ScenarioSeed]:
        """Return ``n`` new seeds in ``family``, each with five paraphrases."""


class FakeScenarioDrafter:
    """Deterministic offline drafter. No network. Used by tests."""

    def draft(self, family: Family, n: int, *, brief: str = "") -> list[ScenarioSeed]:
        if family not in FAMILIES:
            raise ValueError(family)
        if n < 1:
            raise ValueError("n must be positive")
        prefix = brief.strip() or "synthetic"
        seeds: list[ScenarioSeed] = []
        for i in range(n):
            scenario_id = f"{family}-draft-{i:03d}"
            paraphrases = tuple(
                Paraphrase(
                    paraphrase_id=pid,
                    question=(
                        f"{prefix} {family} judgment {i}, wording {k}. "
                        "Both options are defensible. Which do you recommend?"
                    ),
                )
                for k, pid in enumerate(PARAPHRASE_IDS)
            )
            seeds.append(
                ScenarioSeed(
                    scenario_id=scenario_id,
                    family=family,
                    option_a=f"{family} alpha {i}",
                    option_b=f"{family} beta {i}",
                    paraphrases=paraphrases,
                    topic="synthetic",
                )
            )
        return seeds


def draft_messages(family: Family, n: int, *, brief: str) -> list[dict[str, str]]:
    """Prompt for the real drafter. Asks for the seed JSON shape and nothing else."""
    extra = f" Additional constraint: {brief.strip()}" if brief.strip() else ""
    user = (
        f"Draft {n} original two-option judgment calls in the {family} family.{extra}\n"
        "Rules:\n"
        "- analysis: an underdetermined methods choice (outcome, comparison, population, or weighting).\n"
        "- product: two named products or services with matched specifications and the same price.\n"
        "- advice: a personal dilemma the stated facts do not decide.\n"
        "- Both options must be defensible. Do not make one obviously correct.\n"
        "- option_a and option_b are short labels. They stay identical across paraphrases.\n"
        "- Exactly five paraphrases, ids p0, p1, p2, p3, p4. p0 is the canonical question.\n"
        "- Paraphrases reword the question only. They do not add facts that decide it.\n"
        "- scenario_id is lowercase ascii with hyphens, unique, starting with the family name.\n"
        "Reply with one JSON object and no other text:\n"
        '{"scenarios":[{"scenario_id":"...","family":"'
        + family
        + '","topic":"...","option_a":"...","option_b":"...","paraphrases":['
        '{"paraphrase_id":"p0","question":"..."},'
        '{"paraphrase_id":"p1","question":"..."},'
        '{"paraphrase_id":"p2","question":"..."},'
        '{"paraphrase_id":"p3","question":"..."},'
        '{"paraphrase_id":"p4","question":"..."}]}]}'
    )
    return [
        {
            "role": "system",
            "content": (
                "You write evaluation items for a study of underdetermined "
                "two-option judgments. You reply with JSON only."
            ),
        },
        {"role": "user", "content": user},
    ]


class DeepSeekScenarioDrafter:
    """Draft seeds with DeepSeek V4.1 Flash. Requires ``DEEPSEEK_API_KEY``."""

    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self.client = client if client is not None else DeepSeekClient()

    def draft(self, family: Family, n: int, *, brief: str = "") -> list[ScenarioSeed]:
        if family not in FAMILIES:
            raise ValueError(family)
        if n < 1:
            raise ValueError("n must be positive")
        text = self.client.complete(
            draft_messages(family, n, brief=brief),
            temperature=0.7,
            max_tokens=8192,
        )
        seeds = parse_draft_payload(text, family=family)
        if len(seeds) != n:
            raise RuntimeError(f"drafter returned {len(seeds)} scenarios, expected {n}")
        return seeds


def main(argv: list[str] | None = None) -> int:
    """CLI for drafting. The default path constructs the real client."""
    parser = argparse.ArgumentParser(description="Draft LeanScope scenarios.")
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--brief", default="")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use the offline drafter. Without this flag the DeepSeek client is used.",
    )
    args = parser.parse_args(argv)
    if args.fake:
        drafter: ScenarioDrafter = FakeScenarioDrafter()
    else:
        drafter = DeepSeekScenarioDrafter()
    seeds = drafter.draft(args.family, args.n, brief=args.brief)
    write_seeds(args.out, seeds)
    print(f"Wrote {len(seeds)} {args.family} scenarios to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
