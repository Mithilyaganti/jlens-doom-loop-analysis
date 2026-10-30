"""96-generation T4 pilot: planning and a CPU harness.

Ninety-six generations is 24 prompts in each of English, Hindi, Telugu,
and romanized Telugu. The harness times a caller-supplied generator and
scores the texts with D0–D4. It does not load a model.

``plan_go_nogo`` turns a tokens-per-second figure into go / replan / no-go.
Without a measured figure the decision is ``pending``. An assumed figure
is labeled assumed and is not a result.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from loop_lingua.detectors import VARIANTS, Detection, Variant, detect
from loop_lingua.prompts import PILOT_PER_LANGUAGE

PILOT_GENERATIONS = 96
PILOT_LANGUAGES: tuple[str, ...] = ("en", "hi", "te", "te_roman")
FULL_PROMPTS_PER_LANGUAGE = 300
# Planning length for the full study. The pilot itself generates fewer
# tokens so a T4 can measure throughput; see DESIGN_DECISIONS.md.
FULL_MAX_NEW_TOKENS = 2048
PILOT_MAX_NEW_TOKENS = 512
BUDGET_HOURS = 8.0
# Below this, a 4B model on a T4 is not really on the GPU (offload, a
# debug batch, or a stalled process). It is a health floor, not a benchmark.
HEALTH_FLOOR_TOKENS_PER_SECOND = 15.0


@dataclass(frozen=True)
class PilotJob:
    item_id: str
    language: str
    prompt: str


@dataclass(frozen=True)
class Generation:
    item_id: str
    language: str
    text: str
    token_ids: tuple[int, ...]
    new_tokens: int
    elapsed_seconds: float


@dataclass(frozen=True)
class PilotConfig:
    per_language: int = PILOT_PER_LANGUAGE
    languages: tuple[str, ...] = PILOT_LANGUAGES
    pilot_max_new_tokens: int = PILOT_MAX_NEW_TOKENS
    full_prompts_per_language: int = FULL_PROMPTS_PER_LANGUAGE
    full_max_new_tokens: int = FULL_MAX_NEW_TOKENS
    budget_hours: float = BUDGET_HOURS
    health_floor_tokens_per_second: float = HEALTH_FLOOR_TOKENS_PER_SECOND
    model_id: str = "Qwen/Qwen3.5-4B"
    device: str = "T4"

    @property
    def n_generations(self) -> int:
        return self.per_language * len(self.languages)

    @property
    def full_sequences(self) -> int:
        return self.full_prompts_per_language * len(self.languages)

    @property
    def full_new_tokens_upper_bound(self) -> int:
        """Every sequence running out to the cap. Real runs stop sooner."""
        return self.full_sequences * self.full_max_new_tokens


@dataclass
class PilotReport:
    config: PilotConfig
    generations: list[Generation]
    tokens_per_second: float
    loop_counts: dict[str, int]
    loop_counts_by_language: dict[str, dict[str, int]]
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def n_generations(self) -> int:
        return len(self.generations)


@dataclass(frozen=True)
class GoNoGo:
    decision: str
    basis: str
    tokens_per_second: float | None
    estimated_full_hours: float | None
    full_new_tokens_upper_bound: int
    max_new_tokens_that_fit: int | None
    reasons: tuple[str, ...]
    caveats: tuple[str, ...]


_CAVEATS = (
    "Hours assume tokens per second stay constant out to full_max_new_tokens. "
    "Long contexts on a T4 are slower than a short pilot, so a 'go' is an upper bound on speed.",
    "The token total assumes every sequence emits full_max_new_tokens. Early stops make the run shorter.",
    "Decoding in this plan is greedy. Liquid's Antidoom pipeline used temperature 0.01.",
)


def jobs_from_records(
    records: Sequence[Mapping[str, object]],
    pilot_ids: Sequence[str],
    *,
    languages: Sequence[str] = PILOT_LANGUAGES,
) -> list[PilotJob]:
    """Pick the pilot rows out of the matched prompt records."""
    wanted = set(pilot_ids)
    by_key: dict[tuple[str, str], str] = {}
    for row in records:
        item_id = str(row["item_id"])
        language = str(row["language"])
        if item_id in wanted and language in languages:
            prompt = row["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"empty prompt for {item_id} {language}")
            by_key[(item_id, language)] = prompt
    jobs: list[PilotJob] = []
    for item_id in pilot_ids:
        for language in languages:
            key = (item_id, language)
            if key not in by_key:
                raise KeyError(f"missing pilot prompt for {item_id} / {language}")
            jobs.append(PilotJob(item_id=item_id, language=language, prompt=by_key[key]))
    if len(jobs) != len(pilot_ids) * len(languages):
        raise RuntimeError("pilot job count does not match ids × languages")
    return jobs


def plan_go_nogo(
    tokens_per_second: float | None,
    config: PilotConfig | None = None,
    *,
    basis: str = "measured",
) -> GoNoGo:
    """Decide whether the full 300×4 study fits on one T4.

    ``basis`` is ``measured`` only when ``tokens_per_second`` came from the
    harness on the T4. Use ``assumed`` for a what-if, and omit the rate
    (``basis='none'``) when nothing has been timed.
    """
    cfg = config or PilotConfig()
    if basis not in {"measured", "assumed", "none"}:
        raise ValueError("basis must be measured, assumed, or none")
    upper = cfg.full_new_tokens_upper_bound
    if tokens_per_second is None or basis == "none":
        return GoNoGo(
            decision="pending",
            basis="none",
            tokens_per_second=None,
            estimated_full_hours=None,
            full_new_tokens_upper_bound=upper,
            max_new_tokens_that_fit=None,
            reasons=("No tokens/second figure yet. Run the 96-generation pilot on a T4.",),
            caveats=_CAVEATS,
        )
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    if basis == "assumed":
        prefix = "ASSUMED rate, not a measurement. "
    else:
        prefix = ""
    hours = upper / tokens_per_second / 3600
    fit = int(tokens_per_second * cfg.budget_hours * 3600 / cfg.full_sequences)
    reasons: list[str] = []
    if tokens_per_second < cfg.health_floor_tokens_per_second:
        decision = "no-go"
        reasons.append(
            prefix
            + f"{tokens_per_second:.1f} tok/s is below the {cfg.health_floor_tokens_per_second:.0f} tok/s "
            "health floor for a 4B model on a T4."
        )
    elif hours <= cfg.budget_hours:
        decision = "go"
        reasons.append(
            prefix
            + f"Upper bound {hours:.2f} h is within the {cfg.budget_hours:.0f} h budget "
            f"at {cfg.full_max_new_tokens} new tokens."
        )
    else:
        decision = "replan"
        reasons.append(
            prefix
            + f"Upper bound {hours:.2f} h exceeds the {cfg.budget_hours:.0f} h budget. "
            f"At this rate, {fit} new tokens per sequence fit in the budget."
        )
    return GoNoGo(
        decision=decision,
        basis=basis,
        tokens_per_second=tokens_per_second,
        estimated_full_hours=hours,
        full_new_tokens_upper_bound=upper,
        max_new_tokens_that_fit=fit,
        reasons=tuple(reasons),
        caveats=_CAVEATS,
    )


GenerateFn = Callable[[PilotJob], Generation]


def _score(text: str, token_ids: Sequence[int], chars_per_token: float | None) -> dict[Variant, Detection]:
    scored: dict[Variant, Detection] = {}
    for variant in VARIANTS:
        if variant == "D3":
            scored[variant] = detect(text, "D3", token_ids=token_ids)
        elif variant == "D4":
            if chars_per_token is None:
                continue
            scored[variant] = detect(text, "D4", chars_per_token=chars_per_token)
        else:
            scored[variant] = detect(text, variant)
    return scored


def run_harness(
    jobs: Sequence[PilotJob],
    generate: GenerateFn,
    *,
    config: PilotConfig | None = None,
    chars_per_token: float | None = None,
) -> PilotReport:
    """Time ``generate`` and score loops. ``generate`` is injected so tests stay on CPU."""
    cfg = config or PilotConfig()
    if len(jobs) != cfg.n_generations:
        raise ValueError(f"expected {cfg.n_generations} jobs, got {len(jobs)}")
    generations: list[Generation] = []
    loop_counts: dict[str, int] = {variant: 0 for variant in VARIANTS}
    by_language: dict[str, dict[str, int]] = {
        language: {variant: 0 for variant in VARIANTS} for language in cfg.languages
    }
    started = time.perf_counter()
    for job in jobs:
        generation = generate(job)
        if generation.new_tokens < 0 or generation.elapsed_seconds < 0:
            raise ValueError("generation counts must be non-negative")
        generations.append(generation)
        scored = _score(generation.text, generation.token_ids, chars_per_token)
        for variant, detection in scored.items():
            if detection.is_loop:
                loop_counts[variant] += 1
                by_language[generation.language][variant] += 1
    elapsed = time.perf_counter() - started
    total_tokens = sum(item.new_tokens for item in generations)
    total_gen_seconds = sum(item.elapsed_seconds for item in generations)
    if total_gen_seconds <= 0:
        raise ValueError("total generation time is zero; cannot report throughput")
    return PilotReport(
        config=cfg,
        generations=generations,
        tokens_per_second=total_tokens / total_gen_seconds,
        loop_counts=loop_counts,
        loop_counts_by_language=by_language,
        elapsed_seconds=elapsed,
        notes=[
            "Throughput uses the generator's own elapsed time, summed across jobs.",
            "D4 is counted only when chars_per_token is passed in.",
        ],
    )


def describe_plan(config: PilotConfig | None = None) -> str:
    cfg = config or PilotConfig()
    lines = [
        f"Pilot: {cfg.n_generations} generations = {cfg.per_language} prompts × {len(cfg.languages)} languages.",
        f"Languages: {', '.join(cfg.languages)}.",
        f"Pilot cap: {cfg.pilot_max_new_tokens} new tokens. Full-study cap used for the budget: {cfg.full_max_new_tokens}.",
        f"Full study: {cfg.full_sequences} sequences, upper bound {cfg.full_new_tokens_upper_bound} new tokens.",
        f"Budget: {cfg.budget_hours:g} h on one {cfg.device}. Health floor: {cfg.health_floor_tokens_per_second:g} tok/s.",
        f"Model id (not loaded here): {cfg.model_id}.",
        "This process has not generated any tokens.",
    ]
    return "\n".join(lines)


def transformers_greedy_generate(model: object, tokenizer: object, job: PilotJob, *, max_new_tokens: int) -> Generation:
    """Greedy generation adapter for a T4 session.

    Not called by tests, the dry-run, or this repository's CPU suite.
    ``model`` and ``tokenizer`` are the already-loaded Hugging Face objects.
    Confirm ``model.generate`` exists for Qwen3.5's multimodal class before
    relying on it; that check is part of the T4 session, which has not been run.
    """
    import torch

    if not hasattr(tokenizer, "__call__") or not hasattr(tokenizer, "decode"):
        raise TypeError("tokenizer does not look like a Hugging Face tokenizer")
    if not hasattr(model, "generate") or not hasattr(model, "device"):
        raise TypeError("model does not expose generate() and device")
    encoded = tokenizer(job.prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_len = int(inputs["input_ids"].shape[1])
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - started
    new_ids = [int(token) for token in output[0, prompt_len:].tolist()]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return Generation(
        item_id=job.item_id,
        language=job.language,
        text=text,
        token_ids=tuple(new_ids),
        new_tokens=len(new_ids),
        elapsed_seconds=elapsed,
    )
