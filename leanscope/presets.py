"""Size presets for LeanScope builds and GPU runs.

Presets only drop records from the full expansion. They do not reassign the
split lock and they do not re-roll the lean coin flip. A scenario that remains
keeps the split and the lean it has in the full set.

Hour estimates use the same rough L4 model as the full-run note: 50 tokens/s
of decode, 300 new tokens, so one generation is 6 seconds of decode. Recording
activations adds about half of that again (the full 14,000-record greedy run
lands at 35 hours, the middle of the 30-to-40 L4-hour band). Natural-lean
samples are decode only; ``generate_records`` does not cache activations for
them.

    decode hours = generations * 300 / 50 / 3600
    greedy hours = 1.5 * decode hours
    total hours  = greedy hours + natural-sample decode hours
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from leanscope.scenarios import PARAPHRASE_IDS
from leanscope.schema import FAMILIES, SPLITS, STRENGTHS, Record, SplitName
from leanscope.splits import SPLIT_RATIOS, _largest_remainder, scenario_hash

# Rough L4 decode rate used for every preset estimate. Not a measurement.
DECODE_TOKENS_PER_SECOND = 50
DEFAULT_NEW_TOKENS = 300
# Extra forward for activations, as a fraction of greedy decode.
ACTIVATION_DECODE_FACTOR = 1.5

PRESET_NAMES: tuple[str, ...] = ("pilot", "main", "full")

# Full-set second pass, not the default of --preset full. Printed so the
# 100-hour budget stays visible next to the smaller presets.
FULL_OPTIONAL_NATURAL_SAMPLES = 20


@dataclass(frozen=True)
class Preset:
    """Which rows of the full design a budget run keeps."""

    name: str
    paraphrase_ids: tuple[str, ...]
    orders: tuple[str, ...]
    kinds: tuple[str, ...]
    strengths: tuple[str, ...]
    # None keeps every scenario in the file. A count takes that many, split
    # by the existing lock ratios, without moving anyone.
    scenario_count: int | None
    natural_samples: int
    # None draws natural-lean samples for every kept none/trait row.
    # Otherwise only rows whose locked split is in this tuple.
    natural_splits: tuple[SplitName, ...] | None
    summary: str


PILOT = Preset(
    name="pilot",
    paraphrase_ids=("p0", "p1"),
    orders=("AB", "BA"),
    kinds=("none", "assigned", "neutral_matched"),
    strengths=("explicit", "soft"),
    scenario_count=40,
    natural_samples=10,
    natural_splits=None,
    summary=(
        "40 scenarios at the locked 60/15/25 mix, spread evenly across families, "
        "paraphrases p0 and p1, both orders, none plus explicit and soft assigned "
        "and neutral prompts, 10 natural-lean samples on each kept none row"
    ),
)

MAIN = Preset(
    name="main",
    paraphrase_ids=("p0", "p1", "p2"),
    orders=("AB", "BA"),
    kinds=("none", "trait", "assigned", "neutral_matched"),
    strengths=STRENGTHS,
    scenario_count=None,
    natural_samples=10,
    natural_splits=("dev",),
    summary=(
        "all scenarios, paraphrases p0 to p2, both orders, none, trait, and "
        "every assigned and neutral strength, 10 natural-lean samples on the "
        "locked dev split"
    ),
)

FULL = Preset(
    name="full",
    paraphrase_ids=PARAPHRASE_IDS,
    orders=("AB", "BA"),
    kinds=("none", "trait", "assigned", "neutral_matched", "conceal"),
    strengths=STRENGTHS,
    scenario_count=None,
    natural_samples=0,
    natural_splits=None,
    summary="the full expansion; natural-lean samples stay off unless requested",
)

_PRESETS = {preset.name: preset for preset in (PILOT, MAIN, FULL)}


def get_preset(name: str) -> Preset:
    try:
        return _PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"preset must be one of {PRESET_NAMES}, got {name!r}") from exc


def _scenario_meta(records: Sequence[Record]) -> tuple[dict[str, str], dict[str, str]]:
    splits: dict[str, str] = {}
    families: dict[str, str] = {}
    for row in records:
        sid = row.scenario.scenario_id
        previous = splits.get(sid)
        if previous is not None and previous != row.split:
            raise ValueError(f"{sid} has more than one split on its records")
        family = families.get(sid)
        if family is not None and family != row.scenario.family:
            raise ValueError(f"{sid} has more than one family on its records")
        splits[sid] = row.split
        families[sid] = row.scenario.family
    return splits, families


def _even_take(total: int, capacities: Sequence[int]) -> list[int]:
    """Share ``total`` as evenly as the capacities allow.

    Ties in the largest-remainder step go to the later index, matching the
    split lock. A family with fewer scenarios than its share keeps all of
    them, and the spare goes to families that still have room.
    """
    n = len(capacities)
    if n == 0:
        if total == 0:
            return []
        raise ValueError(f"cannot take {total} scenarios from an empty pool")
    if total < 0 or any(cap < 0 for cap in capacities):
        raise ValueError("counts must be non-negative")
    if total > sum(capacities):
        raise ValueError(f"asked for {total} scenarios, and only {sum(capacities)} are available")
    raw = [total / n for _ in range(n)]
    taken = [int(math.floor(x)) for x in raw]
    leftover = total - sum(taken)
    order = sorted(range(n), key=lambda i: (raw[i] - taken[i], i), reverse=True)
    for k in range(leftover):
        taken[order[k]] += 1
    for _ in range(n):
        overflow = 0
        spare: list[int] = []
        for i, cap in enumerate(capacities):
            if taken[i] > cap:
                overflow += taken[i] - cap
                taken[i] = cap
            elif taken[i] < cap:
                spare.append(i)
        if overflow == 0:
            return taken
        for i in sorted(spare, reverse=True):
            if overflow == 0:
                break
            room = capacities[i] - taken[i]
            give = min(room, overflow)
            taken[i] += give
            overflow -= give
        if overflow:
            raise ValueError(f"could not place {overflow} scenarios within family capacities")
    return taken


def select_scenario_ids(
    assignments: Mapping[str, str],
    preset: Preset,
    *,
    families: Mapping[str, str] | None = None,
) -> list[str]:
    """Scenario ids a preset keeps. Assignments are read, never rewritten.

    A counted preset takes a largest-remainder share of each existing split
    (the same 60/15/25 rule as the lock). Inside a split it spreads that share
    across families, and within a family it keeps the ids with the smallest
    split hash. The hash is only a stable order. It is not a new cut.
    """
    ids = sorted(set(assignments))
    if preset.scenario_count is None or preset.scenario_count >= len(ids):
        return ids
    if preset.scenario_count <= 0:
        raise ValueError(f"{preset.name}: scenario_count must be positive")
    by_split: dict[str, list[str]] = {name: [] for name in SPLITS}
    for sid in ids:
        split = assignments[sid]
        if split not in by_split:
            raise ValueError(f"{sid} has split {split!r}, not one of {SPLITS}")
        by_split[split].append(sid)
    for name in SPLITS:
        by_split[name].sort(key=lambda sid: (scenario_hash(sid), sid))
    counts = _largest_remainder(preset.scenario_count, SPLIT_RATIOS)
    chosen: list[str] = []
    for split, count in zip(SPLITS, counts):
        pool = by_split[split]
        if count > len(pool):
            raise ValueError(
                f"{preset.name} asks for {count} {split} scenarios, and the lock has {len(pool)}"
            )
        if not families:
            chosen.extend(pool[:count])
            continue
        grouped = {name: [sid for sid in pool if families[sid] == name] for name in FAMILIES}
        unknown = [sid for sid in pool if families.get(sid) not in FAMILIES]
        if unknown:
            raise ValueError(f"{unknown[:3]} have no known family")
        shares = _even_take(count, [len(grouped[name]) for name in FAMILIES])
        for name, share in zip(FAMILIES, shares):
            chosen.extend(grouped[name][:share])
    return sorted(chosen)


def _condition_kept(row: Record, preset: Preset) -> bool:
    condition = row.condition
    if condition.kind not in preset.kinds:
        return False
    if condition.kind in ("none", "trait"):
        return condition.strength is None
    return condition.strength in preset.strengths


def apply_preset(records: Sequence[Record], preset: Preset) -> list[Record]:
    """Return the subset of ``records`` that this preset generates.

    Item ids, splits, and condition leans are copied through. Nothing is rebuilt.
    """
    splits, families = _scenario_meta(records)
    chosen = set(select_scenario_ids(splits, preset, families=families))
    kept: list[Record] = []
    for row in records:
        if row.scenario.scenario_id not in chosen:
            continue
        if row.scenario.paraphrase_id not in preset.paraphrase_ids:
            continue
        if row.scenario.order not in preset.orders:
            continue
        if not _condition_kept(row, preset):
            continue
        kept.append(row)
    return kept


def natural_sample_ids(records: Sequence[Record], preset: Preset) -> list[str]:
    """None and trait item ids that receive natural-lean samples under this preset."""
    ids: list[str] = []
    for row in records:
        if row.condition.kind not in ("none", "trait"):
            continue
        if preset.natural_splits is not None and row.split not in preset.natural_splits:
            continue
        ids.append(row.item_id)
    return ids


def decode_hours(n_generations: int, *, new_tokens: int = DEFAULT_NEW_TOKENS) -> float:
    if n_generations < 0:
        raise ValueError("n_generations must be non-negative")
    return n_generations * new_tokens / DECODE_TOKENS_PER_SECOND / 3600


def estimate_run(
    records: Sequence[Record],
    preset: Preset,
    *,
    natural_samples: int | None = None,
    new_tokens: int = DEFAULT_NEW_TOKENS,
) -> dict[str, float | int | str]:
    """Record count and L4-hour estimate for the records a preset kept.

    ``natural_samples`` overrides the preset default. Pass the CLI value when
    the user set one.
    """
    samples = preset.natural_samples if natural_samples is None else int(natural_samples)
    if samples < 0:
        raise ValueError("natural_samples must be non-negative")
    sample_ids = natural_sample_ids(records, preset) if samples else []
    natural_generations = len(sample_ids) * samples
    greedy_hours = ACTIVATION_DECODE_FACTOR * decode_hours(len(records), new_tokens=new_tokens)
    natural_hours = decode_hours(natural_generations, new_tokens=new_tokens)
    scenarios = len({row.scenario.scenario_id for row in records})
    return {
        "preset": preset.name,
        "n_records": len(records),
        "n_scenarios": scenarios,
        "natural_samples": samples,
        "natural_rows": len(sample_ids),
        "natural_generations": natural_generations,
        "greedy_hours": greedy_hours,
        "natural_hours": natural_hours,
        "total_hours": greedy_hours + natural_hours,
        "by_split": dict(Counter(row.split for row in {row.scenario.scenario_id: row for row in records}.values())),
    }


def format_estimate(estimate: Mapping[str, float | int | str]) -> str:
    """One line a script can print before any GPU work."""
    return (
        f"preset={estimate['preset']} records={estimate['n_records']} "
        f"scenarios={estimate['n_scenarios']} "
        f"greedy_generations={estimate['n_records']} "
        f"natural_generations={estimate['natural_generations']} "
        f"estimated_l4_hours={float(estimate['total_hours']):.1f}"
    )


def optional_full_natural_line(
    full_records: Sequence[Record],
    *,
    new_tokens: int = DEFAULT_NEW_TOKENS,
) -> str:
    """What ``--preset full --natural-samples 20`` adds on the full file."""
    estimate = estimate_run(
        full_records,
        FULL,
        natural_samples=FULL_OPTIONAL_NATURAL_SAMPLES,
        new_tokens=new_tokens,
    )
    return (
        "optional natural-samples "
        f"{FULL_OPTIONAL_NATURAL_SAMPLES} on the full file adds "
        f"{estimate['natural_generations']} generations and brings the total to "
        f"about {float(estimate['total_hours']):.1f} L4-hours"
    )


def iter_presets() -> Iterable[Preset]:
    return (PILOT, MAIN, FULL)
