"""Build records and run generation through jspace.

``build_records`` is the item constructor behind ``scripts/lean_00_build_items.py``.
``generate_records`` is the runner behind ``scripts/lean_01_generate.py``. It
renders each item with ``jspace.generation.apply_chat_template`` and generates
with ``generate_greedy`` (or ``generate_with_tiered_cache`` when a lens is
loaded and an activations directory is set).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from leanscope.conditions import coin_flip_lean, condition_set
from leanscope.labels import NaturalLeanResult, natural_lean_from_outputs, parse_recommendation
from leanscope.scenarios import ScenarioSeed, expand_seed, render_user_prompt
from leanscope.schema import LeanCondition, Record, Scenario, write_jsonl

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


def make_item_id(scenario: Scenario, condition: LeanCondition) -> str:
    strength = condition.strength or "na"
    lean = condition.lean or "na"
    return (
        f"{scenario.scenario_id}|{scenario.paraphrase_id}|{scenario.order}"
        f"|{condition.kind}|{strength}|{lean}"
    )


def build_records(
    seeds: Sequence[ScenarioSeed],
    assignments: dict[str, str],
    *,
    lean_seed: int = 42,
) -> list[Record]:
    """Expand seeds into records. One coin-flip lean and one split per scenario_id."""
    records: list[Record] = []
    for seed in seeds:
        if seed.scenario_id not in assignments:
            raise KeyError(f"{seed.scenario_id} has no split assignment")
        split = assignments[seed.scenario_id]
        lean = coin_flip_lean(seed.scenario_id, seed=lean_seed)
        conditions = condition_set(lean)
        for scenario in expand_seed(seed):
            for condition in conditions:
                records.append(
                    Record(
                        item_id=make_item_id(scenario, condition),
                        scenario=scenario,
                        condition=condition,
                        output_text="",
                        recommended=None,
                        voiced=None,
                        activations_path=None,
                        split=split,  # type: ignore[arg-type]
                    )
                )
    ids = [row.item_id for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("item_id collision while building records")
    return records


def manifest(records: Sequence[Record], *, lean_seed: int) -> dict[str, Any]:
    """Counts for a build, so a later run can see what the file contains."""
    return {
        "n_records": len(records),
        "n_scenarios": len({row.scenario.scenario_id for row in records}),
        "lean_seed": int(lean_seed),
        "by_family": dict(Counter(row.scenario.family for row in records)),
        "by_kind": dict(Counter(row.condition.kind for row in records)),
        "by_split": dict(Counter(row.split for row in records)),
        "by_order": dict(Counter(row.scenario.order for row in records)),
    }


def render_model_prompt(tokenizer: Any, scenario: Scenario, condition: LeanCondition) -> str:
    """Chat-templated prompt. System text is the condition; user text is the scenario."""
    from jspace.generation import apply_chat_template

    return apply_chat_template(
        tokenizer,
        render_user_prompt(scenario),
        system=condition.system_prompt,
    )


def stable_prompt_id(item_id: str, replicate: int = 0) -> int:
    """Sampling seed input. Stable across runs, distinct per replicate."""
    digest = hashlib.sha256(f"{item_id}:{int(replicate)}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


def _trace_dirname(item_id: str) -> str:
    return _SAFE_ID.sub("_", item_id)


@dataclass
class GenerateConfig:
    max_new_tokens: int = 300
    temperature: float = 0.0
    natural_samples: int = 0
    sample_temperature: float = 0.7
    activations_dir: Path | None = None
    limit: int | None = None


def merge_existing(fresh: Sequence[Record], existing: Sequence[Record]) -> list[Record]:
    """Copy completed outputs onto a freshly built record list, matched by item_id."""
    done = {row.item_id: row for row in existing if row.output_text.strip()}
    merged: list[Record] = []
    for row in fresh:
        previous = done.get(row.item_id)
        if previous is not None:
            row.output_text = previous.output_text
            row.recommended = previous.recommended
            row.voiced = previous.voiced
            row.activations_path = previous.activations_path
        merged.append(row)
    return merged


def generate_records(
    records: Sequence[Record],
    stack: Any,
    config: GenerateConfig | None = None,
    *,
    skip_filled: bool = True,
    already_sampled: set[str] | None = None,
    checkpoint_path: Path | str | None = None,
    checkpoint_every: int = 20,
) -> tuple[list[Record], list[NaturalLeanResult], list[dict[str, Any]]]:
    """Greedy generation for every record, plus optional natural-lean samples.

    Natural-lean samples are drawn only for ``none`` and ``trait`` (no lean
    prompt). Activations are recorded only when ``activations_dir`` is set and
    ``stack.lens`` is loaded; otherwise the text is still generated and
    ``activations_path`` stays None.

    Records that already have ``output_text`` are left alone when
    ``skip_filled`` is set, so a GPU run can resume.
    """
    from jspace.generation import generate_greedy, generate_with_tiered_cache, save_trace

    config = config or GenerateConfig()
    already_sampled = set(already_sampled or ())
    rows = list(records)
    natural_results: list[NaturalLeanResult] = []
    natural_rows: list[dict[str, Any]] = []
    lens = getattr(stack, "lens", None)
    record_activations = config.activations_dir is not None and lens is not None
    if config.activations_dir is not None and lens is None:
        logger.warning(
            "Activations directory %s was set but stack.lens is None. "
            "Generating text only.",
            config.activations_dir,
        )

    limit = None if config.limit is None else int(config.limit)
    processed = 0
    for record in rows:
        if limit is not None and processed >= limit:
            break
        processed += 1
        if not (skip_filled and record.output_text.strip()):
            prompt = render_model_prompt(stack.tokenizer, record.scenario, record.condition)
            prompt_id = stable_prompt_id(record.item_id, 0)
            if record_activations:
                generated = generate_with_tiered_cache(
                    stack,
                    prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    prompt_id=prompt_id,
                )
                trace_dir = Path(config.activations_dir) / _trace_dirname(record.item_id)  # type: ignore[arg-type]
                save_trace(
                    trace_dir,
                    generated,
                    {
                        "item_id": record.item_id,
                        "generated_text": generated.get("generated_text", ""),
                        "seed": generated.get("seed"),
                        "temperature": generated.get("temperature"),
                        "greedy": generated.get("greedy"),
                    },
                )
                record.activations_path = str(trace_dir)
            else:
                generated = generate_greedy(
                    stack,
                    prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    prompt_id=prompt_id,
                )
            record.output_text = str(generated.get("generated_text", ""))
            record.recommended = parse_recommendation(record.output_text, record.scenario)

        if config.natural_samples > 0 and record.condition.kind in ("none", "trait"):
            if record.item_id in already_sampled:
                continue
            samples: list[str] = []
            prompt = render_model_prompt(stack.tokenizer, record.scenario, record.condition)
            for replicate in range(1, int(config.natural_samples) + 1):
                sample = generate_greedy(
                    stack,
                    prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.sample_temperature,
                    prompt_id=stable_prompt_id(record.item_id, replicate),
                )
                text = str(sample.get("generated_text", ""))
                samples.append(text)
                natural_rows.append(
                    {
                        "item_id": record.item_id,
                        "scenario_id": record.scenario.scenario_id,
                        "paraphrase_id": record.scenario.paraphrase_id,
                        "order": record.scenario.order,
                        "family": record.scenario.family,
                        "replicate": replicate,
                        "output_text": text,
                        "recommended": parse_recommendation(text, record.scenario),
                    }
                )
            natural_results.append(natural_lean_from_outputs(samples, record.scenario))
            already_sampled.add(record.item_id)

        if checkpoint_path is not None and checkpoint_every > 0 and processed % checkpoint_every == 0:
            write_jsonl(checkpoint_path, rows)

    if checkpoint_path is not None:
        write_jsonl(checkpoint_path, rows)
    return rows, natural_results, natural_rows


def write_records(path: Path | str, records: Sequence[Record]) -> int:
    return write_jsonl(path, records)
