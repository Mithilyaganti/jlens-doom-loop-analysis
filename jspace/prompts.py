"""Real prompt loading from LiquidAI/antidoom-mix-v1.0 (ShareGPT-style schema)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Component source tags live in row id: antidoom:<component>:<idx>
REASONING_COMPONENT_SOURCES: tuple[str, ...] = (
    "gsm8k_train",
    "math_lighteval_train",
    "math_qa_train",
    "apps_train",
    "open_perfectblend_ultrainteract",
    "open_perfectblend_metamathqa",
    "open_perfectblend_evol_codealpaca",
)

SKIPPED_COMPONENT_SOURCES: tuple[str, ...] = (
    "mmlu_auxiliary_train",
    "commonsense_qa_train",
    "pubmedqa_artificial_train",
    "open_perfectblend_ultrachat200k",
    "open_perfectblend_autoif",
    "open_perfectblend_lmsys_arena",
    "ifstruct_train_generated",
)

DEFAULT_SAMPLE_PATH = Path("results/prompt_sample_ids.json")


def extract_user_text_from_conversations(conversations: Any) -> str | None:
    """Parse antidoom-mix conversations field.

    Schema (verified on HF): List[{'from': 'human'|'gpt'|..., 'value': str}]
    Prefer the last human/user turn.
    """
    if not conversations:
        return None
    if isinstance(conversations, str):
        return conversations.strip() or None
    if not isinstance(conversations, list):
        return None

    last_human: str | None = None
    for msg in conversations:
        if not isinstance(msg, dict):
            if isinstance(msg, str) and msg.strip():
                last_human = msg
            continue
        role = (
            msg.get("from")
            or msg.get("role")
            or msg.get("speaker")
            or ""
        )
        role_l = str(role).lower()
        text = msg.get("value") or msg.get("content") or msg.get("text")
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        if role_l in ("human", "user", "prompter"):
            last_human = text
        elif last_human is None and role_l not in ("gpt", "assistant", "bot", "model"):
            last_human = text
    return last_human


def component_source_from_row(row: dict[str, Any]) -> str | None:
    """Return antidoom component tag from row id (antidoom:<component>:<idx>)."""
    rid = row.get("id")
    if not rid:
        return None
    parts = str(rid).split(":")
    if len(parts) >= 2 and parts[0] == "antidoom":
        return parts[1]
    return None


def _row_to_prompt(row_idx: int, row: dict[str, Any]) -> dict[str, Any] | None:
    text = extract_user_text_from_conversations(row.get("conversations"))
    if text is None:
        for key in ("prompt", "question", "text", "input"):
            if row.get(key):
                text = str(row[key]).strip()
                break
    if not text:
        return None
    comp = component_source_from_row(row)
    return {
        "row_index": row_idx,
        "prompt_id": row_idx,
        "id": row.get("id"),
        "text": text[:4000],
        "source": comp or row.get("source") or "unknown",
        "hf_source": row.get("source"),
        "dataset": "LiquidAI/antidoom-mix-v1.0",
    }


def _per_source_counts(total: int, n_sources: int) -> list[int]:
    base, rem = divmod(total, n_sources)
    return [base + (1 if i < rem else 0) for i in range(n_sources)]


def sample_reasoning_prompts(
    total: int = 200,
    *,
    seed: int = 42,
    sources: tuple[str, ...] = REASONING_COMPONENT_SOURCES,
) -> list[dict]:
    """Stratified random sample from antidoom-mix reasoning component sources only."""
    from datasets import load_dataset

    ds = load_dataset("LiquidAI/antidoom-mix-v1.0", split="train")
    by_source: dict[str, list[int]] = {s: [] for s in sources}
    for i in range(len(ds)):
        comp = component_source_from_row(ds[i])
        if comp in by_source:
            by_source[comp].append(i)

    rng = random.Random(seed)
    counts = _per_source_counts(total, len(sources))
    out: list[dict] = []
    for src, n_take in zip(sources, counts):
        pool = by_source[src]
        if len(pool) < n_take:
            raise RuntimeError(
                f"Not enough rows for {src}: need {n_take}, have {len(pool)}"
            )
        chosen = rng.sample(pool, n_take)
        for row_idx in sorted(chosen):
            item = _row_to_prompt(row_idx, ds[row_idx])
            if item is None:
                raise RuntimeError(f"Unparseable prompt at row {row_idx} ({src})")
            out.append(item)

    rng.shuffle(out)
    logger.info(
        "Sampled %d prompts from %d reasoning sources (seed=%d): %s",
        len(out),
        len(sources),
        seed,
        {s: c for s, c in zip(sources, counts)},
    )
    return out


def save_prompt_sample(
    prompts: list[dict],
    path: Path | str,
    *,
    seed: int = 42,
    total: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, int] = {}
    for p in prompts:
        by_source[p["source"]] = by_source.get(p["source"], 0) + 1
    payload = {
        "dataset": "LiquidAI/antidoom-mix-v1.0",
        "seed": seed,
        "total_requested": total or len(prompts),
        "total_sampled": len(prompts),
        "sources": list(REASONING_COMPONENT_SOURCES),
        "per_source_counts": by_source,
        "prompts": [
            {
                "prompt_id": p["prompt_id"],
                "row_index": p["row_index"],
                "id": p["id"],
                "source": p["source"],
                "hf_source": p.get("hf_source"),
            }
            for p in prompts
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote prompt sample audit file: %s", path)


def load_prompt_sample(path: Path | str) -> list[dict]:
    """Reload full prompt texts from saved sample IDs."""
    from datasets import load_dataset

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt sample file: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    ds = load_dataset("LiquidAI/antidoom-mix-v1.0", split="train")
    out: list[dict] = []
    for entry in meta["prompts"]:
        row_idx = entry["row_index"]
        item = _row_to_prompt(row_idx, ds[row_idx])
        if item is None:
            raise RuntimeError(f"Could not reload prompt row {row_idx}")
        out.append(item)
    logger.info("Reloaded %d prompts from %s", len(out), path)
    return out


def get_or_create_prompt_sample(
    path: Path | str = DEFAULT_SAMPLE_PATH,
    *,
    total: int = 200,
    seed: int = 42,
) -> list[dict]:
    path = Path(path)
    if path.is_file():
        return load_prompt_sample(path)
    prompts = sample_reasoning_prompts(total=total, seed=seed)
    save_prompt_sample(prompts, path, seed=seed, total=total)
    return prompts
