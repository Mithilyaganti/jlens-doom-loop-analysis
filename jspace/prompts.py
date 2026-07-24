"""Real prompt loading from LiquidAI/antidoom-mix-v1.0 (ShareGPT-style schema)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
            # Unknown role but has text — keep as candidate only if nothing else
            last_human = text
    return last_human


# Source substrings that Liquid's mix treats as harder (loops more often).
# Used only to *prefer* order, not to invent prompts.
_HARD_SOURCE_HINTS = (
    "math",
    "gsm",
    "code",
    "humaneval",
    "mbpp",
    "livecode",
    "competition",
    "olympiad",
    "aime",
    "aops",
    "theorem",
    "proof",
    "leetcode",
    "reasoning",
    "think",
)


def _source_hardness(source: str | None) -> int:
    s = (source or "").lower()
    return 1 if any(h in s for h in _HARD_SOURCE_HINTS) else 0


def load_antidoom_mix(max_prompts: int, *, prefer_hard: bool = True) -> list[dict]:
    """Load prompts from LiquidAI/antidoom-mix-v1.0. Raises if empty.

    When prefer_hard=True, scan a larger window and fill the budget preferring
    math/code/reasoning sources first (still real antidoom-mix rows only).
    """
    from datasets import load_dataset

    ds = load_dataset("LiquidAI/antidoom-mix-v1.0", split="train")
    # Scan more rows than needed so we can prefer hard sources without inventing.
    scan_n = min(len(ds), max(max_prompts * 40, 5000))
    hard: list[dict] = []
    soft: list[dict] = []
    for i in range(scan_n):
        row = ds[i]
        text = extract_user_text_from_conversations(row.get("conversations"))
        if text is None:
            for key in ("prompt", "question", "text", "input"):
                if row.get(key):
                    text = str(row[key]).strip()
                    break
        if not text:
            continue
        item = {
            "prompt_id": i,
            "id": row.get("id"),
            "text": text[:4000],
            "source": row.get("source") or "antidoom-mix",
            "dataset": "LiquidAI/antidoom-mix-v1.0",
        }
        if prefer_hard and _source_hardness(item["source"]):
            hard.append(item)
        else:
            soft.append(item)
        if prefer_hard and len(hard) >= max_prompts:
            break
        if not prefer_hard and len(hard) + len(soft) >= max_prompts:
            break

    if prefer_hard:
        prompts = hard[:max_prompts]
        if len(prompts) < max_prompts:
            prompts.extend(soft[: max_prompts - len(prompts)])
    else:
        prompts = (hard + soft)[:max_prompts]

    if not prompts:
        raise RuntimeError(
            "LiquidAI/antidoom-mix-v1.0 loaded but yielded zero parseable prompts. "
            f"columns={ds.column_names}. Refusing to invent substitutes."
        )
    n_hard = sum(1 for p in prompts if _source_hardness(p["source"]))
    logger.info(
        "Loaded %d prompts from antidoom-mix-v1.0 (hard-preferred=%s, hard_count=%d, scanned=%d)",
        len(prompts),
        prefer_hard,
        n_hard,
        scan_n,
    )
    return prompts


HARD_MATH_PROMPTS = [
    "Prove that the square root of 2 is irrational, writing a full step-by-step proof.",
    "Solve: A tank fills in 6 hours and empties in 8. How long to fill if both open? Show all steps.",
    "Find all positive integers n such that n^2 + n + 1 divides n^3 - 1. Explain carefully.",
    "Compute the last two digits of 7^100 using modular arithmetic. Show every step.",
    "A fair coin is flipped until two consecutive heads appear. Expected number of flips? Derive fully.",
    "Prove that there are infinitely many primes. Write a detailed proof.",
    "Solve the system: x+y+z=6, 2x-y+z=3, x+2y-z=2. Show all steps.",
    "Explain the Fourier transform of a Gaussian step by step with derivations.",
    "A frog climbs 3m by day and slips 2m by night from a 10m well. When does it escape? Reason carefully.",
    "Derive the quadratic formula from ax^2+bx+c=0 by completing the square. Every step.",
]

HARD_CODE_PROMPTS = [
    "Write a Python function that finds the longest palindromic substring. Trace an example by hand first.",
    "Implement Dijkstra's algorithm and walk through it on a 6-node graph step by step.",
    "Explain and correct this buggy binary search, then prove correctness of the fixed version.",
    "Write a recursive solution to the N-Queens problem and analyze its complexity carefully.",
    "Design a thread-safe LRU cache and reason about race conditions at each step.",
    "Write a correct implementation of quicksort and prove its average complexity.",
    "Implement topological sort on a DAG and prove it terminates. Show a worked example.",
    "Write merge sort and derive its recurrence T(n)=2T(n/2)+O(n) solution step by step.",
    "Implement a red-black tree insert and walk through rotations on an example sequence.",
    "Write a correct union-find with path compression; analyze amortized complexity carefully.",
]


def with_hard_extras(base: list[dict], *, id_offset: int = 200000) -> list[dict]:
    """Append labeled hard math/coding prompts (PROJECT_SPEC §10.2)."""
    out = list(base)
    for j, t in enumerate(HARD_MATH_PROMPTS):
        out.append(
            {
                "prompt_id": id_offset + j,
                "text": t,
                "source": "hard_math",
                "dataset": "project_spec_section_10.2",
            }
        )
    for j, t in enumerate(HARD_CODE_PROMPTS):
        out.append(
            {
                "prompt_id": id_offset + 1000 + j,
                "text": t,
                "source": "hard_code",
                "dataset": "project_spec_section_10.2",
            }
        )
    return out
