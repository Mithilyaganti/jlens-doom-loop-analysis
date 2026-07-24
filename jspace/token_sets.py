"""Token set construction for Experiment 1."""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from jspace.detection import RESTART_WORDS

logger = logging.getLogger(__name__)


def tokenize_word_variants(tokenizer: Any, word: str) -> list[tuple[int, str]]:
    """Return (token_id, decoded) for leading-space and bare forms of word."""
    variants = []
    for form in (f" {word}", word, f" {word.capitalize()}", word.capitalize()):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            tid = ids[0]
            raw = tokenizer.convert_ids_to_tokens(tid)
            if isinstance(raw, list):
                raw = raw[0]
            variants.append((tid, form))
        elif len(ids) >= 1:
            # multi-token: take last piece as approximation for rare words
            tid = ids[-1]
            variants.append((tid, form + f" (multi→last:{ids})"))
    # dedupe by id
    seen = set()
    out = []
    for tid, form in variants:
        if tid not in seen:
            seen.add(tid)
            out.append((tid, form))
    return out


def build_restart_word_token_ids(tokenizer: Any) -> list[dict[str, Any]]:
    rows = []
    for w in sorted(RESTART_WORDS):
        for tid, form in tokenize_word_variants(tokenizer, w):
            rows.append(
                {
                    "token_id": tid,
                    "word": w,
                    "form": form,
                    "decoded": tokenizer.decode([tid]),
                    "source": "restart_words",
                }
            )
    return rows


def load_trigger_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_trigger_set(
    tokenizer: Any,
    trigger_csv: Path | None,
    *,
    top_n: int = 30,
    target_size: int = 40,
) -> list[dict[str, Any]]:
    """Top-N from empirical table + RESTART_WORDS, deduped by token_id."""
    by_id: dict[int, dict[str, Any]] = {}

    if trigger_csv and Path(trigger_csv).is_file():
        df = load_trigger_table(trigger_csv)
        for _, row in df.head(top_n).iterrows():
            tid = int(row["token_id"])
            by_id[tid] = {
                "token_id": tid,
                "decoded": str(row.get("token", row.get("decoded", ""))),
                "count": int(row.get("count", 0)),
                "source": "empirical",
            }

    for r in build_restart_word_token_ids(tokenizer):
        tid = r["token_id"]
        if tid not in by_id:
            by_id[tid] = {
                "token_id": tid,
                "decoded": r["decoded"],
                "count": 0,
                "source": "restart_words",
            }

    items = list(by_id.values())
    # Prefer empirical first
    items.sort(key=lambda x: (0 if x["source"] == "empirical" else 1, -x.get("count", 0)))
    return items[: max(target_size, len(items))]


def estimate_unigram_from_wikitext(tokenizer: Any, n_docs: int = 500) -> Counter:
    """Approximate unigram counts from wikitext for frequency matching."""
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    except Exception as e:
        logger.warning("Could not load wikitext (%s); using empty counts", e)
        return Counter()

    counts: Counter = Counter()
    for i, row in enumerate(ds):
        if i >= n_docs:
            break
        text = row.get("text") or ""
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        counts.update(ids[:512])
    return counts


def build_frequency_matched_content(
    tokenizer: Any,
    trigger_ids: Sequence[int],
    unigram: Counter,
    *,
    n: int = 30,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Content-ish tokens with similar frequency to triggers (±20%)."""
    rng = random.Random(seed)
    trigger_freqs = [unigram.get(t, 1) for t in trigger_ids]
    if not trigger_freqs:
        trigger_freqs = [10]
    mean_f = sum(trigger_freqs) / len(trigger_freqs)
    lo, hi = mean_f * 0.8, mean_f * 1.2

    # Exclude triggers
    exclude = set(trigger_ids)
    candidates = []
    for tid, c in unigram.items():
        if tid in exclude:
            continue
        if not (lo <= c <= hi or mean_f == 0):
            # also allow broader band if too few
            continue
        tok = tokenizer.decode([tid])
        if not any(ch.isalpha() for ch in tok):
            continue
        # Prefer content-looking: longer, not pure function words
        stripped = tok.strip().lower()
        if stripped in RESTART_WORDS:
            continue
        if len(stripped) < 3:
            continue
        candidates.append((tid, c, tok))

    if len(candidates) < n:
        # relax frequency band
        for tid, c in unigram.most_common(5000):
            if tid in exclude:
                continue
            tok = tokenizer.decode([tid])
            stripped = tok.strip().lower()
            if len(stripped) < 4 or not stripped.isalpha():
                continue
            if stripped in RESTART_WORDS:
                continue
            candidates.append((tid, c, tok))
            if len(candidates) >= n * 5:
                break

    rng.shuffle(candidates)
    out = []
    seen = set()
    for tid, c, tok in candidates:
        if tid in seen:
            continue
        seen.add(tid)
        out.append({"token_id": tid, "decoded": tok, "count": c, "source": "freq_content"})
        if len(out) >= n:
            break
    return out


def build_discourse_controls(
    tokenizer: Any,
    trigger_ids: Sequence[int],
    *,
    n: int = 30,
) -> list[dict[str, Any]]:
    """High-frequency discourse markers NOT in the trigger set."""
    markers = [
        "and",
        "or",
        "when",
        "if",
        "I",
        "you",
        "we",
        "they",
        "he",
        "she",
        "it",
        "a",
        "an",
        "of",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "as",
        "that",
        "which",
        "who",
        "what",
        "not",
        "no",
        "yes",
        "just",
        "only",
        "also",  # may be trigger — filtered
        "very",
        "really",
        "quite",
        "about",
        "into",
        "than",
        "then",
        "there",
        "here",
        "some",
        "any",
        "all",
        "each",
        "other",
        "more",
        "most",
        "such",
        "can",
        "will",
        "would",
        "could",
        "should",
    ]
    exclude = set(trigger_ids)
    out = []
    seen = set()
    for w in markers:
        for tid, form in tokenize_word_variants(tokenizer, w):
            if tid in exclude or tid in seen:
                continue
            seen.add(tid)
            out.append(
                {
                    "token_id": tid,
                    "decoded": tokenizer.decode([tid]),
                    "word": w,
                    "source": "discourse_control",
                }
            )
            if len(out) >= n:
                return out
    return out


def build_random_tokens(
    tokenizer: Any,
    *,
    n: int = 30,
    seed: int = 42,
    exclude: set[int] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    exclude = exclude or set()
    vocab = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer)
    out = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        tid = rng.randrange(0, vocab)
        if tid in exclude or tid in seen:
            continue
        tok = tokenizer.decode([tid])
        stripped = "".join(ch for ch in tok if ch.isalpha())
        if len(stripped) < 3:
            continue
        if not any(ch.isalpha() for ch in tok):
            continue
        seen.add(tid)
        out.append({"token_id": tid, "decoded": tok, "source": "random"})
    return out


def build_all_token_sets(
    tokenizer: Any,
    trigger_csv: Path | None,
    *,
    size: int = 30,
) -> dict[str, list[dict[str, Any]]]:
    triggers = build_trigger_set(tokenizer, trigger_csv, top_n=size, target_size=size)
    t_ids = [t["token_id"] for t in triggers]
    unigram = estimate_unigram_from_wikitext(tokenizer)
    content = build_frequency_matched_content(tokenizer, t_ids, unigram, n=size)
    discourse = build_discourse_controls(tokenizer, set(t_ids) | {c["token_id"] for c in content}, n=size)
    exclude = set(t_ids) | {c["token_id"] for c in content} | {d["token_id"] for d in discourse}
    randoms = build_random_tokens(tokenizer, n=size, exclude=exclude)
    return {
        "trigger": triggers[:size],
        "freq_content": content[:size],
        "discourse": discourse[:size],
        "random": randoms[:size],
    }
