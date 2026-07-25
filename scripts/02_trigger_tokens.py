#!/usr/bin/env python
"""Generate Qwen3.5-4B doom-loop trigger token table (publishable artifact).

Runs near-greedy generation on antidoom-mix prompts, detects loops with
Liquid's character-based detector, extracts first token of first repeat.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("trigger_tokens")

# Default scale for laptop: enough loops for a stable top-30 table.
# Override with env JLENS_MAX_PROMPTS / JLENS_MAX_NEW_TOKENS.
import os

MAX_PROMPTS = int(os.environ.get("JLENS_MAX_PROMPTS", "120"))
# Liquid default.yaml uses max_new_tokens=4000; 2048 is the laptop-feasible floor
# that still produces loops on hard math/code. Override with JLENS_MAX_NEW_TOKENS.
MAX_NEW_TOKENS = int(os.environ.get("JLENS_MAX_NEW_TOKENS", "4000"))
TEMPERATURE = float(os.environ.get("JLENS_TEMPERATURE", "0.01"))


def load_prompts(max_prompts: int) -> list[dict]:
    """Load stratified antidoom-mix reasoning subset (no external prompts)."""
    from jspace.prompts import get_or_create_prompt_sample

    return get_or_create_prompt_sample(total=max_prompts)


def main() -> int:
    from jspace.loading import load_stack, clear_cuda
    from jspace.generation import apply_chat_template, generate_greedy
    from jspace.detection import detect_loop
    from jspace.analysis import write_status, append_results_summary, ensure_dir, save_json
    import pandas as pd

    out = ensure_dir(ROOT / "results")
    summary_path = out / "baseline_pass_summary.json"
    if summary_path.is_file():
        logger.info(
            "Baseline pass already complete — rebuilding status from %s", summary_path
        )
        stats = json.loads(summary_path.read_text(encoding="utf-8"))
        status = f"""# Trigger Token Extraction — Qwen3.5-4B (from baseline pass)

- Prompts: {stats.get('n_prompts')}
- Loops detected: {stats.get('n_loop')} ({stats.get('loop_rate', 0):.1%})
- Table: `results/trigger_tokens_qwen3.5-4b.csv`
- Source: unified baseline pass (`scripts/02_baseline_pass.py`)
"""
        write_status("02_trigger_tokens", status)
        append_results_summary("Trigger Tokens (Qwen3.5-4B)", status)
        return 0 if stats.get("n_loop", 0) > 0 else 2

    generations_path = out / "trigger_gen_log.jsonl"
    stack = load_stack()
    prompts = load_prompts(MAX_PROMPTS)
    logger.info("Running trigger extraction on %d prompts (max_new=%d)", len(prompts), MAX_NEW_TOKENS)

    trigger_counter: Counter = Counter()
    trigger_examples: dict[int, str] = {}
    n_loop = 0
    n_attempt = 0

    with generations_path.open("w", encoding="utf-8") as logf:
        for i, p in enumerate(prompts):
            n_attempt += 1
            chat = apply_chat_template(stack.tokenizer, p["text"])
            try:
                gen = generate_greedy(
                    stack, chat, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
                )
            except Exception as e:
                logger.exception("gen failed prompt %s: %s", p["prompt_id"], e)
                clear_cuda()
                continue

            loop = detect_loop(
                gen["generated_text"],
                token_ids=gen["generated_ids"],
                tokenizer=stack.tokenizer,
            )
            row = {
                "prompt_id": p["prompt_id"],
                "source": p["source"],
                "is_loop": loop.is_loop,
                "trigger_token_id": loop.trigger_token_id,
                "trigger_decoded": loop.trigger_decoded,
                "trigger_token_str": loop.trigger_token_str,
                "n_tokens": len(gen["generated_ids"]),
                "snippet": loop.hit.snippet if loop.hit else None,
                "generated_preview": gen["generated_text"][:500],
            }
            logf.write(json.dumps(row, ensure_ascii=False) + "\n")
            logf.flush()

            if loop.is_loop:
                n_loop += 1
                if loop.trigger_token_id is not None:
                    trigger_counter[loop.trigger_token_id] += 1
                    trigger_examples.setdefault(
                        loop.trigger_token_id, loop.trigger_decoded or ""
                    )
            if (i + 1) % 5 == 0:
                logger.info(
                    "progress %d/%d loops=%d unique_triggers=%d",
                    i + 1,
                    len(prompts),
                    n_loop,
                    len(trigger_counter),
                )
            clear_cuda()

    total_triggers = sum(trigger_counter.values())
    rows = []
    for tid, count in trigger_counter.most_common():
        dec = trigger_examples.get(tid) or stack.tokenizer.decode([tid])
        share = count / total_triggers if total_triggers else 0.0
        rows.append(
            {
                "count": count,
                "share": round(share, 6),
                "token": repr(dec),
                "token_id": tid,
                "decoded": dec,
            }
        )
    df = pd.DataFrame(rows)
    csv_path = out / "trigger_tokens_qwen3.5-4b.csv"
    df.to_csv(csv_path, index=False)

    # Honest empty empirical table if no loops — never invent frequencies.
    if df.empty:
        df = pd.DataFrame(columns=["count", "share", "token", "token_id", "decoded"])
        df.to_csv(csv_path, index=False)
        logger.error(
            "No loops observed in %d attempts — empirical trigger table is empty. "
            "Increase JLENS_MAX_PROMPTS / JLENS_MAX_NEW_TOKENS. "
            "Do not treat _RESTART_WORDS as empirical Qwen triggers.",
            n_attempt,
        )

    rate = n_loop / n_attempt if n_attempt else 0.0
    status = f"""# Trigger Token Extraction — Qwen3.5-4B

- Prompts attempted: {n_attempt}
- Loops detected: {n_loop} ({rate:.1%})
- Unique trigger token IDs: {len(trigger_counter)}
- Max new tokens: {MAX_NEW_TOKENS}, temperature: {TEMPERATURE}
- Table: `results/trigger_tokens_qwen3.5-4b.csv` (empirical only; empty if zero loops)
- Generation log: `results/trigger_gen_log.jsonl`

Top-10 triggers:
```
{df.head(10).to_string(index=False) if len(df) else '(empty — no loops observed)'}
```

This table is the **empirical** Qwen3.5-4B trigger distribution under Liquid's
character-based loop detector (min_repeats=4, min_total_repeated=60 chars).
No frequencies were invented.
"""
    write_status("02_trigger_tokens", status)
    append_results_summary("Trigger Tokens (Qwen3.5-4B)", status)
    save_json(
        {
            "n_attempt": n_attempt,
            "n_loop": n_loop,
            "loop_rate": rate,
            "unique_triggers": len(trigger_counter),
            "empirical_table_empty": bool(df.empty),
        },
        out / "trigger_extraction_stats.json",
    )
    clear_cuda()
    # Non-zero exit if we got zero loops: pipeline can continue Exp1 with
    # operational RESTART_WORDS seed, but the empirical artifact is honestly empty.
    return 0 if n_loop > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
