#!/usr/bin/env python
"""Merge empirical trigger tokens from all generation logs into one CSV."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def collect_loops() -> tuple[Counter, dict[int, str]]:
    counter: Counter = Counter()
    examples: dict[int, str] = {}

    # Trigger extraction log
    for path in [
        ROOT / "results" / "trigger_gen_log.jsonl",
        ROOT / "results" / "trigger_gen_log_partial_easy.jsonl",
    ]:
        if not path.is_file():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("is_loop"):
                continue
            tid = row.get("trigger_token_id")
            if tid is None:
                continue
            counter[int(tid)] += 1
            examples.setdefault(int(tid), row.get("trigger_decoded") or "")

    # Exp2 looping traces
    loop_dir = ROOT / "results" / "exp2" / "looping"
    if loop_dir.is_dir():
        for meta_path in loop_dir.glob("*/meta.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta.get("is_loop"):
                continue
            tid = meta.get("trigger_token_id")
            if tid is None:
                continue
            counter[int(tid)] += 1
            examples.setdefault(int(tid), meta.get("trigger_decoded") or "")

    return counter, examples


def main() -> int:
    from jspace.loading import load_stack

    counter, examples = collect_loops()
    stack = load_stack()
    total = sum(counter.values())
    rows = []
    for tid, count in counter.most_common():
        dec = examples.get(tid) or stack.tokenizer.decode([tid])
        share = count / total if total else 0.0
        rows.append(
            {
                "count": count,
                "share": round(share, 6),
                "token": repr(dec),
                "token_id": tid,
                "decoded": dec,
            }
        )

    out = ROOT / "results" / "trigger_tokens_qwen3.5-4b.csv"
    df = pd.DataFrame(rows, columns=["count", "share", "token", "token_id", "decoded"])
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} empirical triggers from {total} loop events")
    if df.empty:
        print("NOTE: empirical table still empty — Exp1/3 use operational RESTART_WORDS seed.")
    else:
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
