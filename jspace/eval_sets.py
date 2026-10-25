"""Eval sets for the Exp3 surgical-competence check (GSM8K).

Replaces the old 5-question hand-written arithmetic eval with a fixed 50-problem
GSM8K test sample (``data/gsm8k_eval.json``), created once by sampling the HF
``openai/gsm8k`` test split with seed=42 (see :func:`sample_gsm8k_eval`).

Scoring is exact-match on the LAST number in the model output vs the gold final
numeric answer (GSM8K "#### <n>" field). Pure stdlib on purpose so CPU-only
unit tests stay light.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
GSM8K_EVAL_PATH = ROOT / "data" / "gsm8k_eval.json"

# Last number in the text: optional sign, digits with optional thousands
# separators ("1,234"), optional decimal part. Trailing punctuation is
# naturally excluded by the pattern itself.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_last_number(text: str) -> str | None:
    """Return the last number in ``text`` as a comma-stripped string, or None.

    Edge cases covered (and unit-tested in tests/test_intervention.py):
      "The answer is 42."  -> "42"
      "#### 42"            -> "42"
      "1,234"              -> "1234"
      "no numbers here!"   -> None
      ""                   -> None
    """
    if not text:
        return None
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "").lstrip("+")


def canonical_number(value: str | int | float) -> str | None:
    """Canonical string form used for gold-vs-prediction comparison.

    Strips commas/whitespace/plus-signs; integral floats collapse to ints
    ("42.0" -> "42"); anything non-numeric maps to None.
    """
    if value is None:
        return None
    s = str(value).strip().replace(",", "").replace(" ", "").lstrip("+")
    if not s or s in ("-", "."):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if f == int(f):
        return str(int(f))
    return repr(f)


def score_gsm8k_output(output_text: str, gold_answer: str) -> bool:
    """Exact match: last number in the output == gold final numeric answer.

    Counts as wrong when the output contains no number at all.
    """
    pred = extract_last_number(output_text)
    if pred is None:
        return False
    return canonical_number(pred) == canonical_number(gold_answer)


def load_gsm8k_eval(path: Path | str | None = None, *, n: int | None = None) -> list[dict]:
    """Load the fixed GSM8K eval set: [{"question": str, "answer": str}, ...].

    ``answer`` is the final numeric answer string (commas stripped). ``n``
    truncates the set (e.g. JLENS_GSM8K_N smoke runs); None loads everything.
    """
    path = Path(path) if path is not None else GSM8K_EVAL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Create it once with "
            "jspace.eval_sets.sample_gsm8k_eval(50, seed=42) "
            "(requires the `datasets` package and HF access)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    problems = data["problems"] if isinstance(data, dict) else data
    out = []
    for row in problems:
        q = str(row.get("question", "")).strip()
        a = str(row.get("answer", "")).strip()
        if not q or not a:
            raise ValueError(f"Malformed GSM8K eval row in {path}: {row!r}")
        out.append({"question": q, "answer": a.replace(",", "")})
    if n is not None:
        out = out[: max(0, int(n))]
    logger.info("Loaded %d GSM8K eval problems from %s", len(out), path)
    return out


def gsm8k_final_answer(raw_answer: str) -> str:
    """Extract the final numeric answer from a raw GSM8K answer field.

    GSM8K answers look like "<reasoning> ... #### 1,234"; the gold is the
    substring after the last '####' with commas/space stripped.
    """
    if "####" in raw_answer:
        raw_answer = raw_answer.rsplit("####", 1)[1]
    return raw_answer.replace(",", "").strip()


def sample_gsm8k_eval(
    n: int = 50,
    *,
    seed: int = 42,
    path: Path | str | None = None,
) -> Path:
    """Create ``data/gsm8k_eval.json`` by sampling the HF gsm8k TEST split.

    Deterministic given ``seed``: dataset.shuffle(seed=seed).select(range(n)).
    Runs the download exactly once locally; the committed JSON is the
    source of truth afterwards (no network needed at experiment time).
    """
    path = Path(path) if path is not None else GSM8K_EVAL_PATH
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)
    n = min(int(n), len(ds))
    problems = []
    for row in ds.select(range(n)):
        gold = gsm8k_final_answer(str(row["answer"]))
        if not gold:
            raise ValueError(f"GSM8K row without final answer: {row!r}")
        problems.append({"question": str(row["question"]).strip(), "answer": gold})
    payload = {
        "source": "openai/gsm8k",
        "split": "test",
        "seed": int(seed),
        "n": len(problems),
        "note": (
            "Fixed 50-problem surgical-competence eval for Exp3. Sampled once "
            "with datasets.shuffle(seed=42); do not regenerate for analysis."
        ),
        "problems": problems,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d GSM8K problems to %s", len(problems), path)
    return path


def gsm8k_n_from_env(default: int = 50) -> int:
    """JLENS_GSM8K_N knob (default 50)."""
    try:
        return int(os.environ.get("JLENS_GSM8K_N", str(default)))
    except ValueError:
        return default
