"""Reproduce the paper-1 Telugu offset bug on MGSM.

Uses the unmodified ``jspace.detection.TokenState`` and the Qwen3.5-4B
tokenizer. Exits 0 when the tokenizer or the MGSM files are missing, so a
CPU checkout without that download still has the hermetic unit test.

    python -m loop_lingua.reproduce_bug
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loop_lingua.bugreport import report_encoded_texts
from loop_lingua.layout import CORPUS_DIR, MGSM_DIR
from loop_lingua.prompts import load_mgsm_tsvs

TOKENIZER_REPO = "Qwen/Qwen3.5-4B"
TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def tokenizer_json_path() -> Path | None:
    override = os.environ.get("LOOP_LINGUA_TOKENIZER_JSON")
    if override:
        path = Path(override)
        return path if path.is_file() else None
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots"
    if not root.is_dir():
        return None
    pinned = root / TOKENIZER_REVISION / "tokenizer.json"
    if pinned.is_file():
        return pinned
    for snapshot in sorted(root.iterdir()):
        candidate = snapshot / "tokenizer.json"
        if candidate.is_file():
            return candidate
    return None


def _load_questions() -> dict[str, list[str]]:
    paths = {language: MGSM_DIR / f"mgsm_{language}.tsv" for language in ("en", "te")}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"MGSM TSVs are not in {MGSM_DIR}")
    items = load_mgsm_tsvs(paths)
    return {
        "en": [item.questions["en"] for item in items],
        "te": [item.questions["te"] for item in items],
    }


def run() -> dict[str, object]:
    path = tokenizer_json_path()
    if path is None:
        raise FileNotFoundError(
            "Qwen/Qwen3.5-4B tokenizer.json is not cached. "
            "Set LOOP_LINGUA_TOKENIZER_JSON or download the tokenizer file only."
        )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    questions = _load_questions()

    def encode_tokens(text: str) -> list[str]:
        return list(tokenizer.encode(text).tokens)

    reports = {}
    for language, texts in questions.items():
        report = report_encoded_texts(texts, encode_tokens)
        reports[language] = {
            "n_texts": report.n_texts,
            "n_length_mismatches": report.n_length_mismatches,
            "n_offset_mismatches": report.n_offset_mismatches,
        }
    return {
        "tokenizer": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "dataset": "MGSM",
        "definition": (
            "A text mismatches when jspace.detection.TokenState's per-token "
            "decode is not the tokenizer's full-sequence decode."
        ),
        "languages": reports,
    }


def main() -> int:
    try:
        payload = run()
    except FileNotFoundError as exc:
        print(exc)
        return 2
    destination = CORPUS_DIR / "offset_bug_mgsm.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    languages = payload["languages"]
    assert isinstance(languages, dict)
    for language, stats in languages.items():
        assert isinstance(stats, dict)
        print(
            f"{language}: {stats['n_length_mismatches']}/{stats['n_texts']} "
            "texts mis-reconstructed by jspace TokenState"
        )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
