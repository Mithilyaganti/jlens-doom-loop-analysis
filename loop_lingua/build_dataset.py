"""Write the matched prompt corpus from local parquet and MGSM TSVs.

Run from the repo root:

    python -m loop_lingua.build_dataset

Downloads are limited to the MMLU-ProX-Lite test shards (and MGSM TSVs if
the copies in ``loop_lingua/corpus/mgsm`` are missing). No model is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from loop_lingua.layout import CORPUS_DIR, MGSM_DIR
from loop_lingua.licenses import LICENSES
from loop_lingua.pilot import PILOT_LANGUAGES, jobs_from_records
from loop_lingua.prompts import (
    DEFAULT_N,
    DEFAULT_SEED,
    MATCHED_LANGUAGES,
    SOURCE_LANGUAGES,
    ParallelItem,
    build_matched_set,
    load_mgsm_tsvs,
    load_mmlu_prox_lite_parquets,
    pilot_item_ids,
    render_record,
    write_jsonl,
)

MMLU_REPO = "li-lab/MMLU-ProX-Lite"
MGSM_RAW = "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm"


def _mmlu_test_paths() -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    paths: dict[str, Path] = {}
    for language in SOURCE_LANGUAGES:
        downloaded = hf_hub_download(
            MMLU_REPO,
            f"{language}/test-00000-of-00001.parquet",
            repo_type="dataset",
        )
        paths[language] = Path(downloaded)
    return paths


def _ensure_mgsm() -> dict[str, Path]:
    MGSM_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for language in ("en", "te"):
        destination = MGSM_DIR / f"mgsm_{language}.tsv"
        if not destination.exists():
            import urllib.request

            urllib.request.urlretrieve(f"{MGSM_RAW}/mgsm_{language}.tsv", destination)
        paths[language] = destination
    return paths


def _pool_row(item: ParallelItem) -> dict[str, object]:
    eligible = item.eligible()
    return {
        "item_id": eligible.item_id,
        "source": eligible.source,
        "category": eligible.category,
        "languages": sorted(eligible.languages),
    }


def build(corpus_dir: Path | None = None) -> Path:
    root = corpus_dir or CORPUS_DIR
    root.mkdir(parents=True, exist_ok=True)
    mmlu_items, gold = load_mmlu_prox_lite_parquets(_mmlu_test_paths())
    mgsm_items = load_mgsm_tsvs(_ensure_mgsm())
    # Pool order is part of the dropped-source count, not of the sample.
    # Selection sorts inside each category, so concatenation order does not
    # pick winners. MGSM still has to be in the pool so the drop is visible.
    pooled = list(mmlu_items) + list(mgsm_items)
    built = build_matched_set(pooled, n=DEFAULT_N, seed=DEFAULT_SEED)
    write_jsonl(root / "prompts_matched.jsonl", (row.as_dict() for row in built.records))
    write_jsonl(
        root / "gold.jsonl",
        ({"item_id": item_id, "answer": gold[item_id], "source": "mmlu-prox-lite"} for item_id in built.item_ids),
    )
    write_jsonl(root / "pool_index.jsonl", (_pool_row(item) for item in pooled))
    categories = {item_id: next(row.category for row in built.records if row.item_id == item_id) for item_id in built.item_ids}
    pilot_ids = pilot_item_ids(built.item_ids, categories)
    pilot_jobs = jobs_from_records([row.as_dict() for row in built.records], pilot_ids, languages=PILOT_LANGUAGES)
    write_jsonl(
        root / "pilot_96.jsonl",
        (
            {"item_id": job.item_id, "language": job.language, "prompt": job.prompt}
            for job in pilot_jobs
        ),
    )
    slots = {
        "status": "placeholder",
        "n_slots": len(built.tenglish_slots),
        "hand_edited": 0,
        "note": (
            "Hand-edit code-mixed Telugu-English (Tenglish) prompts. "
            "Do not fill these with machine translations or automatic code-mixing. "
            "Each slot points at a matched item so the edit can stay parallel."
        ),
        "slots": [slot.as_dict() for slot in built.tenglish_slots],
    }
    (root / "tenglish_slots.json").write_text(json.dumps(slots, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    supplement = []
    for item in mgsm_items:
        for language in ("en", "te", "te_roman"):
            supplement.append(render_record(item, language).as_dict())
    write_jsonl(root / "mgsm_supplement.jsonl", supplement)
    manifest = {
        "n_per_language": DEFAULT_N,
        "seed": DEFAULT_SEED,
        "languages": list(MATCHED_LANGUAGES),
        "selection_fields": ["item_id", "category", "languages"],
        "required_languages": list(SOURCE_LANGUAGES),
        "item_ids": built.item_ids,
        "category_allocation": built.category_allocation,
        "dropped_sources": built.dropped_sources,
        "pilot_ids": pilot_ids,
        "pilot_generations": len(pilot_jobs),
        "tenglish_slots": len(built.tenglish_slots),
        "held_out": "MMLU-ProX-Lite validation (question ids 0-69) is not in this pool.",
        "licenses": {key: record.as_dict() for key, record in LICENSES.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root


def main() -> None:
    path = build()
    print(f"wrote corpus to {path}")


if __name__ == "__main__":
    main()
