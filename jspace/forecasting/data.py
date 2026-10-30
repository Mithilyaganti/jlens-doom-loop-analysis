"""Load baseline-pass traces and the summary files that describe them.

``scripts/02_baseline_pass.py`` writes one directory per prompt:

    results/exp2_{slug}/looping/p{prompt_id}/meta.json
    results/exp2_{slug}/looping/p{prompt_id}/tokens.pt
    results/exp2_{slug}/nonlooping/p{prompt_id}/...

``meta.json`` carries ``generated_text`` and the stored loop fields.
``tokens.pt`` carries ``generated_ids``. The legacy tree ``results/exp2/`` is
read only when the slug-scoped tree has no traces. The jsonl generation log
stores a 500-character preview, not a trace, and is counted but not labelled.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from jspace.forecasting.metrics import assign_split

ROOT = Path(__file__).resolve().parents[2]


def display_path(path: Path) -> str:
    """Path relative to the repo root when it lives inside the checkout."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass
class TraceRecord:
    prompt_id: int
    generated_text: str | None
    token_ids: list[int] | None
    token_strings: list[str] | None
    source: str | None
    meta: dict
    text_complete: bool
    origin: str
    warnings: list[str] = field(default_factory=list)


def _has_traces(root: Path) -> bool:
    for kind in ("looping", "nonlooping"):
        base = root / kind
        if base.is_dir() and any((p / "meta.json").is_file() for p in base.glob("p*")):
            return True
    return False


def _read_tokens_pt(path: Path) -> tuple[list[int] | None, str | None]:
    try:
        import torch
    except ImportError:
        return None, "torch is not installed; tokens.pt was not read"
    try:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return None, f"failed to read tokens.pt: {type(exc).__name__}: {exc}"
    if not isinstance(obj, dict) or obj.get("generated_ids") is None:
        return None, "tokens.pt has no generated_ids"
    ids = obj["generated_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(x) for x in ids], None


def load_trace_dir(trace_dir: Path) -> TraceRecord | None:
    meta_path = trace_dir / "meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pid = meta.get("prompt_id")
    if pid is None:
        try:
            pid = int(trace_dir.name[1:])
        except ValueError:
            return None
    text = meta.get("generated_text")
    text_complete = isinstance(text, str) and len(text) > 0
    warnings: list[str] = []
    token_ids = None
    tok_path = trace_dir / "tokens.pt"
    if tok_path.is_file():
        token_ids, err = _read_tokens_pt(tok_path)
        if err:
            warnings.append(err)
    source = meta.get("source")
    return TraceRecord(
        prompt_id=int(pid),
        generated_text=text if text_complete else None,
        token_ids=token_ids,
        token_strings=None,
        source=None if source is None else str(source),
        meta=meta,
        text_complete=text_complete,
        origin=str(trace_dir),
        warnings=warnings,
    )


def load_baseline_traces(results: Path, slug: str) -> tuple[list[TraceRecord], dict]:
    """Return usable-looking trace records and a description of where we looked."""
    results = Path(results)
    primary = results / f"exp2_{slug}"
    legacy = results / "exp2"
    if _has_traces(primary):
        roots = [primary]
        root_note = "slug-scoped exp2 tree"
    elif _has_traces(legacy):
        roots = [legacy]
        root_note = "legacy results/exp2 tree (slug-scoped tree has no traces)"
    else:
        roots = [primary, legacy]
        root_note = "no per-trace meta.json files in the slug-scoped or legacy exp2 trees"
    found: list[TraceRecord] = []
    seen: set[int] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for kind in ("looping", "nonlooping"):
            base = root / kind
            if not base.is_dir():
                continue
            for trace_dir in sorted(p for p in base.glob("p*") if (p / "meta.json").is_file()):
                rec = load_trace_dir(trace_dir)
                if rec is None or rec.prompt_id in seen:
                    continue
                seen.add(rec.prompt_id)
                found.append(rec)
    previews = _preview_rows(results, slug)
    info = {
        "searched_roots": [display_path(r) for r in (primary, legacy)],
        "root_note": root_note,
        "n_trace_dirs": len(found),
        "n_with_token_ids": sum(1 for r in found if r.token_ids),
        "n_with_complete_text": sum(1 for r in found if r.text_complete),
        "n_preview_rows": len(previews),
        "preview_rows_are_not_traces": True,
        "preview_prompt_ids": [int(r["prompt_id"]) for r in previews if r.get("prompt_id") is not None],
        "token_read_warnings": [w for r in found for w in r.warnings],
    }
    return found, info


def _preview_rows(results: Path, slug: str) -> list[dict]:
    paths = []
    for path in (results / f"trigger_gen_log_{slug}.jsonl", results / "trigger_gen_log.jsonl"):
        if path.is_file() and path not in paths:
            paths.append(path)
    rows = []
    seen = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row.get("prompt_id")
            if pid in seen:
                continue
            seen.add(pid)
            rows.append(row)
    return rows


def _compact_record(path: Path, results: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    completed = data.get("completed_prompt_ids")
    if completed is None:
        completed = data.get("completed_ids")
    n_completed = data.get("n_completed")
    if n_completed is None and isinstance(completed, list):
        n_completed = len(completed)
    looping = data.get("looping_prompt_ids")
    try:
        rel = str(path.relative_to(results.parent))
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "model_id": data.get("model_id"),
        "status": data.get("status"),
        "n_prompts": data.get("n_prompts", data.get("n_total")),
        "n_completed": n_completed,
        "n_loop": data.get("n_loop"),
        "n_non": data.get("n_non"),
        "loop_rate": data.get("loop_rate", data.get("loop_rate_so_far")),
        "looping_prompt_ids": [int(x) for x in looping] if isinstance(looping, list) else None,
        "n_completed_ids_listed": len(completed) if isinstance(completed, list) else None,
    }


def load_recorded_counts(results: Path, slug: str) -> dict:
    """Read the summary and checkpoint files. These are recorded counts, not relabelled."""
    results = Path(results)
    files = {
        "baseline_summary_slug": results / f"baseline_pass_summary_{slug}.json",
        "baseline_summary_legacy": results / "baseline_pass_summary.json",
        "baseline_checkpoint_slug": results / "checkpoints" / f"baseline_pass_{slug}.json",
        "baseline_checkpoint_legacy": results / "checkpoints" / "baseline_pass.json",
        "exp2_checkpoint": results / "checkpoints" / f"exp2_gen_{slug}.json",
    }
    out = {}
    for key, path in files.items():
        compact = _compact_record(path, results)
        if compact is not None:
            out[key] = compact
    return out


def recorded_baseline_loop_ids(recorded: dict) -> list[int]:
    for key in ("baseline_summary_slug", "baseline_summary_legacy"):
        block = recorded.get(key) or {}
        ids = block.get("looping_prompt_ids")
        if ids:
            return [int(x) for x in ids]
    return []


def load_prompt_sample_splits(sample_path: Path, loop_ids: list[int] | None = None) -> dict | None:
    path = Path(sample_path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts") or []
    ids = [int(p["prompt_id"]) for p in prompts if p.get("prompt_id") is not None]
    counts = Counter(assign_split(i) for i in ids)
    id_set = set(ids)
    loops = [int(i) for i in (loop_ids or [])]
    loop_counts = Counter(assign_split(i) for i in loops if i in id_set)
    missing = [i for i in loops if i not in id_set]
    return {
        "path": display_path(path),
        "n_prompts": len(ids),
        "seed": data.get("seed"),
        "by_split": {k: int(counts.get(k, 0)) for k in ("train", "dev", "test")},
        "recorded_baseline_loops_by_split": {k: int(loop_counts.get(k, 0)) for k in ("train", "dev", "test")},
        "recorded_baseline_loops_missing_from_sample": missing,
    }
