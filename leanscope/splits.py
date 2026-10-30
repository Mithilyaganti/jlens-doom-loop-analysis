"""Hash-locked train/dev/test splits, grouped by scenario_id.

Same discipline as ``jspace.eval_sets.sample_gsm8k_eval``: the assignment is
drawn once from a fixed rule, written to a JSON file, and that file is the
source of truth afterwards. Paraphrase id and option order are not inputs, so
every surface of a scenario stays in one split.

The lock rank-cuts SHA-256(salt || scenario_id) into 60/15/25 (train/dev/test).
That cut is a pure function of the id set and the salt, so it is fixed before
any model output. The pool was re-cut at these ratios when the hand-written
scenarios grew, still before any model output. Ids added later are appended
with a per-id hash bucket and do not move ids already in the file.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from leanscope.schema import SPLITS, SplitName

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK_PATH = ROOT / "data" / "leanscope" / "split_lock.json"

SPLIT_SALT = "leanscope-split-v1"
# train, dev, test. Largest-remainder rounding, ties broken toward the later split.
SPLIT_RATIOS: tuple[float, float, float] = (0.6, 0.15, 0.25)


def scenario_hash(scenario_id: str, *, salt: str = SPLIT_SALT) -> str:
    """Hex SHA-256 of the salted scenario id. Independent of paraphrase and order."""
    text = f"{salt}:{scenario_id}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _largest_remainder(n: int, ratios: tuple[float, float, float]) -> list[int]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if len(ratios) != 3 or any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be three non-negative numbers, got {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios)}")
    raw = [n * r for r in ratios]
    floors = [int(math.floor(x)) for x in raw]
    leftover = n - sum(floors)
    # Higher index wins a tie, so a leftover goes to test before dev before train.
    order = sorted(range(3), key=lambda i: (raw[i] - floors[i], i), reverse=True)
    for k in range(leftover):
        floors[order[k]] += 1
    return floors


def rank_cut(
    scenario_ids: Iterable[str],
    *,
    salt: str = SPLIT_SALT,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> dict[str, SplitName]:
    """Assign every id by sorting on its hash and cutting at ``ratios``.

    Adding or removing an id can move other ids. Use this only to create the
    initial lock. Later additions go through :func:`hash_bucket`.
    """
    ids = sorted(set(scenario_ids))
    ranked = sorted(ids, key=lambda sid: (scenario_hash(sid, salt=salt), sid))
    counts = _largest_remainder(len(ranked), ratios)
    assigned: dict[str, SplitName] = {}
    cursor = 0
    for split, count in zip(SPLITS, counts):
        for sid in ranked[cursor : cursor + count]:
            assigned[sid] = split
        cursor += count
    return assigned


def hash_bucket(scenario_id: str, *, salt: str = SPLIT_SALT) -> SplitName:
    """Per-id bucket in ``[0, 1)`` cut at 0.60 and 0.75. Does not depend on other ids."""
    unit = int(scenario_hash(scenario_id, salt=salt)[:8], 16) / 0x100000000
    if unit < SPLIT_RATIOS[0]:
        return "train"
    if unit < SPLIT_RATIOS[0] + SPLIT_RATIOS[1]:
        return "dev"
    return "test"


def _lock_payload(assignments: Mapping[str, str], *, salt: str) -> dict:
    cleaned = {str(k): _as_split(v) for k, v in sorted(assignments.items())}
    return {
        "mechanism": "sha256-rank-cut",
        "salt": salt,
        "ratios": {"train": SPLIT_RATIOS[0], "dev": SPLIT_RATIOS[1], "test": SPLIT_RATIOS[2]},
        "grouped_by": "scenario_id",
        "note": (
            "Fixed before any model output. Grouped by scenario_id, never by "
            "paraphrase or option order. Do not regenerate this file to chase "
            "a result. Same write-once discipline as "
            "jspace.eval_sets.sample_gsm8k_eval: this file is the source of truth. "
            "New scenario ids are appended with a per-id hash bucket and do not "
            "move ids already listed here."
        ),
        "assignments": cleaned,
    }


def _as_split(value: str) -> SplitName:
    if value not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {value!r}")
    return value  # type: ignore[return-value]


def write_lock(
    path: Path | str,
    scenario_ids: Iterable[str],
    *,
    salt: str = SPLIT_SALT,
) -> dict:
    """Create a new rank-cut lock. Refuses to overwrite an existing file."""
    path = Path(path)
    if path.is_file():
        raise FileExistsError(
            f"{path} already exists. Load it, or extend it with extend_lock. "
            "Refusing to reassign ids that may already have been used."
        )
    payload = _lock_payload(rank_cut(scenario_ids, salt=salt), salt=salt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_lock(path: Path | str | None = None) -> dict:
    path = Path(path) if path is not None else DEFAULT_LOCK_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing split lock {path}. Create it once with "
            "leanscope.splits.write_lock before any result is computed."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("assignments"), dict):
        raise ValueError(f"{path} is not a split lock")
    payload["assignments"] = {
        str(k): _as_split(v) for k, v in payload["assignments"].items()
    }
    return payload


def extend_lock(lock: dict, scenario_ids: Iterable[str]) -> dict:
    """Append ids that are not yet in the lock. Existing assignments stay put."""
    salt = str(lock.get("salt") or SPLIT_SALT)
    assignments = dict(lock["assignments"])
    for sid in scenario_ids:
        sid = str(sid)
        if sid not in assignments:
            assignments[sid] = hash_bucket(sid, salt=salt)
    updated = _lock_payload(assignments, salt=salt)
    updated["mechanism"] = lock.get("mechanism", updated["mechanism"])
    return updated


def save_lock(path: Path | str, lock: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _lock_payload(lock["assignments"], salt=str(lock.get("salt") or SPLIT_SALT))
    payload["mechanism"] = lock.get("mechanism", payload["mechanism"])
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_lock(
    path: Path | str,
    scenario_ids: Iterable[str],
    *,
    update: bool = False,
) -> dict:
    """Load the lock, creating it if absent.

    Missing ids are an error unless ``update`` is set, in which case they are
    appended and the previous assignments are left as they were.
    """
    path = Path(path)
    ids = [str(sid) for sid in scenario_ids]
    if not path.is_file():
        return write_lock(path, ids)
    lock = load_lock(path)
    missing = [sid for sid in ids if sid not in lock["assignments"]]
    if not missing:
        return lock
    if not update:
        raise KeyError(
            f"{len(missing)} scenario ids are not in {path} "
            f"(for example {missing[:3]}). Pass update=True to append them "
            "without moving the ids already locked."
        )
    updated = extend_lock(lock, missing)
    save_lock(path, updated)
    return updated


def split_for(scenario_id: str, lock: Mapping[str, object] | dict) -> SplitName:
    """Look up a scenario's split. Raises if the id was not locked."""
    assignments = lock["assignments"] if isinstance(lock, dict) and "assignments" in lock else lock
    if scenario_id not in assignments:
        raise KeyError(
            f"{scenario_id} is not in the split lock. "
            "Append it with extend_lock; do not reassign the existing ids."
        )
    return _as_split(str(assignments[scenario_id]))
