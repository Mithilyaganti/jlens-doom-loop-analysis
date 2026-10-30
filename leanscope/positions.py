"""Residual capture at the lean-readout positions.

P0 is the last prompt token, before the assistant turn. P1 is the token just
before the first option word in the answer. P2 is the end of a thinking block
and is recorded only when ``record_p2`` is set (or ``LEANSCOPE_RECORD_P2=1``).

Capture is one forward pass: :func:`jspace.generation.capture_residuals_one_pass`
hooks ``stack.layers`` and offloads those positions to CPU inside the hooks.
Option names may be several tokens. If that span is absent from the answer,
P1 falls back to a single-token label (``A`` or ``B``).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from leanscope.schema import Record, Scenario

POSITIONS = ("P0", "P1", "P2")
_THINK_END_MARKERS = ("</think>", "</think>")


def record_p2_from_env() -> bool:
    return os.environ.get("LEANSCOPE_RECORD_P2", "0").strip() == "1"


def render_user_prompt(scenario: Scenario) -> str:
    """List the two options in ``scenario.order``. A and B stay the option identities."""
    if scenario.order == "AB":
        first_label, first_text = "A", scenario.option_a
        second_label, second_text = "B", scenario.option_b
    else:
        first_label, first_text = "B", scenario.option_b
        second_label, second_text = "A", scenario.option_a
    return (
        f"{scenario.question}\n\n"
        f"Option {first_label}: {first_text}\n"
        f"Option {second_label}: {second_text}\n\n"
        "Reply with your final answer as Answer option."
    )


def render_prompt(tokenizer: Any, scenario: Scenario, system_prompt: str) -> str:
    """Render system + user. The system text stays in the string.

    ``jspace.generation.apply_chat_template`` drops the system message when the
    tokenizer has no chat template, and it appends an antidoom answer cue.
    LeanScope renders the judgment prompt itself and uses the tokenizer's
    template only when that call succeeds.
    """
    user = render_user_prompt(scenario)
    system = system_prompt or ""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            pass
    parts = [part for part in (system, user) if part]
    return "\n\n".join(parts)


def encode_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [int(tok) for tok in ids]
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(tok) for tok in ids]


def _dedupe_spans(spans: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for span in spans:
        if span and span not in out:
            out.append(span)
    return out


def option_spans(tokenizer: Any, text: str) -> list[list[int]]:
    """Token spans for an option name.

    The full encoding is kept, including when the name is several tokens.
    Single-token variants from :func:`jspace.token_sets.tokenize_word_variants`
    are added only for a one-word name, so a trailing subword cannot masquerade
    as the whole option.
    """
    raw = (text or "").strip()
    if not raw:
        return []
    spans: list[list[int]] = []
    for form in (raw, f" {raw}", raw.lower(), f" {raw.lower()}"):
        spans.append(encode_ids(tokenizer, form))
    if " " not in raw:
        try:
            from jspace.token_sets import tokenize_word_variants

            for tid, _form in tokenize_word_variants(tokenizer, raw):
                spans.append([int(tid)])
        except Exception:
            pass
    return _dedupe_spans(spans)


def label_spans(tokenizer: Any, label: str) -> list[list[int]]:
    """Single-token encodings of a letter label such as ``A`` or ``B``."""
    spans: list[list[int]] = []
    for form in (label, label.lower(), label.upper(), f" {label}", f" {label.lower()}"):
        ids = encode_ids(tokenizer, form)
        if len(ids) == 1:
            spans.append(ids)
    try:
        from jspace.token_sets import tokenize_word_variants

        for tid, _form in tokenize_word_variants(tokenizer, label):
            spans.append([int(tid)])
    except Exception:
        pass
    return _dedupe_spans(spans)


def resolve_option_token_ids(tokenizer: Any, option_text: str, label: str) -> tuple[list[int], bool]:
    """Option-name token ids, or the single-token label if the name will not encode.

    Returns ``(token_ids, used_label_fallback)``.
    """
    spans = option_spans(tokenizer, option_text)
    if spans:
        return max(spans, key=len), False
    labels = label_spans(tokenizer, label)
    if labels:
        return labels[0], True
    return [], True


def earliest_span(
    token_ids: Sequence[int], start: int, spans: Sequence[Sequence[int]]
) -> tuple[int, list[int]] | None:
    """Earliest ``(index, span)`` at or after ``start``, or None."""
    ids = list(token_ids)
    best: tuple[int, list[int]] | None = None
    for span in spans:
        needle = list(span)
        width = len(needle)
        if width == 0 or start < 0:
            continue
        last = len(ids) - width
        for index in range(start, last + 1):
            if ids[index : index + width] == needle:
                if best is None or index < best[0]:
                    best = (index, needle)
                break
    return best


def find_think_end_position(tokenizer: Any, full_ids: Sequence[int], prompt_len: int) -> int | None:
    """Absolute index of the token that completes a closing think tag in the answer."""
    gen = list(full_ids[prompt_len:])
    if not gen or not hasattr(tokenizer, "decode"):
        return None
    for end in range(1, len(gen) + 1):
        text = tokenizer.decode(gen[:end], skip_special_tokens=False)
        for marker in _THINK_END_MARKERS:
            if marker in text:
                return prompt_len + end - 1
    return None


def locate_positions(
    tokenizer: Any,
    full_ids: Sequence[int],
    prompt_len: int,
    option_a: str,
    option_b: str,
    *,
    record_p2: bool = False,
) -> dict[str, Any]:
    """Locate P0, P1, and (optionally) P2 in ``full_ids``.

    P1 searches the generated ids only, so the option names in the prompt are
    not treated as the answer. Multi-token names are tried first. If neither
    name occurs, single-token labels ``A`` and ``B`` are the fallback.
    """
    ids = [int(tok) for tok in full_ids]
    prompt_len = int(prompt_len)
    p0 = prompt_len - 1 if prompt_len > 0 else None
    hit_a = earliest_span(ids, prompt_len, option_spans(tokenizer, option_a))
    hit_b = earliest_span(ids, prompt_len, option_spans(tokenizer, option_b))
    used_fallback = False
    tagged: list[tuple[int, list[int], str]] = []
    if hit_a is not None:
        tagged.append((hit_a[0], hit_a[1], "A"))
    if hit_b is not None:
        tagged.append((hit_b[0], hit_b[1], "B"))
    if not tagged:
        used_fallback = True
        hit_a = earliest_span(ids, prompt_len, label_spans(tokenizer, "A"))
        hit_b = earliest_span(ids, prompt_len, label_spans(tokenizer, "B"))
        if hit_a is not None:
            tagged.append((hit_a[0], hit_a[1], "A"))
        if hit_b is not None:
            tagged.append((hit_b[0], hit_b[1], "B"))
    p1 = None
    matched = None
    if tagged:
        start, _span, matched = min(tagged, key=lambda item: item[0])
        p1 = start - 1 if start > 0 else None
    p2 = find_think_end_position(tokenizer, ids, prompt_len) if record_p2 else None
    return {
        "P0": p0,
        "P1": p1,
        "P2": p2,
        "label_fallback": used_fallback and p1 is not None,
        "matched_option": matched,
        "prompt_len": prompt_len,
    }


def _last_phrase(text: str, phrase: str) -> int:
    if not phrase:
        return -1
    hay = text.lower()
    needle = phrase.lower().strip()
    if not needle:
        return -1
    if " " in needle:
        return hay.rfind(needle)
    matches = list(re.finditer(rf"\b{re.escape(needle)}\b", hay))
    return matches[-1].start() if matches else -1


def infer_recommendation(text: str, scenario: Scenario) -> str | None:
    """Which option the visible answer ends on, or None if it names neither.

    The later of the two option names wins. Standalone letter labels count
    only when the name itself is absent from that side.
    """
    if not text:
        return None
    candidates: list[tuple[int, str]] = []
    for side, phrase in (("A", scenario.option_a), ("B", scenario.option_b)):
        index = _last_phrase(text, phrase)
        if index >= 0:
            candidates.append((index, side))
    if not candidates:
        for side, pattern in (("A", r"\bA\b"), ("B", r"\bB\b")):
            matches = list(re.finditer(pattern, text))
            if matches:
                candidates.append((matches[-1].start(), side))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def resolve_workspace_layers(
    stack: Any,
    band_path: Path | str | None = None,
    *,
    key_only: bool = False,
) -> list[int]:
    """Workspace-band layer indices.

    A band JSON (``workspace_layers``, optionally ``key_workspace_layers``)
    wins. Otherwise ``stack.workspace_layers``, otherwise the middle third of
    the stack.
    """
    n_layers = int(getattr(stack, "n_layers", len(stack.layers)))
    if band_path is not None and Path(band_path).is_file():
        import json

        data = json.loads(Path(band_path).read_text(encoding="utf-8"))
        raw = data.get("key_workspace_layers") if key_only else None
        if not raw:
            raw = data.get("workspace_layers") or data.get("key_workspace_layers") or []
        picked = [int(layer) for layer in raw if 0 <= int(layer) < n_layers]
        if picked:
            return picked
    existing = getattr(stack, "workspace_layers", None)
    if existing:
        picked = [int(layer) for layer in existing if 0 <= int(layer) < n_layers]
        if picked:
            return picked
    start = n_layers // 3
    end = max(start + 1, (2 * n_layers) // 3)
    return list(range(start, end))


def record_residuals_one_pass(
    stack: Any,
    full_ids: Sequence[int],
    located: dict[str, Any],
    layers: Sequence[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Hook one forward and return ``{position: {layer: vector}}``."""
    from jspace.generation import capture_residuals_one_pass

    named = {name: located.get(name) for name in POSITIONS}
    keep: list[int] = []
    for pos in named.values():
        if pos is None:
            continue
        idx = int(pos)
        if idx not in keep:
            keep.append(idx)
    if not keep:
        return {}
    ids = torch.tensor(list(full_ids), dtype=torch.long)
    captured = capture_residuals_one_pass(
        stack, ids, layers=list(layers), token_positions=keep
    )
    out: dict[str, dict[int, torch.Tensor]] = {}
    for name, pos in named.items():
        if pos is None:
            continue
        row = keep.index(int(pos))
        out[name] = {int(layer): tensor[row].clone() for layer, tensor in captured.items()}
    return out


def _safe_name(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", item_id)


def stable_prompt_id(item_id: str) -> int:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_activation(path: Path | str) -> dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(str(path), map_location="cpu", weights_only=False)


def record_item(
    stack: Any,
    record: Record,
    *,
    activations_dir: Path | str,
    workspace_layers: Sequence[int],
    layers: Sequence[int] | None = None,
    record_p2: bool | None = None,
    max_new_tokens: int = 64,
    recommendation_fn: Callable[[Sequence[int], Record], str | None] | None = None,
) -> Record:
    """Generate one answer and store residuals at P0/P1/(P2).

    ``recommendation_fn`` is used only when the visible text names neither
    option. The CPU fake uses it for silent (conceal) answers. A real run
    leaves it unset and relies on :func:`infer_recommendation`.
    """
    from jspace.generation import generate_greedy

    if record_p2 is None:
        record_p2 = record_p2_from_env()
    prompt = render_prompt(stack.tokenizer, record.scenario, record.condition.system_prompt)
    generated = generate_greedy(
        stack,
        prompt,
        max_new_tokens=max_new_tokens,
        prompt_id=stable_prompt_id(record.item_id),
    )
    full_ids = [int(tok) for tok in generated["full_ids"]]
    prompt_len = int(generated["prompt_len"])
    located = locate_positions(
        stack.tokenizer,
        full_ids,
        prompt_len,
        record.scenario.option_a,
        record.scenario.option_b,
        record_p2=bool(record_p2),
    )
    use_layers = list(layers) if layers is not None else list(workspace_layers)
    residuals = record_residuals_one_pass(stack, full_ids, located, use_layers)
    ids_a, fallback_a = resolve_option_token_ids(stack.tokenizer, record.scenario.option_a, "A")
    ids_b, fallback_b = resolve_option_token_ids(stack.tokenizer, record.scenario.option_b, "B")
    payload = {
        "item_id": record.item_id,
        "positions": {name: located.get(name) for name in POSITIONS},
        "residuals": residuals,
        "layers": [int(layer) for layer in use_layers],
        "workspace_layers": [int(layer) for layer in workspace_layers],
        "option_token_ids": {"A": [int(tok) for tok in ids_a], "B": [int(tok) for tok in ids_b]},
        "label_fallback": bool(located.get("label_fallback") or fallback_a or fallback_b),
        "matched_option": located.get("matched_option"),
        "prompt_len": prompt_len,
    }
    out_dir = Path(activations_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe_name(record.item_id)}.pt"
    torch.save(payload, path)
    text = str(generated.get("generated_text") or "")
    recommended = infer_recommendation(text, record.scenario)
    if recommended is None and recommendation_fn is not None:
        recommended = recommendation_fn(full_ids, record)
    return replace(
        record,
        output_text=text,
        recommended=recommended,
        activations_path=str(path),
    )
