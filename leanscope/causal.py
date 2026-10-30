"""Steer and ablate an option direction at P0 and P1.

Condition design follows ``scripts/05_exp3_causal.py``:

- ``baseline`` — lean prompt, no hook (the pass-1 rerun).
- ``steer_option`` / ``ablate_option`` — the real direction, at one position.
- ``steer_random`` / ``ablate_random`` — a norm-matched random direction.
- ``remove_lean_prompt`` — positive control. The lean system prompt is removed
  and the question is regenerated with no hook. This is the analogue of
  Exp3's logit-suppression control: an intervention that must move the
  outcome if the run is working, independent of the direction.

Directions are unit-length. Steering adds the same scalar multiple of a unit
vector in both the option and the random conditions, so the added residual
has the same L2 norm. Ablation projects that unit direction out
(``jspace.intervention.project_out`` / ``InterventionContext``).

The direction itself comes from a pluggable function. The default is the
difference of means of saved residuals, lean A minus lean B, so a later
detector can replace it without changing the condition list.

P0 is the last prompt token. P1 is the token just before the first option
word in the pass-1 answer. Hooks fire only at that position. On a
``use_cache=True`` decode the bookkeeping follows the onset hook in
``jspace.intervention``: the first forward is the prompt, later forwards are
the new tokens. A full re-forward (the CPU test double) indexes the position
in the current sequence instead.

Flip rate is the fraction of items whose recommendation changed relative to
baseline. The paired test is exact McNemar on "follows the assigned lean",
with Wilson intervals and Cohen's h, all from ``jspace.stats``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from jspace.intervention import (
    InterventionContext,
    project_out,
    random_direction,
)
from jspace.stats import cohens_h, mcnemar_exact, wilson_ci

from leanscope.prompts import parse_option, render_prompt
from leanscope.schema import Record

POSITIONS = ("P0", "P1")
INTERVENTION_CONDITIONS = (
    "steer_option",
    "ablate_option",
    "steer_random",
    "ablate_random",
)
POSITIVE_CONTROL = "remove_lean_prompt"
BASELINE = "baseline"

DirectionFn = Callable[..., torch.Tensor]


@dataclass
class DecodeResult:
    text: str
    recommended: str | None
    prompt_len: int
    generated_ids: list[int]
    n_tokens: int


@dataclass
class Edit:
    mode: str
    direction_by_layer: dict[int, torch.Tensor]
    abs_position: int
    alpha: float
    cache_mode: str


def _as_unit(vector: torch.Tensor) -> torch.Tensor:
    flat = vector.detach().float().cpu().reshape(-1)
    if int(flat.numel()) == 0:
        raise ValueError("direction is empty")
    return flat / flat.norm().clamp_min(1e-12)


def matched_random(reference: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Unit random direction with the same L2 norm as ``reference``.

    ``random_direction`` already returns a unit vector. Scaling by the
    reference norm is what makes a non-unit probe weight comparable; both
    sides are unit-normalized again before the hook so the steer delta has
    L2 norm ``|alpha|`` either way.
    """
    ref = reference.detach().float().cpu().reshape(-1)
    unit = random_direction(int(ref.numel()), seed=int(seed))
    return unit * ref.norm().clamp_min(1e-12)


def control_seed(item_id: str, layer: int, position: str) -> int:
    """Deterministic per-item seed. Not Python's salted ``hash``."""
    acc = 17
    for ch in f"{item_id}|{position}":
        acc = (acc * 131 + ord(ch)) % (2**31 - 1)
    return (acc + int(layer) * 1_000_003 + 17) % (2**31 - 1)


def steer_alpha_against_lean(assigned_lean: str, alpha: float) -> float:
    """Sign the A-minus-B axis so the edit pushes against the assigned lean."""
    magnitude = abs(float(alpha))
    if assigned_lean == "A":
        return -magnitude
    if assigned_lean == "B":
        return magnitude
    raise ValueError("steering against a lean requires assigned lean 'A' or 'B'")


def absolute_position(
    prompt_len: int,
    generated_ids: Sequence[int],
    position: str,
    option_token_ids: set[int] | None = None,
) -> int:
    """Absolute index of P0 or of the token just before the first option token."""
    if prompt_len < 1:
        raise ValueError("prompt_len must be positive")
    if position == "P0":
        return int(prompt_len) - 1
    if position != "P1":
        raise ValueError(f"position must be 'P0' or 'P1', got {position!r}")
    option_ids = {int(t) for t in (option_token_ids or ())}
    for index, token_id in enumerate(generated_ids):
        if int(token_id) in option_ids:
            if index == 0:
                return int(prompt_len) - 1
            return int(prompt_len) + index - 1
    if generated_ids:
        return int(prompt_len) + len(generated_ids) - 1
    return int(prompt_len) - 1


def locate_p1_from_text(
    prompt_len: int,
    generated_ids: Sequence[int],
    text: str,
    tokenizer,
    needles: Sequence[str],
) -> int | None:
    """Map the earliest option-word in ``text`` back to a token just before it."""
    usable = [needle for needle in needles if needle]
    if tokenizer is None or not generated_ids or not usable:
        return None
    for index in range(len(generated_ids)):
        decoded = tokenizer.decode(list(generated_ids[: index + 1]), skip_special_tokens=True)
        if any(needle in decoded for needle in usable):
            if index == 0:
                return int(prompt_len) - 1
            return int(prompt_len) + index - 1
    return None


def load_residual(record: Record, *, position: str, layer: int) -> torch.Tensor:
    """Load one residual vector from ``record.activations_path``.

    The file is a torch dict. Either ``payload[position]`` is ``[n_layers, d]``
    (or ``[d]`` for a single saved layer), or it is a dict keyed by layer.
    Detection records nest that dict under ``payload["residuals"]``.
    """
    if not record.activations_path:
        raise ValueError(f"{record.item_id} has no activations_path")
    try:
        payload = torch.load(record.activations_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(record.activations_path, map_location="cpu")
    block = payload.get(position) if isinstance(payload, dict) else None
    if block is None and isinstance(payload, dict):
        nested = payload.get("residuals")
        if isinstance(nested, dict):
            block = nested.get(position)
    if block is None:
        raise ValueError(f"{record.activations_path} has no {position} residual")
    if isinstance(block, dict):
        if layer in block:
            vec = block[layer]
        elif str(layer) in block:
            vec = block[str(layer)]
        else:
            raise ValueError(f"{record.activations_path} {position} has no layer {layer}")
    else:
        tensor = block if torch.is_tensor(block) else torch.as_tensor(block)
        vec = tensor[layer] if tensor.ndim >= 2 else tensor
    return vec.detach().float().cpu().reshape(-1)


def difference_of_means_direction(
    records: Sequence[Record],
    *,
    layer: int,
    position: str,
) -> torch.Tensor:
    """Mean residual under lean A minus mean residual under lean B.

    This is the persona-vector style default. Records without an assigned
    lean or without an activation file are skipped. The result is not
    normalized; callers unit-normalize before the hook.
    """
    groups: dict[str, list[torch.Tensor]] = {"A": [], "B": []}
    for record in records:
        lean = record.condition.lean
        if lean not in groups or not record.activations_path:
            continue
        groups[lean].append(load_residual(record, position=position, layer=layer))
    if not groups["A"] or not groups["B"]:
        raise ValueError(
            f"difference of means at {position} layer {layer} needs activations "
            f"for both lean A and lean B (got {len(groups['A'])} and {len(groups['B'])})"
        )
    mean_a = torch.stack(groups["A"]).mean(dim=0)
    mean_b = torch.stack(groups["B"]).mean(dim=0)
    return mean_a - mean_b


def make_tracked_position_hook(
    direction: torch.Tensor,
    *,
    abs_position: int,
    mode: str,
    alpha: float,
):
    """Edit one absolute position across an incremental (use_cache) decode.

    The first forward is the whole prompt. Later forwards contain only the
    new tokens. Position counting matches ``make_onset_ablation_hook``.
    Ablation calls ``project_out``. Steering adds ``alpha`` times the unit
    direction.
    """
    if mode not in ("ablate", "steer"):
        raise ValueError(mode)
    unit = _as_unit(direction)
    state = {"n_seen": 0}
    target = int(abs_position)

    def hook(module, inputs, output):
        if torch.is_tensor(output):
            hidden = output
            rest = None
        else:
            hidden = output[0]
            rest = output[1:]
        seq_len = int(hidden.shape[1])
        pos_first = int(state["n_seen"])
        state["n_seen"] = pos_first + seq_len
        local = target - pos_first
        if not (0 <= local < seq_len):
            return output
        hidden = hidden.clone()
        _edit_row(hidden, local, unit, mode=mode, alpha=alpha)
        if rest is None:
            return hidden
        return (hidden, *rest)

    return hook


def make_full_forward_steer_hook(direction: torch.Tensor, *, position: int, alpha: float):
    """Add ``alpha * unit(direction)`` at one index of a full-sequence forward."""
    unit = _as_unit(direction)
    target = int(position)

    def hook(module, inputs, output):
        if torch.is_tensor(output):
            hidden = output
            rest = None
        else:
            hidden = output[0]
            rest = output[1:]
        seq_len = int(hidden.shape[1])
        local = target if target >= 0 else seq_len + target
        if not (0 <= local < seq_len):
            return output
        hidden = hidden.clone()
        _edit_row(hidden, local, unit, mode="steer", alpha=alpha)
        if rest is None:
            return hidden
        return (hidden, *rest)

    return hook


def _edit_row(hidden: torch.Tensor, local: int, unit: torch.Tensor, *, mode: str, alpha: float) -> None:
    row = hidden[:, local, :]
    if mode == "ablate":
        hidden[:, local, :] = project_out(row, unit)
        return
    step = unit.to(device=row.device, dtype=row.dtype)
    hidden[:, local, :] = row + float(alpha) * step


class EditContext:
    """Register the position edit, then remove it.

    Full-sequence ablation goes through ``InterventionContext`` so the CPU
    test exercises the same hook Exp3 uses. Incremental ablation and both
    steer paths use a one-position hook that calls ``project_out`` for the
    ablation case.
    """

    def __init__(self, stack, edit: Edit):
        self.stack = stack
        self.edit = edit
        self._handles: list = []
        self._intervention: InterventionContext | None = None

    def __enter__(self):
        edit = self.edit
        if edit.cache_mode == "full" and edit.mode == "ablate":
            self._intervention = InterventionContext(
                self.stack,
                list(edit.direction_by_layer),
                dict(edit.direction_by_layer),
                position=edit.abs_position,
                strength=1.0,
            )
            self._intervention.__enter__()
            return self
        for layer, direction in edit.direction_by_layer.items():
            if edit.cache_mode == "incremental":
                hook = make_tracked_position_hook(
                    direction,
                    abs_position=edit.abs_position,
                    mode=edit.mode,
                    alpha=edit.alpha,
                )
            else:
                if edit.mode != "steer":
                    raise ValueError(edit.mode)
                hook = make_full_forward_steer_hook(
                    direction, position=edit.abs_position, alpha=edit.alpha
                )
            self._handles.append(self.stack.layers[layer].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        if self._intervention is not None:
            self._intervention.__exit__(*exc)
            self._intervention = None
        for handle in self._handles:
            handle.remove()
        self._handles = []


def contrast_stats(
    baseline: Sequence[str | None],
    condition: Sequence[str | None],
    leans: Sequence[str | None],
) -> dict:
    """Flip rate, exact McNemar, Wilson intervals, and Cohen's h.

    McNemar's ``b`` counts pairs where the baseline follows the assigned lean
    and the condition does not. ``c`` counts the opposite discordance.
    Cohen's h compares the two follow rates, baseline minus condition, the
    same order as Exp3.
    """
    n = len(baseline)
    if n != len(condition) or n != len(leans):
        raise ValueError("baseline, condition, and leans must be the same length")
    n_flips = sum(b != c for b, c in zip(baseline, condition))
    b_only = 0
    c_only = 0
    base_follow = 0
    cond_follow = 0
    for recommended_base, recommended_cond, lean in zip(baseline, condition, leans):
        follows_base = recommended_base is not None and recommended_base == lean
        follows_cond = recommended_cond is not None and recommended_cond == lean
        base_follow += int(follows_base)
        cond_follow += int(follows_cond)
        if follows_base and not follows_cond:
            b_only += 1
        elif follows_cond and not follows_base:
            c_only += 1
    mc = mcnemar_exact(b_only, c_only)
    if n == 0:
        nan = float("nan")
        empty = [nan, nan]
        return {
            "n_pairs": 0,
            "n_flips": 0,
            "flip_rate": nan,
            "flip_rate_wilson_ci": empty,
            "b_baseline_follows_only": 0,
            "c_condition_follows_only": 0,
            "p_exact": mc["p_exact"],
            "odds_ratio_b_over_c": mc["odds_ratio_b_over_c"],
            "rate_baseline_follows": nan,
            "rate_condition_follows": nan,
            "wilson_ci_baseline_follows": empty,
            "wilson_ci_condition_follows": empty,
            "cohens_h": nan,
        }
    rate_base = base_follow / n
    rate_cond = cond_follow / n
    flip_lo, flip_hi = wilson_ci(n_flips, n)
    base_lo, base_hi = wilson_ci(base_follow, n)
    cond_lo, cond_hi = wilson_ci(cond_follow, n)
    return {
        "n_pairs": n,
        "n_flips": int(n_flips),
        "flip_rate": n_flips / n,
        "flip_rate_wilson_ci": [float(flip_lo), float(flip_hi)],
        "b_baseline_follows_only": int(b_only),
        "c_condition_follows_only": int(c_only),
        "p_exact": mc["p_exact"],
        "odds_ratio_b_over_c": mc["odds_ratio_b_over_c"],
        "rate_baseline_follows": rate_base,
        "rate_condition_follows": rate_cond,
        "wilson_ci_baseline_follows": [float(base_lo), float(base_hi)],
        "wilson_ci_condition_follows": [float(cond_lo), float(cond_hi)],
        "cohens_h": cohens_h(rate_base, rate_cond),
    }


def build_contrasts(outcomes: Sequence[Mapping]) -> dict[str, dict]:
    """Pair every condition with the baseline row of the same item."""
    by_item: dict[str, dict[str, Mapping]] = {}
    for row in outcomes:
        by_item.setdefault(str(row["item_id"]), {})[str(row["key"])] = row
    keys = sorted({str(row["key"]) for row in outcomes if row["key"] != BASELINE})
    contrasts: dict[str, dict] = {}
    for key in keys:
        baseline_recs: list[str | None] = []
        condition_recs: list[str | None] = []
        leans: list[str | None] = []
        for cols in by_item.values():
            if BASELINE not in cols or key not in cols:
                continue
            baseline_recs.append(cols[BASELINE]["recommended"])
            condition_recs.append(cols[key]["recommended"])
            leans.append(cols[BASELINE]["assigned_lean"])
        contrasts[key] = contrast_stats(baseline_recs, condition_recs, leans)
    return contrasts


def _bind_direction(
    direction_fn: DirectionFn | None,
    direction_records: Sequence[Record] | None,
) -> tuple[DirectionFn, list[Record]]:
    """Return ``(fn, records)`` for ``fn(records, *, layer, position)``.

    The default ``fn`` is the difference of means on ``direction_records``.
    A detector plugs in by passing its own function with that signature.
    """
    records = list(direction_records or [])
    if direction_fn is not None:
        return direction_fn, records
    if not records:
        raise ValueError(
            "difference-of-means needs direction_records (use the train split). "
            "Pass direction_fn to plug in another direction."
        )

    def _default(source: Sequence[Record], *, layer: int, position: str) -> torch.Tensor:
        return difference_of_means_direction(source, layer=layer, position=position)

    return _default, records


def _call_direction(
    direction_fn: DirectionFn,
    records: Sequence[Record],
    *,
    layer: int,
    position: str,
) -> torch.Tensor:
    vector = direction_fn(records, layer=layer, position=position)
    return _as_unit(vector)


def _prompt_length_hf(stack, prompt: str) -> int:
    encoded = stack.tokenizer(prompt, truncation=True, max_length=2048)
    return int(len(encoded["input_ids"]))


@torch.no_grad()
def _decode_logits(
    stack,
    prompt: str,
    *,
    tokenize: Callable[[str], torch.Tensor],
    option_ids: Mapping[int, str],
    max_new_tokens: int,
) -> DecodeResult:
    ids = tokenize(prompt)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    prompt_len = int(ids.shape[1])
    generated: list[int] = []
    option_set = set(option_ids)
    for _ in range(max_new_tokens):
        out = stack.model(input_ids=ids)
        next_id = int(torch.argmax(out.logits[0, -1]).item())
        generated.append(next_id)
        nxt = torch.tensor([[next_id]], dtype=ids.dtype, device=ids.device)
        ids = torch.cat([ids, nxt], dim=1)
        if next_id in option_set:
            break
    recommended = option_ids.get(generated[-1]) if generated else None
    text = f"Answer: {recommended}" if recommended else ""
    return DecodeResult(
        text=text,
        recommended=recommended,
        prompt_len=prompt_len,
        generated_ids=generated,
        n_tokens=len(generated),
    )


@torch.no_grad()
def _decode_hf(stack, prompt: str, *, max_new_tokens: int) -> DecodeResult:
    from jspace.generation import generate_greedy

    prompt_len = _prompt_length_hf(stack, prompt)
    generated = generate_greedy(stack, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    ids = [int(t) for t in generated["generated_ids"]]
    return DecodeResult(
        text=generated["generated_text"],
        recommended=None,
        prompt_len=int(generated.get("prompt_len", prompt_len)),
        generated_ids=ids,
        n_tokens=len(ids),
    )


def _resolve_recommendation(result: DecodeResult, record: Record) -> str | None:
    if result.recommended in ("A", "B"):
        return result.recommended
    return parse_option(result.text, record.scenario)


def _position_index(
    result: DecodeResult,
    record: Record,
    position: str,
    *,
    option_token_ids: set[int],
    tokenizer,
) -> int:
    if position == "P0":
        return absolute_position(result.prompt_len, result.generated_ids, "P0")
    text_index = None
    if tokenizer is not None:
        scenario = record.scenario
        text_index = locate_p1_from_text(
            result.prompt_len,
            result.generated_ids,
            result.text,
            tokenizer,
            [scenario.option_a, scenario.option_b, "Answer: A", "Answer: B"],
        )
    if text_index is not None:
        return text_index
    return absolute_position(
        result.prompt_len, result.generated_ids, "P1", option_token_ids
    )


def run_causal(
    stack,
    eval_records: Sequence[Record],
    *,
    direction_records: Sequence[Record] | None = None,
    direction_fn: DirectionFn | None = None,
    layers: Sequence[int] = (0,),
    positions: Sequence[str] = POSITIONS,
    alpha: float = 1.0,
    max_new_tokens: int = 32,
    backend: str = "auto",
    tokenize: Callable[[str], torch.Tensor] | None = None,
    option_token_ids: Mapping[int, str] | None = None,
    neutral_system_prompt: str = "",
    per_item_direction: bool = False,
    direction_name: str | None = None,
) -> dict:
    """Run the paired causal conditions on records that carry an assigned lean.

    ``direction_fn(records, *, layer, position) -> Tensor`` is called once per
    layer and position, with the train records. The default is the difference
    of means on that split. Pass a function with the same signature to plug
    in a detector direction. ``per_item_direction`` calls it on each eval
    record alone, which is what the J-lens option direction needs when
    scenarios do not share option text.

    ``backend="logits"`` re-forwards the whole sequence (CPU test double).
    ``backend="hf"`` calls ``jspace.generation.generate_greedy`` with
    incremental hooks. ``auto`` uses HF when the stack has a tokenizer and
    ``model.generate``.
    """
    layer_list = [int(layer) for layer in layers]
    position_list = tuple(positions)
    for position in position_list:
        if position not in POSITIONS:
            raise ValueError(f"unknown position {position!r}")
    n_layers = len(stack.layers)
    for layer in layer_list:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"layer {layer} is outside 0..{n_layers - 1}")

    if backend == "auto":
        backend = "hf" if getattr(stack, "tokenizer", None) is not None and hasattr(stack.model, "generate") else "logits"
    if backend not in ("hf", "logits"):
        raise ValueError(backend)
    if backend == "logits" and (tokenize is None or not option_token_ids):
        raise ValueError("logits backend needs tokenize and option_token_ids")
    cache_mode = "incremental" if backend == "hf" else "full"
    option_map = {int(k): v for k, v in (option_token_ids or {}).items()}
    option_set = set(option_map)
    bound, bound_records = _bind_direction(direction_fn, direction_records)

    def _directions_for(source: Sequence[Record]) -> dict[tuple[int, str], torch.Tensor]:
        built: dict[tuple[int, str], torch.Tensor] = {}
        for position in position_list:
            for layer in layer_list:
                built[(layer, position)] = _call_direction(
                    bound, source, layer=layer, position=position
                )
        return built

    shared_directions = None if per_item_direction else _directions_for(bound_records)

    def decode(prompt: str) -> DecodeResult:
        if backend == "hf":
            return _decode_hf(stack, prompt, max_new_tokens=max_new_tokens)
        assert tokenize is not None
        return _decode_logits(
            stack,
            prompt,
            tokenize=tokenize,
            option_ids=option_map,
            max_new_tokens=max_new_tokens,
        )

    outcomes: list[dict] = []
    for record in eval_records:
        lean = record.condition.lean
        if lean not in ("A", "B"):
            continue
        directions = _directions_for([record]) if per_item_direction else shared_directions
        assert directions is not None
        baseline_prompt = render_prompt(record.scenario, record.condition.system_prompt)
        baseline = decode(baseline_prompt)
        baseline_rec = _resolve_recommendation(baseline, record)
        outcomes.append(
            _row(
                record,
                key=BASELINE,
                condition=BASELINE,
                position=None,
                recommended=baseline_rec,
                result=baseline,
                abs_position=None,
            )
        )
        neutral = decode(render_prompt(record.scenario, neutral_system_prompt))
        outcomes.append(
            _row(
                record,
                key=POSITIVE_CONTROL,
                condition=POSITIVE_CONTROL,
                position=None,
                recommended=_resolve_recommendation(neutral, record),
                result=neutral,
                abs_position=None,
            )
        )
        tokenizer = getattr(stack, "tokenizer", None) if backend == "hf" else None
        for position in position_list:
            index = _position_index(
                baseline,
                record,
                position,
                option_token_ids=option_set,
                tokenizer=tokenizer,
            )
            for mode, condition, use_random in (
                ("steer", "steer_option", False),
                ("ablate", "ablate_option", False),
                ("steer", "steer_random", True),
                ("ablate", "ablate_random", True),
            ):
                by_layer: dict[int, torch.Tensor] = {}
                for layer in layer_list:
                    reference = directions[(layer, position)]
                    if use_random:
                        by_layer[layer] = _as_unit(
                            matched_random(
                                reference,
                                seed=control_seed(record.item_id, layer, position),
                            )
                        )
                    else:
                        by_layer[layer] = reference
                signed = steer_alpha_against_lean(lean, alpha) if mode == "steer" else 0.0
                edit = Edit(
                    mode=mode,
                    direction_by_layer=by_layer,
                    abs_position=index,
                    alpha=signed,
                    cache_mode=cache_mode,
                )
                with EditContext(stack, edit):
                    edited = decode(baseline_prompt)
                outcomes.append(
                    _row(
                        record,
                        key=f"{condition}@{position}",
                        condition=condition,
                        position=position,
                        recommended=_resolve_recommendation(edited, record),
                        result=edited,
                        abs_position=index,
                    )
                )

    contrasts = build_contrasts(outcomes)
    return {
        "protocol": "leanscope_causal_v1",
        "backend": backend,
        "cache_mode": cache_mode,
        "positions": list(position_list),
        "layers": layer_list,
        "alpha": float(alpha),
        "norm_matching": (
            "option and random directions are unit-normalized; "
            "steer adds the same |alpha| along either unit vector"
        ),
        "positive_control": POSITIVE_CONTROL,
        "direction": (
            direction_name
            if direction_name
            else "pluggable"
            if direction_fn is not None
            else "difference_of_means"
        ),
        "n_items": sum(1 for row in outcomes if row["key"] == BASELINE),
        "outcomes": outcomes,
        "contrasts": contrasts,
    }


def _row(
    record: Record,
    *,
    key: str,
    condition: str,
    position: str | None,
    recommended: str | None,
    result: DecodeResult,
    abs_position: int | None,
) -> dict:
    return {
        "key": key,
        "item_id": record.item_id,
        "scenario_id": record.scenario.scenario_id,
        "condition": condition,
        "position": position,
        "abs_position": abs_position,
        "recommended": recommended,
        "assigned_lean": record.condition.lean,
        "n_tokens": result.n_tokens,
        "prompt_len": result.prompt_len,
        "text": result.text,
    }


def json_ready(value):
    """Replace non-finite floats so the report is strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    return value
