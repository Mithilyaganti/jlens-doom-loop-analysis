"""Causal ablation of J-lens trigger directions in residual stream."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Sequence

import torch

from jspace.geometry import jlens_direction, jlens_direction_batch

logger = logging.getLogger(__name__)

DEFAULT_SUPPRESS_K = 25


def suppress_k_from_env(default: int = DEFAULT_SUPPRESS_K) -> int:
    """JLENS_SUPPRESS_K knob: logit-suppression window length in steps."""
    try:
        return int(os.environ.get("JLENS_SUPPRESS_K", str(default)))
    except ValueError:
        return default


def project_out(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Remove unit direction v from residual h. h: [..., d], v: [d]."""
    v = v.to(h.device, dtype=h.dtype)
    # normalize safety
    vn = v / v.norm().clamp_min(1e-12)
    # projection scalar: h · v
    coef = (h * vn).sum(dim=-1, keepdim=True)
    return h - coef * vn


def project_out_many(h: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Sequentially project out rows of directions [n, d] from h [..., d]."""
    out = h
    for i in range(directions.shape[0]):
        out = project_out(out, directions[i])
    return out


def ablate_trigger_direction(
    h_l: torch.Tensor,
    trigger_token_id: int,
    layer_idx: int,
    stack: Any,
) -> torch.Tensor:
    """Project out the trigger token's J-lens direction from residual at layer."""
    v = jlens_direction(stack, trigger_token_id, layer_idx, normalize=True)
    return project_out(h_l, v.to(h_l.dtype))


def orthonormalize(directions: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Gram-Schmidt a list of vectors into an orthonormal basis of their span.

    Near-linearly-dependent residuals (norm < 1e-6 of the original scale) are
    dropped so the projection stays numerically safe.
    """
    basis: list[torch.Tensor] = []
    for d in directions:
        v = d.detach().float()
        v = v / v.norm().clamp_min(1e-12)
        w = v.clone()
        for b in basis:
            w = w - (w @ b) * b
        nrm = w.norm()
        if float(nrm) > 1e-6:
            basis.append(w / nrm)
    return basis


def _project_out_basis(h: torch.Tensor, basis: list[torch.Tensor]) -> torch.Tensor:
    """Remove every orthonormal direction in ``basis`` from h [..., d]."""
    out = h
    for b in basis:
        v = b.to(device=h.device, dtype=h.dtype)
        coef = (out * v).sum(dim=-1, keepdim=True)
        out = out - coef * v
    return out


def make_onset_ablation_hook(
    directions: Sequence[torch.Tensor] | torch.Tensor,
    *,
    prompt_len: int | None = None,
    start_pos: int = 0,
    strength: float = 1.0,
) -> Callable:
    """Onset-timed forward hook: project directions out ONLY at generated
    sequence positions >= start_pos.

    ``start_pos`` is an index into the GENERATED sequence (0 = first generated
    token). ``prompt_len`` is the tokenized prompt length recorded before
    generation; the absolute sequence position of the tokens seen by each hook
    invocation is tracked as the cumulative input length across calls, so both
    use_cache=True incremental decoding (hook sees only the new tokens) and
    single full forwards (use_cache=False) are handled correctly.

    If ``prompt_len`` is None, ``start_pos`` is interpreted as an absolute
    sequence position (full-forward / no-cache use).

    Multiple directions are orthonormalized first, so the output has no
    component along ANY of them (sequential projection alone would not
    guarantee orthogonality to earlier directions).
    """
    if torch.is_tensor(directions):
        dirs = [directions[i] for i in range(directions.shape[0])] if directions.ndim == 2 else [directions]
    else:
        dirs = list(directions)
    if not dirs:
        raise ValueError("make_onset_ablation_hook needs at least one direction")
    if strength != 1.0:
        raise NotImplementedError("strength != 1.0 not supported for onset hooks")
    basis = orthonormalize(dirs)
    if not basis:
        raise ValueError("all directions numerically degenerate; cannot ablate")

    plen = int(prompt_len) if prompt_len is not None else 0
    threshold = int(start_pos)
    state = {"n_seen": 0}

    def hook(module, inputs, output):
        if torch.is_tensor(output):
            h = output
            rest = None
        else:
            h = output[0]
            rest = output[1:]

        seq_len = h.shape[1]
        pos_first = state["n_seen"]  # absolute position of h[:, 0, :]
        state["n_seen"] += seq_len

        # Generation index of the first token in this forward
        # (= absolute index when prompt_len is None).
        gen_first = pos_first - plen

        h = h.clone()
        if gen_first >= threshold:
            h = _project_out_basis(h, basis)
        elif gen_first + seq_len > threshold:
            s = threshold - gen_first  # 0 <= s < seq_len
            h[:, s:, :] = _project_out_basis(h[:, s:, :], basis)
        # else: every position in this forward is before the onset — untouched.

        if rest is None:
            return h
        return (h, *rest)

    return hook


def make_ablation_hook(
    stack: Any,
    layer_idx: int,
    direction: torch.Tensor,
    *,
    position: int | None = None,
    strength: float = 1.0,
) -> Callable:
    """Forward hook that projects out `direction` from residual at layer output.

    If position is set, only ablate that sequence position (absolute index into
    the current forward's sequence). If None, ablate all positions.
    strength: 1.0 = full projection out; 0.5 = half, etc.
    """
    v = direction.detach().float()
    v = v / v.norm().clamp_min(1e-12)

    def hook(module, inputs, output):
        if torch.is_tensor(output):
            h = output
            rest = None
        else:
            h = output[0]
            rest = output[1:]

        h = h.clone()
        v_dev = v.to(device=h.device, dtype=h.dtype)
        if position is not None:
            # position may be beyond current length during generation
            seq_len = h.shape[1]
            pos = position if position >= 0 else seq_len + position
            if 0 <= pos < seq_len:
                coef = (h[:, pos, :] * v_dev).sum(dim=-1, keepdim=True)
                h[:, pos, :] = h[:, pos, :] - strength * coef * v_dev
        else:
            coef = (h * v_dev).sum(dim=-1, keepdim=True)
            h = h - strength * coef * v_dev

        if rest is None:
            return h
        return (h, *rest)

    return hook


def make_multi_direction_hook(
    stack: Any,
    layer_idx: int,
    directions: Sequence[torch.Tensor],
    *,
    position: int | None = None,
    strength: float = 1.0,
) -> Callable:
    """Hook projecting out multiple directions (sequential Gram-Schmidt style)."""
    dirs = []
    for d in directions:
        v = d.detach().float()
        v = v / v.norm().clamp_min(1e-12)
        dirs.append(v)

    def hook(module, inputs, output):
        if torch.is_tensor(output):
            h = output
            rest = None
        else:
            h = output[0]
            rest = output[1:]
        h = h.clone()
        if position is not None:
            seq_len = h.shape[1]
            pos = position if position >= 0 else seq_len + position
            if 0 <= pos < seq_len:
                for v in dirs:
                    v_dev = v.to(device=h.device, dtype=h.dtype)
                    coef = (h[:, pos, :] * v_dev).sum(dim=-1, keepdim=True)
                    h[:, pos, :] = h[:, pos, :] - strength * coef * v_dev
        else:
            for v in dirs:
                v_dev = v.to(device=h.device, dtype=h.dtype)
                coef = (h * v_dev).sum(dim=-1, keepdim=True)
                h = h - strength * coef * v_dev
        if rest is None:
            return h
        return (h, *rest)

    return hook


class InterventionContext:
    """Register ablation hooks on workspace-band layers for one generation.

    Onset-timed mode (Exp3 fixed protocol): pass ``prompt_len`` (tokenized
    prompt length) and ``start_pos`` (index into the GENERATED sequence).
    The hook then projects the directions out only at generated positions
    >= start_pos, tracking positions across incremental (use_cache=True)
    forwards. Without start_pos, the legacy behavior is preserved
    (position=None -> ablate every position).
    """

    def __init__(
        self,
        stack: Any,
        layer_indices: Sequence[int],
        directions_by_layer: dict[int, torch.Tensor] | dict[int, list[torch.Tensor]],
        *,
        position: int | None = None,
        strength: float = 1.0,
        prompt_len: int | None = None,
        start_pos: int | None = None,
    ):
        self.stack = stack
        self.layer_indices = list(layer_indices)
        self.directions_by_layer = directions_by_layer
        self.position = position
        self.strength = strength
        self.prompt_len = prompt_len
        self.start_pos = start_pos
        self._handles: list = []

    def __enter__(self):
        for l in self.layer_indices:
            if l not in self.directions_by_layer:
                continue
            d = self.directions_by_layer[l]
            if isinstance(d, list) or (torch.is_tensor(d) and d.ndim == 2):
                dirs = d if isinstance(d, list) else [d[i] for i in range(d.shape[0])]
            else:
                dirs = [d]
            if self.start_pos is not None:
                if self.prompt_len is None:
                    raise ValueError("start_pos requires prompt_len (tokenized prompt length)")
                hook = make_onset_ablation_hook(
                    dirs,
                    prompt_len=self.prompt_len,
                    start_pos=self.start_pos,
                    strength=self.strength,
                )
            elif len(dirs) > 1:
                hook = make_multi_direction_hook(
                    self.stack, l, dirs, position=self.position, strength=self.strength
                )
            else:
                hook = make_ablation_hook(
                    self.stack, l, dirs[0], position=self.position, strength=self.strength
                )
            handle = self.stack.layers[l].register_forward_hook(hook)
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []


def random_direction(d_model: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g)
    return v / v.norm().clamp_min(1e-12)


try:  # HF base class when available; duck-typed otherwise (CPU tests).
    from transformers import LogitsProcessor as _SUPPRESS_BASE
except Exception:  # pragma: no cover - transformers is always in the venv
    _SUPPRESS_BASE = object  # type: ignore[assignment,misc]


class TriggerSuppressLogitsProcessor(_SUPPRESS_BASE):  # type: ignore[misc,valid-type]
    """Positive control — trigger-token logit suppression.

    Once generation reaches ``start_pos`` (index into the GENERATED sequence),
    add ``bias`` (default -50.0) to the trigger token's logit for the next
    ``k_steps`` decode steps, then stop suppressing. The model may still loop
    afterwards — that is the point: we measure the loop rate.

    HF ``generate`` calls logits processors with the FULL growing input_ids
    each step, so ``input_ids`` length - prompt_len gives the index of the
    token about to be produced — correct position bookkeeping even with
    ``use_cache=True``. Works via ``model.generate(..., logits_processor=[...])``.

    ``k_steps=None`` -> JLENS_SUPPRESS_K env (default 25).
    ``prompt_len=None`` -> snapshot the length at the first call (at step 0
    HF passes exactly the prompt).
    """

    def __init__(
        self,
        trigger_token_id: int,
        *,
        start_pos: int,
        k_steps: int | None = None,
        bias: float = -50.0,
        prompt_len: int | None = None,
    ):
        self.trigger_token_id = int(trigger_token_id)
        self.start_pos = int(start_pos)
        self.k_steps = int(k_steps) if k_steps is not None else suppress_k_from_env()
        self.bias = float(bias)
        self.prompt_len = None if prompt_len is None else int(prompt_len)
        self._first_len: int | None = None
        self.n_suppressed_steps = 0

    def window(self) -> tuple[int, int]:
        """Suppression window as [start, end) generation indices."""
        return (self.start_pos, self.start_pos + self.k_steps)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        seq_len = int(input_ids.shape[-1])
        if self.prompt_len is not None:
            plen = self.prompt_len
        else:
            if self._first_len is None:
                self._first_len = seq_len
            plen = self._first_len
        # 0-based generation index of the token about to be produced.
        next_step = seq_len - plen
        if self.start_pos <= next_step < self.start_pos + self.k_steps:
            scores = scores.clone()
            scores[..., self.trigger_token_id] = scores[..., self.trigger_token_id] + self.bias
            self.n_suppressed_steps += 1
        return scores


def build_trigger_directions(
    stack: Any,
    trigger_token_id: int,
    layer_indices: Sequence[int],
    *,
    extra_token_ids: Sequence[int] = (),
) -> dict[int, list[torch.Tensor]]:
    """Per-trace ablation directions: the trace's trigger token J-direction
    at each layer, optionally followed by extra (e.g. global top-k) trigger
    directions (duplicates of the trace token are skipped)."""
    out: dict[int, list[torch.Tensor]] = {}
    for l in layer_indices:
        dirs = [jlens_direction(stack, trigger_token_id, l, normalize=True)]
        for tid in extra_token_ids:
            if int(tid) == int(trigger_token_id):
                continue
            dirs.append(jlens_direction(stack, int(tid), l, normalize=True))
        out[l] = dirs
    return out


def build_condition_directions(
    stack: Any,
    condition: str,
    layer_indices: Sequence[int],
    *,
    trigger_token_ids: Sequence[int],
    control_token_ids: Sequence[int] | None = None,
    n_directions: int = 1,
    seed: int = 0,
) -> dict[int, list[torch.Tensor]]:
    """Build per-layer directions to ablate for a named experimental condition.

    Conditions:
      - baseline: empty (caller should not register hooks)
      - ablate_trigger
      - ablate_random
      - ablate_control
      - ablate_trigger_sensory / ablate_trigger_motor use same dirs; layers differ
    """
    out: dict[int, list[torch.Tensor]] = {}
    for l in layer_indices:
        if condition in ("baseline", "none", "no_intervention"):
            continue
        if condition in (
            "ablate_trigger",
            "ablate_trigger_sensory",
            "ablate_trigger_motor",
            "ablate_trigger_workspace",
        ):
            ids = list(trigger_token_ids)[:n_directions]
            dirs = [jlens_direction(stack, tid, l, normalize=True) for tid in ids]
        elif condition == "ablate_random":
            dirs = [random_direction(stack.d_model, seed=seed + l * 100 + i) for i in range(n_directions)]
        elif condition == "ablate_control":
            ids = list(control_token_ids or [])[:n_directions]
            if not ids:
                dirs = [random_direction(stack.d_model, seed=seed + 999 + l)]
            else:
                dirs = [jlens_direction(stack, tid, l, normalize=True) for tid in ids]
        else:
            raise ValueError(f"Unknown condition: {condition}")
        out[l] = dirs
    return out
