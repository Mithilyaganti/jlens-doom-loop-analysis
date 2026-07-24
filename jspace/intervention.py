"""Causal ablation of J-lens trigger directions in residual stream."""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import torch

from jspace.geometry import jlens_direction, jlens_direction_batch

logger = logging.getLogger(__name__)


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
    """Register ablation hooks on workspace-band layers for one generation."""

    def __init__(
        self,
        stack: Any,
        layer_indices: Sequence[int],
        directions_by_layer: dict[int, torch.Tensor] | dict[int, list[torch.Tensor]],
        *,
        position: int | None = None,
        strength: float = 1.0,
    ):
        self.stack = stack
        self.layer_indices = list(layer_indices)
        self.directions_by_layer = directions_by_layer
        self.position = position
        self.strength = strength
        self._handles: list = []

    def __enter__(self):
        for l in self.layer_indices:
            if l not in self.directions_by_layer:
                continue
            d = self.directions_by_layer[l]
            if isinstance(d, list) or (torch.is_tensor(d) and d.ndim == 2):
                dirs = d if isinstance(d, list) else [d[i] for i in range(d.shape[0])]
                hook = make_multi_direction_hook(
                    self.stack, l, dirs, position=self.position, strength=self.strength
                )
            else:
                hook = make_ablation_hook(
                    self.stack, l, d, position=self.position, strength=self.strength
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
