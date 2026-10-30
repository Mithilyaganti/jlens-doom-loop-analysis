"""CPU test double for the causal flip test.

Not a language model and not a stand-in for Qwen. The residual at every
position is ``coef * planted``. ``coef`` is +1 when the prompt contains the
lean marker and -0.4 when the lean prompt is absent. Option A wins only when
the last-position projection onto ``planted`` clears a small margin, so
removing that component (or steering against it) changes the argmax, and a
norm-matched random direction does not.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# A wins only when proj > MARGIN. Exact ablation leaves proj at 0, below the margin.
MARGIN = 0.05
LEAN_MARKER = "LEAN_A"


class TinyFlipModel(nn.Module):
    """One identity layer. Hooks on ``layers[0]`` edit the residual the logits read.

    ``decision_at="P0"``: the first generated token is the option, decided by
    the last prompt position.

    ``decision_at="P1"``: the first generated token is a fixed preamble. The
    option is the second token, decided by the preamble position (the token
    just before the option word). Editing P0 does not change that token.
    """

    A_ID = 5
    B_ID = 6
    PREAMBLE_ID = 3
    LEAN_ID = 1
    OTHER_ID = 2

    def __init__(self, planted: torch.Tensor, *, decision_at: str = "P0"):
        super().__init__()
        if decision_at not in ("P0", "P1"):
            raise ValueError("decision_at must be 'P0' or 'P1'")
        vector = planted.detach().float().reshape(-1)
        vector = vector / vector.norm().clamp_min(1e-12)
        self.register_buffer("planted", vector)
        self.decision_at = decision_at
        self.layers = nn.ModuleList([nn.Identity()])
        self.vocab = 8

    def forward(self, input_ids: torch.Tensor, **kwargs):
        batch, seq = input_ids.shape
        width = int(self.planted.numel())
        lean = (input_ids == self.LEAN_ID).any(dim=-1).float()
        coef = torch.where(lean > 0, torch.ones(batch, dtype=torch.float32), torch.full((batch,), -0.4))
        hidden = coef[:, None, None] * self.planted.view(1, 1, -1)
        hidden = hidden.expand(batch, seq, width).clone()
        for layer in self.layers:
            hidden = layer(hidden)
        logits = hidden.new_full((batch, seq, self.vocab), -1.0e4)
        proj = (hidden * self.planted.view(1, 1, -1)).sum(dim=-1)
        if self.decision_at == "P0":
            logits[..., self.A_ID] = proj - MARGIN
            logits[..., self.B_ID] = 0
        else:
            has_preamble = (input_ids == self.PREAMBLE_ID).any(dim=-1)
            floor = torch.where(
                has_preamble[:, None],
                torch.zeros_like(proj),
                torch.full_like(proj, -1.0e4),
            )
            preamble_logit = torch.where(
                has_preamble[:, None],
                torch.full_like(proj, -1.0e4),
                torch.zeros_like(proj),
            )
            logits[..., self.PREAMBLE_ID] = preamble_logit
            logits[..., self.A_ID] = torch.where(has_preamble[:, None], proj - MARGIN, floor)
            logits[..., self.B_ID] = floor
        return _Logits(logits)


class _Logits:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


def tiny_tokenize(text: str) -> torch.Tensor:
    """Four tokens. The last one is the lean marker when the prompt carries it."""
    marker = TinyFlipModel.LEAN_ID if LEAN_MARKER in text else TinyFlipModel.OTHER_ID
    return torch.tensor([[7, 7, 7, marker]], dtype=torch.long)


def option_token_ids() -> dict[int, str]:
    return {TinyFlipModel.A_ID: "A", TinyFlipModel.B_ID: "B"}
