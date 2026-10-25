"""CPU-only unit tests for the new geometry/analysis helpers used by Exp1/Exp2.

Covers: per-token pairwise-cosine extraction, per-token workspace alignment,
SVD-once basis reuse, unembedding-row caching, lazy row providers, and the
Exp2 readout<->generated-token mapping / anchor-finding helpers.

Run: python tests/test_analysis_stats.py
"""

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace.geometry import (  # noqa: E402
    SafetensorsUnembedRows,
    clear_unembed_cache,
    get_unembed_row,
    get_unembed_rows,
    jlens_direction_batch,
    pairwise_cosine,
    per_token_alignment,
    per_token_mean_cosine,
    top_principal_basis,
    workspace_alignment,
)
from jspace.analysis import (  # noqa: E402
    first_occurrence_anchor,
    plot_lines_by_layer,
    position_matched_anchor,
    readout_gen_offset,
)


def _rand_unit(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True)


class _FakeQuantWeight:
    """lm_head.weight stand-in whose dequantize() materializes a full matrix."""

    def __init__(self, full: torch.Tensor):
        self._full = full
        self.dequant_calls = 0

    def dequantize(self) -> torch.Tensor:
        self.dequant_calls += 1
        return self._full.clone()


def _fake_stack(weight, d: int = 16, n_layers: tuple = (0,)):
    torch.manual_seed(0)
    J = {l: torch.eye(d) + 0.01 * torch.randn(d, d) for l in n_layers}
    lens = SimpleNamespace(jacobians=J, source_layers=sorted(J))
    return SimpleNamespace(lens=lens, lm_head=SimpleNamespace(weight=weight), device="cpu")


# ---------------------------------------------------------------------------
# Per-token geometry statistics
# ---------------------------------------------------------------------------


def test_per_token_mean_cosine_matches_pairwise_mean():
    dirs = _rand_unit(7, 12, seed=1)
    per_tok = per_token_mean_cosine(dirs)
    assert per_tok.shape == (7,)
    scalar = float(pairwise_cosine(dirs))
    assert abs(float(per_tok.mean()) - scalar) < 1e-6
    # n<2 -> zeros
    assert torch.equal(per_token_mean_cosine(dirs[:1]), torch.zeros(1))


def test_per_token_alignment_matches_workspace_alignment():
    d = 24
    stack = _fake_stack(torch.zeros(0), d=d, n_layers=(3,))
    dirs = _rand_unit(9, d, seed=2)
    basis = top_principal_basis(stack, 3)
    per_tok = per_token_alignment(dirs, basis[:8])
    assert per_tok.shape == (9,)
    assert float(per_tok.min()) >= 0.0
    # basis=... path must equal the default (in-function SVD) path
    a1 = workspace_alignment(dirs, stack, 3, k=8)
    a2 = workspace_alignment(dirs, stack, 3, k=8, basis=basis)
    assert abs(a1 - a2) < 1e-8
    assert abs(float(per_tok.mean()) - a1) < 1e-6
    # Identity J: full-rank basis keeps ALL variance in-basis.
    eye_stack = SimpleNamespace(lens=SimpleNamespace(jacobians={0: torch.eye(d)}, source_layers=[0]))
    full = per_token_alignment(dirs, top_principal_basis(eye_stack, 0))
    assert torch.allclose(full, torch.ones(9), atol=1e-5)


def test_basis_reuse_is_exact():
    d = 20
    stack = _fake_stack(None, d=d, n_layers=(1,))
    b1 = top_principal_basis(stack, 1)
    b2 = top_principal_basis(stack, 1)
    assert torch.equal(b1, b2)
    assert b1.shape == (d, d)


# ---------------------------------------------------------------------------
# Unembedding-row caching
# ---------------------------------------------------------------------------


def test_get_unembed_row_dequantizes_once():
    try:
        full = torch.randn(50, 8)
        W = _FakeQuantWeight(full)
        stack = _fake_stack(W, d=8)
        r1 = get_unembed_row(stack, 4)
        r2 = get_unembed_row(stack, 4)
        r3 = get_unembed_row(stack, 9)
        assert W.dequant_calls == 1  # cached, not per-call
        assert torch.allclose(r1, full[4].float())
        assert torch.allclose(r2, full[4].float())
        assert torch.allclose(r3, full[9].float())
        rows = get_unembed_rows(stack, [4, 9])
        assert rows.shape == (2, 8)
        assert W.dequant_calls == 1
        # Freeing works and forces a fresh dequant on next use.
        clear_unembed_cache(W)
        get_unembed_row(stack, 4)
        assert W.dequant_calls == 2
    finally:
        clear_unembed_cache()


def test_get_unembed_row_plain_tensor_and_lazy_provider():
    W = torch.randn(40, 6)
    stack = _fake_stack(W, d=6)
    assert torch.allclose(get_unembed_row(stack, 3), W[3].float())
    rows = get_unembed_rows(stack, [3, 0])
    assert torch.allclose(rows, W[[3, 0]].float())

    provider = SafetensorsUnembedRows.from_row_cache({3: W[3].half(), 7: W[7].half()})
    stack2 = _fake_stack(provider, d=6)
    # Row cache stores fp16; compare against the fp16-rounded originals.
    assert torch.allclose(get_unembed_row(stack2, 7), W[7].half().float())
    assert torch.allclose(get_unembed_rows(stack2, [3, 7]), W[[3, 7]].half().float())
    try:
        get_unembed_row(stack2, 1)  # not cached, no shards
        raised = False
    except Exception:
        raised = True
    assert raised


def test_jlens_direction_batch_with_lazy_rows():
    d = 8
    W = torch.randn(30, d)
    provider = SafetensorsUnembedRows.from_row_cache({i: W[i].half() for i in range(30)})
    stack = _fake_stack(provider, d=d, n_layers=(0,))
    dirs = jlens_direction_batch(stack, [1, 2, 3], 0, normalize=True)
    assert dirs.shape == (3, d)
    assert torch.allclose(dirs.norm(dim=-1), torch.ones(3), atol=1e-5)


# ---------------------------------------------------------------------------
# Exp2 anchor/mapping helpers
# ---------------------------------------------------------------------------


def test_readout_gen_offset():
    # Untruncated with matching sizes -> direct mapping.
    assert readout_gen_offset({}, 10, 100, 100) == 0
    # Untruncated mismatch (legacy capped cache) -> ambiguous.
    assert readout_gen_offset({}, 10, 1059, 1024) is None
    # Trigger-centered window starting at 0 -> offset 0.
    meta = {"trunc": {"truncated": True, "window_start": 0, "prompt_len": 299}}
    assert readout_gen_offset(meta, 299, 4000, 964) == 0
    # Tail window past the prompt -> offset = window_start - prompt_len.
    meta = {"trunc": {"truncated": True, "window_start": 2763, "prompt_len": 299}}
    assert readout_gen_offset(meta, 299, 4000, 1536) == 2763 - 299
    # Window covering the prompt start -> offset 0.
    meta = {"trunc": {"truncated": True, "window_start": 0, "prompt_len": 299}}
    assert readout_gen_offset(meta, None, 4000, 1536) == 0
    # Inconsistent sizes -> None.
    meta = {"trunc": {"truncated": True, "window_start": 3900, "prompt_len": 299}}
    assert readout_gen_offset(meta, 299, 4000, 1536) is None


def test_first_occurrence_anchor():
    gen = [5, 6, 5, 7, 6, 5, 8]
    assert first_occurrence_anchor(gen, 6, 0, 7) == 1
    assert first_occurrence_anchor(gen, 8, 0, 7) == 6
    assert first_occurrence_anchor(gen, 9, 0, 7) is None
    # Occurrence before the cached window: first IN-WINDOW occurrence returned.
    assert first_occurrence_anchor(gen, 5, 2, 5) == 0  # gen idx 2 -> readout 0
    assert first_occurrence_anchor(gen, 5, 3, 4) == 2  # gen idx 5 -> readout 2
    # Token only occurs before the window -> None.
    assert first_occurrence_anchor(gen, 7, 4, 4) is None


def test_position_matched_anchor():
    assert position_matched_anchor(0.5, 101) == 50
    assert position_matched_anchor(0.0, 101) == 5  # clamped to min_pos
    assert position_matched_anchor(1.0, 101) == 100
    assert position_matched_anchor(0.3, 11) == 5
    assert position_matched_anchor(0.5, 5) is None  # window too short
    assert position_matched_anchor(0.5, 6) == 5


def test_plot_lines_by_layer_accepts_yerr():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "lines.png"
        layers = [0, 1, 2]
        series = {"a": [0.1, 0.2, 0.3], "b": [0.2, 0.1, 0.4]}
        yerr = {"a": [0.01, 0.02, 0.01]}
        plot_lines_by_layer(
            layers, series, ylabel="y", title="t", out_path=out, yerr=yerr
        )
        assert out.is_file() and out.stat().st_size > 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            except Exception as exc:  # pragma: no cover
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
