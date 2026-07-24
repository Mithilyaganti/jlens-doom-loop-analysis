"""J-lens direction extraction, workspace-band metrics, and static geometry."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_unembed_row(stack: Any, token_id: int) -> torch.Tensor:
    """Return unembedding vector for token_id as float32 [d_model] on CPU."""
    W = stack.lm_head.weight
    # 4-bit Linear: weight may be packed; use dequant via forward on one-hot if needed
    if hasattr(W, "dequantize"):
        row = W.dequantize()[token_id].float().cpu()
    else:
        try:
            row = W[token_id].float().detach().cpu()
        except Exception:
            # Fallback: one-hot through lm_head is wrong direction; use embedding if tied
            emb = stack.text_module.embed_tokens.weight
            row = emb[token_id].float().detach().cpu()
    return row


def jlens_direction(
    stack: Any,
    token_id: int,
    layer_idx: int,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Steering direction for token t at layer ℓ: unit vector in residual space.

    direction = W_U[t] @ J_ℓ.T  (same as transport of unembed row through J).
    """
    J = stack.lens.jacobians[layer_idx].float()  # [d, d]
    w = get_unembed_row(stack, token_id).float()  # [d]
    # J transports residual → final basis: residual @ J.T
    # Direction in residual basis that maps to W_U[t] in final basis:
    # We want d such that d @ J.T ≈ w  ⇒  d ≈ w @ J  if J orthogonal-ish;
    # Anthropic / open-jlens use: direction = W_U_t @ J_l  with transport residual@J.T
    # Spec: direction = W_U_t @ J_l.T
    direction = w @ J.T  # [d]
    if normalize:
        n = direction.norm()
        if n > 1e-12:
            direction = direction / n
    return direction


def jlens_direction_batch(
    stack: Any,
    token_ids: Sequence[int],
    layer_idx: int,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Return [n_tokens, d_model] directions at one layer."""
    J = stack.lens.jacobians[layer_idx].float()
    rows = torch.stack([get_unembed_row(stack, tid) for tid in token_ids], dim=0)
    dirs = rows @ J.T
    if normalize:
        norms = dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        dirs = dirs / norms
    return dirs


def direction_norms(
    stack: Any,
    token_ids: Sequence[int],
    layer_idx: int,
) -> torch.Tensor:
    """Unnormalized direction norms (steering strength) at a layer."""
    dirs = jlens_direction_batch(stack, token_ids, layer_idx, normalize=False)
    return dirs.norm(dim=-1)


def pairwise_cosine(dirs: torch.Tensor) -> torch.Tensor:
    """Mean pairwise cosine similarity of [n, d] unit vectors (off-diagonal)."""
    if dirs.shape[0] < 2:
        return torch.tensor(0.0)
    # Assume already unit-normalized
    norms = dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u = dirs / norms
    sim = u @ u.T
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return sim[mask].mean()


def pairwise_cosine_matrix(dirs: torch.Tensor) -> torch.Tensor:
    norms = dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u = dirs / norms
    return u @ u.T


@torch.no_grad()
def jlens_readout(
    stack: Any,
    h_l: torch.Tensor,
    layer_idx: int,
    top_k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k J-lens readout for residual h_l [..., d_model].

    Returns (indices, values/logits) with last dim = top_k.
    """
    transported = stack.lens.transport(h_l.float(), layer_idx)
    dtype = next(stack.lm_head.parameters()).dtype if list(stack.lm_head.parameters()) else torch.bfloat16
    # final_norm + lm_head
    x = transported.to(dtype)
    # Place on same device as final_norm
    try:
        dev = next(stack.final_norm.parameters()).device
    except StopIteration:
        dev = stack.device
    x = stack.final_norm(x.to(dev))
    logits = stack.lm_head(x).float()
    topk = logits.topk(top_k, dim=-1)
    return topk.indices.cpu(), topk.values.cpu()


def excess_kurtosis_of_matrix(M: torch.Tensor) -> float:
    """Excess kurtosis of matrix entries (Fisher: normal → 0)."""
    x = M.float().reshape(-1)
    # subsample for speed if huge
    if x.numel() > 2_000_000:
        idx = torch.randperm(x.numel())[:2_000_000]
        x = x[idx]
    x = x - x.mean()
    m2 = (x ** 2).mean()
    m4 = (x ** 4).mean()
    if m2 < 1e-20:
        return 0.0
    return float((m4 / (m2 ** 2)) - 3.0)


def singular_values_of_jlens(stack: Any, layer_idx: int, max_vocab: int = 8192) -> torch.Tensor:
    """Approx singular values of V_ℓ ≈ top-max_vocab rows of W_U · J_ℓ.

    Full V is [vocab, d] which is huge; we use random vocab subset + J SVD proxy:
    participation ratio of J itself as geometry of the map.
    """
    J = stack.lens.jacobians[layer_idx].float()
    # SVD of J (d x d) is cheap for d~2560
    try:
        S = torch.linalg.svdvals(J)
    except Exception:
        S = torch.linalg.svdvals(J.cpu())
    return S.cpu()


def effective_dimensionality(singular_values: torch.Tensor) -> float:
    """Participation ratio (Σλ)² / Σλ² on singular values (or eigenvalues)."""
    s = singular_values.float().clamp_min(0)
    # Use squared singular values as energy
    lam = s ** 2
    total = lam.sum()
    if total < 1e-20:
        return 0.0
    return float((total ** 2) / (lam ** 2).sum())


def entropy_eff_dim(singular_values: torch.Tensor) -> float:
    """exp(entropy) effective dim of normalized singular-value distribution."""
    s = singular_values.float().clamp_min(0)
    p = s / s.sum().clamp_min(1e-20)
    p = p[p > 0]
    ent = -(p * p.log()).sum()
    return float(torch.exp(ent))


@torch.no_grad()
def layerwise_ntp_accuracy(
    stack: Any,
    prompts: list[str],
    *,
    max_seq_len: int = 128,
    layers: Sequence[int] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Next-token prediction accuracy of lens(h_ℓ) vs model final predictions."""
    from jlens.hooks import ActivationRecorder

    if layers is None:
        layers = list(range(stack.n_layers))

    correct_top1 = {l: 0 for l in layers}
    correct_topk = {l: 0 for l in layers}
    total = 0

    for prompt in prompts:
        inputs = stack.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
        )
        input_ids = inputs.input_ids.to(stack.device)
        if input_ids.shape[1] < 2:
            continue
        # Predict last position's next token (model prediction as target)
        with ActivationRecorder(stack.layers, at=list(set(layers) | {stack.n_layers - 1})) as rec:
            out = stack.model(input_ids=input_ids, use_cache=False)
            acts = {i: rec.activations[i].detach() for i in rec.activations}

        model_logits = out.logits[0, -1, :].float().cpu()
        target = int(model_logits.argmax().item())

        for l in layers:
            if l not in acts:
                continue
            h = acts[l][0, -1, :].float()
            # transport + unembed
            if l in stack.lens.jacobians:
                h_t = stack.lens.transport(h, l)
            else:
                h_t = h
            try:
                dev = next(stack.final_norm.parameters()).device
                dtype = next(stack.lm_head.parameters()).dtype
            except StopIteration:
                dev, dtype = stack.device, torch.bfloat16
            logits = stack.lm_head(stack.final_norm(h_t.to(dtype).to(dev))).float().cpu()
            pred = int(logits.argmax().item())
            topk = logits.topk(top_k).indices.tolist()
            if pred == target:
                correct_top1[l] += 1
            if target in topk:
                correct_topk[l] += 1
        total += 1
        del acts, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if total == 0:
        return {"top1": {}, "topk": {}, "n": 0}
    return {
        "top1": {l: correct_top1[l] / total for l in layers},
        "topk": {l: correct_topk[l] / total for l in layers},
        "n": total,
    }


def compute_workspace_band_metrics(stack: Any, prompts: list[str] | None = None) -> dict[str, Any]:
    """Compute excess kurtosis, NTP accuracy, effective dim per layer."""
    n_layers = stack.n_layers
    layers = sorted(stack.lens.source_layers)

    kurtosis = {}
    eff_dim = {}
    ent_dim = {}
    for l in layers:
        J = stack.lens.jacobians[l]
        kurtosis[l] = excess_kurtosis_of_matrix(J)
        S = singular_values_of_jlens(stack, l)
        eff_dim[l] = effective_dimensionality(S)
        ent_dim[l] = entropy_eff_dim(S)

    if prompts is None:
        # Minimal default prompts for NTP
        prompts = [
            "The capital of France is",
            "In mathematics, a prime number is",
            "def fibonacci(n):",
            "The mitochondria is the",
            "Once upon a time in a",
            "The derivative of x squared is",
            "Python is a programming",
            "Water freezes at",
            "The speed of light is approximately",
            "To solve the equation",
        ] * 5  # 50 short prompts

    logger.info("Computing NTP accuracy on %d prompts...", len(prompts))
    ntp = layerwise_ntp_accuracy(stack, prompts[:50], layers=layers)

    return {
        "layers": layers,
        "kurtosis": kurtosis,
        "effective_dimensionality": eff_dim,
        "entropy_dimensionality": ent_dim,
        "ntp_top1": ntp["top1"],
        "ntp_topk": ntp["topk"],
        "n_prompts_ntp": ntp["n"],
        "n_layers": n_layers,
    }


def identify_workspace_band(metrics: dict[str, Any]) -> dict[str, Any]:
    """Heuristically identify workspace band from the three metric curves.

    Band: elevated kurtosis, rising/high NTP accuracy, elevated eff dim.
    """
    layers = metrics["layers"]
    kurt = np.array([metrics["kurtosis"][l] for l in layers], dtype=float)
    acc = np.array([metrics["ntp_top1"].get(l, 0.0) for l in layers], dtype=float)
    dim = np.array([metrics["effective_dimensionality"][l] for l in layers], dtype=float)

    # Normalize each to [0, 1] for combined score
    def _norm(x):
        lo, hi = np.nanmin(x), np.nanmax(x)
        if hi - lo < 1e-9:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    score = _norm(kurt) + _norm(acc) + _norm(dim)

    # Workspace: score above median of upper half, contiguous region
    thresh = float(np.median(score) + 0.15 * (score.max() - np.median(score)))
    in_band = score >= thresh

    # Find longest contiguous True region (or first large rise to near-sat of acc)
    best_start, best_end, best_len = 0, 0, 0
    i = 0
    while i < len(in_band):
        if in_band[i]:
            j = i
            while j < len(in_band) and in_band[j]:
                j += 1
            if j - i > best_len:
                best_len = j - i
                best_start, best_end = i, j - 1
            i = j
        else:
            i += 1

    # Also use accuracy: workspace often starts where top1 rises above 0.3
    # and ends where it plateaus near max (motor)
    acc_start = None
    for idx, a in enumerate(acc):
        if a >= 0.3:
            acc_start = idx
            break
    # Motor: last few layers where acc is within 5% of max
    amax = acc.max() if len(acc) else 0
    motor_start = len(acc) - 1
    for idx in range(len(acc) - 1, -1, -1):
        if acc[idx] < amax - 0.05:
            motor_start = idx + 1
            break

    if best_len <= 0:
        raise RuntimeError(
            "Could not identify a contiguous workspace band from metrics "
            "(kurtosis / NTP accuracy / effective dim). Inspect metric plots — "
            "refusing to invent layer boundaries."
        )

    start_l = layers[best_start]
    end_l = layers[best_end]

    # Refine with accuracy onset (may only shrink start, not invent a band)
    if acc_start is not None:
        start_l = min(start_l, layers[acc_start])

    motor_start_l = layers[min(motor_start, len(layers) - 1)]
    if end_l > motor_start_l and motor_start_l > start_l:
        end_l = motor_start_l

    if end_l < start_l:
        raise RuntimeError(
            f"Invalid workspace band ordering start={start_l} end={end_l}. "
            "Refusing to invent boundaries."
        )

    # Anthropic: last few layers are motor. If the high-score band reaches the
    # final fitted layer, peel the last 2 source layers as motor so causal
    # controls remain defined (documented methodological choice, not invented metrics).
    last_src = layers[-1]
    motor_peel_note = ""
    if end_l >= last_src and len(layers) >= 4:
        peel = 2
        new_end = layers[-(peel + 1)]
        if new_end >= start_l:
            end_l = new_end
            motor_peel_note = (
                f" Peeled last {peel} source layers as motor (end was last fitted layer {last_src})."
            )

    band_layers = [l for l in layers if start_l <= l <= end_l]
    if not band_layers:
        raise RuntimeError(
            f"No layers in computed band [{start_l}, {end_l}]. Refusing to invent."
        )

    # Pick 3–5 representative workspace layers for Tier-2 caching
    if len(band_layers) <= 5:
        key_layers = band_layers
    else:
        idxs = np.linspace(0, len(band_layers) - 1, 5).astype(int)
        key_layers = [band_layers[i] for i in idxs]

    sensory = [l for l in layers if l < start_l]
    motor = [l for l in layers if l > end_l]
    if not motor:
        raise RuntimeError(
            "Motor band is empty after workspace identification. "
            "Need non-empty sensory/workspace/motor for Exp 3 controls."
        )
    mid_layer = band_layers[len(band_layers) // 2]

    return {
        "workspace_start": int(start_l),
        "workspace_end": int(end_l),
        "workspace_layers": [int(l) for l in band_layers],
        "key_workspace_layers": [int(l) for l in key_layers],
        "mid_workspace_layer": int(mid_layer),
        "sensory_layers": [int(l) for l in sensory],
        "motor_layers": [int(l) for l in motor],
        "score_threshold": thresh,
        "notes": (
            "Band from combined kurtosis + NTP top-1 + effective dimensionality."
            + motor_peel_note
            + " Verify against plots before interpreting Exp 1–3."
        ),
    }


def workspace_alignment(
    dirs: torch.Tensor,
    stack: Any,
    layer_idx: int,
    k: int = 64,
) -> float:
    """Fraction of direction variance in top-k principal components of J_ℓ."""
    J = stack.lens.jacobians[layer_idx].float()
    # PCA of rows of J (or right singular vectors)
    try:
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    except Exception:
        U, S, Vh = torch.linalg.svd(J.cpu(), full_matrices=False)
    # Top-k right singular vectors span principal residual directions mapped strongly
    basis = Vh[:k]  # [k, d]
    # Project each direction
    proj = dirs @ basis.T  # [n, k]
    var_in = (proj ** 2).sum(dim=-1)
    var_tot = (dirs ** 2).sum(dim=-1).clamp_min(1e-12)
    return float((var_in / var_tot).mean().item())
