"""J-lens direction extraction, workspace-band metrics, and static geometry."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unembedding-row access with caching
#
# bitsandbytes 4-bit lm_head weights only expose rows through W.dequantize(),
# which materializes the ENTIRE [vocab, d] matrix on every call. Per-token
# loops (interventions, Exp2 alignment) used to pay that cost once per token.
# We now dequantize once per weight object and cache it, keyed by id(W) with
# the weight itself kept in the cache entry so the id cannot be reused while
# the entry is alive. Call clear_unembed_cache() to free the memory.
# ---------------------------------------------------------------------------

_UNEMBED_DEQUANT_CACHE: dict[int, tuple[Any, torch.Tensor]] = {}


def clear_unembed_cache(weight: Any = None) -> None:
    """Free the cached dequantized unembedding matrix (all, or one weight's)."""
    if weight is None:
        _UNEMBED_DEQUANT_CACHE.clear()
    else:
        _UNEMBED_DEQUANT_CACHE.pop(id(weight), None)


def _cached_dequant(W: Any) -> torch.Tensor:
    """Dequantize W once per weight object and cache the CPU fp32 result."""
    key = id(W)
    hit = _UNEMBED_DEQUANT_CACHE.get(key)
    if hit is not None and hit[0] is W:
        return hit[1]
    full = W.dequantize().float().cpu()
    _UNEMBED_DEQUANT_CACHE[key] = (W, full)
    logger.debug("Dequantized unembedding matrix once: %s", tuple(full.shape))
    return full


def get_unembed_row(stack: Any, token_id: int) -> torch.Tensor:
    """Return unembedding vector for token_id as float32 [d_model] on CPU.

    4-bit (bnb) weights are dequantized once per weight object and cached
    module-level (see clear_unembed_cache); plain tensor weights and the
    lazy SafetensorsUnembedRows provider are indexed per row directly.
    """
    W = stack.lm_head.weight
    # 4-bit Linear: weight may be packed; dequantize once, cache, index rows
    if hasattr(W, "dequantize"):
        row = _cached_dequant(W)[token_id].float().cpu()
    else:
        try:
            row = W[token_id].float().detach().cpu()
        except Exception:
            # Fallback: one-hot through lm_head is wrong direction; use embedding if tied
            emb = stack.text_module.embed_tokens.weight
            row = emb[token_id].float().detach().cpu()
    return row


def get_unembed_rows(stack: Any, token_ids: Sequence[int]) -> torch.Tensor:
    """Return [n_tokens, d_model] float32 CPU rows for token_ids (batched)."""
    ids = [int(t) for t in token_ids]
    if not ids:
        raise ValueError("token_ids must be non-empty")
    W = stack.lm_head.weight
    idx = torch.tensor(ids, dtype=torch.long)
    if hasattr(W, "dequantize"):
        return _cached_dequant(W)[idx].float().cpu()
    if torch.is_tensor(W):
        return W[idx].float().detach().cpu()
    # Lazy row providers (e.g. SafetensorsUnembedRows)
    return torch.stack([get_unembed_row(stack, t) for t in ids], dim=0)


class SafetensorsUnembedRows:
    """Lazy, row-indexable unembedding matrix backed by local safetensors shards.

    Lets CPU-only analysis (ANALYZE_ONLY passes) read individual W_U rows
    without loading the quantized model or any GPU. Rows are fetched through
    ``get_slice`` so the full [vocab, d] tensor is never materialized; each
    fetched row is memoized. Use :meth:`from_row_cache` for a purely
    dict-backed instance (no shards) built from a saved row cache.
    """

    def __init__(self, shard_paths: Sequence[Path | str], tensor_name: str):
        self._shard_paths = [Path(p) for p in shard_paths]
        self.tensor_name = tensor_name
        self._rows: dict[int, torch.Tensor] = {}
        self._active_shard: Path | None = None
        self._shape: tuple[int, ...] | None = None

    @classmethod
    def from_row_cache(cls, rows: dict[int, torch.Tensor]) -> "SafetensorsUnembedRows":
        """Dict-backed provider over previously fetched rows (no shard reads)."""
        obj = cls([], tensor_name="<row-cache>")
        for tid, row in rows.items():
            obj._rows[int(tid)] = row.float().cpu()
        if obj._rows:
            obj._shape = (max(obj._rows) + 1, next(iter(obj._rows.values())).shape[0])
        return obj

    def _resolve_shard(self) -> Path:
        if self._active_shard is not None:
            return self._active_shard
        from safetensors import safe_open

        last_err: Exception | None = None
        for p in self._shard_paths:
            try:
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    if self.tensor_name in set(f.keys()):
                        self._active_shard = p
                        return p
            except Exception as e:  # pragma: no cover - environment dependent
                last_err = e
        raise FileNotFoundError(
            f"Tensor {self.tensor_name!r} not found in {self._shard_paths} ({last_err})"
        )

    def _load_row(self, tid: int) -> torch.Tensor:
        from safetensors import safe_open

        shard = self._resolve_shard()
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            sl = f.get_slice(self.tensor_name)
            if self._shape is None:
                self._shape = tuple(sl.get_shape())
            return sl[tid].to(torch.float32).cpu()

    def __getitem__(self, token_id: int | torch.Tensor) -> torch.Tensor:
        tid = int(token_id)
        row = self._rows.get(tid)
        if row is None:
            row = self._load_row(tid)
            self._rows[tid] = row
        return row

    def prefetch(self, token_ids: Sequence[int]) -> None:
        for t in token_ids:
            self.__getitem__(t)

    def export_rows(self) -> dict[int, torch.Tensor]:
        """Fetched rows as {token_id: fp16 [d] tensor} for torch.save caching."""
        return {tid: row.half() for tid, row in self._rows.items()}

    @property
    def n_rows_cached(self) -> int:
        return len(self._rows)

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is None:
            self._load_row(0)  # populates shape lazily
        return self._shape  # type: ignore[return-value]

    def detach(self) -> "SafetensorsUnembedRows":
        return self

    def cpu(self) -> "SafetensorsUnembedRows":
        return self


def _hf_cache_snapshots(model_id: str, cache_dir: str | None = None) -> list[Path]:
    """All locally cached snapshot revisions for a model (any of them may hold
    the weights when refs/main has moved to a newer, partially-downloaded rev)."""
    from huggingface_hub import constants

    base = Path(cache_dir) if cache_dir else Path(constants.HF_HUB_CACHE)
    repo_dir = base / ("models--" + model_id.replace("/", "--"))
    if not repo_dir.is_dir():
        return []
    return sorted((repo_dir / "snapshots").glob("*"))


def hf_safetensors_unembed(
    model_id: str, cache_dir: str | None = None
) -> SafetensorsUnembedRows | None:
    """Open W_U rows from locally cached HF safetensors (no download, no model load).

    Prefers a real ``lm_head.weight`` tensor; falls back to the (tied)
    ``model.embed_tokens.weight``. Looks in every cached snapshot revision, so
    a refs/main bump to a newer partial revision cannot hide the weights.
    Returns None when nothing local resolves.
    """
    from huggingface_hub import hf_hub_download

    found: dict[str, list[Path]] = {
        "model.safetensors.index.json": [],
        "model.safetensors": [],
    }
    # Preferred: hub-resolved paths (respect refs/main).
    for fname in list(found):
        try:
            p = hf_hub_download(
                model_id, filename=fname, cache_dir=cache_dir, local_files_only=True
            )
            if p:
                found[fname].append(Path(p))
        except Exception:
            pass
    # Fallback: any cached snapshot revision.
    for snap in _hf_cache_snapshots(model_id, cache_dir):
        for fname in list(found):
            p = snap / fname
            if p.is_file() and p not in found[fname]:
                found[fname].append(p)

    candidates: list[tuple[Path, str]] = []
    for name in ("lm_head.weight", "model.embed_tokens.weight"):
        for index in found["model.safetensors.index.json"]:
            try:
                wmap = json.loads(index.read_text(encoding="utf-8")).get(
                    "weight_map", {}
                )
            except Exception:
                continue
            shard = wmap.get(name)
            if not shard:
                continue
            p = index.parent / shard
            if p.is_file():
                candidates.append((p, name))
        for single in found["model.safetensors"]:
            candidates.append((single, name))

    for shard, name in candidates:
        provider = SafetensorsUnembedRows([shard], name)
        try:
            provider._resolve_shard()
            logger.info("Lazy unembedding rows from %s :: %s", shard.name, name)
            return provider
        except FileNotFoundError:
            continue
    return None


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
    rows = get_unembed_rows(stack, token_ids)
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


def per_token_mean_cosine(dirs: torch.Tensor) -> torch.Tensor:
    """Per-token mean cosine to the OTHER rows of ``dirs`` [n, d].

    The mean of the returned vector equals the scalar ``pairwise_cosine(dirs)``
    (each unordered pair contributes cos(i,j) once to token i and once to j),
    so these values are the per-token observations for bootstrap/permutation
    statistics over tokens.
    """
    n = dirs.shape[0]
    if n < 2:
        return torch.zeros(n, dtype=dirs.dtype)
    sim = pairwise_cosine_matrix(dirs)
    sim = sim.clone()
    sim.fill_diagonal_(0.0)
    return sim.sum(dim=1) / (n - 1)


def per_token_alignment(dirs: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Per-token fraction of direction variance inside a [k, d] orthonormal basis.

    The mean of the returned vector matches ``workspace_alignment`` called with
    the same basis, so these are the per-token observations for token-level
    bootstrap/permutation statistics.
    """
    basis = basis.float()
    proj = dirs.float() @ basis.T  # [n, k]
    var_in = (proj ** 2).sum(dim=-1)
    var_tot = (dirs.float() ** 2).sum(dim=-1).clamp_min(1e-12)
    return var_in / var_tot


def top_principal_basis(stack: Any, layer_idx: int) -> torch.Tensor:
    """Right singular vectors Vh of J_ℓ as [d, d] (rows sorted by singular value).

    The SVD of J does not depend on any token set — compute once per layer and
    reuse across token sets (see workspace_alignment(basis=...)).
    """
    J = stack.lens.jacobians[layer_idx].float()
    try:
        _U, _S, Vh = torch.linalg.svd(J, full_matrices=False)
    except Exception:
        _U, _S, Vh = torch.linalg.svd(J.cpu(), full_matrices=False)
    return Vh.cpu()


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
    *,
    basis: torch.Tensor | None = None,
) -> float:
    """Fraction of direction variance in top-k principal components of J_ℓ.

    ``basis``: precomputed right singular vectors of J_ℓ (see
    top_principal_basis) so the SVD runs once per layer instead of once per
    token set. When omitted, the SVD is computed here (backward compatible).
    """
    if basis is None:
        basis = top_principal_basis(stack, layer_idx)
    basis = basis[:k]  # [k, d]
    return float(per_token_alignment(dirs, basis).mean().item())
