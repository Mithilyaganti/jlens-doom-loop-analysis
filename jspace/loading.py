"""Model + J-lens loading for 8GB VRAM (bitsandbytes NF4)."""

from __future__ import annotations

import gc
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Project root (parent of code/)
ROOT = Path(__file__).resolve().parent.parent
VENDOR_JLENS = ROOT / "vendor" / "open-jlens-data" / "code" / "jacobian-lens"


def _bootstrap_jlens_vendor() -> None:
    """Colab git clones omit vendor/ — auto-fetch open-jlens on first import."""
    marker = VENDOR_JLENS / "jlens" / "__init__.py"
    if marker.is_file():
        if str(VENDOR_JLENS) not in sys.path:
            sys.path.insert(0, str(VENDOR_JLENS))
        return
    try:
        import jlens  # noqa: F401
        return
    except ImportError:
        pass
    from jspace.vendor_bootstrap import ensure_jlens_importable

    ensure_jlens_importable()
    if str(VENDOR_JLENS) not in sys.path:
        sys.path.insert(0, str(VENDOR_JLENS))


_bootstrap_jlens_vendor()

from jspace.model_config import get_active_model

_ACTIVE = get_active_model()
DEFAULT_MODEL = _ACTIVE.model_id
DEFAULT_LENS_PATH = _ACTIVE.lens_path or (ROOT / "lenses" / f"{_ACTIVE.slug}.pt")
DEFAULT_LENS_HF = _ACTIVE.lens_hf_repo
DEFAULT_LENS_HF_FILE = _ACTIVE.lens_hf_file


@dataclass
class LoadedStack:
    """Container for model, tokenizer, lens, and layout metadata."""

    model: Any
    tokenizer: Any
    lens: Any  # JacobianLens | None if not yet fitted
    model_name: str
    n_layers: int
    d_model: int
    device: torch.device
    text_module: Any
    layers: Any
    final_norm: Any
    lm_head: Any
    W_U: torch.Tensor  # [vocab, d_model] unembedding (may be meta/4bit — use lm_head)
    hybrid_architecture: bool = False
    slug: str = ""
    generation_backend: str = "hf"


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_bnb_config() -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    *,
    quantize: bool = True,
    device_map: str | dict = "auto",
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Load HF causal LM in 4-bit NF4 and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model %s (quantize=%s)", model_name, quantize)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.bfloat16,
    }
    if quantize:
        kwargs["quantization_config"] = get_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def download_lens(
    dest: Path | None = None,
    *,
    repo_id: str | None = None,
    filename: str | None = None,
) -> Path:
    """Download pre-fitted J-lens from neuronpedia if not already present."""
    dest = Path(dest or DEFAULT_LENS_PATH)
    repo_id = repo_id or DEFAULT_LENS_HF
    filename = filename or DEFAULT_LENS_HF_FILE
    if not repo_id or not filename:
        raise FileNotFoundError(
            f"No pre-fitted lens published for this model. Expected path: {dest}. "
            "Fit one with scripts/06_fit_lfm_lens.py (or open-jlens fitter)."
        )
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        logger.info("Lens already present at %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    logger.info("Downloading lens %s/%s → %s", repo_id, filename, dest)
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest.parent / "_hf_cache"))
    import shutil

    shutil.copy2(path, dest)
    logger.info("Saved lens to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def load_jacobian_lens(path: Path | str = DEFAULT_LENS_PATH, device: str = "cpu") -> Any:
    """Load a pre-fitted JacobianLens (dict format or raw tensor stack)."""
    from jlens.lens import JacobianLens

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Lens not found: {path}. Run download_lens() first.")

    # Prefer JacobianLens.load (handles official format)
    try:
        lens = JacobianLens.load(str(path))
        # Move jacobians to device lazily later; keep on CPU for RAM
        logger.info(
            "Loaded JacobianLens: d_model=%d, n_prompts=%d, layers=%d",
            lens.d_model,
            lens.n_prompts,
            len(lens.source_layers),
        )
        return lens
    except (ValueError, KeyError) as e:
        logger.warning("JacobianLens.load failed (%s); trying raw tensor formats", e)

    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "J" in raw:
        return JacobianLens(
            jacobians=raw["J"],
            n_prompts=int(raw.get("n_prompts", 0)),
            d_model=int(raw.get("d_model", next(iter(raw["J"].values())).shape[0])),
        )
    if isinstance(raw, torch.Tensor) and raw.ndim == 3:
        # [n_layers, d, d]
        jacobians = {i: raw[i].float() for i in range(raw.shape[0])}
        return JacobianLens(
            jacobians=jacobians,
            n_prompts=0,
            d_model=raw.shape[1],
        )
    if isinstance(raw, dict) and all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in raw):
        jacobians = {int(k): v.float() for k, v in raw.items()}
        d = next(iter(jacobians.values())).shape[0]
        return JacobianLens(jacobians=jacobians, n_prompts=0, d_model=d)
    raise ValueError(f"Unrecognized lens file format: keys={list(raw.keys()) if isinstance(raw, dict) else type(raw)}")


def load_tokenizer_only(model_name: str, *, trust_remote_code: bool = True) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _vllm_tokenizer_only_mode(require_lens: bool, lens_path: Path) -> bool:
    """vLLM generates; skip HF weight load (lens file may exist but is unused here).

    Important for Qwen: pre-fitted ``lenses/qwen3.5-4b.pt`` must NOT force a full
    HF+vLLM double load on Colab T4 — that OOMs. Tier-1/2 hooked gen uses HF, not vLLM.
    """
    import os

    from jspace.vllm_backend import vllm_available

    force = os.environ.get("JLENS_VLLM_TOKENIZER_ONLY", "").strip()
    if force == "0":
        return False
    if force == "1":
        return vllm_available() and not require_lens
    return (
        os.environ.get("JLENS_BACKEND", "").lower() == "vllm"
        and vllm_available()
        and not require_lens
    )


def load_stack_tokenizer_only(model_name: str, active: Any) -> LoadedStack:
    """Tokenizer + config only — for vLLM generation when no J-lens tier cache."""
    from transformers import AutoConfig

    logger.info("Tokenizer-only stack for vLLM backend: %s", model_name)
    tokenizer = load_tokenizer_only(model_name)
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    n_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
    d_model = int(getattr(text_cfg, "hidden_size", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return LoadedStack(
        model=None,
        tokenizer=tokenizer,
        lens=None,
        model_name=model_name,
        n_layers=n_layers,
        d_model=d_model,
        device=device,
        text_module=None,
        layers=None,
        final_norm=None,
        lm_head=None,
        W_U=None,
        hybrid_architecture=active.hybrid_architecture,
        slug=active.slug,
        generation_backend="vllm",
    )


def resolve_layout(model: Any) -> tuple[Any, Any, Any, Any]:
    """Return (text_module, layers ModuleList, final_norm, lm_head)."""
    from jlens.hf import Layout, _find_layout, _resolve_attr_path

    try:
        layout = _find_layout(model)
    except ValueError:
        # LFM2: final norm is `embedding_norm` (not `norm`) — jlens layouts miss this.
        name = type(model).__name__
        if "Lfm2" in name or "LFM" in name or "Lfm" in name:
            for layout in (
                Layout("model", layers="layers", norm="embedding_norm", embed="embed_tokens"),
                Layout("model", layers="layers", norm="norm", embed="embed_tokens"),
            ):
                try:
                    candidate = _resolve_attr_path(model, layout.path)
                except AttributeError:
                    continue
                if not hasattr(candidate, layout.layers):
                    continue
                if not hasattr(model, layout.lm_head):
                    continue
                text_module = candidate
                layers = getattr(text_module, layout.layers)
                final_norm = getattr(text_module, layout.norm, None) or torch.nn.Identity()
                lm_head = getattr(model, layout.lm_head)
                logger.info("Using LFM2 layout fallback: path=%s norm=%s", layout.path, layout.norm)
                return text_module, layers, final_norm, lm_head
        raise
    text_module = _resolve_attr_path(model, layout.path)
    layers = getattr(text_module, layout.layers)
    final_norm = getattr(text_module, layout.norm)
    lm_head = getattr(model, layout.lm_head)
    return text_module, layers, final_norm, lm_head


def load_stack(
    model_name: str | None = None,
    lens_path: Path | str | None = None,
    *,
    quantize: bool | None = None,
    download_if_missing: bool = True,
    require_lens: bool = False,
) -> LoadedStack:
    """Load model + tokenizer (+ J-lens if available).

    For LiquidAI/LFM2-2.6B there is no pre-fitted neuronpedia lens. Generation +
    loop detection work with lens=None; Exp1–3 require fitting first
    (see scripts/06_fit_lfm_lens.py / handoff).
    """
    active = get_active_model()
    model_name = model_name or active.model_id
    if quantize is None:
        quantize = active.hf_quantize
    if lens_path is None:
        lens_path = active.lens_path or DEFAULT_LENS_PATH
    lens_path = Path(lens_path)

    if (
        download_if_missing
        and active.lens_hf_repo
        and active.lens_hf_file
        and not lens_path.is_file()
    ):
        download_lens(
            lens_path,
            repo_id=active.lens_hf_repo,
            filename=active.lens_hf_file,
        )

    if _vllm_tokenizer_only_mode(require_lens, lens_path):
        return load_stack_tokenizer_only(model_name, active)

    model, tokenizer = load_model_and_tokenizer(model_name, quantize=quantize)
    text_module, layers, final_norm, lm_head = resolve_layout(model)

    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_layers = int(getattr(cfg, "num_hidden_layers", len(layers)))
    d_model = int(cfg.hidden_size)

    lens = None
    if lens_path.is_file():
        lens = load_jacobian_lens(lens_path)
        if lens.d_model != d_model:
            raise ValueError(f"Lens d_model={lens.d_model} != model d_model={d_model}")
        if max(lens.source_layers) >= n_layers:
            raise ValueError(
                f"Lens has layer {max(lens.source_layers)} but model has only {n_layers} layers"
            )
    elif require_lens:
        raise FileNotFoundError(
            f"J-lens required but missing at {lens_path}. "
            f"For {active.display_name} fit a lens first (hybrid LFM needs attention-layer adaptation)."
        )
    else:
        logger.warning(
            "No J-lens at %s — generation/detection OK; Exp1–3 geometry blocked until fitted.",
            lens_path,
        )

    W_U = lm_head.weight.detach() if hasattr(lm_head, "weight") else None
    device = next(model.parameters()).device
    logger.info(
        "Stack ready: %s, n_layers=%d, d_model=%d, device=%s, lens=%s, hybrid=%s",
        model_name,
        n_layers,
        d_model,
        device,
        "yes" if lens is not None else "no",
        active.hybrid_architecture,
    )
    return LoadedStack(
        model=model,
        tokenizer=tokenizer,
        lens=lens,
        model_name=model_name,
        n_layers=n_layers,
        d_model=d_model,
        device=device,
        text_module=text_module,
        layers=layers,
        final_norm=final_norm,
        lm_head=lm_head,
        W_U=W_U,
        hybrid_architecture=active.hybrid_architecture,
        slug=active.slug,
        generation_backend="hf",
    )


@torch.no_grad()
def sanity_check_lens(stack: LoadedStack, prompt: str = "The capital of France is") -> dict[str, Any]:
    """Verify final-layer J-lens readout matches model next-token top-k."""
    from jlens.hooks import ActivationRecorder
    from jlens.hf import from_hf

    model = stack.model
    tokenizer = stack.tokenizer
    lens = stack.lens
    final_layer = stack.n_layers - 1

    # Prefer jlens apply path
    try:
        lm = from_hf(model, tokenizer, force_bos=False)
        lens_logits, model_logits, input_ids = lens.apply(
            lm, prompt, layers=[final_layer], positions=[-1], max_seq_len=128
        )
        j_topk = lens_logits[final_layer][0].topk(10)
        m_topk = model_logits[0].topk(10)
    except Exception as e:
        logger.warning("from_hf apply path failed (%s); using manual hooks", e)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(stack.device)
        with ActivationRecorder(stack.layers, at=[final_layer]) as rec:
            model(input_ids=input_ids, use_cache=False)
            h = rec.activations[final_layer][:, -1, :].float()
        # At final layer J ≈ I for a well-fit lens, but still transport
        if final_layer in lens.jacobians:
            h_t = lens.transport(h, final_layer)
        else:
            h_t = h
        logits_j = stack.lm_head(stack.final_norm(h_t.to(stack.lm_head.weight.dtype)))
        out = model(input_ids=input_ids, use_cache=False)
        logits_m = out.logits[:, -1, :]
        j_topk = logits_j[0].float().cpu().topk(10)
        m_topk = logits_m[0].float().cpu().topk(10)

    j_ids = j_topk.indices.tolist() if hasattr(j_topk.indices, "tolist") else j_topk.indices.cpu().tolist()
    m_ids = m_topk.indices.tolist() if hasattr(m_topk.indices, "tolist") else m_topk.indices.cpu().tolist()
    # Flatten if nested
    if j_ids and isinstance(j_ids[0], list):
        j_ids = j_ids[0]
    if m_ids and isinstance(m_ids[0], list):
        m_ids = m_ids[0]

    overlap = len(set(j_ids[:5]) & set(m_ids[:5]))
    top1_match = j_ids[0] == m_ids[0]
    result = {
        "top1_match": top1_match,
        "top5_overlap": overlap,
        "jlens_top5": [tokenizer.decode([i]) for i in j_ids[:5]],
        "model_top5": [tokenizer.decode([i]) for i in m_ids[:5]],
        "jlens_ids": j_ids[:10],
        "model_ids": m_ids[:10],
        "ok": top1_match or overlap >= 3,
    }
    logger.info("Sanity check: top1_match=%s top5_overlap=%d ok=%s", top1_match, overlap, result["ok"])
    logger.info("  jlens: %s", result["jlens_top5"])
    logger.info("  model: %s", result["model_top5"])
    return result
