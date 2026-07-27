#!/usr/bin/env python
"""Fit a Jacobian lens for LiquidAI/LFM2-2.6B (hybrid attn+conv).

Paper / open-jlens recipe, adapted for LFM2:
  - Load model in fp16 (or bf16 if JLENS_FIT_DTYPE=bf16)
  - Fit J_l only on full_attention layers (skip LIV conv blocks)
  - WikiText-103 prompts, max_seq_len=128, skip_first=16
  - Checkpoint every prompt; resume-safe

Env knobs
---------
JLENS_FIT_N_PROMPTS   default 100 (paper: usable; 1000 = full)
JLENS_FIT_DIM_BATCH   default 32 (lower if OOM; raise if VRAM free)
JLENS_FIT_MAX_SEQ     default 128
JLENS_FIT_DTYPE       fp16 | bf16  (default fp16)
JLENS_FIT_OUT         override output .pt path
JLENS_FIT_CKPT        override checkpoint path
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fit_lfm_lens")

# LFM2-2.6B layer_types: conv/conv/full_attention repeating
LFM2_26B_ATTN_LAYERS = (2, 5, 9, 13, 17, 21, 24, 27)


def _attn_layers_from_config(config) -> tuple[int, ...]:
    layer_types = getattr(config, "layer_types", None)
    if not layer_types:
        return LFM2_26B_ATTN_LAYERS
    idxs = tuple(i for i, t in enumerate(layer_types) if "attention" in str(t).lower())
    return idxs or LFM2_26B_ATTN_LAYERS


def main() -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from jspace.analysis import ensure_dir, save_json, write_status
    from jspace.model_config import artifact_paths, get_active_model
    from jspace.vendor_bootstrap import ensure_jlens_importable

    ensure_jlens_importable()
    import jlens
    from jlens.examples import load_wikitext_prompts
    from jlens.hf import Layout, from_hf

    cfg = get_active_model()
    paths = artifact_paths(cfg)
    ensure_dir(paths["results"])
    ensure_dir(ROOT / "lenses")

    out_path = Path(os.environ.get("JLENS_FIT_OUT", str(paths["lens"])))
    ckpt_path = Path(
        os.environ.get(
            "JLENS_FIT_CKPT",
            str(paths["results"] / "checkpoints" / f"jlens_fit_{cfg.slug}.pt"),
        )
    )
    n_prompts = int(os.environ.get("JLENS_FIT_N_PROMPTS", "100"))
    dim_batch = int(os.environ.get("JLENS_FIT_DIM_BATCH", "4"))  # 32 OOMs on Colab T4 15GB
    max_seq = int(os.environ.get("JLENS_FIT_MAX_SEQ", "128"))
    dtype_name = os.environ.get("JLENS_FIT_DTYPE", "fp16").strip().lower()
    dtype = torch.bfloat16 if dtype_name in ("bf16", "bfloat16") else torch.float16

    if out_path.is_file() and os.environ.get("JLENS_FIT_FORCE", "0") != "1":
        logger.info("Lens already present: %s (set JLENS_FIT_FORCE=1 to refit)", out_path)
        write_status("06_fit_lfm_lens", f"# LFM J-lens\n\nAlready present at `{out_path}`.\n")
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for J-lens fit")

    logger.info(
        "Fitting %s | dtype=%s n_prompts=%d dim_batch=%d max_seq=%d",
        cfg.model_id,
        dtype,
        n_prompts,
        dim_batch,
        max_seq,
    )
    logger.info("GPU: %s | VRAM %.1f GB", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9)

    hf_config = AutoConfig.from_pretrained(cfg.model_id, trust_remote_code=True)
    source_layers = _attn_layers_from_config(hf_config)
    logger.info("Attention source_layers (%d): %s", len(source_layers), source_layers)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # LFM2 final norm is embedding_norm (not norm) — required for from_hf
    layout = Layout("model", layers="layers", norm="embedding_norm", embed="embed_tokens")
    lm = from_hf(model, tokenizer, layout=layout, force_bos=False)
    logger.info("Wrapped HFLensModel: n_layers=%d d_model=%d", lm.n_layers, lm.d_model)

    prompts = load_wikitext_prompts(n_prompts)
    logger.info("Loaded %d WikiText prompts", len(prompts))

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lens = jlens.fit(
        lm,
        prompts,
        source_layers=list(source_layers),
        dim_batch=dim_batch,
        max_seq_len=max_seq,
        checkpoint_path=str(ckpt_path),
        checkpoint_every=1,
        resume=True,
    )
    lens.save(str(out_path))
    logger.info("Saved lens → %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    status = {
        "model_id": cfg.model_id,
        "hybrid": cfg.hybrid_architecture,
        "fitted": True,
        "dtype": str(dtype).replace("torch.", ""),
        "n_prompts": len(prompts),
        "dim_batch": dim_batch,
        "max_seq_len": max_seq,
        "source_layers": list(source_layers),
        "lens_path": str(out_path),
        "checkpoint_path": str(ckpt_path),
        "note": (
            "Jacobian fit on full_attention layers only (conv LIV blocks skipped). "
            "Apply with NF4 generation on laptop per PROJECT_SPEC; refit on larger "
            "GPU in fp16/bf16 for paper-grade if results look promising."
        ),
    }
    save_json(status, paths["jlens_fit_status"])
    write_status(
        "06_fit_lfm_lens",
        "# LFM J-lens fit\n\n"
        f"**Fitted** `{out_path}`\n\n"
        f"- dtype: `{status['dtype']}`\n"
        f"- prompts: {status['n_prompts']}\n"
        f"- attn layers: `{source_layers}`\n",
    )
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
