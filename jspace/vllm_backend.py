"""Optional vLLM generation backend (preferred on Colab / Linux).

Matches Liquid Antidoom defaults as closely as practical:
- max_new_tokens=4000
- temperature≈0.01 (near-greedy)
- dtype: prefer fp8 if available, else bfloat16 (Colab); avoid bf16 on 8GB laptop

Windows note: vLLM is often unreliable on Windows. Prefer Colab GPU kernel, or
fall back to HuggingFace generate (jspace.generation.generate_greedy).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_vllm_dtype() -> str:
    """Pick dtype for vLLM: fp8 > bfloat16 > float16 via env override."""
    override = os.environ.get("JLENS_VLLM_DTYPE", "").strip().lower()
    if override:
        return override
    # Default: fp8 on capable GPUs; Colab T4 may need bfloat16/float16
    return os.environ.get("JLENS_VLLM_DTYPE_DEFAULT", "fp8")


class VLLMGenerator:
    """Thin wrapper around vLLM LLM for sequential near-greedy generation."""

    def __init__(
        self,
        model_id: str,
        *,
        dtype: str | None = None,
        max_model_len: int = 6000,
        gpu_memory_utilization: float = 0.85,
        trust_remote_code: bool = True,
    ) -> None:
        from vllm import LLM

        dtype = dtype or resolve_vllm_dtype()
        logger.info(
            "Init vLLM model=%s dtype=%s max_model_len=%d",
            model_id,
            dtype,
            max_model_len,
        )
        kwargs: dict[str, Any] = dict(
            model=model_id,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enable_prefix_caching=False,
        )
        try:
            self.llm = LLM(**kwargs)
        except Exception as e:
            if dtype == "fp8":
                logger.warning("vLLM fp8 failed (%s); retrying bfloat16", e)
                kwargs["dtype"] = "bfloat16"
                self.llm = LLM(**kwargs)
            else:
                raise
        self.model_id = model_id
        self.dtype = kwargs["dtype"]

    def generate(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = 4000,
        temperature: float = 0.01,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        # Near-greedy: very low temperature
        if temperature <= 1e-5:
            sp = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
            )
        else:
            sp = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=max(temperature, 1e-5),
                top_p=1.0,
            )
        outs = self.llm.generate([prompt_text], sp)
        out = outs[0]
        gen_text = out.outputs[0].text
        # token ids if available
        gen_ids = list(out.outputs[0].token_ids) if out.outputs[0].token_ids else []
        prompt_len = len(out.prompt_token_ids) if out.prompt_token_ids else 0
        full_ids = list(out.prompt_token_ids or []) + gen_ids
        return {
            "prompt_text": prompt_text,
            "generated_ids": gen_ids,
            "generated_text": gen_text,
            "prompt_len": prompt_len,
            "full_ids": full_ids,
            "backend": "vllm",
            "dtype": self.dtype,
        }


def make_generator(model_id: str):
    """Factory: vLLM if JLENS_BACKEND=vllm (or auto+available), else None (use HF)."""
    backend = os.environ.get("JLENS_BACKEND", "auto").strip().lower()
    if backend == "hf":
        return None
    if backend == "vllm" or (backend == "auto" and vllm_available()):
        if not vllm_available():
            raise RuntimeError(
                "JLENS_BACKEND=vllm but vllm is not installed. "
                "On Colab: pip install vllm. On Windows laptop prefer JLENS_BACKEND=hf."
            )
        return VLLMGenerator(
            model_id,
            max_model_len=int(os.environ.get("JLENS_MAX_MODEL_LEN", "6000")),
            gpu_memory_utilization=float(os.environ.get("JLENS_GPU_UTIL", "0.85")),
        )
    return None
