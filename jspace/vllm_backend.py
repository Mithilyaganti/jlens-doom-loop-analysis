"""Optional vLLM generation backend (preferred on Colab / Linux).

Matches Liquid Antidoom defaults as closely as practical:
- max_new_tokens=4000
- temperature: 0.0 greedy (deterministic) by default; JLENS_TEMPERATURE>0 gives
  seeded sampling (per-prompt seed via SamplingParams.seed)
- dtype: prefer fp8 if available, else bfloat16 (Colab); avoid bf16 on 8GB laptop

Windows note: vLLM is often unreliable on Windows. Prefer Colab GPU kernel, or
fall back to HuggingFace generate (jspace.generation.generate_greedy).
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)

GREEDY_EPS = 1e-5


def vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def build_sampling_params(
    sp_cls: Any,
    *,
    max_tokens: int,
    temperature: float,
    seed: int | None = None,
) -> Any:
    """Map temperature/seed onto a vLLM SamplingParams (greedy at T<=1e-5)."""
    if temperature <= GREEDY_EPS:
        return sp_cls(max_tokens=max_tokens, temperature=0.0)
    kwargs: dict[str, Any] = dict(
        max_tokens=max_tokens,
        temperature=max(temperature, GREEDY_EPS),
        top_p=1.0,
    )
    if seed is not None:
        kwargs["seed"] = seed
    return sp_cls(**kwargs)


def build_batch_sampling_params(
    sp_cls: Any,
    *,
    count: int,
    max_tokens: int,
    temperature: float,
    seeds: Sequence[int | None] | None = None,
) -> Any:
    """SamplingParams for a batch: one shared object when greedy, else per prompt.

    vLLM seeds live on SamplingParams, so sampled batches need one params
    object per prompt to keep per-prompt seed determinism.
    """
    if temperature <= GREEDY_EPS:
        return sp_cls(max_tokens=max_tokens, temperature=0.0)
    seed_list = list(seeds) if seeds is not None else [None] * count
    if len(seed_list) != count:
        raise ValueError(f"got {len(seed_list)} seeds for {count} prompts")
    return [
        build_sampling_params(
            sp_cls, max_tokens=max_tokens, temperature=temperature, seed=s
        )
        for s in seed_list
    ]


def release_gpu() -> None:
    """Best-effort GPU free after a vLLM engine is dropped or half-initialized."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _request_output_to_dict(
    out: Any,
    prompt_text: str,
    dtype: str,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
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
        "dtype": dtype,
        "temperature": temperature,
        "seed": seed,
        "greedy": temperature <= GREEDY_EPS,
    }


def resolve_vllm_dtype() -> str:
    """Pick dtype for vLLM: fp8 > bfloat16 > float16 via env override."""
    override = os.environ.get("JLENS_VLLM_DTYPE", "").strip().lower()
    if override:
        return override
    # Default: fp8 on capable GPUs; Colab T4 may need bfloat16/float16
    return os.environ.get("JLENS_VLLM_DTYPE_DEFAULT", "fp8")


class VLLMGenerator:
    """Thin wrapper around vLLM LLM for near-greedy generation (greedy by default)."""

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
            enable_prefix_caching=True,
        )
        # T4 KV cache cannot hold a 200-way 4000-token batch. Cap concurrent
        # sequences when the caller sets JLENS_VLLM_MAX_NUM_SEQS (the A1
        # notebook sets this to the same chunk size as JLENS_BASELINE_BATCH_SIZE).
        max_num_seqs = os.environ.get("JLENS_VLLM_MAX_NUM_SEQS", "").strip()
        if max_num_seqs:
            kwargs["max_num_seqs"] = int(max_num_seqs)
        try:
            self.llm = self._try_llm(LLM, kwargs)
        except Exception as e:
            if dtype == "fp8":
                logger.warning("vLLM fp8 failed (%s); retrying bfloat16", e)
                kwargs["dtype"] = "bfloat16"
                try:
                    self.llm = self._try_llm(LLM, kwargs)
                except Exception as e2:
                    logger.warning("vLLM bfloat16 failed (%s); retrying float16", e2)
                    kwargs["dtype"] = "float16"
                    self.llm = self._try_llm(LLM, kwargs)
            elif dtype in ("bfloat16", "bf16"):
                logger.warning("vLLM bfloat16 failed (%s); retrying float16", e)
                kwargs["dtype"] = "float16"
                self.llm = self._try_llm(LLM, kwargs)
            else:
                raise
        self.model_id = model_id
        self.dtype = kwargs["dtype"]

    @staticmethod
    def _try_llm(llm_cls: Any, kwargs: dict[str, Any]) -> Any:
        """Init LLM; drop kwargs this vLLM build does not accept.

        ``enable_prefix_caching`` is retried as False (older vLLM). ``max_num_seqs``
        is dropped entirely when the constructor rejects it.
        """
        while True:
            try:
                return llm_cls(**kwargs)
            except TypeError as e:
                msg = str(e)
                if "enable_prefix_caching" in msg and kwargs.get("enable_prefix_caching") is not False:
                    logger.warning(
                        "vLLM rejects enable_prefix_caching (%s); retrying without it", e
                    )
                    kwargs["enable_prefix_caching"] = False
                    continue
                if "max_num_seqs" in msg and "max_num_seqs" in kwargs:
                    logger.warning("vLLM rejects max_num_seqs (%s); retrying without it", e)
                    kwargs.pop("max_num_seqs", None)
                    continue
                raise

    def _llm_generate(self, prompts: list[str], sampling_params: Any) -> list[Any]:
        if self.llm is None:
            raise RuntimeError(
                "vLLM engine was released after an earlier failure — use the HF backend"
            )
        try:
            return self.llm.generate(prompts, sampling_params)
        except Exception:
            # Free the engine's GPU allocation so any HF fallback can load.
            self.release()
            raise

    def release(self) -> None:
        """Drop the engine and free its GPU memory (before an HF fallback load)."""
        llm = self.llm
        self.llm = None
        del llm
        release_gpu()

    def generate(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = 4000,
        temperature: float = 0.0,
        seed: int | None = None,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        sp = build_sampling_params(
            SamplingParams,
            max_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
        outs = self._llm_generate([prompt_text], sp)
        return _request_output_to_dict(outs[0], prompt_text, self.dtype, temperature, seed)

    def generate_batch(
        self,
        prompt_texts: Sequence[str],
        *,
        max_new_tokens: int = 4000,
        temperature: float = 0.0,
        seeds: Sequence[int | None] | None = None,
        tokenizer: Any = None,
    ) -> list[dict[str, Any]]:
        """Generate for ALL prompts in one continuous-batched LLM.generate call.

        Returns one dict per prompt in input order (vLLM preserves submission
        order). Greedy batches share a single SamplingParams; sampled batches get
        one per prompt so each carries its own seed.
        """
        from vllm import SamplingParams

        prompts = list(prompt_texts)
        params = build_batch_sampling_params(
            SamplingParams,
            count=len(prompts),
            max_tokens=max_new_tokens,
            temperature=temperature,
            seeds=seeds,
        )
        outs = self._llm_generate(prompts, params)
        if len(outs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outs)} outputs for {len(prompts)} prompts"
            )
        seed_list = list(seeds) if seeds is not None else [None] * len(prompts)
        return [
            _request_output_to_dict(out, prompts[i], self.dtype, temperature, seed_list[i])
            for i, out in enumerate(outs)
        ]


def make_generator(model_id: str):
    """Factory: vLLM if requested and installed, else None (use HF).

    Never hard-crash when JLENS_BACKEND=vllm but vllm is missing — fall back to HF
    so Colab runs can continue after a failed/skipped pip install.
    """
    # Prefer HF when explicitly requested, or when vLLM is known broken for this stack.
    # Colab users can force: echo hf_bf16 > /content/jlens_a1_backend.txt
    backend = os.environ.get("JLENS_BACKEND", "auto").strip().lower()
    force_hf = os.environ.get("JLENS_FORCE_HF", "").strip() == "1"
    if force_hf or backend in ("hf", "huggingface", "transformers"):
        return None
    want_vllm = backend == "vllm" or (backend == "auto" and vllm_available())
    if not want_vllm:
        return None
    if not vllm_available():
        logger.warning(
            "JLENS_BACKEND=%s but vllm is not installed — falling back to HuggingFace. "
            "On Colab run: %%pip install -q vllm",
            backend,
        )
        return None
    try:
        return VLLMGenerator(
            model_id,
            max_model_len=int(os.environ.get("JLENS_MAX_MODEL_LEN", "6000")),
            gpu_memory_utilization=float(os.environ.get("JLENS_GPU_UTIL", "0.85")),
        )
    except Exception as e:
        # Common on Colab: vLLM 0.8.x vs new transformers (Qwen2Tokenizer
        # missing all_special_tokens_extended). Fall back to HF bf16/fp16.
        logger.warning(
            "vLLM init failed (%s: %s) — falling back to HuggingFace generate",
            type(e).__name__,
            str(e)[:300],
        )
        # LLM() may have allocated GPU memory before raising — free it so the
        # HF fallback load does not OOM.
        release_gpu()
        return None
