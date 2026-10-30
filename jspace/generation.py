"""Generation utilities with tiered residual/J-lens caching for Exp 2–3.

Determinism: JLENS_TEMPERATURE (default 0.0) at/below 1e-5 decodes greedily;
above it, sampling stays available but every generate call is preceded by
torch.manual_seed on a per-prompt seed derived from JLENS_SAMPLE_SEED (default
42) — seed = base_seed*1000003 + prompt_id (mod 2**31), identical across runs
and backends. The seed/temperature/greedy triple is returned in every generate
dict and recorded in per-prompt meta.json.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from jspace.detection import TokenState, detect_loop, decode_token
from jspace.geometry import jlens_readout
from jspace.loading import clear_cuda

logger = logging.getLogger(__name__)

GREEDY_EPS = 1e-5


def resolve_temperature() -> float:
    """JLENS_TEMPERATURE env; default 0.0 = greedy (deterministic)."""
    return float(os.environ.get("JLENS_TEMPERATURE", "0.0"))


def resolve_base_seed() -> int:
    """JLENS_SAMPLE_SEED env (default 42)."""
    return int(os.environ.get("JLENS_SAMPLE_SEED", "42"))


def is_greedy_temperature(temperature: float) -> bool:
    """Temperatures at/below GREEDY_EPS run as exact greedy decoding."""
    return float(temperature) <= GREEDY_EPS


def per_prompt_seed(base_seed: int, prompt_id: int) -> int:
    """Per-prompt sampling seed, stable across runs and backends."""
    return (int(base_seed) * 1000003 + int(prompt_id)) % (2**31)


def sampling_plan(
    prompt_id: int | None = None,
    *,
    base_seed: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Seed + temperature + greedy flag for one prompt's generate call.

    ``prompt_id=None`` derives the seed as prompt_id 0 (callers without ids
    still get reproducible sampling).
    """
    base = resolve_base_seed() if base_seed is None else int(base_seed)
    temp = resolve_temperature() if temperature is None else float(temperature)
    pid = 0 if prompt_id is None else int(prompt_id)
    return {
        "seed": per_prompt_seed(base, pid),
        "temperature": temp,
        "greedy": is_greedy_temperature(temp),
    }


def build_hf_gen_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    pad_token_id: int | None,
    eos_token_id: int | None,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
) -> dict[str, Any]:
    """HF model.generate kwargs: exact greedy at T<=eps, seeded sampling above.

    ``repetition_penalty`` and ``no_repeat_ngram_size`` are optional decoding
    baselines. Omitted means the key is absent, which is the unsteered path.
    """
    kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=not is_greedy_temperature(temperature),
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_cache=True,
    )
    if not is_greedy_temperature(temperature):
        kwargs["temperature"] = max(float(temperature), GREEDY_EPS)
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = float(repetition_penalty)
    if no_repeat_ngram_size is not None:
        kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    return kwargs


def apply_chat_template(tokenizer: Any, user_prompt: str, system: str = "") -> str:
    """Render chat template if available; else plain text with reasoning cue."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    # Antidoom-style thinking instruction
    body = (
        f"{user_prompt}\n\n"
        'Think through the problem step by step, then respond with your final answer as "Answer: <your answer>".'
    )
    messages.append({"role": "user", "content": body})
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return body


@torch.no_grad()
def generate_greedy(
    stack: Any,
    prompt_text: str,
    *,
    max_new_tokens: int = 4000,
    temperature: float | None = None,
    vllm_generator: Any = None,
    prompt_id: int | None = None,
    logits_processor: Sequence[Any] | None = None,
    repetition_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
) -> dict[str, Any]:
    """Generate completion (greedy by default; seeded sampling at T>0).

    Uses vLLM if generator provided. torch.manual_seed is set from the
    per-prompt seed before every generate call. ``logits_processor`` applies
    to the HF path only (e.g. the Exp3 logit-suppression positive control).
    Repetition penalty and no-repeat n-gram are HF-only decoding baselines.
    They are refused on the vLLM path so a run cannot claim they were applied.
    """
    plan = sampling_plan(prompt_id, temperature=temperature)
    decoding_constraints = repetition_penalty is not None or no_repeat_ngram_size is not None
    if vllm_generator is not None and decoding_constraints:
        raise RuntimeError(
            "repetition_penalty and no_repeat_ngram_size are HF decoding baselines; "
            "refusing to drop them on the vLLM path"
        )
    if vllm_generator is not None:
        return vllm_generator.generate(
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=plan["temperature"],
            seed=plan["seed"],
            tokenizer=stack.tokenizer,
        )

    tokenizer = stack.tokenizer
    model = stack.model
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs.input_ids.to(stack.device)
    attention_mask = inputs.attention_mask.to(stack.device) if "attention_mask" in inputs else None
    prompt_len = input_ids.shape[1]

    torch.manual_seed(plan["seed"])
    gen_kwargs = build_hf_gen_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=plan["temperature"],
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    if logits_processor:
        gen_kwargs["logits_processor"] = list(logits_processor)
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    gen_ids = out[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    result = {
        "prompt_text": prompt_text,
        "generated_ids": gen_ids,
        "generated_text": gen_text,
        "prompt_len": prompt_len,
        "full_ids": out[0].tolist(),
        "backend": "hf",
        "seed": plan["seed"],
        "temperature": plan["temperature"],
        "greedy": plan["greedy"],
    }
    if repetition_penalty is not None:
        result["repetition_penalty"] = float(repetition_penalty)
    if no_repeat_ngram_size is not None:
        result["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
    return result


@torch.no_grad()
def generate_with_tiered_cache(
    stack: Any,
    prompt_text: str,
    *,
    max_new_tokens: int = 4000,
    temperature: float | None = None,
    workspace_layers: Sequence[int] | None = None,
    top_k: int = 10,
    cache_tier3: bool = False,
    vllm_generator: Any = None,
    prompt_id: int | None = None,
) -> dict[str, Any]:
    """Generate with Tier-1 readouts (+ Tier-2 workspace residuals).

    If stack.lens is None, still generates + detects loops but skips J-lens caches.
    """
    gen = generate_greedy(
        stack,
        prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        vllm_generator=vllm_generator,
        prompt_id=prompt_id,
    )
    gen_ids = gen["generated_ids"]
    loop = detect_loop(
        gen["generated_text"], token_ids=gen_ids, tokenizer=stack.tokenizer
    )

    if stack.lens is None:
        return {
            **gen,
            "readouts": None,
            "residuals_ws": None,
            "residuals_full": None,
            "loop": loop,
            "note": "no_jlens_skip_tier_cache",
        }

    full_ids = torch.tensor([gen["full_ids"]], device=stack.device)
    prompt_len = gen["prompt_len"]
    n_gen = len(gen_ids)

    if n_gen == 0:
        return {
            **gen,
            "readouts": None,
            "residuals_ws": None,
            "residuals_full": None,
            "loop": loop,
        }

    ws = list(workspace_layers or [])
    cached = cache_jlens_readouts_for_sequence(
        stack,
        full_ids=full_ids,
        prompt_len=prompt_len,
        trigger_token_index=loop.trigger_token_index,
        workspace_layers=ws,
        top_k=top_k,
        cache_tier3=cache_tier3,
    )
    return {
        **gen,
        **cached,
        "loop": loop,
    }


def _choose_cache_window(
    full_len: int,
    prompt_len: int,
    trigger_token_index: int | None,
    *,
    max_ctx: int = 1536,
    pre_trigger: int = 64,
    post_trigger: int = 16,
) -> tuple[int, int, dict[str, Any]]:
    """Pick a contiguous token window that covers pre-onset context.

    For looping traces, center the window on the trigger so positions −5..0
    are always inside the cached readouts (critical for Exp2).
    """
    max_ctx = min(max_ctx, full_len)
    if full_len <= max_ctx:
        return 0, full_len, {"truncated": False, "orig_len": full_len, "mode": "full"}

    if trigger_token_index is not None and trigger_token_index >= 0:
        # Absolute index in full_ids of the trigger token.
        abs_trig = prompt_len + int(trigger_token_index)
        # End just after trigger (+ small margin); start max_ctx before that.
        end = min(full_len, abs_trig + 1 + post_trigger)
        start = max(0, end - max_ctx)
        # Ensure we still keep enough tokens before trigger for −5..0 analysis.
        need_before = pre_trigger + 8
        if abs_trig - start < need_before:
            start = max(0, abs_trig - need_before)
            end = min(full_len, start + max_ctx)
        mode = "trigger_centered"
    else:
        # Non-loop: keep the most recent context (generation tail).
        end = full_len
        start = full_len - max_ctx
        mode = "tail"

    info = {
        "truncated": True,
        "orig_len": full_len,
        "window_start": start,
        "window_end": end,
        "mode": mode,
        "max_ctx": max_ctx,
    }
    return start, end, info


def resolve_record_mode() -> str:
    """JLENS_RECORD_MODE env: onepass (default, single hooked forward) | chunked."""
    mode = os.environ.get("JLENS_RECORD_MODE", "onepass").strip().lower()
    if mode not in ("onepass", "chunked"):
        logger.warning("Unknown JLENS_RECORD_MODE=%r — using onepass", mode)
        return "onepass"
    return mode


@torch.no_grad()
def cache_jlens_readouts_for_sequence(
    stack: Any,
    *,
    full_ids: torch.Tensor,
    prompt_len: int,
    trigger_token_index: int | None,
    workspace_layers: Sequence[int] | None = None,
    top_k: int = 10,
    cache_tier3: bool = False,
    max_ctx: int = 1536,
) -> dict[str, Any]:
    """Tier-1/2 J-lens cache for an already-generated full token sequence.

    JLENS_RECORD_MODE selects the capture strategy (outputs are identical):
    - onepass (default): one forward with hooks on all needed layers, each
      captured tensor offloaded to CPU inside the hook.
    - chunked: legacy ~ceil(n_layers/6) forwards over 6-layer chunks.
    """
    if full_ids.dim() == 1:
        full_ids = full_ids.unsqueeze(0)
    full_ids = full_ids.to(stack.device)
    full_len = int(full_ids.shape[1])
    ws = list(workspace_layers or [])

    start, end, trunc_info = _choose_cache_window(
        full_len,
        prompt_len,
        trigger_token_index,
        max_ctx=max_ctx,
    )
    ids = full_ids[:, start:end]
    # Positions in `ids` that correspond to generated tokens.
    gen_abs_start = max(prompt_len, start)
    gen_pos_start = gen_abs_start - start  # index within ids
    seq_len = ids.shape[1]
    gen_positions = list(range(gen_pos_start, seq_len))
    n_gp = len(gen_positions)
    if n_gp == 0:
        return {
            "readouts": None,
            "residuals_ws": None,
            "residuals_full": None,
            "trunc": trunc_info,
            "gen_pos_start": gen_pos_start,
            "trigger_local_in_readouts": None,
            "n_gen_positions_cached": 0,
        }

    n_layers = stack.n_layers
    topk_idx = torch.zeros(n_layers, n_gp, top_k, dtype=torch.long)
    topk_val = torch.zeros(n_layers, n_gp, top_k, dtype=torch.float16)
    residuals_ws: dict[int, torch.Tensor] = {}
    residuals_full: dict[int, torch.Tensor] = {}

    def _consume_layer(l: int, h: torch.Tensor) -> None:
        if l in stack.lens.jacobians:
            try:
                idx, val = jlens_readout(stack, h, l, top_k=top_k)
                topk_idx[l] = idx
                topk_val[l] = val.half()
            except Exception as e:
                logger.warning("readout failed layer %d: %s", l, e)
        if l in ws:
            residuals_ws[l] = h.half().cpu()
        if cache_tier3:
            residuals_full[l] = h.half().cpu()

    record_mode = resolve_record_mode()
    if record_mode == "onepass":
        hook_layers = sorted(set(range(n_layers)) | {l for l in ws if 0 <= l < n_layers})
        cpu_acts: dict[int, torch.Tensor] = {}

        def _offload_hook(layer_idx: int) -> Any:
            def hook(module: Any, inputs: Any, output: Any) -> None:
                tensor = output if torch.is_tensor(output) else output[0]
                cpu_acts[layer_idx] = tensor[0].detach().to("cpu", copy=True)

            return hook

        handles = [stack.layers[l].register_forward_hook(_offload_hook(l)) for l in hook_layers]
        try:
            stack.model(input_ids=ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        for l in range(n_layers):
            acts = cpu_acts.pop(l, None)
            if acts is None:
                continue
            h = acts[gen_positions, :].to(stack.device).float()
            _consume_layer(l, h)
            del h, acts
        clear_cuda()
    else:
        from jlens.hooks import ActivationRecorder

        chunk_size = 6
        for c0 in range(0, n_layers, chunk_size):
            chunk = list(range(c0, min(c0 + chunk_size, n_layers)))
            need = set(chunk)
            if ws:
                need |= {l for l in ws if c0 <= l < c0 + chunk_size}
            with ActivationRecorder(stack.layers, at=sorted(need)) as rec:
                stack.model(input_ids=ids, use_cache=False)
                for l in chunk:
                    if l not in rec.activations:
                        continue
                    h = rec.activations[l][0, gen_positions, :].float()
                    _consume_layer(l, h)
                    del h
                del rec.activations
            clear_cuda()

    trigger_local_trunc = None
    if trigger_token_index is not None:
        abs_trig = prompt_len + int(trigger_token_index)
        if start <= abs_trig < end:
            # Index within gen_positions / readout time axis.
            trigger_local_trunc = (abs_trig - start) - gen_pos_start

    trunc_info = {
        **trunc_info,
        "prompt_len": prompt_len,
        "trigger_token_index": trigger_token_index,
        "trigger_abs": (prompt_len + trigger_token_index) if trigger_token_index is not None else None,
        "record_mode": record_mode,
    }
    return {
        "readouts": {"indices": topk_idx, "values": topk_val},
        "residuals_ws": residuals_ws,
        "residuals_full": residuals_full if cache_tier3 else None,
        "trunc": trunc_info,
        "gen_pos_start": gen_pos_start,
        "trigger_local_in_readouts": trigger_local_trunc,
        "n_gen_positions_cached": n_gp,
    }


def loop_result_to_meta(loop: Any, gen: dict[str, Any], prompt_id: Any = None) -> dict[str, Any]:
    meta = {
        "prompt_id": prompt_id,
        "is_loop": bool(loop.is_loop),
        "trigger_token_index": loop.trigger_token_index,
        "trigger_token_id": loop.trigger_token_id,
        "trigger_token_str": loop.trigger_token_str,
        "trigger_decoded": loop.trigger_decoded,
        "start_char": loop.start_char,
        "repeat_start_char": loop.repeat_start_char,
        "end_char": loop.end_char,
        "period": loop.hit.period if loop.hit else None,
        "repeats": loop.hit.repeats if loop.hit else None,
        "snippet": loop.hit.snippet if loop.hit else None,
        "generated_text": gen.get("generated_text", ""),
        "n_tokens": len(gen.get("generated_ids", [])),
        "prompt_text": gen.get("prompt_text", "")[:2000],
        "trigger_local_in_readouts": gen.get("trigger_local_in_readouts"),
        "n_gen_positions_cached": gen.get("n_gen_positions_cached"),
        "trunc": gen.get("trunc"),
        "seed": gen.get("seed"),
        "temperature": gen.get("temperature"),
        "greedy": gen.get("greedy"),
    }
    if meta["is_loop"]:
        # Best effort: detector fields may be absent on unusual hits.
        trig_id = getattr(loop, "trigger_token_id", None)
        if trig_id is not None:
            meta["loop_trigger_token_id"] = trig_id
        trig_text = getattr(loop, "trigger_decoded", None) or getattr(loop, "trigger_token_str", None)
        if trig_text:
            meta["loop_trigger_text"] = trig_text
        trig_pos = getattr(loop, "trigger_token_index", None)
        if trig_pos is not None:
            meta["loop_trigger_pos"] = trig_pos
    return meta


def save_trace(out_dir: Path, trace: dict[str, Any], meta: dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    if trace.get("readouts") is not None:
        torch.save(trace["readouts"], out_dir / "readouts.pt")
    if trace.get("residuals_ws"):
        torch.save(trace["residuals_ws"], out_dir / "residuals_workspace_band.pt")
    if trace.get("residuals_full"):
        torch.save(trace["residuals_full"], out_dir / "residuals_full.pt")
    if "generated_ids" in trace:
        payload = {
            "generated_ids": trace["generated_ids"],
            "prompt_len": trace.get("prompt_len"),
        }
        if trace.get("full_ids") is not None:
            payload["full_ids"] = trace["full_ids"]
        torch.save(payload, out_dir / "tokens.pt")
