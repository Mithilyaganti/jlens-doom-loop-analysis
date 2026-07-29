"""Generation utilities with tiered residual/J-lens caching for Exp 2–3."""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import torch

from jspace.detection import TokenState, detect_loop, decode_token
from jspace.geometry import jlens_readout
from jspace.loading import clear_cuda

logger = logging.getLogger(__name__)


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
    temperature: float = 0.01,
    vllm_generator: Any = None,
) -> dict[str, Any]:
    """Generate completion (near-greedy). Uses vLLM if generator provided."""
    if vllm_generator is not None:
        out = vllm_generator.generate(
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            tokenizer=stack.tokenizer,
        )
        return out

    tokenizer = stack.tokenizer
    model = stack.model
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs.input_ids.to(stack.device)
    attention_mask = inputs.attention_mask.to(stack.device) if "attention_mask" in inputs else None
    prompt_len = input_ids.shape[1]

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 1e-5,
        temperature=max(temperature, 1e-5) if temperature > 1e-5 else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if temperature <= 1e-5:
        gen_kwargs["do_sample"] = False
        gen_kwargs.pop("temperature", None)

    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
    gen_ids = out[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "prompt_text": prompt_text,
        "generated_ids": gen_ids,
        "generated_text": gen_text,
        "prompt_len": prompt_len,
        "full_ids": out[0].tolist(),
        "backend": "hf",
    }


@torch.no_grad()
def generate_with_tiered_cache(
    stack: Any,
    prompt_text: str,
    *,
    max_new_tokens: int = 4000,
    temperature: float = 0.01,
    workspace_layers: Sequence[int] | None = None,
    top_k: int = 10,
    cache_tier3: bool = False,
    vllm_generator: Any = None,
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
    """Tier-1/2 J-lens cache for an already-generated full token sequence."""
    from jlens.hooks import ActivationRecorder

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
    return {
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
    }


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
