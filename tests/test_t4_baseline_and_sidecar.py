"""CPU smoke tests for T4 baseline speedups and the anti-doom sidecar.

Run: python tests/test_t4_baseline_and_sidecar.py
No model, GPU, or network. Synthetic token ids and a fake HF generate only.
"""

from __future__ import annotations

import contextlib
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace.antidoom_sidecar import (  # noqa: E402
    AntiDoomSidecar,
    continuation_to_block,
    sidecar_enabled,
    sidecar_from_env,
)
from jspace.baseline_speed import (  # noqa: E402
    LoopConfirmedStoppingCriteria,
    chunk_pending,
    early_stop_enabled,
    resolve_batch_chunk_size,
)
from jspace.detection import find_inner_repetition  # noqa: E402
from jspace.generation import generate_greedy, loop_result_to_meta  # noqa: E402
from jspace.loading import hf_attn_implementation  # noqa: E402
from jspace.vllm_backend import VLLMGenerator  # noqa: E402

_MISSING = object()


@contextlib.contextmanager
def _env(**kv):
    saved = {k: os.environ.get(k, _MISSING) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is _MISSING:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _RecordingLLM:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)


class _RejectMaxNumSeqs(_RecordingLLM):
    def __init__(self, **kwargs):
        if "max_num_seqs" in kwargs:
            raise TypeError("LLM.__init__() got an unexpected keyword argument 'max_num_seqs'")
        super().__init__(**kwargs)


@contextlib.contextmanager
def _fake_vllm(llm_cls):
    fake = types.ModuleType("vllm")
    fake.SamplingParams = _FakeSamplingParams
    fake.LLM = llm_cls
    saved = sys.modules.get("vllm")
    sys.modules["vllm"] = fake
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = saved


class _Batch:
    def __init__(self, ids: torch.Tensor):
        self.input_ids = ids
        self.attention_mask = torch.ones_like(ids)

    def __contains__(self, key: object) -> bool:
        return key in ("input_ids", "attention_mask")


class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=2048):
        return _Batch(torch.tensor([[2, 3]]))

    def decode(self, ids, skip_special_tokens=True):
        return "".join("x" if int(i) == 9 else f"{int(i) % 10}" for i in ids)


class _Model:
    def __init__(self):
        self.kwargs = None

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.kwargs = kwargs
        extra = torch.full((input_ids.shape[0], 1), 9)
        return torch.cat([input_ids, extra], dim=1)


class _Stack:
    def __init__(self):
        self.tokenizer = _Tok()
        self.model = _Model()
        self.device = "cpu"


def test_chunk_pending_default_and_legacy_all():
    items = list(range(10))
    assert chunk_pending(items, 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert chunk_pending(items, 0) == [items]
    assert chunk_pending(items, 100) == [items]
    assert chunk_pending([], 4) == []
    with _env(JLENS_BASELINE_BATCH_SIZE=None):
        assert resolve_batch_chunk_size(10) == 4
    with _env(JLENS_BASELINE_BATCH_SIZE="all"):
        assert resolve_batch_chunk_size(10) == 10
        assert chunk_pending(items, resolve_batch_chunk_size(len(items))) == [items]
    with _env(JLENS_BASELINE_BATCH_SIZE="0"):
        assert resolve_batch_chunk_size(7) == 7
    with _env(JLENS_BASELINE_BATCH_SIZE="not-a-number"):
        assert resolve_batch_chunk_size(3) == 4


def test_early_stop_env_defaults_on():
    with _env(JLENS_BASELINE_EARLY_STOP=None):
        assert early_stop_enabled() is True
    with _env(JLENS_BASELINE_EARLY_STOP="0"):
        assert early_stop_enabled() is False


def test_confirmed_loop_prefix_stays_a_loop_when_extended():
    """Stopping at the first detector hit cannot flip a true loop to false.

    The same repeated pattern, extended, is still a loop. That is why the
    baseline may stop once the hit exists without changing the loop label.
    """
    first = None
    for k in range(1, 40):
        found, _hit = find_inner_repetition("abcd" * k)
        if found:
            first = k
            break
    assert first is not None and first < 40
    found_long, _ = find_inner_repetition("abcd" * (first + 5))
    assert found_long
    found_short, _ = find_inner_repetition("abcd" * (first - 1))
    assert not found_short


def test_stopping_criteria_fires_only_after_the_detector_confirms():
    tok = _Tok()
    criteria = LoopConfirmedStoppingCriteria(tok, prompt_len=2, check_every=32)
    scores = torch.zeros(1, 4)

    short = torch.tensor([[2, 3] + [9] * 32])
    assert not bool(criteria(short, scores)[0])
    assert criteria.stopped is False

    # 64 identical decoded chars: period 16, 4 repeats, 64 >= 60.
    looping = torch.tensor([[2, 3] + [9] * 64])
    assert bool(criteria(looping, scores)[0])
    assert criteria.stopped is True
    assert criteria.stopped_at == 64

    # Stays stopped on later calls.
    assert bool(criteria(looping, scores)[0])

    fresh = LoopConfirmedStoppingCriteria(tok, prompt_len=2, check_every=16)
    # 48 x's is 3 periods of 16, under the 4-repeat rule.
    almost = torch.tensor([[2, 3] + [9] * 48])
    assert not bool(fresh(almost, scores)[0])
    # Off the check interval: do not decode, do not stop.
    off = torch.tensor([[2, 3] + [9] * 17])
    assert not bool(fresh(off, scores)[0])


def test_generate_greedy_wires_early_stop_and_processors():
    stack = _Stack()
    with _env(JLENS_TEMPERATURE="0", JLENS_SAMPLE_SEED="42", JLENS_EARLY_STOP_EVERY="32"):
        out = generate_greedy(
            stack,
            "hello",
            max_new_tokens=8,
            prompt_id=3,
            early_stop=True,
            logits_processor=["marker-processor"],
        )
    assert out["backend"] == "hf"
    assert out["early_stopped"] is False  # fake generate never invokes the criterion
    assert out["generated_ids"] == [9]
    kwargs = stack.model.kwargs
    assert kwargs["logits_processor"] == ["marker-processor"]
    assert kwargs["do_sample"] is False
    criteria = kwargs["stopping_criteria"]
    assert len(criteria) == 1
    assert criteria[0].prompt_len == 2
    assert criteria[0].check_every == 32

    stack2 = _Stack()
    with _env(JLENS_TEMPERATURE="0"):
        generate_greedy(stack2, "hello", max_new_tokens=4, early_stop=False)
    assert "stopping_criteria" not in stack2.model.kwargs


def test_continuation_blocks_the_next_copy_only():
    assert continuation_to_block([7, 7], min_repeats=4) is None
    assert continuation_to_block([7, 7, 7], min_repeats=4) == 7
    assert continuation_to_block([1, 2, 1, 2], min_repeats=4) is None
    assert continuation_to_block([1, 2, 1, 2, 1, 2], min_repeats=4) == 1
    # Partial third copy of a bigram: next token is the other half.
    assert continuation_to_block([1, 2, 1, 2, 1], min_repeats=3) == 2
    # A prefix that is not part of the period must not hide the tail.
    assert continuation_to_block([9, 1, 2, 1, 2, 1, 2], min_repeats=4) == 1
    assert continuation_to_block([1, 2, 3, 4, 5], min_repeats=4) is None
    # Shortest period wins on a unigram run.
    assert continuation_to_block([4, 4, 4, 4, 4, 4], min_repeats=4, max_period=8) == 4


def test_sidecar_biases_only_the_continuation_token():
    proc = AntiDoomSidecar(prompt_len=2, min_repeats=4, max_period=8, bias=-1.0e4)
    # Prompt tokens repeat; generated tail is three 0s, so block the fourth 0.
    input_ids = torch.tensor([[0, 0, 0, 0, 0]])
    scores = torch.tensor([[5.0, 1.0, 0.5]])
    out = proc(input_ids, scores)
    assert proc.n_suppressed_steps == 1
    assert proc.blocked_token_ids == [0]
    assert torch.equal(out[0, 1:], scores[0, 1:])
    assert float(out[0, 0]) == float(scores[0, 0]) - 1.0e4
    assert int(out.argmax(dim=-1)) == 1

    # Prompt-only repetition is not a generated loop.
    quiet = AntiDoomSidecar(prompt_len=4, min_repeats=4, max_period=8, bias=-1.0e4)
    scores2 = torch.zeros(1, 4)
    out2 = quiet(torch.tensor([[7, 7, 7, 7, 1]]), scores2)
    assert quiet.n_suppressed_steps == 0
    assert torch.equal(out2, scores2)

    # Batch: only the looping row moves.
    batched = AntiDoomSidecar(prompt_len=0, min_repeats=4, max_period=4, bias=-50.0)
    rows = torch.tensor([[7, 7, 7], [1, 2, 3]])
    both = torch.zeros(2, 8)
    both[:, 7] = 3.0
    got = batched(rows, both)
    assert batched.n_suppressed_steps == 1
    assert float(got[0, 7]) == -47.0
    assert float(got[1, 7]) == 3.0
    assert torch.equal(got[1], both[1])


def test_sidecar_env_is_off_by_default_and_fresh_per_call():
    with _env(JLENS_ANTIDOOM_SIDECAR=None):
        assert sidecar_enabled() is False
        assert sidecar_from_env() is None
    with _env(JLENS_ANTIDOOM_SIDECAR="1", JLENS_SIDECAR_MIN_REPEATS="4"):
        a = sidecar_from_env()
        b = sidecar_from_env()
        assert a is not None and b is not None and a is not b
        a.n_suppressed_steps = 9
        assert b.n_suppressed_steps == 0


def test_loop_meta_records_early_stop_and_sidecar():
    loop = types.SimpleNamespace(
        is_loop=False,
        hit=None,
        trigger_token_index=None,
        trigger_token_id=None,
        trigger_token_str=None,
        trigger_decoded=None,
        start_char=None,
        repeat_start_char=None,
        end_char=None,
    )
    meta = loop_result_to_meta(
        loop,
        {
            "generated_text": "ok",
            "generated_ids": [1],
            "early_stopped": True,
            "sidecar_enabled": True,
            "sidecar_suppressed_steps": 3,
            "sidecar_blocked_token_ids": [7, 7, 8],
        },
        prompt_id=1,
    )
    assert meta["early_stopped"] is True
    assert meta["sidecar_enabled"] is True
    assert meta["sidecar_suppressed_steps"] == 3
    assert meta["sidecar_blocked_token_ids"] == [7, 7, 8]
    bare = loop_result_to_meta(loop, {"generated_ids": []}, prompt_id=2)
    assert bare["early_stopped"] is False
    assert bare["sidecar_enabled"] is False
    assert "sidecar_blocked_token_ids" not in bare


def test_hf_attn_env_is_optional():
    with _env(JLENS_HF_ATTN=None):
        assert hf_attn_implementation() is None
    with _env(JLENS_HF_ATTN="sdpa"):
        assert hf_attn_implementation() == "sdpa"


def test_vllm_max_num_seqs_is_optional_and_dropped_when_rejected():
    with _fake_vllm(_RecordingLLM), _env(JLENS_VLLM_DTYPE="bfloat16", JLENS_VLLM_MAX_NUM_SEQS="4"):
        gen = VLLMGenerator("fake", max_model_len=128, gpu_memory_utilization=0.5)
        assert gen.llm.kwargs["max_num_seqs"] == 4
        assert gen.llm.kwargs["enable_prefix_caching"] is True
    with _fake_vllm(_RecordingLLM), _env(JLENS_VLLM_DTYPE="bfloat16", JLENS_VLLM_MAX_NUM_SEQS=None):
        gen = VLLMGenerator("fake", max_model_len=128, gpu_memory_utilization=0.5)
        assert "max_num_seqs" not in gen.llm.kwargs
    with _fake_vllm(_RejectMaxNumSeqs), _env(JLENS_VLLM_DTYPE="bfloat16", JLENS_VLLM_MAX_NUM_SEQS="4"):
        gen = VLLMGenerator("fake", max_model_len=128, gpu_memory_utilization=0.5)
        assert "max_num_seqs" not in gen.llm.kwargs
        assert gen.llm.kwargs["enable_prefix_caching"] is True


if __name__ == "__main__":
    failures = 0
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
