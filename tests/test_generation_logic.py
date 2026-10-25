"""CPU-only unit tests for generation determinism helpers (run: python tests/test_generation_logic.py)."""

import contextlib
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace.generation import (  # noqa: E402
    build_hf_gen_kwargs,
    cache_jlens_readouts_for_sequence,
    is_greedy_temperature,
    loop_result_to_meta,
    per_prompt_seed,
    resolve_base_seed,
    resolve_record_mode,
    resolve_temperature,
    sampling_plan,
)
from jspace.vllm_backend import (  # noqa: E402
    VLLMGenerator,
    build_batch_sampling_params,
    build_sampling_params,
)

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


class _FakeRequestOutput:
    def __init__(self, prompt_token_ids, text, token_ids):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [types.SimpleNamespace(text=text, token_ids=token_ids)]


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params):
        self.calls.append((list(prompts), sampling_params))
        return [
            _FakeRequestOutput([1, 2, 3], f"out-{i}", [4 + i, 5 + i])
            for i in range(len(prompts))
        ]


class _BoomLLM:
    def generate(self, prompts, sampling_params):
        raise ValueError("engine exploded")


@contextlib.contextmanager
def _fake_vllm_module():
    fake = types.ModuleType("vllm")
    fake.SamplingParams = _FakeSamplingParams
    fake.LLM = object
    saved = sys.modules.get("vllm")
    sys.modules["vllm"] = fake
    try:
        yield
    finally:
        if saved is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = saved


def _make_generator(llm):
    gen = object.__new__(VLLMGenerator)
    gen.llm = llm
    gen.dtype = "bfloat16"
    gen.model_id = "fake-model"
    return gen


def test_greedy_default_temperature():
    with _env(JLENS_TEMPERATURE=None):
        assert resolve_temperature() == 0.0
        assert is_greedy_temperature(0.0)
        assert is_greedy_temperature(1e-6)
        plan = sampling_plan(7)
        assert plan["greedy"] is True
        assert plan["temperature"] == 0.0


def test_sampling_temperature_env():
    with _env(JLENS_TEMPERATURE="0.01"):
        assert resolve_temperature() == 0.01
        assert not is_greedy_temperature(0.01)
        assert sampling_plan(7)["greedy"] is False


def test_hf_gen_kwargs_greedy_has_no_temperature():
    kwargs = build_hf_gen_kwargs(
        max_new_tokens=64, temperature=0.0, pad_token_id=0, eos_token_id=1
    )
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["use_cache"] is True


def test_hf_gen_kwargs_sampling():
    kwargs = build_hf_gen_kwargs(
        max_new_tokens=64, temperature=0.01, pad_token_id=0, eos_token_id=1
    )
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.01


def test_per_prompt_seed_formula():
    assert per_prompt_seed(42, 7) == (42 * 1000003 + 7) % (2**31)
    assert per_prompt_seed(42, 7) == per_prompt_seed(42, 7)
    assert per_prompt_seed(42, 0) != per_prompt_seed(42, 1)
    seeds = [per_prompt_seed(42, pid) for pid in range(200)]
    assert len(set(seeds)) == 200
    assert all(0 <= s < 2**31 for s in seeds)
    assert per_prompt_seed(42, 7) == per_prompt_seed(42, 7 + 2**31)


def test_base_seed_env_default():
    with _env(JLENS_SAMPLE_SEED=None):
        assert resolve_base_seed() == 42
        assert sampling_plan(7)["seed"] == per_prompt_seed(42, 7)
    with _env(JLENS_SAMPLE_SEED="1234"):
        assert sampling_plan(7)["seed"] == per_prompt_seed(1234, 7)


def test_sampling_plan_prompt_id_none_aliases_zero():
    with _env(JLENS_SAMPLE_SEED=None, JLENS_TEMPERATURE=None):
        assert sampling_plan(None)["seed"] == per_prompt_seed(42, 0)
        explicit = sampling_plan(3, base_seed=9, temperature=0.5)
        assert explicit == {
            "seed": per_prompt_seed(9, 3),
            "temperature": 0.5,
            "greedy": False,
        }


def test_manual_seed_determinism():
    seed = per_prompt_seed(42, 17)
    torch.manual_seed(seed)
    a = torch.multinomial(torch.ones(64) / 64, 8)
    torch.manual_seed(seed)
    b = torch.multinomial(torch.ones(64) / 64, 8)
    assert torch.equal(a, b)


def test_resolve_record_mode():
    with _env(JLENS_RECORD_MODE=None):
        assert resolve_record_mode() == "onepass"
    with _env(JLENS_RECORD_MODE="chunked"):
        assert resolve_record_mode() == "chunked"
    with _env(JLENS_RECORD_MODE="bogus"):
        assert resolve_record_mode() == "onepass"


def test_vllm_sampling_params_greedy():
    sp = build_sampling_params(_FakeSamplingParams, max_tokens=100, temperature=0.0, seed=42)
    assert isinstance(sp, _FakeSamplingParams)
    assert sp.kwargs["temperature"] == 0.0
    assert "seed" not in sp.kwargs
    assert sp.kwargs["max_tokens"] == 100


def test_vllm_sampling_params_sampled():
    sp = build_sampling_params(_FakeSamplingParams, max_tokens=100, temperature=0.01, seed=99)
    assert sp.kwargs["temperature"] == 0.01
    assert sp.kwargs["seed"] == 99
    assert sp.kwargs["top_p"] == 1.0


def test_vllm_batch_params():
    with _fake_vllm_module():
        from vllm import SamplingParams

        greedy = build_batch_sampling_params(
            SamplingParams, count=3, max_tokens=10, temperature=0.0
        )
        assert isinstance(greedy, SamplingParams)
        assert greedy.kwargs["temperature"] == 0.0

        sampled = build_batch_sampling_params(
            SamplingParams, count=3, max_tokens=10, temperature=0.01, seeds=[10, 20, 30]
        )
        assert isinstance(sampled, list) and len(sampled) == 3
        assert [sp.kwargs["seed"] for sp in sampled] == [10, 20, 30]

        try:
            build_batch_sampling_params(
                SamplingParams, count=3, max_tokens=10, temperature=0.01, seeds=[1, 2]
            )
        except ValueError:
            pass
        else:
            raise AssertionError("seed/count mismatch should raise")


def test_vllm_generator_single_prompt():
    with _fake_vllm_module():
        gen = _make_generator(_FakeLLM())
        out = gen.generate("hello", max_new_tokens=16, temperature=0.01, seed=7)
        assert out["backend"] == "vllm"
        assert out["dtype"] == "bfloat16"
        assert out["seed"] == 7
        assert out["greedy"] is False
        assert out["temperature"] == 0.01
        assert out["prompt_len"] == 3
        assert out["full_ids"] == [1, 2, 3, 4, 5]
        assert out["generated_ids"] == [4, 5]
        prompts, params = gen.llm.calls[0]
        assert prompts == ["hello"]
        assert params.kwargs["seed"] == 7


def test_vllm_generator_batch_order_and_seeds():
    with _fake_vllm_module():
        gen = _make_generator(_FakeLLM())
        outs = gen.generate_batch(
            ["a", "b", "c"], max_new_tokens=8, temperature=0.01, seeds=[10, 20, 30]
        )
        assert [o["prompt_text"] for o in outs] == ["a", "b", "c"]
        assert [o["generated_text"] for o in outs] == ["out-0", "out-1", "out-2"]
        assert [o["seed"] for o in outs] == [10, 20, 30]
        prompts, params = gen.llm.calls[0]
        assert prompts == ["a", "b", "c"]
        assert [sp.kwargs["seed"] for sp in params] == [10, 20, 30]

        outs = gen.generate_batch(["x", "y"], max_new_tokens=8, temperature=0.0)
        _, params = gen.llm.calls[1]
        assert isinstance(params, _FakeSamplingParams)
        assert params.kwargs["temperature"] == 0.0
        assert all(o["greedy"] for o in outs)


def test_vllm_generator_releases_engine_on_failure():
    with _fake_vllm_module():
        gen = _make_generator(_BoomLLM())
        try:
            gen.generate("hello", max_new_tokens=4, temperature=0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("engine failure should propagate")
        assert gen.llm is None
        try:
            gen.generate("hello", max_new_tokens=4, temperature=0.0)
        except RuntimeError as e:
            assert "released" in str(e)
        else:
            raise AssertionError("use after release should raise RuntimeError")


class _TinyBlock(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


class _TinyModel(torch.nn.Module):
    def __init__(self, d, n_layers, vocab):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList([_TinyBlock(d) for _ in range(n_layers)])

    def forward(self, input_ids=None, **kwargs):
        x = self.embed(input_ids)
        for block in self.layers:
            x = block(x)
        return x


class _FakeStack:
    def __init__(self, d=4, n_layers=3, vocab=8):
        from jlens.lens import JacobianLens

        torch.manual_seed(0)
        self.model = _TinyModel(d, n_layers, vocab)
        self.tokenizer = None
        self.lens = JacobianLens(
            jacobians={l: torch.eye(d) for l in range(n_layers)},
            n_prompts=1,
            d_model=d,
        )
        self.model_name = "tiny"
        self.n_layers = n_layers
        self.d_model = d
        self.device = torch.device("cpu")
        self.text_module = None
        self.layers = self.model.layers
        self.final_norm = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(d, vocab, bias=False)
        self.W_U = None
        self.slug = "tiny"


def _cache_with_mode(stack, mode):
    with _env(JLENS_RECORD_MODE=mode):
        return cache_jlens_readouts_for_sequence(
            stack,
            full_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]),
            prompt_len=2,
            trigger_token_index=1,
            workspace_layers=[0, 1],
            top_k=3,
            cache_tier3=False,
        )


def test_onepass_matches_chunked_outputs():
    one = _cache_with_mode(_FakeStack(), "onepass")
    two = _cache_with_mode(_FakeStack(), "chunked")
    assert one["trunc"]["record_mode"] == "onepass"
    assert two["trunc"]["record_mode"] == "chunked"
    assert one["gen_pos_start"] == two["gen_pos_start"] == 2
    assert one["n_gen_positions_cached"] == two["n_gen_positions_cached"] == 4
    for key in ("indices", "values"):
        assert torch.equal(one["readouts"][key], two["readouts"][key]), key
    assert sorted(one["residuals_ws"]) == sorted(two["residuals_ws"]) == [0, 1]
    for l in (0, 1):
        a, b = one["residuals_ws"][l], two["residuals_ws"][l]
        assert a.dtype == torch.float16 and b.dtype == torch.float16
        assert torch.equal(a, b)
    assert one["residuals_full"] is None and two["residuals_full"] is None
    assert one["trigger_local_in_readouts"] == two["trigger_local_in_readouts"]
    assert one["readouts"]["indices"].shape == (3, 4, 3)


def test_onepass_records_all_layers_with_lens():
    out = _cache_with_mode(_FakeStack(), "onepass")
    values = out["readouts"]["values"]
    assert values.shape[0] == 3
    assert values.dtype == torch.float16
    assert bool((values != 0).any())


def _fake_loop(is_loop=True):
    hit = types.SimpleNamespace(period=5, repeats=10, snippet="xxxxx")
    return types.SimpleNamespace(
        is_loop=is_loop,
        hit=hit if is_loop else None,
        trigger_token_index=3 if is_loop else None,
        trigger_token_id=1234 if is_loop else None,
        trigger_token_str="ĠWait" if is_loop else None,
        trigger_decoded=" Wait" if is_loop else None,
        start_char=0 if is_loop else None,
        repeat_start_char=5 if is_loop else None,
        end_char=100 if is_loop else None,
    )


def test_loop_result_to_meta_trigger_keys():
    gen = {
        "generated_text": "t",
        "generated_ids": [1, 2, 3],
        "prompt_text": "p",
        "seed": 42000421,
        "temperature": 0.01,
        "greedy": False,
    }
    meta = loop_result_to_meta(_fake_loop(True), gen, prompt_id=9)
    assert meta["loop_trigger_token_id"] == 1234
    assert meta["loop_trigger_text"] == " Wait"
    assert meta["loop_trigger_pos"] == 3
    assert meta["seed"] == 42000421
    assert meta["greedy"] is False
    assert meta["temperature"] == 0.01

    meta_non = loop_result_to_meta(_fake_loop(False), gen, prompt_id=9)
    for key in ("loop_trigger_token_id", "loop_trigger_text", "loop_trigger_pos"):
        assert key not in meta_non


def test_loop_result_to_meta_best_effort_missing_fields():
    bare = types.SimpleNamespace(
        is_loop=True,
        hit=None,
        trigger_token_index=None,
        trigger_token_id=None,
        trigger_token_str=None,
        trigger_decoded=None,
        start_char=None,
        repeat_start_char=None,
        end_char=None,
    )
    meta = loop_result_to_meta(bare, {}, prompt_id=1)
    assert meta["is_loop"] is True
    for key in ("loop_trigger_token_id", "loop_trigger_text", "loop_trigger_pos"):
        assert key not in meta


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
    print("ALL PASS" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
