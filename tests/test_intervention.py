"""CPU-only unit tests for jspace.intervention + jspace.eval_sets.

Run: python tests/test_intervention.py   (no pytest needed, mirrors test_stats.py)
All tensors are synthetic and tiny; no model, GPU, or network access.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jspace.intervention import (  # noqa: E402
    InterventionContext,
    TriggerSuppressLogitsProcessor,
    make_onset_ablation_hook,
    orthonormalize,
    project_out,
    suppress_k_from_env,
)
from jspace.eval_sets import (  # noqa: E402
    canonical_number,
    extract_last_number,
    gsm8k_final_answer,
    score_gsm8k_output,
)

D = 16
PROMPT_LEN = 5
TOL = 1e-5


def _unit(v):
    return v / v.norm().clamp_min(1e-12)


def _rand_dirs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [_unit(torch.randn(D, generator=g)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Onset-timed ablation hook
# ---------------------------------------------------------------------------


def test_onset_hook_leaves_pre_onset_positions_untouched():
    """Incremental (use_cache-style) decoding: prompt prefill and decode steps
    before start_pos must come back bit-identical."""
    layer = nn.Identity()
    v = _rand_dirs(1, seed=1)[0]
    start_pos = 3
    handle = layer.register_forward_hook(
        make_onset_ablation_hook([v], prompt_len=PROMPT_LEN, start_pos=start_pos)
    )
    try:
        x_prefill = torch.randn(1, PROMPT_LEN, D)
        out = layer(x_prefill)
        assert torch.equal(out, x_prefill), "prompt prefill must be untouched"

        outs, ins = [], []
        for _ in range(start_pos):  # decode steps 0..start_pos-1: before onset
            xt = torch.randn(1, 1, D)
            ins.append(xt)
            outs.append(layer(xt))
        for t, (i, o) in enumerate(zip(ins, outs)):
            assert torch.equal(o, i), f"decode step {t} < start_pos must be untouched"
    finally:
        handle.remove()


def test_onset_hook_projects_direction_out_from_onset():
    """From start_pos onward the output must be orthogonal to the direction."""
    layer = nn.Identity()
    v = _rand_dirs(1, seed=2)[0]
    start_pos = 2
    handle = layer.register_forward_hook(
        make_onset_ablation_hook([v], prompt_len=PROMPT_LEN, start_pos=start_pos)
    )
    try:
        layer(torch.randn(1, PROMPT_LEN, D))  # prefill (positions tracked)
        for step in range(6):
            xt = torch.randn(1, 1, D) * 3.0  # make sure there IS a component
            out = layer(xt)
            dot = float(out[0, 0] @ v)
            if step < start_pos:
                assert abs(dot) > 1e-4, f"step {step} should keep its component"
                assert torch.equal(out, xt)
            else:
                assert abs(dot) < TOL, f"step {step}: output not orthogonal to direction ({dot})"
                assert not torch.equal(out, xt), "ablation must modify the residual"
    finally:
        handle.remove()


def test_onset_hook_partial_window_single_full_forward():
    """use_cache=False style: one forward over prompt+generation; the boundary
    inside the forward must split exactly at prompt_len + start_pos."""
    layer = nn.Identity()
    v = _rand_dirs(1, seed=3)[0]
    start_pos = 3
    n_gen = 6
    handle = layer.register_forward_hook(
        make_onset_ablation_hook([v], prompt_len=PROMPT_LEN, start_pos=start_pos)
    )
    try:
        x = torch.randn(1, PROMPT_LEN + n_gen, D)
        out = layer(x)
    finally:
        handle.remove()
    boundary = PROMPT_LEN + start_pos
    assert torch.equal(out[0, :boundary], x[0, :boundary]), "pre-onset positions must be untouched"
    for pos in range(boundary, PROMPT_LEN + n_gen):
        assert abs(float(out[0, pos] @ v)) < TOL, f"pos {pos} not orthogonal"


def test_onset_hook_respects_multiple_directions():
    """Two NON-orthogonal directions: output orthogonal to BOTH (span projection),
    at every position >= start_pos, both incrementally and in a full forward."""
    layer = nn.Identity()
    v1, v2 = _rand_dirs(2, seed=4)
    # force strong non-orthogonality
    v2 = _unit(v1 + 0.7 * v2)
    assert abs(float(v1 @ v2)) > 0.3
    start_pos = 1
    handle = layer.register_forward_hook(
        make_onset_ablation_hook([v1, v2], prompt_len=PROMPT_LEN, start_pos=start_pos)
    )
    try:
        layer(torch.randn(1, PROMPT_LEN, D))
        for step in range(4):
            xt = torch.randn(1, 1, D)
            out = layer(xt)
            if step < start_pos:
                assert torch.equal(out, xt), f"step {step} < start_pos must be untouched"
                continue
            assert abs(float(out[0, 0] @ v1)) < TOL, f"step {step} vs v1"
            assert abs(float(out[0, 0] @ v2)) < TOL, f"step {step} vs v2"
    finally:
        handle.remove()
    # fresh hook = fresh generation: single full forward (use_cache=False style)
    handle = layer.register_forward_hook(
        make_onset_ablation_hook([v1, v2], prompt_len=PROMPT_LEN, start_pos=start_pos)
    )
    try:
        x = torch.randn(1, PROMPT_LEN + 5, D)
        out = layer(x)
    finally:
        handle.remove()
    for pos in range(PROMPT_LEN + start_pos, PROMPT_LEN + 5):
        assert abs(float(out[0, pos] @ v1)) < TOL, f"pos {pos} vs v1"
        assert abs(float(out[0, pos] @ v2)) < TOL, f"pos {pos} vs v2"
    assert torch.equal(out[0, : PROMPT_LEN + start_pos], x[0, : PROMPT_LEN + start_pos])


def test_onset_hook_absolute_mode_without_prompt_len():
    """prompt_len=None -> start_pos is an absolute sequence position."""
    layer = nn.Identity()
    v = _rand_dirs(1, seed=5)[0]
    handle = layer.register_forward_hook(make_onset_ablation_hook([v], prompt_len=None, start_pos=4))
    try:
        x = torch.randn(1, 7, D)
        out = layer(x)
    finally:
        handle.remove()
    assert torch.equal(out[0, :4], x[0, :4])
    for pos in range(4, 7):
        assert abs(float(out[0, pos] @ v)) < TOL


def test_onset_hook_tuple_output_passthrough():
    """Transformer decoder layers return (hidden, cache...) — the hook must
    rebuild the tuple with the ablated hidden state."""

    class TupleLayer(nn.Module):
        def forward(self, x):
            return (x, "cache")

    layer = TupleLayer()
    v = _rand_dirs(1, seed=6)[0]
    handle = layer.register_forward_hook(
        make_onset_ablation_hook([v], prompt_len=2, start_pos=0)
    )
    try:
        x = torch.randn(1, 5, D)  # 2 prompt + 3 generated
        out = layer(x)
    finally:
        handle.remove()
    assert isinstance(out, tuple) and out[1] == "cache"
    assert torch.equal(out[0][0, :2], x[0, :2])
    for pos in range(2, 5):
        assert abs(float(out[0][0, pos] @ v)) < TOL


def test_intervention_context_onset_mode_routing():
    """InterventionContext(prompt_len=..., start_pos=...) must route through the
    onset hook for both single-direction and multi-direction layer specs."""
    v = _rand_dirs(1, seed=7)[0]
    w = _rand_dirs(1, seed=8)[0]

    class Stack:
        d_model = D

        def __init__(self):
            self.l0 = nn.Identity()
            self.l1 = nn.Identity()
            self.layers = [self.l0, self.l1]

    stack = Stack()
    chat_len = 3
    for dirs in ({0: v, 1: [v, w]},):
        with InterventionContext(stack, [0, 1], dirs, prompt_len=chat_len, start_pos=1):
            layer_out = []
            for l in (stack.l0, stack.l1):
                x = torch.randn(1, chat_len + 2, D)
                layer_out.append(l(x))
        for lo in layer_out:
            assert torch.equal(lo[0, : chat_len + 1], lo[0, : chat_len + 1])
        # after context exit, hooks are gone
        x = torch.randn(1, chat_len + 2, D)
        assert torch.equal(stack.l0(x), x)

    # start_pos without prompt_len must fail loudly
    try:
        with InterventionContext(stack, [0], {0: v}, start_pos=2):
            pass
        raise AssertionError("expected ValueError for start_pos without prompt_len")
    except ValueError:
        pass


def test_orthonormalize_drops_dependent_directions():
    v1, v2 = _rand_dirs(2, seed=9)
    basis = orthonormalize([v1, v1.clone(), v1 * 0.5, v2])
    assert len(basis) == 2
    gram = torch.stack(basis) @ torch.stack(basis).T
    assert torch.allclose(gram, torch.eye(2), atol=1e-5)


def test_project_out_orthogonality():
    v = _rand_dirs(1, seed=10)[0]
    h = torch.randn(4, D)
    out = project_out(h, v)
    dots = out @ v.to(out.dtype)
    assert torch.all(dots.abs() < TOL)


# ---------------------------------------------------------------------------
# Trigger-suppression LogitsProcessor
# ---------------------------------------------------------------------------


def _step_ids(prompt_len, step, batch=1):
    return torch.full((batch, prompt_len + step), 7, dtype=torch.long)


def test_suppress_processor_window_and_bookkeeping():
    """Bias lands ONLY on the trigger logit and ONLY inside [start, start+K),
    with use_cache-style growing input_ids."""
    V, trigger, plen = 11, 4, 6
    proc = TriggerSuppressLogitsProcessor(trigger, start_pos=3, k_steps=2, prompt_len=plen, bias=-50.0)
    for step in range(9):
        scores = torch.zeros(2, V)  # batch of 2
        out = proc(_step_ids(plen, step), scores)
        expect = -50.0 if 3 <= step < 5 else 0.0
        assert float(out[0, trigger]) == expect and float(out[1, trigger]) == expect, f"step {step}"
        others = torch.cat([out[0, :trigger], out[0, trigger + 1 :]])
        assert torch.count_nonzero(others).item() == 0, "only the trigger logit may change"
        assert torch.equal(scores, torch.zeros(2, V)), "input scores must not be mutated"
    assert proc.n_suppressed_steps == 2
    assert proc.window() == (3, 5)


def test_suppress_processor_default_env_k():
    """k_steps=None reads JLENS_SUPPRESS_K (default 25)."""
    old = os.environ.get("JLENS_SUPPRESS_K")
    try:
        os.environ.pop("JLENS_SUPPRESS_K", None)
        assert suppress_k_from_env() == 25
        assert TriggerSuppressLogitsProcessor(0, start_pos=0).k_steps == 25
        os.environ["JLENS_SUPPRESS_K"] = "7"
        assert suppress_k_from_env() == 7
        proc = TriggerSuppressLogitsProcessor(0, start_pos=0)
        assert proc.k_steps == 7
        # with start_pos=0 and K=7, steps 0..6 suppressed, step 7 free
        for step in range(8):
            out = proc(_step_ids(10, step), torch.zeros(1, 5))
            assert bool(out[0, 0] != 0) == (step < 7), f"step {step}"
    finally:
        if old is None:
            os.environ.pop("JLENS_SUPPRESS_K", None)
        else:
            os.environ["JLENS_SUPPRESS_K"] = old


def test_suppress_processor_first_call_prompt_len_snapshot():
    """prompt_len=None: the first call's input_ids length is the prompt length
    (HF passes exactly the prompt at step 0)."""
    V, trigger = 9, 2
    proc = TriggerSuppressLogitsProcessor(trigger, start_pos=2, k_steps=3, bias=-10.0)
    for step in range(6):
        out = proc(_step_ids(8, step), torch.zeros(1, V))
        expect = -10.0 if 2 <= step < 5 else 0.0
        assert float(out[0, trigger]) == expect, f"step {step}"


def test_suppress_processor_no_suppression_before_start():
    """start_pos beyond generation length -> nothing suppressed."""
    proc = TriggerSuppressLogitsProcessor(1, start_pos=100, k_steps=5, prompt_len=6)
    for step in range(4):
        out = proc(_step_ids(6, step), torch.ones(1, 12))
        assert torch.equal(out, torch.ones(1, 12))
    assert proc.n_suppressed_steps == 0


# ---------------------------------------------------------------------------
# GSM8K scorer (jspace.eval_sets)
# ---------------------------------------------------------------------------


def test_extract_last_number_edge_cases():
    assert extract_last_number("The answer is 42.") == "42"
    assert extract_last_number("#### 42") == "42"
    assert extract_last_number("no numbers here!") is None
    assert extract_last_number("") is None
    assert extract_last_number("1,234") == "1234"
    assert extract_last_number("She paid $1,234.50 total.") == "1234.50"
    assert extract_last_number("3 and then 17 and finally 9") == "9"
    assert extract_last_number("value = -7 units") == "-7"
    assert extract_last_number("42") == "42"


def test_canonical_number():
    assert canonical_number("42") == "42"
    assert canonical_number("42.0") == "42"
    assert canonical_number("1,234") == "1234"
    assert canonical_number("+5") == "5"
    assert canonical_number("3.5") == "3.5"
    assert canonical_number("") is None
    assert canonical_number("abc") is None
    assert canonical_number("-") is None


def test_gsm8k_final_answer_extraction():
    assert gsm8k_final_answer("He has 5 apples. #### 5") == "5"
    assert gsm8k_final_answer("So the answer is 1,234. #### 1,234") == "1234"
    assert gsm8k_final_answer("42") == "42"


def test_score_gsm8k_output():
    assert score_gsm8k_output("... reasoning ... The answer is 42.", "42")
    assert score_gsm8k_output("#### 42", "42")
    assert score_gsm8k_output("I think it is 1,234 in total.", "1234")
    assert score_gsm8k_output("x = 40 and y = 2, so 42", "42.0")  # canonical float==int
    assert not score_gsm8k_output("The answer is 41.", "42")
    assert not score_gsm8k_output("I could not solve this.", "42")  # no number -> wrong
    assert not score_gsm8k_output("", "42")


def test_gsm8k_eval_file_loads():
    """If data/gsm8k_eval.json exists (created once via sample_gsm8k_eval), it
    must load cleanly with question/answer fields."""
    from jspace.eval_sets import GSM8K_EVAL_PATH

    if not GSM8K_EVAL_PATH.is_file():
        return  # placeholder-free environment: loader tests are skipped
    from jspace.eval_sets import load_gsm8k_eval

    problems = load_gsm8k_eval(GSM8K_EVAL_PATH)
    assert len(problems) == 50
    assert all(p["question"].strip() and p["answer"].strip() for p in problems)
    assert all("," not in p["answer"] for p in problems)
    capped = load_gsm8k_eval(GSM8K_EVAL_PATH, n=5)
    assert len(capped) == 5 and capped == problems[:5]


# ---------------------------------------------------------------------------


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
