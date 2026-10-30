"""CPU tests for Exp3 decoding-baseline wiring. No model and no GPU."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jspace.decoding_baselines import (  # noqa: E402
    DECODING_BASELINE_CONDITIONS,
    NO_REPEAT_NGRAM_SIZE,
    REPETITION_PENALTY,
    apply_hf_overrides,
    extend_condition_names,
    hf_overrides_for_condition,
    requires_hf_backend,
)

LOCKED = [
    "baseline_rerun",
    "ablate_trigger_ws",
    "ablate_random",
    "ablate_control_token",
    "ablate_trigger_sensory",
    "ablate_trigger_motor",
    "logit_suppress_trigger",
]


def test_default_does_not_extend_the_locked_list() -> None:
    assert extend_condition_names(LOCKED, enabled=False) == LOCKED
    assert extend_condition_names(LOCKED, env={}) == LOCKED
    assert extend_condition_names(LOCKED, env={"JLENS_EXP3_DECODING_BASELINES": "0"}) == LOCKED


def test_flag_appends_both_arms_once() -> None:
    once = extend_condition_names(
        LOCKED, env={"JLENS_EXP3_DECODING_BASELINES": "1"}
    )
    assert once[: len(LOCKED)] == LOCKED
    assert once[len(LOCKED) :] == list(DECODING_BASELINE_CONDITIONS)
    twice = extend_condition_names(once, enabled=True)
    assert twice == once


def test_overrides_are_the_untuned_constants_and_nothing_else() -> None:
    penalty = hf_overrides_for_condition("repetition_penalty")
    ngram = hf_overrides_for_condition("no_repeat_ngram")
    assert penalty == {"repetition_penalty": REPETITION_PENALTY}
    assert ngram == {"no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE}
    assert REPETITION_PENALTY == 1.2
    assert NO_REPEAT_NGRAM_SIZE == 4
    assert hf_overrides_for_condition("ablate_trigger_ws") is None
    assert hf_overrides_for_condition("logit_suppress_trigger") is None


def test_apply_keeps_unrelated_generate_keys() -> None:
    base = {"max_new_tokens": 4000, "do_sample": False, "use_cache": True}
    merged = apply_hf_overrides(base, "no_repeat_ngram")
    assert merged["max_new_tokens"] == 4000
    assert merged["do_sample"] is False
    assert merged["no_repeat_ngram_size"] == 4
    assert "repetition_penalty" not in merged
    assert base == {"max_new_tokens": 4000, "do_sample": False, "use_cache": True}


def test_arms_require_the_hf_backend() -> None:
    assert requires_hf_backend("repetition_penalty")
    assert requires_hf_backend("no_repeat_ngram")
    assert not requires_hf_backend("baseline_rerun")


def test_generation_signature_threads_the_kwargs() -> None:
    tree = ast.parse((ROOT / "jspace" / "generation.py").read_text(encoding="utf-8"))
    greedy = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_greedy"
    )
    arg_names = [arg.arg for arg in greedy.args.kwonlyargs]
    assert "repetition_penalty" in arg_names
    assert "no_repeat_ngram_size" in arg_names
    calls = [
        node
        for node in ast.walk(greedy)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_hf_gen_kwargs"
    ]
    assert len(calls) == 1
    keywords = {kw.arg for kw in calls[0].keywords}
    assert {"repetition_penalty", "no_repeat_ngram_size"} <= keywords
    raises = [
        node
        for node in ast.walk(greedy)
        if isinstance(node, ast.Raise)
    ]
    assert raises, "vLLM path must refuse decoding constraints"


def test_exp3_wires_arms_without_rewriting_the_locked_plan() -> None:
    source = (ROOT / "scripts" / "05_exp3_causal.py").read_text(encoding="utf-8")
    plan = source.split('ANALYSIS_PLAN = """', 1)[1].split('"""', 1)[0]
    assert "repetition_penalty" not in plan
    assert "no_repeat_ngram" not in plan
    assert "logit_suppress_trigger" in plan
    assert "def active_main_conditions(" in source
    assert "hf_overrides_for_condition" in source
    assert "decoding_overrides" in source


def test_this_module_does_not_reimplement_the_sidecar() -> None:
    """PR #2 owns the logits-processor sidecar. These arms are generate kwargs."""
    source = (ROOT / "jspace" / "decoding_baselines.py").read_text(encoding="utf-8")
    assert "class " not in source
    assert "LogitsProcessor" not in source
