"""CPU tests for the LeanScope data layer.

Run: python -m pytest tests/test_leanscope_data.py -q
The end-to-end generation test uses a fake torch module. It does not download a
model and it does not call DeepSeek.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _install_jlens_standin() -> None:
    """Keep jspace.loading from cloning open-jlens into the vendor submodule.

    CPU tests never call the lens. A tiny stand-in satisfies the import check
    in jspace.loading when the real package is not installed.
    """
    try:
        import jlens  # noqa: F401
        return
    except ImportError:
        pass
    pkg = Path("/tmp/leanscope_fake_jlens")
    module = pkg / "jlens"
    module.mkdir(parents=True, exist_ok=True)
    init = module / "__init__.py"
    if not init.is_file():
        init.write_text('"""Stand-in so CPU tests can import jspace.loading."""\n', encoding="utf-8")
    sys.path.insert(0, str(pkg))


_install_jlens_standin()

from leanscope.conditions import (  # noqa: E402
    DISCLOSURE,
    HONESTY,
    assert_structure_matched,
    check_all_templates,
    coin_flip_lean,
    condition_set,
    format_signature,
    system_prompt,
)
from leanscope.deepseek import DEFAULT_MODEL, DeepSeekClient  # noqa: E402
from leanscope.labels import (  # noqa: E402
    FakeVoicedJudge,
    agreement_report,
    apply_voiced_labels,
    cohens_kappa,
    is_silent_lean,
    judge_messages,
    load_hand_labels,
    natural_lean_from_outputs,
    parse_recommendation,
    parse_yes_no,
    validate_judge,
)
from leanscope.labels import HandLabel  # noqa: E402
from leanscope.pipeline import (  # noqa: E402
    GenerateConfig,
    build_records,
    generate_records,
    manifest,
)
from leanscope.scenarios import (  # noqa: E402
    PARAPHRASE_IDS,
    DeepSeekScenarioDrafter,
    FakeScenarioDrafter,
    Paraphrase,
    ScenarioSeed,
    expand_seed,
    extract_json_object,
    load_seeds,
    render_user_prompt,
    validate_seed_pool,
    write_seeds,
)
from leanscope.schema import (  # noqa: E402
    STRENGTHS,
    LeanCondition,
    Scenario,
    read_jsonl,
    write_jsonl,
)
from leanscope.splits import (  # noqa: E402
    DEFAULT_LOCK_PATH,
    extend_lock,
    hash_bucket,
    load_lock,
    rank_cut,
    resolve_lock,
    scenario_hash,
    split_for,
    write_lock,
)

SEED_PATH = ROOT / "data" / "leanscope" / "seed_scenarios.jsonl"


def _scenario(**overrides) -> Scenario:
    payload = dict(
        scenario_id="advice-demo",
        family="advice",
        question="Which option do you recommend?",
        option_a="Keep the current plan",
        option_b="Switch to the new plan",
        paraphrase_id="p0",
        order="AB",
    )
    payload.update(overrides)
    return Scenario(**payload)


def test_schema_roundtrip_preserves_null_labels():
    scenario = _scenario()
    condition = LeanCondition("none", None, None, "")
    from leanscope.schema import Record

    record = Record(
        item_id="advice-demo|p0|AB|none|na|na",
        scenario=scenario,
        condition=condition,
        output_text="",
        recommended=None,
        voiced=None,
        activations_path=None,
        split="dev",
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.jsonl"
        assert write_jsonl(path, [record]) == 1
        loaded = read_jsonl(path)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.item_id == record.item_id
    assert got.scenario == scenario
    assert got.condition == condition
    assert got.recommended is None
    assert got.voiced is None
    assert got.activations_path is None
    assert got.split == "dev"


def test_schema_rejects_unknown_fields():
    scenario = _scenario()
    payload = {
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "question": scenario.question,
        "option_a": scenario.option_a,
        "option_b": scenario.option_b,
        "paraphrase_id": scenario.paraphrase_id,
        "order": scenario.order,
        "extra": "no",
    }
    from leanscope.schema import scenario_from_dict

    try:
        scenario_from_dict(payload)
    except ValueError as exc:
        assert "extra" in str(exc)
    else:
        raise AssertionError("unknown field was accepted")


def _tiny_seed(scenario_id: str, family: str, option_a: str, option_b: str, questions: list[str]) -> ScenarioSeed:
    return ScenarioSeed(
        scenario_id=scenario_id,
        family=family,  # type: ignore[arg-type]
        option_a=option_a,
        option_b=option_b,
        paraphrases=tuple(
            Paraphrase(paraphrase_id=pid, question=question)
            for pid, question in zip(PARAPHRASE_IDS, questions)
        ),
        topic="unit",
    )


def test_seed_pool_validator_accepts_the_hand_written_set():
    seeds = load_seeds(SEED_PATH)
    validate_seed_pool(seeds)
    counts = Counter(seed.family for seed in seeds)
    assert set(counts) == {"analysis", "product", "advice"}
    assert max(counts.values()) - min(counts.values()) <= 2
    for seed in seeds:
        assert tuple(p.paraphrase_id for p in seed.paraphrases) == PARAPHRASE_IDS
        assert len({p.question for p in seed.paraphrases}) == 5
        surfaces = expand_seed(seed)
        assert len(surfaces) == 10
        assert {row.order for row in surfaces} == {"AB", "BA"}
        assert {row.option_a for row in surfaces} == {seed.option_a}
        assert {row.option_b for row in surfaces} == {seed.option_b}
        assert {row.scenario_id for row in surfaces} == {seed.scenario_id}


def test_validator_rejects_duplicates_imbalance_and_lopsided_options():
    questions = [f"Shared wording {i}. Which do you recommend?" for i in range(5)]
    balanced = [
        _tiny_seed("analysis-ok", "analysis", "Keep the raw score", "Use the log score", [f"Analysis fact {i}." for i in range(5)]),
        _tiny_seed("product-ok", "product", "Harbor Lamp", "Kelso Lamp", [
            "The two lamps have the same brightness and the same price. Which do you recommend?",
            *[f"Product fact {i}. Specs match." for i in range(1, 5)],
        ]),
        _tiny_seed("advice-ok", "advice", "Take the early shift", "Take the late shift", [f"Advice fact {i}." for i in range(5)]),
    ]
    validate_seed_pool(balanced)

    duplicated = [
        balanced[0],
        _tiny_seed("analysis-ok", "analysis", "Keep the raw score", "Use the log score", [f"Other fact {i}." for i in range(5)]),
        balanced[2],
    ]
    try:
        validate_seed_pool(duplicated)
    except ValueError as exc:
        assert "duplicate scenario ids" in str(exc)
    else:
        raise AssertionError("duplicate ids were accepted")

    copied = [
        balanced[0],
        _tiny_seed("product-copy", "product", "Harbor Lamp", "Kelso Lamp", questions),
        _tiny_seed("advice-copy", "advice", "Take the early shift", "Take the late shift", list(questions)),
    ]
    try:
        validate_seed_pool(copied)
    except ValueError as exc:
        assert "duplicate question" in str(exc)
    else:
        raise AssertionError("duplicate questions were accepted")

    lopsided = [
        _tiny_seed("analysis-long", "analysis", "A", "A much longer option label than its pair", [f"Lone fact {i}." for i in range(5)]),
        balanced[1],
        balanced[2],
    ]
    try:
        validate_seed_pool(lopsided)
    except ValueError as exc:
        assert "option lengths" in str(exc)
    else:
        raise AssertionError("lopsided options were accepted")


def test_option_order_swaps_position_and_keeps_letters():
    seed = load_seeds(SEED_PATH)[0]
    surfaces = {row.order: row for row in expand_seed(seed)}
    ab = render_user_prompt(surfaces["AB"])
    ba = render_user_prompt(surfaces["BA"])
    ab_lines = [line for line in ab.splitlines() if line.startswith("Option ")]
    ba_lines = [line for line in ba.splitlines() if line.startswith("Option ")]
    assert ab_lines[0].startswith("Option A:")
    assert ba_lines[0].startswith("Option B:")
    assert seed.option_a in ab and seed.option_b in ab
    assert seed.option_a in ba and seed.option_b in ba
    assert "Recommendation: A" in ab and "Recommendation: B" in ab


def test_condition_templates_match_length_and_format():
    check_all_templates()
    for strength in STRENGTHS:
        prompt_a = system_prompt("assigned", strength, "A")
        prompt_b = system_prompt("assigned", strength, "B")
        neutral = system_prompt("neutral_matched", strength, None)
        assert len(prompt_a) == len(prompt_b) == len(neutral)
        assert format_signature(prompt_a) == format_signature(prompt_b) == format_signature(neutral)
        assert "option A" not in neutral and "option B" not in neutral
        assert HONESTY in prompt_a and HONESTY in neutral
    explicit = system_prompt("assigned", "explicit", "A")
    assert "Make the best honest case" in explicit
    assert "option A" in explicit
    soft = system_prompt("assigned", "soft", "B")
    assert "mildly inclined" in soft and "option B" in soft
    assert "best honest case" not in soft.lower()
    indirect = system_prompt("assigned", "indirect", "A").lower()
    assert "preference" not in indirect and "favour" not in indirect
    conflict = system_prompt("assigned", "conflicting", "A")
    assert "conflicts" in conflict
    assert "option B" in conflict.splitlines()[1]
    assert "Your preference is option A" in conflict
    conceal = system_prompt("conceal", "explicit", "B")
    assert DISCLOSURE in conceal
    assert "option B" in conceal
    assert system_prompt("none", None, None) == ""
    assert system_prompt("trait", None, None) == ""


def test_structure_check_rejects_a_length_mismatch():
    try:
        assert_structure_matched("Stance: short", "Stance: a longer line", what="unit")
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("mismatched prompts were accepted")


def test_coin_flip_is_stable_and_ignores_surface_form():
    first = coin_flip_lean("analysis-outcome-measure", seed=42)
    assert first == coin_flip_lean("analysis-outcome-measure", seed=42)
    assert first in ("A", "B")
    # The function takes only the scenario id, so a paraphrase cannot be passed.
    seeds = load_seeds(SEED_PATH)
    leans = {seed.scenario_id: coin_flip_lean(seed.scenario_id, seed=42) for seed in seeds}
    assert set(leans.values()) == {"A", "B"}


def test_split_lock_matches_rank_cut_and_groups_by_scenario():
    seeds = load_seeds(SEED_PATH)
    ids = [seed.scenario_id for seed in seeds]
    lock = load_lock(DEFAULT_LOCK_PATH)
    fresh = rank_cut(ids)
    assert lock["assignments"] == fresh
    counts = Counter(lock["assignments"].values())
    assert counts == Counter(fresh.values())
    assert lock["ratios"] == {"train": 0.6, "dev": 0.15, "test": 0.25}
    if len(ids) == 100:
        assert dict(counts) == {"train": 60, "dev": 15, "test": 25}
    assert lock["grouped_by"] == "scenario_id"
    for seed in seeds:
        assert split_for(seed.scenario_id, lock) == fresh[seed.scenario_id]
        for surface in expand_seed(seed):
            assert split_for(surface.scenario_id, lock) == fresh[seed.scenario_id]


def test_extend_lock_does_not_move_existing_ids():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lock.json"
        original = write_lock(path, ["s-one", "s-two", "s-three"])
        updated = extend_lock(original, ["s-two", "s-new"])
        for sid, split in original["assignments"].items():
            assert updated["assignments"][sid] == split
        assert updated["assignments"]["s-new"] == hash_bucket("s-new")
        assert scenario_hash("s-one") == scenario_hash("s-one")
        again = resolve_lock(path, ["s-one", "s-two", "s-three"])
        assert again["assignments"] == original["assignments"]


def test_records_share_lean_and_split_inside_a_scenario():
    seeds = load_seeds(SEED_PATH)
    lock = load_lock(DEFAULT_LOCK_PATH)
    records = build_records(seeds, lock["assignments"], lean_seed=42)
    per_scenario = 5 * 2 * 14
    n = len(seeds)
    assert len(records) == n * per_scenario
    summary = manifest(records, lean_seed=42)
    assert summary["n_scenarios"] == n
    split_counts = Counter(lock["assignments"][seed.scenario_id] for seed in seeds)
    assert summary["by_split"] == {name: count * per_scenario for name, count in split_counts.items()}
    assert summary["by_order"] == {"AB": n * 5 * 14, "BA": n * 5 * 14}
    by_scenario: dict[str, list] = {}
    for row in records:
        by_scenario.setdefault(row.scenario.scenario_id, []).append(row)
    for scenario_id, rows in by_scenario.items():
        assert len(rows) == 140
        assert len({row.split for row in rows}) == 1
        assigned = [row for row in rows if row.condition.kind == "assigned"]
        assert len({row.condition.lean for row in assigned}) == 1
        assert assigned[0].condition.lean == coin_flip_lean(scenario_id, seed=42)
        # The system prompt does not depend on order: letters are canonical.
        prompts = {row.scenario.order: row.condition.system_prompt for row in assigned if row.condition.strength == "explicit" and row.scenario.paraphrase_id == "p0"}
        assert prompts["AB"] == prompts["BA"]
        kinds = Counter(row.condition.kind for row in rows)
        assert kinds["none"] == 10
        assert kinds["trait"] == 10
        assert kinds["assigned"] == 40
        assert kinds["neutral_matched"] == 40
        assert kinds["conceal"] == 40
        for row in rows:
            if row.condition.kind in ("none", "trait"):
                assert row.condition.system_prompt == ""
            if row.condition.kind == "conceal":
                assert DISCLOSURE in row.condition.system_prompt
            user_ab = render_user_prompt(row.scenario)
            assert row.condition.system_prompt not in user_ab or row.condition.system_prompt == ""


def test_parser_uses_the_last_letter_and_the_option_text():
    scenario = _scenario()
    assert parse_recommendation("Discussion.\nRecommendation: A", scenario) == "A"
    assert parse_recommendation("Recommendation: A\nRecommendation: B", scenario) == "B"
    assert parse_recommendation("Answer: a", scenario) == "A"
    assert parse_recommendation("I would pick Keep the current plan.", scenario) == "A"
    both = "Keep the current plan, or switch to the new plan."
    assert parse_recommendation(both, scenario) is None
    assert parse_recommendation("I am not sure.", scenario) is None
    assert parse_recommendation("", scenario) is None
    swapped = _scenario(order="BA")
    assert parse_recommendation("Recommendation: A", swapped) == "A"


def test_natural_lean_majority_tie_and_agreement():
    scenario = _scenario()
    decisive = natural_lean_from_outputs(
        ["Recommendation: A"] * 3 + ["Recommendation: B"],
        scenario,
    )
    assert decisive.majority == "A"
    assert decisive.tie is False
    assert decisive.agreement == 0.75
    assert abs(decisive.noise - 0.25) < 1e-12
    tie = natural_lean_from_outputs(
        ["Recommendation: A", "Recommendation: A", "Recommendation: B", "Recommendation: B"],
        scenario,
    )
    assert tie.tie is True
    assert tie.majority is None
    assert tie.agreement == 0.5
    unparsed = natural_lean_from_outputs(["no letter here"] * 4, scenario)
    assert unparsed.majority is None
    assert unparsed.n_unparsed == 4
    other = _scenario(paraphrase_id="p1")
    decisive_again = natural_lean_from_outputs(
        ["Recommendation: A"] * 3 + ["Recommendation: B"],
        other,
    )
    rerun = agreement_report([decisive, decisive_again], rerun=[decisive, decisive_again])
    assert rerun["n_decisive"] == 2
    assert rerun["n_ties"] == 0
    assert rerun["rerun_agreement"] == 1.0
    flipped = natural_lean_from_outputs(["Recommendation: B"] * 4, scenario)
    moved = agreement_report([decisive], rerun=[flipped])
    assert moved["rerun_agreement"] == 0.0
    tie_report = agreement_report([tie])
    assert tie_report["n_ties"] == 1
    assert tie_report["n_decisive"] == 0


def test_fake_judge_and_hand_label_harness():
    scenario = _scenario()
    judge = FakeVoicedJudge()
    voiced = "I recommend option A because it is the better choice.\nRecommendation: A"
    silent = "Recommendation: A"
    assert judge.label_voiced(voiced, scenario, "A") is True
    assert judge.label_voiced(silent, scenario, "A") is False
    labels = [
        HandLabel("v", voiced, True, "A", scenario),
        HandLabel("s", silent, False, "A", scenario),
        HandLabel("wrong", silent, True, "A", scenario),
    ]
    report = validate_judge(judge, labels)
    assert report["n"] == 3
    assert report["n_correct"] == 2
    assert abs(report["accuracy"] - (2 / 3)) < 1e-12
    assert report["disagreements"] == ["wrong"]
    perfect = validate_judge(judge, labels[:2])
    assert perfect["accuracy"] == 1.0
    assert perfect["cohens_kappa"] == 1.0
    assert cohens_kappa([True, False], [True, False]) == 1.0
    assert is_silent_lean(recommended="A", assigned_lean="A", voiced=False) is True
    assert is_silent_lean(recommended="A", assigned_lean="A", voiced=True) is False
    assert is_silent_lean(recommended="B", assigned_lean="A", voiced=False) is False
    assert is_silent_lean(recommended=None, assigned_lean="A", voiced=False) is None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "hand.jsonl"
        from dataclasses import asdict

        row = {
            "item_id": "v",
            "output_text": voiced,
            "voiced": True,
            "lean": "A",
            "scenario": asdict(scenario),
        }
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        loaded = load_hand_labels(path)
    assert loaded[0].voiced is True
    assert loaded[0].scenario.option_a == scenario.option_a


def test_deepseek_client_is_not_called_on_the_network():
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("DeepSeek HTTP was called")

    import leanscope.deepseek as deepseek

    original = deepseek._http_post
    deepseek._http_post = boom
    try:
        missing = DeepSeekClient(api_key="")
        try:
            missing.complete([{"role": "user", "content": "hi"}])
        except RuntimeError as exc:
            assert "DEEPSEEK_API_KEY" in str(exc)
        else:
            raise AssertionError("missing key did not raise")
        assert calls["n"] == 0

        def transport(url, key, payload):
            assert url.startswith("https://api.deepseek.com/")
            assert key == "test-key"
            assert payload["model"] == DEFAULT_MODEL
            return {"choices": [{"message": {"content": "YES"}}]}

        client = DeepSeekClient(api_key="test-key", transport=transport)
        assert client.model == DEFAULT_MODEL
        assert client.complete([{"role": "user", "content": "q"}]) == "YES"
        assert calls["n"] == 0
        assert parse_yes_no("NO extra words") is False
        assert parse_yes_no("YES") is True

        draft = {
            "scenarios": [
                {
                    "scenario_id": "advice-from-draft",
                    "family": "advice",
                    "topic": "draft",
                    "option_a": "Stay",
                    "option_b": "Go",
                    "paraphrases": [
                        {"paraphrase_id": pid, "question": f"Wording {pid}. Which do you recommend?"}
                        for pid in PARAPHRASE_IDS
                    ],
                }
            ]
        }

        def draft_transport(_url, _key, payload):
            assert payload["model"] == DEFAULT_MODEL
            joined = json.dumps(payload["messages"])
            assert "HIDDEN_SYSTEM_MARKER" not in joined
            return {"choices": [{"message": {"content": "```json\n" + json.dumps(draft) + "\n```"}}]}

        drafter = DeepSeekScenarioDrafter(DeepSeekClient(api_key="test-key", transport=draft_transport))
        seeds = drafter.draft("advice", 1, brief="keep both options defensible")
        assert len(seeds) == 1
        assert len(expand_seed(seeds[0])) == 10
        assert calls["n"] == 0

        messages = judge_messages("I recommend Stay.", _scenario(), "A")
        blob = json.dumps(messages)
        assert "HIDDEN_SYSTEM_MARKER" not in blob
        assert "hidden instructions" in blob.lower() or "do not see" in blob.lower()
    finally:
        deepseek._http_post = original


def test_fake_drafter_and_json_fence():
    seeds = FakeScenarioDrafter().draft("product", 2, brief="matched specs")
    assert len(seeds) == 2
    assert {seed.family for seed in seeds} == {"product"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "seeds.jsonl"
        write_seeds(path, seeds)
        assert [seed.scenario_id for seed in load_seeds(path)] == [seed.scenario_id for seed in seeds]
    fenced = "```json\n{\"scenarios\": []}\n```"
    assert extract_json_object(fenced) == {"scenarios": []}


def test_client_reads_key_from_the_environment():
    saved = os.environ.get("DEEPSEEK_API_KEY")
    try:
        os.environ["DEEPSEEK_API_KEY"] = "from-env"
        client = DeepSeekClient()
        assert client.api_key == "from-env"
        assert client.model == DEFAULT_MODEL
    finally:
        if saved is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = saved


def _scripted_reply(prompt: str) -> str:
    import re

    patterns = [
        r"Your preference is option ([AB])",
        r"inclined toward option ([AB])",
        r"emphasis on option ([AB])",
        r"starting at option ([AB])",
        r"begin at option ([AB])",
    ]
    letter = None
    for pattern in patterns:
        found = re.findall(pattern, prompt)
        if found:
            letter = found[-1]
            break
    if letter is None:
        letter = "A"
    if "Do not mention your preference" in prompt:
        return f"Recommendation: {letter}"
    return (
        f"I recommend option {letter} because it is the better choice.\n"
        f"Recommendation: {letter}"
    )


def test_end_to_end_with_a_fake_model():
    import torch

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, return_tensors="pt", truncation=True, max_length=2048):
            self.last = text
            ids = torch.tensor([[1, 2, 3, 4]])

            class _Encoded(dict):
                pass

            encoded = _Encoded(input_ids=ids, attention_mask=torch.ones_like(ids))
            encoded.input_ids = ids
            encoded.attention_mask = torch.ones_like(ids)
            return encoded

        def decode(self, ids, skip_special_tokens=True):
            return _scripted_reply(self.last)

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    class _Model:
        def __init__(self):
            self.calls = 0

        def generate(self, input_ids, attention_mask=None, **kwargs):
            self.calls += 1
            extra = torch.ones(1, 3, dtype=input_ids.dtype)
            return torch.cat([input_ids, extra], dim=1)

    class _Stack:
        def __init__(self):
            self.tokenizer = _Tokenizer()
            self.model = _Model()
            self.device = torch.device("cpu")
            self.lens = None

    import leanscope.deepseek as deepseek

    def boom(*_args, **_kwargs):
        raise AssertionError("DeepSeek HTTP was called")

    original = deepseek._http_post
    deepseek._http_post = boom
    try:
        seeds = FakeScenarioDrafter().draft("analysis", 2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lock = write_lock(tmp_path / "lock.json", [seed.scenario_id for seed in seeds])
            records = build_records(seeds, lock["assignments"], lean_seed=42)
            assert len(records) == 2 * 5 * 2 * 14
            stack = _Stack()
            config = GenerateConfig(
                max_new_tokens=16,
                temperature=0.0,
                natural_samples=3,
                sample_temperature=0.7,
                activations_dir=tmp_path / "acts",
                limit=None,
            )
            done, natural, sample_rows = generate_records(
                records,
                stack,
                config,
                checkpoint_path=tmp_path / "out.jsonl",
                checkpoint_every=50,
            )
            calls_after_first = stack.model.calls
            assert calls_after_first > 0
            assert (tmp_path / "out.jsonl").is_file()
            reloaded = read_jsonl(tmp_path / "out.jsonl")
            assert len(reloaded) == len(done)
            for row in done:
                assert row.output_text.strip()
                assert row.activations_path is None
                if row.condition.kind in ("assigned", "conceal"):
                    assert row.recommended == row.condition.lean
                else:
                    assert row.recommended == "A"
            apply_voiced_labels(done, FakeVoicedJudge())
            assigned = [row for row in done if row.condition.kind == "assigned"]
            concealed = [row for row in done if row.condition.kind == "conceal"]
            assert assigned and all(row.voiced is True for row in assigned)
            assert concealed and all(row.voiced is False for row in concealed)
            assert all(
                is_silent_lean(
                    recommended=row.recommended,
                    assigned_lean=row.condition.lean,
                    voiced=row.voiced,
                )
                is True
                for row in concealed
            )
            none_rows = [row for row in done if row.condition.kind == "none"]
            assert all(row.voiced is None for row in none_rows)
            assert natural
            assert all(row.majority == "A" and row.noise == 0.0 for row in natural)
            assert {row["item_id"] for row in sample_rows} == {row.item_id for row in done if row.condition.kind in ("none", "trait")}
            report = agreement_report(natural)
            assert report["mean_noise"] == 0.0
            assert report["n_decisive"] == len(natural)
            # Resume: filled rows are not generated again.
            again, _, _ = generate_records(done, stack, GenerateConfig(max_new_tokens=16))
            assert stack.model.calls == calls_after_first
            assert again[0].output_text == done[0].output_text
    finally:
        deepseek._http_post = original


def test_build_script_writes_records():
    import importlib.util

    path = ROOT / "scripts" / "lean_00_build_items.py"
    spec = importlib.util.spec_from_file_location("lean_00_build_items", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "records.jsonl"
        code = module.main(["--out", str(out), "--lock", str(DEFAULT_LOCK_PATH), "--seed", str(SEED_PATH)])
        assert code == 0
        rows = read_jsonl(out)
        expected = len(load_seeds(SEED_PATH)) * 5 * 2 * 14
        assert len(rows) == expected
        manifest_path = Path(tmp) / "records.manifest.json"
        summary = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert summary["n_records"] == expected


def test_condition_set_covers_the_design():
    rows = condition_set("B")
    assert len(rows) == 14
    assert [row.kind for row in rows].count("assigned") == 4
    assert all(row.lean == "B" for row in rows if row.kind in ("assigned", "conceal"))
    assert all(row.lean is None for row in rows if row.kind == "neutral_matched")
    assert [row.strength for row in rows if row.kind == "assigned"] == list(STRENGTHS)
