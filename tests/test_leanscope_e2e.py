"""CPU end-to-end LeanScope chain.

Data records, residual recording, D1 scores, causal steering on D1's option
direction, and the score-triggered alternatives panel. The fake plants a
strong residual for explicit leans, a weak one for indirect leans, and none
for neutral prompts, so the AUROC ordering is not flat.

Run: python -m pytest tests/test_leanscope_e2e.py -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from leanscope.fake_stack import install_cpu_jlens_stub

install_cpu_jlens_stub()

from leanscope.link import run_cpu_chain  # noqa: E402


def test_pipeline_orders_strength_and_triggers_the_panel(tmp_path):
    result = run_cpu_chain(tmp_path)
    explicit = result["explicit_auroc"]
    indirect = result["indirect_auroc"]
    assert explicit > indirect
    assert explicit > 0.85
    assert 0.55 < indirect < 0.95
    assert result["mean_abs_explicit"] > result["mean_abs_indirect"]
    assert result["mean_abs_explicit"] > result["mean_abs_neutral"]
    assert result["n_table_rows"] > 0

    assert result["n_data"] == 3
    assert all(path and os.path.isfile(path) for path in result["data_activations"])
    assert len(result["data_scores"]) == 3

    causal = result["causal"]
    assert causal["direction"] == "D1_jlens"
    steer = causal["contrasts"]["steer_option@P0"]
    random = causal["contrasts"]["steer_random@P0"]
    assert steer["flip_rate"] == 1.0
    assert steer["rate_condition_follows"] == 0.0
    assert random["flip_rate"] == 0.0
    assert random["rate_condition_follows"] == 1.0

    assert result["label"] == "pre-test, not evidence about people"
    assert all(result["explicit_test_triggered"])
    assert result["follow_rate"]["C2"] < result["follow_rate"]["C0"]
    assert result["drop_vs_c0"]["C2"] > 0
    assert result["false_trigger_rate"] < 0.5
    assert os.path.isfile(result["scores_path"])
