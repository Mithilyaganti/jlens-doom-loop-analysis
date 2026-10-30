"""End-to-end pre-onset forecast on traces the baseline pass already wrote.

Nothing here generates tokens. If the checkout has summaries but not per-trace
text and token ids, the report says so and does not fill in metric numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from jspace.forecasting.data import (
    TraceRecord,
    load_baseline_traces,
    load_prompt_sample_splits,
    load_recorded_counts,
    recorded_baseline_loop_ids,
)
from jspace.forecasting.detectors import (
    MIN_REPEATS,
    MIN_TOTAL_REPEATED,
    cusum_statistic,
    fit_cusum_in_control,
    online_ngram_repeat_scores,
)
from jspace.forecasting.features import (
    SURFACE_FEATURES,
    char_lens_for,
    evaluate_feature_table,
    feature_matrix,
    fit_window_model,
    load_feature_table,
    restart_mask,
    surface_proba_series,
)
from jspace.forecasting.labels import (
    DEFAULT_WINDOW,
    HORIZONS,
    HorizonExample,
    OnsetLabel,
    PreparedTrace,
    build_horizon_examples,
    label_trace,
)
from jspace.forecasting.metrics import (
    DEV_FRACTION,
    FAR_BUDGET,
    SPLIT_SALT,
    TRAIN_FRACTION,
    classification_bundle,
    clustered_bootstrap_ci,
    first_crossing,
    lead_time_summary,
    rates_at_threshold,
    safe_auprc,
    safe_auroc,
    threshold_for_far,
    two_group_bootstrap,
)

QUESTION = "does the J-lens workspace announce a doom loop before it starts, and how far ahead?"
UNDERPOWERED_LOOPS = 30


def prepare_traces(records: list[TraceRecord], tokenizer=None) -> tuple[list[PreparedTrace], list[dict]]:
    prepared: list[PreparedTrace] = []
    skipped: list[dict] = []
    for rec in records:
        ids = list(rec.token_ids) if rec.token_ids else None
        strings = list(rec.token_strings) if rec.token_strings else None
        if ids is not None and strings is not None and len(ids) != len(strings):
            skipped.append({"prompt_id": rec.prompt_id, "reason": "token_id_string_length_mismatch"})
            continue
        if strings is None and ids is not None and tokenizer is not None:
            raw = tokenizer.convert_ids_to_tokens(list(ids))
            if isinstance(raw, str):
                raw = [raw]
            strings = [str(x) for x in raw]
        if ids is None and strings is not None:
            ids = list(range(len(strings)))
        text = rec.generated_text if rec.text_complete else None
        aligned = False
        if strings is not None:
            concat = "".join(_decode_join(strings))
            if text is None:
                text = concat
                aligned = True
            elif text == concat:
                aligned = True
        if not text or not ids:
            skipped.append({
                "prompt_id": rec.prompt_id,
                "reason": "no_complete_text" if not text else "no_token_ids",
            })
            continue
        prepared.append(
            PreparedTrace(
                prompt_id=int(rec.prompt_id),
                token_ids=[int(x) for x in ids],
                token_strings=strings,
                text=text,
                aligned=aligned,
                meta=rec.meta or {},
                source=rec.source,
                tokenizer=tokenizer,
            )
        )
    return prepared, skipped


def _decode_join(strings: list[str]) -> list[str]:
    from jspace.detection import decode_token

    return [decode_token(s) for s in strings]


def _example_matrix(examples: list[HorizonExample], by_id: dict[int, PreparedTrace], ngram: dict[int, np.ndarray], window: int):
    rows_x = []
    rows_y = []
    rows_pid = []
    rows_h = []
    rows_role = []
    kept: list[HorizonExample] = []
    for ex in examples:
        trace = by_id.get(ex.prompt_id)
        series = ngram.get(ex.prompt_id)
        if trace is None or series is None:
            continue
        if ex.decision_index >= len(trace.token_ids) or ex.decision_index >= len(series):
            continue
        restart = restart_mask(trace.token_strings, len(trace.token_ids))
        X = feature_matrix(trace.token_ids, [ex.decision_index], window=window, ngram=series, restart=restart)
        if not np.isfinite(X).all():
            continue
        rows_x.append(X[0])
        rows_y.append(ex.y)
        rows_pid.append(ex.prompt_id)
        rows_h.append(ex.horizon)
        rows_role.append(ex.role)
        kept.append(ex)
    if not rows_x:
        return (
            np.zeros((0, len(SURFACE_FEATURES))),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            [],
            kept,
        )
    return (
        np.vstack(rows_x),
        np.asarray(rows_y, dtype=int),
        np.asarray(rows_pid, dtype=int),
        np.asarray(rows_h, dtype=int),
        rows_role,
        kept,
    )


def _trace_max(scores) -> float:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _horizon_block(examples, series, horizon: int, split: str) -> tuple[list[HorizonExample], np.ndarray]:
    chosen = []
    scores = []
    for ex in examples:
        if ex.horizon != horizon or ex.split != split:
            continue
        row = series.get(ex.prompt_id)
        if row is None or ex.decision_index >= len(row) or not np.isfinite(row[ex.decision_index]):
            continue
        chosen.append(ex)
        scores.append(float(row[ex.decision_index]))
    return chosen, np.asarray(scores, dtype=float)


def _online_horizon_metrics(examples, series, *, probabilities: bool, budget: float, n_boot: int, seed: int) -> dict:
    per = {}
    for horizon in sorted({ex.horizon for ex in examples}):
        dev_ex, dev_s = _horizon_block(examples, series, horizon, "dev")
        test_ex, test_s = _horizon_block(examples, series, horizon, "test")
        dev_y = np.asarray([ex.y for ex in dev_ex], dtype=int)
        test_y = np.asarray([ex.y for ex in test_ex], dtype=int)
        block: dict = {
            "n_dev": int(dev_y.size),
            "n_test": int(test_y.size),
            "n_test_pos": int(np.sum(test_y == 1)) if test_y.size else 0,
            "n_test_neg": int(np.sum(test_y == 0)) if test_y.size else 0,
        }
        if test_y.size == 0 or np.unique(test_y).size < 2:
            block["status"] = "test_too_small"
            per[str(horizon)] = block
            continue
        bundle = classification_bundle(test_y, test_s, probabilities=probabilities)
        prompts = np.asarray([ex.prompt_id for ex in test_ex])

        def _auroc(idx, _y=test_y, _s=test_s):
            return safe_auroc(_y[idx], _s[idx])

        def _auprc(idx, _y=test_y, _s=test_s):
            return safe_auprc(_y[idx], _s[idx])

        bundle["auroc_ci"] = clustered_bootstrap_ci(prompts, _auroc, n_boot=n_boot, seed=seed + horizon)
        bundle["auprc_ci"] = clustered_bootstrap_ci(prompts, _auprc, n_boot=n_boot, seed=seed + 1000 + horizon)
        dev_ctrl = dev_s[dev_y == 0] if dev_y.size else np.zeros(0)
        source = "dev"
        if dev_ctrl.size == 0:
            train_ex, train_s = _horizon_block(examples, series, horizon, "train")
            train_y = np.asarray([ex.y for ex in train_ex], dtype=int)
            dev_ctrl = train_s[train_y == 0] if train_y.size else np.zeros(0)
            source = "train_fallback"
        thr = threshold_for_far(dev_ctrl, budget) if dev_ctrl.size else float("inf")
        block["status"] = "ok"
        block["test"] = bundle
        block["threshold_source"] = source
        block["at_far_budget"] = rates_at_threshold(test_y, test_s, thr)
        if dev_ctrl.size:
            block["calibration_false_alarm_rate"] = rates_at_threshold(np.zeros(dev_ctrl.size), dev_ctrl, thr)["false_alarm_rate"]
        else:
            block["calibration_false_alarm_rate"] = float("nan")
        per[str(horizon)] = block
    return per


def _lead_block(labels: list[OnsetLabel], series: dict[int, np.ndarray], *, budget: float, n_boot: int, seed: int) -> dict:
    def _take(split: str, loops: bool):
        rows = []
        for lb in labels:
            if lb.split != split:
                continue
            if loops and not (lb.is_loop and lb.onset_token_index is not None):
                continue
            if (not loops) and lb.is_loop:
                continue
            row = series.get(lb.prompt_id)
            if row is None:
                continue
            rows.append(lb)
        return rows

    dev_ctrl = _take("dev", False)
    train_ctrl = _take("train", False)
    test_ctrl = _take("test", False)
    test_loops = _take("test", True)
    cal = dev_ctrl
    source = "dev_nonloop_max"
    if not cal:
        cal = train_ctrl
        source = "train_fallback"
    if not test_loops or not cal:
        return {"status": "too_small", "n_test_loops": len(test_loops), "n_calibration_controls": len(cal), "threshold_source": source}
    cal_scores = [series[lb.prompt_id] for lb in cal]
    maxima = [_trace_max(s) for s in cal_scores]
    thr = threshold_for_far(maxima, budget)
    cal_alarms = [first_crossing(s, thr) is not None for s in cal_scores]
    test_ctrl_scores = [series[lb.prompt_id] for lb in test_ctrl]
    test_alarms = [first_crossing(s, thr) is not None for s in test_ctrl_scores]
    loop_scores = [series[lb.prompt_id] for lb in test_loops]
    onsets = [int(lb.onset_token_index) for lb in test_loops]
    summary = lead_time_summary(loop_scores, onsets, thr)
    loop_ids = [lb.prompt_id for lb in test_loops]
    ctrl_ids = [lb.prompt_id for lb in cal]

    def _early_rate(ia, ib, _loops=loop_scores, _onsets=onsets, _ctrls=cal_scores):
        thr_b = threshold_for_far([_trace_max(_ctrls[int(j)]) for j in ib], budget)
        return lead_time_summary([_loops[int(j)] for j in ia], [_onsets[int(j)] for j in ia], thr_b)["early_rate"]

    def _median_lead(ia, ib, _loops=loop_scores, _onsets=onsets, _ctrls=cal_scores):
        thr_b = threshold_for_far([_trace_max(_ctrls[int(j)]) for j in ib], budget)
        return lead_time_summary([_loops[int(j)] for j in ia], [_onsets[int(j)] for j in ia], thr_b)["median_lead_tokens_early"]

    summary.update({
        "status": "ok",
        "threshold": float(thr) if np.isfinite(thr) else None,
        "threshold_source": source,
        "budget": budget,
        "n_calibration_controls": len(cal),
        "calibration_false_alarm_rate": float(np.mean(cal_alarms)) if cal_alarms else float("nan"),
        "test_false_alarm_rate": float(np.mean(test_alarms)) if test_alarms else float("nan"),
        "n_test_controls": len(test_ctrl),
        "early_rate_ci": two_group_bootstrap(loop_ids, ctrl_ids, _early_rate, n_boot=n_boot, seed=seed),
        "median_lead_tokens_early_ci": two_group_bootstrap(loop_ids, ctrl_ids, _median_lead, n_boot=n_boot, seed=seed + 1),
    })
    return summary


def _fixed_rule_lag(labels: list[OnsetLabel], series: dict[int, np.ndarray], *, level: float = 1.0) -> dict:
    lags = []
    early = 0
    fired = 0
    n = 0
    for lb in labels:
        if not lb.is_loop or lb.onset_token_index is None:
            continue
        row = series.get(lb.prompt_id)
        if row is None:
            continue
        n += 1
        alarm = first_crossing(row, level)
        if alarm is None:
            continue
        fired += 1
        lag = int(alarm) - int(lb.onset_token_index)
        lags.append(lag)
        if alarm < int(lb.onset_token_index):
            early += 1
    return {
        "level": level,
        "n_loops": n,
        "n_fired": fired,
        "n_before_onset": early,
        "median_lag_tokens": float(np.median(lags)) if lags else float("nan"),
        "note": "Descriptive: score >= 1 is four repeats (and 60 characters when character lengths are known). Not a false-alarm-budget threshold.",
    }


def _split_counts(labels: list[OnsetLabel]) -> dict:
    counts: dict[str, int] = {}
    for split in ("train", "dev", "test"):
        group = [lb for lb in labels if lb.split == split]
        counts[split] = len(group)
        counts[f"{split}_loops"] = sum(1 for lb in group if lb.is_loop and lb.onset_token_index is not None)
        counts[f"{split}_controls"] = sum(1 for lb in group if not lb.is_loop)
    return counts


def _ngram_for(trace: PreparedTrace) -> np.ndarray:
    char_lens = char_lens_for(trace.token_strings) if trace.aligned else None
    return online_ngram_repeat_scores(trace.token_ids, token_char_lens=char_lens)


def _score_map(traces: list[PreparedTrace]) -> dict[int, np.ndarray]:
    return {tr.prompt_id: _ngram_for(tr) for tr in traces}


def _cusum_map(ngram: dict[int, np.ndarray], labels: list[OnsetLabel]) -> tuple[dict[int, np.ndarray], dict]:
    train_ctrl = [
        ngram[lb.prompt_id]
        for lb in labels
        if lb.split == "train" and not lb.is_loop and lb.prompt_id in ngram
    ]
    ref = fit_cusum_in_control(train_ctrl)
    mapped = {
        pid: cusum_statistic(scores, mu=ref["mu"], allowance=ref["allowance"])
        for pid, scores in ngram.items()
    }
    return mapped, ref


def _surface_series(kind: str, examples, by_id, ngram, window: int) -> tuple[dict[int, np.ndarray] | None, str | None]:
    train = [ex for ex in examples if ex.split == "train"]
    if not train:
        return None, "no training windows"
    X, y, _pids, _horizons, _roles, _kept = _example_matrix(train, by_id, ngram, window)
    if y.size == 0 or np.unique(y).size < 2:
        return None, "training windows do not contain both classes"
    model, err = fit_window_model(X, y, kind)
    if model is None:
        return None, err
    out = {}
    for pid, trace in by_id.items():
        restart = restart_mask(trace.token_strings, len(trace.token_ids))
        out[pid] = surface_proba_series(model, trace.token_ids, ngram[pid], restart, window)
    return out, None


def _labelling_summary(labels: list[OnsetLabel], examples: list[HorizonExample], match_info: dict, n_input: int, skipped: list[dict]) -> dict:
    reasons = Counter(row["reason"] for row in skipped)
    by_h: dict[str, dict[str, int]] = {}
    for ex in examples:
        slot = by_h.setdefault(str(ex.horizon), {"pre_onset": 0, "control": 0})
        slot[ex.role] = slot.get(ex.role, 0) + 1
    lags = [
        int(lb.confirmation_token_index) - int(lb.onset_token_index)
        for lb in labels
        if lb.is_loop and lb.onset_token_index is not None and lb.confirmation_token_index is not None
    ]
    return {
        "n_input": n_input,
        "n_usable": len(labels),
        "n_skipped": len(skipped),
        "skip_reasons": dict(reasons),
        "n_loop": sum(1 for lb in labels if lb.is_loop),
        "n_loop_with_onset": sum(1 for lb in labels if lb.is_loop and lb.onset_token_index is not None),
        "n_control": sum(1 for lb in labels if not lb.is_loop),
        "n_disagreement_with_stored_flag": sum(1 for lb in labels if lb.disagrees_with_stored),
        "n_onset_index_mismatch": sum(1 for lb in labels if lb.onset_index_mismatch),
        "onset_sources": dict(Counter(lb.onset_source for lb in labels)),
        "ineligible_by_horizon": match_info.get("ineligible_by_horizon", {}),
        "unmatched_positives": match_info.get("unmatched_positives", 0),
        "examples_by_horizon": by_h,
        "antidoom_confirmation_lag_tokens": {
            "n": len(lags),
            "median": float(np.median(lags)) if lags else float("nan"),
            "note": "confirmation token minus onset token, on usable loops whose text aligns with token strings. Positive means the character rule fires after onset.",
        },
    }


def _data_status(labelling: dict, split_counts: dict) -> tuple[str, bool, bool]:
    """Return status, underpowered, metrics_computed."""
    n_loops = int(labelling["n_loop_with_onset"])
    n_ctrl = int(labelling["n_control"])
    test_loops = int(split_counts.get("test_loops", 0))
    test_ctrl = int(split_counts.get("test_controls", 0))
    if n_loops == 0 or n_ctrl == 0 or test_loops == 0 or test_ctrl == 0:
        return "insufficient", True, False
    underpowered = n_loops < UNDERPOWERED_LOOPS or test_loops < 8
    status = "underpowered" if underpowered else "ok"
    return status, underpowered, True


def analyze_traces(
    records: list[TraceRecord],
    *,
    tokenizer=None,
    horizons: tuple[int, ...] = HORIZONS,
    window_len: int = DEFAULT_WINDOW,
    budget: float = FAR_BUDGET,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Label traces and, when both classes reach the test split, score detectors."""
    prepared, skipped = prepare_traces(records, tokenizer=tokenizer)
    labels = [label_trace(tr) for tr in prepared]
    examples, match_info = build_horizon_examples(labels, horizons=horizons, window_len=window_len)
    labelling = _labelling_summary(labels, examples, match_info, len(records), skipped)
    split_counts = _split_counts(labels)
    status, underpowered, compute = _data_status(labelling, split_counts)
    notes = []
    if not compute and labelling["n_usable"] == 0:
        notes.append(
            "No forecasting metrics were computed. No usable trace was available: a usable trace needs the full generated text and the token ids from the baseline pass."
        )
    elif not compute:
        notes.append(
            "No forecasting metrics were computed. Usable traces do not put both a loop and a non-loop in the test split, so a reported AUROC or lead time would not be a test-set estimate."
        )
    elif underpowered:
        notes.append(
            f"Underpowered: {labelling['n_loop_with_onset']} usable loops with an onset (test loops {split_counts.get('test_loops', 0)}). "
            f"Point estimates are reported with prompt-clustered intervals; they are not a stable answer to the paper question until there are at least {UNDERPOWERED_LOOPS} onset-labelled loops."
        )
    result = {
        "question": QUESTION,
        "horizons": list(horizons),
        "window_len": window_len,
        "far_budget": budget,
        "min_repeats": MIN_REPEATS,
        "min_total_repeated_chars": MIN_TOTAL_REPEATED,
        "split": {
            "salt": SPLIT_SALT,
            "train_fraction": TRAIN_FRACTION,
            "dev_fraction": DEV_FRACTION,
            "test_fraction": round(1.0 - TRAIN_FRACTION - DEV_FRACTION, 2),
            "counts": split_counts,
        },
        "labelling": labelling,
        "data_status": status,
        "underpowered": underpowered,
        "metrics_computed": compute,
        "notes": notes,
        "detectors": None,
        "feature_sets": None,
    }
    if not compute:
        return result

    by_id = {tr.prompt_id: tr for tr in prepared}
    ngram = _score_map(prepared)
    char_applied = all(tr.aligned and tr.token_strings for tr in prepared)
    cusum, ref = _cusum_map(ngram, labels)
    surface_maps = {}
    surface_errors = {}
    for kind in ("logreg", "lightgbm"):
        mapped, err = _surface_series(kind, examples, by_id, ngram, window_len)
        surface_maps[kind] = mapped
        surface_errors[kind] = err

    def _pack(name, series, *, probabilities: bool, fixed_rule: bool = False):
        block = {
            "probabilities": probabilities,
            "per_horizon": _online_horizon_metrics(examples, series, probabilities=probabilities, budget=budget, n_boot=n_boot, seed=seed),
            "lead_time": _lead_block(labels, series, budget=budget, n_boot=n_boot, seed=seed),
        }
        if fixed_rule:
            test_labels = [lb for lb in labels if lb.split == "test"]
            block["fixed_rule_on_test"] = _fixed_rule_lag(test_labels, series, level=1.0)
            block["char_criterion_applied"] = char_applied
        return block

    detectors = {
        "ngram_repeat": _pack("ngram_repeat", ngram, probabilities=False, fixed_rule=True),
        "cusum_ngram": _pack("cusum_ngram", cusum, probabilities=False),
    }
    detectors["cusum_ngram"]["reference"] = ref
    detectors["cusum_ngram"]["reference_note"] = "mu and allowance were fit on training non-loop n-gram scores only. allowance = 0.5 * sigma."
    feature_sets = {}
    X, y, pids, horizons_arr, _roles, _kept = _example_matrix(examples, by_id, ngram, window_len)
    for kind in ("logreg", "lightgbm"):
        if surface_errors[kind]:
            detectors[f"surface_{kind}"] = {"status": "fit_failed", "reason": surface_errors[kind], "probabilities": True}
            feature_sets["surface_repetition_" + kind] = {"status": "fit_failed", "reason": surface_errors[kind]}
            continue
        detectors[f"surface_{kind}"] = _pack(f"surface_{kind}", surface_maps[kind], probabilities=True)
        feature_sets["surface_repetition_" + kind] = evaluate_feature_table(
            pids, y, X, kind=kind, horizons=horizons_arr, budget=budget, n_boot=n_boot, seed=seed,
        )
    # The CUSUM harness is the plug-in point: any per-token score can replace ngram.
    detectors["cusum_ngram"]["plugs_into"] = "cusum_statistic(per_token_scores, mu, allowance)"
    result["detectors"] = detectors
    result["feature_sets"] = feature_sets
    result["surface_features"] = list(SURFACE_FEATURES)
    return result


def _status_notes(load_info: dict, recorded: dict, sample: dict | None, analysis: dict) -> list[str]:
    notes = list(analysis.get("notes") or [])
    if load_info.get("n_trace_dirs", 0) == 0:
        notes.append(
            "The checkout has no per-trace generations. Baseline traces live under results/exp2_{slug}/{looping,nonlooping}/p*/ (meta.json + tokens.pt) and are gitignored. Onset token indices cannot be recomputed from the summary files."
        )
    if load_info.get("n_preview_rows", 0):
        notes.append(
            f"The generation log has {load_info['n_preview_rows']} preview rows (generated_preview is truncated to 500 characters). Those rows were not labelled."
        )
    legacy_ckpt = (recorded.get("baseline_checkpoint_legacy") or {})
    legacy_summary = (recorded.get("baseline_summary_legacy") or {})
    if legacy_ckpt.get("n_completed") is not None and legacy_summary.get("n_completed") is not None:
        if int(legacy_ckpt["n_completed"]) != int(legacy_summary["n_completed"]):
            notes.append(
                f"{legacy_ckpt['path']} lists {legacy_ckpt['n_completed']} completed ids, which does not match {legacy_summary['path']} ({legacy_summary['n_completed']} completed, {legacy_summary.get('n_loop')} loops). The summary counts were not recomputed."
            )
    exp2 = recorded.get("exp2_checkpoint") or {}
    if exp2.get("n_loop") is not None and load_info.get("n_trace_dirs", 0) == 0:
        notes.append(
            f"The Exp2 checkpoint records {exp2.get('n_completed')} completed hooked traces and {exp2.get('n_loop')} loops, but those trace directories are not in the checkout."
        )
    if sample and sample.get("recorded_baseline_loops_by_split"):
        by = sample["recorded_baseline_loops_by_split"]
        notes.append(
            "Hash-locked split of the recorded baseline looping prompt ids "
            f"(train {by.get('train', 0)}, dev {by.get('dev', 0)}, test {by.get('test', 0)}). "
            "A test fold that small cannot carry a four-horizon forecast even after the traces are present. "
            "These ids were not re-labelled; the onset detector did not see their text."
        )
    return notes


def summarize_results(
    results_dir,
    slug: str = "lfm2-2.6b",
    *,
    tokenizer=None,
    n_boot: int = 1000,
    seed: int = 0,
    window_len: int = DEFAULT_WINDOW,
    extra_dir: Path | None = None,
) -> dict:
    results_dir = Path(results_dir)
    traces, load_info = load_baseline_traces(results_dir, slug)
    recorded = load_recorded_counts(results_dir, slug)
    loop_ids = recorded_baseline_loop_ids(recorded)
    sample = load_prompt_sample_splits(results_dir / "prompt_sample_ids.json", loop_ids)
    analysis = analyze_traces(traces, tokenizer=tokenizer, n_boot=n_boot, seed=seed, window_len=window_len)
    extra = _extra_feature_sets(extra_dir or (results_dir / "forecast" / "feature_tables"), n_boot=n_boot, seed=seed)
    notes = _status_notes(load_info, recorded, sample, analysis)
    summary = {
        "question": QUESTION,
        "model_slug": slug,
        "load": load_info,
        "recorded": recorded,
        "prompt_sample": sample,
        **{k: v for k, v in analysis.items() if k not in {"question", "notes"}},
        "extra_feature_sets": extra,
        "notes": notes,
    }
    if extra:
        summary["feature_sets"] = {**(summary.get("feature_sets") or {}), **{row["name"]: row["metrics"] for row in extra}}
    return to_jsonable(summary)


def _extra_feature_sets(directory: Path, *, n_boot: int, seed: int) -> list[dict]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("*.npz")):
        table = load_feature_table(path)
        metrics = {
            kind: evaluate_feature_table(
                table["prompt_ids"],
                table["y"],
                table["X"],
                kind=kind,
                horizons=table["horizon"],
                n_boot=n_boot,
                seed=seed,
            )
            for kind in ("logreg", "lightgbm")
        }
        rows.append({
            "name": table["name"],
            "path": table["path"],
            "n_rows": int(table["y"].shape[0]),
            "n_features": int(table["X"].shape[1]),
            "feature_names": table["feature_names"],
            "split_recomputed_from_prompt_id": True,
            "metrics": metrics,
        })
    return rows


def write_summary(path, summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, dict) and "point" in value:
        return f"{_fmt(value.get('point'), digits)} [{_fmt(value.get('lo'), digits)}, {_fmt(value.get('hi'), digits)}]"
    return str(value)


def _horizon_line(block: dict | None) -> str:
    if not block or block.get("status") != "ok":
        reason = (block or {}).get("status", "missing")
        return f"n/a ({reason})"
    test = block.get("test") or {}
    at = block.get("at_far_budget") or {}
    return (
        f"AUROC {_fmt(test.get('auroc'))} ({_fmt(test.get('auroc_ci'))}), "
        f"AUPRC {_fmt(test.get('auprc'))}, "
        f"ECE {_fmt(test.get('ece'))}, "
        f"recall@5% FAR {_fmt(at.get('recall'))} (n_pos {at.get('n_pos')}, n_neg {at.get('n_neg')})"
    )


def render_report(summary: dict) -> str:
    """Short markdown report. Numbers come only from ``summary``."""
    slug = summary.get("model_slug", "")
    status = summary.get("data_status", "unknown")
    lines = [
        f"# Pre-onset doom-loop forecast — {slug}",
        "",
        f"Question: {summary.get('question')}",
        "",
        f"Data status: **{status}**.",
        "",
    ]
    if not summary.get("metrics_computed"):
        lines.append("No forecasting metrics were computed.")
        lines.append("")
    for note in summary.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    recorded = summary.get("recorded") or {}
    lines.append("## Recorded counts (not re-estimated)")
    lines.append("")
    if not recorded:
        lines.append("No baseline summary or checkpoint file was found.")
    else:
        lines.append("| File | Completed | Loops |")
        lines.append("| --- | --- | --- |")
        for key, block in recorded.items():
            lines.append(f"| `{block.get('path', key)}` | {block.get('n_completed')} | {block.get('n_loop')} |")
    sample = summary.get("prompt_sample") or {}
    if sample:
        by = sample.get("by_split") or {}
        loops = sample.get("recorded_baseline_loops_by_split") or {}
        lines.append("")
        lines.append(
            f"Prompt sample: {sample.get('n_prompts')} ids, seed {sample.get('seed')}. "
            f"Hash-locked split (`{SPLIT_SALT}`, 60/20/20): "
            f"train {by.get('train')}, dev {by.get('dev')}, test {by.get('test')}."
        )
        lines.append(
            "Recorded baseline looping ids in that split: "
            f"train {loops.get('train')}, dev {loops.get('dev')}, test {loops.get('test')}."
        )
    labelling = summary.get("labelling") or {}
    lines.extend([
        "",
        "## Labelling",
        "",
        f"Usable traces: {labelling.get('n_usable', 0)} of {labelling.get('n_input', 0)} loaded. "
        f"Loops with an onset token: {labelling.get('n_loop_with_onset', 0)}. "
        f"Non-loops: {labelling.get('n_control', 0)}.",
        f"Window length: {summary.get('window_len')} tokens. Horizons: {summary.get('horizons')}.",
        f"Character rule: at least {summary.get('min_repeats', MIN_REPEATS)} repeats and at least {summary.get('min_total_repeated_chars', MIN_TOTAL_REPEATED)} characters.",
        "",
    ])
    if summary.get("metrics_computed") and summary.get("detectors"):
        lines.extend(["## Test-set detectors", ""])
        lines.append("Horizon metrics use the score at `onset - H`. Lead time is `onset - alarm`, positive when the alarm is strictly before onset. The 5% false-alarm threshold is fit on dev non-loops and frozen before the test loops are scored.")
        lines.append("")
        for name, block in summary["detectors"].items():
            lines.append(f"### {name}")
            lines.append("")
            if block.get("status") == "fit_failed":
                lines.append(f"Fit failed: {block.get('reason')}")
                lines.append("")
                continue
            for horizon, hblock in (block.get("per_horizon") or {}).items():
                lines.append(f"- H={horizon}: {_horizon_line(hblock)}")
            lead = block.get("lead_time") or {}
            if lead.get("status") == "ok":
                lines.append(
                    f"- Lead time at 5% FAR: early rate {_fmt(lead.get('early_rate'))} "
                    f"({_fmt(lead.get('early_rate_ci'))}), "
                    f"median early lead {_fmt(lead.get('median_lead_tokens_early'))} tokens "
                    f"({_fmt(lead.get('median_lead_tokens_early_ci'))}), "
                    f"median signed lead {_fmt(lead.get('median_signed_lead_tokens'))}, "
                    f"dev FAR {_fmt(lead.get('calibration_false_alarm_rate'))}, "
                    f"test FAR {_fmt(lead.get('test_false_alarm_rate'))}."
                )
            else:
                lines.append(f"- Lead time: n/a ({lead.get('status', 'missing')}).")
            fixed = block.get("fixed_rule_on_test")
            if fixed:
                lines.append(
                    f"- Fixed n-gram rule (score reaches 1): fired on {fixed.get('n_fired')} / {fixed.get('n_loops')} test loops, "
                    f"before onset on {fixed.get('n_before_onset')}, median lag {_fmt(fixed.get('median_lag_tokens'))} tokens."
                )
            lines.append("")
        if summary.get("feature_sets"):
            lines.append("Feature-table models (logistic regression and LightGBM) are the `surface_*` rows above, plus any `.npz` tables found under `results/forecast/feature_tables/`.")
            lines.append("")
    else:
        lines.extend([
            "## Detectors",
            "",
            "The n-gram rule, the surface logistic / LightGBM models, and the CUSUM harness were not scored. There is no test-set estimate to report.",
            "",
        ])
    lines.extend([
        "## Running this on Qwen traces",
        "",
        "Leave the generation scripts alone. After `scripts/02_baseline_pass.py` has written `results/exp2_qwen3.5-4b/{looping,nonlooping}/p*/meta.json` and `tokens.pt`, run:",
        "",
        "```",
        "python scripts/07_forecast_report.py --slug qwen3.5-4b",
        "```",
        "",
        "Hidden-state or J-lens features, once a hooked run writes them, go in `results/forecast/feature_tables/*.npz` with arrays `prompt_ids`, `y`, `X`, `feature_names`, and optional `horizon`. The split is recomputed from the prompt id (`jspace-forecast-v1`, 60/20/20). CUSUM accepts any per-token score through `cusum_statistic`.",
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"
