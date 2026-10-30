"""Window features and the logistic / LightGBM interface.

The surface set is computed from token ids alone, so it runs on a baseline
trace before any hooked J-lens pass. Hidden-state and J-lens features use the
same evaluator: a ``.npz`` table of rows already aligned to prompt ids. The
hash-locked split is recomputed from those ids; a split column in the file is
not read.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jspace.detection import RESTART_WORDS, decode_token
from jspace.forecasting.metrics import (
    FAR_BUDGET,
    assign_split,
    classification_bundle,
    clustered_bootstrap_ci,
    rates_at_threshold,
    safe_auprc,
    safe_auroc,
    threshold_for_far,
)

SURFACE_FEATURES = (
    "unique_ratio",
    "top1_share",
    "unigram_entropy_norm",
    "adjacent_repeat_rate",
    "max_identical_run",
    "prefix_ngram_score",
    "period2_hit_rate",
    "restart_rate",
)


def char_lens_for(token_strings: list[str] | None) -> list[int] | None:
    if not token_strings:
        return None
    return [len(decode_token(s)) for s in token_strings]


def restart_mask(token_strings: list[str] | None, n: int) -> np.ndarray | None:
    if token_strings is None:
        return None
    mask = np.zeros(n, dtype=np.float64)
    for i, raw in enumerate(token_strings[:n]):
        dec = decode_token(raw).strip().lower()
        if dec in RESTART_WORDS:
            mask[i] = 1.0
    return mask


def feature_matrix(
    token_ids,
    decision_indices,
    *,
    window: int,
    ngram: np.ndarray,
    restart: np.ndarray | None = None,
) -> np.ndarray:
    """One row per decision index. Row ``t`` uses only ``token_ids[: t + 1]``.

    ``ngram[t]`` must already be the causal prefix score at ``t``. Statistics
    other than that score are restricted to the ``window`` tokens ending at
    ``t`` inclusive.
    """
    ids = np.asarray(token_ids, dtype=np.int64)
    ngram = np.asarray(ngram, dtype=float)
    indices = [int(t) for t in decision_indices]
    X = np.full((len(indices), len(SURFACE_FEATURES)), np.nan, dtype=np.float64)
    log_w = math.log(window) if window > 1 else 1.0
    for row, t in enumerate(indices):
        start = t - window + 1
        if start < 0 or t < 0 or t >= ids.size or t >= ngram.size:
            continue
        w = ids[start : t + 1]
        _uniq, counts = np.unique(w, return_counts=True)
        X[row, 0] = _uniq.size / float(window)
        X[row, 1] = float(counts.max()) / float(window)
        p = counts.astype(np.float64) / float(window)
        X[row, 2] = float(-np.sum(p * np.log(p))) / log_w
        if window > 1:
            X[row, 3] = float(np.mean(w[1:] == w[:-1]))
        best = run = 1
        prev = int(w[0])
        for tok in w[1:].tolist():
            tok = int(tok)
            if tok == prev:
                run += 1
                if run > best:
                    best = run
            else:
                run = 1
                prev = tok
        X[row, 4] = best / float(window)
        X[row, 5] = float(ngram[t])
        if window > 2:
            X[row, 6] = float(np.mean(w[2:] == w[:-2]))
        else:
            X[row, 6] = 0.0
        if restart is not None:
            X[row, 7] = float(np.mean(restart[start : t + 1]))
        else:
            X[row, 7] = 0.0
    return X


def fit_window_model(X, y, kind: str):
    """Fit logistic regression or LightGBM. Returns ``(model, error)``."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y, dtype=int).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        return None, "feature matrix shape does not match labels"
    if y.size < 4 or np.unique(y).size < 2:
        return None, "need both classes and at least 4 training rows"
    if kind == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=0,
        )
    elif kind == "lightgbm":
        n_pos = max(int(np.sum(y == 1)), 1)
        n_neg = max(int(np.sum(y == 0)), 1)
        clf = LGBMClassifier(
            n_estimators=40,
            num_leaves=7,
            min_child_samples=5,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=0,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_row_wise=True,
            scale_pos_weight=n_neg / n_pos,
        )
    else:
        return None, f"unknown model kind: {kind}"
    model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    try:
        model.fit(X, y)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return model, None


def positive_proba(model, X) -> np.ndarray:
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(X) == 0:
        return np.zeros(0, dtype=float)
    proba = model.predict_proba(X)
    classes = list(model.named_steps["clf"].classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def surface_proba_series(model, token_ids, ngram, restart, window: int) -> np.ndarray:
    """Causal probability at every token. Positions before ``window`` stay NaN."""
    n = len(token_ids)
    out = np.full(n, np.nan, dtype=np.float64)
    if model is None or n < window:
        return out
    idxs = np.arange(window - 1, n)
    X = feature_matrix(token_ids, idxs, window=window, ngram=ngram, restart=restart)
    good = np.isfinite(X).all(axis=1)
    if not np.any(good):
        return out
    out[idxs[good]] = positive_proba(model, X[good])
    return out


def _eval_split(X, y, prompt_ids, train_m, dev_m, test_m, *, kind, budget, n_boot, seed, tag):
    n_train = int(np.sum(train_m))
    if n_train == 0 or np.unique(y[train_m]).size < 2:
        return {
            "status": "too_small",
            "reason": f"{tag} train split does not contain both classes",
            "n_train": n_train,
            "n_dev": int(np.sum(dev_m)),
            "n_test": int(np.sum(test_m)),
        }
    model, err = fit_window_model(X[train_m], y[train_m], kind)
    out = {
        "status": "ok",
        "n_train": n_train,
        "n_dev": int(np.sum(dev_m)),
        "n_test": int(np.sum(test_m)),
        "model": model,
    }
    if model is None:
        out["status"] = "fit_failed"
        out["reason"] = err
        out["model"] = None
        return out
    if int(np.sum(test_m)) == 0 or np.unique(y[test_m]).size < 2:
        out["status"] = "test_too_small"
        out["reason"] = f"{tag} test split does not contain both classes"
        out["test"] = None
        return out
    test_p = positive_proba(model, X[test_m])
    bundle = classification_bundle(y[test_m], test_p, probabilities=True)
    test_y = y[test_m]
    test_prompts = prompt_ids[test_m]

    def _auroc(idx):
        return safe_auroc(test_y[idx], test_p[idx])

    def _auprc(idx):
        return safe_auprc(test_y[idx], test_p[idx])

    bundle["auroc_ci"] = clustered_bootstrap_ci(test_prompts, _auroc, n_boot=n_boot, seed=seed)
    bundle["auprc_ci"] = clustered_bootstrap_ci(test_prompts, _auprc, n_boot=n_boot, seed=seed + 1)
    dev_ctrl = dev_m & (y == 0)
    if int(np.sum(dev_ctrl)) == 0:
        cal_m = train_m & (y == 0)
        source = "train_fallback"
    else:
        cal_m = dev_ctrl
        source = "dev"
    cal_p = positive_proba(model, X[cal_m]) if int(np.sum(cal_m)) else np.zeros(0)
    thr = threshold_for_far(cal_p, budget) if cal_p.size else float("inf")
    out["test"] = bundle
    out["at_far_budget"] = rates_at_threshold(y[test_m], test_p, thr)
    out["threshold_source"] = source
    out["threshold"] = out["at_far_budget"]["threshold"]
    if cal_p.size:
        out["calibration_false_alarm_rate"] = rates_at_threshold(y[cal_m], cal_p, thr)["false_alarm_rate"]
    else:
        out["calibration_false_alarm_rate"] = float("nan")
    return out


def evaluate_feature_table(
    prompt_ids,
    y,
    X,
    *,
    kind: str,
    horizons=None,
    budget: float = FAR_BUDGET,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Fit on the hash-locked train split and score the test split.

    ``horizons``, when given, also produces one model per horizon. The pooled
    model trains on every horizon row. Splits are recomputed from ``prompt_ids``.
    """
    prompt_ids = np.asarray(prompt_ids)
    y = np.asarray(y, dtype=int).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.shape[0] or prompt_ids.shape[0] != y.shape[0]:
        raise ValueError("prompt_ids, y, and X must have the same number of rows")
    splits = np.array([assign_split(int(p)) for p in prompt_ids])
    result = {
        "kind": kind,
        "probabilities": True,
        "n_rows": int(y.size),
        "n_features": int(X.shape[1]),
        "split_counts": {name: int(np.sum(splits == name)) for name in ("train", "dev", "test")},
    }
    result["pooled"] = _strip_model(
        _eval_split(
            X, y, prompt_ids, splits == "train", splits == "dev", splits == "test",
            kind=kind, budget=budget, n_boot=n_boot, seed=seed, tag="pooled",
        )
    )
    if horizons is not None:
        horizons_arr = np.asarray(horizons)
        per = {}
        for horizon in sorted({int(h) for h in horizons_arr.tolist()}):
            mask = horizons_arr == horizon
            per[str(horizon)] = _strip_model(
                _eval_split(
                    X, y, prompt_ids,
                    mask & (splits == "train"),
                    mask & (splits == "dev"),
                    mask & (splits == "test"),
                    kind=kind, budget=budget, n_boot=n_boot, seed=seed + horizon, tag=f"H={horizon}",
                )
            )
        result["per_horizon"] = per
    return result


def _strip_model(block: dict) -> dict:
    """Drop the fitted estimator so results stay JSON-serializable."""
    return {k: v for k, v in block.items() if k != "model"}


def save_feature_table(path, *, name: str, prompt_ids, y, X, feature_names, horizon=None) -> None:
    """Write a feature table for a later hooked run.

    Required arrays: ``prompt_ids``, ``y`` (0/1), ``X`` shaped ``(n, d)``,
    ``feature_names`` length ``d``. Optional ``horizon`` (int per row). Do not
    store a split: it is recomputed from the prompt id.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=np.float64)
    names = np.asarray(list(feature_names))
    if names.shape[0] != X.shape[1]:
        raise ValueError("feature_names length must match X.shape[1]")
    payload = {
        "name": np.asarray(name),
        "prompt_ids": np.asarray(prompt_ids, dtype=np.int64),
        "y": np.asarray(y, dtype=np.int64),
        "X": X,
        "feature_names": names,
    }
    if horizon is not None:
        payload["horizon"] = np.asarray(horizon, dtype=np.int64)
    np.savez_compressed(path, **payload)


def load_feature_table(path) -> dict:
    with np.load(path, allow_pickle=False) as blob:
        table = {
            "name": str(blob["name"]),
            "prompt_ids": np.asarray(blob["prompt_ids"], dtype=np.int64),
            "y": np.asarray(blob["y"], dtype=np.int64),
            "X": np.asarray(blob["X"], dtype=np.float64),
            "feature_names": [str(x) for x in blob["feature_names"].tolist()],
            "horizon": np.asarray(blob["horizon"], dtype=np.int64) if "horizon" in blob.files else None,
            "path": str(path),
        }
    if table["X"].ndim != 2 or table["X"].shape[0] != table["y"].shape[0]:
        raise ValueError(f"bad feature table shape in {path}")
    if len(table["feature_names"]) != table["X"].shape[1]:
        raise ValueError(f"feature name count does not match X in {path}")
    return table
