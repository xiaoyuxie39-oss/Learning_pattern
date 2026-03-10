#!/usr/bin/env python3
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from . import step01_features as stage3_features

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:  # pragma: no cover - fallback handled at runtime
    ExplainableBoostingClassifier = None  # type: ignore[assignment]

TOP_KS = [10, 20, 30, 50]
LINEAR_WARNING_MODELS = {"logistic_l2", "elasticnet"}
SPURIOUS_MATMUL_WARNING_FRAGMENTS = (
    "divide by zero encountered in matmul",
    "overflow encountered in matmul",
    "invalid value encountered in matmul",
)
WARNING_CACHE: dict[str, set[str]] = defaultdict(set)


def append_warning(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_key = str(path.resolve())
    if message in WARNING_CACHE[cache_key]:
        return
    WARNING_CACHE[cache_key].add(message)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def finite_matrix_or_raise(name: str, values: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values after sanitization")
    return out


def should_ignore_warning(model_name: str, category_name: str, message: str) -> bool:
    base_model = model_name.split(".", 1)[0]
    if base_model not in LINEAR_WARNING_MODELS:
        return False
    if category_name != "RuntimeWarning":
        return False
    return any(fragment in message for fragment in SPURIOUS_MATMUL_WARNING_FRAGMENTS)


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return np.nan
    topk = min(k, len(y_true))
    idx = np.argsort(-score)[:topk]
    return float(np.mean(y_true[idx]))


def recall_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    n_pos = int(np.sum(y_true == 1))
    if len(y_true) == 0 or n_pos == 0:
        return np.nan
    topk = min(k, len(y_true))
    idx = np.argsort(-score)[:topk]
    return float(np.sum(y_true[idx]) / n_pos)


def enrichment_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    base_rate = float(np.mean(y_true)) if len(y_true) else np.nan
    if not np.isfinite(base_rate) or base_rate <= 0:
        return np.nan
    p_at_k = precision_at_k(y_true, score, k)
    return float(p_at_k / base_rate)


def auc_proxy(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(y_true) == 0:
        return np.nan
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy()
    sum_pos = float(ranks[y_true == 1].sum())
    return float((sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def make_estimator(
    model_kind: str,
    random_seed: int,
    *,
    branch_name: str = "mainline",
    pair_limit: int = 4,
    triple_limit: int = 0,
) -> Any:
    if model_kind == "logistic_l2":
        return LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=0.25,
            max_iter=2000,
            random_state=random_seed,
        )
    if model_kind == "elasticnet":
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=3000,
            tol=1e-4,
            learning_rate="optimal",
            random_state=random_seed,
        )
    if model_kind == "gbdt":
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            min_samples_leaf=10,
            random_state=random_seed,
        )
    if model_kind == "ebm":
        if ExplainableBoostingClassifier is None:
            raise RuntimeError("interpret_not_available")
        interaction_budget = 0 if branch_name == "mainline" else min(pair_limit + (triple_limit if branch_name == "mainline_plus_gated_3way" else 0), 8)
        return ExplainableBoostingClassifier(
            interactions=interaction_budget,
            max_bins=64,
            max_rounds=300,
            learning_rate=0.03,
            outer_bags=4,
            inner_bags=0,
            n_jobs=1,
            random_state=random_seed,
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -35.0, 35.0)))
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def fit_with_warning_capture(
    model_name: str,
    estimator_factory: Callable[[int, str], Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_seed: int,
    warning_log: Path,
    branch_name: str,
) -> tuple[np.ndarray, Any, list[str]]:
    captured: list[str] = []
    estimator = estimator_factory(random_seed, branch_name)
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        estimator.fit(X_train, y_train)
        score = predict_scores(estimator, X_test)
    score = finite_matrix_or_raise(f"{model_name}.score", np.asarray(score, dtype=float))
    for item in warning_list:
        if should_ignore_warning(model_name, item.category.__name__, str(item.message)):
            continue
        message = f"{model_name}: {item.category.__name__}: {item.message}"
        captured.append(message)
        append_warning(warning_log, message)
    return np.asarray(score, dtype=float), estimator, captured


def evaluate_model_branch(
    model_spec: Any,
    branch_data: Any,
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    *,
    random_seed: int,
    warning_log: Path,
    pair_limit: int = 4,
    triple_limit: int = 0,
    top_ks: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any, pd.DataFrame]:
    if branch_data.skipped_by_config:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, pd.DataFrame()
    top_k_values = list(top_ks or TOP_KS)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    audit_trace_rows: list[dict[str, Any]] = []
    estimator_factory = lambda seed, branch_name: make_estimator(
        model_spec.kind,
        seed,
        branch_name=branch_name,
        pair_limit=pair_limit,
        triple_limit=triple_limit,
    )
    row_ids = df["id"].astype(str) if "id" in df.columns else pd.Series([str(i) for i in range(len(df))])
    for sp in splits:
        split_id = int(sp["split_id"])
        train_idx = sp["train_idx"]
        test_idx = sp["test_idx"]
        X_train, X_test = stage3_features.stabilize_design_matrices(branch_data.X.iloc[train_idx], branch_data.X.iloc[test_idx])
        y_train = y[train_idx].astype(int)
        y_test = y[test_idx].astype(int)
        scores, _, _ = fit_with_warning_capture(
            model_spec.name,
            estimator_factory,
            X_train,
            y_train,
            X_test,
            random_seed + split_id,
            warning_log,
            branch_data.branch_name,
        )
        for k in top_k_values:
            p_val = precision_at_k(y_test, scores, k)
            r_val = recall_at_k(y_test, scores, k)
            e_val = enrichment_at_k(y_test, scores, k)
            metrics_rows.append({"split_id": split_id, "metric": "P", "k": k, "value": p_val})
            metrics_rows.append({"split_id": split_id, "metric": "Recall", "k": k, "value": r_val})
            metrics_rows.append({"split_id": split_id, "metric": "Enrichment", "k": k, "value": e_val})
        auc_val = auc_proxy(y_test, scores)
        metrics_rows.append({"split_id": split_id, "metric": "AUC_proxy", "k": 0, "value": auc_val})
        for local_idx, global_idx in enumerate(test_idx):
            prediction_rows.append(
                {
                    "model": model_spec.name,
                    "branch": branch_data.branch_name,
                    "split_id": split_id,
                    "row_idx": int(global_idx),
                    "row_id": row_ids.iloc[global_idx],
                    "company": df.iloc[global_idx]["company"],
                    "coverage_tier": df.iloc[global_idx]["coverage_tier"],
                    "y_true": int(y_test[local_idx]),
                    "score": float(scores[local_idx]),
                }
            )
        audit_trace_rows.append(
            {
                "model": model_spec.name,
                "branch": branch_data.branch_name,
                "split_id": split_id,
                "metric": "AUC_proxy",
                "value": auc_val,
            }
        )

    metric_df = pd.DataFrame(metrics_rows)
    pred_df = pd.DataFrame(prediction_rows)
    fold_metrics = metric_df.copy()
    fold_metrics.insert(0, "branch", branch_data.branch_name)
    fold_metrics.insert(0, "model", model_spec.name)
    audit_trace = pd.DataFrame(audit_trace_rows)

    full_model = make_estimator(
        model_spec.kind,
        random_seed,
        branch_name=branch_data.branch_name,
        pair_limit=pair_limit,
        triple_limit=triple_limit,
    )
    X_full_frame, full_scaling_stats = stage3_features.stabilize_feature_frame(branch_data.X)
    X_full = finite_matrix_or_raise("X_full", X_full_frame.to_numpy(dtype=float))
    X_full = np.clip(X_full, -10.0, 10.0)
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        full_model.fit(X_full, y.astype(int))
    for item in warning_list:
        if should_ignore_warning(model_spec.name, item.category.__name__, str(item.message)):
            continue
        append_warning(warning_log, f"{model_spec.name}: {item.category.__name__}: {item.message}")

    return metric_df, pred_df, fold_metrics, audit_trace, full_model, full_scaling_stats
