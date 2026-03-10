#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import step01_features as stage3_features
from . import step03_models as stage3_models

DEFAULT_INPUT_SHIFT_FEATURES = ("power_mw", "rack_kw_typical", "pue")
DEFAULT_MATCH_STRATA = ("power_mw_bin", "rack_kw_typical_bin", "pue_bin")


def _safe_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    return values


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    a = np.sort(a)
    b = np.sort(b)
    xs = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, xs, side="right") / max(1, len(a))
    cdf_b = np.searchsorted(b, xs, side="right") / max(1, len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _wasserstein_approx(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    qs = np.linspace(0.0, 1.0, 51)
    aq = np.quantile(a, qs)
    bq = np.quantile(b, qs)
    return float(np.mean(np.abs(aq - bq)))


def _mmd_rbf(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    x = a.reshape(-1, 1)
    y = b.reshape(-1, 1)
    z = np.concatenate([x, y], axis=0)
    diffs = z[:, None, :] - z[None, :, :]
    sqdist = np.sum(diffs * diffs, axis=2)
    tri = sqdist[np.triu_indices_from(sqdist, k=1)]
    sigma2 = float(np.median(tri)) if len(tri) else 1.0
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = 1.0
    gamma = 1.0 / (2.0 * sigma2)

    kxx = np.exp(-gamma * ((x - x.T) ** 2))
    kyy = np.exp(-gamma * ((y - y.T) ** 2))
    kxy = np.exp(-gamma * ((x - y.T) ** 2))

    m = len(x)
    n = len(y)
    xx = (np.sum(kxx) - np.trace(kxx)) / max(1, (m * (m - 1)))
    yy = (np.sum(kyy) - np.trace(kyy)) / max(1, (n * (n - 1)))
    xy = np.sum(kxy) / max(1, (m * n))
    return float(xx + yy - 2.0 * xy)


def _directional_sign_rate(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return np.nan
    pos = float(np.mean(arr > 0))
    neg = float(np.mean(arr < 0))
    return max(pos, neg)


def negative_control_audit(
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
) -> tuple[bool, str, pd.DataFrame]:
    if branch_data.skipped_by_config:
        return True, "skipped_by_config", pd.DataFrame()
    estimator_factory = lambda seed, branch_name: stage3_models.make_estimator(
        model_spec.kind,
        seed,
        branch_name=branch_name,
        pair_limit=pair_limit,
        triple_limit=triple_limit,
    )
    rng = np.random.default_rng(random_seed + 1000)
    rows: list[dict[str, Any]] = []
    deltas: list[float] = []
    for sp in splits[: min(10, len(splits))]:
        split_id = int(sp["split_id"])
        train_idx = sp["train_idx"]
        test_idx = sp["test_idx"]
        y_train = y[train_idx].astype(int).copy()
        rng.shuffle(y_train)
        X_train, X_test = stage3_features.stabilize_design_matrices(branch_data.X.iloc[train_idx], branch_data.X.iloc[test_idx])
        y_test = y[test_idx].astype(int)
        scores, _, _ = stage3_models.fit_with_warning_capture(
            f"{model_spec.name}.negative_control",
            estimator_factory,
            X_train,
            y_train,
            X_test,
            random_seed + 9000 + split_id,
            warning_log,
            branch_data.branch_name,
        )
        p20 = stage3_models.precision_at_k(y_test, scores, 20)
        prevalence = float(np.mean(y_test)) if len(y_test) else np.nan
        delta = float(p20 - prevalence) if np.isfinite(p20) and np.isfinite(prevalence) else np.nan
        if np.isfinite(delta):
            deltas.append(delta)
        rows.append({"split_id": split_id, "metric": "mean_p20_minus_prevalence", "value": delta})
    trace = pd.DataFrame(rows)
    if not deltas:
        return True, "no_valid_negative_control_values", trace
    mean_delta = float(np.mean(deltas))
    fail = mean_delta > 0.05
    return (not fail), f"mean_p20_minus_prevalence={mean_delta:.4f}", trace


def _build_input_shift_rows(split_id: int, grp: pd.DataFrame, input_features: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    c2 = grp[grp["coverage_tier"] == "C2"]
    c3 = grp[grp["coverage_tier"] == "C3"]
    for feature in input_features:
        if feature not in grp.columns:
            continue
        c2_values = _safe_numeric(c2[feature])
        c3_values = _safe_numeric(c3[feature])
        rows.append(
            {
                "split_id": int(split_id),
                "feature_name": feature,
                "c2_n": int(len(c2_values)),
                "c3_n": int(len(c3_values)),
                "ks_stat": _ks_statistic(c2_values, c3_values),
                "wasserstein": _wasserstein_approx(c2_values, c3_values),
                "mmd_rbf": _mmd_rbf(c2_values, c3_values),
            }
        )
    return rows


def _build_score_shift_row(split_id: int, grp: pd.DataFrame, top_k: int) -> dict[str, Any]:
    c2 = grp[grp["coverage_tier"] == "C2"]
    c3 = grp[grp["coverage_tier"] == "C3"]
    c2_scores = _safe_numeric(c2["score"])
    c3_scores = _safe_numeric(c3["score"])
    top_n = min(int(top_k), len(grp))
    top = grp.nlargest(top_n, "score") if top_n > 0 else pd.DataFrame(columns=grp.columns)
    c2_share = float(np.mean(top["coverage_tier"] == "C2")) if top_n > 0 else np.nan
    c3_share = float(np.mean(top["coverage_tier"] == "C3")) if top_n > 0 else np.nan
    return {
        "split_id": int(split_id),
        "c2_n": int(len(c2_scores)),
        "c3_n": int(len(c3_scores)),
        "score_ks_stat": _ks_statistic(c2_scores, c3_scores),
        "score_wasserstein": _wasserstein_approx(c2_scores, c3_scores),
        "score_mmd_rbf": _mmd_rbf(c2_scores, c3_scores),
        "topk": int(top_n),
        "topk_c2_share": c2_share,
        "topk_c3_share": c3_share,
        "topk_c3_minus_c2": float(c3_share - c2_share) if np.isfinite(c2_share) and np.isfinite(c3_share) else np.nan,
    }


def _match_control_row(
    split_id: int,
    grp: pd.DataFrame,
    *,
    top_k: int,
    match_strata: Sequence[str],
    raw_p20_diff: float,
    raw_enrichment_diff: float,
    raw_diff_eps: float = 0.02,
) -> dict[str, Any]:
    out = {
        "split_id": int(split_id),
        "matched_valid": False,
        "match_strata": "|".join(match_strata),
        "matched_c2_n": 0,
        "matched_c3_n": 0,
        "raw_p20_c3_minus_c2": raw_p20_diff,
        "matched_p20_c3_minus_c2": np.nan,
        "raw_enrichment20_c3_minus_c2": raw_enrichment_diff,
        "matched_enrichment20_c3_minus_c2": np.nan,
        "matched_reduction_abs": np.nan,
        "matched_reduction_rate": np.nan,
        "matched_diff_reduction_abs": np.nan,
        "matched_diff_reduction_rate": np.nan,
    }
    for col in match_strata:
        if col not in grp.columns:
            return out
    c2 = grp[grp["coverage_tier"] == "C2"].copy()
    c3 = grp[grp["coverage_tier"] == "C3"].copy()
    if len(c2) < 10 or len(c3) < 10:
        return out

    for col in match_strata:
        c2[col] = c2[col].fillna("__MISSING__").astype(str)
        c3[col] = c3[col].fillna("__MISSING__").astype(str)
    c2_groups = c2.groupby(list(match_strata), dropna=False)
    c3_groups = c3.groupby(list(match_strata), dropna=False)
    c3_map = {key: idxs.to_numpy(dtype=int) for key, idxs in c3_groups.groups.items()}
    rng = np.random.default_rng(int(split_id) + 2026)
    matched_c2_idx: list[int] = []
    matched_c3_idx: list[int] = []
    for key, idxs_c2 in c2_groups.groups.items():
        idxs_c3 = c3_map.get(key)
        if idxs_c3 is None:
            continue
        idxs_c2_arr = np.asarray(idxs_c2, dtype=int)
        idxs_c3_arr = np.asarray(idxs_c3, dtype=int)
        n = int(min(len(idxs_c2_arr), len(idxs_c3_arr)))
        if n <= 0:
            continue
        pick_c2 = rng.choice(idxs_c2_arr, size=n, replace=False) if len(idxs_c2_arr) > n else idxs_c2_arr
        pick_c3 = rng.choice(idxs_c3_arr, size=n, replace=False) if len(idxs_c3_arr) > n else idxs_c3_arr
        matched_c2_idx.extend(pick_c2.tolist())
        matched_c3_idx.extend(pick_c3.tolist())
    if len(matched_c2_idx) < 5 or len(matched_c3_idx) < 5:
        return out

    c2_m = grp.loc[matched_c2_idx]
    c3_m = grp.loc[matched_c3_idx]
    k = int(min(top_k, len(c2_m), len(c3_m)))
    if k <= 0:
        return out
    p20_c2 = stage3_models.precision_at_k(c2_m["y_true"].to_numpy(dtype=int), c2_m["score"].to_numpy(dtype=float), k)
    p20_c3 = stage3_models.precision_at_k(c3_m["y_true"].to_numpy(dtype=int), c3_m["score"].to_numpy(dtype=float), k)
    e20_c2 = stage3_models.enrichment_at_k(c2_m["y_true"].to_numpy(dtype=int), c2_m["score"].to_numpy(dtype=float), k)
    e20_c3 = stage3_models.enrichment_at_k(c3_m["y_true"].to_numpy(dtype=int), c3_m["score"].to_numpy(dtype=float), k)
    matched_p20_diff = float(p20_c3 - p20_c2) if np.isfinite(p20_c2) and np.isfinite(p20_c3) else np.nan
    matched_e20_diff = float(e20_c3 - e20_c2) if np.isfinite(e20_c2) and np.isfinite(e20_c3) else np.nan
    reduction_abs = (
        float(abs(raw_p20_diff) - abs(matched_p20_diff))
        if np.isfinite(raw_p20_diff) and np.isfinite(matched_p20_diff)
        else np.nan
    )
    reduction_rate = np.nan
    if np.isfinite(reduction_abs) and np.isfinite(raw_p20_diff) and abs(raw_p20_diff) >= float(raw_diff_eps):
        reduction_rate = float(reduction_abs / max(abs(raw_p20_diff), 1e-9))
    out.update(
        {
            "matched_valid": True,
            "matched_c2_n": int(len(c2_m)),
            "matched_c3_n": int(len(c3_m)),
            "matched_p20_c3_minus_c2": matched_p20_diff,
            "matched_enrichment20_c3_minus_c2": matched_e20_diff,
            "matched_reduction_abs": reduction_abs,
            "matched_reduction_rate": reduction_rate,
            "matched_diff_reduction_abs": reduction_abs,
            "matched_diff_reduction_rate": reduction_rate,
        }
    )
    return out


def tier_stability_audit(
    pred_df: pd.DataFrame,
    *,
    feature_df: pd.DataFrame | None = None,
    input_shift_features: Sequence[str] = DEFAULT_INPUT_SHIFT_FEATURES,
    match_strata: Sequence[str] = DEFAULT_MATCH_STRATA,
    top_k: int = 20,
    common_k_min: int = 15,
    min_test_c2_n: int = 25,
    min_test_c3_n: int = 20,
    raw_diff_eps: float = 0.02,
) -> tuple[bool, str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty_trace = pd.DataFrame(
        columns=[
            "split_id",
            "metric",
            "value",
            "common_k_min",
            "min_test_c2_n",
            "min_test_c3_n",
            "raw_diff_eps",
            "n_splits_used",
            "tier2d_level",
        ]
    )
    empty_input = pd.DataFrame(columns=["split_id", "feature_name", "c2_n", "c3_n", "ks_stat", "wasserstein", "mmd_rbf"])
    empty_score = pd.DataFrame(
        columns=["split_id", "c2_n", "c3_n", "score_ks_stat", "score_wasserstein", "score_mmd_rbf", "topk", "topk_c2_share", "topk_c3_share", "topk_c3_minus_c2"]
    )
    empty_perf = pd.DataFrame(
        columns=[
            "row_type",
            "split_id",
            "p20_c2",
            "p20_c3",
            "p20_c3_minus_c2",
            "enrichment20_c2",
            "enrichment20_c3",
            "enrichment20_c3_minus_c2",
            "abs_p20_diff",
            "n_splits",
            "median_abs_p20_diff",
            "p95_abs_p20_diff",
            "sign_rate",
            "k_used",
            "tier2d_level",
            "n_splits_used",
            "common_k_min",
            "min_test_c2_n",
            "min_test_c3_n",
            "raw_diff_eps",
            "skipped_due_to_small_tier_n",
            "skipped_small_tier_count",
            "skipped_common_k_count",
            "matched_reduction_abs_median",
            "matched_reduction_rate_median",
            "matched_reduction_median",
        ]
    )
    empty_matched = pd.DataFrame(
        columns=[
            "row_type",
            "split_id",
            "matched_valid",
            "match_strata",
            "matched_c2_n",
            "matched_c3_n",
            "raw_p20_c3_minus_c2",
            "matched_p20_c3_minus_c2",
            "raw_enrichment20_c3_minus_c2",
            "matched_enrichment20_c3_minus_c2",
            "matched_reduction_abs",
            "matched_reduction_rate",
            "matched_diff_reduction_abs",
            "matched_diff_reduction_rate",
            "common_k_min",
            "min_test_c2_n",
            "min_test_c3_n",
            "raw_diff_eps",
            "n_splits_used",
            "tier2d_level",
        ]
    )
    if pred_df.empty:
        return True, "PASS", "no_predictions", "all", empty_trace, empty_input, empty_score, empty_perf, empty_matched

    meta_df: pd.DataFrame | None = None
    if feature_df is not None and (not feature_df.empty) and ("row_idx" in pred_df.columns):
        cols = [col for col in set(input_shift_features) | set(match_strata) if col in feature_df.columns]
        if cols:
            meta_df = feature_df.reset_index(drop=True).copy()
            meta_df["row_idx"] = np.arange(len(meta_df))
            meta_df = meta_df[["row_idx", *cols]]

    input_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    diffs: list[float] = []
    signed_diffs: list[float] = []
    skipped_due_to_small_tier_n = 0
    skipped_due_to_common_k = 0

    for split_id, grp_base in pred_df.groupby("split_id"):
        grp = grp_base.copy()
        if meta_df is not None:
            grp = grp.merge(meta_df, on="row_idx", how="left")
        c2 = grp[grp["coverage_tier"] == "C2"]
        c3 = grp[grp["coverage_tier"] == "C3"]
        c2_n = int(len(c2))
        c3_n = int(len(c3))
        if c2_n < int(min_test_c2_n) or c3_n < int(min_test_c3_n):
            skipped_due_to_small_tier_n += 1
            continue
        common_k = int(min(int(top_k), c2_n, c3_n))
        if common_k < int(common_k_min):
            skipped_due_to_common_k += 1
            continue
        p2 = stage3_models.precision_at_k(c2["y_true"].to_numpy(dtype=int), c2["score"].to_numpy(dtype=float), common_k)
        p3 = stage3_models.precision_at_k(c3["y_true"].to_numpy(dtype=int), c3["score"].to_numpy(dtype=float), common_k)
        e2 = stage3_models.enrichment_at_k(c2["y_true"].to_numpy(dtype=int), c2["score"].to_numpy(dtype=float), common_k)
        e3 = stage3_models.enrichment_at_k(c3["y_true"].to_numpy(dtype=int), c3["score"].to_numpy(dtype=float), common_k)
        if not (np.isfinite(p2) and np.isfinite(p3)):
            continue
        p20_diff = float(p3 - p2)
        e20_diff = float(e3 - e2) if np.isfinite(e2) and np.isfinite(e3) else np.nan
        diffs.append(abs(p20_diff))
        signed_diffs.append(p20_diff)
        perf_rows.append(
            {
                "row_type": "split",
                "split_id": int(split_id),
                "p20_c2": float(p2),
                "p20_c3": float(p3),
                "p20_c3_minus_c2": p20_diff,
                "enrichment20_c2": float(e2) if np.isfinite(e2) else np.nan,
                "enrichment20_c3": float(e3) if np.isfinite(e3) else np.nan,
                "enrichment20_c3_minus_c2": e20_diff,
                "abs_p20_diff": float(abs(p20_diff)),
                "n_splits": np.nan,
                "median_abs_p20_diff": np.nan,
                "p95_abs_p20_diff": np.nan,
                "sign_rate": np.nan,
                "k_used": int(common_k),
                "tier2d_level": "",
                "n_splits_used": np.nan,
                "common_k_min": int(common_k_min),
                "min_test_c2_n": int(min_test_c2_n),
                "min_test_c3_n": int(min_test_c3_n),
                "raw_diff_eps": float(raw_diff_eps),
                "skipped_due_to_small_tier_n": np.nan,
                "skipped_small_tier_count": np.nan,
                "skipped_common_k_count": np.nan,
                "matched_reduction_abs_median": np.nan,
                "matched_reduction_rate_median": np.nan,
                "matched_reduction_median": np.nan,
            }
        )
        score_rows.append(_build_score_shift_row(int(split_id), grp, int(top_k)))
        if meta_df is not None:
            input_rows.extend(_build_input_shift_rows(int(split_id), grp, input_shift_features))
            matched_row = _match_control_row(
                int(split_id),
                grp,
                top_k=int(common_k),
                match_strata=match_strata,
                raw_p20_diff=p20_diff,
                raw_enrichment_diff=e20_diff,
                raw_diff_eps=float(raw_diff_eps),
            )
            matched_row.update(
                {
                    "common_k_min": int(common_k_min),
                    "min_test_c2_n": int(min_test_c2_n),
                    "min_test_c3_n": int(min_test_c3_n),
                    "raw_diff_eps": float(raw_diff_eps),
                    "n_splits_used": np.nan,
                    "tier2d_level": "",
                }
            )
            matched_rows.append(matched_row)

    if not diffs:
        input_df = pd.DataFrame(input_rows) if input_rows else empty_input.copy()
        score_df = pd.DataFrame(score_rows) if score_rows else empty_score.copy()
        matched_df = pd.DataFrame(matched_rows) if matched_rows else empty_matched.copy()
        detail = (
            "insufficient_data"
            f";common_k_min={int(common_k_min)}"
            f";min_test_c2_n={int(min_test_c2_n)}"
            f";min_test_c3_n={int(min_test_c3_n)}"
            f";raw_diff_eps={float(raw_diff_eps):.4f}"
            f";n_splits_used=0"
            f";skipped_due_to_small_tier_n={str(skipped_due_to_small_tier_n > 0).lower()}"
            f";skipped_small_tier_count={int(skipped_due_to_small_tier_n)}"
            f";skipped_common_k_count={int(skipped_due_to_common_k)}"
        )
        return True, "PASS", detail, "all", empty_trace, input_df, score_df, empty_perf, matched_df

    p95_abs_diff = float(np.percentile(diffs, 95))
    median_abs_diff = float(np.median(diffs))
    mean_signed = float(np.mean(signed_diffs)) if signed_diffs else 0.0
    sign_rate = _directional_sign_rate(signed_diffs)

    matched_df = pd.DataFrame(matched_rows) if matched_rows else pd.DataFrame(columns=empty_matched.columns)
    if not matched_df.empty:
        matched_df["row_type"] = "split"
        valid_reduction = matched_df[
            matched_df["matched_valid"].astype(bool)
            & matched_df["matched_diff_reduction_rate"].apply(lambda x: np.isfinite(float(x)) if pd.notna(x) else False)
        ]
        valid_reduction_rate = pd.to_numeric(valid_reduction["matched_reduction_rate"], errors="coerce")
        valid_reduction_rate = valid_reduction_rate[np.isfinite(valid_reduction_rate)]
        valid_reduction_abs = pd.to_numeric(matched_df.loc[matched_df["matched_valid"].astype(bool), "matched_reduction_abs"], errors="coerce")
        valid_reduction_abs = valid_reduction_abs[np.isfinite(valid_reduction_abs)]
        matched_reduction_rate_median = float(np.median(valid_reduction_rate.to_numpy(dtype=float))) if not valid_reduction_rate.empty else np.nan
        matched_reduction_abs_median = float(np.median(valid_reduction_abs.to_numpy(dtype=float))) if not valid_reduction_abs.empty else np.nan
    else:
        matched_reduction_rate_median = np.nan
        matched_reduction_abs_median = np.nan

    matched_reduction_median = matched_reduction_abs_median

    directional_shift = bool(np.isfinite(sign_rate) and sign_rate >= 0.60)
    if (not directional_shift) or (p95_abs_diff <= 0.20):
        tier2d_level = "PASS"
    elif p95_abs_diff <= 0.30:
        tier2d_level = "WARN"
    else:
        tier2d_level = "FAIL"
    fail = tier2d_level == "FAIL"
    shift_large = p95_abs_diff > 0.20
    matched_persistent = bool((not np.isfinite(matched_reduction_abs_median)) or (matched_reduction_abs_median < 0.0))

    applicability_domain = "all"
    if fail:
        applicability_domain = "C3_only" if mean_signed > 0 else "C2_only"

    detail = (
        f"n={len(diffs)},p95_abs_diff={p95_abs_diff:.4f},sign_rate={sign_rate:.4f},"
        f"tier2d_level={tier2d_level},"
        f"matched_reduction_abs_median={matched_reduction_abs_median:.4f},"
        f"matched_reduction_rate_median={matched_reduction_rate_median:.4f},"
        f"gates=(shift_large={str(shift_large).lower()},directional={str(directional_shift).lower()},matched_persistent={str(matched_persistent).lower()}),"
        f"common_k_min={int(common_k_min)},min_test_c2_n={int(min_test_c2_n)},min_test_c3_n={int(min_test_c3_n)},raw_diff_eps={float(raw_diff_eps):.4f},"
        f"n_splits_used={int(len(diffs))},"
        f"skipped_due_to_small_tier_n={str(skipped_due_to_small_tier_n > 0).lower()},"
        f"skipped_small_tier_count={int(skipped_due_to_small_tier_n)},"
        f"skipped_common_k_count={int(skipped_due_to_common_k)}"
    )

    perf_df = pd.DataFrame(perf_rows) if perf_rows else empty_perf.copy()
    summary_row = {
        "row_type": "summary",
        "split_id": -1,
        "p20_c2": np.nan,
        "p20_c3": np.nan,
        "p20_c3_minus_c2": np.nan,
        "enrichment20_c2": np.nan,
        "enrichment20_c3": np.nan,
        "enrichment20_c3_minus_c2": np.nan,
        "abs_p20_diff": np.nan,
        "n_splits": int(len(diffs)),
        "median_abs_p20_diff": median_abs_diff,
        "p95_abs_p20_diff": p95_abs_diff,
        "sign_rate": sign_rate,
        "k_used": np.nan,
        "tier2d_level": tier2d_level,
        "n_splits_used": int(len(diffs)),
        "common_k_min": int(common_k_min),
        "min_test_c2_n": int(min_test_c2_n),
        "min_test_c3_n": int(min_test_c3_n),
        "raw_diff_eps": float(raw_diff_eps),
        "skipped_due_to_small_tier_n": bool(skipped_due_to_small_tier_n > 0),
        "skipped_small_tier_count": int(skipped_due_to_small_tier_n),
        "skipped_common_k_count": int(skipped_due_to_common_k),
        "matched_reduction_abs_median": matched_reduction_abs_median,
        "matched_reduction_rate_median": matched_reduction_rate_median,
        "matched_reduction_median": matched_reduction_median,
    }
    perf_df = pd.concat([perf_df, pd.DataFrame([summary_row])], ignore_index=True)

    if matched_df.empty:
        matched_df = empty_matched.copy()
    else:
        matched_summary = {
            "row_type": "summary",
            "split_id": -1,
            "matched_valid": bool((matched_df["matched_valid"].astype(bool)).sum() > 0),
            "match_strata": "|".join(match_strata),
            "matched_c2_n": int(pd.to_numeric(matched_df["matched_c2_n"], errors="coerce").fillna(0).sum()),
            "matched_c3_n": int(pd.to_numeric(matched_df["matched_c3_n"], errors="coerce").fillna(0).sum()),
            "raw_p20_c3_minus_c2": np.nan,
            "matched_p20_c3_minus_c2": np.nan,
            "raw_enrichment20_c3_minus_c2": np.nan,
            "matched_enrichment20_c3_minus_c2": np.nan,
            "matched_reduction_abs": matched_reduction_abs_median,
            "matched_reduction_rate": matched_reduction_rate_median,
            "matched_diff_reduction_abs": np.nan,
            "matched_diff_reduction_rate": matched_reduction_rate_median,
            "common_k_min": int(common_k_min),
            "min_test_c2_n": int(min_test_c2_n),
            "min_test_c3_n": int(min_test_c3_n),
            "raw_diff_eps": float(raw_diff_eps),
            "n_splits_used": int(len(diffs)),
            "tier2d_level": tier2d_level,
        }
        matched_df = pd.concat([matched_df, pd.DataFrame([matched_summary])], ignore_index=True)

    trace_rows = [
        [
            "split_id",
            "metric",
            "value",
            "common_k_min",
            "min_test_c2_n",
            "min_test_c3_n",
            "raw_diff_eps",
            "n_splits_used",
            "tier2d_level",
        ]
    ]
    for row in perf_rows:
        trace_rows.append(
            [
                int(row["split_id"]),
                "p20_c3_minus_c2",
                float(row["p20_c3_minus_c2"]),
                int(common_k_min),
                int(min_test_c2_n),
                int(min_test_c3_n),
                float(raw_diff_eps),
                int(len(diffs)),
                tier2d_level,
            ]
        )
    trace = pd.DataFrame(trace_rows[1:], columns=trace_rows[0])
    input_df = pd.DataFrame(input_rows) if input_rows else empty_input.copy()
    score_df = pd.DataFrame(score_rows) if score_rows else empty_score.copy()
    return (not fail), tier2d_level, detail, applicability_domain, trace, input_df, score_df, perf_df, matched_df


def candidate_consistency_audit(rulebook_support: pd.DataFrame, candidate_trace: pd.DataFrame) -> tuple[bool, str]:
    if rulebook_support.empty:
        return True, "no_rules"
    if candidate_trace.empty:
        return True, "no_candidate_trace"
    known_ids = set(candidate_trace["candidate_id"].tolist())
    unknown = rulebook_support[~rulebook_support["rule_id"].isin(known_ids)] if "rule_id" in rulebook_support.columns else pd.DataFrame()
    missing_invalid = rulebook_support[
        rulebook_support["condition_text"].astype(str).str.contains("__MISSING__|missing_flag")
    ]
    if unknown.empty and missing_invalid.empty:
        return True, "ok"
    return False, f"unknown_rules={len(unknown)},missing_rulebook_rows={len(missing_invalid)}"
