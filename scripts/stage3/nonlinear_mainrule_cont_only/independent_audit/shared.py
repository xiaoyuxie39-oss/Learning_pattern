#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
NONLINEAR_DIR = SCRIPT_DIR.parent
STAGE3_DIR = NONLINEAR_DIR.parent
if str(NONLINEAR_DIR) not in sys.path:
    sys.path.insert(0, str(NONLINEAR_DIR))
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from common import ensure_dir  # noqa: E402
from v2_shared import (  # noqa: E402
    RunContext,
    build_run_context,
    enrichment_at_k,
    precision_at_k,
    top_ks_from_execution,
    write_json,
)

DEFAULT_INPUT_SHIFT_FEATURES = ("power_mw", "rack_kw_typical", "pue")
DEFAULT_MATCH_STRATA = ("power_mw_bin", "rack_kw_typical_bin", "pue_bin")
CONDITION_PATTERN = re.compile(
    r"^\s*(?P<feature>[A-Za-z0-9_]+)\s*(?P<op>>=|<=|>|<|==)\s*(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)


@dataclass
class AuditContext:
    base_ctx: RunContext
    audit_root: Path
    branch_root: Path
    rule_root: Path


def build_audit_context(manifest_arg: str | Path) -> AuditContext:
    base_ctx = build_run_context(manifest_arg)
    audit_root = base_ctx.part2_out_dir / "nonlinear_cont_only_v2_independent_audit"
    branch_root = audit_root / "branch_group_oos"
    rule_root = audit_root / "rule_group_oos"
    ensure_dir(branch_root)
    ensure_dir(rule_root)
    return AuditContext(
        base_ctx=base_ctx,
        audit_root=audit_root,
        branch_root=branch_root,
        rule_root=rule_root,
    )


def load_trainable_audit_feature_view(ctx: AuditContext) -> pd.DataFrame:
    feature_view_file = ctx.base_ctx.part1_out_dir / "interaction_feature_view.csv"
    if not feature_view_file.exists():
        raise FileNotFoundError(f"Missing audit feature view: {feature_view_file}")
    df = pd.read_csv(feature_view_file)
    if "base_non_missing_count" not in df.columns:
        raise RuntimeError("interaction_feature_view.csv missing base_non_missing_count")
    trainable = df[df["base_non_missing_count"] >= 2].copy().reset_index(drop=True)
    trainable["row_idx"] = np.arange(len(trainable))
    return trainable


def source_model_branch_dir(ctx: AuditContext, model_name: str, feature_mode: str, branch_name: str) -> Path:
    return ctx.base_ctx.out_root / "models" / str(model_name) / str(feature_mode) / str(branch_name)


def audit_model_branch_dir(ctx: AuditContext, model_name: str, feature_mode: str, branch_name: str) -> Path:
    out_dir = ctx.branch_root / str(model_name) / str(feature_mode) / str(branch_name)
    ensure_dir(out_dir)
    return out_dir


def top_ks_for_audit(ctx: AuditContext) -> list[int]:
    return top_ks_from_execution(ctx.base_ctx.execution)


def load_predictions(ctx: AuditContext, model_name: str, feature_mode: str, branch_name: str) -> pd.DataFrame:
    pred_file = source_model_branch_dir(ctx, model_name, feature_mode, branch_name) / "predictions_oof.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_file}")
    pred_df = pd.read_csv(pred_file)
    required = {"split_id", "row_idx", "company", "coverage_tier", "y_true", "score"}
    missing = sorted(required - set(pred_df.columns))
    if missing:
        raise RuntimeError(f"predictions_oof.csv missing required columns: {missing}")
    return pred_df


def auc_proxy(y_true: Sequence[int], scores: Sequence[float]) -> float:
    y = pd.Series(y_true, dtype="float64")
    s = pd.Series(scores, dtype="float64")
    valid = y.notna() & s.notna()
    y = y[valid].astype(int)
    s = s[valid].astype(float)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return np.nan
    ranks = s.rank(method="average")
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - (pos * (pos + 1) / 2.0)) / (pos * neg))


def compute_group_oos_metrics(
    pred_df: pd.DataFrame,
    *,
    model_name: str,
    branch_name: str,
    top_ks: Sequence[int],
    scopes: Sequence[str] = ("overall", "C2", "C3"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_rows: list[dict[str, Any]] = []
    for split_id, split_df in pred_df.groupby("split_id", sort=True):
        for scope in scopes:
            if scope == "overall":
                eval_df = split_df.copy()
            else:
                eval_df = split_df[split_df["coverage_tier"].astype(str) == str(scope)].copy()
            if eval_df.empty:
                continue
            y = eval_df["y_true"].to_numpy(dtype=int)
            scores = eval_df["score"].to_numpy(dtype=float)
            prevalence = float(np.mean(y)) if len(y) else np.nan
            split_rows.append(
                {
                    "model": str(model_name),
                    "branch": str(branch_name),
                    "scope": str(scope),
                    "split_id": int(split_id),
                    "metric": "AUC_proxy",
                    "k": 0,
                    "value": auc_proxy(y, scores),
                    "n_rows": int(len(eval_df)),
                    "n_positive": int(y.sum()),
                    "prevalence": prevalence,
                }
            )
            for k in top_ks:
                k_used = int(min(int(k), len(eval_df)))
                split_rows.append(
                    {
                        "model": str(model_name),
                        "branch": str(branch_name),
                        "scope": str(scope),
                        "split_id": int(split_id),
                        "metric": "P",
                        "k": int(k),
                        "value": precision_at_k(y, scores, k_used),
                        "n_rows": int(len(eval_df)),
                        "n_positive": int(y.sum()),
                        "prevalence": prevalence,
                    }
                )
                split_rows.append(
                    {
                        "model": str(model_name),
                        "branch": str(branch_name),
                        "scope": str(scope),
                        "split_id": int(split_id),
                        "metric": "Enrichment",
                        "k": int(k),
                        "value": enrichment_at_k(y, scores, k_used),
                        "n_rows": int(len(eval_df)),
                        "n_positive": int(y.sum()),
                        "prevalence": prevalence,
                    }
                )
    split_df = pd.DataFrame(split_rows)
    if split_df.empty:
        empty_cols = ["model", "branch", "scope", "metric", "k", "mean", "std", "ci_low_95", "ci_high_95", "n_valid_splits"]
        return split_df, pd.DataFrame(columns=empty_cols)

    summary_rows: list[dict[str, Any]] = []
    for (model, branch, scope, metric, k), grp in split_df.groupby(
        ["model", "branch", "scope", "metric", "k"], dropna=False
    ):
        vals = pd.to_numeric(grp["value"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        summary_rows.append(
            {
                "model": str(model),
                "branch": str(branch),
                "scope": str(scope),
                "metric": str(metric),
                "k": int(k),
                "mean": float(vals.mean()) if not vals.empty else np.nan,
                "std": float(vals.std(ddof=0)) if not vals.empty else np.nan,
                "ci_low_95": float(vals.quantile(0.025)) if not vals.empty else np.nan,
                "ci_high_95": float(vals.quantile(0.975)) if not vals.empty else np.nan,
                "n_valid_splits": int(len(vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["model", "branch", "scope", "metric", "k"]
    ).reset_index(drop=True)
    return split_df.sort_values(["model", "branch", "scope", "split_id", "metric", "k"]).reset_index(drop=True), summary_df


def compute_delta_ci(
    split_metric_df: pd.DataFrame,
    *,
    model_name: str,
    left_branch: str,
    right_branch: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = split_metric_df[
        (split_metric_df["model"].astype(str) == str(model_name))
        & (split_metric_df["branch"].astype(str) == str(left_branch))
    ].copy()
    right = split_metric_df[
        (split_metric_df["model"].astype(str) == str(model_name))
        & (split_metric_df["branch"].astype(str) == str(right_branch))
    ].copy()
    key_cols = ["scope", "split_id", "metric", "k"]
    left = left[key_cols + ["value"]].rename(columns={"value": "value_left"})
    right = right[key_cols + ["value"]].rename(columns={"value": "value_right"})
    merged = left.merge(right, on=key_cols, how="inner")
    if merged.empty:
        empty_cols = ["model", "comparison", "scope", "metric", "k", "delta_mean", "delta_ci_low_95", "delta_ci_high_95", "n_valid_splits"]
        return merged, pd.DataFrame(columns=empty_cols)
    merged["comparison"] = f"{left_branch}_vs_{right_branch}"
    merged["model"] = str(model_name)
    merged["delta_value"] = pd.to_numeric(merged["value_left"], errors="coerce") - pd.to_numeric(merged["value_right"], errors="coerce")

    summary_rows: list[dict[str, Any]] = []
    for (model, comparison, scope, metric, k), grp in merged.groupby(["model", "comparison", "scope", "metric", "k"], dropna=False):
        vals = pd.to_numeric(grp["delta_value"], errors="coerce")
        vals = vals[np.isfinite(vals)]
        summary_rows.append(
            {
                "model": str(model),
                "comparison": str(comparison),
                "scope": str(scope),
                "metric": str(metric),
                "k": int(k),
                "delta_mean": float(vals.mean()) if not vals.empty else np.nan,
                "delta_ci_low_95": float(vals.quantile(0.025)) if not vals.empty else np.nan,
                "delta_ci_high_95": float(vals.quantile(0.975)) if not vals.empty else np.nan,
                "n_valid_splits": int(len(vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["model", "comparison", "scope", "metric", "k"]
    ).reset_index(drop=True)
    return merged.sort_values(["model", "comparison", "scope", "split_id", "metric", "k"]).reset_index(drop=True), summary_df


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
                "feature_name": str(feature),
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
    p20_c2 = precision_at_k(c2_m["y_true"].to_numpy(dtype=int), c2_m["score"].to_numpy(dtype=float), k)
    p20_c3 = precision_at_k(c3_m["y_true"].to_numpy(dtype=int), c3_m["score"].to_numpy(dtype=float), k)
    e20_c2 = enrichment_at_k(c2_m["y_true"].to_numpy(dtype=int), c2_m["score"].to_numpy(dtype=float), k)
    e20_c3 = enrichment_at_k(c3_m["y_true"].to_numpy(dtype=int), c3_m["score"].to_numpy(dtype=float), k)
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
            meta_df = feature_df[["row_idx", *cols]].copy()

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
        p2 = precision_at_k(c2["y_true"].to_numpy(dtype=int), c2["score"].to_numpy(dtype=float), common_k)
        p3 = precision_at_k(c3["y_true"].to_numpy(dtype=int), c3["score"].to_numpy(dtype=float), common_k)
        e2 = enrichment_at_k(c2["y_true"].to_numpy(dtype=int), c2["score"].to_numpy(dtype=float), common_k)
        e3 = enrichment_at_k(c3["y_true"].to_numpy(dtype=int), c3["score"].to_numpy(dtype=float), common_k)
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

    trace_rows = []
    for row in perf_rows:
        trace_rows.append(
            {
                "split_id": int(row["split_id"]),
                "metric": "p20_c3_minus_c2",
                "value": float(row["p20_c3_minus_c2"]),
                "common_k_min": int(common_k_min),
                "min_test_c2_n": int(min_test_c2_n),
                "min_test_c3_n": int(min_test_c3_n),
                "raw_diff_eps": float(raw_diff_eps),
                "n_splits_used": int(len(diffs)),
                "tier2d_level": tier2d_level,
            }
        )
    trace_df = pd.DataFrame(trace_rows) if trace_rows else empty_trace.copy()
    input_df = pd.DataFrame(input_rows) if input_rows else empty_input.copy()
    score_df = pd.DataFrame(score_rows) if score_rows else empty_score.copy()
    return (not fail), tier2d_level, detail, applicability_domain, trace_df, input_df, score_df, perf_df, matched_df


def summary_value(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return np.nan
    summary = df[df["row_type"] == "summary"]
    if summary.empty:
        return np.nan
    return float(summary.iloc[0][col])


def compile_rule_mask(df: pd.DataFrame, condition_text: str) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    parts = [part.strip() for part in str(condition_text).split(" and ") if part.strip()]
    if not parts:
        raise ValueError(f"Empty condition_text: {condition_text!r}")
    for part in parts:
        matched = CONDITION_PATTERN.match(part)
        if matched is None:
            raise ValueError(f"Unsupported rule condition: {part!r}")
        feature = str(matched.group("feature"))
        op = str(matched.group("op"))
        value = float(matched.group("value"))
        if feature not in df.columns:
            raise KeyError(f"Missing feature {feature!r} for rule condition: {condition_text!r}")
        series = pd.to_numeric(df[feature], errors="coerce")
        if op == ">=":
            current = series >= value
        elif op == "<=":
            current = series <= value
        elif op == ">":
            current = series > value
        elif op == "<":
            current = series < value
        elif op == "==":
            current = series == value
        else:
            raise ValueError(f"Unsupported operator {op!r}")
        mask &= current.fillna(False)
    return mask


def load_published_gbdt_interaction_rules(ctx: AuditContext) -> pd.DataFrame:
    pair_file = ctx.base_ctx.out_root / "rules_publish_realvalue.csv"
    three_file = ctx.base_ctx.out_root / "rules_3way_publish_realvalue.csv"
    frames: list[pd.DataFrame] = []
    if pair_file.exists():
        pair_df = pd.read_csv(pair_file)
        pair_df = pair_df[
            (pair_df["model"].astype(str) == "gbdt")
            & (pair_df["branch"].astype(str) == "mainline_plus_pairwise")
            & (pair_df["rule_family"].astype(str) == "interaction")
        ].copy()
        if not pair_df.empty:
            pair_branch_rules = ctx.base_ctx.out_root / "models" / "gbdt" / "cont_only" / "mainline_plus_pairwise" / "rules_interaction_realvalue.csv"
            if pair_branch_rules.exists():
                ref_df = pd.read_csv(pair_branch_rules)
                join_cols = [col for col in ["feature_a", "feature_b", "condition_text"] if col in pair_df.columns and col in ref_df.columns]
                if join_cols:
                    enrich_cols = [
                        col for col in ["rule_id", "support_pos", "coverage", "cutpoint_source"]
                        if col in ref_df.columns and col not in pair_df.columns
                    ]
                    if enrich_cols:
                        pair_df = pair_df.merge(
                            ref_df[join_cols + enrich_cols].drop_duplicates(subset=join_cols),
                            on=join_cols,
                            how="left",
                        )
            pair_df["feature_c"] = ""
            pair_df["source_branch"] = "mainline_plus_pairwise"
            pair_df["publish_file"] = str(pair_file)
            frames.append(pair_df)
    if three_file.exists():
        three_df = pd.read_csv(three_file)
        if not three_df.empty:
            three_df = three_df.copy()
            three_df["source_branch"] = "mainline_plus_3way"
            three_df["publish_file"] = str(three_file)
            if "publish_scope" not in three_df.columns:
                three_df["publish_scope"] = "threeway_gain_positive"
            frames.append(three_df)
    if not frames:
        return pd.DataFrame()
    rule_df = pd.concat(frames, ignore_index=True, sort=False)
    if "rule_id" not in rule_df.columns:
        rule_df["rule_id"] = np.nan
    missing_rule_id = rule_df["rule_id"].isna() | (rule_df["rule_id"].astype(str).str.strip() == "")
    if missing_rule_id.any():
        def _cell(row: pd.Series, key: str) -> str:
            value = row.get(key, "")
            return "" if pd.isna(value) else str(value)
        rule_df.loc[missing_rule_id, "rule_id"] = rule_df.loc[missing_rule_id].apply(
            lambda row: "::".join(
                [
                    _cell(row, "model"),
                    _cell(row, "branch"),
                    _cell(row, "rule_family"),
                    _cell(row, "feature_a"),
                    _cell(row, "feature_b"),
                    _cell(row, "feature_c"),
                ]
            ).rstrip(":"),
            axis=1,
        )
    for col in ["feature_b", "feature_c", "publish_scope", "publish_reason"]:
        if col not in rule_df.columns:
            rule_df[col] = ""
    keep_cols = [
        "rule_id",
        "model",
        "branch",
        "source_branch",
        "rule_family",
        "feature_a",
        "feature_b",
        "feature_c",
        "condition_text",
        "support_n",
        "support_pos",
        "coverage",
        "enrichment",
        "model_score",
        "rule_source",
        "publish_scope",
        "publish_reason",
        "publish_file",
    ]
    return rule_df[keep_cols].copy().reset_index(drop=True)


def write_audit_metadata(ctx: AuditContext, out_dir: Path, payload: dict[str, Any]) -> None:
    ensure_dir(out_dir)
    write_json(out_dir / "audit_run_summary.json", payload)
