#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from stage3_workflow_isolated import step04_audits as stage3_audits


def _summary_value(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return float("nan")
    summary = df[df["row_type"] == "summary"]
    if summary.empty:
        return float("nan")
    return float(summary.iloc[0][col])


def evaluate_variants(
    pred_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    variants: list[tuple[str, dict[str, Any]]] = [
        (
            "baseline_fixedk_legacy_floor",
            {
                "common_k_min": 0,
                "min_test_c2_n": 10,
                "min_test_c3_n": 10,
                "raw_diff_eps": 0.0,
            },
        ),
        (
            "commonk_min_floor_default",
            {
                "common_k_min": 15,
                "min_test_c2_n": 25,
                "min_test_c3_n": 20,
                "raw_diff_eps": 0.02,
            },
        ),
    ]
    rows: list[dict[str, Any]] = []
    for variant_name, cfg in variants:
        (
            tier_pass,
            tier_level,
            tier_detail,
            applicability_domain,
            _trace,
            _input,
            _score,
            perf_shift,
            matched_shift,
        ) = stage3_audits.tier_stability_audit(
            pred_df,
            feature_df=feature_df,
            top_k=int(top_k),
            common_k_min=int(cfg["common_k_min"]),
            min_test_c2_n=int(cfg["min_test_c2_n"]),
            min_test_c3_n=int(cfg["min_test_c3_n"]),
            raw_diff_eps=float(cfg["raw_diff_eps"]),
        )
        rows.append(
            {
                "variant": variant_name,
                "tier2d_pass": bool(tier_pass),
                "tier2d_level": str(tier_level),
                "applicability_domain": str(applicability_domain),
                "n_splits_used": int(_summary_value(perf_shift, "n_splits_used")),
                "p95_abs_diff": _summary_value(perf_shift, "p95_abs_p20_diff"),
                "sign_rate": _summary_value(perf_shift, "sign_rate"),
                "matched_reduction_abs_median": _summary_value(perf_shift, "matched_reduction_abs_median"),
                "matched_reduction_rate_median": _summary_value(perf_shift, "matched_reduction_rate_median"),
                "matched_metric_median": _summary_value(perf_shift, "matched_reduction_median"),
                "matched_valid_any": (
                    bool(int(_summary_value(matched_shift, "matched_valid")))
                    if (not matched_shift.empty and pd.notna(_summary_value(matched_shift, "matched_valid")))
                    else False
                ),
                "tier_detail": tier_detail,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tier2d audit variants on existing prediction outputs.")
    parser.add_argument("--pred-csv", required=True, help="Path to oof_predictions.csv")
    parser.add_argument("--feature-view-csv", required=True, help="Path to interaction_feature_view.csv")
    parser.add_argument("--out-csv", required=True, help="Path to output csv")
    parser.add_argument("--model", default="gbdt", help="Filter by model name")
    parser.add_argument("--feature-mode", default="cont_plus_bin", help="Filter by feature_mode")
    parser.add_argument("--branch", default="mainline", help="Filter by branch")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k used in tier2d audit")
    args = parser.parse_args()

    pred_csv = Path(args.pred_csv)
    feature_csv = Path(args.feature_view_csv)
    out_csv = Path(args.out_csv)

    pred_df = pd.read_csv(pred_csv)
    feature_df = pd.read_csv(feature_csv)
    feature_df = feature_df[feature_df["base_non_missing_count"] >= 2].copy().reset_index(drop=True)

    subset = pred_df.copy()
    if "model" in subset.columns:
        subset = subset[subset["model"].astype(str) == str(args.model)]
    if "feature_mode" in subset.columns:
        subset = subset[subset["feature_mode"].astype(str) == str(args.feature_mode)]
    if "branch" in subset.columns:
        subset = subset[subset["branch"].astype(str) == str(args.branch)]
    if subset.empty:
        raise SystemExit("No rows matched --model/--feature-mode/--branch filters.")

    out = evaluate_variants(subset, feature_df, top_k=int(args.top_k))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
