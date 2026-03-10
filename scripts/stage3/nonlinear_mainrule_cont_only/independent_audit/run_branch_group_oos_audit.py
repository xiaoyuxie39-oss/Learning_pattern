#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared import (
    audit_model_branch_dir,
    build_audit_context,
    compute_delta_ci,
    compute_group_oos_metrics,
    load_predictions,
    load_trainable_audit_feature_view,
    source_model_branch_dir,
    summary_value,
    tier_stability_audit,
    top_ks_for_audit,
    write_audit_metadata,
)


def _metric_mean(summary_df: pd.DataFrame, scope: str, metric: str, k: int) -> float:
    subset = summary_df[
        (summary_df["scope"].astype(str) == str(scope))
        & (summary_df["metric"].astype(str) == str(metric))
        & (summary_df["k"].astype(int) == int(k))
    ]
    if subset.empty:
        return np.nan
    return float(subset.iloc[0]["mean"])


def _tier_overview_row(
    *,
    model_name: str,
    feature_mode: str,
    branch_name: str,
    tier_pass: bool,
    tier_level: str,
    detail: str,
    applicability_domain: str,
    perf_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "model": str(model_name),
        "feature_mode": str(feature_mode),
        "branch": str(branch_name),
        "tier_pass": bool(tier_pass),
        "tier2d_level": str(tier_level),
        "applicability_domain": str(applicability_domain),
        "p95_abs_p20_diff": summary_value(perf_df, "p95_abs_p20_diff"),
        "median_abs_p20_diff": summary_value(perf_df, "median_abs_p20_diff"),
        "sign_rate": summary_value(perf_df, "sign_rate"),
        "matched_reduction_abs_median": summary_value(perf_df, "matched_reduction_abs_median"),
        "matched_reduction_rate_median": summary_value(perf_df, "matched_reduction_rate_median"),
        "n_splits_used": summary_value(perf_df, "n_splits_used"),
        "detail": str(detail),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent branch-level group OOS + tier2d audit for nonlinear cont_only v2")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    parser.add_argument("--model", default="gbdt", help="Model name, default gbdt")
    parser.add_argument("--feature-mode", default="cont_only", help="Feature mode, default cont_only")
    parser.add_argument(
        "--branches",
        default="mainline,mainline_plus_pairwise,mainline_plus_3way",
        help="Comma-separated branch list",
    )
    args = parser.parse_args()

    ctx = build_audit_context(args.manifest)
    model_name = str(args.model)
    feature_mode = str(args.feature_mode)
    branch_names = [part.strip() for part in str(args.branches).split(",") if part.strip()]
    feature_df = load_trainable_audit_feature_view(ctx)
    top_ks = top_ks_for_audit(ctx)
    primary_k = int(ctx.base_ctx.execution.get("primary_k", 20))
    common_k_min = int(ctx.base_ctx.execution.get("tier2d_common_k_min", 15))
    min_test_c2_n = int(ctx.base_ctx.execution.get("min_test_c2_n", 25))
    min_test_c3_n = int(ctx.base_ctx.execution.get("min_test_c3_n", 20))
    raw_diff_eps = float(ctx.base_ctx.execution.get("tier2d_raw_diff_eps", 0.02))

    branch_split_frames: list[pd.DataFrame] = []
    branch_summary_frames: list[pd.DataFrame] = []
    tier_rows: list[dict[str, Any]] = []
    brief_lines: list[str] = [
        "# Branch Group OOS Audit Brief",
        "",
        f"- source_run_id: {ctx.base_ctx.run_id}",
        f"- model: {model_name}",
        f"- feature_mode: {feature_mode}",
        f"- primary_k: {primary_k}",
        f"- top_ks: {', '.join(str(v) for v in top_ks)}",
        "",
    ]

    for branch_name in branch_names:
        source_dir = source_model_branch_dir(ctx, model_name, feature_mode, branch_name)
        out_dir = audit_model_branch_dir(ctx, model_name, feature_mode, branch_name)
        pred_df = load_predictions(ctx, model_name, feature_mode, branch_name)

        split_metric_df, summary_df = compute_group_oos_metrics(
            pred_df,
            model_name=model_name,
            branch_name=branch_name,
            top_ks=top_ks,
        )
        split_metric_df.to_csv(out_dir / "group_oos_metrics_by_split.csv", index=False)
        summary_df.to_csv(out_dir / "group_oos_metrics_ci.csv", index=False)

        (
            tier_pass,
            tier_level,
            tier_detail,
            applicability_domain,
            trace_df,
            input_shift_df,
            score_shift_df,
            perf_shift_df,
            matched_df,
        ) = tier_stability_audit(
            pred_df,
            feature_df=feature_df,
            top_k=primary_k,
            common_k_min=common_k_min,
            min_test_c2_n=min_test_c2_n,
            min_test_c3_n=min_test_c3_n,
            raw_diff_eps=raw_diff_eps,
        )
        trace_df.to_csv(out_dir / "tier2d_audit_trace.csv", index=False)
        input_shift_df.to_csv(out_dir / "tier2d_input_shift_metrics.csv", index=False)
        score_shift_df.to_csv(out_dir / "tier2d_score_shift_metrics.csv", index=False)
        perf_shift_df.to_csv(out_dir / "tier2d_perf_shift_metrics.csv", index=False)
        matched_df.to_csv(out_dir / "tier2d_matched_control.csv", index=False)

        tier_row = _tier_overview_row(
            model_name=model_name,
            feature_mode=feature_mode,
            branch_name=branch_name,
            tier_pass=bool(tier_pass),
            tier_level=str(tier_level),
            detail=str(tier_detail),
            applicability_domain=str(applicability_domain),
            perf_df=perf_shift_df,
        )
        tier_rows.append(tier_row)
        branch_split_frames.append(split_metric_df)
        branch_summary_frames.append(summary_df)

        brief_lines.extend(
            [
                f"## {branch_name}",
                f"- overall P@{primary_k}: {_metric_mean(summary_df, 'overall', 'P', primary_k):.4f}",
                f"- overall Enrichment@{primary_k}: {_metric_mean(summary_df, 'overall', 'Enrichment', primary_k):.4f}",
                f"- overall AUC_proxy: {_metric_mean(summary_df, 'overall', 'AUC_proxy', 0):.4f}",
                f"- C2 P@{primary_k}: {_metric_mean(summary_df, 'C2', 'P', primary_k):.4f}",
                f"- C3 P@{primary_k}: {_metric_mean(summary_df, 'C3', 'P', primary_k):.4f}",
                f"- tier2d: {tier_level} | pass={str(bool(tier_pass)).lower()} | applicability_domain={applicability_domain}",
                f"- tier2d_detail: {tier_detail}",
                "",
            ]
        )
        write_audit_metadata(
            ctx,
            out_dir,
            {
                "source_run_id": ctx.base_ctx.run_id,
                "source_dir": str(source_dir),
                "audit_dir": str(out_dir),
                "model": model_name,
                "feature_mode": feature_mode,
                "branch": branch_name,
                "primary_k": primary_k,
                "top_ks": list(top_ks),
                "tier2d_level": tier_level,
                "tier_pass": bool(tier_pass),
                "applicability_domain": applicability_domain,
            },
        )

    all_split_df = pd.concat(branch_split_frames, ignore_index=True) if branch_split_frames else pd.DataFrame()
    all_summary_df = pd.concat(branch_summary_frames, ignore_index=True) if branch_summary_frames else pd.DataFrame()
    tier_overview_df = pd.DataFrame(tier_rows).sort_values(["model", "feature_mode", "branch"]).reset_index(drop=True)

    root_dir = ctx.branch_root / model_name / feature_mode
    root_dir.mkdir(parents=True, exist_ok=True)
    all_split_df.to_csv(root_dir / "branch_group_oos_metrics_by_split.csv", index=False)
    all_summary_df.to_csv(root_dir / "branch_group_oos_metrics_ci.csv", index=False)
    tier_overview_df.to_csv(root_dir / "branch_tier2d_overview.csv", index=False)

    delta_split_frames: list[pd.DataFrame] = []
    delta_summary_frames: list[pd.DataFrame] = []
    comparisons = [
        ("mainline_plus_pairwise", "mainline"),
        ("mainline_plus_3way", "mainline_plus_pairwise"),
    ]
    for left_branch, right_branch in comparisons:
        if left_branch not in branch_names or right_branch not in branch_names:
            continue
        delta_split_df, delta_summary_df = compute_delta_ci(
            all_split_df,
            model_name=model_name,
            left_branch=left_branch,
            right_branch=right_branch,
        )
        if not delta_split_df.empty:
            delta_split_frames.append(delta_split_df)
        if not delta_summary_df.empty:
            delta_summary_frames.append(delta_summary_df)

    all_delta_split_df = pd.concat(delta_split_frames, ignore_index=True) if delta_split_frames else pd.DataFrame()
    all_delta_summary_df = pd.concat(delta_summary_frames, ignore_index=True) if delta_summary_frames else pd.DataFrame()
    all_delta_split_df.to_csv(root_dir / "branch_group_oos_delta_by_split.csv", index=False)
    all_delta_summary_df.to_csv(root_dir / "branch_group_oos_delta_ci.csv", index=False)

    if not all_delta_summary_df.empty:
        brief_lines.append("## Delta")
        for comparison in all_delta_summary_df["comparison"].drop_duplicates().tolist():
            for scope in ["overall", "C2", "C3"]:
                p_row = all_delta_summary_df[
                    (all_delta_summary_df["comparison"].astype(str) == str(comparison))
                    & (all_delta_summary_df["scope"].astype(str) == str(scope))
                    & (all_delta_summary_df["metric"].astype(str) == "P")
                    & (all_delta_summary_df["k"].astype(int) == primary_k)
                ]
                e_row = all_delta_summary_df[
                    (all_delta_summary_df["comparison"].astype(str) == str(comparison))
                    & (all_delta_summary_df["scope"].astype(str) == str(scope))
                    & (all_delta_summary_df["metric"].astype(str) == "Enrichment")
                    & (all_delta_summary_df["k"].astype(int) == primary_k)
                ]
                if p_row.empty and e_row.empty:
                    continue
                p_mean = float(p_row.iloc[0]["delta_mean"]) if not p_row.empty else np.nan
                e_mean = float(e_row.iloc[0]["delta_mean"]) if not e_row.empty else np.nan
                brief_lines.append(f"- {comparison} | {scope} | dP@{primary_k}={p_mean:.4f} | dEnrichment@{primary_k}={e_mean:.4f}")
        brief_lines.append("")

    brief_file = root_dir / "branch_audit_brief.md"
    brief_file.write_text("\n".join(brief_lines).rstrip() + "\n", encoding="utf-8")
    write_audit_metadata(
        ctx,
        root_dir,
        {
            "source_run_id": ctx.base_ctx.run_id,
            "source_output_root": str(ctx.base_ctx.out_root),
            "audit_dir": str(root_dir),
            "model": model_name,
            "feature_mode": feature_mode,
            "branches": branch_names,
            "primary_k": primary_k,
            "top_ks": list(top_ks),
            "tier2d_common_k_min": common_k_min,
            "min_test_c2_n": min_test_c2_n,
            "min_test_c3_n": min_test_c3_n,
            "raw_diff_eps": raw_diff_eps,
        },
    )
    print(str(root_dir))


if __name__ == "__main__":
    main()
