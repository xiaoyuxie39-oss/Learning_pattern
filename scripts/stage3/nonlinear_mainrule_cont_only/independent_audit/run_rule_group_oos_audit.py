#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared import (
    build_audit_context,
    compile_rule_mask,
    load_predictions,
    load_published_gbdt_interaction_rules,
    load_trainable_audit_feature_view,
    write_audit_metadata,
)


def _branch_rule_dir(root_dir: Path, source_branch: str) -> Path:
    out_dir = root_dir / str(source_branch)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _evaluate_rule_by_split(rule_row: pd.Series, eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_id, split_df in eval_df.groupby("split_id", sort=True):
        for scope in ["overall", "C2", "C3"]:
            if scope == "overall":
                scope_df = split_df.copy()
            else:
                scope_df = split_df[split_df["coverage_tier"].astype(str) == str(scope)].copy()
            if scope_df.empty:
                continue
            mask = compile_rule_mask(scope_df, str(rule_row["condition_text"]))
            hit_df = scope_df[mask].copy()
            test_n = int(len(scope_df))
            test_pos = int(scope_df["y_true"].sum())
            active_n = int(len(hit_df))
            active_pos = int(hit_df["y_true"].sum()) if active_n > 0 else 0
            coverage = float(active_n / test_n) if test_n > 0 else np.nan
            precision_rule = float(active_pos / active_n) if active_n > 0 else np.nan
            prevalence = float(test_pos / test_n) if test_n > 0 else np.nan
            enrichment = float(precision_rule / prevalence) if active_n > 0 and prevalence > 0 else np.nan
            rows.append(
                {
                    "rule_id": str(rule_row["rule_id"]),
                    "source_branch": str(rule_row["source_branch"]),
                    "scope": str(scope),
                    "split_id": int(split_id),
                    "test_n": test_n,
                    "test_pos": test_pos,
                    "active_n": active_n,
                    "active_pos": active_pos,
                    "coverage": coverage,
                    "precision_rule": precision_rule,
                    "prevalence": prevalence,
                    "lift_vs_base": float(precision_rule - prevalence) if np.isfinite(precision_rule) and np.isfinite(prevalence) else np.nan,
                    "enrichment": enrichment,
                    "is_active_split": bool(active_n > 0),
                    "is_positive_split": bool(np.isfinite(enrichment) and enrichment > 1.0),
                }
            )
    return pd.DataFrame(rows)


def _evaluate_rule_by_company(rule_row: pd.Series, eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for company, company_df in eval_df.groupby("company", sort=True):
        mask = compile_rule_mask(company_df, str(rule_row["condition_text"]))
        hit_df = company_df[mask].copy()
        test_n = int(len(company_df))
        test_pos = int(company_df["y_true"].sum())
        active_n = int(len(hit_df))
        active_pos = int(hit_df["y_true"].sum()) if active_n > 0 else 0
        prevalence = float(test_pos / test_n) if test_n > 0 else np.nan
        precision_rule = float(active_pos / active_n) if active_n > 0 else np.nan
        enrichment = float(precision_rule / prevalence) if active_n > 0 and prevalence > 0 else np.nan
        rows.append(
            {
                "rule_id": str(rule_row["rule_id"]),
                "source_branch": str(rule_row["source_branch"]),
                "company": str(company),
                "n_splits_present": int(company_df["split_id"].nunique()),
                "test_n": test_n,
                "test_pos": test_pos,
                "active_n": active_n,
                "active_pos": active_pos,
                "coverage": float(active_n / test_n) if test_n > 0 else np.nan,
                "precision_rule": precision_rule,
                "prevalence": prevalence,
                "enrichment": enrichment,
                "is_active_company": bool(active_n > 0),
                "is_positive_company": bool(np.isfinite(enrichment) and enrichment > 1.0),
            }
        )
    return pd.DataFrame(rows)


def _summarize_rule(rule_row: pd.Series, split_df: pd.DataFrame, company_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    company_active = company_df[company_df["is_active_company"].astype(bool)].copy()
    for scope, scope_df in split_df.groupby("scope", sort=True):
        active = scope_df[scope_df["is_active_split"].astype(bool)].copy()
        coverage_vals = pd.to_numeric(scope_df["coverage"], errors="coerce")
        active_enrichment = pd.to_numeric(active["enrichment"], errors="coerce")
        active_enrichment = active_enrichment[np.isfinite(active_enrichment)]
        active_precision = pd.to_numeric(active["precision_rule"], errors="coerce")
        active_precision = active_precision[np.isfinite(active_precision)]
        active_support = pd.to_numeric(active["active_n"], errors="coerce")
        active_support = active_support[np.isfinite(active_support)]
        active_support_pos = pd.to_numeric(active["active_pos"], errors="coerce")
        active_support_pos = active_support_pos[np.isfinite(active_support_pos)]
        rows.append(
            {
                "rule_id": str(rule_row["rule_id"]),
                "source_branch": str(rule_row["source_branch"]),
                "rule_family": str(rule_row["rule_family"]),
                "scope": str(scope),
                "n_total_splits": int(scope_df["split_id"].nunique()),
                "n_active_splits": int(active["split_id"].nunique()),
                "active_split_rate": float(scope_df["is_active_split"].mean()) if not scope_df.empty else np.nan,
                "positive_split_rate": float(active["is_positive_split"].mean()) if not active.empty else np.nan,
                "coverage_mean": float(coverage_vals.mean()) if not coverage_vals.empty else np.nan,
                "coverage_median": float(coverage_vals.median()) if not coverage_vals.empty else np.nan,
                "coverage_ci_low_95": float(coverage_vals.quantile(0.025)) if not coverage_vals.empty else np.nan,
                "coverage_ci_high_95": float(coverage_vals.quantile(0.975)) if not coverage_vals.empty else np.nan,
                "support_n_median_active": float(active_support.median()) if not active_support.empty else np.nan,
                "support_pos_median_active": float(active_support_pos.median()) if not active_support_pos.empty else np.nan,
                "precision_mean_active": float(active_precision.mean()) if not active_precision.empty else np.nan,
                "precision_median_active": float(active_precision.median()) if not active_precision.empty else np.nan,
                "enrichment_mean_active": float(active_enrichment.mean()) if not active_enrichment.empty else np.nan,
                "enrichment_median_active": float(active_enrichment.median()) if not active_enrichment.empty else np.nan,
                "enrichment_ci_low_95_active": float(active_enrichment.quantile(0.025)) if not active_enrichment.empty else np.nan,
                "enrichment_ci_high_95_active": float(active_enrichment.quantile(0.975)) if not active_enrichment.empty else np.nan,
                "n_total_companies": int(company_df["company"].nunique()),
                "n_active_companies": int(company_active["company"].nunique()),
                "active_company_rate": float(company_df["is_active_company"].mean()) if not company_df.empty else np.nan,
                "positive_company_rate": float(company_active["is_positive_company"].mean()) if not company_active.empty else np.nan,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent rule-level group OOS audit for published cont_only interaction rules")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    parser.add_argument("--model", default="gbdt", help="Model name, default gbdt")
    parser.add_argument("--feature-mode", default="cont_only", help="Feature mode, default cont_only")
    args = parser.parse_args()

    ctx = build_audit_context(args.manifest)
    model_name = str(args.model)
    feature_mode = str(args.feature_mode)
    root_dir = ctx.rule_root / model_name / feature_mode
    root_dir.mkdir(parents=True, exist_ok=True)

    feature_df = load_trainable_audit_feature_view(ctx)
    rule_catalog = load_published_gbdt_interaction_rules(ctx)
    if rule_catalog.empty:
        raise RuntimeError("No published gbdt interaction rules found for rule-level audit.")

    branch_eval_frames: dict[str, pd.DataFrame] = {}
    for source_branch in sorted(rule_catalog["source_branch"].astype(str).unique()):
        pred_df = load_predictions(ctx, model_name, feature_mode, source_branch)
        merge_cols = [
            "row_idx",
            *[col for col in feature_df.columns if col != "row_idx" and col not in pred_df.columns],
        ]
        branch_eval_frames[source_branch] = pred_df.merge(feature_df[merge_cols], on="row_idx", how="left")

    split_frames: list[pd.DataFrame] = []
    company_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    brief_lines: list[str] = [
        "# Rule Group OOS Audit Brief",
        "",
        f"- source_run_id: {ctx.base_ctx.run_id}",
        f"- model: {model_name}",
        f"- feature_mode: {feature_mode}",
        f"- published_rule_count: {len(rule_catalog)}",
        "",
    ]

    for source_branch, branch_rule_df in rule_catalog.groupby("source_branch", sort=True):
        branch_dir = _branch_rule_dir(root_dir, str(source_branch))
        branch_rule_df.to_csv(branch_dir / "published_rule_catalog.csv", index=False)

        branch_split_frames: list[pd.DataFrame] = []
        branch_company_frames: list[pd.DataFrame] = []
        branch_summary_rows: list[dict[str, Any]] = []
        eval_df = branch_eval_frames[str(source_branch)]

        brief_lines.append(f"## {source_branch}")
        for _, rule_row in branch_rule_df.iterrows():
            split_df = _evaluate_rule_by_split(rule_row, eval_df)
            company_df = _evaluate_rule_by_company(rule_row, eval_df)
            summary_for_rule = _summarize_rule(rule_row, split_df, company_df)
            branch_split_frames.append(split_df)
            branch_company_frames.append(company_df)
            branch_summary_rows.extend(summary_for_rule)

            overall_summary = next((row for row in summary_for_rule if str(row["scope"]) == "overall"), None)
            if overall_summary is not None:
                brief_lines.extend(
                    [
                        f"- {rule_row['rule_id']}",
                        f"  condition: {rule_row['condition_text']}",
                        f"  active_split_rate={float(overall_summary['active_split_rate']):.4f} | positive_split_rate={float(overall_summary['positive_split_rate']):.4f}",
                        f"  enrichment_mean_active={float(overall_summary['enrichment_mean_active']):.4f} | active_company_rate={float(overall_summary['active_company_rate']):.4f}",
                    ]
                )
        brief_lines.append("")

        branch_split_df = pd.concat(branch_split_frames, ignore_index=True) if branch_split_frames else pd.DataFrame()
        branch_company_df = pd.concat(branch_company_frames, ignore_index=True) if branch_company_frames else pd.DataFrame()
        branch_summary_df = pd.DataFrame(branch_summary_rows).sort_values(["rule_id", "scope"]).reset_index(drop=True)
        branch_split_df.to_csv(branch_dir / "rule_group_oos_by_split.csv", index=False)
        branch_company_df.to_csv(branch_dir / "rule_group_oos_by_company.csv", index=False)
        branch_summary_df.to_csv(branch_dir / "rule_group_oos_summary.csv", index=False)
        write_audit_metadata(
            ctx,
            branch_dir,
            {
                "source_run_id": ctx.base_ctx.run_id,
                "audit_dir": str(branch_dir),
                "model": model_name,
                "feature_mode": feature_mode,
                "source_branch": str(source_branch),
                "published_rule_count": int(len(branch_rule_df)),
            },
        )

        if not branch_split_df.empty:
            split_frames.append(branch_split_df)
        if not branch_company_df.empty:
            company_frames.append(branch_company_df)
        summary_rows.extend(branch_summary_rows)

    split_df = pd.concat(split_frames, ignore_index=True) if split_frames else pd.DataFrame()
    company_df = pd.concat(company_frames, ignore_index=True) if company_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows).sort_values(["source_branch", "rule_id", "scope"]).reset_index(drop=True)

    rule_catalog.to_csv(root_dir / "published_rule_catalog.csv", index=False)
    split_df.to_csv(root_dir / "rule_group_oos_by_split.csv", index=False)
    company_df.to_csv(root_dir / "rule_group_oos_by_company.csv", index=False)
    summary_df.to_csv(root_dir / "rule_group_oos_summary.csv", index=False)
    (root_dir / "rule_audit_brief.md").write_text("\n".join(brief_lines).rstrip() + "\n", encoding="utf-8")
    write_audit_metadata(
        ctx,
        root_dir,
        {
            "source_run_id": ctx.base_ctx.run_id,
            "source_output_root": str(ctx.base_ctx.out_root),
            "audit_dir": str(root_dir),
            "model": model_name,
            "feature_mode": feature_mode,
            "published_rule_count": int(len(rule_catalog)),
        },
    )
    print(str(root_dir))


if __name__ == "__main__":
    main()
