#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v2_rules import (
    build_cutpoint_alignment_rows,
    build_cutpoint_map,
    build_ebm_interaction_rules,
    build_gbdt_interaction_rules,
    build_main_effect_rules,
)
from v2_shared import (
    add_pair_features,
    build_cont_only_feature_frame,
    build_run_context,
    evaluate_branch_with_splits,
    extract_metric_mean,
    feature_profile,
    load_cont_only_training_frame,
    load_or_create_splits,
    model_branch_dir,
    primary_k_from_execution,
    resolve_model_branch_override,
    select_top_pair_features,
    top_ks_from_execution,
    write_json,
)

RULE_COLUMNS = [
    "rule_id",
    "model",
    "branch",
    "rule_family",
    "feature_a",
    "feature_b",
    "condition_text",
    "support_n",
    "support_pos",
    "coverage",
    "enrichment",
    "model_score",
    "rule_source",
    "cutpoint_source",
]


def _nonlinear_v2_cfg(execution: dict[str, Any]) -> dict[str, Any]:
    cfg = execution.get("nonlinear_cont_only_v2", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "pair_limit": int(execution.get("pair_limit", cfg.get("pair_limit", 4))),
        "min_support_n": int(cfg.get("min_support_n", 10)),
        "min_support_pos": int(cfg.get("min_support_pos", 2)),
        "min_enrichment": float(cfg.get("min_enrichment", 1.0)),
        "max_rules_per_feature": int(cfg.get("max_rules_per_feature", 3)),
        "require_model_cutpoints": bool(cfg.get("require_model_cutpoints", True)),
        "allow_physical_fallback": bool(cfg.get("allow_physical_fallback", False)),
        "use_joint_pair_interval_search": bool(cfg.get("use_joint_pair_interval_search", False)),
    }


def run_model_branch(
    *,
    model_name: str,
    branch_name: str,
    manifest: str,
    force_rebuild_splits: bool = False,
    max_splits: int = 0,
) -> dict[str, Any]:
    ctx = build_run_context(manifest)
    cfg = _nonlinear_v2_cfg(ctx.execution)
    branch_override = resolve_model_branch_override(ctx.execution, model_name, branch_name)
    pair_limit = int(branch_override.get("pair_limit", cfg["pair_limit"]))
    model_params = branch_override.get("model_params", {})
    if not isinstance(model_params, dict):
        model_params = {}
    top_ks = top_ks_from_execution(ctx.execution)
    primary_k = primary_k_from_execution(ctx.execution)

    raw_df, y, input_file = load_cont_only_training_frame(ctx)
    splits, split_cfg = load_or_create_splits(ctx, raw_df, y, force_rebuild=force_rebuild_splits)
    if int(max_splits) > 0:
        splits = splits[: int(max_splits)]

    base_X = build_cont_only_feature_frame(raw_df)

    selected_pairs: list[tuple[str, str, float]] = []
    if model_name == "gbdt" and branch_name == "mainline_plus_pairwise":
        selected_pairs = select_top_pair_features(base_X, y, max_pairs=pair_limit)
        X = add_pair_features(base_X, selected_pairs)
    else:
        X = base_X

    branch_result = evaluate_branch_with_splits(
        model_name=model_name,
        branch_name=branch_name,
        X=X,
        raw_df=raw_df,
        y=y,
        splits=splits,
        random_seed=ctx.random_seed,
        pair_limit=pair_limit,
        top_ks=top_ks,
        model_params=model_params,
    )

    cut_map = build_cutpoint_map(
        model_name=model_name,
        model=branch_result.full_model,
        feature_names=branch_result.feature_columns,
        scaling_stats=branch_result.scaling_stats,
        source_df=raw_df,
    )
    cutpoint_alignment = build_cutpoint_alignment_rows(cut_map, raw_df)

    main_rules = build_main_effect_rules(
        model_name=model_name,
        branch_name=branch_name,
        model=branch_result.full_model,
        feature_names=branch_result.feature_columns,
        source_df=raw_df,
        y=y,
        cut_map=cut_map,
        min_support_n=int(cfg["min_support_n"]),
        min_support_pos=int(cfg["min_support_pos"]),
        min_enrichment=float(cfg["min_enrichment"]),
        max_rules_per_feature=int(cfg["max_rules_per_feature"]),
        require_model_cutpoints=bool(cfg["require_model_cutpoints"]),
        allow_physical_fallback=bool(cfg["allow_physical_fallback"]),
    )

    interaction_rules = pd.DataFrame(columns=RULE_COLUMNS)
    if branch_name == "mainline_plus_pairwise":
        if model_name == "ebm":
            interaction_rules = build_ebm_interaction_rules(
                model=branch_result.full_model,
                model_name=model_name,
                branch_name=branch_name,
                feature_names=branch_result.feature_columns,
                source_df=raw_df,
                y=y,
                cut_map=cut_map,
                pair_limit=pair_limit,
                min_support_n=int(cfg["min_support_n"]),
                min_support_pos=int(cfg["min_support_pos"]),
                min_enrichment=float(cfg["min_enrichment"]),
                require_model_cutpoints=bool(cfg["require_model_cutpoints"]),
                allow_physical_fallback=bool(cfg["allow_physical_fallback"]),
            )
        elif model_name == "gbdt":
            interaction_rules = build_gbdt_interaction_rules(
                model=branch_result.full_model,
                model_name=model_name,
                branch_name=branch_name,
                feature_names=branch_result.feature_columns,
                source_df=raw_df,
                y=y,
                cut_map=cut_map,
                pair_limit=pair_limit,
                min_support_n=int(cfg["min_support_n"]),
                min_support_pos=int(cfg["min_support_pos"]),
                min_enrichment=float(cfg["min_enrichment"]),
                require_model_cutpoints=bool(cfg["require_model_cutpoints"]),
                allow_physical_fallback=bool(cfg["allow_physical_fallback"]),
                use_joint_interval_search=bool(cfg["use_joint_pair_interval_search"]),
            )

    if main_rules.empty:
        main_rules = pd.DataFrame(columns=RULE_COLUMNS)
    if interaction_rules.empty:
        interaction_rules = pd.DataFrame(columns=RULE_COLUMNS)

    rule_frames: list[pd.DataFrame] = []
    if not main_rules.empty:
        rule_frames.append(main_rules)
    if not interaction_rules.empty:
        rule_frames.append(interaction_rules)
    if rule_frames:
        rules_all = pd.concat(rule_frames, ignore_index=True)
        rules_all = rules_all.sort_values(
            ["rule_family", "model_score", "enrichment", "support_n"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    else:
        rules_all = pd.DataFrame(columns=RULE_COLUMNS)

    out_dir = model_branch_dir(ctx, model_name, branch_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    branch_result.metrics_by_split.to_csv(out_dir / "metrics_by_split.csv", index=False)
    branch_result.metrics_ci.to_csv(out_dir / "metrics_topk_ci.csv", index=False)
    branch_result.predictions_oof.to_csv(out_dir / "predictions_oof.csv", index=False)
    main_rules.to_csv(out_dir / "rules_main_effect_realvalue.csv", index=False)
    interaction_rules.to_csv(out_dir / "rules_interaction_realvalue.csv", index=False)
    rules_all.to_csv(out_dir / "rules_realvalue.csv", index=False)
    cutpoint_alignment.to_csv(out_dir / "model_cutpoint_alignment.csv", index=False)
    branch_result.scaling_stats.to_csv(out_dir / "feature_scaling_stats.csv", index=False)

    if selected_pairs:
        pd.DataFrame(
            [
                {
                    "source_col_a": a,
                    "source_col_b": b,
                    "selection_score_abs_corr": score,
                }
                for a, b, score in selected_pairs
            ]
        ).to_csv(out_dir / "selected_pair_features.csv", index=False)

    p_mean = extract_metric_mean(branch_result.metrics_ci, "P", primary_k)
    e_mean = extract_metric_mean(branch_result.metrics_ci, "Enrichment", primary_k)

    summary = {
        "run_id": ctx.run_id,
        "model": model_name,
        "branch": branch_name,
        "feature_mode": "cont_only",
        "input_file": str(input_file),
        "output_dir": str(out_dir),
        "n_rows_trainable": int(len(raw_df)),
        "n_positive": int(y.sum()),
        "n_splits": int(len(splits)),
        "top_ks": list(top_ks),
        "primary_k": int(primary_k),
        "p_at_primary_k_mean": p_mean,
        "enrichment_at_primary_k_mean": e_mean,
        "feature_profile": feature_profile(X),
        "pair_limit": pair_limit,
        "selected_pair_feature_count": int(len(selected_pairs)),
        "split_config": split_cfg,
        "rule_config": {**cfg, "pair_limit": pair_limit, "model_params": model_params},
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary


def run_job_cli(*, model_name: str, branch_name: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run nonlinear cont_only v2 job: {model_name}/{branch_name}")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    parser.add_argument("--force-rebuild-splits", action="store_true", help="Rebuild shared splits even if cached")
    parser.add_argument("--max-splits", type=int, default=0, help="Debug only: cap number of splits for quick smoke runs")
    args = parser.parse_args()

    summary = run_model_branch(
        model_name=model_name,
        branch_name=branch_name,
        manifest=args.manifest,
        force_rebuild_splits=bool(args.force_rebuild_splits),
        max_splits=int(args.max_splits),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
