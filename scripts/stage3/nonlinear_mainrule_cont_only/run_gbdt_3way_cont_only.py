#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from v2_rules import (
    build_cutpoint_alignment_rows,
    build_cutpoint_map,
    build_gbdt_interaction_rules,
    build_gbdt_triple_rules,
    build_main_effect_rules,
)
from v2_shared import (
    CONTINUOUS_SOURCE_COLS,
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
    "feature_c",
    "condition_text",
    "support_n",
    "support_pos",
    "coverage",
    "enrichment",
    "model_score",
    "rule_source",
    "cutpoint_source",
]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def select_top_triple_features(
    X_pair: pd.DataFrame,
    y: np.ndarray,
    selected_pairs: list[tuple[str, str, float]],
    *,
    max_triples: int,
    min_coverage_rate: float,
    second_triple_min_coverage_rate: float,
    second_triple_min_support_n: int,
    second_triple_min_support_pos: int,
) -> list[tuple[str, str, str, float, float, int, int]]:
    rows: list[tuple[str, str, str, float, float, int, int]] = []
    if int(max_triples) <= 0:
        return rows
    if not selected_pairs:
        return rows

    y_int = y.astype(int)
    total_n = int(len(y_int))

    for src_a, src_b, _ in selected_pairs:
        col_a = f"cont::{src_a}"
        col_b = f"cont::{src_b}"
        if col_a not in X_pair.columns or col_b not in X_pair.columns:
            continue
        for src_c in CONTINUOUS_SOURCE_COLS:
            if src_c in {src_a, src_b}:
                continue
            col_c = f"cont::{src_c}"
            if col_c not in X_pair.columns:
                continue
            prod = (X_pair[col_a] * X_pair[col_b] * X_pair[col_c]).to_numpy(dtype=float)
            score = abs(_safe_corr(prod, y.astype(float)))
            nonzero = prod != 0
            support_n = int(nonzero.sum())
            support_pos = int(y_int[nonzero].sum()) if support_n > 0 else 0
            coverage_rate = float(support_n / total_n) if total_n > 0 else 0.0
            rows.append((src_a, src_b, src_c, float(score), coverage_rate, support_n, support_pos))

    dedup: dict[tuple[str, str, str], tuple[float, float, int, int]] = {}
    for a, b, c, score, coverage_rate, support_n, support_pos in rows:
        key = tuple(sorted([a, b, c]))
        prev = dedup.get(key)
        if prev is None or score > float(prev[0]):
            dedup[key] = (score, coverage_rate, support_n, support_pos)

    out = [
        (k[0], k[1], k[2], float(v[0]), float(v[1]), int(v[2]), int(v[3]))
        for k, v in dedup.items()
        if float(v[1]) >= float(min_coverage_rate)
    ]
    out = sorted(out, key=lambda r: r[3], reverse=True)

    if not out:
        return []

    # Default policy: keep top-1 triple. Add top-2 only if it passes strong gate.
    keep: list[tuple[str, str, str, float, float, int, int]] = [out[0]]
    if int(max_triples) >= 2 and len(out) >= 2:
        _, _, _, _, coverage, support_n, support_pos = out[1]
        if (
            float(coverage) >= float(second_triple_min_coverage_rate)
            and int(support_n) >= int(second_triple_min_support_n)
            and int(support_pos) >= int(second_triple_min_support_pos)
        ):
            keep.append(out[1])
    return keep


def add_triple_features(
    X_pair: pd.DataFrame,
    selected_triples: list[tuple[str, str, str, float, float, int, int]],
) -> pd.DataFrame:
    out = X_pair.copy()
    for src_a, src_b, src_c, *_ in selected_triples:
        col_a = f"cont::{src_a}"
        col_b = f"cont::{src_b}"
        col_c = f"cont::{src_c}"
        if col_a not in out.columns or col_b not in out.columns or col_c not in out.columns:
            continue
        out[f"tx::{src_a}__{src_b}__{src_c}"] = (out[col_a] * out[col_b] * out[col_c]).astype(float)
    return out


def _config(execution: dict[str, Any]) -> dict[str, Any]:
    common = execution.get("nonlinear_cont_only_v2", {})
    if not isinstance(common, dict):
        common = {}
    cfg = execution.get("nonlinear_cont_only_3way", {})
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "pair_limit": int(execution.get("pair_limit", common.get("pair_limit", 4))),
        "triple_limit": int(cfg.get("triple_limit", 2)),
        "min_triple_coverage_rate": float(cfg.get("min_triple_coverage_rate", 0.30)),
        "second_triple_min_coverage_rate": float(cfg.get("second_triple_min_coverage_rate", 0.35)),
        "second_triple_min_support_n": int(cfg.get("second_triple_min_support_n", 30)),
        "second_triple_min_support_pos": int(cfg.get("second_triple_min_support_pos", 3)),
        "min_support_n": int(cfg.get("min_support_n", common.get("min_support_n", 10))),
        "min_support_pos": int(cfg.get("min_support_pos", common.get("min_support_pos", 2))),
        "min_enrichment": float(cfg.get("min_enrichment", common.get("min_enrichment", 1.0))),
        "max_rules_per_feature": int(cfg.get("max_rules_per_feature", common.get("max_rules_per_feature", 3))),
        "require_model_cutpoints": bool(cfg.get("require_model_cutpoints", common.get("require_model_cutpoints", True))),
        "allow_physical_fallback": bool(cfg.get("allow_physical_fallback", common.get("allow_physical_fallback", False))),
        "use_joint_pair_interval_search": bool(cfg.get("use_joint_pair_interval_search", common.get("use_joint_pair_interval_search", False))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GBDT mainline_plus_3way on cont_only input")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    parser.add_argument("--force-rebuild-splits", action="store_true", help="Rebuild shared splits even if cached")
    parser.add_argument("--max-splits", type=int, default=0, help="Debug only: cap splits")
    args = parser.parse_args()

    ctx = build_run_context(args.manifest)
    cfg = _config(ctx.execution)
    branch_override = resolve_model_branch_override(ctx.execution, "gbdt", "mainline_plus_3way")
    pair_limit = int(branch_override.get("pair_limit", cfg["pair_limit"]))
    model_params = branch_override.get("model_params", {})
    if not isinstance(model_params, dict):
        model_params = {}
    top_ks = top_ks_from_execution(ctx.execution)
    primary_k = primary_k_from_execution(ctx.execution)

    raw_df, y, input_file = load_cont_only_training_frame(ctx)
    splits, split_cfg = load_or_create_splits(ctx, raw_df, y, force_rebuild=bool(args.force_rebuild_splits))
    if int(args.max_splits) > 0:
        splits = splits[: int(args.max_splits)]

    base_X = build_cont_only_feature_frame(raw_df)
    selected_pairs = select_top_pair_features(base_X, y, max_pairs=pair_limit)
    X_pair = add_pair_features(base_X, selected_pairs)
    selected_triples = select_top_triple_features(
        X_pair,
        y,
        selected_pairs,
        max_triples=int(cfg["triple_limit"]),
        min_coverage_rate=float(cfg["min_triple_coverage_rate"]),
        second_triple_min_coverage_rate=float(cfg["second_triple_min_coverage_rate"]),
        second_triple_min_support_n=int(cfg["second_triple_min_support_n"]),
        second_triple_min_support_pos=int(cfg["second_triple_min_support_pos"]),
    )
    X_3way = add_triple_features(X_pair, selected_triples)

    branch_name = "mainline_plus_3way"
    model_name = "gbdt"
    result = evaluate_branch_with_splits(
        model_name=model_name,
        branch_name=branch_name,
        X=X_3way,
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
        model=result.full_model,
        feature_names=result.feature_columns,
        scaling_stats=result.scaling_stats,
        source_df=raw_df,
    )
    cutpoint_alignment = build_cutpoint_alignment_rows(cut_map, raw_df)

    main_rules = build_main_effect_rules(
        model_name=model_name,
        branch_name=branch_name,
        model=result.full_model,
        feature_names=result.feature_columns,
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
    pair_rules = build_gbdt_interaction_rules(
        model=result.full_model,
        model_name=model_name,
        branch_name=branch_name,
        feature_names=result.feature_columns,
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
    triple_rules = build_gbdt_triple_rules(
        model=result.full_model,
        model_name=model_name,
        branch_name=branch_name,
        feature_names=result.feature_columns,
        source_df=raw_df,
        y=y,
        cut_map=cut_map,
        triple_limit=int(cfg["triple_limit"]),
        min_support_n=int(cfg["min_support_n"]),
        min_support_pos=int(cfg["min_support_pos"]),
        min_enrichment=float(cfg["min_enrichment"]),
        require_model_cutpoints=bool(cfg["require_model_cutpoints"]),
        allow_physical_fallback=bool(cfg["allow_physical_fallback"]),
    )

    if main_rules.empty:
        main_rules = pd.DataFrame(columns=RULE_COLUMNS)
    if pair_rules.empty:
        pair_rules = pd.DataFrame(columns=RULE_COLUMNS)
    if triple_rules.empty:
        triple_rules = pd.DataFrame(columns=RULE_COLUMNS)

    all_frames = [df for df in [main_rules, pair_rules, triple_rules] if not df.empty]
    if all_frames:
        rules_all = pd.concat(all_frames, ignore_index=True)
        rules_all = rules_all.sort_values(
            ["rule_family", "model_score", "enrichment", "support_n"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    else:
        rules_all = pd.DataFrame(columns=RULE_COLUMNS)

    out_dir = model_branch_dir(ctx, model_name, branch_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.metrics_by_split.to_csv(out_dir / "metrics_by_split.csv", index=False)
    result.metrics_ci.to_csv(out_dir / "metrics_topk_ci.csv", index=False)
    result.predictions_oof.to_csv(out_dir / "predictions_oof.csv", index=False)
    main_rules.to_csv(out_dir / "rules_main_effect_realvalue.csv", index=False)
    pair_rules.to_csv(out_dir / "rules_interaction_realvalue.csv", index=False)
    triple_rules.to_csv(out_dir / "rules_interaction_3way_realvalue.csv", index=False)
    rules_all.to_csv(out_dir / "rules_realvalue.csv", index=False)
    cutpoint_alignment.to_csv(out_dir / "model_cutpoint_alignment.csv", index=False)
    result.scaling_stats.to_csv(out_dir / "feature_scaling_stats.csv", index=False)

    pd.DataFrame(
        [{"source_col_a": a, "source_col_b": b, "selection_score_abs_corr": s} for a, b, s in selected_pairs]
    ).to_csv(out_dir / "selected_pair_features.csv", index=False)
    pd.DataFrame(
        [
            {
                "selected_rank": int(idx + 1),
                "source_col_a": a,
                "source_col_b": b,
                "source_col_c": c,
                "selection_score_abs_corr": score,
                "coverage_rate_nonzero": coverage,
                "support_n_nonzero": support_n,
                "support_pos_nonzero": support_pos,
            }
            for idx, (a, b, c, score, coverage, support_n, support_pos) in enumerate(selected_triples)
        ]
    ).to_csv(out_dir / "selected_triple_features.csv", index=False)

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
        "p_at_primary_k_mean": extract_metric_mean(result.metrics_ci, "P", primary_k),
        "enrichment_at_primary_k_mean": extract_metric_mean(result.metrics_ci, "Enrichment", primary_k),
        "feature_profile": feature_profile(X_3way),
        "pair_limit": pair_limit,
        "selected_pair_feature_count": int(len(selected_pairs)),
        "selected_triple_feature_count": int(len(selected_triples)),
        "rule_config": {**cfg, "pair_limit": pair_limit, "model_params": model_params},
        "split_config": split_cfg,
    }
    write_json(out_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
