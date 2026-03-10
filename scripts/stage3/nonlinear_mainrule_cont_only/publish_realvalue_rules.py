#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from v2_shared import build_run_context, primary_k_from_execution, write_json


def _load_metric_mean(path: pd.PathLike[str] | str, metric: str, k: int) -> float:
    p = pd.read_csv(path)
    sub = p[(p["metric"] == str(metric)) & (p["k"] == int(k))]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish compact real-value rulebook for nonlinear cont_only v2")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    args = parser.parse_args()

    ctx = build_run_context(args.manifest)
    primary_k = primary_k_from_execution(ctx.execution)

    decision_file = ctx.out_root / "interaction_gain_decision.csv"
    if not decision_file.exists():
        raise FileNotFoundError(f"Missing interaction decision file: {decision_file}")
    decisions = pd.read_csv(decision_file)
    publish_cfg = ctx.execution.get("nonlinear_cont_only_v2_publish", {})
    if not isinstance(publish_cfg, dict):
        publish_cfg = {}
    allow_rule_level_publish_when_gain_negative = bool(
        publish_cfg.get("allow_pairwise_rule_level_when_gain_negative", False)
    )
    rule_level_models_when_gain_negative = {
        str(v)
        for v in list(publish_cfg.get("rule_level_models_when_gain_negative", ["ebm"]))
    }
    explanation_only_models = {str(v) for v in list(publish_cfg.get("explanation_only_models", []))}

    models = ["ebm", "gbdt"]
    rows: list[dict[str, object]] = []
    brief_lines: list[str] = [
        "# Nonlinear cont_only v2 Decision Brief",
        "",
        f"- primary_k: {int(primary_k)}",
        "- interaction pass rule: delta_mean(P@K)>0 and delta_mean(Enrichment@K)>0",
        "",
        "## Model Summary",
    ]

    for model in models:
        main_dir = ctx.out_root / "models" / model / "cont_only" / "mainline"
        pair_dir = ctx.out_root / "models" / model / "cont_only" / "mainline_plus_pairwise"
        main_rules_file = main_dir / "rules_main_effect_realvalue.csv"
        pair_rules_file = pair_dir / "rules_interaction_realvalue.csv"
        main_metrics_file = main_dir / "metrics_topk_ci.csv"
        pair_metrics_file = pair_dir / "metrics_topk_ci.csv"

        if not main_rules_file.exists():
            raise FileNotFoundError(f"Missing mainline rules for {model}: {main_rules_file}")
        if not pair_rules_file.exists():
            raise FileNotFoundError(f"Missing pairwise rules for {model}: {pair_rules_file}")

        main_rules = pd.read_csv(main_rules_file)
        pair_rules = pd.read_csv(pair_rules_file)

        decision_row = decisions[decisions["model"].astype(str) == model]
        gain_positive = bool(decision_row.iloc[0]["interaction_gain_positive"]) if not decision_row.empty else False
        delta_p = float(decision_row.iloc[0]["delta_p_at_k_mean"]) if not decision_row.empty else np.nan
        delta_e = float(decision_row.iloc[0]["delta_enrichment_at_k_mean"]) if not decision_row.empty else np.nan
        explanation_only = model in explanation_only_models

        p_main = _load_metric_mean(main_metrics_file, "P", int(primary_k)) if main_metrics_file.exists() else np.nan
        e_main = _load_metric_mean(main_metrics_file, "Enrichment", int(primary_k)) if main_metrics_file.exists() else np.nan
        p_pair = _load_metric_mean(pair_metrics_file, "P", int(primary_k)) if pair_metrics_file.exists() else np.nan
        e_pair = _load_metric_mean(pair_metrics_file, "Enrichment", int(primary_k)) if pair_metrics_file.exists() else np.nan

        if explanation_only:
            interaction_mode = "explanation_only"
        elif gain_positive:
            interaction_mode = "predictive_gain"
        else:
            interaction_mode = "mainline_only"

        brief_lines.extend(
            [
                f"### {model}",
                f"- mainline: P@{primary_k}={p_main:.4f} | Enrichment@{primary_k}={e_main:.4f}",
                f"- pairwise: P@{primary_k}={p_pair:.4f} | Enrichment@{primary_k}={e_pair:.4f}",
                f"- delta: dP={delta_p:.4f} | dEnrichment={delta_e:.4f}",
                f"- interaction_gain_positive: {str(gain_positive).lower()}",
                f"- interaction_mode: {interaction_mode}",
                "",
            ]
        )

        if not main_rules.empty:
            for r in main_rules.itertuples(index=False):
                rows.append(
                    {
                        "model": model,
                        "branch": "mainline",
                        "publish_scope": "always_mainline",
                        "rule_family": str(getattr(r, "rule_family", "main_effect")),
                        "feature_a": str(getattr(r, "feature_a", "")),
                        "feature_b": str(getattr(r, "feature_b", "")),
                        "condition_text": str(getattr(r, "condition_text", "")),
                        "support_n": float(getattr(r, "support_n", np.nan)),
                        "enrichment": float(getattr(r, "enrichment", np.nan)),
                        "model_score": float(getattr(r, "model_score", np.nan)),
                        "rule_source": str(getattr(r, "rule_source", "")),
                    }
                )

        if explanation_only and (not pair_rules.empty):
            for r in pair_rules.itertuples(index=False):
                rows.append(
                    {
                        "model": model,
                        "branch": "mainline_plus_pairwise",
                        "publish_scope": "pairwise_explanation_only",
                        "rule_family": str(getattr(r, "rule_family", "interaction")),
                        "feature_a": str(getattr(r, "feature_a", "")),
                        "feature_b": str(getattr(r, "feature_b", "")),
                        "condition_text": str(getattr(r, "condition_text", "")),
                        "support_n": float(getattr(r, "support_n", np.nan)),
                        "enrichment": float(getattr(r, "enrichment", np.nan)),
                        "model_score": float(getattr(r, "model_score", np.nan)),
                        "rule_source": str(getattr(r, "rule_source", "")),
                    }
                )
        elif gain_positive and (not pair_rules.empty):
            for r in pair_rules.itertuples(index=False):
                rows.append(
                    {
                        "model": model,
                        "branch": "mainline_plus_pairwise",
                        "publish_scope": "pairwise_only_if_gain_positive",
                        "rule_family": str(getattr(r, "rule_family", "interaction")),
                        "feature_a": str(getattr(r, "feature_a", "")),
                        "feature_b": str(getattr(r, "feature_b", "")),
                        "condition_text": str(getattr(r, "condition_text", "")),
                        "support_n": float(getattr(r, "support_n", np.nan)),
                        "enrichment": float(getattr(r, "enrichment", np.nan)),
                        "model_score": float(getattr(r, "model_score", np.nan)),
                        "rule_source": str(getattr(r, "rule_source", "")),
                    }
                )
        elif (
            allow_rule_level_publish_when_gain_negative
            and (model in rule_level_models_when_gain_negative)
            and (not pair_rules.empty)
        ):
            for r in pair_rules.itertuples(index=False):
                rows.append(
                    {
                        "model": model,
                        "branch": "mainline_plus_pairwise",
                        "publish_scope": "pairwise_rule_level_even_if_model_gain_negative",
                        "rule_family": str(getattr(r, "rule_family", "interaction")),
                        "feature_a": str(getattr(r, "feature_a", "")),
                        "feature_b": str(getattr(r, "feature_b", "")),
                        "condition_text": str(getattr(r, "condition_text", "")),
                        "support_n": float(getattr(r, "support_n", np.nan)),
                        "enrichment": float(getattr(r, "enrichment", np.nan)),
                        "model_score": float(getattr(r, "model_score", np.nan)),
                        "rule_source": str(getattr(r, "rule_source", "")),
                    }
                )

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(
            ["model", "publish_scope", "rule_family", "model_score", "enrichment", "support_n"],
            ascending=[True, True, True, False, False, False],
        ).reset_index(drop=True)

    out_df.to_csv(ctx.out_root / "rules_publish_realvalue.csv", index=False)
    (ctx.out_root / "run_decision_brief.md").write_text("\n".join(brief_lines).rstrip() + "\n", encoding="utf-8")

    summary = {
        "output_dir": str(ctx.out_root),
        "primary_k": int(primary_k),
        "published_rule_count": int(len(out_df)),
        "published_file": str(ctx.out_root / "rules_publish_realvalue.csv"),
        "brief_file": str(ctx.out_root / "run_decision_brief.md"),
    }
    write_json(ctx.out_root / "publish_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
