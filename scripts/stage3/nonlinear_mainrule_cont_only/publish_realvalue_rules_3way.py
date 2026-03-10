#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from v2_shared import build_run_context, primary_k_from_execution, write_json


def _load_metric_mean(path: str, metric: str, k: int) -> float:
    df = pd.read_csv(path)
    sub = df[(df["metric"] == str(metric)) & (df["k"] == int(k))]
    if sub.empty:
        return np.nan
    return float(sub.iloc[0]["mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish 3way real-value rules (GBDT cont_only)")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    args = parser.parse_args()

    ctx = build_run_context(args.manifest)
    primary_k = primary_k_from_execution(ctx.execution)

    decision_file = ctx.out_root / "interaction_gain_3way_decision.csv"
    if not decision_file.exists():
        raise FileNotFoundError(f"Missing 3way decision file: {decision_file}")
    decision = pd.read_csv(decision_file)
    if decision.empty:
        raise RuntimeError("interaction_gain_3way_decision.csv is empty")

    d = decision.iloc[0]
    gain_positive = bool(d.get("threeway_gain_positive", False))
    gain_recommended = bool(d.get("threeway_gain_recommended", False))

    pair_metrics = ctx.out_root / "models" / "gbdt" / "cont_only" / "mainline_plus_pairwise" / "metrics_topk_ci.csv"
    three_metrics = ctx.out_root / "models" / "gbdt" / "cont_only" / "mainline_plus_3way" / "metrics_topk_ci.csv"
    rules_3way = ctx.out_root / "models" / "gbdt" / "cont_only" / "mainline_plus_3way" / "rules_interaction_3way_realvalue.csv"

    if not rules_3way.exists():
        raise FileNotFoundError(f"Missing 3way rules file: {rules_3way}")

    df_3way = pd.read_csv(rules_3way)
    if gain_positive and not df_3way.empty:
        publish_df = df_3way.copy()
        publish_df["publish_reason"] = "threeway_gain_positive"
    else:
        publish_df = pd.DataFrame(columns=list(df_3way.columns) + ["publish_reason"])

    publish_path = ctx.out_root / "rules_3way_publish_realvalue.csv"
    publish_df.to_csv(publish_path, index=False)

    p_pair = _load_metric_mean(str(pair_metrics), "P", int(primary_k)) if pair_metrics.exists() else np.nan
    e_pair = _load_metric_mean(str(pair_metrics), "Enrichment", int(primary_k)) if pair_metrics.exists() else np.nan
    p_three = _load_metric_mean(str(three_metrics), "P", int(primary_k)) if three_metrics.exists() else np.nan
    e_three = _load_metric_mean(str(three_metrics), "Enrichment", int(primary_k)) if three_metrics.exists() else np.nan

    brief_lines = [
        "# 3way Follow-up Decision Brief",
        "",
        f"- primary_k: {int(primary_k)}",
        "- comparison: gbdt mainline_plus_3way vs gbdt mainline_plus_pairwise",
        "",
        "## Top-K Summary",
        f"- pairwise: P@{primary_k}={p_pair:.4f} | Enrichment@{primary_k}={e_pair:.4f}",
        f"- 3way: P@{primary_k}={p_three:.4f} | Enrichment@{primary_k}={e_three:.4f}",
        f"- delta: dP={float(d.get('delta_p_at_k_mean', np.nan)):.4f} | dEnrichment={float(d.get('delta_enrichment_at_k_mean', np.nan)):.4f}",
        f"- threeway_gain_positive: {str(gain_positive).lower()}",
        f"- threeway_gain_recommended: {str(gain_recommended).lower()}",
        "",
        "## Publish",
        f"- published_3way_rule_count: {int(len(publish_df))}",
    ]
    (ctx.out_root / "run_decision_brief_3way.md").write_text("\n".join(brief_lines).rstrip() + "\n", encoding="utf-8")

    summary = {
        "output_dir": str(ctx.out_root),
        "primary_k": int(primary_k),
        "threeway_gain_positive": gain_positive,
        "threeway_gain_recommended": gain_recommended,
        "published_3way_rule_count": int(len(publish_df)),
        "published_file": str(publish_path),
    }
    write_json(ctx.out_root / "publish_3way_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
