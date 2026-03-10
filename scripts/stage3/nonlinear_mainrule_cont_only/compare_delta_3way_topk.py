#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from v2_shared import build_run_context, primary_k_from_execution, top_ks_from_execution, write_json


def summarize_ci(values: np.ndarray) -> tuple[float, float, float, int]:
    vals = values[np.isfinite(values)] if len(values) else np.asarray([], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(len(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GBDT 3way vs pairwise Top-K delta")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    args = parser.parse_args()

    ctx = build_run_context(args.manifest)
    top_ks = top_ks_from_execution(ctx.execution)
    primary_k = primary_k_from_execution(ctx.execution)

    pair_path = ctx.out_root / "models" / "gbdt" / "cont_only" / "mainline_plus_pairwise" / "metrics_by_split.csv"
    three_path = ctx.out_root / "models" / "gbdt" / "cont_only" / "mainline_plus_3way" / "metrics_by_split.csv"
    if (not pair_path.exists()) or (not three_path.exists()):
        raise FileNotFoundError(f"Missing metrics file: {pair_path} or {three_path}")

    pair_df = pd.read_csv(pair_path)
    three_df = pd.read_csv(three_path)

    rows_split: list[dict[str, object]] = []
    for metric in ["P", "Enrichment"]:
        for k in top_ks:
            base = pair_df[(pair_df["metric"] == metric) & (pair_df["k"] == int(k))][["split_id", "value"]].rename(columns={"value": "pairwise_value"})
            tri = three_df[(three_df["metric"] == metric) & (three_df["k"] == int(k))][["split_id", "value"]].rename(columns={"value": "triple_value"})
            merged = base.merge(tri, on="split_id", how="inner")
            if merged.empty:
                continue
            merged["delta"] = merged["triple_value"] - merged["pairwise_value"]
            for row in merged.itertuples(index=False):
                rows_split.append(
                    {
                        "metric": metric,
                        "k": int(k),
                        "split_id": int(row.split_id),
                        "pairwise_value": float(row.pairwise_value),
                        "triple_value": float(row.triple_value),
                        "delta": float(row.delta),
                    }
                )

    by_split = pd.DataFrame(rows_split)
    if by_split.empty:
        raise RuntimeError("No delta rows generated for 3way-vs-pairwise")

    rows_summary: list[dict[str, object]] = []
    for (metric, k), grp in by_split.groupby(["metric", "k"], dropna=False):
        mean, low, high, n = summarize_ci(grp["delta"].to_numpy(dtype=float))
        rows_summary.append(
            {
                "metric": str(metric),
                "k": int(k),
                "delta_mean": mean,
                "delta_ci_low_95": low,
                "delta_ci_high_95": high,
                "n_valid_splits": int(n),
                "delta_positive": bool(np.isfinite(mean) and mean > 0),
            }
        )
    summary_df = pd.DataFrame(rows_summary)

    p_row = summary_df[(summary_df["metric"] == "P") & (summary_df["k"] == int(primary_k))]
    e_row = summary_df[(summary_df["metric"] == "Enrichment") & (summary_df["k"] == int(primary_k))]
    p_delta = float(p_row.iloc[0]["delta_mean"]) if not p_row.empty else np.nan
    e_delta = float(e_row.iloc[0]["delta_mean"]) if not e_row.empty else np.nan

    gate_cfg = ctx.execution.get("nonlinear_cont_only_3way", {})
    if not isinstance(gate_cfg, dict):
        gate_cfg = {}
    min_dp = float(gate_cfg.get("min_delta_p_for_3way", 0.005))
    min_de = float(gate_cfg.get("min_delta_enrichment_for_3way", 0.02))

    decision = {
        "model": "gbdt",
        "primary_k": int(primary_k),
        "delta_p_at_k_mean": p_delta,
        "delta_enrichment_at_k_mean": e_delta,
        "threeway_gain_positive": bool(np.isfinite(p_delta) and np.isfinite(e_delta) and p_delta > 0 and e_delta > 0),
        "threeway_gain_recommended": bool(np.isfinite(p_delta) and np.isfinite(e_delta) and p_delta >= min_dp and e_delta >= min_de),
        "recommended_threshold": {
            "min_delta_p_for_3way": min_dp,
            "min_delta_enrichment_for_3way": min_de,
        },
    }

    by_split.to_csv(ctx.out_root / "delta_3way_vs_pair_topk_by_split.csv", index=False)
    summary_df.to_csv(ctx.out_root / "delta_3way_vs_pair_topk_summary.csv", index=False)
    pd.DataFrame([decision]).to_csv(ctx.out_root / "interaction_gain_3way_decision.csv", index=False)

    out = {
        "output_dir": str(ctx.out_root),
        "primary_k": int(primary_k),
        "delta_summary_rows": int(len(summary_df)),
        "decision": decision,
    }
    write_json(ctx.out_root / "delta_3way_run_summary.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
