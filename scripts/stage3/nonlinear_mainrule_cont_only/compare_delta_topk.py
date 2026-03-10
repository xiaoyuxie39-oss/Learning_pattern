#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from v2_shared import (
    build_run_context,
    primary_k_from_execution,
    top_ks_from_execution,
    write_json,
)


def summarize_ci(values: np.ndarray) -> tuple[float, float, float, int]:
    vals = values[np.isfinite(values)] if len(values) else np.asarray([], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(len(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pairwise vs mainline Top-K delta for nonlinear cont_only v2")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    args = parser.parse_args()

    ctx = build_run_context(args.manifest)
    top_ks = top_ks_from_execution(ctx.execution)
    primary_k = primary_k_from_execution(ctx.execution)

    models = ["ebm", "gbdt"]
    by_split_rows: list[dict[str, object]] = []

    for model in models:
        main_path = ctx.out_root / "models" / model / "cont_only" / "mainline" / "metrics_by_split.csv"
        pair_path = ctx.out_root / "models" / model / "cont_only" / "mainline_plus_pairwise" / "metrics_by_split.csv"
        if (not main_path.exists()) or (not pair_path.exists()):
            raise FileNotFoundError(f"Missing metrics for {model}: {main_path} or {pair_path}")

        main_df = pd.read_csv(main_path)
        pair_df = pd.read_csv(pair_path)

        for metric in ["P", "Enrichment"]:
            for k in top_ks:
                base = main_df[(main_df["metric"] == metric) & (main_df["k"] == int(k))][["split_id", "value"]].rename(columns={"value": "base_value"})
                pair = pair_df[(pair_df["metric"] == metric) & (pair_df["k"] == int(k))][["split_id", "value"]].rename(columns={"value": "pair_value"})
                merged = base.merge(pair, on="split_id", how="inner")
                if merged.empty:
                    continue
                merged["delta"] = merged["pair_value"] - merged["base_value"]
                for row in merged.itertuples(index=False):
                    by_split_rows.append(
                        {
                            "model": model,
                            "metric": metric,
                            "k": int(k),
                            "split_id": int(row.split_id),
                            "mainline_value": float(row.base_value),
                            "pairwise_value": float(row.pair_value),
                            "delta": float(row.delta),
                        }
                    )

    delta_by_split = pd.DataFrame(by_split_rows)
    if delta_by_split.empty:
        raise RuntimeError("No delta rows generated. Check model outputs.")

    summary_rows: list[dict[str, object]] = []
    for (model, metric, k), grp in delta_by_split.groupby(["model", "metric", "k"], dropna=False):
        mean, low, high, n = summarize_ci(grp["delta"].to_numpy(dtype=float))
        summary_rows.append(
            {
                "model": str(model),
                "metric": str(metric),
                "k": int(k),
                "delta_mean": mean,
                "delta_ci_low_95": low,
                "delta_ci_high_95": high,
                "n_valid_splits": int(n),
                "delta_positive": bool(np.isfinite(mean) and mean > 0),
            }
        )
    delta_summary = pd.DataFrame(summary_rows)

    decision_rows: list[dict[str, object]] = []
    for model in models:
        p_row = delta_summary[(delta_summary["model"] == model) & (delta_summary["metric"] == "P") & (delta_summary["k"] == int(primary_k))]
        e_row = delta_summary[(delta_summary["model"] == model) & (delta_summary["metric"] == "Enrichment") & (delta_summary["k"] == int(primary_k))]
        p_delta = float(p_row.iloc[0]["delta_mean"]) if not p_row.empty else np.nan
        e_delta = float(e_row.iloc[0]["delta_mean"]) if not e_row.empty else np.nan
        gain_positive = bool(np.isfinite(p_delta) and np.isfinite(e_delta) and p_delta > 0 and e_delta > 0)
        decision_rows.append(
            {
                "model": model,
                "primary_k": int(primary_k),
                "delta_p_at_k_mean": p_delta,
                "delta_enrichment_at_k_mean": e_delta,
                "interaction_gain_positive": gain_positive,
                "rule": "delta_mean(P@K)>0 and delta_mean(Enrichment@K)>0",
            }
        )
    decision_df = pd.DataFrame(decision_rows)

    delta_by_split.to_csv(ctx.out_root / "delta_topk_by_split.csv", index=False)
    delta_summary.to_csv(ctx.out_root / "delta_topk_summary.csv", index=False)
    decision_df.to_csv(ctx.out_root / "interaction_gain_decision.csv", index=False)

    summary = {
        "output_dir": str(ctx.out_root),
        "primary_k": int(primary_k),
        "models": models,
        "delta_summary_rows": int(len(delta_summary)),
        "decision_rows": int(len(decision_df)),
    }
    write_json(ctx.out_root / "delta_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
