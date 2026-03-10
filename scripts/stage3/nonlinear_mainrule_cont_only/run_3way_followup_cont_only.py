#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from v2_shared import build_run_context, write_json

SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(script_name: str, manifest: str, max_splits: int = 0) -> None:
    cmd = ["python3", str((SCRIPT_DIR / script_name).resolve()), "--manifest", manifest]
    if int(max_splits) > 0 and script_name.startswith("run_"):
        cmd.extend(["--max-splits", str(int(max_splits))])
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3way follow-up for cont_only nonlinear pipeline")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    parser.add_argument("--max-splits", type=int, default=0, help="Debug only: cap splits for 3way run")
    parser.add_argument("--force-run", action="store_true", help="Force run 3way even when gate check fails")
    args = parser.parse_args()

    ctx = build_run_context(args.manifest)
    gate_cfg = ctx.execution.get("nonlinear_cont_only_3way", {})
    if not isinstance(gate_cfg, dict):
        gate_cfg = {}

    enforce_gate_b = bool(gate_cfg.get("enforce_gate_b", False))
    min_dp = float(gate_cfg.get("min_delta_p_for_3way", 0.005))
    min_de = float(gate_cfg.get("min_delta_enrichment_for_3way", 0.02))

    decision_file = ctx.out_root / "interaction_gain_decision.csv"
    if not decision_file.exists():
        raise FileNotFoundError(
            f"Missing pairwise decision file: {decision_file}. "
            "Please run run_nonlinear_cont_only_suite_v2.py first."
        )

    decision_df = pd.read_csv(decision_file)
    gbdt = decision_df[decision_df["model"].astype(str) == "gbdt"]
    if gbdt.empty:
        raise RuntimeError("No gbdt row in interaction_gain_decision.csv")

    row = gbdt.iloc[0]
    gate_a_pass = bool(row.get("interaction_gain_positive", False))
    dp = float(row.get("delta_p_at_k_mean", float("nan")))
    de = float(row.get("delta_enrichment_at_k_mean", float("nan")))
    gate_b_pass = bool((dp >= min_dp) and (de >= min_de))

    should_run = bool(args.force_run) or (gate_a_pass and ((not enforce_gate_b) or gate_b_pass))

    followup_plan = {
        "gate_a_pairwise_positive": gate_a_pass,
        "gate_b_threshold": {
            "min_delta_p_for_3way": min_dp,
            "min_delta_enrichment_for_3way": min_de,
            "enforce_gate_b": enforce_gate_b,
            "gate_b_pass": gate_b_pass,
        },
        "force_run": bool(args.force_run),
        "should_run": should_run,
        "source_pairwise_delta": {
            "delta_p_at_k_mean": dp,
            "delta_enrichment_at_k_mean": de,
        },
    }

    if should_run:
        run_step("run_gbdt_3way_cont_only.py", args.manifest, max_splits=int(args.max_splits))
        run_step("compare_delta_3way_topk.py", args.manifest)
        run_step("publish_realvalue_rules_3way.py", args.manifest)
        followup_plan["status"] = "executed"
    else:
        followup_plan["status"] = "skipped_by_gate"

    write_json(ctx.out_root / "threeway_followup_plan.json", followup_plan)
    print(json.dumps(followup_plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
