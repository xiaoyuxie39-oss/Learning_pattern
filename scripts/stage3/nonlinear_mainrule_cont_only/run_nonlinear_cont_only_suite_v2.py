#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_step(script_name: str, manifest: str, max_splits: int = 0) -> None:
    cmd = ["python3", str((SCRIPT_DIR / script_name).resolve()), "--manifest", manifest]
    if int(max_splits) > 0 and script_name.startswith("run_") and script_name.endswith("_cont_only.py"):
        cmd.extend(["--max-splits", str(int(max_splits))])
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent nonlinear cont_only v2 suite (no dependency on 02_model_execution_and_audit.py)")
    parser.add_argument("--manifest", required=True, help="Path to run manifest yaml")
    parser.add_argument("--max-splits", type=int, default=0, help="Debug only: cap splits for model scripts")
    args = parser.parse_args()

    steps = [
        "run_ebm_mainline_cont_only.py",
        "run_ebm_pairwise_cont_only.py",
        "run_gbdt_mainline_cont_only.py",
        "run_gbdt_pairwise_cont_only.py",
        "compare_delta_topk.py",
        "publish_realvalue_rules.py",
    ]

    for step in steps:
        run_step(step, args.manifest, max_splits=int(args.max_splits))

    summary = {
        "manifest": str(args.manifest),
        "suite": "nonlinear_cont_only_v2",
        "steps": steps,
        "max_splits": int(args.max_splits),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
