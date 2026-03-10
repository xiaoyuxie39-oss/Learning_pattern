#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Any

STEP_SEQUENCE = [
    ("step01_features", "prepare_base_features", "year_to_continuous", "stabilize_design_matrices"),
    ("step02_candidates", "generate_pairwise_candidates", "generate_triple_candidates", "classify_candidates"),
    ("step03_models", "make_estimator", "fit_with_warning_capture", "evaluate_model_branch"),
    ("step04_audits", "negative_control_audit", "tier_stability_audit", "candidate_consistency_audit"),
    ("step05_reporting", "render_release_rule", "build_rulebook_support_from_candidates", "build_artifact_completeness_report"),
]


def ensure_import_paths() -> None:
    here = Path(__file__).resolve().parent
    stage3_dir = here.parent
    root = repo_root()
    for path in [str(here), str(stage3_dir), str(root)]:
        if path not in sys.path:
            sys.path.insert(0, path)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def stage3_script_path() -> Path:
    return repo_root() / "scripts" / "stage3" / "02_model_execution_and_audit.py"


def validate_workflow_modules() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    module_dir = Path(__file__).resolve().parent
    for index, (module_name, *required_exports) in enumerate(STEP_SEQUENCE, start=1):
        module_path = module_dir / f"{module_name}.py"
        if not module_path.exists():
            rows.append(
                {
                    "order": index,
                    "module": module_name,
                    "required_exports": list(required_exports),
                    "missing_exports": list(required_exports),
                    "ok": False,
                    "error": f"module_file_missing:{module_path}",
                }
            )
            continue
        try:
            tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
        except Exception as exc:
            rows.append(
                {
                    "order": index,
                    "module": module_name,
                    "required_exports": list(required_exports),
                    "missing_exports": list(required_exports),
                    "ok": False,
                    "error": f"parse_error:{exc}",
                }
            )
            continue
        defined_symbols = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        missing = [name for name in required_exports if name not in defined_symbols]
        rows.append(
            {
                "order": index,
                "module": module_name,
                "required_exports": list(required_exports),
                "missing_exports": missing,
                "ok": len(missing) == 0,
            }
        )
    return rows


def print_validation(rows: list[dict[str, Any]]) -> None:
    print("[stage3-workflow] ordered module validation")
    for row in rows:
        status = "OK" if row["ok"] else "FAIL"
        print(f"- step{row['order']:02d} {row['module']}: {status}")
        if row["missing_exports"]:
            print(f"  missing: {', '.join(row['missing_exports'])}")


def run_stage3_pipeline(manifest: str) -> int:
    script = stage3_script_path()
    if not script.exists():
        print(f"[stage3-workflow] missing execution script: {script}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(script), "--manifest", manifest]
    print("[stage3-workflow] executing:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage3 compatibility entrypoint for the ordered helper modules")
    parser.add_argument("--manifest", required=True, help="Path to run_manifest.yaml (repo relative or absolute)")
    parser.add_argument("--check-only", action="store_true", help="Only validate ordered workflow modules")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_import_paths()
    rows = validate_workflow_modules()
    print_validation(rows)
    if not all(row["ok"] for row in rows):
        raise SystemExit(2)
    if args.check_only:
        print("[stage3-workflow] check-only completed")
        return
    code = run_stage3_pipeline(args.manifest)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
