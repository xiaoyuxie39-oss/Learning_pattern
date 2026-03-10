#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE3_DIR = REPO_ROOT / "scripts" / "stage3"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from common import append_log, ensure_dir, load_simple_yaml, read_json, resolve_repo_path  # noqa: E402
from stage3_workflow_isolated import step01_features as stage3_features  # noqa: E402
from stage3_workflow_isolated import step02_candidates as stage3_candidates  # noqa: E402
from stage3_workflow_isolated import step03_models as stage3_models  # noqa: E402


REFERENCE_NONLINEAR_MODELS = ("ebm", "gbdt")
LINEAR_CONTROL_MODELS = ("logistic_l2", "elasticnet")
KEY_ALIGNMENT_COLUMNS = (
    "id",
    "company",
    "coverage_tier",
    "llm_ai_dc_label",
    "accel_model",
    "accel_count",
    "base_non_missing_count",
)
MODEL_CONFIGS = (
    {"name": "logistic_l2", "pair_limit": 2},
    {"name": "elasticnet", "pair_limit": 2},
    {"name": "ebm", "pair_limit": 3},
    {"name": "gbdt", "pair_limit": 2},
)


def ensure_file(path: Path, content: str = "") -> None:
    ensure_dir(path.parent)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def load_stage3_part2_module() -> Any:
    module_path = STAGE3_DIR / "02_model_execution_and_audit.py"
    spec = importlib.util.spec_from_file_location("stage3_part2_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_default_out_root(mainline_root: Path) -> Path:
    return REPO_ROOT / "artifacts" / f"stage3_bin_only_control_against_{mainline_root.name}_{timestamp_slug()}"


def load_reference_splits(split_indices_path: Path) -> list[dict[str, Any]]:
    raw = read_json(split_indices_path)
    splits: list[dict[str, Any]] = []
    for item in raw:
        splits.append(
            {
                "split_id": int(item["split_id"]),
                "repeat_id": int(item.get("repeat_id", item["split_id"])),
                "train_idx": np.asarray(item["train_idx"], dtype=int),
                "test_idx": np.asarray(item["test_idx"], dtype=int),
            }
        )
    return splits


def assert_trainable_alignment(cont_df: pd.DataFrame, bin_df: pd.DataFrame) -> dict[str, Any]:
    cont_trainable = cont_df[cont_df["base_non_missing_count"] >= 2].reset_index(drop=True)
    bin_trainable = bin_df[bin_df["base_non_missing_count"] >= 2].reset_index(drop=True)
    if len(cont_trainable) != len(bin_trainable):
        raise RuntimeError(
            f"Trainable row count mismatch: cont_only={len(cont_trainable)} bin_only={len(bin_trainable)}"
        )
    checks: dict[str, bool] = {}
    for col in KEY_ALIGNMENT_COLUMNS:
        if col not in cont_trainable.columns or col not in bin_trainable.columns:
            continue
        checks[col] = cont_trainable[col].astype(str).equals(bin_trainable[col].astype(str))
    failed = [col for col, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Trainable row order mismatch for columns: {failed}")
    return {
        "cont_rows": int(len(cont_df)),
        "bin_rows": int(len(bin_df)),
        "cont_trainable_rows": int(len(cont_trainable)),
        "bin_trainable_rows": int(len(bin_trainable)),
        "checked_columns": list(checks.keys()),
        "all_checked_columns_match": all(checks.values()) if checks else False,
    }


def patch_aligned_estimators() -> Any:
    original = stage3_models.make_estimator

    def aligned_make_estimator(
        model_kind: str,
        random_seed: int,
        *,
        branch_name: str = "mainline",
        pair_limit: int = 4,
        triple_limit: int = 0,
    ) -> Any:
        if model_kind == "gbdt":
            default_max_depth = 2 if branch_name == "mainline" else 3
            default_max_iter = 220 if branch_name == "mainline" else 320
            return stage3_models.HistGradientBoostingClassifier(
                max_depth=default_max_depth,
                learning_rate=0.05,
                max_iter=default_max_iter,
                l2_regularization=1.0,
                min_samples_leaf=10,
                random_state=random_seed,
            )
        if model_kind == "ebm":
            if stage3_models.ExplainableBoostingClassifier is None:
                raise RuntimeError("interpret_not_available")
            interaction_budget = 0 if branch_name == "mainline" else min(max(1, int(pair_limit)), 8)
            return stage3_models.ExplainableBoostingClassifier(
                interactions=interaction_budget,
                max_bins=64,
                max_rounds=300,
                learning_rate=0.03,
                outer_bags=4,
                inner_bags=0,
                n_jobs=1,
                random_state=random_seed,
            )
        return original(
            model_kind,
            random_seed,
            branch_name=branch_name,
            pair_limit=pair_limit,
            triple_limit=triple_limit,
        )

    stage3_models.make_estimator = aligned_make_estimator
    return original


def copy_reference_split_bundle(src_dir: Path, dst_dir: Path, metadata: dict[str, Any]) -> None:
    ensure_dir(dst_dir)
    for name in ("company_holdout_splits.csv", "company_holdout_splits_meta.json", "split_indices.json"):
        shutil.copy2(src_dir / name, dst_dir / name)
    (dst_dir / "split_reference.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def recompute_selection_flags(selection_df: pd.DataFrame, stage3_part2: Any) -> pd.DataFrame:
    if selection_df.empty:
        return selection_df
    selection_df = selection_df.copy()
    selection_df["ci_evidence_ok"] = (
        (selection_df["delta_p20_ci_low"].fillna(-np.inf) >= 0)
        & (selection_df["delta_enrichment20_ci_low"].fillna(-np.inf) >= 0)
    )
    selection_df["status_completed"] = selection_df["status"].astype(str).eq("COMPLETED")
    selection_df["warning_free"] = ~selection_df["warning_present"].fillna(True)
    selection_df["tier2d_level_rank"] = selection_df["tier2d_level"].map({"PASS": 2, "WARN": 1, "FAIL": 0}).fillna(0).astype(int)
    selection_df["primary_sort"] = list(
        zip(
            selection_df["status_completed"].astype(int),
            selection_df["hard_audit_pass_count"],
            selection_df["rulebook_reproducible"].astype(int),
            selection_df["warning_free"].astype(int),
            selection_df["interaction_upgrade_eligible"].astype(int),
            selection_df["ci_evidence_ok"].astype(int),
            selection_df["p20_ci_low"].fillna(-9999.0),
            selection_df["p20_mean"].fillna(-9999.0),
            selection_df["enrichment20_ci_low"].fillna(-9999.0),
            selection_df["enrichment20_mean"].fillna(-9999.0),
            selection_df["auc_ci_low"].fillna(-9999.0),
            selection_df["auc_mean"].fillna(-9999.0),
            selection_df["delta_p20_ci_low"].fillna(-9999.0),
            selection_df["delta_enrichment20_ci_low"].fillna(-9999.0),
            selection_df["tier2d_level_rank"],
            selection_df["tier2d_status"].astype(str).eq("PASS").astype(int),
        )
    )
    selection_df = selection_df.sort_values(
        [
            "status_completed",
            "hard_audit_pass_count",
            "rulebook_reproducible",
            "warning_free",
            "interaction_upgrade_eligible",
            "ci_evidence_ok",
            "p20_ci_low",
            "p20_mean",
            "enrichment20_ci_low",
            "enrichment20_mean",
            "auc_ci_low",
            "auc_mean",
            "delta_p20_ci_low",
            "delta_enrichment20_ci_low",
            "tier2d_level_rank",
            "tier2d_status",
        ],
        ascending=[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    selection_df["is_branch_winner"] = False
    selection_df["is_primary_winner"] = False
    selection_df["is_control_winner"] = False

    branch_subset = selection_df[selection_df["branch"].isin([stage3_part2.MAINLINE_BRANCH, stage3_part2.PAIRWISE_BRANCH])].copy()
    if not branch_subset.empty:
        for _, grp in branch_subset.groupby(["model", "feature_mode"], sort=False):
            selection_df.loc[int(grp.index[0]), "is_branch_winner"] = True

    control_candidates: list[int] = []
    for model_name in sorted(selection_df[selection_df["feature_mode"] == stage3_part2.CONTROL_WINNER_FEATURE_MODE]["model"].astype(str).unique().tolist()):
        idx = stage3_part2.pick_model_internal_winner(
            selection_df,
            model_name=model_name,
            feature_mode=stage3_part2.CONTROL_WINNER_FEATURE_MODE,
            allow_pair_override=True,
        )
        if idx is not None:
            control_candidates.append(int(idx))
    control_candidates_ci = [idx for idx in control_candidates if bool(selection_df.loc[idx, "ci_evidence_ok"])]
    control_idx = stage3_part2.first_ranked_index(selection_df, control_candidates_ci or control_candidates)
    if control_idx is not None:
        selection_df.loc[control_idx, "is_control_winner"] = True

    selection_df["winner_role"] = "none"
    selection_df.loc[selection_df["is_branch_winner"], "winner_role"] = "branch_winner"
    selection_df.loc[selection_df["is_control_winner"], "winner_role"] = "control_winner"
    selection_df["selected_as_primary"] = False
    return selection_df


def csv_is_empty(path: Path) -> bool:
    try:
        return pd.read_csv(path).empty
    except Exception:
        return False


def prune_outputs(out_root: Path) -> list[str]:
    removed: list[str] = []
    for path in sorted(out_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".en.md"):
            path.unlink()
            removed.append(str(path.relative_to(REPO_ROOT)))
            continue
        if path.suffix.lower() == ".csv" and csv_is_empty(path):
            path.unlink()
            removed.append(str(path.relative_to(REPO_ROOT)))
    return removed


def load_topk_metric(metrics_path: Path, metric: str, k: int) -> tuple[float, float]:
    if not metrics_path.exists():
        return np.nan, np.nan
    df = pd.read_csv(metrics_path)
    row = df[(df["metric"].astype(str) == metric) & (df["k"].astype(int) == int(k))]
    if row.empty:
        return np.nan, np.nan
    first = row.iloc[0]
    return float(first["mean"]), float(first["ci_low_95"])


def load_cont_pair_delta(delta_path: Path, model: str) -> dict[str, float]:
    if not delta_path.exists():
        return {"delta_p20_mean": np.nan, "delta_p20_ci_low": np.nan, "delta_enrichment20_mean": np.nan, "delta_enrichment20_ci_low": np.nan}
    df = pd.read_csv(delta_path)
    out: dict[str, float] = {}
    for metric, key_prefix in (("P", "delta_p20"), ("Enrichment", "delta_enrichment20")):
        row = df[(df["model"].astype(str) == model) & (df["metric"].astype(str) == metric) & (df["k"].astype(int) == 20)]
        if row.empty:
            out[f"{key_prefix}_mean"] = np.nan
            out[f"{key_prefix}_ci_low"] = np.nan
        else:
            out[f"{key_prefix}_mean"] = float(row.iloc[0]["delta_mean"])
            out[f"{key_prefix}_ci_low"] = float(row.iloc[0]["delta_ci_low_95"])
    return out


def summarize_top_pairs(pair_file: Path, limit: int = 3) -> list[str]:
    if not pair_file.exists():
        return []
    df = pd.read_csv(pair_file)
    if df.empty:
        return []
    cols = [col for col in ("source_col_a", "source_col_b", "feature_a", "feature_b") if col in df.columns]
    if "source_col_a" in df.columns and "source_col_b" in df.columns:
        return [f"{row.source_col_a} x {row.source_col_b}" for row in df.head(limit).itertuples(index=False)]
    if "feature_a" in df.columns and "feature_b" in df.columns:
        return [f"{row.feature_a} x {row.feature_b}" for row in df.head(limit).itertuples(index=False)]
    return []


def build_run_decision_markdown(
    selection_df: pd.DataFrame,
    *,
    mainline_root: Path,
    alignment_summary: dict[str, Any],
    pair_limit_by_model: dict[str, int],
) -> str:
    control_rows = selection_df[selection_df["is_control_winner"] == True] if not selection_df.empty else pd.DataFrame()
    control = control_rows.iloc[0] if not control_rows.empty else (selection_df.iloc[0] if not selection_df.empty else pd.Series(dtype=object))
    lines = [
        "# Bin Only Control Run Decision",
        "",
        f"- reference_mainline_root: {mainline_root}",
        f"- reference_mainline_run_id: {mainline_root.name}",
        "- feature_mode: bin_only",
        "- branches: mainline, mainline_plus_pairwise",
        "- split_reuse_mode: exact_split_indices_reuse",
        f"- alignment_all_checked_columns_match: {str(bool(alignment_summary.get('all_checked_columns_match', False))).lower()}",
        f"- trainable_rows: {alignment_summary.get('bin_trainable_rows', 'NA')}",
        f"- control_winner_model: {control.get('model', 'NA')}",
        f"- control_winner_branch: {control.get('branch', 'NA')}",
        f"- control_winner_p20_mean: {float(control.get('p20_mean', np.nan)) if not control.empty else np.nan}",
        f"- control_winner_enrichment20_mean: {float(control.get('enrichment20_mean', np.nan)) if not control.empty else np.nan}",
        f"- control_winner_tier2d_level: {control.get('tier2d_level', 'NA')}",
        f"- control_winner_interaction_mode: {control.get('interaction_mode', 'NA')}",
        f"- pair_limit_by_model: {json.dumps(pair_limit_by_model, ensure_ascii=False)}",
    ]
    return "\n".join(lines) + "\n"


def build_comparison_report(
    out_root: Path,
    *,
    mainline_root: Path,
    selection_df: pd.DataFrame,
    pair_limit_by_model: dict[str, int],
) -> str:
    cont_root = mainline_root / "part2" / "nonlinear_cont_only_v2"
    delta_topk_path = cont_root / "delta_topk_summary.csv"
    lines = [
        "# Bin Only vs Cont Mainline Comparison",
        "",
        f"- reference_mainline_root: {mainline_root}",
        f"- reference_cont_pipeline_root: {cont_root}",
        "- split_alignment: exact reuse of reference split_indices.json",
        "- audit_chain_alignment: negative_control + controlled_missingness_parallel + tier2d/C2C3_stability + rule_candidate_consistency",
        f"- pair_limit_by_model: {json.dumps(pair_limit_by_model, ensure_ascii=False)}",
        "",
        "## Nonlinear Same-Family Comparison",
        "",
        "| model | branch | cont_P20 | bin_P20 | cont_E20 | bin_E20 | cont_dP20_low | bin_dP20_low | cont_dE20_low | bin_dE20_low | bin_tier2d | bin_interaction_mode |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for model in REFERENCE_NONLINEAR_MODELS:
        cont_main_metrics = cont_root / "models" / model / "cont_only" / "mainline" / "metrics_topk_ci.csv"
        cont_pair_metrics = cont_root / "models" / model / "cont_only" / "mainline_plus_pairwise" / "metrics_topk_ci.csv"
        cont_pair_delta = load_cont_pair_delta(delta_topk_path, model)
        for branch_name, cont_metrics_path in (
            ("mainline", cont_main_metrics),
            ("mainline_plus_pairwise", cont_pair_metrics),
        ):
            cont_p20, _ = load_topk_metric(cont_metrics_path, "P", 20)
            cont_e20, _ = load_topk_metric(cont_metrics_path, "Enrichment", 20)
            row = selection_df[
                (selection_df["model"].astype(str) == model)
                & (selection_df["feature_mode"].astype(str) == "bin_only")
                & (selection_df["branch"].astype(str) == branch_name)
            ]
            if row.empty:
                bin_row = pd.Series(dtype=object)
            else:
                bin_row = row.iloc[0]
            if branch_name == "mainline":
                cont_dp20_low = 0.0
                cont_de20_low = 0.0
            else:
                cont_dp20_low = cont_pair_delta["delta_p20_ci_low"]
                cont_de20_low = cont_pair_delta["delta_enrichment20_ci_low"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        model,
                        branch_name,
                        f"{cont_p20:.4f}" if np.isfinite(cont_p20) else "NA",
                        f"{float(bin_row.get('p20_mean', np.nan)):.4f}" if not bin_row.empty and np.isfinite(float(bin_row.get("p20_mean", np.nan))) else "NA",
                        f"{cont_e20:.4f}" if np.isfinite(cont_e20) else "NA",
                        f"{float(bin_row.get('enrichment20_mean', np.nan)):.4f}" if not bin_row.empty and np.isfinite(float(bin_row.get("enrichment20_mean", np.nan))) else "NA",
                        f"{cont_dp20_low:.4f}" if np.isfinite(cont_dp20_low) else "NA",
                        f"{float(bin_row.get('delta_p20_ci_low', np.nan)):.4f}" if not bin_row.empty and np.isfinite(float(bin_row.get("delta_p20_ci_low", np.nan))) else "NA",
                        f"{cont_de20_low:.4f}" if np.isfinite(cont_de20_low) else "NA",
                        f"{float(bin_row.get('delta_enrichment20_ci_low', np.nan)):.4f}" if not bin_row.empty and np.isfinite(float(bin_row.get("delta_enrichment20_ci_low", np.nan))) else "NA",
                        str(bin_row.get("tier2d_level", "NA")) if not bin_row.empty else "NA",
                        str(bin_row.get("interaction_mode", "NA")) if not bin_row.empty else "NA",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Linear Control Layer",
            "",
            "| model | branch | bin_P20 | bin_E20 | bin_dP20_low | bin_dE20_low | tier2d | interaction_mode |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for model in LINEAR_CONTROL_MODELS:
        for branch_name in ("mainline", "mainline_plus_pairwise"):
            row = selection_df[
                (selection_df["model"].astype(str) == model)
                & (selection_df["feature_mode"].astype(str) == "bin_only")
                & (selection_df["branch"].astype(str) == branch_name)
            ]
            if row.empty:
                continue
            bin_row = row.iloc[0]
            lines.append(
                "| "
                + " | ".join(
                    [
                        model,
                        branch_name,
                        f"{float(bin_row['p20_mean']):.4f}" if np.isfinite(float(bin_row["p20_mean"])) else "NA",
                        f"{float(bin_row['enrichment20_mean']):.4f}" if np.isfinite(float(bin_row["enrichment20_mean"])) else "NA",
                        f"{float(bin_row['delta_p20_ci_low']):.4f}" if np.isfinite(float(bin_row["delta_p20_ci_low"])) else "NA",
                        f"{float(bin_row['delta_enrichment20_ci_low']):.4f}" if np.isfinite(float(bin_row["delta_enrichment20_ci_low"])) else "NA",
                        str(bin_row.get("tier2d_level", "NA")),
                        str(bin_row.get("interaction_mode", "NA")),
                    ]
                )
                + " |"
            )

    cont_gbdt_pairs = summarize_top_pairs(cont_root / "models" / "gbdt" / "cont_only" / "mainline_plus_pairwise" / "selected_pair_features.csv")
    bin_gbdt_publishable = summarize_top_pairs(out_root / "models" / "gbdt" / "bin_only" / "mainline_plus_pairwise" / "pair_rulebook_publishable_c3only.csv")
    bin_gbdt_unstable = summarize_top_pairs(out_root / "models" / "gbdt" / "bin_only" / "mainline_plus_pairwise" / "pair_rulebook_explanation_unstable_c3only.csv")
    lines.extend(
        [
            "",
            "## Pair Direction Snapshot",
            "",
            f"- cont_only gbdt top pairs: {', '.join(cont_gbdt_pairs) if cont_gbdt_pairs else 'none'}",
            f"- bin_only gbdt publishable pairs: {', '.join(bin_gbdt_publishable) if bin_gbdt_publishable else 'none'}",
            f"- bin_only gbdt unstable explanation pairs: {', '.join(bin_gbdt_unstable) if bin_gbdt_unstable else 'none'}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_run_manifest_payload(
    *,
    run_id: str,
    out_root: Path,
    mainline_root: Path,
    part1_out_dir: Path,
    pair_limit_by_model: dict[str, int],
    alignment_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run": {
            "run_id": run_id,
            "stage": "stage3_bin_only_control",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "reference_mainline": {
            "run_root": str(mainline_root),
            "run_id": mainline_root.name,
            "split_source": str(mainline_root / "part2" / "nonlinear_cont_only_v2" / "splits"),
        },
        "paths": {
            "part1_out_dir": str(part1_out_dir),
            "part2_out_dir": str(out_root),
        },
        "execution": {
            "feature_modes": ["bin_only"],
            "branch_subset": ["mainline", "mainline_plus_pairwise"],
            "model_subset": [cfg["name"] for cfg in MODEL_CONFIGS],
            "pair_limit_by_model": pair_limit_by_model,
            "split_reuse_mode": "exact_split_indices_reuse",
            "audit_chain": [
                "negative_control",
                "controlled_missingness_parallel",
                "tier2d/C2C3_stability",
                "rule_candidate_consistency",
            ],
        },
        "alignment_summary": alignment_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated bin_only control run against a completed cont_only mainline run.")
    parser.add_argument(
        "--mainline-root",
        default="artifacts/stage3_nonlinear_mainrule_n1_monotonic_chain_20260306_223004",
        help="Reference cont_only mainline run root.",
    )
    parser.add_argument(
        "--out-root",
        default="",
        help="Output root for the isolated bin_only control run. Defaults to a timestamped artifacts path.",
    )
    parser.add_argument(
        "--part1-out-dir",
        default="",
        help="Optional override for the shared Stage3 part1 directory. Defaults to the path declared by the reference mainline manifest.",
    )
    args = parser.parse_args()

    stage3_part2 = load_stage3_part2_module()
    original_make_estimator = patch_aligned_estimators()
    try:
        mainline_root = resolve_repo_path(REPO_ROOT, args.mainline_root)
        if not mainline_root.exists():
            raise FileNotFoundError(f"Reference mainline root not found: {mainline_root}")

        mainline_manifest = mainline_root / "manifest_monotonic_chain.yaml"
        if not mainline_manifest.exists():
            raise FileNotFoundError(f"Reference mainline manifest not found: {mainline_manifest}")
        manifest = load_simple_yaml(mainline_manifest)
        reference_execution = manifest.get("execution", {})
        reference_run = manifest.get("run", {})
        reference_paths = manifest.get("paths", {})

        if args.part1_out_dir:
            part1_out_dir = resolve_repo_path(REPO_ROOT, args.part1_out_dir)
        else:
            part1_out_dir = resolve_repo_path(REPO_ROOT, str(reference_paths.get("part1_out_dir")))
        out_root = resolve_repo_path(REPO_ROOT, args.out_root) if args.out_root else make_default_out_root(mainline_root)
        logs_dir = out_root / "logs"
        ensure_dir(out_root)
        ensure_dir(logs_dir)
        run_log = logs_dir / "run.log"
        append_log(run_log, f"reference_mainline_root={mainline_root}")
        append_log(run_log, f"part1_out_dir={part1_out_dir}")

        bin_file = part1_out_dir / "interaction_feature_view_bin_only.csv"
        cont_file = part1_out_dir / "interaction_feature_view_cont_only.csv"
        gate_file = part1_out_dir / "gate_c_acceptance.json"
        for path in (bin_file, cont_file, gate_file):
            if not path.exists():
                raise FileNotFoundError(f"Required input missing: {path}")
        gate = read_json(gate_file)
        if gate.get("gate_c_status") != "PASS":
            raise RuntimeError(f"Gate C not PASS: {gate}")

        bin_df = pd.read_csv(bin_file)
        cont_df = pd.read_csv(cont_file)
        alignment_summary = assert_trainable_alignment(cont_df, bin_df)
        append_log(run_log, "trainable_alignment=PASS")

        df_trainable = bin_df[bin_df["base_non_missing_count"] >= 2].copy().reset_index(drop=True)
        y = stage3_part2.is_strict_positive(df_trainable).to_numpy(dtype=int)

        split_src_dir = mainline_root / "part2" / "nonlinear_cont_only_v2" / "splits"
        split_indices_path = split_src_dir / "split_indices.json"
        splits = load_reference_splits(split_indices_path)
        split_reference_meta = {
            "source_mainline_root": str(mainline_root),
            "source_split_dir": str(split_src_dir),
            "split_count": int(len(splits)),
            "source_run_id": str(reference_run.get("run_id", mainline_root.name)),
            "alignment_summary": alignment_summary,
        }
        copy_reference_split_bundle(split_src_dir, out_root / "splits", split_reference_meta)
        append_log(run_log, f"split_reuse_count={len(splits)}")

        include_continuous, include_bins = stage3_features.feature_mode_flags("bin_only")
        base_X, base_feature_meta = stage3_features.prepare_base_features(
            df_trainable.copy(),
            include_missing_indicators=True,
            include_year_sensitivity=False,
            include_continuous=include_continuous,
            include_bins=include_bins,
        )
        feature_profile = stage3_features.summarize_feature_frame(base_X)
        base_feature_sets = {"bin_only": (base_X, base_feature_meta, feature_profile)}

        pair_candidates = stage3_candidates.generate_pairwise_candidates(
            df_trainable,
            y,
            splits,
            candidate_scope="C2C3",
        )
        triple_candidates = pair_candidates.iloc[0:0].copy()
        l0_threshold = stage3_part2.build_primary_threshold(reference_execution)
        unstable_explanation_config = {
            "enabled": True,
            "support_n_min": 10,
            "support_pos_min": 2,
            "enrichment_min": 1.2,
            "selection_freq_min": 0.50,
            "top_n": 10,
        }
        resolved_rulebook_policy = stage3_part2.resolve_nonlinear_cont_only_rulebook_policy(
            reference_execution,
            n_rows=int(len(df_trainable)),
            n_pos=int(y.sum()),
        )
        sensitivity_refs = {
            "year": "not_generated_in_isolated_bin_control",
            "missingness": "not_generated_in_isolated_bin_control",
            "threshold": "not_generated_in_isolated_bin_control",
            "bridge": "bin_vs_cont_mainline_comparison.md",
        }

        pair_limit_by_model = {cfg["name"]: int(cfg["pair_limit"]) for cfg in MODEL_CONFIGS}
        all_explain_frames: list[pd.DataFrame] = []
        all_metrics_frames: list[pd.DataFrame] = []
        all_metrics_c3_frames: list[pd.DataFrame] = []
        all_audit_frames: list[pd.DataFrame] = []
        all_selection_records: list[dict[str, Any]] = []

        root_warning_log = out_root / "run_warning.log"
        ensure_file(root_warning_log)

        stage3_part2.TRIPLE_LIMIT = 0
        stage3_part2.PAIR_DISCOVERY_LIMIT = 12
        stage3_part2.TRIPLE_DISCOVERY_LIMIT = 0
        stage3_part2.ENFORCE_CROSS_SIGNAL_PUBLISHABLE = True
        stage3_part2.EXPORT_DEBUG_ARTIFACTS = False

        for model_cfg in MODEL_CONFIGS:
            model_name = str(model_cfg["name"])
            model_pair_limit = int(model_cfg["pair_limit"])
            append_log(run_log, f"model_start={model_name} pair_limit={model_pair_limit}")
            stage3_part2.PAIR_LIMIT = model_pair_limit
            model_specs = stage3_part2.build_model_specs(model_subset={model_name})
            explain_selection_df = stage3_part2.build_explain_model_selection_table(
                model_specs,
                ["bin_only"],
                [stage3_part2.MAINLINE_BRANCH, stage3_part2.PAIRWISE_BRANCH],
                threshold=l0_threshold,
                pair_candidates=pair_candidates,
                triple_candidates=triple_candidates,
                enable_3way=False,
                cont_plus_bin_model_subset=set(),
            )
            if not explain_selection_df.empty:
                all_explain_frames.append(explain_selection_df)
            (
                _comparison_rows,
                metrics_summary,
                metrics_c3_summary,
                audit_summary_root,
                selection_records,
                _model_level_predictions,
                _model_level_fold_metrics,
                _model_level_audit_trace,
            ) = stage3_part2.execute_model_zoo(
                out_root,
                df_trainable,
                y,
                splits,
                random_seed=int(reference_run.get("random_seed", 20260305)),
                threshold=l0_threshold,
                pair_candidates=pair_candidates,
                triple_candidates=triple_candidates,
                base_feature_sets=base_feature_sets,
                feature_modes=["bin_only"],
                enable_3way=False,
                sensitivity_refs=sensitivity_refs,
                unstable_explanation_config=unstable_explanation_config,
                model_subset={model_name},
                cont_plus_bin_model_subset=set(),
                branch_subset=[stage3_part2.MAINLINE_BRANCH, stage3_part2.PAIRWISE_BRANCH],
                explain_selection_df=explain_selection_df,
                nonlinear_cont_only_rulebook_policy=resolved_rulebook_policy,
                tier2d_common_k_min=int(reference_execution.get("common_k_min", 15)),
                tier2d_min_test_c2_n=int(reference_execution.get("min_test_c2_n", 25)),
                tier2d_min_test_c3_n=int(reference_execution.get("min_test_c3_n", 20)),
                tier2d_raw_diff_eps=float(reference_execution.get("raw_diff_eps", 0.02)),
            )
            all_metrics_frames.append(metrics_summary)
            all_metrics_c3_frames.append(metrics_c3_summary)
            all_audit_frames.append(audit_summary_root)
            all_selection_records.extend(selection_records)
            append_log(run_log, f"model_done={model_name} rows={len(selection_records)}")

        explain_selected_df = pd.concat(all_explain_frames, ignore_index=True) if all_explain_frames else pd.DataFrame(columns=stage3_part2.EXPLAIN_MODEL_SELECTION_COLUMNS)
        stage3_part2.write_interaction_selected_from_explain_model(out_root, explain_selected_df)

        selection_df = pd.DataFrame(all_selection_records)
        selection_df = recompute_selection_flags(selection_df, stage3_part2)
        metrics_summary_df = pd.concat(all_metrics_frames, ignore_index=True) if all_metrics_frames else pd.DataFrame()
        metrics_c3_summary_df = pd.concat(all_metrics_c3_frames, ignore_index=True) if all_metrics_c3_frames else pd.DataFrame()
        audit_summary_df = pd.concat(all_audit_frames, ignore_index=True) if all_audit_frames else pd.DataFrame()

        metrics_summary_df.to_csv(out_root / "interaction_metrics_summary_ci.csv", index=False)
        metrics_c3_summary_df.to_csv(out_root / "interaction_metrics_c3_slice_ci.csv", index=False)
        audit_summary_df.to_csv(out_root / "interaction_audit_linkage_summary.csv", index=False)
        selection_df.to_csv(out_root / "model_selection_summary.csv", index=False)
        (out_root / "model_zoo_comparison.md").write_text(stage3_part2.model_zoo_markdown(selection_df, language="zh"), encoding="utf-8")
        stage3_part2.build_warning_summary(out_root, selection_df)

        run_id = out_root.name
        manifest_payload = build_run_manifest_payload(
            run_id=run_id,
            out_root=out_root,
            mainline_root=mainline_root,
            part1_out_dir=part1_out_dir,
            pair_limit_by_model=pair_limit_by_model,
            alignment_summary=alignment_summary,
        )
        stage3_part2.write_json(out_root / "run_manifest.json", manifest_payload)

        run_decision_md = build_run_decision_markdown(
            selection_df,
            mainline_root=mainline_root,
            alignment_summary=alignment_summary,
            pair_limit_by_model=pair_limit_by_model,
        )
        (out_root / "run_decision.md").write_text(run_decision_md, encoding="utf-8")

        comparison_md = build_comparison_report(
            out_root,
            mainline_root=mainline_root,
            selection_df=selection_df,
            pair_limit_by_model=pair_limit_by_model,
        )
        (out_root / "bin_vs_cont_mainline_comparison.md").write_text(comparison_md, encoding="utf-8")

        removed = prune_outputs(out_root)
        stage3_part2.write_json(out_root / "prune_summary.json", {"removed_files": removed, "removed_count": len(removed)})
        append_log(run_log, f"prune_removed={len(removed)}")
        append_log(run_log, "run_complete=PASS")
    finally:
        stage3_models.make_estimator = original_make_estimator


if __name__ == "__main__":
    main()
