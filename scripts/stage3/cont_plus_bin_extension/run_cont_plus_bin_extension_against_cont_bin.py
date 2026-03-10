#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
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

from common import append_log, ensure_dir, load_simple_yaml, resolve_repo_path, write_json  # noqa: E402
from stage3_workflow_isolated import step01_features as stage3_features  # noqa: E402
from stage3_workflow_isolated import step02_candidates as stage3_candidates  # noqa: E402
from stage3_workflow_isolated import step03_models as stage3_models  # noqa: E402
from stage3_workflow_isolated import step04_audits as stage3_audits  # noqa: E402
from stage3_workflow_isolated import step05_reporting as stage3_reporting  # noqa: E402


MAINLINE_BRANCH = "mainline"
PAIRWISE_BRANCH = "mainline_plus_pairwise"
TRIPLE_BRANCH = "mainline_plus_gated_3way"
BRANCH_ORDER = [MAINLINE_BRANCH, PAIRWISE_BRANCH, TRIPLE_BRANCH]
TOP_K_DEFAULT = [10, 20, 30]
KEY_ALIGNMENT_COLUMNS = (
    "id",
    "company",
    "coverage_tier",
    "llm_ai_dc_label",
    "accel_model",
    "accel_count",
    "base_non_missing_count",
)
PAIR_DISCOVERY_LIMIT = 12
TRIPLE_DISCOVERY_LIMIT = 8
PAIR_CANDIDATE_SCOPE = "C2C3"
PAIR_PUBLISH_SCOPE = "C3_only"
TRIPLE_CANDIDATE_SCOPE = "C2C3"
SOFT_AUDIT_NAME = "tier2d/C2C3_stability"
HARD_AUDITS = ["negative_control", "controlled_missingness_parallel", "rule_candidate_consistency"]
MODEL_CONFIGS = (
    {"name": "logistic_l2", "pair_limit": 2, "triple_limit": 0, "nonlinear": False},
    {"name": "gbdt", "pair_limit": 2, "triple_limit": 1, "nonlinear": True},
)
THRESHOLD_PRESETS: dict[str, dict[str, float]] = {
    "L0": {"support_n": 20, "support_pos": 3, "pair_freq": 0.80, "triple_freq": 0.90},
    "L1": {"support_n": 18, "support_pos": 2, "pair_freq": 0.75, "triple_freq": 0.85},
    "L2": {"support_n": 15, "support_pos": 2, "pair_freq": 0.70, "triple_freq": 0.80},
}
DEFAULT_UNSTABLE_EXPLANATION = {
    "enabled": True,
    "support_n_min": 10,
    "support_pos_min": 2,
    "enrichment_min": 1.2,
    "selection_freq_min": 0.50,
    "top_n": 10,
}


@dataclass
class ThresholdSpec:
    threshold_id: str
    support_n: int
    support_pos: int
    pair_freq: float
    triple_freq: float
    sensitivity_only: bool = False
    pair_support_n: int | None = None
    pair_support_pos: int | None = None
    triple_support_n: int | None = None
    triple_support_pos: int | None = None


@dataclass
class ModelSpec:
    name: str
    kind: str
    nonlinear: bool
    available: bool
    skip_reason: str = ""


@dataclass
class BranchData:
    branch_name: str
    feature_mode: str
    X: pd.DataFrame
    feature_profile: dict[str, Any]
    rulebook_support: pd.DataFrame
    rulebook_support_engineered_comparison: pd.DataFrame
    rulebook_legacy: pd.DataFrame
    rulebook_debug: pd.DataFrame
    pair_rulebook_publishable_c3only: pd.DataFrame
    pair_rulebook_explanation_unstable_c3only: pd.DataFrame
    candidate_trace: pd.DataFrame
    rulebook_model_derived_sensitivity: pd.DataFrame
    model_derived_cutpoint_alignment: pd.DataFrame
    linear_continuous_effects: pd.DataFrame
    linear_pairwise_effects: pd.DataFrame
    linear_vs_engineered_direction_check: pd.DataFrame
    skipped_by_config: bool
    skip_reason: str = ""


@dataclass
class BranchArtifacts:
    feature_mode: str
    feature_profile: dict[str, Any]
    metrics_by_split: pd.DataFrame
    metrics_ci: pd.DataFrame
    metrics_c3_slice_ci: pd.DataFrame
    delta_ci: pd.DataFrame
    audit_summary: pd.DataFrame
    rulebook_support: pd.DataFrame
    rulebook_support_engineered_comparison: pd.DataFrame
    rulebook_legacy: pd.DataFrame
    rulebook_debug: pd.DataFrame
    pair_rulebook_publishable_c3only: pd.DataFrame
    pair_rulebook_explanation_unstable_c3only: pd.DataFrame
    rulebook_model_derived_sensitivity: pd.DataFrame
    model_derived_cutpoint_alignment: pd.DataFrame
    linear_continuous_effects: pd.DataFrame
    linear_pairwise_effects: pd.DataFrame
    linear_vs_engineered_direction_check: pd.DataFrame
    decision_payload: dict[str, Any]
    conclusion_md: str
    mechanism_md: str
    status: str
    skipped_by_config: bool
    candidate_trace: pd.DataFrame
    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    audit_trace: pd.DataFrame
    input_shift_metrics: pd.DataFrame
    score_shift_metrics: pd.DataFrame
    perf_shift_metrics: pd.DataFrame
    tier_shift_matched_control: pd.DataFrame


def ensure_file(path: Path, content: str = "") -> None:
    ensure_dir(path.parent)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_default_out_root(mainline_root: Path) -> Path:
    return REPO_ROOT / "artifacts" / f"stage3_cont_plus_bin_extension_against_{mainline_root.name}_{timestamp_slug()}"


def fmt_float(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def load_json_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_strict_positive(df: pd.DataFrame) -> pd.Series:
    label_ok = df["llm_ai_dc_label"].isin(["ai_specific", "ai_optimized"])
    accel_ok = df["accel_model"].notna() | df["accel_count"].notna()
    return (label_ok & accel_ok).astype(int)


def load_reference_splits(split_indices_path: Path) -> list[dict[str, Any]]:
    raw = load_json_any(split_indices_path)
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


def assert_trainable_alignment(cont_df: pd.DataFrame, bin_df: pd.DataFrame, cont_plus_bin_df: pd.DataFrame) -> dict[str, Any]:
    def trainable(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["base_non_missing_count"].fillna(0).astype(float) >= 2].reset_index(drop=True)

    cont_t = trainable(cont_df)
    bin_t = trainable(bin_df)
    cpb_t = trainable(cont_plus_bin_df)
    sizes = {"cont_only": len(cont_t), "bin_only": len(bin_t), "cont_plus_bin": len(cpb_t)}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f"Trainable row count mismatch: {sizes}")
    checks: dict[str, bool] = {}
    for col in KEY_ALIGNMENT_COLUMNS:
        if col not in cont_t.columns or col not in bin_t.columns or col not in cpb_t.columns:
            continue
        checks[f"cont_vs_bin::{col}"] = cont_t[col].astype(str).equals(bin_t[col].astype(str))
        checks[f"cont_vs_cpb::{col}"] = cont_t[col].astype(str).equals(cpb_t[col].astype(str))
    failed = [col for col, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Trainable row order mismatch: {failed}")
    return {
        "cont_rows": int(len(cont_df)),
        "bin_rows": int(len(bin_df)),
        "cont_plus_bin_rows": int(len(cont_plus_bin_df)),
        "trainable_rows": int(len(cont_t)),
        "checked_columns": list(checks.keys()),
        "all_checked_columns_match": all(checks.values()) if checks else False,
    }


def copy_reference_split_bundle(src_dir: Path, dst_dir: Path, metadata: dict[str, Any]) -> None:
    ensure_dir(dst_dir)
    for name in ("company_holdout_splits.csv", "company_holdout_splits_meta.json", "split_indices.json"):
        shutil.copy2(src_dir / name, dst_dir / name)
    (dst_dir / "split_reference.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_bundle_matches(reference_split_dir: Path, candidate_split_dir: Path) -> bool:
    ref_csv = reference_split_dir / "company_holdout_splits.csv"
    cand_csv = candidate_split_dir / "company_holdout_splits.csv"
    if not ref_csv.exists() or not cand_csv.exists():
        return False
    return ref_csv.read_text(encoding="utf-8") == cand_csv.read_text(encoding="utf-8")


def patch_aligned_estimators() -> Any:
    original = stage3_models.make_estimator

    def aligned_make_estimator(
        model_kind: str,
        random_seed: int,
        *,
        branch_name: str = MAINLINE_BRANCH,
        pair_limit: int = 4,
        triple_limit: int = 0,
    ) -> Any:
        if model_kind == "gbdt":
            if branch_name == MAINLINE_BRANCH:
                return stage3_models.HistGradientBoostingClassifier(
                    max_depth=2,
                    learning_rate=0.05,
                    max_iter=220,
                    l2_regularization=1.0,
                    min_samples_leaf=10,
                    random_state=random_seed,
                )
            if branch_name == TRIPLE_BRANCH:
                return stage3_models.HistGradientBoostingClassifier(
                    max_depth=2,
                    learning_rate=0.05,
                    max_iter=260,
                    l2_regularization=2.0,
                    min_samples_leaf=15,
                    random_state=random_seed,
                )
            return stage3_models.HistGradientBoostingClassifier(
                max_depth=3,
                learning_rate=0.05,
                max_iter=320,
                l2_regularization=1.0,
                min_samples_leaf=10,
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


def build_primary_threshold(execution: dict[str, Any]) -> ThresholdSpec:
    threshold_id = str(execution.get("primary_threshold_id", "L0")).strip().upper() or "L0"
    preset = THRESHOLD_PRESETS.get(threshold_id, THRESHOLD_PRESETS["L0"])
    pair_support_n = int(execution.get("pair_support_n", preset["support_n"]))
    pair_support_pos = int(execution.get("pair_support_pos", preset["support_pos"]))
    pair_freq = float(execution.get("pair_selection_freq", preset["pair_freq"]))
    triple_support_n = int(execution.get("triple_support_n", pair_support_n))
    triple_support_pos = int(execution.get("triple_support_pos", pair_support_pos))
    triple_freq = float(execution.get("triple_selection_freq", preset["triple_freq"]))
    return ThresholdSpec(
        threshold_id=threshold_id,
        support_n=pair_support_n,
        support_pos=pair_support_pos,
        pair_freq=pair_freq,
        triple_freq=triple_freq,
        pair_support_n=pair_support_n,
        pair_support_pos=pair_support_pos,
        triple_support_n=triple_support_n,
        triple_support_pos=triple_support_pos,
    )


def summarize_metric_ci(metric_df: pd.DataFrame, model_name: str, branch_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metric_df.empty:
        return pd.DataFrame(rows)
    for (metric, k), grp in metric_df.groupby(["metric", "k"], dropna=False):
        vals = grp["value"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            mean = low = high = std = np.nan
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=0))
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
        rows.append(
            {
                "model": model_name,
                "branch": branch_name,
                "metric": metric,
                "k": int(k),
                "mean": mean,
                "std": std,
                "ci_low_95": low,
                "ci_high_95": high,
                "n_valid_splits": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def summarize_c3_slice_ci(pred_df: pd.DataFrame, model_name: str, branch_name: str, top_ks: list[int]) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame()
    metric_rows: list[dict[str, Any]] = []
    for split_id, grp in pred_df.groupby("split_id"):
        c3 = grp[grp["coverage_tier"] == "C3"]
        if c3.empty:
            continue
        y_true = c3["y_true"].to_numpy(dtype=int)
        score = c3["score"].to_numpy(dtype=float)
        for k in top_ks:
            metric_rows.append({"split_id": int(split_id), "metric": "P", "k": k, "value": stage3_models.precision_at_k(y_true, score, k)})
            metric_rows.append({"split_id": int(split_id), "metric": "Enrichment", "k": k, "value": stage3_models.enrichment_at_k(y_true, score, k)})
        metric_rows.append({"split_id": int(split_id), "metric": "AUC_proxy", "k": 0, "value": stage3_models.auc_proxy(y_true, score)})
    metric_df = pd.DataFrame(metric_rows)
    return summarize_metric_ci(metric_df, model_name, branch_name) if not metric_df.empty else pd.DataFrame()


def summarize_delta_ci(base_metric_df: pd.DataFrame, metric_df: pd.DataFrame, model_name: str, branch_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metric_df.empty:
        return pd.DataFrame(rows)
    if branch_name == MAINLINE_BRANCH:
        for (metric, k), grp in metric_df.groupby(["metric", "k"], dropna=False):
            rows.append(
                {
                    "model": model_name,
                    "branch": branch_name,
                    "metric": metric,
                    "k": int(k),
                    "mean": 0.0,
                    "ci_low_95": 0.0,
                    "ci_high_95": 0.0,
                    "n_valid_splits": int(len(grp)),
                }
            )
        return pd.DataFrame(rows)
    merged = base_metric_df.merge(metric_df, on=["split_id", "metric", "k"], suffixes=("_base", "_branch"))
    merged["delta"] = merged["value_branch"] - merged["value_base"]
    for (metric, k), grp in merged.groupby(["metric", "k"], dropna=False):
        vals = grp["delta"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            mean = low = high = np.nan
        else:
            mean = float(np.mean(vals))
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
        rows.append(
            {
                "model": model_name,
                "branch": branch_name,
                "metric": metric,
                "k": int(k),
                "mean": mean,
                "ci_low_95": low,
                "ci_high_95": high,
                "n_valid_splits": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def get_delta_summary(delta_ci: pd.DataFrame, metric: str, k: int) -> tuple[float, float, float]:
    row = delta_ci[(delta_ci["metric"].astype(str) == metric) & (delta_ci["k"].astype(int) == int(k))]
    if row.empty:
        return np.nan, np.nan, np.nan
    first = row.iloc[0]
    return float(first["mean"]), float(first["ci_low_95"]), float(first["ci_high_95"])


def continuous_source_for_feature(feature_name: str) -> str:
    mapping = {
        "power_mw_bin": "cont::power_mw",
        "power_mw": "cont::power_mw",
        "rack_kw_typical_bin": "cont::rack_kw_typical",
        "rack_kw_typical": "cont::rack_kw_typical",
        "rack_kw_peak_bin": "cont::rack_kw_peak",
        "rack_kw_peak": "cont::rack_kw_peak",
        "pue_bin": "cont::pue",
        "pue": "cont::pue",
        "building_sqm_bin": "cont::building_sqm",
        "building_sqm": "cont::building_sqm",
        "whitespace_sqm": "cont::whitespace_sqm",
        "rack_density_area_w_per_sf_dc": "cont::rack_density_area_w_per_sf_dc",
    }
    return mapping.get(feature_name, "")


def candidate_to_feature_frame(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    if candidates.empty:
        return feat
    for row in candidates.itertuples(index=False):
        cond = pd.Series(True, index=df.index)
        if getattr(row, "feature_a", ""):
            cond &= df[str(row.feature_a)].astype(str) == str(row.value_a)
        if getattr(row, "feature_b", ""):
            cond &= df[str(row.feature_b)].astype(str) == str(row.value_b)
        if getattr(row, "feature_c", ""):
            cond &= df[str(row.feature_c)].astype(str) == str(row.value_c)
        if getattr(row, "type", "") == "triple":
            cond &= df["coverage_tier"].isin(["C2", "C3"])
        feat[str(row.candidate_id)] = cond.astype(float)
    return feat


def candidate_to_nonlinear_feature_frame(df: pd.DataFrame, base_X: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    if candidates.empty:
        return feat
    for row in candidates.itertuples(index=False):
        values: list[pd.Series] = []
        for feature_name, value in [(row.feature_a, row.value_a), (row.feature_b, row.value_b), (row.feature_c, row.value_c)]:
            if not feature_name:
                continue
            cont_source = continuous_source_for_feature(str(feature_name))
            if cont_source and cont_source in base_X.columns:
                values.append(pd.to_numeric(base_X[cont_source], errors="coerce").fillna(0.0).astype(float))
            else:
                values.append((df[str(feature_name)].astype(str) == str(value)).astype(float))
        if not values:
            continue
        feature_series = values[0].copy()
        for component in values[1:]:
            feature_series = feature_series * component
        if getattr(row, "type", "") == "triple":
            feature_series = feature_series * df["coverage_tier"].isin(["C2", "C3"]).astype(float)
        feat[f"cx::{row.candidate_id}"] = feature_series.astype(float)
    return feat


def shortlist_candidate_pool(candidates: pd.DataFrame, *, pair_limit: int, triple_limit: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    df = candidates.copy()
    if "selection_freq" not in df.columns:
        df["selection_freq"] = 0.0
    df = df.sort_values(
        [
            "type",
            "same_signal_group",
            "primary_condition_contains_missing",
            "selection_freq",
            "enrichment",
            "support_pos",
            "support_n",
        ],
        ascending=[True, True, True, False, False, False, False],
    ).reset_index(drop=True)
    keep_frames: list[pd.DataFrame] = []
    pair_df = df[df["type"] == "pair"].head(max(0, int(pair_limit)))
    triple_df = df[df["type"] == "triple"].head(max(0, int(triple_limit)))
    if not pair_df.empty:
        keep_frames.append(pair_df)
    if not triple_df.empty:
        keep_frames.append(triple_df)
    if not keep_frames:
        return df.iloc[0:0].copy()
    return pd.concat(keep_frames, ignore_index=True)


def build_model_specs(model_subset: set[str] | None = None) -> list[ModelSpec]:
    specs = [
        ModelSpec(name="logistic_l2", kind="logistic_l2", nonlinear=False, available=True),
        ModelSpec(name="gbdt", kind="gbdt", nonlinear=True, available=True),
    ]
    if not model_subset:
        return specs
    return [spec for spec in specs if spec.name in model_subset]


def compute_prediction_rule_audit(rulebook_support: pd.DataFrame) -> tuple[bool, str]:
    if rulebook_support.empty:
        return True, "no_rules"
    prediction_rows = rulebook_support[rulebook_support["rule_type"] == "prediction"]
    if prediction_rows.empty:
        return True, "no_prediction_rules"
    invalid = prediction_rows[
        prediction_rows["condition_text"].astype(str).str.contains("__MISSING__")
        | prediction_rows["condition_text"].astype(str).str.contains("missing_flag")
    ]
    if invalid.empty:
        return True, "bad_prediction_rules=0"
    return False, f"bad_prediction_rules={len(invalid)}"


def extract_mechanism_markdown(model_spec: ModelSpec, model: Any, feature_names: list[str], top_n: int = 10) -> str:
    lines = [f"# {model_spec.name} Mechanism Extraction", ""]
    if model is None:
        lines.append("- skipped: no fitted model")
        return "\n".join(lines) + "\n"
    importance = stage3_reporting.model_feature_importance_series(model, feature_names)
    if importance.empty:
        lines.append("- skipped: importance interface unavailable")
        return "\n".join(lines) + "\n"
    lines.append("## Top Features")
    ordered = importance.reindex(importance.abs().sort_values(ascending=False).index)
    for name, value in ordered.head(top_n).items():
        lines.append(f"- {name}: {float(value):.6f}")
    return "\n".join(lines) + "\n"


def build_branch_data(
    model_spec: ModelSpec,
    branch_name: str,
    feature_mode: str,
    df: pd.DataFrame,
    base_X: pd.DataFrame,
    feature_profile: dict[str, Any],
    threshold: ThresholdSpec,
    selected_pairs: pd.DataFrame,
    pair_trace: pd.DataFrame,
    selected_triples: pd.DataFrame,
    triple_trace: pd.DataFrame,
) -> BranchData:
    if branch_name == MAINLINE_BRANCH:
        return BranchData(
            branch_name=branch_name,
            feature_mode=feature_mode,
            X=base_X.copy(),
            feature_profile=dict(feature_profile),
            rulebook_support=pd.DataFrame(columns=stage3_reporting.SUPPORT_RULEBOOK_COLUMNS),
            rulebook_support_engineered_comparison=pd.DataFrame(columns=stage3_reporting.SUPPORT_RULEBOOK_COLUMNS),
            rulebook_legacy=pd.DataFrame(columns=stage3_reporting.LEGACY_RULEBOOK_COLUMNS),
            rulebook_debug=pd.DataFrame(),
            pair_rulebook_publishable_c3only=pd.DataFrame(columns=stage3_reporting.C3_PAIR_RULEBOOK_COLUMNS),
            pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=stage3_reporting.UNSTABLE_EXPLANATION_PAIR_COLUMNS),
            candidate_trace=pd.DataFrame(),
            rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
            model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
            linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
            linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
            linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
            skipped_by_config=False,
        )
    if branch_name == TRIPLE_BRANCH and not model_spec.nonlinear:
        return BranchData(
            branch_name=branch_name,
            feature_mode=feature_mode,
            X=base_X.copy(),
            feature_profile=dict(feature_profile),
            rulebook_support=pd.DataFrame(columns=stage3_reporting.SUPPORT_RULEBOOK_COLUMNS),
            rulebook_support_engineered_comparison=pd.DataFrame(columns=stage3_reporting.SUPPORT_RULEBOOK_COLUMNS),
            rulebook_legacy=pd.DataFrame(columns=stage3_reporting.LEGACY_RULEBOOK_COLUMNS),
            rulebook_debug=pd.DataFrame(),
            pair_rulebook_publishable_c3only=pd.DataFrame(columns=stage3_reporting.C3_PAIR_RULEBOOK_COLUMNS),
            pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=stage3_reporting.UNSTABLE_EXPLANATION_PAIR_COLUMNS),
            candidate_trace=pd.DataFrame(),
            rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
            model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
            linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
            linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
            linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
            skipped_by_config=True,
            skip_reason="linear_model_3way_disabled",
        )

    branch_pairs = selected_pairs.copy()
    prediction_pairs = branch_pairs[branch_pairs["rule_type"].astype(str) == "prediction"].copy() if not branch_pairs.empty else pd.DataFrame()
    pair_frame = candidate_to_nonlinear_feature_frame(df, base_X, prediction_pairs) if model_spec.nonlinear else candidate_to_feature_frame(df, prediction_pairs)
    c3_pair_rulebook = stage3_reporting.build_c3_only_pair_rulebook(
        df,
        is_strict_positive(df).to_numpy(dtype=int),
        branch_pairs,
        publish_scope=PAIR_PUBLISH_SCOPE,
        default_candidate_train_scope=PAIR_CANDIDATE_SCOPE,
    )

    if branch_name == PAIRWISE_BRANCH:
        support = stage3_reporting.build_rulebook_support_from_candidates(branch_pairs, threshold=threshold, branch_name=branch_name)
        legacy = stage3_reporting.build_legacy_rulebook_from_pairs(branch_pairs)
        return BranchData(
            branch_name=branch_name,
            feature_mode=feature_mode,
            X=pd.concat([base_X, pair_frame], axis=1),
            feature_profile=dict(feature_profile),
            rulebook_support=support,
            rulebook_support_engineered_comparison=pd.DataFrame(columns=stage3_reporting.SUPPORT_RULEBOOK_COLUMNS),
            rulebook_legacy=legacy,
            rulebook_debug=branch_pairs,
            pair_rulebook_publishable_c3only=c3_pair_rulebook,
            pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=stage3_reporting.UNSTABLE_EXPLANATION_PAIR_COLUMNS),
            candidate_trace=pair_trace,
            rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
            model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
            linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
            linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
            linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
            skipped_by_config=False,
        )

    branch_triples = selected_triples.copy()
    prediction_triples = branch_triples[branch_triples["rule_type"].astype(str) == "prediction"].copy() if not branch_triples.empty else pd.DataFrame()
    triple_frame = candidate_to_nonlinear_feature_frame(df, base_X, prediction_triples) if model_spec.nonlinear else candidate_to_feature_frame(df, prediction_triples)
    support = stage3_reporting.build_rulebook_support_from_candidates(
        pd.concat([branch_pairs, branch_triples], ignore_index=True),
        threshold=threshold,
        branch_name=branch_name,
    )
    legacy = stage3_reporting.build_legacy_rulebook_from_pairs(branch_pairs)
    return BranchData(
        branch_name=branch_name,
        feature_mode=feature_mode,
        X=pd.concat([base_X, pair_frame, triple_frame], axis=1),
        feature_profile=dict(feature_profile),
        rulebook_support=support,
        rulebook_support_engineered_comparison=pd.DataFrame(columns=stage3_reporting.SUPPORT_RULEBOOK_COLUMNS),
        rulebook_legacy=legacy,
        rulebook_debug=pd.concat([branch_pairs, branch_triples], ignore_index=True),
        pair_rulebook_publishable_c3only=c3_pair_rulebook,
        pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=stage3_reporting.UNSTABLE_EXPLANATION_PAIR_COLUMNS),
        candidate_trace=pd.concat([pair_trace, triple_trace], ignore_index=True),
        rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
        model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
        linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
        linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
        linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
        skipped_by_config=False,
    )


def evaluate_branch_audits(
    model_spec: ModelSpec,
    branch_data: BranchData,
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    pred_df: pd.DataFrame,
    warning_log: Path,
    random_seed: int,
    *,
    pair_limit: int,
    triple_limit: int,
    common_k_min: int,
    min_test_c2_n: int,
    min_test_c3_n: int,
    raw_diff_eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, Any]] = []
    audit_trace_frames: list[pd.DataFrame] = []
    pass_flag, detail = compute_prediction_rule_audit(branch_data.rulebook_support)
    audit_rows.append({"audit_name": "controlled_missingness_parallel", "status": "PASS" if pass_flag else "FAIL", "details": detail, "severity": "hard"})

    neg_pass, neg_detail, neg_trace = stage3_audits.negative_control_audit(
        model_spec,
        branch_data,
        df,
        y,
        splits,
        random_seed=random_seed,
        warning_log=warning_log,
        pair_limit=pair_limit,
        triple_limit=triple_limit,
    )
    audit_rows.append({"audit_name": "negative_control", "status": "PASS" if neg_pass else "FAIL", "details": neg_detail, "severity": "hard"})
    if not neg_trace.empty:
        neg_trace.insert(0, "audit_name", "negative_control")
        audit_trace_frames.append(neg_trace)

    (
        tier_pass,
        tier_level,
        tier_detail,
        applicability_domain,
        tier_trace,
        input_shift_metrics,
        score_shift_metrics,
        perf_shift_metrics,
        tier_shift_matched_control,
    ) = stage3_audits.tier_stability_audit(
        pred_df,
        feature_df=df,
        top_k=20,
        common_k_min=common_k_min,
        min_test_c2_n=min_test_c2_n,
        min_test_c3_n=min_test_c3_n,
        raw_diff_eps=raw_diff_eps,
    )
    tier_summary = perf_shift_metrics[perf_shift_metrics["row_type"] == "summary"] if not perf_shift_metrics.empty and "row_type" in perf_shift_metrics.columns else pd.DataFrame()
    n_splits_used = int(tier_summary["n_splits_used"].iloc[0]) if not tier_summary.empty and "n_splits_used" in tier_summary.columns and pd.notna(tier_summary["n_splits_used"].iloc[0]) else 0
    audit_rows.append(
        {
            "audit_name": SOFT_AUDIT_NAME,
            "status": "PASS" if tier_pass else "FAIL",
            "details": tier_detail,
            "severity": "soft",
            "tier2d_level": tier_level,
            "common_k_min": int(common_k_min),
            "min_test_c2_n": int(min_test_c2_n),
            "min_test_c3_n": int(min_test_c3_n),
            "raw_diff_eps": float(raw_diff_eps),
            "n_splits_used": int(n_splits_used),
        }
    )
    if not tier_trace.empty:
        tier_trace.insert(0, "audit_name", SOFT_AUDIT_NAME)
        audit_trace_frames.append(tier_trace)

    consistency_pass, consistency_detail = stage3_audits.candidate_consistency_audit(branch_data.rulebook_support, branch_data.candidate_trace)
    audit_rows.append({"audit_name": "rule_candidate_consistency", "status": "PASS" if consistency_pass else "FAIL", "details": consistency_detail, "severity": "hard"})

    audit_summary = pd.DataFrame(audit_rows)
    audit_trace = pd.concat(audit_trace_frames, ignore_index=True) if audit_trace_frames else pd.DataFrame()
    return (
        audit_summary,
        audit_trace,
        applicability_domain,
        tier_level,
        input_shift_metrics,
        score_shift_metrics,
        perf_shift_metrics,
        tier_shift_matched_control,
    )


def build_branch_conclusion(
    model_name: str,
    feature_mode: str,
    branch_name: str,
    metrics_ci: pd.DataFrame,
    delta_ci: pd.DataFrame,
    audit_summary: pd.DataFrame,
    rulebook_support: pd.DataFrame,
    unstable_explanation_pairs: pd.DataFrame,
    *,
    status: str,
    skip_reason: str,
    applicability_domain: str,
) -> str:
    p20_row = metrics_ci[(metrics_ci["metric"].astype(str) == "P") & (metrics_ci["k"].astype(int) == 20)]
    e20_row = metrics_ci[(metrics_ci["metric"].astype(str) == "Enrichment") & (metrics_ci["k"].astype(int) == 20)]
    auc_row = metrics_ci[(metrics_ci["metric"].astype(str) == "AUC_proxy") & (metrics_ci["k"].astype(int) == 0)]
    dp20_mean, dp20_low, dp20_high = get_delta_summary(delta_ci, "P", 20)
    de20_mean, de20_low, de20_high = get_delta_summary(delta_ci, "Enrichment", 20)
    p20 = p20_row.iloc[0] if not p20_row.empty else pd.Series(dtype=float)
    e20 = e20_row.iloc[0] if not e20_row.empty else pd.Series(dtype=float)
    auc = auc_row.iloc[0] if not auc_row.empty else pd.Series(dtype=float)
    lines = [
        f"# {model_name} / {feature_mode} / {branch_name}",
        "",
        f"- status: {status}",
        f"- skip_reason: {skip_reason or 'NA'}",
        f"- applicability_domain: {applicability_domain}",
        f"- P@20: {fmt_float(float(p20.get('mean', np.nan)))}",
        f"- Enrichment@20: {fmt_float(float(e20.get('mean', np.nan)))}",
        f"- AUC_proxy: {fmt_float(float(auc.get('mean', np.nan)))}",
        f"- delta_p20_mean: {fmt_float(dp20_mean)}",
        f"- delta_p20_ci_low: {fmt_float(dp20_low)}",
        f"- delta_p20_ci_high: {fmt_float(dp20_high)}",
        f"- delta_enrichment20_mean: {fmt_float(de20_mean)}",
        f"- delta_enrichment20_ci_low: {fmt_float(de20_low)}",
        f"- delta_enrichment20_ci_high: {fmt_float(de20_high)}",
        "",
        "## Audits",
    ]
    if audit_summary.empty:
        lines.append("- none")
    else:
        for row in audit_summary.itertuples(index=False):
            lines.append(f"- {row.audit_name}: {row.status} ({row.details})")
    lines.extend(["", "## Prediction Rules"])
    pred_rules = rulebook_support[rulebook_support["rule_type"].astype(str) == "prediction"].head(8) if not rulebook_support.empty and "rule_type" in rulebook_support.columns else pd.DataFrame()
    if pred_rules.empty:
        lines.append("- none")
    else:
        for row in pred_rules.itertuples(index=False):
            lines.append(f"- {row.condition_text} | coverage={fmt_float(float(row.coverage), 4)} | enrich={fmt_float(float(row.enrichment), 4)}")
    lines.extend(["", "## Unstable Explanation Pairs"])
    if unstable_explanation_pairs.empty:
        lines.append("- none")
    else:
        for row in unstable_explanation_pairs.head(8).itertuples(index=False):
            lines.append(f"- {row.condition_text} | enrich_c3={fmt_float(float(row.enrichment_c3), 4)} | why={row.why_unstable}")
    return "\n".join(lines) + "\n"


def make_placeholder_branch_outputs(branch_name: str, feature_mode: str, feature_profile: dict[str, Any], skip_reason: str) -> BranchArtifacts:
    audit_summary = pd.DataFrame(
        [
            {"audit_name": "controlled_missingness_parallel", "status": "PASS", "details": skip_reason, "severity": "hard"},
            {"audit_name": "negative_control", "status": "PASS", "details": skip_reason, "severity": "hard"},
            {"audit_name": SOFT_AUDIT_NAME, "status": "PASS", "details": skip_reason, "severity": "soft", "tier2d_level": "PASS"},
            {"audit_name": "rule_candidate_consistency", "status": "PASS", "details": skip_reason, "severity": "hard"},
        ]
    )
    decision = {
        "status": "SKIPPED",
        "skip_reason": skip_reason,
        "feature_mode": feature_mode,
        "hard_audits_pass": 3,
        "tier2d_status": "PASS",
        "tier2d_level": "PASS",
        "interaction_mode": "explanation_only",
        "interaction_upgrade_eligible": False,
        "warning_present": False,
        "rulebook_reproducible": True,
        "prediction_rule_count": 0,
        "publish_scope_final": "C2C3",
        "applicability_domain": "all",
    }
    return BranchArtifacts(
        feature_mode=feature_mode,
        feature_profile=dict(feature_profile),
        metrics_by_split=pd.DataFrame(),
        metrics_ci=pd.DataFrame(columns=["model", "branch", "metric", "k", "mean", "std", "ci_low_95", "ci_high_95", "n_valid_splits"]),
        metrics_c3_slice_ci=pd.DataFrame(columns=["model", "branch", "metric", "k", "mean", "std", "ci_low_95", "ci_high_95", "n_valid_splits"]),
        delta_ci=pd.DataFrame(columns=["model", "branch", "metric", "k", "mean", "ci_low_95", "ci_high_95", "n_valid_splits"]),
        audit_summary=audit_summary,
        rulebook_support=pd.DataFrame(columns=stage3_reporting.SUPPORT_RULEBOOK_COLUMNS),
        rulebook_support_engineered_comparison=pd.DataFrame(columns=stage3_reporting.SUPPORT_RULEBOOK_COLUMNS),
        rulebook_legacy=pd.DataFrame(columns=stage3_reporting.LEGACY_RULEBOOK_COLUMNS),
        rulebook_debug=pd.DataFrame(),
        pair_rulebook_publishable_c3only=pd.DataFrame(columns=stage3_reporting.C3_PAIR_RULEBOOK_COLUMNS),
        pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=stage3_reporting.UNSTABLE_EXPLANATION_PAIR_COLUMNS),
        rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
        model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
        linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
        linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
        linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
        decision_payload=decision,
        conclusion_md=f"# {branch_name} skipped\n\n- reason: {skip_reason}\n",
        mechanism_md="# mechanism skipped\n",
        status="SKIPPED",
        skipped_by_config=True,
        candidate_trace=pd.DataFrame(),
        oof_predictions=pd.DataFrame(),
        fold_metrics=pd.DataFrame(),
        audit_trace=pd.DataFrame(),
        input_shift_metrics=pd.DataFrame(),
        score_shift_metrics=pd.DataFrame(),
        perf_shift_metrics=pd.DataFrame(),
        tier_shift_matched_control=pd.DataFrame(),
    )


def interaction_upgrade_gate(
    branch_name: str,
    *,
    allow_interaction_promotion: bool,
    status: str,
    hard_pass_count: int,
    warning_present: bool,
    rulebook_reproducible: bool,
    delta_p20_ci_low: float,
    delta_enrichment20_ci_low: float,
) -> tuple[bool, str]:
    if branch_name == MAINLINE_BRANCH:
        return True, "baseline_only"
    if not allow_interaction_promotion:
        return False, "explanation_only"
    eligible = (
        status == "COMPLETED"
        and hard_pass_count == len(HARD_AUDITS)
        and (not warning_present)
        and bool(rulebook_reproducible)
        and np.isfinite(delta_p20_ci_low)
        and np.isfinite(delta_enrichment20_ci_low)
        and delta_p20_ci_low >= 0
        and delta_enrichment20_ci_low >= 0
    )
    return eligible, ("upgrade_eligible" if eligible else "explanation_only")


def execute_model_matrix(
    out_root: Path,
    df_trainable: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    *,
    random_seed: int,
    threshold: ThresholdSpec,
    pair_candidates: pd.DataFrame,
    triple_candidates: pd.DataFrame,
    base_feature_sets: dict[str, tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]],
    model_configs: tuple[dict[str, Any], ...],
    unstable_explanation_config: dict[str, Any],
    top_ks: list[int],
    tier2d_common_k_min: int,
    tier2d_min_test_c2_n: int,
    tier2d_min_test_c3_n: int,
    tier2d_raw_diff_eps: float,
    run_log: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_rows: list[dict[str, Any]] = []
    metrics_frames: list[pd.DataFrame] = []
    metrics_c3_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    feature_mode = "cont_plus_bin"
    base_X, base_feature_meta, feature_profile = base_feature_sets[feature_mode]

    for model_cfg in model_configs:
        model_name = str(model_cfg["name"])
        pair_limit = int(model_cfg["pair_limit"])
        triple_limit = int(model_cfg["triple_limit"])
        model_spec = build_model_specs({model_name})[0]
        model_dir = out_root / "models" / model_name / feature_mode
        ensure_dir(model_dir)
        append_log(run_log, f"model_start={model_name} pair_limit={pair_limit} triple_limit={triple_limit}")

        pair_pool = shortlist_candidate_pool(pair_candidates, pair_limit=PAIR_DISCOVERY_LIMIT, triple_limit=TRIPLE_DISCOVERY_LIMIT)
        selected_pairs, pair_trace = stage3_candidates.classify_candidates(
            pair_pool,
            threshold,
            pair_limit=pair_limit,
            triple_limit=triple_limit,
            enforce_cross_signal_publishable=True,
        )
        triple_pool = shortlist_candidate_pool(triple_candidates, pair_limit=0, triple_limit=TRIPLE_DISCOVERY_LIMIT)
        selected_triples, triple_trace = stage3_candidates.classify_candidates(
            triple_pool,
            threshold,
            pair_limit=pair_limit,
            triple_limit=triple_limit,
            enforce_cross_signal_publishable=True,
        )

        mainline_metric_df = pd.DataFrame()
        for branch_name in BRANCH_ORDER:
            branch_dir = model_dir / branch_name
            ensure_dir(branch_dir)
            branch_warning_log = branch_dir / "run_warning.log"
            ensure_file(branch_warning_log)

            branch_data = build_branch_data(
                model_spec,
                branch_name,
                feature_mode,
                df_trainable,
                base_X,
                feature_profile,
                threshold,
                selected_pairs,
                pair_trace,
                selected_triples,
                triple_trace,
            )

            if branch_data.skipped_by_config:
                branch_result = make_placeholder_branch_outputs(branch_name, feature_mode, feature_profile, branch_data.skip_reason or "skipped_by_config")
            else:
                metric_df, pred_df, fold_metric_df, audit_trace_df, full_model, full_scaling_stats = stage3_models.evaluate_model_branch(
                    model_spec,
                    branch_data,
                    df_trainable,
                    y,
                    splits,
                    random_seed=random_seed,
                    warning_log=branch_warning_log,
                    pair_limit=pair_limit,
                    triple_limit=triple_limit,
                    top_ks=top_ks,
                )
                if branch_name == MAINLINE_BRANCH:
                    mainline_metric_df = metric_df.copy()
                metrics_ci = summarize_metric_ci(metric_df, model_name, branch_name)
                metrics_c3_ci = summarize_c3_slice_ci(pred_df, model_name, branch_name, top_ks)
                delta_ci = summarize_delta_ci(mainline_metric_df if not mainline_metric_df.empty else metric_df, metric_df, model_name, branch_name)

                if branch_name == MAINLINE_BRANCH:
                    support_rulebook, legacy_rulebook, debug_rulebook = stage3_reporting.build_mainline_rulebook(
                        model_spec,
                        df_trainable,
                        base_X,
                        base_feature_meta,
                        y,
                        full_model,
                    )
                    branch_data.rulebook_support = support_rulebook
                    branch_data.rulebook_legacy = legacy_rulebook
                    branch_data.rulebook_debug = debug_rulebook
                    (
                        branch_data.rulebook_model_derived_sensitivity,
                        branch_data.model_derived_cutpoint_alignment,
                    ) = stage3_reporting.build_model_derived_sensitivity_rulebook(
                        model_spec,
                        full_model,
                        df_trainable,
                        base_X,
                        y,
                        full_scaling_stats,
                        min_support_n=10,
                        min_support_pos=2,
                        min_enrichment=1.05,
                        max_rules_per_feature=3,
                    )

                branch_data.linear_continuous_effects = stage3_reporting.build_linear_continuous_effects(
                    model_spec,
                    full_model,
                    branch_data.X.columns.tolist(),
                    full_scaling_stats,
                )
                branch_data.linear_pairwise_effects = stage3_reporting.build_linear_pairwise_effects(
                    model_spec,
                    full_model,
                    branch_data.X.columns.tolist(),
                    branch_data.rulebook_debug if isinstance(branch_data.rulebook_debug, pd.DataFrame) else pd.DataFrame(),
                )
                branch_data.linear_vs_engineered_direction_check = stage3_reporting.build_linear_vs_engineered_direction_check(
                    model_spec,
                    branch_data.linear_continuous_effects,
                )

                (
                    audit_summary,
                    audit_trace,
                    applicability_domain,
                    tier2d_level,
                    input_shift_metrics,
                    score_shift_metrics,
                    perf_shift_metrics,
                    tier_shift_matched_control,
                ) = evaluate_branch_audits(
                    model_spec,
                    branch_data,
                    df_trainable,
                    y,
                    splits,
                    pred_df,
                    branch_warning_log,
                    random_seed,
                    pair_limit=pair_limit,
                    triple_limit=triple_limit,
                    common_k_min=tier2d_common_k_min,
                    min_test_c2_n=tier2d_min_test_c2_n,
                    min_test_c3_n=tier2d_min_test_c3_n,
                    raw_diff_eps=tier2d_raw_diff_eps,
                )
                tier2d_status = (
                    audit_summary.loc[audit_summary["audit_name"] == SOFT_AUDIT_NAME, "status"].iloc[0]
                    if (not audit_summary.empty and (audit_summary["audit_name"] == SOFT_AUDIT_NAME).any())
                    else "PASS"
                )
                tier2d_detail = (
                    audit_summary.loc[audit_summary["audit_name"] == SOFT_AUDIT_NAME, "details"].iloc[0]
                    if (not audit_summary.empty and (audit_summary["audit_name"] == SOFT_AUDIT_NAME).any())
                    else ""
                )
                publish_scope_final = "C2C3" if tier2d_level in {"PASS", "WARN"} else "C3_only"
                _, dp20_low, _ = get_delta_summary(delta_ci, "P", 20)
                _, de20_low, _ = get_delta_summary(delta_ci, "Enrichment", 20)
                if branch_name != MAINLINE_BRANCH:
                    debug_df = branch_data.rulebook_debug if isinstance(branch_data.rulebook_debug, pd.DataFrame) else pd.DataFrame()
                    if model_spec.nonlinear:
                        branch_data.pair_rulebook_publishable_c3only = stage3_reporting.build_c3_only_pair_rulebook(
                            df_trainable,
                            y,
                            debug_df,
                            publish_scope=publish_scope_final,
                            default_candidate_train_scope=PAIR_CANDIDATE_SCOPE,
                        )
                    else:
                        branch_data.pair_rulebook_publishable_c3only = pd.DataFrame(columns=stage3_reporting.C3_PAIR_RULEBOOK_COLUMNS)
                        if not branch_data.rulebook_support.empty and "rule_type" in branch_data.rulebook_support.columns:
                            branch_data.rulebook_support["rule_type"] = "triage"
                            if "notes" in branch_data.rulebook_support.columns:
                                branch_data.rulebook_support["notes"] = branch_data.rulebook_support["notes"].astype(str).apply(
                                    lambda s: "; ".join([piece for piece in [s if s != "nan" else "", "linear_pairwise_explanation_only=true", "publishable_pair_rulebook=disabled"] if piece])
                                )
                    branch_data.pair_rulebook_explanation_unstable_c3only = stage3_reporting.build_unstable_explanation_pair_rulebook(
                        df_trainable,
                        y,
                        debug_df,
                        delta_p20_ci_low=dp20_low,
                        delta_enrichment20_ci_low=de20_low,
                        tier2d_status=tier2d_status,
                        tier2d_detail=tier2d_detail,
                        publish_scope=publish_scope_final,
                        default_candidate_train_scope=PAIR_CANDIDATE_SCOPE,
                        config=unstable_explanation_config,
                    )
                    if model_spec.nonlinear:
                        stage3_reporting.annotate_nonlinear_interaction_rulebooks(model_spec, branch_name, branch_data, full_model)

                decision_payload = {
                    "status": "COMPLETED",
                    "skip_reason": "",
                    "feature_mode": feature_mode,
                    "hard_audits_pass": int((audit_summary[audit_summary["audit_name"].isin(HARD_AUDITS)]["status"] == "PASS").sum()),
                    "tier2d_status": tier2d_status,
                    "tier2d_level": tier2d_level,
                    "warning_present": bool(branch_warning_log.exists() and branch_warning_log.stat().st_size > 0),
                    "prediction_rule_count": int((branch_data.rulebook_support["rule_type"] == "prediction").sum()) if not branch_data.rulebook_support.empty and "rule_type" in branch_data.rulebook_support.columns else 0,
                    "publish_scope_final": publish_scope_final,
                    "applicability_domain": applicability_domain,
                    "delta_p20_ci_low": dp20_low,
                    "delta_enrichment20_ci_low": de20_low,
                }
                conclusion_md = build_branch_conclusion(
                    model_name,
                    feature_mode,
                    branch_name,
                    metrics_ci,
                    delta_ci,
                    audit_summary,
                    branch_data.rulebook_support,
                    branch_data.pair_rulebook_explanation_unstable_c3only,
                    status="COMPLETED",
                    skip_reason="",
                    applicability_domain=applicability_domain,
                )
                mechanism_md = extract_mechanism_markdown(model_spec, full_model, branch_data.X.columns.tolist()) if model_spec.nonlinear else "# linear model\n"
                branch_result = BranchArtifacts(
                    feature_mode=feature_mode,
                    feature_profile=dict(feature_profile),
                    metrics_by_split=metric_df,
                    metrics_ci=metrics_ci,
                    metrics_c3_slice_ci=metrics_c3_ci,
                    delta_ci=delta_ci,
                    audit_summary=audit_summary,
                    rulebook_support=branch_data.rulebook_support,
                    rulebook_support_engineered_comparison=branch_data.rulebook_support_engineered_comparison,
                    rulebook_legacy=branch_data.rulebook_legacy,
                    rulebook_debug=branch_data.rulebook_debug,
                    pair_rulebook_publishable_c3only=branch_data.pair_rulebook_publishable_c3only,
                    pair_rulebook_explanation_unstable_c3only=branch_data.pair_rulebook_explanation_unstable_c3only,
                    rulebook_model_derived_sensitivity=branch_data.rulebook_model_derived_sensitivity,
                    model_derived_cutpoint_alignment=branch_data.model_derived_cutpoint_alignment,
                    linear_continuous_effects=branch_data.linear_continuous_effects,
                    linear_pairwise_effects=branch_data.linear_pairwise_effects,
                    linear_vs_engineered_direction_check=branch_data.linear_vs_engineered_direction_check,
                    decision_payload=decision_payload,
                    conclusion_md=conclusion_md,
                    mechanism_md=mechanism_md,
                    status="COMPLETED",
                    skipped_by_config=False,
                    candidate_trace=branch_data.candidate_trace,
                    oof_predictions=pred_df,
                    fold_metrics=fold_metric_df,
                    audit_trace=audit_trace if not audit_trace.empty else audit_trace_df,
                    input_shift_metrics=input_shift_metrics,
                    score_shift_metrics=score_shift_metrics,
                    perf_shift_metrics=perf_shift_metrics,
                    tier_shift_matched_control=tier_shift_matched_control,
                )

            warning_present = bool(branch_warning_log.exists() and branch_warning_log.stat().st_size > 0)
            hard_pass_count = int((branch_result.audit_summary[branch_result.audit_summary["audit_name"].isin(HARD_AUDITS)]["status"] == "PASS").sum()) if not branch_result.audit_summary.empty else 0
            tier2d_status = (
                branch_result.audit_summary.loc[branch_result.audit_summary["audit_name"] == SOFT_AUDIT_NAME, "status"].iloc[0]
                if not branch_result.audit_summary.empty and (branch_result.audit_summary["audit_name"] == SOFT_AUDIT_NAME).any()
                else "PASS"
            )
            tier2d_level = (
                str(branch_result.audit_summary.loc[branch_result.audit_summary["audit_name"] == SOFT_AUDIT_NAME, "tier2d_level"].iloc[0])
                if not branch_result.audit_summary.empty and "tier2d_level" in branch_result.audit_summary.columns and (branch_result.audit_summary["audit_name"] == SOFT_AUDIT_NAME).any()
                else "PASS"
            )
            dp20_mean, dp20_low, dp20_high = get_delta_summary(branch_result.delta_ci, "P", 20)
            de20_mean, de20_low, de20_high = get_delta_summary(branch_result.delta_ci, "Enrichment", 20)
            branch_p20 = branch_result.metrics_ci[(branch_result.metrics_ci["metric"].astype(str) == "P") & (branch_result.metrics_ci["k"].astype(int) == 20)]
            branch_e20 = branch_result.metrics_ci[(branch_result.metrics_ci["metric"].astype(str) == "Enrichment") & (branch_result.metrics_ci["k"].astype(int) == 20)]
            branch_auc = branch_result.metrics_ci[(branch_result.metrics_ci["metric"].astype(str) == "AUC_proxy") & (branch_result.metrics_ci["k"].astype(int) == 0)]
            rulebook_reproducible = True
            if not branch_result.rulebook_support.empty:
                pred_rules = branch_result.rulebook_support[branch_result.rulebook_support["rule_type"].astype(str) == "prediction"].copy()
                missing_ok = not branch_result.rulebook_support["condition_text"].astype(str).str.contains("__MISSING__|missing_flag").any()
                cross_signal_ok = True
                for pred in pred_rules.itertuples(index=False):
                    if not stage3_reporting.has_feature(getattr(pred, "feature_b", "")) and not stage3_reporting.has_feature(getattr(pred, "feature_c", "")):
                        continue
                    groups = stage3_reporting.distinct_signal_groups(str(getattr(pred, "feature_a", "")), str(getattr(pred, "feature_b", "")), str(getattr(pred, "feature_c", "")))
                    if stage3_reporting.has_feature(getattr(pred, "feature_c", "")) and len(groups) < 3:
                        cross_signal_ok = False
                        break
                    if (not stage3_reporting.has_feature(getattr(pred, "feature_c", ""))) and len(groups) < 2:
                        cross_signal_ok = False
                        break
                rulebook_reproducible = bool(missing_ok and cross_signal_ok)
            interaction_upgrade_eligible, interaction_mode = interaction_upgrade_gate(
                branch_name,
                allow_interaction_promotion=model_spec.nonlinear,
                status=branch_result.status,
                hard_pass_count=hard_pass_count,
                warning_present=warning_present,
                rulebook_reproducible=rulebook_reproducible,
                delta_p20_ci_low=dp20_low,
                delta_enrichment20_ci_low=de20_low,
            )
            branch_result.decision_payload["interaction_upgrade_eligible"] = bool(interaction_upgrade_eligible)
            branch_result.decision_payload["interaction_mode"] = interaction_mode
            branch_result.decision_payload["warning_present"] = bool(warning_present)
            branch_result.decision_payload["rulebook_reproducible"] = bool(rulebook_reproducible)

            branch_result.metrics_by_split.to_csv(branch_dir / "metrics_by_split.csv", index=False)
            branch_result.metrics_ci.to_csv(branch_dir / "metrics_ci.csv", index=False)
            branch_result.metrics_c3_slice_ci.to_csv(branch_dir / "metrics_c3_slice_ci.csv", index=False)
            branch_result.delta_ci.to_csv(branch_dir / "delta_ci.csv", index=False)
            branch_result.audit_summary.to_csv(branch_dir / "audit_summaries.csv", index=False)
            branch_result.rulebook_support.to_csv(branch_dir / "rulebook_support.csv", index=False)
            branch_result.rulebook_support_engineered_comparison.to_csv(branch_dir / "rulebook_support_engineered_comparison.csv", index=False)
            branch_result.rulebook_legacy.to_csv(branch_dir / "rulebook_legacy_pair_tier2plus.csv", index=False)
            branch_result.pair_rulebook_publishable_c3only.to_csv(branch_dir / "pair_rulebook_publishable_c3only.csv", index=False)
            branch_result.pair_rulebook_explanation_unstable_c3only.to_csv(branch_dir / "pair_rulebook_explanation_unstable_c3only.csv", index=False)
            branch_result.rulebook_model_derived_sensitivity.to_csv(branch_dir / "rulebook_model_derived_sensitivity.csv", index=False)
            branch_result.model_derived_cutpoint_alignment.to_csv(branch_dir / "model_derived_cutpoint_alignment.csv", index=False)
            branch_result.linear_continuous_effects.to_csv(branch_dir / "linear_continuous_effects.csv", index=False)
            branch_result.linear_pairwise_effects.to_csv(branch_dir / "linear_pairwise_effects.csv", index=False)
            branch_result.linear_vs_engineered_direction_check.to_csv(branch_dir / "linear_vs_engineered_direction_check.csv", index=False)
            branch_result.input_shift_metrics.to_csv(branch_dir / "input_shift_metrics.csv", index=False)
            branch_result.score_shift_metrics.to_csv(branch_dir / "score_shift_metrics.csv", index=False)
            branch_result.perf_shift_metrics.to_csv(branch_dir / "perf_shift_metrics.csv", index=False)
            branch_result.tier_shift_matched_control.to_csv(branch_dir / "tier_shift_matched_control.csv", index=False)
            branch_result.rulebook_debug.to_csv(branch_dir / "rulebook_debug.csv", index=False)
            branch_result.candidate_trace.to_csv(branch_dir / "candidate_selection_trace.csv", index=False)
            branch_result.oof_predictions.to_csv(branch_dir / "predictions_oof.csv", index=False)
            branch_result.fold_metrics.to_csv(branch_dir / "fold_metrics.csv", index=False)
            branch_result.audit_trace.to_csv(branch_dir / "audit_trace.csv", index=False)
            (branch_dir / "run_decision.md").write_text(json.dumps(branch_result.decision_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (branch_dir / "run_conclusion_analysis_and_improvement.md").write_text(branch_result.conclusion_md, encoding="utf-8")
            (branch_dir / "rulebook_mechanism_extraction.md").write_text(branch_result.mechanism_md, encoding="utf-8")

            metrics_frames.append(branch_result.metrics_ci.assign(model=model_name, feature_mode=feature_mode))
            metrics_c3_frames.append(branch_result.metrics_c3_slice_ci.assign(model=model_name, feature_mode=feature_mode))
            audit_frames.append(branch_result.audit_summary.assign(model=model_name, feature_mode=feature_mode, branch=branch_name))

            selection_rows.append(
                {
                    "model": model_name,
                    "feature_mode": feature_mode,
                    "branch": branch_name,
                    "status": branch_result.status,
                    "skipped_by_config": bool(branch_result.skipped_by_config),
                    "n_cont": int(feature_profile.get("n_cont", 0)),
                    "n_bin_onehot": int(feature_profile.get("n_bin_onehot", 0)),
                    "n_missing_indicators": int(feature_profile.get("n_missing_indicators", 0)),
                    "x_columns_sha256": str(feature_profile.get("x_columns_sha256", "")),
                    "hard_audit_pass_count": hard_pass_count,
                    "tier2d_status": tier2d_status,
                    "tier2d_level": tier2d_level,
                    "publish_scope_final": str(branch_result.decision_payload.get("publish_scope_final", "C2C3")),
                    "rulebook_reproducible": bool(rulebook_reproducible),
                    "prediction_rule_count": int(branch_result.decision_payload.get("prediction_rule_count", 0)),
                    "delta_p20_mean": dp20_mean,
                    "delta_p20_ci_low": dp20_low,
                    "delta_p20_ci_high": dp20_high,
                    "delta_enrichment20_mean": de20_mean,
                    "delta_enrichment20_ci_low": de20_low,
                    "delta_enrichment20_ci_high": de20_high,
                    "p20_mean": float(branch_p20["mean"].iloc[0]) if not branch_p20.empty else np.nan,
                    "p20_ci_low": float(branch_p20["ci_low_95"].iloc[0]) if not branch_p20.empty else np.nan,
                    "enrichment20_mean": float(branch_e20["mean"].iloc[0]) if not branch_e20.empty else np.nan,
                    "enrichment20_ci_low": float(branch_e20["ci_low_95"].iloc[0]) if not branch_e20.empty else np.nan,
                    "auc_mean": float(branch_auc["mean"].iloc[0]) if not branch_auc.empty else np.nan,
                    "auc_ci_low": float(branch_auc["ci_low_95"].iloc[0]) if not branch_auc.empty else np.nan,
                    "warning_present": bool(warning_present),
                    "interaction_mode": interaction_mode,
                    "interaction_upgrade_eligible": bool(interaction_upgrade_eligible),
                    "publishable_c3_pair_count": int(len(branch_result.pair_rulebook_publishable_c3only)),
                    "unstable_explanation_pair_count": int(len(branch_result.pair_rulebook_explanation_unstable_c3only)),
                    "applicability_domain": str(branch_result.decision_payload.get("applicability_domain", "all")),
                    "skip_reason": str(branch_result.decision_payload.get("skip_reason", "")),
                }
            )
            append_log(run_log, f"branch_done model={model_name} branch={branch_name} status={branch_result.status}")

    selection_df = pd.DataFrame(selection_rows)
    if not selection_df.empty:
        selection_df["ci_evidence_ok"] = (selection_df["delta_p20_ci_low"].fillna(-np.inf) >= 0) & (selection_df["delta_enrichment20_ci_low"].fillna(-np.inf) >= 0)
        selection_df["status_completed"] = selection_df["status"].astype(str).eq("COMPLETED")
        selection_df["warning_free"] = ~selection_df["warning_present"].fillna(True)
        selection_df["tier2d_level_rank"] = selection_df["tier2d_level"].map({"PASS": 2, "WARN": 1, "FAIL": 0}).fillna(0).astype(int)
        selection_df = selection_df.sort_values(
            [
                "model",
                "feature_mode",
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
            ],
            ascending=[True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        ).reset_index(drop=True)
        selection_df["winner_role"] = "none"
        for _, grp in selection_df.groupby(["model", "feature_mode"], sort=False):
            selection_df.loc[int(grp.index[0]), "winner_role"] = "branch_winner"
    return (
        selection_df,
        pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame(),
        pd.concat(metrics_c3_frames, ignore_index=True) if metrics_c3_frames else pd.DataFrame(),
        pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame(),
    )


def summarize_branch_delta_from_metrics(base_metrics: pd.DataFrame, branch_metrics: pd.DataFrame, branch_name: str) -> pd.DataFrame:
    return summarize_delta_ci(base_metrics, branch_metrics, "gbdt", branch_name)


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
        if path.suffix.lower() == ".csv" and csv_is_empty(path):
            path.unlink()
            removed.append(str(path.relative_to(REPO_ROOT)))
    return removed


def build_model_zoo_markdown(selection_df: pd.DataFrame) -> str:
    lines = [
        "# Cont Plus Bin Model Zoo Comparison",
        "",
        "| model | branch | status | p20 | e20 | dP20_low | dE20_low | tier2d | interaction_mode | winner_role |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in selection_df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.model),
                    str(row.branch),
                    str(row.status),
                    fmt_float(float(getattr(row, "p20_mean", np.nan)), 4),
                    fmt_float(float(getattr(row, "enrichment20_mean", np.nan)), 4),
                    fmt_float(float(getattr(row, "delta_p20_ci_low", np.nan)), 4),
                    fmt_float(float(getattr(row, "delta_enrichment20_ci_low", np.nan)), 4),
                    str(getattr(row, "tier2d_level", "NA")),
                    str(getattr(row, "interaction_mode", "NA")),
                    str(getattr(row, "winner_role", "none")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_warning_summary(out_root: Path, selection_df: pd.DataFrame) -> dict[str, Any]:
    root_warning_log = out_root / "run_warning.log"
    root_lines = root_warning_log.read_text(encoding="utf-8").splitlines() if root_warning_log.exists() else []
    branches: list[dict[str, Any]] = []
    for row in selection_df.itertuples(index=False):
        branch_log = out_root / "models" / str(row.model) / str(row.feature_mode) / str(row.branch) / "run_warning.log"
        lines = branch_log.read_text(encoding="utf-8").splitlines() if branch_log.exists() else []
        branches.append(
            {
                "model": str(row.model),
                "feature_mode": str(row.feature_mode),
                "branch": str(row.branch),
                "warning_count": len(lines),
                "warning_present": len(lines) > 0,
                "sample": lines[:5],
            }
        )
    payload = {
        "root_warning_count": len(root_lines),
        "root_warning_present": len(root_lines) > 0,
        "root_sample": root_lines[:10],
        "branch_warnings": branches,
        "generated_at": stage3_reporting.now_iso(),
    }
    write_json(out_root / "warning_summary.json", payload)
    return payload


def build_run_manifest_payload(
    *,
    run_id: str,
    out_root: Path,
    mainline_root: Path,
    bin_control_root: Path,
    part1_out_dir: Path,
    alignment_summary: dict[str, Any],
    pair_limit_by_model: dict[str, int],
) -> dict[str, Any]:
    return {
        "run": {
            "run_id": run_id,
            "stage": "stage3_cont_plus_bin_extension",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "reference_roots": {
            "cont_mainline_root": str(mainline_root),
            "bin_control_root": str(bin_control_root),
        },
        "paths": {
            "part1_out_dir": str(part1_out_dir),
            "part2_out_dir": str(out_root),
        },
        "execution": {
            "feature_mode": "cont_plus_bin",
            "model_subset": [cfg["name"] for cfg in MODEL_CONFIGS],
            "branch_subset": BRANCH_ORDER,
            "pair_limit_by_model": pair_limit_by_model,
            "split_reuse_mode": "exact_split_indices_reuse",
            "audit_chain": ["negative_control", "controlled_missingness_parallel", SOFT_AUDIT_NAME, "rule_candidate_consistency"],
        },
        "alignment_summary": alignment_summary,
    }


def build_run_decision_markdown(selection_df: pd.DataFrame, *, mainline_root: Path, bin_control_root: Path, alignment_summary: dict[str, Any]) -> str:
    gbdt_rows = selection_df[(selection_df["model"].astype(str) == "gbdt") & (selection_df["winner_role"].astype(str) == "branch_winner")]
    logistic_rows = selection_df[(selection_df["model"].astype(str) == "logistic_l2") & (selection_df["winner_role"].astype(str) == "branch_winner")]
    gbdt = gbdt_rows.iloc[0] if not gbdt_rows.empty else pd.Series(dtype=object)
    logistic = logistic_rows.iloc[0] if not logistic_rows.empty else pd.Series(dtype=object)
    lines = [
        "# Cont Plus Bin Extension Run Decision",
        "",
        f"- reference_cont_mainline_root: {mainline_root}",
        f"- reference_bin_control_root: {bin_control_root}",
        "- feature_mode: cont_plus_bin",
        "- branches: mainline, mainline_plus_pairwise, mainline_plus_gated_3way",
        "- split_reuse_mode: exact_split_indices_reuse",
        f"- alignment_all_checked_columns_match: {str(bool(alignment_summary.get('all_checked_columns_match', False))).lower()}",
        f"- trainable_rows: {alignment_summary.get('trainable_rows', 'NA')}",
        f"- gbdt_branch_winner: {gbdt.get('branch', 'NA')}",
        f"- gbdt_branch_winner_p20: {fmt_float(float(gbdt.get('p20_mean', np.nan)), 4) if not gbdt.empty else 'NA'}",
        f"- gbdt_branch_winner_e20: {fmt_float(float(gbdt.get('enrichment20_mean', np.nan)), 4) if not gbdt.empty else 'NA'}",
        f"- logistic_branch_winner: {logistic.get('branch', 'NA')}",
        f"- logistic_branch_winner_p20: {fmt_float(float(logistic.get('p20_mean', np.nan)), 4) if not logistic.empty else 'NA'}",
        f"- logistic_branch_winner_e20: {fmt_float(float(logistic.get('enrichment20_mean', np.nan)), 4) if not logistic.empty else 'NA'}",
    ]
    return "\n".join(lines) + "\n"


def load_selection_summary(root: Path) -> pd.DataFrame:
    candidates = [root / "model_selection_summary.csv", root / "part2" / "model_selection_summary.csv"]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def load_generic_branch_row(root: Path, model: str, feature_mode: str, branch: str) -> pd.Series:
    df = load_selection_summary(root)
    if df.empty:
        return pd.Series(dtype=object)
    subset = df[
        (df["model"].astype(str) == model)
        & (df["feature_mode"].astype(str) == feature_mode)
        & (df["branch"].astype(str) == branch)
    ]
    return subset.iloc[0] if not subset.empty else pd.Series(dtype=object)


def load_metrics_topk_value(path: Path, metric: str, k: int) -> float:
    if not path.exists():
        return np.nan
    df = pd.read_csv(path)
    row = df[(df["metric"].astype(str) == metric) & (df["k"].astype(int) == int(k))]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["mean"])


def load_delta_topk_value(path: Path, metric: str, k: int) -> float:
    if not path.exists():
        return np.nan
    df = pd.read_csv(path)
    row = df[(df["metric"].astype(str) == metric) & (df["k"].astype(int) == int(k))]
    if row.empty:
        return np.nan
    if "delta_ci_low_95" in row.columns:
        return float(row.iloc[0]["delta_ci_low_95"])
    return np.nan


def load_dedicated_cont_gbdt_row(mainline_root: Path, branch: str) -> dict[str, Any]:
    cont_root = mainline_root / "part2" / "nonlinear_cont_only_v2"
    branch_dir = cont_root / "models" / "gbdt" / "cont_only" / ("mainline_plus_3way" if branch == TRIPLE_BRANCH else branch)
    metrics_path = branch_dir / "metrics_topk_ci.csv"
    run_summary_path = branch_dir / "run_summary.json"
    interaction_gain_path = cont_root / "interaction_gain_decision.csv"
    interaction_gain_3way_path = cont_root / "interaction_gain_3way_decision.csv"
    if branch == MAINLINE_BRANCH:
        interaction_mode = "baseline_only"
        dp20_low = 0.0
        de20_low = 0.0
    elif branch == PAIRWISE_BRANCH:
        gain_df = pd.read_csv(interaction_gain_path) if interaction_gain_path.exists() else pd.DataFrame()
        gain_row = gain_df[gain_df["model"].astype(str) == "gbdt"] if not gain_df.empty else pd.DataFrame()
        interaction_mode = "predictive_gain" if not gain_row.empty and bool(gain_row.iloc[0]["interaction_gain_positive"]) else "explanation_only"
        dp20_low = load_delta_topk_value(cont_root / "delta_topk_summary.csv", "P", 20)
        de20_low = load_delta_topk_value(cont_root / "delta_topk_summary.csv", "Enrichment", 20)
    else:
        gain_df = pd.read_csv(interaction_gain_3way_path) if interaction_gain_3way_path.exists() else pd.DataFrame()
        interaction_mode = "predictive_gain" if not gain_df.empty and bool(gain_df.iloc[0]["threeway_gain_positive"]) else "explanation_only"
        dp20_low = load_delta_topk_value(cont_root / "delta_3way_vs_pair_topk_summary.csv", "P", 20)
        de20_low = load_delta_topk_value(cont_root / "delta_3way_vs_pair_topk_summary.csv", "Enrichment", 20)
    return {
        "source": "current_cont_mainline_dedicated",
        "model": "gbdt",
        "feature_mode": "cont_only",
        "branch": branch,
        "status": "COMPLETED" if metrics_path.exists() else "MISSING",
        "p20_mean": load_metrics_topk_value(metrics_path, "P", 20),
        "enrichment20_mean": load_metrics_topk_value(metrics_path, "Enrichment", 20),
        "delta_p20_ci_low": dp20_low,
        "delta_enrichment20_ci_low": de20_low,
        "tier2d_level": "NA",
        "interaction_mode": interaction_mode,
        "artifact": str(branch_dir),
        "run_summary_exists": run_summary_path.exists(),
    }


def choose_row_with_fallback(primary_row: pd.Series, fallback_row: pd.Series, *, primary_source: str, fallback_source: str, branch: str) -> dict[str, Any]:
    row = primary_row if not primary_row.empty else fallback_row
    source = primary_source if not primary_row.empty else fallback_source
    if row.empty:
        return {
            "source": "missing",
            "branch": branch,
            "status": "MISSING",
            "p20_mean": np.nan,
            "enrichment20_mean": np.nan,
            "delta_p20_ci_low": np.nan,
            "delta_enrichment20_ci_low": np.nan,
            "tier2d_level": "NA",
            "interaction_mode": "NA",
            "artifact": "",
        }
    return {
        "source": source,
        "branch": branch,
        "status": str(row.get("status", "NA")),
        "p20_mean": float(row.get("p20_mean", np.nan)) if pd.notna(row.get("p20_mean", np.nan)) else np.nan,
        "enrichment20_mean": float(row.get("enrichment20_mean", np.nan)) if pd.notna(row.get("enrichment20_mean", np.nan)) else np.nan,
        "delta_p20_ci_low": float(row.get("delta_p20_ci_low", np.nan)) if pd.notna(row.get("delta_p20_ci_low", np.nan)) else np.nan,
        "delta_enrichment20_ci_low": float(row.get("delta_enrichment20_ci_low", np.nan)) if pd.notna(row.get("delta_enrichment20_ci_low", np.nan)) else np.nan,
        "tier2d_level": str(row.get("tier2d_level", "NA")),
        "interaction_mode": str(row.get("interaction_mode", "NA")),
        "artifact": "",
    }


def build_comparison_report(
    out_root: Path,
    *,
    mainline_root: Path,
    bin_control_root: Path,
    legacy_aligned_root: Path | None,
    legacy_split_aligned: bool,
) -> str:
    current_selection = pd.read_csv(out_root / "model_selection_summary.csv") if (out_root / "model_selection_summary.csv").exists() else pd.DataFrame()
    current_bin_selection = load_selection_summary(bin_control_root)
    legacy_root = legacy_aligned_root if legacy_aligned_root and legacy_split_aligned else None
    lines = [
        "# Cont Plus Bin vs Cont / Bin Comparison",
        "",
        f"- reference_cont_mainline_root: {mainline_root}",
        f"- reference_bin_control_root: {bin_control_root}",
        f"- current_extension_root: {out_root}",
        f"- legacy_aligned_generic_root: {legacy_root if legacy_root else 'not_used'}",
        f"- legacy_split_aligned: {str(bool(legacy_split_aligned)).lower()}",
        "- split_alignment: exact reuse of current cont_mainline split_indices.json for the new cont_plus_bin extension",
        "- audit_chain_alignment: negative_control + controlled_missingness_parallel + tier2d/C2C3_stability + rule_candidate_consistency",
        "",
        "## GBDT View",
        "",
        "| feature_mode | branch | source | status | P@20 | E20 | delta_low_ref | interaction_mode | tier2d |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for branch in BRANCH_ORDER:
        cont_row = load_dedicated_cont_gbdt_row(mainline_root, branch)
        lines.append(
            "| "
            + " | ".join(
                [
                    "cont_only",
                    branch,
                    str(cont_row["source"]),
                    str(cont_row["status"]),
                    fmt_float(float(cont_row["p20_mean"]), 4),
                    fmt_float(float(cont_row["enrichment20_mean"]), 4),
                    fmt_float(float(cont_row["delta_p20_ci_low"]) if branch != TRIPLE_BRANCH else float(cont_row["delta_p20_ci_low"]), 4),
                    str(cont_row["interaction_mode"]),
                    str(cont_row["tier2d_level"]),
                ]
            )
            + " |"
        )
    for branch in BRANCH_ORDER:
        primary_row = load_generic_branch_row(bin_control_root, "gbdt", "bin_only", branch)
        fallback_row = load_generic_branch_row(legacy_root, "gbdt", "bin_only", branch) if legacy_root else pd.Series(dtype=object)
        row = choose_row_with_fallback(primary_row, fallback_row, primary_source="current_bin_control", fallback_source="legacy_aligned_generic", branch=branch)
        lines.append(
            "| "
            + " | ".join(
                [
                    "bin_only",
                    branch,
                    str(row["source"]),
                    str(row["status"]),
                    fmt_float(float(row["p20_mean"]), 4),
                    fmt_float(float(row["enrichment20_mean"]), 4),
                    fmt_float(float(row["delta_p20_ci_low"]), 4),
                    str(row["interaction_mode"]),
                    str(row["tier2d_level"]),
                ]
            )
            + " |"
        )
    for branch in BRANCH_ORDER:
        row_df = current_selection[
            (current_selection["model"].astype(str) == "gbdt")
            & (current_selection["feature_mode"].astype(str) == "cont_plus_bin")
            & (current_selection["branch"].astype(str) == branch)
        ]
        row = row_df.iloc[0] if not row_df.empty else pd.Series(dtype=object)
        lines.append(
            "| "
            + " | ".join(
                [
                    "cont_plus_bin",
                    branch,
                    "current_extension",
                    str(row.get("status", "MISSING")),
                    fmt_float(float(row.get("p20_mean", np.nan)), 4),
                    fmt_float(float(row.get("enrichment20_mean", np.nan)), 4),
                    fmt_float(float(row.get("delta_p20_ci_low", np.nan)), 4),
                    str(row.get("interaction_mode", "NA")),
                    str(row.get("tier2d_level", "NA")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Logistic View",
            "",
            "| feature_mode | branch | source | status | P@20 | E20 | delta_low_ref | interaction_mode | tier2d |",
            "|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for branch in BRANCH_ORDER:
        row = load_generic_branch_row(legacy_root, "logistic_l2", "cont_only", branch) if legacy_root else pd.Series(dtype=object)
        picked = choose_row_with_fallback(row, pd.Series(dtype=object), primary_source="legacy_aligned_generic", fallback_source="missing", branch=branch)
        lines.append(
            "| "
            + " | ".join(
                [
                    "cont_only",
                    branch,
                    str(picked["source"]),
                    str(picked["status"]),
                    fmt_float(float(picked["p20_mean"]), 4),
                    fmt_float(float(picked["enrichment20_mean"]), 4),
                    fmt_float(float(picked["delta_p20_ci_low"]), 4),
                    str(picked["interaction_mode"]),
                    str(picked["tier2d_level"]),
                ]
            )
            + " |"
        )
    for branch in BRANCH_ORDER:
        primary_row = load_generic_branch_row(bin_control_root, "logistic_l2", "bin_only", branch)
        fallback_row = load_generic_branch_row(legacy_root, "logistic_l2", "bin_only", branch) if legacy_root else pd.Series(dtype=object)
        picked = choose_row_with_fallback(primary_row, fallback_row, primary_source="current_bin_control", fallback_source="legacy_aligned_generic", branch=branch)
        lines.append(
            "| "
            + " | ".join(
                [
                    "bin_only",
                    branch,
                    str(picked["source"]),
                    str(picked["status"]),
                    fmt_float(float(picked["p20_mean"]), 4),
                    fmt_float(float(picked["enrichment20_mean"]), 4),
                    fmt_float(float(picked["delta_p20_ci_low"]), 4),
                    str(picked["interaction_mode"]),
                    str(picked["tier2d_level"]),
                ]
            )
            + " |"
        )
    for branch in BRANCH_ORDER:
        row_df = current_selection[
            (current_selection["model"].astype(str) == "logistic_l2")
            & (current_selection["feature_mode"].astype(str) == "cont_plus_bin")
            & (current_selection["branch"].astype(str) == branch)
        ]
        row = row_df.iloc[0] if not row_df.empty else pd.Series(dtype=object)
        lines.append(
            "| "
            + " | ".join(
                [
                    "cont_plus_bin",
                    branch,
                    "current_extension",
                    str(row.get("status", "MISSING")),
                    fmt_float(float(row.get("p20_mean", np.nan)), 4),
                    fmt_float(float(row.get("enrichment20_mean", np.nan)), 4),
                    fmt_float(float(row.get("delta_p20_ci_low", np.nan)), 4),
                    str(row.get("interaction_mode", "NA")),
                    str(row.get("tier2d_level", "NA")),
                ]
            )
            + " |"
        )

    current_delta_3way = out_root / "delta_3way_vs_pair_topk_summary.csv"
    if current_delta_3way.exists():
        df = pd.read_csv(current_delta_3way)
        p20_low = load_delta_topk_value(current_delta_3way, "P", 20)
        e20_low = load_delta_topk_value(current_delta_3way, "Enrichment", 20)
        lines.extend(
            [
                "",
                "## Current Extension 3way vs Pair",
                "",
                f"- gbdt delta_p20_ci_low(3way_vs_pair): {fmt_float(p20_low, 4)}",
                f"- gbdt delta_enrichment20_ci_low(3way_vs_pair): {fmt_float(e20_low, 4)}",
                f"- rows_available: {int(len(df))}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent cont_plus_bin extension runner aligned to the completed cont/bin results.")
    parser.add_argument(
        "--mainline-root",
        default="artifacts/stage3_nonlinear_mainrule_n1_monotonic_chain_20260306_223004",
        help="Reference cont_only mainline run root.",
    )
    parser.add_argument(
        "--bin-control-root",
        default="artifacts/stage3_bin_only_control_against_stage3_nonlinear_mainrule_n1_monotonic_chain_20260306_223004_20260307_101114",
        help="Reference isolated bin_only control root.",
    )
    parser.add_argument(
        "--legacy-aligned-root",
        default="artifacts/stage3/stage3_r5_part1_20260305_175809_0305/part2",
        help="Aligned legacy generic part2 root used only for missing comparison cells.",
    )
    parser.add_argument(
        "--out-root",
        default="",
        help="Output root for this isolated cont_plus_bin extension run.",
    )
    parser.add_argument(
        "--part1-out-dir",
        default="",
        help="Optional override for the shared Stage3 part1 directory. Defaults to the path declared by the reference mainline manifest.",
    )
    args = parser.parse_args()

    original_make_estimator = patch_aligned_estimators()
    try:
        mainline_root = resolve_repo_path(REPO_ROOT, args.mainline_root)
        bin_control_root = resolve_repo_path(REPO_ROOT, args.bin_control_root)
        legacy_aligned_root = resolve_repo_path(REPO_ROOT, args.legacy_aligned_root) if args.legacy_aligned_root else None
        if not mainline_root.exists():
            raise FileNotFoundError(f"Reference mainline root not found: {mainline_root}")
        if not bin_control_root.exists():
            raise FileNotFoundError(f"Reference bin control root not found: {bin_control_root}")

        manifest_path = mainline_root / "manifest_monotonic_chain.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Reference manifest not found: {manifest_path}")
        manifest = load_simple_yaml(manifest_path)
        reference_execution = manifest.get("execution", {})
        reference_run = manifest.get("run", {})
        reference_paths = manifest.get("paths", {})

        if args.part1_out_dir:
            part1_out_dir = resolve_repo_path(REPO_ROOT, args.part1_out_dir)
        else:
            part1_out_dir = resolve_repo_path(REPO_ROOT, str(reference_paths.get("part1_out_dir")))
        out_root = resolve_repo_path(REPO_ROOT, args.out_root) if args.out_root else make_default_out_root(mainline_root)
        ensure_dir(out_root)
        ensure_dir(out_root / "logs")
        run_log = out_root / "logs" / "run.log"
        root_warning_log = out_root / "run_warning.log"
        ensure_file(root_warning_log)

        append_log(run_log, f"reference_mainline_root={mainline_root}")
        append_log(run_log, f"reference_bin_control_root={bin_control_root}")
        append_log(run_log, f"part1_out_dir={part1_out_dir}")

        cont_file = part1_out_dir / "interaction_feature_view_cont_only.csv"
        bin_file = part1_out_dir / "interaction_feature_view_bin_only.csv"
        cpb_file = part1_out_dir / "interaction_feature_view_cont_plus_bin.csv"
        gate_file = part1_out_dir / "gate_c_acceptance.json"
        for path in (cont_file, bin_file, cpb_file, gate_file):
            if not path.exists():
                raise FileNotFoundError(f"Required input missing: {path}")
        gate = load_json_any(gate_file)
        if gate.get("gate_c_status") != "PASS":
            raise RuntimeError(f"Gate C not PASS: {gate}")

        cont_df = pd.read_csv(cont_file)
        bin_df = pd.read_csv(bin_file)
        cpb_df = pd.read_csv(cpb_file)
        alignment_summary = assert_trainable_alignment(cont_df, bin_df, cpb_df)
        append_log(run_log, "trainable_alignment=PASS")

        df_trainable = cpb_df[cpb_df["base_non_missing_count"].fillna(0).astype(float) >= 2].copy().reset_index(drop=True)
        y = is_strict_positive(df_trainable).to_numpy(dtype=int)

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

        include_continuous, include_bins = stage3_features.feature_mode_flags("cont_plus_bin")
        base_X, base_feature_meta = stage3_features.prepare_base_features(
            df_trainable.copy(),
            include_missing_indicators=True,
            include_year_sensitivity=False,
            include_continuous=include_continuous,
            include_bins=include_bins,
        )
        feature_profile = stage3_features.summarize_feature_frame(base_X)
        base_feature_sets = {"cont_plus_bin": (base_X, base_feature_meta, feature_profile)}

        pair_candidates = stage3_candidates.generate_pairwise_candidates(
            df_trainable,
            y,
            splits,
            candidate_scope=PAIR_CANDIDATE_SCOPE,
        )
        pair_candidates.to_csv(out_root / "interaction_candidates_pairwise.csv", index=False)
        pair_candidates.to_csv(out_root / "pair_candidates_discovery.csv", index=False)
        threshold = build_primary_threshold(reference_execution)
        selected_pairs_l0, _ = stage3_candidates.classify_candidates(
            shortlist_candidate_pool(pair_candidates, pair_limit=PAIR_DISCOVERY_LIMIT, triple_limit=TRIPLE_DISCOVERY_LIMIT),
            threshold,
            pair_limit=max(int(cfg["pair_limit"]) for cfg in MODEL_CONFIGS),
            triple_limit=max(int(cfg["triple_limit"]) for cfg in MODEL_CONFIGS),
            enforce_cross_signal_publishable=True,
        )
        hierarchy_pairs = selected_pairs_l0[(selected_pairs_l0["type"].astype(str) == "pair") & (selected_pairs_l0["rule_type"].astype(str) == "prediction")].copy()
        triple_candidates = stage3_candidates.generate_triple_candidates(
            df_trainable,
            y,
            splits,
            hierarchy_pairs,
            candidate_scope=TRIPLE_CANDIDATE_SCOPE,
        )
        triple_candidates.to_csv(out_root / "interaction_candidates_3way.csv", index=False)

        top_ks = list(reference_execution.get("top_ks", TOP_K_DEFAULT))
        unstable_explanation_config = dict(DEFAULT_UNSTABLE_EXPLANATION)
        pair_limit_by_model = {cfg["name"]: int(cfg["pair_limit"]) for cfg in MODEL_CONFIGS}
        selection_df, metrics_summary_df, metrics_c3_summary_df, audit_summary_df = execute_model_matrix(
            out_root,
            df_trainable,
            y,
            splits,
            random_seed=int(reference_run.get("random_seed", 20260305)),
            threshold=threshold,
            pair_candidates=pair_candidates,
            triple_candidates=triple_candidates,
            base_feature_sets=base_feature_sets,
            model_configs=MODEL_CONFIGS,
            unstable_explanation_config=unstable_explanation_config,
            top_ks=top_ks,
            tier2d_common_k_min=int(reference_execution.get("common_k_min", 15)),
            tier2d_min_test_c2_n=int(reference_execution.get("min_test_c2_n", 25)),
            tier2d_min_test_c3_n=int(reference_execution.get("min_test_c3_n", 20)),
            tier2d_raw_diff_eps=float(reference_execution.get("raw_diff_eps", 0.02)),
            run_log=run_log,
        )

        metrics_summary_df.to_csv(out_root / "interaction_metrics_summary_ci.csv", index=False)
        metrics_c3_summary_df.to_csv(out_root / "interaction_metrics_c3_slice_ci.csv", index=False)
        audit_summary_df.to_csv(out_root / "interaction_audit_linkage_summary.csv", index=False)
        selection_df.to_csv(out_root / "model_selection_summary.csv", index=False)
        (out_root / "model_zoo_comparison.md").write_text(build_model_zoo_markdown(selection_df), encoding="utf-8")
        build_warning_summary(out_root, selection_df)

        gbdt_main_metrics = out_root / "models" / "gbdt" / "cont_plus_bin" / MAINLINE_BRANCH / "metrics_by_split.csv"
        gbdt_pair_metrics = out_root / "models" / "gbdt" / "cont_plus_bin" / PAIRWISE_BRANCH / "metrics_by_split.csv"
        gbdt_three_metrics = out_root / "models" / "gbdt" / "cont_plus_bin" / TRIPLE_BRANCH / "metrics_by_split.csv"
        if gbdt_pair_metrics.exists() and gbdt_three_metrics.exists():
            pair_df = pd.read_csv(gbdt_pair_metrics)
            three_df = pd.read_csv(gbdt_three_metrics)
            delta_three_vs_pair = summarize_branch_delta_from_metrics(pair_df, three_df, TRIPLE_BRANCH)
            delta_three_vs_pair.to_csv(out_root / "delta_3way_vs_pair_topk_summary.csv", index=False)
        if gbdt_main_metrics.exists() and gbdt_pair_metrics.exists():
            main_df = pd.read_csv(gbdt_main_metrics)
            pair_df = pd.read_csv(gbdt_pair_metrics)
            delta_pair_vs_main = summarize_branch_delta_from_metrics(main_df, pair_df, PAIRWISE_BRANCH)
            delta_pair_vs_main.to_csv(out_root / "delta_pair_vs_main_topk_summary.csv", index=False)

        manifest_payload = build_run_manifest_payload(
            run_id=out_root.name,
            out_root=out_root,
            mainline_root=mainline_root,
            bin_control_root=bin_control_root,
            part1_out_dir=part1_out_dir,
            alignment_summary=alignment_summary,
            pair_limit_by_model=pair_limit_by_model,
        )
        write_json(out_root / "run_manifest.json", manifest_payload)
        (out_root / "run_decision.md").write_text(
            build_run_decision_markdown(
                selection_df,
                mainline_root=mainline_root,
                bin_control_root=bin_control_root,
                alignment_summary=alignment_summary,
            ),
            encoding="utf-8",
        )

        legacy_split_aligned = False
        if legacy_aligned_root and legacy_aligned_root.exists():
            legacy_split_dir = legacy_aligned_root / "splits"
            legacy_split_aligned = split_bundle_matches(split_src_dir, legacy_split_dir)
            append_log(run_log, f"legacy_split_aligned={str(legacy_split_aligned).lower()}")

        comparison_md = build_comparison_report(
            out_root,
            mainline_root=mainline_root,
            bin_control_root=bin_control_root,
            legacy_aligned_root=legacy_aligned_root,
            legacy_split_aligned=legacy_split_aligned,
        )
        (out_root / "cont_plus_bin_vs_cont_bin_mainline_comparison.md").write_text(comparison_md, encoding="utf-8")

        removed = prune_outputs(out_root)
        write_json(out_root / "prune_summary.json", {"removed_files": removed, "removed_count": len(removed)})
        append_log(run_log, f"prune_removed={len(removed)}")
        append_log(run_log, "run_complete=PASS")
    finally:
        stage3_models.make_estimator = original_make_estimator


if __name__ == "__main__":
    main()
