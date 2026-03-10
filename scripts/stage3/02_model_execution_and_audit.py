#!/usr/bin/env python3
"""Stage3 Part II main entrypoint.

This script owns the pipeline-level orchestration:
- load the manifest and Stage3 Part I outputs
- build the model x feature_mode x branch execution matrix
- run split generation, sensitivity runs, and root-level summaries

Step-specific logic lives in `stage3_workflow_isolated/step01~step05.py` and
is imported below. Extend those modules for feature/candidate/model/audit/
reporting behavior instead of reintroducing local duplicates here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from stage3_workflow_isolated import step04_audits as stage3_audits
from stage3_workflow_isolated import step02_candidates as stage3_candidates
from common import append_log, ensure_dir, load_simple_yaml, read_json, repo_root_from_file, resolve_repo_path
from stage3_workflow_isolated import step01_features as stage3_features
from stage3_workflow_isolated import step03_models as stage3_models
from stage3_workflow_isolated import step05_reporting as stage3_reporting

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:  # pragma: no cover - fallback handled at runtime
    ExplainableBoostingClassifier = None  # type: ignore[assignment]

CAT_BASE_COLS = [
    "power_mw_bin",
    "rack_kw_typical_bin",
    "pue_bin",
    "cooling_norm",
    "liquid_cool_binary",
    "building_sqm_bin",
]
CONTINUOUS_SOURCE_COLS = [
    "power_mw",
    "rack_kw_typical",
    "rack_kw_peak",
    "rack_density_area_w_per_sf_dc",
    "pue",
    "building_sqm",
    "whitespace_sqm",
]
NUM_BASE_COLS = [
    "power_mw_is_missing",
    "rack_kw_typical_is_missing",
    "pue_is_missing",
    "cooling_is_missing",
    "liquid_cool_is_missing",
    "building_sqm_is_missing",
    "base_non_missing_count",
]
MISSING_FLAG_PREFIXES = (
    "power_mw_is_missing",
    "rack_kw_typical_is_missing",
    "pue_is_missing",
    "cooling_is_missing",
    "liquid_cool_is_missing",
    "building_sqm_is_missing",
    "year_is_missing",
)
LEAKAGE_PREFIXES = ("llm_", "accel_")
LEAKAGE_EXACT = {"stage", "type", "level", "year"}
PAIR_LIMIT = 4
TRIPLE_LIMIT = 0
PAIR_DISCOVERY_LIMIT = 12
TRIPLE_DISCOVERY_LIMIT = 0
PAIR_CANDIDATE_SCOPE = "C2C3"
PAIR_PUBLISH_SCOPE = "C3_only"
TRIPLE_CANDIDATE_SCOPE = "C2C3"
DEFAULT_FEATURE_MODES = ["cont_only", "bin_only"]
DEFAULT_CONT_PLUS_BIN_MODEL_SUBSET = {"gbdt", "logistic_l2"}
PRIMARY_WINNER_FEATURE_MODE = "cont_only"
CONTROL_WINNER_FEATURE_MODE = "bin_only"
PRIMARY_WINNER_MODELS = {"gbdt", "ebm"}
MAINLINE_BRANCH = "mainline"
PAIRWISE_BRANCH = "mainline_plus_pairwise"
TRIPLE_BRANCH = "mainline_plus_gated_3way"
TOP_KS = [10, 20, 30, 50]
SOFT_AUDIT_NAME = "tier2d/C2C3_stability"
HARD_AUDITS = ["negative_control", "controlled_missingness_parallel", "rule_candidate_consistency"]
TIER2D_COMMON_K_MIN = 15
TIER2D_MIN_TEST_C2_N = 25
TIER2D_MIN_TEST_C3_N = 20
TIER2D_RAW_DIFF_EPS = 0.02
SPLIT_MIN_TEST_C2_N = 25
SPLIT_MIN_TEST_C3_N = 20
SIGNAL_GROUP_MAPPING = {
    "power_mw": "power",
    "power_mw_bin": "power",
    "rack_kw_typical": "rack_power_density",
    "rack_kw_typical_bin": "rack_power_density",
    "rack_kw_peak": "rack_power_density",
    "rack_kw_peak_bin": "rack_power_density",
    "rack_density_area_w_per_sf_dc": "rack_power_density",
    "rack_density": "rack_power_density",
    "cooling_norm": "cooling",
    "liquid_cool_binary": "cooling",
    "liquid_immersion": "cooling",
    "water_based_air": "cooling",
    "hybrid_air_liquid": "cooling",
    "liquid_direct_or_loop": "cooling",
    "pue": "efficiency",
    "pue_bin": "efficiency",
    "building_sqm_bin": "space",
    "building_sqm": "space",
    "whitespace_sqm": "space",
}
ENFORCE_CROSS_SIGNAL_PUBLISHABLE = True
EXPORT_DEBUG_ARTIFACTS = True
LEGACY_RULEBOOK_COLUMNS = [
    "pair_rank",
    "feature_a",
    "feature_b",
    "condition_text",
    "coverage",
    "enrichment",
    "stability_freq",
    "support",
    "support_min",
    "pos_hits",
    "pos_hits_min",
    "support_fold_count",
    "rule_strength",
    "rule_use",
    "downgrade_reason",
    "candidate_train_scope",
    "publish_scope",
    "notes",
]
SUPPORT_RULEBOOK_COLUMNS = [
    "rule_rank",
    "rule_type",
    "feature_a",
    "feature_b",
    "feature_c",
    "condition_text",
    "coverage",
    "enrichment",
    "stability_freq",
    "rule_tier_min",
    "support",
    "pos_hits",
    "notes",
    "rule_id",
]
C3_PAIR_RULEBOOK_COLUMNS = [
    "rule_rank",
    "feature_a",
    "feature_b",
    "condition_text",
    "coverage_c3",
    "enrichment_c3",
    "support_c2",
    "enrichment_c2",
    "support_c3",
    "delta_enrichment_c3_minus_c2",
    "stability_freq",
    "candidate_train_scope",
    "publish_scope",
    "signal_group_a",
    "signal_group_b",
    "notes",
    "rule_id",
]
UNSTABLE_EXPLANATION_PAIR_COLUMNS = [
    "rule_rank",
    "feature_a",
    "feature_b",
    "condition_text",
    "coverage_c3",
    "enrichment_c3",
    "stability_freq",
    "support_n_by_fold_min",
    "support_pos_by_fold_min",
    "candidate_train_scope",
    "publish_scope",
    "signal_group_a",
    "signal_group_b",
    "evidence_status",
    "explanation_only",
    "why_unstable",
    "notes",
    "rule_id",
]
EXPLAIN_MODEL_SELECTION_COLUMNS = [
    "source_explain_model_name",
    "target_model_name",
    "feature_mode",
    "target_branch_name",
    "feature_a",
    "feature_b",
    "feature_c",
    "support_n_by_fold_min",
    "support_pos_by_fold_min",
    "selection_freq",
    "stability_freq",
    "notes",
]
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
DEFAULT_NONLINEAR_CONT_ONLY_RULEBOOK_POLICY = {
    "primary_source": "model_derived",
    "engineered_comparison_enabled": True,
    "min_support_n": "auto",
    "min_support_pos": "auto",
    "min_enrichment": "auto",
    "max_rules_per_feature": 3,
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
class BranchArtifacts:
    feature_mode: str
    feature_profile: dict[str, Any]
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
    conclusion_zh: str
    conclusion_en: str
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
    applicability_domain: str = "all"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(safe_json_dumps(payload) + "\n", encoding="utf-8")


def ensure_file(path: Path, content: str = "") -> None:
    ensure_dir(path.parent)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def git_hash(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return ""


def is_strict_positive(df: pd.DataFrame) -> pd.Series:
    label_ok = df["llm_ai_dc_label"].isin(["ai_specific", "ai_optimized"])
    accel_ok = df["accel_model"].notna() | df["accel_count"].notna()
    return (label_ok & accel_ok).astype(int)


def normalize_pair_candidate_scope(scope: Any) -> str:
    text = str(scope or "C2C3").strip().upper()
    if text in {"C2C3", "C2_C3", "C2+C3", "TIER2PLUS", "TIER-2+"}:
        return "C2C3"
    if text in {"C1C2C3", "C1_C2_C3", "C1+C2+C3", "TIER1PLUS", "TIER-1+"}:
        return "C1C2C3"
    return "C2C3"


def normalize_triple_candidate_scope(scope: Any) -> str:
    text = str(scope or "C2C3").strip().upper()
    if text in {"DISABLED", "OFF", "NONE", "FALSE", "0"}:
        return "DISABLED"
    if text in {"C2C3", "C2_C3", "C2+C3"}:
        return "C2C3"
    return "C2C3"


def infer_default_pair_publish_scope(manifest: dict[str, Any]) -> str:
    fallback_scope = str(manifest.get("fallback_policy", {}).get("publish_scope", "")).strip().lower()
    if "c1c2c3" in fallback_scope or "c1_c2_c3" in fallback_scope:
        return "C1C2C3"
    if "c2c3" in fallback_scope or "c2_c3" in fallback_scope:
        return "C2C3"
    if "c3" in fallback_scope:
        return "C3_only"
    return "C3_only"


def normalize_pair_publish_scope(scope: Any, manifest: dict[str, Any]) -> str:
    text = str(scope or "").strip().upper()
    if not text:
        return infer_default_pair_publish_scope(manifest)
    if text in {"C3_ONLY", "C3"}:
        return "C3_only"
    if text in {"C2C3", "C2_C3", "C2+C3"}:
        return "C2C3"
    if text in {"C1C2C3", "C1_C2_C3", "C1+C2+C3", "TIER1PLUS", "TIER-1+"}:
        return "C1C2C3"
    return infer_default_pair_publish_scope(manifest)


def compute_tier_balance_metrics(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    soft_config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = soft_config or {}
    enabled = bool(config.get("enabled", False))
    test_tiers = df.iloc[test_idx]["coverage_tier"].astype(str)
    c2_n = int((test_tiers == "C2").sum())
    c3_n = int((test_tiers == "C3").sum())
    c23_total = c2_n + c3_n
    c2_share = float(c2_n / c23_total) if c23_total > 0 else np.nan
    c3_share = float(c3_n / c23_total) if c23_total > 0 else np.nan
    share_range = config.get("test_c2_c3_share_range", [0.30, 0.70])
    low = float(share_range[0]) if isinstance(share_range, list) and len(share_range) >= 1 else 0.30
    high = float(share_range[1]) if isinstance(share_range, list) and len(share_range) >= 2 else 0.70
    min_c2 = int(config.get("min_test_c2_n", 0))
    min_c3 = int(config.get("min_test_c3_n", 0))
    share_ok = bool(c23_total > 0 and np.isfinite(c2_share) and np.isfinite(c3_share) and low <= c2_share <= high and low <= c3_share <= high)
    counts_ok = bool(c2_n >= min_c2 and c3_n >= min_c3)
    soft_ok = bool((share_ok or counts_ok) if enabled else True)
    closeness = 1.0 - abs((c2_share if np.isfinite(c2_share) else 0.0) - 0.5) * 2.0 if c23_total > 0 else 0.0
    score = float((2.0 if counts_ok else 0.0) + (1.0 if share_ok else 0.0) + max(0.0, closeness))
    return {
        "test_c2_n": c2_n,
        "test_c3_n": c3_n,
        "test_c2_share": c2_share,
        "test_c3_share": c3_share,
        "tier_balance_score": score,
        "soft_tier_balance_satisfied": soft_ok,
        "share_range_satisfied": share_ok,
        "count_floor_satisfied": counts_ok,
    }


def fmt_float(value: float, digits: int = 6) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def get_threshold_grid() -> list[ThresholdSpec]:
    grid: list[ThresholdSpec] = []
    idx = 1
    for support_n in [15, 20]:
        for support_pos in [2, 3]:
            for pair_freq in [0.70, 0.80]:
                triple_freq = 0.80 if pair_freq <= 0.70 else 0.90
                grid.append(
                    ThresholdSpec(
                        threshold_id=f"GRID_{idx}",
                        support_n=support_n,
                        support_pos=support_pos,
                        pair_freq=pair_freq,
                        triple_freq=triple_freq,
                        sensitivity_only=True,
                    )
                )
                idx += 1
    return grid


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
        sensitivity_only=False,
        pair_support_n=pair_support_n,
        pair_support_pos=pair_support_pos,
        triple_support_n=triple_support_n,
        triple_support_pos=triple_support_pos,
    )


def _int_or_auto(value: Any, auto_value: int, *, minimum: int = 1) -> int:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return int(max(minimum, auto_value))
    if value is None:
        return int(max(minimum, auto_value))
    return int(max(minimum, int(value)))


def _float_or_auto(value: Any, auto_value: float, *, minimum: float = 0.0) -> float:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return float(max(minimum, auto_value))
    if value is None:
        return float(max(minimum, auto_value))
    return float(max(minimum, float(value)))


def resolve_nonlinear_cont_only_rulebook_policy(
    execution: dict[str, Any],
    *,
    n_rows: int,
    n_pos: int,
) -> dict[str, Any]:
    raw = execution.get("nonlinear_cont_only_rulebook", {})
    if not isinstance(raw, dict):
        raw = {}
    base_rate = float(n_pos / n_rows) if n_rows > 0 else 0.0
    auto_support_n = max(10, int(math.ceil(float(max(1, n_rows)) * 0.03)))
    auto_support_pos = max(2, int(math.ceil(float(auto_support_n) * max(0.08, base_rate * 0.40))))
    auto_min_enrichment = 1.10 if n_rows < 1000 else 1.05
    policy = dict(DEFAULT_NONLINEAR_CONT_ONLY_RULEBOOK_POLICY)
    policy.update(raw)
    policy_resolved = {
        "primary_source": str(policy.get("primary_source", "model_derived")).strip().lower() or "model_derived",
        "engineered_comparison_enabled": bool(policy.get("engineered_comparison_enabled", True)),
        "min_support_n": _int_or_auto(policy.get("min_support_n", "auto"), auto_support_n, minimum=1),
        "min_support_pos": _int_or_auto(policy.get("min_support_pos", "auto"), auto_support_pos, minimum=1),
        "min_enrichment": _float_or_auto(policy.get("min_enrichment", "auto"), auto_min_enrichment, minimum=0.0),
        "max_rules_per_feature": _int_or_auto(policy.get("max_rules_per_feature", 3), 3, minimum=1),
    }
    return policy_resolved


def build_company_splits(
    df: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    *,
    n_repeats: int = 100,
    test_company_frac: float = 0.20,
    min_test_pos: int = 5,
    min_train_pos: int = 20,
    min_test_c2_n: int = 25,
    min_test_c3_n: int = 20,
    soft_tier_balance: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    rng = np.random.default_rng(seed)
    companies = df["company"].fillna("__UNKNOWN_COMPANY__").astype(str).unique()
    n_test_companies = max(1, int(round(len(companies) * test_company_frac)))

    candidate_splits: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(1000, n_repeats * 60)
    company_series = df["company"].fillna("__UNKNOWN_COMPANY__").astype(str)
    tier_series = df["coverage_tier"].astype(str)
    seen_signatures: set[tuple[str, ...]] = set()

    while len(candidate_splits) < max(n_repeats * 4, n_repeats) and attempts < max_attempts:
        attempts += 1
        sampled = rng.choice(companies, size=n_test_companies, replace=False)
        test_companies = set(sampled.tolist())
        signature = tuple(sorted(test_companies))
        if signature in seen_signatures:
            continue
        test_mask = company_series.isin(test_companies).to_numpy()
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        train_pos = int(y[train_idx].sum())
        test_pos = int(y[test_idx].sum())
        if train_pos < min_train_pos or test_pos < min_test_pos:
            continue
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        test_tiers = tier_series.iloc[test_idx]
        test_c2_n = int((test_tiers == "C2").sum())
        test_c3_n = int((test_tiers == "C3").sum())
        if test_c2_n < int(min_test_c2_n) or test_c3_n < int(min_test_c3_n):
            continue
        seen_signatures.add(signature)
        candidate_splits.append(
            {
                "split_id": len(candidate_splits),
                "repeat_id": len(candidate_splits),
                "random_seed": seed,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "train_pos": train_pos,
                "test_pos": test_pos,
                "train_n": len(train_idx),
                "test_n": len(test_idx),
                "test_c2_n": test_c2_n,
                "test_c3_n": test_c3_n,
                "train_companies": sorted(set(company_series.iloc[train_idx].tolist())),
                "test_companies": sorted(test_companies),
            }
        )
    if not candidate_splits:
        return [], attempts
    selected = sorted(
        candidate_splits,
        key=lambda row: (
            int(row.get("test_pos", 0)),
            -int(row.get("test_n", 0)),
        ),
        reverse=True,
    )[:n_repeats]
    splits: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        row = dict(row)
        row["split_id"] = idx
        row["repeat_id"] = idx
        splits.append(row)
    return splits, attempts


def save_splits(
    part2_out_dir: Path,
    df: pd.DataFrame,
    splits: list[dict[str, Any]],
    *,
    seed: int,
    input_hash: str,
    min_test_c2_n: int = 25,
    min_test_c3_n: int = 20,
    soft_tier_balance: dict[str, Any] | None = None,
) -> tuple[Path, Path, str]:
    split_dir = part2_out_dir / "splits"
    ensure_dir(split_dir)
    rows: list[dict[str, Any]] = []
    split_summaries: list[dict[str, Any]] = []
    row_id_col = "id" if "id" in df.columns else None
    companies = df["company"].fillna("__UNKNOWN_COMPANY__").astype(str)
    row_ids = df[row_id_col].astype(str).tolist() if row_id_col else [str(i) for i in range(len(df))]
    for sp in splits:
        split_id = int(sp["split_id"])
        holdout = set(sp["test_idx"].tolist())
        for idx in range(len(df)):
            rows.append(
                {
                    "split_id": split_id,
                    "fold_id": 0,
                    "group_company": companies.iloc[idx],
                    "is_holdout": int(idx in holdout),
                    "row_id": row_ids[idx],
                }
            )
        split_summaries.append(
            {
                "split_id": split_id,
                "random_seed": seed,
                "repeat_id": int(sp["repeat_id"]),
                "train_company_list": sp["train_companies"],
                "test_company_list": sp["test_companies"],
                "train_n": int(sp["train_n"]),
                "test_n": int(sp["test_n"]),
                "train_pos": int(sp["train_pos"]),
                "test_pos": int(sp["test_pos"]),
                "test_c2_n": int(sp.get("test_c2_n", 0)),
                "test_c3_n": int(sp.get("test_c3_n", 0)),
            }
        )
    split_csv = split_dir / "company_holdout_splits.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)
    meta = {
        "n_splits": len(splits),
        "n_folds": 1,
        "group_key": "company",
        "seed": seed,
        "holdout_policy": "company_holdout_20pct_shared",
        "time_created": stage3_reporting.now_iso(),
        "input_hash": input_hash,
        "min_test_c2_n": int(min_test_c2_n),
        "min_test_c3_n": int(min_test_c3_n),
        "split_summaries": split_summaries,
    }
    split_meta = split_dir / "company_holdout_splits_meta.json"
    write_json(split_meta, meta)
    split_hash = sha256_text(split_csv.read_text(encoding="utf-8") + safe_json_dumps(meta))
    return split_csv, split_meta, split_hash


def shortlist_candidate_pool(
    candidates: pd.DataFrame,
    *,
    pair_limit: int,
    triple_limit: int,
) -> pd.DataFrame:
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
    if not pair_df.empty:
        keep_frames.append(pair_df)
    triple_df = df[df["type"] == "triple"].head(max(0, int(triple_limit)))
    if not triple_df.empty:
        keep_frames.append(triple_df)
    if not keep_frames:
        return df.iloc[0:0].copy()
    return pd.concat(keep_frames, ignore_index=True)


def candidate_to_feature_frame(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    if candidates.empty:
        return feat
    for row in candidates.itertuples(index=False):
        cond = pd.Series(True, index=df.index)
        if row.feature_a:
            cond &= df[row.feature_a].astype(str) == str(row.value_a)
        if row.feature_b:
            cond &= df[row.feature_b].astype(str) == str(row.value_b)
        if row.feature_c:
            cond &= df[row.feature_c].astype(str) == str(row.value_c)
        if row.type == "triple":
            cond &= df["coverage_tier"].isin(["C2", "C3"])
        feat[row.candidate_id] = cond.astype(float)
    return feat


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


def candidate_to_nonlinear_feature_frame(
    df: pd.DataFrame,
    base_X: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
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
        if row.type == "triple":
            feature_series = feature_series * df["coverage_tier"].isin(["C2", "C3"]).astype(float)
        feat[f"cx::{row.candidate_id}"] = feature_series.astype(float)
    return feat


def filter_explain_model_selection(
    explain_selection_df: pd.DataFrame | None,
    *,
    model_name: str,
    feature_mode: str,
    branch_name: str,
) -> pd.DataFrame:
    if explain_selection_df is None or explain_selection_df.empty:
        return pd.DataFrame()
    required = {"target_model_name", "feature_mode", "target_branch_name"}
    if not required.issubset(explain_selection_df.columns):
        return pd.DataFrame()
    out = explain_selection_df[
        (explain_selection_df["target_model_name"].astype(str) == model_name)
        & (explain_selection_df["feature_mode"].astype(str) == feature_mode)
        & (explain_selection_df["target_branch_name"].astype(str) == branch_name)
    ].copy()
    return out.reset_index(drop=True)


def feature_mode_allowed_for_model(
    model_name: str,
    feature_mode: str,
    cont_plus_bin_model_subset: set[str] | None,
) -> bool:
    if str(feature_mode) != "cont_plus_bin":
        return True
    if cont_plus_bin_model_subset is None:
        return True
    return str(model_name) in cont_plus_bin_model_subset


def build_explain_model_selection_table(
    model_specs: Sequence["ModelSpec"],
    feature_modes: Sequence[str],
    branch_order: Sequence[str],
    *,
    threshold: "ThresholdSpec",
    pair_candidates: pd.DataFrame,
    triple_candidates: pd.DataFrame,
    enable_3way: bool,
    cont_plus_bin_model_subset: set[str] | None = None,
) -> pd.DataFrame:
    pair_pool = shortlist_candidate_pool(
        pair_candidates,
        pair_limit=PAIR_DISCOVERY_LIMIT,
        triple_limit=TRIPLE_DISCOVERY_LIMIT,
    )
    selected_pairs, _ = stage3_candidates.classify_candidates(
        pair_pool,
        threshold,
        pair_limit=PAIR_LIMIT,
        triple_limit=TRIPLE_LIMIT,
        enforce_cross_signal_publishable=ENFORCE_CROSS_SIGNAL_PUBLISHABLE,
    )

    selected_triples = pd.DataFrame(columns=triple_candidates.columns)
    if enable_3way and "mainline_plus_gated_3way" in branch_order:
        triple_pool = shortlist_candidate_pool(
            triple_candidates,
            pair_limit=0,
            triple_limit=TRIPLE_DISCOVERY_LIMIT,
        )
        selected_triples, _ = stage3_candidates.classify_candidates(
            triple_pool,
            threshold,
            pair_limit=PAIR_LIMIT,
            triple_limit=TRIPLE_LIMIT,
            enforce_cross_signal_publishable=ENFORCE_CROSS_SIGNAL_PUBLISHABLE,
        )

    candidate_cols = list(dict.fromkeys(list(selected_pairs.columns) + list(selected_triples.columns)))
    rows: list[pd.DataFrame] = []

    def annotate(source_df: pd.DataFrame, *, source_model_name: str, target_model_name: str, current_feature_mode: str, target_branch_name: str) -> pd.DataFrame:
        if source_df.empty:
            return pd.DataFrame(columns=EXPLAIN_MODEL_SELECTION_COLUMNS + candidate_cols)
        out = source_df.copy()
        out.insert(0, "source_explain_model_name", source_model_name)
        out.insert(1, "target_model_name", target_model_name)
        out.insert(2, "feature_mode", current_feature_mode)
        out.insert(3, "target_branch_name", target_branch_name)
        return out

    for model_spec in model_specs:
        for current_feature_mode in feature_modes:
            if not feature_mode_allowed_for_model(model_spec.name, current_feature_mode, cont_plus_bin_model_subset):
                continue
            if "mainline_plus_pairwise" in branch_order:
                rows.append(
                    annotate(
                        selected_pairs,
                        source_model_name=model_spec.name,
                        target_model_name=model_spec.name,
                        current_feature_mode=current_feature_mode,
                        target_branch_name="mainline_plus_pairwise",
                    )
                )
            if enable_3way and "mainline_plus_gated_3way" in branch_order:
                rows.append(
                    annotate(
                        pd.concat([selected_pairs, selected_triples], ignore_index=True),
                        source_model_name=model_spec.name,
                        target_model_name=model_spec.name,
                        current_feature_mode=current_feature_mode,
                        target_branch_name="mainline_plus_gated_3way",
                    )
                )

    if not rows:
        return pd.DataFrame(columns=EXPLAIN_MODEL_SELECTION_COLUMNS + candidate_cols)
    out = pd.concat(rows, ignore_index=True)
    for col in EXPLAIN_MODEL_SELECTION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out


def summarize_c3_slice_ci(pred_df: pd.DataFrame, model_name: str, branch_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if pred_df.empty:
        return pd.DataFrame(rows)
    metric_rows: list[dict[str, Any]] = []
    for split_id, grp in pred_df.groupby("split_id"):
        c3 = grp[grp["coverage_tier"] == "C3"]
        if c3.empty:
            continue
        y_true = c3["y_true"].to_numpy(dtype=int)
        score = c3["score"].to_numpy(dtype=float)
        for k in TOP_KS:
            metric_rows.append({"split_id": int(split_id), "metric": "P", "k": k, "value": stage3_models.precision_at_k(y_true, score, k)})
            metric_rows.append({"split_id": int(split_id), "metric": "Enrichment", "k": k, "value": stage3_models.enrichment_at_k(y_true, score, k)})
        metric_rows.append({"split_id": int(split_id), "metric": "AUC_proxy", "k": 0, "value": stage3_models.auc_proxy(y_true, score)})
    metric_df = pd.DataFrame(metric_rows)
    if metric_df.empty:
        return pd.DataFrame(rows)
    return summarize_metric_ci(metric_df, model_name, branch_name)


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


def summarize_delta_ci(base_metric_df: pd.DataFrame, metric_df: pd.DataFrame, model_name: str, branch_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metric_df.empty:
        return pd.DataFrame(rows)
    if branch_name == "mainline":
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


def build_model_specs(model_subset: set[str] | None = None) -> list[ModelSpec]:
    specs = [
        ModelSpec(name="logistic_l2", kind="logistic_l2", nonlinear=False, available=True),
        ModelSpec(name="elasticnet", kind="elasticnet", nonlinear=False, available=True),
        ModelSpec(
            name="ebm",
            kind="ebm",
            nonlinear=True,
            available=ExplainableBoostingClassifier is not None,
            skip_reason="interpret_not_available" if ExplainableBoostingClassifier is None else "",
        ),
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
    importances: pd.Series
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_).reshape(-1)
        importances = pd.Series(values, index=feature_names)
        lines.append("- source: coefficient")
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).reshape(-1)
        importances = pd.Series(values, index=feature_names)
        lines.append("- source: feature_importances_")
    elif hasattr(model, "term_names_") and hasattr(model, "term_importances"):
        values = np.asarray(model.term_importances())
        names = list(model.term_names_)
        importances = pd.Series(values.reshape(-1), index=names)
        lines.append("- source: term_importances")
    else:
        lines.append("- skipped: importance interface unavailable")
        return "\n".join(lines) + "\n"
    lines.append("")
    lines.append("## Top Features")
    ordered = importances.reindex(importances.abs().sort_values(ascending=False).index)
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
    pair_candidates: pd.DataFrame,
    triple_candidates: pd.DataFrame,
    *,
    enable_3way: bool,
    selected_from_explain_model: pd.DataFrame | None = None,
) -> BranchData:
    if branch_name == "mainline":
        return BranchData(
            branch_name=branch_name,
            feature_mode=feature_mode,
            X=base_X.copy(),
            feature_profile=dict(feature_profile),
            rulebook_support=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
            rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
            rulebook_legacy=pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS),
            rulebook_debug=pd.DataFrame(),
            pair_rulebook_publishable_c3only=pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS),
            pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
            candidate_trace=pd.DataFrame(columns=["candidate_id", "type", "threshold_id", "selection_freq_for_threshold", "prediction_candidate", "rule_type", "downgrade_reason"]),
            rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
            model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
            linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
            linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
            linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
            skipped_by_config=False,
        )
    if branch_name == TRIPLE_BRANCH and (not model_spec.nonlinear):
        return BranchData(
            branch_name=branch_name,
            feature_mode=feature_mode,
            X=base_X.copy(),
            feature_profile=dict(feature_profile),
            rulebook_support=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
            rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
            rulebook_legacy=pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS),
            rulebook_debug=pd.DataFrame(),
            pair_rulebook_publishable_c3only=pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS),
            pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
            candidate_trace=pd.DataFrame(),
            rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
            model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
            linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
            linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
            linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
            skipped_by_config=True,
            skip_reason="linear_model_3way_disabled",
        )

    pair_pool = shortlist_candidate_pool(
        pair_candidates,
        pair_limit=PAIR_DISCOVERY_LIMIT,
        triple_limit=TRIPLE_DISCOVERY_LIMIT,
    )
    selected_pairs, pair_trace = stage3_candidates.classify_candidates(
        pair_pool,
        threshold,
        pair_limit=PAIR_LIMIT,
        triple_limit=TRIPLE_LIMIT,
        enforce_cross_signal_publishable=ENFORCE_CROSS_SIGNAL_PUBLISHABLE,
    )
    branch_pairs = selected_pairs.copy()
    if selected_from_explain_model is not None and not selected_from_explain_model.empty:
        pair_subset = selected_from_explain_model[selected_from_explain_model["type"].astype(str) == "pair"].copy()
        if not pair_subset.empty:
            branch_pairs = pair_subset
    prediction_pairs = branch_pairs[branch_pairs["rule_type"] == "prediction"]
    pair_frame = (
        candidate_to_nonlinear_feature_frame(df, base_X, prediction_pairs)
        if model_spec.nonlinear
        else candidate_to_feature_frame(df, prediction_pairs)
    )
    c3_pair_rulebook = stage3_reporting.build_c3_only_pair_rulebook(
        df,
        is_strict_positive(df).to_numpy(dtype=int),
        branch_pairs,
        publish_scope=PAIR_PUBLISH_SCOPE,
        default_candidate_train_scope=PAIR_CANDIDATE_SCOPE,
    )

    if branch_name == "mainline_plus_pairwise":
        support = stage3_reporting.build_rulebook_support_from_candidates(branch_pairs, threshold=threshold, branch_name=branch_name)
        legacy = stage3_reporting.build_legacy_rulebook_from_pairs(branch_pairs)
        return BranchData(
            branch_name=branch_name,
            feature_mode=feature_mode,
            X=pd.concat([base_X, pair_frame], axis=1),
            feature_profile=dict(feature_profile),
            rulebook_support=support,
            rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
            rulebook_legacy=legacy,
            rulebook_debug=branch_pairs,
            pair_rulebook_publishable_c3only=c3_pair_rulebook,
            pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
            candidate_trace=pair_trace,
            rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
            model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
            linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
            linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
            linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
            skipped_by_config=False,
        )

    if not enable_3way:
        return BranchData(
            branch_name=branch_name,
            feature_mode=feature_mode,
            X=base_X.copy(),
            feature_profile=dict(feature_profile),
            rulebook_support=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
            rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
            rulebook_legacy=pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS),
            rulebook_debug=pd.DataFrame(),
            pair_rulebook_publishable_c3only=pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS),
            pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
            candidate_trace=pair_trace,
            rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
            model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
            linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
            linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
            linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
            skipped_by_config=True,
            skip_reason="skipped_by_config=true",
        )

    triple_pool = shortlist_candidate_pool(
        triple_candidates,
        pair_limit=0,
        triple_limit=TRIPLE_DISCOVERY_LIMIT,
    )
    selected_triples, triple_trace = stage3_candidates.classify_candidates(
        triple_pool,
        threshold,
        pair_limit=PAIR_LIMIT,
        triple_limit=TRIPLE_LIMIT,
        enforce_cross_signal_publishable=ENFORCE_CROSS_SIGNAL_PUBLISHABLE,
    )
    branch_triples = selected_triples.copy()
    if selected_from_explain_model is not None and not selected_from_explain_model.empty:
        triple_subset = selected_from_explain_model[selected_from_explain_model["type"].astype(str) == "triple"].copy()
        branch_triples = triple_subset if not triple_subset.empty else pd.DataFrame(columns=selected_triples.columns)
    prediction_triples = branch_triples[branch_triples["rule_type"] == "prediction"] if not branch_triples.empty else pd.DataFrame(columns=branch_triples.columns)
    triple_frame = (
        candidate_to_nonlinear_feature_frame(df, base_X, prediction_triples)
        if model_spec.nonlinear
        else candidate_to_feature_frame(df, prediction_triples)
    )
    support = stage3_reporting.build_rulebook_support_from_candidates(
        pd.concat([branch_pairs, branch_triples], ignore_index=True), threshold=threshold, branch_name=branch_name
    )
    legacy = stage3_reporting.build_legacy_rulebook_from_pairs(branch_pairs)
    candidate_trace = pd.concat([pair_trace, triple_trace], ignore_index=True)
    return BranchData(
        branch_name=branch_name,
        feature_mode=feature_mode,
        X=pd.concat([base_X, pair_frame, triple_frame], axis=1),
        feature_profile=dict(feature_profile),
        rulebook_support=support,
        rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_legacy=legacy,
        rulebook_debug=pd.concat([branch_pairs, branch_triples], ignore_index=True),
        pair_rulebook_publishable_c3only=c3_pair_rulebook,
        pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
        candidate_trace=candidate_trace,
        rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
        model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
        linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
        linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
        linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
        skipped_by_config=False,
    )


def get_delta_summary(delta_ci: pd.DataFrame, metric: str, k: int) -> tuple[float, float, float]:
    row = delta_ci[(delta_ci["metric"] == metric) & (delta_ci["k"] == k)]
    if row.empty:
        return np.nan, np.nan, np.nan
    first = row.iloc[0]
    return float(first["mean"]), float(first["ci_low_95"]), float(first["ci_high_95"])


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
    if not stage3_reporting.is_interaction_branch(branch_name):
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


def row_passes_base_gate(row: pd.Series) -> bool:
    return bool(
        str(row.get("status", "")) == "COMPLETED"
        and int(row.get("hard_audit_pass_count", 0)) == len(HARD_AUDITS)
        and bool(row.get("rulebook_reproducible", False))
        and (not bool(row.get("warning_present", True)))
    )


def pick_model_internal_winner(
    selection_df: pd.DataFrame,
    *,
    model_name: str,
    feature_mode: str,
    allow_pair_override: bool,
) -> int | None:
    subset = selection_df[
        (selection_df["model"].astype(str) == str(model_name))
        & (selection_df["feature_mode"].astype(str) == str(feature_mode))
        & (selection_df["branch"].isin([MAINLINE_BRANCH, PAIRWISE_BRANCH]))
    ]
    if subset.empty:
        return None
    mainline = subset[subset["branch"] == MAINLINE_BRANCH]
    pairwise = subset[subset["branch"] == PAIRWISE_BRANCH]
    main_idx = int(mainline.index[0]) if not mainline.empty else None
    pair_idx = int(pairwise.index[0]) if not pairwise.empty else None

    chosen_idx: int | None = None
    if main_idx is not None and row_passes_base_gate(selection_df.loc[main_idx]):
        chosen_idx = main_idx
    if pair_idx is None or (not allow_pair_override):
        return chosen_idx

    pair_row = selection_df.loc[pair_idx]
    pair_is_promotable = bool(
        row_passes_base_gate(pair_row)
        and bool(pair_row.get("interaction_upgrade_eligible", False))
        and bool(pair_row.get("ci_evidence_ok", False))
    )
    if not pair_is_promotable:
        return chosen_idx
    if chosen_idx is None:
        return pair_idx

    main_row = selection_df.loc[chosen_idx]
    pair_better = bool(
        float(pair_row.get("delta_p20_ci_low", -np.inf)) > 0
        and float(pair_row.get("delta_enrichment20_ci_low", -np.inf)) > 0
        and float(pair_row.get("p20_ci_low", -np.inf)) >= float(main_row.get("p20_ci_low", -np.inf))
        and float(pair_row.get("enrichment20_ci_low", -np.inf)) >= float(main_row.get("enrichment20_ci_low", -np.inf))
    )
    return pair_idx if pair_better else chosen_idx


def first_ranked_index(selection_df: pd.DataFrame, candidate_idxs: Sequence[int]) -> int | None:
    idx_set = {int(idx) for idx in candidate_idxs}
    if not idx_set:
        return None
    for idx in selection_df.index.tolist():
        if int(idx) in idx_set:
            return int(idx)
    return None


def read_top_pair_rule_ids(part2_out_dir: Path, model_name: str, feature_mode: str, *, top_n: int = 5) -> set[str]:
    pair_path = stage3_reporting.model_branch_dir(part2_out_dir, model_name, feature_mode, PAIRWISE_BRANCH) / "pair_rulebook_publishable_c3only.csv"
    if (not pair_path.exists()) or pair_path.stat().st_size <= 0:
        return set()
    df = pd.read_csv(pair_path)
    if df.empty:
        return set()
    ordered = df.copy()
    if "rule_rank" in ordered.columns:
        ordered = ordered.sort_values("rule_rank", ascending=True)
    if "rule_id" in ordered.columns:
        ids = ordered["rule_id"].astype(str).head(max(1, int(top_n))).tolist()
        return {rid for rid in ids if rid and rid != "nan"}
    pairs = ordered[["feature_a", "feature_b"]].astype(str).head(max(1, int(top_n)))
    return {f"{row.feature_a}::{row.feature_b}" for row in pairs.itertuples(index=False)}


def compute_bridge_recommendation(
    selection_df: pd.DataFrame,
    part2_out_dir: Path,
) -> tuple[bool, list[str], float]:
    if selection_df.empty:
        return False, ["selection_empty"], np.nan
    primary_rows = selection_df[selection_df["is_primary_winner"] == True]
    control_rows = selection_df[selection_df["is_control_winner"] == True]
    if primary_rows.empty or control_rows.empty:
        return False, ["missing_primary_or_control_winner"], np.nan
    primary = primary_rows.iloc[0]
    control = control_rows.iloc[0]
    reasons: list[str] = []
    primary_type = "nonlinear" if str(primary.get("model", "")) in PRIMARY_WINNER_MODELS else "linear_or_other"
    control_type = "nonlinear" if str(control.get("model", "")) in PRIMARY_WINNER_MODELS else "linear_or_other"
    if primary_type != control_type:
        reasons.append("winner_model_type_mismatch")

    primary_pair_ids = read_top_pair_rule_ids(part2_out_dir, str(primary["model"]), PRIMARY_WINNER_FEATURE_MODE, top_n=5)
    control_pair_ids = read_top_pair_rule_ids(part2_out_dir, str(control["model"]), CONTROL_WINNER_FEATURE_MODE, top_n=5)
    union = primary_pair_ids | control_pair_ids
    inter = primary_pair_ids & control_pair_ids
    overlap = float(len(inter) / len(union)) if union else np.nan
    ci_boundary = bool(
        float(primary.get("delta_p20_ci_low", -np.inf)) < 0.02
        or float(primary.get("delta_enrichment20_ci_low", -np.inf)) < 0.02
        or float(control.get("delta_p20_ci_low", -np.inf)) < 0.02
        or float(control.get("delta_enrichment20_ci_low", -np.inf)) < 0.02
    )
    if union and np.isfinite(overlap) and overlap < 0.20 and ci_boundary:
        reasons.append("top_pair_overlap_low_with_ci_boundary")

    cont_pairs_max = float(
        selection_df[
            (selection_df["feature_mode"] == PRIMARY_WINNER_FEATURE_MODE)
            & (selection_df["branch"] == PAIRWISE_BRANCH)
        ]["publishable_c3_pair_count"].fillna(0).max()
    ) if (
        (selection_df["feature_mode"] == PRIMARY_WINNER_FEATURE_MODE) & (selection_df["branch"] == PAIRWISE_BRANCH)
    ).any() else 0.0
    bin_pairs_max = float(
        selection_df[
            (selection_df["feature_mode"] == CONTROL_WINNER_FEATURE_MODE)
            & (selection_df["branch"] == PAIRWISE_BRANCH)
        ]["publishable_c3_pair_count"].fillna(0).max()
    ) if (
        (selection_df["feature_mode"] == CONTROL_WINNER_FEATURE_MODE) & (selection_df["branch"] == PAIRWISE_BRANCH)
    ).any() else 0.0
    if cont_pairs_max < 4 and bin_pairs_max >= 4:
        reasons.append("cont_publishable_pairs_insufficient_bin_sufficient")

    return bool(len(reasons) > 0), reasons, overlap


def build_branch_conclusion(
    model_name: str,
    branch_name: str,
    feature_mode: str,
    metrics_ci: pd.DataFrame,
    metrics_c3_slice_ci: pd.DataFrame,
    delta_ci: pd.DataFrame,
    audit_summary: pd.DataFrame,
    rulebook_support: pd.DataFrame,
    unstable_explanation_pairs: pd.DataFrame,
    *,
    sensitivity_refs: dict[str, str],
    status: str,
    skipped_reason: str,
    applicability_domain: str,
) -> tuple[str, str, dict[str, Any]]:
    p20_mean_row = metrics_ci[(metrics_ci["metric"] == "P") & (metrics_ci["k"] == 20)]
    e20_mean_row = metrics_ci[(metrics_ci["metric"] == "Enrichment") & (metrics_ci["k"] == 20)]
    auc_row = metrics_ci[(metrics_ci["metric"] == "AUC_proxy") & (metrics_ci["k"] == 0)]
    c3_p20_row = metrics_c3_slice_ci[(metrics_c3_slice_ci["metric"] == "P") & (metrics_c3_slice_ci["k"] == 20)]
    c3_e20_row = metrics_c3_slice_ci[(metrics_c3_slice_ci["metric"] == "Enrichment") & (metrics_c3_slice_ci["k"] == 20)]
    c3_auc_row = metrics_c3_slice_ci[(metrics_c3_slice_ci["metric"] == "AUC_proxy") & (metrics_c3_slice_ci["k"] == 0)]
    p20 = p20_mean_row.iloc[0] if not p20_mean_row.empty else pd.Series(dtype=float)
    e20 = e20_mean_row.iloc[0] if not e20_mean_row.empty else pd.Series(dtype=float)
    auc = auc_row.iloc[0] if not auc_row.empty else pd.Series(dtype=float)
    c3_p20 = c3_p20_row.iloc[0] if not c3_p20_row.empty else pd.Series(dtype=float)
    c3_e20 = c3_e20_row.iloc[0] if not c3_e20_row.empty else pd.Series(dtype=float)
    c3_auc = c3_auc_row.iloc[0] if not c3_auc_row.empty else pd.Series(dtype=float)
    dp20_mean, dp20_low, dp20_high = get_delta_summary(delta_ci, "P", 20)
    de20_mean, de20_low, de20_high = get_delta_summary(delta_ci, "Enrichment", 20)
    dauc_mean, dauc_low, dauc_high = get_delta_summary(delta_ci, "AUC_proxy", 0)

    pred_rules = rulebook_support[rulebook_support["rule_type"] == "prediction"].head(8)
    triage_rules = rulebook_support[rulebook_support["rule_type"] == "triage"].head(8)
    audit_lines_zh = [f"- {row.audit_name}: {row.status} ({row.details})" for row in audit_summary.itertuples(index=False)]
    audit_lines_en = [f"- {row.audit_name}: {row.status} ({row.details})" for row in audit_summary.itertuples(index=False)]
    pred_lines_zh = [f"- {row.condition_text} | coverage={fmt_float(row.coverage, 4)} | enrich={fmt_float(row.enrichment, 4)}" for row in pred_rules.itertuples(index=False)] or ["- none"]
    triage_lines_zh = [f"- {row.condition_text} | notes={row.notes}" for row in triage_rules.itertuples(index=False)] or ["- none"]
    pred_lines_en = [f"- {row.condition_text} | coverage={fmt_float(row.coverage, 4)} | enrich={fmt_float(row.enrichment, 4)}" for row in pred_rules.itertuples(index=False)] or ["- none"]
    triage_lines_en = [f"- {row.condition_text} | notes={row.notes}" for row in triage_rules.itertuples(index=False)] or ["- none"]
    unstable_rules = unstable_explanation_pairs.head(8)
    unstable_lines_zh = [
        f"- {row.condition_text} | enrich_c3={fmt_float(row.enrichment_c3, 4)} | evidence={row.evidence_status} | why={row.why_unstable}"
        for row in unstable_rules.itertuples(index=False)
    ] or ["- none"]
    unstable_lines_en = [
        f"- {row.condition_text} | enrich_c3={fmt_float(row.enrichment_c3, 4)} | evidence={row.evidence_status} | why={row.why_unstable}"
        for row in unstable_rules.itertuples(index=False)
    ] or ["- none"]

    zh_lines = [
        f"# {model_name} / {branch_name} / {feature_mode} 结果解读",
        "",
        f"- status: {status}",
        f"- skip_reason: {skipped_reason or 'NA'}",
        f"- applicability_domain: {applicability_domain}",
        f"- P@20: {fmt_float(float(p20.get('mean', np.nan)))} | CI95=[{fmt_float(float(p20.get('ci_low_95', np.nan)))}, {fmt_float(float(p20.get('ci_high_95', np.nan)))}]",
        f"- Enrichment@20: {fmt_float(float(e20.get('mean', np.nan)))} | CI95=[{fmt_float(float(e20.get('ci_low_95', np.nan)))}, {fmt_float(float(e20.get('ci_high_95', np.nan)))}]",
        f"- AUC_proxy: {fmt_float(float(auc.get('mean', np.nan)))} | CI95=[{fmt_float(float(auc.get('ci_low_95', np.nan)))}, {fmt_float(float(auc.get('ci_high_95', np.nan)))}]",
        f"- C3 P@20: {fmt_float(float(c3_p20.get('mean', np.nan)))} | CI95=[{fmt_float(float(c3_p20.get('ci_low_95', np.nan)))}, {fmt_float(float(c3_p20.get('ci_high_95', np.nan)))}]",
        f"- C3 Enrichment@20: {fmt_float(float(c3_e20.get('mean', np.nan)))} | CI95=[{fmt_float(float(c3_e20.get('ci_low_95', np.nan)))}, {fmt_float(float(c3_e20.get('ci_high_95', np.nan)))}]",
        f"- C3 AUC_proxy: {fmt_float(float(c3_auc.get('mean', np.nan)))} | CI95=[{fmt_float(float(c3_auc.get('ci_low_95', np.nan)))}, {fmt_float(float(c3_auc.get('ci_high_95', np.nan)))}]",
        f"- ΔP@20: {fmt_float(dp20_mean)} | CI95=[{fmt_float(dp20_low)}, {fmt_float(dp20_high)}]",
        f"- ΔEnrichment@20: {fmt_float(de20_mean)} | CI95=[{fmt_float(de20_low)}, {fmt_float(de20_high)}]",
        f"- ΔAUC: {fmt_float(dauc_mean)} | CI95=[{fmt_float(dauc_low)}, {fmt_float(dauc_high)}]",
        "",
        "## 审计链",
        *audit_lines_zh,
        "",
        "## Prediction Rules Top N",
        *pred_lines_zh,
        "",
        "## Triage Rules Top N",
        *triage_lines_zh,
        "",
        "## Unstable Explanation Pairs Top N",
        *unstable_lines_zh,
        "",
        "## 敏感性摘要引用",
        f"- +year: {sensitivity_refs['year']}",
        f"- missingness ablation: {sensitivity_refs['missingness']}",
        f"- threshold grid: {sensitivity_refs['threshold']}",
        f"- legacy bridge: {sensitivity_refs['bridge']}",
        f"- model-derived sensitivity rulebook: rulebook_model_derived_sensitivity.csv",
        f"- model-derived cutpoint alignment: model_derived_cutpoint_alignment.csv",
        f"- linear continuous effects: linear_continuous_effects.csv",
    ]
    en_lines = [
        f"# {model_name} / {branch_name} / {feature_mode} Conclusion",
        "",
        f"- status: {status}",
        f"- skip_reason: {skipped_reason or 'NA'}",
        f"- applicability_domain: {applicability_domain}",
        f"- P@20: {fmt_float(float(p20.get('mean', np.nan)))} | CI95=[{fmt_float(float(p20.get('ci_low_95', np.nan)))}, {fmt_float(float(p20.get('ci_high_95', np.nan)))}]",
        f"- Enrichment@20: {fmt_float(float(e20.get('mean', np.nan)))} | CI95=[{fmt_float(float(e20.get('ci_low_95', np.nan)))}, {fmt_float(float(e20.get('ci_high_95', np.nan)))}]",
        f"- AUC_proxy: {fmt_float(float(auc.get('mean', np.nan)))} | CI95=[{fmt_float(float(auc.get('ci_low_95', np.nan)))}, {fmt_float(float(auc.get('ci_high_95', np.nan)))}]",
        f"- C3 P@20: {fmt_float(float(c3_p20.get('mean', np.nan)))} | CI95=[{fmt_float(float(c3_p20.get('ci_low_95', np.nan)))}, {fmt_float(float(c3_p20.get('ci_high_95', np.nan)))}]",
        f"- C3 Enrichment@20: {fmt_float(float(c3_e20.get('mean', np.nan)))} | CI95=[{fmt_float(float(c3_e20.get('ci_low_95', np.nan)))}, {fmt_float(float(c3_e20.get('ci_high_95', np.nan)))}]",
        f"- C3 AUC_proxy: {fmt_float(float(c3_auc.get('mean', np.nan)))} | CI95=[{fmt_float(float(c3_auc.get('ci_low_95', np.nan)))}, {fmt_float(float(c3_auc.get('ci_high_95', np.nan)))}]",
        f"- ΔP@20: {fmt_float(dp20_mean)} | CI95=[{fmt_float(dp20_low)}, {fmt_float(dp20_high)}]",
        f"- ΔEnrichment@20: {fmt_float(de20_mean)} | CI95=[{fmt_float(de20_low)}, {fmt_float(de20_high)}]",
        f"- ΔAUC: {fmt_float(dauc_mean)} | CI95=[{fmt_float(dauc_low)}, {fmt_float(dauc_high)}]",
        "",
        "## Audit Chain",
        *audit_lines_en,
        "",
        "## Prediction Rules Top N",
        *pred_lines_en,
        "",
        "## Triage Rules Top N",
        *triage_lines_en,
        "",
        "## Unstable Explanation Pairs Top N",
        *unstable_lines_en,
        "",
        "## Sensitivity References",
        f"- +year: {sensitivity_refs['year']}",
        f"- missingness ablation: {sensitivity_refs['missingness']}",
        f"- threshold grid: {sensitivity_refs['threshold']}",
        f"- legacy bridge: {sensitivity_refs['bridge']}",
        f"- model-derived sensitivity rulebook: rulebook_model_derived_sensitivity.csv",
        f"- model-derived cutpoint alignment: model_derived_cutpoint_alignment.csv",
        f"- linear continuous effects: linear_continuous_effects.csv",
    ]
    decision = {
        "status": status,
        "skip_reason": skipped_reason,
        "feature_mode": feature_mode,
        "applicability_domain": applicability_domain,
        "rule_publish_gate": "rule_level_L0_support_stability",
        "branch_promotion_gate": "delta_ci_plus_warning_plus_hard_audits",
        "c3_p20_mean": float(c3_p20.get("mean", np.nan)) if not c3_p20.empty else np.nan,
        "c3_enrichment20_mean": float(c3_e20.get("mean", np.nan)) if not c3_e20.empty else np.nan,
        "delta_p20_ci_low": dp20_low,
        "delta_enrichment20_ci_low": de20_low,
        "hard_audits_pass": int((audit_summary[audit_summary["audit_name"].isin(HARD_AUDITS)]["status"] == "PASS").sum()) if not audit_summary.empty else 0,
        "tier2d_status": (
            audit_summary.loc[audit_summary["audit_name"] == SOFT_AUDIT_NAME, "status"].iloc[0]
            if (not audit_summary.empty and (audit_summary["audit_name"] == SOFT_AUDIT_NAME).any())
            else "PASS"
        ),
        "tier2d_level": (
            str(audit_summary.loc[audit_summary["audit_name"] == SOFT_AUDIT_NAME, "tier2d_level"].iloc[0])
            if (
                not audit_summary.empty
                and "tier2d_level" in audit_summary.columns
                and (audit_summary["audit_name"] == SOFT_AUDIT_NAME).any()
            )
            else "PASS"
        ),
        "prediction_rule_count": int((rulebook_support["rule_type"] == "prediction").sum()) if not rulebook_support.empty else 0,
        "unstable_explanation_pair_count": int(len(unstable_explanation_pairs)),
    }
    return "\n".join(zh_lines) + "\n", "\n".join(en_lines) + "\n", decision


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
    common_k_min: int = TIER2D_COMMON_K_MIN,
    min_test_c2_n: int = TIER2D_MIN_TEST_C2_N,
    min_test_c3_n: int = TIER2D_MIN_TEST_C3_N,
    raw_diff_eps: float = TIER2D_RAW_DIFF_EPS,
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
        pair_limit=PAIR_LIMIT,
        triple_limit=TRIPLE_LIMIT,
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
    if not tier_summary.empty and "n_splits_used" in tier_summary.columns and pd.notna(tier_summary["n_splits_used"].iloc[0]):
        n_splits_used = int(tier_summary["n_splits_used"].iloc[0])
    elif not tier_summary.empty and "n_splits" in tier_summary.columns and pd.notna(tier_summary["n_splits"].iloc[0]):
        n_splits_used = int(tier_summary["n_splits"].iloc[0])
    else:
        n_splits_used = 0
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


def make_placeholder_branch_outputs(branch_name: str, feature_mode: str, feature_profile: dict[str, Any], skip_reason: str) -> BranchArtifacts:
    metrics_ci = pd.DataFrame(columns=["model", "branch", "metric", "k", "mean", "std", "ci_low_95", "ci_high_95", "n_valid_splits"])
    metrics_c3_slice_ci = pd.DataFrame(columns=["model", "branch", "metric", "k", "mean", "std", "ci_low_95", "ci_high_95", "n_valid_splits"])
    delta_ci = pd.DataFrame(columns=["model", "branch", "metric", "k", "mean", "ci_low_95", "ci_high_95", "n_valid_splits"])
    audit_summary = pd.DataFrame(
        [
            {"audit_name": "controlled_missingness_parallel", "status": "PASS", "details": skip_reason, "severity": "hard"},
            {"audit_name": "negative_control", "status": "PASS", "details": skip_reason, "severity": "hard"},
            {"audit_name": SOFT_AUDIT_NAME, "status": "PASS", "details": skip_reason, "severity": "soft", "tier2d_level": "PASS"},
            {"audit_name": "rule_candidate_consistency", "status": "PASS", "details": skip_reason, "severity": "hard"},
        ]
    )
    decision_payload = {
        "status": "SKIPPED",
        "skip_reason": skip_reason,
        "applicability_domain": "all",
        "hard_audits_pass": 3,
        "tier2d_status": "PASS",
        "tier2d_level": "PASS",
        "publish_scope_final": "C2C3",
        "prediction_rule_count": 0,
    }
    conclusion_zh = f"# {branch_name} 跳过\n\n- reason: {skip_reason}\n"
    conclusion_en = f"# {branch_name} skipped\n\n- reason: {skip_reason}\n"
    return BranchArtifacts(
        feature_mode=feature_mode,
        feature_profile=dict(feature_profile),
        metrics_ci=metrics_ci,
        metrics_c3_slice_ci=metrics_c3_slice_ci,
        delta_ci=delta_ci,
        audit_summary=audit_summary,
        rulebook_support=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_legacy=pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS),
        rulebook_debug=pd.DataFrame(),
        pair_rulebook_publishable_c3only=pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS),
        pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
        rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
        model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
        linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
        linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
        linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
        decision_payload=decision_payload,
        conclusion_zh=conclusion_zh,
        conclusion_en=conclusion_en,
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


def execute_model_zoo(
    part2_out_dir: Path,
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    *,
    random_seed: int,
    threshold: ThresholdSpec,
    pair_candidates: pd.DataFrame,
    triple_candidates: pd.DataFrame,
    base_feature_sets: dict[str, tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]],
    feature_modes: list[str],
    enable_3way: bool,
    sensitivity_refs: dict[str, str],
    unstable_explanation_config: dict[str, Any] | None = None,
    model_subset: set[str] | None = None,
    cont_plus_bin_model_subset: set[str] | None = None,
    branch_subset: list[str] | None = None,
    explain_selection_df: pd.DataFrame | None = None,
    nonlinear_cont_only_rulebook_policy: dict[str, Any] | None = None,
    tier2d_common_k_min: int = TIER2D_COMMON_K_MIN,
    tier2d_min_test_c2_n: int = TIER2D_MIN_TEST_C2_N,
    tier2d_min_test_c3_n: int = TIER2D_MIN_TEST_C3_N,
    tier2d_raw_diff_eps: float = TIER2D_RAW_DIFF_EPS,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    model_specs = build_model_specs(model_subset=model_subset)
    branch_order = branch_subset or ["mainline", "mainline_plus_pairwise", "mainline_plus_gated_3way"]
    comparison_rows: list[dict[str, Any]] = []
    metrics_frames: list[pd.DataFrame] = []
    metrics_c3_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    model_level_predictions: list[pd.DataFrame] = []
    model_level_fold_metrics: list[pd.DataFrame] = []
    model_level_audit_trace: list[pd.DataFrame] = []
    resolved_rulebook_policy = dict(nonlinear_cont_only_rulebook_policy or DEFAULT_NONLINEAR_CONT_ONLY_RULEBOOK_POLICY)

    for model_spec in model_specs:
        model_dir = part2_out_dir / "models" / model_spec.name
        ensure_dir(model_dir)
        model_prediction_frames: list[pd.DataFrame] = []
        model_fold_frames: list[pd.DataFrame] = []
        model_trace_frames: list[pd.DataFrame] = []
        model_candidate_frames: list[pd.DataFrame] = []
        for feature_mode in feature_modes:
            if not feature_mode_allowed_for_model(model_spec.name, feature_mode, cont_plus_bin_model_subset):
                continue
            base_X, base_feature_meta, feature_profile = base_feature_sets[feature_mode]
            feature_mode_dir = model_dir / feature_mode
            ensure_dir(feature_mode_dir)
            mainline_metric_df = pd.DataFrame()

            for branch_name in branch_order:
                branch_dir = feature_mode_dir / branch_name
                ensure_dir(branch_dir)
                branch_warning_log = branch_dir / "run_warning.log"
                ensure_file(branch_warning_log)

                if not model_spec.available:
                    branch_result = make_placeholder_branch_outputs(branch_name, feature_mode, feature_profile, model_spec.skip_reason or "dependency_unavailable")
                else:
                    branch_data = build_branch_data(
                        model_spec,
                        branch_name,
                        feature_mode,
                        df,
                        base_X,
                        feature_profile,
                        threshold,
                        pair_candidates,
                        triple_candidates,
                        enable_3way=enable_3way,
                        selected_from_explain_model=filter_explain_model_selection(
                            explain_selection_df,
                            model_name=model_spec.name,
                            feature_mode=feature_mode,
                            branch_name=branch_name,
                        ),
                    )
                    if branch_data.skipped_by_config:
                        branch_result = make_placeholder_branch_outputs(branch_name, feature_mode, feature_profile, branch_data.skip_reason or "skipped_by_config")
                    else:
                        metric_df, pred_df, fold_metric_df, audit_trace_df, full_model, full_scaling_stats = stage3_models.evaluate_model_branch(
                            model_spec,
                            branch_data,
                            df,
                            y,
                            splits,
                            random_seed=random_seed,
                            warning_log=branch_warning_log,
                            pair_limit=PAIR_LIMIT,
                            triple_limit=TRIPLE_LIMIT,
                            top_ks=TOP_KS,
                        )
                        if branch_name == "mainline":
                            mainline_metric_df = metric_df.copy()
                        metrics_ci = summarize_metric_ci(metric_df, model_spec.name, branch_name)
                        metrics_c3_ci = summarize_c3_slice_ci(pred_df, model_spec.name, branch_name)
                        delta_ci = summarize_delta_ci(mainline_metric_df if not mainline_metric_df.empty else metric_df, metric_df, model_spec.name, branch_name)
                        if branch_name == "mainline":
                            support_rulebook, legacy_rulebook, debug_rulebook = stage3_reporting.build_mainline_rulebook(
                                model_spec,
                                df,
                                base_X,
                                base_feature_meta,
                                y,
                                full_model,
                            )
                            branch_data.rulebook_support = support_rulebook
                            branch_data.rulebook_legacy = legacy_rulebook
                            branch_data.rulebook_debug = debug_rulebook
                            branch_data.rulebook_support_engineered_comparison = pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS)
                            (
                                branch_data.rulebook_model_derived_sensitivity,
                                branch_data.model_derived_cutpoint_alignment,
                            ) = stage3_reporting.build_model_derived_sensitivity_rulebook(
                                model_spec,
                                full_model,
                                df,
                                base_X,
                                y,
                                full_scaling_stats,
                                min_support_n=int(resolved_rulebook_policy.get("min_support_n", 10)),
                                min_support_pos=int(resolved_rulebook_policy.get("min_support_pos", 2)),
                                min_enrichment=float(resolved_rulebook_policy.get("min_enrichment", 1.0)),
                                max_rules_per_feature=int(resolved_rulebook_policy.get("max_rules_per_feature", 3)),
                            )
                            use_model_derived_as_primary = bool(
                                model_spec.nonlinear
                                and str(feature_mode) == "cont_only"
                                and str(resolved_rulebook_policy.get("primary_source", "model_derived")) == "model_derived"
                            )
                            if use_model_derived_as_primary and not branch_data.rulebook_model_derived_sensitivity.empty:
                                if bool(resolved_rulebook_policy.get("engineered_comparison_enabled", True)):
                                    branch_data.rulebook_support_engineered_comparison = support_rulebook.copy()
                                branch_data.rulebook_support = stage3_reporting.promote_model_derived_as_primary_support(
                                    branch_data.rulebook_model_derived_sensitivity
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
                            df,
                            y,
                            splits,
                            pred_df,
                            branch_warning_log,
                            random_seed,
                            common_k_min=tier2d_common_k_min,
                            min_test_c2_n=tier2d_min_test_c2_n,
                            min_test_c3_n=tier2d_min_test_c3_n,
                            raw_diff_eps=tier2d_raw_diff_eps,
                        )
                        if applicability_domain != "all" and tier2d_level == "FAIL" and not branch_data.rulebook_support.empty:
                            mask = branch_data.rulebook_support["rule_type"] == "prediction"
                            branch_data.rulebook_support.loc[mask, "notes"] = branch_data.rulebook_support.loc[mask, "notes"].astype(str).apply(
                                lambda s: "; ".join([piece for piece in [s, f"applicability_domain={applicability_domain}", "downgrade_reason=tier_shift"] if piece and piece != "nan"])
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
                        if stage3_reporting.is_interaction_branch(branch_name):
                            debug_df = branch_data.rulebook_debug if isinstance(branch_data.rulebook_debug, pd.DataFrame) else pd.DataFrame()
                            if model_spec.nonlinear:
                                branch_data.pair_rulebook_publishable_c3only = stage3_reporting.build_c3_only_pair_rulebook(
                                    df,
                                    y,
                                    debug_df,
                                    publish_scope=publish_scope_final,
                                    default_candidate_train_scope=PAIR_CANDIDATE_SCOPE,
                                )
                            else:
                                branch_data.pair_rulebook_publishable_c3only = pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS)
                                if not branch_data.rulebook_support.empty and "rule_type" in branch_data.rulebook_support.columns:
                                    branch_data.rulebook_support["rule_type"] = "triage"
                                    if "notes" in branch_data.rulebook_support.columns:
                                        branch_data.rulebook_support["notes"] = branch_data.rulebook_support["notes"].astype(str).apply(
                                            lambda s: "; ".join(
                                                [
                                                    piece
                                                    for piece in [
                                                        s if s != "nan" else "",
                                                        "linear_pairwise_explanation_only=true",
                                                        "publishable_pair_rulebook=disabled",
                                                    ]
                                                    if piece
                                                ]
                                            )
                                        )
                            branch_data.pair_rulebook_explanation_unstable_c3only = stage3_reporting.build_unstable_explanation_pair_rulebook(
                                df,
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
                        conclusion_zh, conclusion_en, decision_payload = build_branch_conclusion(
                            model_spec.name,
                            branch_name,
                            feature_mode,
                            metrics_ci,
                            metrics_c3_ci,
                            delta_ci,
                            audit_summary,
                            branch_data.rulebook_support,
                            branch_data.pair_rulebook_explanation_unstable_c3only,
                            sensitivity_refs=sensitivity_refs,
                            status="COMPLETED",
                            skipped_reason="",
                            applicability_domain=applicability_domain,
                        )
                        conclusion_zh += (
                            "\n## Supplementary Outputs\n"
                            "- model-derived sensitivity rulebook: rulebook_model_derived_sensitivity.csv\n"
                            "- cutpoint alignment: model_derived_cutpoint_alignment.csv\n"
                            "- engineered threshold comparison: rulebook_support_engineered_comparison.csv\n"
                            "- linear continuous effects: linear_continuous_effects.csv\n"
                            "- linear pairwise effects: linear_pairwise_effects.csv\n"
                            "- linear direction check: linear_vs_engineered_direction_check.csv\n"
                            "- tier input shift: input_shift_metrics.csv\n"
                            "- tier score shift: score_shift_metrics.csv\n"
                            "- tier perf shift: perf_shift_metrics.csv\n"
                            "- tier matched control: tier_shift_matched_control.csv\n"
                        )
                        conclusion_en += (
                            "\n## Supplementary Outputs\n"
                            "- model-derived sensitivity rulebook: rulebook_model_derived_sensitivity.csv\n"
                            "- cutpoint alignment: model_derived_cutpoint_alignment.csv\n"
                            "- engineered threshold comparison: rulebook_support_engineered_comparison.csv\n"
                            "- linear continuous effects: linear_continuous_effects.csv\n"
                            "- linear pairwise effects: linear_pairwise_effects.csv\n"
                            "- linear direction check: linear_vs_engineered_direction_check.csv\n"
                            "- tier input shift: input_shift_metrics.csv\n"
                            "- tier score shift: score_shift_metrics.csv\n"
                            "- tier perf shift: perf_shift_metrics.csv\n"
                            "- tier matched control: tier_shift_matched_control.csv\n"
                        )
                        mechanism_md = extract_mechanism_markdown(model_spec, full_model, branch_data.X.columns.tolist()) if model_spec.nonlinear else "# linear model\n"
                        branch_result = BranchArtifacts(
                            feature_mode=feature_mode,
                            feature_profile=dict(feature_profile),
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
                            conclusion_zh=conclusion_zh,
                            conclusion_en=conclusion_en,
                            mechanism_md=mechanism_md,
                            status="COMPLETED",
                            skipped_by_config=False,
                            candidate_trace=branch_data.candidate_trace,
                            oof_predictions=pred_df,
                            fold_metrics=fold_metric_df,
                            audit_trace=audit_trace_df if audit_trace.empty else audit_trace,
                            input_shift_metrics=input_shift_metrics,
                            score_shift_metrics=score_shift_metrics,
                            perf_shift_metrics=perf_shift_metrics,
                            tier_shift_matched_control=tier_shift_matched_control,
                        )
                        branch_result.decision_payload["tier2d_level"] = tier2d_level
                        branch_result.decision_payload["publish_scope_final"] = publish_scope_final

                warning_present = branch_warning_log.exists() and branch_warning_log.stat().st_size > 0
                branch_p20 = branch_result.metrics_ci[(branch_result.metrics_ci["metric"] == "P") & (branch_result.metrics_ci["k"] == 20)]
                branch_e20 = branch_result.metrics_ci[(branch_result.metrics_ci["metric"] == "Enrichment") & (branch_result.metrics_ci["k"] == 20)]
                branch_auc = branch_result.metrics_ci[(branch_result.metrics_ci["metric"] == "AUC_proxy") & (branch_result.metrics_ci["k"] == 0)]
                branch_c3_p20 = branch_result.metrics_c3_slice_ci[(branch_result.metrics_c3_slice_ci["metric"] == "P") & (branch_result.metrics_c3_slice_ci["k"] == 20)]
                branch_c3_e20 = branch_result.metrics_c3_slice_ci[(branch_result.metrics_c3_slice_ci["metric"] == "Enrichment") & (branch_result.metrics_c3_slice_ci["k"] == 20)]
                p20_mean, p20_low, p20_high = get_delta_summary(branch_result.delta_ci, "P", 20)
                e20_mean, e20_low, e20_high = get_delta_summary(branch_result.delta_ci, "Enrichment", 20)
                auc_mean, auc_low, auc_high = get_delta_summary(branch_result.delta_ci, "AUC_proxy", 0)
                hard_pass_count = int(
                    (branch_result.audit_summary[branch_result.audit_summary["audit_name"].isin(HARD_AUDITS)]["status"] == "PASS").sum()
                ) if not branch_result.audit_summary.empty else 0
                tier2d_status = (
                    branch_result.audit_summary.loc[branch_result.audit_summary["audit_name"] == SOFT_AUDIT_NAME, "status"].iloc[0]
                    if not branch_result.audit_summary.empty and (branch_result.audit_summary["audit_name"] == SOFT_AUDIT_NAME).any()
                    else "PASS"
                )
                tier2d_level = (
                    str(branch_result.audit_summary.loc[branch_result.audit_summary["audit_name"] == SOFT_AUDIT_NAME, "tier2d_level"].iloc[0])
                    if (
                        not branch_result.audit_summary.empty
                        and "tier2d_level" in branch_result.audit_summary.columns
                        and (branch_result.audit_summary["audit_name"] == SOFT_AUDIT_NAME).any()
                    )
                    else ("FAIL" if str(tier2d_status).upper() != "PASS" else "PASS")
                )
                publish_scope_final = str(
                    branch_result.decision_payload.get(
                        "publish_scope_final",
                        "C2C3" if tier2d_level in {"PASS", "WARN"} else "C3_only",
                    )
                )
                rulebook_reproducible = True
                if not branch_result.rulebook_support.empty:
                    pred_mask = branch_result.rulebook_support["rule_type"] == "prediction"
                    pred_rules = branch_result.rulebook_support.loc[pred_mask].copy()
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
                    delta_p20_ci_low=p20_low,
                    delta_enrichment20_ci_low=e20_low,
                )
                branch_result.decision_payload["warning_present"] = bool(warning_present)
                branch_result.decision_payload["rulebook_reproducible"] = bool(rulebook_reproducible)
                branch_result.decision_payload["interaction_upgrade_eligible"] = bool(interaction_upgrade_eligible)
                branch_result.decision_payload["interaction_mode"] = interaction_mode
                branch_result.decision_payload["feature_mode"] = feature_mode
                branch_result.decision_payload["feature_profile"] = dict(feature_profile)
                branch_result.decision_payload["tier2d_level"] = tier2d_level
                branch_result.decision_payload["publish_scope_final"] = publish_scope_final
                if stage3_reporting.is_interaction_branch(branch_name):
                    branch_result.conclusion_zh += f"\n- interaction_mode: {interaction_mode}\n"
                    branch_result.conclusion_en += f"\n- interaction_mode: {interaction_mode}\n"
                branch_result.metrics_ci.to_csv(branch_dir / "metrics_ci.csv", index=False)
                branch_result.metrics_c3_slice_ci.to_csv(branch_dir / "metrics_c3_slice_ci.csv", index=False)
                branch_result.delta_ci.to_csv(branch_dir / "delta_ci.csv", index=False)
                branch_result.audit_summary.to_csv(branch_dir / "audit_summaries.csv", index=False)
                branch_result.rulebook_support.to_csv(branch_dir / "rulebook_support.csv", index=False)
                branch_result.rulebook_support_engineered_comparison.to_csv(
                    branch_dir / "rulebook_support_engineered_comparison.csv", index=False
                )
                branch_result.rulebook_legacy.to_csv(branch_dir / "rulebook_legacy_pair_tier2plus.csv", index=False)
                branch_result.pair_rulebook_publishable_c3only.to_csv(branch_dir / "pair_rulebook_publishable_c3only.csv", index=False)
                if not branch_result.pair_rulebook_publishable_c3only.empty and "publish_scope" in branch_result.pair_rulebook_publishable_c3only.columns:
                    publish_scope = str(branch_result.pair_rulebook_publishable_c3only["publish_scope"].iloc[0])
                    scope_tag = publish_scope.lower().replace("+", "").replace("/", "").replace("_", "")
                    scoped_name = f"pair_rulebook_publishable_{scope_tag}.csv"
                    if scoped_name != "pair_rulebook_publishable_c3only.csv":
                        branch_result.pair_rulebook_publishable_c3only.to_csv(branch_dir / scoped_name, index=False)
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
                if EXPORT_DEBUG_ARTIFACTS:
                    branch_result.rulebook_debug.to_csv(branch_dir / "rulebook_debug.csv", index=False)
                (branch_dir / "run_decision.md").write_text(safe_json_dumps(branch_result.decision_payload) + "\n", encoding="utf-8")
                (branch_dir / "run_conclusion_analysis_and_improvement.md").write_text(branch_result.conclusion_zh, encoding="utf-8")
                (branch_dir / "run_conclusion_analysis_and_improvement.en.md").write_text(branch_result.conclusion_en, encoding="utf-8")
                if model_spec.nonlinear:
                    (branch_dir / "rulebook_mechanism_extraction.md").write_text(branch_result.mechanism_md, encoding="utf-8")
                else:
                    (branch_dir / "rulebook_mechanism_extraction.md").write_text("# linear model\n", encoding="utf-8")

                metrics_frames.append(branch_result.metrics_ci)
                metrics_c3_frames.append(branch_result.metrics_c3_slice_ci)
                audit_summary_df = branch_result.audit_summary.copy()
                if not audit_summary_df.empty:
                    audit_summary_df.insert(0, "feature_mode", feature_mode)
                    audit_summary_df.insert(0, "branch", branch_name)
                    audit_summary_df.insert(0, "model", model_spec.name)
                    audit_frames.append(audit_summary_df)

                if not branch_result.oof_predictions.empty:
                    pred_copy = branch_result.oof_predictions.copy()
                    pred_copy.insert(1, "feature_mode", feature_mode)
                    model_prediction_frames.append(pred_copy)
                if not branch_result.fold_metrics.empty:
                    fold_copy = branch_result.fold_metrics.copy()
                    fold_copy.insert(1, "feature_mode", feature_mode)
                    model_fold_frames.append(fold_copy)
                if not branch_result.audit_trace.empty:
                    trace = branch_result.audit_trace.copy()
                    trace.insert(0, "feature_mode", feature_mode)
                    trace.insert(0, "branch", branch_name)
                    trace.insert(0, "model", model_spec.name)
                    model_trace_frames.append(trace)
                if not branch_result.candidate_trace.empty:
                    candidate_trace = branch_result.candidate_trace.copy()
                    candidate_trace.insert(0, "feature_mode", feature_mode)
                    candidate_trace.insert(0, "branch", branch_name)
                    candidate_trace.insert(0, "model", model_spec.name)
                    model_candidate_frames.append(candidate_trace)
                selected_as_primary = False
                downgrade_reasons = []
                if branch_result.status != "COMPLETED":
                    downgrade_reasons.append(branch_result.decision_payload.get("skip_reason", branch_result.status))
                if hard_pass_count < len(HARD_AUDITS):
                    downgrade_reasons.append("hard_audit_failed")
                if not rulebook_reproducible:
                    downgrade_reasons.append("rulebook_not_reproducible")
                if warning_present:
                    downgrade_reasons.append("warning_present")
                if tier2d_level == "FAIL":
                    downgrade_reasons.append("tier_shift")
                if np.isfinite(p20_low) and np.isfinite(e20_low) and (p20_low < 0 or e20_low < 0):
                    downgrade_reasons.append("ci_not_robust")
                if stage3_reporting.is_interaction_branch(branch_name) and not interaction_upgrade_eligible:
                    downgrade_reasons.append("interaction_explanation_only")
                selection_rows.append(
                    {
                        "model": model_spec.name,
                        "feature_mode": feature_mode,
                        "branch": branch_name,
                        "status": branch_result.status,
                        "n_cont": int(feature_profile.get("n_cont", 0)),
                        "n_bin_onehot": int(feature_profile.get("n_bin_onehot", 0)),
                        "n_missing_indicators": int(feature_profile.get("n_missing_indicators", 0)),
                        "x_columns_sha256": str(feature_profile.get("x_columns_sha256", "")),
                        "hard_audit_pass_count": hard_pass_count,
                        "tier2d_status": tier2d_status,
                        "tier2d_level": tier2d_level,
                        "publish_scope_final": publish_scope_final,
                        "rulebook_reproducible": bool(rulebook_reproducible),
                        "prediction_rule_count": branch_result.decision_payload.get("prediction_rule_count", 0),
                        "delta_p20_mean": p20_mean,
                        "delta_p20_ci_low": p20_low,
                        "delta_p20_ci_high": p20_high,
                        "delta_enrichment20_mean": e20_mean,
                        "delta_enrichment20_ci_low": e20_low,
                        "delta_enrichment20_ci_high": e20_high,
                        "delta_auc_mean": auc_mean,
                        "delta_auc_ci_low": auc_low,
                        "delta_auc_ci_high": auc_high,
                        "p20_mean": float(branch_p20["mean"].iloc[0]) if not branch_p20.empty else np.nan,
                        "p20_ci_low": float(branch_p20["ci_low_95"].iloc[0]) if not branch_p20.empty else np.nan,
                        "enrichment20_mean": float(branch_e20["mean"].iloc[0]) if not branch_e20.empty else np.nan,
                        "enrichment20_ci_low": float(branch_e20["ci_low_95"].iloc[0]) if not branch_e20.empty else np.nan,
                        "auc_mean": float(branch_auc["mean"].iloc[0]) if not branch_auc.empty else np.nan,
                        "auc_ci_low": float(branch_auc["ci_low_95"].iloc[0]) if not branch_auc.empty else np.nan,
                        "c3_p20_mean": float(branch_c3_p20["mean"].iloc[0]) if not branch_c3_p20.empty else np.nan,
                        "c3_enrichment20_mean": float(branch_c3_e20["mean"].iloc[0]) if not branch_c3_e20.empty else np.nan,
                        "applicability_domain": branch_result.decision_payload.get("applicability_domain", "all"),
                        "warning_present": warning_present,
                        "interaction_mode": interaction_mode,
                        "interaction_upgrade_eligible": bool(interaction_upgrade_eligible),
                        "publishable_c3_pair_count": int(len(branch_result.pair_rulebook_publishable_c3only)),
                        "unstable_explanation_pair_count": int(len(branch_result.pair_rulebook_explanation_unstable_c3only)),
                        "selected_as_primary": selected_as_primary,
                        "downgrade_reason": "|".join(dict.fromkeys([x for x in downgrade_reasons if x])),
                        "skipped_by_config": branch_result.skipped_by_config,
                    }
                )
                comparison_rows.append(
                    {
                        "model": model_spec.name,
                        "feature_mode": feature_mode,
                        "branch": branch_name,
                        "status": branch_result.status,
                        "selected_as_primary": selected_as_primary,
                        "negative_control": (
                            branch_result.audit_summary.loc[branch_result.audit_summary["audit_name"] == "negative_control", "status"].iloc[0]
                            if not branch_result.audit_summary.empty and (branch_result.audit_summary["audit_name"] == "negative_control").any()
                            else "NA"
                        ),
                        "controlled_missingness": (
                            branch_result.audit_summary.loc[branch_result.audit_summary["audit_name"] == "controlled_missingness_parallel", "status"].iloc[0]
                            if not branch_result.audit_summary.empty and (branch_result.audit_summary["audit_name"] == "controlled_missingness_parallel").any()
                            else "NA"
                        ),
                        "tier2d": tier2d_status,
                        "tier2d_level": tier2d_level,
                        "interaction_mode": interaction_mode,
                        "rulebook_reproducible": rulebook_reproducible,
                        "delta_p20_ci_low": p20_low,
                        "delta_enrichment20_ci_low": e20_low,
                        "applicability_domain": branch_result.decision_payload.get("applicability_domain", "all"),
                        "publish_scope_final": publish_scope_final,
                        "downgrade_reason": selection_rows[-1]["downgrade_reason"],
                    }
                )

        pred_out = pd.concat(model_prediction_frames, ignore_index=True) if model_prediction_frames else pd.DataFrame()
        fold_out = pd.concat(model_fold_frames, ignore_index=True) if model_fold_frames else pd.DataFrame()
        trace_out = pd.concat(model_trace_frames, ignore_index=True) if model_trace_frames else pd.DataFrame()
        candidate_out = pd.concat(model_candidate_frames, ignore_index=True) if model_candidate_frames else pd.DataFrame()
        pred_out.to_csv(model_dir / "oof_predictions.csv", index=False)
        fold_out.to_csv(model_dir / "fold_metrics.csv", index=False)
        candidate_out.to_csv(model_dir / "candidate_selection_trace.csv", index=False)
        trace_out.to_csv(model_dir / "audit_trace.csv", index=False)
        model_level_predictions.append(pred_out)
        model_level_fold_metrics.append(fold_out)
        model_level_audit_trace.append(trace_out)

    selection_df = pd.DataFrame(selection_rows)
    if not selection_df.empty:
        selection_df["ci_evidence_ok"] = (selection_df["delta_p20_ci_low"].fillna(-np.inf) >= 0) & (selection_df["delta_enrichment20_ci_low"].fillna(-np.inf) >= 0)
        selection_df["status_completed"] = (selection_df["status"].astype(str) == "COMPLETED")
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
                (selection_df["tier2d_status"] == "PASS").astype(int),
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

        branch_subset = selection_df[selection_df["branch"].isin([MAINLINE_BRANCH, PAIRWISE_BRANCH])].copy()
        if not branch_subset.empty:
            for _, grp in branch_subset.groupby(["model", "feature_mode"], sort=False):
                best_idx = int(grp.index[0])
                selection_df.loc[best_idx, "is_branch_winner"] = True

        primary_candidates: list[int] = []
        for model_name in sorted(PRIMARY_WINNER_MODELS):
            idx = pick_model_internal_winner(
                selection_df,
                model_name=model_name,
                feature_mode=PRIMARY_WINNER_FEATURE_MODE,
                allow_pair_override=True,
            )
            if idx is not None:
                primary_candidates.append(int(idx))
        primary_candidates_ci = [idx for idx in primary_candidates if bool(selection_df.loc[idx, "ci_evidence_ok"])]
        primary_idx = first_ranked_index(selection_df, primary_candidates_ci or primary_candidates)
        if primary_idx is not None:
            selection_df.loc[primary_idx, "is_primary_winner"] = True

        control_candidates: list[int] = []
        for model_name in sorted(selection_df[selection_df["feature_mode"] == CONTROL_WINNER_FEATURE_MODE]["model"].astype(str).unique().tolist()):
            idx = pick_model_internal_winner(
                selection_df,
                model_name=model_name,
                feature_mode=CONTROL_WINNER_FEATURE_MODE,
                allow_pair_override=True,
            )
            if idx is not None:
                control_candidates.append(int(idx))
        control_candidates_ci = [idx for idx in control_candidates if bool(selection_df.loc[idx, "ci_evidence_ok"])]
        control_idx = first_ranked_index(selection_df, control_candidates_ci or control_candidates)
        if control_idx is not None:
            selection_df.loc[control_idx, "is_control_winner"] = True

        selection_df["winner_role"] = "none"
        selection_df.loc[selection_df["is_branch_winner"], "winner_role"] = "branch_winner"
        selection_df.loc[selection_df["is_control_winner"], "winner_role"] = "control_winner"
        selection_df.loc[selection_df["is_primary_winner"], "winner_role"] = "primary_winner"
        selection_df["selected_as_primary"] = selection_df["is_primary_winner"].astype(bool)
    return (
        comparison_rows,
        pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame(),
        pd.concat(metrics_c3_frames, ignore_index=True) if metrics_c3_frames else pd.DataFrame(),
        pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame(),
        selection_df.to_dict("records"),
        model_level_predictions,
        model_level_fold_metrics,
        model_level_audit_trace,
    )


def run_year_sensitivity(
    model_spec: ModelSpec,
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    *,
    random_seed: int,
    warning_log: Path,
) -> pd.DataFrame:
    base_no_year, _ = stage3_features.prepare_base_features(df.copy(), include_missing_indicators=True, include_year_sensitivity=False)
    base_with_year, _ = stage3_features.prepare_base_features(df.copy(), include_missing_indicators=True, include_year_sensitivity=True)
    base_no_year_profile = stage3_features.summarize_feature_frame(base_no_year)
    base_with_year_profile = stage3_features.summarize_feature_frame(base_with_year)
    branch_base = BranchData(
        branch_name="mainline_plus_pairwise",
        feature_mode="cont_plus_bin",
        X=base_no_year,
        feature_profile=base_no_year_profile,
        rulebook_support=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_legacy=pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS),
        rulebook_debug=pd.DataFrame(),
        pair_rulebook_publishable_c3only=pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS),
        pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
        candidate_trace=pd.DataFrame(),
        rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
        model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
        linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
        linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
        linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
        skipped_by_config=False,
    )
    branch_year = BranchData(
        branch_name="mainline_plus_pairwise_year",
        feature_mode="cont_plus_bin",
        X=base_with_year,
        feature_profile=base_with_year_profile,
        rulebook_support=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_legacy=pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS),
        rulebook_debug=pd.DataFrame(),
        pair_rulebook_publishable_c3only=pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS),
        pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
        candidate_trace=pd.DataFrame(),
        rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
        model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
        linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
        linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
        linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
        skipped_by_config=False,
    )
    base_metrics, _, _, _, _, _ = stage3_models.evaluate_model_branch(
        model_spec,
        branch_base,
        df,
        y,
        splits,
        random_seed=random_seed,
        warning_log=warning_log,
        pair_limit=PAIR_LIMIT,
        triple_limit=TRIPLE_LIMIT,
        top_ks=TOP_KS,
    )
    year_metrics, _, _, _, _, _ = stage3_models.evaluate_model_branch(
        model_spec,
        branch_year,
        df,
        y,
        splits,
        random_seed=random_seed + 100,
        warning_log=warning_log,
        pair_limit=PAIR_LIMIT,
        triple_limit=TRIPLE_LIMIT,
        top_ks=TOP_KS,
    )
    base_ci = summarize_metric_ci(base_metrics, model_spec.name, "no_year")
    year_ci = summarize_metric_ci(year_metrics, model_spec.name, "with_year")
    delta_ci = summarize_delta_ci(base_metrics, year_metrics, model_spec.name, "with_year")
    out = pd.concat([base_ci, year_ci, delta_ci.assign(branch="delta_with_year_vs_no_year")], ignore_index=True)
    out.insert(0, "sensitivity", "+year")
    return out


def run_missingness_ablation(
    model_spec: ModelSpec,
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    *,
    random_seed: int,
    warning_log: Path,
) -> pd.DataFrame:
    subset = df[df["coverage_tier"].isin(["C2", "C3"])].copy()
    if subset.empty:
        return pd.DataFrame([{"sensitivity": "missingness_ablation", "status": "SKIPPED", "details": "no_C2C3_rows"}])
    subset_y = is_strict_positive(subset).to_numpy(dtype=int)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(subset.index.tolist())}
    subset = subset.reset_index(drop=True)
    subset_splits: list[dict[str, Any]] = []
    for sp in splits:
        train_idx = [idx for idx in sp["train_idx"] if idx < len(df) and df.iloc[idx]["coverage_tier"] in {"C2", "C3"}]
        test_idx = [idx for idx in sp["test_idx"] if idx < len(df) and df.iloc[idx]["coverage_tier"] in {"C2", "C3"}]
        if len(train_idx) < 20 or len(test_idx) < 10:
            continue
        new_train = np.array([old_to_new[idx] for idx in train_idx if idx in old_to_new], dtype=int)
        new_test = np.array([old_to_new[idx] for idx in test_idx if idx in old_to_new], dtype=int)
        if len(new_train) == 0 or len(new_test) == 0:
            continue
        if subset_y[new_train].sum() < 10 or subset_y[new_test].sum() < 3:
            continue
        subset_splits.append({"split_id": sp["split_id"], "train_idx": new_train, "test_idx": new_test})
    if not subset_splits:
        return pd.DataFrame([{"sensitivity": "missingness_ablation", "status": "SKIPPED", "details": "insufficient_C2C3_splits"}])
    with_missing, _ = stage3_features.prepare_base_features(subset.copy(), include_missing_indicators=True, include_year_sensitivity=False)
    without_missing, _ = stage3_features.prepare_base_features(subset.copy(), include_missing_indicators=False, include_year_sensitivity=False)
    with_missing_profile = stage3_features.summarize_feature_frame(with_missing)
    without_missing_profile = stage3_features.summarize_feature_frame(without_missing)
    branch_with = BranchData(
        branch_name="with_missing",
        feature_mode="cont_plus_bin",
        X=with_missing,
        feature_profile=with_missing_profile,
        rulebook_support=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_legacy=pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS),
        rulebook_debug=pd.DataFrame(),
        pair_rulebook_publishable_c3only=pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS),
        pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
        candidate_trace=pd.DataFrame(),
        rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
        model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
        linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
        linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
        linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
        skipped_by_config=False,
    )
    branch_without = BranchData(
        branch_name="without_missing",
        feature_mode="cont_plus_bin",
        X=without_missing,
        feature_profile=without_missing_profile,
        rulebook_support=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
        rulebook_legacy=pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS),
        rulebook_debug=pd.DataFrame(),
        pair_rulebook_publishable_c3only=pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS),
        pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
        candidate_trace=pd.DataFrame(),
        rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
        model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
        linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
        linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
        linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
        skipped_by_config=False,
    )
    metrics_with, _, _, _, _, _ = stage3_models.evaluate_model_branch(
        model_spec,
        branch_with,
        subset,
        subset_y,
        subset_splits,
        random_seed=random_seed,
        warning_log=warning_log,
        pair_limit=PAIR_LIMIT,
        triple_limit=TRIPLE_LIMIT,
        top_ks=TOP_KS,
    )
    metrics_without, _, _, _, _, _ = stage3_models.evaluate_model_branch(
        model_spec,
        branch_without,
        subset,
        subset_y,
        subset_splits,
        random_seed=random_seed + 200,
        warning_log=warning_log,
        pair_limit=PAIR_LIMIT,
        triple_limit=TRIPLE_LIMIT,
        top_ks=TOP_KS,
    )
    with_ci = summarize_metric_ci(metrics_with, model_spec.name, "with_missing")
    without_ci = summarize_metric_ci(metrics_without, model_spec.name, "without_missing")
    delta_ci = summarize_delta_ci(without_metrics:=metrics_without, metric_df:=metrics_with, model_name=model_spec.name, branch_name="with_missing")
    out = pd.concat([without_ci, with_ci, delta_ci.assign(branch="delta_with_vs_without_missing")], ignore_index=True)
    out.insert(0, "sensitivity", "missingness_ablation_tier2plus")
    return out


def build_rulebook_bridge_summary(selection_df: pd.DataFrame, part2_out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in selection_df.iterrows():
        model = row["model"]
        feature_mode = row.get("feature_mode", "cont_plus_bin")
        branch = row["branch"]
        branch_dir = stage3_reporting.model_branch_dir(part2_out_dir, str(model), str(feature_mode), str(branch))
        legacy_path = branch_dir / "rulebook_legacy_pair_tier2plus.csv"
        support_path = branch_dir / "rulebook_support.csv"
        legacy_df = pd.read_csv(legacy_path) if legacy_path.exists() and legacy_path.stat().st_size > 0 else pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS)
        support_df = pd.read_csv(support_path) if support_path.exists() and support_path.stat().st_size > 0 else pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS)
        rows.append(
            {
                "model": model,
                "feature_mode": feature_mode,
                "branch": branch,
                "legacy_schema_ok": list(legacy_df.columns) == LEGACY_RULEBOOK_COLUMNS if not legacy_df.empty else True,
                "support_schema_ok": all(col in support_df.columns for col in SUPPORT_RULEBOOK_COLUMNS),
                "legacy_rule_count": int(len(legacy_df)),
                "support_rule_count": int(len(support_df)),
            }
        )
    return pd.DataFrame(rows)


def build_warning_summary(part2_out_dir: Path, selection_df: pd.DataFrame) -> dict[str, Any]:
    root_warning_log = part2_out_dir / "run_warning.log"
    root_lines = root_warning_log.read_text(encoding="utf-8").splitlines() if root_warning_log.exists() else []
    branches: list[dict[str, Any]] = []
    for _, row in selection_df.iterrows():
        branch_log = stage3_reporting.model_branch_dir(part2_out_dir, str(row["model"]), str(row.get("feature_mode", "cont_plus_bin")), str(row["branch"])) / "run_warning.log"
        lines = branch_log.read_text(encoding="utf-8").splitlines() if branch_log.exists() else []
        branches.append(
            {
                "model": row["model"],
                "feature_mode": row.get("feature_mode", "cont_plus_bin"),
                "branch": row["branch"],
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
    write_json(part2_out_dir / "warning_summary.json", payload)
    return payload


def run_threshold_grid(
    model_spec: ModelSpec,
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    *,
    random_seed: int,
    pair_candidates: pd.DataFrame,
    triple_candidates: pd.DataFrame,
    base_X: pd.DataFrame,
    enable_3way: bool,
    warning_log: Path,
    tier2d_common_k_min: int = TIER2D_COMMON_K_MIN,
    tier2d_min_test_c2_n: int = TIER2D_MIN_TEST_C2_N,
    tier2d_min_test_c3_n: int = TIER2D_MIN_TEST_C3_N,
    tier2d_raw_diff_eps: float = TIER2D_RAW_DIFF_EPS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    feature_profile = stage3_features.summarize_feature_frame(base_X)
    for threshold in get_threshold_grid():
        branch_data = build_branch_data(
            model_spec,
            "mainline_plus_pairwise",
            "cont_plus_bin",
            df,
            base_X,
            feature_profile,
            threshold,
            pair_candidates,
            triple_candidates,
            enable_3way=enable_3way,
        )
        metric_df, pred_df, _, _, _, _ = stage3_models.evaluate_model_branch(
            model_spec,
            branch_data,
            df,
            y,
            splits,
            random_seed=random_seed + 500,
            warning_log=warning_log,
            pair_limit=PAIR_LIMIT,
            triple_limit=TRIPLE_LIMIT,
            top_ks=TOP_KS,
        )
        base_branch = BranchData(
            branch_name="mainline",
            feature_mode="cont_plus_bin",
            X=base_X,
            feature_profile=feature_profile,
            rulebook_support=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
            rulebook_support_engineered_comparison=pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS),
            rulebook_legacy=pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS),
            rulebook_debug=pd.DataFrame(),
            pair_rulebook_publishable_c3only=pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS),
            pair_rulebook_explanation_unstable_c3only=pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS),
            candidate_trace=pd.DataFrame(),
            rulebook_model_derived_sensitivity=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_RULEBOOK_COLUMNS),
            model_derived_cutpoint_alignment=pd.DataFrame(columns=stage3_reporting.MODEL_DERIVED_CUTPOINT_COLUMNS),
            linear_continuous_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_CONTINUOUS_EFFECT_COLUMNS),
            linear_pairwise_effects=pd.DataFrame(columns=stage3_reporting.LINEAR_PAIRWISE_EFFECT_COLUMNS),
            linear_vs_engineered_direction_check=pd.DataFrame(columns=stage3_reporting.LINEAR_DIRECTION_CHECK_COLUMNS),
            skipped_by_config=False,
        )
        base_metrics, _, _, _, _, _ = stage3_models.evaluate_model_branch(
            model_spec,
            base_branch,
            df,
            y,
            splits,
            random_seed=random_seed + 600,
            warning_log=warning_log,
            pair_limit=PAIR_LIMIT,
            triple_limit=TRIPLE_LIMIT,
            top_ks=TOP_KS,
        )
        delta_ci = summarize_delta_ci(base_metrics, metric_df, model_spec.name, "grid")
        audit_summary, _, applicability_domain, tier2d_level, _, _, _, _ = evaluate_branch_audits(
            model_spec,
            branch_data,
            df,
            y,
            splits,
            pred_df,
            warning_log,
            random_seed,
            common_k_min=tier2d_common_k_min,
            min_test_c2_n=tier2d_min_test_c2_n,
            min_test_c3_n=tier2d_min_test_c3_n,
            raw_diff_eps=tier2d_raw_diff_eps,
        )
        dp20_mean, dp20_low, dp20_high = get_delta_summary(delta_ci, "P", 20)
        de20_mean, de20_low, de20_high = get_delta_summary(delta_ci, "Enrichment", 20)
        rows.append(
            {
                "threshold_grid_id": threshold.threshold_id,
                "support_n": threshold.support_n,
                "support_pos": threshold.support_pos,
                "selection_freq_pair": threshold.pair_freq,
                "selection_freq_triple": threshold.triple_freq,
                "prediction_pair_count": int((branch_data.rulebook_support["rule_type"] == "prediction").sum()) if not branch_data.rulebook_support.empty else 0,
                "delta_p20_mean": dp20_mean,
                "delta_p20_ci_low": dp20_low,
                "delta_p20_ci_high": dp20_high,
                "delta_enrichment20_mean": de20_mean,
                "delta_enrichment20_ci_low": de20_low,
                "delta_enrichment20_ci_high": de20_high,
                "hard_audit_pass_count": int((audit_summary[audit_summary["audit_name"].isin(HARD_AUDITS)]["status"] == "PASS").sum()) if not audit_summary.empty else 0,
                "tier2d_status": audit_summary.loc[audit_summary["audit_name"] == SOFT_AUDIT_NAME, "status"].iloc[0] if not audit_summary.empty and (audit_summary["audit_name"] == SOFT_AUDIT_NAME).any() else "PASS",
                "tier2d_level": tier2d_level,
                "applicability_domain": applicability_domain,
                "sensitivity_only": True,
            }
        )
    return pd.DataFrame(rows)


def model_zoo_markdown(selection_df: pd.DataFrame, *, language: str = "zh") -> str:
    title = "# Model Zoo Comparison" if language == "en" else "# 模型横向对比"
    lines = [
        title,
        "",
        "| model | feature_mode | branch | status | hard_audits | rulebook_ok | tier2d | interaction_mode | dP20_low | dE20_low | winner_role | primary | control | branch_winner | downgrade |",
        "|---|---|---|---:|---:|---:|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in selection_df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.model),
                    str(getattr(row, "feature_mode", "cont_plus_bin")),
                    str(row.branch),
                    str(row.status),
                    str(row.hard_audit_pass_count),
                    str(bool(row.rulebook_reproducible)),
                    f"{row.tier2d_status}/{getattr(row, 'tier2d_level', 'PASS')}",
                    str(getattr(row, "interaction_mode", "NA")),
                    fmt_float(float(row.delta_p20_ci_low)),
                    fmt_float(float(row.delta_enrichment20_ci_low)),
                    str(getattr(row, "winner_role", "none")),
                    str(bool(row.selected_as_primary)),
                    str(bool(getattr(row, "is_control_winner", False))),
                    str(bool(getattr(row, "is_branch_winner", False))),
                    str(row.downgrade_reason or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def run_level_conclusion(
    selection_df: pd.DataFrame,
    threshold_grid_used: bool,
    *,
    part2_out_dir: Path,
    root_warning_present: bool,
) -> tuple[str, str, dict[str, Any]]:
    primary_rows = selection_df[selection_df["is_primary_winner"] == True] if not selection_df.empty else pd.DataFrame()
    primary = primary_rows.iloc[0] if not primary_rows.empty else (selection_df.iloc[0] if not selection_df.empty else pd.Series(dtype=object))
    control_rows = selection_df[selection_df["is_control_winner"] == True] if not selection_df.empty else pd.DataFrame()
    if not control_rows.empty:
        control = control_rows.iloc[0]
    else:
        control_candidates = selection_df[selection_df["feature_mode"] == CONTROL_WINNER_FEATURE_MODE] if not selection_df.empty else pd.DataFrame()
        control = control_candidates.iloc[0] if not control_candidates.empty else pd.Series(dtype=object)
    mainline_upgrade = bool(
        (not primary.empty)
        and primary.get("status") == "COMPLETED"
        and primary.get("hard_audit_pass_count", 0) == len(HARD_AUDITS)
        and bool(primary.get("rulebook_reproducible", False))
        and not bool(primary.get("warning_present", False))
        and (not stage3_reporting.is_interaction_branch(str(primary.get("branch", ""))) or bool(primary.get("interaction_upgrade_eligible", False)))
        and not root_warning_present
    )
    applicability = primary.get("applicability_domain", "all") if not primary.empty else "none"
    tier2d_level = str(primary.get("tier2d_level", "PASS")) if not primary.empty else "PASS"
    publish_scope = primary.get("publish_scope_final", applicability) if not primary.empty else "none"
    bridge_recommended, bridge_reason_codes, top_pair_overlap = compute_bridge_recommendation(selection_df, part2_out_dir)
    zh_lines = [
        "# Stage3 Part II Run Decision",
        "",
        f"- mainline_upgrade: {str(mainline_upgrade).lower()}",
        f"- primary_winner_model: {primary.get('model', 'NA')}",
        f"- primary_winner_feature_mode: {primary.get('feature_mode', 'NA')}",
        f"- primary_winner_branch: {primary.get('branch', 'NA')}",
        f"- control_winner_model: {control.get('model', 'NA') if not control.empty else 'NA'}",
        f"- control_winner_feature_mode: {control.get('feature_mode', 'NA') if not control.empty else 'NA'}",
        f"- control_winner_branch: {control.get('branch', 'NA') if not control.empty else 'NA'}",
        f"- tier2d_level: {tier2d_level}",
        f"- applicability_domain: {applicability}",
        f"- publish_scope: {publish_scope}",
        f"- c3_p20_mean: {fmt_float(float(primary.get('c3_p20_mean', np.nan)) if not primary.empty else np.nan)}",
        f"- c3_enrichment20_mean: {fmt_float(float(primary.get('c3_enrichment20_mean', np.nan)) if not primary.empty else np.nan)}",
        f"- bridge_recommended: {str(bridge_recommended).lower()}",
        f"- bridge_reason_codes: {'|'.join(bridge_reason_codes) if bridge_reason_codes else 'none'}",
        f"- bridge_top_pair_overlap: {fmt_float(top_pair_overlap)}",
        f"- threshold_grid_used: {str(threshold_grid_used).lower()}",
        f"- root_warning_present: {str(root_warning_present).lower()}",
        f"- selected_branch_warning_present: {str(bool(primary.get('warning_present', False)) if not primary.empty else False).lower()}",
        f"- downgrade_reason: {primary.get('downgrade_reason', 'NA')}",
    ]
    en_lines = [
        "# Stage3 Part II Run Decision",
        "",
        f"- mainline_upgrade: {str(mainline_upgrade).lower()}",
        f"- primary_winner_model: {primary.get('model', 'NA')}",
        f"- primary_winner_feature_mode: {primary.get('feature_mode', 'NA')}",
        f"- primary_winner_branch: {primary.get('branch', 'NA')}",
        f"- control_winner_model: {control.get('model', 'NA') if not control.empty else 'NA'}",
        f"- control_winner_feature_mode: {control.get('feature_mode', 'NA') if not control.empty else 'NA'}",
        f"- control_winner_branch: {control.get('branch', 'NA') if not control.empty else 'NA'}",
        f"- tier2d_level: {tier2d_level}",
        f"- applicability_domain: {applicability}",
        f"- publish_scope: {publish_scope}",
        f"- c3_p20_mean: {fmt_float(float(primary.get('c3_p20_mean', np.nan)) if not primary.empty else np.nan)}",
        f"- c3_enrichment20_mean: {fmt_float(float(primary.get('c3_enrichment20_mean', np.nan)) if not primary.empty else np.nan)}",
        f"- bridge_recommended: {str(bridge_recommended).lower()}",
        f"- bridge_reason_codes: {'|'.join(bridge_reason_codes) if bridge_reason_codes else 'none'}",
        f"- bridge_top_pair_overlap: {fmt_float(top_pair_overlap)}",
        f"- threshold_grid_used: {str(threshold_grid_used).lower()}",
        f"- root_warning_present: {str(root_warning_present).lower()}",
        f"- selected_branch_warning_present: {str(bool(primary.get('warning_present', False)) if not primary.empty else False).lower()}",
        f"- downgrade_reason: {primary.get('downgrade_reason', 'NA')}",
    ]
    payload = {
        "mainline_upgrade": mainline_upgrade,
        "selected_model": primary.get("model", "NA") if not primary.empty else "NA",
        "selected_feature_mode": primary.get("feature_mode", "NA") if not primary.empty else "NA",
        "selected_branch": primary.get("branch", "NA") if not primary.empty else "NA",
        "primary_winner_model": primary.get("model", "NA") if not primary.empty else "NA",
        "primary_winner_feature_mode": primary.get("feature_mode", "NA") if not primary.empty else "NA",
        "primary_winner_branch": primary.get("branch", "NA") if not primary.empty else "NA",
        "control_winner_model": control.get("model", "NA") if not control.empty else "NA",
        "control_winner_feature_mode": control.get("feature_mode", "NA") if not control.empty else "NA",
        "control_winner_branch": control.get("branch", "NA") if not control.empty else "NA",
        "tier2d_level": tier2d_level,
        "applicability_domain": applicability,
        "publish_scope": publish_scope,
        "c3_p20_mean": float(primary.get("c3_p20_mean", np.nan)) if not primary.empty else np.nan,
        "c3_enrichment20_mean": float(primary.get("c3_enrichment20_mean", np.nan)) if not primary.empty else np.nan,
        "bridge_recommended": bool(bridge_recommended),
        "bridge_reason_codes": list(bridge_reason_codes),
        "bridge_top_pair_overlap": float(top_pair_overlap) if np.isfinite(top_pair_overlap) else np.nan,
        "threshold_grid_used": threshold_grid_used,
        "root_warning_present": root_warning_present,
        "selected_branch_warning_present": bool(primary.get("warning_present", False)) if not primary.empty else False,
        "downgrade_reason": primary.get("downgrade_reason", "NA") if not primary.empty else "NA",
    }
    return "\n".join(zh_lines) + "\n", "\n".join(en_lines) + "\n", payload


def upsert_run_section(md_path: Path, run_id: str, section_markdown: str) -> None:
    ensure_dir(md_path.parent)
    start_marker = f"<!-- STAGE3_RUN:{run_id}:START -->"
    end_marker = f"<!-- STAGE3_RUN:{run_id}:END -->"
    wrapped = f"{start_marker}\n{section_markdown.rstrip()}\n{end_marker}\n"
    if not md_path.exists():
        md_path.write_text("# Stage3 Execution Log\n\n" + wrapped, encoding="utf-8")
        return
    text = md_path.read_text(encoding="utf-8")
    if start_marker in text and end_marker in text:
        pre, rest = text.split(start_marker, 1)
        _, post = rest.split(end_marker, 1)
        new_text = pre.rstrip() + "\n\n" + wrapped + post.lstrip("\n")
    else:
        base = text.rstrip()
        if base:
            base += "\n\n"
        new_text = base + wrapped
    md_path.write_text(new_text, encoding="utf-8")


def build_run_manifest_json(
    manifest: dict[str, Any],
    repo_root: Path,
    part2_out_dir: Path,
    splits_hash: str,
    *,
    split_min_test_c2_n: int = SPLIT_MIN_TEST_C2_N,
    split_min_test_c3_n: int = SPLIT_MIN_TEST_C3_N,
    tier2d_common_k_min: int = TIER2D_COMMON_K_MIN,
    tier2d_min_test_c2_n: int = TIER2D_MIN_TEST_C2_N,
    tier2d_min_test_c3_n: int = TIER2D_MIN_TEST_C3_N,
    tier2d_raw_diff_eps: float = TIER2D_RAW_DIFF_EPS,
) -> dict[str, Any]:
    input_csv = resolve_repo_path(repo_root, str(manifest.get("paths", {}).get("input_csv", "data/AI_database_chrome__llm_final_R1.csv")))
    label_spec = "g3_strict_v1: llm_ai_dc_label in {ai_specific, ai_optimized} AND (accel_model non-missing OR accel_count non-missing)"
    feature_spec = {
        "cat_base_cols": CAT_BASE_COLS,
        "continuous_source_cols": CONTINUOUS_SOURCE_COLS,
        "num_base_cols": NUM_BASE_COLS,
        "leakage_prefixes": list(LEAKAGE_PREFIXES),
        "leakage_exact": sorted(LEAKAGE_EXACT),
        "threshold_presets": THRESHOLD_PRESETS,
        "pair_limit": PAIR_LIMIT,
        "triple_limit": TRIPLE_LIMIT,
        "pair_discovery_limit": PAIR_DISCOVERY_LIMIT,
        "triple_discovery_limit": TRIPLE_DISCOVERY_LIMIT,
        "pair_candidate_scope": PAIR_CANDIDATE_SCOPE,
        "pair_publish_scope": PAIR_PUBLISH_SCOPE,
        "pair_publish_scope_policy": "tier2d_PASS_or_WARN=>C2C3; tier2d_FAIL=>C3_only",
        "triple_candidate_scope": TRIPLE_CANDIDATE_SCOPE,
        "split_min_test_c2_n": int(split_min_test_c2_n),
        "split_min_test_c3_n": int(split_min_test_c3_n),
        "tier2d_common_k_min": int(tier2d_common_k_min),
        "tier2d_min_test_c2_n": int(tier2d_min_test_c2_n),
        "tier2d_min_test_c3_n": int(tier2d_min_test_c3_n),
        "tier2d_raw_diff_eps": float(tier2d_raw_diff_eps),
        "signal_group_mapping": SIGNAL_GROUP_MAPPING,
        "enforce_cross_signal_publishable": ENFORCE_CROSS_SIGNAL_PUBLISHABLE,
        "feature_modes_available": list(getattr(stage3_features, "FEATURE_MODES", ["cont_plus_bin"])),
    }
    payload = {
        "run_id": manifest.get("run", {}).get("run_id", "unknown"),
        "input_csv_path": str(input_csv),
        "input_csv_hash": sha256_file(input_csv) if input_csv.exists() else "",
        "label_spec_hash": sha256_text(label_spec),
        "feature_spec_hash": sha256_text(safe_json_dumps(feature_spec)),
        "splits_hash": splits_hash,
        "code_git_hash": git_hash(repo_root),
        "timestamp": stage3_reporting.now_iso(),
        "commandline_args": sys.argv,
        "manifest_snapshot": manifest,
    }
    write_json(part2_out_dir / "run_manifest.json", payload)
    return payload


def update_run_manifest_feature_registry(part2_out_dir: Path, selection_df: pd.DataFrame) -> None:
    manifest_path = part2_out_dir / "run_manifest.json"
    payload = read_json(manifest_path) if manifest_path.exists() else {}
    registry_rows: dict[str, dict[str, Any]] = {}
    if not selection_df.empty:
        dedup_cols = [col for col in ["model", "feature_mode", "branch", "n_cont", "n_bin_onehot", "n_missing_indicators", "x_columns_sha256"] if col in selection_df.columns]
        for row in selection_df[dedup_cols].drop_duplicates().itertuples(index=False):
            model_name = str(getattr(row, "model", ""))
            feature_mode = str(getattr(row, "feature_mode", ""))
            branch_name = str(getattr(row, "branch", ""))
            registry_key = f"{model_name}/{feature_mode}/{branch_name}"
            registry_rows[registry_key] = {
                "model_name": model_name,
                "feature_mode": feature_mode,
                "branch_name": branch_name,
                "n_cont": int(getattr(row, "n_cont", 0)),
                "n_bin_onehot": int(getattr(row, "n_bin_onehot", 0)),
                "n_missing_indicators": int(getattr(row, "n_missing_indicators", 0)),
                "X_columns_sha256": str(getattr(row, "x_columns_sha256", "")),
            }
    payload["feature_set_registry"] = registry_rows
    write_json(manifest_path, payload)


def write_interaction_selected_from_explain_model(part2_out_dir: Path, selection_df: pd.DataFrame) -> None:
    out = selection_df.copy() if selection_df is not None else pd.DataFrame()
    for col in EXPLAIN_MODEL_SELECTION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    extra_cols = [col for col in out.columns if col not in EXPLAIN_MODEL_SELECTION_COLUMNS]
    ordered_cols = EXPLAIN_MODEL_SELECTION_COLUMNS + extra_cols
    out = out.reindex(columns=ordered_cols)
    out.to_csv(part2_out_dir / "interaction_selected_from_explain_model.csv", index=False)


def assert_artifacts_complete(part2_out_dir: Path) -> None:
    report_path = part2_out_dir / "artifact_completeness_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    missing = report.get("missing_files", [])
    bad_schema = [item["check"] for item in report.get("schema_checks", []) if not item.get("pass")]
    if missing or bad_schema or not report.get("overall_pass", False):
        print("Artifact completeness check failed.")
        if missing:
            print("Missing files:")
            for item in missing:
                print(f"- {item}")
        if bad_schema:
            print("Schema failures:")
            for item in bad_schema:
                print(f"- {item}")
        raise SystemExit(2)


def main() -> None:
    global ENFORCE_CROSS_SIGNAL_PUBLISHABLE, EXPORT_DEBUG_ARTIFACTS, PAIR_LIMIT, TRIPLE_LIMIT, PAIR_DISCOVERY_LIMIT, TRIPLE_DISCOVERY_LIMIT
    global PAIR_CANDIDATE_SCOPE, PAIR_PUBLISH_SCOPE, TRIPLE_CANDIDATE_SCOPE
    parser = argparse.ArgumentParser(description="Stage3 Part II - Model Execution And Audit")
    parser.add_argument("--manifest", required=True, help="Path to run_manifest.yaml")
    parser.add_argument("--min-test-c2-n", type=int, default=None, help="Override min C2 rows in each test split")
    parser.add_argument("--min-test-c3-n", type=int, default=None, help="Override min C3 rows in each test split")
    parser.add_argument("--common-k-min", type=int, default=None, help="Override common_k_min for tier2d audit")
    parser.add_argument("--tier2d-min-test-c2-n", type=int, default=None, help="Override tier2d min C2 rows per split")
    parser.add_argument("--tier2d-min-test-c3-n", type=int, default=None, help="Override tier2d min C3 rows per split")
    parser.add_argument("--raw-diff-eps", type=float, default=None, help="Override raw_diff eps guard for matched reduction rate")
    args = parser.parse_args()

    repo_root = repo_root_from_file(__file__)
    manifest_path = resolve_repo_path(repo_root, args.manifest)
    manifest = load_simple_yaml(manifest_path)

    paths = manifest.get("paths", {})
    run = manifest.get("run", {})
    execution = manifest.get("execution", {})

    part1_out_dir = resolve_repo_path(repo_root, str(paths.get("part1_out_dir")))
    part2_out_dir = resolve_repo_path(repo_root, str(paths.get("part2_out_dir")))
    log_dir = resolve_repo_path(repo_root, str(paths.get("log_dir")))
    ensure_dir(part2_out_dir)
    ensure_dir(log_dir)

    part2_log = log_dir / "part2.log"
    root_warning_log = part2_out_dir / "run_warning.log"
    ensure_file(root_warning_log)
    append_log(part2_log, f"run_id={run.get('run_id', 'unknown')} part2_start")

    gate_file = part1_out_dir / "gate_c_acceptance.json"
    feature_view_file = part1_out_dir / "interaction_feature_view.csv"
    if not gate_file.exists():
        raise FileNotFoundError(f"Missing Gate C file: {gate_file}")
    if not feature_view_file.exists():
        raise FileNotFoundError(f"Missing interaction feature view: {feature_view_file}")

    gate = read_json(gate_file)
    if gate.get("gate_c_status") != "PASS":
        raise RuntimeError(f"Gate C not PASS: {gate}")

    df = pd.read_csv(feature_view_file)
    required_cols = set(CAT_BASE_COLS + NUM_BASE_COLS + ["company", "coverage_tier", "base_non_missing_count", "llm_ai_dc_label", "accel_model", "accel_count"])
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise RuntimeError(f"Part II missing required columns from Part I output: {missing_cols}")

    df_trainable = df[df["base_non_missing_count"] >= 2].copy().reset_index(drop=True)
    y = is_strict_positive(df_trainable).to_numpy(dtype=int)

    random_seed = int(run.get("random_seed", 20260303))
    n_repeats = int(execution.get("n_repeats", 100))
    min_valid_splits = int(execution.get("min_valid_splits", 80))
    enable_3way = bool(execution.get("enable_3way", False))
    ENFORCE_CROSS_SIGNAL_PUBLISHABLE = bool(execution.get("enforce_cross_signal_publishable", True))
    EXPORT_DEBUG_ARTIFACTS = bool(execution.get("export_debug_artifacts", True))
    PAIR_LIMIT = int(execution.get("pair_limit", PAIR_LIMIT))
    TRIPLE_LIMIT = int(execution.get("triple_limit", TRIPLE_LIMIT))
    PAIR_DISCOVERY_LIMIT = int(execution.get("pair_discovery_limit", PAIR_DISCOVERY_LIMIT))
    TRIPLE_DISCOVERY_LIMIT = int(execution.get("triple_discovery_limit", TRIPLE_DISCOVERY_LIMIT))
    PAIR_CANDIDATE_SCOPE = normalize_pair_candidate_scope(execution.get("pair_candidate_scope", PAIR_CANDIDATE_SCOPE))
    PAIR_PUBLISH_SCOPE = normalize_pair_publish_scope(execution.get("pair_publish_scope", PAIR_PUBLISH_SCOPE), manifest)
    TRIPLE_CANDIDATE_SCOPE = normalize_triple_candidate_scope(execution.get("triple_candidate_scope", TRIPLE_CANDIDATE_SCOPE))
    model_subset_raw = execution.get("model_subset", [])
    branch_subset_raw = execution.get("branch_subset", [])
    feature_modes_raw = execution.get("feature_modes", list(DEFAULT_FEATURE_MODES))
    cont_plus_bin_model_subset_raw = execution.get("cont_plus_bin_model_subset", sorted(DEFAULT_CONT_PLUS_BIN_MODEL_SUBSET))
    model_subset = set(model_subset_raw) if isinstance(model_subset_raw, list) and model_subset_raw else None
    branch_subset = list(branch_subset_raw) if isinstance(branch_subset_raw, list) and branch_subset_raw else None
    feature_modes = list(feature_modes_raw) if isinstance(feature_modes_raw, list) and feature_modes_raw else list(DEFAULT_FEATURE_MODES)
    if isinstance(cont_plus_bin_model_subset_raw, list):
        cont_plus_bin_model_subset = {str(item) for item in cont_plus_bin_model_subset_raw if str(item)}
    else:
        cont_plus_bin_model_subset = set(DEFAULT_CONT_PLUS_BIN_MODEL_SUBSET)
    nonlinear_cont_only_rulebook_policy = resolve_nonlinear_cont_only_rulebook_policy(
        execution,
        n_rows=int(len(df_trainable)),
        n_pos=int(y.sum()),
    )
    os.environ["LOKY_MAX_CPU_COUNT"] = str(int(execution.get("loky_max_cpu_count", 1)))
    split_soft_constraints = manifest.get("split_soft_constraints", {})
    split_min_test_c2_n = int(
        args.min_test_c2_n
        if args.min_test_c2_n is not None
        else execution.get(
            "min_test_c2_n",
            split_soft_constraints.get("min_test_c2_n", SPLIT_MIN_TEST_C2_N) if isinstance(split_soft_constraints, dict) else SPLIT_MIN_TEST_C2_N,
        )
    )
    split_min_test_c3_n = int(
        args.min_test_c3_n
        if args.min_test_c3_n is not None
        else execution.get(
            "min_test_c3_n",
            split_soft_constraints.get("min_test_c3_n", SPLIT_MIN_TEST_C3_N) if isinstance(split_soft_constraints, dict) else SPLIT_MIN_TEST_C3_N,
        )
    )
    tier2d_common_k_min = int(args.common_k_min if args.common_k_min is not None else execution.get("common_k_min", TIER2D_COMMON_K_MIN))
    tier2d_min_test_c2_n = int(
        args.tier2d_min_test_c2_n
        if args.tier2d_min_test_c2_n is not None
        else execution.get("tier2d_min_test_c2_n", execution.get("min_test_c2_n", TIER2D_MIN_TEST_C2_N))
    )
    tier2d_min_test_c3_n = int(
        args.tier2d_min_test_c3_n
        if args.tier2d_min_test_c3_n is not None
        else execution.get("tier2d_min_test_c3_n", execution.get("min_test_c3_n", TIER2D_MIN_TEST_C3_N))
    )
    tier2d_raw_diff_eps = float(args.raw_diff_eps if args.raw_diff_eps is not None else execution.get("raw_diff_eps", TIER2D_RAW_DIFF_EPS))
    if not isinstance(manifest.get("execution"), dict):
        manifest["execution"] = {}
    manifest_execution = manifest["execution"]
    manifest_execution["pair_candidate_scope"] = PAIR_CANDIDATE_SCOPE
    manifest_execution["pair_publish_scope"] = PAIR_PUBLISH_SCOPE
    manifest_execution["min_test_c2_n"] = int(split_min_test_c2_n)
    manifest_execution["min_test_c3_n"] = int(split_min_test_c3_n)
    manifest_execution["common_k_min"] = int(tier2d_common_k_min)
    manifest_execution["tier2d_min_test_c2_n"] = int(tier2d_min_test_c2_n)
    manifest_execution["tier2d_min_test_c3_n"] = int(tier2d_min_test_c3_n)
    manifest_execution["raw_diff_eps"] = float(tier2d_raw_diff_eps)
    manifest_execution["publish_scope_policy"] = "tier2d_PASS_or_WARN=>C2C3; tier2d_FAIL=>C3_only"
    manifest_execution["cont_plus_bin_model_subset"] = sorted(cont_plus_bin_model_subset)
    manifest_execution["nonlinear_cont_only_rulebook"] = dict(nonlinear_cont_only_rulebook_policy)
    if not isinstance(split_soft_constraints, dict):
        split_soft_constraints = {}
    split_soft_constraints["min_test_c2_n"] = int(split_min_test_c2_n)
    split_soft_constraints["min_test_c3_n"] = int(split_min_test_c3_n)
    manifest["split_soft_constraints"] = split_soft_constraints
    unstable_explanation_config = manifest.get("unstable_explanation", {})

    splits, attempts = build_company_splits(
        df_trainable,
        y,
        seed=random_seed,
        n_repeats=n_repeats,
        min_test_c2_n=split_min_test_c2_n,
        min_test_c3_n=split_min_test_c3_n,
        soft_tier_balance=split_soft_constraints,
    )
    append_log(
        part2_log,
        f"split_sampling attempts={attempts} valid_splits={len(splits)} min_test_c2_n={split_min_test_c2_n} min_test_c3_n={split_min_test_c3_n}",
    )
    append_log(
        part2_log,
        f"tier2d_params common_k_min={tier2d_common_k_min} min_test_c2_n={tier2d_min_test_c2_n} min_test_c3_n={tier2d_min_test_c3_n} raw_diff_eps={tier2d_raw_diff_eps:.4f}",
    )
    append_log(
        part2_log,
        (
            f"feature_modes={','.join(feature_modes)} "
            f"cont_plus_bin_model_subset={','.join(sorted(cont_plus_bin_model_subset)) if cont_plus_bin_model_subset else 'NONE'}"
        ),
    )
    append_log(
        part2_log,
        (
            "nonlinear_cont_only_rulebook="
            + safe_json_dumps(nonlinear_cont_only_rulebook_policy).replace("\n", " ")
        ),
    )
    if len(splits) < min_valid_splits:
        stage3_models.append_warning(root_warning_log, f"valid_split_count_below_threshold={len(splits)}")
        raise SystemExit(2)

    split_csv, split_meta, splits_hash = save_splits(
        part2_out_dir,
        df_trainable,
        splits,
        seed=random_seed,
        input_hash=sha256_file(feature_view_file),
        min_test_c2_n=split_min_test_c2_n,
        min_test_c3_n=split_min_test_c3_n,
        soft_tier_balance=split_soft_constraints,
    )
    append_log(part2_log, f"splits_saved csv={split_csv} meta={split_meta}")

    base_feature_sets: dict[str, tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]] = {}
    for feature_mode in feature_modes:
        include_continuous, include_bins = stage3_features.feature_mode_flags(feature_mode)
        feature_X, feature_meta = stage3_features.prepare_base_features(
            df_trainable.copy(),
            include_missing_indicators=True,
            include_year_sensitivity=False,
            include_continuous=include_continuous,
            include_bins=include_bins,
        )
        base_feature_sets[feature_mode] = (
            feature_X,
            feature_meta,
            stage3_features.summarize_feature_frame(feature_X),
        )
    base_X, base_feature_meta, _ = base_feature_sets["cont_plus_bin" if "cont_plus_bin" in base_feature_sets else feature_modes[0]]
    pair_candidates = stage3_candidates.generate_pairwise_candidates(
        df_trainable,
        y,
        splits,
        candidate_scope=PAIR_CANDIDATE_SCOPE,
    )
    pair_candidates.to_csv(part2_out_dir / "interaction_candidates_pairwise.csv", index=False)
    pair_candidates.to_csv(part2_out_dir / "pair_candidates_discovery.csv", index=False)

    l0_threshold = build_primary_threshold(execution)
    selected_pairs_l0, _ = stage3_candidates.classify_candidates(
        pair_candidates,
        l0_threshold,
        pair_limit=PAIR_LIMIT,
        triple_limit=TRIPLE_LIMIT,
        enforce_cross_signal_publishable=ENFORCE_CROSS_SIGNAL_PUBLISHABLE,
    )
    hierarchy_pairs = selected_pairs_l0[(selected_pairs_l0["type"] == "pair") & (selected_pairs_l0["rule_type"] == "prediction")].copy()
    triple_candidates = stage3_candidates.generate_triple_candidates(
        df_trainable,
        y,
        splits,
        hierarchy_pairs,
        candidate_scope=TRIPLE_CANDIDATE_SCOPE,
    )
    triple_candidates.to_csv(part2_out_dir / "interaction_candidates_3way.csv", index=False)

    sensitivity_refs = {
        "year": str((part2_out_dir / "interaction_sensitivity_year_summary.csv").relative_to(repo_root)),
        "missingness": str((part2_out_dir / "interaction_ablation_missingflags_tier2plus.csv").relative_to(repo_root)),
        "threshold": str((part2_out_dir / "threshold_grid_summary.csv").relative_to(repo_root)),
        "bridge": str((part2_out_dir / "rulebook_bridge_summary.csv").relative_to(repo_root)),
    }
    branch_order = branch_subset or ["mainline", "mainline_plus_pairwise", "mainline_plus_gated_3way"]
    model_specs = build_model_specs(model_subset=model_subset)
    explain_selection_df = build_explain_model_selection_table(
        model_specs,
        feature_modes,
        branch_order,
        threshold=l0_threshold,
        pair_candidates=pair_candidates,
        triple_candidates=triple_candidates,
        enable_3way=enable_3way,
        cont_plus_bin_model_subset=cont_plus_bin_model_subset,
    )
    write_interaction_selected_from_explain_model(part2_out_dir, explain_selection_df)

    comparison_rows, metrics_summary, metrics_c3_summary, audit_summary_root, selection_records, _, _, _ = execute_model_zoo(
        part2_out_dir,
        df_trainable,
        y,
        splits,
        random_seed=random_seed,
        threshold=l0_threshold,
        pair_candidates=pair_candidates,
        triple_candidates=triple_candidates,
        base_feature_sets=base_feature_sets,
        feature_modes=feature_modes,
        enable_3way=enable_3way,
        sensitivity_refs=sensitivity_refs,
        unstable_explanation_config=unstable_explanation_config,
        model_subset=model_subset,
        cont_plus_bin_model_subset=cont_plus_bin_model_subset,
        branch_subset=branch_subset,
        explain_selection_df=explain_selection_df,
        nonlinear_cont_only_rulebook_policy=nonlinear_cont_only_rulebook_policy,
        tier2d_common_k_min=tier2d_common_k_min,
        tier2d_min_test_c2_n=tier2d_min_test_c2_n,
        tier2d_min_test_c3_n=tier2d_min_test_c3_n,
        tier2d_raw_diff_eps=tier2d_raw_diff_eps,
    )
    selection_df = pd.DataFrame(selection_records)
    metrics_summary.to_csv(part2_out_dir / "interaction_metrics_summary_ci.csv", index=False)
    metrics_c3_summary.to_csv(part2_out_dir / "interaction_metrics_c3_slice_ci.csv", index=False)
    audit_summary_root.to_csv(part2_out_dir / "interaction_audit_linkage_summary.csv", index=False)
    selection_df.to_csv(part2_out_dir / "model_selection_summary.csv", index=False)

    model_specs = build_model_specs()
    logistic_spec = next(spec for spec in model_specs if spec.name == "logistic_l2")
    year_summary = run_year_sensitivity(logistic_spec, df_trainable, y, splits, random_seed=random_seed, warning_log=root_warning_log)
    year_summary.to_csv(part2_out_dir / "interaction_sensitivity_year_summary.csv", index=False)
    missingness_summary = run_missingness_ablation(logistic_spec, df_trainable, y, splits, random_seed=random_seed, warning_log=root_warning_log)
    missingness_summary.to_csv(part2_out_dir / "interaction_ablation_missingflags_tier2plus.csv", index=False)

    need_threshold_grid = True
    if not selection_df.empty:
        top_l0 = selection_df[
            (selection_df["status"] == "COMPLETED")
            & (selection_df["branch"] == "mainline_plus_pairwise")
            & (selection_df["model"] == "logistic_l2")
            & (selection_df["feature_mode"] == ("cont_plus_bin" if "cont_plus_bin" in feature_modes else feature_modes[0]))
        ]
        if top_l0.empty:
            top_l0 = selection_df[(selection_df["status"] == "COMPLETED") & (selection_df["branch"] == "mainline_plus_pairwise") & (selection_df["model"] == "logistic_l2")]
        if not top_l0.empty:
            row = top_l0.iloc[0]
            need_threshold_grid = bool((row["prediction_rule_count"] == 0) or (row["delta_p20_ci_low"] < 0) or (row["delta_enrichment20_ci_low"] < 0))
    threshold_grid_summary = run_threshold_grid(
        logistic_spec,
        df_trainable,
        y,
        splits,
        random_seed=random_seed,
        pair_candidates=pair_candidates,
        triple_candidates=triple_candidates,
        base_X=base_X,
        enable_3way=enable_3way,
        warning_log=root_warning_log,
        tier2d_common_k_min=tier2d_common_k_min,
        tier2d_min_test_c2_n=tier2d_min_test_c2_n,
        tier2d_min_test_c3_n=tier2d_min_test_c3_n,
        tier2d_raw_diff_eps=tier2d_raw_diff_eps,
    ) if need_threshold_grid else pd.DataFrame([{"threshold_grid_id": "DISABLED", "sensitivity_only": True, "details": "not_triggered"}])
    threshold_grid_summary.to_csv(part2_out_dir / "threshold_grid_summary.csv", index=False)

    bridge_summary = build_rulebook_bridge_summary(selection_df, part2_out_dir)
    bridge_summary.to_csv(part2_out_dir / "rulebook_bridge_summary.csv", index=False)
    build_warning_summary(part2_out_dir, selection_df)

    if not selection_df.empty:
        model_zoo_md = model_zoo_markdown(selection_df, language="zh")
        model_zoo_en_md = model_zoo_markdown(selection_df, language="en")
    else:
        model_zoo_md = "# 模型横向对比\n\n- no completed model runs\n"
        model_zoo_en_md = "# Model Zoo Comparison\n\n- no completed model runs\n"
    (part2_out_dir / "model_zoo_comparison.md").write_text(model_zoo_md, encoding="utf-8")
    (part2_out_dir / "model_zoo_comparison.en.md").write_text(model_zoo_en_md, encoding="utf-8")

    root_warning_present = root_warning_log.exists() and root_warning_log.stat().st_size > 0
    run_decision_zh, run_decision_en, run_payload = run_level_conclusion(
        selection_df,
        need_threshold_grid,
        part2_out_dir=part2_out_dir,
        root_warning_present=root_warning_present,
    )
    (part2_out_dir / "run_decision.md").write_text(run_decision_zh, encoding="utf-8")
    (part2_out_dir / "run_conclusion_analysis_and_improvement.md").write_text(run_decision_zh + "\n" + model_zoo_md, encoding="utf-8")
    (part2_out_dir / "run_conclusion_analysis_and_improvement.en.md").write_text(run_decision_en + "\n" + model_zoo_en_md, encoding="utf-8")

    build_run_manifest_json(
        manifest,
        repo_root,
        part2_out_dir,
        splits_hash,
        split_min_test_c2_n=split_min_test_c2_n,
        split_min_test_c3_n=split_min_test_c3_n,
        tier2d_common_k_min=tier2d_common_k_min,
        tier2d_min_test_c2_n=tier2d_min_test_c2_n,
        tier2d_min_test_c3_n=tier2d_min_test_c3_n,
        tier2d_raw_diff_eps=tier2d_raw_diff_eps,
    )
    update_run_manifest_feature_registry(part2_out_dir, selection_df)
    completeness_report = stage3_reporting.build_artifact_completeness_report(part2_out_dir, selection_df)

    workspace_execution_md_rel = str(execution.get("workspace_execution_md", "process/stage3_execution_log.md"))
    workspace_execution_md = resolve_repo_path(repo_root, workspace_execution_md_rel)
    run_section = "\n".join(
        [
            f"## {run.get('run_id', 'unknown')}",
            "",
            f"- run_root: {paths.get('run_root', '')}",
            f"- selected_model: {run_payload['selected_model']}",
            f"- selected_feature_mode: {run_payload.get('selected_feature_mode', 'NA')}",
            f"- selected_branch: {run_payload['selected_branch']}",
            f"- primary_winner_model: {run_payload.get('primary_winner_model', 'NA')}",
            f"- control_winner_model: {run_payload.get('control_winner_model', 'NA')}",
            f"- mainline_upgrade: {str(run_payload['mainline_upgrade']).lower()}",
            f"- tier2d_level: {run_payload.get('tier2d_level', 'PASS')}",
            f"- applicability_domain: {run_payload['applicability_domain']}",
            f"- publish_scope: {run_payload.get('publish_scope', run_payload['applicability_domain'])}",
            f"- bridge_recommended: {str(bool(run_payload.get('bridge_recommended', False))).lower()}",
            f"- bridge_reason_codes: {'|'.join(run_payload.get('bridge_reason_codes', [])) if isinstance(run_payload.get('bridge_reason_codes', []), list) and run_payload.get('bridge_reason_codes', []) else 'none'}",
            f"- threshold_grid_used: {str(run_payload['threshold_grid_used']).lower()}",
            f"- completeness_pass: {str(completeness_report['overall_pass']).lower()}",
            f"- run_decision: {(part2_out_dir / 'run_decision.md').as_posix()}",
            f"- model_zoo_comparison: {(part2_out_dir / 'model_zoo_comparison.md').as_posix()}",
        ]
    )
    upsert_run_section(workspace_execution_md, str(run.get("run_id", "unknown")), run_section)

    append_log(part2_log, f"part2_end selected_model={run_payload['selected_model']} selected_branch={run_payload['selected_branch']}")
    assert_artifacts_complete(part2_out_dir)


if __name__ == "__main__":
    main()
