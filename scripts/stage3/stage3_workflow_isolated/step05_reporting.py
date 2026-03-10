#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import step03_models as stage3_models

CONTINUOUS_SOURCE_COLS = [
    "power_mw",
    "rack_kw_typical",
    "rack_kw_peak",
    "rack_density_area_w_per_sf_dc",
    "pue",
    "building_sqm",
    "whitespace_sqm",
]
LOG1P_CONTINUOUS_SOURCE_COLS = {
    "power_mw",
    "rack_kw_typical",
    "rack_kw_peak",
    "rack_density_area_w_per_sf_dc",
    "building_sqm",
    "whitespace_sqm",
}
PREFERRED_CONTINUOUS_RELEASE_COLS = [
    "power_mw",
    "rack_kw_typical",
    "pue",
    "building_sqm",
    "rack_kw_peak",
    "whitespace_sqm",
]
CONTINUOUS_RELEASE_CANDIDATE_COLS = [*PREFERRED_CONTINUOUS_RELEASE_COLS, "rack_density_area_w_per_sf_dc"]
DEBUG_ONLY_CONTINUOUS_DISCOVERY_COLS: list[str] = []
BIN_TO_CONTINUOUS_SOURCE = {
    "power_mw_bin": "power_mw",
    "rack_kw_typical_bin": "rack_kw_typical",
    "pue_bin": "pue",
    "building_sqm_bin": "building_sqm",
}
PUBLISH_SCOPE_TO_TIERS = {
    "C3_only": {"C3"},
    "C2C3": {"C2", "C3"},
    "C1C2C3": {"C1", "C2", "C3"},
}
CONTINUOUS_RELEASE_THRESHOLDS = {
    "power_mw": [20.0, 100.0],
    "rack_kw_typical": [35.0, 80.0],
    "pue": [1.15, 1.25],
    "building_sqm": [12000.0, 40000.0],
    "rack_kw_peak": [100.0, 300.0],
    "whitespace_sqm": [3000.0, 20000.0],
    "rack_density_area_w_per_sf_dc": [150.0, 300.0],
}
ENGINEERED_DIRECTION_PRIOR = {
    "power_mw": "positive",
    "rack_kw_typical": "positive",
    "rack_kw_peak": "positive",
    "rack_density_area_w_per_sf_dc": "positive",
    "pue": "negative",
    "building_sqm": "positive",
    "whitespace_sqm": "positive",
}
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
ROOT_REQUIRED_FILES = [
    "model_zoo_comparison.md",
    "model_zoo_comparison.en.md",
    "interaction_audit_linkage_summary.csv",
    "interaction_candidates_pairwise.csv",
    "pair_candidates_discovery.csv",
    "interaction_candidates_3way.csv",
    "interaction_metrics_c3_slice_ci.csv",
    "interaction_sensitivity_year_summary.csv",
    "interaction_ablation_missingflags_tier2plus.csv",
    "rulebook_bridge_summary.csv",
    "threshold_grid_summary.csv",
    "run_manifest.json",
    "artifact_completeness_report.json",
    "model_selection_summary.csv",
    "interaction_metrics_summary_ci.csv",
    "warning_summary.json",
    "run_decision.md",
    "run_conclusion_analysis_and_improvement.md",
    "run_conclusion_analysis_and_improvement.en.md",
    "run_warning.log",
    "splits/company_holdout_splits.csv",
    "splits/company_holdout_splits_meta.json",
]
BRANCH_REQUIRED_FILES = [
    "metrics_ci.csv",
    "metrics_c3_slice_ci.csv",
    "delta_ci.csv",
    "audit_summaries.csv",
    "rulebook_support.csv",
    "rulebook_support_engineered_comparison.csv",
    "rulebook_legacy_pair_tier2plus.csv",
    "pair_rulebook_publishable_c3only.csv",
    "pair_rulebook_explanation_unstable_c3only.csv",
    "rulebook_model_derived_sensitivity.csv",
    "model_derived_cutpoint_alignment.csv",
    "linear_continuous_effects.csv",
    "linear_pairwise_effects.csv",
    "linear_vs_engineered_direction_check.csv",
    "input_shift_metrics.csv",
    "score_shift_metrics.csv",
    "perf_shift_metrics.csv",
    "tier_shift_matched_control.csv",
    "run_decision.md",
    "run_conclusion_analysis_and_improvement.md",
    "run_conclusion_analysis_and_improvement.en.md",
    "run_warning.log",
]
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
MODEL_DERIVED_RULEBOOK_COLUMNS = [
    "rule_rank",
    "feature_a",
    "condition_text",
    "coverage",
    "enrichment",
    "support",
    "pos_hits",
    "cutpoint_source",
    "grid_type",
    "alignment_note",
    "model_term_score",
    "notes",
    "rule_id",
]
MODEL_DERIVED_CUTPOINT_COLUMNS = [
    "feature_a",
    "physical_grid",
    "data_driven_grid",
    "model_cutpoints",
    "model_cutpoints_std",
    "model_cutpoints_raw",
    "value_space",
    "cutpoint_source",
    "alignment_note",
    "scaling_note",
    "n_non_missing",
]
LINEAR_CONTINUOUS_EFFECT_COLUMNS = [
    "feature_name",
    "source_col",
    "coef_std",
    "odds_ratio_per_1sd",
    "mean_raw",
    "std_raw",
    "q10_raw",
    "q50_raw",
    "q90_raw",
    "partial_logit_q10",
    "partial_logit_q50",
    "partial_logit_q90",
    "delta_logit_q10_to_q50",
    "delta_logit_q50_to_q90",
    "delta_logit_q10_to_q90",
    "scaling_note",
]
LINEAR_PAIRWISE_EFFECT_COLUMNS = [
    "feature_name",
    "rule_id",
    "feature_a",
    "feature_b",
    "feature_c",
    "condition_text",
    "coef",
    "odds_ratio",
    "support_n",
    "support_pos",
    "selection_freq",
    "stability_freq",
    "evidence_status",
    "notes",
]
LINEAR_DIRECTION_CHECK_COLUMNS = [
    "feature_name",
    "source_col",
    "coef_std",
    "direction_linear",
    "direction_engineered_prior",
    "direction_consistency",
    "thresholds_engineered",
    "notes",
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
DEFAULT_UNSTABLE_EXPLANATION = {
    "enabled": True,
    "support_n_min": 10,
    "support_pos_min": 2,
    "enrichment_min": 1.2,
    "selection_freq_min": 0.50,
    "top_n": 10,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def fmt_release_number(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.6g}"


def parse_interval_label(label: str) -> tuple[float | None, float | None] | None:
    text = str(label).strip()
    if not text.startswith("[") or not text.endswith(")"):
        return None
    body = text[1:-1]
    if "," not in body:
        return None
    low_txt, high_txt = [part.strip() for part in body.split(",", 1)]
    low = None if low_txt in {"-inf", "-infinity"} else float(low_txt)
    high = None if high_txt in {"inf", "infinity"} else float(high_txt)
    return low, high


def published_feature_name(feature_name: str) -> str:
    return BIN_TO_CONTINUOUS_SOURCE.get(feature_name, feature_name)


def normalize_publish_scope(scope: str | None) -> str:
    text = str(scope or "C3_only").strip().upper()
    if text in {"C3_ONLY", "C3"}:
        return "C3_only"
    if text in {"C2C3", "C2_C3", "C2+C3"}:
        return "C2C3"
    if text in {"C1C2C3", "C1_C2_C3", "C1+C2+C3", "TIER1PLUS", "TIER-1+"}:
        return "C1C2C3"
    return "C3_only"


def normalize_candidate_train_scope(scope: str | None) -> str:
    text = str(scope or "C2C3").strip().upper()
    if text in {"C2C3", "C2_C3", "C2+C3", "TIER2PLUS", "TIER-2+"}:
        return "C2C3"
    if text in {"C1C2C3", "C1_C2_C3", "C1+C2+C3", "TIER1PLUS", "TIER-1+"}:
        return "C1C2C3"
    return "C2C3"


def render_single_release_condition(feature_name: str, value: str) -> str:
    published = published_feature_name(feature_name)
    if value == "__MISSING__":
        return f"{published} is missing"
    interval = parse_interval_label(str(value))
    if interval is not None and (
        feature_name in BIN_TO_CONTINUOUS_SOURCE
        or feature_name in CONTINUOUS_RELEASE_THRESHOLDS
        or feature_name in CONTINUOUS_SOURCE_COLS
    ):
        low, high = interval
        pieces: list[str] = []
        if low is not None:
            pieces.append(f"{published} >= {fmt_release_number(low)}")
        if high is not None:
            pieces.append(f"{published} < {fmt_release_number(high)}")
        if pieces:
            return " and ".join(pieces)
    if feature_name.endswith("_is_missing") and str(value) in {"1", "1.0", "True", "true"}:
        base = feature_name.removesuffix("_is_missing")
        return f"{published_feature_name(base)} is missing"
    if feature_name == "liquid_cool_binary" and str(value) in {"1", "1.0"}:
        return "liquid_cool == Y"
    if feature_name == "liquid_cool_binary" and str(value) in {"0", "0.0"}:
        return "liquid_cool == N"
    return f"{published} == {value}"


def render_release_rule(
    feature_a: str,
    value_a: str,
    feature_b: str = "",
    value_b: str = "",
    feature_c: str = "",
    value_c: str = "",
) -> tuple[str, str, str, str]:
    published_a = published_feature_name(feature_a) if feature_a else ""
    published_b = published_feature_name(feature_b) if feature_b else ""
    published_c = published_feature_name(feature_c) if feature_c else ""
    parts = []
    if feature_a:
        parts.append(render_single_release_condition(feature_a, value_a))
    if feature_b:
        parts.append(render_single_release_condition(feature_b, value_b))
    if feature_c:
        parts.append(render_single_release_condition(feature_c, value_c))
    return published_a, published_b, published_c, " and ".join([p for p in parts if p])


def rule_row_value(row: Any, attr: str, default: str = "") -> str:
    value = getattr(row, attr, default)
    if value is None:
        return default
    return str(value)


def fold_count_key(support_n: int, support_pos: int) -> str:
    return f"support_fold_count_n{support_n}_p{support_pos}"


def signal_group_of(feature_name: str) -> str:
    if not feature_name:
        return ""
    if feature_name in SIGNAL_GROUP_MAPPING:
        return SIGNAL_GROUP_MAPPING[feature_name]
    if feature_name.startswith("cont::"):
        return SIGNAL_GROUP_MAPPING.get(feature_name.split("::", 1)[1], "")
    if "=" in feature_name:
        return SIGNAL_GROUP_MAPPING.get(feature_name.split("=", 1)[0], "")
    return ""


def distinct_signal_groups(*feature_names: str) -> set[str]:
    return {group for group in (signal_group_of(name) for name in feature_names) if group}


def has_feature(value: Any) -> bool:
    text = str(value).strip()
    return text not in {"", "nan", "None"}


def format_threshold_list(values: Sequence[float]) -> str:
    clean = [float(v) for v in values if np.isfinite(v)]
    if not clean:
        return ""
    return ",".join(fmt_release_number(v) for v in clean)


def compute_data_driven_thresholds(source_df: pd.DataFrame, source_col: str) -> list[float]:
    if source_col not in source_df.columns:
        return []
    series = pd.to_numeric(source_df[source_col], errors="coerce")
    if "coverage_tier" in source_df.columns:
        series = series[source_df["coverage_tier"].isin(["C2", "C3"])]
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 20:
        return []
    quantiles = series.quantile([1 / 3, 2 / 3]).tolist()
    thresholds = sorted({float(v) for v in quantiles if np.isfinite(v)})
    return thresholds


def alignment_note_for_thresholds(physical: Sequence[float], comparison: Sequence[float]) -> str:
    a = [float(v) for v in physical if np.isfinite(v)]
    b = [float(v) for v in comparison if np.isfinite(v)]
    if not a and not b:
        return "both_empty"
    if not a or not b:
        return "missing_counterpart"
    if len(a) == len(b) and all(abs(x - y) <= max(1.0, 0.1 * max(abs(x), abs(y), 1.0)) for x, y in zip(a, b)):
        return "approximate"
    overlap = len({round(x, 3) for x in a} & {round(y, 3) for y in b})
    return "partial_overlap" if overlap else "conflict"


def build_scaling_lookup(scaling_stats: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if scaling_stats is None or scaling_stats.empty:
        return {}
    lookup: dict[str, dict[str, float]] = {}
    for row in scaling_stats.itertuples(index=False):
        source_col = str(getattr(row, "source_col", ""))
        mean = float(getattr(row, "mean", np.nan))
        std = float(getattr(row, "std", np.nan))
        if not source_col or (not np.isfinite(mean)) or (not np.isfinite(std)) or std <= 0:
            continue
        lookup[source_col] = {"mean": mean, "std": std}
    return lookup


def to_transformed_value(source_col: str, raw_value: float) -> float:
    value = float(raw_value)
    if source_col in LOG1P_CONTINUOUS_SOURCE_COLS:
        value = float(np.log1p(max(value, 0.0)))
    return value


def to_raw_value(source_col: str, transformed_value: float) -> float:
    value = float(transformed_value)
    if source_col in LOG1P_CONTINUOUS_SOURCE_COLS:
        value = float(np.expm1(value))
        value = max(value, 0.0)
    return value


def denormalize_cutpoints(source_col: str, cutpoints_std: Sequence[float], scaling_lookup: dict[str, dict[str, float]]) -> list[float]:
    stats = scaling_lookup.get(source_col)
    if stats is None:
        return []
    mean = float(stats["mean"])
    std = float(stats["std"])
    out: list[float] = []
    for cp in cutpoints_std:
        if not np.isfinite(cp):
            continue
        transformed = float(cp) * std + mean
        raw = to_raw_value(source_col, transformed)
        if np.isfinite(raw):
            out.append(float(raw))
    return sorted({float(v) for v in out if np.isfinite(v)})


def normalize_cutpoints(source_col: str, cutpoints_raw: Sequence[float], scaling_lookup: dict[str, dict[str, float]]) -> list[float]:
    stats = scaling_lookup.get(source_col)
    if stats is None:
        return []
    mean = float(stats["mean"])
    std = float(stats["std"])
    if (not np.isfinite(std)) or std <= 0:
        return []
    out: list[float] = []
    for cp in cutpoints_raw:
        if not np.isfinite(cp):
            continue
        transformed = to_transformed_value(source_col, float(cp))
        z = (transformed - mean) / std
        if np.isfinite(z):
            out.append(float(z))
    return sorted({float(v) for v in out if np.isfinite(v)})


def _extract_ebm_cutpoints(model: Any, feature_names: Sequence[str], source_col: str) -> list[float]:
    bins_attr = getattr(model, "bins_", None)
    if bins_attr is None:
        return []
    feature_name = f"cont::{source_col}"
    if feature_name not in feature_names:
        return []
    idx = list(feature_names).index(feature_name)
    if idx >= len(bins_attr):
        return []
    raw = bins_attr[idx]
    if isinstance(raw, list) and raw:
        raw = raw[0]
    try:
        arr = np.asarray(raw, dtype=float).reshape(-1)
    except Exception:
        return []
    return sorted({float(v) for v in arr if np.isfinite(v)})


def _extract_gbdt_cutpoints(model: Any, feature_names: Sequence[str], source_col: str) -> list[float]:
    predictors = getattr(model, "_predictors", None)
    feature_name = f"cont::{source_col}"
    if predictors is None or feature_name not in feature_names:
        return []
    feat_idx = list(feature_names).index(feature_name)
    cuts: list[float] = []
    try:
        for stage in predictors:
            for pred in stage:
                nodes = getattr(pred, "nodes", None)
                if nodes is None:
                    continue
                feature_idx = nodes["feature_idx"]
                thresholds = nodes["num_threshold"]
                for fidx, thr in zip(feature_idx, thresholds):
                    if int(fidx) == feat_idx and np.isfinite(thr):
                        cuts.append(float(thr))
    except Exception:
        return []
    if not cuts:
        return []
    rounded = [round(v, 6) for v in cuts]
    freq = Counter(rounded)
    top = [float(item[0]) for item in freq.most_common(3)]
    median_cut = float(np.median(cuts))
    merged = sorted({*top, median_cut})
    return merged[:3]


def extract_model_cutpoints(
    model_spec: Any,
    model: Any,
    feature_names: Sequence[str],
    source_df: pd.DataFrame,
    source_col: str,
    scaling_lookup: dict[str, dict[str, float]],
) -> tuple[list[float], list[float], str, list[float], str, str]:
    physical = CONTINUOUS_RELEASE_THRESHOLDS.get(source_col, [])
    data_driven = compute_data_driven_thresholds(source_df, source_col)
    if model_spec.kind == "ebm":
        cutpoints_std = _extract_ebm_cutpoints(model, feature_names, source_col)
        if cutpoints_std:
            cutpoints_raw = denormalize_cutpoints(source_col, cutpoints_std, scaling_lookup)
            if cutpoints_raw:
                return (
                    cutpoints_raw,
                    cutpoints_std,
                    "ebm_bins",
                    data_driven,
                    alignment_note_for_thresholds(physical, cutpoints_raw),
                    "cutpoint_space=raw; source_space=standardized; transform=std_to_raw_with_log1p_inverse_if_needed",
                )
    if model_spec.kind == "gbdt":
        cutpoints_std = _extract_gbdt_cutpoints(model, feature_names, source_col)
        if cutpoints_std:
            cutpoints_raw = denormalize_cutpoints(source_col, cutpoints_std, scaling_lookup)
            if cutpoints_raw:
                return (
                    cutpoints_raw,
                    cutpoints_std,
                    "gbdt_split_distribution",
                    data_driven,
                    alignment_note_for_thresholds(physical, cutpoints_raw),
                    "cutpoint_space=raw; source_space=standardized; transform=std_to_raw_with_log1p_inverse_if_needed",
                )
    if data_driven:
        return (
            data_driven,
            normalize_cutpoints(source_col, data_driven, scaling_lookup),
            "quantile_fallback",
            data_driven,
            alignment_note_for_thresholds(physical, data_driven),
            "cutpoint_space=raw; source_space=raw_data",
        )
    return (
        list(physical),
        normalize_cutpoints(source_col, physical, scaling_lookup),
        "physical_grid_fallback",
        data_driven,
        alignment_note_for_thresholds(physical, []),
        "cutpoint_space=raw; source_space=physical_grid",
    )


def build_rulebook_support_from_candidates(classified: pd.DataFrame, *, threshold: Any, branch_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if classified.empty:
        return pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS)
    filtered = classified[
        (~classified["primary_condition_contains_missing"].fillna(False))
        & (~classified.get("contains_missing_flag", pd.Series(False, index=classified.index)).fillna(False))
    ].copy()
    if filtered.empty:
        return pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS)
    ordered = filtered.sort_values(
        ["rule_type", "stability_freq", "enrichment", "support_pos", "support_n"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    for rank, row in enumerate(ordered.itertuples(index=False), start=1):
        candidate_scope = normalize_candidate_train_scope(getattr(row, "candidate_train_scope", "C2C3"))
        notes = []
        if isinstance(row.notes, str) and row.notes:
            notes.append(row.notes)
        if isinstance(row.downgrade_reason, str) and row.downgrade_reason:
            notes.append(f"downgrade_reason={row.downgrade_reason}")
        if row.rule_type == "triage":
            notes.append("diagnostic only")
        if row.type == "pair":
            notes.append("candidate_coverage_scope=C2C3_only" if candidate_scope == "C2C3" else "candidate_coverage_scope=all_trainable_tiers(C1,C2,C3)")
        elif row.type == "triple":
            notes.append("candidate_coverage_scope=C2C3_only")
        if branch_name == "mainline_plus_gated_3way" and row.type == "triple":
            notes.append("hierarchical=pair_L0")
        notes.append(f"threshold_id={threshold.threshold_id}")
        published_a, published_b, published_c, condition_text = render_release_rule(
            rule_row_value(row, "feature_a"),
            rule_row_value(row, "value_a"),
            rule_row_value(row, "feature_b"),
            rule_row_value(row, "value_b"),
            rule_row_value(row, "feature_c"),
            rule_row_value(row, "value_c"),
        )
        rows.append(
            {
                "rule_rank": rank,
                "rule_type": row.rule_type,
                "feature_a": published_a or row.feature_a,
                "feature_b": published_b or row.feature_b,
                "feature_c": published_c or row.feature_c,
                "condition_text": condition_text or row.primary_condition.replace("=", " == "),
                "coverage": float(row.coverage),
                "enrichment": float(row.enrichment) if np.isfinite(row.enrichment) else np.nan,
                "stability_freq": float(row.stability_freq),
                "rule_tier_min": row.rule_tier_min,
                "support": float(row.support_n),
                "pos_hits": float(row.support_pos),
                "notes": "; ".join(notes),
                "rule_id": row.candidate_id,
            }
        )
    return pd.DataFrame(rows, columns=SUPPORT_RULEBOOK_COLUMNS)


def build_legacy_rulebook_from_pairs(classified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if classified.empty:
        return pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS)
    pair_df = classified[
        (classified["type"] == "pair")
        & (~classified["primary_condition_contains_missing"].fillna(False))
        & (~classified.get("contains_missing_flag", pd.Series(False, index=classified.index)).fillna(False))
    ].copy()
    pair_df = pair_df.sort_values(["rule_type", "stability_freq", "enrichment", "support_pos"], ascending=[True, False, False, False]).reset_index(drop=True)
    for rank, row in enumerate(pair_df.itertuples(index=False), start=1):
        candidate_scope = normalize_candidate_train_scope(getattr(row, "candidate_train_scope", "C2C3"))
        notes = [
            f"support-folds={getattr(row, fold_count_key(20, 3), 0)}/{max(1, len(pair_df))}",
            f"tier_min={row.rule_tier_min}",
            f"use={row.rule_type}",
            ("candidate_coverage_scope=C2C3_only" if candidate_scope == "C2C3" else "candidate_coverage_scope=all_trainable_tiers(C1,C2,C3)"),
        ]
        if isinstance(row.downgrade_reason, str) and row.downgrade_reason:
            downgrade_reason = row.downgrade_reason
        else:
            downgrade_reason = ""
        rows.append(
            {
                "pair_rank": rank,
                "feature_a": row.feature_a,
                "feature_b": row.feature_b,
                "condition_text": row.primary_condition.replace("=", " == "),
                "coverage": float(row.coverage),
                "enrichment": float(row.enrichment) if np.isfinite(row.enrichment) else np.nan,
                "stability_freq": float(row.stability_freq),
                "support": float(row.support_n),
                "support_min": int(row.support_n_by_fold_min),
                "pos_hits": float(row.support_pos),
                "pos_hits_min": int(row.support_pos_by_fold_min),
                "support_fold_count": int(getattr(row, fold_count_key(20, 3), 0)),
                "rule_strength": (
                    "strong" if row.rule_type == "prediction" and row.stability_freq >= 0.80 else "medium" if row.stability_freq >= 0.60 else "weak"
                ),
                "rule_use": row.rule_type,
                "downgrade_reason": downgrade_reason,
                "candidate_train_scope": candidate_scope,
                "publish_scope": "discovery_only",
                "notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows, columns=LEGACY_RULEBOOK_COLUMNS)


def build_c3_only_pair_rulebook(
    df: pd.DataFrame,
    y: np.ndarray,
    classified_pairs: pd.DataFrame,
    *,
    publish_scope: str = "C3_only",
    default_candidate_train_scope: str = "C2C3",
) -> pd.DataFrame:
    if classified_pairs.empty:
        return pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS)
    resolved_publish_scope = normalize_publish_scope(publish_scope)
    publish_tiers = PUBLISH_SCOPE_TO_TIERS.get(resolved_publish_scope, {"C3"})
    publish_mask = df["coverage_tier"].isin(publish_tiers)
    if int(publish_mask.sum()) == 0:
        return pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS)
    publish_df = df.loc[publish_mask].copy()
    publish_y = y[publish_mask.to_numpy()]
    base_rate_publish = float(np.mean(publish_y)) if len(publish_y) else np.nan
    c2_mask = df["coverage_tier"] == "C2"
    c3_mask = df["coverage_tier"] == "C3"
    c2_df = df.loc[c2_mask].copy()
    c3_df = df.loc[c3_mask].copy()
    y_c2 = y[c2_mask.to_numpy()]
    y_c3 = y[c3_mask.to_numpy()]
    base_rate_c2 = float(np.mean(y_c2)) if len(y_c2) else np.nan
    base_rate_c3 = float(np.mean(y_c3)) if len(y_c3) else np.nan
    rows: list[dict[str, Any]] = []
    subset = classified_pairs[(classified_pairs["type"] == "pair") & (classified_pairs["rule_type"] == "prediction")].copy()
    rank = 0
    for row in subset.itertuples(index=False):
        cond = (publish_df[row.feature_a].astype(str) == str(row.value_a)) & (publish_df[row.feature_b].astype(str) == str(row.value_b))
        support_n = int(cond.sum())
        if support_n == 0:
            continue
        rank += 1
        support_pos = int(publish_y[cond.to_numpy()].sum()) if support_n else 0
        coverage_c3 = float(support_n / len(publish_df)) if len(publish_df) else np.nan
        support_rate = float(support_pos / support_n) if support_n else np.nan
        enrichment_c3 = (
            float(support_rate / base_rate_publish)
            if support_n and np.isfinite(base_rate_publish) and base_rate_publish > 0
            else np.nan
        )
        cond_c2 = (c2_df[row.feature_a].astype(str) == str(row.value_a)) & (c2_df[row.feature_b].astype(str) == str(row.value_b))
        cond_c3 = (c3_df[row.feature_a].astype(str) == str(row.value_a)) & (c3_df[row.feature_b].astype(str) == str(row.value_b))
        support_c2 = int(cond_c2.sum())
        support_c3 = int(cond_c3.sum())
        support_pos_c2 = int(y_c2[cond_c2.to_numpy()].sum()) if support_c2 else 0
        support_pos_c3 = int(y_c3[cond_c3.to_numpy()].sum()) if support_c3 else 0
        support_rate_c2 = float(support_pos_c2 / support_c2) if support_c2 else np.nan
        support_rate_c3 = float(support_pos_c3 / support_c3) if support_c3 else np.nan
        enrichment_c2 = (
            float(support_rate_c2 / base_rate_c2)
            if support_c2 and np.isfinite(base_rate_c2) and base_rate_c2 > 0
            else np.nan
        )
        enrichment_c3_tier = (
            float(support_rate_c3 / base_rate_c3)
            if support_c3 and np.isfinite(base_rate_c3) and base_rate_c3 > 0
            else np.nan
        )
        delta_enrichment = (
            float(enrichment_c3_tier - enrichment_c2)
            if np.isfinite(enrichment_c3_tier) and np.isfinite(enrichment_c2)
            else np.nan
        )
        candidate_train_scope = normalize_candidate_train_scope(getattr(row, "candidate_train_scope", default_candidate_train_scope))
        notes = [
            f"threshold_id={getattr(row, 'threshold_id', 'L0')}",
            f"publish_scope={resolved_publish_scope}",
            f"applicability_domain={resolved_publish_scope}",
        ]
        if isinstance(row.downgrade_reason, str) and row.downgrade_reason:
            notes.append(f"downgrade_reason={row.downgrade_reason}")
        published_a, published_b, _, condition_text = render_release_rule(
            str(row.feature_a),
            str(row.value_a),
            str(row.feature_b),
            str(row.value_b),
        )
        rows.append(
            {
                "rule_rank": rank,
                "feature_a": published_a or row.feature_a,
                "feature_b": published_b or row.feature_b,
                "condition_text": condition_text or row.primary_condition.replace("=", " == "),
                "coverage_c3": coverage_c3,
                "enrichment_c3": enrichment_c3,
                "support_c2": support_c2,
                "enrichment_c2": enrichment_c2,
                "support_c3": support_c3,
                "delta_enrichment_c3_minus_c2": delta_enrichment,
                "stability_freq": float(row.stability_freq),
                "candidate_train_scope": candidate_train_scope,
                "publish_scope": resolved_publish_scope,
                "signal_group_a": row.signal_group_a,
                "signal_group_b": row.signal_group_b,
                "notes": "; ".join(notes),
                "rule_id": row.candidate_id,
            }
        )
    return pd.DataFrame(rows, columns=C3_PAIR_RULEBOOK_COLUMNS)


def unstable_evidence_status(
    delta_p20_ci_low: float,
    delta_enrichment20_ci_low: float,
    tier2d_status: str,
    tier2d_detail: str,
) -> tuple[str, str]:
    ci_unstable = (
        np.isfinite(delta_p20_ci_low)
        and np.isfinite(delta_enrichment20_ci_low)
        and (delta_p20_ci_low < 0 or delta_enrichment20_ci_low < 0)
    )
    tier_shift = str(tier2d_status).upper() != "PASS"
    if not ci_unstable and not tier_shift:
        return "", ""
    if ci_unstable and tier_shift:
        status = "both"
    elif ci_unstable:
        status = "unstable_ci"
    else:
        status = "tier_shift"
    reasons: list[str] = []
    if ci_unstable:
        reasons.append(
            f"delta_ci_crosses_zero(dp20_low={delta_p20_ci_low:.6f},de20_low={delta_enrichment20_ci_low:.6f})"
        )
    if tier_shift:
        detail = str(tier2d_detail or "").replace("\n", " ").strip()
        reasons.append(f"tier2d_fail({detail or 'tier_shift_detected'})")
    return status, "; ".join(reasons)


def build_unstable_explanation_pair_rulebook(
    df: pd.DataFrame,
    y: np.ndarray,
    classified_pairs: pd.DataFrame,
    *,
    delta_p20_ci_low: float,
    delta_enrichment20_ci_low: float,
    tier2d_status: str,
    tier2d_detail: str,
    publish_scope: str = "C3_only",
    default_candidate_train_scope: str = "C2C3",
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = dict(DEFAULT_UNSTABLE_EXPLANATION)
    if config:
        cfg.update(config)
    if not bool(cfg.get("enabled", True)):
        return pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS)
    evidence_status, why_unstable = unstable_evidence_status(
        delta_p20_ci_low,
        delta_enrichment20_ci_low,
        tier2d_status,
        tier2d_detail,
    )
    if not evidence_status or classified_pairs.empty:
        return pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS)
    resolved_publish_scope = normalize_publish_scope(publish_scope)
    publish_tiers = PUBLISH_SCOPE_TO_TIERS.get(resolved_publish_scope, {"C3"})
    publish_mask = df["coverage_tier"].isin(publish_tiers)
    if int(publish_mask.sum()) == 0:
        return pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS)
    publish_df = df.loc[publish_mask].copy()
    publish_y = y[publish_mask.to_numpy()]
    base_rate_publish = float(np.mean(publish_y)) if len(publish_y) else np.nan
    subset = classified_pairs[classified_pairs["type"] == "pair"].copy()
    if subset.empty:
        return pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS)
    subset = subset[
        (~subset["primary_condition_contains_missing"].fillna(False))
        & (~subset.get("contains_missing_flag", pd.Series(False, index=subset.index)).fillna(False))
        & (~subset["same_signal_group"].fillna(False))
        & (subset["known_signal_group_count"].fillna(0).astype(int) >= 2)
        & (subset["support_n_by_fold_min"].fillna(0).astype(int) >= int(cfg["support_n_min"]))
        & (subset["support_pos_by_fold_min"].fillna(0).astype(int) >= int(cfg["support_pos_min"]))
        & (subset["stability_freq"].fillna(0.0).astype(float) >= float(cfg["selection_freq_min"]))
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS)
    rows: list[dict[str, Any]] = []
    for row in subset.itertuples(index=False):
        cond = (publish_df[row.feature_a].astype(str) == str(row.value_a)) & (publish_df[row.feature_b].astype(str) == str(row.value_b))
        support_n_c3 = int(cond.sum())
        if support_n_c3 == 0:
            continue
        support_pos_c3 = int(publish_y[cond.to_numpy()].sum()) if support_n_c3 else 0
        coverage_c3 = float(support_n_c3 / len(publish_df)) if len(publish_df) else np.nan
        support_rate_c3 = float(support_pos_c3 / support_n_c3) if support_n_c3 else np.nan
        enrichment_c3 = (
            float(support_rate_c3 / base_rate_publish)
            if support_n_c3 and np.isfinite(base_rate_publish) and base_rate_publish > 0
            else np.nan
        )
        effective_enrichment = enrichment_c3 if np.isfinite(enrichment_c3) else float(row.enrichment)
        if not np.isfinite(effective_enrichment) or effective_enrichment < float(cfg["enrichment_min"]):
            continue
        candidate_train_scope = normalize_candidate_train_scope(getattr(row, "candidate_train_scope", default_candidate_train_scope))
        published_a, published_b, _, condition_text = render_release_rule(
            str(row.feature_a),
            str(row.value_a),
            str(row.feature_b),
            str(row.value_b),
        )
        notes = [
            f"publish_scope={resolved_publish_scope}",
            f"applicability_domain={resolved_publish_scope}",
            "explanation_only=true",
            f"threshold_id={getattr(row, 'threshold_id', 'L0')}",
            f"source_rule_type={row.rule_type}",
        ]
        if isinstance(row.downgrade_reason, str) and row.downgrade_reason:
            notes.append(f"candidate_downgrade_reason={row.downgrade_reason}")
        rows.append(
            {
                "feature_a": published_a or row.feature_a,
                "feature_b": published_b or row.feature_b,
                "condition_text": condition_text or row.primary_condition.replace("=", " == "),
                "coverage_c3": coverage_c3,
                "enrichment_c3": enrichment_c3,
                "stability_freq": float(row.stability_freq),
                "support_n_by_fold_min": int(row.support_n_by_fold_min),
                "support_pos_by_fold_min": int(row.support_pos_by_fold_min),
                "candidate_train_scope": candidate_train_scope,
                "publish_scope": resolved_publish_scope,
                "signal_group_a": row.signal_group_a,
                "signal_group_b": row.signal_group_b,
                "evidence_status": evidence_status,
                "explanation_only": True,
                "why_unstable": why_unstable,
                "notes": "; ".join(notes),
                "rule_id": row.candidate_id,
            }
        )
    if not rows:
        return pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["enrichment_c3", "stability_freq", "support_n_by_fold_min"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["rule_id"], keep="first").head(int(cfg["top_n"])).reset_index(drop=True)
    out.insert(0, "rule_rank", np.arange(1, len(out) + 1))
    return out[UNSTABLE_EXPLANATION_PAIR_COLUMNS]


def build_continuous_main_effect_rules(
    source_df: pd.DataFrame,
    y: np.ndarray,
    importance: pd.Series,
    source_cols: Sequence[str] | None = None,
    *,
    debug_only: bool = False,
) -> list[dict[str, Any]]:
    base_rate = float(np.mean(y)) if len(y) else np.nan
    rows: list[dict[str, Any]] = []
    for source_col in list(source_cols or PREFERRED_CONTINUOUS_RELEASE_COLS):
        feature_name = f"cont::{source_col}"
        if source_col not in source_df.columns:
            continue
        score = float(importance.get(feature_name, 0.0))
        raw_series = pd.to_numeric(source_df[source_col], errors="coerce")
        if raw_series.notna().sum() == 0:
            continue
        thresholds = CONTINUOUS_RELEASE_THRESHOLDS.get(source_col, [])
        bounds = [None, *thresholds, None]
        interval_rows: list[dict[str, Any]] = []
        for low, high in zip(bounds[:-1], bounds[1:]):
            cond = raw_series.notna()
            if low is not None:
                cond &= raw_series >= float(low)
            if high is not None:
                cond &= raw_series < float(high)
            support_n = int(cond.sum())
            if support_n < 10:
                continue
            support_pos = int(y[cond.to_numpy()].sum()) if support_n else 0
            if support_pos < 2:
                continue
            support_rate = float(support_pos / support_n) if support_n else np.nan
            enrichment = float(support_rate / base_rate) if support_n and np.isfinite(base_rate) and base_rate > 0 else np.nan
            if not np.isfinite(enrichment):
                continue
            _, _, _, condition_text = render_release_rule(
                source_col,
                f"[{fmt_release_number(low) if low is not None else '-inf'},{fmt_release_number(high) if high is not None else 'inf'})",
            )
            interval_rows.append(
                {
                    "rule_type": "triage" if debug_only else "prediction",
                    "feature_a": source_col,
                    "feature_b": "",
                    "feature_c": "",
                    "condition_text": condition_text,
                    "coverage": float(support_n / len(source_df)) if len(source_df) else np.nan,
                    "enrichment": enrichment,
                    "stability_freq": 1.0,
                    "rule_tier_min": "Tier-2+",
                    "support": float(support_n),
                    "pos_hits": float(support_pos),
                    "notes": (
                        "source=continuous_debug_discovery; threshold_source=release_grid; debug_only=true; publishable=false"
                        if debug_only
                        else "source=continuous_main_effect; threshold_source=release_grid"
                    ),
                    "rule_id": f"cont_rule::{source_col}::{condition_text}",
                    "rule_score": abs(score),
                    "release_priority": 1 if source_col in PREFERRED_CONTINUOUS_RELEASE_COLS else 0,
                }
            )
        interval_rows = sorted(interval_rows, key=lambda item: (item["enrichment"], item["support"]), reverse=True)[:3]
        rows.extend(interval_rows)
    return rows


def build_mainline_rulebook(
    model_spec: Any,
    source_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_meta: dict[str, dict[str, Any]],
    y: np.ndarray,
    model: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if model is None:
        empty_support = pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS)
        empty_legacy = pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS)
        empty_debug = pd.DataFrame()
        return empty_support, empty_legacy, empty_debug

    if hasattr(model, "coef_"):
        weights = np.asarray(model.coef_).reshape(-1)
        importance = pd.Series(weights, index=feature_frame.columns)
    elif hasattr(model, "feature_importances_"):
        importance = pd.Series(np.asarray(model.feature_importances_).reshape(-1), index=feature_frame.columns)
    elif hasattr(model, "term_names_") and hasattr(model, "term_importances"):
        values = np.asarray(model.term_importances())
        importance = pd.Series(values.reshape(-1), index=list(model.term_names_))
    else:
        importance = pd.Series(dtype=float)

    base_rate = float(np.mean(y)) if len(y) else np.nan
    candidate_rows: list[dict[str, Any]] = []
    skip_bin_features = {
        feature_name
        for feature_name, source_col in BIN_TO_CONTINUOUS_SOURCE.items()
        if model_spec.nonlinear and source_col in PREFERRED_CONTINUOUS_RELEASE_COLS
    }
    for feature_name in feature_frame.columns:
        meta = feature_meta.get(feature_name, {})
        if meta.get("kind") not in {"rule_indicator", "missing_indicator", "year_sensitivity", "year_sensitivity_continuous"}:
            continue
        if meta.get("contains_missing") or meta.get("is_missing_flag") or str(meta.get("rule_tier_min", "")).lower() == "sensitivity_only":
            continue
        if meta.get("feature_a") in skip_bin_features:
            continue
        indicator = feature_frame[feature_name].to_numpy(dtype=float)
        support_n = int(np.sum(indicator > 0.5))
        if support_n == 0:
            continue
        support_pos = int(y[indicator > 0.5].sum())
        support_rate = support_pos / support_n if support_n else np.nan
        enrichment = support_rate / base_rate if support_n and np.isfinite(base_rate) and base_rate > 0 else np.nan
        notes = []
        rule_type = "prediction"
        value_a = str(meta.get("value_a", ""))
        if value_a:
            published_a, _, _, condition_text = render_release_rule(str(meta.get("feature_a", feature_name)), value_a)
        else:
            published_a, condition_text = str(meta.get("feature_a", feature_name)), str(meta.get("condition_text", feature_name))
        candidate_rows.append(
            {
                "feature_name": feature_name,
                "rule_score": float(importance.get(feature_name, 0.0)),
                "rule_type": rule_type,
                "feature_a": published_a or meta.get("feature_a", feature_name),
                "feature_b": meta.get("feature_b", ""),
                "feature_c": meta.get("feature_c", ""),
                "condition_text": condition_text or meta.get("condition_text", feature_name),
                "coverage": support_n / len(feature_frame),
                "enrichment": enrichment,
                "stability_freq": 1.0,
                "rule_tier_min": meta.get("rule_tier_min", "Tier-2+"),
                "support": float(support_n),
                "pos_hits": float(support_pos),
                "notes": "; ".join(notes),
                "rule_id": feature_name,
            }
        )
    debug_extra_rows: list[dict[str, Any]] = []
    if model_spec.nonlinear:
        candidate_rows.extend(
            build_continuous_main_effect_rules(
                source_df,
                y,
                importance,
                source_cols=CONTINUOUS_RELEASE_CANDIDATE_COLS,
            )
        )
        debug_extra_rows = build_continuous_main_effect_rules(
            source_df,
            y,
            importance,
            source_cols=DEBUG_ONLY_CONTINUOUS_DISCOVERY_COLS,
            debug_only=True,
        )
    if not candidate_rows:
        return pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS), pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS), pd.DataFrame()
    support = pd.DataFrame(candidate_rows)
    support["continuous_release_priority"] = support.apply(
        lambda row: int(
            str(row.get("rule_id", "")).startswith("cont_rule::")
            and int(row.get("release_priority", 0)) > 0
        ),
        axis=1,
    )
    support = support.sort_values(
        ["rule_type", "continuous_release_priority", "rule_score", "enrichment", "coverage"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    support.insert(0, "rule_rank", np.arange(1, len(support) + 1))
    support = support[SUPPORT_RULEBOOK_COLUMNS + ["rule_score", "continuous_release_priority"]]
    debug = support.copy()
    if debug_extra_rows:
        debug_extra = pd.DataFrame(debug_extra_rows)
        debug_extra["continuous_release_priority"] = 0
        debug_extra.insert(0, "rule_rank", np.arange(len(debug) + 1, len(debug) + len(debug_extra) + 1))
        debug_extra = debug_extra[SUPPORT_RULEBOOK_COLUMNS + ["rule_score", "continuous_release_priority"]]
        debug = pd.concat([debug, debug_extra], ignore_index=True)
        debug = debug.sort_values(
            ["rule_type", "continuous_release_priority", "rule_score", "enrichment", "coverage"],
            ascending=[True, False, False, False, False],
        ).reset_index(drop=True)
        debug["rule_rank"] = np.arange(1, len(debug) + 1)
    support = support[SUPPORT_RULEBOOK_COLUMNS]
    legacy = pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS)
    return support, legacy, debug


def build_model_derived_sensitivity_rulebook(
    model_spec: Any,
    model: Any,
    source_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    y: np.ndarray,
    scaling_stats: pd.DataFrame | None = None,
    *,
    min_support_n: int = 10,
    min_support_pos: int = 2,
    min_enrichment: float = 1.0,
    max_rules_per_feature: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model is None or feature_frame.empty:
        return pd.DataFrame(columns=MODEL_DERIVED_RULEBOOK_COLUMNS), pd.DataFrame(columns=MODEL_DERIVED_CUTPOINT_COLUMNS)
    if getattr(model_spec, "kind", "") not in {"ebm", "gbdt"}:
        return pd.DataFrame(columns=MODEL_DERIVED_RULEBOOK_COLUMNS), pd.DataFrame(columns=MODEL_DERIVED_CUTPOINT_COLUMNS)
    if not any(str(col).startswith("cont::") for col in feature_frame.columns):
        return pd.DataFrame(columns=MODEL_DERIVED_RULEBOOK_COLUMNS), pd.DataFrame(columns=MODEL_DERIVED_CUTPOINT_COLUMNS)
    base_rate = float(np.mean(y)) if len(y) else np.nan
    if not np.isfinite(base_rate) or base_rate <= 0:
        return pd.DataFrame(columns=MODEL_DERIVED_RULEBOOK_COLUMNS), pd.DataFrame(columns=MODEL_DERIVED_CUTPOINT_COLUMNS)

    scaling_lookup = build_scaling_lookup(scaling_stats)
    importances = model_feature_importance_series(model, list(feature_frame.columns))
    rule_rows: list[dict[str, Any]] = []
    cut_rows: list[dict[str, Any]] = []

    for source_col in CONTINUOUS_RELEASE_CANDIDATE_COLS:
        if source_col not in source_df.columns:
            continue
        raw_series = pd.to_numeric(source_df[source_col], errors="coerce")
        non_missing = raw_series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(non_missing) < 10:
            continue
        model_cutpoints_raw, model_cutpoints_std, cutpoint_source, data_driven_grid, alignment_note, scaling_note = extract_model_cutpoints(
            model_spec,
            model,
            list(feature_frame.columns),
            source_df,
            source_col,
            scaling_lookup,
        )
        cut_rows.append(
            {
                "feature_a": source_col,
                "physical_grid": format_threshold_list(CONTINUOUS_RELEASE_THRESHOLDS.get(source_col, [])),
                "data_driven_grid": format_threshold_list(data_driven_grid),
                "model_cutpoints": format_threshold_list(model_cutpoints_raw),
                "model_cutpoints_std": format_threshold_list(model_cutpoints_std),
                "model_cutpoints_raw": format_threshold_list(model_cutpoints_raw),
                "value_space": "raw",
                "cutpoint_source": cutpoint_source,
                "alignment_note": alignment_note,
                "scaling_note": scaling_note,
                "n_non_missing": int(len(non_missing)),
            }
        )
        if not model_cutpoints_raw:
            continue
        bounds = [None, *model_cutpoints_raw, None]
        feature_name = f"cont::{source_col}"
        model_term_score = float(abs(importances.get(feature_name, 0.0))) if not importances.empty else 0.0
        interval_rows: list[dict[str, Any]] = []
        for low, high in zip(bounds[:-1], bounds[1:]):
            cond = raw_series.notna()
            if low is not None:
                cond &= raw_series >= float(low)
            if high is not None:
                cond &= raw_series < float(high)
            support_n = int(cond.sum())
            if support_n < int(max(1, min_support_n)):
                continue
            support_pos = int(y[cond.to_numpy()].sum()) if support_n else 0
            if support_pos < int(max(1, min_support_pos)):
                continue
            support_rate = float(support_pos / support_n) if support_n else np.nan
            enrichment = float(support_rate / base_rate) if support_n else np.nan
            if (not np.isfinite(enrichment)) or (enrichment < float(min_enrichment)):
                continue
            _, _, _, condition_text = render_release_rule(
                source_col,
                f"[{fmt_release_number(low) if low is not None else '-inf'},{fmt_release_number(high) if high is not None else 'inf'})",
            )
            interval_rows.append(
                {
                    "feature_a": source_col,
                    "condition_text": condition_text,
                    "coverage": float(support_n / len(source_df)) if len(source_df) else np.nan,
                    "enrichment": enrichment,
                    "support": float(support_n),
                    "pos_hits": float(support_pos),
                    "cutpoint_source": cutpoint_source,
                    "grid_type": "model_derived_sensitivity",
                    "alignment_note": alignment_note,
                    "model_term_score": model_term_score,
                    "notes": (
                        f"source=model_derived_cutpoints; cutpoint_source={cutpoint_source}; "
                        f"cutpoint_space=raw; scaling_note={scaling_note}; "
                        f"physical_grid={format_threshold_list(CONTINUOUS_RELEASE_THRESHOLDS.get(source_col, []))}; "
                        f"data_driven_grid={format_threshold_list(data_driven_grid)}"
                    ),
                    "rule_id": f"model_derived::{source_col}::{condition_text}",
                }
            )
        top_k = int(max(1, max_rules_per_feature))
        rule_rows.extend(sorted(interval_rows, key=lambda item: (item["model_term_score"], item["enrichment"], item["support"]), reverse=True)[:top_k])

    rulebook = pd.DataFrame(rule_rows, columns=MODEL_DERIVED_RULEBOOK_COLUMNS)
    if not rulebook.empty:
        rulebook = rulebook.sort_values(
            ["model_term_score", "enrichment", "support"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        if "rule_rank" in rulebook.columns:
            rulebook["rule_rank"] = np.arange(1, len(rulebook) + 1)
        else:
            rulebook.insert(0, "rule_rank", np.arange(1, len(rulebook) + 1))
        rulebook = rulebook[MODEL_DERIVED_RULEBOOK_COLUMNS]
    else:
        rulebook = pd.DataFrame(columns=MODEL_DERIVED_RULEBOOK_COLUMNS)

    cutpoints = pd.DataFrame(cut_rows, columns=MODEL_DERIVED_CUTPOINT_COLUMNS)
    return rulebook, cutpoints


def promote_model_derived_as_primary_support(rulebook_model_derived: pd.DataFrame) -> pd.DataFrame:
    if rulebook_model_derived is None or rulebook_model_derived.empty:
        return pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS)
    rows: list[dict[str, Any]] = []
    ordered = rulebook_model_derived.copy()
    if "rule_rank" in ordered.columns:
        ordered = ordered.sort_values("rule_rank", ascending=True)
    for rec in ordered.itertuples(index=False):
        notes = str(getattr(rec, "notes", "") or "")
        extra = "source=promoted_model_derived_primary; engineered_comparison=rulebook_support_engineered_comparison.csv"
        notes = f"{notes}; {extra}" if notes else extra
        rows.append(
            {
                "rule_rank": 0,
                "rule_type": "prediction",
                "feature_a": str(getattr(rec, "feature_a", "")),
                "feature_b": "",
                "feature_c": "",
                "condition_text": str(getattr(rec, "condition_text", "")),
                "coverage": float(getattr(rec, "coverage", np.nan)),
                "enrichment": float(getattr(rec, "enrichment", np.nan)),
                "stability_freq": 1.0,
                "rule_tier_min": "Tier-2+",
                "support": float(getattr(rec, "support", np.nan)),
                "pos_hits": float(getattr(rec, "pos_hits", np.nan)),
                "notes": notes,
                "rule_id": str(getattr(rec, "rule_id", "")),
            }
        )
    out = pd.DataFrame(rows, columns=SUPPORT_RULEBOOK_COLUMNS)
    if not out.empty:
        out["rule_rank"] = np.arange(1, len(out) + 1)
    return out


def model_feature_importance_series(model: Any, feature_names: Sequence[str]) -> pd.Series:
    if model is None:
        return pd.Series(dtype=float)
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_).reshape(-1)
        return pd.Series(values, index=list(feature_names))
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).reshape(-1)
        return pd.Series(values, index=list(feature_names))
    if hasattr(model, "term_names_") and hasattr(model, "term_importances"):
        values = np.asarray(model.term_importances())
        names = list(model.term_names_)
        return pd.Series(values.reshape(-1), index=names)
    return pd.Series(dtype=float)


def build_linear_continuous_effects(
    model_spec: Any,
    model: Any,
    feature_names: Sequence[str],
    scaling_stats: pd.DataFrame,
) -> pd.DataFrame:
    if getattr(model_spec, "kind", "") not in {"logistic_l2", "elasticnet"}:
        return pd.DataFrame(columns=LINEAR_CONTINUOUS_EFFECT_COLUMNS)
    if model is None or not hasattr(model, "coef_") or scaling_stats.empty:
        return pd.DataFrame(columns=LINEAR_CONTINUOUS_EFFECT_COLUMNS)

    coef_series = model_feature_importance_series(model, feature_names)
    rows: list[dict[str, Any]] = []
    for stat in scaling_stats.itertuples(index=False):
        feature_name = str(stat.feature_name)
        coef_std = float(coef_series.get(feature_name, np.nan))
        if not np.isfinite(coef_std):
            continue
        std_raw = float(getattr(stat, "std", np.nan))
        mean_raw = float(getattr(stat, "mean", np.nan))
        q10_raw = float(getattr(stat, "q10_raw", np.nan))
        q50_raw = float(getattr(stat, "q50_raw", np.nan))
        q90_raw = float(getattr(stat, "q90_raw", np.nan))
        if not np.isfinite(std_raw) or std_raw <= 0:
            std_raw = 1.0

        def _z(raw_value: float) -> float:
            if not np.isfinite(raw_value):
                return np.nan
            return float(np.clip((raw_value - mean_raw) / std_raw, -6.0, 6.0))

        z10 = _z(q10_raw)
        z50 = _z(q50_raw)
        z90 = _z(q90_raw)
        p10 = float(coef_std * z10) if np.isfinite(z10) else np.nan
        p50 = float(coef_std * z50) if np.isfinite(z50) else np.nan
        p90 = float(coef_std * z90) if np.isfinite(z90) else np.nan
        rows.append(
            {
                "feature_name": feature_name,
                "source_col": str(getattr(stat, "source_col", feature_name.split("::", 1)[-1])),
                "coef_std": coef_std,
                "odds_ratio_per_1sd": float(np.exp(np.clip(coef_std, -20.0, 20.0))),
                "mean_raw": mean_raw,
                "std_raw": std_raw,
                "q10_raw": q10_raw,
                "q50_raw": q50_raw,
                "q90_raw": q90_raw,
                "partial_logit_q10": p10,
                "partial_logit_q50": p50,
                "partial_logit_q90": p90,
                "delta_logit_q10_to_q50": float(p50 - p10) if np.isfinite(p10) and np.isfinite(p50) else np.nan,
                "delta_logit_q50_to_q90": float(p90 - p50) if np.isfinite(p50) and np.isfinite(p90) else np.nan,
                "delta_logit_q10_to_q90": float(p90 - p10) if np.isfinite(p10) and np.isfinite(p90) else np.nan,
                "scaling_note": "coef_std uses standardized cont:: features; odds_ratio_per_1sd=exp(coef_std); partial_logit_* is single-feature contribution on standardized scale.",
            }
        )
    effects = pd.DataFrame(rows, columns=LINEAR_CONTINUOUS_EFFECT_COLUMNS)
    if not effects.empty:
        effects = effects.sort_values(["delta_logit_q10_to_q90", "coef_std"], ascending=[False, False], key=lambda s: np.abs(pd.to_numeric(s, errors="coerce"))).reset_index(drop=True)
    return effects


def build_linear_pairwise_effects(
    model_spec: Any,
    model: Any,
    feature_names: Sequence[str],
    rulebook_debug: pd.DataFrame | None,
) -> pd.DataFrame:
    if getattr(model_spec, "kind", "") not in {"logistic_l2", "elasticnet"}:
        return pd.DataFrame(columns=LINEAR_PAIRWISE_EFFECT_COLUMNS)
    if model is None or not hasattr(model, "coef_"):
        return pd.DataFrame(columns=LINEAR_PAIRWISE_EFFECT_COLUMNS)
    if rulebook_debug is None or rulebook_debug.empty or "candidate_id" not in rulebook_debug.columns:
        return pd.DataFrame(columns=LINEAR_PAIRWISE_EFFECT_COLUMNS)

    coef_series = model_feature_importance_series(model, feature_names)
    candidate_lookup = {
        str(getattr(row, "candidate_id", "")): row
        for row in rulebook_debug.itertuples(index=False)
        if str(getattr(row, "candidate_id", ""))
    }
    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        feature = str(feature_name)
        if feature.startswith("cont::") or feature.endswith("_is_missing") or feature == "base_non_missing_count":
            continue
        if feature not in candidate_lookup:
            continue
        coef = float(coef_series.get(feature, np.nan))
        if not np.isfinite(coef):
            continue
        row = candidate_lookup[feature]
        feature_a = str(getattr(row, "feature_a", ""))
        feature_b = str(getattr(row, "feature_b", ""))
        feature_c = str(getattr(row, "feature_c", ""))
        condition_text = str(getattr(row, "primary_condition", "") or "").replace("=", " == ")
        if not condition_text:
            condition_text = " and ".join([item for item in [feature_a, feature_b, feature_c] if item])
        notes = [
            "linear_pairwise_explanation_only=true",
            "not_for_publishable_rulebook=true",
        ]
        downgrade_reason = str(getattr(row, "downgrade_reason", "") or "")
        if downgrade_reason:
            notes.append(f"downgrade_reason={downgrade_reason}")
        rows.append(
            {
                "feature_name": feature,
                "rule_id": feature,
                "feature_a": feature_a,
                "feature_b": feature_b,
                "feature_c": feature_c,
                "condition_text": condition_text,
                "coef": coef,
                "odds_ratio": float(np.exp(np.clip(coef, -20.0, 20.0))),
                "support_n": float(getattr(row, "support_n", np.nan)),
                "support_pos": float(getattr(row, "support_pos", np.nan)),
                "selection_freq": float(getattr(row, "selection_freq", np.nan)),
                "stability_freq": float(getattr(row, "stability_freq", np.nan)),
                "evidence_status": "explanation_only",
                "notes": "; ".join(notes),
            }
        )
    out = pd.DataFrame(rows, columns=LINEAR_PAIRWISE_EFFECT_COLUMNS)
    if not out.empty:
        out = out.sort_values(
            ["coef", "stability_freq", "selection_freq", "support_n"],
            ascending=[False, False, False, False],
            key=lambda s: np.abs(pd.to_numeric(s, errors="coerce")),
        ).reset_index(drop=True)
    return out


def build_linear_vs_engineered_direction_check(
    model_spec: Any,
    linear_continuous_effects: pd.DataFrame,
) -> pd.DataFrame:
    if getattr(model_spec, "kind", "") not in {"logistic_l2", "elasticnet"}:
        return pd.DataFrame(columns=LINEAR_DIRECTION_CHECK_COLUMNS)
    if linear_continuous_effects.empty:
        return pd.DataFrame(columns=LINEAR_DIRECTION_CHECK_COLUMNS)

    rows: list[dict[str, Any]] = []
    for effect in linear_continuous_effects.itertuples(index=False):
        source_col = str(getattr(effect, "source_col", ""))
        coef_std = float(getattr(effect, "coef_std", np.nan))
        if not source_col or not np.isfinite(coef_std):
            continue
        if abs(coef_std) < 1e-6:
            direction_linear = "neutral"
        else:
            direction_linear = "positive" if coef_std > 0 else "negative"
        expected = str(ENGINEERED_DIRECTION_PRIOR.get(source_col, "unknown"))
        if expected == "unknown" or direction_linear == "neutral":
            consistency = "indeterminate"
        elif expected == direction_linear:
            consistency = "consistent"
        else:
            consistency = "conflict"
        rows.append(
            {
                "feature_name": str(getattr(effect, "feature_name", f"cont::{source_col}")),
                "source_col": source_col,
                "coef_std": coef_std,
                "direction_linear": direction_linear,
                "direction_engineered_prior": expected,
                "direction_consistency": consistency,
                "thresholds_engineered": format_threshold_list(CONTINUOUS_RELEASE_THRESHOLDS.get(source_col, [])),
                "notes": "diagnostic_only=true; not_used_for_winner_or_publish_scope=true",
            }
        )
    out = pd.DataFrame(rows, columns=LINEAR_DIRECTION_CHECK_COLUMNS)
    if not out.empty:
        out = out.sort_values(["direction_consistency", "coef_std"], ascending=[True, False], key=lambda s: np.abs(pd.to_numeric(s, errors="coerce")) if s.name == "coef_std" else s).reset_index(drop=True)
    return out


def is_interaction_branch(branch_name: str) -> bool:
    return branch_name in {"mainline_plus_pairwise", "mainline_plus_gated_3way"}


def annotate_nonlinear_interaction_rulebooks(
    model_spec: Any,
    branch_name: str,
    branch_data: Any,
    model: Any,
) -> None:
    if (not model_spec.nonlinear) or (not is_interaction_branch(branch_name)):
        return
    if branch_data.rulebook_support.empty:
        return
    importances = model_feature_importance_series(model, list(branch_data.X.columns))
    score_map: dict[str, float] = {}
    for rule_id in branch_data.rulebook_support["rule_id"].astype(str).tolist():
        score_map[rule_id] = float(abs(importances.get(f"cx::{rule_id}", 0.0))) if not importances.empty else 0.0
    if not any(score_map.values()):
        try:
            scores = stage3_models.predict_scores(model, branch_data.X.to_numpy(dtype=float))
        except Exception:
            scores = np.asarray([])
        if scores.size:
            for rule_id in branch_data.rulebook_support["rule_id"].astype(str).tolist():
                feat_name = f"cx::{rule_id}"
                if feat_name not in branch_data.X.columns:
                    continue
                values = pd.to_numeric(branch_data.X[feat_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                active = values > 0
                inactive = ~active
                if int(active.sum()) < 5 or int(inactive.sum()) < 5:
                    continue
                score_map[rule_id] = float(abs(scores[active].mean() - scores[inactive].mean()))

    def _apply_score(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        if df.empty or id_col not in df.columns:
            return df
        out = df.copy()
        out["interaction_model_score"] = out[id_col].astype(str).map(lambda x: score_map.get(x, 0.0)).astype(float)
        if "notes" in out.columns:
            out["notes"] = out.apply(
                lambda row: "; ".join(
                    [
                        piece
                        for piece in [
                            str(row.get("notes", "")),
                            (
                                f"source=continuous_interaction_release; interaction_model_score={row['interaction_model_score']:.6f}"
                                if row["interaction_model_score"] > 0
                                else ""
                            ),
                        ]
                        if piece and piece != "nan"
                    ]
                ),
                axis=1,
            )
        out = out.sort_values(
            [
                "rule_type" if "rule_type" in out.columns else id_col,
                "interaction_model_score",
                "stability_freq" if "stability_freq" in out.columns else id_col,
                "enrichment" if "enrichment" in out.columns else ("enrichment_c3" if "enrichment_c3" in out.columns else id_col),
            ],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        if "rule_rank" in out.columns:
            out["rule_rank"] = np.arange(1, len(out) + 1)
        return out.drop(columns=["interaction_model_score"])

    branch_data.rulebook_support = _apply_score(branch_data.rulebook_support, "rule_id")
    branch_data.pair_rulebook_publishable_c3only = _apply_score(branch_data.pair_rulebook_publishable_c3only, "rule_id")
    branch_data.pair_rulebook_explanation_unstable_c3only = _apply_score(branch_data.pair_rulebook_explanation_unstable_c3only, "rule_id")


def _safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_safe_json_dumps(payload) + "\n", encoding="utf-8")


def model_branch_dir(part2_out_dir: Path, model: str, feature_mode: str, branch: str) -> Path:
    return part2_out_dir / "models" / str(model) / str(feature_mode) / str(branch)


def build_artifact_completeness_report(part2_out_dir: Path, selection_df: pd.DataFrame) -> dict[str, Any]:
    missing_files: list[str] = []
    schema_checks: list[dict[str, Any]] = []
    file_checks: list[dict[str, Any]] = []

    for rel_path in ROOT_REQUIRED_FILES:
        exists = True if rel_path == "artifact_completeness_report.json" else (part2_out_dir / rel_path).exists()
        file_checks.append({"path": rel_path, "exists": exists, "scope": "root"})
        if not exists:
            missing_files.append(rel_path)

    for row in selection_df.itertuples(index=False):
        branch_dir = model_branch_dir(part2_out_dir, str(row.model), str(getattr(row, "feature_mode", "cont_plus_bin")), str(row.branch))
        for rel_path in BRANCH_REQUIRED_FILES:
            target = branch_dir / rel_path
            exists = target.exists()
            file_checks.append({"path": str(target.relative_to(part2_out_dir)), "exists": exists, "scope": "branch"})
            if not exists:
                missing_files.append(str(target.relative_to(part2_out_dir)))
        mech = branch_dir / "rulebook_mechanism_extraction.md"
        mech_exists = mech.exists()
        file_checks.append({"path": str(mech.relative_to(part2_out_dir)), "exists": mech_exists, "scope": "branch"})
        if not mech_exists:
            missing_files.append(str(mech.relative_to(part2_out_dir)))

        support_path = branch_dir / "rulebook_support.csv"
        if support_path.exists() and support_path.stat().st_size > 0:
            support_df = pd.read_csv(support_path)
        else:
            support_df = pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS)
        schema_checks.append(
            {
                "check": f"support_rulebook_schema::{row.model}::{row.branch}",
                "pass": all(col in support_df.columns for col in SUPPORT_RULEBOOK_COLUMNS),
            }
        )
        missing_anywhere_ok = True if support_df.empty else ~support_df["condition_text"].astype(str).str.contains("__MISSING__|missing_flag").any()
        schema_checks.append(
            {
                "check": f"support_rulebook_no_missing_anywhere::{row.model}::{row.branch}",
                "pass": bool(missing_anywhere_ok),
            }
        )
        pred_rows = support_df[support_df["rule_type"] == "prediction"].copy() if not support_df.empty else pd.DataFrame(columns=support_df.columns)
        cross_signal_ok = True
        if not pred_rows.empty:
            for pred in pred_rows.itertuples(index=False):
                if not has_feature(getattr(pred, "feature_b", "")) and not has_feature(getattr(pred, "feature_c", "")):
                    continue
                groups = distinct_signal_groups(str(getattr(pred, "feature_a", "")), str(getattr(pred, "feature_b", "")), str(getattr(pred, "feature_c", "")))
                if has_feature(getattr(pred, "feature_c", "")) and len(groups) < 3:
                    cross_signal_ok = False
                    break
                if (not has_feature(getattr(pred, "feature_c", ""))) and len(groups) < 2:
                    cross_signal_ok = False
                    break
        schema_checks.append(
            {
                "check": f"support_rulebook_cross_signal_prediction::{row.model}::{row.branch}",
                "pass": cross_signal_ok,
            }
        )
        support_engineered_path = branch_dir / "rulebook_support_engineered_comparison.csv"
        if support_engineered_path.exists() and support_engineered_path.stat().st_size > 0:
            support_engineered_df = pd.read_csv(support_engineered_path)
        else:
            support_engineered_df = pd.DataFrame(columns=SUPPORT_RULEBOOK_COLUMNS)
        schema_checks.append(
            {
                "check": f"support_engineered_comparison_schema::{row.model}::{row.branch}",
                "pass": all(col in support_engineered_df.columns for col in SUPPORT_RULEBOOK_COLUMNS),
            }
        )
        legacy_path = branch_dir / "rulebook_legacy_pair_tier2plus.csv"
        if legacy_path.exists() and legacy_path.stat().st_size > 0:
            legacy_df = pd.read_csv(legacy_path)
        else:
            legacy_df = pd.DataFrame(columns=LEGACY_RULEBOOK_COLUMNS)
        schema_checks.append(
            {
                "check": f"legacy_rulebook_schema::{row.model}::{row.branch}",
                "pass": list(legacy_df.columns) == LEGACY_RULEBOOK_COLUMNS if not legacy_df.empty else True,
            }
        )
        legacy_missing_ok = True if legacy_df.empty else ~legacy_df["condition_text"].astype(str).str.contains("__MISSING__|missing_flag").any()
        schema_checks.append(
            {
                "check": f"legacy_rulebook_no_missing_anywhere::{row.model}::{row.branch}",
                "pass": bool(legacy_missing_ok),
            }
        )
        c3_path = branch_dir / "pair_rulebook_publishable_c3only.csv"
        if c3_path.exists() and c3_path.stat().st_size > 0:
            c3_df = pd.read_csv(c3_path)
        else:
            c3_df = pd.DataFrame(columns=C3_PAIR_RULEBOOK_COLUMNS)
        schema_checks.append(
            {
                "check": f"c3_pair_rulebook_schema::{row.model}::{row.branch}",
                "pass": all(col in c3_df.columns for col in C3_PAIR_RULEBOOK_COLUMNS),
            }
        )
        c3_missing_ok = True if c3_df.empty else ~c3_df["condition_text"].astype(str).str.contains("__MISSING__|missing_flag").any()
        schema_checks.append(
            {
                "check": f"c3_pair_rulebook_no_missing_anywhere::{row.model}::{row.branch}",
                "pass": bool(c3_missing_ok),
            }
        )
        if not c3_df.empty:
            schema_checks.append(
                {
                    "check": f"c3_pair_rulebook_cross_signal::{row.model}::{row.branch}",
                    "pass": bool((c3_df["signal_group_a"].astype(str) != c3_df["signal_group_b"].astype(str)).all()),
                }
            )
        unstable_path = branch_dir / "pair_rulebook_explanation_unstable_c3only.csv"
        if unstable_path.exists() and unstable_path.stat().st_size > 0:
            unstable_df = pd.read_csv(unstable_path)
        else:
            unstable_df = pd.DataFrame(columns=UNSTABLE_EXPLANATION_PAIR_COLUMNS)
        schema_checks.append(
            {
                "check": f"unstable_pair_rulebook_schema::{row.model}::{row.branch}",
                "pass": all(col in unstable_df.columns for col in UNSTABLE_EXPLANATION_PAIR_COLUMNS),
            }
        )
        if not unstable_df.empty:
            missing_primary_ok = ~unstable_df["condition_text"].astype(str).str.contains("__MISSING__|missing_flag").any()
            cross_signal_ok = bool((unstable_df["signal_group_a"].astype(str) != unstable_df["signal_group_b"].astype(str)).all())
            explanation_only_ok = bool(unstable_df["explanation_only"].astype(bool).all())
            schema_checks.append(
                {
                    "check": f"unstable_pair_rulebook_constraints::{row.model}::{row.branch}",
                    "pass": bool(missing_primary_ok and cross_signal_ok and explanation_only_ok),
                }
            )
        model_derived_path = branch_dir / "rulebook_model_derived_sensitivity.csv"
        if model_derived_path.exists() and model_derived_path.stat().st_size > 0:
            model_derived_df = pd.read_csv(model_derived_path)
        else:
            model_derived_df = pd.DataFrame(columns=MODEL_DERIVED_RULEBOOK_COLUMNS)
        schema_checks.append(
            {
                "check": f"model_derived_rulebook_schema::{row.model}::{row.branch}",
                "pass": all(col in model_derived_df.columns for col in MODEL_DERIVED_RULEBOOK_COLUMNS),
            }
        )
        model_derived_missing_ok = True if model_derived_df.empty else ~model_derived_df["condition_text"].astype(str).str.contains("__MISSING__|missing_flag").any()
        schema_checks.append(
            {
                "check": f"model_derived_rulebook_no_missing_anywhere::{row.model}::{row.branch}",
                "pass": bool(model_derived_missing_ok),
            }
        )
        model_cut_path = branch_dir / "model_derived_cutpoint_alignment.csv"
        if model_cut_path.exists() and model_cut_path.stat().st_size > 0:
            model_cut_df = pd.read_csv(model_cut_path)
        else:
            model_cut_df = pd.DataFrame(columns=MODEL_DERIVED_CUTPOINT_COLUMNS)
        schema_checks.append(
            {
                "check": f"model_derived_cutpoint_schema::{row.model}::{row.branch}",
                "pass": all(col in model_cut_df.columns for col in MODEL_DERIVED_CUTPOINT_COLUMNS),
            }
        )
        linear_effect_path = branch_dir / "linear_continuous_effects.csv"
        if linear_effect_path.exists() and linear_effect_path.stat().st_size > 0:
            linear_effect_df = pd.read_csv(linear_effect_path)
        else:
            linear_effect_df = pd.DataFrame(columns=LINEAR_CONTINUOUS_EFFECT_COLUMNS)
        schema_checks.append(
            {
                "check": f"linear_continuous_effect_schema::{row.model}::{row.branch}",
                "pass": all(col in linear_effect_df.columns for col in LINEAR_CONTINUOUS_EFFECT_COLUMNS),
            }
        )
        linear_pair_path = branch_dir / "linear_pairwise_effects.csv"
        if linear_pair_path.exists() and linear_pair_path.stat().st_size > 0:
            linear_pair_df = pd.read_csv(linear_pair_path)
        else:
            linear_pair_df = pd.DataFrame(columns=LINEAR_PAIRWISE_EFFECT_COLUMNS)
        schema_checks.append(
            {
                "check": f"linear_pairwise_effect_schema::{row.model}::{row.branch}",
                "pass": all(col in linear_pair_df.columns for col in LINEAR_PAIRWISE_EFFECT_COLUMNS),
            }
        )
        direction_check_path = branch_dir / "linear_vs_engineered_direction_check.csv"
        if direction_check_path.exists() and direction_check_path.stat().st_size > 0:
            direction_df = pd.read_csv(direction_check_path)
        else:
            direction_df = pd.DataFrame(columns=LINEAR_DIRECTION_CHECK_COLUMNS)
        schema_checks.append(
            {
                "check": f"linear_direction_check_schema::{row.model}::{row.branch}",
                "pass": all(col in direction_df.columns for col in LINEAR_DIRECTION_CHECK_COLUMNS),
            }
        )

    overall_pass = (len(missing_files) == 0) and all(item["pass"] for item in schema_checks)
    report = {
        "overall_pass": overall_pass,
        "missing_files": missing_files,
        "file_checks": file_checks,
        "schema_checks": schema_checks,
        "generated_at": now_iso(),
    }
    _write_json(part2_out_dir / "artifact_completeness_report.json", report)
    return report
