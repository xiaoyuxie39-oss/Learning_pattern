#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

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
LOG1P_CONTINUOUS_SOURCE_COLS = {
    "power_mw",
    "rack_kw_typical",
    "rack_kw_peak",
    "rack_density_area_w_per_sf_dc",
    "building_sqm",
    "whitespace_sqm",
}
FEATURE_MODES = ("cont_only", "bin_only", "cont_plus_bin")
NUM_BASE_COLS = [
    "power_mw_is_missing",
    "rack_kw_typical_is_missing",
    "pue_is_missing",
    "cooling_is_missing",
    "liquid_cool_is_missing",
    "building_sqm_is_missing",
    "base_non_missing_count",
]
LEAKAGE_PREFIXES = ("llm_", "accel_")
LEAKAGE_EXACT = {"stage", "type", "level", "year"}
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


def log1p_nonnegative(values: pd.Series) -> pd.Series:
    clipped = pd.to_numeric(values, errors="coerce").clip(lower=0)
    return np.log1p(clipped)


def finite_matrix_or_raise(name: str, values: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values after sanitization")
    return out


def feature_mode_flags(feature_mode: str) -> tuple[bool, bool]:
    mode = str(feature_mode or "cont_plus_bin").strip().lower()
    if mode == "cont_only":
        return True, False
    if mode == "bin_only":
        return False, True
    if mode == "cont_plus_bin":
        return True, True
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def summarize_feature_frame(feature_df: pd.DataFrame) -> dict[str, Any]:
    cols = [str(col) for col in feature_df.columns]
    n_cont = int(sum(col.startswith("cont::") for col in cols))
    n_bin_onehot = int(sum("=" in col for col in cols))
    n_missing_indicators = int(sum(col.endswith("_is_missing") for col in cols))
    return {
        "n_features_total": int(len(cols)),
        "n_cont": n_cont,
        "n_bin_onehot": n_bin_onehot,
        "n_missing_indicators": n_missing_indicators,
        "x_columns_sha256": hashlib.sha256("\n".join(cols).encode("utf-8")).hexdigest(),
    }


def compute_continuous_scaling_stats(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in feature_df.columns:
        if not str(col).startswith("cont::"):
            continue
        series = pd.to_numeric(feature_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean = float(series.mean()) if len(series) else 0.0
        std = float(series.std(ddof=0)) if len(series) else 1.0
        if (not np.isfinite(std)) or std < 1e-8:
            std = 1.0
        rows.append(
            {
                "feature_name": str(col),
                "source_col": str(col).split("::", 1)[1],
                "mean": mean,
                "std": std,
                "q10_raw": float(series.quantile(0.10)) if len(series) else np.nan,
                "q50_raw": float(series.quantile(0.50)) if len(series) else np.nan,
                "q90_raw": float(series.quantile(0.90)) if len(series) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stabilize_feature_frame(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = feature_df.copy()
    stats = compute_continuous_scaling_stats(feature_df)
    if not stats.empty:
        stats_by_feature = {str(row.feature_name): row for row in stats.itertuples(index=False)}
        for col in [c for c in out.columns if str(c).startswith("cont::")]:
            row = stats_by_feature.get(str(col))
            if row is None:
                continue
            series = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            out[col] = np.clip((series - float(row.mean)) / float(row.std), -6.0, 6.0)
    return out, stats


def stabilize_design_matrices(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train = X_train_df.copy()
    test = X_test_df.copy()
    stats = compute_continuous_scaling_stats(train)
    if not stats.empty:
        stats_by_feature = {str(row.feature_name): row for row in stats.itertuples(index=False)}
        for col in [c for c in train.columns if str(c).startswith("cont::")]:
            row = stats_by_feature.get(str(col))
            if row is None:
                continue
            train_col = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            test_col = pd.to_numeric(test[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            train[col] = np.clip((train_col - float(row.mean)) / float(row.std), -6.0, 6.0)
            test[col] = np.clip((test_col - float(row.mean)) / float(row.std), -6.0, 6.0)
    X_train = finite_matrix_or_raise("X_train", train.to_numpy(dtype=float))
    X_test = finite_matrix_or_raise("X_test", test.to_numpy(dtype=float))
    return np.clip(X_train, -10.0, 10.0), np.clip(X_test, -10.0, 10.0)


def year_to_continuous(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    year_numeric = pd.to_numeric(series, errors="coerce")
    year_missing = year_numeric.isna().astype(float)
    year_filled = year_numeric.replace([np.inf, -np.inf], np.nan)
    year_filled = year_filled.fillna(year_filled.median() if year_filled.notna().any() else 0.0).astype(float)
    return year_filled, year_missing


def prepare_base_features(
    df: pd.DataFrame,
    *,
    include_missing_indicators: bool = True,
    include_year_sensitivity: bool = False,
    include_continuous: bool = True,
    include_bins: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if not include_continuous and not include_bins:
        raise ValueError("At least one of include_continuous/include_bins must be True")
    meta: dict[str, dict[str, Any]] = {}
    if include_bins:
        for col in CAT_BASE_COLS:
            if col not in df.columns:
                raise ValueError(f"Missing required feature column: {col}")
            values = df[col].fillna("__MISSING__").astype(str)
            df[col] = values
            meta[col] = {"kind": "categorical"}

    if include_missing_indicators:
        num_cols = list(NUM_BASE_COLS)
    else:
        num_cols = ["base_non_missing_count"]
    for col in num_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")
        meta[col] = {"kind": "numeric"}

    cat_df = pd.get_dummies(df[CAT_BASE_COLS], prefix=CAT_BASE_COLS, prefix_sep="=", dtype=float) if include_bins else pd.DataFrame(index=df.index)
    num_df = df[num_cols].fillna(0).astype(float)
    cont_df = pd.DataFrame(index=df.index)
    if include_continuous:
        for src in CONTINUOUS_SOURCE_COLS:
            if src not in df.columns:
                continue
            series = pd.to_numeric(df[src], errors="coerce")
            if src in LOG1P_CONTINUOUS_SOURCE_COLS:
                series = log1p_nonnegative(series)
            series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
            feature_name = f"cont::{src}"
            cont_df[feature_name] = series
            meta[feature_name] = {
                "kind": "continuous_main_fit",
                "source_col": src,
                "feature_a": src,
                "feature_b": "",
                "feature_c": "",
                "contains_missing": False,
                "is_missing_flag": False,
                "rule_tier_min": "Tier-2+",
                "signal_group": signal_group_of(src),
            }
    feat_df = pd.concat([num_df, cont_df, cat_df], axis=1)

    for col in cat_df.columns:
        feature_name, _, value = col.partition("=")
        meta[col] = {
            "kind": "rule_indicator",
            "condition_text": f"{feature_name} == {value}",
            "feature_a": feature_name,
            "value_a": value,
            "feature_b": "",
            "feature_c": "",
            "contains_missing": value == "__MISSING__",
            "is_missing_flag": False,
            "rule_tier_min": "Tier-2+",
        }

    for col in num_cols:
        if col.endswith("_is_missing"):
            meta[col] = {
                "kind": "missing_indicator",
                "condition_text": f"{col} == 1",
                "feature_a": col,
                "feature_b": "",
                "feature_c": "",
                "contains_missing": True,
                "is_missing_flag": True,
                "rule_tier_min": "diagnostic_only",
            }
        elif col == "base_non_missing_count":
            meta[col] = {
                "kind": "numeric",
                "condition_text": "base_non_missing_count",
                "feature_a": col,
                "feature_b": "",
                "feature_c": "",
                "contains_missing": False,
                "is_missing_flag": False,
                "rule_tier_min": "Tier-2+",
                "signal_group": "",
            }

    if include_year_sensitivity:
        year_continuous, year_missing = year_to_continuous(
            df["year"] if "year" in df.columns else pd.Series(np.nan, index=df.index)
        )
        feat_df["cont::year"] = year_continuous.astype(float)
        feat_df["year_is_missing"] = year_missing.astype(float)
        meta["cont::year"] = {
            "kind": "year_sensitivity_continuous",
            "condition_text": "year (continuous sensitivity)",
            "source_col": "year",
            "feature_a": "year",
            "feature_b": "",
            "feature_c": "",
            "contains_missing": False,
            "is_missing_flag": False,
            "rule_tier_min": "sensitivity_only",
            "signal_group": "",
        }
        meta["year_is_missing"] = {
            "kind": "year_sensitivity",
            "condition_text": "year_is_missing == 1",
            "feature_a": "year_is_missing",
            "feature_b": "",
            "feature_c": "",
            "contains_missing": True,
            "is_missing_flag": True,
            "rule_tier_min": "sensitivity_only",
        }

    leaked = [c for c in df.columns if c.startswith(LEAKAGE_PREFIXES) or c in LEAKAGE_EXACT]
    if leaked:
        pass

    feat_df = feat_df.astype(float)
    return feat_df, meta
