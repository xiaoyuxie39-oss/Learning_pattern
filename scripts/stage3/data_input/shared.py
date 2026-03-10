#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from common import append_log, ensure_dir, load_simple_yaml, resolve_repo_path, write_json

FEATURE_MODES = ("cont_only", "bin_only", "cont_plus_bin")
MODE_OUTPUT_FILENAMES = {
    "cont_only": "interaction_feature_view_cont_only.csv",
    "bin_only": "interaction_feature_view_bin_only.csv",
    "cont_plus_bin": "interaction_feature_view_cont_plus_bin.csv",
}

REQUIRED_INPUT_COLS = [
    "power_mw",
    "rack_kw_typical",
    "pue",
    "cooling",
    "liquid_cool",
    "building_sqm",
    "rack_kw_peak",
    "whitespace_sqm",
    "rack_density_area_w_per_sf_dc",
]
EXTENDED_NUMERIC_INPUT_COLS = [
    "rack_kw_peak",
    "whitespace_sqm",
    "rack_density_area_w_per_sf_dc",
]
NUMERIC_BIN_SPECS = {
    "power_mw": {
        "bin_col": "power_mw_bin",
        "bins": [-np.inf, 20, 100, np.inf],
        "labels": ["[-inf,20)", "[20,100)", "[100,inf)"],
    },
    "rack_kw_typical": {
        "bin_col": "rack_kw_typical_bin",
        "bins": [-np.inf, 35, 80, np.inf],
        "labels": ["[-inf,35)", "[35,80)", "[80,inf)"],
    },
    "pue": {
        "bin_col": "pue_bin",
        "bins": [-np.inf, 1.15, 1.25, np.inf],
        "labels": ["[-inf,1.15)", "[1.15,1.25)", "[1.25,inf)"],
    },
    "building_sqm": {
        "bin_col": "building_sqm_bin",
        "bins": [-np.inf, 12000, 40000, np.inf],
        "labels": ["[-inf,12000)", "[12000,40000)", "[40000,inf)"],
    },
    "rack_kw_peak": {
        "bin_col": "rack_kw_peak_bin",
        "bins": [-np.inf, 100, 300, np.inf],
        "labels": ["[-inf,100)", "[100,300)", "[300,inf)"],
    },
    "whitespace_sqm": {
        "bin_col": "whitespace_sqm_bin",
        "bins": [-np.inf, 3000, 20000, np.inf],
        "labels": ["[-inf,3000)", "[3000,20000)", "[20000,inf)"],
    },
    "rack_density_area_w_per_sf_dc": {
        "bin_col": "rack_density_area_w_per_sf_dc_bin",
        "bins": [-np.inf, 150, 300, np.inf],
        "labels": ["[-inf,150)", "[150,300)", "[300,inf)"],
    },
}
SMALL_BIN_MIN_N = 10
SMALL_BIN_MIN_SHARE = 0.01

BASE_MISSING_FLAGS = [
    "power_mw_is_missing",
    "rack_kw_typical_is_missing",
    "pue_is_missing",
    "cooling_is_missing",
    "liquid_cool_is_missing",
    "building_sqm_is_missing",
]
EXTENDED_MISSING_FLAGS = [f"{col}_is_missing" for col in EXTENDED_NUMERIC_INPUT_COLS]

CONTINUOUS_MODE_COLS = [
    "power_mw",
    "rack_kw_typical",
    "pue",
    "building_sqm",
    *EXTENDED_NUMERIC_INPUT_COLS,
]
BIN_MODE_COLS = [
    "power_mw_bin",
    "rack_kw_typical_bin",
    "pue_bin",
    "cooling_bin",
    "liquid_cool_bin",
    "building_sqm_bin",
    "rack_kw_peak_bin",
    "whitespace_sqm_bin",
    "rack_density_area_w_per_sf_dc_bin",
    "cooling_norm",
    "liquid_cool_binary",
]
SHARED_MODE_COLS = [
    "base_non_missing_count",
    "coverage_tier",
    "triage_only_row",
    *BASE_MISSING_FLAGS,
    *EXTENDED_MISSING_FLAGS,
]
MANDATORY_PASSTHROUGH_COLS = [
    "id",
    "company",
    "llm_ai_dc_label",
    "accel_model",
    "accel_count",
    "year",
]


class BuildFailure(RuntimeError):
    """Raised when Part I hard prerequisites fail before Gate C can run."""


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def parse_numeric(value: Any) -> tuple[float, bool, bool, str]:
    if is_blank(value):
        return np.nan, False, False, ""

    raw = str(value).strip().replace(",", "")
    if raw.lower() in {"na", "n/a", "unknown"}:
        return np.nan, False, False, ""

    m = re.match(r"^[<>]=?\s*([-+]?\d*\.?\d+)$", raw)
    if not m:
        m = re.match(r"^[≤≥]\s*([-+]?\d*\.?\d+)$", raw)
    if m:
        return float(m.group(1)), True, False, "coerced_from_inequality"

    try:
        return float(raw), False, False, ""
    except ValueError:
        pass

    nums = re.findall(r"[-+]?\d*\.?\d+", raw)
    if len(nums) == 1:
        return float(nums[0]), False, False, "coerced_from_text"

    return np.nan, False, True, "numeric_parse_error"


def normalize_cooling(value: Any) -> tuple[Any, bool, bool, str]:
    if is_blank(value):
        return np.nan, False, False, ""

    raw = str(value).strip().lower()
    if raw in {"unknown", "unk"}:
        return np.nan, False, False, ""

    aliases = {
        "air": "air",
        "water_based_air": "water_based_air",
        "water based air": "water_based_air",
        "hybrid_air_liquid": "hybrid_air_liquid",
        "hybrid air liquid": "hybrid_air_liquid",
        "hybrid": "hybrid_air_liquid",
        "liquid_direct_or_loop": "liquid_direct_or_loop",
        "liquid direct or loop": "liquid_direct_or_loop",
        "direct_to_chip": "liquid_direct_or_loop",
        "liquid_immersion": "liquid_immersion",
        "immersion": "liquid_immersion",
    }

    if raw in aliases:
        return aliases[raw], False, False, ""

    return np.nan, False, True, "invalid_cooling_value"


def normalize_liquid_cool(value: Any) -> tuple[Any, bool, bool, str]:
    if is_blank(value):
        return np.nan, False, False, ""

    raw = str(value).strip().upper()
    if raw in {"Y", "YES", "TRUE", "1"}:
        return 1, False, False, ""
    if raw in {"N", "NO", "FALSE", "0"}:
        return 0, False, False, ""

    return np.nan, False, True, "invalid_liquid_cool_value"


def to_bin(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    out = pd.cut(series, bins=bins, right=False, labels=labels).astype("object")
    return out.fillna("__MISSING__")


def merge_small_numeric_bins(
    series: pd.Series,
    ordered_labels: list[str],
    *,
    min_n: int = SMALL_BIN_MIN_N,
    min_share: float = SMALL_BIN_MIN_SHARE,
) -> tuple[pd.Series, list[dict[str, str]]]:
    out = series.astype("object").copy()
    total = len(out)
    if total == 0:
        return out, []
    counts = out.value_counts(dropna=False).to_dict()
    actions: list[dict[str, str]] = []
    if len(ordered_labels) <= 1:
        return out, actions

    for idx, label in enumerate(ordered_labels):
        count = int(counts.get(label, 0))
        share = float(count / total) if total else 0.0
        if count == 0 or (count >= min_n and share >= min_share):
            continue
        if idx == 0:
            target = ordered_labels[1]
        elif idx == len(ordered_labels) - 1:
            target = ordered_labels[-2]
        else:
            prev_label = ordered_labels[idx - 1]
            next_label = ordered_labels[idx + 1]
            prev_count = int(counts.get(prev_label, 0))
            next_count = int(counts.get(next_label, 0))
            target = prev_label if prev_count >= next_count else next_label
        out = out.replace(label, target)
        counts[target] = int(counts.get(target, 0)) + count
        counts[label] = 0
        actions.append({"from_bucket": label, "to_bucket": target, "strategy": "merge_adjacent"})
    return out, actions


def merge_small_categorical_bins(
    series: pd.Series,
    *,
    min_n: int = SMALL_BIN_MIN_N,
    min_share: float = SMALL_BIN_MIN_SHARE,
    other_label: str = "__OTHER__",
) -> tuple[pd.Series, list[dict[str, str]]]:
    out = series.astype("object").copy()
    total = len(out)
    if total == 0:
        return out, []
    vc = out.value_counts(dropna=False)
    small_labels = [
        str(label)
        for label, count in vc.items()
        if str(label) != "__MISSING__" and (int(count) < min_n or float(count / total) < min_share)
    ]
    if not small_labels:
        return out, []
    out = out.replace(small_labels, other_label)
    actions = [{"from_bucket": label, "to_bucket": other_label, "strategy": "merge_other"} for label in small_labels]
    return out, actions


def coverage_tier(non_missing_count: int) -> str:
    if non_missing_count < 2:
        return "C0"
    if non_missing_count in {2, 3}:
        return "C1"
    if non_missing_count in {4, 5}:
        return "C2"
    return "C3"


def expected_columns_for_mode(feature_mode: str) -> list[str]:
    mode = str(feature_mode).strip().lower()
    if mode == "cont_only":
        return [*MANDATORY_PASSTHROUGH_COLS, *CONTINUOUS_MODE_COLS, *SHARED_MODE_COLS]
    if mode == "bin_only":
        return [*MANDATORY_PASSTHROUGH_COLS, *BIN_MODE_COLS, *SHARED_MODE_COLS]
    if mode == "cont_plus_bin":
        return [
            *MANDATORY_PASSTHROUGH_COLS,
            *CONTINUOUS_MODE_COLS,
            *BIN_MODE_COLS,
            *SHARED_MODE_COLS,
        ]
    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def select_feature_view_for_mode(full_df: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    mode = str(feature_mode).strip().lower()
    if mode == "cont_plus_bin":
        return full_df.copy()
    wanted = set(expected_columns_for_mode(mode))
    ordered = [col for col in full_df.columns if col in wanted]
    return full_df.loc[:, ordered].copy()


def _build_full_feature_view(
    df: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    part1_log: Path,
) -> dict[str, Any]:
    cleaning_report_rows: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []

    numeric_targets = [
        "power_mw",
        "rack_kw_typical",
        "pue",
        "building_sqm",
        *[col for col in EXTENDED_NUMERIC_INPUT_COLS if col in df.columns],
    ]
    for col in numeric_targets:
        before_non_missing = int(df[col].notna().sum())
        parsed_values: list[float] = []
        coerce_ineq_count = 0
        parse_error_count = 0

        for idx, raw in df[col].items():
            parsed, coerced_ineq, is_error, reason = parse_numeric(raw)
            parsed_values.append(parsed)
            if coerced_ineq:
                coerce_ineq_count += 1
            if is_error:
                parse_error_count += 1
                exception_rows.append(
                    {
                        "row_index": idx,
                        "source_id": df.iloc[idx].get("id", ""),
                        "column": col,
                        "raw_value": "" if is_blank(raw) else str(raw),
                        "reason": reason,
                    }
                )

        df[col] = parsed_values
        after_non_missing = int(pd.Series(parsed_values).notna().sum())
        cleaning_report_rows.append(
            {
                "column": col,
                "type": "numeric",
                "total_rows": len(df),
                "non_missing_before": before_non_missing,
                "non_missing_after": after_non_missing,
                "coerce_from_ineq_count": coerce_ineq_count,
                "parse_error_count": parse_error_count,
            }
        )

    cooling_values: list[Any] = []
    cooling_invalid_count = 0
    for idx, raw in df["cooling"].items():
        normalized, _, is_error, reason = normalize_cooling(raw)
        cooling_values.append(normalized)
        if is_error:
            cooling_invalid_count += 1
            exception_rows.append(
                {
                    "row_index": idx,
                    "source_id": df.iloc[idx].get("id", ""),
                    "column": "cooling",
                    "raw_value": "" if is_blank(raw) else str(raw),
                    "reason": reason,
                }
            )
    df["cooling_norm"] = cooling_values
    cleaning_report_rows.append(
        {
            "column": "cooling",
            "type": "categorical",
            "total_rows": len(df),
            "non_missing_before": int(df["cooling"].notna().sum()),
            "non_missing_after": int(pd.Series(cooling_values).notna().sum()),
            "coerce_from_ineq_count": 0,
            "parse_error_count": cooling_invalid_count,
        }
    )

    liquid_values: list[Any] = []
    liquid_invalid_count = 0
    for idx, raw in df["liquid_cool"].items():
        normalized, _, is_error, reason = normalize_liquid_cool(raw)
        liquid_values.append(normalized)
        if is_error:
            liquid_invalid_count += 1
            exception_rows.append(
                {
                    "row_index": idx,
                    "source_id": df.iloc[idx].get("id", ""),
                    "column": "liquid_cool",
                    "raw_value": "" if is_blank(raw) else str(raw),
                    "reason": reason,
                }
            )
    df["liquid_cool_binary"] = liquid_values
    cleaning_report_rows.append(
        {
            "column": "liquid_cool",
            "type": "categorical",
            "total_rows": len(df),
            "non_missing_before": int(df["liquid_cool"].notna().sum()),
            "non_missing_after": int(pd.Series(liquid_values).notna().sum()),
            "coerce_from_ineq_count": 0,
            "parse_error_count": liquid_invalid_count,
        }
    )

    df["power_mw_is_missing"] = df["power_mw"].isna().astype(int)
    df["rack_kw_typical_is_missing"] = df["rack_kw_typical"].isna().astype(int)
    df["pue_is_missing"] = df["pue"].isna().astype(int)
    df["cooling_is_missing"] = df["cooling_norm"].isna().astype(int)
    df["liquid_cool_is_missing"] = df["liquid_cool_binary"].isna().astype(int)
    df["building_sqm_is_missing"] = df["building_sqm"].isna().astype(int)
    for col in EXTENDED_NUMERIC_INPUT_COLS:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)

    threshold_version = manifest.get("versions", {}).get("threshold_version", "r1_default_v1")
    if threshold_version != "r1_default_v1":
        append_log(part1_log, f"threshold_version={threshold_version} (using r1_default_v1 fallback)")

    df["cooling_norm"] = df["cooling_norm"].astype("object").fillna("__MISSING__")
    df["liquid_cool_binary"] = df["liquid_cool_binary"].astype("object").where(
        pd.Series(df["liquid_cool_binary"]).notna(), "__MISSING__"
    )
    df["cooling_bin"] = df["cooling_norm"]
    df["liquid_cool_bin"] = df["liquid_cool_binary"]

    for source_col, spec in NUMERIC_BIN_SPECS.items():
        if source_col not in df.columns:
            continue
        df[spec["bin_col"]] = to_bin(
            df[source_col],
            bins=list(spec["bins"]),
            labels=list(spec["labels"]),
        )
        df[spec["bin_col"]], _ = merge_small_numeric_bins(
            df[spec["bin_col"]],
            ordered_labels=list(spec["labels"]),
        )

    df["cooling_bin"], _ = merge_small_categorical_bins(df["cooling_bin"])
    df["liquid_cool_bin"], _ = merge_small_categorical_bins(df["liquid_cool_bin"])

    base_non_missing_cols = [
        "power_mw",
        "rack_kw_typical",
        "pue",
        "cooling_norm",
        "liquid_cool_binary",
        "building_sqm",
    ]

    def _count_non_missing(row: pd.Series) -> int:
        cnt = 0
        for col in base_non_missing_cols:
            v = row[col]
            if col in {"cooling_norm", "liquid_cool_binary"}:
                if v != "__MISSING__":
                    cnt += 1
            else:
                if pd.notna(v):
                    cnt += 1
        return cnt

    df["base_non_missing_count"] = df.apply(_count_non_missing, axis=1)
    df["coverage_tier"] = df["base_non_missing_count"].apply(coverage_tier)
    df["triage_only_row"] = (df["coverage_tier"] == "C0").astype(int)

    required_derived_cols = [
        "power_mw_bin",
        "rack_kw_typical_bin",
        "pue_bin",
        "cooling_bin",
        "liquid_cool_bin",
        "building_sqm_bin",
        "cooling_norm",
        "liquid_cool_binary",
        "power_mw_is_missing",
        "rack_kw_typical_is_missing",
        "pue_is_missing",
        "cooling_is_missing",
        "liquid_cool_is_missing",
        "building_sqm_is_missing",
        "base_non_missing_count",
        "coverage_tier",
    ]
    for col in EXTENDED_NUMERIC_INPUT_COLS:
        if col in df.columns:
            required_derived_cols.append(NUMERIC_BIN_SPECS[col]["bin_col"])
            required_derived_cols.append(f"{col}_is_missing")

    gate_fail_reasons: list[str] = []
    gate_schema_rows: list[dict[str, Any]] = []

    missing_required_cols = [c for c in required_derived_cols if c not in df.columns]
    if missing_required_cols:
        gate_fail_reasons.append(f"missing_required_derived_columns:{','.join(missing_required_cols)}")
        gate_schema_rows.append(
            {
                "check_name": "C1_required_derived_columns",
                "status": "FAIL",
                "details": f"missing:{','.join(missing_required_cols)}",
            }
        )
    else:
        gate_schema_rows.append(
            {
                "check_name": "C1_required_derived_columns",
                "status": "PASS",
                "details": "all_required_columns_present",
            }
        )

    present_supplemental_cols = [c for c in EXTENDED_NUMERIC_INPUT_COLS if c in df.columns]
    supplemental_missing_flags = [
        f"{c}_is_missing" for c in present_supplemental_cols if f"{c}_is_missing" in df.columns
    ]
    supplemental_bin_cols = [
        NUMERIC_BIN_SPECS[c]["bin_col"]
        for c in present_supplemental_cols
        if NUMERIC_BIN_SPECS[c]["bin_col"] in df.columns
    ]
    gate_schema_rows.append(
        {
            "check_name": "C1_extended_input_columns",
            "status": "PASS",
            "details": (
                "none_present"
                if not present_supplemental_cols
                else (
                    f"present:{','.join(present_supplemental_cols)};"
                    f"bin_cols:{','.join(supplemental_bin_cols)};"
                    f"missing_flags:{','.join(supplemental_missing_flags)}"
                )
            ),
        }
    )

    whitelist = manifest.get("execution", {}).get("constant_feature_whitelist", [])
    if not isinstance(whitelist, list):
        whitelist = []

    constant_rows: list[dict[str, Any]] = []
    constant_blockers: list[str] = []
    constant_check_cols = sorted(
        {
            col
            for col in df.columns
            if col.endswith("_bin") or col.endswith("_is_missing")
        }
    )

    for col in constant_check_cols:
        vc = df[col].value_counts(dropna=False, normalize=True)
        top_share = float(vc.iloc[0]) if not vc.empty else 1.0
        top_value = str(vc.index[0]) if not vc.empty else "__EMPTY__"
        invalid_constant = top_share >= 0.99
        whitelisted = col in whitelist
        blocks = invalid_constant and not whitelisted
        if blocks:
            constant_blockers.append(col)
        constant_rows.append(
            {
                "column": col,
                "top_value": top_value,
                "top_share": round(top_share, 6),
                "invalid_constant_feature": bool(invalid_constant),
                "whitelisted": bool(whitelisted),
                "blocks_part2": bool(blocks),
            }
        )

    if constant_blockers:
        gate_fail_reasons.append(f"constant_feature_blockers:{','.join(constant_blockers)}")
        gate_schema_rows.append(
            {
                "check_name": "C2_constant_feature",
                "status": "FAIL",
                "details": f"blockers:{','.join(constant_blockers)}",
            }
        )
    else:
        gate_schema_rows.append(
            {
                "check_name": "C2_constant_feature",
                "status": "PASS",
                "details": "no_blocking_constant_features",
            }
        )

    missing_dominance_rows: list[dict[str, Any]] = []
    missing_share_by_field = {
        "power_mw": float(df["power_mw_is_missing"].mean()),
        "rack_kw_typical": float(df["rack_kw_typical_is_missing"].mean()),
        "pue": float(df["pue_is_missing"].mean()),
        "cooling_norm": float(df["cooling_is_missing"].mean()),
        "liquid_cool_binary": float(df["liquid_cool_is_missing"].mean()),
        "building_sqm": float(df["building_sqm_is_missing"].mean()),
    }

    valid_default_base = 0
    for field, miss_share in missing_share_by_field.items():
        dominant = miss_share > 0.60
        default_ok = not dominant
        if default_ok:
            valid_default_base += 1
        missing_dominance_rows.append(
            {
                "base_field": field,
                "missing_share": round(miss_share, 6),
                "is_dominant_missing": bool(dominant),
                "default_interaction_base": bool(default_ok),
            }
        )

    if valid_default_base < 4:
        gate_fail_reasons.append("missing_dominance_valid_default_base_lt_4")
        gate_schema_rows.append(
            {
                "check_name": "C3_missing_dominance",
                "status": "FAIL",
                "details": f"valid_default_base={valid_default_base}",
            }
        )
    else:
        gate_schema_rows.append(
            {
                "check_name": "C3_missing_dominance",
                "status": "PASS",
                "details": f"valid_default_base={valid_default_base}",
            }
        )

    bin_health_rows: list[dict[str, Any]] = []
    unprocessed_small_buckets: list[str] = []
    for col in sorted([col for col in df.columns if col.endswith("_bin")]):
        vc = df[col].value_counts(dropna=False)
        for bucket, n in vc.items():
            share = n / len(df) if len(df) else 0.0
            is_small = (n < 10) or (share < 0.01)
            handled = (not is_small) or (str(bucket) == "__OTHER__")
            bin_health_rows.append(
                {
                    "column": col,
                    "bucket": str(bucket),
                    "n": int(n),
                    "share": round(float(share), 6),
                    "is_small_bucket": bool(is_small),
                    "handled": bool(handled),
                }
            )
            if is_small and not handled:
                unprocessed_small_buckets.append(f"{col}:{bucket}")

    if unprocessed_small_buckets:
        gate_fail_reasons.append("small_bins_unprocessed")
        gate_schema_rows.append(
            {
                "check_name": "C4_bin_health",
                "status": "FAIL",
                "details": ";".join(unprocessed_small_buckets[:20]),
            }
        )
    else:
        gate_schema_rows.append(
            {
                "check_name": "C4_bin_health",
                "status": "PASS",
                "details": "no_small_bins",
            }
        )

    trainable_ratio = float((df["base_non_missing_count"] >= 2).mean()) if len(df) else 0.0
    if trainable_ratio < 0.75:
        gate_fail_reasons.append("trainable_ratio_lt_0.75")
        gate_schema_rows.append(
            {
                "check_name": "C5_row_level_coverage",
                "status": "FAIL",
                "details": f"trainable_ratio={trainable_ratio:.4f}",
            }
        )
    else:
        gate_schema_rows.append(
            {
                "check_name": "C5_row_level_coverage",
                "status": "PASS",
                "details": f"trainable_ratio={trainable_ratio:.4f}",
            }
        )

    gate_status = "PASS" if not gate_fail_reasons else "FAIL"
    gate_payload = {
        "gate_c_status": gate_status,
        "fail_reasons": gate_fail_reasons,
        "trainable_ratio": round(trainable_ratio, 6),
        "valid_default_base_fields": valid_default_base,
    }

    return {
        "full_df": df,
        "cleaning_report_rows": cleaning_report_rows,
        "exception_rows": exception_rows,
        "gate_schema_rows": gate_schema_rows,
        "constant_rows": constant_rows,
        "missing_dominance_rows": missing_dominance_rows,
        "bin_health_rows": bin_health_rows,
        "gate_payload": gate_payload,
    }


def run_mode_build(
    *,
    repo_root: Path,
    manifest_path: Path,
    feature_mode: str,
    write_legacy_view: bool = False,
) -> dict[str, Any]:
    mode = str(feature_mode).strip().lower()
    if mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    manifest = load_simple_yaml(manifest_path)
    paths = manifest.get("paths", {})
    run = manifest.get("run", {})

    input_csv = resolve_repo_path(repo_root, str(paths.get("input_csv", "data/AI_database_chrome__llm_final_R1.csv")))
    part1_out_dir = resolve_repo_path(repo_root, str(paths.get("part1_out_dir")))
    log_dir = resolve_repo_path(repo_root, str(paths.get("log_dir")))

    ensure_dir(part1_out_dir)
    ensure_dir(log_dir)
    part1_log = log_dir / "part1.log"
    append_log(part1_log, f"run_id={run.get('run_id', 'unknown')} part1_start feature_mode={mode}")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    missing_input_cols = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing_input_cols:
        gate_payload = {
            "gate_c_status": "FAIL",
            "fail_reasons": [f"missing_required_input_columns:{','.join(missing_input_cols)}"],
        }
        write_json(part1_out_dir / "gate_c_acceptance.json", gate_payload)
        raise BuildFailure(f"Missing required input columns: {missing_input_cols}")

    build_payload = _build_full_feature_view(df, manifest=manifest, part1_log=part1_log)
    full_df = build_payload["full_df"]
    gate_payload = build_payload["gate_payload"]
    gate_status = str(gate_payload.get("gate_c_status", "FAIL"))

    mode_df = select_feature_view_for_mode(full_df, mode)
    expected_cols = expected_columns_for_mode(mode)
    mode_contract_rows = [
        {
            "feature_mode": mode,
            "column": col,
            "is_present": bool(col in mode_df.columns),
            "is_required": True,
        }
        for col in expected_cols
    ]

    mode_file = part1_out_dir / MODE_OUTPUT_FILENAMES[mode]
    mode_df.to_csv(mode_file, index=False)

    if mode == "cont_plus_bin" or write_legacy_view:
        mode_df.to_csv(part1_out_dir / "interaction_feature_view.csv", index=False)

    pd.DataFrame(build_payload["cleaning_report_rows"]).to_csv(part1_out_dir / "cleaning_report.csv", index=False)
    pd.DataFrame(build_payload["exception_rows"]).to_csv(part1_out_dir / "cleaning_exceptions.csv", index=False)
    pd.DataFrame(build_payload["gate_schema_rows"]).to_csv(part1_out_dir / "gate_c_schema_report.csv", index=False)
    pd.DataFrame(build_payload["constant_rows"]).to_csv(part1_out_dir / "gate_c_constant_feature_report.csv", index=False)
    pd.DataFrame(build_payload["missing_dominance_rows"]).to_csv(part1_out_dir / "gate_c_missing_dominance_report.csv", index=False)
    pd.DataFrame(build_payload["bin_health_rows"]).to_csv(part1_out_dir / "bin_health_report.csv", index=False)
    pd.DataFrame(mode_contract_rows).to_csv(part1_out_dir / f"feature_mode_contract_{mode}.csv", index=False)

    write_json(part1_out_dir / "gate_c_acceptance.json", gate_payload)
    write_json(
        part1_out_dir / f"feature_view_build_summary_{mode}.json",
        {
            "feature_mode": mode,
            "input_csv": str(input_csv),
            "output_csv": str(mode_file),
            "legacy_interaction_feature_view_written": bool(mode == "cont_plus_bin" or write_legacy_view),
            "rows": int(len(mode_df)),
            "columns": int(len(mode_df.columns)),
            "gate_c_status": gate_status,
            "gate_fail_reasons": list(gate_payload.get("fail_reasons", [])),
        },
    )

    append_log(part1_log, f"part1_end feature_mode={mode} gate_c_status={gate_status}")

    if gate_status != "PASS":
        raise SystemExit(2)

    return {
        "feature_mode": mode,
        "output_file": str(mode_file),
        "gate_c_status": gate_status,
        "rows": int(len(mode_df)),
        "columns": int(len(mode_df.columns)),
    }
