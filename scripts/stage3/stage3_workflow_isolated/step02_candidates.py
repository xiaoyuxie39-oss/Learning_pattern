#!/usr/bin/env python3
from __future__ import annotations

from itertools import combinations
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
PAIR_SCOPE_TO_TIERS = {
    "C1C2C3": {"C1", "C2", "C3"},
    "C2C3": {"C2", "C3"},
}
TRIPLE_SCOPE_TO_TIERS = {
    "C2C3": {"C2", "C3"},
    "DISABLED": set(),
}
MISSING_FLAG_PREFIXES = (
    "power_mw_is_missing",
    "rack_kw_typical_is_missing",
    "pue_is_missing",
    "cooling_is_missing",
    "liquid_cool_is_missing",
    "building_sqm_is_missing",
    "year_is_missing",
)
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


def distinct_signal_groups(*feature_names: str) -> set[str]:
    return {group for group in (signal_group_of(name) for name in feature_names) if group}


def selection_freq_key(support_n: int, support_pos: int) -> str:
    return f"selection_freq_n{support_n}_p{support_pos}"


def fold_count_key(support_n: int, support_pos: int) -> str:
    return f"support_fold_count_n{support_n}_p{support_pos}"


def normalize_pair_candidate_scope(scope: str) -> str:
    text = str(scope or "C2C3").strip().upper()
    if text in {"C2C3", "C2_C3", "C2+C3", "TIER2PLUS", "TIER-2+"}:
        return "C2C3"
    if text in {"C1C2C3", "C1_C2_C3", "C1+C2+C3", "TIER1PLUS", "TIER-1+"}:
        return "C1C2C3"
    return "C2C3"


def normalize_triple_candidate_scope(scope: str) -> str:
    text = str(scope or "C2C3").strip().upper()
    if text in {"DISABLED", "OFF", "NONE", "FALSE", "0"}:
        return "DISABLED"
    if text in {"C2C3", "C2_C3", "C2+C3"}:
        return "C2C3"
    return "C2C3"


def make_frequency_columns(fold_n: list[int], fold_pos: list[int], n_splits: int) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for support_n in [15, 20]:
        for support_pos in [2, 3]:
            count = int(sum((n >= support_n) and (p >= support_pos) for n, p in zip(fold_n, fold_pos)))
            results[fold_count_key(support_n, support_pos)] = count
            results[selection_freq_key(support_n, support_pos)] = float(count / n_splits) if n_splits else 0.0
    return results


def _candidate_common(
    *,
    candidate_id: str,
    rule_family: str,
    feature_a: str,
    feature_b: str,
    feature_c: str,
    value_a: str,
    value_b: str,
    value_c: str,
    cond: pd.Series,
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    complete_case: pd.Series,
    tier_min: str,
) -> dict[str, Any]:
    signal_group_a = signal_group_of(feature_a)
    signal_group_b = signal_group_of(feature_b)
    signal_group_c = signal_group_of(feature_c)
    distinct_group_count = len(distinct_signal_groups(feature_a, feature_b, feature_c))
    known_group_count = len([g for g in [signal_group_a, signal_group_b, signal_group_c] if g])
    same_signal_group = False
    if rule_family == "pair":
        same_signal_group = bool(signal_group_a and signal_group_b and signal_group_a == signal_group_b)
    elif rule_family == "triple":
        non_empty = [g for g in [signal_group_a, signal_group_b, signal_group_c] if g]
        same_signal_group = len(non_empty) != len(set(non_empty))
    support_n = int(cond.sum())
    support_pos = int(y[cond.to_numpy()].sum())
    fold_n: list[int] = []
    fold_pos: list[int] = []
    for sp in splits:
        train_idx = sp["train_idx"]
        cond_tr = cond.iloc[train_idx].to_numpy()
        y_tr = y[train_idx]
        n_fold = int(cond_tr.sum())
        pos_fold = int(y_tr[cond_tr].sum()) if n_fold > 0 else 0
        fold_n.append(n_fold)
        fold_pos.append(pos_fold)
    base_rate = float(np.mean(y)) if len(y) else np.nan
    support_rate = (support_pos / support_n) if support_n else np.nan
    enrichment = (support_rate / base_rate) if support_n and np.isfinite(base_rate) and base_rate > 0 else np.nan
    freq_cols = make_frequency_columns(fold_n, fold_pos, len(splits))
    values = [value_a, value_b, value_c]
    values = [v for v in values if v != ""]
    primary_contains_missing = any(v == "__MISSING__" for v in values)
    indicator_rate = support_n / len(df) if len(df) else 0.0
    is_constant_flag = max(indicator_rate, 1.0 - indicator_rate) >= 0.99
    row = {
        "candidate_id": candidate_id,
        "type": rule_family,
        "feature_a": feature_a,
        "feature_b": feature_b,
        "feature_c": feature_c,
        "value_a": value_a,
        "value_b": value_b,
        "value_c": value_c,
        "primary_condition": " & ".join(
            [
                piece
                for piece in [
                    f"{feature_a}={value_a}" if feature_a else "",
                    f"{feature_b}={value_b}" if feature_b else "",
                    f"{feature_c}={value_c}" if feature_c else "",
                ]
                if piece
            ]
        ),
        "support_n": support_n,
        "support_pos": support_pos,
        "support_n_by_fold_min": int(min(fold_n)) if fold_n else 0,
        "support_pos_by_fold_min": int(min(fold_pos)) if fold_pos else 0,
        "complete_case_coverage": round(float(complete_case.mean()) if len(complete_case) else 0.0, 6),
        "missing_share": round(float(1.0 - complete_case.mean()) if len(complete_case) else 1.0, 6),
        "coverage": round(float(indicator_rate), 6),
        "enrichment": round(float(enrichment), 6) if np.isfinite(enrichment) else np.nan,
        "primary_condition_contains_missing": bool(primary_contains_missing),
        "contains_missing_flag": any(x.startswith(MISSING_FLAG_PREFIXES) for x in [feature_a, feature_b, feature_c] if x),
        "is_constant_flag": bool(is_constant_flag),
        "signal_group_a": signal_group_a,
        "signal_group_b": signal_group_b,
        "signal_group_c": signal_group_c,
        "distinct_signal_group_count": distinct_group_count,
        "known_signal_group_count": known_group_count,
        "same_signal_group": same_signal_group,
        "rule_tier_min": tier_min,
        **freq_cols,
    }
    return row


def generate_pairwise_candidates(
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    *,
    candidate_scope: str = "C1C2C3",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scope_key = normalize_pair_candidate_scope(candidate_scope)
    scope_tiers = PAIR_SCOPE_TO_TIERS.get(scope_key, PAIR_SCOPE_TO_TIERS["C1C2C3"])
    scope_mask = df["coverage_tier"].isin(scope_tiers) if "coverage_tier" in df.columns else pd.Series(True, index=df.index)
    tier_label = "Tier-1+" if scope_key == "C1C2C3" else "Tier-2+"
    for col1, col2 in combinations(CAT_BASE_COLS, 2):
        complete_case = scope_mask & (df[col1] != "__MISSING__") & (df[col2] != "__MISSING__")
        grouped = df.loc[scope_mask, [col1, col2]].groupby([col1, col2], dropna=False).size().reset_index(name="support_n")
        for _, grp in grouped.iterrows():
            v1 = str(grp[col1])
            v2 = str(grp[col2])
            cond = (df[col1] == v1) & (df[col2] == v2) & scope_mask
            rows.append(
                _candidate_common(
                    candidate_id=f"pair::{col1}={v1}__{col2}={v2}",
                    rule_family="pair",
                    feature_a=col1,
                    feature_b=col2,
                    feature_c="",
                    value_a=v1,
                    value_b=v2,
                    value_c="",
                    cond=cond,
                    df=df,
                    y=y,
                    splits=splits,
                    complete_case=complete_case,
                    tier_min=tier_label,
                )
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["candidate_train_scope"] = scope_key
    out["selection_freq"] = out[selection_freq_key(20, 3)]
    return out.sort_values(["enrichment", "support_pos", "support_n"], ascending=[False, False, False]).reset_index(drop=True)


def generate_triple_candidates(
    df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    pair_candidates_for_hierarchy: pd.DataFrame,
    *,
    candidate_scope: str = "C2C3",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scope_key = normalize_triple_candidate_scope(candidate_scope)
    if scope_key == "DISABLED":
        return pd.DataFrame(rows)
    if pair_candidates_for_hierarchy.empty:
        return pd.DataFrame(rows)
    eligible_rows = df["coverage_tier"].isin(TRIPLE_SCOPE_TO_TIERS.get(scope_key, {"C2", "C3"}))
    for _, pair in pair_candidates_for_hierarchy.iterrows():
        pair_cond = (df[pair["feature_a"]] == pair["value_a"]) & (df[pair["feature_b"]] == pair["value_b"])
        remaining = [c for c in CAT_BASE_COLS if c not in {pair["feature_a"], pair["feature_b"]}]
        for col3 in remaining:
            complete_case = (
                (df[pair["feature_a"]] != "__MISSING__")
                & (df[pair["feature_b"]] != "__MISSING__")
                & (df[col3] != "__MISSING__")
                & eligible_rows
            )
            grouped = df.loc[eligible_rows, [col3]].groupby(col3, dropna=False).size().reset_index(name="n")
            for _, grp in grouped.iterrows():
                v3 = str(grp[col3])
                cond = pair_cond & (df[col3] == v3) & eligible_rows
                rows.append(
                    _candidate_common(
                        candidate_id=(
                            f"triple::{pair['feature_a']}={pair['value_a']}__"
                            f"{pair['feature_b']}={pair['value_b']}__{col3}={v3}"
                        ),
                        rule_family="triple",
                        feature_a=str(pair["feature_a"]),
                        feature_b=str(pair["feature_b"]),
                        feature_c=col3,
                        value_a=str(pair["value_a"]),
                        value_b=str(pair["value_b"]),
                        value_c=v3,
                        cond=cond,
                        df=df,
                        y=y,
                        splits=splits,
                        complete_case=complete_case,
                        tier_min="C2/C3",
                    )
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["candidate_train_scope"] = scope_key
    out["selection_freq"] = out[selection_freq_key(20, 3)]
    return out.sort_values(["enrichment", "support_pos", "support_n"], ascending=[False, False, False]).reset_index(drop=True)


def classify_candidates(
    candidates: pd.DataFrame,
    threshold: Any,
    *,
    pair_limit: int = 4,
    triple_limit: int = 0,
    enforce_cross_signal_publishable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        empty = candidates.copy()
        for col, default in {
            "stability_freq": 0.0,
            "prediction_candidate": False,
            "downgrade_reason": "",
            "rule_type": "",
            "prediction_eligible": False,
            "notes": "",
            "threshold_id": threshold.threshold_id,
        }.items():
            empty[col] = default
        trace_cols = [
            "candidate_id",
            "type",
            "threshold_id",
            "support_n_by_fold_min",
            "support_pos_by_fold_min",
            "stability_freq",
            "signal_group_a",
            "signal_group_b",
            "signal_group_c",
            "same_signal_group",
            "distinct_signal_group_count",
            "prediction_candidate",
            "rule_type",
            "downgrade_reason",
        ]
        for col in trace_cols:
            if col not in empty.columns:
                empty[col] = ""
        trace = empty[trace_cols].copy()
        trace["selection_freq_for_threshold"] = 0.0
        trace = trace[
            [
                "candidate_id",
                "type",
                "threshold_id",
                "support_n_by_fold_min",
                "support_pos_by_fold_min",
                "selection_freq_for_threshold",
                "stability_freq",
                "signal_group_a",
                "signal_group_b",
                "signal_group_c",
                "same_signal_group",
                "distinct_signal_group_count",
                "prediction_candidate",
                "rule_type",
                "downgrade_reason",
            ]
        ]
        return empty, trace

    df = candidates.copy()
    def _int_or_default(value: Any, default: int) -> int:
        if value is None:
            return int(default)
        try:
            return int(value)
        except Exception:
            return int(default)

    pair_support_n = _int_or_default(getattr(threshold, "pair_support_n", None), _int_or_default(getattr(threshold, "support_n", 20), 20))
    pair_support_pos = _int_or_default(getattr(threshold, "pair_support_pos", None), _int_or_default(getattr(threshold, "support_pos", 3), 3))
    triple_support_n = _int_or_default(getattr(threshold, "triple_support_n", None), pair_support_n)
    triple_support_pos = _int_or_default(getattr(threshold, "triple_support_pos", None), pair_support_pos)

    selection_cutoff = threshold.pair_freq
    triple_cutoff = threshold.triple_freq
    reasons: list[list[str]] = [[] for _ in range(len(df))]
    prediction_flags: list[bool] = []
    stability_values: list[float] = []

    for idx, row in enumerate(df.itertuples(index=False)):
        is_triple = row.type == "triple"
        support_n_required = triple_support_n if is_triple else pair_support_n
        support_pos_required = triple_support_pos if is_triple else pair_support_pos
        freq_required = triple_cutoff if is_triple else selection_cutoff
        freq_col = selection_freq_key(support_n_required, support_pos_required)
        freq_value = float(getattr(row, freq_col, 0.0))
        stability_values.append(freq_value)
        eligible = True
        if int(row.support_n_by_fold_min) < support_n_required:
            reasons[idx].append(f"support_n_by_fold_min_lt_{support_n_required}")
            eligible = False
        if int(row.support_pos_by_fold_min) < support_pos_required:
            reasons[idx].append(f"support_pos_by_fold_min_lt_{support_pos_required}")
            eligible = False
        if freq_value < freq_required:
            reasons[idx].append(f"selection_freq_lt_{freq_required:.2f}")
            eligible = False
        if bool(row.primary_condition_contains_missing) or bool(getattr(row, "contains_missing_flag", False)):
            reasons[idx].append("missing_primary_condition")
            eligible = False
        if bool(row.is_constant_flag):
            reasons[idx].append("is_constant_flag=true")
            eligible = False
        if enforce_cross_signal_publishable:
            if int(getattr(row, "known_signal_group_count", 0)) == 0:
                reasons[idx].append("unknown_signal_group")
                eligible = False
            elif bool(getattr(row, "same_signal_group", False)):
                reasons[idx].append("same_signal_group")
                eligible = False
            elif is_triple and int(getattr(row, "distinct_signal_group_count", 0)) < 3:
                reasons[idx].append("distinct_signal_group_count_lt_3")
                eligible = False
        prediction_flags.append(eligible)

    df["stability_freq"] = stability_values
    df["prediction_candidate"] = prediction_flags
    df["downgrade_reason"] = ["|".join(x) for x in reasons]
    df = df.sort_values(
        ["prediction_candidate", "stability_freq", "enrichment", "support_pos", "support_n"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    selected_ids: set[str] = set()
    for rule_type, limit in [("pair", pair_limit), ("triple", triple_limit)]:
        subset = df[(df["type"] == rule_type) & (df["prediction_candidate"])].head(limit)
        selected_ids.update(subset["candidate_id"].tolist())
        overflow = df[(df["type"] == rule_type) & (df["prediction_candidate"]) & (~df["candidate_id"].isin(selected_ids))]
        if not overflow.empty:
            mask = df["candidate_id"].isin(overflow["candidate_id"])
            df.loc[mask, "prediction_candidate"] = False
            df.loc[mask, "downgrade_reason"] = df.loc[mask, "downgrade_reason"].fillna("")
            df.loc[mask, "downgrade_reason"] = df.loc[mask, "downgrade_reason"].astype(str).apply(
                lambda s: "|".join([p for p in [s, "truncated_by_rank_limit"] if p and p != "nan"])
            )

    df["rule_type"] = np.where(df["candidate_id"].isin(selected_ids), "prediction", "triage")
    df["prediction_eligible"] = df["rule_type"] == "prediction"
    df["notes"] = ""
    df.loc[df["rule_type"] == "triage", "notes"] = "diagnostic only"
    df["threshold_id"] = threshold.threshold_id

    trace = df[
        [
            "candidate_id",
            "type",
            "threshold_id",
            "support_n_by_fold_min",
            "support_pos_by_fold_min",
            "stability_freq",
            "signal_group_a",
            "signal_group_b",
            "signal_group_c",
            "same_signal_group",
            "distinct_signal_group_count",
            "prediction_candidate",
            "rule_type",
            "downgrade_reason",
        ]
    ].copy()
    trace.insert(5, "selection_freq_for_threshold", trace["stability_freq"].astype(float))
    return df, trace
