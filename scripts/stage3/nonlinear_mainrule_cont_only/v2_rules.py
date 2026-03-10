#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v2_shared import CONTINUOUS_SOURCE_COLS, LOG1P_CONTINUOUS_SOURCE_COLS

PHYSICAL_THRESHOLDS = {
    "power_mw": [20.0, 100.0],
    "rack_kw_typical": [35.0, 80.0],
    "pue": [1.15, 1.25],
    "building_sqm": [12000.0, 40000.0],
    "rack_kw_peak": [100.0, 300.0],
    "whitespace_sqm": [3000.0, 20000.0],
    "rack_density_area_w_per_sf_dc": [150.0, 300.0],
}


@dataclass
class CutpointInfo:
    source_col: str
    model_cutpoints_raw: list[float]
    model_cutpoints_std: list[float]
    cutpoint_source: str
    data_driven_grid: list[float]


def fmt_num(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    if float(v).is_integer():
        return str(int(v))
    return f"{float(v):.6g}"


def format_threshold_list(values: Sequence[float]) -> str:
    clean = [float(v) for v in values if np.isfinite(v)]
    return ",".join(fmt_num(v) for v in clean)


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


def model_feature_importance_series(model: Any, feature_names: Sequence[str]) -> pd.Series:
    if model is None:
        return pd.Series(dtype=float)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_).reshape(-1)
        return pd.Series(values, index=list(feature_names))
    if hasattr(model, "term_names_") and hasattr(model, "term_importances"):
        values = np.asarray(model.term_importances())
        names = [str(n) for n in list(model.term_names_)]
        return pd.Series(values.reshape(-1), index=names)
    return pd.Series(dtype=float)


def compute_data_driven_thresholds(source_df: pd.DataFrame, source_col: str) -> list[float]:
    if source_col not in source_df.columns:
        return []
    series = pd.to_numeric(source_df[source_col], errors="coerce")
    if "coverage_tier" in source_df.columns:
        series = series[source_df["coverage_tier"].isin(["C2", "C3"])]
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 20:
        return []
    q = series.quantile([1 / 3, 2 / 3]).tolist()
    return sorted({float(v) for v in q if np.isfinite(v)})


def to_raw_value(source_col: str, transformed_value: float) -> float:
    value = float(transformed_value)
    if source_col in LOG1P_CONTINUOUS_SOURCE_COLS:
        value = float(np.expm1(value))
        value = max(value, 0.0)
    return value


def denormalize_cutpoints(source_col: str, cutpoints_std: Sequence[float], scaling_stats: pd.DataFrame) -> list[float]:
    if scaling_stats.empty:
        return []
    sub = scaling_stats[scaling_stats["source_col"] == source_col]
    if sub.empty:
        return []
    mean = float(sub.iloc[0]["mean"])
    std = float(sub.iloc[0]["std"])
    if (not np.isfinite(std)) or std <= 0:
        return []
    rows: list[float] = []
    for cp in cutpoints_std:
        if not np.isfinite(cp):
            continue
        transformed = float(cp) * std + mean
        raw = to_raw_value(source_col, transformed)
        if np.isfinite(raw):
            rows.append(float(raw))
    return sorted({float(v) for v in rows if np.isfinite(v)})


def extract_ebm_cutpoints_std(model: Any, feature_names: Sequence[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    bins_attr = getattr(model, "bins_", None)
    if bins_attr is None:
        return out
    for idx, feature_name in enumerate(feature_names):
        if not str(feature_name).startswith("cont::"):
            continue
        if idx >= len(bins_attr):
            continue
        raw = bins_attr[idx]
        if isinstance(raw, list) and raw:
            raw = raw[0]
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except Exception:
            continue
        source_col = str(feature_name).split("::", 1)[1]
        out[source_col] = sorted({float(v) for v in arr if np.isfinite(v)})
    return out


def extract_gbdt_cutpoints_std(model: Any, feature_names: Sequence[str]) -> dict[str, list[float]]:
    out_raw: dict[str, list[float]] = {}
    predictors = getattr(model, "_predictors", None)
    if predictors is None:
        return {}
    for stage in predictors:
        for pred in stage:
            nodes = getattr(pred, "nodes", None)
            if nodes is None:
                continue
            feature_idx = nodes["feature_idx"]
            thresholds = nodes["num_threshold"]
            for fidx, thr in zip(feature_idx, thresholds):
                if int(fidx) < 0 or (not np.isfinite(thr)):
                    continue
                if int(fidx) >= len(feature_names):
                    continue
                feature_name = str(feature_names[int(fidx)])
                if not feature_name.startswith("cont::"):
                    continue
                source_col = feature_name.split("::", 1)[1]
                out_raw.setdefault(source_col, []).append(float(thr))

    out: dict[str, list[float]] = {}
    for source_col, values in out_raw.items():
        if not values:
            continue
        rounded = [round(v, 6) for v in values]
        freq = Counter(rounded)
        top = [float(item[0]) for item in freq.most_common(3)]
        median_v = float(np.median(values))
        out[source_col] = sorted({*top, median_v})[:3]
    return out


def build_cutpoint_map(
    *,
    model_name: str,
    model: Any,
    feature_names: Sequence[str],
    scaling_stats: pd.DataFrame,
    source_df: pd.DataFrame,
) -> dict[str, CutpointInfo]:
    if model_name == "ebm":
        std_map = extract_ebm_cutpoints_std(model, feature_names)
        source_name = "ebm_bins"
    elif model_name == "gbdt":
        std_map = extract_gbdt_cutpoints_std(model, feature_names)
        source_name = "gbdt_split_distribution"
    else:
        std_map = {}
        source_name = "unknown"

    out: dict[str, CutpointInfo] = {}
    for source_col in CONTINUOUS_SOURCE_COLS:
        std_points = std_map.get(source_col, [])
        raw_points = denormalize_cutpoints(source_col, std_points, scaling_stats)
        out[source_col] = CutpointInfo(
            source_col=source_col,
            model_cutpoints_raw=raw_points,
            model_cutpoints_std=list(std_points),
            cutpoint_source=source_name if raw_points else "missing_model_cutpoints",
            data_driven_grid=compute_data_driven_thresholds(source_df, source_col),
        )
    return out


def build_cutpoint_alignment_rows(cut_map: dict[str, CutpointInfo], source_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_col in CONTINUOUS_SOURCE_COLS:
        info = cut_map.get(source_col)
        if info is None:
            continue
        non_missing = pd.to_numeric(source_df.get(source_col, pd.Series([], dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum()
        physical = PHYSICAL_THRESHOLDS.get(source_col, [])
        rows.append(
            {
                "feature_a": source_col,
                "physical_grid": format_threshold_list(physical),
                "data_driven_grid": format_threshold_list(info.data_driven_grid),
                "model_cutpoints_raw": format_threshold_list(info.model_cutpoints_raw),
                "model_cutpoints_std": format_threshold_list(info.model_cutpoints_std),
                "cutpoint_source": info.cutpoint_source,
                "alignment_note": alignment_note_for_thresholds(physical, info.model_cutpoints_raw),
                "n_non_missing": int(non_missing),
            }
        )
    return pd.DataFrame(rows)


def _intervals(thresholds: Sequence[float]) -> list[tuple[float | None, float | None]]:
    clean = sorted({float(v) for v in thresholds if np.isfinite(v)})
    bounds: list[float | None] = [None, *clean, None]
    return list(zip(bounds[:-1], bounds[1:]))


def _condition_from_bounds(source_col: str, low: float | None, high: float | None) -> str:
    parts: list[str] = []
    if low is not None:
        parts.append(f"{source_col} >= {fmt_num(low)}")
    if high is not None:
        parts.append(f"{source_col} < {fmt_num(high)}")
    return " and ".join(parts)


def _mask_from_bounds(series: pd.Series, low: float | None, high: float | None) -> pd.Series:
    cond = series.notna()
    if low is not None:
        cond &= series >= float(low)
    if high is not None:
        cond &= series < float(high)
    return cond


def _best_interval(
    source_df: pd.DataFrame,
    y: np.ndarray,
    source_col: str,
    thresholds: Sequence[float],
    *,
    min_support_n: int,
    min_support_pos: int,
) -> dict[str, Any] | None:
    if source_col not in source_df.columns:
        return None
    series = pd.to_numeric(source_df[source_col], errors="coerce")
    base_rate = float(np.mean(y)) if len(y) else np.nan
    if (not np.isfinite(base_rate)) or base_rate <= 0:
        return None

    best: dict[str, Any] | None = None
    for low, high in _intervals(thresholds):
        cond = _mask_from_bounds(series, low, high)
        support_n = int(cond.sum())
        if support_n < int(min_support_n):
            continue
        support_pos = int(y[cond.to_numpy()].sum())
        if support_pos < int(min_support_pos):
            continue
        support_rate = float(support_pos / support_n)
        enrichment = float(support_rate / base_rate)
        row = {
            "low": low,
            "high": high,
            "support_n": support_n,
            "support_pos": support_pos,
            "coverage": float(support_n / len(source_df)) if len(source_df) else np.nan,
            "enrichment": enrichment,
            "condition_text": _condition_from_bounds(source_col, low, high),
            "mask": cond,
        }
        if best is None:
            best = row
        else:
            key_new = (float(row["enrichment"]), int(row["support_n"]))
            key_old = (float(best["enrichment"]), int(best["support_n"]))
            if key_new > key_old:
                best = row
    return best


def build_main_effect_rules(
    *,
    model_name: str,
    branch_name: str,
    model: Any,
    feature_names: Sequence[str],
    source_df: pd.DataFrame,
    y: np.ndarray,
    cut_map: dict[str, CutpointInfo],
    min_support_n: int,
    min_support_pos: int,
    min_enrichment: float,
    max_rules_per_feature: int,
    require_model_cutpoints: bool,
    allow_physical_fallback: bool,
) -> pd.DataFrame:
    importance = model_feature_importance_series(model, feature_names)
    rows: list[dict[str, Any]] = []

    for source_col in CONTINUOUS_SOURCE_COLS:
        if source_col not in source_df.columns:
            continue
        info = cut_map.get(source_col)
        model_thresholds = list(info.model_cutpoints_raw) if info else []
        if model_thresholds:
            thresholds = model_thresholds
            source_name = str(info.cutpoint_source)
        elif allow_physical_fallback:
            thresholds = list(PHYSICAL_THRESHOLDS.get(source_col, []))
            source_name = "physical_grid_fallback"
        elif require_model_cutpoints:
            continue
        else:
            thresholds = list(PHYSICAL_THRESHOLDS.get(source_col, []))
            source_name = "physical_grid"

        if not thresholds:
            continue

        candidate_rows: list[dict[str, Any]] = []
        series = pd.to_numeric(source_df[source_col], errors="coerce")
        base_rate = float(np.mean(y)) if len(y) else np.nan
        if (not np.isfinite(base_rate)) or base_rate <= 0:
            continue

        for low, high in _intervals(thresholds):
            cond = _mask_from_bounds(series, low, high)
            support_n = int(cond.sum())
            if support_n < int(min_support_n):
                continue
            support_pos = int(y[cond.to_numpy()].sum())
            if support_pos < int(min_support_pos):
                continue
            enrichment = float((support_pos / support_n) / base_rate)
            if enrichment < float(min_enrichment):
                continue
            candidate_rows.append(
                {
                    "rule_id": f"{model_name}::{branch_name}::main::{source_col}::{_condition_from_bounds(source_col, low, high)}",
                    "model": model_name,
                    "branch": branch_name,
                    "rule_family": "main_effect",
                    "feature_a": source_col,
                    "feature_b": "",
                    "condition_text": _condition_from_bounds(source_col, low, high),
                    "support_n": support_n,
                    "support_pos": support_pos,
                    "coverage": float(support_n / len(source_df)) if len(source_df) else np.nan,
                    "enrichment": enrichment,
                    "model_score": float(abs(importance.get(f"cont::{source_col}", 0.0))),
                    "rule_source": f"{model_name}_main_effect",
                    "cutpoint_source": source_name,
                }
            )

        candidate_rows = sorted(candidate_rows, key=lambda r: (r["enrichment"], r["support_n"]), reverse=True)[: int(max_rules_per_feature)]
        rows.extend(candidate_rows)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "rule_id",
                "model",
                "branch",
                "rule_family",
                "feature_a",
                "feature_b",
                "condition_text",
                "support_n",
                "support_pos",
                "coverage",
                "enrichment",
                "model_score",
                "rule_source",
                "cutpoint_source",
            ]
        )
    out = out.sort_values(["model_score", "enrichment", "support_n"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def _parse_ebm_pair_terms(model: Any, feature_names: Sequence[str]) -> list[tuple[str, str, float]]:
    feature_list = [str(v) for v in list(feature_names)]

    def _resolve_ebm_term_part(part: str) -> str:
        text = str(part).strip()
        if text.startswith("feature_"):
            idx_text = text.split("_", 1)[1] if "_" in text else ""
            if idx_text.isdigit():
                idx = int(idx_text)
                if 0 <= idx < len(feature_list):
                    return str(feature_list[idx])
        return text

    names = [str(v) for v in list(getattr(model, "term_names_", []))]
    scores = np.asarray(getattr(model, "term_importances", lambda: [])())
    rows: list[tuple[str, str, float]] = []
    if len(names) == 0 or scores.size == 0:
        return rows
    for name, score in zip(names, scores.reshape(-1)):
        text = str(name)
        parts: list[str] = []
        if " & " in text:
            parts = [p.strip() for p in text.split(" & ")]
        elif " x " in text:
            parts = [p.strip() for p in text.split(" x ")]
        if len(parts) != 2:
            continue
        part_a = _resolve_ebm_term_part(parts[0])
        part_b = _resolve_ebm_term_part(parts[1])
        if not part_a.startswith("cont::") or not part_b.startswith("cont::"):
            continue
        src_a = part_a.split("::", 1)[1]
        src_b = part_b.split("::", 1)[1]
        if src_a not in CONTINUOUS_SOURCE_COLS or src_b not in CONTINUOUS_SOURCE_COLS:
            continue
        rows.append((src_a, src_b, float(abs(score))))
    return rows


def _pair_rule_from_best_intervals(
    *,
    model_name: str,
    branch_name: str,
    source_df: pd.DataFrame,
    y: np.ndarray,
    source_a: str,
    source_b: str,
    score: float,
    cut_map: dict[str, CutpointInfo],
    min_support_n: int,
    min_support_pos: int,
    min_enrichment: float,
    require_model_cutpoints: bool,
    allow_physical_fallback: bool,
    rule_source: str,
) -> dict[str, Any] | None:
    info_a = cut_map.get(source_a)
    info_b = cut_map.get(source_b)
    thr_a = list(info_a.model_cutpoints_raw) if info_a else []
    thr_b = list(info_b.model_cutpoints_raw) if info_b else []
    source_name_a = str(info_a.cutpoint_source) if info_a else "missing_model_cutpoints"
    source_name_b = str(info_b.cutpoint_source) if info_b else "missing_model_cutpoints"

    if (not thr_a or not thr_b) and not allow_physical_fallback and require_model_cutpoints:
        return None
    if not thr_a:
        thr_a = list(PHYSICAL_THRESHOLDS.get(source_a, []))
        source_name_a = "physical_grid_fallback"
    if not thr_b:
        thr_b = list(PHYSICAL_THRESHOLDS.get(source_b, []))
        source_name_b = "physical_grid_fallback"

    best_a = _best_interval(source_df, y, source_a, thr_a, min_support_n=min_support_n, min_support_pos=min_support_pos)
    best_b = _best_interval(source_df, y, source_b, thr_b, min_support_n=min_support_n, min_support_pos=min_support_pos)
    if best_a is None or best_b is None:
        return None

    cond = best_a["mask"] & best_b["mask"]
    support_n = int(cond.sum())
    if support_n < int(min_support_n):
        return None
    support_pos = int(y[cond.to_numpy()].sum())
    if support_pos < int(min_support_pos):
        return None
    base_rate = float(np.mean(y)) if len(y) else np.nan
    if (not np.isfinite(base_rate)) or base_rate <= 0:
        return None
    enrichment = float((support_pos / support_n) / base_rate)
    if enrichment < float(min_enrichment):
        return None

    condition_text = f"{best_a['condition_text']} and {best_b['condition_text']}"
    return {
        "rule_id": f"{model_name}::{branch_name}::pair::{source_a}__{source_b}",
        "model": model_name,
        "branch": branch_name,
        "rule_family": "interaction",
        "feature_a": source_a,
        "feature_b": source_b,
        "condition_text": condition_text,
        "support_n": support_n,
        "support_pos": support_pos,
        "coverage": float(support_n / len(source_df)) if len(source_df) else np.nan,
        "enrichment": enrichment,
        "model_score": float(score),
        "rule_source": rule_source,
        "cutpoint_source": f"{source_a}:{source_name_a}|{source_b}:{source_name_b}",
    }


def _pair_rule_from_one_sided_cutpoints(
    *,
    model_name: str,
    branch_name: str,
    source_df: pd.DataFrame,
    y: np.ndarray,
    source_a: str,
    source_b: str,
    score: float,
    cut_map: dict[str, CutpointInfo],
    min_support_n: int,
    min_support_pos: int,
    min_enrichment: float,
    require_model_cutpoints: bool,
    allow_physical_fallback: bool,
    rule_source: str,
) -> dict[str, Any] | None:
    info_a = cut_map.get(source_a)
    info_b = cut_map.get(source_b)
    thr_a = list(info_a.model_cutpoints_raw) if info_a else []
    thr_b = list(info_b.model_cutpoints_raw) if info_b else []
    source_name_a = str(info_a.cutpoint_source) if info_a else "missing_model_cutpoints"
    source_name_b = str(info_b.cutpoint_source) if info_b else "missing_model_cutpoints"

    if (not thr_a or not thr_b) and not allow_physical_fallback and require_model_cutpoints:
        return None
    if not thr_a:
        thr_a = list(PHYSICAL_THRESHOLDS.get(source_a, []))
        source_name_a = "physical_grid_fallback"
    if not thr_b:
        thr_b = list(PHYSICAL_THRESHOLDS.get(source_b, []))
        source_name_b = "physical_grid_fallback"
    if not thr_a or not thr_b:
        return None

    series_a = pd.to_numeric(source_df[source_a], errors="coerce") if source_a in source_df.columns else pd.Series(dtype=float)
    series_b = pd.to_numeric(source_df[source_b], errors="coerce") if source_b in source_df.columns else pd.Series(dtype=float)
    if series_a.empty or series_b.empty:
        return None

    base_rate = float(np.mean(y)) if len(y) else np.nan
    if (not np.isfinite(base_rate)) or base_rate <= 0:
        return None

    def _one_sided_conditions(series: pd.Series, source_col: str, thresholds: Sequence[float]) -> list[tuple[pd.Series, str]]:
        out: list[tuple[pd.Series, str]] = []
        clean = sorted({float(v) for v in thresholds if np.isfinite(v)})
        for t in clean:
            cond_lt = series.notna() & (series < float(t))
            cond_ge = series.notna() & (series >= float(t))
            out.append((cond_lt, f"{source_col} < {fmt_num(float(t))}"))
            out.append((cond_ge, f"{source_col} >= {fmt_num(float(t))}"))
        return out

    conds_a = _one_sided_conditions(series_a, source_a, thr_a)
    conds_b = _one_sided_conditions(series_b, source_b, thr_b)
    if not conds_a or not conds_b:
        return None

    best: dict[str, Any] | None = None
    for mask_a, text_a in conds_a:
        if int(mask_a.sum()) < int(min_support_n):
            continue
        for mask_b, text_b in conds_b:
            cond = mask_a & mask_b
            support_n = int(cond.sum())
            if support_n < int(min_support_n):
                continue
            support_pos = int(y[cond.to_numpy()].sum())
            if support_pos < int(min_support_pos):
                continue
            enrichment = float((support_pos / support_n) / base_rate)
            if enrichment < float(min_enrichment):
                continue
            row = {
                "rule_id": f"{model_name}::{branch_name}::pair::{source_a}__{source_b}",
                "model": model_name,
                "branch": branch_name,
                "rule_family": "interaction",
                "feature_a": source_a,
                "feature_b": source_b,
                "condition_text": f"{text_a} and {text_b}",
                "support_n": support_n,
                "support_pos": support_pos,
                "coverage": float(support_n / len(source_df)) if len(source_df) else np.nan,
                "enrichment": enrichment,
                "model_score": float(score),
                "rule_source": rule_source,
                "cutpoint_source": f"{source_a}:{source_name_a}|{source_b}:{source_name_b}",
            }
            if best is None:
                best = row
                continue
            new_key = (float(row["enrichment"]), int(row["support_n"]))
            old_key = (float(best["enrichment"]), int(best["support_n"]))
            if new_key > old_key:
                best = row
    return best


def build_ebm_interaction_rules(
    *,
    model: Any,
    model_name: str,
    branch_name: str,
    feature_names: Sequence[str],
    source_df: pd.DataFrame,
    y: np.ndarray,
    cut_map: dict[str, CutpointInfo],
    pair_limit: int,
    min_support_n: int,
    min_support_pos: int,
    min_enrichment: float,
    require_model_cutpoints: bool,
    allow_physical_fallback: bool,
) -> pd.DataFrame:
    pair_terms = _parse_ebm_pair_terms(model, feature_names)
    if not pair_terms:
        return pd.DataFrame()

    dedup: dict[tuple[str, str], float] = {}
    for a, b, score in pair_terms:
        key = tuple(sorted([a, b]))
        dedup[key] = max(float(score), float(dedup.get(key, 0.0)))

    ordered = sorted([(k[0], k[1], v) for k, v in dedup.items()], key=lambda x: x[2], reverse=True)[: int(pair_limit)]
    rows: list[dict[str, Any]] = []
    for src_a, src_b, score in ordered:
        row = _pair_rule_from_one_sided_cutpoints(
            model_name=model_name,
            branch_name=branch_name,
            source_df=source_df,
            y=y,
            source_a=src_a,
            source_b=src_b,
            score=score,
            cut_map=cut_map,
            min_support_n=min_support_n,
            min_support_pos=min_support_pos,
            min_enrichment=min_enrichment,
            require_model_cutpoints=require_model_cutpoints,
            allow_physical_fallback=allow_physical_fallback,
            rule_source="ebm_fij_term_importance_cutpoint_onesided",
        )
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model_score", "enrichment", "support_n"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def _pair_rule_from_joint_intervals(
    *,
    model_name: str,
    branch_name: str,
    source_df: pd.DataFrame,
    y: np.ndarray,
    source_a: str,
    source_b: str,
    score: float,
    cut_map: dict[str, CutpointInfo],
    min_support_n: int,
    min_support_pos: int,
    min_enrichment: float,
    require_model_cutpoints: bool,
    allow_physical_fallback: bool,
    rule_source: str,
) -> dict[str, Any] | None:
    info_a = cut_map.get(source_a)
    info_b = cut_map.get(source_b)
    thr_a = list(info_a.model_cutpoints_raw) if info_a else []
    thr_b = list(info_b.model_cutpoints_raw) if info_b else []
    source_name_a = str(info_a.cutpoint_source) if info_a else "missing_model_cutpoints"
    source_name_b = str(info_b.cutpoint_source) if info_b else "missing_model_cutpoints"

    if (not thr_a or not thr_b) and not allow_physical_fallback and require_model_cutpoints:
        return None
    if not thr_a:
        thr_a = list(PHYSICAL_THRESHOLDS.get(source_a, []))
        source_name_a = "physical_grid_fallback"
    if not thr_b:
        thr_b = list(PHYSICAL_THRESHOLDS.get(source_b, []))
        source_name_b = "physical_grid_fallback"
    if not thr_a or not thr_b:
        return None

    base_rate = float(np.mean(y)) if len(y) else np.nan
    if (not np.isfinite(base_rate)) or base_rate <= 0:
        return None

    series_a = pd.to_numeric(source_df[source_a], errors="coerce")
    series_b = pd.to_numeric(source_df[source_b], errors="coerce")
    cand_a = [(low, high, _mask_from_bounds(series_a, low, high), _condition_from_bounds(source_a, low, high)) for low, high in _intervals(thr_a)]
    cand_b = [(low, high, _mask_from_bounds(series_b, low, high), _condition_from_bounds(source_b, low, high)) for low, high in _intervals(thr_b)]
    if not cand_a or not cand_b:
        return None

    best: dict[str, Any] | None = None
    for low_a, high_a, mask_a, text_a in cand_a:
        for low_b, high_b, mask_b, text_b in cand_b:
            cond = mask_a & mask_b
            support_n = int(cond.sum())
            if support_n < int(min_support_n):
                continue
            support_pos = int(y[cond.to_numpy()].sum())
            if support_pos < int(min_support_pos):
                continue
            enrichment = float((support_pos / support_n) / base_rate)
            if enrichment < float(min_enrichment):
                continue
            row = {
                "support_n": support_n,
                "support_pos": support_pos,
                "enrichment": enrichment,
                "text_a": text_a,
                "text_b": text_b,
            }
            if best is None:
                best = row
                continue
            key_new = (float(row["enrichment"]), int(row["support_n"]), int(row["support_pos"]))
            key_old = (float(best["enrichment"]), int(best["support_n"]), int(best["support_pos"]))
            if key_new > key_old:
                best = row

    if best is None:
        return None

    return {
        "rule_id": f"{model_name}::{branch_name}::pair::{source_a}__{source_b}",
        "model": model_name,
        "branch": branch_name,
        "rule_family": "interaction",
        "feature_a": source_a,
        "feature_b": source_b,
        "condition_text": f"{best['text_a']} and {best['text_b']}",
        "support_n": int(best["support_n"]),
        "support_pos": int(best["support_pos"]),
        "coverage": float(int(best["support_n"]) / len(source_df)) if len(source_df) else np.nan,
        "enrichment": float(best["enrichment"]),
        "model_score": float(score),
        "rule_source": rule_source,
        "cutpoint_source": f"{source_a}:{source_name_a}|{source_b}:{source_name_b}",
    }


def build_gbdt_interaction_rules(
    *,
    model: Any,
    model_name: str,
    branch_name: str,
    feature_names: Sequence[str],
    source_df: pd.DataFrame,
    y: np.ndarray,
    cut_map: dict[str, CutpointInfo],
    pair_limit: int,
    min_support_n: int,
    min_support_pos: int,
    min_enrichment: float,
    require_model_cutpoints: bool,
    allow_physical_fallback: bool,
    use_joint_interval_search: bool = False,
) -> pd.DataFrame:
    importance = model_feature_importance_series(model, feature_names)
    candidates: list[tuple[str, str, float]] = []
    for feat in feature_names:
        name = str(feat)
        if not name.startswith("cx::"):
            continue
        body = name.split("::", 1)[1]
        if "__" not in body:
            continue
        src_a, src_b = body.split("__", 1)
        if src_a not in CONTINUOUS_SOURCE_COLS or src_b not in CONTINUOUS_SOURCE_COLS:
            continue
        candidates.append((src_a, src_b, float(abs(importance.get(name, 0.0)))))

    candidates = sorted(candidates, key=lambda row: row[2], reverse=True)[: int(pair_limit)]
    rows: list[dict[str, Any]] = []
    for src_a, src_b, score in candidates:
        if bool(use_joint_interval_search):
            row = _pair_rule_from_joint_intervals(
                model_name=model_name,
                branch_name=branch_name,
                source_df=source_df,
                y=y,
                source_a=src_a,
                source_b=src_b,
                score=score,
                cut_map=cut_map,
                min_support_n=min_support_n,
                min_support_pos=min_support_pos,
                min_enrichment=min_enrichment,
                require_model_cutpoints=require_model_cutpoints,
                allow_physical_fallback=allow_physical_fallback,
                rule_source="gbdt_rule_extraction_from_pair_features_joint2d",
            )
        else:
            row = _pair_rule_from_best_intervals(
                model_name=model_name,
                branch_name=branch_name,
                source_df=source_df,
                y=y,
                source_a=src_a,
                source_b=src_b,
                score=score,
                cut_map=cut_map,
                min_support_n=min_support_n,
                min_support_pos=min_support_pos,
                min_enrichment=min_enrichment,
                require_model_cutpoints=require_model_cutpoints,
                allow_physical_fallback=allow_physical_fallback,
                rule_source="gbdt_rule_extraction_from_pair_features",
            )
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model_score", "enrichment", "support_n"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def _triple_rule_from_best_intervals(
    *,
    model_name: str,
    branch_name: str,
    source_df: pd.DataFrame,
    y: np.ndarray,
    source_a: str,
    source_b: str,
    source_c: str,
    score: float,
    cut_map: dict[str, CutpointInfo],
    min_support_n: int,
    min_support_pos: int,
    min_enrichment: float,
    require_model_cutpoints: bool,
    allow_physical_fallback: bool,
) -> dict[str, Any] | None:
    def _resolve_thresholds(source_col: str) -> tuple[list[float], str]:
        info = cut_map.get(source_col)
        thresholds = list(info.model_cutpoints_raw) if info else []
        source_name = str(info.cutpoint_source) if info else "missing_model_cutpoints"
        if thresholds:
            return thresholds, source_name
        if allow_physical_fallback:
            return list(PHYSICAL_THRESHOLDS.get(source_col, [])), "physical_grid_fallback"
        if require_model_cutpoints:
            return [], source_name
        return list(PHYSICAL_THRESHOLDS.get(source_col, [])), "physical_grid"

    thr_a, source_name_a = _resolve_thresholds(source_a)
    thr_b, source_name_b = _resolve_thresholds(source_b)
    thr_c, source_name_c = _resolve_thresholds(source_c)
    if not thr_a or not thr_b or not thr_c:
        return None

    base_rate = float(np.mean(y)) if len(y) else np.nan
    if (not np.isfinite(base_rate)) or base_rate <= 0:
        return None

    series_a = pd.to_numeric(source_df[source_a], errors="coerce")
    series_b = pd.to_numeric(source_df[source_b], errors="coerce")
    series_c = pd.to_numeric(source_df[source_c], errors="coerce")

    cand_a = [(low, high, _mask_from_bounds(series_a, low, high), _condition_from_bounds(source_a, low, high)) for low, high in _intervals(thr_a)]
    cand_b = [(low, high, _mask_from_bounds(series_b, low, high), _condition_from_bounds(source_b, low, high)) for low, high in _intervals(thr_b)]
    cand_c = [(low, high, _mask_from_bounds(series_c, low, high), _condition_from_bounds(source_c, low, high)) for low, high in _intervals(thr_c)]
    if not cand_a or not cand_b or not cand_c:
        return None

    best: dict[str, Any] | None = None
    for low_a, high_a, mask_a, text_a in cand_a:
        for low_b, high_b, mask_b, text_b in cand_b:
            mask_ab = mask_a & mask_b
            if int(mask_ab.sum()) < int(min_support_n):
                continue
            for low_c, high_c, mask_c, text_c in cand_c:
                cond = mask_ab & mask_c
                support_n = int(cond.sum())
                if support_n < int(min_support_n):
                    continue
                support_pos = int(y[cond.to_numpy()].sum())
                if support_pos < int(min_support_pos):
                    continue
                enrichment = float((support_pos / support_n) / base_rate)
                if enrichment < float(min_enrichment):
                    continue
                row = {
                    "low_a": low_a,
                    "high_a": high_a,
                    "low_b": low_b,
                    "high_b": high_b,
                    "low_c": low_c,
                    "high_c": high_c,
                    "text_a": text_a,
                    "text_b": text_b,
                    "text_c": text_c,
                    "support_n": support_n,
                    "support_pos": support_pos,
                    "enrichment": enrichment,
                }
                if best is None:
                    best = row
                    continue
                key_new = (float(row["enrichment"]), int(row["support_n"]), int(row["support_pos"]))
                key_old = (float(best["enrichment"]), int(best["support_n"]), int(best["support_pos"]))
                if key_new > key_old:
                    best = row

    if best is None:
        return None

    condition_text = f"{best['text_a']} and {best['text_b']} and {best['text_c']}"
    return {
        "rule_id": f"{model_name}::{branch_name}::triple::{source_a}__{source_b}__{source_c}",
        "model": model_name,
        "branch": branch_name,
        "rule_family": "interaction_3way",
        "feature_a": source_a,
        "feature_b": source_b,
        "feature_c": source_c,
        "condition_text": condition_text,
        "support_n": int(best["support_n"]),
        "support_pos": int(best["support_pos"]),
        "coverage": float(int(best["support_n"]) / len(source_df)) if len(source_df) else np.nan,
        "enrichment": float(best["enrichment"]),
        "model_score": float(score),
        "rule_source": "gbdt_rule_extraction_from_triple_features",
        "cutpoint_source": (
            f"{source_a}:{source_name_a}|{source_b}:{source_name_b}|{source_c}:{source_name_c}"
        ),
    }


def build_gbdt_triple_rules(
    *,
    model: Any,
    model_name: str,
    branch_name: str,
    feature_names: Sequence[str],
    source_df: pd.DataFrame,
    y: np.ndarray,
    cut_map: dict[str, CutpointInfo],
    triple_limit: int,
    min_support_n: int,
    min_support_pos: int,
    min_enrichment: float,
    require_model_cutpoints: bool,
    allow_physical_fallback: bool,
) -> pd.DataFrame:
    importance = model_feature_importance_series(model, feature_names)
    candidates: list[tuple[str, str, str, float]] = []
    for feat in feature_names:
        name = str(feat)
        if not name.startswith("tx::"):
            continue
        body = name.split("::", 1)[1]
        parts = body.split("__")
        if len(parts) != 3:
            continue
        src_a, src_b, src_c = parts
        if src_a not in CONTINUOUS_SOURCE_COLS or src_b not in CONTINUOUS_SOURCE_COLS or src_c not in CONTINUOUS_SOURCE_COLS:
            continue
        candidates.append((src_a, src_b, src_c, float(abs(importance.get(name, 0.0)))))

    candidates = sorted(candidates, key=lambda row: row[3], reverse=True)[: int(triple_limit)]
    rows: list[dict[str, Any]] = []
    for src_a, src_b, src_c, score in candidates:
        row = _triple_rule_from_best_intervals(
            model_name=model_name,
            branch_name=branch_name,
            source_df=source_df,
            y=y,
            source_a=src_a,
            source_b=src_b,
            source_c=src_c,
            score=score,
            cut_map=cut_map,
            min_support_n=min_support_n,
            min_support_pos=min_support_pos,
            min_enrichment=min_enrichment,
            require_model_cutpoints=require_model_cutpoints,
            allow_physical_fallback=allow_physical_fallback,
        )
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model_score", "enrichment", "support_n"], ascending=[False, False, False]).reset_index(drop=True)
    return out
