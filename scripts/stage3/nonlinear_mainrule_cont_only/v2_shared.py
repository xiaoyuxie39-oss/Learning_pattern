#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:  # pragma: no cover
    ExplainableBoostingClassifier = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE3_DIR = SCRIPT_DIR.parent
if str(STAGE3_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE3_DIR))

from common import ensure_dir, load_simple_yaml, resolve_repo_path  # noqa: E402

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
MISSING_INDICATOR_COLS = [
    "power_mw_is_missing",
    "rack_kw_typical_is_missing",
    "pue_is_missing",
    "cooling_is_missing",
    "liquid_cool_is_missing",
    "building_sqm_is_missing",
    "rack_kw_peak_is_missing",
    "whitespace_sqm_is_missing",
    "rack_density_area_w_per_sf_dc_is_missing",
    "base_non_missing_count",
]
DEFAULT_TOP_KS = [10, 20, 30]


@dataclass
class RunContext:
    repo_root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    part1_out_dir: Path
    part2_out_dir: Path
    out_root: Path
    run_id: str
    random_seed: int
    execution: dict[str, Any]


@dataclass
class BranchRunResult:
    metrics_by_split: pd.DataFrame
    metrics_ci: pd.DataFrame
    predictions_oof: pd.DataFrame
    full_model: Any
    scaling_stats: pd.DataFrame
    feature_columns: list[str]


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_context(manifest_arg: str | Path) -> RunContext:
    repo_root = SCRIPT_DIR.parents[2]
    manifest_path = resolve_repo_path(repo_root, manifest_arg)
    manifest = load_simple_yaml(manifest_path)
    paths = manifest.get("paths", {})
    run = manifest.get("run", {})
    execution = manifest.get("execution", {}) if isinstance(manifest.get("execution", {}), dict) else {}

    part1_out_dir = resolve_repo_path(repo_root, str(paths.get("part1_out_dir", "")))
    part2_out_dir = resolve_repo_path(repo_root, str(paths.get("part2_out_dir", "")))
    out_root = part2_out_dir / "nonlinear_cont_only_v2"
    ensure_dir(out_root)

    return RunContext(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manifest=manifest,
        part1_out_dir=part1_out_dir,
        part2_out_dir=part2_out_dir,
        out_root=out_root,
        run_id=str(run.get("run_id", "stage3_nonlinear_cont_only_v2")),
        random_seed=int(run.get("random_seed", 20260305)),
        execution=execution,
    )


def top_ks_from_execution(execution: dict[str, Any]) -> list[int]:
    raw = execution.get("top_ks", DEFAULT_TOP_KS)
    if isinstance(raw, list):
        values = [int(v) for v in raw if int(v) > 0]
    else:
        values = DEFAULT_TOP_KS
    return values or list(DEFAULT_TOP_KS)


def primary_k_from_execution(execution: dict[str, Any]) -> int:
    return int(execution.get("primary_k", 20))


def resolve_model_branch_override(
    execution: dict[str, Any],
    model_name: str,
    branch_name: str,
) -> dict[str, Any]:
    raw = execution.get("nonlinear_cont_only_model_branch_overrides", {})
    if not isinstance(raw, dict):
        return {}
    model_cfg = raw.get(str(model_name), {})
    if not isinstance(model_cfg, dict):
        return {}
    branch_cfg = model_cfg.get(str(branch_name), {})
    if not isinstance(branch_cfg, dict):
        return {}
    return dict(branch_cfg)


def strict_positive_label(df: pd.DataFrame) -> np.ndarray:
    label_ok = df["llm_ai_dc_label"].isin(["ai_specific", "ai_optimized"])
    accel_ok = df["accel_model"].notna() | df["accel_count"].notna()
    return (label_ok & accel_ok).astype(int).to_numpy(dtype=int)


def load_cont_only_training_frame(ctx: RunContext) -> tuple[pd.DataFrame, np.ndarray, Path]:
    cont_only_file = ctx.part1_out_dir / "interaction_feature_view_cont_only.csv"
    if not cont_only_file.exists():
        raise FileNotFoundError(
            f"Missing cont_only input view: {cont_only_file}. "
            "Please run scripts/stage3/data_input/build_cont_only_input.py first."
        )

    df = pd.read_csv(cont_only_file)
    required = {
        "company",
        "coverage_tier",
        "base_non_missing_count",
        "llm_ai_dc_label",
        "accel_model",
        "accel_count",
        *CONTINUOUS_SOURCE_COLS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"cont_only input missing required columns: {missing}")

    trainable = df[df["base_non_missing_count"] >= 2].copy().reset_index(drop=True)
    y = strict_positive_label(trainable)
    return trainable, y, cont_only_file


def feature_profile(X: pd.DataFrame) -> dict[str, Any]:
    cols = [str(c) for c in X.columns]
    n_cont = int(sum(c.startswith("cont::") for c in cols))
    n_pair = int(sum(c.startswith("cx::") for c in cols))
    n_missing = int(sum(c.endswith("_is_missing") for c in cols))
    return {
        "n_features_total": int(len(cols)),
        "n_cont": n_cont,
        "n_pair_features": n_pair,
        "n_missing_indicators": n_missing,
        "columns": cols,
    }


def log1p_nonnegative(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").clip(lower=0)
    return np.log1p(numeric)


def build_cont_only_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in MISSING_INDICATOR_COLS:
        if col not in df.columns:
            continue
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    for src in CONTINUOUS_SOURCE_COLS:
        if src not in df.columns:
            continue
        series = pd.to_numeric(df[src], errors="coerce")
        if src in LOG1P_CONTINUOUS_SOURCE_COLS:
            series = log1p_nonnegative(series)
        series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        out[f"cont::{src}"] = series

    return out.astype(float)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-8 or sy < 1e-8:
        return 0.0
    c = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def select_top_pair_features(
    base_X: pd.DataFrame,
    y: np.ndarray,
    *,
    max_pairs: int,
) -> list[tuple[str, str, float]]:
    cont_cols = [c for c in base_X.columns if str(c).startswith("cont::")]
    rows: list[tuple[str, str, float]] = []
    for col_a, col_b in combinations(cont_cols, 2):
        prod = (base_X[col_a] * base_X[col_b]).to_numpy(dtype=float)
        score = abs(_safe_corr(prod, y.astype(float)))
        rows.append((str(col_a).split("::", 1)[1], str(col_b).split("::", 1)[1], float(score)))
    rows = sorted(rows, key=lambda r: r[2], reverse=True)
    return rows[: max(0, int(max_pairs))]


def add_pair_features(
    base_X: pd.DataFrame,
    selected_pairs: list[tuple[str, str, float]],
) -> pd.DataFrame:
    out = base_X.copy()
    for src_a, src_b, _ in selected_pairs:
        col_a = f"cont::{src_a}"
        col_b = f"cont::{src_b}"
        if col_a not in out.columns or col_b not in out.columns:
            continue
        out[f"cx::{src_a}__{src_b}"] = (out[col_a] * out[col_b]).astype(float)
    return out


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return np.nan
    topk = min(int(k), len(y_true))
    idx = np.argsort(-scores)[:topk]
    return float(np.mean(y_true[idx]))


def enrichment_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    base_rate = float(np.mean(y_true)) if len(y_true) else np.nan
    if not np.isfinite(base_rate) or base_rate <= 0:
        return np.nan
    return float(precision_at_k(y_true, scores, k) / base_rate)


def summarize_metric_ci(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics_df.empty:
        return pd.DataFrame(columns=["metric", "k", "mean", "ci_low_95", "ci_high_95", "n_valid_splits"])
    for (metric, k), grp in metrics_df.groupby(["metric", "k"], dropna=False):
        vals = grp["value"].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            mean = low = high = np.nan
        else:
            mean = float(np.mean(vals))
            low = float(np.percentile(vals, 2.5))
            high = float(np.percentile(vals, 97.5))
        rows.append(
            {
                "metric": str(metric),
                "k": int(k),
                "mean": mean,
                "ci_low_95": low,
                "ci_high_95": high,
                "n_valid_splits": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def _fit_scaler(train_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = train_df.mean(axis=0)
    std = train_df.std(axis=0, ddof=0).replace(0.0, 1.0)
    std = std.where(std >= 1e-8, 1.0)
    return mean, std


def _scale_df(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    out = (df - mean) / std
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out.astype(float)


def _to_matrix(df: pd.DataFrame) -> np.ndarray:
    out = np.asarray(df.to_numpy(dtype=float), dtype=float)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -10.0, 10.0)


def build_model(
    model_name: str,
    branch_name: str,
    *,
    random_seed: int,
    pair_limit: int,
    model_params: dict[str, Any] | None = None,
) -> Any:
    params = dict(model_params or {})

    if model_name == "ebm":
        if ExplainableBoostingClassifier is None:
            raise RuntimeError("interpret package unavailable: cannot run EBM")
        interactions = int(
            params.get(
                "interactions",
                0 if branch_name == "mainline" else min(max(1, int(pair_limit)), 8),
            )
        )
        return ExplainableBoostingClassifier(
            interactions=interactions,
            max_bins=int(params.get("max_bins", 64)),
            max_rounds=int(params.get("max_rounds", 300)),
            learning_rate=float(params.get("learning_rate", 0.03)),
            outer_bags=int(params.get("outer_bags", 4)),
            inner_bags=int(params.get("inner_bags", 0)),
            n_jobs=int(params.get("n_jobs", 1)),
            random_state=random_seed,
        )

    if model_name == "gbdt":
        default_max_depth = 2 if branch_name == "mainline" else 3
        default_max_iter = 220 if branch_name == "mainline" else 320
        gbdt_kwargs: dict[str, Any] = {
            "max_depth": params.get("max_depth", default_max_depth),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "max_iter": int(params.get("max_iter", default_max_iter)),
            "l2_regularization": float(params.get("l2_regularization", 1.0)),
            "min_samples_leaf": int(params.get("min_samples_leaf", 10)),
            "random_state": random_seed,
        }
        if "max_leaf_nodes" in params:
            gbdt_kwargs["max_leaf_nodes"] = params["max_leaf_nodes"]
        if gbdt_kwargs["max_depth"] is not None:
            gbdt_kwargs["max_depth"] = int(gbdt_kwargs["max_depth"])
        return HistGradientBoostingClassifier(
            **gbdt_kwargs,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def predict_scores(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba.reshape(-1), dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -35.0, 35.0)))
    return np.asarray(model.predict(X), dtype=float)


def evaluate_branch_with_splits(
    *,
    model_name: str,
    branch_name: str,
    X: pd.DataFrame,
    raw_df: pd.DataFrame,
    y: np.ndarray,
    splits: list[dict[str, Any]],
    random_seed: int,
    pair_limit: int,
    top_ks: list[int],
    model_params: dict[str, Any] | None = None,
) -> BranchRunResult:
    metric_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    row_ids = raw_df["id"].astype(str) if "id" in raw_df.columns else pd.Series([str(i) for i in range(len(raw_df))])

    for split in splits:
        split_id = int(split["split_id"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)

        X_train_df = X.iloc[train_idx].copy()
        X_test_df = X.iloc[test_idx].copy()
        y_train = y[train_idx].astype(int)
        y_test = y[test_idx].astype(int)

        mean, std = _fit_scaler(X_train_df)
        X_train = _to_matrix(_scale_df(X_train_df, mean, std))
        X_test = _to_matrix(_scale_df(X_test_df, mean, std))

        model = build_model(
            model_name,
            branch_name,
            random_seed=random_seed + split_id,
            pair_limit=pair_limit,
            model_params=model_params,
        )
        model.fit(X_train, y_train)
        scores = predict_scores(model, X_test)

        for k in top_ks:
            metric_rows.append(
                {
                    "split_id": split_id,
                    "metric": "P",
                    "k": int(k),
                    "value": precision_at_k(y_test, scores, k),
                }
            )
            metric_rows.append(
                {
                    "split_id": split_id,
                    "metric": "Enrichment",
                    "k": int(k),
                    "value": enrichment_at_k(y_test, scores, k),
                }
            )

        for local_idx, global_idx in enumerate(test_idx):
            pred_rows.append(
                {
                    "split_id": split_id,
                    "row_idx": int(global_idx),
                    "row_id": str(row_ids.iloc[global_idx]),
                    "company": str(raw_df.iloc[global_idx].get("company", "")),
                    "coverage_tier": str(raw_df.iloc[global_idx].get("coverage_tier", "")),
                    "y_true": int(y_test[local_idx]),
                    "score": float(scores[local_idx]),
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_ci = summarize_metric_ci(metrics_df)
    pred_df = pd.DataFrame(pred_rows)

    mean_all, std_all = _fit_scaler(X)
    X_full = _to_matrix(_scale_df(X, mean_all, std_all))
    full_model = build_model(
        model_name,
        branch_name,
        random_seed=random_seed,
        pair_limit=pair_limit,
        model_params=model_params,
    )
    full_model.fit(X_full, y.astype(int))

    scaling_rows: list[dict[str, Any]] = []
    for col in X.columns:
        if not str(col).startswith("cont::"):
            continue
        source_col = str(col).split("::", 1)[1]
        scaling_rows.append(
            {
                "feature_name": str(col),
                "source_col": source_col,
                "mean": float(mean_all.get(col, 0.0)),
                "std": float(std_all.get(col, 1.0)),
            }
        )
    scaling_stats = pd.DataFrame(scaling_rows)

    return BranchRunResult(
        metrics_by_split=metrics_df,
        metrics_ci=metrics_ci,
        predictions_oof=pred_df,
        full_model=full_model,
        scaling_stats=scaling_stats,
        feature_columns=[str(c) for c in X.columns],
    )


def _split_store_dir(ctx: RunContext) -> Path:
    return ctx.out_root / "splits"


def build_company_splits(
    df: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    *,
    n_repeats: int,
    test_company_frac: float = 0.20,
    min_test_pos: int = 5,
    min_train_pos: int = 20,
    min_test_c2_n: int = 25,
    min_test_c3_n: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    rng = np.random.default_rng(seed)
    companies = df["company"].fillna("__UNKNOWN_COMPANY__").astype(str).unique()
    n_test_companies = max(1, int(round(len(companies) * float(test_company_frac))))

    candidate_splits: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(1000, int(n_repeats) * 60)

    company_series = df["company"].fillna("__UNKNOWN_COMPANY__").astype(str)
    tier_series = df["coverage_tier"].astype(str)
    seen_signatures: set[tuple[str, ...]] = set()

    while len(candidate_splits) < max(int(n_repeats) * 4, int(n_repeats)) and attempts < max_attempts:
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
        if train_pos < int(min_train_pos) or test_pos < int(min_test_pos):
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
    )[: int(n_repeats)]

    splits: list[dict[str, Any]] = []
    for idx, row in enumerate(selected):
        item = dict(row)
        item["split_id"] = idx
        item["repeat_id"] = idx
        splits.append(item)
    return splits, attempts


def save_splits_bundle(ctx: RunContext, df: pd.DataFrame, splits: list[dict[str, Any]], cfg: dict[str, Any]) -> None:
    split_dir = _split_store_dir(ctx)
    ensure_dir(split_dir)

    row_id_col = "id" if "id" in df.columns else None
    row_ids = df[row_id_col].astype(str).tolist() if row_id_col else [str(i) for i in range(len(df))]
    companies = df["company"].fillna("__UNKNOWN_COMPANY__").astype(str)

    long_rows: list[dict[str, Any]] = []
    split_meta_rows: list[dict[str, Any]] = []
    split_idx_rows: list[dict[str, Any]] = []

    for sp in splits:
        split_id = int(sp["split_id"])
        test_idx = np.asarray(sp["test_idx"], dtype=int)
        train_idx = np.asarray(sp["train_idx"], dtype=int)
        holdout = set(test_idx.tolist())

        for idx in range(len(df)):
            long_rows.append(
                {
                    "split_id": split_id,
                    "fold_id": 0,
                    "group_company": str(companies.iloc[idx]),
                    "is_holdout": int(idx in holdout),
                    "row_id": row_ids[idx],
                }
            )

        split_meta_rows.append(
            {
                "split_id": split_id,
                "repeat_id": int(sp.get("repeat_id", split_id)),
                "train_n": int(sp.get("train_n", len(train_idx))),
                "test_n": int(sp.get("test_n", len(test_idx))),
                "train_pos": int(sp.get("train_pos", 0)),
                "test_pos": int(sp.get("test_pos", 0)),
                "test_c2_n": int(sp.get("test_c2_n", 0)),
                "test_c3_n": int(sp.get("test_c3_n", 0)),
                "train_company_list": list(sp.get("train_companies", [])),
                "test_company_list": list(sp.get("test_companies", [])),
            }
        )

        split_idx_rows.append(
            {
                "split_id": split_id,
                "repeat_id": int(sp.get("repeat_id", split_id)),
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
            }
        )

    pd.DataFrame(long_rows).to_csv(split_dir / "company_holdout_splits.csv", index=False)
    write_json(
        split_dir / "company_holdout_splits_meta.json",
        {
            "group_key": "company",
            "n_splits": int(len(splits)),
            "seed": int(ctx.random_seed),
            "min_test_c2_n": int(cfg["min_test_c2_n"]),
            "min_test_c3_n": int(cfg["min_test_c3_n"]),
            "split_summaries": split_meta_rows,
        },
    )
    write_json(split_dir / "split_indices.json", split_idx_rows)


def load_splits_bundle(ctx: RunContext) -> list[dict[str, Any]]:
    split_file = _split_store_dir(ctx) / "split_indices.json"
    if not split_file.exists():
        return []
    raw = read_json(split_file)
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


def resolve_split_config(execution: dict[str, Any]) -> dict[str, int]:
    return {
        "n_repeats": int(execution.get("n_repeats", 100)),
        "min_valid_splits": int(execution.get("min_valid_splits", 80)),
        "min_test_c2_n": int(execution.get("min_test_c2_n", 25)),
        "min_test_c3_n": int(execution.get("min_test_c3_n", 20)),
    }


def load_or_create_splits(
    ctx: RunContext,
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    force_rebuild: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = resolve_split_config(ctx.execution)

    if not force_rebuild:
        existing = load_splits_bundle(ctx)
        if existing:
            return existing, cfg

    splits, attempts = build_company_splits(
        df,
        y,
        seed=ctx.random_seed,
        n_repeats=cfg["n_repeats"],
        min_test_c2_n=cfg["min_test_c2_n"],
        min_test_c3_n=cfg["min_test_c3_n"],
    )
    if len(splits) < int(cfg["min_valid_splits"]):
        raise RuntimeError(
            f"valid splits below threshold: got={len(splits)} "
            f"required={cfg['min_valid_splits']} attempts={attempts}"
        )

    save_splits_bundle(ctx, df, splits, cfg)
    cfg["sampling_attempts"] = int(attempts)
    return splits, cfg


def model_branch_dir(ctx: RunContext, model_name: str, branch_name: str) -> Path:
    return ctx.out_root / "models" / str(model_name) / "cont_only" / str(branch_name)


def extract_metric_mean(metrics_ci: pd.DataFrame, metric: str, k: int) -> float:
    if metrics_ci.empty:
        return np.nan
    subset = metrics_ci[(metrics_ci["metric"] == str(metric)) & (metrics_ci["k"] == int(k))]
    if subset.empty:
        return np.nan
    return float(subset.iloc[0]["mean"])
