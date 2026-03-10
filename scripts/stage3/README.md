# Stage3 Script Layout

`scripts/stage3` is organized around one production flow, one data-input module, and one helper package.

## Main Flow

1. `01_data_prep_and_feature_derivation.py`
   Stage3 Part I compatibility entrypoint. It now delegates to `data_input/build_cont_plus_bin_input.py` logic and keeps legacy output `interaction_feature_view.csv` for Part II.

2. `02_model_execution_and_audit.py`
   Stage3 Part II main entrypoint. It owns manifest loading, split generation, the `model x feature_mode x branch` execution matrix, sensitivity runs, run-level aggregation, and final artifact registration.

3. `common.py`
   Shared path, YAML, JSON, and logging helpers used by Part I and Part II.

## Data Input Module

`data_input/` isolates Part I feature-view generation by feature mode.

1. `data_input/shared.py`
   Shared cleaning, normalization, binning, Gate C checks, and output writers.

2. `data_input/build_cont_only_input.py`
   Generates `interaction_feature_view_cont_only.csv`.

3. `data_input/build_bin_only_input.py`
   Generates `interaction_feature_view_bin_only.csv`.

4. `data_input/build_cont_plus_bin_input.py`
   Generates `interaction_feature_view_cont_plus_bin.csv`, and by default also writes legacy alias `interaction_feature_view.csv`.

### Mode Script Usage

```bash
python3 scripts/stage3/data_input/build_cont_only_input.py --manifest <run_manifest.yaml>
python3 scripts/stage3/data_input/build_bin_only_input.py --manifest <run_manifest.yaml>
python3 scripts/stage3/data_input/build_cont_plus_bin_input.py --manifest <run_manifest.yaml>
```

## Nonlinear cont_only Isolated Pipeline

`nonlinear_mainrule_cont_only/` now has an independent `v2` execution path for nonlinear-only runs:

- `run_ebm_mainline_cont_only.py`
- `run_ebm_pairwise_cont_only.py`
- `run_gbdt_mainline_cont_only.py`
- `run_gbdt_pairwise_cont_only.py`
- `compare_delta_topk.py`
- `publish_realvalue_rules.py`
- `run_nonlinear_cont_only_suite_v2.py` (thin orchestrator)
- `run_3way_followup_cont_only.py` (GBDT 3way follow-up, cont_only only)

This `v2` path does not depend on `02_model_execution_and_audit.py`.

## Helper Package

`stage3_workflow_isolated/` is not a second production pipeline. It is the extracted step library that `02_model_execution_and_audit.py` imports.

1. `step01_features.py`
   Base feature assembly, feature-mode switches, continuous scaling, and feature summaries.

2. `step02_candidates.py`
   Pair/triple interaction candidate generation and threshold-based candidate classification.

3. `step03_models.py`
   Estimator construction, warning capture, branch evaluation, and fold metrics.

4. `step04_audits.py`
   Negative control, tier stability, and candidate consistency audits.

5. `step05_reporting.py`
   Rulebook construction, model-derived sensitivity outputs, linear continuous effects, and artifact completeness checks.

6. `step00_pipeline_entry.py`
   Compatibility entrypoint. It validates the exported step contracts, then delegates execution to `02_model_execution_and_audit.py`.

## Grid Policy For Bin Release

- `bin` 发布主口径仍应以 `physical_grid` 为主，保证跨批次可比与可复核。
- `data_driven_grid` 作为敏感性/对齐补充，写入 model-derived 对齐与诊断产物，不替代发布主阈值。

## Current Ownership Rule

- Add or modify pipeline orchestration in `02_model_execution_and_audit.py`.
- Add or modify step-specific logic in `stage3_workflow_isolated/step01~step05.py`.
- Add or modify Part I data input logic in `data_input/`.
- Do not reintroduce copied step implementations into `02_model_execution_and_audit.py`.
