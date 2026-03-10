# Learning_pattern

## Current Scope

This repository currently keeps the active Stage3 workflow and its supporting
documentation in the main tree.

- Active source lives under `scripts/`
- Active normative docs live under `Doc/`
- Run outputs stay under `artifacts/`
- Workspace execution ledger stays under `process/`
- Historical plans, one-off manifests, archived wrappers, and old run materials
  are folded into `.history/`

## Current Mainline Layout

### Documentation

- `Doc/methodologyStage3.md`
  Main Stage3 playbook and path contract.
- `Doc/methodology_stage3/01_data_prep_and_feature_derivation.md`
  Part I specification.
- `Doc/methodology_stage3/02_model_execution_and_audit.md`
  Part II specification.
- `Doc/methodology_stage3/nonlinear_mainrule_cont_only/README.md`
  Independent cont_only nonlinear v2 notes.

### Stage3 Scripts

- `scripts/stage3/01_data_prep_and_feature_derivation.py`
  Stable Part I compatibility entrypoint.
- `scripts/stage3/02_model_execution_and_audit.py`
  Stable Part II orchestration entrypoint.
- `scripts/stage3/data_input/`
  Feature-view builders by mode (`cont_only`, `bin_only`, `cont_plus_bin`).
- `scripts/stage3/stage3_workflow_isolated/`
  Extracted step library used by the main Part II entrypoint.
- `scripts/stage3/nonlinear_mainrule_cont_only/`
  Independent nonlinear cont_only v2 pipeline.
- `scripts/stage3/bin_only_control/`
  Isolated bin-only control run helpers.
- `scripts/stage3/cont_plus_bin_extension/`
  Isolated cont+bin extension helpers.

### Workspace Files

- `process/stage3_execution_log.md`
  Shared execution ledger referenced by current manifests and Stage3 scripts.
- `artifacts/`
  Local run outputs and derived results. These are not the main source tree.

## Entry Points

```bash
python3 scripts/stage3/01_data_prep_and_feature_derivation.py --manifest <run_manifest.yaml>
python3 scripts/stage3/02_model_execution_and_audit.py --manifest <run_manifest.yaml>
python3 scripts/stage3/stage3_workflow_isolated/step00_pipeline_entry.py --manifest <run_manifest.yaml> --check-only
```

## Archive Policy

- Keep only reusable source code in `scripts/`
- Keep only long-lived normative docs and templates in `Doc/`
- Move dated summaries, stage plans, one-off manifests, and round-specific
  wrappers into `.history/`
- Do not treat `.history/` as the runtime source of active Stage3 code
