# Stage3 Ordered Workflow Helpers

This folder contains the extracted step modules used by the Stage3 Part II main
script: `scripts/stage3/02_model_execution_and_audit.py`.

It is a helper package, not a separate production pipeline.

## Ordered Modules

1. `step00_pipeline_entry.py` (compatibility entry)
2. `step01_features.py`
3. `step02_candidates.py`
4. `step03_models.py`
5. `step04_audits.py`
6. `step05_reporting.py`

## Run

```bash
python3 scripts/stage3/stage3_workflow_isolated/step00_pipeline_entry.py \
  --manifest Doc/methodology_stage3/0304/run_manifest_smoke_0304_v5.yaml
```

## Validate Order Contract Only

```bash
python3 scripts/stage3/stage3_workflow_isolated/step00_pipeline_entry.py \
  --manifest Doc/methodology_stage3/0304/run_manifest_smoke_0304_v5.yaml \
  --check-only
```

## Design

- `step01~step05` are the authoritative implementations for step-level logic.
- `02_model_execution_and_audit.py` remains the authoritative production entrypoint.
- `step00_pipeline_entry.py` only validates step exports, then dispatches to `02_model_execution_and_audit.py`.
