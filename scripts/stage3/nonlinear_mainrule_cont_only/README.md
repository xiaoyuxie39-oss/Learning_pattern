# Nonlinear Mainrule (cont_only) - v2 Independent Pipeline

This folder now provides an **independent cont_only nonlinear pipeline** for:

- `EBM` and `GBDT` only
- `mainline` and `mainline_plus_pairwise` branches only
- real-value continuous rule publishing only
- no dependency on `scripts/stage3/02_model_execution_and_audit.py`

## Design Goal

- one script, one responsibility
- model training only uses `interaction_feature_view_cont_only.csv`
- no linear-model execution or linear-model output
- interaction promotion uses Top-K deltas (`ΔP@K`, `ΔEnrichment@K`) as primary gate
- stability/audits are not hard gates in this v2 flow

## Scripts

1. `run_ebm_mainline_cont_only.py`
   Run EBM mainline on cont_only.

2. `run_ebm_pairwise_cont_only.py`
   Run EBM pairwise on cont_only.

3. `run_gbdt_mainline_cont_only.py`
   Run GBDT mainline on cont_only.

4. `run_gbdt_pairwise_cont_only.py`
   Run GBDT pairwise on cont_only.

5. `compare_delta_topk.py`
   Compute `pairwise - mainline` deltas for `P@K` and `Enrichment@K`.

6. `publish_realvalue_rules.py`
   Publish compact real-value rules and decision brief.

7. `run_nonlinear_cont_only_suite_v2.py`
   Thin orchestrator that only dispatches the six scripts above.

8. `run_gbdt_3way_cont_only.py`
   Run GBDT `mainline_plus_3way` on cont_only. Triple feature selection now uses
   `min_triple_coverage_rate` gate (default `0.30`), and default keeps only
   top-1 triple; top-2 is included only when it passes strong gate.

9. `compare_delta_3way_topk.py`
   Compare `3way - pairwise` deltas for `P@K` and `Enrichment@K`.

10. `publish_realvalue_rules_3way.py`
    Publish compact 3way real-value rules when 3way delta is positive.

11. `run_3way_followup_cont_only.py`
    Thin follow-up orchestrator for 3way (gate-aware, cont_only only).

## Base Entrypoints (Underlying Scripts)

```bash
python3 scripts/stage3/nonlinear_mainrule_cont_only/run_nonlinear_cont_only_suite_v2.py \
  --manifest Doc/methodology_stage3/nonlinear_mainrule_cont_only/run_manifest_suite_n1_template.yaml
```

Optional smoke mode:

```bash
python3 scripts/stage3/nonlinear_mainrule_cont_only/run_nonlinear_cont_only_suite_v2.py \
  --manifest Doc/methodology_stage3/nonlinear_mainrule_cont_only/run_manifest_suite_n1_template.yaml \
  --max-splits 5
```

## 3way Follow-up

```bash
python3 scripts/stage3/nonlinear_mainrule_cont_only/run_3way_followup_cont_only.py \
  --manifest Doc/methodology_stage3/nonlinear_mainrule_cont_only/run_manifest_suite_n1_template.yaml
```

3way selection config (optional, in manifest `execution.nonlinear_cont_only_3way`):

```yaml
min_triple_coverage_rate: 0.30
triple_limit: 2
second_triple_min_coverage_rate: 0.35
second_triple_min_support_n: 30
second_triple_min_support_pos: 3
```

## Key Outputs

Under:
`{paths.part2_out_dir}/nonlinear_cont_only_v2/`

- `models/ebm/cont_only/mainline/*`
- `models/ebm/cont_only/mainline_plus_pairwise/*`
- `models/gbdt/cont_only/mainline/*`
- `models/gbdt/cont_only/mainline_plus_pairwise/*`
- `delta_topk_summary.csv`
- `interaction_gain_decision.csv`
- `rules_publish_realvalue.csv`
- `run_decision_brief.md`
