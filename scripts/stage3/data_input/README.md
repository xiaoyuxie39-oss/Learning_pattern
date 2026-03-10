# Stage3 Data Input Scripts

This folder isolates Stage3 Part I input generation by feature mode.

## Scripts

1. `build_cont_only_input.py`
   Output: `interaction_feature_view_cont_only.csv`

2. `build_bin_only_input.py`
   Output: `interaction_feature_view_bin_only.csv`

3. `build_cont_plus_bin_input.py`
   Output: `interaction_feature_view_cont_plus_bin.csv`
   Compatibility alias (default): `interaction_feature_view.csv`

All scripts share the same cleaning, binning, and Gate C logic from `shared.py` and also write:

- `cleaning_report.csv`
- `cleaning_exceptions.csv`
- `gate_c_schema_report.csv`
- `gate_c_constant_feature_report.csv`
- `gate_c_missing_dominance_report.csv`
- `bin_health_report.csv`
- `gate_c_acceptance.json`
- `feature_mode_contract_<mode>.csv`
- `feature_view_build_summary_<mode>.json`

## Example

```bash
python3 scripts/stage3/data_input/build_cont_only_input.py --manifest Doc/methodology_stage3/nonlinear_mainrule_cont_only/run_manifest_suite_n1_template.yaml
python3 scripts/stage3/data_input/build_bin_only_input.py --manifest Doc/methodology_stage3/nonlinear_mainrule_cont_only/run_manifest_suite_n1_template.yaml
python3 scripts/stage3/data_input/build_cont_plus_bin_input.py --manifest Doc/methodology_stage3/nonlinear_mainrule_cont_only/run_manifest_suite_n1_template.yaml
```
