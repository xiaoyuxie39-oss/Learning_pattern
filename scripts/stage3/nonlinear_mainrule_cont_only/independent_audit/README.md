# Independent Audit Layer For `cont_only` v2

This folder contains a self-contained post-hoc audit layer for:

- branch-level `tier2d + group OOS` checks
- rule-level group OOS checks for published interaction rules

Design rules:

- read existing `nonlinear_cont_only_v2` outputs only
- write to a separate audit root under `part2/`
- do not mutate or overwrite training / publish artifacts
- do not depend on `.history/` modules at runtime

Scripts:

- `run_branch_group_oos_audit.py`
  - audits branch-level `group OOS` and `tier2d`
  - default target: `gbdt/cont_only`
- `run_rule_group_oos_audit.py`
  - audits published pair / 3way interaction rules on held-out rows
  - default target: published `gbdt` interaction rules

Outputs:

- `{part2_out_dir}/nonlinear_cont_only_v2_independent_audit/branch_group_oos/...`
- `{part2_out_dir}/nonlinear_cont_only_v2_independent_audit/rule_group_oos/...`
