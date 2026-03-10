# Nonlinear Mainrule (cont_only) 运行说明（已切换到 v2）

当前推荐入口已切换为独立的 `v2` 流程，不再使用旧 Part2 通用管线。

## 推荐入口

- `scripts/stage3/nonlinear_mainrule_cont_only/run_nonlinear_cont_only_suite_v2.py`

### 示例

```bash
python3 scripts/stage3/nonlinear_mainrule_cont_only/run_nonlinear_cont_only_suite_v2.py \
  --manifest Doc/methodology_stage3/nonlinear_mainrule_cont_only/run_manifest_suite_n1_template.yaml
```

## 模板

- `run_manifest_suite_n1_template.yaml`
- `run_manifest_suite_n2_template.yaml`（历史模板，v2 默认不需要）
