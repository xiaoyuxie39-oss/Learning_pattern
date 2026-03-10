# Methodology Stage3 Playbook (Pre-Run Standard)

## 0. 文档用途
- 本文件是 Stage3 的执行前规范入口，不记录单次试运行复盘。
- 目标：形成可复跑、可审计、可追溯的标准执行流程。

## 1. 文档拆分与命名
1. `Doc/methodologyStage3.md`：总纲、路径合约、执行方式（本文件）。
2. `Doc/methodology_stage3/01_data_prep_and_feature_derivation.md`：数据清洗与派生规范。
3. `Doc/methodology_stage3/02_model_execution_and_audit.md`：模型执行与审计规范。

命名规则：
1. 流程文档使用两位序号前缀（`01`、`02`）保证执行顺序。
2. 产物文件名统一 `snake_case`，只允许小写字母、数字和下划线。
3. 规范文档只写“必须做什么”；运行结果只写入 run 目录。

## 2. 统一路径合约（必须）
1. 仓库根目录：`$REPO_ROOT`（执行时当前项目根目录）。
2. 固定默认输入数据：
`$REPO_ROOT/data/AI_database_chrome__llm_final_R1.csv`
3. 脚本标准入口路径（可由 manifest 覆盖）：
- `$REPO_ROOT/scripts/stage3/01_data_prep_and_feature_derivation.py`
- `$REPO_ROOT/scripts/stage3/02_model_execution_and_audit.py`
4. Stage3 运行产物根目录：
`$REPO_ROOT/artifacts/stage3/{run_id}/`
5. 运行内标准子目录：
- `manifest/`
- `logs/`
- `part1/`
- `part2/`
6. 工作区执行总账（跨 run 追加）：
`$REPO_ROOT/process/stage3_execution_log.md`

## 3. Run ID 与目录命名（必须）
1. `run_id` 格式：
`stage3_r1_YYYYMMDD_HHMMSS_{shortsha}`
2. 示例：
`stage3_r1_20260303_173500_a1b2c3d`
3. 禁止将时间戳拼接到每个输出文件名；时间维度只体现在 `run_id` 目录层。

## 4. run_manifest 规范（必须）
每次运行前必须创建：
`$REPO_ROOT/artifacts/stage3/{run_id}/manifest/run_manifest.yaml`

最小字段：
```yaml
run:
  stage: stage3
  run_id: stage3_r1_YYYYMMDD_HHMMSS_shortsha
  random_seed: 20260303

paths:
  input_csv: data/AI_database_chrome__llm_final_R1.csv
  run_root: artifacts/stage3/{run_id}
  log_dir: artifacts/stage3/{run_id}/logs
  part1_out_dir: artifacts/stage3/{run_id}/part1
  part2_out_dir: artifacts/stage3/{run_id}/part2

versions:
  schema_version: r1_schema_v1
  threshold_version: r1_default_v1

execution:
  part1_entry: scripts/stage3/01_data_prep_and_feature_derivation.py
  part2_entry: scripts/stage3/02_model_execution_and_audit.py
```

## 5. 标准执行方式（必须顺序）
1. 初始化 run 目录与 manifest。
2. 执行 Part I：
`python3 {part1_entry} --manifest {manifest_path}`
3. 校验 `{part1_out_dir}/gate_c_acceptance.json == PASS`。
4. 执行 Part II：
`python3 {part2_entry} --manifest {manifest_path}`
5. 任一 Gate 失败：立即停止，禁止 fallback、禁止跳步。

## 6. 输出与发布规范
1. Part I 输出仅写入 `part1_out_dir`，Part II 输出仅写入 `part2_out_dir`。
2. 禁止把运行产物写入 `Doc/`、`data/`、`.cache/`。
3. 运行结论必须落盘到：
`{part2_out_dir}/run_decision.md`
4. 可读总结必须落盘到：
`{part2_out_dir}/stage3_human_readable_results_interaction.md`
5. 本轮结论分析与改进方案必须落盘到：
`{part2_out_dir}/run_conclusion_analysis_and_improvement.md`
6. 每轮结束后必须同步更新工作区执行总账：
`$REPO_ROOT/process/stage3_execution_log.md`

## 7. 执行前检查清单
1. `input_csv` 路径存在且可读。
2. `schema_version`、`threshold_version` 已写入 manifest。
3. Part I 与 Part II 读取同一 `manifest_path`。
4. 输出目录已创建且为空（或明确允许覆写策略）。

## 8. 一句话原则
方法学文档负责流程与门禁；运行目录负责结果与复盘；manifest 负责参数与路径真值。

## 9. 收尾闭环（必须）
1. 每次运行结束必须形成“结论 + 原因 + 意义 + 下轮动作”四段式说明。
2. 说明内容一份写入本轮 `part2` 目录（便于 run 内审计），一份写入工作区执行总账（便于跨 run 对比）。
3. 若 `mainline_upgrade=false`，必须在说明中给出可执行改进项与下一轮验收标准。
