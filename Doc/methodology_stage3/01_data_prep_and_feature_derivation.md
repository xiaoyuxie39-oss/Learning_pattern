# Stage3 - Part I: Data Prep And Feature Derivation

## 0. 目标与边界
- 本文仅定义数据清洗、特征派生与 Gate C 验收。
- 本文通过后，才可进入 Part II。
- 本文不包含模型训练、指标结论和上线决策。

## 1. 输入与版本约束
1. 输入数据路径来自 `run_manifest.yaml -> paths.input_csv`（默认值见总纲）。
2. Part I 输出目录来自 `run_manifest.yaml -> paths.part1_out_dir`。
3. Part I 日志目录来自 `run_manifest.yaml -> paths.log_dir`。
4. 固定 schema 版本：由 `run_manifest.yaml -> versions.schema_version` 指定，Part II 必须复用。
5. 固定阈值版本：默认采用本文 `r1_default_v1`；若覆盖，必须写入 `run_manifest.yaml -> versions.threshold_version`。
6. 标准脚本入口（可覆盖）：`execution.part1_entry`。
7. 输入列合约（缺一即失败）：
`power_mw`, `rack_kw_typical`, `pue`, `cooling`, `liquid_cool`, `building_sqm`, `rack_kw_peak`, `whitespace_sqm`, `rack_density_area_w_per_sf_dc`。

## 2. Step A: 原始清洗与标准化
1. 数值清洗：
- 将可解析文本转为数值（例如 `pue="<1.2"` 解析为 `1.2`，并记录 `coerce_from_ineq=true`）。
- 无法解析项进入 `{part1_out_dir}/cleaning_exceptions.csv`，不得静默丢弃。
- `power_mw`, `rack_kw_typical`, `pue`, `building_sqm`, `rack_kw_peak`, `whitespace_sqm`, `rack_density_area_w_per_sf_dc` 必须执行同一套数值清洗规则。
2. 类别清洗：
- `cooling` 归一化为 `cooling_norm`，允许值：
`air`, `water_based_air`, `hybrid_air_liquid`, `liquid_direct_or_loop`, `liquid_immersion`。
- `cooling in {"unknown", "", null}` 统一视为缺失。
- `liquid_cool` 归一化为 `liquid_cool_binary`：
`Y -> 1`, `N -> 0`, 其余值（含空）全部设为缺失并记录异常。
3. 缺失策略：
- 保留缺失状态，并为关键字段生成显式 missing 指示。
4. 输出：
- `{part1_out_dir}/cleaning_report.csv`
- `{part1_out_dir}/cleaning_exceptions.csv`

## 3. Step B: 工程特征视图生成与落盘（强制）
1. 必须落盘：`{part1_out_dir}/interaction_feature_view.csv`。
2. 默认分箱（`r1_default_v1`，可在 manifest 覆盖）：
- `power_mw_bin`: `[-inf,20)`, `[20,100)`, `[100,inf)`
- `rack_kw_typical_bin`: `[-inf,35)`, `[35,80)`, `[80,inf)`
- `pue_bin`: `[-inf,1.15)`, `[1.15,1.25)`, `[1.25,inf)`
- `building_sqm_bin`: `[-inf,12000)`, `[12000,40000)`, `[40000,inf)`
- `rack_kw_peak_bin`: `[-inf,100)`, `[100,300)`, `[300,inf)`
- `whitespace_sqm_bin`: `[-inf,3000)`, `[3000,20000)`, `[20000,inf)`
- `rack_density_area_w_per_sf_dc_bin`: `[-inf,150)`, `[150,300)`, `[300,inf)`
- `cooling_bin`: 直接复用标准化后的 `cooling_norm` 桶
- `liquid_cool_bin`: 直接复用标准化后的 `liquid_cool_binary` 桶
3. 必须包含派生列：
- `power_mw_bin`
- `rack_kw_typical_bin`
- `pue_bin`
- `cooling_bin`
- `liquid_cool_bin`
- `building_sqm_bin`
- `rack_kw_peak_bin`
- `whitespace_sqm_bin`
- `rack_density_area_w_per_sf_dc_bin`
- `cooling_norm`
- `liquid_cool_binary`
4. 必须包含对应缺失列：
- `power_mw_is_missing`
- `rack_kw_typical_is_missing`
- `pue_is_missing`
- `cooling_is_missing`
- `liquid_cool_is_missing`
- `building_sqm_is_missing`
- `rack_kw_peak_is_missing`
- `whitespace_sqm_is_missing`
- `rack_density_area_w_per_sf_dc_is_missing`
5. 必须原样保留所有数值输入列的清洗后连续值到 `interaction_feature_view.csv`，并生成对应 `*_is_missing`；不得只保留额外补充字段而遗漏六个主底座数值字段。至少包括：
- `power_mw`, `power_mw_is_missing`
- `rack_kw_typical`, `rack_kw_typical_is_missing`
- `pue`, `pue_is_missing`
- `building_sqm`, `building_sqm_is_missing`
- `rack_kw_peak`, `rack_kw_peak_is_missing`
- `whitespace_sqm`, `whitespace_sqm_is_missing`
- `rack_density_area_w_per_sf_dc`, `rack_density_area_w_per_sf_dc_is_missing`
6. `rack_kw_peak`, `whitespace_sqm`, `rack_density_area_w_per_sf_dc` 不得进入 `base_non_missing_count`，也不得改变 `coverage_tier` 定义。
7. 必须包含行级覆盖列：
- `base_non_missing_count`（六个主底座字段的非缺失计数）
- `coverage_tier`
8. `coverage_tier` 定义：
- `C0`: `base_non_missing_count < 2`（仅 triage，不进训练）
- `C1`: `base_non_missing_count in {2,3}`（低信息）
- `C2`: `base_non_missing_count in {4,5}`（主分析）
- `C3`: `base_non_missing_count = 6`（高信息）
9. 强约束：
- `{part1_out_dir}/interaction_feature_view.csv` 是后续交互与训练唯一输入表。
- 禁止在 Part II 直接读取原始主表补算字段。

## 4. Gate C: Schema + 质量验收（硬门禁）
`Gate C` 未通过时，流程必须中断。

### C1 派生列合约
1. 所有派生列、`*_is_missing`、`base_non_missing_count`、`coverage_tier` 必须存在。
2. 缺任一列：`gate_c_status=FAIL`。
3. 九个输入列对应的清洗列、bin 列与 `*_is_missing` 必须按规范写入 `interaction_feature_view.csv`；其中 `rack_kw_peak`, `whitespace_sqm`, `rack_density_area_w_per_sf_dc` 不改变六字段主底座与 `coverage_tier` 契约，但其信息不得遗漏。

### C2 常量特征拦截
1. 任一 `*_bin` 或 `*_is_missing` 单值占比 `>=99%`：
标记 `invalid_constant_feature=true`。
2. 默认阻断进入 Part II。
3. 如需放行，仅允许显式 `constant_feature_whitelist`。

### C3 缺失主导风险
1. 统计候选底座字段 `__MISSING__` 占比。
2. 占比 `>60%` 的字段默认 `triage_only`，不进默认交互底座。
3. 若默认底座有效字段数 `<4`，判定 `gate_c_status=FAIL`。

### C4 分箱质量
1. 任何 `*_bin` 出现小桶：样本 `<10` 或占比 `<1%`。
2. 小桶必须先合并或降级：
- 数值分箱默认并入相邻桶。
- 类别分箱默认并入 `__OTHER__`。
3. 存在未处理小桶：`gate_c_status=FAIL`。

### C5 行级覆盖门禁
1. `coverage_tier=C0` 的记录必须标记 `triage_only_row=true`，禁止进入训练。
2. 可训练样本定义为 `base_non_missing_count >= 2`。
3. 可训练样本占比 `<75%` 时，`gate_c_status=FAIL`。

## 5. Gate C 输出物（必需）
1. `interaction_feature_view.csv`
2. `cleaning_report.csv`
3. `cleaning_exceptions.csv`
4. `gate_c_schema_report.csv`
5. `gate_c_constant_feature_report.csv`
6. `gate_c_missing_dominance_report.csv`
7. `bin_health_report.csv`
8. `gate_c_acceptance.json`（PASS/FAIL + 失败原因）
以上文件均必须写入 `{part1_out_dir}/`，不得写入仓库根目录。

## 6. 失败处置
1. Gate C FAIL：停止执行。
2. 禁止 fallback。
3. 禁止进入交互构造、训练与评估。
