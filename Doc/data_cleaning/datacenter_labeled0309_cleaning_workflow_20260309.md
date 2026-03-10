# datacenter_labeled0309.csv 数据清洗流程

## 1. 目标与边界

本流程面向 [data/datacenter_labeled0309.csv](/Users/xiaoyu/VScode/Learning_pattern/data/datacenter_labeled0309.csv) 的设施级数据清洗与层级去重，目标是为后续 infra-only AI data center 识别提供一份：

- 基础设施字段标准化
- 层级关系可审计
- 不重复计数
- 对冲突信息保守处理
- 可回溯到原始 `id` 和原始记录

本流程当前不做最终 AI 归类重判，只做清洗、字段统一、层级解析、冲突标记和分析层去重。

## 2. 关键约束

- 基础设施优先于营销文本、公司 reputation 和泛 AI 描述。
- site-specific evidence 优先于 campus-level evidence，campus-level evidence 优先于 company-level evidence。
- 不静默覆盖值；每个解析动作都要留下规则痕迹。
- `campus` 与 `campus_facility` 不允许在同一分析层重复计数。
- 原始 `notes` 不进入最终 `clean_master` 和 `clean_model` 表，避免主表冗余；但原始追溯通过源 CSV、`id`、`urls` 与审计表保留。

## 3. 字段范围

### 3.1 身份与层级字段

- `id`
- `name`
- `company`
- `parent`
- `type`
- `level`
- `location`
- `year`
- `stage`

### 3.2 基础设施字段

- `power_mw`
- `rack_count`
- `rack_kw`
- `rack_density_kw_per_rack_dc`
- `rack_density_kw_per_rack_dc_note`
- `rack_density_area_w_per_sf_dc`
- `rack_density_area_w_per_sf_dc_note`
- `cooling`
- `liquid_cool`
- `pue`
- `building_sqm`
- `whitespace_sqm`

### 3.3 证据与注释字段

- `verdict`
- `reasons`
- `urls`
- `ai_terms`

`notes` 仅保留在源数据中，不写入清洗主产物。

## 4. 标准化规则

### 4.1 文本

- 去除首尾空格
- 合并重复空白
- 统一常见大小写和标点噪声
- 保留楼栋号、Phase、DC 编号等有辨识度的差异

### 4.2 空值

以下统一视为缺失：`""`、`null`、`None`、`N/A`、`NA`、`unknown`、`unspecified`、`0.0` 型 parent 占位值。

### 4.3 布尔与类别

- `liquid_cool` 统一为 `Y / N / <null>`
- `cooling` 收敛到：`air / water / evaporative / liquid / immersion / hybrid / mixed / unknown`
- `type`、`level` 保留原层级语义，但修正明显格式问题

### 4.4 数值

- `power_mw` 统一为 MW
- `rack_kw` 与 `rack_density_kw_per_rack_dc` 统一为 kW/rack
- `rack_density_area_w_per_sf_dc` 统一为 W/sq ft
- `building_sqm`、`whitespace_sqm` 统一为平方米
- `pue` 统一为无量纲数值

所有数值字段同时保留 `*_raw` 列，用于解析追溯。

### 4.5 位置

位置统一为 `city, state/province, country` 形式；仅做大小写、空白、明显拼写噪声清理，不虚构缺失地理信息。

## 5. 层级解析与去重策略

### 5.1 parent link 规范化

- `parent` 先做数值样式归一：如 `13494.0 -> 13494`
- `0`、`0.0` 视为无父节点
- 只有当 `parent_id` 能映射到现有 `id`，且 parent-child 关系可信时，才进入自动层级判断
- 对 `level=campus_facility` 但 `parent` 为空或仅为 `0/0.0` 的记录，默认按 standalone leaf 保守保留，不直接送入 review queue

可信 parent link 判定参考现有 [scripts/data_process_script/dedupe_campus_hierarchy.py](/Users/xiaoyu/VScode/Learning_pattern/scripts/data_process_script/dedupe_campus_hierarchy.py)：

- `company` 一致
- 且满足以下至少一条：
- `location` 完整一致
- `location` 的尾部行政区匹配
- `name_base` 归一后匹配

### 5.2 campus 与 campus_facility 的默认处理

默认 analytic unit 优先保留 `campus`，并尽量移除同组 `campus_facility`，但必须先通过组内一致性校验。

### 5.3 允许删除 campus_facility 的条件

若某个 `campus` 组满足以下条件，则：

- `campus` 保留在 `clean_master` 与 `clean_model`
- 组内 `campus_facility` 保留在 `clean_master` 审计视图中，但从 `clean_model` 抑制
- 对外分析时仅计 `campus`

组内一致性条件：

1. parent-child link 可信
2. child 没有显著违背 parent 总量逻辑
3. child 的类别值不与 parent 发生实质冲突
4. child 不是唯一更强、更新、且不可安全上卷的基础设施证据

### 5.4 “不违背 campus 逻辑”的判定

#### 可加总字段

对以下字段做总量一致性检查：

- `power_mw`
- `rack_count`
- `building_sqm`
- `whitespace_sqm`

规则：

- 若 parent 与 child 均有值，检查 `sum(child) <= parent * 1.10`
- 超过阈值则标记为组冲突，不自动删除 child

#### 非加总连续字段

对以下字段仅检查兼容，不做求和：

- `rack_kw`
- `pue`
- `rack_density_kw_per_rack_dc`
- `rack_density_area_w_per_sf_dc`

规则：

- parent 缺失而 child 一致时，可保守上卷到 parent
- parent 非缺失且 child 明显冲突时，不自动覆盖 parent

#### 类别字段

对以下字段做语义兼容检查：

- `cooling`
- `liquid_cool`

规则：

- `air` 可兼容 parent=`hybrid/mixed`
- `liquid`、`immersion`、`water` 与 parent=`air` 且无混合说明时，视为冲突
- `liquid_cool=Y` 与 parent=`N` 视为冲突，除非 parent=`mixed/hybrid`

### 5.5 child 对 parent 的保守回填

如果 parent 缺失关键基础设施字段，而 child 在组内提供一致且不冲突的值，则可将 child 值保守回填到 parent，并记录：

- `backfilled_from_child_fields`
- `merged_child_ids`
- `suppression_reason`

不允许把明显冲突或高不确定 child 值直接覆盖 parent。

### 5.6 必须保留 review 的情况

以下情况不自动删 child，而是写入 review queue：

- child 总量之和显著超过 parent
- parent 与 child 在 `cooling/liquid_cool` 上冲突
- child 是更细粒度且唯一包含关键 infra 值的记录，但无法安全上卷
- `parent` 指向不存在记录或非占位型 link 不可信
- 可能是 building/phase 的真实独立设施，而不是重复记录

## 6. exact duplicate 与 fuzzy duplicate

- exact duplicate：按规范化后的 `company + name + location` 自动识别，可直接合并
- fuzzy duplicate：只生成 review queue，不自动合并
- 层级去重优先于 fuzzy duplicate 合并

## 7. 冲突与 plausibility 检查

至少检查以下异常：

- `whitespace_sqm > building_sqm`
- `rack_kw * rack_count` 明显高于 `power_mw`
- child 总量显著超过 campus 总量
- 高密度但无液冷或混合冷却支撑
- 冷却描述互相矛盾

异常记录不直接删除，只打：

- `conflict_flag`
- `conflict_notes`
- `plausibility_flags`

## 8. 输出文件

本次清洗产物隔离写入 [artifacts/datacenter_labeled0309_cleaning_20260309](/Users/xiaoyu/VScode/Learning_pattern/artifacts/datacenter_labeled0309_cleaning_20260309)：

- `clean_master.csv`：全量清洗表，保留层级与审计字段，不含 `notes`
- `clean_model.csv`：分析层去重表，不含被 parent 抑制的 `campus_facility`
- `review_queue.csv`：人工复核队列，聚合层级冲突、模糊重复和异常
- `profile_summary.json`：清洗前后与异常统计

## 9. 审计字段

主表至少包含以下审计列：

- `dedup_group_id`
- `hierarchy_status`
- `parent_id_clean`
- `keep_for_master`
- `keep_for_model`
- `suppressed_by_parent`
- `suppression_reason`
- `merged_child_ids`
- `backfilled_from_child_fields`
- `conflict_flag`
- `conflict_notes`
- `plausibility_flags`

## 10. 执行顺序

1. 读取源 CSV
2. 文本、空值、布尔、类别、数值统一
3. 规范化 `parent` 并建立可信 parent-child 关系
4. 识别 exact duplicate
5. 对 `campus` 组执行一致性校验
6. 对可安全上卷的 child 信息回填到 campus
7. 对满足条件的 `campus_facility` 执行 model 层抑制
8. 运行 plausibility checks
9. 生成 `clean_master`、`clean_model`、`review_queue` 与 `profile_summary`
