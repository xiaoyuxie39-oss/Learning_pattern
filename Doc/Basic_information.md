# AI Data Center Database (R1) — Basic Information

## 1. Overview & Scope
本文件描述 `data/AI_database_chrome__llm_final_R1.csv` 的基础信息、字段语义、可用边界与审计风险。它是 data card / schema notes，不是建模流水线说明。

本数据库定位为：基于公开证据搜查后，再经过 LLM 判断形成的 **AI-labeled data centers database**。  
研究对象是 **AI-oriented / accelerator-dense / high power-density compatible** 基础设施取向识别（infra-only 的可观察对象）。  
该短语指：在仅用基础设施可观测字段条件下，更可能支持高密度加速计算部署的设施特征取向（不等同于负载占比或利用率）。

## 2. R1 Snapshot and Pre-analysis Checklist
### 2.1 R1 快照（按当前 CSV）
1. 记录数：414（不含表头）。
2. 字段数：24。
3. 明确存在噪声：标签误差、营销措辞污染、证据粒度不一致。
4. 明确存在结构性缺失：跨来源、跨公司披露口径差异导致高缺失字段集中。

### 2.2 字段可用性分层（用于建模前优先级）

| Tier | 字段 | 非缺失率 | 角色 |
|---|---|---|---|
| **T1** (>=70%) | `power_mw` | 90.8% | 主交互底座 |
| | `cooling` | 79.2% | 主交互底座 |
| **T2** (40-70%) | `liquid_cool` | 65.0% | 主交互底座 |
| | `building_sqm` | 59.9% | 主交互底座 |
| | `rack_kw_typical` | 54.3% | 主交互底座 |
| | `pue` | 46.6% | 主交互底座（边缘） |
| **T3** (<40%) | `whitespace_sqm` | 40.3% | 低覆盖，仅 triage |
| | `rack_kw_peak` | 21.0% | 低覆盖，仅 triage |
| | `rack_density_area_w_per_sf_dc` | 15.5% | 低覆盖，仅 triage |
| | `rack_count` | 7.0% | 极低覆盖，仅参考 |

注：`whitespace_sqm` 处于 40% 附近边界，默认按低覆盖字段处理更稳健。

### 2.3 缺失非随机性
缺失来自系统性披露差异（公司策略、来源类型、生命周期、粒度），**不是随机丢失**。缺失模式本身可能成为模型 shortcut。

### 2.4 口径混用风险
`IT load vs facility load`、`planned vs operational`、`campus vs facility` 口径可同时混杂。横向比较容量、密度、能效前需先审计。

### 2.5 标签约束（执行前速览）
1. 正类口径：`g3_strict_v1`（high-confidence evidence positives）。
2. 范式：PU learning / ranking（严格正 vs 其余 unlabeled）。
3. 硬禁输入：`llm_*`、`accel_*`、`stage`、`type`、`level`、`year` 不进主模型。

## 3. Record Granularity and Identifiers
### 3.1 一行记录代表什么
R1 是混合粒度记录，不是单一“机房楼栋”口径；`level` 与 `parent` 的简要说明见下表。

| 字段 | 建议用途 | 主要局限 |
|---|---|---|
| `id` | 记录主键（当前文件内唯一） | 仅保证当前快照唯一，跨版本需做版本化映射。 |
| `name` | 人工核验、证据回溯 | 命名风格不统一，可能含营销名称或阶段名。 |
| `company` | 分组分析、company-holdout 切分 | 不宜作为主模型特征，易学习公司披露/规模 shortcut。 |
| `location` | 地理分层、人工排查 | 地名粒度不一致（城市/州/国家混杂）。 |
| `type` | 元信息、分层描述 | 披露口径混杂；方法学中为主模型禁用字段。 |
| `year` | 时间背景说明 | 缺失高（68.1%），且含规划年份（最高到 2030），不等同“已投运年份”；主模型禁用。 |
| `stage` | 元信息/项目阶段参考 | 离散值含 `-1/0/1/2/3` 与空值，跨来源定义不一致；主模型禁用。 |
| `level` | 记录粒度标识（site/campus/facility） | 混合粒度不可直接横向比较容量；主模型禁用。 |
| `parent` | 父级关系辅助字段 | 高缺失且 ID 来源混杂，只能作弱关联线索，不构成完整层级树。 |

方法学硬约束（R1_2 方案）与上述一致：主模型禁用 `stage`, `type`, `level`, `year`，并禁用 `llm_*` 与 `accel_*` 作为输入。

## 4. Field Dictionary (grouped by: Power / Racks / Density / Cooling / Space / Metadata)
### 4.1 Power
字段：`power_mw`  
物理含义：通常表示站点/园区可用或规划 IT 电力规模（单位 MW）。  
观测覆盖（R1）：90.8% 非缺失。典型量级在“几十到数百 MW”；中位数约 50 MW（该数值会随清洗迭代变化）。  
常见缺失原因：未披露容量、披露仅有总接入电力而非 IT 口径、园区与单楼口径混用。  
风险提示：存在极值/异常值，可能来自粒度、口径或单位混用，需在比较前审计。  
与该取向的关系（谨慎）：高功率可能是必要条件之一，但单字段不能直接判定该取向。

### 4.2 Racks
字段：`rack_count`, `rack_kw_typical`, `rack_kw_peak`  
物理含义：
1. `rack_count`：机架规模（数量）。
2. `rack_kw_typical`：典型机架功率密度（kW/rack，常态运行）。
3. `rack_kw_peak`：峰值/上限机架功率密度（kW/rack，瞬时或设计上限）。

观测覆盖（R1）：
1. `rack_count`：7.0% 非缺失，信息最稀疏，已披露样本多在 `10^2-10^3` 量级（中位数约 1000，随清洗迭代变化）。
2. `rack_kw_typical`：54.3% 非缺失，常见为数十 kW/rack（中位数约 50）。
3. `rack_kw_peak`：21.0% 非缺失，通常高于 `typical`（中位数约 200）。
4. `typical` vs `peak`：前者更接近日常运行，后者更接近设计上限，不宜混用。
5. 上述中位数仅用于量级示意，会随清洗迭代变化。

常见缺失原因：供应商只披露“可支持密度”不披露典型值、仅披露单个营销峰值、站点级与园区级混填。  
风险提示：该组字段存在口径混合与异常值，横向比较前需统一粒度与单位。  
与该取向的关系（谨慎）：机架功率密度与兼容性相关，但必须与电力、热管理和空间联合解释。

### 4.3 Density
字段：`rack_density_area_w_per_sf_dc`  
物理含义：面积归一化电力密度（字段名指向 W/ft²）。  
观测覆盖（R1）：15.5% 非缺失；典型量级约 `10^2` W/ft²（中位数约 165，随清洗迭代变化）。  
常见缺失原因：面积口径不统一（GFA vs whitespace）、单位换算未标准化、很多来源不直接披露该指标。  
风险提示：该字段缺失高且存在极值/异常值，提示定义或单位可能混杂；跨记录比较前必须先做口径审计。  
与该取向的关系（谨慎）：较高面积功率密度可作为线索，但不能单字段下结论。

### 4.4 Cooling / Thermal
字段：`cooling`, `liquid_cool`, `pue`  
物理含义：
1. `cooling`：冷却方案类别（如 `air`, `water_based_air`, `hybrid_air_liquid`, `liquid_direct_or_loop`, `liquid_immersion`）。
2. `liquid_cool`：液冷能力标记（`Y/N/空`）。
3. `pue`：能效指标（Power Usage Effectiveness，通常 >1）。

观测覆盖（R1）：
1. `cooling`：79.2% 非缺失，但仍有 `unknown/空`。
2. `liquid_cool`：65.0% 非缺失，`Y/N/空` 混合。
3. `pue`：46.6% 非缺失，典型值在 1.1-1.3 附近（中位数约 1.17，随清洗迭代变化）。

常见缺失原因：厂商只披露“支持液冷”而不披露具体技术、PUE 只披露设计值或最佳值、不同气候区不可直接横比。  
风险提示：设计值、宣传值与运营值可能混杂，且地区条件不同导致可比性受限。  
与该取向的关系（谨慎）：液冷与较低 PUE 往往与高密部署兼容，但不等同于该取向已被验证。

### 4.5 Space
字段：`building_sqm`, `whitespace_sqm`  
物理含义：
1. `building_sqm`：建筑总面积（常为总建面/GFA，单位 m²）。
2. `whitespace_sqm`：IT 可用白空间面积（单位 m²）。

观测覆盖（R1）：
1. `building_sqm`：59.9% 非缺失，常见量级约 `10^4` m²（中位数约 22,675，随清洗迭代变化）。
2. `whitespace_sqm`：40.3% 非缺失，常见量级约 `10^3-10^4` m²（中位数约 7,200）。

常见缺失原因：只披露土地/园区规模、不区分一期与总规划、白空间定义差异（含/不含机电辅助区）。  
风险提示：园区级与单体级面积混合时，空间字段会显著影响可比性。  
与该取向的关系（谨慎）：空间字段本身不代表该取向，应与功率、热管理与密度联合分析。

### 4.6 Metadata
字段：`id`, `name`, `company`, `location`, `type`, `year`, `stage`, `parent`, `level`  
用途：检索、分层、审计、人工复核、样本切分。  
限制：不应把元信息当“物理机制特征”；尤其 `type/year/stage/level` 在方法学中属于主模型硬禁用字段。

## 5. Labels, Evidence Fields, and Usage Constraints
### 5.1 标签字段（LLM 判断结果）
字段：`llm_ai_dc_label`, `llm_ai_dc_confidence`  
含义：
1. `llm_ai_dc_label`：LLM 基于证据文本给出的 AI 相关类别标签（R1 观测到 `ai_specific`, `ai_optimized`, `ai_capable_marketing`, `non_ai`, `ai_label`）。
2. `llm_ai_dc_confidence`：对应标签置信度分数（0.30 到 0.92）。

风险：该标签是“证据+模型判断”产物，不是地面真值，受提示词、来源文本质量和营销叙事影响。

### 5.2 证据增强字段（锚点候选）
字段：`accel_vendor`, `accel_model`, `accel_count`  
作用：用于人工核验、严格正类锚点构建、证据链补强。  
局限：披露偏向明显（`accel_count` 缺失 91.8%，`accel_model` 缺失 54.8%）。

### 5.3 使用约束（必须）
1. 主模型必须 `infra-only`，不得把 `llm_*` 与 `accel_*` 作为输入特征。
2. `stage/type/level/year` 同样不进入主模型。
3. `llm_*` 与 `accel_*` 可用于构造“高可信正锚点”或人工核验，不可用于泄露式提分。
4. 实践口径可理解为：严格正锚点（high-confidence evidence positives） vs 其余 unlabeled，采用 PU/ranking 视角；`g3_strict_v1` 是这类口径的角色描述。

## 6. Missingness & Noise: Known Issues and Risks
### 6.1 缺失最严重字段（R1 观测）
1. `rack_count`: 93.0% 缺失。
2. `accel_count`: 91.8% 缺失。
3. `parent`: 90.1% 缺失。
4. `rack_density_area_w_per_sf_dc`: 84.5% 缺失。
5. `rack_kw_peak`: 79.0% 缺失。
6. `year`: 68.1% 缺失。
7. `whitespace_sqm`: 59.7% 缺失。
8. `pue`: 53.1% 缺失。

### 6.2 为什么“缺失不是随机”
R1 缺失更可能来自系统性披露差异，而非随机丢失：
1. 公司披露策略不同（上市公司/大型平台更常披露能效与先进冷却）。
2. 来源类型不同（新闻稿、招商页、技术白皮书的信息粒度差异大）。
3. 生命周期不同（规划项目常有总功率，无白空间和机架实配）。
4. 粒度不同（campus 与 standalone_site 在容量口径上天然不对齐）。

### 6.3 shortcut 风险与审计必要性
缺失模式本身可能成为模型捷径（学到“谁会披露”，而非“谁更具该取向”）。  
因此，允许使用缺失指示变量时，必须同时进行缺失消融与审计联动（含 controlled missingness、negative control、分层稳定性），防止把披露模式误判为物理规律。

## 7. Why Multi-field Combinations and Interactions Are Considered (and Why Guardrails)
### 7.1 为什么需要组合而非单字段
单一字段通常不足以支持可解释、可外推判断。更接近物理机制的线索通常来自联合条件，例如：  
`高功率 + 高机架功率密度 + 液冷能力 + 合理空间约束`。

### 7.2 交互与多字段组合的定位（受控）
1. 交互用于机制假设的解释线索与 rulebook 总结，不是默认提分工具。  
2. 只有支持度门槛与稳定性/负控/受控缺失审计共同通过时，才可上升为可外推规则结论。  
3. 缺失模式不能被当作交互规律本体。

## 8. Practical Interpretation Guide (What You Can Conclude / Cannot Conclude)
### 8.1 建议的理解框架（原则）
1. 优先看“组合机制”而非单字段高低值。
2. 组合规律必须通过稳定性与负控/受控缺失审计。
3. 不把 `__MISSING__` 当作可迁移规律本体。
4. 不使用 `llm_*`/`accel_*` 作为主模型输入，避免标签与证据泄露。

### 8.2 两层输出口径
1. `Score`：用于排序与优先级，回答“先看谁”。
2. `Rulebook`：用于可审计解释，回答“为什么被排在前面”。

### 8.3 你可以与不可以得出的结论
可以：
1. 识别“更可能具备 AI-oriented / accelerator-dense / high power-density compatible 取向”的对象优先级。
2. 形成可追溯的解释线索并支持人工复核。

不可以：
1. 将该库直接解释为 AI 负载占比或 GPU 实际利用率数据库。
2. 将 LLM 标签视为无噪声真值。
3. 在缺少审计链的情况下，把交互或缺失模式直接宣称为可外推规律。
