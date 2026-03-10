# Stage3 - Part II: Model Execution And Audit

## 0. 目标与硬冻结
- 本文定义 Gate C 通过后的交互构造、模型执行、审计链、主模型选择与产物规范。
- 硬冻结：`infra-only + PU/ranking + company holdout` 不得变更。
- 必做敏感性（不改变主结论）：
  - `+year` 敏感性（仅用于稳健性：`continuous year + year_is_missing` 不进入主模型）
  - missingness ablation（Tier-2+：with vs without missing indicators；用于 shortcut 风险判定）
  - legacy compatibility（输出与 R1_1 相同 schema 的 pair-rulebook 以便横向对齐）

## 1. Gate 前置条件与输入契约（必须）
1. 前置条件：`{part1_out_dir}/gate_c_acceptance.json` 必须为 `PASS`。
2. Part II 仅允许读取以下 Part I 产物：
- `{part1_out_dir}/interaction_feature_view.csv`
- `{part1_out_dir}/gate_c_acceptance.json`
3. `run_manifest.yaml` 中以下字段必须存在且可追溯：
- `paths.part1_out_dir`
- `paths.part2_out_dir`
- `paths.log_dir`
- `execution.part2_entry`
- `versions.schema_version`
- `versions.threshold_version`
4. 强制 schema 一致：Part II 字段列表、schema 版本、threshold 版本必须与 Part I 一致。
5. 训练样本仅允许 `base_non_missing_count >= 2`（`coverage_tier in {C1,C2,C3}`）。
6. `coverage_tier=C0` 仅允许 `triage/interpretation`，禁止进入训练。

## 2. 标签契约与泄露控制（必须）
1. 严格正锚点（`g3_strict_v1`）：
`llm_ai_dc_label in {ai_specific, ai_optimized} AND (accel_model 非缺失 OR accel_count 非缺失)`。
2. PU 映射：
- `strict_positive=1`：满足 `g3_strict_v1`
- `unlabeled=0`：其余全部样本
3. 泄露硬禁（不得进入任何主模型）：
- `llm_*`, `accel_*`, `stage`, `type`, `level`, `year`
4. 泄露字段仅可用于：标签构造、分层审计、人工复核。

## 3. 实验分支与共享切分（必须）
### 3.1 分支命名（统一）
1. `mainline`
2. `mainline_plus_pairwise`
3. `mainline_plus_gated_3way`（默认关闭，可由 manifest 显式开启）

### 3.1.1 feature_mode 维度（必须）
1. Part II 必须显式区分训练特征模式（`feature_mode`），不得从 rulebook 文本反推模型训练输入。
2. 允许的 `feature_mode`：
- `cont_only`：仅连续值 + missing indicators
- `bin_only`：仅 bin/one-hot + missing indicators
- `cont_plus_bin`：连续值 + bin/one-hot + missing indicators
3. `feature_mode` 是与 `branch` 正交的执行维度；执行矩阵应理解为：`model × feature_mode × branch`。
4. 相对 `mainline` 的 delta CI，必须在同一 `feature_mode` 内计算，不得跨 `feature_mode` 混算。
5. 所有输出目录、选择摘要与 run manifest 必须显式记录 `feature_mode`。
6. `run_manifest.json` 对特征摘要的标准字段名固定为 `feature_set_registry`，不得改用其他别名。
7. 默认执行策略（recommended）：
- 主矩阵默认仅运行 `cont_only + bin_only`；
- `cont_plus_bin` 仅作为桥接稳定性补跑模式；
- `cont_plus_bin` 默认仅对 `gbdt` 与 `logistic_l2` 开启（可由 manifest 的 `cont_plus_bin_model_subset` 覆盖）。

### 3.2 company holdout 共享切分（硬约束）
1. 三分支必须共享同一组 `company holdout splits` 与同一重复次数。
2. 分组字段固定 `company`，测试公司占比默认约 `20%`。
3. 默认 `n_repeats=100`（可由 manifest 覆盖）。
4. 每次切分必须满足：`test_strict_positive >= 5`、`train_strict_positive >= 20`、`test_C2_n >= 25`、`test_C3_n >= 20`，否则重采样。
5. 有效切分数 `<80`：本轮运行失败。
6. 切分必须落盘，至少包含：
- `{part2_out_dir}/splits/company_holdout_splits.csv`
- `{part2_out_dir}/splits/company_holdout_splits_meta.json`
7. 落盘内容至少包含：`split_id`, `random_seed`, `repeat_id`, `train_company_list`, `test_company_list`, `train_n`, `test_n`, `train_pos`, `test_pos`, `test_c2_n`, `test_c3_n`。

### 3.2.1 split 采样附注
1. shared company holdout splits 仅按 `3.2` 的硬约束生成与复用。
2. `min_test_c2_n`、`min_test_c3_n` 必须真正参与切分筛选（不允许“配置存在但采样未生效”）。
3. 不允许为平衡 tiers 改为 row-level 随机切分。

## 4. 交互构造与入模硬门槛（必须）
### 4.1 交互构造输入
1. 交互构造仅允许使用 `{part1_out_dir}/interaction_feature_view.csv`。
2. 禁止在 Part II 直接回读原始主表补算交互字段。

### 4.1.1 signal-group 映射（必须）
1. Part II 必须定义并可追溯 `signal_group_mapping`；可以写在 manifest 对应配置版本、或脚本常量中，但必须在运行时固定。
2. 推荐最小映射（示例）：
- `power`：`power_mw`, `power_mw_bin`
- `rack_power_density`：`rack_kw_typical`, `rack_kw_peak`, `rack_kw_typical_bin`, `rack_kw_peak_bin`, `rack_density_area_w_per_sf_dc`, `rack_density`
- `cooling`：`cooling_norm`, `liquid_cool_binary`, `liquid_immersion`, `water_based_air`, `hybrid_air_liquid`, `liquid_direct_or_loop`
- `efficiency`：`pue`, `pue_bin`
- `space`：`building_sqm_bin`, `building_sqm`, `whitespace_sqm`
3. 同一基础信号的连续值、bin 值、别名字段，必须映射到同一 `signal_group`；禁止通过换写法绕过约束。
4. 未映射字段默认不得进入 publishable interaction；仅可保留在 debug/triage，并标记 `downgrade_reason=unknown_signal_group`。

### 4.1.2 不同信号交互硬约束（必须）
1. `pair` 候选必须满足：`signal_group(feature_a) != signal_group(feature_b)`；否则不得进入 publishable candidate pool。
2. `triple` 候选必须满足：三个特征至少来自 3 个不同 `signal_group`；任意两维同组即不得进入 publishable candidate pool。
3. 同组字段（例如 `cooling_norm` 与 `liquid_cool_binary`）不得形成 publishable interaction；如保留，仅可进入 debug/triage，并标记 `downgrade_reason=same_signal_group`。
4. 解释模型产出的交互在回注主模型前，必须再次执行本约束；不得因 explain-model 路径绕过。
5. 发布态 rulebook（含 `rulebook_support.csv` / legacy 对齐态 / supplementary rulebook）禁止出现违反本约束的 `prediction` 行。

### 4.2 hierarchical 与数量上限
1. 三阶交互只能从“已通过稳定门槛的二阶交互”扩展（hierarchical）。
2. 交互必须区分“发现层”与“发布/注入层”：
- 发现层可保留更宽的候选池（当前建议：`pair <= 12`，`triple = 0`）用于 debug、unstable explanation 与非线性交互排序；
- 发布/注入层继续收紧（当前建议：`pair <= 4`，`triple = 0`）。
3. 超出上限的候选必须按门槛优先级排序截断并记录截断原因。

### 4.3 入模门槛与区间梯度（必须）
> 目标是优先得到“可复核机制规则”，允许一定波动，但必须通过审计链并给出敏感性区间解释。

#### 4.3.1 默认门槛（recommended）
1. 二阶候选进入 prediction，默认同时满足：
- `support_n_by_fold_min >= 20`
- `support_pos_by_fold_min >= 3`
- `selection_freq >= 0.80`
2. 三阶候选进入 prediction，默认同时满足：
- `support_n_by_fold_min >= 20`
- `support_pos_by_fold_min >= 3`
- `selection_freq >= 0.90`
- 来自已通过门槛的二阶项（hierarchical）
3. 任一条件不满足：自动降级 `triage`。

#### 4.3.2 区间梯度（threshold grid）策略（当默认门槛导致“无规则/无增益”时启用）
当出现以下任一情况，可启动区间梯度尝试，并把结果作为“敏感性区间”写入解释文档：
- prediction 规则数为 0（pair 或 triple 任一为 0）；
- 主指标（默认 K=20）提升证据不足且需要判断是否为门槛过严导致。

区间梯度必须满足：不改变硬冻结（infra-only + PU/ranking + company holdout），且每一次网格尝试都复用同一组共享 splits。

默认网格（不超过 2×2×2，避免过度搜索）：
- `support_n_by_fold_min ∈ {15, 20}`
- `support_pos_by_fold_min ∈ {2, 3}`
- `selection_freq_pair ∈ {0.70, 0.80}`
- `selection_freq_triple ∈ {0.80, 0.90}`

输出要求：
- 每个网格点生成独立产物目录（见第 9 节），并在 `model_selection_summary.csv` 中记录 `threshold_grid_id`。
- 解释要求：必须报告“规则数量—性能—审计链”三者的折中，并明确指出主结论采用的门槛点与理由。

### 4.4 三阶门控优先级（必须）
1. 优先级固定：`coverage gating (C2/C3) > hierarchy gating`。
2. `coverage gating`：仅在 `coverage_tier in {C2,C3}` 激活三阶候选。
3. `hierarchy gating`：仅在相关二阶条件成立时激活第三维。
4. 若覆盖门控不通过，不得因 hierarchy 通过而放行。

### 4.5 missing 主条件禁令
1. missing 相关主条件（`missing_flag_*`、`__MISSING__`）不得进入任何 rulebook 家族。
2. rulebook 家族至少包括：`rulebook_support.csv`、`rulebook_legacy_pair_tier2plus.csv`、`pair_rulebook_publishable_c3only.csv`、`pair_rulebook_explanation_unstable_c3only.csv`、`rulebook_model_derived_sensitivity.csv`。
3. missing 仅允许保留在敏感性/审计/ablation 产物中，例如 `interaction_ablation_missingflags_tier2plus.csv`、`interaction_sensitivity_year_summary.csv`、audit trace。
4. 若候选或 debug trace 中存在 missing 主条件：不得升级为 triage/publishable rulebook，必须停留在 sensitivity/debug 层，并记录 `downgrade_reason=missing_primary_condition`。

## 5. Model Zoo（多模型）执行规范

### 5.1 允许模型列表
1. `logistic_l2`
2. `logistic_elasticnet`
3. `ebm_main_plus_limited_interactions`
4. `gbdt_low_depth_regularized`（HistGB/XGBoost，强正则）

### 5.1.1 交互规则指标解释模型（Interaction Discovery / Explanation Models）
> 目的：用于“发现/解释交互机制规则”，不等同于主预测模型（Main Predictive Model）。允许使用不同模型来评估交互强度与可解释形状，但必须满足 infra-only 与审计链。

#### A) 两类模型角色（必须区分）
1. 主预测模型（Main Predictive Model）：用于输出最终排序分数（PU/ranking）与主结论。
2. 交互规则指标解释模型（Interaction Explanation Model）：用于识别与解释交互候选（pair/triple）的“机制性证据”，输出 rulebook 与解释材料。

两者可以相同，也可以不同；但主结论只能来自通过第 7 节选择链的主预测模型。

#### B) 允许的解释模型（recommended）
- `ebm_interaction_discovery`：EBM 主效应 + 限制交互（pair 优先），输出 shape、交互强度与可解释阈值。
- `gbdt_interaction_probe_low_depth`：低深度强正则 GBDT 用于交互探测（仅用于解释/筛选，不直接作为主结论）。
- `logistic_interaction_screen`：对候选 0/1 交互做稀疏筛选（L1/ElasticNet）用于稳定性统计。

#### C) 解释模型输出契约（必须）
每个解释模型必须在共享 company splits 上运行，并额外输出：
1. `{part2_out_dir}/models/{explain_model_name}/{feature_mode}/{branch_name}/interaction_explanation_summary.md`
2. `{part2_out_dir}/models/{explain_model_name}/{feature_mode}/{branch_name}/interaction_explanation_summary.en.md`
3. `{part2_out_dir}/models/{explain_model_name}/{feature_mode}/{branch_name}/interaction_probe_artifacts/`（可包含：shape 图、分裂阈值、交互强度表）

解释模型必须导出与第 9.2 节一致的 `rulebook_support.csv` 与 `rulebook_model_derived_sensitivity.csv`。

#### D) 解释→主模型回注（Interaction Injection）规则（必须）
1. 解释模型只负责“提出候选交互与其机制解释”，不得绕过第 4 节的门槛与第 6 节审计链。
2. 进入主预测模型的交互项，必须满足：
- 通过第 4.3 门槛（或 4.3.2 网格点的选择门槛）
- 通过第 6 节审计链（negative control / controlled missingness / tier2d-C2C3）
- 满足第 4.5 与第 5.3 的 missing 禁令（missing 仅可保留在敏感性/审计产物，不得进入任何 rulebook 家族）
3. 回注方式：
- 将入选交互写入单一运行级文件 `{part2_out_dir}/interaction_selected_from_explain_model.csv`。
- 该文件至少必须包含以下列：
  - `source_explain_model_name`
  - `target_model_name`
  - `feature_mode`
  - `target_branch_name`
  - `feature_a`
  - `feature_b`
  - `feature_c`（若无三阶可留空）
  - `support_n_by_fold_min`
  - `support_pos_by_fold_min`
  - `selection_freq`
  - `stability_freq`
  - `notes`
- 主预测模型在 `mainline_plus_pairwise` / `mainline_plus_gated_3way` 分支读取该文件时，必须按 `target_model_name == current_model_name`、`feature_mode == current_feature_mode`、`target_branch_name == current_branch_name` 精确过滤后再回注，不得跨 `feature_mode` 或跨模型串用候选。
4. 若解释模型提出的交互无法通过审计或门槛：必须降级为 triage，仅用于人工解释，不得入主模型。

### 5.1.2 两段式训练与发布（必须）
1. `fit` 阶段：
- 主线模型优先使用连续值特征进行训练；
- 交互规则发现仍可保留 bin-based pair/triple candidates；
- `EBM/GBDT` 优先使用连续交互，但必须限制自由度并施加强正则。
- `prepare_base_features()` 必须支持 `include_continuous / include_bins` 开关，用于显式生成 `cont_only / bin_only / cont_plus_bin` 三类特征模式；
- 每个 `model × feature_mode × branch` 必须记录实际输入特征摘要：`n_cont`, `n_bin_onehot`, `n_missing_indicators`, `X_columns_sha256`。
- 上述特征摘要必须统一写入 `run_manifest.json.feature_set_registry`；registry key 固定为 `{model_name}/{feature_mode}/{branch_name}`，且 value 内必须重复写出 `model_name`、`feature_mode`、`branch_name` 以避免并行执行时歧义。
2. `publish` 阶段：
- 发布层必须区分两个 rulebook 家族：
  - `rulebook_support.csv`：发布态主规则集；
  - `rulebook_model_derived_sensitivity.csv`：模型驱动切点/形状规则；
- 对 `nonlinear + cont_only + mainline`，`rulebook_support.csv` 必须以 `model_derived` 规则为主发布来源；工程阈值规则仅可作为对比集落盘到 `rulebook_support_engineered_comparison.csv`，不得覆盖主发布域决策；
- `rulebook_model_derived_sensitivity.csv` / `model_derived_cutpoint_alignment.csv` 默认仅对 `EBM/GBDT`（且实际输入包含连续特征）产出；线性模型不强制产出 model-derived 内容；
- `nonlinear + cont_only` 的 model-derived 规则过滤阈值应按数据规模自适应（可由 manifest 覆盖），至少包括：`min_support_n`、`min_support_pos`、`min_enrichment`、`max_rules_per_feature`；
- 连续交互形状或树分裂点进入发布层时，必须自动提炼为不超过 `3` 个阈值条件；
- `EBM/GBDT` 的 `mainline` 必须优先把主效应连续特征提炼为阈值规则；当前优先字段：`power_mw`、`rack_kw_typical`、`pue`、`building_sqm`、`rack_kw_peak`、`whitespace_sqm`；
- 交互分支的发布层必须优先把 `*_bin` 条件转换成对应连续区间表达；若无法稳定转换，才保留原 bin 文本；
- `rack_density_area_w_per_sf_dc` 允许进入连续阈值发布候选层，但不属于默认优先发布字段；仅在达到支持度/富集要求且未被同 `signal_group` 更强字段覆盖时进入发布态；
- 非线性交互（`EBM/GBDT`）在发布时必须优先参考拟合后的 interaction importance，对已入选交互做排序与注释；发布文本仍必须压缩为可复核阈值区间，而不是直接暴露原始 shape；
- 若无法稳定压缩为可复核阈值规则，则该交互仅可保留为 `triage/explanation_only`。
3. 连续阈值发布必须采用双层网格：
- 默认物理网格（physical grid）：跨来源一致、口径稳定，作为 `rulebook_support.csv` 的主发布基准；
- 数据驱动网格（data-driven grid，可选）：基于 `Tier-2+` 完整样本的分位点/等频切分，仅用于 `rulebook_model_derived_sensitivity.csv` 或 cutpoint 对齐输出；
- 必须额外输出 `model_derived_cutpoint_alignment.csv`，记录 `physical_grid / data_driven_grid / model_cutpoints / alignment_note`，其中 `alignment_note ∈ {consistent, approximate, conflict}`。
- 对 `EBM/GBDT` 从模型中提取的 cutpoints，发布前必须先从标准化空间反变换回原始值空间（含 `log1p` 特征的逆变换）；发布层 `condition_text` 与对齐比较均必须在原始值空间完成。
- `model_derived_cutpoint_alignment.csv` 必须同时保留标准化空间与原始值空间 cutpoints（例如 `model_cutpoints_std`、`model_cutpoints_raw`）用于审计追溯。
4. 不允许把“连续值拟合结果”直接以不可复核的原始 shape 替代发布态 rulebook。
5. 线性模型（`logistic_l2`、`logistic_elasticnet`）必须额外输出 `linear_continuous_effects.csv`：
- 从标准化后的 `cont::` 特征系数提取 `coef_std`；
- 输出 `odds_ratio_per_1sd = exp(coef_std)`；
- 结合训练时标准化的均值/标准差，在若干分位点上计算单特征 `partial_logit` 变化；
- 必须注明系数解释依赖标准化尺度，不得把不同尺度下的原始系数直接横向比较。
6. 线性模型在交互分支还必须输出 `linear_pairwise_effects.csv`（交互项系数/贡献，仅机制提示）；可选输出 `linear_vs_engineered_direction_check.csv` 作为方向一致性诊断（仅诊断，不参与 winner/发布域决策）。

### 5.1.3 按模型单独判定交互升级（必须）
1. 每个模型必须独立判断其 `mainline_plus_pairwise` / `mainline_plus_gated_3way` 是否相对同模型 `mainline` 构成升级。
2. 对某一模型的交互分支，只有同时满足以下条件，才允许作为该模型的“主线可升级分支”：
- `ΔP@20` 的 `95%CI` 下界 `>= 0`
- `ΔEnrichment@20` 的 `95%CI` 下界 `>= 0`
- `warning=0`
- 审计链 PASS
3. 必须显式区分：
- `rule_publish_gate`：规则级发布门槛，仅控制某条规则能否进入 `rulebook_support.csv` 或 `pair_rulebook_publishable_c3only.csv`；
- `branch_promotion_gate`：分支级升级门槛，仅控制 `pairwise/gated_3way` 是否允许成为该模型主线升级分支。
4. 若不满足上述条件：
- 该模型交互分支必须降级为 `explanation_only`
- 仍必须输出 `rulebook_support.csv` 与 `pair_rulebook_publishable_c3only.csv`
- 不得用某一模型的交互增益去为另一模型的主线升级背书
5. 主模型选择时，interaction upgrade 资格必须按模型独立计算，不得跨模型共享。
6. 线性模型（`logistic_l2`、`logistic_elasticnet`）的交互分支默认固定为 `explanation_only`；仅非线性模型允许进入交互升级判定。
7. 线性模型的 `mainline_plus_pairwise` 可训练用于机制提示，但必须禁止发布态 pair rulebook：
- `pair_rulebook_publishable_c3only.csv` 对线性交互分支必须为空（或全部降级为非发布态）；
- 线性交互相关规则仅可进入 explanation/sensitivity 层（例如 `pair_rulebook_explanation_unstable_c3only.csv`、`linear_pairwise_effects.csv`）。
8. 线性模型的 `mainline_plus_gated_3way` 必须实际 `skip`（`skipped_by_config=true`），不得进入训练与发布流程。

### 5.1.4 unstable explanation-only 交互补充输出（supplementary）
1. 当某模型的交互分支未通过 `5.1.3` 主线升级门槛时，允许额外输出 `pair_rulebook_explanation_unstable_c3only.csv` 作为补充解释层。
2. 该文件不得参与主模型选择、不得用于 `mainline_upgrade` 判定、不得替代 `rulebook_support.csv` 或 `pair_rulebook_publishable_c3only.csv`。
3. 允许 `CI` 证据不过线、允许 `tier2d/C2C3` FAIL，但仍必须满足：
- 主条件不得包含 `missing_flag_*` 或 `__MISSING__`
- 仅允许跨 `signal_group` 的 pair
- 最低支持度默认不低于：`support_n_by_fold_min >= 10` 且 `support_pos_by_fold_min >= 2`
- 最低规则富集默认不低于：`rule-level enrichment >= 1.2`
4. 文件必须新增或保留以下字段：
- `evidence_status = unstable_ci | tier_shift | both`
- `explanation_only = true`
- `why_unstable`（例如 `delta_ci_crosses_zero(...)`、`tier2d_fail(...)`）
5. 推荐限制数量上限（例如 `top 10`），并按 `enrichment -> stability_freq -> support_n_by_fold_min` 排序。
6. 本补充文件的定位是“保留弱但有机制意义的交互线索”，不得被表述为稳健主结论。

### 5.2 执行矩阵
1. 每个模型必须在可用 `feature_mode` 与分支上执行：`model × feature_mode × branch`。
2. `mainline_plus_gated_3way` 未开启时，必须在记录中标注 `skipped_by_config=true`。
3. 所有模型实验必须复用第 3 节共享切分，不得自行重采样。
4. 推荐默认运行节奏：
- Suite A：`cont_only` × 4 模型 × `[mainline, mainline_plus_pairwise]`
- Suite B：`bin_only` × 4 模型 × `[mainline, mainline_plus_pairwise]`
- Suite C：`Primary winner` 非线性模型在 `cont_only` 上做 `mainline_plus_gated_3way` follow-up
5. `cont_plus_bin` 不进入默认全矩阵；若需桥接补跑，默认仅运行 `gbdt + logistic_l2`（可由 manifest 覆盖）。
6. Suite D（`cont_plus_bin` bridge）触发条件应基于 `Primary winner` 与 `Control winner` 的机制分歧，而非仅以“winner 是否不同”做粗判；推荐至少包含：
- `Primary winner` vs `Control winner` 的模型类型或 top-pair 机制显著不一致（如 top-pair overlap 低且 CI 在边界区）；
- 或 `cont_only` 可发布 pair 不足而 `bin_only` 足够。
7. 主规则隔离专线（推荐）：
- Suite N1：`model_subset=[ebm, gbdt]`、`feature_modes=[cont_only]`、`branch_subset=[mainline, mainline_plus_pairwise]`；
- Suite N2：仅对 N1 的 `Primary winner` 运行 `mainline_plus_gated_3way`；
- 隔离模板与 wrapper 统一放在：
  - `Doc/methodology_stage3/nonlinear_mainrule_cont_only/`
  - `scripts/stage3/nonlinear_mainrule_cont_only/`

### 5.3 每个模型的强制输出
每个模型必须输出以下产物，否则该模型结果无效：（包含主预测模型与解释模型；解释模型还需满足 5.1.1 的额外产物契约）
1. OOS 指标与 95%CI（Top-K 指标）：`P@10/20/30/50`, `Enrichment@10/20/30/50`, `AUC proxy`
2. 相对 `mainline` 的 `delta CI`（至少含 `ΔP@20`, `ΔEnrichment@20`, `ΔAUC`）
3. `C3-slice` supplementary 指标与 95%CI（至少含 `P@20`, `Enrichment@20`, `AUC proxy`）
4. 审计链结果：`negative_control`, `controlled_missingness`, `tier2d/C2C3`
5. 可复核 rulebook：
- 若模型是非线性（如 EBM/GBDT），必须把解释结果提炼为“机制型可复核规则”（条件、支持度、稳定性、降级原因）
- 若无法导出可复核规则，该模型不得成为“主结论模型”
- rulebook 必须输出统一 schema（见 9.2），以支持跨模型横向对齐。
- missing 相关条件不得进入任何 rulebook 家族（`rulebook_support.csv`、`rulebook_legacy_pair_tier2plus.csv`、`pair_rulebook_publishable_c3only.csv`、`pair_rulebook_explanation_unstable_c3only.csv`、`rulebook_model_derived_sensitivity.csv`）；missing 仅允许保留在敏感性/审计/ablation 产物。
6. 线性模型交互分支的发布约束（必须）：
- `mainline_plus_pairwise` 可训练，但 `pair_rulebook_publishable_c3only.csv` 必须为空（或等价禁发状态）；
- `mainline_plus_gated_3way` 必须 `skipped_by_config=true`；
- 线性交互仅可保留在解释层（`pair_rulebook_explanation_unstable_c3only.csv`、`linear_pairwise_effects.csv`）。

### 5.4 解释性结论文档（必须）
1. 每个 `model × feature_mode × branch` 必须独立生成一份“结果意义与结论解释”文档：
- `{part2_out_dir}/models/{model_name}/{feature_mode}/{branch_name}/run_conclusion_analysis_and_improvement.md`

1. 文档至少包含：
- 关键指标（K=20 为主）与 95%CI；与 baseline 的 delta CI；
- 通过/未通过的审计项与原因（negative control / controlled missingness / tier2d-C2C3）；
- rulebook 的前 N 条（prediction 与 triage 分开）；
- 若启用了区间梯度（4.3.2），需给出敏感性区间总结。
1. 缺失任一文档，该实验结果视为无效（不得进入主模型选择）。

2. 工作区必须生成一个横向综合比较文档（覆盖所有模型与分支）：
- `{part2_out_dir}/model_zoo_comparison.md`

内容至少包含：统一表格对比（P@20/Enrichment@20/AUC proxy + CI）、审计通过情况、rulebook 可复核性状态、以及最终主模型选择说明。

## 6. 审计链（必须并联）
1. `controlled_missingness_parallel`
2. `negative_control`
3. `tier2d/C2C3` 分层稳定性
4. 候选-规则一致性审计（同一阈值、同一降级原因）
5. 任一硬审计 FAIL：
- 标记 `bias_risk_interaction=true`
- 该实验仅可保留为 `triage/interpretation`
- 禁止作为主线升级依据
6. `tier2d/C2C3` 的评估口径必须使用 `common_k`：
- 每个 split 先取 `common_k = min(top_k, C2_n, C3_n)`；
- 若 `common_k < common_k_min`（默认 15）或 `C2_n/C3_n` 不满足最小样本门槛（默认 `25/20`），该 split 不参与 tier2d 统计；
- 审计输出必须记录：`common_k_min`, `min_test_c2_n`, `min_test_c3_n`, `raw_diff_eps`, `n_splits_used`。
7. `tier2d/C2C3` 必须输出三段分级：
- `PASS`: `p95_abs_diff <= 0.20` 或 `sign_rate < 0.60`
- `WARN`: `0.20 < p95_abs_diff <= 0.30` 且 `sign_rate >= 0.60`
- `FAIL`: `p95_abs_diff > 0.30` 且 `sign_rate >= 0.60`
8. matched-control 必须以 `reduction_abs = abs(raw_diff) - abs(matched_diff)` 为主统计；`rate` 仅作补充并加 `raw_diff_eps` 防护（默认 `0.02`）。
9. `tier2d/C2C3` 属于域外推审计：
- 若 `tier2d_level=FAIL`，不自动否决主线模型，但发布域必须降至 `C3_only`；
- 若 `tier2d_level in {PASS, WARN}`，发布域可为 `C2C3`；
- 解释文档与 `run_decision.md` 必须显式写出 `tier2d_level` 与 `publish_scope`。

## 7. Main Model Selection（主模型选择，必须）
1. 候选集合：通过基础运行门槛的全部 `model × feature_mode × branch` 实验。
2. winner 分层必须显式区分，避免 `cont_only/bin_only` 与 `mainline/pairwise` 歧义：
- `Branch winner`：同一 `model + feature_mode` 内 `mainline vs pairwise` 的内部诊断胜者，仅用于诊断；
- `Primary winner`：仅从 `cont_only` 的非线性模型（`gbdt/ebm`）中选择；默认 `mainline`，仅当 `pairwise` 在稳定门槛下显著优于 `mainline` 才可替代；仅 `Primary winner` 可触发 `3way`；
- `Control winner`：仅在 `bin_only` 内选择，用于跨模型可比对照与 bridge 触发，不直接替代 `Primary winner`。
3. `Primary winner` 的排序规则：先按 company-holdout 的 `P@20 / Enrichment@20` CI 排序（先看 CI 下界，再看均值）。
4. 允许成为 `Primary winner` 的必要条件（必须同时满足）：
- CI 不明显跨负向（默认：`ΔP@20` 与 `ΔEnrichment@20` 的 95%CI 下界 `>= 0`；容差可由 manifest 配置）
- 硬审计全部 PASS
- 可导出机制型可复核规则（rulebook 完整）
5. 若 `tier2d/C2C3` 为 `FAIL` 但硬审计通过、warning=0、rulebook 可复核：
- 允许选择该模型作为域内主模型；
- `publish_scope` 必须写为 `C3_only`；
- 主文案不得暗示其自然外推到 Tier-2+ 全域。
6. 若 `tier2d/C2C3` 为 `WARN`：
- 允许主模型保留；
- `publish_scope` 允许写为 `C2C3`，但必须在 rulebook 中附带分 tier 证据（`support_C2/support_C3/enrichment_C2/enrichment_C3/delta_enrichment`）。
7. 若候选分支属于 `mainline_plus_pairwise` / `mainline_plus_gated_3way`，还必须满足 5.1.3 的“按模型单独判定交互升级”条件。
8. 若性能最高模型不满足任一必要条件：
- 降级为补充结果（supplementary）
- 不得作为主结论模型
- 在 `run_decision.md` 中写明降级原因

## 8. 数值稳定性与可复跑（必须）
1. logistic/优化器必须使用数值稳定实现：
- 稳定 `logloss` / `expit`
- `logit clip`（避免极值溢出）
- 可选梯度裁剪（建议默认开启）
2. 所有 overflow / invalid warning 必须记为 `run_warning` 并落盘。
3. 存在未清理 `run_warning` 时，本轮判定为“需重跑”，不得发布主结论。
4. 可复跑约束：
- 固定并落盘 `random_seed`
- 落盘共享 splits
- 落盘完整 manifest 快照与关键阈值版本

## 9. 输出物清单（更新）
### 9.1 运行级输出（`{part2_out_dir}/`）
1. `splits/company_holdout_splits.csv`
2. `splits/company_holdout_splits_meta.json`
3. `interaction_candidates_pairwise.csv`
4. `interaction_candidates_3way.csv`（仅三阶开启时）
5. `model_selection_summary.csv`（含候选排序、主模型选择、降级原因）
6. `interaction_metrics_summary_ci.csv`（聚合视图）
7. `interaction_metrics_c3_slice_ci.csv`（`C3-slice` supplementary 聚合视图）
8. `interaction_audit_linkage_summary.csv`（聚合视图）
9. `interaction_sensitivity_year_summary.csv`（+year 敏感性对比摘要）
10. `interaction_ablation_missingflags_tier2plus.csv`（Tier-2+ missingness ablation 摘要）
11. `threshold_grid_summary.csv`（若启用 4.3.2，记录每个网格点的规则数/指标/审计）
12. `interaction_selected_from_explain_model.csv`（若启用 5.1.1 的解释→回注；必须为单文件，并至少包含 `source_explain_model_name`、`target_model_name`、`feature_mode`、`target_branch_name`、`feature_a`、`feature_b`、`feature_c`、`support_n_by_fold_min`、`support_pos_by_fold_min`、`selection_freq`、`stability_freq`、`notes`）
13. `warning_summary.json`
14. `run_decision.md`
15. `run_conclusion_analysis_and_improvement.md`
16. `run_conclusion_analysis_and_improvement.en.md`
17. `run_warning.log`（无 warning 时也必须存在，可为空）
18. 工作区执行总账追加：`$REPO_ROOT/process/stage3_execution_log.md`
19. `run_manifest.json` 必须额外记录 `feature_set_registry`：
- 标准字段名固定为 `feature_set_registry`
- registry key 固定为 `{model_name}/{feature_mode}/{branch_name}`
- 每个已执行 `model × feature_mode × branch` 的 value 至少包含：`model_name`, `feature_mode`, `branch_name`, `n_cont`, `n_bin_onehot`, `n_missing_indicators`, `X_columns_sha256`

### 9.2 模型分支级输出（`{part2_out_dir}/models/{model_name}/{feature_mode}/{branch_name}/`）
1. `metrics_ci.csv`
2. `metrics_c3_slice_ci.csv`
3. `delta_ci.csv`（相对同模型 `mainline`）
4. `audit_summaries.csv`
5. `rulebook_support.csv`（发布态统一 schema；兼容 legacy 时另见 `rulebook_legacy_pair_tier2plus.csv`）
   - 必含列：
     - `rule_rank`（从 1 开始）
     - `feature_a`, `feature_b`（pair 必填；triple 时可用 `feature_c`）
     - `condition_text`（人类可读条件表达）
     - `coverage`（规则覆盖率）
     - `enrichment`（相对 baseline 的富集度/提升倍数）
     - `stability_freq`（跨 repeats/splits 的入选频率）
     - `rule_tier_min`（如 `Tier-2+`，或 `C2/C3`）
     - `rule_type`（`prediction` / `triage`）
     - `notes`（包含 downgrade_reason、口径说明：typical vs peak 等）
   - 连续交互或树分裂点进入发布态时，必须压缩为不超过 `3` 个阈值条件。
   - 禁止：任意行出现 `missing_flag_*` 或 `__MISSING__` 主条件；missing 只允许存在于敏感性/审计产物，不得出现在任何 rulebook 家族。
   - 禁止：`prediction` 行出现同组交互（`signal_group(feature_a)==signal_group(feature_b)`，或 triple 非三组）；如出现必须改为 `triage`。
6. `pair_rulebook_publishable_c3only.csv`
7. `pair_rulebook_explanation_unstable_c3only.csv`（supplementary，`explanation_only=true`）
8. `rulebook_model_derived_sensitivity.csv`（补充解释层；输出模型学习到的 cutpoints / shape 压缩规则，不参与主模型升级）
9. `rulebook_support_engineered_comparison.csv`（仅对 `nonlinear + cont_only + mainline` 必需；工程阈值对比集）
10. `model_derived_cutpoint_alignment.csv`（至少记录 `physical_grid / data_driven_grid / model_cutpoints / model_cutpoints_std / model_cutpoints_raw / alignment_note / scaling_note`）
11. `rulebook_legacy_pair_tier2plus.csv`
12. `run_decision.md`
13. `run_conclusion_analysis_and_improvement.md`
14. `run_conclusion_analysis_and_improvement.en.md`
15. `run_warning.log`
16. `rulebook_mechanism_extraction.md`（非线性模型必需）
17. `linear_continuous_effects.csv`（线性模型必需；非线性模型可为空 schema）
18. `linear_pairwise_effects.csv`（线性模型在交互分支必需；解释层，仅机制提示）
19. `linear_vs_engineered_direction_check.csv`（线性模型可选；仅诊断，不参与 winner/发布域决策）
20. `interaction_explanation_summary.md`（解释模型必需）
21. `interaction_explanation_summary.en.md`（解释模型必需）
22. `interaction_probe_artifacts/`（解释模型必需）

### 9.3 模型级聚合输出（`{part2_out_dir}/models/{model_name}/`）
1. `oof_predictions.csv`
2. `fold_metrics.csv`
3. `candidate_selection_trace.csv`
4. `audit_trace.csv`
5. 模型级聚合输出必须包含 `feature_mode` 列，用于区分 `cont_only / bin_only / cont_plus_bin`

## 10. 发布前检查与失败处置
1. 发布前必须满足：
- 输入可追溯（Part I 输入、schema、threshold）
- 共享切分可复跑
- 主模型选择链完整且可复核
- 审计链与 rulebook 完整
- publishable interaction 满足不同信号组约束（不得出现 same-signal `prediction` 规则）
2. 任一关键约束失败（Gate、审计 FAIL、可复核规则缺失、未清理 warning）：
- 禁止主线升级
- 可保留补充结果，但必须在 `run_decision.md` 与结论文档标记风险

说明：若仅为“增益 CI 跨 0 / 波动较大”但审计链 PASS 且 rulebook 可复核，则允许作为补充结果发布，并必须在 `model_zoo_comparison.md` 中以“敏感性区间/阈值网格”方式解释不确定性；不得将其写成确定性主结论。

3. 禁止 fallback、禁止跳步、禁止以人工口头结论替代落盘产物。
