# Hegel Machine v10.2 系统审计闭合、勘误账本与同工具归因报告（2026-11-16）

> [!IMPORTANT]
> **审计基准与硬约束执行声明：**
> - **被审计锚点提交：** `a66cd82b22f444044dbda98f43782c96f833a4f0`（审计输入包：`markdown/hegel_v10.2/`，含 `GPT_diagnosis_v10.2.md` 及 `01–04` 施工合同与 `probe_certificate_domain.py`）。
> - **Phase 6 范围冻结：** 严格执行 `02_下一施工合同.md` 第 2.4 节要求——未新增三维／移动网格 PDE、未新增 SOS/Positivstellensatz 求解器、未新增通用定理生成器或新的专设物理家族；保留现有 `v10` 工具集并将全部工作聚焦于**可信度修复（交付物 A）**、**同工具因果归因（交付物 B）**与**冻结全6阶段生命周期（交付物 C）**。
> - **零新增付费 LLM 调用：** 全部修复、重测、三臂同工具归因与冻结6阶段前瞻实验均在 **`model_calls = 0, total_tokens = 0`** 下完成。
> - **历史证据零覆盖保护：** `docs/evidence/v10_world_structure_20261116/` 下的全部 13 份历史 JSON 回执保持 **100% 原始字节未改动**（已逐文件记录只读 SHA256 内容摘要）；所有新增与重测回执统一写入独立目录 `docs/evidence/v10_2_audit_remediation_20261116/`。

---

## 一、新增证据回执清单（`docs/evidence/v10_2_audit_remediation_20261116/`）

| 序号 | 回执文件 | 对应交付物 / 发现项 | 核心核验内容 | 在线 LLM 调用 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `pre_fix_a66cd82b_domain_probe_20261116.json` | A1 (`F01`) | 在修复前锚点 `1f9f872a...` 上运行 `probe_certificate_domain.py`，复现 `both_opposite_certificates_accepted = true` | `0` |
| 2 | `post_fix_domain_probe_20261116.json` | A1 (`F01, F02`) | 在修复后 `algebraic_proof_certificates.py` 上运行同一探测脚本，确认 `both_opposite_certificates_accepted = false`、`potential_lean_compiled = false` | `0` |
| 3 | `sde_and_proof_remediation_20261116.json` | A1, A2, A3 (`F01, F02, F07, F08, F10–F12, F14`) | 三种 `state_bounds_mode` 语义测试、`7/7` 计费有限元数不变量、共享核验器 Rollout 对照、`learned_model -> certificate` 在线桥接、SDE 5B-1 含噪历史一步条件漂移风险 vs 无噪孪生迁移 MSE 分离重测 | `0` |
| 4 | `equal_tool_attribution_20261116.json` | B1, B2, B3 (`F03–F06, F17`) | Study B1 三臂同工具知识归因（含强 `archive_full_tools` 诚实平手）、Study B2 单因子前提依赖线索隔离＋局部补丁 vs 从零合成独立因子表、Study B3 跨 4 类不同物理机制的双臂实测探针修复对比 | `0` |
| 5 | `frozen_v4_lifecycle_20261116.json` | C1, C2, C3 (`#9–#13, #15, #16, #18, #21`) | `1` 条开发定标链（`dev_chain_700000`）＋ `4` 条连续预注册零挑选前瞻物理源链（`711000–714000`）的冻结全6阶段生命周期三臂同工具评测 | `0` |
| 6 | `errata_and_sha256_ledger_20261116.json` | A4 (`F01–F20, F16`) | 13 份未改动 `v10` 历史回执的只读 SHA256 摘要、实验角色与独立物理源重分类，以及 `F01–F20` 完整勘误映射表 | `0` |

---

## 二、20 项审计发现（`F01–F20`）逐项闭合与勘误矩阵

| 编号 | 级别 | 原报告表述或代码问题（`a66cd82b`） | 审计核实的真实状态 | `v10.2` 代码／口径修复与收紧结论 | 验证回执与测试 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`F01`** | **P0** | 同一控制题可同时通过不可达势证书与可达控制证人 | `PotentialLowerBoundCertificate` 使用了 `state_bounds`，而 `ReachabilityWitnessCertificate` 与 `BudgetedControlLab` 未强制同一盒约束且未证明可达包络 | 新增 `validate_control_task_semantics` 与 `verify_reachable_box_enclosure`，统一三种模式：`physical_legality`（全链路强制盒约束）、`analysis_region`（未证包络返回 `conditional_impossibility_domain_unverified`, `verified=False`）、`unbounded`（禁止虚构默认盒，拒绝非零盒乘子） | `post_fix_domain_probe_20261116.json`；`test_acceptance_01_and_02` |
| **`F02`** | **P1** | 将 Lean 4 导出字符串表述为“自包含零 `sorry` 形式证明” | 精确有理校验实际由 Python `Fraction` 与 `SymPy QQ` 执行；Lean 字符串未经过 `lean` 编译器检查 | 撤回“Lean 已编译定理”表述；所有导出显式标记 `lean_status="draft_only"`、`lean_compiled=False`、`trusted_computing_base="python_fractions_and_sympy_QQ"`，并在草稿头部注明 `proof_verified=false in Lean` | `test_acceptance_05_06_07_08` |
| **`F03`** | **P1** | 将 Phase 3A/4A/4B/5A 求解器开关消融混入“持久黑格尔学习优越性” | 基线臂同时关闭了 `manifold_pde_discovery` / `ar2` / `vdp` 等求解器能力开关 | 将 Phase 3A/4A/4B/5A 重分类为**求解器与模型类能力消融**；在 Study B1 与交付物 C 中执行严格同工具三臂归因（`cold_full_tools` vs `archive_full_tools` vs `structured_full_tools`） | `equal_tool_attribution_20261116.json` |
| **`F04`** | **P1** | 将 Phase 2B/3B Stage A 称为“`PremiseDependencyGraph` 单因子隔离” | 同时改变了父承诺、已解析角色、违背前提提示与局部槽位语法 vs 完整 AST | 将 Phase 2B/3B Stage A 重标定为**结构化修订工作流多因子对比**；新增 Study B2 严格单因子依赖线索隔离（共享同一父承诺、同一数据、同一补丁语法，仅切换违背前提线索）及独立的 `local_patch` vs `from_scratch` 因子表 | `equal_tool_attribution_20261116.json`；`test_acceptance_13_14_15_16_19_20_21` |
| **`F05`** | **P1** | 将 Cohort 5C-3 表述为修复 4 类不同物理失配机制 | 原始 5C-3 仅测试了 `boundary_port_gain` 的 4 个参数水平（`0.60, 1.00, 1.25, 1.50`） | 修正 5C-3 历史描述为“单一端口增益机制的 4 个参数实例”；并在 Study B3 真正实现并执行跨 **4 类不同物理机制**（`boundary_port_gain`、`bulk_sink_dissipation`、`coupling_shift`、`direction_reversal`）的双臂修复评测 | `equal_tool_attribution_20261116.json` |
| **`F06`** | **P1** | 将 `89.92%` 探针减少作为实测基线对比报告 | 基线探针数来自名义公式 $2(3N+1)+N(N-1)$，未实际运行无引导修复代理 | 将 `89.92%` 降级标注为名义全扫描公式估计；在 Study B3 运行真实双臂同工具代理并记录 `lab.probe_queries`（实测由 `101` 步降至 `42` 步，**-58.42%**） | `equal_tool_attribution_20261116.json` |
| **`F07`** | **P1** | Rollout 基线硬编码 `certified_proof = False` | 原始 Rollout 即使找到合法控制序列也无法调用证人核验器 | 新增 `certify_rollout_baseline` 共享 `SymbolicProofKernel.verify_solvability_certificate`，并将 5C-2 明确归类为精确 LP／证人求解器相对启发式采样的能力对比 | `sde_and_proof_remediation_20261116.json` |
| **`F08`** | **P1** | 将 `process_noise_std=0` 无噪孪生 MSE 等同于含噪条件漂移风险，弱化原生 `2/4` 通过率 | 关闭过程噪声改变了历史信息集 $\mathcal{F}_t$ 的边际分布；原生在线确认完成率为 `2/4` | 新增独立后验评估器 `sde_conditional_drift_eval.py`，分离四项指标：**原生完成率 `2/4`**、**实时含噪轨迹 MSE `5.42e-3`**、**真实含噪历史一步条件漂移 MSE `6.12e-4`（`4/4 <= 2e-3`）**、**无噪孪生迁移 MSE `3.32e-4`** | `sde_and_proof_remediation_20261116.json`；`test_acceptance_17` |
| **`F09`** | **P1** | Phase 2B/3B 保存确认记录时误传入开发验证集行数 `36` 与 `validation_loss` | 独立 `fresh48` 门控确实已执行，但 `remember_confirmed_selection` 元数据写串了 `validation` 字段 | 修复 `unified_research.py` 及 Phase 2B/3B 脚本，严格分离 `fit_rows`、`selection_rows`、`selection_loss`、`confirmation_rows=48`、`confirmation_loss=fresh48_mse` 与幂等去重 `confirmation_ref` | `test_acceptance_09_11_and_f15` |
| **`F10`** | **P1** | Phase 5C 符号消元、LP 求解与证书核验未接入 `SearchBudget` | 相关函数无 `budget` 参数，耗尽预算无法优雅中断 | 在 `SymbolicProofKernel`、`synthesize_emergent_invariants`、`prove_task_solvability`、`certify_rollout_baseline`、`repair_proof_from_discrepancy` 全链路接入 `SearchBudget`，预算耗尽返回 `status="unknown", reason="budget_exhausted"` | `test_acceptance_13_14_15_16_19_20_21` |
| **`F11`** | **P2** | 将有限测试元数集（如 $N\in\{3,5,8\}$）表述为“任意元数定理” | `SymPy` 对每个有限测试 $N$ 逐一展开校验，未做符号归纳证明 | 显式设置 `all_arity_proved = False`，并输出 `verified_on_finite_arities` 列表 | `test_acceptance_05_06_07_08` |
| **`F12`** | **P2** | 单向矩单调性仅检验通量恒等式，未在内核强制非负锥与有向边号前提 | 符号不等式成立依赖状态非负锥 $q_i \ge 0$ 与活跃边位移同号，原仅写在注释中 | 在 `SymbolicProofKernel` 强制核查 `domain_obligations`（`donor_state_cone == "nonnegative"`、有向边位移号校验及有理锥生成元逐项非负核验），缺失或违背一律拒绝 | `test_acceptance_05_06_07_08` |
| **`F13`** | **P2** | 混淆人工预置前提图／求解器库与运行时自主学得承诺 | 前提节点本体与求解器族系由开发者编写；运行时自主学习角色绑定、参数区间、作用域承诺与局部组合 | 在证书与报告中严格区分 `declared_premises`、`empirically_supported_premises`、`refuted_premises`、`formal_obligations` | 本文档第 二、三 节 |
| **`F14`** | **P2** | Phase 5C 输入手工 `system_spec`/`model` 而非运行时学得承诺 | 原始 5C 属于 `supplied_model` 证明器定标 | 新增 `model_origin`（`"supplied_model"` vs `"learned_model"`）并实现 `extract_control_model_from_learned_commitment`，打通从 `AnonymousObjectLab` 在线发现到可解／不可达证书生成的端到端桥接 | `sde_and_proof_remediation_20261116.json` |
| **`F15`** | **P2** | 交接文档将 `AR(2)` AST 误写为 `children=(const(a1), const(a2), drive)` | 真实代码 `_ar2_role_term` 为 `Program("accumulate", value=a1, name="ar2"|"ar2_lag1", children=(drive, const(a2)))` | 修正全部技术报告与交接文档中的 AST 契约描述，并新增序列化往返单元测试锁定 | `test_acceptance_09_11_and_f15` |
| **`F16`** | **P2** | 将 `28` 个专项测试与 `13` 份回执笼统表述为全仓回归与 13 项前瞻验证 | 仓库含海量历史测试，且 13 份回执涵盖历史普查、求解器消融、多因子工作流等不同角色 | 在 `errata_and_sha256_ledger_20261116.json` 逐份标注 `study_role` 与独立物理源数量，并分离报告专项测试（`52/52`）与全仓测试收集规模（`17,520` 项） | `errata_and_sha256_ledger_20261116.json` |
| **`F17`** | **P2** | 将普通档案基线低成本重构等价方法称为“同构漏洞（isomorphism bug）” | 从共享原始概念低成本重构等价承诺是合法的强基线表现，平手是真实科学结论 | 撤回“同构漏洞”表述；在交付物 B 与 C 中允许 `archive_full_tools` 按需重构等价承诺并计入重构算力，如实报告其与预巩固承诺的近似平手结果 | `equal_tool_attribution_20261116.json`；`frozen_v4_lifecycle_20261116.json` |
| **`F18`** | **P2** | 将循环格点平移称为全群 $\operatorname{Aut}(\mathbb{T}^2)$、跨网格称为连续极限 PDE | 实际检验的是离散周期平移子群 $\mathbb{Z}_{N_x}\times\mathbb{Z}_{N_y}$ 及有限离散格点／图动力学 | 全部数学表述收紧为“有限离散格点／图动力系统及离散周期平移子群 $\mathbb{Z}_{N_x}\times\mathbb{Z}_{N_y}$ 不变性” | 本文档及更新后报告 |
| **`F19`** | **正向** | 完整拟合列跨元数实例化、零权重列保留与前提级局部修订 | 经源码核查属实（`world_structure_commitments.py`、`premise_dependency_audit.py`） | 完整保留，并进一步补齐经验一致性门控 `is_commitment_empirically_consistent` 与联合版本空间亚微米重分割（`relation_uncertainty.py`） | `frozen_v4_lifecycle_20261116.json` |
| **`F20`** | **正向** | 同工具 `602 -> 603` 搜索预算下降 `66.47%` 与零额外 LLM 跨元数复用 | 经回执核查属实 | 完整保留，并精确限定为 `SearchBudget` 记账算力下降及独立源／目标双账本报告 | `v10_cow_602_603_new_ledger_20261116.json` |

---

## 三、交付物 A 定量核验：证明域合法性统一、在线桥接与 SDE 条件漂移重测

### 3.1 GPT-6 域反例探测（`probe_certificate_domain.py`）修复前后与三模式对照

| 运行版本 / `state_bounds_mode` | `PotentialLowerBoundCertificate` | `ReachabilityWitnessCertificate` | `BudgetedControlLab` 轨迹合法性 | `both_opposite_certificates_accepted` |
| :--- | :--- | :--- | :--- | :--- |
| **修复前锚点 `a66cd82b`**（`1f9f872a...`） | `verified = True` | `verified = True`（漏检出盒！） | 未拦截出盒状态 | **`True`（P0 冲突复现）** |
| **修复后：`physical_legality`**（物理硬边界 `[-1, 0]^2`） | `verified = True`（盒内真不可达） | `verified = False`（拦截 `step_0` 出盒） | `winnable_verified_on_lab = False` | **`False`（冲突彻底消除）** |
| **修复后：`analysis_region`**（仅分析盒，未证可达包络） | `verified = False`（`conditional_impossibility_domain_unverified`） | `verified = True`（成本 `1/5 <= 1/2`） | `winnable_verified_on_lab = True` | **`False`（拒绝未证无条件不可能）** |
| **修复后：`unbounded`**（无状态盒声明 `state_bounds=None`） | `verified = False`（拒绝非零盒乘子） | `verified = True`（求解器直接产出可达证人） | `winnable_verified_on_lab = True` | **`False`（不虚构默认盒）** |

- **从在线学习承诺到形式证书桥接（`F14`）：** 在 `AnonymousObjectLab(objects=3, readout="total_event", relation_direction="right", relation_radius=0.95)` 上由 `UnifiedResearchAgent` 自主完成传感器角色解耦、前提审计与世界承诺巩固（`learning_status = "completed"`，在线学习算力 `6,920`），通过 `extract_control_model_from_learned_commitment` 提取 `model_origin = "learned_model"` 控制模型并生成预算充足可达证人（`B = 1.6 -> winnable`）与低预算不可达证书（`B = 0.4 -> impossible`，`certified_min_budget = 1.0`），同时诚实标记 `environment_coverage_verified = False`（证明对所提取模型精确成立，环境等价性取决于经验前提审计）。

### 3.2 Phase 5B-1 强过程噪声 SDE 四指标分离重测（`sde_and_proof_remediation_20261116.json`）

设信息集 $\mathcal{F}_t$ 严格仅含事前特征与历史步（`post_step_after_sensors_exposed_in_prediction = False`），在真实含噪轨迹分布上评估一步条件漂移风险：

$$\mathcal{R}_{\text{drift}}(\widehat{m}) = \mathbb{E}_{\mathcal{F}_t \sim P_{\sigma_p > 0}}\!\left[\left(\widehat{m}(\mathcal{F}_t) - \mathbb{E}[Y_t \mid \mathcal{F}_t]\right)^2\right]$$

| 对照臂（`Estimation-Method Ablation`，4题） | 原生在线确认 `native_completed` | 实时含噪轨迹均值 MSE | **含噪历史精确条件漂移 MSE**（达标 `<=2e-3`） | 含噪历史 64-MC 条件漂移 MSE | 无噪孪生系统迁移 MSE | 潜极点均值绝对误差 | 均值 `SearchBudget` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `deterministic_v10_no_sde_audit` | `0/4` | `1.002e-01` | `9.248e-02`（`0/4`） | `9.276e-02` | `9.311e-02` | — | `500,282.0` |
| `stochastic_v10_naive_ols` | `1/4` | `1.191e-02` | `6.276e-03`（`1/4`） | `6.434e-03` | `6.188e-03` | `0.2060`（Hurwicz-Nickell 偏置） | `546,435.0` |
| **`v10_phase5b_stochastic_2sls_iv`** | **`2/4`** | **`5.421e-03`** | **`6.119e-04`（`4/4`，较 OLS 降 `10.26×`）** | **`6.990e-04`** | **`3.317e-04`** | **`0.0181`（降 `11.37×`）** | **`425,124.2`** |

---

## 四、交付物 B 定量核验：严格同工具知识归因、单因子隔离与四机制实测修复

### 4.1 Study B1：三臂严格同工具知识归因（4 题跨元数偏移相关目标 ＋ 1 题不匹配机制负向控制）

三臂均开启完全相同的 `v10` 全套求解器＋有界开放语法局部补丁搜索（`model_calls = 0`）；`archive_full_tools` 保留完整原始源先验并获准在目标题按需重构世界承诺（重构算力计入目标题总账，`F17`）：

| 同工具对照臂 | 4题相关目标通过率 (`<=1e-7`) | 均值观测行数 | 均值测量通道数 | 均值按需重构算力 | 4题目标总算力 (`sum_total_target_compute`) | 不匹配控制题（`b1_control_vdp_n4`）表现 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cold_full_tools`（全工具无历史记忆） | **`0/4 (0.0%)`** | `295.75` | `4,975.5` | `0.0` | `1,487,155` | `completed`（全工具原生 VdP 求解器独立解出，无虚假负迁移） |
| `archive_full_tools`（全工具＋原始先验按需重构） | **`3/4 (75.0%)`** | `204.50` (**-30.85%**) | `3,050.25` | `190.0` | `975,874` (**-34.38%**) | `completed`（不匹配先验被经验一致性门控安全跳过） |
| **`structured_full_tools`（全工具＋预巩固世界承诺）** | **`3/4 (75.0%)`** | `205.00` (**-30.68%**) | `3,059.25` | `0.0` | **`975,294`（较 `cold` -34.42%；较 `archive` 省 `580` 算力）** | `completed`（不匹配承诺零误触发） |

> [!NOTE]
> **对 `F17` 的诚实归因结论：** 在相同全套工具下，跨任务保留先验知识相对 `cold_full_tools` 带来决定性收益（完成率 `0/4 -> 3/4`，总目标算力下降 `34.4%`）；同时，当知识库规模较小时（1–2 条源先验），强基线 `archive_full_tools` 在目标端按需重构等价承诺仅需约 `190` 算力／题，因此 `structured_full_tools` 与 `archive_full_tools` 呈现**真实的近似平手**（仅节省预巩固摊销算力 `580` 单位）。第四题 `b1_fq_n5_s2` 在 300k 目标算力上限内因匿名角色消歧耗时未完成确认，保留为诚实未过样本。

### 4.2 Study B2：`PremiseDependencyGraph` 严格单因子消融与“局部补丁 vs 从零合成”独立因子表

在 6 台失配仪器（覆盖 `P_exchange_form`、`P_temporal_readout`、`P_latent_transition_drive` 三类前提违背）上，固定完全相同的父承诺、共享观测包（`96` 训练行 + `36` 选择行 + `167` 转移记录）与 `43` 候选局部补丁语法，**唯一自变量**为是否传入 `PremiseDependencyGraph` 审计出的违背前提槽位线索（各臂信息清单 SHA256 已在回执中逐题记录）：

| 预算制度与单因子对照臂 | 验证集支持数 (`supported`) | 独立 `fresh48 <= 1e-7` 通过数 | `budget_interrupted` 中断数 | 均值搜索算力 (`SearchBudget`) |
| :--- | :--- | :--- | :--- | :--- |
| **紧预算 `16k`**：`same_grammar_no_dependency_cue` | `0/6` | `0/6` | `6/6` | `15,993.5` |
| **紧预算 `16k`**：`same_grammar_with_dependency_cue` | `0/6` | `0/6` | `6/6` | `15,993.5`（共享角色消歧即耗尽 `16k`，诚实平手） |
| **充足预算 `220k`**：`same_grammar_no_dependency_cue` | `2/6` | **`2/6 (33.3%)`** | `4/6` | `168,137.3` |
| **充足预算 `220k`**：**`same_grammar_with_dependency_cue`** | **`6/6`** | **`4/6 (66.7%)`** | **`0/6`** | **`54,075.0`（算力下降 `-67.84%`）** |

同时单独剥离**局部补丁语法（`local_patch_no_dependency_cue`）vs 全语法从零枚举（`from_scratch_no_dependency_cue`）**（`80k` 算力、均无依赖线索，4题）：`from_scratch` 通过 **`0/4`**（尝试 `942–2,966` 项候选均耗尽预算），`local_patch` 通过 **`2/4`**——证明 Phase 2B/3B 的总收益实由“父承诺局部补丁缩减搜索空间”与“前提依赖线索定向槽位排序”两项独立因子共同贡献。

### 4.3 Study B3：跨 4 类不同物理机制的双臂实测证明失配修复（8 题：$N \in \{4, 5\}$）

| 失配物理机制（每机制 $N=4,5$ 各1题） | 无引导全工具代理实测 `lab.probe_queries` | `SufficientCover` 依赖引导代理实测 `lab.probe_queries` | 修复后证书核验通过率 | 实测探针节省率 |
| :--- | :--- | :--- | :--- | :--- |
| 机制 1：边界端口非线性增益（`boundary_port_gain`） | `15 + 18 = 33` | `4 + 5 = 9` | `2/2` vs `2/2` | `-72.73%` |
| 机制 2：体相耗散漏汇（`bulk_sink_dissipation`） | `11 + 13 = 24` | `5 + 6 = 11` | `2/2` vs `2/2` | `-54.17%` |
| 机制 3：耦合系数偏移（`coupling_shift`） | `10 + 12 = 22` | `5 + 6 = 11` | `2/2` vs `2/2` | `-50.00%` |
| 机制 4：有向交互反转（`direction_reversal`） | `10 + 12 = 22` | `5 + 6 = 11` | `2/2` vs `2/2` | `-50.00%` |
| **8 题总计（4 类独立机制）** | **总计 `101` 步（均值 `12.625`）** | **总计 `42` 步（均值 `5.250`）** | **`8/8` vs `8/8`（100%）** | **实测 `-58.42%`**（替代原公式估计 `89.92%`） |

---

## 五、交付物 C 定量核验：冻结全 6 阶段 `v4` 知识生命周期（`frozen_v4_lifecycle_20261116.json`）

实验严格按物理源链为独立样本单位（`1` 条开发定标链 `dev_chain_700000` 单独披露且不计入前瞻统计；`4` 条连续预注册种子链 `711000, 712000, 713000, 714000` 在目标暴露前 SHA256 冻结提议目录、策略配置与源承诺，零事后选种）：
- **Stage 1（源任务结构形成，$N=4, \text{right}, \text{donor\_positive\_gap}$）** $\to$ **Stage 2（目标暴露前 SHA256 冻结，验证源浮点数直接代入 Stage 3 均失败 `MSE > 2e-3`）** $\to$ **Stage 3（跨元数＋双参数偏移迁移，$N=5, \text{right}$）** $\to$ **Stage 4–5（无提示反例证伪 $N=5, \text{left}$，旧承诺 $C_0$ 降级为 `status="conditional"` 并局部修订生成 $C_1$）** $\to$ **Stage 6（原条件跨元数回归 $N=6, \text{right}$，检验是否拒绝不兼容新承诺 $C_1$ 并通过等价类重整直接恢复 $C_0$）**。

### 5.1 开发定标链与 4 条连续预注册前瞻链逐链明细（`structured_lifecycle_v10` vs 基线）

| 链编号与角色 | Stage 1 源形成 (`N=4`) | Stage 3 偏移迁移 (`N=5, right`) | Stage 4–5 反例与局部修订 (`N=5, left`) | Stage 6 原条件回归 (`N=6, right`) | `cold_full_tools` 后3阶段通过 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dev_chain_700000`（开发定标链，不计入前瞻均值） | **通过**（`8.53e-21`，源直代失败 `4.36e-3`） | **通过**（`1.30e-21`） | **通过**（`1.04e-20`，$C_0 \to \text{conditional}$） | **通过**（`9.76e-19`，`selected_channels=[]`，**恢复 $C_0$**） | `0/3` |
| `prospective_chain_1`（`711000`，冻结前瞻） | **通过**（`6.93e-21`，源直代失败 `3.91e-3`） | **通过**（`1.54e-18`，`channels=[]`） | **通过**（`1.20e-18`，`channels=[]`，$C_0 \to \text{conditional}$） | 未完成在线确认（但**运行时成功拒绝 $C_1$、恢复 $C_0$**，`fresh48 MSE = 4.06e-21`） | `0/3` |
| `prospective_chain_2`（`712000`，冻结前瞻） | **未通过**（初测采样稀疏未过 `1e-7` 门控，冻结 `0` 承诺） | 未通过（无源承诺可用，三臂诚实同分） | 未通过（无源承诺可用） | 未通过（标签 `no_retained_knowledge_available`） | `0/3` |
| `prospective_chain_3`（`713000`，冻结前瞻） | **通过**（`4.98e-22`，源直代失败 `2.15e-3`） | **通过**（`1.67e-18`，`channels=[]`） | **通过**（`1.33e-21`，$C_0 \to \text{conditional}$） | **通过**（`1.50e-18`，`channels=[]`，**恢复 $C_0$**） | `0/3` |
| `prospective_chain_4`（`714000`，冻结前瞻） | **通过**（`1.52e-22`，源直代失败 `2.80e-3`） | **通过**（`4.41e-19`，`channels=[]`） | **通过**（`4.20e-19`，`channels=[]`，$C_0 \to \text{conditional}$） | 未通过（`N=6` 18传感器角色消歧耗尽 `520k` 预算，标签 `retained_knowledge_not_invoked`） | `0/3` |

### 5.2 4 条冻结前瞻链三臂汇总账本（严格分离源形成成本、冻结后目标成本与全序列累计成本）

| 前瞻评估维度（样本单位：$n=4$ 条独立前瞻物理源链，共 12 个冻结后阶段） | `cold_full_tools` | `archive_full_tools`（强重构基线） | **`structured_lifecycle_v10`** |
| :--- | :--- | :--- | :--- |
| **Stage 3 跨元数双参数偏移迁移通过数** | `0/4 (0.0%)` | **`3/4 (75.0%)`**（源形成成功链 `3/3`） | **`3/4 (75.0%)`**（源形成成功链 `3/3`） |
| **Stage 4–5 无提示反例降级＋局部修订通过数** | `0/4 (0.0%)` | **`3/4 (75.0%)`**（$C_0$ 全部转为 `conditional`） | **`3/4 (75.0%)`**（$C_0$ 全部转为 `conditional`） |
| **Stage 6 原条件 $N=6$ 回归恢复 $C_0$ 数 / 最终确认通过数** | `0/4` 恢复 / `0/4` 确认 | **`2/4` 恢复 $C_0$** / **`1/4` 确认** | **`2/4` 恢复 $C_0$** / **`1/4` 确认** |
| **冻结后阶段总通过数（12 阶段）** | **`0/12 (0.0%)`** | **`7/12 (58.3%)`**（有源承诺链 `7/9 = 77.8%`） | **`7/12 (58.3%)`**（有源承诺链 `7/9 = 77.8%`） |
| **单链冻结后目标阶段均值算力（`post_freeze_target_compute`）** | `1,488,940.5` | `1,140,655.8` (**-23.39%**) | **`1,139,826.8`（较 `cold` -23.45%；较 `archive` -829.0）** |
| **单链含 Stage 1 全序列累计均值算力（`cumulative_compute`）** | `1,540,861.5` | `1,192,576.8` (**-22.60%**) | **`1,191,882.8`（较 `cold` -22.65%；较 `archive` -694.0）** |
| **单链冻结后均值观测行数 / 全序列累计均值观测行数** | `1,209.25` / `1,460.25` | `979.25` / `1,230.25` | `1,013.50` / `1,264.50` |

> [!TIP]
> **生命周期机制修复洞察（研发链定位、前瞻链零微调验证）：**
> 1. **匿名仪器签名隔离（`_unique_task_signature`）：** 此前所有 `AnonymousObjectLab` 实例共享 `task_id = "anonymous_process"`，导致 Stage 4 反例会将 $C_0$ 在所有后续仪器上误判为同任务拉黑；引入基于控制通道名的确定性仪器签名后，Stage 6 得以在回归原物理条件时合法恢复 $C_0$。
> 2. **联合版本空间亚微米重分割（`relation_uncertainty.py`）：** 当半径 $r_0$ 先收敛至 $\le 10^{-6}$ 而阈值 $c_0$ 的后续二分探针恰好落入 $r_0$ 的亚微米残差胞腔内时，允许兄弟参数在该窄胞腔内继续按新切点精细重分割，并补齐接收端焦点对称谓词 `or(source_focus, target_focus)`（同时修复了历史回归测试 `test_cognition_conditional_relation_intervals.py`）。

---

## 六、22 项行为验收测试（`#1–#22`）与测试规模诚实披露（`F16`）

- **专项回归与验收测试（`52/52` 全部通过）：**
  - `tests/test_v10_2_audit_acceptance.py`（8 个集成测试函数完整覆盖合同第 5 节 **`#1–#22`** 全部行为验收标准及 `F15` AST 往返契约与 SHA256 账本完整性校验）；
  - `tests/test_v10_*.py`（全部 Phase 1–5 专项测试）＋ `tests/test_historical_capabilities.py`（历史能力锁测试）＋ `tests/test_cognition_conditional_relation_intervals.py`、`tests/test_cognition_condition_questions.py`、`tests/test_cognition_competing_object_experiments.py` 合计 **`52/52`（100% 通过）**。
- **全仓库测试规模区分（`F16`）：** 仓库 `tests/` 下共收集 **`17,520`** 项参数化历史测试（排除 2 个历史遗留导入缺失脚本），与专项 `v10/v10.2` 验收套件严格区分报告。
- **静态反作弊扫描（100% 通过）：** 零持久化 API Key、运行时认知模块对实验室私有属性 `._` 零读取（SDE 真实条件漂移真值核验已隔离至独立评估模块 `sde_conditional_drift_eval.py`）、`docs/evidence/v10_world_structure_20261116/` 全部 13 份文件 `git status` 零改动。
- **验收边界重申（`#22`）：** 通过全部行为验收测试证明了代码层的信任边界一致、算力记账闭合与实验因子隔离合法；系统的前瞻科学能力严格以上述冻结回执中的实测分布（含 `prospective_chain_2` 源形成未过、`N=6` 部分预算耗尽以及与强 `archive_full_tools` 的近似平手）为准，不做任何外推夸大。
