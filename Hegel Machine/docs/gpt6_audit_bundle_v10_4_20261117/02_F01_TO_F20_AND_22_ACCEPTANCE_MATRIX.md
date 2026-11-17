# F01–F20 审计发现闭合矩阵与 22 项行为验收逐条追溯表

固定对比基线：`a66cd82b22f444044dbda98f43782c96f833a4f0`（GPT-6 审计输入包 `markdown/hegel_v10.2/`） $\to$ `85b6055b11312a53720561f6697f6054b4ab8873`（`v10.2` 审计闭合） $\to$ `c0bad267a4f211cda03f388e9386908e73c27016`（`v10.3` 规模索引与高元数角色标定） $\to$ `v10.4`（奥卡姆极小性门控与统一审计包）。

---

## 一、20 项审计发现（`F01–F20`）逐项闭合表

| 编号 | 级别 | `a66cd82b` 原问题 / 过度表述 | 代码与实验修复（含本包 `bundled_sources/` 对应路径） | 核验回执（本包 `receipts/`）与测试 |
| :--- | :--- | :--- | :--- | :--- |
| **`F01`** | **P0** | `PotentialLowerBoundCertificate` 使用 `state_bounds`，而 `ReachabilityWitnessCertificate` 与 `BudgetedControlLab` 未强制同一盒约束且未证明可达包络，导致同一题同时接受可达与不可达 | `src/hegel_machine/cognition/algebraic_proof_certificates.py` 新增 `validate_control_task_semantics` 与 `verify_reachable_box_enclosure`：统一 `physical_legality`（全链路强制盒约束）、`analysis_region`（未证包络返回 `status="conditional_impossibility_domain_unverified"`, `verified=False`）、`unbounded`（禁止默认盒，拒绝非零盒对偶乘子） | `v10_2_audit_remediation_20261116/post_fix_domain_probe_20261116.json`（`both_opposite_certificates_accepted = False`）；`test_acceptance_01_and_02` |
| **`F02`** | **P1** | 将 Lean 4 导出字符串表述为“自包含零 `sorry` 已证定理” | 撤回“Lean 已编译证明”表述；`emit_lean4_certificate` 头部显式输出 `lean_status=draft_only, proof_verified=false in Lean`，全部返回字典标记 `lean_compiled=False`、`trusted_computing_base="python_fractions_and_sympy_QQ"` | `post_fix_domain_probe_20261116.json`；`test_acceptance_05_06_07_08` |
| **`F03`** | **P1** | 将 Phase 3A/4A/4B/5A 求解器开关消融混入“持久黑格尔学习优越性” | 在 `errata_and_sha256_ledger_20261116.json` 将 Phase 3A/4A/4B/5A 重分类为 `solver_and_representation_capability_ablation`；在 `equal_tool_attribution.py` 执行严格同工具三臂对照（`cold_full_tools` vs `archive_full_tools` vs `structured_full_tools`） | `equal_tool_attribution_20261116.json`（Study B1） |
| **`F04`** | **P1** | 将 Phase 2B/3B Stage A 平铺 LLM vs 门控 LLM 称为单因子依赖隔离 | 重分类 Stage A 为结构化修订多因子工作流；新增 `run_single_factor_dependency_ablation_arm`（固定同一父承诺、同一观测包、同一 43 项局部补丁目录，唯一切换 `use_support_dependency_cue`）及独立的 `local_patch` vs `from_scratch` 因子表 | `equal_tool_attribution_20261116.json`（Study B2：`220k` 下 `4/6` vs `2/6`，算力 `-67.84%`；`16k` 下诚实平手 `0/6` vs `0/6`） |
| **`F05`** | **P1** | 将 Cohort 5C-3 描述为修复 4 类不同故障机制 | 勘误原 5C-3 为“`boundary_port_gain` 单一机制的 4 个参数水平”；并在 Study B3 实现并实测 **4 类不同物理机制**（`boundary_port_gain`、`bulk_sink_dissipation`、`coupling_shift`、`direction_reversal`） | `equal_tool_attribution_20261116.json`（Study B3：8 题全过） |
| **`F06`** | **P1** | 将 `89.92%` 探针减少作为实测基线结果报告 | 将 `89.92%` 明确标注为名义公式 $2(3N+1)+N(N-1)$ 估计；在 Study B3 运行真实无引导全工具修复代理测量实际 `lab.probe_queries`（`101 -> 42`，实测 **-58.42%**） | `equal_tool_attribution_20261116.json`（Study B3） |
| **`F07`** | **P1** | Rollout 基线硬编码 `certified_proof=False` 且无证人核验入口 | 新增 `certify_rollout_baseline` 共享 `SymbolicProofKernel.verify_solvability_certificate`，并将 5C-2 归类为精确 LP 求解器相对采样的能力对比 | `sde_and_proof_remediation_20261116.json` |
| **`F08`** | **P1** | 将 `process_noise_std=0` 无噪孪生误差等同于带噪条件漂移风险 | 新增 `sde_conditional_drift_eval.py`，严格分离 4 项指标：原生完成率 `2/4`、实时含噪轨迹 MSE `5.421e-3`、**真实含噪历史一步条件漂移 MSE `6.119e-4`（`4/4 <= 2e-3`，较 OLS `6.276e-3` 降 `10.26x`）**、无噪孪生迁移 MSE `3.317e-4` | `sde_and_proof_remediation_20261116.json`；`test_acceptance_17` |
| **`F09`** | **P1** | Phase 2B/3B 确认后保存记录误写 `validation_rows=36` 与 `validation_loss` | 修复 `unified_research.py` `remember_confirmed_selection` 及 Phase 2B/3B 脚本，严格区分 `fit_rows`、`selection_rows`、`selection_loss`、`confirmation_rows=48`、`confirmation_loss=fresh48_mse` 与幂等 `confirmation_ref` | `test_acceptance_09_11_and_f15` |
| **`F10`** | **P1** | Phase 5C 符号消元、LP 与证书核验未接入 `SearchBudget` | 在 `SymbolicProofKernel`、`synthesize_emergent_invariants`、`prove_task_solvability`、`certify_rollout_baseline`、`repair_proof_from_discrepancy` 全链路扣减 `SearchBudget`，耗尽时返回 `status="unknown", reason="budget_exhausted"` | `test_acceptance_13_14_15_16_19_20_21` |
| **`F11`** | **P2** | 将有限测试元数集表述为任意 $N$ 定理 | 返回 `all_arity_proved=False` 与显式 `verified_on_finite_arities` 列表 | `sde_and_proof_remediation_20261116.json`；`test_acceptance_05_06_07_08` |
| **`F12`** | **P2** | 单向矩单调性仅检查恒等式，未在内核强制非负锥与有向边号约束 | `SymbolicProofKernel.verify_invariant_certificate` 强制核查 `domain_obligations`（`donor_state_cone == "nonnegative"`、有向边位移号及有理锥生成元逐项非负），缺失或违背即拒绝 | `test_acceptance_05_06_07_08` |
| **`F13`** | **P2** | 混淆开发者编写的前提图／求解器族与运行时自主学得对象 | 在证书与报告中分层标记 `declared_premises`、`empirically_supported_premises`、`refuted_premises`、`formal_obligations` | `01_UNIFIED_AUDIT_DOSSIER_v10_2_to_v10_4.md` |
| **`F14`** | **P2** | Phase 5C 仅输入手工 `system_spec` 而未测试学得模型桥接 | 新增 `model_origin`（`"supplied_model"` vs `"learned_model"`）与 `extract_control_model_from_learned_commitment`，实测从 `AnonymousObjectLab` 在线学习到可解／不可达证书闭环，并标记 `environment_coverage_verified=False` | `sde_and_proof_remediation_20261116.json`；`test_acceptance_05_06_07_08` |
| **`F15`** | **P2** | 文档误写 `AR(2)` AST 为 `children=(const(a1), const(a2), drive)` | 修正文档契约为真实代码 `Program("accumulate", value=a1, name="ar2"|"ar2_lag1", children=(drive, const(a2)))`，并用序列化往返测试锁定 | `test_acceptance_09_11_and_f15` |
| **`F16`** | **P2** | 将专项测试与 13 份回执笼统写为全仓回归与 13 项独立前瞻研究 | 在 `errata_and_sha256_ledger_20261116.json` 与 `04_SOURCE_AND_RECEIPT_MANIFEST.json` 逐份标注 `study_role` 与 `independent_physical_sources`，并区分专项套件（`58/58`）与全仓收集规模（`17,520` 项） | `errata_and_sha256_ledger_20261116.json`；`04_SOURCE_AND_RECEIPT_MANIFEST.json` |
| **`F17`** | **P2** | 将强档案基线重构等价方法称为“同构病灶” | 撤回该表述；允许 `archive_full_tools` 合法按需重构等价承诺并计费，在小库（$M \le 4$）如实报告平手，在大规模冲突库（$M = 36, 72$）通过 `v10.3/v10.4` 实测检验真实规模分化 | `equal_tool_attribution_20261116.json`；`library_scaling_curve_20261117.json`；`occam_minimality_ablation_20261117.json` |
| **`F18`** | **P2** | 将循环格点平移称为全群 $\operatorname{Aut}(\mathbb{T}^2)$、跨有限网格称为连续 PDE 极限 | 全部收紧为有限离散格点／图动力系统及离散周期平移子群 $\mathbb{Z}_{N_x}\times\mathbb{Z}_{N_y}$ 不变性 | `01_UNIFIED_AUDIT_DOSSIER_v10_2_to_v10_4.md` |
| **`F19`** | **正向** | 完整拟合列跨元数实例化、零权重列保留与前提级局部修订 | 完整保留，并在 `v10.2–v10.4` 补齐 `_unique_task_signature`、联合版本空间亚微米重分割、`PremiseCommitmentIndex` 与奥卡姆极小性四层门控 | `frozen_v4_lifecycle_20261116.json`；`occam_minimality_ablation_20261117.json` |
| **`F20`** | **正向** | 同工具 `602 -> 603` 搜索预算下降 `66.47%` 与零额外 LLM 跨元数复用 | 完整保留，严格限定为 `SearchBudget` 记账算力下降及独立源／目标双账本 | `v10_cow_602_603_new_ledger_20261116.json` |

---

## 二、`02_下一施工合同.md` 第 5 节 22 项行为验收逐条核验表

全部 22 项行为验收均由 `tests/test_v10_2_audit_acceptance.py` 自动化锁定（零网络、零付费调用）：

| 验收号 | 合同要求摘要 | 实现函数 / 检查点 | 自动化验收测试函数 |
| :--- | :--- | :--- | :--- |
| **#1** | 同一任务语义不能同时得到认可的可达证人与无条件不可达结论 | `SymbolicProofKernel.verify_solvability_certificate` + `validate_control_task_semantics` | `test_acceptance_01_and_02_gpt6_domain_counterexample_never_accepts_opposite_certificates` |
| **#2** | 报告给定盒域反例必须由合法性检查或未证域状态正确处理 | `physical_legality` 拒证人、`analysis_region` 返 `conditional_impossibility_domain_unverified`、`unbounded` 拒盒乘子 | `test_acceptance_01_and_02_gpt6_domain_counterexample_never_accepts_opposite_certificates` |
| **#3** | 初态域外、动作导致域外、负成本、维数不合与目标截断不被静默接受 | `validate_control_task_semantics` + `BudgetedControlLab.execute_trace` | `test_acceptance_03_and_04_malformed_inputs_and_rationalization_audit` |
| **#4** | 有理化改变数值时，输出所证明的实际有理模型及其范围 | `rationalize_model_with_audit` 记录 `exact_input_preserved`、`max_rationalization_delta`、`certified_rational_model` | `test_acceptance_03_and_04_malformed_inputs_and_rationalization_audit` |
| **#5** | 不含定理的 Lean 文本不能标成 `proof_verified` | `lean_status="draft_only"`, `lean_compiled=False`, `proof_verified=false in Lean` | `test_acceptance_05_06_07_08_lean_draft_finite_arities_cone_obligations_and_model_origin` |
| **#6** | 有限 $N$ 核验不能返回 `all_arity_proved` | `all_arity_proved=False`, `verified_on_finite_arities=[...]` | `test_acceptance_05_06_07_08_lean_draft_finite_arities_cone_obligations_and_model_origin` |
| **#7** | 单调性／非负性所需方向和域前提不只存在于注释 | `domain_obligations` 内核强制检查非负状态锥与有向边位移号 | `test_acceptance_05_06_07_08_lean_draft_finite_arities_cone_obligations_and_model_origin` |
| **#8** | `supplied_model` 工具试验和 `learned_model` 真实研究有不同来源标记 | `model_origin` 字段 + `extract_control_model_from_learned_commitment` | `test_acceptance_05_06_07_08_lean_draft_finite_arities_cone_obligations_and_model_origin` |
| **#9** | 确认行、选择行和拟合行可区分；已有真实确认不丢失也不重复计数 | `remember_confirmed_selection` 幂等 `confirmation_ref` + 分离 `fit/selection/confirmation` 行数与误差 | `test_acceptance_09_11_and_f15_confirmation_metadata_idempotency_full_columns_and_ar2_ast` |
| **#10** | 旧世界假设被反驳后其计算可保留，但失效支持不能继续用于当前决策 | `is_commitment_empirically_consistent` + `counterexample_task_ids` + `_unique_task_signature` | `test_acceptance_10_12_18_refutation_scope_local_invalidation_and_alternative_explanations` |
| **#11** | 完整列和参数共享经过保存／恢复／重绑仍保持；零权重列不丢失 | `consolidate_world_structure_commitments` + `instantiate_commitment_on_schema` 保留 `0.0` 系数列 | `test_acceptance_09_11_and_f15_confirmation_metadata_idempotency_full_columns_and_ar2_ast` |
| **#12** | 局部变更使所有真实依赖失效，但无关结果可复用；同名步骤不混淆具体实例 | 前提依赖图局部槽位失效传播 + `commitment_id` 实例级隔离 | `test_acceptance_10_12_18_refutation_scope_local_invalidation_and_alternative_explanations` |
| **#13** | 同工具普通基线能合法调用所有新增求解器和检查器 | `cold_full_tools` / `archive_full_tools` 共享完全相同的 `ResearchConfig` 求解器开关、`certify_rollout_baseline` 及 `occam_premise_minimality` | `test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators` |
| **#14** | 依赖消融不同时给处理组额外角色答案或不同的候选语法 | `run_single_factor_dependency_ablation_arm` 核验双臂 `information_manifest` 除 `violated_premise_cue` 外 100% 一致 | `test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators` |
| **#15** | 无实际调用的对象不生成“使用后无益”标签 | `build_action_chain_telemetry` 严格区分 `no_retained_knowledge_available`、`retained_knowledge_not_invoked`、`invoked_*` | `test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators` |
| **#16** | 未完成、未交付、行政中断、科学反驳分别统计，不给缺失 MSE 填 `0` | `classify_episode_outcome` 保留 `heldout_mse=None` 绝不填 `0.0` | `test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators` |
| **#17** | 噪声孪生误差不被自动输出成真实带噪历史的条件期望风险 | `evaluate_sde_conditional_drift_and_twin_metrics` 分离输出四项独立字段且核验 `after` 传感器零泄露 | `test_acceptance_17_sde_noisy_conditional_drift_vs_noiseless_twin_separation` |
| **#18** | 新代的错误知识不会因 `source_status=supported` 永久通过；未解决替代解释保留 | `is_commitment_empirically_consistent` 在目标端复核实时转移审计并保留 `surviving_bijections` | `test_acceptance_10_12_18_refutation_scope_local_invalidation_and_alternative_explanations` |
| **#19** | 原 API 无私有字段访问只是必要检查；运行时允许信息清单和传输内容也要可审计 | 静态 AST／正则私有属性扫描 + `build_information_manifest` SHA256 载荷审计 | `test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators` |
| **#20** | 所有调用和成本都来自真实事件；公式估算与实测字段分开 | Study B3 分离 `nominal_full_scan_formula_estimate` 与 `measured_probe_queries`；Markdown 表格由回执脚本直接读取生成 | `test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators` |
| **#21** | 来源／目标首个通过者选择如实保留失败分母，不回写原结果 | `frozen_v4_lifecycle_20261116.json` 完整保留 `prospective_chain_2` 失败分母（`3/4` 源形成、`7/12` 后冻结阶段） | `test_acceptance_13_14_15_16_19_20_21_attribution_manifest_budget_and_denominators` |
| **#22** | 测试全通过与科学效果不混称；局部专项通过不称全仓兼容已证 | 专项套件（`58/58`）与全仓收集规模（`17,520` 项）分开报告，并明确区分行为合规验收与前瞻实测通过率 | `test_acceptance_22_ledger_integrity_and_no_v10_evidence_overwrites` |
