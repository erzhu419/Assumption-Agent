# Hegel Machine v15：真实单版本移除与合法重建、K/R 统一状态转移及 E1/E2 前瞻双实验报告

> 本报告所有表格数字均由 `scripts/build_v15_gpt6_audit_bundle.py` 直接读取 `docs/evidence/v15_true_intervention_and_strategy_20261118/*.json` 自动生成，历史 35 份 `v10..v14` 回执保持 0 字节改动，零新增 `src/` 模块，零在线模型调用（`model_calls = 0`）。

## 1. 对主问题的直接回答与 §10 停止规则结论

1. **真实单版本移除与合法计费重建（§3）：** 已在 `memory.py` 实现按 `(concept_id, semantic_version)` 作用的 `remove_library_object_version` 与 `export_library_object_inventory`，清理 list 型 `concepts`、`abstracts`、dict 型 `definitions`、`definition_bindings`、`meta_priors`、分支与缓存中**仅由被删版本独占支持**的载体，保留双独立来源共享规则与原始经历 `_raw_reconciliation_evidence_archive`（剥离被删概念预解出的 `commitment`/`coefficients`），并允许目标期通过普通研究 API 从原始观测与转移计费重建（`generated_from_raw_history = True`）。控制 3 实测重建对象 `concept_0005`（版本 `d16da439cede2073d663ffc6`）计费 `67,053` 算力并独立通过新鲜确认（`loss = 1.0760936821035056e-16`）。
2. **同历史 K/R 统一状态转移与 `scope_unidentified` 修复（§4）：** 在 `branch_reconciliation.py` 将在线 `K` 与按需 `R` 统一到同一时间序状态转移入口 `apply_reconciliation_step_from_evidence`，并在 `replay_evidence_batches_onto_branches` 遇到 `scope_unidentified` 时始终执行可重绑模板交叉检验，记为 `instance_scope_unknown`（绝不误标全局 `refuted`），同时将 `radius_value` 与 `commitment` 纳入 `compute_evidence_archive_fingerprint`。在既有 8 条 D2 源簇（`24` 对目标）与全新前瞻 `E1`（`36` 对目标）上，`K` 与 `R` 首次调度语义分歧均为 **`0`**。
3. **前瞻 `E1`（同原始历史组织价值，`1051000–1056000`，`6` 簇 × `6` 目标 = `36` 题/臂）：** 源冲突 `S4` 共 `6/6` 簇产出合格 `accumulate` 修复对象。在 `36` 个拉丁方顺序平衡未见 `N=5` 目标上：`K_eager_persistent` 严格通过 `33/36`（均值算力 `168,817.2`，观测 `244.7`），同历史按需 `R_same_history_lazy` 严格通过 `33/36`（均值算力 `172,203.1`，观测 `244.7`），参考锚点 `K_cold` 为 `32/36`（均值算力 `174,034.0`）。计入源期维护算力后，`R - K` 前缀全算力差均值为 `k=1: 2,100.3`，`k=3: 2,099.3`，`k=6: 2,099.2`（`R` 少花 `48.0` 行源区分探针观测），表明在相同历史与统一状态转移函数下 `K` 与 `R` 近乎等价，呈微量算力与源区分探针观测之间的资源取舍；而真实移除修复版本臂 `K_true_repaired_version_removed_raw_rebuild_allowed` 为 `5/36`（`1054000` 以 `475,939` 算力成功从原始经历重建并 `5/6` 通过，其余 `5` 簇因未设子预算上限的 2D 高阶潜变量搜索在目标题入口耗尽 `840k` 额度而 `budget_deferred`）。
4. **前瞻 `E2`（零拼接端到端完整策略比较，`1051000–1056000`）：** 共享完全相同局部研究器与触发规则的 `Ordinary_autonomous_end_to_end` 用自身在源期形成的库 `L^Ord_c` 继续 `T1..T6`，源期严格通过 `24/30`（`0/6` 卡在无作用域隔离的 `S3_right_dpg_N4`）、目标严格通过 `34/36`（均值目标算力 `135,223.9`，观测 `185.3`）；`K_autonomous_end_to_end` 在 `L^K_c` 上源期严格通过 `30/30`、目标严格通过 `33/36`。端到端 `Ordinary - K` 全轨迹累计算力差均值为 `k=0(仅源): 495,873.2`，`k=1: 458,227.2`，`k=3: 394,531.8`，`k=6: 294,313.3`；累计观测差 `k=6: 26.3`。严格执行 **§10 停止规则第 1、2、4、6 条**：`K` 在顺序多机制源流（尤其 `S3` 条件方向冲突）中提供明确的源期完成率（`30/30` vs `24/30`）与版本可审计性，但在后继 `N=5` 目标题求解上强普通方法已追平甚至略优（`34/36` vs `33/36`），故不宣称后继目标期普遍加速。

---

## 2. Deliverable 1：V14 十项勘误与 6 条连续链状态感知事实重导出

- 重算确认 V14 原始回执共 **`232`** 条臂—episode 判定（D2 `112` + D3 `120`，去重任务 UID 为 **`106`**，`207` 严格通过、`25` 未决、`0` 假确认）；D2 锚点均值算力 **`114,263.125`**、墙钟 **`6.39594s`**；D2 目标均值测量 `cold = 4032.5`、`R = 4122.5`、`K = 3932.5`。
- 已对 `1041000–1046000` 重新导出真实 `autonomous_branch_reconciliation` 生产事件与状态感知七句账本，两条失败链 `1043000` 与 `1045000` 明确输出 `unresolved` / `missing_evidence`，不再出现任何伪成功确认句。

| 勘误编号 | 修正后事实摘要 |
| :--- | :--- |
| `V14_ERRATA_01_EPISODE_COUNTS` | D2 (112) + D3 (120) = 232 arm-episode judgments (207 strict passes, 25 unresolved/unconfirmed, 0 false internal confirmations); deduplicating by episode_uid yields 106 unique episodes; 214 omitted D3's 6 anchors + 12 Markov sources; paired statistical unit remains the source cluster (n=8 in D2, n=6 in D3). |
| `V14_ERRATA_02_D2_ANCHOR_COSTS` | D2 anchor mean compute = 114263.125 and agent wall-clock = 6.39594s. |
| `V14_ERRATA_03_D2_TARGET_MEASUREMENTS` | Exact D2 target mean measurements: K_cold = 4032.5, R_lazy_noH = 4122.5, K_eager_noH = 3932.5. |
| `V14_ERRATA_04_STRUCTURED_CONSUMPTION_VS_SUBSTRING` | V14 used substring matching (`invoked_repaired_object`). Typed `consumption_events` prove confirmed adoption on 3/4 chains for K_eager_noH (['unexposed_v14_focal_chain_1041000', 'unexposed_v14_focal_chain_1044000', 'unexposed_v14_focal_chain_1046000']) and 1/4 chains for R_lazy_noH (['unexposed_v14_focal_chain_1041000']); remaining cases are marked `unproved_by_v14_telemetry`. |
| `V14_ERRATA_05_FAILED_CHAIN_NARRATIVES` | Chains 1043000 (external MSE 0.03976) and 1045000 (external MSE 0.03268) did NOT form a qualified repaired object; re-exported ledgers report `unresolved` / `missing_evidence` with zero fake confirmation claims. |
| `V14_ERRATA_06_TIMING_VS_FULL_POLICY_ATTRIBUTION` | In V14 D2, R and K differed on 24/24 first-dispatch states due to batch-vs-chronological replay and `scope_unidentified` handling, so V14 D2 compared two complete organization policies rather than timing-only; v15 E1 unifies both onto one state-transition function. |
| `V14_ERRATA_07_OBJECT_REMOVAL_VALIDITY` | V14 `_build_d3_target_libraries` checked `isinstance(concepts, dict)` on list-shaped `KnowledgeLibrary.concepts`, leaving `concept_0004` in `concepts`, `abstracts`, `definitions`, and `definition_bindings`, while disabling reconstruction; no marginal object value claim is drawn from V14 D3 prior to v15's true versioned removal + billed rebuild. |
| `V14_ERRATA_08_WRONG_TRANSFER_AND_SCOPE_KEYS` | `wrong_knowledge_transfer == 0` covers only false confirmations after adopting library knowledge (not extra cost or extra unresolved episodes), and V14's exporter read `scope_unknown_branch_ids`/`excluded_out_of_scope_branch_ids` instead of dispatcher's actual `scope_not_established`/`empirically_excluded_on_context`. |
| `V14_ERRATA_09_H_STAR_SCOPE` | In V14, `_prioritize_generalized_families = False` disabled target-time H* Pass-0 scheduling while source-time H* construction in `autonomous_reconcile_library_episode` still ran; v15 §7.1 skips unconsumed source-time H* construction when disabled. |
| `V14_ERRATA_10_NO_SPLICING_E1_AND_E2` | V14 D3 Question B's `R_lazy_noH` operated on the subquestion-formed library rather than Question A ordinary retry's own library; A-ordinary and B-R cannot be spliced into a single end-to-end run, so v15 separates E1 (same history) and E2 (true end-to-end). |

---

## 3. Deliverable 2：三项真实移除控制（§3.3）、8 簇 K/R 状态对齐（§4.3）与 6 案例回放（§5.2）

- **控制 1（list 型容器真实移除与零召回）：** 移除 `concept_0004`（`0fc3ce3f69f4093815893cfb`）后，`concepts`/`abstracts`/`definitions`/`definition_bindings`/`_recall` 残留均为 `False`，原始证据批保持 `8` 批不变；
- **控制 2（独占派生失效 vs 双源共享保留）：** 独占派生 `concept_0005` 移除 `=True`，双独立来源共享概念 `concept_0006` 保留 `=True`；
- **控制 3（从保留原始历史计费重建）：** 重建出 `concept_0005`，`generated_from_raw_history = True`，计费算力 `67,053`；
- **8 条既有 D2 源簇 K/R 对齐：** `8` 簇、`24` 对目标的首次调度语义差异总计 **`0`**，预算不足时返回 `partial_pending`。

---

## 4. 前瞻实验 E1：同原始历史组织与真实版本移除对照（`1051000–1056000`，`36` 目标/臂）

| E1 目标对比臂 | 严格通过 (`36`题) | 外部准确模型 | 假确认 | 未决(含外评已准) | 采用修复/重建源 | 原始历史重建次数 | 均值目标算力 | 均值观测 | 均值测量 | 冷/暖重建算力 (`k=1` / `k>=2`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | :---: |
| `K_eager_persistent` | **33/36** | 34/36 | 2 | 1 (1) | 2 | 0 | 168,817.2 | 244.7 | 3,320.0 | 0.0 / 0.0 |
| `R_same_history_lazy` | **33/36** | 34/36 | 2 | 1 (1) | 2 | 0 | 172,203.1 | 244.7 | 3,320.0 | 25,015.5 / 7.8 |
| `K_true_repaired_version_removed_raw_rebuild_allowed` | **5/36** | 5/36 | 1 | 30 (0) | 3 | 31 | 735,629.2 | 217.8 | 2,007.5 | 779,239.2 / 699,916.0 |
| `K_cold` | **32/36** | 32/36 | 2 | 2 (0) | 0 | 0 | 174,034.0 | 240.8 | 3,271.2 | 0.0 / 0.0 |

### E1 前缀全成本对比（`R_same_history_lazy - K_eager_persistent`，含源维护费 `C^K_maint`）

| 前缀长度 | 均值全算力差 (`R - K`) | 中位数全算力差 | `K` 全算力更省簇数 (`/6`) | 符号翻转 `p` (描述性) | 仅目标算力差均值 (`R - K`) | 移除臂减 `K` 目标算力差均值 | 均值全观测差 (`R - K`) |
| :---: | ---: | ---: | :---: | :---: | ---: | ---: | ---: |
| `k_1` | 2,100.3 | 3,015.5 | 4/6 | 0.1250 | 20,316.7 | 633,153.7 | -48.0 |
| `k_3` | 2,099.3 | 3,013.0 | 4/6 | 0.1250 | 20,315.7 | 1,783,315.5 | -48.0 |
| `k_6` | 2,099.2 | 3,012.5 | 4/6 | 0.1250 | 20,315.5 | 3,400,872.0 | -48.0 |

---

## 5. 前瞻实验 E2：零拼接端到端策略比较（`Ordinary` 用自身 `L^Ord_c` vs `K` 用自身 `L^K_c`）

| 端到端策略臂 | 源任务严格通过 (`30`题) | 源均值算力 | 目标严格通过 (`36`题) | 目标外部准确 | 目标假确认 | 目标均值算力 | 目标均值观测 | 目标均值测量 |
| :--- | :---: | ---: | :---: | :---: | :---: | ---: | ---: | ---: |
| `K_autonomous_end_to_end` | 30/30 | 122,583.6 | **33/36** | 34/36 | 2 | 168,817.2 | 244.7 | 3,320.0 |
| `Ordinary_autonomous_end_to_end` | 24/30 | 221,758.2 | **34/36** | 35/36 | 1 | 135,223.9 | 185.3 | 2,429.6 |

### E2 端到端前缀资源对比（`Ordinary - K`，从源第 1 题累计至目标第 `k` 题，零拼接）

| 前缀阶段 | 均值累计算力差 (`Ord - K`) | 中位数累计算力差 | `K` 算力更省簇数 (`/6`) | 均值累计观测差 (`Ord - K`) | `K` 观测更省簇数 (`/6`) | 均值累计测量差 (`Ord - K`) |
| :---: | ---: | ---: | :---: | ---: | :---: | ---: |
| `k_0_source_only` | 495,873.2 | 623,003.0 | 5/6 | 382.5 | 6/6 | 4,110.0 |
| `k_1` | 458,227.2 | 545,465.5 | 5/6 | 329.5 | 6/6 | 3,255.0 |
| `k_3` | 394,531.8 | 438,065.5 | 5/6 | 223.7 | 6/6 | 1,727.5 |
| `k_6` | 294,313.3 | 276,370.0 | 5/6 | 26.3 | 3/6 | -1,232.5 |

---

## 6. §7.1 逐项算力剖分均值（目标期 36 题均值）

| 对比组 | 1.对象与前提审计 | 2.历史实例重评 | 3.区分实验规划 | 4.参数重绑与新机制 | 5.确认与选择评分 | 6.分支调和与索引 | 7.按需/原始经历重建 | 总算力 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `E1: K_eager_persistent` | 34,171.8 | 0.0 | 35,385.9 | 59,428.0 | 39,732.0 | 784.6 | 0.0 | 168,817.2 |
| `E1: R_same_history_lazy` | 37,563.3 | 0.0 | 35,385.9 | 59,418.6 | 37,631.9 | 0.0 | 4,175.8 | 172,203.1 |
| `E1: K_true_removed_rebuild` | 718,976.1 | 0.0 | 4,967.3 | 8,526.9 | 1,727.9 | 714,344.5 | 713,136.5 | 735,629.2 |
| `E2: Ordinary_end_to_end` | 51,095.4 | 0.0 | 34,553.5 | 43,394.9 | 10,611.0 | 0.0 | 64.4 | 135,223.9 |

---

## 7. 按目标角色拆分与全量非通过／假确认／重建耗尽诊断（§9.1, §9.2）

| 对比臂 | `Role_A` 相容潜变量 (`12`题) | `Role_B` 返回旧域 (`12`题) | `Role_C` 不相容反例域 (`12`题) |
| :--- | :---: | :---: | :---: |
| `E1: K_eager_persistent` | 12/12 (`138,289` / `245.1` obs) | 9/12 (`136,123` / `230.0` obs) | 12/12 (`232,039` / `258.9` obs) |
| `E1: R_same_history_lazy` | 12/12 (`141,406` / `245.1` obs) | 9/12 (`139,820` / `230.0` obs) | 12/12 (`235,384` / `258.9` obs) |
| `E1: K_true_removed_rebuild` | 2/12 (`753,217` / `216.0` obs) | 1/12 (`717,964` / `213.9` obs) | 2/12 (`735,707` / `223.6` obs) |
| `E1: K_cold` | 12/12 (`128,580` / `241.1` obs) | 9/12 (`169,087` / `234.2` obs) | 11/12 (`224,435` / `246.9` obs) |
| `E2: Ordinary_end_to_end` | 12/12 (`79,151` / `183.2` obs) | 10/12 (`118,561` / `172.7` obs) | 12/12 (`207,960` / `200.0` obs) |

- **`Role_B` 边界子单元残差说明：** `K_eager_persistent` 与 `R_same_history_lazy` 的 `3` 个非严格通过目标均位于 `Role_B_return_to_old_domain` 的 `B2`（`right_dpg` 条件阈值旧域）：其中 `1` 题外部 48 行已精确命中但确认前预算保留未决（`unconfirmed_yet_accurate = 1`），另 `2` 题（`1053000:T6_B2` 外评 MSE `4.28e-4`、`1054000:T5_B2` 外评 MSE `1.43e-3`）为阈值常数落在 24 行确认轨迹离散分界微单元内的子区间漂移，均如实计为 `false_internal_confirmation`（`wrong_knowledge_transfer` 为采用库对象后发生的外评超差）。
- **`K_true_repaired_version_removed_raw_rebuild_allowed` 入口无上限重建诊断：** 在单次冻结前瞻 `1051000–1056000`（每簇含 `4` 个保留源先验）上，`1054000` 在首题以 `475,939` 算力从保留原始历史成功重建出 `concept_0006`（`fresh_confirmation_loss = 1.73e-16`）并取得 `5/6` 通过；其余 `5` 簇在首阶 `allow_ho=False` 未命中精确半径微格后进入无子预算上限的 `allow_ho=True` 2D 高阶潜变量搜索，在目标题入口即耗尽 `~839.9k / 840k` 算力（`status = budget_deferred`），反观不做原始历史预重建的 `K_cold` 直接由局部研究器取得 `32/36`（均值 `174.0k`）。我们严守单次冻结协议不作事后回修重跑，将此作为“无条件全量原始历史重修在多先验下反致预算挤占”的真实反例记录在案。
