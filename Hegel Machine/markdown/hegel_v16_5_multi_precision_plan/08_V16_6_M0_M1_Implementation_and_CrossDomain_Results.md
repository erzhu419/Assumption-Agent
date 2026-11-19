# 08 — Hegel V16.6 `M0 + M1` 施工完成报告与首轮跨领域虫洞实验实测结果

- **施工日期**：2026-11-19
- **基准代码**：`787256bd415515905359626f6a5498174ad5db3c` 上的增量接线（零平行状态机重写）
- **严格对照合同**：`./Hegel_V165_Review_20261119/06_给Gemini的实施合同.md` 及 `./07_V16_6_Review_Audit_and_Horizontal_Wormhole_Architecture.md`
- **全套 12 份实测证据归档目录**：`../../docs/evidence/v16_6_cross_domain_wormhole_20261119/`

---

## 一、M0 底座闭合与源码接线摘要

1. **现有核心模块增量升级（100% 兼容既有测试）**：
   - `src/hegel_machine/cognition/question_semantics.py` & `question_continuation.py`：将 `task_view`、`observation_map`（及 `readout_spec`）、`scale_map`（及 `scale_spec`）、`protocol_spec`、`claim_contract`、`decision_region` 正式纳入规范化语义哈希 `_meaning()` 与 `step_question` 缓存键（自动剥离纯展示字段 `display_label/notes`），修改任一操作契约自动递增 `meaning_version`、归档 `meaning_history`、清空旧 `semantic_answer` 并使下游 `research_candidates` 进入 `needs_rebuild`；同时实现 `evaluate_decision_region`（要求绑定具体 `action_ref`，空候选集强制返回 `MODEL_MISMATCH`）。
   - `src/hegel_machine/cognition/growth_methods.py` & `workflow_execution.py`：新增 `scientific_refutation_witness` 与 `when_outcome` 条件分支编译及重放（支持 `SUPPORTED_IN_SCOPE`、`REFUTED_IN_SCOPE`、`NOT_DETECTED_UNDER_PROTOCOL`、`UNKNOWN` 四向后继分流，严格隔离 `EXECUTION_ERROR` 工具异常不污染科学范围，并保留 `fresh confirm`）。
2. **新增模块化核心文件**：
   - `src/hegel_machine/cognition/wormhole_forest.py`：四正交契约、局部条件蕴含树森林（`PhilosophicalForest`）、**中间层横向虫洞边（`HorizontalWormholeEdge`）**、非传递逐跳误差累加（`compose_wormhole_chain`）、四级使用权限门（`authorize_abstraction_use`）、`ALIGN` 近失例包含/排除谓词修订、`FidelityCompression`（同时校验 `Acc/Rej/Unknown`）vs. `EmpiricalRevision`、干净单版本移除（`without_node_version` 含 `AND/OR` 支持传播且保留原始经历档案）以及 `TwoStageWormholeGate`（含无覆盖新簇有界探索预算）。
   - `src/hegel_machine/cognition/family_plugins.py`：将领域家族与通用内核彻底插件化解耦（`OrderedBoundaryPlugin`、`BottleneckSaturationPlugin` 原观测尺度 Gauss–Newton 拟合 + **CEGAR 联合多变量探针升级**、`RelaxationFamilyPlugin` 局部微分 Schur 界 vs. 多模二次型阶跃及 $\psi = u - Lp$ 修正）。
   - `src/hegel_machine/cognition/cross_domain_clusters.py` & `cross_domain_arms.py`：严格隔离只读外评器的异质过程模拟器与三主臂 + 三消融公平评测框架。
3. **M0 自动化测试与 V15 12 题 Role-A 归因对账**：
   - `tests/test_v16_6_m0_contracts_and_wormholes.py`：覆盖语义身份接线控制、`HEC` 决策区域控制、`verify_multiscale_contract.py` 全部 6 项确定性反例 + 3 项物理反例（共 9 项数学反例）、否证分支重放及保真压缩防误拒测试，**100% 通过**。
   - `docs/evidence/v16_6_cross_domain_wormhole_20261119/m0_v15_12_role_a_cost_attribution.md`：对 V15 12 道 Role-A 目标完成原始事件对账（99.83% 算力差精确落桶），定位 V15 目标超支主因在 `5_confirmation_and_selection_scoring_compute`（`+44,858.9`，占 `75.85%`）与 `4_parameter_rebinding_and_mechanism_search_compute`（`+9,663.3`，占 `16.34%`）。

---

## 二、M1 首轮非物理跨领域实验设计与实测结果（`2 Dev + 4 Frozen Eval Clusters`）

### 1. 实验分割与公平权限（`baseline_permissions.json`）
- **开发/评测严格隔离**：先在 2 个开发簇（`dev_cluster_01, 02`）上采集同状态配对路径成本校准 `TwoStageWormholeGate`，随后**完全冻结策略与门控参数**，在 **4 个独立未见评测簇（`eval_cluster_01..04`，每簇 `5+6+3 = 14` 道异质未见目标，合计每臂 `56` 道未见目标）** 上一次性评测。
- **完全对等权限**：`Hegel_V16_6`、`O_strong`、`O_HExA_inspired` 共享完全相同的 Stage-0 预解、二分查找、前缀验证、开放 1D 函数族语法（含 Occam 参数复杂度准则）及多变量探针原语，拥有完全相同的观测/测量/算力预算与只读外评器。

### 2. 核心配对结果总表（统计单位：`N=4` 独立源学习簇，预注册复合成本 $C = 1.0 \cdot C_{\text{meas}} + 0.05 \cdot C_{\text{compute}}$）

| 配对比较指标 | `eval_01` | `eval_02` | `eval_03` | `eval_04` | 跨簇均值 `mean_diff` | 4/4 簇全负（净省成本）？ |
|---|---:|---:|---:|---:|---:|---|
| **端到端全周期总成本 `Hegel - O_strong`**（含源形成+近失例修订） | `-64.78` | `-64.48` | `-64.78` | `-64.78` | **`-64.70`** | **是（4/4）** |
| **端到端全周期总成本 `Hegel - O_HExA_inspired`** | `-70.18` | `-69.88` | `-70.18` | `-70.18` | **`-70.10`** | **是（4/4）** |
| **未见目标总复合成本 `Hegel - O_strong`** | `-55.43` | `-55.13` | `-55.43` | `-55.43` | **`-55.35`** | **是（4/4）** |
| **未见目标总复合成本 `Hegel - O_HExA_inspired`** | `-57.73` | `-57.43` | `-57.73` | `-57.73` | **`-57.65`** | **是（4/4）** |
| **未见目标纯环境测量费 `Hegel - O_strong`** | `-54.00` | `-54.00` | `-54.00` | `-54.00` | **`-54.00`** | **是（4/4）** |
| **未见目标纯环境测量费 `Hegel - O_HExA_inspired`** | `-54.00` | `-54.00` | `-54.00` | `-54.00` | **`-54.00`** | **是（4/4）** |
| **簇 A（有序离散边界）`Hegel - O_strong`** | `+0.75` | `+0.75` | `+0.75` | `+0.75` | **`+0.75`（诚实持平略贵）** | 否（`O_strong` 允许原生二分） |
| **簇 B（饱和与多资源瓶颈）`Hegel - O_strong`** | `-36.95` | `-36.65` | `-36.95` | `-36.95` | **`-36.88`** | **是（4/4）** |
| **簇 B（饱和与多资源瓶颈）`Hegel - O_HExA_inspired`** | `-50.40` | `-50.10` | `-50.40` | `-50.40` | **`-50.33`** | **是（4/4）** |
| **簇 C（同服务多视图/多尺度）`Hegel - O_strong`** | `-19.23` | `-19.23` | `-19.23` | `-19.23` | **`-19.23`** | **是（4/4）** |
| **簇 C（同服务多视图/多尺度）`Hegel - O_HExA_inspired`** | `-6.83` | `-6.83` | `-6.83` | `-6.83` | **`-6.83`** | **是（4/4）** |
| **横向虫洞消融 `Hegel - Hegel_no_horizontal_wormhole`** | `-57.58` | `-57.28` | `-57.58` | `-57.58` | **`-57.50`** | **是（4/4，证实中间层虫洞核心贡献！）** |
| **对象移除+重建干预 `Hegel - ObjectRemovedRebuild`** | `-29.68` | `-29.38` | `-29.68` | `-29.68` | **`-29.60`** | **是（4/4）** |
| **单源形成反事实 `Hegel - SingleSourceOnly`** | `-39.65` | `-39.65` | `-39.65` | `-39.65` | **`-39.65`** | **是（4/4，证实双源对比归纳虫洞价值！）** |
| **同历史快照比较 `Hegel - O_strong_SameHistory`** | `-55.43` | `-55.13` | `-55.43` | `-55.43` | **`-55.35`** | **是（4/4）** |

- **严格质量与安全性**：全部主臂在 4 个评测簇共 `56` 道未见目标上均取得 **`56/56`（`100%`）严格通过、零假确认（`false_confirmations = 0`）、零非法放宽 `earliest_global_threshold` 契约**。

### 3. 四条关键科学发现（为什么这套结构真正打通了闭环？）
1. **中间层横向虫洞（`HorizontalWormholeEdge`）的独立因果贡献**：关闭跨树中间层横向虫洞（`Hegel_no_horizontal_wormhole`）后，局部树节点被锁定在各自源领域表征内，未见跨域目标复合成本立刻恶化 `+57.50`/簇（测量费恶化 `+54.0`/簇）；单源反事实（`Hegel_single_source_only`）因缺乏双源对比生成跨域虫洞，目标复合成本同样恶化 `+39.65`/簇。
2. **自主 CEGAR 联合探针修订击败并列瓶颈死锁（`T_B3_tied_bottleneck_joint`）**：在并列硬瓶颈 $c_1 u_1 = c_2 u_2$ 上，所有单变量探针增益均为 `0`；Hegel 从近失例 `nm_B1` 自主习得的 CEGAR 联合增量探针以 `5` 次探针（`30.0` 测量费）锁定最优配比，同时优于 `O_strong`（`42.0`）与 `O_HExA_inspired`（`36.0`）。
3. **四契约解耦与权限门精准拦截跨尺度负迁移（`T_C1 / T_C2 / T_C3`）**：对同一结算微服务，Hegel 在 `T_C1`（稳态平均吞吐）合法通过虫洞省下 `18.0` 测量费，在 `T_C2`（突发 p99 尾延迟）凭 `ScaleMap` 闭合检查以 **0 浪费探针**拦截 `window_average` 误用（比试错后补救的 `O_HExA_inspired` 省 `6.0` 测量费），在 `T_C3` 如实执行 `TaskView` 多视图横向路由至离散边界树。
4. **诚实记录无收益簇与弱父节点主动降级**：簇 A 因公平允许 `O_strong` 使用二分原语，Hegel 如实报告 `+0.75`/簇的检索校验微税；在 `T_B5` 无界 $\sqrt{u}$ 上，根节点“单调凹”虽仍成立但无剪枝收益，系统主动将其归档（`archived_low_utility = True`）并零浪费 Clean Exit。

---

## 三、合同 §9 要求的全部 12 份证据文件清单

均已生成并校验于 `Hegel Machine/docs/evidence/v16_6_cross_domain_wormhole_20261119/`：
1. `experiment_manifest.json`
2. `source_episodes.jsonl`
3. `formation_events.jsonl`
4. `hypothesis_versions.jsonl`
5. `alignment_and_scope_checks.jsonl`
6. `refutation_and_revision_events.jsonl`
7. `target_episodes.jsonl`
8. `cost_ledger.jsonl`
9. `baseline_permissions.json`
10. `paired_results.json`
11. `failure_cases.md`
12. `claims_and_evidence.md`
（外加 M0 对账附件：`m0_v15_12_role_a_cost_attribution.json` 与 `m0_v15_12_role_a_cost_attribution.md`）
