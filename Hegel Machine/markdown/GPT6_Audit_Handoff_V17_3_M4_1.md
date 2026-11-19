# Hegel Machine `V17.3 (M4.1)` 独立红队复核交接书（GPT-6 Audit Handoff）

> **仓库基线与静态证据只读说明**：本复核包构建于前序已审计提交 **`11961f702739ab272661ff59b77fb5bc8e3589b8`** 之上，逐项回应 `markdown/Hegel_M4_Audit/GPT_diagnosis_M4.md`、`M4_AUDIT.md` 及三套审计探针（`m31/probe_boundaries.py`、`author/probe_author.py`、`multiscale/probe_multiscale_lifecycle.py`）。前序全部静态证据目录（`docs/evidence/v16_8_*` 至 `docs/evidence/v17_2_*`）及两套红队审计目录（`markdown/hegel_M3_Audit/`、`markdown/Hegel_M4_Audit/`）均保持 **100% 原始字节级只读未改动**；本轮 `V17.3 (M4.1)` 全新静态证据独立归档于 `docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120/`。

---

## 1. 复核范围、源码 SHA-256 清单与证据地图

本轮 `V17.3 (M4.1)` 包含紧密咬合的四项工程与实验交付：
1. **Part 1 — `Hegel_M4_Audit` 全量 11 项代码边界、生命周期接线与统计口径闭环**：
   - 闭合单文件重复 `def predict` / 模块属性遮蔽（`math.exp = ...`）AST 纯度漏洞、无收据且省略 `target_id` 时的布尔越权旁路、工作区内绝对路径沙箱确认映射、安全低负载失败后单标志实证更新、四参数组合算子 `exp_relax -> oscillatory_damping -> additive_shunt` 参数撞号（`p0..p3` 严格唯一连续）、`authored_method.json` 防伪完整性核验（拦截未确认 `metadata_only` 与篡改 `operator_spec`）、`program_steps` 合法指令解释器、两跳游走子路径 `c1_eval` 的 Tier-3 终审与未知前提拦截、`FidelityCompression` 验证结果正式提交至 keeper 及同维异律冲突拦截，以及节点修订后纵向 `coarse_graining_edges` 存活保持与显式复核恢复（`6 -> 4 -> 6`）。
2. **Part 2 — 完整四格 `M3.1` $2\times 2$ 检索因子分解实验（补齐第 4 格 `Alpha0_NoPool`）**：
   - 在 `4` 个冻结评测簇（`m3_1_eval_01..04`）上完整跑满 `{Walk (alpha=0.65), NoWalk (alpha=0.0)} × {WormholePool, NoPool}` 全部四格（每格 `124` 次工作区运行，四格合计 `496` 次运行）。
3. **Part 3 — 可执行动力学 $\Pi_s$ 快慢轨迹重整化与区域多点盒闭合证书（`ExecutableFastSlowRenormalizationLab`）**：
   - 底层使用确定性四阶 Runge-Kutta（`dt = 0.01`）积分双时间尺度刚性微分方程 $(x_{\text{slow}}(t), x_{\text{fast}}(t))$，耦合可执行 adiabatic Schur 消元粗微分方程（`dt_coarse = 0.05`）与轨迹尾窗慢流形投影算子 $\Pi_s$；
   - 以区域多点盒探针签发带显式 `verified_input_box` 边界的 `ScaleMap` 闭合证书（拦截盒外查询并捕获单点探针漏检的域内局部微观共振凸起），并通过 `With_CoarseGraining_Edges` vs `Without_CoarseGraining_Edges` 因果删除对照证明 `coarse_graining_edges` 在 `HippoRAGWormholeIndex` 中具有不可替代的跨尺度寻路效用（`72.61` vs `97.78`，删除后成本上升 `+34.7%`）。
4. **Part 4 — 端到端统一持续发现—入林—自主检索—生命周期链（`Stage 0 -> Stage 1 -> Stage 2 -> Stage 3`）**：
   - 全部 6 个评测臂（含零删点公平 `Frozen_After_Stage0`）从 **100% 字节级一致的 `Stage 0` 纯原子森林**出发；
   - `Stage 1` 由原生 `CodeAgent` 从原子算子出发经残差 `stdout` 动态合成 **3 类互异复合机理**（`Haldane+Shunt`、**4 参数** `Oscillatory+Shunt`、`Murray+Cap`），经 `load_verified_authored_method` 防伪校验后直接注册为 `PhilosophicalForest` 正式节点并完成 **3 层 `R3` 旧作用域保守性核查**；
   - `Stage 2` 执行在线 `FidelityCompression` 落库，并通过 **3 轮真实付费 DEV 子进程隔离确认失败**淘汰高词频表面诱饵（彻底移除合成 `record_utility_feedback` 循环）；
   - `Stage 3` 在 **零宿主 `lens_hint`**（完全由付费试探数据自主标定坐标置换与仿射透镜）、均衡覆盖全部 3 类新复合机理的 `4` 随机种子未见目标集（每臂 `48` 域内 + `8` OOS = `56` 次运行，六臂合计 **`336` 次运行**）上证明：`Hegel_M4_1_UnifiedForest`（`36.3477`）**同时严格击败逆序全历史 `Ordinary_AllHistory_Reversed`（`38.5167`）与正序全历史 `Ordinary_AllHistory_Forward`（`38.9250`）**，以及 `Forest_Alpha0`（`41.6863`）、`Flat_BM25`（`41.7020`）和公平 `Frozen_After_Stage0`（`48.1063`）。

### 核心源码、脚本、测试与报告 SHA-256 清单

| 相对仓库路径 (`Hegel Machine/`) | 角色说明 | `SHA-256` |
| :--- | :--- | :--- |
| `src/hegel_machine/cognition/wormhole_forest.py` | 无收据无 `target_id` 拦截、`compress_compatible_nodes` 落库提交与异律冲突拦截、纵向边失效标记与 `revalidate_vertical_edges_after_revision` | `f686f33eb7dcf6e5e14342628e665a0354a14d4330a1b0cea1a0715aa4db22b0` |
| `src/hegel_machine/cognition/open_workspace_labs.py` | `_normalize_workspace_relative_model_path` 工作区内绝对路径规范化与越界拦截 | `f956fa62c99dc7de95f77d5c5cfefd11fb75f06ddb886a64d6bcc3f4c04f71fa` |
| `src/hegel_machine/cognition/open_workspace_wormhole.py` | 重复 `def predict / fit` 与非 `ast.Name` 模块属性赋值纯度拦截、确认沙箱绝对路径规范化 | `811c4fd4486e17b3a1aa8bcc4ea9195e1dec73922b39cbe32eec18c4eb070af2` |
| `src/hegel_machine/cognition/hipporag_wormhole_index.py` | 图构建跳过 `stale_endpoint_revision == True` 的 `local_implication_edges` 与 `coarse_graining_edges` | `b3f7e1b1d89eeda49de2d5e9cc007979de816560e28d38c438fde8bda83f18b4` |
| `src/hegel_machine/cognition/m3_scale_universe.py` | 安全低负载确认失败单标志实证更新、新增 `hipporag_alpha0_nopool` 检索模式 | `68f04bb7409876c8bf40d8f09a4d2aeaf2c2aaa867c8ccc8643b2704a7041c85` |
| `src/hegel_machine/cognition/m3_1_walk_factorization.py` | 支持 `hipporag_alpha0_nopool` 完整 $2\times 2$ 四格检索因子分解 | `082996c9901c74fe9495c94998803c91011ffd66478f3c890b23a86b9dc84b92` |
| `src/hegel_machine/cognition/m4_compositional_discovery.py` | `_iter_child_exprs` 覆盖 `cos_mul.param`、`strict_unique=True` 四参数无撞号编译、保留 `multi_start_param`、`load_verified_authored_method` 防伪校验 | `2ac0c704f0d20c5e94a2535e828101fbb525f06b99886cfd66d8a0164cd89e1e` |
| `src/hegel_machine/cognition/m4_multiscale_renormalization.py` | `_validate_and_interpret_m4_program_steps` 指令解释器、子路径 `c1_eval` Tier-3 终审、多起点双向符号透镜标定、纵向边复核接入 | `b73014d08b4e63bbdadf0fb9688f733e9bead9ff58c6ebce90eb1da2b0138d25` |
| `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py` | **新增核心模块**：RK4 快慢轨迹 + 可执行 Schur $\Pi_s$ 区域闭合实验室、端到端统一 `Stage 0 -> Stage 3` 持续发现入林与 6 臂无提示迁移链 | `fb154c373b29583edf9c2ac5d02b48a2e61986c33d7dd4e1d047c131309c0f9e` |
| `scripts/run_v17_3_m4_1_unified_study.py` | `V17.3 (M4.1)` 四部分全量复算、持久化工作区归档与 `SHA256SUMS.txt` 生成脚本 | `b57f14f992d4e869e96470bb28a81d82ef6f5c9923d588185a7fe45bf21cd738` |
| `tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py` | `Hegel_M4_Audit` 全部边界探针回归、$2\times 2$ 四格、可执行 $\Pi_s$ 与统一持续链测试 | `78a64188772c7b29c8972f1d72542462f79d04894035a1868257a9b217291dfa` |
| `markdown/hegel_v16_5_multi_precision_plan/15_V17_3_M4_1_Audit_Closure_Executable_Pi_and_Unified_Chain_Report.md` | `V17.3 (M4.1)` 完整技术与实验报告 | `786fef21029ba76328415d66477956811032f0066782cf3f24a2d87ce4848f89` |

### 新增只读证据包导航
- **`V17.3 (M4.1)` 证据目录**：`docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120/`
  - `summary.json`（总览指标与全生命周期账本）
  - `part1_audit_boundary_and_lifecycle_closures.json`（11 项审计探针闭环复验）
  - `part2_m3_1_complete_2x2_retrieval_factorization.json`（`496` 次 $2\times 2$ 四格检索实测）
  - `part3_executable_pi_s_fast_slow_renormalization.json`（RK4 快慢轨迹、区域盒闭合与 `coarse_graining_edges` 因果删除对照）
  - `part4_unified_continual_discovery_forest_chain.json` + `per_target_receipts_336_runs.jsonl`（4 种子 × 14 目标 × 6 臂逐题收据与成本明细）
  - `persisted_stage1_authored_workspaces/`（完整保留 `S1_monod_parent`、`W1_haldane_shunt`、`W2_osc_shunt_4p`、`W3_murray_capped` 四个 `CodeAgent` 发现工作区的 `authored_method.json`、`authored_model.py`、`confirmed_training_observations.json`）
  - `SHA256SUMS.txt`（本次发布的 `52` 个归档文件的逐字节校验和）

---

## 2. 第一部分：`Hegel_M4_Audit` 11 项工程边界与统计口径逐项闭环表

| 审计编号与原探针定位 | 根因修复机制 | 临时目录复跑原探针 & 回归测试结果 |
| :--- | :--- | :--- |
| **B1. 重复定义 `predict` 与 `math.exp` 属性遮蔽**<br>`m31/probe_boundaries.py` | `canonicalize_python_model_ast` 计数顶层 `predict / fit` 定义（`>1` 判为 `UNSUPPORTED_IMPURE_PYTHON`，摘要 `impure:<sha256>`），并拦截所有非 `ast.Name` 的属性/下标顶层赋值及任意裸语句。 | `duplicate_predict_definition` 与 `module_attribute_shadowing` 均为 `same_digest = False`、`statuses = ["UNSUPPORTED_IMPURE_PYTHON", "UNSUPPORTED_IMPURE_PYTHON"]`。 |
| **B2. 无收据且省略 `target_id` 时布尔放行**<br>`m31/probe_boundaries.py` | `authorize_abstraction_use`：凡 `goal_kind == "open_workspace_mechanism_prediction"` 或 `family_plugin.endswith("_ast")` 且 `evidence_receipt is None`，无条件置 `confirmed = False`。 | `no_receipt_no_target_boolean_true` 返回 `granted = False`（降级回 `bounded_trial`）。 |
| **B3. 确认沙箱绝对路径边界**<br>`m31/probe_boundaries.py` | `open_workspace_labs.py`、`open_workspace_wormhole.py`、`m4_compositional_discovery.py` 统一将工作区内绝对路径转为相对沙箱路径；越界路径直接抛 `ValueError`。 | `confirmation_rmse: {"relative_path": 9.0, "absolute_path": 9.0}`（此前 `absolute_path` 为 `null`）。 |
| **B4. 安全试探高负载 4 标志广播**<br>`m3_scale_universe.py:2400` | 按 `(active_arity, symmetric_roles, sig:low_regime_*_onset)` 严格仅置位当前被证伪起始类对应的**唯一单个**高负载标志。 | 消除跨机理标志广播污染，`M3.1` 全套测试保持 `100%` 通过。 |
| **B5. `osc -> shunt` 四参数撞号 `p2`**<br>`author/probe_author.py` | `_iter_child_exprs` 统一遍历含 `cos_mul.param` 的全部子节点，`_collect_params(..., strict_unique=True)` 强制连续唯一索引，且 `additive_secondary_shunt` 继承子算子 `multi_start_param`。 | `expected_independent_params = 4, actual_distinct_params = 4`（`_PARAMS = {'p0': 1.5, 'p1': 1.2, 'p2': 1.8, 'p3': 0.6}`，`VERIFIED_PURE_AST`）。 |
| **B6. `metadata_only` 与篡改 `operator_spec` 产物校验**<br>`author/probe_author.py` | `authored_method.json` 绑定 `authored_model_source` 与 `confirmation_digest`；`load_verified_authored_method` 核验 `confirmed_ok is True`、代码 SHA256、纯 AST 骨架摘要及重编译一致性。 | `metadata_only_rejected_properly = True`；篡改 `operator_spec` 保留旧 `.py` 同样被拒绝召回。 |
| **B7. `program_steps` 未核验未知 opcode**<br>`multiscale/probe_multiscale_lifecycle.py` | `_validate_and_interpret_m4_program_steps` 校验受支持 opcode 集合、`operator_tag` 与节点动作一致、`active_arity` 一致及 `fresh_points >= 1`。 | 原探针 `nonsense_all_equal = False`（全部 5 题 `confirmed_via_recall: True -> False` 且回退开销真实上升）。 |
| **B8. 子路径 `c1_eval` 跳过 Tier-3 终审与未知前提检查**<br>`multiscale/probe_multiscale_lifecycle.py` | 子分支仅从实测 `pilot + c0_rows` 更新观测特征，并强制通过 `authorize_abstraction_use(requested_tier="local_solver_guidance", evidence_receipt=...)` 终审。 | 原探针 `child_recall_despite_final_deny = False`；`unverified_child_scope_bypass`（追加未测前提 `audit_boundary_measured`）返回 `confirmed_via_recall = False`。 |
| **B9. `FidelityCompression` 未落库及同维异律误合**<br>`multiscale/probe_multiscale_lifecycle.py` | ① `fidelity_compress_nodes` 拦截重叠 `ACC` 下异律节点合并（`law_a != law_b` 记入冲突）；② `compress_compatible_nodes` 将验证通过的 `claim_contract`（含 `disjunctive_branches`）注册提交至 keeper 并执行逐查询后验断言。 | `fidelity_evaluator_wrong_law_control.accepted = False`（冲突索引 `[0, 1]`）；`fidelity_committed_node_vs_validated_node`：`validated_merged_B == committed_keeper_B == "ACC"`。 |
| **B10. 节点修订后纵向 `coarse_graining_edges` 失活**<br>`multiscale/probe_multiscale_lifecycle.py` | 纯协议修订（`ScaleMap` 与定律指纹未变）自动保持粗粒化边绑定新版本（`6 -> 6`）；当 `ScaleMap` 或定律变更时标 `stale_endpoint_revision = True`（`6 -> 4`），并通过 `revalidate_vertical_edges_after_revision` 复核恢复（`4 -> 6`）。 | 协议修订后 `coarse_graining_edges = 6`；`ScaleMap` 变更后 `4` $\to$ 显式复核后恢复 `6`。 |
| **B11. `M3.1` 第 4 格补齐与 `13_V17_1` 报告 `C4` 勘误**<br>`GPT_diagnosis_M4.md` | 新增 `hipporag_alpha0_nopool` 跑满四格；正式勘误说明 `13_V17_1` 中 `C4`：`Hegel` 目标均值 `53.225` 对比公平同 Pilot 全历史普通臂 `57.075`（净省 `-3.85`），而 `74.70` 系同表 `without_node_version` 节点删除消融对照。 | 账本口径与四格矩阵 100% 闭合对齐。 |

---

## 3. 第二部分：完整 `M3.1` $2\times 2$ 四格检索因子分解结果

> **计数透明说明**：每个冻结评测簇中 `suite["targets"]` 含 `27` 题（含 `T01..T24` 及簇内 `3` 道桥接题），外加 `suite["walk_targets"]` 的 `4` 道独立安全试探题 `T25..T28`（其中 `T25..T28` 在汇总列表中合计评估 `8 × 4 = 32` 次），故每格总运行数为 `(27 + 4) × 4 = 124` 次，四格合计 `496` 次目标工作区运行：

| 因子格（`Walk × WormholePool`） | 总运行数 | `strict_pass` | `T01..T06` 跨域桥接复合成本 | `T25..T28` 纯图游走召回胜出 | `T25..T28` 复合成本 | 全集平均测量费 | 全集平均计算费 | **全集平均复合成本** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Walk_Pool` ($\alpha=0.65$, `Pool=True`)** | `124` | `1.0000` | **`39.9702`** | **`28 / 32` (`87.5%`)** | **`44.7579`** | **`36.4516`** | **`38.9854`** | **`38.4009`** |
| `Walk_NoPool` ($\alpha=0.65$, `Pool=False`) | `124` | `1.0000` | `40.8660` | `28 / 32` (`87.5%`) | `44.7579` | `36.6129` | `39.8725` | `38.6065` |
| **`Alpha0_NoPool` ($\alpha=0.0$, `Pool=False`，补齐第 4 格)** | `124` | `1.0000` | `39.8483` | `0 / 32` (`0.0%`) | `64.3025` | `40.4839` | `58.0323` | `43.3855` |
| `Alpha0_Pool` ($\alpha=0.0$, `Pool=True`) | `124` | `1.0000` | `39.9317` | `0 / 32` (`0.0%`) | `64.3025` | `40.4839` | `58.3548` | `43.4016` |

---

## 4. 第三部分：可执行 $\Pi_s$ 重整化与端到端统一持续发现链的精确边界披露

### 4.1 可执行动力学 $\Pi_s$ 实验台（`ExecutableFastSlowRenormalizationLab`）的精确主张与边界
- **真实动力学投影**：细通道 `micro_fine`（`12.0/点`）以 RK4（`dt = 0.01`）真实积分双时间尺度刚性 ODE；粗通道 `coarse_macro`（`4.0/点`）执行快变量绝热消元（$\tau_f \dot{x}_f \to 0 \implies x_f = a_f u - g_{fs} x_s$）后的 1D Schur 慢方程积分及轨迹尾窗慢流形投影。
- **因果边与区域闭合实测**：
  - 在慢流形绝热盒 `[2.2, 5.2]` 上，`With_CoarseGraining_Edges` 沿 `ode_micro_fast_port --Pi_s--> ode_meso_schur_reduced` 游走命中粗算子（复合成本 **`72.6095`**），严格优于删除粗粒化边的 `Without_CoarseGraining_Edges`（**`97.7765`**，失去召回、评估 5 候选）及 `MicroOnly_AlwaysFine`（**`108.6095`**）；盒外查询点（$T=0.12 \notin [2.2, 5.2]$）被 `is_point_inside_verified_box` 严格拦截。
  - 在 $T \approx 3.55$ 含局部高斯微观共振包的反例上，单点边界探针发生假确认（`false_confirmation = True`），而 3 点区域盒探针捕获内部缺陷 `0.8584 >> 0.06`，阻断粗闭合证书并诚实返回 `scoped_unknown`（`false_confirmation = False`）。
- **主动诚实边界说明**：区域多点盒探针覆盖四角与内部中心分位点，能有效捕获有限宽度的域内非绝热破裂；但在无先验全局解析 Lipschitz 常数假设的黑盒采样设定下，任何有限点探针在理论上均不可能 100% 排除恰好藏在网格间隙内的任意窄测度尖峰（需更高密度采样预算换取更窄局部检测半径）。

### 4.2 端到端统一持续发现链（`336` 次多随机种子运行）的精确主张与边界
- **完全拉平的实验控制**：
  1. **同一起点**：全部 6 臂（含 `Frozen_After_Stage0`）从完全相同的 `Stage 0` `10` 节点纯原子森林出发，**零手工删除子节点**；
  2. **零宿主提示**：`Stage 3` 全部未见目标均不传入 `lens_hint` 或人工 `role_names`，由各臂在已购 `pilot_rows` 上自主执行 $S_d$ 角色排列与多起点双向符号仿射透镜标定；
  3. **真实 DEV 诱饵淘汰**：表面诱饵在 `3` 轮真实付费 DEV 子进程隔离确认中因 `confirm_rmse > tol` 被 Minton 账本自动归档，零合成循环；
  4. **`R3` 三层明确区分**：分别报告 ① 原冻结 `S1` `authored_model.py` 实例零重拟合直测（`RMSE = 0.0`）、② 新复合族重拟合（`RMSE = 0.0`）、③ 父子作用域分流路由（`100% Pass`）。
- **四种子（`48` 域内 + `8` OOS 题 / 臂）实测汇总**：
  - **`Hegel_M4_1_UnifiedForest`**：`48/48` 通过、`48/48` 召回确认、`16/16` 两跳游走胜出、`0/56` 假确认（`8/8` OOS `scoped_unknown`），平均评估候选 **`1.6667`**、测量费 **`35.00`**、计算费 **`26.955`**、总复合成本 **`36.3477`**；
  - **同时严格击败两种顺序的全历史普通臂**：`Ordinary_AllHistory_Reversed`（候选 `3.6667`、测量 `36.67`、计算 `37.00`、复合 **`38.5167`**）与 `Ordinary_AllHistory_Forward`（候选 `10.6667`、测量 `35.00`、计算 `78.50`、复合 **`38.9250`**）；
  - **同时严格击败无游走与扁平检索及公平冻结起点**：`Forest_Alpha0_SameHistory`（**`41.6863`**）、`Flat_BM25_SameHistory`（**`41.7020`**）、`Frozen_After_Stage0`（召回 `0.0000`、复合 **`48.1063`**）。
- **主动诚实边界说明**：① 离线可复现评测仍采用确定性残差驱动 `OpenCompositionalCodeProvider` 在 `ExprNode` 算子代数上做动态多轮合成（未发起外部商业 LLM API 网络调用，但实验台完整遵循 `CodeAgent` 工具协议）；② 在多随机种子评测脚本 `run_v17_3_m4_1_unified_study.py` 中，`Stage 1` 组合发现工作区在主种子运行并持久化至 `persisted_stage1_authored_workspaces/`，随后在全部 4 个独立未见目标种子（`20261120, 20261221, 20261322, 20261423`）上通过 `load_verified_authored_method` 实盘加载核验迁移表现。

---

## 5. 一键独立复现与红队核验命令（只读安全）

在仓库 `Hegel Machine/` 根目录下可直接运行以下命令完成全量复核（所有临时探针均在 `mktemp -d` 中执行，绝不污染只读证据与审计目录）：

```bash
# 1. 校验 V17.3 (M4.1) 静态证据包全部 55 个归档文件的 SHA-256 完整性
(cd docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120 && sha256sum -c SHA256SUMS.txt)

# 2. 运行 M0 -> M1 -> M2 -> M2.5 -> M3 -> M3.1 -> M4 -> M4.1 全套 33 项回归测试（约 3 分钟，33 passed）
PYTHONPATH=src PYTHONHASHSEED=0 python3 -m pytest \
  tests/test_v16_6_m0_contracts_and_wormholes.py \
  tests/test_v16_6_m1_cross_domain_wormhole_study.py \
  tests/test_v16_8_m2_open_workspace_wormholes.py \
  tests/test_v16_9_m2_audit_closure_and_open_revision.py \
  tests/test_v17_0_m3_hipporag_wormhole_scaling.py \
  tests/test_v17_1_m3_1_audit_closure_and_walk.py \
  tests/test_v17_2_m4_multiscale_and_open_discovery.py \
  tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py -v

# 3. 单独快速运行 V17.3 (M4.1) 新增 5 项红队闭环、可执行 Pi_s 与统一链测试（约 40 秒）
PYTHONPATH=src PYTHONHASHSEED=0 python3 -m pytest \
  tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py -v

# 4. 在隔离临时目录复跑上一轮 Hegel_M4_Audit 原始边界探针（保持 markdown/Hegel_M4_Audit 100% 只读）
TMP_RECHECK=$(mktemp -d)
cp -r markdown/Hegel_M4_Audit "$TMP_RECHECK/"
PYTHONPATH="src:$TMP_RECHECK/Hegel_M4_Audit/m31" PYTHONHASHSEED=0 python3 -c "
import pathlib, sys
import probe_boundaries as pb
pb.REPO = pathlib.Path('.').resolve()
pb.OUT = pathlib.Path('$TMP_RECHECK/boundaries_recheck.json')
pb.main()
print(pb.OUT.read_text())
"
rm -rf "$TMP_RECHECK"

# 5. 完整重算 V17.3 (M4.1) 全部四部分实证研究（约 3 分 15 秒）
PYTHONPATH=src PYTHONHASHSEED=0 python3 scripts/run_v17_3_m4_1_unified_study.py
```

---

## 6. 提请 GPT-6 重点审阅的 5 个问题

1. **11 项代码与契约边界闭环核验**：针对 `Hegel_M4_Audit` 探针发现的重复 `predict` / 属性遮蔽、无收据无 `target_id` 布尔放行、绝对路径确认、四参数 `osc+shunt` 撞号、`metadata_only` 产物防伪、`program_steps` 指令解释、子路径 Tier-3 终审、`FidelityCompression` 落库提交与异律拒绝，以及纵向粗粒化边复核恢复，当前源码实现是否已彻底闭合全部漏洞且无新引入旁路？
2. **多机理均衡目标集对正/逆序全历史基线的结构性区分**：在均衡覆盖 `3` 类互异新复合机理且零宿主 `lens_hint` 的 `336` 次评测中，`Hegel_M4_1_UnifiedForest` 同时严格击败 `Ordinary_AllHistory_Forward` 与 `Ordinary_AllHistory_Reversed`，是否消除了此前单目标下由历史列表顺序造成的解释歧义？
3. **可执行动力学 $\Pi_s$ 与 `coarse_graining_edges` 因果性**：双时间尺度 RK4 + 可执行 Schur 消元 $\Pi_s$ + 带输入盒边界的区域多点闭合证书，以及删除 `coarse_graining_edges` 后成本上升 `+34.7%` 的因果对照，是否扎实补齐了多尺度重整化支柱的操作语义？
4. **统一对象链与 `R1 / R3` 生命周期自洽性**：从字节级一致 `Stage 0` 起步、`CodeAgent` 组合发现入林、三层 `R3`（含原 `S1` 冻结实例零重拟合）到真实付费 DEV 诱饵淘汰与 `Stage 3` 自主迁移，这条不断裂的对象链与 `unified_full_lifecycle_ledger` 是否已满足 `M4` 里程碑的闭环标准？
5. **下一阶段（`M5`：接入在线前沿大模型后端与外部真实科学数据集）的架构建议**：在当前离线确定性契约、安全沙箱、多尺度重整化与森林检索底座全部夯实后，下一步挂载真实在线 LLM `CodeProvider`（存在随机采样噪声、语法越界、非参数化数值代码倾向）挑战外部真实科学发现基准时，最关键的防护设计是什么？
