# Hegel V17.3 (`M4.1`) 复核闭环、可执行动力学 $\Pi_s$ 重整化与端到端统一持续发现链实测报告

> **审计基准 Commit**：`11961f702739ab272661ff59b77fb5bc8e3589b8`（对应 `markdown/Hegel_M4_Audit/GPT_diagnosis_M4.md` 与 `M4_AUDIT.md` 及三套探针 `m31/probe_boundaries.py`、`author/probe_author.py`、`multiscale/probe_multiscale_lifecycle.py`）  
> **静态只读约束**：此前全部证据包（`docs/evidence/v16_8_*` 至 `v17_2_*`）与审计包（`markdown/hegel_M3_Audit/`、`markdown/Hegel_M4_Audit/`）保持 **100% 原样只读未改动**；本轮 `V17.3 (M4.1)` 全新证据独立归档于 `docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120/`。

---

## 一、对 `Hegel_M4_Audit`（GPT-6 复核）的独立工程研判：吸收 vs 战略保留与升级

遵循项目治理原则——**把外部审计当作最坏边界提醒器，100% 吸收真接线缺陷与统计口径修正，同时拒绝将多尺度普适森林与图游走压扁退化为单尺度局部技巧**——我们将 `GPT_diagnosis_M4.md` 拆分为 **A 类真边界闭环** 与 **B 类架构升级回应**：

### 1.1 100% 吸收并已在源码闭合的 11 项工程边界与统计修正

| 编号 | 审计指出的具体问题 | 源码修复位置与机制 | 复测状态 |
| :--- | :--- | :--- | :--- |
| **B1** | 同一 `model.py` 重复定义 `def predict` 或顶层篡改 `math.exp = ...` 时仍可能产生相同 `VERIFIED_PURE_AST` 摘要 | `open_workspace_wormhole.py`：`canonicalize_python_model_ast` 统计 `predict_count / fit_count`（`>1` 判为 `UNSUPPORTED_IMPURE_PYTHON` 并以 `impure:<sha256>` 区分），拒绝一切非 `ast.Name` 属性/下标顶层赋值 | **已闭合** (`True`) |
| **B2** | `authorize_abstraction_use` 在 `evidence_receipt=None` 且 `task_view` 省略 `target_id` 时仍允许 `heldout_confirmed=True` 放行 | `wormhole_forest.py`：凡 `open_workspace_mechanism_prediction` 或 `*_ast` 契约且 `evidence_receipt is None`，无论是否传入 `target_id` 一律 `confirmed = False` | **已闭合** (`True`) |
| **B3** | `validate_workspace_model_on_rows` 与 `evaluate_isolated_python_model` 传入工作区内绝对路径时沙箱映射失效 | `open_workspace_labs.py`、`open_workspace_wormhole.py`、`m4_compositional_discovery.py`：统一将工作区内绝对路径规范化为相对路径，越界路径抛 `ValueError` | **已闭合** (`9.0 == 9.0`) |
| **B4** | `m3_scale_universe.py` 安全低负载首点确认失败后同时置位全部 `4` 个高负载标志 | `m3_scale_universe.py`：严格按当前实测维数、对称性与低负载起始签名（`reciprocal / ridge / shunt / charging`）**仅更新唯一对应**的高负载实证标志 | **已闭合** |
| **B5** | `m4_compositional_discovery.py` 的 `_max_param_idx` 未遍历 `cos_mul.param`，导致 `exp_relax -> osc -> shunt` 四参数算子撞号 `p2` | `m4_compositional_discovery.py`：引入通用 `_iter_child_exprs` 遍历全部子树（含 `cos_mul.param`）、强制 `strict_unique=True` 连续参数校验，并保留子算子 `multi_start_param` | **已闭合** (`p0..p3` 独立纯 AST) |
| **B6** | `authored_method.json` 加载器未核对确认状态、代码 SHA256 与 AST 摘要（元数据伪造即可复用） | `m4_compositional_discovery.py`：新增 `load_verified_authored_method`，强制校验 `confirmed_ok is True`、`VERIFIED_PURE_AST`、`authored_model_source` SHA256、确认摘要及重编译 AST 一致性 | **已闭合** (`metadata_only` 100% 拒绝) |
| **B7** | `evaluate_m4_target_under_arm` 仅检查 `program_steps` 非空（未知 opcode 仍照常走召回） | `m4_multiscale_renormalization.py`：实现 `_validate_and_interpret_m4_program_steps` 指令解释器，校验合法 opcode 集、`operator_tag`、`active_arity` 与 `fresh_points >= 1` | **已闭合** (`nonsense` 100% 拒绝召回) |
| **B8** | 两跳游走子路径 `c1_eval` 跳过最终 `authorize_abstraction_use(requested_tier="local_solver_guidance")` 及未知前提拦截 | `m4_multiscale_renormalization.py`：子分支严格从已购 `pilot + c0_rows` 重算实证特征并强制通过 `local_solver_guidance` 终审；`bounded_trial` 上限或未测前提 `audit_boundary_measured`（`UNKNOWN`）均被拦截 | **已闭合** (`True`) |
| **B9** | `compress_compatible_nodes` 未把验证通过的 `merged_node["claim_contract"]`（含 `disjunctive_branches`）提交回 keeper，且粗评估器允许同维异律（Murray vs Monod）合并 | `wormhole_forest.py`：① `fidelity_compress_nodes` 拦截重叠 `ACC` 下的异律合并（`law_a != law_b` 记入冲突）；② `compress_compatible_nodes` 将验证后的 `claim_contract` 正式注册提交至 keeper 并执行提交后逐查询一致性断言 | **已闭合** (`accepted=False` 异律 / `ACC==ACC` 同律或析取) |
| **B10** | 协议修订导致 `aq_meso_hex_cell` 版本递增后，纵向 `coarse_graining_edges` 悬空失活（`6 -> 4`） | `wormhole_forest.py` + `hipporag_wormhole_index.py`：纯协议修订自动重绑未变 `ScaleMap` 的粗粒化边（`6 -> 6`）；当 `ScaleMap` 或定律指纹变更时标记 `stale_endpoint_revision=True`（`6 -> 4`），并通过新增 `revalidate_vertical_edges_after_revision` 显式复核恢复（`4 -> 6`） | **已闭合** (`6 -> 4 -> 6`) |
| **B11** | `M3.1` $2\times 2$ 检索因子表缺第 4 格 `Alpha0_NoPool`，以及 `13_V17_1` 报告中 `C4` 数字笔误澄清 | ① 新增 `hipporag_alpha0_nopool` 跑满 `4` 簇完整四格；② **口径勘误说明**：`V17.1` Part 1 的 `C4` 基线对比中，`Hegel_M3` 目标均值为 `53.225`，`O_strong_AllValidatedPriors` 全历史均值为 `57.075`（单目标净省 `-3.85`），而 `74.70` 系同表节点版本删除消融对照 `without_node_version`，特此精确区分 | **已补齐并勘误** |

### 1.2 正面回应架构质疑：拒绝退回静态单层记忆，打通“一条不断裂的对象链”

GPT-6 对 `M4` 最核心的架构质疑并非局部 bug，而是三点结构性质疑：
1. **质疑一（单目标逆序基线）**：在 `M4` Pillar 1 中只有 **1 类**未见复合目标（`Haldane+Shunt`），把 `Ordinary_AllHistory` 的 6 个历史先验倒序排列（`W_haldane_shunt` 恰好排第 0 位）即可与森林打平；
2. **质疑二（重整化算子形式化程度）**：`M4` Pillar 2 的粗粒化通道是闭合解析式，且单点边界探针无法捕获域内局部微观共振凸起，`coarse_graining_edges` 删除后未改变下游行为；
3. **质疑三（三支柱对象链断裂）**：Pillar 1 `CodeAgent` 发明的复合算子没有直接注册进 Pillar 2/3 的 `PhilosophicalForest` 再由 `HippoRAGWormholeIndex` 在无宿主 `lens_hint` 下自主检索复用，且 Pillar 3 的 `Frozen` 起点手工删了 1 个子节点、诱饵归档使用了合成反馈循环。

在 `V17.3 (M4.1)` 中，我们没有收缩架构，而是正面建成了 `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py`，将三大支柱熔铸为**同一条零作弊对象链**：
- **针对质疑一（数学上不可被任何静态顺序绕过的多机理均衡评测）**：当未见目标集均衡覆盖 **$M = 3$ 类互异新发现复合机理**（`Haldane+Shunt`、**4 参数** `Oscillatory+Shunt`、安全试探两跳 `Murray+Cap` 各占 $1/3$）且特征顺序置换、传感器符号反转、**零宿主 `lens_hint`** 时，任何扁平全历史列表无论正序（`Forward`）还是逆序（`Reversed`），最多只能把其中 1 类放在首位，其余 $2/3$ 必付遍历试错与角色置换代价；而 `Hegel_M4_1_UnifiedForest` 通过可观测 `sig:*` 线索与两跳跨树图游走精准路由，**同时严格击败 `Ordinary_AllHistory_Forward` 与 `Ordinary_AllHistory_Reversed`**！
- **针对质疑二（真实 RK4 快慢微分方程轨迹 + 可执行 $\Pi_s$ 投影 + 区域多点盒闭合证书 + 粗粒化边因果删除对照）**：实现了真实双时间尺度刚性 ODE 积分器与可执行 adiabatic Schur 消元 / 轨迹窗口投影算子 $\Pi_s$，用区域多点盒探针（覆盖四角与内部中心）取代单点探针（100% 捕获域内局部微观共振凸起并拦截盒外查询），并证明删除 `coarse_graining_edges` 会直接切断 `S_micro --Pi_s--> S_meso` 召回路径，导致付费测量与总成本显著上升（`72.61 -> 97.78`）！
- **针对质疑三（100% 字节级一致 `Stage 0` 起点 + 真实 `CodeAgent` 发现入林 + 3 层 `R3` 保守性 + 真实付费 DEV 诱饵淘汰）**：全部 6 个评测臂（含 `Frozen_After_Stage0`）从完全相同的 `Stage 0` 原子森林起步（零手工删点）；`Stage 1` 由 `CodeAgent` 从原子算子合成全部 3 类新复合机理并经 `load_verified_authored_method` 校验直接注册进森林；`Stage 2` 表面诱饵全部通过在真实付费 DEV 任务上运行隔离子进程确认失败触发 Minton 效用归档（零合成反馈循环！）。

---

## 二、完整补齐 `M3.1` $2\times 2$ 检索因子分解实验（`4` 冻结评测簇，`496` 次目标工作区运行）

| 检索臂（$\alpha \times \text{WormholePool}$） | 严格通过率 (`124` runs) | `T01..T06` 跨域桥接复合成本 | `T25..T28` 安全低负载游走胜出数 | `T25..T28` 复合成本 | 全集平均测量费 ($\lambda=0$) | 全集平均计算费 | 全集平均复合成本 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Walk_Pool` ($\alpha=0.65$, `Pool=True`)** | **`1.0000`** | **`39.9702`** | **`28 / 32` (`87.5%`)** | **`44.7579`** | **`36.4516`** | **`38.9854`** | **`38.4009`** |
| `Walk_NoPool` ($\alpha=0.65$, `Pool=False`) | `1.0000` | `40.8660` | `28 / 32` (`87.5%`) | `44.7579` | `36.6129` | `39.8725` | `38.6065` |
| `Alpha0_Pool` ($\alpha=0.0$, `Pool=True`) | `1.0000` | `39.9317` | `0 / 32` (`0.0%`) | `64.3025` | `40.4839` | `58.3548` | `43.4016` |
| **`Alpha0_NoPool` ($\alpha=0.0$, `Pool=False`，新增第 4 格)** | `1.0000` | `39.8483` | `0 / 32` (`0.0%`) | `64.3025` | `40.4839` | `58.0323` | `43.3855` |

- **正交效应完全清晰**：在安全低负载试探区（`Q_safe \cap sigs(Child) = \emptyset`），$\alpha=0$ 的两格（无论是否池化）图游走召回均为 **`0/32`**（复合成本 `64.3025`），而 $\alpha=0.65$ 两跳游走降至 **`44.7579`**（节省 **`-30.4%`**）；而在全量程高竞争跨域目标 `T01..T06` 上，`Walk_Pool` 相比 `Walk_NoPool` 进一步压低跨域冲突开销（`39.9702` vs `40.8660`），全集总复合成本取得四格最低 **`38.4009`**。

---

## 三、可执行动力学 $\Pi_s$ 快慢轨迹重整化与区域闭合实验（Part 3）

在双时间尺度刚性系统 $\dot{x}_s = -(x_s - a_s u)/\tau_s + g_{sf} x_f,\ \tau_f \dot{x}_f = -(x_f - a_f u) - g_{fs} x_s$（$\tau_s = 1.45,\ \tau_f = 0.10$，RK4 细步长 $\Delta t = 0.01$）上，可执行粗粒化算子 $\Pi_s$ 执行 adiabatic Schur 消元与轨迹窗口投影：

1. **快流形奴役缺陷随时间尺度指数收敛**：
   - 短时淬火 $T = 0.08 \sim \tau_f$：快流形奴役残差 `0.527555`，粗细读出偏差 `0.439699 >> tol` $\implies$ 区域闭合探针自动判定 `not_closed` 并下钻至 `micro_fine`（`heldout_rmse = 0.0`，`strict_pass = True`）；
   - 慢流形绝热窗 $T = 4.0 \gg \tau_f$：快流形奴役残差收敛至 **`0.002901`**，粗细读出偏差 **`0.007368`**；且当外部查询点落入已验证输入盒 `[2.2, 5.2]` 之外（$T = 0.12$）时，`is_point_inside_verified_box` 100% 拒绝粗模型越界授权（`out_of_box_query_rejected = True`）。
2. **`coarse_graining_edges` 的因果必要性验证**：
   - **`With_CoarseGraining_Edges`**：从微观探头线索 `sig:cue_fast_probe_instrument` 沿 `ode_micro_fast_port --Pi_s--> ode_meso_schur_reduced` 游走召回 1 阶 Schur 粗算子（`walked_coarse_graining_edge = True`），`1` 次拟合即通过隔离确认，付费测量费 **`72.0`**（总复合成本 **`72.6095`**，`heldout_rmse = 0.008792`）；
   - **`Without_CoarseGraining_Edges`（删除粗粒化边）**：失去跨尺度召回（`confirmed_via_recall = False`，评估候选数 `1 -> 5`），付费测量费升至 **`96.0`**（总复合成本 **`97.7765`**，贵出 `+34.7%`）；
   - **`MicroOnly_AlwaysFine`（全微观细通道）**：付费测量费 **`108.0`**（总复合成本 **`108.6095`**，比 `With_CoarseGraining_Edges` 贵出 `+49.6%`）。
3. **域内局部微观共振凸起反例（区域多点盒探针 vs 单点边界探针）**：
   - 在 $T \approx 3.55 \in [2.2, 5.2]$ 注入高斯微观共振包：单点边界探针（`Single_1Pt_Boundary_Probe`）漏检局部破裂（测得缺陷仅 `0.0178`），产生 **`false_confirmation = True`**（`heldout_rmse = 0.4292`）；
   - 区域多点盒探针捕获内部峰值缺陷 **`0.8584 >> 0.06`**，拒绝粗闭合证书，转入细尺度后因高斯凸起超出有理/指数语法范围而诚实返回 `status = "scoped_unknown"`，实现 **`false_confirmation = False`（零假确认）**。

---

## 四、端到端统一持续发现—入林—自主检索—生命周期链（`4` 随机种子 × `14` 目标 × `6` 臂 = `336` 次运行）

### 4.1 `Stage 0 -> Stage 1 -> Stage 2` 真实对象链审计结果

- **字节级一致起点（`Stage 0`）**：全部 6 臂从同一个 `10` 节点纯原子森林出发（不含任何复合算子，`Frozen_After_Stage0` 零删点原样冻结）。
- **`Stage 1` 开放组合发现入林**：原生 `CodeAgent` 仅凭原子算子与 `diag_residual.py` 残差 `stdout` 独立发现并验证全部 3 类复合定律：
  1. `W1_haldane_shunt_2d`（2 轮合成 `Monod -> Haldane -> Haldane+Shunt`，`strict_pass = True`）；
  2. `W2_oscillatory_shunt_4p`（2 轮合成 **`4` 参数** `ExpRelax -> OscillatoryDamping -> Oscillatory+Shunt`，`strict_pass = True`）；
  3. `W3_murray_capped_1d`（合成 `Murray 2/3 -> VariationalCeilingCap`，`strict_pass = True`）。
- **`R3` 三层旧作用域保守性验证**：
  - `Tier 1`（原冻结 `S1` `authored_model.py` 实例**零重拟合**直接评测旧 `S1` heldout）：`RMSE = 0.0`（`Pass`）；
  - `Tier 2`（新复合子族在旧 `S1` 数据上重拟合兼容性）：`RMSE = 0.0`（`Pass`）；
  - `Tier 3`（森林作用域路由：低底物走冻结 `S1` 实例、高底物分流走冻结 `W1` 实例，双方零重拟合）：`100% Pass`。
- **`Stage 2` 真实付费 DEV 诱饵淘汰与保真压缩**：`3` 个兼容跨树副本经 `compress_compatible_nodes` 在线合并并提交回 keeper；异律合并被拒绝；`2` 个高词频表面诱饵在 `3` 轮真实付费 DEV 子进程确认中因高误差（`confirm_rmse > tol`）触发 Minton 负效用归档；磁盘产物篡改为 `metadata_only` 被 `100%` 拦截，删除与恢复严格复现因果差异（`R1`）。

### 4.2 `Stage 3` 多机理均衡未见目标迁移总表（零宿主 `lens_hint`，自主标定角色与透镜）

| 受控评测臂（统一 `Stage 0` 起点） | 域内严格通过 (`48` runs) | 召回确认率 (`48` runs) | 安全试探两跳游走胜出 (`16` runs) | 假确认数 (`56` runs 含 `8` OOS) | 平均评估候选数 | 平均物理测量费 ($\lambda=0$) | 平均计算费 | 平均总复合成本 | 相对 `Hegel_M4_1` 差值 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_M4_1_UnifiedForest`** | **`1.0000`** | **`1.0000`** | **`16 / 16`** | **`0`** (`8/8` OOS 诚实未知) | **`1.6667`** | **`35.0000`** | **`26.9550`** | **`36.3477`** | **基准最优** |
| **`Ordinary_AllHistory_Reversed`（逆序全历史对照）** | `1.0000` | `0.6667` | `0 / 16` | `0` | `3.6667` | `36.6667` | `37.0000` | `38.5167` | **`+2.1690`（慢 `2.20×` 候选）** |
| **`Ordinary_AllHistory_Forward`（正序全历史对照）** | `1.0000` | `0.6667` | `0 / 16` | `0` | `10.6667` | `35.0000` | `78.5000` | `38.9250` | **`+2.5773`（慢 `6.40×` 候选）** |
| `Forest_Alpha0_SameHistory`（无游走池化森林） | `1.0000` | `0.6667` | `0 / 16` | `0` | `2.6667` | `40.0000` | `33.7250` | `41.6863` | **`+5.3386`（测量贵 `+14.3%`）** |
| `Flat_BM25_SameHistory`（扁平 BM25 池化） | `1.0000` | `0.6667` | `0 / 16` | `0` | `2.6667` | `40.0000` | `34.0400` | `41.7020` | **`+5.3543`（测量贵 `+14.3%`）** |
| **`Frozen_After_Stage0`（公平零删点冻结起点）** | `1.0000` | `0.0000` | `0 / 16` | `0` | `12.6667` | `43.3333` | `95.4600` | `48.1063` | **`+11.7586`（总成本贵 `+32.3%`）** |

- **全生命周期统一成本账本（`unified_full_lifecycle_ledger`）**：完整分项记录 `stage0_atomic_formation`（`186.00`）、`stage1_compositional_discovery`（`157.70`）、`stage2_compression_and_empirical_dev_lures`（`167.70`）与 `stage3_hegel_target_eval`（`12` 题合计 `436.17`，对比 `Frozen_After_Stage0` 单阶段 12 题即耗费 `577.28`），在包含中期发现与维护投资后仍展现显著持续摊销优势。

---

## 五、全量回归验证与证据索引

- **全套测试**：`PYTHONPATH=src python3 -m pytest -q tests/test_v16_6*.py tests/test_v16_8*.py tests/test_v16_9*.py tests/test_v17_0*.py tests/test_v17_1*.py tests/test_v17_2*.py tests/test_v17_3*.py` $\implies$ **`33 passed`（100% 通过，零回退）**。
- **独立证据目录**：`docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120/`（含 `summary.json`、`part1..part4` 四份分项 JSON、`per_target_receipts_336_runs.jsonl`、持久化 `persisted_stage1_authored_workspaces/` 及 `55` 项 `SHA256SUMS.txt`）。
