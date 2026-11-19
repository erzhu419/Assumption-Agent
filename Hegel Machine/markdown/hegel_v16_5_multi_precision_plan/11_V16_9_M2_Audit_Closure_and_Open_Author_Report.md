# Hegel V16.9（M2.5）：对 `Hegel_M2_Audit` 的选择性吸收、开放 `CodeAgent` 修订闭环与公平对照实测报告

## 一、总体决策原则：全盘吸收硬核工程与归因修正，坚定保留面向 M3 的核心架构

针对 `markdown/Hegel_M2_Audit/GPT_diagnosis_m2.md`（及 `M2_AUDIT.md`、`MECHANISM_AUDIT.md`、`M2_RESULTS_AUDIT.md` 与 5 个便携探针脚本）的独立审计，我们遵循**“严抓真实漏洞与基线公平，拒绝因小库暂时权衡而拆毁大库骨架”**的决策原则，将审计项明确划分为两类处理：

### 1. 100% 吸收并闭合的工程正确性与归因问题（`F1`–`F11` 全数修复）
1. **确认失败诚实退出与确认快照冻结（`F5`）**：`solve_open_workspace_direct` 严格要求新点确认 `rmse <= tolerance`（取消 `1.2x` 放宽），通过确认后直接冻结被确认的 `model.py` 与训练前缀快照（不再用未确认的全量重拟合覆盖已确认模型）；当面对有限模板库外的 `y = 5 + sin(u)`（`tol = 0.01`）确认失败时，诚实返回 `status = "scoped_unknown"`（保留已购数据与 `unconfirmed_candidate.py`），`false_confirmation = False`。
2. **有状态 `fit(observations)` 的确认前缀严格隔离（`F11`）**：在调用 `lab.acquire(confirm_pts, phase="confirm")` 前先冻结 `training_prefix`，`validate_workspace_model_on_rows` 仅将 `training_prefix` 传入 `evaluate_python_model`。合法记忆器 `memorize.py` 在 `u=3` 新确认点上预测 `0.0`、`RMSE = 9.0`，被彻底拦截。
3. **收据（`evidence_receipt`）多重身份与有限误差绑定（`F7`）**：`wormhole_forest._validate_evidence_receipt` 强制核验具体 `claim_id`（禁止通配符 `'*'`）、有限非负 `observed_error <= tolerance`、`target_id`、`node_version`、`model_sha256` 与 `confirmation_digest`。旧目标/旧版本、缺误差、`NaN` 误差收据全部返回 `granted = False`。
4. **纯数据流 AST 规范化与旁注一致性互验（`F2`）**：重写 `canonicalize_python_model_ast`，直接内联 `predict(row)` 内的纯局部赋值（如 `f = row['features']`、`latent = ...`），仅剥离经校验非零的外层仿射读出，保留幂指数精确符号（`EXP_2` vs `EXP_-2`），并与 `_CORE_LAW_EXPR` 双向一致性校验（不一致时标记 `ANNOTATION_CODE_MISMATCH` 并绑定唯一源码哈希）。`probe_ast.py` 三组碰撞控制 `digests_equal` 全部翻转为 `False`，而勒夏特列（`+1`）与楞次（`-1` + 偏置）仍稳定收敛至同一 `VERIFIED_PURE_AST` 摘要。
5. **坐标排列不变性（$S_d$ 对称实验设计，`F6`）、真实执行 `program_steps`（`F8`）、独立克隆 RNG（`F9`）与真实子进程计时（`F10`）**：
   - 二维采用 5 点 $S_2$ 对称 pilot、三维采用 6 点 $S_3$ 对称 pilot，`T_M2A1` 逆序 `feature_names` 与原顺序取得完全一致的测量费（`35.0`）、计算费（`30.0`）与通过率（`RMSE ≈ 0.027`）；
   - 目标转移真实解释执行 `program_steps`（删除 `calibrate_observation_lens` 与 `subprocess_confirm_and_refit` 会实际跳过校准与收据生成，导致 `local_solver_guidance` 拒绝并触发回退，复合成本从 `36.5` 升至 `58.0`）；
   - `OpenWorkspaceTaskLab.__deepcopy__` 独立克隆 `_noise_rng` 状态（`identical_callable = False`，同点首观测精确相等）；
   - 移除 `0.02s` 合成占位，全程记录 `time.perf_counter()` 实测子进程耗时 `measured_python_wall_seconds`，并与逻辑代理费率分开入账；
   - `HExA` 采用确定性字典序破平局（`PYTHONHASHSEED=0` 与 `3` 输出逐字节一致），DEV 门控校准键与目标最终决策签名（含 `sig:m2_2d_capped_ceiling`）完全对齐。
6. **全历史公平基线与实测两跳桥接（回应第 3、4 节核心归因批评）**：
   - `O_strong_AllValidatedPriors` 放开保留同元数**全部已验证历史 schema（含单源及近失例）**，实际解析工作区 `shared/*.py` 并按额外候选计费；
   - `O_HExA_CodeSkills` 在目标阶段真实消费 `mistake_bank` 错误恢复模板；
   - 去掉目标专用提示字段 `two_hop_bridge_only`，改由 3 个真实一维源任务（`src_M2C1`、`src_M2C2`、`src_M2C3`）实测拟合残差自动形成两跳桥接，同时并列评测同门控同路径元数据的 `Flat_EqualMetadata_PathAware` 与传递等价簇 `Flat_TransitiveCluster`、以及保留跨领域 local 节点的 `Hegel_M2_no_wormhole_allow_local`。
7. **接入原生 `CodeAgent` 完成两条非模板开放作者反例修订链（`open_python_revision.py`，闭合 `F1` 与 `F3`）**：
   - 通过 `OpenWorkspaceCodeLabAdapter` 直接驱动 `CodeAgent`（`query / run / write / evaluate / finish`），编写含 `scipy.optimize.least_squares` 有状态 `fit(observations)` 与纯函数 `predict(row)` 的真实模块，在十类预写模板之外完成两条完整修订链：
     - **链 O1**：单调 Monod 饱和 $\to$ 遭遇高底物抑制 + 旁路分流反例 witness（旧模型 `RMSE = 0.975` 失败）$\to$ 诊断非单调翻转与旁路相关性 $\to$ 合成 Haldane 抑制加分流模块（向后兼容源数据 `RMSE = 0.00023`，前缀隔离新点确认通过）$\to$ 冻结后零搜索迁移至**列逆序且符号反转**的微藻光生物反应器目标 `T_O1`；
     - **链 O2**：过阻尼阶跃响应 $\to$ 遭遇欠阻尼 MEMS 悬臂梁超调振铃反例 witness（旧模型 `RMSE = 1.789` 失败）$\to$ 引入 `cos(w*t)` 振荡阻尼项（向后兼容源数据 `RMSE = 0.00024`）$\to$ 冻结后零搜索迁移至**列逆序且符号反转**的电源母线 LC 振铃目标 `T_O2`。

### 2. 主动保留并拒绝保守回退的设计（面向 M3 的核心资产）
- **保留哲学森林 + 横向虫洞（`PhilosophicalForest` + `HorizontalWormholeEdge`）**：在小规模 7 节点无同签名干扰碰撞的受控库中，`Hegel_M2` 与允许直接跨领域遍历 local 节点的 `NoWhAllowLocal` 成本持平（`0.00`），与全元数据平面表 `Flat_EqualMetadata_PathAware` 仅差线性扫描费（`-1.0125`）；同时单批 10 题尚未摊回双源形成或源期反例核验的前置投入（需约 17~22 题摊回）。**我们如实记录这一小库边界，绝不因此退回平面列表**——因为横边排除谓词、非传递误差合成与中间层图拓扑的核心战场正是 **M3（$K \ge 32$ 大规模异质干扰簇并存 + HippoRAG 个性化 PageRank 虫洞检索）**。

---

## 二、V16.9 聚焦反例复核结果（`focused_audit_recheck.json`）

| 审计探针项 | V16.8（审计前） | V16.9（修复后实测） | 状态 |
|---|---|---|---|
| `probe_ast`: 同旁注异代码 / 无旁注局部变量 / `x**2` vs `x**-2` | 3 组均碰撞（`True`） | **3 组 `digests_equal` 均为 `False`**（且保留 `+1/-1` 透镜归一） | 已闭合 |
| `probe_confirmation_training_leak`: 有状态 `memorize.py` | 确认 `RMSE = 0.0` | **确认 `RMSE = 9.0`**（仅见确认前 2 行训练前缀） | 已闭合 |
| `probe_runtime`: 模板库外 `5 + sin(u)` 确认失败 | 返回 `completed`（假确认） | **返回 `scoped_unknown`**（`false_confirmation = False`） | 已闭合 |
| `probe_runtime`: `T_M2A1` 逆序 `feature_names` | `RMSE = 0.049` 失败 | **`strict_pass = True`（`RMSE = 0.0271`，测量费 `35.0` 与原序完全一致）** | 已闭合 |
| `probe_runtime`: 删除 `program_steps` 校准与确认步骤 | 行为与成本完全不变 | **真实改变行为与授权**（跳过校准/收据被拒转回退，成本 `36.5 -> 58.0`） | 已闭合 |
| `probe_runtime`: `deepcopy(lab)` 闭包 RNG | `identical_callable = True` | **`identical_callable = False`，同点首观测精确相等（`equal = True`）** | 已闭合 |
| `probe_runtime`: 旧目标/旧版本/缺误差/`NaN` 收据 | 3 项均误通过 | **全部拒绝（`[False, False, False]`）** | 已闭合 |

---

## 三、V16.9 四簇受控公平对照与开放作者修订实测总表

### 1. 受控 9 臂四簇冻结评测（`360/360` 严格通过，`0` 假确认）

| 配对比较（均值单位：每 10 题评测簇） | 测量费差 `ΔMeas` | 逻辑计算差 `ΔCompute` | 目标复合差 `ΔTarget` (`λ=0.05`) | 含源期端到端差 `ΔE2E` | 准确科学解释 |
|---|---:|---:|---:|---:|---|
| **`Hegel_M2 - O_strong_AllValidatedPriors`** | **`0.00`** | **`-179.00`** | **`-8.95`** (4/4 负) | **`+6.05`** | 对称 5 点 pilot 消除冗余探针后测量打平；目标期省去多候选遍历（实测子进程 `0.87s` vs `1.08s`）；单批 10 题尚未摊回源期 `+15.0` 反例核验投入（约 17 题打平） |
| **`Hegel_M2 - O_strong_TwoDomainCompressed`** | `-70.00` | `-146.00` | **`-77.30`** (4/4 负) | `-62.30` | 保留作历史压缩敏感性诊断对照 |
| **`Hegel_M2 - O_HExA_CodeSkills`（消费错误库+确定破平）** | `0.00` | `-103.00` | **`-5.15`** (4/4 负) | `+6.05` | 确定性破平局 + `mistake_bank` 重定向后仍省去错误技能先试开销 |
| **`Hegel_M2 - Flat_TransitiveCluster`（同 Stage1 门控）** | `-15.00` | `-48.75` | **`-17.44`** (4/4 负) | `-17.44` | 实测两跳误差界拦截非传递近似简化（无目标提示字段） |
| **`Hegel_M2 - Flat_EqualMetadata_PathAware`** | `0.00` | `-20.25` | **`-1.01`** (4/4 负) | `-1.01` | 同元数据同路径界下仅剩平面表线性扫描开销，为 M3 规模化检索确立零偏置基线 |
| **`Hegel_M2 - NoWhAllowLocal`（保留跨领域 local 节点）** | `0.00` | `0.00` | **`0.00`** | `0.00` | 如实证明 7 节点无冲突小库中 local 内容已足够，横边排除与图路由面向 M3 大库冲突消解 |
| **`Hegel_M2 - NoCrossDomainTransfer`** | `-115.00` | `-106.50` | **`-120.33`** (4/4 负) | `-120.33` | 跨领域历史复用整体开关净价值 |

### 2. 原生 `CodeAgent` 开放算子反例修订链（4 评测种子 $\times$ 2 非模板链 $\times$ 3 臂 = `24/24` 严格通过，`0` 假确认）

| 未见逆序反号目标 | `Hegel_OpenRevision_Forest` (Meas / Comp / Wall / Events) | `Continuing_Ordinary_CodeSession`（同工具同 `shared/` 历史） | `Ablated_RemoveRevisedNode`（移除新修订节点对照） |
|---|---|---|---|
| **`T_O1`（微藻光反应器：Haldane 底物抑制 + 旁路分流）** | **`50.0` / `53.23` / `1.63s` / `3` 步** | `60.0` / `66.23` / `2.21s` / `9` 步（**Hegel 净省 `-13.00`**） | `65.0` / `74.70` / `3.21s` / `13` 步（**证明移除新知识显著改变后继行为**） |
| **`T_O2`（电源母线 LC 欠阻尼阶跃振铃）** | **`50.0` / `53.23` / `1.36s` / `3` 步** | `60.0` / `71.23` / `4.30s` / `17` 步（**Hegel 净省 `-18.00`，墙钟快 `3.1x`**） | `65.0` / `77.45` / `4.19s` / `17` 步（**证明移除新知识显著改变后继行为**） |

---

## 四、代码与全套证据索引

- 升级与新增模块：`src/hegel_machine/cognition/wormhole_forest.py`、`open_workspace_labs.py`、`open_workspace_wormhole.py`、`open_python_revision.py`
- 回归测试（`17/17` 全绿）：`tests/test_v16_9_m2_audit_closure_and_open_revision.py`（及 `M0`、`M1`、`M2` 测试集）
- 完整 15 份实测证据目录：`docs/evidence/v16_9_m2_audited_open_workspace_20261119/`（原 V16.8 `9115` 证据目录原样保留供静态溯源比对）
