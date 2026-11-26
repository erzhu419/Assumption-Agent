# Hegel M8.9.1 确定性审计闭环与 M9.0a 零提示自愿可执行复用对照实验报告（`docs/45_ZH`）

- **工作分支**：`exp/m6-causal-wormhole-sandbox`
- **三阶段严格提交隔离（Clean Commit Separation）**：
  - **Commit 1 (`be2a3b29fb978f19d94b9b9b8a19956ba96d3857`)**：`M8.9.1` 全部确定性工程闭环（`R1..R7`）、独立完备 Box-$\ell_0$ 证书核（`m8_9_1_box_l0_certificate_kernel.py`）、封存 `M8.9` `hs01` 非破坏性独立重评回执（`18/18` 全通过，伪造下界负控严格拦截）、`m7_4` 真实 20 折 `LOTO` 路由器回执（`14/20` vs `13/20`）、`57/57` 回归测试全绿及 `docs/44b` 正式勘误增补。
  - **Commit 2 (`99c8d82fec81bc4607879faf45a37ffef62a1331`)**：`M9.0a` 零提示分层基准（`Hegel-M90a-VoluntaryReuse-4`：`vr01..vr04`）、正负对照容器验证套件（`8/8` 测试全绿）与并发交错实验运行器（`run_m9_0a_voluntary_reuse_and_knowledge_isolation.py`），在实测启动前完整干净冻结（`git_dirty_at_start == false`）。
  - **Commit 3 (`HEAD`)**：单次无断点并发完成的 `24`-episode（`3 实验臂 × 4 任务族 × 2 独立重复`）`gpt-6.2-sol` 封存证据（`docs/evidence/m9_0a_voluntary_reuse_and_knowledge_isolation_20261126/`，含 `778` 个文件 `SHA256SUMS.txt`）与中英文 `docs/45` 报告。

---

## 1. GPT-6 审计诊断处置与主权决策综述（`Hegel_M8_9_Audit`）

### 1.1 全部采纳并闭环的确定性工程缺陷（`M8.9.1`, `R1..R7`）

针对 `markdown/Hegel_M8_9_Audit` 中的全部可复现反例，已在 `Commit 1` 中逐项修复并通过 `57/57` 单元测试：

1. **`R1`（完备 Box-$\ell_0$ 子立方体证书核与封存 `hs01` 独立重评）**：
   - 新增独立验证核 `src/hegel_machine/cognition/m8_9_1_box_l0_certificate_kernel.py`（`verify_box_l0_subcube_certificate`），合法接受 `1 <= len(leaves) <= max_leaves`，对每个叶节点通过全部 $2^p$ 支撑上的连续箱约束凸 QP（$p \le 16$）或携带证明的 Fenchel 透视对偶（`dual_nu`）/ PSD 对角下界（`minorant_diagonal`, `minorant_center`）见证（任意 $p$）检验下界完备性，并正交分离输出 `solution_correct`、`partition_valid`、`leaf_bounds_valid`、`gap_satisfied` 与 `certificate_valid`。
   - 同步加固 `hs01` 与 `PT06` 验证器。
   - 在不改动任何 `M8.9` 封存文件的前提下，对 `M8.9` 全部 `6` 份唯一 SHA-256 `hs01` 提交物在原始 3 个私有实例（`99101, 99102, 99103`，$p = 14, 14, 15$，共 `65,536` 次精确凸 QP 求解）上执行独立重评：**`18/18` 个 `(artifact, instance)` 配对全部通过（最大间隙 $\le 4.25 \times 10^{-7}$）**，重评后 `M8.9` 三组实验臂在 `Primary 1x` 与 `Full Policy` 下均达到 **`8/8`（`100%`）**（与原始 `7/8`、`8/8` 标签并列展示），同时 GPT-6 伪造下界负控被确定性拒绝（`leaf_bounds_valid == False`）。
2. **`R2`（内容寻址资格校验与多版本编译溯源）**：
   - `enrich_and_harden_l1_primitive_contracts` 改为按 `(stem, actual_code_sha256, synthetic_unit_test_sha256)` 寻址并仅向字节一致副本写回回执；`sublate_autogenous_forest_into_compositional_primitives` 按 `(filename, recomputed_source_sha256)` 收集多版本 `L2` 源码并写入 `verified_code_sha256`；`verify_l1_module_qualification_record` 对缺失/空/`False`/SHA 失配回执严格 fail-closed 拦截；`M8` 容器修复注入真实前置条件与预取契约，并显式标注 `"repaired_digest_kind": "raw_single_file_sha256"`。
3. **`R3`（两阶段原子回写与双向入口可达留存判定）**：
   - `auto_package_wormhole_imports_in_container` 升级为主机端全文件构建与 SHA 核验通过后再向容器提交的两阶段原子回写；`inspect_final_deliverable_skill_retention` 升级为有序辅助闭包双向集合相等 + `SimpleNamespace` 导出映射核验 + 作用域感知的静态调用图可达性（排除未调用嵌套定义体），并为全部任务显式绑定 `declared_callable_entrypoints`。
4. **`R4`（实质性验证与命题绑定 `nogood` 消解）**：
   - `is_substantive_verification_command` 拒绝纯 `python3 -c "import sol"` 冒烟命令；`nogood` 消解见证按 `risk_kind`（如 `target_identifiability` 强制要求锚点/规范不变性见证）与 `scope` 校验。
5. **`R5..R7`（路由器特征顺序与哈希绑定、真实 `LOTO`、规划器签名纯化及知识隔离臂）**：
   - 修正 `CONTAINER_STATE_FEATURE_NAMES` 第 `2..3` 列顺序并在 `from_json_dict` 中强制校验 `feature_builder_sha256`；归档 `m7_4` 真实 20 折 `LOTO`（当前特征 `14/20` vs 放宽后 `13/20`）；从 `inc_sig` 剥离后验账本字段并支持时序 `ObservationContext`；采用显式未知量规模（`>= 4000 unknowns`）替代单词 `"iterative"` 触发，并新增 `progressive_knowledge_only` 仅知识参照臂。

### 1.2 超越保守停滞的主权实验推进（`M9.0a`）

GPT-6 倾向于先停留在确定性修复，我们则在干净冻结 `Commit 1` 与 `Commit 2` 后，直接在冻结 `v5` 技能库（`6` 个合格 `L1` 模块、`17` 个函数，`0` 在线编译）上使用 32 并发 `gpt-6.2-sol` CLI（`auth_mode == "chatgpt"`，`max_turns = 28`，题目提示词 **零技能名暗示**）执行了决定性的 **`3 实验臂 × 4 任务族 × 2 独立重复 = 24` 交错对照实验（`M9.0a`）**：
1. **`bare_no_library`（Arm 1）**：空森林/空档案（`0` 参考卡，`0` 可执行技能）；
2. **`knowledge_only_reference`（Arm 2）**：只读 `progressive_knowledge_only` 参考卡 + `/tmp/hegel_knowledge_ref/describe_ref.py` + `.reference.txt` 算法参考笔记（**不挂载任何可 `import` 的 `hegel_wormhole_skills` Python 包**，精准隔离“纯数学/算法知识效应”）；
3. **`progressive_executable_library`（Arm 3）**：渐进契约索引卡 + `--describe` CLI + 可直接 `import` 的 `/tmp/hegel_wormhole_skills` 可执行包 + 两阶段原子静态闭包打包器（精准隔离“超越纯知识的可执行代码复用效应”）。

---

## 2. `M9.0a` 溯源完整性与预运行任务难度核查

完整记录于 `docs/evidence/m9_0a_voluntary_reuse_and_knowledge_isolation_20261126/m9_0a_voluntary_reuse_master_summary.json` 与 `m9_0a_pre_run_hardness_and_applicability_audit.json`：

- **单次干净调用溯源**：
  - `code_commit_at_start == code_commit_at_end == "99c8d82fec81bc4607879faf45a37ffef62a1331"`
  - `source_tree_py_sha256_at_start == source_tree_py_sha256_at_end == "ddd58dd7199024647b1d30af999a5a70e031c9b590073d0c72198bae1922b966"`
  - `git_dirty_at_start = false`、`git_dirty_at_end = false`、`resumed_episode_count = 0`、`fresh_executed_episode_count = 24`、`single_clean_unresumed_invocation = true`、`zero_skill_naming_hints = true`
  - 并发遥测：24 个交错 worker 并发提交，峰值并发 Codex 调用 `24`，总调用 `296` 次，总墙钟耗时 **`857.22s`（`14.29` 分钟）**，`SHA256SUMS.txt` 完整封印 `778` 个文件。
- **四类分层适用性体制（`Hegel-M90a-VoluntaryReuse-4`）**：
  1. **`vr01__proof_carrying_box_l0_minorant_bnb`（精确子问题匹配 + 非平凡搜索）**：私有维度 $p = (20, 21, 22)$（支撑空间达 `105 万 .. 419 万`，彻底阻断 $2^p$ 暴力枚举），实测根结点未分支对角下界间隙 `(0.410, 0.526, 0.529)`、Fenchel 对偶间隙 `(1.238, 1.366, 1.390)`，强制要求叶节点携带多项式时间可检验对偶/下界见证。
  2. **`vr02__large_robin_spd_preconditioned_cg_multi_rhs`（部分子例程匹配 + 大规模迭代 PDE）**：`72 × 72 = 5,184` 未知量、`4,000` 倍高对比度电导率及四壁串联 Robin 边界（`conservative_grid_operator` 的零幽灵 Dirichlet 边界在此失配，而 `galerkin_preconditioner` + `solve_residual_checked_cg` 与 `fit_ridge_nuisance` + `profile_candidate_amplitudes` 精确匹配内部子问题）。
  3. **`vr03__prehistory_viscoelastic_control_panel_calibration`（跨窗口时序记忆 + 分组对照面板组合）**：`6` 中心 × `3` 分层队列，含非零前置应变历史、不规则时间步、`11.1%` 柯西尖峰、`5` 列参考对照面板（`12%` 缺失 `NaN`）与严重辛普森悖论。
  4. **`vr04__indefinite_helmholtz_wrapped_phase_inversion`（前置条件违背负控）**：`32 × 32 = 1,024` 复对称不定 Helmholtz 散射（Hermitian 部含 `18` 个严格负特征值，`min_eig = -0.1933`，违背一切实数/SPD/CG 前置条件）+ $2\pi$ 缠绕相位与离群脉冲。

---

## 3. `M9.0a` 核心实验结果（严格分离 `Primary 1x` 与 `Full Policy`）

### 3.1 每臂全部 `8` 个 Episode 总体表现（`4 任务 × 2 独立重复`）

| 指标 | Arm 1: `bare_no_library` | Arm 2: `knowledge_only_reference` | Arm 3: `progressive_executable_library` |
| :--- | :---: | :---: | :---: |
| **`Primary 1x` 严格通过率（`N=8`，重复间 SD）** | **`8/8` (`100.0%`, SD `0.0`)** | **`8/8` (`100.0%`, SD `0.0`)** | **`8/8` (`100.0%`, SD `0.0`)** |
| **`Full Policy` 严格通过率（`N=8`，重复间 SD）** | **`8/8` (`100.0%`, SD `0.0`)** | **`8/8` (`100.0%`, SD `0.0`)** | **`8/8` (`100.0%`, SD `0.0`)** |
| **`Primary 1x` 平均轮数（`max_turns = 28`）** | `9.00` | `9.62` | `10.62` |
| **`Primary 1x` 总 Token 数（`8` ep 合计）** | `1,626,388` | `2,344,308`（较 Arm 1 `+44.14%`） | `2,563,061`（较 Arm 2 `+9.33%`；较 Arm 1 `+57.59%`） |
| **`Primary 1x` 折算计费 Token 数** | `1,329,268` | `2,011,860` | `2,180,309` |
| **`Primary 1x` 平均墙钟耗时（秒）** | `323.75s` | `342.25s` | `378.10s` |
| **`Full Policy` 平均名义算力单位（`1.0u..2.0u`）** | `1.750u`（触发 `6/8` 第二分支*） | `1.375u`（触发 `3/8` 第二分支*） | `1.250u`（触发 `2/8` 第二分支*） |
| **`Full Policy` 总 Token 数（`8` ep 合计）** | `2,212,947` | `2,630,802`（较 Arm 1 `+18.88%`） | `2,723,347`（较 Arm 2 `+3.52%`；较 Arm 1 `+23.06%`） |
| **`Full Policy` 平均墙钟耗时（秒）** | `526.30s` | `440.86s` | `427.98s` |

> **关于 `Full Policy` 第二分支触发差异的方法学自审声明（`*`）**：核查 `00_episode_summary.json` 可知，在无宿主专属公开探针的非 `TB-Science` 新任务上，当 `staged_module_count == 0`（`Arm 1`）时，`M8OfflineContainerPlanner` 将 `public_verification_state` 标为 `"UNKNOWN"` 并触发第二分支（`6/8`）；而在 `Arms 2 & 3` 中因存在挂载目录或动态技能调用，状态转为 `SUPPORTED_IN_PARTIAL_SCOPE`（触发降至 `3/8` 与 `2/8`）。因此，**`Primary 1x`（三臂完全对称的 `1.00u` 单分支）才是严格无混杂的主比较口径**，绝不将 `Full Policy` 的 `1.750u -> 1.250u` 夸大归因为技能库自身的纠错增益。

### 3.2 六阶段自愿采纳与前置条件克制漏斗

| 漏斗阶段 | Arm 1: `bare_no_library` | Arm 2: `knowledge_only_reference` | Arm 3: `progressive_executable_library` |
| :--- | :---: | :---: | :---: |
| **Turn-1 推荐 `L1` 模块数（均值）** | `0.00` | `2.50` | `2.50` |
| **磁盘可导入可执行 `L1` 模块数** | `0.00` | **`0.00`**（挂载 `6.00` 份 `.reference.txt`） | **`6.00`**（`/tmp/hegel_wormhole_skills`） |
| **可按需 `--describe` 查询目录 `L1` 数** | `0.00` | `6.00` | `6.00` |
| **正匹配任务（`vr01..vr03`, `N=6`）`--describe` 查询** | `0/6`（`0` 次） | `3/6`（`10` 次；`4/6` 读取 `.reference.txt`） | **`6/6`（`100%`，共 `17` 次）** |
| **正匹配任务（`vr01..vr03`, `N=6`）动态调用 `L1`** | `0/6`（`0` 次） | `0/6`（`0` 次） | **`5/6`（`83.3%`，共 `58` 次）** |
| **正匹配任务（`vr01..vr03`, `N=6`）双向 AST 验证留存** | `0/6` | `0/6` | **`4/6`（`66.7%`，`100%` AST 一致）** |
| **正匹配任务（`vr01..vr03`, `N=6`）声明入口可达 `L1`** | `0/6` | `0/6` | **`4/6`（`66.7%`）** |
| **负控任务（`vr04`, `N=2`）违规调用 / 错误留存** | `0/2` / `0/2` | `0/2` / `0/2` | **`0/2`（`0` 次调用）/ `0/2`（`0` 留存）** |

### 3.3 各任务族零提示自愿采纳与克制明细（每臂每题 `2` 次独立重复）

| 任务族（每格 `N=2` 独立重复） | `Arm 1` `Primary 1x` 通过 / 均轮 / 总 Token | `Arm 2` `Primary 1x` 通过 / 均轮 / 总 Token | `Arm 3` `Primary 1x` 通过 / 均轮 / 总 Token | `Arm 3` 零提示契约查询、部分子例程选择与入口可达留存实况 |
| :--- | :---: | :---: | :---: | :--- |
| **`vr01`（证明携带 Box-L0, $p=20..22$）** | `2/2` / `11.0t` / `525,149` | `2/2` / `13.5t` / `913,742` | `2/2` / `14.5t` / `924,916` | `2/2` 执行 `--describe`（`4` 次）；**`2/2` 动态调用 `L1`**（`6` 次：`gram_diagonal_minorant`, `row_incidence_blocks`）；**`1/2`（`rep_02`）在最终提交物入口可达留存** `gram_diagonal_minorant`。 |
| **`vr02`（`5,184` 未知量 Robin SPD + 匹配）** | `2/2` / `7.0t` / `284,663` | `2/2` / `7.0t` / `378,151` | `2/2` / `9.5t` / `540,649`（`rep_02`: `7t`, `186,256`） | `2/2` 执行 `--describe`（`8` 次）；**准确避开边界失配的 `conservative_grid_operator`**，**`2/2` 动态调用（`44` 次）且 `2/2` 入口可达留存** `galerkin_preconditioner` + `solve_residual_checked_cg`！ |
| **`vr03`（前置记忆粘弹性 + 对照面板）** | `2/2` / `10.0t` / `494,907` | `2/2` / `9.5t` / `587,075` | `2/2` / `10.5t` / `669,335` | `2/2` 执行 `--describe`（`5` 次）；**`1/2`（`rep_02`）动态调用（`8` 次）且入口可达留存** `align_reference_panel`、`balanced_cell_weights`、`weighted_column_fill`。 |
| **`vr04`（复不定 Helmholtz 负控）** | `2/2` / `8.0t` / `321,669` | `2/2` / `8.5t` / `465,340` | `2/2` / `8.0t` / `428,161` | **`0/2` 查询、`0/2` 动态调用、`0/2` 留存 —— 前置条件违背场景 `100%` 干净弃权！** |

---

## 4. 严格校准的科学结论与下一阶段（`M9.1`）演进方向

1. **`M8.9.1` + `M9.0a` 已确凿证实的工程与机制结论**：
   - **验证核与审计卫生全面闭环**：`R1..R7` 全部反例已完成代码修复与测试固化；`M8.9` 封存 `hs01` 提交物在完备连续凸 QP 证书核下 `18/18` 全通过。
   - **零提示自愿可执行复用与函数级前置条件辨识真实成立**：在题目完全不提任何技能名的条件下，`gpt-6.2-sol` 在正匹配任务上 `6/6` 主动调用 `--describe`、`5/6` 动态执行 `L1` 原语（`58` 次调用）、在 `vr02` 上 `2/2` 精准跳过零幽灵 Dirichlet 算子而复用 AMG 预条件子与残差核验 CG、`4/6` 将完整双向 AST 校验一致的 `L1` 闭包保留进最终入口可达产物，并在复不定 Helmholtz 负控 `vr04` 上实现 `0/2` 违规调用。
2. **`M9.0a` 同样诚实揭示的经济净收益边界（天花板饱和与 Turn-1 提示税）**：
   - 在 `Bare`（Arm 1）单分支 `9.0` 轮即达到 **`8/8`（`100%`）通过率天花板** 的单文件自包含任务上，无条件在 Turn-1 注入参考卡既无法进一步提升通过率（`100% -> 100%`），也会带来净 Token 开销（`Primary 1x`：Arm 2 较 Arm 1 `+44.14%`，Arm 3 较 Arm 1 `+57.59%`）。
   - 三臂正交拆解清晰表明：**其中 `77%` 的主分支 Token 增量（`+57.59%` 中的 `+44.14%`）在“仅知识臂”（`Arm 2 - Arm 1`）就已产生**（源于无条件 Turn-1 卡片上下文与参考文档读取），而真正由可执行导入与打包后重验带来的边际开销仅 **`+9.33%`**（`Arm 3 - Arm 2`，整策略下仅 `+3.52%`）。
3. **`M9.1` 的明确架构方向（按需零首轮税检索 + 非饱和高难多组件基准）**：
   - **取消无条件 Turn-1 卡片注入（Zero-Turn-1-Tax Demand-Triggered Retrieval）**：默认以 `0` 额外技能 Token 启动（与 `Bare` 完全一致），仅当主分支遭遇实证 `nogood` / 残差瓶颈或主动发起轻量检索时才按需拉取契约与挂载模块，从而在常规可解任务上将首轮提示税精确压至 `0%`，同时将可执行复用聚焦于 `Bare 1x < 100%` 的真实多模块硬瓶颈任务。
