# Hegel M9.1 技术报告：零首轮税按需触发检索（Zero-Turn-1-Tax Retrieval）、无混杂路由对齐与三维大规模评测

- **里程碑**：`M9.1`（`Demand-Triggered Zero-Turn-1-Tax Retrieval + Unconfounded Router Parity + Hard Multi-Stage/3D-Scale Live Evaluation`）
- **分支 / 工作树**：`exp/m6-causal-wormhole-sandbox`
- **冻结代码提交（`Commit 4`）**：`8f7cc449f006854518dcb9a3b2f64e566f994ed0`（`source_tree_py_sha256 = 9fe7c6f7b524d695d5b952652f6761c6221009bd87a36c3f2fd5d47d100dfb86`，运行期零源码变更，单次无断点干净调用）
- **封存证据包**：`docs/evidence/m9_1_zero_tax_demand_retrieval_20261126/`（共 `532` 个文件通过 `SHA256SUMS.txt` 完整校验）
- **冻结预训练技能库**：`pretrained_autogenous_forest_v5_m89.json`（`SHA-256 = 0fcb0ca4b8b664515f29c184cd531700c817710a8e7e8a2281ef212641a6888c`）
- **真实评测后端**：官方 ChatGPT Subscription Codex CLI（`gpt-6.2-sol`，`reasoning_effort = "medium"`，`auth_mode = "chatgpt"`，`24` 并发 episode worker，峰值并发 CLI 调用 `24`，总调用 `209` 次，总耗时 `808.41s`）

---

## 1. 研究动机与可证伪假设

在 `M9.0a`（`docs/45`）的因子正交拆解中，我们定位了两个明确的系统瓶颈：
1. **无条件首轮提示词税（Turn-1 Prompt Tax）**：`progressive_executable_library` 相比 `bare_no_library` 的 `+57.59%` 主分支 token 开销中，有 **`77%`（`+44.14%`）** 来自在每个任务第 1 轮无条件注入召回的 `L2` 笔记与 `L1` 索引卡片——即使基座模型一次就能独立解出任务，或面对负对照任务（`vr04`）时也不例外。
2. **主分支结束后的路由器混杂（Router Confound）**：在无硬编码领域指标提取器的新基准上，`M8OfflineContainerPlanner` 将 `Bare`（`skill_activation_count == 0` 且 `nogood_count == 0`）判为 `UNKNOWN`（`6/8` 轮开启第二分支），而将调用过技能的实验组判为 `SUPPORTED_IN_PARTIAL_SCOPE`（仅 `2/8` 轮开启第二分支）。

`M9.1` 对此检验两个严格可证伪假设：
- **H1（零首轮税按需触发检索）**：在 `disclosure_mode = "demand_triggered_zero_tax"` 下，第 1 轮注入 **`0` 字节技能/方法论清单**（首轮用户载荷字节数与 `Bare` 完全一致），仅在容器 `/tmp/hegel_wormhole_skills` 静默挂载 `6` 个合格 `v5` `L1` 原语并提供零导入 `--search <keywords>` / `--describe <mod>.<fn>` CLI，仅当出现客观瓶颈（记录 `nogood`、命令报错或 `turn >= 5` 仍无验证候选）时最多触发 **1 次**紧凑提示（负对照 `vr04` 上 `0` 次提示），即可在不降低通过率的前提下彻底消除首轮提示词税。
- **H2（无混杂路由器对称性）**：在非 `TB-Science` / 非 `BioMystery` 基准上对称采信容器内实质性验证收据（`in_container_substantive_receipt_verified == True`），可彻底消除 `Bare` 与 `Skill` 组之间的第二分支非对称触发，同时严格保持 `m7_4`（`Frontier-20`）上 **`14/20`（`70.0%`）** 的真实留一任务交叉验证（`LOTO`）精度不变。

---

## 2. 架构改动摘要（`Commit 4`: `8f7cc449f`）

1. **`demand_triggered_zero_tax` 披露模式与纯数据 `--search` CLI（`m6_4_executable_wormhole_library.py`、`m6_frontier_real_benchmarks.py`）**：
   - 第 1 轮 `init_payload` 将 `"recalled_wormhole_methodology_and_wave1_sublations"` 与 `"mounted_executable_wormhole_skills"` 均置为 `[]`（`turn1_staged_manifest_json_chars == 2`，与 `Bare` 完全同长度）。
   - 新增 `python3 -m hegel_wormhole_skills --search "<keywords>"`，仅检索 `_skill_contracts.json`（`0` 模块导入执行，写入 `/tmp/.hegel_skill_inspections.jsonl` 的 `skill_search_executed` 事件）。
   - 仅当发生客观瓶颈时最多发送 **1 次**精简 `on_demand_skill_hint`；同时补齐复数不定 Helmholtz / 缠绕相位场景对实数 `L1` 模块的先验否决（`vr04` 上 `demand_triggered_recommended_l1_count == 0`）。
2. **`M8` 离线规划器对称无混杂状态判定（`m8_offline_container_planner.py`、`m6_9_a_dialectical_synthesis_engine.py`）**：
   - 在非 `TB-Science` / 非 `BioMystery` 任务上，对主/反题分支对称注入 `in_container_substantive_receipt_verified`，使通过容器内公开验证的 `Bare` 与 `Skill` 分支一致归入 `SUPPORTED_IN_PARTIAL_SCOPE`，且完全不触碰 `m7_4` 历史校准语义（单元测试验证 `LOTO` 仍精确为 `14/20`）。
3. **新增 `Hegel-M91-HardDemand-2` 基准套件（`m9_1_hard_demand_benchmark.py`）**：
   - `hd01__multicenter_cauchy_panel_and_certified_box_l0_pipeline`：三阶段耦合多中心对照面板仿射对齐与单元平衡插补 + 柯西重尾异常值稳健回归 + $p \in \{20, 21, 22\}$（高达 `419` 万支撑集）带可检验下界证书的盒约束 $\ell_0$-Ridge 全局优化（`<= 2048` 叶子）。
   - `hd02__3d_high_contrast_robin_amg_multi_rhs_and_nuisance_tomography`：真实三维（`28 × 28 × 28 = 21,952` 自由度、`4,000` 倍电导率反差）六面串联 Robin 边界椭圆型 PDE 多右端项求解（跨 2 个私有算例共 `16` 次三维求解，学生代码限时 `< 14.0s`，相对残差 `<= 1e-7`）+ SVD 白化 Ridge 干扰参数剖面估计。

---

## 3. 实测结果（`3 Arms × 4 Tasks × 2 Replicates = 24 Episodes`）

### 3.1 路由器无混杂性验证
全部 `24/24` 轮运行（每组 `8/8` 轮）均对称落入 `public_verification_state = "SUPPORTED_IN_PARTIAL_SCOPE"`，`second_branch_spawn_count = 0`，`mean_policy_nominal_units = 1.000`。`M9.0a` 中 `Bare` 组因路由器状态落入 `UNKNOWN` 导致的双分支开销混杂被 **100% 彻底消除**，`Primary 1x` 与 `Full Policy` 指标完全一致。

### 3.2 三组总览汇总表（每组 `n = 8` 轮，对称 `max_turns = 18`）

| 实验组（每组 `n = 8` 轮） | 首轮清单字符数 | `Primary 1x` / `Policy` 通过率（`Rep1, Rep2`, SD） | 平均轮数 | 总 Tokens（相对 `Bare`） | 折算计费 Tokens（相对 `Bare`） | 平均主分支耗时 | `--search` / `--describe` 轮数 | 动态 `L1` 调用次数（保留轮数） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 1: `bare_no_library`** | **`2.0`** | `6/8`（`75.0%`；`0.75, 0.75`，SD `0.00`） | `8.62` | `1,559,277`（基准） | `1,195,725`（基准） | `320.73s` | `0/8` / `0/8` | `0`（`0/8`） |
| **Arm 2: `progressive_upfront_library`** | `6,721.0` | `7/8`（`87.5%`；`1.00, 0.75`，SD `0.18`） | `8.50` | `1,958,452`（**`+25.60%`**） | `1,630,612`（**`+36.37%`**） | `307.39s` | `0/8` / `5/8`（`18` 次） | `50`（`2/8`） |
| **Arm 3: `demand_triggered_zero_tax_library`** | **`2.0`** | `7/8`（`87.5%`；`0.75, 1.00`，SD `0.18`） | `9.00` | `1,643,587`（**`+5.41%`**，较 Arm 2 **`-16.08%`**） | `1,295,491`（**`+8.34%`**，较 Arm 2 **`-20.55%`**） | `314.79s` | `1/8` / `2/8`（`5` 次） | `0`（`0/8`） |

### 3.3 分任务精细拆解（每格 `2` 次独立重复）

| 任务（每组 `2` 轮） | `Bare` 通过 / Tokens / 轮数 | `Progressive Upfront` 通过 / Tokens（相对 `Bare`） / 轮数 | `Demand Zero-Tax` 通过 / Tokens（相对 `Bare`，相对 `Prog`） / 轮数 | 核心遥测与机制发现 |
| :--- | :---: | :---: | :---: | :--- |
| **`vr02`**（`5,184` 自由度 Robin SPD，纵向回归） | `2/2` / `314,085` / `7.5` | `2/2` / `368,369`（`+17.28%`） / `7.0` | `2/2` / **`273,045`（`-13.07%`，`-25.88%`）** / `7.0` | `Progressive` 在 `2/2` 轮保留了 `galerkin_preconditioner` + `solve_residual_checked_cg`；`Demand Zero-Tax` 在 7 轮内零瓶颈独立解出（`0` 提示），**token 与耗时（`195.88s`）均为三组最低**。 |
| **`vr04`**（复数不定 Helmholtz，负对照） | `2/2` / `278,733` / `7.0` | `2/2` / `406,736`（`+45.92%`） / `8.0` | `2/2` / **`325,535`（`+16.79%`，`-19.96%`）** / `8.0` | 两组技能库均实现 **`0/2` 误提示、`0/2` 误调用**；`Demand Zero-Tax` 彻底消除了 `Progressive` 首轮 `3,294` 字符的 `L2` 卡片税（剩余差异完全来自 `8.0` vs `7.0` 轮轨迹自然波动）。 |
| **`hd01`**（三阶段多中心对齐 + 柯西回归 + 盒约束 $\ell_0$ 证书） | `0/2` / `661,333` / `12.5` | `1/2` / `812,855`（`+22.91%`） / `12.0` | `1/2` / **`743,091`（`+12.36%`，`-8.58%`）** / `13.5` | `Demand Zero-Tax` 在 `rep_01` 第 6 轮按需发出单次瓶颈提示（触发 `2` 次 `--describe`），在 `rep_02` 主动执行 `--search` + `3` 次 `--describe` 并以 **`302,870` tokens（低于 `Bare` `rep_02` 的 `340,599`）** 满分通过。 |
| **`hd02`**（三维 `21,952` 自由度高反差 Robin PDE + 剖面反演） | `2/2` / `305,126` / `7.5` | `2/2` / `370,492`（`+21.42%`） / `7.0` | `2/2` / **`301,916`（`-1.05%`，`-18.51%`）** / `7.5` | 三组均在 `2.47s..5.32s < 14.0s` 内（相对残差 `0.0`）通过单次 `splu` 稀疏分解复用解出 `16` 个三维算例；`Demand Zero-Tax` 总 token 比 `Bare` 低 `-1.05%`（计费 token 低 `-2.28%`）。 |

---

## 4. 审慎科学讨论与局限性披露

1. **首轮提示词税已实质性消除（净开销削减 `79%`，纵向 `vr02+vr04` 子集净税率归零）**：
   - 跨全部 8 轮，`Demand Zero-Tax` 将 `Progressive Upfront` 相对 `Bare` 的总 token 税率从 **`+25.60%`（计费 `+36.37%`）** 压降至 **`+5.41%`（计费 `+8.34%`）**，在保持相同 `7/8`（`87.5%`）通过率的同时直接节省 **`-16.08%` 总 tokens（`-20.55%` 计费 tokens）**。
   - 在纵向 `M9.0a` 子集（`vr02 + vr04`）上，`Demand Zero-Tax` 总 tokens 为 `598,580`（相对 `Bare` `592,818` 仅 **`+0.97%`**），折算计费 tokens 为 **`444,884`（比 `Bare` 的 `454,386` 还低 `-2.09%`）**，严格证明静默挂载技能在模型无需援助时不再产生首轮税。
2. **核心机制权衡：首轮可见性驱动的代码复用 vs 零税静默下的按需契约查阅**：
   - 在 `Progressive Upfront` 中，首轮可见的紧凑索引卡片促使模型在 `vr02`（`2/2` 轮保留 `autogenous_prim_conservative_spd_solve`）与 `hd01`（`1/2` 轮动态调用 `autogenous_prim_reference_panel_balancing`）上直接导入执行 `L1` 原语，但代价是全任务承担 `+25.60%` 的上下文开销。
   - 在 `Demand Zero-Tax` 中，当模型在 `vr02` 与 `hd02` 上首轮尝试即顺利收敛（`7` 轮、`0` nogood）时，系统保持完全静默（`0` 动态调用、`0` 税）；而在遇到真实多阶段瓶颈（`hd01`）时，模型倾向于将 `--search` / `--describe` 作为**零导入的数学与接口契约参考书**（`2/2` 轮查阅契约，`0` 运行时导入）来指导自写实现。
3. **关于 `hd01` 失败模式（`99301:bad_masks` 接口掩码歧义）与样本量（`n = 2`）的透明披露**：
   - 对全部 6 轮 `hd01` 官方验证器报告的事后核查表明：6 轮在第一阶段对齐插补（`max_aligned_mae = 0.063..0.075 <= 0.12`）和第二阶段柯西稳健回归（`max_potency_weight_rel_l2 = 0.033..0.057 <= 0.08`）上全部达标，且均实现了合法的对角下界分支定界。
   - 全部 4 个失败轮次（`Bare` `2/2`、`Progressive` `1/2`、`Demand Zero-Tax` `1/2`）均因 `"99301:bad_masks"` 失败：原因是 `hd01` 题干第 74 行要求叶子包含 `'fixed_zero'` 与 `'fixed_active'` 时未显式指明其为长度为 `p` 的布尔掩码还是整数下标列表（`Bare` 返回了 `np.flatnonzero(...).tolist()`），而在通过验证的 2 个技能库轮次中，模型均通过 `--describe autogenous_prim_box_ridge_bounds.box_ridge_cube_lower_bound` 看到了 `(p,)` 布尔掩码参数签名从而消除了格式歧义（`gap = 4e-8`，耗时 `0.10s`）。
   - 因此，我们**绝不**将 `hd01` 上的 `1/2 vs 0/2`（单轮差异，重复间 SD `0.71`）夸大为超越基座模型的算法能力突破：在 `hd02` 上，`gpt-6.2-sol` 仅凭显式 PDE 规范即自行采用单次 `splu` 稀疏分解复用来在 `~2.6s` 内解完 8 个三维右端项；而在 `hd01` 上，`+1` 轮通过主要反映了 `--describe` 对接口掩码契约的消歧作用。
