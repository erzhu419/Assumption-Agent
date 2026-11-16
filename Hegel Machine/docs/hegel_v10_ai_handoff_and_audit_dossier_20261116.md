# Hegel Machine v10 — 面向外部 AI 审阅与交接的全量工程台账、机制原理与验证审计档案（AI-to-AI Handoff & Audit Dossier）

> **目标读者（Target Reader）**：外部审阅／协作／接手开发的 AI 智能体（如 GPT-6、Codex、Claude、Gemini 等自主科研与代码审计模型）。
> **文档定位**：在 `docs/hegel_v10_technical_report_20261116.md`（学术专著）与路线合成报告的基础上，以**零模糊、代码与回执逐项可核验（code-exact & receipt-verifiable）**的标准，完整记录自接手本项目（承接 `v9 §118` 及 `106110602 -> 106110603` 三臂 `0/3` 未交付预测器瓶颈）以来：
> 1. **具体做了什么**（新增／修改的全部模块、AST 原语、JSON 承诺契约、`ResearchConfig` 开关、主循环钩子与 Git 提交链）；
> 2. **为什么这么做**（对历史 `v0.1–v9` 失败根因的代码级法医解剖、各阶段数学不可能性定理、对 AWM 论文与 GPT-6 `§117` 建议的取舍边界，以及开发途中真实踩中并修复的底层工程 Bug）；
> 3. **定量结果如何**（全部 **13 份前瞻 JSON 回执**的逐臂、逐队列、逐指标真实数据，含符号计算账本、在线 LLM Token 账本、形式化证明账本，以及全部非 100% 边界单题的透明归因）；
> 4. **是否均已验证及如何复核**（28/28 自动化回归测试、篡改证书拦截测试、无噪 Oracle 漂移对照、COW 历史封账零改动校验，以及供审阅 AI 直接运行的核查命令）。

---

## 0. 仓库坐标、Git 提交谱系与硬性认识论不变量（System State & Hard Invariants）

### 0.1 工作区与完整 Git 提交链
- **仓库根目录**：`/home/erzhu419/mine_code/Asumption Agent/Hegel Machine`（Windows UNC：`\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Asumption Agent\Hegel Machine`）
- **开发分支**：`codex/hegel-cognitive-mainline`（严格仅本地提交，**未执行 `git push`**）
- **接手前基线提交**：`686921d1`（`feat(hegel): connect parent-bound method composition and update evidence`），当时工作区尚有未提交的 `v9 §§112–118` 证据回执及 `markdown/hegel_v10/` 审阅材料。
- **接手后完整提交链（按时间顺序）**：
  1. `ae5e8b63` — `chore(hegel): sync v9 §§112-118 evidence receipts and v10 design materials`（18 文件，`+4,737` 行；先将接手前已有 `v9` 收尾材料与 `v10` 设计输入干净归档，确保 `v10` 代码改动边界清晰可审计）；
  2. `5af9576a` — `feat(hegel): implement v10 world commitments, premise audit, open-grammar gate, active threshold probing, and POMDP latent growth`（对应 `§§119–121`：COW 历史 135 工作区普查、真实 `602 -> 603` 新账本后继正收益验证、Phase 1 跨尺度世界承诺与前提审计、Phase 2A 条件边与非守恒修订、Phase 2B 在线 `gpt-5.6-sol` 前提门控开放语法合成与零调用迁移、Phase 2C 开放阈值主动二分与一阶 POMDP 生长；67 文件，`+26,778 / -222` 行）；
  3. `7335e0c4` — `feat(hegel): add v10 Phase 3 higher-order/coupled POMDP discovery and open-grammar POMDP synthesis`（对应 `§122`：Phase 3A 二阶阻尼谐振 `AR(2)` 与跨对象空间耦合 POMDP、Phase 3B 在线 `gpt-5.6-sol` 非线性双线性／二次能量 POMDP 受控合成与零调用迁移；20 文件，`+5,712 / -78` 行）；
  4. `d6522772` — `feat(hegel): add v10 Phase 4 curved hypersurface probing, continuum PDE subgraph growth, supercritical VdP POMDP, and monograph`（对应 `§123`：Phase 4A 三参数耦合曲面超平面主动边界设计、Phase 4B 无位置连续介质场一维环 PDE 子图生长与超临界 Van der Pol $\rho > 1$ 极限环 POMDP、Phase 4C 独立技术专著；23 文件，`+7,511 / -57` 行）；
  5. `1949b9a9` — `feat(hegel): add v10 Phase 5 manifold/graph PDEs, stochastic SDE 2SLS-IV audit, active resonance, and algebraic proof certificates`（对应 `§124`：Phase 5A 二维环面／有界 `Neumann<->Dirichlet`／不规则图拉普拉斯 PDE、Phase 5B 强过程噪声 SDE 两阶段最小二乘工具变量 `2SLS-IV` 审计与近临界极限环主动共振、Phase 5C 零公理有理代数证明内核 `SymbolicProofKernel`／Farkas 不可达证明／AWM §4.3 差异触发前提修复；25 文件，`+8,845 / -106` 行）。
- **外部独立挂载盘与未暂存文件保护状态**：
  - 独立实验虚拟盘 `D:\Hegel-Active-Experiments-20261013\hegel-snapshot-native.vhdx`（挂载于 WSL `/mnt/hegel-snapshot-cow`）：历史已封账的 5 份成本与状态回执（累计 `[4,409,116, 5,849,706]` 计算、`7,080` 行）在实验前后经脚本校验 **`size_bytes` 与 `mtime_ns` 100% 零改动**；新增复核单独写入 `/mnt/hegel-snapshot-cow/hegel-v10-cow-602-603-successor-20261116`（`0.13 MB`）。
  - 仓库内历史外部压缩包 `markdown/BAPR_HRO_6e46267_V47_diagnosis_and_prototype_20261114.zip` 始终保持未暂存（untracked）。

### 0.2 全周期严格执行的 5 条硬性反自欺／认识论不变量
任何审阅 AI 在检查 `src/hegel_machine/cognition/` 代码时，均可直接验证以下不变量未被任何捷径破坏：
1. **零私有物理状态读取（Zero Oracle Leakage）**：智能体、前提审计器、主动探针规划器与世界承诺巩固器**绝不读取**任何实验室实例的私有属性（如 `_radius`、`_coupling`、`_latent_decay`、`_latent_freq`、`_latent_nonlinearity`、`_condition_threshold`、`_condition_curvature`、`_physical_controls`、`_measurement_names`、`_site_ids`、`_graph_edges` 等）。所有推断仅依赖公开 `TaskSpec`（`feature_names`、`feature_bounds`、`max_probe_length`）与付费调用 `laboratory.acquire(probe, channels)` 返回的 `observations` 及公开传感器 `before -> after` `transitions`。
2. **双预算全额计费（`SearchBudget` + `Observation / Measurement Budget`）**：每次线性回归、候选程序求值、双射筛查、算子矩阵求解与等价门控检验均调用 `budget.spend(...)` 扣减 `SearchBudget`；每次实验探针均计入 `observation_cost`（步数）与 `measurement_cost`（通道费率 × 步数）。
3. **符号科研账本与在线 LLM Token 账本严格物理隔离**：
   - 13 份前瞻回执中，**11 份纯符号实验回执以及全部跨尺度后继迁移阶段（Phase 2B Stage B、Phase 3B Stage B）均强制 `model_calls == 0`、`total_tokens == 0`**；
   - 仅在专门验证“超出全部内置符号原语库的开放算子发现”的 Phase 2B Stage A 与 Phase 3B Stage A 中，通过 `open_grammar_author.py` 在线调用 `POST /v1/responses`（模型 `gpt-5.6-sol`），从环境变量瞬时读取 `OPENAI_API_KEY` / `OPENAI_BASE_URL`，**零密钥落盘**，并逐次记录 `input_tokens`、`output_tokens`、`total_tokens`。
4. **未放宽的确认门槛与零测量纯控制升级要求**：
   - 确定性装置独立未曝光 48 步轨迹确认门槛保持 `minimum_improvement = 1e-7`（`fresh48_mse <= 1e-7`）不变；
   - 确认通过后，智能体仍须克服未知三次执行器畸变 $q = c_q + \kappa c_q^3$ 与离散焦点取整 $\text{round}(\text{focus})$，经 `fit_instrument_model` + `_distinguish_instruments` 将预测器从付费细传感器通道 `['fine']` 升级为**零在线测量成本的纯控制通道 `selected_channels == []`**。
5. **旧基线零削弱与默认向后兼容**：`ResearchConfig` 中 `world_structure_commitments: bool = False`、`transition_premise_audit: bool = False` 默认关闭；不开启时 `v4–v9` 的 `Archive` 与 `Fixed-Common` 行为逐字节不变。

---

## 1. 接手时的法医级根因审计：为什么 `v0.1–v9`（`§§1–118`）始终未证明后继正收益？

在动手写任何 `v10` 代码前，我们先将 `markdown/initial thought/`（`黑格尔机.md`、`Hegel_assumption.md`、`黑格尔机和泛函分析.md`、`广义对位关系.md`）、`markdown/Meta prior.md`、`docs/hegel_machine_next_construction_questions_20261027.md`（`§§1–118`）、`markdown/hegel_v10/`（AWM 论文 PDF、`GPT_diagnosis_v10.md`、`复核范围与v9最小修订建议.md`）与 `src/hegel_machine/cognition/` 源码及 `v9` 实验回执做了全量交叉审计，定位出导致 `501 / 601 / 602 / 603` 停滞的 **4 个精确到代码行的根因**：

### 1.1 四个代码与账本级病灶
1. **病灶 1（`601` 与 `603` 三臂同构假象）**：
   - 在 `composed_abc_20261101.json`（`601`）与 `position_method_successor_20261107.json`（`603`）中，`archive`（A 臂）和 `fixed_pr2`（B 臂）被传入了与 `pr3`（C 臂）完全相同的 `raw_sources / program_sources`，并在任务入口调用廉价确定性重建函数（如 `reconstruct_partial_role_source` 在 `603` 上仅耗 **15 单位计算**，在 `601` 上耗 `4,831` 计算）。结果进入真正决策点时，A/B/C 三臂持有逐字节一致的方法池，C 臂反而因 Common 序列化多花 `+26,462` 计算。
2. **病灶 2（无世界结构内容的动作拼接宏被直接父稿 `Pi1` 吸收，且表面可用性条件组外泛化失败）**：
   - 在 `501`（$N=4$）上，虽然组合程序 `revise_then_reidentify`（`Pi3 -> Pi1`）赢了单独的固定 `Pi3`（`1.13e-20` vs `0.00795`），但在 `common_object_research.py:338-381` 中，把普通 `Pi1` 直接绑到 333 池最优保留父稿 `selected_before` 上，产出了**完全相同的终测程序且少花 `1,109` 计算**（`20,368` vs `21,477`）；
   - 在 `method_conditions.py`（`§§85–99`）中，调用前切换条件退化为检查某个语法绑定字段是否存在，未曝光源组外选对率仅 `1/2`。
3. **病灶 3（`602 -> 603` 窄过程前置条件 `remaining_owner_count == 2` 永不触发＋回调串题＋空候选崩溃）**：
   - `object_dependency_methods.py:214-265` 的 `form_position_owner_method` 硬编码要求首轮位置实验后 `remaining_owner_count == 2` 且固定复用角 `[0, 2]`。在独立五对象源 `106110603` 上，A 臂首轮四角实验一步将位置归属从 `5 -> 1`（`cr_retained = ['m1_458488']`），导致 `== 2` 前置条件调用次数为 `0`；
   - B/C 两臂因早期选择器子题留下的 `completed_relation` 串题，将位置题误判为已完成并重复购买角 1 达 **57 次**（耗尽 `216` 行观测），最终三臂在五对象结构拟合返回空列表时于 `_record_program_comparison` 触发 `NoneType.get` 异常退出（`0/3` 交付预测器）。
   - **履行 `§117` 第二步的全盘只读普查**：我们扫描了 `/mnt/hegel-snapshot-cow` 全部 **135** 个历史工作区与 **229** 份位置状态制品（固化于 `paid_position_round1_census_20261116.json`），证实除训练源 `106110602` 自身（`5 -> 2 -> 1`）外，满足首轮后 `len(retained_owners) == 2` 的独立非源现场数严格为 **0**。因此彻底停止窄过程补丁回溯。
4. **病灶 4（真正卡死 `602/603` 及跨尺度后继的深层科学断层）**：
   - **固定元数阻断知识迁移**：`v8` 库全部 `333` 条确认先验均绑定在 $N=4$ 具体槽位上；`structural_discovery.py:709` 与 `unified_structure_transfer.py:32-34` 一见 `len(source_schema.objects) != len(schema.objects)` 直接跳过迁移！
   - **`object_scope_revision.py` 休眠且锁死源半径常数**：仓库虽有 `v0.74` 的全对象重写器，但（a）从未接入部分角色竞争入口 `competing_object_programs.py`，（b）锁死了源装置的固定数值 `radius.value`，目标装置物理半径一旦变化即失配；
   - **退回盲搜时只搜无向核**：`_discover_structure` 默认只构造无向 `neighbor_sum(..., name=None)`，在有向传递的 `602`（`relation_direction="right"`）上训练 MSE 卡死在 `0.00149 >> 1e-7`；
   - **`window_total_event` 单步试点对焦点 `f` 完全失明**：在 `structure_lab.py:190-211` 中，`event = flows[-1] - flows[-2] if len(flows) > 1 else 0.0`。任何单步重置试点（`position == 0`）上 `event == 0`，输出退化为与 `focus` 无关的 `total`，导致旧焦点试点连续 25 次返回 `uninformative_pilot`，留下 $5! \times 5! = 14,400$ 的 `(x, f)` 排列悬崖！

### 1.2 对 AWM 论文（arXiv:2610.40503v2）与 GPT-6 `§117` 诊断的精准吸收与拒绝边界
- **核心吸收**：将第二循环（方法与抽象学习）从“无内容的阶段动作宏”彻底迁移到**有物理内容、跨元数参数化的世界结构承诺（World-Structure Commitments）**与**显式支撑前提依赖图（`PremiseDependencyGraph`）**之上，并完整走通 AWM 的三大闭环：
  1. **探索与主动边界辨识（Stage 01）** $\to$ 多归属对比、一维／多维耦合曲面二分、因果起振与近临界相位共振、半整数切点夹逼；
  2. **严格保持性巩固 vs. 保守科学修订（Stage 02 + App. C.6）** $\to$ 四重无损等价门控（含 $\operatorname{Rej}(T) \supseteq \operatorname{Rej}(S)$ 与 $\operatorname{Aut}(\mathcal{M})$ 规范不变性）＋ 16 节点前提图逐层消去与局部修订；
  3. **形式化可解／不可达证明与差异触发前提修复（Stage 03 + §4.3）** $\to$ 零公理有理多项式内核 `SymbolicProofKernel`、跨承诺零空间消元涌现守恒律、Farkas 对偶势函数不可达证书及 `SufficientCover` 靶向前提修复。
- **明确拒绝**：绝不在 `Program` 之外另起炉灶搞平行玩具 DSL，绝不重演早期 `M3` 静态证书包裹 50,000 棵 AST 暴力枚举的老路，绝不碰历史封账回执一字节。

---

## 2. `v10` 核心契约、AST 算子、JSON Schema 与运行时调用链全景

### 2.1 `ResearchConfig` 新增配置开关全表（`src/hegel_machine/cognition/models.py:167-180`）

| 字段名 | 类型 | 默认值 | 所属阶段与精确控制语义 |
| :--- | :--- | :--- | :--- |
| `world_structure_commitments` | `bool` | `False` | 主开关：启用跨元数世界承诺巩固、等价门控与实例化（默认 `False` 保证历史基线逐字节不变） |
| `transition_premise_audit` | `bool` | `False` | 主开关：启用 `PremiseDependencyGraph` 在已付转移与轨迹上的分层前提审计 |
| `expanded_premise_revisions` | `bool` | `True` | 允许条件边（`source_greater`/`less`）与非守恒（`sink`/`nonlocal`）保守修订（Phase 2A+，需主开关开启） |
| `active_threshold_probing` | `bool` | `True` | 将自定义谓词比较阈值 `c_k`（`predicate_threshold`）纳入 `RelationIntervalResearch` 主动二分（Phase 2C-1+） |
| `latent_state_discovery` | `bool` | `True` | 启用一阶 POMDP `P_markov_state` 证伪、水库递推消去与跨尺度极点重标定（Phase 2C-2+） |
| `higher_order_latent_discovery` | `bool` | `True` | 启用二阶复共轭谐振 `AR(2)`（`P_latent_order`）与跨对象空间隐耦合（`P_latent_coupling`）（Phase 3A+） |
| `curved_hypersurface_probing` | `bool` | `True` | 启用三参数耦合曲面 `(r_0, c_0, c_1)` 凸多胞体裁剪与直径极值最大杠杆主动探针（Phase 4A+） |
| `continuum_field_discovery` | `bool` | `True` | 启用 `P_role_partition` 证伪与无位置传感器一维周期 PDE 环重构及 `ring_sum` 承诺（Phase 4B-1+） |
| `nonlinear_limit_cycle_discovery` | `bool` | `True` | 启用原点线性不稳定（$\rho > 1.0$）超临界 Van der Pol 三次极限环 POMDP 辨识（Phase 4B-2+） |
| `manifold_pde_discovery` | `bool` | `True` | 扩展无位置场至二维环面 $\mathbb{Z}_{N_x}\times\mathbb{Z}_{N_y}$、有界 `Neumann`/`Dirichlet` 链与不规则图拉普拉斯 PDE（Phase 5A） |
| `stochastic_premise_audit` | `bool` | `True` | 启用内禀过程噪声 SDE 检测（`P_stochastic_process_noise`）与统计前提审计（Phase 5B-1） |
| `stochastic_estimation_method` | `str` | `"2sls_iv"` | 选择 `"2sls_iv"`（滞后控制工具变量＋外生脉冲投影）或消融对照 `"naive_ols"`（Phase 5B-1） |
| `active_resonance_probing` | `bool` | `True` | 启用近临界极限环（$\rho \approx 1.0$）固有频率估计与多步同相共振激励探针（Phase 5B-2） |
| `algebraic_proof_synthesis` | `bool` | `True` | 启用 `SymbolicProofKernel` 跨承诺不变量合成、Farkas 不可达证明与差异触发修复（Phase 5C） |

### 2.2 `Program` 语言新增与扩展算子精确语义（`language.py` & `simplification.py`）

1. **`relation_sum`（跨元数开放边谓词与有向／对称交互算子，Phase 2A/2B/3A/4A）**：
   - 结构：`Program("relation_sum", name=direction, children=(radius_node, predicate_ast, transfer_ast, *per_object_tuples))`
   - 语义：对满足距离方向 `direction in {"right", "left", "symmetric"}`（$< r_0$）且边谓词 `predicate_ast(edge_q_src, edge_q_dst, edge_x_src, edge_x_dst)` 为真的有序对象对累加 `transfer_ast`（有向模式下源减目标加，严格反对称守恒；支持耦合隐状态槽位 `edge_z_src`, `edge_z_dst`）。
2. **`accumulate`（层次化因果隐状态水库算子，Phase 2C/3A/4B-2）**：
   - `name=None`：一阶指数递推 $z_t = p z_{t-1} + d_t$（`value = p`）；
   - `name="ar2"` / `"ar2_lag1"`：`children=(const(a1), const(a2), drive)`，计算二阶阻尼谐振 $z_t = a_1 z_{t-1} + a_2 z_{t-2} + d_{t \text{ or } t-1}$（$a_1 = 2\rho\cos\omega,\, a_2 = -\rho^2$）；
   - `name="vdp"` / `"vdp_lag1"`：`children=(const(a1), const(a2), const(beta), drive)`，计算超临界三次饱和极限环 $\operatorname{clip}_{[-6,6]}(a_1 z_{t-1} + a_2 z_{t-2} - \beta z_{t-1}^3 + u_{t \text{ or } t-1})$。
3. **`ring_sum`（一维 $\mathbb{Z}_N$ 循环规范不变 PDE 差分算子，Phase 4B-1）**：
   - `name in {"laplacian", "grad_back", "grad_fwd"}`，`children=(f_0, q_0, ..., f_{N-1}, q_{N-1})`（任意 $N \ge 3$）。
4. **`manifold_sum`（高维环面、非周期边值与一般图拉普拉斯算子，Phase 5A）**：
   - 二维环面：`name="torus2d_{lap_x|lap_y|grad_bx|grad_by}:{Nx}x{Ny}"`，满足 $\operatorname{Aut}(\mathbb{T}^2) = \mathbb{Z}_{N_x}\times\mathbb{Z}_{N_y}$ 平移规范不变；
   - 一维非周期有界链：`chain_lap_neumann`（端点零通量反射 $q_1-q_0, q_{N-2}-q_{N-1}$）、`chain_lap_dirichlet`（端点零值吸收 $q_1-2q_0, q_{N-2}-2q_{N-1}$）、`chain_adv_zeroflux`、`chain_boundary_sum`（Dirichlet 全局墙汇通量 $q_0 + q_{N-1}$）、`chain_focal_boundary`；
   - 一般不规则加权图：`name="graph_laplacian"`，`children` 为每条规范边 `(const(w_uv), f_u, q_u, f_v, q_v)` 五元组序列，计算 $\sum_{(u,v)\in E} w_{uv}(f_u - f_v)(q_v - q_u)$。

### 2.3 `UnifiedResearchAgent.run()` 内部主循环完整接线路径
审阅 AI 可沿 `src/hegel_machine/cognition/unified_research.py` 以下精确调用链核对运行时行为：
1. **表征与对象／流形归纳（`_research_objects`, `L519–845`）**：
   - 采购初始 12 行随机依赖观测与控制隔离探针后，调用 `_audit_object_transition_premises`（`L226–325`）：
     * 若 `continuum_field_discovery` 开启且 `is_position_free_continuum_candidate(model)` 为真（`position_candidates == {}` 且存在 1-to-1 `(q, u)` 通道）：优先路由至 `audit_manifold_pde_premises`（Phase 5A）或 `audit_continuum_field_premises`（Phase 4B-1），证伪 `P_role_partition` 并重构环／二维环面／有界链／一般图 `ObjectSchema`；缺选择器保持行时由 `_object_probe` 调用 `propose_manifold_probe` / `propose_continuum_probe`；
     * 若存在 `x` 传感器且 `stochastic_premise_audit` 检出重置行过程噪声方差显著（`is_stochastic == True`）：路由至 `audit_stochastic_transition_premises`（Phase 5B-1），执行 `2SLS-IV` 隐状态估计、转移消去、Wald 守恒检验、$\chi^2/F$ 位置双射与半径区间恢复及实现态 `after` 精确焦点配对；
     * 否则路由至确定性／POMDP `audit_transition_premises`（Phase 1–4B-2），逐层审计 `P_markov_state` $\to$ `AR(1)/AR(2)/Coupled/VdP` 消去 $\to$ `P_conservation` $\to$ `P_exchange_form` $\to$ `P_interaction_direction` $\to$ `P_position_ownership` $\to$ `P_locality_radius` $\to$ `P_focus_membership`；
   - 角色确认后，若存在未收窄的三参数耦合曲面多胞体（`verified_hypersurface_polytope`），在 `L720–806` 循环调用 `plan_coupled_hypersurface_probe` 执行先验半径下界扩张、直径极值最大杠杆切片及亚微米半径二分收口；若为近临界极限环且激发不足，在 `L807–835` 调用 `plan_near_critical_resonance_probe` 执行同相共振激励。
2. **世界结构承诺巩固、跨尺度实例化与科学前提修订（`_propose_schema_structures`, `L326–517`）**：
   - 从 `self.library.meta_priors` 调用 `consolidate_manifold_pde_commitments` / `consolidate_continuum_field_commitments` / `consolidate_world_structure_commitments`（内部强制执行 `check_equivalence_reorganization` 四重无损门控）；
   - 若前提审计无冲突触发器（`triggers == []`），调用 `instantiate_*_commitment` 在新对象数 $N_T$ 上直接跨尺度实例化，并在可行区间内重标定连续参数 $(r_0, c_0, c_1, \alpha, \rho, \omega, \beta)$ 与线性读出列（包括在源任务上系数为 0 的基底列）；
   - 若存在局部前提证伪触发器（`triggers != []`），调用 `propose_scientific_premise_revision`（或 `Neumann <-> Dirichlet` 边值修订），仅局部替换被实证推翻的前提节点并保留其余结构骨架。
3. **冻结确认、SDE 重复轨迹噪声挑战与纯控制仪器升级（`L6760–7450`）**：
   - 在独立新轨迹上执行 `confirm_frozen_program`（确定性门槛 `loss <= 1e-7`；随机 SDE 下经 `assess_uncertainty` 排除全部可预测结构／时序残差后，由 `challenge_noise` 购买同控制序列独立重复轨迹运行 `assess_noise_replication`，检验乘积残差无偏性）；
   - 通过确认后，调用 `remember_confirmed_selection` 写回 `meta_priors`，并经 `fit_instrument_model` + `_distinguish_instruments` 校准控制到传感器的非线性映射，交付 `selected_channels = []`。

---

## 3. 全阶段工程演进明细台账（COW + Phase 1 至 Phase 5C）

### 3.1 真实历史 COW 挂载普查与 `602 -> 603` 新账本后继正收益验证（`§§119–120`）
- **做了什么**：编写 `scripts/run_v10_cow_602_603_study.py`，只读普查 `/mnt/hegel-snapshot-cow` 135 个历史工作区；在 `object_interventions.py` 新增支持任意 $2 \le |S| \le N$ 存活归属的 `plan_generalized_position_followup`（保持旧 `len == 2` 代码零改动）；从真实 `v8` 333 先验库出发，在原注册紧预算下测试 `106110602` 与 `106110603` 的无 v10 基线、冷启动 v10 与 `602 -> 603` 顺序后继 v10。
- **为什么这么做**：`602 -> 603` 是整个项目此前卡死的核心里程碑（`§116` 三臂 `0/3`），必须在**原封不动的历史先验库与原注册限额**（`603`: `obs<=256, compute<=200,000, meas<=6,144`）下正面证明后继正收益。
- **定量结果（`v10_cow_602_603_new_ledger_20261116.json`，`model_calls = 0`）**：
  - `baseline_without_v10`：`602` `unresolved`（`MSE = 3.26e-2`, `compute = 296,318`），`603` `unresolved`（`MSE = 5.68e-2`, `compute = 146,002`）$\to$ **`0/2`**；
  - `v10_from_v8_archive`（冷启动自 v8 库）：`602` `completed`（`MSE = 9.01e-19`, `[]`, `compute = 128,064`），`603` `completed`（`MSE = 1.17e-20`, 停留在 `['fine']`, `compute = 199,912`，距 20 万上限仅剩 88 单位无力校准纯控制）$\to$ **`2/2`**；
  - **`v10_sequential_after_602`（继承 `602` 后的真实后继）**：`603` `completed`，**科学计算从 `199,912` 骤降至 `67,028`（节省 `66.47%` / `2.98x` 加速，比失败基线 `146,002` 还低 `54.09%`），查询数 `14 -> 10`，随机终测 `MSE = 8.04e-19`，且成功将通道从 `['fine']` 升级为 `selected_channels = []`（纯控制终测 `8.18e-20`）**。
- **验证状态**：**已验证**（含 5 份 COW 历史封账文件前后 `size_bytes` 与 `mtime_ns` 逐字节零变化校验）。

---

### 3.2 Phase 1：四重无损等价门控、支撑前提依赖审计与空库 $N=4 \to N=5$ 迁移（`§119`）
- **做了什么**：创建 `world_structure_commitments.py` 与 `premise_dependency_audit.py`，接入 `UnifiedResearchAgent` 与 `CommonObjectResearchAgent`；编写 `scripts/run_v10_world_structure_study.py` 与 `tests/test_v10_world_structure_and_premises.py`。
- **为什么这么做 & 修复的关键问题**：
  - 为防止 `v9 §77` 式“合并丢列／静默粗化”，门控函数 `check_equivalence_reorganization` 强制同时满足四项条件：完整拟合列契约（含零权重读出列）、符号共享参数组、支持集预测一致（$\le 10^{-9}$）与反例拒绝集单调不收缩 $\operatorname{Rej}(T) \supseteq \operatorname{Rej}(S)$；
  - 修复了 $N=5$ `window_total_event` 因单步试点 `event == 0` 导致的 25 次 `uninformative_pilot` 死循环，改为在已购多步连续轨迹上直接审计 `P_focus_membership`。
- **定量结果（`v10_ablation_and_cross_scale_successor_20261116.json`，`model_calls = 0`）**：
  - 等价门控消融：合法跨元数提升 $N \in \{3,5,6\}$ 零误差通过；3 类退化粗化（删零权重列、`right -> symmetric` 粗化、收缩滞后阶数）**`3/3`（100%）拦截**；
  - 冲突前提隔离（对应 AWM App. C.6）：6 组五对象冲突装置上，平铺轨迹 MSE 定位 **`0/6`**，显式 `PremiseDependencyGraph` 定位 **`6/6`（100%）** 并将守恒装置位置域一步收窄 `[5,5,5,5,5] -> [1,1,1,1,1]`；
  - 空库 $N=4$（`S0+S1` 耗算 `345,857`）$\to$ 5 台未曝光 $N=5$ 装置（3 台同向变半径＋2 台反向 `left` 修订）三臂对照：
    * `baseline_archive`：**`0/5`**（`MSE = 5.79e-2`，目标总算 `1,419,682`）；
    * `commitment_only`（仅开跨尺度承诺、关闭前提审计）：**`0/5`**（`MSE = 5.79e-2`，目标总算 `1,420,972`）——**直接证伪“光靠公式跨元数重写就能迁移”，证明没有前提审计仍会淹没在 $14,400$ 角色排列中**；
    * **`v10_commitment_plus_premise_audit`**：**`5/5`（100%）**（`MSE = 7.06e-19`，目标总算 **`1,026,583`［省 `27.69%`］**，含 `S0+S1` 全周期总算 `1,372,440` 仍低于基线纯目标耗算）。
- **验证状态**：**已验证**（`test_v10_world_structure_and_premises.py` 4/4 通过＋独立三臂回执）。

---

### 3.3 Phase 2A、2B、2C：条件边与非守恒修订、在线 `gpt-5.6-sol` 开放语法门控、主动阈值二分与一阶 POMDP（`§§120–121`）
- **做了什么**：
  - **Phase 2A**：扩展 `P_exchange_form`（`source_greater` / `source_less`）与 `P_conservation`（分离局部比例耗散 `sink` $\kappa q_i$ 与全局非局域耦合 `nonlocal` $\frac{\kappa}{N}\sum_j q_j$）及 $N=3\to 4\to 5\to 6$ 四代连续演化（`scripts/run_v10_phase2a_scientific_revisions.py`）；
  - **Phase 2B**：构建 `open_grammar_author.py`（`POST /v1/responses` SSE 流式解析、AST 沙箱校验、零密钥落盘），在超出全部内置符号表的 `donor_positive_gap`（$q_s > 0 \land q_s - q_d > 0.25$）与 `focal_quadratic_total_event`（含 $0.35 q_{\text{focus}}^2$）上运行严格单因子对照（平铺 LLM vs. v10 前提门控 LLM），随后在 Stage B 未曝光 $N=5$ 装置上**强制 `0` 次 LLM 调用**验证跨尺度巩固（`scripts/run_v10_phase2b_open_grammar_llm.py`）；
  - **Phase 2C**：将开放语法比较常数 `c_0`（`predicate_threshold`）与半径 `r_0` 统一接入 `RelationIntervalResearch` 坐标交替主动二分，并在 `PremiseDependencyGraph` 新增 `P_markov_state` 一阶隐状态（`transition_memory` 与 `focal_memory`）发现、水库消去与极点重标定（`scripts/run_v10_phase2c_active_threshold_and_pomdp.py`）。
- **开发途中暴露并修复的底层 Bug**：
  - 在 Phase 2C-1 中，当 `event_delta` 连续两步同时因错误阈值漏检时，差分相消偶发产生零误差假象；我们在 `relation_uncertainty.py` 增加了非破坏性兄弟区间救回（sibling rescue）并在 `program_condition_sets.py:solve_compound_region` 实现了焦点布尔选择器展开，使复合区域求解器能精确命中激发边界。
- **定量结果**：
  - **Phase 2A（`v10_phase2a_expanded_revisions_20261116.json`，`0` LLM）**：$N=4\to 5$ 八题三臂：`baseline` **`0/8`**，仅方向修订 **`0/8`**，**Phase 2A `8/8`（100%，`MSE = 3.09e-19`，省算 `37.82%`）**；$N=3\to 4\to 5\to 6$ 四代演化 **`4/4`**（$N=6$ 一步将 $6! = 720$ 压至 `1` 个存活双射，`MSE = 2.10e-20`）。
  - **Phase 2B（`v10_phase2b_open_grammar_llm_20261116.json`）**：
    * Stage A（$N=4$ 4 题，每题共享完全相同的 `203` 行已购观测）：纯符号 v10 **`0/4`**（`0` 调用）；平铺 `gpt-5.6-sol` **`0/4`**（`12` 次调用，`177,608` tokens，`MSE = 0.948`）；**v10 前提门控 `gpt-5.6-sol` `4/4`（100%，仅 `5` 次调用［`-58.3%`］，`34,527` tokens［`-80.56%`］，`MSE = 1.34e-21`）**；
    * Stage B（$N=5$ 跨尺度后继，**`0` LLM 调用 / `0` token**）：未整合父库 **`0/4`**，**巩固承诺臂 `4/4`（100%，`MSE = 7.88e-20`，省算 `45.35%`）**。
  - **Phase 2C（`v10_phase2c_active_threshold_and_pomdp_20261116.json`，`0` LLM）**：
    * Cohort 1（16 初测行，4 台偏移阈值）：冻结源 **`0/4`**，被动中点 **`1/4`**（均值区间宽 `0.1006`），**主动二分 `4/4`（100%，`MSE = 4.38e-20`，区间宽 `4.94e-6`［收窄 `20,356x`］）**；匿名 $N=5$ 双参数偏移题 `810599` 端到端 `completed` 且达 `selected_channels = []`（`4.94e-20`）；
    * Cohort 2（6 台匿名一阶 POMDP）：平铺 `continuous_state` **`0/6`**，马尔可夫 v10 **`0/6`**，**v10 POMDP 前提生长 `6/6`（100%，`MSE = 3.63e-12`，全 6 题省算 `56.72%`，$N=5$ 迁移省算 `71.86%` 且 `2/2` 达 `[]`）**。
- **验证状态**：**已验证**（`test_v10_phase2_revisions_and_open_author.py` 4/4 + `test_v10_phase2c_active_threshold_and_pomdp.py` 4/4 通过＋3 份回执）。

---

### 3.4 Phase 3A & 3B：二阶阻尼谐振 `AR(2)`、跨对象空间耦合 POMDP 与开放语法非线性隐状态合成（`§122`）
- **做了什么**：
  - **Phase 3A**：在 `language.py` 扩展 `accumulate[ar2 | ar2_lag1]`，在 `premise_dependency_audit.py` 新增 `P_latent_order` 与 `P_latent_coupling`；利用全局守恒相消 $\sum_i \Phi_i^{\text{spatial}} \equiv 0$ 对净变化总和 $S_t$ 建立四列齐次线性递推一步消去二阶谐振隐状态；对焦点型隐状态建立无模型假设的 $S_{N-1}$ 对称破缺焦点对比与 Frisch-Waugh-Lovell 正交投影二维极坐标求解器，并在 `world_structure_commitments.py` 实现 `value_summand` 邻域提升（`scripts/run_v10_phase3a_higher_order_and_coupled_pomdp.py`）；
  - **Phase 3B**：新增 `P_latent_transition_drive` 与 `screen_custom_latent_transition_drive`，将 `open_grammar_author.py` 接入超全符号库的两类非线性隐状态 POMDP——双线性做功转移记忆 $z_{i,t} = \alpha z_{i,t-1} + q_{i,t} u_{i,t}$ 与二次能量焦点记忆 $z_{i,t} = \alpha z_{i,t-1} + q_{i,t}^2$（`scripts/run_v10_phase3b_open_grammar_pomdp_llm.py`）。
- **定量结果**：
  - **Phase 3A（`v10_phase3a_higher_order_and_coupled_pomdp_20261116.json`，`0` LLM）**：8 台匿名 POMDP 装置（覆盖 `oscillatory_transition_memory`、`oscillatory_focal_memory`、`coupled_transition_memory`、`coupled_focal_memory` 各 $N=4 \to N=5$）：平铺基线 **`0/8`**，一阶 POMDP（Phase 2C）**`0/8`**，**Phase 3A `8/8`（100%，`MSE = 1.62e-13`，省算 `69.60%`，$N=5$ 迁移省算 `66.62%` 且 `4/4` 全达 `selected_channels = []`）**。
  - **Phase 3B（`v10_phase3b_open_grammar_pomdp_llm_20261116.json`）**：
    * Stage A（$N=4$ 4 题，上限 `max_llm_calls = 2`）：完整符号 POMDP **`0/4`**（`0` 调用）；同授 `accumulate` 的无门控平铺 `gpt-5.6-sol` **`0/4`**（`8` 次调用，`121,362` tokens，`MSE = 0.3401`）；**v10 POMDP 前提门控 `gpt-5.6-sol` `3/4`（75%，`5` 次调用，`29,218` tokens［省 `75.92%`］：双线性做功 `2/2` 单次命中 `2.96e-21`，二次能量焦点 `960402` 单次命中 `4.13e-17`，`960401` 压至 `2.20e-2`）**；
    * Stage B（4 台未曝光偏移极点与半径的 $N=5$ 装置，**`0` LLM 调用 / `0` token**）：未整合父库 **`0/4`**，**巩固非线性 POMDP 承诺 `4/4`（100%，`MSE = 1.10e-17`，省算 `49.77%`，`4/4` 全达 `selected_channels = []`）**。
- **验证状态**：**已验证**（`test_v10_phase3_higher_order_and_open_pomdp.py` 4/4 通过＋2 份回执）。

---

### 3.5 Phase 4A、4B、4C：三参数耦合曲面主动实验设计、无位置连续场一维环 PDE 与超临界极限环 POMDP（`§123`）
- **做了什么 & 为什么这么做**：
  1. **Phase 4A（`curved_hypersurface_probing.py`）**：
     - **数学动机（定理 2：一维二分的几何零空间）**：对四维边空间中的三参数耦合不等式 `distance_attenuated_gap`（$0 < \Delta x < r_0 \land q_s > 0 \land (q_s - q_d) - c_1(\Delta x)^2 > c_0$）与 `bilinear_product_barrier`（$0 < \Delta x < r_0 \land q_s > q_d \land q_s^2 - c_1 q_s q_d > c_0$），固定名义背景的一维二分恒残留沿 $(-\psi_1(e_{\text{nom}}), 1)^\top$ 的 $O(1)$ 零空间；且落入 $(r_{\text{lo}}, r_{\text{hi}})$ 内的静默边若误用于裁剪 $(c_0, c_1)$ 会直接切掉真值！
     - **核心实现与底层 Bug 修复**：实现有向拓扑排序多边（$\ge 3$ 并发边）DAG 精确回溯解码、单调认证半径二分、**严格仅使用 $\Delta x \le r_{\text{lo}}$ 认证半径内静默边**的 Sutherland-Hodgman 二维凸多胞体裁剪＋Chebyshev 内切中心、先验半径下界扩张（推高 $r_{\text{lo}} \uparrow r^*$ 解锁杠杆锥）＋多胞体直径极值顶点最大杠杆中点切片合成；同时修复了 `program_condition_sets.py:solve_compound_region` 中单侧不等式 $q_s > 0$ 旧等式拉回目标 $q_s \approx 10^{-5}$ 与强激励带 $[0.55, 0.95]$ 反向拉扯卡在 `0.375` 的 Bug，升级为**单侧铰链残差 $\max(0, \ell + m - v)$**。
  2. **Phase 4B-1（`continuum_field_lab.py`）**：构建完全移除位置传感器／控制器（`x = empty`）的匿名一维周期对流-扩散-反应 PDE 装置 `ContinuumFieldLab`，自动证伪 `P_role_partition`，以 $O(N^3)$ 算子回归＋迎风不对称性 $D + c_{\text{adv}} > D$ 重构有向哈密顿环（避免 $N=7$ 时 $5,040 \times 5,040 = 25,401,600$ 枚举），沉淀 $\mathbb{Z}_N$ 循环不变 `Program("ring_sum")` 承诺，并在 `unified_research.py` 动态扩容多传感器仪器校准子预算。
  3. **Phase 4B-2（`premise_dependency_audit.py`）**：针对原点线性不稳定（$\rho > 1.0 \implies |a_2| = \rho^2 > 1$）靠 $-\beta z_{t-1}^3$ 自激收敛至极限环的 Van der Pol 振子（因 $\sum_i z_i^3 \neq (\sum_i z_i)^3$ 无法宏观齐次消去），证明并实现**因果起振定理**（单控制扫描首个非零跳变步恒有 $q_{i,\tau^*+1} = u_{i,\tau^*}$，$O(N)$ 零搜索配对 `u -> q`）＋重置轨迹 $t=0,1$ 纯空间解耦＋四列三次最小二乘消去。
- **定量结果（`model_calls = 0`）**：
  - **Phase 4A（`v10_phase4a_curved_hypersurface_probing_20261116.json`）**：Stage 4A-1（固定 16 步微预算 8 题对抗集）：冻结源 **`0/8`**，被动多胞体中心 **`0/8`**（直径 `0.2400`，$L_2$ 误差 `0.0421`），一维坐标二分 **`1/8`**（直径 `0.2236`），**耦合直径探针 `7/8`（87.5%，直径 `0.003044`［压缩 `78.8x`］，$L_2$ 误差 `7.30e-4`［降 `57.7x`］）**；Stage 4A-2（开启亚微米半径收口，4 台匿名 $N=4\to 5$ 三参数同时偏移装置）：平铺基线 **`0/4`**，**v10 耦合世界承诺 `4/4`（100%，全达 `selected_channels = []`，`MSE = 1.13e-19`，省算 `59.41%`）**；
  - **Phase 4B（`v10_phase4b_continuum_pde_and_vdp_20261116.json`）**：Cohort 4B-1（$N=4\to 6, 5\to 7$ 连续场 PDE 8 题）：平铺与固定角色 v10 均 **`0/8`**，**v10 连续场 `8/8`（100%，`MSE = 4.31e-20`，省算 `71.16%`，4 项跨分辨率迁移全达 `[]`）**；Cohort 4B-2（超临界 VdP 4 题）：平铺与线性 `AR(2)` 均 **`0/4`**，**v10 超临界 VdP 达 `3/4`（其中 $N=4\to 5$ 迁移 `2/2` 全达 `[]`，总均省算 `64.69%`；唯一冷启动未过题 `990401` 在 Phase 5B-2 查明根因并彻底解决）**。
- **验证状态**：**已验证**（`test_v10_phase4_curved_hypersurface_and_continuum_vdp.py` 4/4 通过＋2 份回执）。

---

### 3.6 Phase 5A、5B、5C：二维环面／边值／不规则图 PDE、强过程噪声 SDE 工具变量审计＋近临界主动共振、零公理代数证明内核与 AWM 全对齐（`§124`）

Phase 5 将 Phase 4 报告末尾列出的全部三项开放科学边界逐一闭合：

#### （1）Phase 5A：二维周期环面 $\mathbb{Z}_{N_x}\times\mathbb{Z}_{N_y}$、非周期 `Neumann <-> Dirichlet` 边值修订与一般未知图拉普拉斯 PDE（`manifold_pde_lab.py`）
- **做了什么**：实现 `ManifoldPDELab`、`Program("manifold_sum")`、二维笛卡尔环积零分支坐标重构、有界链 `P_boundary_condition` 辨识与 `Neumann <-> Dirichlet` 保守修订、一般连通不规则加权图拉普拉斯重构及 `ManifoldPDECommitment`。
- **开发与实测中捕获并彻底解决的 3 个深层工程问题**：
  1. **`4x4` 二维环面（$N=16$）角色探针组合爆炸**：旧 `propose_object_probe` 在 `positions == {}` 时仍试图逐对购买 $16 \times 16 = 256$ 个 `(u, q)` 角点（直接撞穿 `row_limit = 192`）。在 `continuum_field_lab.py:extract_continuum_u_links` 中识别 `len(role_candidates) == N` 且全部 `additive_supported` 的 1-to-1 映射并在并发 PDE 转移上核验，零多余角点直接进入流形算子重构；
  2. **48 细传感器（$N=16$）纯控制通道半整数切点漂移**：随机 `focus` 采样在半整数取整边界 $k + 0.5$ 附近留空，导致 `instrument_models._fit_membership` 切偏选择器区间而卡在 `['fine']`。在 `propose_manifold_probe` 中对每个桶 $k$ 交替采样 $k - 0.44$ 与 $k + 0.44$，**同时满足两步保持（供 `event_delta` 对齐）并将每个切点 $k + 0.5$ 严格夹逼在 $[k + 0.44, k + 0.56]$ 内**；
  3. **不规则图（度数序列 `[4,4,4,3,2,2,2,2,1]`）枢纽节点幅度偏置 vs. 单保持行余弦退化**：在 $N=9$ 不规则图迁移题 `520901`／`520902` 上，初版未归一化 `sum(|yr * feat|)` 将高残差选择器全部吸向度为 4 的枢纽顶点（产生 4-轮换局部极小），而简单改用向量余弦相似度又会在 $N=16$ 单保持行（`df = 1`）上退化为 `|a*b|/(|a|*|b|) == 1.0`！最终在 `continuum_field_lab.py:_resolve_continuum_focus_mapping` 中设计了**逐读出族精确保持步平方预测残差矩阵 $(y_{\text{res}} - \widehat{f}_k)^2$ ＋ $O(N^3)$ 匈牙利算法（`scipy.optimize.linear_sum_assignment`）**：真格点 $k^*$ 在 `df = 1` 下仍严格取 `0.0` 零代价，而错误的高度数枢纽顶点因幅度大反而承受巨额平方惩罚，一举使 `4x4` 环面与 $N=9$ 不规则图全部一次命中全局最优！
- **定量结果（`v10_phase5a_manifold_and_graph_pde_20261116.json`，`0` LLM）**：
  - 覆盖 `3x3=9 -> 4x4=16` 二维环面（$16! \approx 2.09 \times 10^{13}$）、`N=5 -> N=8` `Neumann <-> Dirichlet` 双向边值反转、`N=6 -> N=9` 不规则图（`branched_wheel` 与 `ladder_pendant`）共 12 题三臂对照：
  - `flat_baseline_no_continuum`：**`0/12`**（均值 MSE `0.1787`，均值计算 `711,585.8`）；
  - `ring_only_v10_phase4b`（仅一维环）：**`0/12`**（均值 MSE `0.1698`，均值计算 `705,805.2`）；
  - **`v10_phase5a_manifold_pde_and_commitment`**：**`12/12`（100.0%），12 题全数达成 `selected_channels = []`，均值 MSE `1.08e-19`，均值计算 `291,148.1`（省算 `59.08%`），均值观测 `220.1` 行（省 `21.66%`）**。

#### （2）Phase 5B：强过程噪声 SDE 两阶段最小二乘工具变量（`2SLS-IV`）前提审计与近临界极限环主动共振（`stochastic_premise_audit.py`）
- **做了什么 & 修复的 4 个关键底层问题**：
  1. **消解 Hurwicz-Nickell 内生性偏置**：在 `StructureLab` 注入状态转移高斯过程噪声 $\varepsilon_{i,t} \sim \mathcal{N}(0, \sigma_p^2)$ 后，$S_t = \alpha S_{t-1} + \eta U_{t-1} + (E_t - \alpha E_{t-1})$ 中 $\operatorname{Cov}(S_{t-1}, E_t - \alpha E_{t-1}) = -\alpha N\sigma_p^2 < 0$ 使朴素 OLS 严重低估极点（如 `530402` 真值 $\alpha^* = 0.80$ 时 OLS 仅得 `0.4133`）。`fit_2sls_iv_latent_transition` 以严格外生的滞后控制 $(U_{t-2}, U_{t-3}, U_{t-4})$ 作工具变量执行 2SLS 并衔接无穷滞后外生脉冲投影优化，输出 Hausman 内生性统计量与 99% Wald 置信区间；
  2. **防止确定性极限环误触发随机分支**：将 `estimate_process_noise_stats` 严格限制在 `position == 0, history == ()` 的初始重置步（此处一切确定性实验室恒有 `latent_state == 0`），彻底杜绝把 VdP 隐状态尾迹误判为过程噪声；
  3. **强过程噪声下的机器精度零误差焦点配对**：在 `530402` 调试中发现，当 `focus_t != focus_{t-1}` 时，旧单列逐对象差分 `fl_cur[qi] - fl_prev[qi]` 减错了上一时刻对象导致 3-轮换错配；改为对完整候选排列 `perm` 计算 `fl_cur[perm[fi_t]] - fl_prev[perm[fi_{t-1}]]` 后，因公开 `after[q]` **已经包含了当期真实实现的过程噪声 $\varepsilon_{i,t}$**，真排列 `perm*` 在任意强度的 SDE 噪声下均满足 `1e-30` 机器精度恒等匹配！
  4. **查明并彻底攻克 Phase 4B 遗留题 `990401`（$\rho = 1.02$）**：逐行追踪发现 `990401` 在 39 行时已解出 `4.65e-24` VdP 递推，但追加 12 行角色确认后（51 行）`premise_dependency_audit.py:2862` 的旧 10 单元子采样恰好跳过了包含 $r_0^* = 0.96$ 的距离区间 `(0.9002, 0.9828)`！将距离等价类保留上限提至 `256` 并增设相位相干主动共振探针 `plan_near_critical_resonance_probe`；同时修复 `unified_uncertainty.py:157` 中 16 维多对象特征下 `claims_cap = 18000` 过小导致 `insufficient_budget` 早退的问题。
- **定量结果（`v10_phase5b_stochastic_sde_and_resonance_20261116.json`，`0` LLM）**：
  - **Cohort 5B-1（4 台强过程噪声 SDE 装置，$\sigma_p \in [0.025, 0.035]$，含 $N=4\to 5$ 与二阶谐振 SDE）**：
    * `deterministic_v10_no_sde_audit`：无噪条件漂移达标（$\le 2\times 10^{-3}$）**`0/4`**（均值漂移 MSE `8.03e-2`，`completed = 0/4`）；
    * `stochastic_v10_naive_ols`：条件漂移达标 **`1/4`**（均值漂移 MSE `2.33e-2`，极点绝对误差均值 `0.1944`，`completed = 1/4`；在 `530402` 上因 OLS 偏置留有可复现偏差被原生 `assess_noise_replication` 实测拒绝！）；
    * **`v10_phase5b_stochastic_2sls_iv`**：**条件漂移达标 `4/4`（100%，均值无噪条件漂移 MSE `3.32e-4` —— 较朴素 OLS 降低 `70.1x`、较确定性 v10 降低 `242.1x`；极点绝对误差均值 `0.0181` 降低 `10.7x`）**，且在未放宽任何确认规则下于 `530501`（$N=5$）与 `530402`（$\alpha^*=0.80$）直接通过独立重复轨迹噪声挑战达成原生 `completed`（均值计算 `424,956.2` 省 `12.01%`）。
  - **Cohort 5B-2（4 台近临界超临界 VdP 极限环 $\rho \in \{1.008, 1.015, 1.02, 1.04\}$，含原 `990401`）**：
    * `flat_baseline` **`0/4`**（`MSE = 11.19`），`linear_ar2_pomdp` **`0/4`**（`MSE = 10.68`）；
    * **`v10_phase5b_active_resonance_vdp`**：**`4/4`（100.0%），4 题全达 `selected_channels = []`，均值 MSE `2.03e-17`，均值计算 `69,074.8`（省算 `83.57%`）**。

#### （3）Phase 5C：零公理有理代数证明内核、跨承诺涌现守恒律、Farkas 不可达证书与 AWM §4.3 差异触发前提修复（`algebraic_proof_certificates.py`）
- **做了什么 & 严苛内核拦截实例**：
  - 构建纯有理数多项式零公理证明内核 `SymbolicProofKernel`（基于 `fractions.Fraction` 与 `sympy.Poly(..., domain='QQ')` 展开正规形判零及有理 Farkas 乘子非负检验，**零浮点容差、零未证 `sorry`**，并内置标准 Lean 4 代码生成器 `emit_lean4_certificate`）；
  - **内核真实拦截实例**：在 `ctrl_5404`（左向传输 $q'_3 = (1-\gamma)q_3 + u_3$）初测时，若默认状态域取双侧 `[-10, 10]`，当 $q_3 < 0$ 时 $-\gamma q_3 > 0$ 会提供虚假正增量，`SymbolicProofKernel` 拒绝签发无条件仿射上界证明！显式引入物理非负密度域 `state_bounds = [0.0, 10.0]` 并将 `P_nonnegative_state_domain` 记入证书依赖集 `used_premise_ids` 后，8 道控制题实现 100% 精确对偶闭合。
- **定量结果（`v10_phase5c_algebraic_proofs_and_impossibility_20261116.json`，`0` LLM）**：
  - **Cohort 5C-1（跨承诺符号通量消元涌现守恒律）**：覆盖对象-水库耦合 POMDP（$Q + \frac{\eta}{1-\alpha}Z$）、双物种化学计量可逆反应运输（$q A_{\text{tot}} + p B_{\text{tot}}$）、Neumann 守恒、Dirichlet Gauss-Green 墙平衡及二维环面守恒共 **7 条定理**，跨元数 $N \in \{3..9\}$ **核验通过 `7/7`（100%）**；对系数施加有理微扰的 7 份篡改证书 **拦截拒绝 `7/7`（100%）**；
  - **Cohort 5C-2（AWM Stage 03 有界预算可达证人 vs. Farkas 不可达证书，8 题）**：256 条随机／贪心 Rollout 搜索形式化认证 **`0/8`**（启发式仅命中 `1/4` 可达，且永远无法证明不可达）；**v10 `SymbolicProofKernel` 精确认证 `8/8`（100%：`4/4` 可达题给出有理控制序列并在 `BudgetedControlLab` 实测通关；`4/4` 不可达题给出闭式有理 Farkas 势函数证书证明预算缺口 $\Delta_{\text{gap}} \in \mathbb{Q}_{>0}$）**；
  - **Cohort 5C-3（AWM §4.3 证明-现实差异触发 `SufficientCover` 靶向前提修复，4 题）**：面对隐藏耗散汇、耦合漂移、初始库存亏损与执行器限幅降额 4 类实测冲突，仅沿证书 `used_premise_ids` 合成 4 步靶向探针即 **`4/4`（100%）修复并实测通过**，重算的真实最小预算 $B^*$ 最大绝对误差 **`0.0`**，较无依赖图盲探（均值 `39.7` 步）**减少 `89.92%` 实验步**。
- **验证状态**：**已验证**（`test_v10_phase5_manifold_sde_and_proofs.py` 5/5 通过＋3 份回执）。

---

## 4. 全量验证核查表、非 100% 细项透明台账与 AWM 对齐矩阵

### 4.1 13 份前瞻证据回执一键索引表（`docs/evidence/v10_world_structure_20261116/`）

| 序号 | 回执文件 | 阶段 | LLM 调用 / Token | 核心对照结果摘要 | 验证状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | [paid_position_round1_census_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/paid_position_round1_census_20261116.json) | `§117` 普查 | `0` / `0` | 135 工作区／229 制品只读核查：非源 `len(owners)==2` 现场为 `0`；5 份封账文件零改动 | 已验证 |
| 2 | [v10_cow_602_603_new_ledger_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_cow_602_603_new_ledger_20261116.json) | COW `602->603` | `0` / `0` | 基线 `0/2`；`602->603` 顺序后继使 `603` 省算 **`66.47%`**（`199,912 -> 67,028`）并升级 `['fine'] -> []` | 已验证 |
| 3 | [v10_ablation_and_cross_scale_successor_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_ablation_and_cross_scale_successor_20261116.json) | Phase 1 | `0` / `0` | 门控拦截 `3/3` 粗化；前提定位 `6/6` vs `0/6`；$N=4\to 5$ **`5/5`** vs 基线 `0/5` & 仅承诺 `0/5`（省算 `27.69%`） | 已验证 |
| 4 | [v10_phase2a_expanded_revisions_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase2a_expanded_revisions_20261116.json) | Phase 2A | `0` / `0` | 条件边＋非守恒 $N=4\to 5$ **`8/8`** vs `0/8`（省算 `37.82%`）；$N=3\to 6$ 四代演化 **`4/4`** | 已验证 |
| 5 | [v10_phase2b_open_grammar_llm_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase2b_open_grammar_llm_20261116.json) | Phase 2B | Stage A: `5` / `34,527`<br/>Stage B: **`0` / `0`** | Stage A **`4/4`**（省 `80.56%` token）vs 平铺 LLM `0/4`（`177,608` tok）；Stage B $N=5$ 零调用 **`4/4`** vs `0/4` | 已验证 |
| 6 | [v10_phase2c_active_threshold_and_pomdp_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase2c_active_threshold_and_pomdp_20261116.json) | Phase 2C | `0` / `0` | 主动阈值二分 **`4/4`** vs 被动 `1/4`（收窄 `20,356x`）；一阶 POMDP **`6/6`** vs `0/6`（省算 `56.72%`，$N=5$ 全达 `[]`） | 已验证 |
| 7 | [v10_phase3a_higher_order_and_coupled_pomdp_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase3a_higher_order_and_coupled_pomdp_20261116.json) | Phase 3A | `0` / `0` | 二阶 `AR(2)`＋空间耦合 POMDP **`8/8`** vs 一阶 POMDP `0/8` & 基线 `0/8`（省算 `69.60%`，$N=5$ `4/4` 达 `[]`） | 已验证 |
| 8 | [v10_phase3b_open_grammar_pomdp_llm_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase3b_open_grammar_pomdp_llm_20261116.json) | Phase 3B | Stage A: `5` / `29,218`<br/>Stage B: **`0` / `0`** | Stage A **`3/4`**（省 `75.92%` token）vs 平铺 LLM `0/4`（`121,362` tok）；Stage B $N=5$ 零调用 **`4/4`**（全达 `[]`） | 已验证 |
| 9 | [v10_phase4a_curved_hypersurface_probing_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase4a_curved_hypersurface_probing_20261116.json) | Phase 4A | `0` / `0` | 16 步微预算 **`7/8`** vs 被动 `0/8` & 一维 `1/8`（直径压缩 `78.8x`）；匿名 $N=4\to 5$ **`4/4`**（全达 `[]`，省算 `59.41%`） | 已验证 |
| 10 | [v10_phase4b_continuum_pde_and_vdp_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase4b_continuum_pde_and_vdp_20261116.json) | Phase 4B | `0` / `0` | 一维连续场 PDE **`8/8`** vs `0/8`（省算 `71.16%`，迁移全达 `[]`）；超临界 VdP **`3/4`**（$N=5$ 迁移 **`2/2`** 达 `[]`） | 已验证 |
| 11 | [v10_phase5a_manifold_and_graph_pde_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase5a_manifold_and_graph_pde_20261116.json) | Phase 5A | `0` / `0` | 二维环面 $4\times 4$／`Neumann<->Dirichlet`／不规则图 PDE **`12/12`（100%，全达 `[]`，省算 `59.08%`）** vs 两基线 `0/12` | 已验证 |
| 12 | [v10_phase5b_stochastic_sde_and_resonance_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase5b_stochastic_sde_and_resonance_20261116.json) | Phase 5B | `0` / `0` | SDE `2SLS-IV` 条件漂移 **`4/4`**（`3.32e-4`，较 OLS 降 `70.1x`）；近临界 VdP（含 `990401`）**`4/4`（全达 `[]`，省算 `83.57%`）** | 已验证 |
| 13 | [v10_phase5c_algebraic_proofs_and_impossibility_20261116.json](file://wsl.localhost/Ubuntu-22.04/home/erzhu419/mine_code/Asumption%20Agent/Hegel%20Machine/docs/evidence/v10_world_structure_20261116/v10_phase5c_algebraic_proofs_and_impossibility_20261116.json) | Phase 5C | `0` / `0` | 涌现守恒律 **`7/7`**（篡改拦截 `7/7`）；可达／Farkas 不可达认证 **`8/8`** vs Rollout `0/8`；差异修复 **`4/4`**（省 `89.92%` 探针） | 已验证 |

### 4.2 全部非 100% 细项指标的逐题透明审计（零挑选种子、零事后改门槛）
为便于审阅 AI 严格核查，以上 13 份回执中出现过的**每一处未达 100% 子指标**均完整保留在原回执中并说明如下：
1. **Phase 3B Stage A 单题 `960401`（`3/4`，同族 `960402` 过且 Stage B `4/4`）**：在预注册 `max_llm_calls = 2` 上限内，焦点读出型隐状态没有 `before -> after` 逐对象转移证人，短窗口内 $q^2$ 与滞后交叉项样本共线使 `960401` 停在 `2.20e-2`；未通过调大调用上限或换种子掩盖。
2. **Phase 4A Stage 4A-1 微预算单题 `970405`（`7/8`，完整智能体 Stage 4A-2 `4/4`）**：固定 16 步微预算关闭了细半径二分收口，曲面多胞体直径已缩至 `0.001105`（曲面参数误差仅 `3.48e-4`），仅留 `0.00337` 半径余量；在开启 `allow_fine_radius_polish=True` 的完整智能体 `Stage 4A-2` 中 `4/4` 全部通过。
3. **Phase 4B Cohort 4B-2 单题 `990401`（原回执保留 `3/4`，已在 Phase 5B-2 查明根因并达成 `4/4`）**：未篡改已发布的 Phase 4B 回执，而是在 Phase 5B-2 中定位到 51 行后距离子采样上限 `10` 的精确缺陷，修复并在相同装置参数下验证 `990401 -> 990501` 双双以 `selected_channels = []` 通过。
4. **Phase 5B Cohort 5B-1 强噪声在线重复确认配额（Oracle 条件漂移 `4/4`［100%］达标，原生在线 `completed` 为 `2/4`）**：`2SLS-IV` 在全部 4 题上均精确恢复真实条件漂移（无噪漂移 MSE 为 `2.70e-4`、`5.58e-4`、`3.54e-4`、`1.45e-4`），并在 `530501` 与 `530402` 上直接通过 `UnifiedResearchAgent` 原生重复轨迹挑战；`530401` 与二阶振荡 SDE `530403` 因多轮残差挑战耗尽 `520` 行预算上限而诚实保持 `unresolved`（绝不为刷 `completed` 而篡改 `assess_noise_replication` 统计门限）。

### 4.3 与 AWM 论文（arXiv:2610.40503v2）全要素对齐核查矩阵

| AWM 论文核心章节与机制 | Hegel Machine v10 对应模块与实现状态 | 对齐度与超越点 |
| :--- | :--- | :--- |
| **Stage 01 — Exploration**（主动探索环境边界与状态转移） | `object_interventions.py`、`relation_uncertainty.py`、`curved_hypersurface_probing.py`、`manifold_pde_lab.py`、`stochastic_premise_audit.py` | **100% 闭合且超越**：从离散具名接口扩展至匿名打乱传感器、非线性执行器、高维三参数耦合曲面、近临界极限环同相共振与无位置流形切点夹逼 |
| **Stage 02 — World-Model Consolidation**（将经验巩固为可执行代数世界模型） | `world_structure_commitments.py`、`continuum_field_lab.py`、`manifold_pde_lab.py` | **100% 闭合且超越**：引入四重无损等价门控（零权重列保持＋$\operatorname{Rej}(T)\supseteq\operatorname{Rej}(S)$＋$\operatorname{Aut}(\mathcal{M})$ 规范群不变性），严格分离等价重组与科学修订 |
| **Appendix C.6 — Premise Support Dependencies**（暴露支撑前提依赖提升冲突定位） | `premise_dependency_audit.py`、`stochastic_premise_audit.py`、`open_grammar_author.py` | **100% 闭合且超越**：16 节点动态生长前提图（涵盖离散对象、一／二阶／耦合／极限环 POMDP、流形边值及随机 SDE `2SLS-IV`），并门控在线 LLM 节省 `78.68%` tokens |
| **Stage 03 — Proofs & Impossibility Certificates**（Eq. 3 & Lemma A.3 可达证人与势函数不可达证明） | `algebraic_proof_certificates.py`（`SymbolicProofKernel`、`synthesize_emergent_invariants`、`prove_task_solvability`） | **100% 闭合**：零浮点容差 `Fraction + QQ[x]` 内核、跨承诺零空间消元涌现守恒律（`7/7`）、有界预算控制可达证人与闭式 Farkas 对偶不可达证书（`8/8` vs Rollout `0/8`）、导出无 `sorry` Lean 4 |
| **§4.3 — Discrepancy-Triggered Repair**（证明与现实冲突时沿 `SufficientCover` 靶向修复） | `algebraic_proof_certificates.py:repair_proof_from_discrepancy` | **100% 闭合**：仅沿证书 `used_premise_ids` 合成靶向边界探针，4 类偏移 `4/4` 修复并实测验证，最小预算误差 `0.0`，减少 `89.92%` 探针 |

---

## 5. 供审阅／接手 AI 直接执行的自动化验证命令与扩展钩子

### 5.1 一键运行全套 28 项回归与单元／端到端测试（`28/28` 100% 通过，零网络调用）
```bash
cd "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine"

PYTHONPATH=src:. python3 -m pytest \
  tests/test_v10_world_structure_and_premises.py \
  tests/test_v10_phase2_revisions_and_open_author.py \
  tests/test_v10_phase2c_active_threshold_and_pomdp.py \
  tests/test_v10_phase3_higher_order_and_open_pomdp.py \
  tests/test_v10_phase4_curved_hypersurface_and_continuum_vdp.py \
  tests/test_v10_phase5_manifold_sde_and_proofs.py \
  tests/test_historical_capabilities.py -v
```

### 5.2 反作弊静态审计命令（验证零私有状态偷看、零密钥落盘、默认向后兼容）
```bash
cd "/home/erzhu419/mine_code/Asumption Agent/Hegel Machine"

# 1. 验证仓库无任何持久化 API Key
! grep -rnE "sk-[A-Za-z0-9]{20,}" src/ scripts/ docs/ tests/ && echo "PASS: Zero persisted API keys"

# 2. 验证认知推理／审计／探针／证明模块零读取实验室私有下划线物理字段
! grep -rnE "\._(radius|coupling|latent_decay|latent_freq|latent_nonlinearity|condition_threshold|condition_curvature|physical_controls|measurement_names|site_ids|graph_edges)" \
  src/hegel_machine/cognition/world_structure_commitments.py \
  src/hegel_machine/cognition/premise_dependency_audit.py \
  src/hegel_machine/cognition/curved_hypersurface_probing.py \
  src/hegel_machine/cognition/stochastic_premise_audit.py \
  src/hegel_machine/cognition/algebraic_proof_certificates.py \
  src/hegel_machine/cognition/open_grammar_author.py \
  src/hegel_machine/cognition/unified_research.py && echo "PASS: Zero private apparatus attribute reads"

# 3. 批量核查全部 13 份 JSON 回执结构完整性与双账本分离字段
python3 -c "
import glob, json
paths = sorted(glob.glob('docs/evidence/v10_world_structure_20261116/*.json'))
assert len(paths) == 13, f'Expected 13 receipts, got {len(paths)}'
for p in paths:
    d = json.load(open(p, encoding='utf-8'))
    print(f'{p.split(\"/\")[-1]:58s} | keys={list(d.keys())[:5]}')
print('PASS: All 13 JSON receipts parsed cleanly')
"
```

### 5.3 后续接手 AI 的标准扩展接口（Extension Hooks）
若后续 AI 继续在 `v10` 基线上推进新研究，建议直接挂载以下现成协议入口而无需改动主控骨架：
1. **接入外部真实物理仿真器／SciML 基准**：只需实现四方法实验室协议 `spec_for(channels)`、`acquire(probe, channels)`、`random_probe(seed, length, name)`、`evaluation_rows(channels, seed, length, trajectories)`（参考 `manifold_pde_lab.py:ManifoldPDELab`），即可直接交由 `UnifiedResearchAgent(cfg, library=lib).run(lab)` 全自动运行；
2. **扩展时变动网格或三维流形算子**：在 `language.py:predict_program` 的 `op == "manifold_sum"` 分支追加新网格差分核，并在 `manifold_pde_lab.py:audit_manifold_pde_premises` 增加对应算子谱／邻接识别器；
3. **从线性／仿射 Farkas 证书扩展至多项式平方和（SOS / Positivstellensatz）非凸证明**：在 `algebraic_proof_certificates.py:SymbolicProofKernel` 中保留现有 `Fraction` + `sympy.Poly(..., domain='QQ')` 底座，新增有理 Cholesky 分解验证器 $\sum_m c_m g_m(\mathbf{x})^2$（$c_m \in \mathbb{Q}_{>0}$），即可保持零浮点容差与无 `sorry` Lean 4 导出性质。
