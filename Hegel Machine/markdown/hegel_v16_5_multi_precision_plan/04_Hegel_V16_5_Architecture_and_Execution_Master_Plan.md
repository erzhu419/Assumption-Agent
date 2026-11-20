# 04 — Hegel V16.5 / V16.6 总体架构与实施主计划：局部哲学树森林 + 中间层横向虫洞 + 两阶段成本与权限门控

- **基准提交**: `787256bd415515905359626f6a5498174ad5db3c` (`codex/hegel-cognitive-mainline`)
- **终审校准配套文档**: 详见 **[07_V16_6_Review_Audit_and_Horizontal_Wormhole_Architecture.md](./07_V16_6_Review_Audit_and_Horizontal_Wormhole_Architecture.md)**（吸收 `Hegel_V165_Review_20261119` 全部数学、代码接线与实验修订）。
- **核心设计信条**: **仔细慎重守底线（四契约解耦、语义哈希接线、四级使用权限、逐跳误差预算、外评只读、全账互斥），大胆乐观拓上限（“管中窥豹”——各领域构建局部条件蕴含树，跨学科通过中间层节点之间的横向虫洞 `HorizontalWormholeEdge` 走曲面捷径，覆盖算法调试、计算机系统、运筹经济、生化与物理等一般领域）**。

---

## 一、北极星问题与三阶递进完成标准

**唯一主问题**：
> 从真实研究经历中自主提炼出的**多精度假设格与可执行诊断程序**，在强普通方法（`O_strong`）及紧凑经验技能库（`O_HExA`）拥有**完全相同的模型、工具、原始经历、主动采样、程序归纳、缓存及可选直接求解/重建权限**的条件下，能否在预先定义的**跨表征与跨领域（共享部分机制层级的）未见任务分布**上，实现同等严格质量下的**全生命周期净成本降低**或**固定预算严格成功率提升**？

三阶独立验收（严禁用前一阶替代后一阶，也不因后一阶尚在调优而抹杀前一阶成果）：
1. **Tier-A（自主多精度机制链闭合）**：由源经历分歧触发候选对应与多精度假设提案 $\to$ 源内反例自检过滤错误瞬时律 $\to$ 冻结入库 $\to$ 在未见目标上真实执行 `QuestionSpec` 探针，观察到**正向跳启动（$\mathcal{L}_3$）**与**遇反例优雅回退（$\mathcal{L}_3 \to \mathcal{L}_2 / \mathcal{L}_1$）**并改变父研究动作。
2. **Tier-B（预登记实例级净节省）**：在预登记未见目标上，同等确认标准下真实取消原搜索工作并获得净计算/观测节省；同时完整公开所属簇全部成功与失败样本。
3. **Tier-C（冻结分布级净效用）**：在按源簇隔离的冻结测试集（含同机制跨表征、$\mathcal{L}_3$ 全共性跨物理、$\mathcal{L}_2/\mathcal{L}_1$ 部分共性近邻、以及表面相似硬负例）上，相对 `O_strong` 与 `O_HExA` 取得统计可辨识的成本—质量帕累托优势。

---

## 二、核心架构升级：直接建在现有源码接口上的四大组件（零平行重写）

源码核查确认，`src/hegel_machine/cognition/` 已有 `question_semantics.py`、`question_agenda.py`、`question_continuation.py`、`growth_methods.py`、`research_methods.py`、`method_revision.py`。V16.5 **不新建平行的 CQM 状态机**，而是在现有六大模块上做精准增量升级：

```mermaid
flowchart TD
    T["新目标任务到达 (绑定 TaskView & 输出合同)"] --> S0{"Stage-0: SATzilla 短预解门<br/>(共享快速原语，极简题 0 检索/对齐税直解)"}
    S0 -- "已达确认阈值" --> EXIT0["直接完成 (计入预解与确认成本)"]
    S0 -- "未解，存在结构残差" --> PILOT["有界基础探针 & ProbeSignature 提取<br/>(仅含已购观测特征，禁塞隐藏机制名)"]
    PILOT --> RETR["跨树中间层虫洞召回 (Permission Tier 1: retrieve_candidate)<br/>源局部树中间层节点 M_A <== HorizontalWormholeEdge ==> 目标候选 M_B"]
    RETR --> GATE{"Stage-1: 同状态剩余路径成本门控 + 新簇有界探索臂<br/>UCB(C_transfer_rem) < LCB(C_direct_rem) - margin 或属探索预算?"}
    GATE -- "否" --> DIRECT["保留已购观测，切换原生直接研究"]
    GATE -- "是 (进入 Tier 2: bounded_trial)" --> ALIGN["角色、ObservationMap 与 ScaleMap 校验<br/>(调用对应家族插件：有序边界 / 容量饱和 / 松弛记忆)"]
    ALIGN --> PROBE["执行中间层节点 M_B 的区分性 QuestionSpec<br/>(检查存活候选是否落入同一行动决策区域 R_a，空集转 UNKNOWN)"]
    PROBE -- "满足子叶更强前提 (Tier 3: local_solver_guidance)" --> LEAF["沿局部树纵向深入叶节点加速求解 + 留出观测独立确认"]
    PROBE -- "叶前提被反例否证 / 近失例" --> DIAG{"五因竞争诊断与双向调整"}
    DIAG -- "抽象丢失关键状态 (如并列瓶颈/双模瞬态)" --> CEGAR["CEGAR 状态/联合探针细化 + ALIGN 虫洞条件分裂"]
    DIAG -- "子叶过强但父节点 M_B 仍独立成立且省成本" --> BACKOFF["沿局部树纵向回退父节点 M_B 求解 + 独立确认"]
    DIAG -- "连父节点亦失配或无净收益" --> CLEAN_EXIT["诚实退出虫洞 (Clean Exit) -> 带已购数据回直接求解"]
```

### 1. 表示层：局部条件蕴含树森林 + 中间层横向虫洞图（通用内核与家族插件解耦）
不再强求所有学科共用一条全局线性 `L0..L4` 阶梯，也不再把物理共轭变量 `(u, z, p, F)` 写死在通用主干上，而是采用**“通用四契约内核 + 中间层横向虫洞 + 家族插件”**架构（详见 `07`）：
- **四个正交契约对象**：`TaskView`（预测/决策目标，如 `some_adjacent_flip` vs `earliest_global_threshold`、平均吞吐 vs 尾延迟）、`ObservationMap`（保留 `LensTransform` 单位/符号/传感器变换）、`ScaleMap`（聚合/窗口/快变量消元算子 $\Pi_s$ + 闭合范围 `steady_state_only | transient_approx` + `proposed_discarded_variables`）、`ClaimContract`（协议、容差与四级权限 `retrieve_candidate | bounded_trial | local_solver_guidance | validated_pruning`）。
- **纵向边（局部树内）与横向边（跨树中间层虫洞，M1 即一等启用！）**：
  - `local_implies / coarse_grains_to`（纵向）：仅在单棵局部树内经验证成立时传播；
  - `HorizontalWormholeEdge(M_A <-> M_B)`（横向）：直接连接不同领域局部树的**中间层节点**，携带 `probe_signature_refs`、`role_and_observation_map`、`align_near_miss_conditions`（近失例包含/排除谓词）与**逐跳累加误差预算 `cumulative_error_budget`**（绝不对 $\varepsilon$-近似边取无损传递闭包！）。
- **前提与证据依赖边（区分 `AND` 与 `OR`）**：
  - `requires_premise_and`：多前提联合依赖（如内在守恒 **AND** 传感器标定；未知目标上原始和漂移作为竞争解释待检，不预设唯一原因）；
  - `supported_by_or`：独立源簇替代支持（注意控制同源相关性）。
- **修复 `question_semantics.py` 身份接线与 `growth_methods.py` / `workflow_execution.py` 分支重放**：
  - 将 `task_view`、`observation_map`、`scale_map`、`protocol_spec`、`claim_contract` 接入 `question_semantics.py` 的规范化语义摘要与 `question_continuation.py` 缓存键（确保修改协议/读出/尺度必升 `meaning_version` 并清空旧答案，通过 `verify_question_identity_integration.py` 控制验证！）；
  - 在 `growth_methods.py` 与 `workflow_execution.py` 实现最小类型化 `outcome -> successor` 分支重放，区分 `execution_error` 与 `scientific_refutation_witness`，保留 `fresh confirm` 与真实父消费状态。

### 2. 问题语义层：协议完备的 `QuestionSpec` 与决策区域停止准则
在 `question_semantics.py` 现有字段基础上，每个诊断问题必须绑定：
- `protocol_spec`：`{controlled_vars, held_boundary_vars, reset_requirement, waveform_type, rise_time_bound, sampling_grid}`；
- `readout_spec`：`{raw_channel, conjugate_transform, sign_convention, noise_tolerance}`；
- `outcome_partition`：显式包含 `SUPPORTED_IN_SCOPE`、`REFUTED_IN_SCOPE`、`BOUNDARY_DEGENERATE`、`SENSOR_MISMATCH`、`UNKNOWN`、`PROTOCOL_ERROR`；
- `decision_region_map`（借鉴 Javdani et al. 2014 `HEC`）：一旦存活候选全部落入同一精度决策区域 $V(S) \subseteq r_{\mathcal{L}_k}$，立即停止额外探针，进入该层求解器，绝不为区分不影响当前行动的微观细节浪费预算。

### 3. 分支治理层：拆分“保真压缩”与“经验纠错”两套合同
- **保真重组/合并（`FidelityCompression`）**：仅当两分支在重叠范围内预测相容且声称不改变知识语义时启用，强制执行 AWM 式拒绝保持 `Rej(T_merged) ⊇ Rej(T_1) ∪ Rej(T_2)` 且消费者回归测试全过。
- **经验纠错/精度修订（`EmpiricalRevision`）**：当新证据发现旧反例系由读出错配（如把总磁链误当共轭电流）或未复位历史引起时，生成新 `meaning_version`，**允许撤回旧错误拒绝**并重验下游依赖对象，完整保留变更原因链。

### 4. 检索与门控层：SATzilla 两阶段成本门控 + MAC/FAC-Hippo 虫洞索引
1. **Stage-0 短预解门（Pre-Pilot Presolver，解决简单题“迁移税反噬”）**：
   - 借鉴 `SATzilla`（Xu et al., 2008 §3.5），在分配固定 $\le 0.05 B$ 的极小预算内先跑一次原生轻量无记忆/一阶基线检查；若残差已低于确认门槛，直接确认退出，**真正实现简单题 0 检索/对齐税**。
2. **MAC/FAC + HippoRAG 多精度二分图召回（候选库扩大后启用）**：
   - 严禁在 Pilot 阶段把未经验证的机制词（如 `lenz`、`equilibrium`）塞入查询三元组；
   - 图节点仅由**已实测的协议观测特征签名**（如 `sig:endpoint_matched_history_gap`、`sig:step_monotone_attenuation`、`sig:equal_spacing_exp_ratio`）与源经历段落、多精度假设节点（$\mathcal{L}_1\sim\mathcal{L}_4$）相连；
   - 第一轮固定 PPR 阻尼与重启权重（不与准入选择器同时训练，杜绝选择后乐观偏差），输出 Top-$K$ 候选 `(HypothesisNode, PrecisionTier, CandidatePortBinding)`。
3. **Stage-1 配对经验成本门控（Auditable Empirical Cost Selector）**：
   - 在开发源簇上，从相同决策前快照采集 `Direct` vs. `Transfer(Tier L_k)` 的完整路径配对代价（含绑定、探针、失败回退及独立确认：`C_transfer_full`）；
   - 仅在满足预设严格完成率与零新增假确认约束下，当经验区间满足：
     $$\widehat{\operatorname{UCB}}\left(C_{\text{transfer\_full}}(s, \mathcal{L}_k)\right) < \widehat{\operatorname{LCB}}\left(C_{\text{direct\_remaining}}(s)\right) - \delta_{\text{margin}}$$
     时准入执行，并设置硬探针止损上限（如附加探针开销 $\le 0.20 B_{\text{rem}}$），超限立即带已购数据无缝回退原生求解。样本量较小时诚实标注为**经开发簇校准的经验保守门控**，不虚称逐状态有限样本定理下界。

---

## 三、慎重而进取的四阶段推进路线（`M0 -> M3`）

我们既吸取 GPT-6“严控同时变化变量、先修 P0 账本”的工程审慎，又坚决拒绝“缩回同分布单域温室、放弃跨表征与多精度跨学科验证”的保守退却，制定如下紧凑四步路线：

### Milestone M0（前置底座）：P0 八项边界闭合 + 三大数学反例回归门
1. **闭合 V15 遗留 P0 项**：版本不可变身份、`exact-protocol` 严绑初态与前史、AND/OR 支持传播分离、干净载体移除（清理派生副本但保留原始历史与独立推导）、可选有界重建、按版本/范围隔离的失败缓存、全流程互斥计费账本。
2. **完成 V15 12 道 A 目标逐题轨迹拆解**：精确量化此前目标端多花预算究竟出在检索、绑定、重复拟合还是确认阶段。
3. **三大反例写入自动化单元测试**：
   - `test_rl_unreset_history_sign_reversal`（验证 $0<t<\ln 2$ 瞬时乘积反例不会误杀 $\mathcal{L}_2/\mathcal{L}_3$ 共轭模型）；
   - `test_rl_smooth_pulse_ratio_exceeds_one`（验证 $\Phi_e = A\theta^2 e^{-\theta}$ 下 $\theta/3 > 1$ 自动拦截无条件点态比值律，同时通过微分算子检验）；
   - `test_nonlinear_sensor_conservation_separation`（验证 $y_i=c_i^2$ 触发传感器标定分支 `SENSOR_MISMATCH` 而非误判质量不守恒）。

### Milestone M1（核心突破切片）：领域无关的自主多精度形成与跨表征/跨任务族迁移（正面迎战 `O_strong` 与 `O_HExA`）
- **核心原则（绝不局限在物理！）**：勒夏特列—楞次只是帮助我们看清“角色/透镜对齐 + 协议条件化 + 多精度假设格 + 遇反例优雅回退”的显微镜（详见 `05_Universal_Hegel_Dialectical_Pattern_Beyond_Physics.md`）。代码内核与 M1 主实验**完全领域无关（Domain-Agnostic）**，一行物理专用公式都不硬编码！
- **M1 覆盖三大通用非物理/跨领域抽象机制簇（每簇均含 $\mathcal{L}_3$ 高精度全共性目标 + $\mathcal{L}_2/\mathcal{L}_1$ 部分共性回退目标 + 透镜错位例 + 硬负例）**：
  1. **状态记忆与内生缓冲簇**（涵盖缓存/队列背压、经济/生态内生适应、阻尼松弛）：检验同终点异前史 $\mathcal{L}_1$ 探针、阶跃缓和界 $\mathcal{L}_2$ 与单模率参数模板 $\mathcal{L}_3 \to \mathcal{L}_2$ 双模回退；
  2. **瓶颈容量饱和与木桶限制簇**（涵盖计算机系统 Amdahl 并行加速、运筹排队吞吐、生化酶催化饱和 $\to$ 微服务/农业 Liebig 最小因子硬瓶颈回退）：检验单因子边际递减探针 $\mathcal{L}_1$、凹上包络界 $\mathcal{L}_2$ 与双倒数线性模板 $\frac{1}{y}=\alpha+\frac{\beta}{u}$（$\mathcal{L}_3$）遇到折线硬瓶颈时优雅回退至 $\mathcal{L}_2$ 瓶颈定位；
  3. **离散跳变/悬崖定位与折半诊断簇**（涵盖软件回归 `git bisect`、连续控制混沌悬崖如 `HExA`、数据流水线变点检测）：检验异号区间夹逼 $\mathcal{L}_1/\mathcal{L}_2$、折半探测程序 $\mathcal{L}_3$ 在全局非单调（多处反转）目标上从 $\mathcal{L}_4$（唯一阈值）优雅回退至 $\mathcal{L}_2/\mathcal{L}_3$ 仍以 $O(\log N)$ 锁定有效边界！
- **自主形成协议**：不预塞问题树或正确分支；让代理在源任务流中经历“朴素无记忆/线性/全量扫描失败 $\to$ 主动分叉探测 $\to$ 多精度假设提炼 $\to$ 源内反例自检 $\to$ 冻结入库”。
- **三臂公平主对照**：
  1. `Hegel V16.5`（通用多精度假设与诊断 DAG + 两阶段成本门控）；
  2. `O_strong`（强普通研究器：同模型、同原语、同原始历史、主动实验、允许按需归纳/重建函数并缓存、允许直接求解）；
  3. `O_HExA`（参照 `HExA 2606.29315v2` 实现的紧凑“自然语言策略 + 错误恢复 Bank”基线，从完全相同的源经历对比提炼并冻结，计入全部蒸馏与注入成本）。
- **两项局部因果干预**：(a) 预登记同历史快照移除对象（测对象边际贡献）；(b) 提案前源经历移除/匹配替换（测双源对比经历对多精度抽象形成的真实因果作用）。

### Milestone M2（旗舰跨学科语义虫洞验证）：穿透不同学科外壳的真实迁移测试
将同一套通用引擎（不改一行代码）投入披着真实异质学科外衣的封存测试集，验证“不走学科平面、通过 `ProbeSignature` 虫洞走高维曲面”的真实加速：
- **虫洞测试组 W1（系统工程 $\leftrightarrow$ 生物化学 $\to$ 运筹/生态瓶颈）**：从并行计算扩容轨迹与酶反应实验轨迹自主提炼瓶颈多精度格，迁移求解未见的分布式微服务限流与多资源配比优化问题；
- **虫洞测试组 W2（软件工程调试 $\leftrightarrow$ 物理参数控制 $\to$ 复杂流水线故障隔离）**：从离散回归排查与连续悬崖控制提炼折半夹逼多精度格，迁移求解含非单调干扰的多阶段流水线异常定位；
- **虫洞测试组 W3（化学平衡 + 电磁感应 $\to$ 黏弹力学 / 经济内生缓冲 + 未知传感器标定解耦）**：作为经典跨自然科学与社会系统代表，检验对偶读出透镜（如 $-I \leftrightarrow \delta\mu_A$）、$\mathcal{L}_3 \to \mathcal{L}_2$ 双模态回退、$R=0$ 守恒边界分叉以及 `observes_via` 非线性观测校准解耦。

### Milestone M3（规模化虫洞检索消融）：同内容 `Flat vs. Multi-Precision Hippo Graph`
- 当跨源簇累积经历达到 $K \ge 32$（跨越记忆、松弛、守恒扩散、振荡、无记忆代数及硬负例多家族）导致候选召回出现实测稀释时，冻结完全相同的多精度对象库、元数据与 Stage-1 门控，**仅替换检索通道（BM25/Dense Flat vs. 协议签名多精度图 PPR）**，计入建图费用，量化关系虫洞在规模化记忆下的独立边际收益。

### Milestone M4（开放组合式科学发现 × 多尺度重整化 $\Pi_s$ × 持续保真压缩生命周期）
- 打破预置算子枚举，打通开放 Python AST 组合演化器、双通道多尺度快慢消元证书（$\Pi_s$ Schur 约化与闭合破裂下钻）以及 $K \ge 60$ 持续流中的在线 `FidelityCompression`、零重拟合作用域分发器与全生命周期回本验证（已完成 `M4.0`–`M4.5` 闭环）。

### Milestone M5（收官终章：泛化跨时刻多尺度时间折叠 × 部分共同基底子树先验与少样本方差治理）
> **定位说明**：在 `M0`–`M4.5` 已彻底夯实防泄漏计量账本与在线持续写回底座（`M5.0` 桥接流）的基础上，**`M5` 不再做碎片化补丁叠加，而是将原方案最具灵气的两大核心思想完整合一，作为整代 Hegel Machine 的收官里程碑（完整设计规范详见 [20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md](./20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md)）**：
1. **泛化跨时刻相位（`过去 / 现在 / 将来`）与多尺度时间折叠原生入树**：
   - 摒弃脆弱的固定采样步长 `t-1, t, t+1`，在每个尺度 $\tau$ 的 `ScaleMap × ObservationMap` 中定义相对该尺度的泛化时态三相位 $(\mathcal{P}_\tau, \mathcal{N}_\tau, \mathcal{F}_\tau)$；
   - 局部树纵向深度自然对应跨时刻信息需求：`现在截面层` $\to$ `过去 vs 现在/将来变迁对比层（如楞次定律/勒夏特列反抗）` $\to$ `过去->现在->将来路径记忆与快慢反转层（同现在异过去走向异将来）`；
   - 跨尺度纵向边 `coarse_grains_to` 实现**时间折叠**：微观尺度完整的 `过去->现在->将来` 拉扯折叠为宏观慢尺度 `现在` 的等效本构属性，闭合破裂时反向展开时间。
2. **渐进式部分共同基底子树先验（`SharedBaseSubtreePrior`）与四重少样本方差控制**：
   - 将单节点横向虫洞升级为**自上而下逐层拼合的共同基底子树对齐**（匹配 1 层慢结构给弱方向先验，叠加匹配第 2 层连通结构解锁强联合先验，第 3 层差异处干净分叉）；
   - 以**“慢尺度做树干快尺度留树叶 + 单源只传定性骨架/多异质源三角测量缩参数区间 + 最高方差分歧点优先探针 + 反例局部劈叶隔离”**四重机制彻底解决极少样本结构先验的高方差问题，并在静态无时间基线与扁平单点记忆必然定性失效的跨领域任务上完成终局验证。

---

## 四、全生命周期账本与停止准则

1. **互斥成本向量**：每一步事件唯一归入
   $$C_{\text{total}} = C_{\text{source}} + C_{\text{formation}} + C_{\text{index/maint}} + \sum_{j=1}^{N} \left(C_{\text{presolve}} + C_{\text{pilot}} + C_{\text{retrieve}} + C_{\text{bind}} + C_{\text{probe}} + C_{\text{solve}} + C_{\text{confirm}} + C_{\text{fallback}}\right)_j.$$
   同时单列目标阶段条件成本与含源阶段摊销曲线（有限前缀 $N=1, 3, 6, 12$）。
2. **主指标与统计单位**：以**独立源学习簇（Source Cluster）**为统计推断单位，报告固定预算严格成功率、满足预设无量纲精度下的总成本、首次合法确认成本、假确认率（False Confirmation Rate）、真机制误剪率（True-Mechanism Pruning Error）以及 $\mathcal{L}_1/\mathcal{L}_2/\mathcal{L}_3/\mathcal{L}_4$ 各精度层级的实际触发与回退命中率。
3. **诚实诊断分流**：
   - 若 $\mathcal{L}_3$ 跳启动与 $\mathcal{L}_2$ 回退均跑通且在跨表征/跨学科簇显著优于 `O_strong` 与 `O_HExA` $\implies$ 核心论点成立，进入 M3 规模化扩展；
   - 若在全共性 $\mathcal{L}_3$ 上与 `O_HExA` 持平，但在部分共性近邻（双模/非线性/边界退化）上显著胜出 $\implies$ 精准定位 Hegel 多精度 DAG 相对扁平技能库的核心优势在于**抗负迁移与优雅精度降级**！
