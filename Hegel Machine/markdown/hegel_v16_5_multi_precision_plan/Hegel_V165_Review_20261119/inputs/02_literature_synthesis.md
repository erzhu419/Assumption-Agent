# 02 — 55 篇本地归档参考文献深度精读综合：支撑“通用多精度假设抽取”的四大理论支柱

- **本地文献库位置**: `Hegel Machine/markdown/Hegel_V16_Full_Package/references/`（共 **55 份已校验 PDF**，详见 `REFERENCES_INDEX.md`）。
- **精读目标**: 不仅核实 GPT-6 在反馈包中引用的边界限制，更重要的是从一手原文中挖出 GPT-6 因过度保守而未充分展开的**正向数学与算法武器**——特别是如何从任意领域具体案例中“管中窥豹”，在**部分共性（Partial Commonality）**下安全、可验证地抽取**多精度假设（Multi-Precision Hypotheses）**。

---

## 支柱一：部分结构对应、开槽图式归纳与分层等价类（为什么“局部不匹配”绝不等于“毫无共性”？）

| 核心文献（本地 PDF） | 原文关键机制与定理定位 | 对 Hegel“通用多精度假设格”的直接支撑 | 必须遵守的严谨边界 |
|---|---|---|---|
| **ISS** (`ISS_2609.32247v1.pdf`, Khanzadeh et al. 2026) | **§3 Def. 1, Prop. 1 & Prop. 6; §4 Thm. 2 & Prop. 4**：定义干预分离代价 $k_{\mathcal{Q},c}(P,P')=\min_{q\in W_{PP'}} c(q)$，严格证明其满足**超度量不等式** $k(P,P'')\ge\min\{k(P,P'),k(P',P')\}$，故 $r$-局部等价 $\equiv^r_{\mathcal{Q},\varepsilon}$ 随协议半径 $r$ 与容差 $\varepsilon$ 形成**嵌套等价类树**。 | **多精度假设格的第一性数学证明**：精细干预（大 $r$/小 $\varepsilon$）下的反例只切分底层细等价类，**完全不破坏粗协议/宽容差下的上层等价类证书**！同时 §6 Result 3 证明必须在分歧探针之外保留少量类外挑战探针防共同漏真机制。 | 预算耗尽不等于获得等价证书；证书以真机制落在当前精度容许集内为条件。 |
| **Gick & Holyoak 1983** (`GickHolyoak1983_SchemaInduction.pdf`) | **Part II, Exps. 4–6 (pp. 21–33)**：对比两则远域类比可触发**消去归纳（Eliminative Induction）**，把跨域差异分为三类：映射恒等核、**保结构差异（化为开放参数槽 Open Slots）**、**破结构差异（要求上移抽象层或分叉）**。实验表明**中等精度图式（Intermediate Abstraction）**迁移率最高（无提示 45% vs. 单例 21%），过粗空话或过细表面细节均失败。 | 解释了为什么双源对比能“管中窥豹”：不同领域的非本质差异自动变成参数槽（如 $\rho\in[0,1]$ 或瓶颈系数 $\alpha$），而破结构差异恰好标定精度边界！ | 双案例负责**生成高潜候选图式**，不能代替目标域的实测检验。 |
| **Gentner 1983 & SME 1989 & MAC/FAC 1995** (`Gentner1983`, `SME_1989`, `MACFAC_1995`) | **SME §3.2 (pp. 11–24)**：从局部匹配构造一对一且参数连通一致的多个部分全局映射 `Gmaps`，按高阶关系深度计算系统性分数（SES），并导出带结构支持的**候选推论（Candidate Inferences）**。**MAC/FAC §2–3**：廉价特征向量粗筛（MAC）+ 少量候选结构对齐（FAC）。 | 支持**部分子结构映射**（无需全系统同构！）与“两阶段召回—对齐”架构：用廉价协议行为指纹做 MAC/Hippo 粗召回，只对 Top-$K$ 跑带类型端口对齐。 | 结构匹配高分只是“值得优先检验的候选”，不等于目标域真值成立。 |
| **SMDP Homomorphisms & Relativized Options** (`SMDP_Homomorphisms_RavindranBarto2003.pdf`, `Options_1999`, `StochasticBisimulation_2003`) | **Ravindran & Barto §3 Eq.(1)-(2), §6.2–6.3 (p. 1014)**：引入状态依赖动作重编码 $g_s(a)$；更重要的是明确指出全系统同态往往平凡无解，但限制在子目标起始集 $\mathcal{I}$ 上的**相对化局部同态（Relativized Options $\mathcal{O}=\langle h, M_{\mathcal{O}}, \mathcal{I}, \beta\rangle$）**广泛存在且可安全迁移子策略！ | **打破全局同构苛求**：两学科无需全状态同构，只要在某个诊断子程序（Option）的起始集与结果划分上满足局部同态，该诊断程序即可跨领域无损复用！ | 状态必须包含影响后续转移的前史、剩余预算与已检版本，不能遗漏隐状态。 |
| **Seven Sketches & Open Networks** (`FongSpivak2018`, `BaezPollard_1704.02051`) | **Fong & Spivak §1.3–1.4（Galois 连接与生成效应）、§6.2.3 Def. 6.19（Pushout）**：Galois 伴随描述粗细精度格之间的保序投影与回拉；余极限/Pushout 在**给定共享接口态射 $S \leftarrow M \rightarrow T$** 后给出满足普遍性质的粘合对象。 | 明确分工：**归纳与探针负责发现接口 $M$ 与映射**；范畴结构与 Galois 连接负责保证跨精度回退（$\mathcal{L}_k \to \mathcal{L}_{k-1}$）与多组件拼接时的类型一致性！ | 严禁宣称“范畴余极限能不经实验自动猜出物理/领域对应”。 |
| **SciAnalogyAR** (`SciAnalogyAR_2605.11258v1.pdf`) | **§3.2–3.3, §6–7**：抽取共享高阶关系做跨域部分映射以提升科学提案多样性。 | 佐证远域关系虫洞能生成局部搜索难以触及的候选方向，但必须接上 Hegel 的执行与否证闭环。 | 论文仅评多样性/合理性，未做端到端严格验证与成本竞争。 |

---

## 支柱二：面向决策区域的主动提问、非近视探针选择与序贯否证（如何把“假设”变成“高效省算力的可执行问题”？）

| 核心文献（本地 PDF） | 原文关键机制与定理定位 | 对 Hegel V16.5 的直接工程启发 | 必须遵守的严谨边界 |
|---|---|---|---|
| **Decision-Region Active Learning (HEC)** (`Javdani2014_ActiveLearningDecisionMaking.pdf`) | **§2–3 Thm. 1–3**：提出**重叠决策区域判定问题（DRD）**与超边切割算法（HEC），证明自适应次模性。只要所有存活假设落入**同一决策区域 $V(S)\subseteq r$**，无需继续区分微观模型即可最优停止并采取行动！ | **直接粉碎“同一曲线可能有不同底层物理实现所以没用”的保守顾虑**：只要不同底层实现在当前精度层对应同一个后继求解动作（同一决策区域 $r_{\mathcal{L}_k}$），立即停探针直接加速！ | 候选集若整体漏掉真机制，需靠独立确认集兜底。 |
| **Selecting Computations** (`Hay2012_SelectingComputations_UAI.pdf`) | **§2 Thm. 4, 7, 9; §4 Def. 14 & Thm. 16（Blinkered Policy）**：证明一步近视 VOI（Myopic Policy）会**过早停止**（因单次探测未必立刻翻转最优动作，但两步互补探测能大幅改判），提出按行动分支解耦的 **Blinkered 多步价值估计**，显著优于近视法与 UCB1。 | 诊断程序不能每跑半步就按近视收益腰斩；应以完整 `2–3` 步有界诊断子程序（如“复位阶跃 + 等距比值检验”）为单位做有上限估值与准入！ | 必须设硬预算上限 $\le \alpha B_{\text{rem}}$，防止无界元计算。 |
| **Inquisitive Semantics & Question Program Gen & Reiter Diagnosis** (`Ciardelli2013`, `RotheLakeGureckis2017`, `Reiter1987`, `Piriyakulkij2024`) | **Ciardelli §3 & §6.2**：问题语义允许重叠备选，不局限于无噪硬分割；**Reiter §4–5 Thm. 4.4 & 5.7**：由前提冲突集（Conflict Sets）最小命中集定位故障前提，且新观测可能揭示多前提同时失效；**Rothe 2017 / Piriyakulkij 2024**：将问句表示为可执行程序并结合后验更新。 | 当叶假设失败时，沿 `requires_premise_and`（如“内在守恒 AND 线性观测透镜”）构造冲突集，优先区分是**观测透镜错位**还是**精细前提失效**，驱动精准回退！ | 命中集本身不自动给出实验成本最优序，需结合 VOI/HEC 选探针。 |
| **Modern BED & Adaptive Submodularity & POPPER** (`ModernBED_2302.14545v2`, `AdaptiveSubmodularity_1003.3967v5`, `POPPER_2502.09858v1`) | **Rainforth §2 & §5.3**：模型错设下的实验设计风险；**POPPER §2–3**：子零假设蕴含条件与序贯 $e$-值累积 $E_i = \prod e_s \ge 1/\alpha$，设计端与执行端数据隔离。 | 在含噪任务中严格隔离“提案/拟合数据”与“独立确认数据”；确定性反例直接做语义否证，随机噪声场景按条件有效统计检验控制假确认。 | 反复偷看同一批数据挑检验会破坏条件有效性；未拒绝不等于证实全称真。 |

---

## 支柱三：经验技能库、程序压缩与非参数持续学习的成败分野（为什么扁平库在跨任务与部分共性上会崩？）

| 核心文献（本地 PDF） | 原文关键机制与实证发现 | 对 Hegel V16.5 的核心启示 |
|---|---|---|
| **Continual-ARC** (`ContinualARC_2611.14505v3.pdf`, 2026-11) | **§2.3–2.4, §4 Table 1 & Fig. 3**：严格拆分同任务新实例保持（`S_seen`）与未见新任务泛化（`S_held`）。发现：朴素示例缓存（Demo-Cache）击败多数复杂笔记 Harness；**每任务存精确程序（Program Library）虽在 `S_seen` 高达 0.81，但在未见任务 `S_held` 上仅为 0.03–0.05（几乎为零）！** | **一针见血解释了为什么只存叶级精确程序（$\mathcal{L}_4$）无法迁移到未见新任务！** 同时警示：实验必须包含认真优化的缓存/强普通基线，且评测重心必须放在跨任务泛化而非同任务复现！ |
| **Negative Transfer in Memory CL** (`NegativeTransferMemory_2604.27003v1.pdf`, Hu et al. 2026) | **§3.1–3.4 Table 3 & Fig. 2–4**：发现惊人的**任务内 vs. 跨任务反转定律**——细粒度原始轨迹（`Raw`）在同任务内占优（80.5% vs. 65.5%），但跨任务迁移（`A->B`）时产生严重负迁移（`FWT = -9.5pp`，**难题子集 `ΔNL` 暴跌 `-26.1pp`！**）；而去除了任务表面动作绑定、保留高阶策略的抽象表征（`Insight`）将跨任务迁移翻正为 `+6.5pp ~ +9.0pp` 并消除反向遗忘！ | **直接证明了“只测同分布单域切片”会系统性偏袒低抽象基线、掩盖抽象的真正威力！** 更说明当源与目标仅共享部分结构时，生搬硬套细粒度流程在难题上危害最大——这正是 Hegel 多精度回退（$\mathcal{L}_3 \to \mathcal{L}_2/\mathcal{L}_1$）相对于扁平库的决胜点！ |
| **HExA** (`HExA_2606.29315v2.pdf`, Chandra et al. 2026-10) | **§3 Alg. 1, pp. 5–6 (`S1 Default -> S2 Diagnose Cliff -> S3 Bisect + AVOID`), §4.4**：对比成功/失败轨迹提炼策略技能与错误恢复三元组 $(\delta,\rho,\alpha)$，跨任务零样本合成使投石机成功率从 11% 升至 44%。 | **Hegel 最贴近的现代强竞争基线（`O_HExA`）**！HExA 证明了“默认动作+异常诊断+纠偏动作”的巨大潜力，但其 Bank 是纯自然语言扁平条目，无可执行前提检查、无观测透镜校准、无反例驱动的多精度版本回退。正面击败或互补超越 `O_HExA` 将极具说服力！ |
| **Algebraic World Model v1/v2 & Agent Workflow Memory** (`AWM_2610.40503v1/v2`, `AgentWorkflowMemory_ICML2025`) | **Pan et al. §2.3 & App. A.2–A.3**（核对 v2 与 v1 方法完全一致）：用带显式 `scope` 的 case/coproduct 组织规则，严格区分**保真压缩（保持诊断拒绝集 `Rej`）**与**环境实证纠错（撤回旧错误假设并重证依赖）**。**Wang et al. ICML 2025 §2.3**：从成功轨迹抽离具体实体为槽位 `{slot}` 形成子工作流。 | 吸收两种 AWM 之长：既有槽位化工作流复用，又有可执行 `scope` 门控与“保真合并 vs. 实证纠错”双轨治理。 |
| **DreamCoder, Stitch & LILO** (`DreamCoder_2021`, `Stitch_2023`, `LILO_2310.19791v3/v4`) | **Stitch §4 Eq.(8)** 压缩效用扣除抽象定义成本；**LILO §3–4**：程序合成 + Stitch 重写 + AutoDoc 可读命名说明，且允许无库基线事后归纳做公平对比。 | 诊断子程序必须能编译回基础探针原语（可展开验证），同时附带清晰语义契约供选择器调用。 |
| **2025–2026 RSI 与技能演化族** (`PASTBench`, `Recuris`, `AIDE2`, `MementoSkills`, `AgentCL`, `ACE`, `MemSkill`, `EvoScientist`) | **PAST-Bench**：分离任务得分与状态持久化/读取/更新机制回执；**Recuris**：失败触发局部修补与锚点回归门；**AIDE²**：严格隔离开发集、选择集与最终未见测试集；**Memento-Skills**：证明 BM25 词面匹配远逊于行为对齐路由（Recall@1 0.32 vs 0.60）；**ACE**：增量更新避免全量重写丢细节；**EvoScientist**：分开方向记忆与执行记忆。 | 全面融入 V16.5 实验合同：按源簇隔离开发/选择/冻结测试；增量版本化更新；同时输出机制追踪回执与端到端全成本净收益；以行为探针指纹取代纯词面检索。 |

---

## 支柱四：效用问题（Utility Problem）、算法组合门控与跨学科系统范例

| 核心文献（本地 PDF） | 原文关键机制与公式定位 | 对 Hegel V16.5 的落地约束 |
|---|---|---|
| **Minton 1988 (PRODIGY/EBL)** (`Minton1988_EBL_UtilityProblem.pdf`) | **§2.1 & §4.3 (p. 566)**：严格提出 **Utility Problem**——正确知识也可能因累计匹配测试费拖慢系统：$\text{Utility} = (\text{AvgSavings}\times\text{ApplicFreq}) - \text{AvgMatchCost}$。指出单纯靠更快索引不能根治，必须限制昂贵前置条件并在实战中淘汰负效用规则！ | 每个假设节点维护实测 `(ApplicFreq, AvgSavings, MatchProbeCost)` 账本；长期净效用为负的节点自动降级为归档只读（保留科学记录但退出默认在线调度）。 |
| **SATzilla** (`SATzilla_Xu2008_JAIR.pdf`) | **§2 Steps 5–12, §3.3 & §3.5 (pp. 570, 576–577)**：在线求解前先无条件运行极短 **Pre-solver**（5 秒解决 ~45% 简单实例，0 特征开销！），未解实例才在硬超时上限内提特征并由经验硬度模型选求解器，超时自动切 Backup Solver。 | 直接催生 V16.5 的 **Stage-0 短预解门 + 有上限探针提取 + 保底回退**，彻底解决简单题被检索/对齐手续费反噬的顽疾！ |
| **CQL, SPIBB, IQL, HIQL & OPE** (`CQL_2020`, `SPIBB_2019`, `IQL_2021`, `HIQL_2023`, `OPE_2017`) | **CQL §3.1 Thm. 3.1 vs. 3.2**：区分策略期望值下界与逐状态动作点态下界；**SPIBB §2–3**：在样本支持数 $N(s,a) < N_\wedge$ 的低覆盖区强制回退基线策略；**OPE**：非配对观测日志存在选择偏差。 | 开发阶段采用**同快照克隆双路运行**采集无偏配对成本差；小样本期采用 SPIBB 精神——覆盖不足时不乱开虫洞，有覆盖且配对区间显著占优时才放行！ |
| **物理/热力学/控制精细边界文献** (`vanDerSchaft2015`, `BoydVandenberghe2004`, `DeVoe2014`, `Siebert_Ch6`, `MIT_6_763_Lec02/10/17`, `HippoRAG1/2`) | **van der Schaft §11.2**：统一端口能量与非线性平衡反应网耗散结构；**Boyd App. A.5.5**：多维及奇异矩阵 Schur 补极小化定理；**DeVoe §11.9**：恒温恒容惰性气体零耦合边界；**Siebert §6.3**：三极点负反馈 $K\ge 8$ 右半平面失稳；**MIT 6.763**：区分 $R=0$ 完美导体磁链守恒、超导磁通子量子化与涡旋钉扎；**HippoRAG 1/2**：短语—段落双层图上的 PPR 多跳关联召回。 | 为跨学科验证中的物理/化学/控制算例提供滴水不漏的边界定义，并确立 HippoRAG 作为“行为探针签名跨领域召回引擎”的清晰角色。 |
