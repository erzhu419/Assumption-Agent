# 02 — 55 篇本地归档参考文献深度精读综合：支撑“通用多精度假设抽取”的四大理论支柱

- **本地文献库位置**: `Hegel Machine/markdown/Hegel_V16_Full_Package/references/`（共 **55 份已校验 PDF**，详见 `REFERENCES_INDEX.md`）。
- **精读目标**: 不仅核实 GPT-6 在反馈包中引用的边界限制，更重要的是从一手原文中挖出 GPT-6 因过度保守而未充分展开的**正向数学与算法武器**——特别是如何从任意领域具体案例中“管中窥豹”，在**部分共性（Partial Commonality）**下安全、可验证地抽取**多精度假设（Multi-Precision Hypotheses）**。

---

## 支柱一：部分结构对应、开槽图式归纳与分层等价类（为什么“局部不匹配”绝不等于“毫无共性”？）

| 核心文献（本地 PDF） | 原文关键机制与定理定位 | 对 Hegel“通用多精度假设格”的直接支撑 | 必须遵守的严谨边界 |
|---|---|---|---|
| **ISS** (`ISS_2609.32247v1.pdf`, Khanzadeh et al. 2026) & **Girard–Pappas 2007** (`58_Girard_Pappas_2007...pdf`) | **ISS §3 Def. 1, Prop. 1 & Prop. 6; §4 Prop. 4**：精确查询回答等价满足**超度量不等式** $k(P,P'')\ge\min\{k(P,P'),k(P',P'')\}$，形成嵌套等价类树；连续容差证书要求存活候选在相关查询上受控且真机制在候选集内。**Girard & Pappas Prop. 2**：近似双模拟距离在串联组合时误差相加。 | 支撑**局部树内的精确查询蕴含层次**；同时明确警告：**正容差 $\varepsilon$-邻近关系不具备传递性**（$0, 0.75\varepsilon, 1.5\varepsilon$ 首尾超差），因此跨树横向虫洞严禁取传递闭包，必须逐跳累加误差预算或检查簇直径！ | 预算耗尽不等于获得等价证书；近似虫洞不可当作无损等价关系。 |
| **Gick & Holyoak 1983** & **ALIGN 2015** (`GickHolyoak1983`, `56_McLure_Forbus_2015_ALIGN...pdf`) | **Gick & Holyoak Exps. 4–6**：双案例消去归纳区分恒等核、**开放参数槽**与破结构差异。**ALIGN (AAAI 2015, pp. 566–569)**：利用关系对齐检索结构相近但标签相反的**近失例（Near-Misses）**，自动提炼包含/排除边界条件，并允许后续正例修订过强条件、支持同一概念多簇并存。 | 直接支撑 Hegel 的**双源候选生成 + 近失例驱动虫洞条件分裂**：不用等大样本统计显著才提假设，遇到近失例精准长出范围谓词而非推倒整棵树！ | 双案例与 ALIGN 机制负责候选图式与边界修订，未见任务净收益仍须实测。 |
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
| **Continual-ARC** (`ContinualARC_2611.14505v3.pdf`, 2026-11) | **§2.3–2.4 Eq.(4), §4 Table 1 & App. A**：严格拆分同任务保持（`S_seen`）与未见任务获取（`S_held`，指标定义为**相对 reset 条件节省的反馈预算比例**，非准确率）。Table 1 中整题程序库基线 `S_seen=0.81`，而 `S_held=0.03–0.05`（区间跨零）；App. A 指出该特定基线仅存整题程序而未抽低层共享原语。 | 说明**逐题缓存完整叶级解程序不能替代跨题抽象获取**；同时要求公平评测中**必须允许 `O_strong` 普通基线抽取、组合并缓存可复用程序原语**！ |
| **Negative Transfer in Memory CL** (`NegativeTransferMemory_2604.27003v1.pdf`, Hu et al. 2026) | **§2–3, Tables 3–5**：揭示同任务表现不能预测跨任务迁移——ALFWorld `A->B` 中 `Raw` 的 `FWT = -9.5pp`（难题 `ΔNL = -26.1pp`），`Insight` 翻正为 `+6.5pp`（BabyAI `+9.0pp`）；但反向 `B->A` 的 `Insight FWT = -2.5pp`，且 Table 4 `BWT = -1.0 / -3.5pp`（并非普遍反转律或消除遗忘）。 | 证明**必须实测跨任务双向迁移、部分共性与负迁移风险**，不能只拿同任务表现推断跨域收益，也不能假设抽象永远正迁移——必须配成本与范围门控！ |
| **HExA** (`HExA_2606.29315v2.pdf`, Chandra et al. 2026-10) & **CEGAR** (`57_Clarke_2003_CEGAR...pdf`) | **HExA §4.4, App. B.3–B.4, B.7–B.8**：从成败轨迹对比提炼带**适用条件、来源、错误诊断与修正**的技能库，支持跨轮改写组合（论文主结果用静态注入，B.7 另定义可选任务内试验模式），特定跨任务投石机设置由 11% 升至 44%。**CEGAR (Clarke et al. 2003)**：区分真实反例与因抽象丢状态导致的伪反例，按需细化状态表示。 | **`O_HExA-inspired` 必须保留条件、错误恢复、跨轮演化与自然语言尺度/协议表达能力**（绝不人为削弱成傻瓜扁平库）；Hegel 的待测差异在于显式可执行合同、中间层横向虫洞误差治理及“纵向父回退 + CEGAR 补状态”双向修订！ |
| **Algebraic World Model v1/v2 & Agent Workflow Memory** (`AWM_2610.40503v1/v2`, `AgentWorkflowMemory_ICML2025`) | **Pan et al. v1/v2 §2.2–2.4**（所核相关方法段落一致）：用带显式 `scope` 的 case/coproduct 组织规则，区分**保真压缩（在对齐域与查询集上保持 `Acc/Rej/Unknown` 语义，避免新增大量误拒）**与**环境实证纠错（升版本撤回旧错误拒绝并重验依赖）**。**Wang et al. ICML 2025 / arXiv v1 §2.2–2.3**：抽离具体实体为参数槽 `{slot}` 复用子工作流（准入依赖 LM 预测器，非无误环境证书）。 | 吸收两种 AWM 之长：既有槽位化子程序复用，又有可执行 `scope` 门控与“保真合并 vs. 实证纠错”双轨治理。 |
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
