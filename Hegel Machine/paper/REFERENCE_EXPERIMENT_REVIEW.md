# 从 reference 文献反推黑格尔机的论文实验

核读日期：2026-10-05。对象是 Asumption Agent 下所有名为 reference 的目录及其子目录中的 PDF，并对照当前 [英文论文](manuscript.md) 和 [研究计划](RESEARCH_PLAN.md)。这份文件保留文献与实验设计分析。三项后续工作已实施：在线递归与四格迁移完成正式离线评估，同模型对照因服务不可用中断；结果见 [实施报告](../docs/hegel_machine_online_inquiry_20261005.md) 和 [研究计划](RESEARCH_PLAN.md)。下文“还需要”指提出实验设计时的缺口。

## 结论与优先级

**当前补充三项：真实在线递归研究的机制对照、同一模型的普通 Agent 对照，以及冻结后的未见结构迁移。** 正确假设、独立预测、支持研究对决策的作用是主终点；预算作为共同约束，成本作为次要结果。三项共用一个窄实验环境。

现有结果已经支撑一篇范围有限的“可修订研究知识”论文：保留失败模型中的可用计算结构，能够帮助后续修订和旧条件回返。若要把“自主提出子假设、递归研究、让研究方法自身改变”升为实证中心，还需要证明这些动作对解决父问题具有独立作用。新增实验应围绕这个差别组织。

| 优先级 | 要回答的问题 | 本轮建议 |
|---|---|---|
| 第一 | 子问题研究是不是实际计算过程，而不是拟合后的解释？ | 一个窄任务族，在线递归对同预算平铺自适应研究；附一次冻结旧方法的干预。 |
| 第二 | 收益是否超出普通代码试错和文字反思？ | 在同一任务族接一个实际 LLM，比较普通 Code/ReAct、文字经验、黑格尔机。使用 ruoli.dev 的 gpt-6.1-sol。 |
| 第三 | 学到的是任务规律，还是当前生成器中的捷径？ | 冻结知识后，留出结构组合，加入“表面相似但机制不同”的任务。 |

**三项在同一研究接口上衔接实现。** 已完成的保留、回返、坐标变化和新产物删除实验继续作为证据，不再换一批种子重复同一结论。

## 文献范围与实际类型

扫描得到 **88 个 PDF 文件，共 3,449 页**：64 个文档文件，以及论文仓库中的 24 个单页图稿。64 个文档中，HippoRAG 有两个版本；还包含综述、立场论文、数学专著、教程和整本会议论文集。因此不能称为“88 篇自进化论文”。

文献中的实证大致检验五件不同的事：改写 Agent 的代码或运行流程；积累并使用经验；增加题内推理计算；生成与检验假设；学习便于预测和行动的表示。范畴论、信息几何等材料主要提供形式语言。这里没有一组结果可以直接推出机器具有主观自我意识。

下文页码均为**本地 PDF 页序**。逐文档清单在文末，文件名与正文标题不一致的情况已标明。

## 最值得借鉴的实验，以及它们真正证明了什么

### Red Queen Gödel Machine：机制消融、固定外部结果与成本

[Red Queen Gödel Machine](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf>) 同时演化 Agent 和评估器，覆盖代码、论文生成/评审和奥赛证明/评分。它用锚定数据约束评估器，并在消融中拆开评估器替换、对抗样本池与历史清除；比较固定评估器的 HGM-H。主实验与机制消融使用的底模不同，应分别解读（pp.6–13、19–25）。

| 实验 | 实际结果与口径 | 对我们的启发 |
|---|---|---|
| 代码生成 | Polyglot 终测 RQGM 119/166，HGM-H 116/166；论文另报找到候选所需搜索 token 的减少。 | 同一外部题集下同时报告正确率与找到答案的成本，而不是只报内部奖励。 |
| 论文生成 | 四个评审模型对每位 writer 的 100 篇生成稿评分；平均接受率 21.8%→38.8%，generalist 搜索约 44.6M 对 42.5M token。Specialist 40.5%，搜索约 221.8M。 | 评审接受率是该实验的结果变量；不能换称真实论文录用、科学发现或低成本普遍优势。 |
| 数学证明 | Specialist 平均分 4.33，对静态 IMO25 的 4.07；满分通过率却为 48.3%，静态为 55%。HGM-H 与 generalist 均为 3.73，未超过静态基线。 | 多个终点可能给出不同排序，平均进步与严格成功要分别展示。 |
| 机制拆分 | 固定评估器、替换、对抗池、清除的消融；训练、验证、终测分开，报告搜索预算。 | 证明哪一步改变了结果，比再增加一个完整系统成绩重要。 |

表中搜索 token 使用论文的加权口径：输入 token 加五倍输出 token，包含生成与评估。代码搜索每臂给定 12,288 次任务评估额度，但各领域的 token 花费仍须单独比较。其主设置的八条 headline 搜索运行不是大量独立演化重复；不同 writer 自己的评审器接受率也不能直接横比（pp.20–25，Tables 3、5、7）。我们应借鉴机制干预与质量—成本关系；当前实验室有确定的外部预测目标，**没有必要为了模仿这篇文章而改成评估目标共演化**。

### 最接近“假设研究”的两篇

[POPPER](</home/erzhu419/mine_code/Asumption Agent/reference/Automated Hypothesis Validation with Agentic Sequential Falsification.pdf>) 从给定主假设及数据表结构出发，让 Agent 生成子假设检验、执行统计代码、检查与主假设的关联，再顺序汇总证据。它在 DiscoveryBench 与生物学 TargetVal 上比较 CodeGen、ReAct、Self-Refine，并拆掉关联检查、替换证据汇总方法，测检出能力及错误接受（pp.5–8，Tables 3–4）。DiscoveryBench 的一列 Type-I error 是 0.103±0.020，不能复述成所有均值严格低于 0.1。它主要验证给定假设，而非学习跨任务可执行规律。

[Hypothesis-Driven Reasoning](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/22_hypothesis_driven_reasoning_llms_aaai2026.pdf>) 已经做了“提取因子→生成规律→检查→用于新例”的合成视觉规则实验：150 个新多物体测试、三个难度、每档三次；有 Reflexion、oracle 规则及去因子/去迭代消融（pp.5–7，Tables 2–5）。因此，单独宣称“提出并验证假设”不足以区分我们的贡献。

我们的差别应落实在：**支持条件能成为新研究问题，子问题取得的证据改变父问题的实验与解释；得到的程序和研究步骤能够带着适用条件被后续任务调用或修订。** 这句话的每个动作都要在可执行比较中有对应结果。

### 记忆、技能和元推理：共同强调使用效果，但成本口径不同

[Reflexion](</home/erzhu419/mine_code/Asumption Agent/reference/Reflexion Language Agents with Verbal Reinforcement Learning.pdf>) 测重试与文字反思；[ExpeL](</home/erzhu419/mine_code/Asumption Agent/reference/ExpeL LLM Agents Are Experiential Learners.pdf>) 拆分实例检索和经验摘要；[Voyager](</home/erzhu419/mine_code/Asumption Agent/reference/Voyager An Open-Ended Embodied Agent with Large Language Models.pdf>) 把技能库移交另一控制器，并在新世界测未见目标；[SELF-DISCOVER](</home/erzhu419/mine_code/Asumption Agent/reference/SELF-DISCOVER Large Language Models Self-Compose Reasoning Structures.pdf>) 消融选择、适配和实现推理模块。它们说明，仅证明保存了经验，不能代替证明经验被有效使用。

其中一些结论很克制：Reflexion 在 MBPP Python 上出现退步；Voyager 的新任务每项仅三次；SELF-DISCOVER 主要按调用次数比较，而每次调用长度不同；ExpeL 的完整方法通常增加 token。[Meta-Harness](</home/erzhu419/mine_code/Asumption Agent/reference/Meta-Harness End-to-End Optimization of Model Harnesses.pdf>) 在部分领域使用同提议模型、同候选评估次数和冻结后的新题，这更接近我们应采用的对照，但其 TerminalBench 搜索与终评使用同 89 题，不能混作迁移证据。

[经验管理研究](</home/erzhu419/mine_code/Asumption Agent/reference/How Memory Management Impacts LLM Agents An Empirical Study of Experience-Following Behavior.pdf>) 的错误传播与删除实验、[MRT](</home/erzhu419/mine_code/Asumption Agent/reference/Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning.pdf>) 的成功率—生成 token 曲线，分别对应我们“新知识是否值得保留”和“继续研究是否值得投入”的两个真实问题。二者可以放进同一有限任务序列，无需另造一个终身学习大工程。

## 与当前证据的对应

| 论文要表达的能力 | 当前已经观察到的结果 | 还需要回答的问题 |
|---|---|---|
| 失败后保留可用结构 | 30 个实例的保留干预：保留 180/180，删除 159/180；完整序列计入前缀获取成本。 | 窄机制证据已成立；不需要再做同型删除实验。 |
| 区分不变、系数变与关系变 | 30 个实例、270 个研究结果均通过新鲜确认与终测；旧条件返回调用 C。 | 条件族已验证，尚不能代表一般研究选择。 |
| 从新证据修订具体假设 | 812676 中旧 C 失败，D 保留 C 并修订关系边界；独立终测通过。 | 当前依赖图由已完成拟合步骤编译，不能据此推出在线生成多级子研究。 |
| 知识与研究方法共同积累 | Joint-growth 的确认且终测通过为 232/240，对任务局部研究 215/240；方法保留有后继成本收益。 | 清空方法库证明“有方法”的价值，尚未单独证明“改写方法”优于“继续使用旧方法”。 |
| 新知识有独立后继价值 | 新 concept_0004 的 12 例删除干预输出相同；保留每例额外筛选计算 1,920。 | 此批产物冗余，结论已经明确；不能把旧知识带来的成功归给它。 |
| 比普通 Agent 更好 | 离线 Full 180/180，任务局部 179/180，主要节省观测和计费计算。 | 当前固定基底对照存在过完备拟合弱点；离线研究器消融还不是同一 LLM 的强 Code Agent 对照。 |
| 长期净收益 | 获取成本已计入；joint-growth 计费计算降低 22.42%，实际 CPU 增加 17.92%。 | 收益依赖资源种类和任务复现频率；还不是普遍的长期加速。 |

来源分别为 [四臂比较](../artifacts/paper_composed_comparison_20261005/summary.json)、[保留干预](../artifacts/paper_scoped_retention_20261005/summary.json)、[组件变化](../artifacts/paper_revision_choice_20261005/summary.json)、[joint-growth](../artifacts/growth_paper_20261006_main_30/summary.json) 及论文 Sections 5.8–5.9 / Appendix E。这里的不同分母对应不同实验，不能合并成一个总成功率。

## 优先实验一：子问题真的改变父问题

**问题。** 当一个响应规律失配时，Agent 能否发现自己尚不知道的是仪器含义、适用边界或局部机制，先研究这个支持条件，再据其结果选择父问题的实验与修订？

**最窄任务。** 复用现有匿名仪器与关系变化实验室。让初始观测兼容两种解释，例如“响应机制变了”与“观测坐标的绑定变了”。存在合法校准/干预动作可将它们区分。将“辨认坐标绑定”作为父研究的子问题；在一部分实例中，该子问题又需要解决一个校准条件。算法不能得到实验生成器的变化标签。

同时包含旧方法仍适用、一级支持条件失效、二级支持条件失效的实例。所有臂都能使用相同测量动作、候选语言和初始资料；平铺臂也可以主动选择校准，不能通过禁止关键动作人为制造递归优势。深度是支持关系的变化，不是规定必须走多少次循环。

**主比较只设两个研究组织方式。** 完整方法在线创建未决子问题并消费其结果；平铺研究器拥有相同假设和动作空间，通过反复提议、执行、评估直接解决目标。另在共同前缀的一个方法失配阶段加入“旧研究程序冻结”的诊断分支：仍允许相同底层动作及普通回退研究，仅冻结持久方法的改写。这样能分开在线组织作用和旧方法更新作用，避免完整因子矩阵。

当前需要补的实现是：未决依赖在采样之前驱动调度、子研究返回可用结果、父研究根据结果重新选择实验。先打通一个两层案例；事后生成更多图节点不满足这项实验。

**建议规模与报告。** 用 6 个开发实例确认任务和接口，再冻结至少 30 条独立任务链，平衡上述三种情况；开发例不进入终测。30 是起步设计，不是显著性保证。主终点是同一资源上限内通过新鲜确认且通过独立终测的任务比例；同时报告全程观测、测量、计算及实际 CPU。以任务链为配对单位，不能把一条链中的数百观测当成独立重复。

展示一个实际执行轨迹：父假设失配→产生支持问题→取得区分证据→改变下一项实验→得到修订→新鲜确认。对子结果暂不返回的配对干预，检查父研究是否改了行为以及终测/成本是否变化。正确填写子问题文字而不改变决策，不算机制收益。

本批使用共同的充分资源上限，重点报告规律和预测是否正确、研究步骤是否实际改变；资源曲线不作为本批主目标。失败样本分别说明观测不能区分、子问题选择错误、研究过早停止或错误接受；分析用于解释该实验，不自动扩展成四条修复项目。

若在线与平铺表现相当，就将贡献写为一种可解释的研究组织方式，不声称递归带来计算优势。若冻结旧方法也相同，则保留“方法可执行”结论，暂不声称改写本身产生独立收益。

## 优先实验二：同一 LLM 的普通 Agent 对照

**共用实验一的任务、工具、独立终测和预算口径。** 三个主臂足够：普通 Code/ReAct；带跨任务文字经验的 Reflexion/ExpeL 风格代理；带可执行假设、支持问题和可修订研究程序的黑格尔机。前两者若使用本地适配实现，应明确称风格对照，不冒充复现原论文的完整成绩。

模型在各臂承担相同层面的候选提议或代码生成工作。各臂均可使用相同原始数据和实验原语；任何专用求解器与人工给定先验要么共同可用，要么作为处理差异明确报告。普通代理要能正常主动试错，不能用当前拟合失灵的 fixed_library 替代。

固定模型版本、工具权限、实验预算、累计输入/输出 token 与重试额度；分别记录 tokens、环境调用和执行时间，不把调用次数当 token，也不把三种资源折成未经解释的综合分。少量开发任务只用于把三臂接通，终测任务按同一预定清单执行。学习前缀和经验生成计入完整序列成本，前缀可物理共用但每个反事实分支都承担同样的获取成本。

主结果报告确认且终测成功、错误确认、各资源花费与直接调用旧知识的比例。实验一已隔离在线/平铺机制，这里不再乘出大量模块开关。实际模型固定为 ruoli.dev 的 gpt-6.1-sol；接口测试与真实模型成绩分开记录。

## 实验三：冻结后的结构迁移

冻结源任务学得的响应和研究程序，留出**组合结构**而不只是随机种子。目标任务覆盖结构相同/不同与表面表示相近/不同的四种组合；特别包含表面相近但旧机制不适用的目标。[ARN](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gscl_intrinsic_candidates_20260730/papers/arn_2024_tacl_1_59.pdf>) 提供表面干扰的设计启发，[组合泛化研究](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/14_compositional_structure_generalization_iclr2025.pdf>) 说明保存组件并不保证新组合成功，[组合接口研究](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/16_luketina25a.pdf>) 提供冻结旧模块、只学新接口的对照范式。

在共同已付前缀后比较保留新程序、移除该程序但保留祖先、任务局部研究。并列报告直接复用、必要局部修订、错误迁移及恢复。现有 concept_0004 的冗余负结果继续保留；新的迁移任务按结构问题预先选择，而不是挑到能让新知识取胜为止。有限序列即可展示累计净成本，不需要以开放世界终身收益为完成门槛。

若需要一个独立公开数据领域，候选是 [DiscoveryBench 官方仓库](https://github.com/allenai/discoverybench)：它提供数据驱动发现任务和 coder/react 基线；[任务元数据](https://github.com/allenai/discoverybench/blob/main/discoverybench/README.md) 包含领域、工作流和数据说明。应先按任务及资源条件预选一小组，再接原始评价；这只是扩展建议，尚未选定名单或下载数据。它在已有数据上做分析，不自动支持购买新物理实验，也不能单独证明跨任务程序复用。当前不另开该适配工程。

## 论文如何因此更完整

主文保留四个证据位置即可：已有保留/修订干预；新增在线递归的因果比较；同模型普通代理比较；一条从失败到子研究、修订、确认和回用的可执行轨迹。成本曲线并入对应结果。多个同型返回实验、历史修复、实现细目放入附录。

大框架可以连接预测、RL、鲁棒性和因果，但实证论断聚焦于一个问题：**把支持假设及研究程序当作可修订知识，是否改变了研究的成功率和成本？** 这比在四个领域各跑一个没有隔离机制的小演示更能支撑当前论文。

## 核读边界

本次查看了全部文件的身份与主题，对方法论文核读实验设置、主结果和相关附录；没有重跑这些论文，也没有逐页证明其理论结论。数学书按目录、导言及相关章节审读。名为 Categorical Information Geometry 的 641 页文件实际是整本 GSI 2023 Part I：本次读目录及对应章节（PDF pp.298–307），并识别全书含其他数值研究，不将其余每章都声称为已逐篇精读。所有数字对应本地版本；文献实验的强弱不替代我们自己的结果。下面清单中的“无原创实验”指没有该文新组织的 Agent/算法经验比较，不否认其数学结果或引用案例。

## 全部文档的实验清单

以下编号对应扫描顺序；完整文档与图稿分别列出。短标题链接到本地实际文件。

### reconstruction 与 reconstruction_v2 根目录

| 编号及文档 | 作者做了什么实验/对照 | 结果与本论文用途 | 本地定位 |
|---|---|---|---|
| 01 [Self-Evolving Agents Survey](</home/erzhu419/mine_code/Asumption Agent/reconstruction/reference/A Survey of Self-Evolving Agents.pdf>) | 综述；整理演化对象、时机、应用和评价协议，没有统一新实验。 | 既有结果的底模/预算不同；支持区分题内反思、跨题学习、保留与成本。 | pp.40–47，Tables 6–11 |
| 02 [Agentic Harness Engineering](</home/erzhu419/mine_code/Asumption Agent/reconstruction/reference/Agentic Harness Engineering Observability-Driven Automatic Evolution of Coding-Agent Harnesses.pdf>) | Terminal-Bench 89题优化10轮，同起点比ACE/Training-Free GRPO；冻结后测SWE Verified 500题；拆记忆、工具、中间件、prompt。 | 优化集69.7%→77.0%，迁移75.2%→75.6%；单条演化轨迹。借鉴冻结后新题及组件干预。 | pp.6–10、16，Tables 1–4 |
| 03 [Discrete World Models via Regularization](</home/erzhu419/mine_code/Asumption Agent/reconstruction/reference/Discrete World Models via Regularization.pdf>) | MNIST 8-puzzle、IceSlider，干净/噪声数据；AE、βVAE、DeepCubeAI及正则消融；冻结编码器测符号状态/一步预测，10次。 | 测表示与一步动态，未直接验证端到端规划或递归研究。表示质量与任务收益应分开。 | pp.8–13，Tables 1–3 |
| 04 [EmbodiSkill](</home/erzhu419/mine_code/Asumption Agent/reconstruction/reference/EmbodiSkill Skill-Aware Reflection for Self-Evolving Embodied Agents.pdf>) | ALFWorld、EmbodiedBench，冻结演化后的技能；无记忆及多种记忆系统；静态/粗粒度/定向修订消融。 | ALFWorld一配置93.28%，导航61.33%；定向修订优于静态，未给完整同token演化成本。 | pp.9–12，Tables 1–3 |
| 05 [HippoRAG 2 / From RAG to Memory](</home/erzhu419/mine_code/Asumption Agent/reconstruction/reference/From RAG to Memory Non-Parametric Continual Learning for Large Language Models.pdf>) | 七套事实、多跳、长篇QA；同底模检索对照；查询链接/段落/过滤消融，四阶段语料扩张。 | 结构检索改善部分任务；复杂QA仍随扩库下降，建库另有时间及LLM成本。 | pp.6–9、19，Tables 1–4、12 |
| 06 [Generative Recursive Reasoning / GRAM](</home/erzhu419/mine_code/Asumption Agent/reconstruction/reference/Generative Recursive Reasoning.pdf>) | Sudoku、ARC、N-Queens、图着色、MNIST；比TRM/HRM等，拆深监督、层级递归、随机引导；测步数与多样性。 | 是潜在状态迭代而非显式子假设论证；比较深度/宽度计算，但步数相同未必实际计算相同。 | pp.7–10、18–21，Tables 1、3、7–8 |
| 07 [Huxley–Gödel Machine](</home/erzhu419/mine_code/Asumption Agent/reconstruction/reference/HUXLEY-GÖDEL MACHINE HUMAN-LEVEL CODING AGENT DEVELOPMENT BY AN APPROXIMATION OF THE OPTIMAL SELF-IMPROVING MACHINE.pdf>) | 同祖先、800任务评估比DGM/SICA；后继演化收益相关；放大优化集，移除重叠题后在207道SWE Lite新题测迁移。 | 迁移34.8%→40.1%，人工SWE-agent39.6%；当前分数与后继潜力不同，整体非同token对照。 | pp.7–10，Tables 1–4 |
| 08 [HippoRAG（arXiv版）](</home/erzhu419/mine_code/Asumption Agent/reconstruction/reference/HippoRAG Neurobiologically Inspired Long-Term Memory for Large Language Models.pdf>) | 三套多跳QA各1000题；比稠密检索/IRCoT，拆OpenIE、PPR、同义边与特异度。 | 2Wiki显著提高，HotpotQA F1低于ColBERTv2；检索成功不能等同可执行规律发现。与11为同一工作。 | pp.5–8、28，Tables 2、4–6、17–18 |
| 09 [What's your hypothesis?](</home/erzhu419/mine_code/Asumption Agent/reconstruction/reference/s41592-026-03104-y(1).pdf>) | Nature Methods两页社论，评论AI科学假设案例；没有作者新运行的对照实验。 | 其中提出的假设与既有人类发现对应；不要将报道当成新的独立验证。 | pp.1–2 |
| 10 [The Red Queen Gödel Machine](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf>) | 代码、论文写作/评审、奥赛证明/评分；固定评估器与共演化比较，替换/对抗池/清除消融，分列搜索成本。 | 代码119/166对116/166；写作看模型接受率；证明均分与满分率排序不同。详见前文。 | pp.6–13、19–25，Tables 1–7 |

### 广义结构对应材料

| 编号及文档 | 作者做了什么实验/对照 | 结果与本论文用途 | 本地定位 |
|---|---|---|---|
| 11 [HippoRAG（NeurIPS版）](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/papers/01_hipporag_neurips2024.pdf>) | 08的会议版本，多出附加材料/checklist；所核对实验内容相同。 | 单独保留文件定位，不重复计为另一项验证或另一组样本。 | pp.5–8、28 |
| 12 [Functorial Data Migration](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/papers/02_functorial_data_migration_arxiv1009.1166.pdf>) | 范畴化数据库与数据迁移理论；连接、投影、并集等构造示例，无Agent经验比较。 | 为接口与结构保持提供形式语言，不能替代迁移实验。 | pp.1–2、5–13、21起 |
| 13 [Categorical Deep Learning（Position）](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/papers/03_categorical_deep_learning_icml2024.pdf>) | 用范畴结构解释等变、RNN、参数绑定等架构；无新组织的Agent基线实验。 | 帮助定义结构约束，不要求我们的首篇实现全部范畴机制。 | pp.1–2及方法章节 |
| 14 [HAKE](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/papers/04_hake_aaai2020.pdf>) | WN18RR、FB15k237、YAGO3-10链接预测，MRR/Hits；TransE/RotatE等及模长/相位/偏置消融。 | 验证知识图谱表示；部分基线引自既有论文，不测自主假设或递归检验。 | pp.5–7，Tables 3–5 |
| 15 [Hypergraph PageRank](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/papers/05_hypergraph_pagerank_icml2024.pdf>) | 噪声矩形/球面数据，2/7/15维、50次；UCI半监督分类；图/超图AUC、收敛误差与时间对照。 | 展示结构表示与标签传播价值；配套9个图稿不是9篇实验论文。 | pp.7–8、23，Figure 1、Tables 2–4 |
| 16 [A Taxonomy of Categories for Relations](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/papers/06_taxonomy_categories_relations_lmcs2026.pdf>) | 纯理论分类，关系范畴及其丰富化、Kleisli构造；无经验算法对照。 | 可用来表达关系组合与类型，非自进化效果证据。 | pp.1–4及分类定理 |

### 建议路线中的先验、表示与假设材料

| 编号及文档 | 作者做了什么实验/对照 | 结果与本论文用途 | 本地定位 |
|---|---|---|---|
| 26 [Geometric Deep Learning](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/01_geometric_deep_learning_arxiv2104.13478.pdf>) | 理论教程/综述，用群、局部性、尺度与等变性统一架构；列举已有视觉、分子与图应用。 | 没有统一新训练实验；结构先验的形式化与先验自主发现是两件事。 | pp.105–129 |
| 27 [Learning Markov State Abstractions for Deep RL](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/02_learning_markov_state_abstractions_arxiv2106.04379.pdf>) | 6×6网格300种子、冻结表示测控制；六个DMC域50万环境步，主要10种子；表示损失/预训练消融。 | 表示充分性必须通过后续控制验证；部分任务不胜RAD，环境样本和更新数均需匹配。 | pp.9–10、26–34，Figures 3、11 |
| 28 [CROSS: Efficient Low-Rank Tensor Completion](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/04_cross_low_rank_tensor_completion_arxiv1611.01129.pdf>) | 合成低秩张量每设定1000次；误差、采样、噪声与CPU；ADHD影像10幅各100次，比LRTC/HOSVD。 | 给定低秩先验可省采样；并未自主提出或撤回先验。 | pp.19–25，Figures 3–6、Table 1 |
| 29 [Bayesian Optimization with Monotonicity](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/05_bayesian_optimization_monotonicity.pdf>) | 2D/5D目标与五参数纤维实验，比普通BO，50次均值/标准误；逐轮选择单调方向。 | 人工给定单调性帮助寻优；缺失错误单调先验和撤回对照。 | pp.4–5，Figures 1–2 |
| 30 [Beyond Convexity: Submodularity（教程）](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/06_submodularity_tutorial_slides.pdf>) | 150页幻灯片，特征选择、主动学习、传感器与水网布置案例，来自已有研究。 | 边际递减条件下的选择理论，不代表任意知识收益都次模。 | pp.90、135 |
| 31 [Physics-Informed Architectures for Dynamics](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/07_physics_informed_architectures_arxiv2109.06407.pdf>) | 双摆及五个Brax机器人；无先验MLP与逐级加入动力分解、对称和接触约束，比较预测与滚动误差。 | 人工正确约束的价值；未测试失配先验撤回。双摆重复次数在图与附录口径不同。 | pp.7–9、16，Figures 3–5 |
| 32 [Perspectives on Contractivity](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/08_contractivity_perspectives_arxiv2404.11707.pdf>) | 理论观点，收缩、稳定、鲁棒与模块组合的条件及示意例；无统一新经验对照。 | 可以说明支持条件；收敛本身不意味着答案正确。 | pp.1、3–9 |
| 33 [User-friendly Introduction to PAC-Bayes Bounds](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/12_pac_bayes_introduction_arxiv2110.11216.pdf>) | 理论教程，推导泛化界并介绍既有MNIST非空泛界实验。 | 不是本教程新跑的Agent实验；采用先验仍有数据依赖和适用条件。 | pp.55–60 |
| 34 [Generalized Computable Universal Priors](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/13_generalized_computable_universal_priors_arxiv_cs0503026.pdf>) | 纯理论，计算性层级、可支配关系及收敛/不收敛构造。 | 不提供有限Agent成功率或预算保证。 | p.9 Table 5、pp.18–19 |
| 35 [When Does Compositional Structure Yield Compositional Generalization?](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/14_compositional_structure_generalization_iclr2025.pdf>) | 核理论加MNIST/CIFAR实验；改变训练组合、间距、插值/外推，10次类别重排；ConvNet/ResNet/ViT。 | 组合表示可能仍失败或学到捷径；需要结构留出而非只换随机种子。 | pp.6、9–10、32–33，Figures 5、11–12 |
| 36 [Learning Robust Models Using Independent Causal Mechanisms](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/15_independent_mechanism_analysis_arxiv2010.07167.pdf>) | 1365组合SCM、未见干预；Flow/ANM、ICP/ERM/CERM、gate消融；Colored-MNIST跨环境。 | Colored-MNIST未见环境68.5%对ERM10.2%，不变约束过强也退化。文件名不是正文标题。 | pp.6–8、15，Figures 4–5 |
| 37 [Compositional Interfaces for Compositional Generalization](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/16_luketina25a.pdf>) | 240种观测×动作×指令组合；随机/困难留出和新感知模块；同架构独立训练，固定192k轨迹比较覆盖。 | 冻结旧模块、只学新接口可省数据；困难留出仍低于独立训练。正文CoLLAs 2024，文件名带25。 | pp.6–9、15，Figures 3–7 |
| 38 [Change-point Detection with Deep Learning（综述）](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/17_change_point_detection_deep_learning_review.pdf>) | 整理在线/离线变点、真实数据集与方法，无作者统一新对照。 | 机制变化应同时看延迟、误报、漏检；发现变点不等于找到失效原因。 | pp.3–5、17，Tables 1–3 |
| 39 [Provable Separation of Scales in Maximal Update Parameterization](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/18_hong25h.pdf>) | 理论附单一宽度10000的两层MLP/CIFAR-10实验，300 epoch扫学习率，测损失与权重变化。 | 概念验证很窄，跨宽度定量实验留待未来；不能借其标题声称已验证跨尺度研究。 | p.13 |
| 40 [Symmetry-aware Learning（综述）](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/19_symmetry_aware_learning_review.pdf>) | 汇编104篇论文与5项资源，比较已有对称架构性能/部署开销，无标准化重跑。 | 正确、近似和错误对称应区分；汇编成本不是我们的预算依据。 | pp.5、25–30，Tables 5–7 |
| 41 [When Physics Meets Machine Learning（综述）](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/20_physics_meets_machine_learning_survey.pdf>) | 按物理知识与数据/架构/损失注入方式分类案例，无统一原创实验。 | 自动选择合适物理知识仍是缺口；不是堆入更多正确先验即可解决。 | pp.15–16 |
| 42 [Topological Data Analysis and Deep Learning（综述）](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/21_topological_data_deep_learning_review.pdf>) | 持续同调、Laplacian/Dirac、sheaf、多尺度方法及应用/软件综述。 | 不变表示也可能抹掉任务必需区别；无统一新Agent比较。 | pp.45–46、51 |
| 43 [Hypothesis-Driven Reasoning for LLMs](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/22_hypothesis_driven_reasoning_llms_aaai2026.pdf>) | 合成视觉异常规则、三难度、150多物体测试、三次重复；Reflexion、oracle及去因子/迭代消融。 | 直接相关的生成—检查—使用先例；没有同调用预算或主动购买区分实验。 | pp.5–7，Tables 2–5 |
| 44 [Hypothesis Generation and Updating in LLMs](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/papers/25_hypothesis_generation_updating_arxiv2605.05851.pdf>) | 数字概念任务8+255刺激集、逐步给例，8种LLM及thinking；分别测自由生成、规则评价、逐数预测与域外推。 | 三种读出可能不一致；文字假设好看不能替代实际执行正确。 | pp.3–8、20，Figures 4–6 |

### 类比和关系推理材料

| 编号及文档 | 作者做了什么实验/对照 | 结果与本论文用途 | 本地定位 |
|---|---|---|---|
| 60 [ANALOBENCH](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gscl_intrinsic_candidates_20260730/papers/analobench_2024_emnlp_main_725.pdf>) | 340故事、47类，1/10/30句；四选一与故事库检索，多个LLM及人工对照。 | 文本加长使类比匹配退化；只测识别/检索，不测迁移后的可检验预测。 | pp.5–8，Tables 1–4 |
| 61 [ARN](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gscl_intrinsic_candidates_20260730/papers/arn_2024_tacl_1_59.pdf>) | 1096类比三元组；近/远结构对应×近/远表面干扰；SBERT/LLM零/少样本，120题人工子集。 | LLM平均57.3%对人工96%；远类比均值39.6%低于二选一随机线。适合构造错误迁移反例。 | pp.8–12，Table 2 |
| 62 [SCAR](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gscl_intrinsic_candidates_20260730/papers/scar_2023_findings_emnlp_160.pdf>) | 400系统类比、632系统、1614概念映射，中英双语；零/少样本、背景、解释与人工映射比较。 | 测试角色映射；还需后继执行验证，不能只凭对应关系的语言评分。 | pp.4、6–7，Tables 3、5 |
| 63 [CLUTRR](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gscl_law_native_candidates_20260802/papers/clutrr_emnlp2019_D19-1458.pdf>) | 家族关系故事；短链训练、长链测试，留出释义/逻辑子句，噪声条件；LSTM/RN/MAC/BERT与真实图GAT。 | 测组合长度与噪声鲁棒性；真实图控制额外获得结构信息，应标明。 | pp.6–8，Figure 5、Table 2 |

### 根 reference 的 Agent、元学习和数学材料

| 编号及文档 | 作者做了什么实验/对照 | 结果与本论文用途 | 本地定位 |
|---|---|---|---|
| 64 [Algebra Unveils Deep Learning / Neuroalgebraic Geometry](</home/erzhu419/mine_code/Asumption Agent/reference/Algebra Unveils Deep Learning An Invitation to Neuroalgebraic Geometry.pdf>) | 立场/理论，维数、次数、奇异性与网络表达/学习之间的对应。 | 无新组织的自进化实验；用于形式化而非本项目经验依据。 | pp.1–2、9，Table 1 |
| 65 [POPPER: Agentic Sequential Falsification](</home/erzhu419/mine_code/Asumption Agent/reference/Automated Hypothesis Validation with Agentic Sequential Falsification.pdf>) | DiscoveryBench与TargetVal；给定H生成子检验，执行代码与顺序证据；CodeGen/ReAct/Self-Refine及关联/统计消融，5次。 | 同时测检出与错误接受；9位生物学家小样本比较。给定假设验证不等同自主产生可迁移理论。 | pp.5–8，Tables 3–4 |
| 66 [Categorical and Geometric Methods in Learning](</home/erzhu419/mine_code/Asumption Agent/reference/CATEGORICAL AND GEOMETRIC METHODS IN STATISTICAL, MANIFOLD, AND MACHINE LEARNING.pdf>) | 概率态射、统计/流形/机器学习的理论与数学例子。 | 无统一Agent任务、预算、基线；属于形式基础。 | pp.1–3及方法章节 |
| 67 [GSI 2023 Part I / Categorical Information Geometry章](</home/erzhu419/mine_code/Asumption Agent/reference/Categorical Information Geometry.pdf>) | 文件实际641页会议论文集；相应章节统一Markov范畴、散度、熵与数据处理不等式。 | 目标章节是理论；全书其他章节有数值实验。本次按目录及目标章核读，不把整本当一篇AI论文。 | 目录pp.26–33；目标章pp.298–307 |
| 68 [Categorical Probability and Stochastic Dominance](</home/erzhu419/mine_code/Asumption Agent/reference/Categorical Probability and Stochastic Dominance in Metric Spaces.pdf>) | 209页数学学位论文；Kantorovich monad、度量与凸序等定理。 | 没有Agent自演化经验对照；实验比较的数学基础与真实运行需分开。 | pp.5–8及主体定理 |
| 69 [Continual Learning of LLMs（综述）](</home/erzhu419/mine_code/Asumption Agent/reference/Continual Learning of Large Language Models A Comprehensive Survey.pdf>) | 整理持续预训练/微调、任务与数据集；评价协议、遗忘和迁移指标。 | 没有作者新方法的统一重跑；启发全过程前向/后向迁移和遗忘评价。 | pp.23–25，Table 4 |
| 70 [EvolveR](</home/erzhu419/mine_code/Asumption Agent/reference/EvolveR Self-Evolving LLM Agents through an Experience-Driven Lifecycle.pdf>) | NQ/HotpotQA经验训练，另五套QA测域外；Qwen0.5B/1.5B/3B；直接回答、RAG/SFT/Search-R1与检索/内化/教师消融。 | 3B均EM .382，禁检索 .340；检索与参数学习应分开归因，未报全生命周期回本。 | pp.7–10、18，Tables 1–3、9 |
| 71 [ExpeL](</home/erzhu419/mine_code/Asumption Agent/reference/ExpeL LLM Agents Are Experiential Learners.pdf>) | HotpotQA/ALFWorld/WebShop四折；同GPT-3.5 Act/ReAct、仅检索/仅摘要；成功失败经验、随机检索等消融。 | 迁往FEVER时GPT-4改写摘要；完整方法常增token。分开实例与抽象经验的贡献。 | pp.7–10、38，Tables 1–3、6 |
| 72 [From AI for Science to Agentic Science](</home/erzhu419/mine_code/Asumption Agent/reference/From AI for Science to Agentic Science.pdf>) | 84页科学Agent综述，能力、闭环与既有领域案例。 | 案例汇编不是新跑实验；适合定位窄而完整的科学研究闭环。 | pp.13–27及案例表 |
| 73 [GenericAgent](</home/erzhu419/mine_code/Asumption Agent/reference/GenericAgent A Token-Efficient Self-Evolving LLM Agent via Contextual Information Density Maximization (V1.0).pdf>) | SOP-Bench/Lifelong AgentBench/RealFin、5项同Sonnet长任务；无/完整/冗余/压缩记忆，重复任务成本。 | 同模型5任务都完成，token有差；压缩有时优于完整记忆，重复任务不等于新结构迁移。 | pp.14–21，Tables 2、4–8 |
| 74 [How Memory Management Impacts LLM Agents](</home/erzhu419/mine_code/Asumption Agent/reference/How Memory Management Impacts LLM Agents An Empirical Study of Experience-Following Behavior.pdf>) | RegAgent/EHRAgent/AgentDriver/IoT；固定/全收/粗审/严审，周期/效用删除，分布重排及容量限制。 | 全收可传播错误；严格评审由真值模拟，不是免费自主能力；差评审会误删。 | pp.3–8、19，Tables 1–2、4 |
| 75 [Information Geometry（Ay等专著）](</home/erzhu419/mine_code/Asumption Agent/reference/Information Geometry (Nihat Ay, Jürgen Jost, Hông Vân Lê etc.).pdf>) | 411页数学专著，统计模型、Fisher结构、充分统计量、几何与定理。 | 不是自演化基准论文；按导言/目录和相关理论定位。 | pp.5–12及相关章节 |
| 76 [Internalizing Meta-Experience / MEL](</home/erzhu419/mine_code/Asumption Agent/reference/Internalizing Meta-Experience into Memory for Guided Reinforcement Learning in Large Language Models.pdf>) | DAPO-Math-17k训练，AIME/AMC/MATH500/OlympiadBench；同框架GRPO，另接RFT/REINFORCE++；多个规模。 | 14B Pass@1平均56.30→61.03；主要测参数内化，元经验生产成本未独立摊销。 | pp.6–8，Table 1、Figures 3–5 |
| 77 [LeWorldModel](</home/erzhu419/mine_code/Asumption Agent/reference/LeWorldModel Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.pdf>) | TwoRoom/Reacher/PushT/OGBenchCube像素世界模型及MPC规划；DINO-WM/PLDM/GCBC等；状态探针、预期违背、架构消融。 | 附录3训练种子；表示探针之外测实际控制，提醒预测质量不必转成行动成功。 | pp.7–9、21–27，附录Table 5 |
| 78 [Lifelong Learning of LLM-based Agents（Roadmap）](</home/erzhu419/mine_code/Asumption Agent/reference/Lifelong Learning of Large Language Model based Agents A Roadmap.pdf>) | 综述路线，感知/记忆/行动与平均性能、全过程性能、遗忘、前后向迁移。 | 无自己新跑的完整算法；指标可以借鉴，长期收益仍须计入获取和维护费用。 | pp.1–3、30–33，式14–18 |
| 79 [Memory-R1](</home/erzhu419/mine_code/Asumption Agent/reference/Memory-R1 Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning.pdf>) | LoCoMo 152训练问，LoCoMo/MSC/LongMemEval；同骨干RAG/A-Mem/Mem0/MemoryOS/SFT，管理/回答RL及蒸馏消融。 | 管理与使用可分开干预；LLM评审奖励提高评审分却降低词面指标，成本未全摊销。 | pp.5–8、18，Tables 1–5 |
| 80 [Meta-Harness](</home/erzhu419/mine_code/Asumption Agent/reference/Meta-Harness End-to-End Optimization of Model Harnesses.pdf>) | 分类、数学检索、TerminalBench；同提议模型及候选评估量的搜索对照，分数/摘要/原轨迹消融；部分领域冻结后新题。 | 有精度—上下文前沿；TerminalBench搜索与终评同89题，不能视作全部领域的独立泛化。 | pp.5–9，Tables 2–7 |
| 81 [Meta Reinforcement Fine-Tuning / MRT](</home/erzhu419/mine_code/Asumption Agent/reference/Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning.pdf>) | 数学推理，对结果奖励GRPO/STaR/长度惩罚/RISE；4K–16K及32K预算、并行和回溯。 | 同精度省约1.2–1.6倍生成token；推理节省不代表额外训练已回本。 | pp.11–17，Table 1、Figures 7–12 |
| 82 [Bayesian Meta-Reasoning（Position）](</home/erzhu419/mine_code/Asumption Agent/reference/Position LLMs Need a Bayesian Meta-Reasoning Framework for More Robust and Generalizable Reasoning.pdf>) | 任务层/元层概念框架，先验更新、监控、自知与调节；无整套实现的原创效果实验。 | 可帮助问题定位，不能说其元层已获实证，也不要求我们必须改成贝叶斯实现。 | pp.3–8，Figures 2–5、Table 1 |
| 83 [ReMA](</home/erzhu419/mine_code/Asumption Agent/reference/ReMA Learning to Meta-think for LLMs with Multi-agent Reinforcement Learning.pdf>) | MATH与多套域外数学、RewardBench→JudgeBench；同底模CoT/普通RL/单模型元思考/分层双代理，多轮与长度消融。 | 数学有增益，评审分支非全胜；多轮可重复或空答，层级本身不保证提高。 | pp.6–9、27，Tables 1–2、Figures 3–6 |
| 84 [Reflexion](</home/erzhu419/mine_code/Asumption Agent/reference/Reflexion Language Agents with Verbal Reinforcement Learning.pdf>) | 134 ALFWorld、100 HotpotQA、HumanEval/MBPP/Leetcode；重试、轨迹记忆、内部测试和文字反思消融。 | HumanEval Python91%，MBPP Python却降至77%；内部测试误通过会误导研究。成本主要按重试数。 | pp.5–8，Tables 1–3、Figures 3–4 |
| 85 [Representable Markov Categories and Comparison of Statistical Experiments](</home/erzhu419/mine_code/Asumption Agent/reference/Representable Markov Categories and Comparison of Statistical Experiments in Categorical Probability.pdf>) | 证明二阶随机占优和Blackwell–Sherman–Stein等价关系；纯理论。 | 标题的experiments是统计信息通道，不是作者运行了AI实验；用于信息可区分性的形式表达。 | pp.37–54 |
| 86 [SELF-DISCOVER](</home/erzhu419/mine_code/Asumption Agent/reference/SELF-DISCOVER Large Language Models Self-Compose Reasoning Structures.pdf>) | BBH/T4D/200 MATH，从39已有模块生成结构；CoT/Plan-and-Solve/投票，SELECT/ADAPT/IMPLEMENT及跨模型迁移。 | GPT-4 BBH81%对CoT75%；调用数更少不等于token更少，未测反例驱动的持续方法改写。 | pp.4–8、13，Table 1–2 |
| 87 [Category-theoretic Classical and Quantum Information Geometry](</home/erzhu419/mine_code/Asumption Agent/reference/Towards a category-theoretic foundation of Classical and Quantum Information Geometry.pdf>) | 定义NCP、协方差函子及相关数学结果预告。 | 8页理论文章，没有Agent经验比较或自我意识实验。 | pp.1–8 |
| 88 [Voyager](</home/erzhu419/mine_code/Asumption Agent/reference/Voyager An Open-Ended Embodied Agent with Large Language Models.pdf>) | Minecraft160提示额度比ReAct/Reflexion/AutoGPT；技能/课程/反馈消融；新世界4目标各3次、50提示上限，技能库跨控制器。 | 新目标完整方法均3/3，但重复很少；技能可复用与完整控制器可分开，成本不是完整token。 | pp.6–9，Tables 1–2、Figures 7–9 |

### 24 个图稿文件

这些文件属于上表第15和35项的代码仓库，是论文图片及草稿，不另计为24项独立研究。均为单页。

| 编号 | 图稿 |
|---|---|
| 17 | [Page 1.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/repos/05_hypergraph_diffusions/figures/ICML/Page 1.pdf>) |
| 18 | [error_vs_iteration.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/repos/05_hypergraph_diffusions/figures/ICML/error_vs_iteration.pdf>) |
| 19 | [estimated_labels_example.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/repos/05_hypergraph_diffusions/figures/ICML/estimated_labels_example.pdf>) |
| 20 | [first_draft.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/repos/05_hypergraph_diffusions/figures/ICML/first_draft.pdf>) |
| 21 | [AUC_hist.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/repos/05_hypergraph_diffusions/figures/NEURIPS/AUC_hist.pdf>) |
| 22 | [AUC_hist_for_drawing.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/repos/05_hypergraph_diffusions/figures/NEURIPS/AUC_hist_for_drawing.pdf>) |
| 23 | [draft_2.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/repos/05_hypergraph_diffusions/figures/NEURIPS/draft_2.pdf>) |
| 24 | [estimated_labels_example.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/repos/05_hypergraph_diffusions/figures/NEURIPS/estimated_labels_example.pdf>) |
| 25 | [manifold_learning_4_23.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/generalized_structural_correspondence_20260730/repos/05_hypergraph_diffusions/figures/NEURIPS/manifold_learning_4_23.pdf>) |
| 45 | [fig-1.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-1.pdf>) |
| 46 | [fig-2.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-2.pdf>) |
| 47 | [fig-3.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-3.pdf>) |
| 48 | [fig-4.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-4.pdf>) |
| 49 | [fig-5.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-5.pdf>) |
| 50 | [fig-addition-cifar.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-addition-cifar.pdf>) |
| 51 | [fig-addition-dispersed.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-addition-dispersed.pdf>) |
| 52 | [fig-addition-mnist.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-addition-mnist.pdf>) |
| 53 | [fig-addition-rich.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-addition-rich.pdf>) |
| 54 | [fig-cdm-cifar.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-cdm-cifar.pdf>) |
| 55 | [fig-cdm-mnist.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-cdm-mnist.pdf>) |
| 56 | [fig-cdm-rich.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-cdm-rich.pdf>) |
| 57 | [fig-partial.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-partial.pdf>) |
| 58 | [fig-sim.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-sim.pdf>) |
| 59 | [fig-te-deep.pdf](</home/erzhu419/mine_code/Asumption Agent/reconstruction_v2/reference/gpt_advice_roadmap_20260729/repos/14_compositional-generalization/figures/fig-te-deep.pdf>) |
