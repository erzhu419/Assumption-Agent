# 结构迁移研究：一手来源与核对范围

检索与核对日期：2026-11-19（Asia/Shanghai；UTC 为 2026-11-18）。主代理需自行 open 下列 URL，方可在主答复引用对应网页；本表给出已存在的 web ref 方便定位，不编造引用。大多数来源是稳定理论，不以搜索引擎的“近来抓取/上传时间”冒充论文发表年。

正文为 `01_structural_transfer_report.md`。S1–S17 对应其中引用。除作者提供的 AWM v1 外，未在正文大量引用原文；公式和例子主要以概念复述与明确标记的推导/设计建议呈现。

## S1 — Gentner，结构映射理论（1983）

- 作者：Dedre Gentner。
- 题名：Structure-Mapping: A Theoretical Framework for Analogy。
- 正式出版：Cognitive Science 7(2), 155–170，1983。
- 一手原文：<https://groups.psych.northwestern.edu/gentner/papers/Gentner83.2b.pdf>
- 已读 ref：`turn513947681view1`。
- 定位：摘要、关系与属性区分、systematicity 段落。
- 支持：关系系统而非总体特征重合、类比与字面相似区分。
- 不支持：对任何新领域的真实性/效用保证；把全部结构相似等同完全图同构。

## S2 — SME（1989）与作者项目说明

- 作者：Brian Falkenhainer、Kenneth D. Forbus、Dedre Gentner。
- 题名：The Structure-Mapping Engine: Algorithm and Examples。
- 正式出版：Artificial Intelligence 41, 1–63，1989。
- 原文：<https://www.qrg.northwestern.edu/papers/Files/smeff2(searchable).pdf>
- 原文 ref：`turn427825720view0`。
- 作者项目：<https://qrg.northwestern.edu/ideas/smeidea.htm>
- 项目 ref：`turn513947681view0`。
- 定位：原文§2–3；项目“SME: The Structure-Mapping Engine”，尤其输出组成。
- 支持：局部匹配组成结构一致对应，输出候选推论、对应与结构评分；匹配模块可嵌入更大系统。
- 不支持：结构评分等于候选真值；召回、适用与行动成本已经被同时解决。

## S3 — MAC/FAC（1995）

- 作者：Kenneth D. Forbus、Dedre Gentner、Keith Law。
- 题名：MAC/FAC: A Model of Similarity-Based Retrieval。
- 出版：Cognitive Science 19(2), 141–205，1995。
- 原文：<https://groups.psych.northwestern.edu/gentner/papers/ForbusGentnerLaw94.2b.pdf>
- 原文 ref：`turn733924961view1`、`turn233516272view5`。
- 作者项目：<https://www.qrg.northwestern.edu/ideas/macfac.htm>
- 项目 ref：`turn733924961view0`。
- 定位：摘要、§2.1 structural consistency、两阶段检索。
- 年份注意：扫描首页印1994，但作者修注与官方索引明确应为1995。
- 支持：廉价内容向量粗筛、SME精排；一对一对应与连通约束。
- 不支持：当前Hegel非用此架构不可，或PPR自然实现同样验证。

## S4 — SMDP 同态（2003）

- 作者：Balaraman Ravindran、Andrew G. Barto。
- 题名：SMDP Homomorphisms: An Algebraic Approach to Abstraction in Semi-Markov Decision Processes。
- 正式会议页：<https://ijcai.org/proceedings/2003/144>
- 原文：<https://ijcai.org/proceedings/2003/0144.pdf>
- refs：`turn513947681view4`、`turn814463484view2`、`turn392160917view0`。
- 定位：§3 定义及式(1)(2)，§4 策略转移，§5结构映射搜索，§6.2–6.3局部/option特定抽象。
- 支持：状态映射、状态依赖行动重编码、聚合转移/回报/持续时间保持；部分任务可有抽象而全系统没有。
- 边界：是模型满足精确条件的保证，不是只见有限数据就已证明全称同态；一般映射搜索也可能组合爆炸。
- 2002 SARA原文的作者站访问403，因此最终以可完整阅读的2003正式论文为主。

## S5 — 随机双模拟与模型最小化（2003）

- 作者：Robert Givan、Thomas Dean、Matthew Greig。
- 题名：Equivalence notions and model minimization in Markov decision processes。
- 出版：Artificial Intelligence 147, 163–223，2003。
- 原文大学镜像：<https://ics.uci.edu/~dechter/courses/ics-295/fall-2019/papers/givan-dean-greig.pdf>
- refs：`turn733924961view2`、`turn693832847view4`。
- 定位：§3 stochastic bisimulation、Theorems 5–7、§4.2–4.3。
- 支持：保持块转移概率与回报的商模型可转移最优策略。
- 边界：不能用无权、无奖励的拓扑同构替代该关系。

## S6 — 近似状态抽象（2016）

- 作者：David Abel、David Hershkowitz、Michael Littman。
- 题名：Near Optimal Behavior via Approximate State Abstraction。
- 正式会议页：<https://proceedings.mlr.press/v48/abel16.html>
- ref：`turn268006284view2`。
- 阅读范围：官方摘要与研究定位；正文没有借用未详读定理的具体常数。
- 支持：近似抽象可在特定定义和假设下权衡压缩与行为损失。
- 边界：任意embedding相似度没有自动获得此界限。

## S7 — 面向决策的主动实验（2014）

- 作者：Shervin Javdani、Yuxin Chen、Amin Karbasi、Andreas Krause、Drew Bagnell、Siddhartha Srinivasa。
- 题名：Near Optimal Bayesian Active Learning for Decision Making。
- 正式页：<https://proceedings.mlr.press/v33/javdani14.html>
- PDF：<https://proceedings.mlr.press/v33/javdani14.pdf>
- refs：`turn733924961view7`、`turn814463484view4`、`turn233516272view3`。
- 定位：摘要、§1–2，decision regions、hypotheses与tests。
- 支持：尽快收敛到同一决策区域，而非必将全部模型辨清；问题必须有结果语义。
- 边界：不是所有提问策略都具有论文算法的近最优保证。

## S8 — 有噪声且相关的实验（2017）

- 作者：Yuxin Chen、Hamed Hassani、Andreas Krause。
- 题名：Near-optimal Bayesian Active Learning with Correlated and Noisy Tests。
- 正式页：<https://proceedings.mlr.press/v54/chen17b.html>
- ref：`turn733924961view8`。
- 阅读范围：正式论文摘要及所述模型/算法条件。
- 支持：实验相关和噪声可使贪心不确定性减少启发式失效；必须建模结果依赖。
- 边界：正文建议没有声称Hegel已实现ECED或拥有其定理保证。

## S9 — DreamCoder（2020预印本／2021 PLDI）

- 作者：Kevin Ellis等。
- 详读版本：Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning，arXiv:2006.08381v1。
- 完整HTML：<https://arxiv.org/html/2006.08381v1>
- refs：`turn251834511view2`、`turn411941158view2`。
- 正式PLDI题名：DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning。
- 2021正式记录：<https://dspace.mit.edu/entities/publication/86d768ab-558f-4fb1-b92e-9288de643ccb>
- 正式记录 ref：`turn268006284view0`；PDF下载429，未假装已独立读到该下载。
- 定位：2020版本介绍、wake/sleep学习、held-out及memorization消融段落。
- 支持：程序库与搜索策略共同学习、组合抽象、实际新任务测试。
- 边界：正文不跨版本搬数字，不把文中特定枚举复杂度例子当Hegel普通基线复杂度。

## S10 — Stitch（2023）

- 作者：Matthew/Maddy Bowers、Theo X. Olausson、Lionel Wong、Gabriel Grand、Joshua B. Tenenbaum、Kevin Ellis、Armando Solar-Lezama（不同官方页面姓名显示不同）。
- 题名：Top-Down Synthesis for Library Learning。
- 正式POPL页：<https://popl23.sigplan.org/details/POPL-2023-popl-research-papers/41/Top-Down-Synthesis-For-Library-Learning>
- 原文HTML：<https://arxiv.org/html/2211.16605v2>
- refs：`turn814463484view3`、`turn427825720view1`、`turn693832847view6`。
- 定位：§3增量匹配/上界剪枝；§4式(8)压缩效用；§6评估指标。
- 支持：从程序语料合成共享抽象，重写压缩减抽象成本，anytime实现。
- 边界：训练语料压缩质量不自动成为未见研究效用或总成本优势。

## S11 — 范畴论一手教材（2018/2019）

- 作者：Brendan Fong、David I. Spivak。
- 题名：Seven Sketches in Compositionality: An Invitation to Applied Category Theory。
- 作者PDF：<https://dspivak.net/7Sketches.pdf>
- refs：`turn733924961view3`、`turn814463484view5`。
- 官方课程：<https://ocw.mit.edu/courses/18-s097-applied-category-theory-january-iap-2019/pages/lecture-videos-and-readings/>
- 版本：作者PDF更新时间2018-10-12；Cambridge教材2019。
- 定位：§3.5 (co)limits；§6.2.3 Definition 6.19、Examples 6.25/6.29；§6.2.4 finite colimits。
- 支持：给定图式/映射后的普遍性质；pushout按给定关系识别元素，可发生大量粘合。
- 边界：不提供无需选择正确物理映射的科学发现算法。

## S12 — Algebraic World Model（用户v1主输入；v2方法网页复核）

- 作者：Guancheng Pan、Qixin Tan、Yang Yuan、Andrew Chi-Chih Yao。
- 题名：Algebraic World Model: From Exploration to Proofs。
- 用户文件：`upload/2610.40503v1(1).pdf`，29页；既有详读 `analysis_mathrsi/awm_transfer.md`。
- v1 SHA256：`25984dbb8e6c80fee90fc9f9a8bbe6624b8c71d51a0dab4907cd795f060bcd8d`。
- v2方法页面：<https://arxiv.org/html/2610.40503v2>
- refs：`turn733924961view9`、`turn814463484view6`、`turn693832847view7`、`turn233516272view2`。
- 定位：§2.3、A.2–A.3；v1 pp.15–17结构变换，pp.24–25实际规则纠错。
- 支持：case/coproduct组织、任务保真与结构证明分开、固定诊断集拒绝保持；纠错与重组应区分。
- 边界：本次不逐页diff两版本，不搬未核数字；没有把colimit当自主抽象发现的充分机制。

## S13 — DeVoe 化学热力学

- 作者：Howard DeVoe。
- 题名：Thermodynamics and Chemistry，2nd ed., version 5，May 2014。
- 原教材大学镜像：<https://faculty.washington.edu/gdrobny/v5-screen.pdf>
- refs：`turn693832847view0`、`turn233516272view4`。
- 定位：印刷pp.356–358，§11.9，Eqs.11.9.7–11.9.9及紧随的误用反例。
- 可阅读的作者教材转排页：<https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/DeVoes_Thermodynamics_and_Chemistry/11%3A_Reactions_and_Other_Chemical_Processes/11.09%3A_Effects_of_Temperature_and_Pressure_on_Equilibrium_Position>
- 转排页 refs：`turn814463484view1`、`turn693832847view5`。
- 支持：Gibbs曲率、反应焓/体积与平衡响应；恒V惰性气体、局部微分与积分热差别。
- 边界：这些是指定约束下的公式，不是所有开放/远离平衡系统都存在标量负反馈。

## S14 — Faraday 原理的教材形式

- 机构：OpenStax，University Physics Volume 2，§13.1。
- URL：<https://openstax.org/books/university-physics-volume-2/pages/13-1-faradays-law>
- ref：`turn268006284view1`。
- 定位：磁通面积积分式13.1与感应电动势式13.2。
- 支持：通量含场、面积、方向；电动势随通量变化。
- 边界：BA只在合适场与几何条件简化成立。

## S15 — Lenz 方向规则

- 机构：OpenStax，University Physics Volume 2，§13.2。
- URL：<https://openstax.org/books/university-physics-volume-2/pages/13-2-lenzs-law>
- ref：`turn733924961view5`。
- 支持：反抗引起感应的磁通变化，非简单反抗磁场本身；能量解释。
- 边界：不能只测闭路电流符号便推出任意瞬态或开路状态的统一判据。

## S16 — RL 动态与储能

- 机构：OpenStax，University Physics Volume 2，§14.4。
- URL：<https://openstax.org/books/university-physics-volume-2/pages/14-4-rl-circuits>
- ref：`turn733924961view6`。
- 支持：电流动态依赖电阻、电感和初态，增长/衰减及自感。
- 正文中乘I得到能量账是清楚标注的初等推导，不是宣称原文已证明LeChatelier与Lenz等同。

## S17 — 跨物理系统的能量结构（2015）

- 作者：Arjan van der Schaft。
- 题名：Physical Network Systems and Model Reduction，Mathematical Control Theory II，Chapter 11。
- DOI：10.1007/978-3-319-21003-2_11。
- 作者大学存档：<https://pure.rug.nl/ws/portalfiles/portal/64854097/Schaft2015_Chapter_PhysicalNetworkSystemsAndModel.pdf>
- refs：`turn251834511view0`、`turn411941158view0`、`turn427825720view2`。
- 定位：§11.2.2结构抽象，§11.2.4详细平衡化学网络 Eqs.11.17–11.21，§11.2.6 RLC Eqs.11.24–11.25。
- 支持：有条件的化学和电路能量/耗散/互联描述；不同物理系统可以有公共可执行结构。
- 边界：非全部化学/电磁/有源系统的统一负反馈律；端口和条件需具体给出。

## 检索到但未用来扩张结论的材料

- Willems 1972 原论文官方作者站PDF可打开，但浏览提取为0行；未声称已全文核实。通用耗散思路由可读的一手 van der Schaft 章节支持。
- 搜索返回若干2025/2026神经组合性/科学发现综述，非当前理论机制所必需，未拿来制造“最新进展已经证明Hegel方案”的结论。
- 没有根据二手解读、博客或关键词命中为具体定理背书。

## 对主答复的引用建议

优先引用少量最有解释力的来源：SME作者项目（候选推论边界）、MAC/FAC（分阶段检索）、SMDP原文（行为/成本保持）、Javdani（面向决策的实验）、Stitch（可执行抽象成本）、AWM（保真与纠错）、DeVoe与Faraday（物理边界）。若正文采用能量结构替代，引用S17，并明确这是供Hegel测试的设计方向，不是现有系统成果。

