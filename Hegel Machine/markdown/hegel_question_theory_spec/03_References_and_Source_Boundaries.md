# 参考来源、版本与引用边界

检索日期：2026-10-20。以下链接是原始论文、正式出版页或作者材料。这里列出的是本次设计所使用的文献，不是关于全部自动科学发现的系统综述，也不意味着各工作已经解决黑格尔机的完整目标。

2026 年文献按所核查的 arXiv 版本标记；本文没有独立复现外部论文结果。数值目标、算法计划与接口建议不是这些文献的实证结论。

## A. 问题形成、假设与研究控制

### R01 — Question Asking as Program Generation

Anselm Rothe, Brenden M. Lake, Todd M. Gureckis. NeurIPS 2017.  
https://arxiv.org/abs/1711.06351

用途：将问题表示为可能世界上返回答案的组合程序。限制：有给定语法的受限游戏研究；不是任意科学问题形成系统。

### R02 — Inquisitive Semantics: A New Notion of Meaning

Ivano Ciardelli, Jeroen Groenendijk, Floris Roelofsen. Language and Linguistics Compass, 2013, 7(9):459–476. DOI:10.1111/lnc3.12037.  
作者出版目录：https://www.florisroelofsen.com/  
出版链接：https://doi.org/10.1111/lnc3.12037

用途：区分陈述的信息与问题要求解决的事项。本轮能读取作者出版目录；DOI直开存在访问失败。未基于未读章节提出更强技术主张。

### R03 — Goal-Driven Autonomy in a Navy Strategy Simulation

Matthew Molineaux, Matthew Klenk, David W. Aha. AAAI 2010, 1548–1554. DOI:10.1609/aaai.v24i1.7576.  
https://ojs.aaai.org/index.php/AAAI/article/view/7576

用途：异常、解释、目标生成与目标管理分开；目标本身可以被研究。限制：所验证环境与已有规划结构有限。

### R04 — A Theory of Diagnosis from First Principles

Raymond Reiter. Artificial Intelligence, 32(1):57–95, 1987. DOI:10.1016/0004-3702(87)90062-2.  
https://www.sciencedirect.com/science/article/pii/0004370287900622

用途：从系统描述与冲突观测提出竞争诊断。限制：适用逻辑和系统描述是前提；数值拟合失败不是自动的逻辑不可满足证明。

### R05 — An Assumption-Based TMS

Johan de Kleer. Artificial Intelligence, 28(2):127–162, 1986. DOI:10.1016/0004-3702(86)90080-9.  
https://www.sciencedirect.com/science/article/pii/0004370286900809

用途：不同假设环境中的支持维护和上下文切换。限制：支持管理器不是候选发现器。

### R06 — Structure-Mapping: A Theoretical Framework for Analogy

Dedre Gentner. Cognitive Science, 7(2):155–170, 1983. DOI:10.1207/s15516709cog0702_3.  
https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog0702_3

用途：以关系结构和角色对应区别于表面对象相似。限制：提出类比不等于验证目标系统服从该结构。

## B. 语言／代码假设、主动实验与理论对象学习

### R07 — Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning

Wasu Top Piriyakulkij 等. NeurIPS 2024.  
https://proceedings.neurips.cc/paper_files/paper/2024/hash/5f1b350fc0c2affd56f465faa36be343-Abstract-Conference.html

用途：语言规则、可执行预测、概率更新和主动实验的组合。不要从有限候选直接推断完整开放假设空间已被覆盖。

### R08 — Induction and Inquiry via Probabilistic Reasoning over Language and Code

Wasu Top Piriyakulkij, Sam Acquaviva, Cassidy Langenfeld, Joshua Tenenbaum, Kevin Ellis. arXiv:2609.01815v1, 2026-09-01，预印本。  
https://arxiv.org/html/2609.01815v1

用途：顺序更新语言／代码候选，并用候选之间的可检验差异支持提问。理论谱的长期跨域生长仍是另一个目标。

### R09 — Model Discovery Agent: LLM-assisted Bayesian Experiment Design for Data-efficient Discovery of Mechanistic World Models

Kevin Murphy. arXiv:2608.09696v5, 2026-09-27，预印本。  
https://arxiv.org/html/2608.09696v5

用途：区分结构、参数和隐轨迹推断；失配触发结构扩展；实验设计与推断分工。尤其参考A.8对有限池、近似证据及截断的限制。不可将受限池内归一化权重称为全模型空间的可信概率。

### R10 — Automated Hypothesis Validation with Agentic Sequential Falsifications

Kexin Huang, Ying Jin, Ryan Li, Michael Y. Li, Emmanuel Candès, Jure Leskovec. ICML/PMLR 267:25372–25437, 2025。  
https://proceedings.mlr.press/v267/huang25n.html

用途：高层假设对应的可执行子检验与序贯验证。限制：统计有效性取决于检验的数据与方法前提；不是自由文本真值器。

### R11 — LILO: Learning Interpretable Libraries by Compressing and Documenting Code

Gabriel Grand 等. ICLR 2024；读取 arXiv:2310.19791v4。  
https://arxiv.org/html/2310.19791

用途：程序搜索、压缩、命名和文档化共同形成可用库。自然语言名称帮助使用抽象，但不是抽象真实性证据。

### R12 — Top-Down Synthesis for Library Learning

Matthew Bowers 等（按所读取arXiv出版记录）. POPL 2023；arXiv:2211.16605v2，Stitch。  
https://arxiv.org/abs/2211.16605

用途：从给定程序语料中寻找共享函数抽象。限制：原始代码结构的共享不能单独证明跨物理系统的共同规律。

### R13 — Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information

Willie Neiswanger, Ke Alexander Wang, Stefano Ermon. ICML/PMLR 139:8005–8015, 2021.  
https://proceedings.mlr.press/v139/neiswanger21a.html

用途：围绕真正所求的算法输出而不是全部世界细节选择信息。应用概率式信息价值前要有合理的预测分布。

## C. 收缩、预测状态与行为距离

### R14 — Policy Gradient Methods for Reinforcement Learning with Function Approximation

Richard S. Sutton 等. NeurIPS 1999.  
https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html

用途：说明RL还可以直接对策略优化；“所有RL都只做收缩证明”不成立。

### R15 — Metrics for Finite Markov Decision Processes

Norm Ferns, Prakash Panangaden, Doina Precup. UAI 2004；arXiv存档1207.4114。  
https://arxiv.org/abs/1207.4114

相关连续状态扩展：Bisimulation Metrics for Continuous Markov Decision Processes, SIAM Journal on Computing 2011, DOI:10.1137/10080484X。  
https://doi.org/10.1137/10080484X

用途：奖励／转移语义支持行为度量；收缩用于其固定点分析。原始任务、动作映射、模型估计和不确定性仍需处理。

### R16 — Predictive Representations of State

Michael L. Littman 等. NeurIPS 2001，按原出版页核查题名与年份。  
https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html

用途：以多步行动条件未来观测描述状态。有限测试集合的等价不能无限推广到所有未来。

## D. 先验、正则与函数空间

### R17 — A General Framework for Updating Belief Distributions

Pier Giovanni Bissiri, Chris C. Holmes, Stephen G. Walker. JRSS Series B, 78(5):1103–1130, 2016. DOI:10.1111/rssb.12158；arXiv:1306.6430。  
https://arxiv.org/abs/1306.6430  
https://doi.org/10.1111/rssb.12158

用途：损失与KL的变分更新、广义Bayes。温度与相关性需要解释，不是任意损失直接得到校准真值概率。

### R18 — The Bayesian Learning Rule

Mohammad Emtiyaz Khan, Håvard Rue. JMLR, 24(281):1–46, 2023.  
https://jmlr.org/papers/v24/22-0291.html

用途：概率推断与优化的联系。本文的MAP、Gibbs及先验转换推导是明确限定下的数学说明。

### R19 — Gaussian Processes and Reproducing Kernel Hilbert Spaces: Connections and Equivalences

Motonobu Kanagawa 等. arXiv:2506.17366, 2025.  
https://arxiv.org/abs/2506.17366v3

用途：核正则、GP和RKHS的联系及区别。不能给无限维函数空间随意套普通Lebesgue密度。

## E. 逻辑假设生成与失败约束

### R20 — Meta-interpretive Learning of Higher-order Dyadic Datalog: Predicate Invention Revisited

Stephen H. Muggleton, Dianhuan Lin, Alireza Tamaddoni-Nezhad. Machine Learning 100(1):49–73, 2015. DOI:10.1007/s10994-014-5471-y。  
作者机构原始记录：https://openresearch.surrey.ac.uk/esploro/outputs/journalArticle/Meta-interpretive-learning-of-higher-order-dyadic-datalog/99537123102346

用途：在明确逻辑语法与元规则下生成递归程序和新谓词。限制：元规则及逻辑片段是重要先验，不能当成完全无先验发现。

### R21 — Learning Programs by Learning from Failures

Andrew Cropper, Rolf Morel. Machine Learning 110:801–856, 2021. DOI:10.1007/s10994-020-05934-z。  
https://link.springer.com/article/10.1007/s10994-020-05934-z  
https://arxiv.org/abs/2005.02259

用途：generate–test–constrain；失败用于约束后续假设空间，而不只丢掉一个候选。严格的剪枝结论依赖适用逻辑与数据设定。

注意：该文的 **Popper ILP系统** 与R10的 **2025 POPPER统计假设验证Agent** 是不同工作，不能把两者作者、算法和保证混用。

## F. 本项目与原始构想

固定仓库链接基础：

https://github.com/erzhu419/Assumption-Agent/tree/468256bfc7e9d0c6077f29a0475e7e065d24c6d6

### G1 — 分支快照

本轮GitHub接口确认 `468256bfc7e9d0c6077f29a0475e7e065d24c6d6`；本地未检出／执行整个项目，也未修改仓库。

### G2 — 当前假设与子研究说明

`Hegel Machine/docs/hegel_machine_hypothesis_research_guide_20261020.md`

本轮读取当前说明，确认已有对象、provider=None内层边界、既有触发与科学判同缺口。

### G3 — 当前实验与反复问题总览

`Hegel Machine/docs/hegel_machine_executable_evidence_repair_20261013.md`

该名称含10月13日，但当前版本已更新到10月20日。不能把它与下面旧诊断的终态混同。当前成功、正式负对照与开发未决均按各自范围保留。

### G4 — 原始在线失败诊断

上传文件《黑格尔机：在线研究为何未形成合格科学解释》，2026-10-13，229行。

零填充碰撞、参数域未消费和宿主事后2×2分析是历史诊断。它不证明当前代码仍缺全部相关接口，也不代表模型自主发现了宿主修复。

### G5 — 关键源码

- `Hegel Machine/src/hegel_machine/cognition/online_inquiry_llm.py`：本轮重新读取问题身份函数。
- `common_archive_inquiry.py`：前一轮同一快照核对的父返回与live研究调度。
- `structural_proposal.py`、`structural_inquiry.py`、`world_relation_research.py`：同一快照下的结构提案、Program、候选银行与条件证据语义。
- `inquiry_revision.py`、`executable_evidence.py`、`representation_diagnostics.py`：由当前指南定位的既有接入点。
- `anonymous_method_learning.py`、`temporal_method_learning.py`、`historical_capabilities.py`：当前运行、方法和历史能力入口。

本稿没有声称重新逐行审计所有这些文件；施工映射来自本轮直接读取、同一快照此前已核对的源码及当前指南。Codex开始前须再次核对工作树。

### O1 — 《黑格尔机》原始构想

上传MD区分固定语言内研究与语言扩展，要求经验、假设、元先验、尺度和反例积累。早期完成度打分和fixture机制示范不能作为本轮新发现证据。

### O2 — 《广义对位关系》v0.2

原文件明确文末2026-07-30 v0.2优先。其第8节限定勒夏特列—楞次为有范围的远域类比，不无条件推出扰动幅度降低。此次通过Files工具重新定位并读取了该条款。

### O3 — 《黑格尔机和泛函分析》及Meta prior

提供尺度、可观测性、扩张／简化和支持前提的原始问题。本文数学澄清与现有事实分开，未把早期GPT推测当作已证明定理。

## G. 引用和复现范围

- 仓库中的结果数字只按已有报告叙述，不是本次重跑。
- 外部论文只支持其自己范围内的机制，不自动支持本方案组合有效。
- 配套检查脚本仅检验简单数学／语义例子，不执行Hegel项目，也不构成自主发现证据。
- 本次没有调用Luna、产生真实实验标签、修改旧协议、增加账户预算或向GitHub提交文件。
