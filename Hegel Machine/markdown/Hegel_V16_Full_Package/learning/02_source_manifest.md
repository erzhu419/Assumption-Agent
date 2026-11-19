# V16 经验学习与实验设计：来源清单

报告日期为 **2026-11-19（Asia/Shanghai）**；检索与核对截至当地 00:15，对应 **2026-11-18 16:15 UTC**。仅使用原论文、会议论文集和作者/研究机构网页支持技术判断。没有独立复现作者实验，也没有审核这些外部仓库的全部实现。搜索发现而未读取正文的资料不作为本文结论依据；未检索到某结果不表示该结果不存在。

## A. 近期经验学习与 RSI

### LR01 — PAST-Bench

- 标题：*PAST-Bench: Benchmarking the Foundations of Recursive Self-Improvement in Personal Agents*。
- 固定版本：[arXiv:2608.04003v1](https://arxiv.org/html/2608.04003v1)，2026-08-04。
- 公开项目链接：[Gen-Verse/PAST-Bench](https://github.com/Gen-Verse/PAST-Bench)。本次未执行项目。
- 已读定位：§3.2 评测流程，§4.1 设置，Appendix D.6 成本表。尤其正文说明 persistence-off 不可访问该家族此前形成的状态；任务结果和机制证据分开。
- 证据边界：本次确认论文设计和作者结果陈述；不把其消融解释成“普通方法保留相同原始历史后仍无法重建”。

### LR02 — Recuris

- 标题：*Recursive Experiential–Working Memory Evolution for Long-Horizon Agent Harnesses*。
- 固定版本：[arXiv:2608.24876v1](https://arxiv.org/html/2608.24876v1)，2026-08-25。
- 公开项目链接：[Gen-Verse/Recuris](https://github.com/Gen-Verse/Recuris)。本次未执行项目。
- 已读定位：§2.2 的状态与调用，§2.3.1–2.3.4 的定位、局部修改、准入与固定外环；§2.3.5 的预算匹配重试对照。
- 证据边界：作者明确将失败定位作为修复决定而非严格因果识别。开发锚点的修改准入不能升级为对每个新任务的收益保证。

### LR03 — AIDE²

- 标题：*Recursive self-improvement of AI research agents*。
- 固定版本：[arXiv:2609.26457v1](https://arxiv.org/html/2609.26457v1)，2026-09-22；[元数据](https://arxiv.org/abs/2609.26457)。
- 已读定位：§2.1–2.2 的内外层预算和评分，§3.2 选择集与 §3.3 未见评测、§3.6 更高层改进的边界。
- 证据边界：成绩及递归修改次数为作者报告。主要借鉴不同层级的开发/选择/评价隔离，不据此推断 Hegel 必须修改研究器代码才算进步。

### LR04 — Memento-Skills

- 标题：*Memento-Skills: Let Agents Design Agents*。
- 固定版本：[arXiv:2603.18743v1 PDF](https://arxiv.org/pdf/2603.18743v1)，2026-03-19。
- 公开项目链接：[Memento-Teams/Memento-Skills](https://github.com/Memento-Teams/Memento-Skills)。本次未执行项目。
- 已读定位：§2.3，PDF 第 11–12 页的合成查询、InfoNCE 与一步 Q 解释；§3.1，第 15–16 页的训练/测试划分和跨任务迁移限制。
- 证据边界：HTML 转换存在空节，实质判断来自 PDF。技能匹配奖励不等于完整研究资源净收益；训练峰值和独立测试不能混用。

### LR05 — AgentCL

- 标题：*AgentCL: Toward Rigorous Evaluation of Continual Learning in Language Agents*。
- 固定版本：[arXiv:2606.02461v3](https://arxiv.org/html/2606.02461v3)。[元数据](https://arxiv.org/abs/2606.02461)：初稿 2026-06-01，本版 2026-09-26。
- 已读定位：任务关系与两阶段评估、plasticity/stability/generalization 定义，dependent/conventional/independent 流设计。
- 证据边界：其相关流是用于诊断的构造；不由有意设置的任务依赖直接推论任意现实分布收益。未独立重算成绩。

### LR06 — Continual-ARC

- 标题：*Continual-ARC: Measuring Skill Acquisition over a Working Lifetime*。
- 固定版本：[arXiv:2611.14505v3 PDF](https://arxiv.org/pdf/2611.14505v3)。[元数据](https://arxiv.org/abs/2611.14505)：初稿 2026-11-10，本版 2026-11-15。
- 已读定位：摘要、流与冻结测试协议、示例缓存基线与结果；Appendix A，PDF 第 21 页关于反事实 reset、生成器和 Compute 的说明。
- 证据边界：仍为预印本；作者报告的有限未见增益限定于被测任务、模型与 harness。论文明确记录 compute 但不将其折算计入主评分。不能用它宣称完整成本上某类学习普遍更优。

### LR07 — ACE（补充机制）

- 标题：*Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models*。
- 本次读取的固定正文：[arXiv:2510.04618v1](https://arxiv.org/html/2510.04618v1)，2025-10-06。
- [当前元数据](https://arxiv.org/abs/2510.04618)另核到 v3，2026-03-29；本文只借鉴 v1 的增量 playbook 更新思想，没有声称审核了最新版本所有实验。
- 证据边界：作为局部记忆维护的补充参考，不用于论证 Hegel 的价值置信界或当前目标收益。

### LR08 — MemSkill（补充机制）

- 标题：*MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents*。
- 固定版本：[arXiv:2602.02474v3](https://arxiv.org/html/2602.02474v3)。[元数据](https://arxiv.org/abs/2602.02474)：初稿 2026-02-02，本版 2026-11-06。
- 已读定位：controller/executor/designer 的方法定义、相关技能选择与技能演化。
- 证据边界：没有逐项审核其实验和实现；仅支持“记忆操作可以成为选择/改进对象”的方法观察，不支持确定性 LCB。

## B. 价值选择、离线评价和探针设计的基础

### LR09 — Options

- Sutton, Precup, Singh，*Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning*，Artificial Intelligence 112，1999。
- [论文 PDF，UCI 托管原文](https://ics.uci.edu/~dechter/courses/ics-295/winter-2018/papers/Sutton-Precup-Singh-AIJ99.pdf)；[作者官方论文页](https://web.eecs.umich.edu/~baveja/Options.html)。
- 已读定位：PDF 第 6–7 页，起始集、策略、终止条件，Markov 与 semi-Markov options。
- 证据边界：该形式化允许长动作，并不自动提供 option 的经验价值或保证学习效率。

### LR10 — CQL

- Kumar et al.，*Conservative Q-Learning for Offline Reinforcement Learning*，NeurIPS 2020。
- [官方会议论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf)。
- 已读定位：引言、§3，特别是 Theorem 3.2 附近对策略期望值下界和逐状态动作下界的区别。
- 证据边界：没有把定理条件迁移到 Hegel，也不声称 CQL 不存在可用保守变体；批评针对 V16 未提供相应条件的直接移用。

### LR11 — SPIBB

- Laroche, Trichelair, Tachet des Combes，*Safe Policy Improvement with Baseline Bootstrapping*，ICML 2019。
- [官方会议页](https://proceedings.mlr.press/v97/laroche19a.html)；[官方 PDF](https://proceedings.mlr.press/v97/laroche19a/laroche19a.pdf)。
- 已读定位：§2 基线与覆盖不足的状态动作约束；§3.2 的数据收集假设；结论对带保证版本与经验变体的区分。
- 证据边界：仅借鉴证据不足时约束偏离基线。不能把样本支持不足时回退解释成无需开销或每道题绝不变差。

### LR12 — Off-policy evaluation

- Wang, Agarwal, Dudík，*Optimal and Adaptive Off-policy Evaluation in Contextual Bandits*，ICML 2017。
- [官方会议 PDF](https://proceedings.mlr.press/v70/wang17a/wang17a.pdf)。
- 已读定位：引言与问题设定，直接法、重要性加权、双重稳健估计及风险讨论。
- 证据边界：用于说明只观察选择动作的反馈时需要的建模与覆盖前提。本文没有对 Hegel 账本实施 OPE，没有输出虚构的策略价值置信区间。

### LR13 — SATzilla

- Xu, Hutter, Hoos, Leyton-Brown，*SATzilla: Portfolio-based Algorithm Selection for SAT*，JAIR 32，2008。
- [论文 PDF，作者研究机构托管](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/08-JAIR-Satzilla.pdf)；[官方项目](https://www.cs.ubc.ca/labs/algorithms/Projects/SATzilla/)。
- 已读定位：§2 的离线构建与在线步骤；§3.3 特征成本、§3.5 预解、§3.6 备用求解器。
- 证据边界：借鉴完整计费和超时回退的算法选择设计，不把 SAT 的成绩或复杂度结论迁移为 Hegel 效果。

### LR14 — Adaptive submodularity

- Golovin, Krause，*Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization*，JAIR 42，2011。
- [固定 arXiv 正文 v5](https://arxiv.org/html/1003.3967v5)；[作者机构论文索引](https://research.google/pubs/adaptive-submodularity-theory-and-applications-in-active-learning-and-stochastic-optimization/)。
- 已读定位：§4 代价归一的贪心选择、§5 保证条件、§9 主动学习与噪声边界。
- 证据边界：不是对任意科学假设空间的信息增益保证。三次二值问答的 87.5% 示意是本文在明确均匀、均衡条件下的简单推导，不是该论文报告的 Hegel 实验。

## C. 项目自身证据

### HL01 — V16 上传方案

- 本地原文：`upload/hegel_v16_wormhole_synthesis_and_master_plan.md`。
- 已完整读取。其引用的本机 WSL 路径不是已读材料；本文没有假定那些链接的正文与摘要一致。
- 直接审查的陈述包括优势 LCB 公式、零税直解、复杂度与不可替代性、探针剪枝目标、匿名化排除先验、P0–P3 路线。

### HL02 — V15 独立复算

- `analysis_v15/results_protocol.md`。
- `analysis_v15/recomputed_v15.json`。
- `analysis_v15/recompute_v15.py`。
- 证据范围：保存回执的 MSE、状态、分母、成本和轨迹核对；并非重跑科学实验。本文重新读取既有诊断，并未重新执行 V15 全量研究。

## D. 不作为结论依据的检索结果

广搜还发现其他持续学习、技能与 RSI 预印本及综述。因本次未完成原文方法与实验核对，没有将它们当作“新成果已证实”的依据，也未把搜索结果数量用于评价哪个方向更先进。MathRSI、AWM、HippoRAG 的专项审计由其他工作分支负责，本文没有重复扩张这些结论。
