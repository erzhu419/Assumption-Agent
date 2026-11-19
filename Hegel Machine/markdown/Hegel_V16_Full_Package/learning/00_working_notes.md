# V16 经验学习、选择器与离线价值估计：中间研究笔记

报告日期为2026-11-19（Asia/Shanghai）；检索实际时间为2026-11-18 16:02以后UTC，对应当地11月19日00:02以后。本文是过程笔记，不是最终实验结论。

## 已读用户计划

`upload/hegel_v16_wormhole_synthesis_and_master_plan.md`。

优先待审主张：

1. “V15已彻底跨过真假性审查”与六个尚存证据/身份/干预缺口相冲突，应承认实际形成/采用并区分剩余证据边界。
2. “底层复杂度证明58k直解时任何抽象检索都会抵消”没有复杂度推导。某两题的当前实现成本不构成所有选择器/提问树的下界。
3. `LCB(Q_option)-Qhat_direct`通常不是优势的LCB；必须同时控制基线误差，或直接估计成对优势。
4. 确定性可重放不产生统计保守保证；旧账本未见动作缺乏支持，且V15相同目标的K/Ord历史不同，不是同状态的两个option反事实。
5. “否则零税直解”忽略已支出的pilot、检索、绑定、gate评估和回退。至多证明额外收费有上限、或者pilot是双方共同已付成本。
6. 2–3探针剪90%必须定义候选语义集合、先验/质量与oracle保留；计数剪枝容易通过重复AST和错误删真解做高。3个平衡二值探针在均匀8叶下只对应87.5%缩减，并非一般90%保证；多值probe可更多但必须计实际观测/测量。
7. 应保留中等难度和easy负迁移检查，困难不是价值出现的必要条件。高维扩张可能同时扩大角色映射、探针和验证成本，且强普通必须允许因子化、主动实验、同知识摘要/缓存。

## 第一轮检索来源（均需后续读正文，不把摘要当复现）

- PAST-Bench, arXiv:2608.04003v1, 2026-08-04：匹配持久化开/关、任务成功与机制证据分别评价，适合Hegel。
- Recuris, arXiv:2608.24876v1, 2026-08-25：working-state驱动skill选择、局部更新与validation gate；不等于Hegel具有相同收益。
- AIDE², arXiv:2609.26457v1, 2026-09-22：自身代码迭代、隐藏选择评价、持出任务；与知识形成复用不同，不应成为Hegel必须满足的新门槛。
- Memento-Skills, arXiv:2603.18743v1, 2026-03-19：技能read/write和InfoNCE router的one-step-Q解释；需检查真实跨任务transfer，不能把相似性分数当Q的置信界。
- ACE, arXiv:2510.04618v1：delta context/playbook，不必增加新模型训练即可试经验策略积累。
- CQL, NeurIPS2020；SPIBB, ICML2019：悲观估计及基线回退均有数据覆盖/模型/样本条件，不能仅借名称给任意启发式赋保证。
- Dudík et al. 2011/2014 DR policy evaluation：若只观察日志策略选过动作，需要propensity/support或明确模型假设。
- SATzilla, JAIR2008：算法选择要给特征计算收费，并有timeout/backup。
- Golovin & Krause, JAIR2011 adaptive submodularity：贪心信息选择的保证依赖具体性质，不能推出固定probe数。

## 近期追加检索（需读官方正文）

- Continual-ARC arXiv:2611.14505（后核元数据：初稿2026-11-10，v3为11月15日）：同任务新实例与未见任务分开、付费反馈，摘要报告多数harness不胜简单示例缓存；非常适合防止只讲正结果。尚未复现。
- Memory-Centric Continual Learning arXiv:2610.29357（2026-10-21）：框架/评价提议，不是可据以移植的已验证收益算法。
- Continual Learning Exam arXiv:2610.33956（2026-10-25）：专业任务梯级，不能直接把单次反馈修订收益等同跨任务新知识收益。
- AgentCL arXiv:2606.02461（检索提示有9月修订）：区分dependent/conventional/independent任务流，提示应该明确迁移分布及负迁移。
- MemSkill arXiv:2602.02474v3（检索提示11月6修订）：记忆操作controller/executor/evolution，待核正文。

## 本次验证边界

当前仅原始文献阅读与数学/协议分析；没有复现作者模型实验、没有付费API、没有更改Hegel仓库。已完成`01_learning_and_evaluation_review.md`与`02_source_manifest.md`，其中逐项标作者报告、本次推论及固定版本阅读边界。Dudík早期DR论文未扩为全文审查；最终基础参考采用已读的Wang/Agarwal/Dudík 2017官方OPE论文。
