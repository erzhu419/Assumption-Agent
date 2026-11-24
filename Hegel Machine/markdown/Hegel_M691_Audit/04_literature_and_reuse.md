# 相关研究与已有能力复用

以下只讨论与本轮实际缺口直接有关的原始论文段落，未把另一个任务集上的结论当作本项目成败的证据。完整多尺度、潜状态和部分共同结构仍应保留；这里借鉴的是形成、消费、验证和资源分配的实验合同。

## LILO 让抽象来自真实程序并进入后继求解

[LILO 原文与方法](https://arxiv.org/html/2310.19791v4) 的 synthesis、compression、AutoDoc 组成闭环，抽象来自真实解程序。其消融表明匿名抽象甚至可能干扰模型使用；文档和名字帮助消费，但语义说明也可能出错。

对本项目的直接启示是：不能把真实 producer 替换成“函数数目×drive+state”再声称抽出了共同机制。保留复杂源程序，提取实际共享片段、生成准确接口和条件文档，再检查目标任务是否使用。解释应服务于准确绑定，不意味着压掉复杂物理过程。可借其冻结最终库、固定下游搜索设置的办法，隔离库本身价值与原求解模型的影响。

## Skill∞ 区分技能改善和改进方法本身的改善

[Skill∞ 原文](https://arxiv.org/html/2610.35434v1) 将任务技能和优化技能的 meta-skill 分开，通过执行轨迹、候选评估与独立审阅组织改进；附录把 activation 记为 Yes、No、Unknown。其独立审阅关注候选是否实质变化、可用并实际激活。论文同时承认部分过程依赖模型遵守指令，并非所有要求都有工具强制。

本项目可复用这种“已调用/未调用/未知”的诚实分栏，并把 source→binding→adoption→gain 串联。若要证明递归改进，应记录哪条新学改进策略进一步改变后继技能发现、验证或选择，再在新的目标上验证。新增函数、贴上 Skill∞ 名称或增加函数调用计数，不能单独提供这层证据。

## RSI Tax 要计入形成与被拒改动的成本

[The RSI Tax 原文](https://arxiv.org/html/2611.13524v1) 按连续任务流在反馈回流之前记分，并将改进过程中的验证、探索、写入成本与任务执行成本一起核算。后继是否检索到产物，是使用证据，不是正确性或收益的同义词；未被检索也不意味着该知识永久没有价值。

本轮比较应因此分清反题目标任务 tokens 和它继承的双轨 Wave-1 形成成本。“消除 RSI Tax”不符合目前实际账本。应缓存可复用的有效验证，减少无作用的重复检查，并保留丰富知识结构。不能把论文某些设置里简短笔记更省钱，推广成这里应放弃记忆核、代码技能或哲学树。

## 测试时计算应按任务分配

[Scaling LLM Test-Time Compute Optimally 原文](https://arxiv.org/html/2408.03314v1) 讨论固定计算预算下的生成与验证，指出顺序修订与并行多启动的适合比例随任务难度变化。

本项目下一步可测试公开证据触发第二轨、分歧驱动的额外探针和已有候选的局部修订。需要在固定总 tokens 与工具预算下比较，不能从同样 28 轮推断成本匹配。本轮“反题17、合题18”的结构提示存在优化资源配置的空间，尚不能直接据此硬编码只给 AMR 第二轨。

## 仓库已有能力的复用位置

| 所需能力 | 现有入口 | 本轮缺口 |
|---|---|---|
| 保留经过评价的真实程序 | `code_agent.retain_evaluated_python_model`、`archive_confirmed_python_model` | Frontier 应绑定实际 producer 和实测证据，不应取最后一个 .py 且填零误差 |
| 角色 单位 时间对齐 | `branch_reconciliation.align_semantics_roles_units_and_time` | 对齐真实输入/状态/读出与时间关系，而不是空 demo |
| 部分共同结构 | `anti_unify_program_family`、`extract_shared_base_subtree_from_sources` | 以真实局部语义为输入，保留未解释部分和竞争补全 |
| 已付证据下裁决 | `adjudicate_branches_on_paid_evidence` | 真实 task ABI、相同评价数据与来源版本需要贯通 |
| 前提撤回与失效传播 | `propagate_premise_retraction_invalidation`、真实 dispatch | 在实际 archive/forest 中改变资格，而不是只对临时空支持图调用 |
| 多尺度读出与探针 | `predict_via_pi_fold_mode`、`select_m5_discriminating_probe` | 明确 source retrospective 和 target causal cutoff，返回可消费模型与状态 |
| 存储 执行 计时 | `checkpoint_storage`、`experiment_runtime`、`parallel_prediction` | 用同一版本/增量/恢复合同，不再为新 runner 复制一套 |

历史查询记录随专项报告保留。这个项目当前更需要把成熟旧能力与真实 Frontier 对象连起来，再测边际收益；无需为了下一轮再命名并复制一套架构。
