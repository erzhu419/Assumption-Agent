# Hegel M5 计划与实现审计 63b940a03

审计对象：提交 63b940a03205d0f5b36bf2403d274d74709a17fe，版本工作总结称 V18.0 / M5 Final。审计日期：2026 年 11 月 20 日。对照前序提交 e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b。

## 一 总体判断

**新版计划总体成立，保留跨时刻动态、多尺度折叠、跨领域部分共同结构及软参数先验，都是合理选择。目前实现不足以支持“M5 已完整收官”的结论。**主要差距在观测、抽取、实验、修订和后继消费之间：若干部件已经实现，但主基准通过预置机制签名、固定源知识和按任务名执行的脚本跨过了这些部件。本次还复现了热启动确认失败不回退、时间原点不变性破坏、绑定条件默认全满足等具体功能错误。

这不是一次把所有既有成果重新判为无效的审计。提交内的数值拟合、训练与评估行分离、逐题回执、独立子进程加载，都有真实实现。六个文件 SHA256 全匹配，640 条回执与汇总完全对得上。需要收窄的是这些运行所能证明的能力，以及成本差的归因。

本报告的判据是新版计划自己写下的合同，不以是否采纳上轮建议为标准。无量纲软热启动、内部潜状态不强求人类命名、有限里程碑这三项保留决定均可接受。后续工作应接通既有能力，不应删去两条核心思路。

## 二 审阅范围与验证边界

先完整阅读修订后的 20 总计划、21 工作总结及 00/03/06/07 的变更，再逐项审阅本提交全部 27 个变更文件。代码范围包括新增 2,367 行 M5 主模块、wormhole_forest 的新增合同与算法、M4.4 序列化增量、运行脚本与全部新增测试；向前追踪了实际调用的 anti-unification、HippoRAG、VOI、scope dispatcher、M5.0 写回器和持久化接口。导入本提交的上轮审计资料作为历史证据复用，不把旧问题重新冒充新发现。

本轮复算提交中的六个受校验文件和全部 640 条主基准回执；重跑新增 test_m5_all_12_mathematical_and_redteam_controls，结果通过；另运行少量 API、真实目标拟合、快照删项、旧实例授权等定向控制。没有重新运行整个八臂 640 条主基准，也没有把报告声称的 51/51 历史测试写成本轮重新完成的测试。工作树源代码与提交证据未修改。

审计脚本和机器可读结果随附。强制探针返回值、禁用函数等控制只发生在审计进程内，用于检查调用依赖。数值能力、签名和暖冷成本的配对控制保留真实拟合；错误率口径控制另用固定预测桩隔离标签判定逻辑，不用于能力或成本统计。J19 删除对照使用相同数据与 RNG；其 44 nfev 是本小控制的结果，不与原 J17→J18→J19 连续消耗 RNG 后的 nfev 混比。

## 三 修订计划的评价

### 3.1 应保留的改进

| 设计选择 | 判断 | 对项目的意义 |
|---|---|---|
| 快动态经条件和范围验证后可进入基底 | 保留 | 共享性由条件化机制决定，慢尺度可作排序线索，不能成为唯一准入条件。 |
| 时间折叠允许常数、慢动力学、记忆核或潜状态摘要 | 保留 | 可以保留同现在而前史不同的影响，不必把粗粒化压成静态。 |
| 区分采样步长、内生时间、提取窗、预测时距与因果截止点 | 保留 | 让观察、回顾和预测各自有清楚的时间语义。 |
| 共同子结构同时支持因果链与 AND 前提 | 保留 | 跨领域部分共性能够逐层使用，不需要把并列背景伪装成因果先后。 |
| 任务风险 VOI 加 HEC 停止 | 保留 | 目标相关的实验价值比裸方差更适合指导有代价的探测。 |
| 反例可修订最小受影响上游子图 | 保留 | 可以保留源域有效知识，同时修订目标映射或失效前提。 |
| 软参数热启动与宽域回退 | 保留 | 合法地利用历史参数经验，同时允许异质目标逃出源区间。 |
| 匿名定量轨、语义展示轨与有限 J 流 | 保留 | 能分别检验能力、说明结构和检验持续复用，前提是复用同一执行链。 |

这些修订见 [计划裁决矩阵](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L12-L21)、[两条核心支柱](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L25-L113)。它们没有要求削弱表达力；当前最大风险是实现绕开了已经保留的表达力。

### 3.2 计划还需要写清的三个合同

**先验收紧指什么。**计划同时写候选空间包含关系与保留宽域兜底。二者可以兼容：优先搜索池可以收紧，完整可接受模型域仍保留。应把优先级、局部优化区间、硬排除与最终授权分别标记，避免将软先验意外实现成最终参数边界。当前热启动反例正说明这一区别会影响实际完成率。

**灰区外侧为何可授权。**从一个支持点和一个反例点得到半轴范围，需要已有单调性或连通作用域前提。若关系非单调，支持点以下也未必全安全。无需增加庞大证明系统；在现有 scope contract 中注明适用的单调性、已验证区间及条件即可。

**有限验收能声称什么。**四种子和封存任务中的零假确认是有价值的有限实测，不等于一般性零漏解、任意机制迁移或普遍 RSI。M5 完成可以被定义为有限合同闭合，但“同原始历史”“自动提炼”“同内容组织消融”“冷恢复消费所学知识”都应在该有限合同内真实发生。见 [实验及验收合同](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L117-L133)。

## 四 已经成立的结果

主基准确实调用 least_squares 和 RK4。拟合器只接收训练行；生成器检查训练/探针、确认、heldout 的输入键不重叠。最终 heldout 在一个候选已经通过确认之后才用于评估，没有发现它直接反馈进本次候选选择或拟合。需要区分：同一组 confirmation 被反复用于模型选择，所以它承担 validation 职能；独立 heldout 仍是合法的最终测试。

| 实验臂 | 域内通过 | OOS unknown | 回执假确认 | 候选数 | 回执 nfev | M | C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hegel Full | 72/72 | 8/8 | 0 | 152 | 2,952 | 1,096 | 1,164.1795 |
| Strong AllHistory | 72/72 | 8/8 | 0 | 408 | 9,056 | 1,920 | 2,076.6160 |
| Memoryless Static | 20/72 | 8/8 | 0 | 140 | 578 | 960 | 977.1580 |
| Blind Parameter Copy | 18/72 | 8/8 | 54 | 72 | 0* | 600 | 614.4520 |
| Forest Subtree PPR | 72/72 | 8/8 | 0 | 152 | 2,947 | 1,096 | 1,164.1245 |
| EqualAuth SingleNode | 72/72 | 8/8 | 0 | 156 | 3,747 | 1,124 | 1,197.6445 |
| EqualAuth Flat BM25 | 72/72 | 8/8 | 0 | 192 | 4,087 | 1,124 | 1,203.2665 |
| EqualAuth RawVariance | 72/72 | 8/8 | 0 | 152 | 2,940 | 1,180 | 1,243.7275 |

C 是项目定义的复合计量单位 C = M + 0.05K，不是实测运行秒数或货币。Full 和 Blind 的生命周期列另加常数 28。*Blind 的 nfev=0 存在漏记，详见第七节；此处忠实列出回执值，不代表认定其从未拟合。来源：[提交回执](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120/part2_track1_anonymous_receipts.jsonl)；本轮复算见 evidence_recomputed.json。

在组 II 的绝热任务上，Forest 与 Flat 的 M 均为 64，这项账面平权修订已经生效。后文指出的混杂来自其他策略差异，不否定这项修订本身。

## 五 阻碍 M5 完成验收的主要问题

### F01 主基准直接获得机制线索和隐藏状态

位置：[数据生成](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L988-L1080)、[任务签名](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1254-L1323)、[实际消费](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1692-L1743)。

数据生成器根据已知任务族直接填写 endpoint_matched_history_gap、endogenous_adaptation_candidate、mass_sum_drift、C1+C2 等签名。主求解器用它们检索、授权、选择首试模型、激活热先验和决定探针类别。这里没有从已购 pilot 观测计算这些签名的过程。变量名匿名不能移除这种由实验器提供的机制信息；本轮也没有据此声称发生了大模型预训练记忆泄漏，主实验本身没有在线 LLM 发现步骤。

记忆任务中的 mem_conv 是独立抽取的随机数，生成真值时用作隐藏状态 z0，拟合器又直接把同一数值作为 z0。tau_scale 同样由任务参数直接传给拟合。已知状态特征下的拟合是合法局部实验，但无法替代从原始历史估计潜状态和尺度的实验。

依赖控制把 extract_causal_temporal_regimes 与 predict_via_pi_fold_mode 同时换成一调用即抛错的函数，B_memory_kernel_fold 仍成功，确认 RMSE 0.000022、heldout RMSE 0.000074。这证明主基准不消费这两个新增入口。C_fast_but_transferable 还只是将 u 乘 0.004 再在静态公式中除回去，H 未随之变化，不能单独作为快时间动力学迁移的证据。

另一个控制固定所有数值行，只删预置签名：候选数从 1 变 5、nfev 从 8 变 117，但两次都通过；由于少买一个预设探针，C 反从 14.54825 降为 13.83700。因此签名帮助路由，但不能把删签名描述成必然失能，也不能假定现有追加采样对每个任务都是必要的。

应接入已有历史状态估计和观测特征入口，为每个用于决策的签名、潜状态、尺度保存合法观测来源；已知仪器标定可以公开，未知机制量应从付费前缀估计。保留现有机制已知条件作为部件基线即可。控制：temporal_runtime_controls.json、evidence_targeted_controls.json。

### F02 共享子树的自动抽取尚未产生实际执行结构

位置：[抽取函数](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L230-L263)、[预置源与子树](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1524-L1558)。

函数确实调用 anti_unify_programs，但 shared_nodes、shared_relations、角色 schema、开放槽、来源及参数样本均由调用者填写。反统一结果只保存为 repr 和槽位元数据，没有用于生成实际模型。

主源程序只有 7.5*u_norm 与 5.2*u_norm。它们的共同 AST 不能自行给出 C1∧C2、u→x→y、饱和项、平方根分支或外漏分支；这些内容在代码里另外写入。三条 source metadata 中也只有两棵传入的源 AST。整个 seed builder 直接注册 SUPPORTED_IN_SCOPE 节点和粗粒化证书，source_formation_cost=28.0 是情景常数，没有对应的源形成账本。

所以当前可保留的结论是“给定源知识库后的检索和参数拟合”；“同原始源历史出发的自动提炼与完整生命周期收益”尚未兑现。计划要求的源 A/B 自主形成不在这次主运行中。全候选失败后也只是退出固定八模型枚举，没有接回计划写明的开放组合发现。

修复应复用 branch_reconciliation 的真实已确认分支与参数家族接口，让源经历产物成为子树和候选生成的输入。无需另写一个新求解器。参见 [已有经确认的反统一提案](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py#L2072-L2094)与[已有观测驱动修订步骤](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py#L4038-L4087)。

### F03 选中的探针没有决定实际获得的观测

位置：[共享基底探针执行](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1745-L1770)、[慢反转探针执行](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1779-L1806)。

VOI 价值器真实运行，问题出在调用方。共同子树路径不论选 relevant、nuisance，甚至返回 HEC stop，都会把 p_open 的观测加入训练；选择 nuisance 只多记 3.5M。反转路径除 HEC stop 外固定加入 slow probe 和两条 extra_dynamic 行；选 nuisance 同样只额外记费。没有先执行 nuisance、读回观测、更新候选、再重新选择 slow probe 的过程。

| 同一子树任务的强制选择 | 实际训练输入 | nfev | heldout RMSE | M |
|---|---|---:|---:|---:|
| relevant probe | 6 行 | 8 | 0 | 14.0 |
| nuisance probe | 与上一行逐字一致 | 8 | 0 | 17.5 |
| HEC stop 且不选 probe | 与上一行逐字一致 | 8 | 0 | 14.0 |

反转任务的 slow 与 nuisance 两次也收到逐字相同的 8 行训练数据，nfev 均 7，M 分别 19.5 与 23。RawVariance 当前主要是同有效数据加一笔干扰探针费，无法检验错误实验选择如何影响信息、后验和后续规划。J02 虽名为 HEC_EarlyStop，提交回执 M=19.5，实际仍买了慢探针。

最小修复是让唯一 acquisition 接口只消费 selected_probe_id，未购买行对 solver 不可见；再复用已有候选更新与 VOI 继续规划。HEC 停止后仍可做必要目标确认，但不能继续无条件购买已决定停止的分歧探针。验证应检查实际 acquisition IDs 和训练行，不只检查选择器返回字符串。控制：temporal_runtime_controls.json。

### F04 联合流的新增知识与冷恢复结果缺少因果依赖

位置：[J 的预置共同结构](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1887-L1923)、[按任务名修订](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1987-L2005)、[冷恢复重造先验](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L2027-L2048)。

sub_joint_fast_slow 在 J01 之前用两棵手写 Program 和手填关系注册，实际求解器不引用它。J09 的修订发生在该任务任一 arm 求解之前，条件是精确 task_id，归因原因和新增分支也是固定字符串。J09/J10/J19 则根据预置 mass_sum_drift 签名直接选择库中已有的 LinearLeak 类。现有索引建图没有把 shared_base_subtrees 转为参与检索的节点或关系；mutation token 触发重建，不等于新增子树已被消费。

冷子进程确实启动、加载快照并校验摘要，但随后再次调用 _build_seed_forest_and_priors 重造参数 prior。快照有 7 个节点、2 个共享子树、1 条子树历史；raw_episode_archive、questions 和 node_history 都为空。这一快照支持结构保存测试，不能被描述为已保存整段在线学习经历。

| 对真实 post_j16 快照的定向控制 | 观察结果 |
|---|---|
| 删除 J09 新增 confirmed_external_sink_leak，再删除 sub_joint_fast_slow | 相同 J19 的完整 receipt 和拟合调用不变：1 候选、44 nfev、确认/heldout RMSE 均 0。 |
| 清空来源、证据依赖与多样性，再按生产冷启方式构建 prior | J17 仍用同一 warm box，完整 receipt 不变，9 nfev。 |
| 经公开 revision API 降权为 retrieve_candidate，保持摘要有效 | J17 仍执行同一 warm prior 与同一拟合。 |

这些控制没有证明数值预测错误；它们证明新增分支、恢复来源和子树权限尚未约束实际行为。普通 node 的授权检查在后面运行，不能补上前面已经直接使用的 subtree prior。

有限修复是在当前 J 流中，让源提炼和一次失败归因真实产生可执行分支；后续候选由所选子树开放槽/版本生成。先验应保存在版本绑定状态中，或由持久源估计重建；冷启动禁用固定 seed 知识重造。然后重做删新知识或回退版本的配对控制。控制：cold_subtree_controls_results.json。

### F05 两组成本比较均有未隔离的策略差异

位置：[Strong 的固定策略](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1648-L1681)、[森林计价与路由](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1683-L1776)、[固定形成费](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1553-L1558)。

Strong 拥有八种动态模型，确实比静态消融强；但它固定先买 5 pilot+4 extra+3 confirm，每行 2M，每题 24M。森林初始买 8 行，同一 dataset 行字典却按普通 1.5M、coarse 1M 计价。Strong 没有按已买数据决定是否补 extra，也不消费原始源历史形成先验。若这些单价代表不同仪器，应有不同采集过程、精度或授权证据；当前实现只是相同数据上的定价支路。

Strong−Full 的目标成本差为 912.4365C，其中 M 差 824，占 90.31%；0.05K 差 88.4365，占 9.69%。只做会计敏感性分析：保持 Strong 的 12 行和已记录拟合结果不变，按 Hegel 各族单价重计，M 从 1920 降为 1344。单价这一项即可解释 576C，即原差额的 63.13%。剩余 336.4365 仍混有固定采样量、模型顺序和冷启动政策。这不是重新跑过公平 Strong 的预测结果。

组 II 的粗通道 64M 平权确实成立，但在 C1+C2 签名命中的任务上，Flat/Single 同时失去 warm prior，并被强制追加两行数据；Flat 还被插入特定回退模型顺序。Flat−PPR 的 39.142C 中，28M（71.54%）来自采样费用差，剩余 11.142 仍混合了索引、先验和候选排序。它可以作为完整子树模块消融，尚不是同结论、同先验、同执行政策下的纯组织消融。

暖冷优化停止规则还不同：warm 的单次局部拟合结束后，只要训练误差过门槛就返回；cold 即使第一个局部解已达标，仍执行其余四个起点。同模型同五行数据的控制，warm 为 8 nfev，cold 为 10+17+11+15+9=62；第一个冷解 RMSE 已约 1.07e−9。该例同达标规则下可支持 8 对 10，不能把 8 对 62 全解释为先验带来的搜索难度下降。

建议保留现表并准确标注其策略；端到端表统一 acquisition、价格、停止和历史可用性，形成费从真实账本加总。同内容表把结论、参数先验、证书和候选分叉以平坦记录同样提供给 Flat，仅替换索引/导航。无需制造无限强对手或扩充任务族。复算和控制：evidence_recomputed.json、evidence_targeted_controls.json。

### F06 热启动确认失败后不会重试同模型宽域

位置：[训练阈值触发回退](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L391-L446)、[确认失败后进入下一模型](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1844-L1883)。

warm 训练 RMSE 小于 0.85*tolerance 时即返回；若后续目标 confirmation 失败，pipeline 直接尝试另一模型类。保留 wide_fallback_box 字段并没有保证这一分支会执行。

定向任务仍属于已支持的 C_shared_structure_hetero_params，真参数 vmax=7.8、rho=0.14、branch=0.32 均在 hard bounds 内；pilot 和 probe 的 s_grad 在 1 附近小幅变化，独立确认覆盖 0.25–2.2。两次使用同数据与 RNG。

| 指标 | 原 warm 流程 | 仅关闭 warm box |
|---|---:|---:|
| 正确 M6 类训练 RMSE | 0.03309146 | 3.63e−16 |
| 正确 M6 类确认 RMSE | 0.63387170 | 0 |
| 最终尝试模型数 | 8 | 1 |
| 最终状态 | scoped_unknown | confirmed_ok |
| heldout RMSE | 未提交 | 0 |
| 总 C | 23.64475 | 17.59025 |

这是具体的域内完成率损失，原流程仍诚实返回 unknown，不能称为一次假确认。应在确认反驳后先触发同模型 wide refit，必要时补辨识实验，再决定是否放弃结构。如果把失败确认点纳入拟合，就应另留新鲜确认。软先验应继续保留。控制：warm_pipeline_control.json。

### F07 联合绑定把部分相容误当作完整先验成立

位置：[绑定器](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L2216-L2314)、[调用方只检查 valid](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1730-L1743)。

空 satisfied_shared_node_ids 被解释为全部共享节点满足。对真实 seed 子树传空列表，返回两个条件 matched、独立证据 2、active relations 3、valid=True。只给 C1、不满足 C2 也仍 valid=True；实际 consumer 没有核验所需 AND 是否成立。当前主基准外层会检查 C1/C2 两个签名，并硬填两个 satisfied IDs；这些 API 负控制不能被解读为原 640 条回执已发生缺 C2 的假确认。

seed schema 的 required roles 没有列入两条因果边共同的 x_pool。删除这个中间绑定后，active_relation_count 从 3 变 1，却仍 valid=True。不给 edge_bindings 也可仅凭角色名字激活关系。时间尺度目前只验证 normalized_tau_ratio 是正的有限数，没有校验各边读出、时间锚点及尺度映射。

部分匹配本应保留，不能为修这个问题改成整棵树全有或全无。应分别返回 matched_subgraph、成立的条件和可激活先验；仅 C1 成立时使用 C1 对应候选，C1∧C2 的强化先验要有两者证据。空条件不能代表全满足，缺失中间角色不能被完整链路消费。控制：contracts_controls_results.json。

### F08 灰区守卫尚未接回实际旧写回合同

位置：[新增灰区 guard](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L428-L473)、[仍生成 midpoint 的旧写回器](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_0_closure_and_continual_sublation.py#L439-L461)、[J 的固定公式二分](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L2009-L2017)。

新增 guard 能拦带 gray_zone_bounds 的手工合同，C10 这一局部测试有效。但旧 OutcomeDrivenForestConsolidator 仍将 max_supported 与 min_refuted 的中点写成 b<=theta_emp，没有生成 gray_zone_bounds。J 的二分也只是对已知 surge 公式算一个数，没有购买实际边界观测或回写合同。

用本提交保存的真实 v17.7 快照及 W4 frozen instance，经当前 ForestScopeDispatcher 查询 s=4、b=1.8，仍得到 dispatched_ok、local_solver_guidance、fit_called=False；合同为 b<=1.85，没有 gray_zone_bounds。这个点处于先前支持界 1.45 与反例界 2.25 之间，因此旧授权缺口仍在。

此控制只证明范围授权未修复，没有绑定相同实例参数的目标真值，不能据旧 C10 的解析遗漏项推断这次实际预测必错。有限修复是把支持界、反例界、灰区和实测探针结果接入原写回器及旧快照迁移。控制：legacy_gray_control_results.json。

## 六 需要修复的局部数学与持久化问题

### F09 集合多样性仍可被同源换皮提高

Canonical 排序确实解决了插入顺序问题；单一 lineage 的三个克隆也确实仍为 1。但多 lineage 情况下，代码先对同 lineage 的 surface_cues 取并集，再计算不同 lineage 的 Jaccard 相似度。A、B 原本都有 cue same，D_eff=1.1111111111；增加一个 lineage 仍是 A、cue 只是 renamed 的记录后，独立 lineage 数仍为 2，D_eff 却升为 1.9047619048，跨过 1.35 阈值，实际把 skeleton_only 改为 soft_warm_start。见 [cue 合并及谱计算](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L2148-L2205)与[参数 pooling 门槛](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L2332-L2342)。

另一个接口问题是显式 pairwise_surface_sim 不保证正半定。对三 lineage，矩阵 [[1,t,t],[t,1,0],[t,0,1]] 的每项都在允许范围内；t=1 时 D_eff=5.2857142857，超过 lineage 数 3；t=0.87 时为 −12.5420944559；t=1.25/√2 时 K+0.25I 奇异并抛错。把成对相似度各自截进 [0,1] 不等于得到合法 Gram 矩阵。

应在 lineage 层使用稳定机制特征或规范化表示，同源纯重命名不改变对外距离；显式相似度要通过具有 PSD 保证的 kernel 或明确的相关矩阵校验。仅裁剪最终 D_eff 不能修复奇异反演。这些反例未出现在原 640 条回执使用的固定来源集合中，不能据此宣布原表全部污染。控制：contracts_controls_results.json。

### F10 记忆折叠不保持时间原点平移不变性

位置：[初态余波及斜率分支](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L186-L212)。初态衰减使用 exp(−t0/τ)，没有减去该初态所属的历史起点。

控制为 dz/dt=−z、初态 z=1、前缀时间 0/0.1/0.2、预测时距 H=0.3。原预测 0.60653066 正确；保持全部观测值、参数、Δt 和 H 不变，仅将时间戳与 t0 全部加 100，预测变成 0，正确值仍应为 0.60653066。应记录初态时间锚点，并明确最后观测时刻早于 t0 时的推进规则。

C03 正好走斜率重构分支，w_slope=1，卷积估计完全被覆盖；其通过不能验证记忆卷积及初态项。该测试报告的同现在 x(t0)=0 也不准确：x=0 在初始化 t=0，真正决策 t0=0.02 时两轨迹的 x 已一正一负。

指数记忆核使用增广潜状态与二维状态空间 realization 是合理选择，不需要为了可解释性改回显式常数。需要修的是时间锚点，并分别覆盖卷积、斜率估计两条路径。控制：temporal_runtime_controls.json。

### F11 子树摘要没有覆盖影响来源治理的完整状态

位置：[子树摘要字段](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1806-L1833)、[快照验证范围](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L236-L270)。当前摘要覆盖子树结构和 permission，但不包含 supporting_sources、diversity_metrics、probe_signatures、唯一证据依赖和 stale 标志；共享子树历史也没有逐版本验证。

控制清空来源、将 D_eff 改为一百万、替换 probe_signatures，当前 verifier 仍报告零 mismatch；把历史版本的 shared_relations 清空也仍是零 mismatch。这是摘要覆盖范围的局限，不是一次攻击证据。应在 F04 真正接通持久 prior 时，把决定它的来源估计、版本和撤回状态一并校验，保留科学内容身份与来源身份的清晰分层。

## 七 测试与统计口径

| 问题 | 实际行为 | 对当前结果的准确影响 |
|---|---|---|
| 假确认阈值不一致 | Static/Strong 用 heldout>1.5tol；Hegel 用 >tol；strict_pass 都用 tol。 | 相同 .055 误差、tol=.045 时会得到不同假确认标签。但原表 Strong/Static 没有此边界样本，因此不会改变本批零计数。 |
| Blind 的 nfev 漏记 | 非 C/非 reversal 路径真实拟合 M1，回执仍写 0。 | 控制实际 3、报告 0；K 已含拟合费，不能说整份 C 漏计。 |
| Static 候选数漏记 | 第二个候选通过也返回 candidate_count=1。 | 低估部分成功任务的尝试次数；不是加总脚本错误。 |
| RNG 派生不稳定 | 用含字符串的 hash((seed,arm)) 派生随机种子。 | 新进程 hash salt 不同，固定四个任务种子不足以精确重现 nfev。Full 与 PPR 的微小差异不应解释为算法差异。 |
| J 对 Strong 的测试断言失效 | 条件表达式 else 分支直接返回 baseline 正成本。 | 当前实际 Hegel 成本确实较低，但该断言没有检验成本大小。 |
| confirm 被重复用于选型 | 同三行确认给多个候选打分并决定继续。 | 应称 validation；最终 heldout 仍独立，不能笼统称训练或 heldout 泄漏。 |

对应源码：[Static 与 Strong 统计](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1623-L1680)、[Blind nfev](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1598-L1614)、[Hegel 统计](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1860-L1873)、[RNG 派生](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L2118-L2128)、[条件表达式断言](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/tests/test_v18_0_m5_temporal_multiscale_and_base_subtree.py#L95-L100)。控制见 evidence_targeted_controls.json。建议统一阈值、统计实际执行次数、稳定派生 seed，并让比较双方共享停止规则和可比随机数。

12 项控制的通过应逐项解释。C01、C04、C06、C07 等确实覆盖有意义的局部行为；但 C09 用固定差值和手写原因调用修订 API，C10 对已知解析式二分，C11 检查被调用的 anti-unification 槽位，C12 直接构造误差并填 zero_false_confirmation/false_confirmation_count 常量。它们不能各自替代主路径中的自动归因、真实边界采集、子树消费和最终确认门禁。C03 的替代分支问题已在 F10 说明。

这就是为什么新增测试全过仍可与本次反例同时成立：它们验证的范围不同。将本轮几个消费依赖控制并入原有测试即可，不必堆积大量只镜像实现的测试。

## 八 语义孪生展示的正确定位

三个展示仍有价值，应保留。其数据和结构来自手设同构 ODE、解析函数、角色字典及预设城市分支，适合说明时间归一化、反转和部分共性；它们不构成真实电磁/化学实验或真实城市经济机制验证。见 [三个语义案例](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L2203-L2324)。

“同引擎”需要收窄：案例复用了同一模块中的模拟、VOI、绑定和拟合部件，但没有全部调用主匿名 pipeline；twin 2 选择探针后没有实际执行探针来锁定分支，twin 3 另做五个解析样本的拟合，没有由所选一针直接驱动分叉写回。因此可称语义化的合成同构示例，不宜写成三条完整自主研究轨迹。

如果要证明完全相同执行机制，最简方式是把匿名任务的同一结构化 execution trace 通过独立展示映射改名；改变展示标签后，观测、动作、拟合、确认和修订回执保持一致。无需放弃长沙/南昌、楞次/勒夏特列等直观表达。

## 九 有限修复顺序

### 9.1 优先级

| 优先级 | 范围 | 为什么现在处理 |
|---|---|---|
| P1 完成声明阻断 | F01 原始观测入口、F02 源提炼、F03 探针执行、F04 J 与冷恢复消费、F05 公平对照 | 这些缺口直接影响新版计划核心能力与节省幅度是否得到验证。 |
| P1 已复现行为或旧修复未闭合 | F06 warm 确认失败回退；F08 灰区写回/恢复 | 已有域内任务损失，以及真实旧合同仍授权灰区点。 |
| P2 当前正向夹具遮住的合同与数学问题 | F07 部分绑定状态、F09 D_eff、F10 时间原点、F11 摘要范围 | 反例已明确；应随相关执行链接通修复，无需因它们推翻全部架构。 |
| P2 计量与叙述 | 第七节统计/seed/断言、第八节展示范围 | 小改即可提高复现性和报告准确度。 |

### 9.2 四个有边界的工作包

**包 A：让动作真正控制信息。**给当前固定任务接统一 acquisition；未买数据不可见，selected_probe_id 决定实际返回；pilot/prefix 产生机制特征和状态估计。复用 probing.py、既有 POMDP 与前提审计接口，不新造研究引擎。先以一条记忆任务、一条反转任务和一条共同基底任务完成接线。

**包 B：让历史真正产生并控制后继。**沿用已确认 Program、branch_reconciliation 与原结果驱动写回器，产生真实 source AST、共享关系、参数估计和开放分支；把其实际引用写入候选/探针/确认回执。当前 J 流只需一条明确受新知识帮助的后继，做保留/删除新知识对照，观察候选、动作、成本或完成率的可解释变化。冷恢复允许加载算法与静态 schema，但学得参数和修订必须来自恢复状态。

**包 C：补齐已承诺的回退与范围合同。**确认反驳 warm 后先试同类 wide；部分匹配保留候选作用，但不得激活未证实的 AND 先验；修记忆初态时间原点；灰区元数据接回生产写回与旧状态恢复。多样性和持久化校验随同来源先验链修复。保留潜状态、宽域兜底、快动态和多类型关系。

**包 D：在原有限评测上重新归因。**统一测量价格、优化停止、错误阈值和 RNG；组 I 真实加总源形成及维护账本，组 II 提供同结论、同 prior、同证书的平坦表示。现有 80 题/臂与 J 足以作为下一轮验收主体；在接线正确前增加种子或场景不会解决这些问题。

### 9.3 可作为收官门槛的最小证据

1. 每个用于路由的特征、潜状态和尺度都有可追溯的已购观测或公开标定来源。
2. 强制选择不同探针时，实际 acquisition 和训练数据按选择改变；HEC stop 不再无条件买分歧探针。
3. 共享结构与参数来自已确认源经历；修改源结构会按规则改变派生子树，而不只是改变 repr 元数据。
4. 至少一项在线修订经过实际失败、竞争归因和确认；明确受其帮助的后继能够显示该项知识的因果作用。
5. 新进程不靠固定参数 bundle 重造所学内容；撤回、降权和版本变化能改变对应 prior 的消费。
6. 本报告的 warm 域内失败、灰区授权、空条件绑定和时间平移控制闭合；比较使用相同计量与停止规则。

这些门槛不要求给每个潜变量一个人类名字，也不要求系统在所有领域永不失败。它们检验的是新版计划中最有价值的结构是否真正进入了观察、行动与学习过程。

## 十 历史能力复用定位

本轮已查阅 reuse hub，并通过 historical --query memory / POMDP 查询既有路线。最相关的存量能力包括：language.Program 的历史/累积算子；premise_dependency_audit 与 world_structure_commitments 的马尔可夫充分性、隐状态和耦合前提；branch_reconciliation 的角色/单位/时间对齐、反统一家族及付费证据修订；probing.value_forecast_experiments；M5.0 的结果驱动写回与旧 scope dispatcher。

因此主要诊断应写作新 M5 runner 的输入与消费未接通，以及个别合同实现错误。不能把已有算法改名成下一轮新能力，也不能因为新 runner 未采用它就说整个项目没有该能力。修复后应在现有 historical_capabilities 注册入口、证据和剩余边界，继续使用共享计量与存储接口。

## 十一 附件及复现

| 文件 | 内容 |
|---|---|
| AUDIT_REPORT_ZH.md | 本完整审计正文。 |
| recompute_evidence.py / evidence_recomputed.json | 六文件校验、640 回执及八臂/任务族成本复算。 |
| evidence_targeted_controls.py / evidence_targeted_controls.json | 签名删项、冷暖停止、计数、错误阈值和 RNG 控制。 |
| temporal_runtime_controls.py / temporal_runtime_controls.json | 探针消费依赖、禁用时间入口、时间原点控制。 |
| warm_pipeline_control.py / warm_pipeline_control.json | 真正域内 warm 确认失败不触发 wide 的配对控制。 |
| contracts_controls.py / contracts_controls_results.json | 绑定、D_eff、摘要及子树消费控制。 |
| cold_subtree_controls.py / cold_subtree_controls_results.json | 真实 J 快照的删分支、来源撤回和合法降权控制。 |
| legacy_gray_control.py / legacy_gray_control_results.json | 当前 dispatcher 消费旧真实 W4 合同的灰区授权控制。 |
| REVIEW_MANIFEST.json / README.md / SHA256SUMS.txt | 固定版本、审阅范围、复现环境、运行办法与附件校验。 |

将仓库固定到本提交，使用项目 Python 依赖。所有控制默认只读源码与证据；新结果输出到审计目录，避免覆盖提交内实验资料。可移植脚本支持 HEGEL_AUDIT_REPO_ROOT，读 v18 快照的脚本另支持 HEGEL_AUDIT_EVIDENCE_DIR。具体命令见 README.md。

**当前阶段建议定位：完成了多尺度与共享子树的多个可执行部件，并给出预置机制知识条件下的集成演示；M5 的自动源形成、观测驱动的联合成长及同内容净收益验收仍需上述有限闭合。**以这一定义继续推进，可以保留全部表达力，也能让下一轮结果清楚地回答哲学树本身贡献了什么。

