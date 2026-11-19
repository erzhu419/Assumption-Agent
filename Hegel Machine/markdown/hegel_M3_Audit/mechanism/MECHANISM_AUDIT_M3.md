# M3 机制专项审计：612d8eea5

固定提交：`612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`。本报告只判断该提交的 M3 实现及证据，不把分支之后的变化混入结论。代码位置均相对于 `Hegel Machine/`。

## 1. 结论与审计范围

M3 确有工程进展：可观测查询从付费 pilot 产生，五个森林实验臂共享冻结节点、契约和原始信息，真正执行 Python 拟合、前缀隔离确认与 heldout 评价。此前“扁平基线丢掉元数据”“关横边顺便禁掉跨域读取”等强混杂已明显收缩。因此不能把整个 M3 简化成硬编码答案或假实验。

但当前最强的科学结论需要降一级：**实验证明的是这个固定合成任务族内，一套“特异性种子 + 同机理连通分量汇总 + 图传播 + 契约筛选 + 两候选预算”的组合相对若干具体检索器有收益，尚未识别 PPR 随机游走本身的增益。** 对提交中唯一持久化的 eval04 冻结图做因子控制后，完全关闭随机游走、保留同一分量汇总，23 个可检索目标的授权 Recall@2 仍为 23/23，Recall@1 反而从 16/23 提升到 21/23。两题真实 target consumer 对照进一步确认该现象能传递到实际模型和测量成本。

另有三个应在扩大 M3 结论前收口的实现问题：M3 存在但不执行 `program_steps`；ScaleMap 的“经验校准”目前是收费后返回常数；HExA 近失错误库虽被拼入列表，却在默认两个试用槽内全部不可达。AST 检查的新严格化还与 M3 旧准入条件冲突，导致 9/36 个 `mismatch:` 节点仍为 SUPPORTED；这些是合法代码被规范化器误标后继续放行，不能等同于九个错误物理模型。

审计已全文阅读新索引、新 M3 universe、M3 测试、报告 12，追踪 `PhilosophicalForest`、授权、链组合、实际 target consumer；遵守仓库 AGENTS 先读 reuse hub，并执行 `cognition historical --query retrieval --compact`，该查询未返回可复用项。没有修改仓库，没有调用 LLM，没有重复完整 M3。运行范围只有：一个已提交冻结图上的廉价排名控制，及 T13/T20 的 9 次目标控制。完整 M3 复放与总账核对由另一专项负责。

## 2. 已经成立的部分

### 2.1 目标真值没有被直接送入本轮查询

`extract_m3_observable_signatures`（`src/hegel_machine/cognition/m3_scale_universe.py:904–1114`）依据 `lab.observations`、校准读出与公开的 ScaleMap 构造 signature。目标的 `ground_truth_schema_for_audit_log_only` 在 `solve_target` / `_continue_after_pilot` 只是最终日志字段（2123、2176、2340、2367 行），没有进入排序或拟合分支。

所以目前没有证据支持“目标 oracle 标签直接泄漏到检索”。手工设计有限特征算子、有限函数语法本身也不构成泄漏；它们是受控检索实验可以合理使用的任务定义。

不过 `validate_observable_query_signatures`（索引文件 41–53 行）只验证 `sig:` 前缀，不能在接口层证明来源是观测。把 `truth:foo` 改写成 `sig:truth:foo` 就不再触发它。更准确的承诺应是“当前调用链由可观测提取器生成查询”，而非“字符串守卫证明了零泄漏”。若以后开放外部作者，应传带来源和观测摘要的结构化 signature 回执。

### 2.2 五个森林臂的共享信息有实际落实

`clone_with_mode`（M3 universe 1658–1669 行）深复制同一森林、同一 Stage-1 gate 与同一 learned scale certificates；`solve_target` / `_continue_after_pilot`（2117–2370 行）是共用执行路径。Flat 拷贝节点与 incident wormhole/path 元数据（索引 402–431 行），VerticalPPR 也保留 incident 元数据，只关闭横边传播和由横边生成的分量汇总。

这足以认可“五个森林臂之间较强的等信息比较”。但报告 12 第 7 行将七个臂统称“严格单因子”不准确：另外两个 ordinary/HExA 基线的历史组织、检索和求解流程也不同。报告第 14 行限定到五个森林克隆时则基本对应实现。

### 2.3 PPR 计算本身符合概率传播公式

索引 209–217 行按每行边权总和归一化；293–304 行构造特异性 reset；306–321 行执行

\[
p_{t+1}=(1-\alpha)r+\alpha P^T p_t.
\]

因此在合法非负参数下质量守恒。共现特征只用于 reset，没有作为中间游走顶点；这确实避免了高频 `arity` 特征变成游走中的跨树捷径。图权重和终端子节点自保留权重仍属手工设计的排序策略，但不影响此数学判断。

### 2.4 确认和产物冻结比旧实现可靠

M3 在新确认观测前保存 `fit_prefix`，随后使用该前缀验证、冻结候选（2268–2353 行），并以 target、claim、node version、model SHA、confirmation digest 组成回执（2294–2321 行）。这确实避免了先把确认数据加入训练再声称独立确认的旧问题。域外任务可以返回 `scoped_unknown`，也不应要求这一步突然变成无限语法发现器。

## 3. 核心因子控制：目前不需要随机游走也能得到完整 Top-2

### 3.1 方法和可复核范围

脚本：`probe_retrieval_factorization.py`。

结果：`retrieval_factorization_result.json`。

读取提交的 `docs/evidence/v17_0_m3_hipporag_wormhole_scale_20261119/wormhole_forest_graph.json`，保留全部节点、边、契约与版本。该文件只保存最后一个评测簇 `m3_eval_04`，因此以下结论仅覆盖该图。用同一 seed `7027404` 重建付费 pilot 和查询，T14 在检索前被 ScaleMap 阻挡，剩 23 个查询。原 PPR 的 signatures、raw top-4、authorized top-4 与提交证据逐项比对，**0 mismatch**。

所有 ranker 均不读取 truth。truth 仅用于事后计分。Flat 分量汇总控制复用完全相同的逐节点 BM25 公式、IDF 和平均文长，再对 PPR 使用的同一 live-wormhole component 求和；这是排名因子控制，不宣称该辅助实现的墙钟代表一个优化后的生产 ranker。

| 同一 eval04 冻结图上的变体 | 授权 R@1 | 授权 R@2 | 授权 R@3 | 授权 R@4 |
|---|---:|---:|---:|---:|
| 原 PPR，alpha=.8，原分量汇总 | 16/23 | 23/23 | 23/23 | 23/23 |
| **alpha=0，保留原分量汇总** | **21/23** | **23/23** | 23/23 | 23/23 |
| 原 walk，取消分量汇总 | 15/23 | 21/23 | 21/23 | 23/23 |
| 关闭横向 walk，保留原分量汇总 | 17/23 | 22/23 | 22/23 | 23/23 |
| 原 Flat BM25 | 18/23 | 21/23 | **23/23** | 23/23 |
| Flat BM25，对相同分量求和 | 17/23 | **23/23** | 23/23 | 23/23 |
| 原 Flat，仅反向重标识 node IDs | 20/23 | 22/23 | 23/23 | 23/23 |
| 原 PPR，仅反向重标识 node IDs | 17/23 | 23/23 | 23/23 | 23/23 |

### 3.2 为什么这个控制改变解释

PPR 的最终分数不是单节点概率。索引 330–331 行将同一连通分量成员概率相加，355–365 行再按 schema 去重。Flat 则对每个文档独立打分，然后只保留每个 schema 得分最高的文档（473–505 行）。因此原比较同时包含：reset 权重方式、传播、同机理跨实例汇总及去重代表选择。

横向同机理汇总是合理的“多来源支持”策略，可以保留，也可能正是目前最有价值的部分。上述结果**不否认森林结构或虫洞等价类有用**；它们准确否定的是“当前 perfect Top-2 必须依赖随机游走”的解释。报告 12 第 47–49 行所谓“牺牲 Top-1 换 Top-2 的真实图扩散权衡”尚未成立：alpha=0 同时保住 Top-2 并显著改善 Top-1。

同样，NoWormholes 同时取消横向转移与横边导出的分量汇总（索引 163–207 行）。它是一个合理的横向机制整体消融，却不能独立量化 walk 的贡献。把原收益称为“特异性检索与跨实例汇总的组合收益”更符合现有证据。

### 3.3 名称不进入 query，但名称仍会影响平局排序

节点 ID 在 M3 universe 1740–1742 行由 `surface_domain` 组成。Flat 的最终平局按 `node_id` 字典序（索引 495 行），PPR 也把它作为末级平局规则（356 行）。三个 distractor 的 domain 使用 `aa_` 前缀（M3 universe 1331–1333 行）。我们不推测这种命名的意图，但仅反向重新标识所有节点并同步边端点后，Flat R@2 从 21/23 变成 22/23；T13 的正确 schema 从授权第 3 变第 2。

这不是目标真值泄漏的证据，是**语义无关标识符造成的排序敏感性**。在等分较多、试用窗口只有两个的场景，应使用内容稳定的平局规则，或者将同分候选作为同一预算层处理，并把平局敏感性记录出来。

## 4. 将排名差额连接到实际目标执行

脚本：`probe_target_controls.py`。

结果：`target_controls_result.json`。

只重跑原 Flat R@2 未命中的两个目标 T13/T20，比较原 PPR、alpha=0 同 pooling、Flat 两槽、Flat 三槽；另在 T13 删除全部 36 个节点的 `program_steps`。总共 9 次 target 调用，无 source 重训。

完整冻结 Stage-1 gate 的数值记录没有持久化，因此本控制**明确固定 admission=true**，依据是原两题、原 PPR/Flat 均已进入 recall loop。其余全部调用原 `solve_target`。对两题原两臂检查模型 SHA、selected schema、strict pass、heldout RMSE、观测数、measurement、compute、composite、是否经 recall 确认，**0 mismatch**。所以它是有明确门控范围的事后机制对照，不是新一轮全协议评测。

| 目标与变体 | 严格通过 | measurement | logical compute | composite | 错误候选数 | 经 recall 确认 |
|---|---|---:|---:|---:|---:|---|
| T13 原 PPR | 是 | 30 | 29.30 | 31.4650 | 0 | 是 |
| T13 alpha=0 同 pooling | 是 | 30 | 27.83 | 31.3915 | 0 | 是 |
| T13 Flat 两槽 | 是 | 45 | 91.88 | 49.5940 | 2 | 否，转全量 |
| T13 Flat 三槽 | 是 | 30 | 49.38 | 32.4690 | 2 | 是 |
| T13 PPR 删除所有 program_steps | 是 | 30 | 29.30 | 31.4650 | 0 | 是 |
| T20 原 PPR | 是 | 25 | 27.30 | 26.3650 | 0 | 是 |
| T20 alpha=0 同 pooling | 是 | 25 | 25.83 | 26.2915 | 0 | 是 |
| T20 Flat 两槽 | 是 | 25 | 37.38 | 26.8690 | 1 | 是 |
| T20 Flat 三槽 | 是 | 25 | 37.38 | 26.8690 | 1 | 是 |

T13 中，alpha=0、Flat 三槽与原 PPR 产出**同一个 model SHA**，heldout RMSE 都为 `.001192`。Flat 原先被挤到第 3 的正确 schema，只要多容纳一个便宜拟合，就能消除这一题全部 15 measurement 的差额，仍保留两次无效拟合带来的 compute 劣势。源码 2257 行直接截断 `authorized_candidates[:max_recalled_trials]`，所以训练残差立即失败的候选也会耗掉候选槽。接下来更直接的改进是“拟合计算预算与新观测预算分开”，并非先调一组复杂的自适应 alpha。

T20 必须单独解释：真实生成家族 `rational_monod_1d` 在原 Flat 排第 3，但第 2 的 `congestion_rollover_1d` 已在目标容差内通过。因此 schema Recall miss **并非真实任务失败，也不必然多买观测**。alpha=0 在此选中了 rational，heldout RMSE `.001808`；原 PPR/Flat 选 rollover，RMSE `.003875`；全部严格通过。报告应同时区分“生成家族识别”与“目标容差内任务成功”，不要混成一个指标。

这些是诊断性反例，不表示 alpha=0 在未测分布普遍优于 PPR，也不表示 Flat 三槽必定更好。它们已足以说明原强归因需要补上极小的必要控制。

## 5. M3 的程序对象仍未成为被执行的方法

源节点注册了三步 `program_steps`（1770–1774 行），修订节点也一样（1961–1965 行）。但实际 recall 消费者 2257–2273 行只从 `claim_contract.conclusion_or_action.schema_tag` 读标签，再调用固定的 `run_m3_workspace_synthesizer`，之后走固定确认流程。整个 M3 target 路径不解释 `program_steps`。

删除全部节点 program_steps 的 T13 控制与原 PPR 的模型字节、SHA、观测、严格质量和全部逻辑成本完全相同，验证它们在这里仍是描述性字段。M2.5 为可执行方法语义做的修复不能自动算到新 M3 consumer 头上。

这不会抹掉“历史检索能缩小候选函数族”的收益，但它限制了哲学/RSI 主张：当前迁移的主要是函数族候选和排除规则，尚未证明历史方法程序被抽象、重组再执行。最小修补是复用已有受限程序解释器，使允许的 op 真正决定校准、角色绑定、拟合和确认；加一条方法步骤扰动能改变执行计划或被明确拒绝的控制即可，不需要重造一个通用 agent。

## 6. 图形成、误差与授权还存在语义缺口

### 6.1 AST 的新拒绝语义未被 M3 准入消费

`run_m3_workspace_synthesizer` 将 `canonicalize_python_model_ast` 的结果展开返回（769–779 行）；源形成只检查 `strict_pass`、digest 非空和 schema 非空（1712 行），随后将所有节点标为 `SUPPORTED_IN_SCOPE`（1759–1767 行）。`mismatch:...` 是非空字符串，所以照常入图和索引。

提交的 eval04 图有 9/36 个 mismatch digest：linear 两个、exp 两个、log 一个、cobb 两个、inverse-linear 一个、sqrt 一个。这些没有形成它们本应拥有的同 AST 横边；而且带参数差异的 mismatch digest 会把同一数学族拆成不同身份。

与闭环审计交叉验证：S01/S02 原 seed 实际复放均 strict pass，digest 与提交逐字相同。原因是纯数据流规范化剥离外层 affine 时连 core 的正数比例/截距一起剥离，而 annotation 侧保留相应符号形式。故应修正 canonicalizer 的两侧规则，并让 M3 对 canonicalization status 做显式分流：行为有效但身份未核对的候选可留作 local evidence，不能直接享受已核对 AST 等价类的语义。不要把它们误报成九个伪科学模型。

### 6.2 当前 hop budget 是拟合残差，不是实际映射的传播界

同 AST 横边的误差取两源 heldout/confirmation RMSE 的最大值，Lipschitz 写为 1（1788–1808 行）。特殊 monod→rollover 桥先把 monod 拟合到 rollover 的付费观测，再取 `train_rmse` 作为 hop budget（1813–1855 行），Lipschitz 同样写 1。

这比完全拍常数的桥边有进步：桥拟合真的执行，数据是已购买的源观测。但是：源模型各自拟合残差、近似桥的训练 RMSE、以及某个已定义 observation/parameter map 的误差上界，是不同对象。当前 `role_and_observation_map` 主要存 schema/digest/来源标记，没有执行参数输运或验证复合映射。固定 L=1 也没有给出对应变换和范数的依据。

`compose_wormhole_chain` 本身的公式（`wormhole_forest.py:387–419`）是正确的误差传播模板；它不能把未经建立的预算自动变成保证。此外，预算超过容差表示“不足以继承该层保证”，不是证明候选物理上无效。该 helper 的 docstring 和结果本就允许降到 bounded_trial，M3 2200–2225 行却在其特殊分支直接跳过候选。这可作为节省预算的策略，但报告应称“保守预算筛选”，不应称经证明的错误候选剪枝。

### 6.3 本轮多跳路径没有来自非直接命中节点的 BFS 发现

索引 `_shortest_wormhole_path_from_direct_hits` 在候选属于直接 signature 命中时，直接返回第一个 incident edge 或手工 `chain_parent_wormhole`（230–246 行）；只有非直接命中才到 BFS（251–265 行）。由于共有的 arity signature，eval04 全部 **129/129** 个返回候选已在直接命中集合，**0** 个候选需要 BFS。

实际概率传播仍发生，不能因此说“没有图计算”。但这批结果并没有演示从一个只有间接关系、无法直接 cue 命中的节点发现复合路径。T13 的二跳检查还有显式 `high_load_rollover_detected && schema == rational_monod_1d` 分支，并按 `bridge_from_schema` 找预设桥（2200–2218 行）。目标的 rollover 标志来自 pilot，避免了旧版目标标签路由；桥的候选选择与两跳组合仍是该机理对的手写适配。更窄且准确的结果是：这个已指定的近似桥在目标容差下可以被合约筛掉。

### 6.4 ScaleMap 的“经验校准”当前没有实验数据

M3 universe 1376–1379 行：`scale_probe_fn` 先计费 2 observation、10 measurement、4 compute，然后直接返回 `mean_flux_diff=.0012, peak_crest_diff=.84`。函数没有采样、oracle、微观轨迹、window average 或误差计算。

2026–2045 行据此生成 `verified_discarded_in_scope`，随后 `_effective_scale_map`（2076–2086 行）按操作名 `window_average` 全局合并。此处是**固定 fixture 被当成经验认证**，比“文档表述稍夸大”更具体。五个共享森林臂得到相同 fixture，所以它不单独使 PPR 比 Flat 获得更多信息；但它不能支撑 learned ScaleMap certificate、跨目标通用遗忘变量许可等主张。

最小修复可以很小：复用实际微观/宏观 source，产生可复核 rows 和差异统计，证书绑定观测协议、窗口、变量范围、任务目标、容差及数据摘要。若本阶段不想扩工作量，先把它改名为预设测试 fixture，并将经验认证从完成清单移出。无需因此废弃整个 ScaleMap gate。

### 6.5 反例已经实测，但修订仍部分由预设家族路由

近失对 incumbent 实际拟合，记录 witness RMSE（1870–1895 行），比旧的“零实验反例”有所改进。但修订准入（1905 行）未检查 witness 是否超过容差；三条 exclusion predicate 由修订后的 schema 名直接选择（1908–1915 行）；`zero_limit_reduction_to_parent` 被作为成立事实注册（1968–1975 行），没有对应极限变换检验。双时间尺度 exclusion 更是由“库里存在 bimodal schema”触发（2003–2024 行）。

当前这四个预设近失可以作为受控正例；一般化为 CEGAR 或哲学自我修订接口前，应要求真实冲突才标 refutation，分开记录“新候选有效”和“它保留/推广父契约”两种证据。此处不要求提前实现无限概念发现。

## 7. ordinary / HExA 基线的结论范围

`M3StrongOrdinaryAllPriors` 确实保存全部通过的历史 schema，修好了“只保留多数类先验”的限制（2419–2428 行）。但它只按 arity 返回完整 schema 集，不用观测 shape 排序；复制到 target 的 shared Python 文件没有被执行。因此它是“全历史候选池 + 无条件多族拟合”的基线，而不是已完整实现条件方法复用的最强 ordinary scientist。与它比较能说明调度价值，不能单独证明图结构不可替代。

HExA 的情况更具体。2465–2466 行按出现次数排序 skill_bank；2495–2496 行先列全部 skill，再追加 mistake；2507 行只试前两个。依据提交的 eval04 source 回执重建：

| arity | 基础技能数 | 近失写入的 mistake | 默认两个槽实际候选 |
|---|---:|---|---|
| 1 | 8 | 无 | congestion_rollover、exp_washout |
| 2 | 9 | liebig_capped、underdamped_ringing、haldane_shunt | bimodal_relaxation、cobb_douglas |
| 3 | 2 | state_memory | damped_driven_with_leak、tri_resource_liebig |

也就是说，所有四个近失产生的新 mistake schema 都排在两个槽之后，在这套基准的默认路径**没有任何一个能作为错误库候选被试用**。这是旧问题的部分修补：代码现在读取 mistake_bank，但实际执行仍未享用错误库。只需让错误见证/shape 条件参与路由，或分配独立 mistake 试用槽；不应把现有差额归为 HExA 类机制本身的失败。

## 8. K=36、独立性与测试覆盖的准确表述

已提交 eval04 图：36 个节点、32 个 tree、23 个 schema、27 个 cue、179 条 observation links、4 条纵边、11 条横边。32 个源产生 32 棵树，4 个近失在原树中增加子节点。项目要求 K≥32 时，树数门槛确实达到；但 K=36 更准确指 source/near-miss episodes 与节点数，不是 36 棵独立树，也不是 36 个独立机理。

当前有多个科学生境名称，但数学规律来自固定的 23 家族手写语法，源方程及参数跨 cluster 基本固定；target seed 主要改变噪声与一个 `.95–1.05` 的公用系数（1382 行起）。四个随机种子可以是独立噪声重复，但不是四套独立图拓扑、函数语法或新科学世界。它们支持这个任务族内的稳定性，不支持对通用开放 RSI 分布的强外推。只测一个 K 也不能估计随 K 增长的扩展曲线。

现有 M3 测试有三组：字符串前缀 guard；手工构图的传播与概率守恒正例；真实 36 episode 形成、若干 gate/OOS/置换案例。它们对实现回归有用，但没有测试 alpha=0、分量汇总因子、平局标识符、program_steps 执行、ScaleMap 数据来源、AST status 准入或 mistake 可达性。“测试通过”不能自动补齐这些科学归因。没必要把所有潜在问题变成大规模测试矩阵，优先加入本报告已给出失败见证的少数控制即可。

## 9. 最小修复与下一步决策

1. **先把已有结果说准。** 保留同信息五臂收益、域外诚实和确认隔离结果；将“纯图扩散因果效应”改为完整检索组合的收益；说明 K 的口径、单一任务族、每簇均值与随机种子范围。把 ScaleMap 固定 fixture 单列。
2. **补最少因子控制。** 使用已形成的冻结图保留 alpha=0 同 pooling、无 pooling 的 walk、Flat 同 component 汇总；不需要调大网格。记录确认通过的功能等价候选，防止把 schema Recall miss 当任务失败。T13 单题增加第三个便宜拟合已经是有判别力的预算控制。
3. **修接口错接。** M3 执行已有程序解释器；canonicalizer 两侧一致，M3 显式消费 status；HExA mistake 能到达实际试用槽。每项一条针对性行为控制即可。
4. **把证书与启发式分开。** residual 可用于 edge ranking，未经实证/推导的 hop budget 不应升级成 guarantee；ScaleMap 固定值可当测试输入，不能当 learned empirical certificate。未获保证时可以保留 bounded trial，不必硬性宣判候选无效。
5. **随后再谈扩规模。** 等上述最小收口完成，再加一套结构不同的冻结任务族或两个 K 截点，检验 pooling/graph 是否在机制增加时持续获益。当前无需全面重写、无限语法、全行业 benchmark 或先清空所有旧架构。

这轮最值得保留的发现是：历史抽象的组织和许可筛选已能在一个透明的有限世界里改变实际科研成本。下一步要明确究竟是跨实例支持汇总、随机游走、候选预算还是方法程序带来变化；已有证据足以让这个问题用很小的补丁和控制被回答。

## 附：本专项可复核产物

- `probe_retrieval_factorization.py`：只读重建 eval04 图、查询与八个排名控制。
- `retrieval_factorization_result.json`：每题完整排名、汇总、重建一致性、直接命中统计和图概况。
- `probe_target_controls.py`：两个目标共九次原 target consumer 控制，注明固定 admission 的边界。
- `target_controls_result.json`：提交、评价、成本、诊断、观测坐标与原件核对。
- `target_control_workspaces/`：上述实际 Python 模型、拟合输入、确认产物；用于核对 SHA 和数据前缀。

AST 误标的原 source 重放证据见并行专项 `closure/m3_ast_rejection_results.json`，本专项未重复其运行。
