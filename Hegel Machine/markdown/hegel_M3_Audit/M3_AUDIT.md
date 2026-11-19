# Hegel Machine M3 审计与 M4 路线建议

审计固定提交：`612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`。比较基线：`9115f9adf9dad1f52f106a2ab14babe8b68243dd`。审计日期：2026 年 11 月 19 日。

**总体判断：M3 的受控实验结果可以复现，等信息检索对照和确认流程有实质进步；目前足以支持“历史知识的组织与许可筛选，在这套有限任务族里可以降低科研成本”。工作总结中的三项更强结论尚未成立：V16.9 已完成非预写模板的开放作者、M3 的净收益已被归因于 PPR 随机游走、K 从 7 到 36 已证明一般规模反转。**

建议保留现有森林、虫洞、版本化契约、有限语法基准和全部历史证据。下一步先完成一轮范围有限的 M3.1：修确认上下文、AST 身份与消费者接线，补检索因子控制，统一计费和基线消费；随后开展小规模、真正由反馈驱动的开放作者合流实验。现在直接扩大到无预置机理的大规模真实科学数据，会让作者能力、知识组织、预算策略和测量协议同时变化，难以定位成败原因。

本报告区分三类证据：原提交事实及完整重放；本次事后诊断性控制；后续设计建议。诊断控制不会被当作新的预注册主实验。

## 1. 原实验的哪些结果已经核实

### 1.1 完整重放与收据核对

| 验证项 | V16.9 M2.5 | V17.0 M3 |
| --- | ---: | ---: |
| 原始 source 收据 | 160 | 512 |
| 原始主实验 target 收据 | 360 | 672 |
| 原始 ledger 收据 | 538 | 1,252 |
| 本次完整重放主任务严格通过 | 360 / 360 | 672 / 672 |
| 主任务假确认 | 0 | 0 |
| 额外实验 | 开放修订目标 24 / 24 通过 | 8 / 8 域外控制返回 scoped_unknown |
| 关键成对比较及 DEV 汇总 | 与提交精确一致 | 与提交精确一致 |
| 本次完整重放总墙钟 | 195.798 秒 | 183.367 秒 |

逐条复核了 record 对应、phase 加总、`C = M + 0.05 K`、目标簇汇总、主要 paired comparisons 以及 manifest 模块哈希。已记录收据内部没有发现算术矛盾。M3 的 8 个 OOS 只运行 Hegel，是诚实退出控制；它们不等于又成功解出了 8 道科学任务，其费用 416.82 单列在诊断证据中。[D1] [D2]

M0 到 M3 的 20 项里程碑回归测试通过。另查本次改动的 effective_history：8 项存储测试通过，包括新增目录归档测试；1 项旧的真实 7z 子进程测试因当前环境没有 7z 可执行文件而未通过。这属于环境依赖，不能据此判定本次代码有存储回归。测试日志另记录一项 anyio 与 pytest 的兼容性弃用警告，因此本环境不能复述“零警告全绿”。[D9]

原 V16.8 证据目录与 9115 提交比较没有任何变化；两个新增研究均独立落盘。所有本次重放与诊断在独立工作目录进行，未修改仓库源码，也没有调用付费模型。

### 1.2 应明确承认的改善

确认失败返回 unknown、成功产物及训练前缀冻结、旧的三个 AST 碰撞反例修复、对称 pilot、独立噪声 RNG 克隆、实测子进程计时，以及 ordinary 保留全部已验证候选，都有实际实现与行为证据。不能继续照搬上一轮原封不动的批评。[S3] [S4] [S9]

M3 中 PPR、两种 Flat、Vertical 和 Unchecked 五个森林臂确实克隆同一森林、契约、ScaleMap 元数据和 Stage-1 gate，并使用同一目标消费者。Flat 在当前主实验中保留了同一节点的契约、incident/chain-parent 路径元数据及排除条件，Vertical 仍能读跨域 local 内容。这比上一轮更有说服力。[S1] [S2]

查询签名确由付费 pilot、读出校准与公开协议生成。检查到的目标 truth_schema 只写诊断日志，没有进入排序或拟合决策。有限语法、手工几何特征本身是受控检索研究的合理设定。`sig:` 前缀守卫只检查字符串格式；“当前数据流没有发现真值泄漏”比“守卫证明任意未来输入零泄漏”准确。[S1] [S2]

## 2. 应最先修的接口问题：确认标签仍可从文件进入 fit

**函数参数的训练前缀已隔离，模型工作目录中的训练文件尚未隔离。**

现有 CodeAgent 约定允许程序读取 observations.json 和 model_observations.json。`lab.acquire(confirm, workspace=...)` 先把确认观测加入历史，再同步写入这两份文件。验证器虽把旧 prefix 作为 fit 参数，模型子进程的 cwd 仍是原工作区。因此，合法地按文件读取训练数据的模型会见到确认标签。[S4] [S5] [S6]

本次用真实规律 y=u²、训练点 u=1,2、确认点 u=3 做了三个最小控制：

| 模型 fit 的数据来源 | wrapper 报告的训练前缀行数 | 确认 RMSE |
| --- | ---: | ---: |
| 函数参数 observations | 2 | 9 |
| observations.json | 2 | **0** |
| model_observations.json | 2 | **0** |

更进一步，通过新开放入口完整运行 query → write → evaluate → finish → fresh confirm，文件读取记忆器得到 completed、confirmation_rmse=0、prefix_count=2；外评新点 u=4 时预测 0、RMSE=16、false_confirmation=true。[D3]

这是已经复现的生命周期缺陷。原五个 scipy 模板只用 fit 参数，没有证据说明发表的 O1/O2 或 M3 模板利用了该通道。因此，原结果与接口问题应分别保留判断。

**最低充分修复：** 购买确认点前冻结模型、依赖和训练上下文；其中两份标准观测文件都只包含 prefix。确认标签存到该拟合上下文不可见的记录中，模型只接收确认输入。若以后利用全部已付数据重新拟合，产生新产物身份并重新确认。验收只需让原参数记忆器与两种文件记忆器都无法通过新点确认，同时正常 scipy 模型仍通过。可复用已有上下文归档能力，无须扩成任意程序安全项目。

收据还有相关的范围问题。validator 对 target、version、model hash、confirmation digest 是“提供期望值后才核验”；新开放链的 receipt 和 task_view 没有 confirmation digest，旧 heldout_confirmed 布尔兼容入口也仍存在。M3 主路径正常提供了较完整绑定，未发现其原实验复用过期收据；但“所有入口强制五元组绑定”不成立。新实验认证入口应显式要求完整身份与实际确认记录，兼容模式可以继续保留给旧调用。[S6] [S7] [D3]

## 3. V16.9 已接入 CodeAgent，但开放作者主张仍未闭合

### 3.1 生成候选的关键决策仍在宿主脚本

`open_python_revision.py` 真正执行 CodeAgent 工具回路，fit 真正调用 scipy.optimize.least_squares。这些工作成立。其 provider 叫 ScriptedOpenGrammarCodeProvider，依次消费宿主提供的 query_batches 与 candidate_specs；renderer 只有五个完整公式分支：Monod、Haldane 一维、Haldane 加旁路、过阻尼阶跃、欠阻尼振铃。[S7]

O1/O2 的候选、顺序和修订签名在实验开始前列好。所有候选先渲染，随后“诊断脚本”输出样本数、最大最小值及 span；run 的 stdout 不参与下一结构选择。本次向两个克隆 provider 提供相互矛盾的诊断，下一条 write 内容逐字相同。provider 的 usage 也是预设计数，不能作为在线 LLM token 使用量的证据。[S7] [D3]

因此，现有证据支持“真实 CodeAgent 接线、有状态拟合、五个新公式族内的反例修订和跨域复用”。“超过旧十类模板库”成立；“非预写模板的自主算子生成”还需单独验证。项目历史中已有在线开放作者路径，可优先接通既有能力。

旧模型确实在 witness 上失败，新族也确实通过新点确认。向后兼容检查会把新公式族重新拟合到旧源数据，再评旧源；这证明公式族能表示旧规律，不能解释成 witness 上同一组固定参数未经变化就保留旧源全部知识。不同物理上下文允许不同参数是合理的，报告应准确声明保留的是哪一层对象。[S7]

### 3.2 开放链普通组被无条件安排额外查询

ordinary 在试已有候选前被固定安排多买两点；删除修订节点的 ablation 被固定安排多买三点。这些观测没有由“现有数据拟合失败”的实际反馈触发。[S7]

本次仅在 seed 5026101 删除 ordinary 的额外两点；候选内容、顺序、拟合、确认点、容差与外评不变，其他臂不变：

| 目标 | Hegel 测量费 / 复合费 | ordinary 原测量费 / 复合费 | ordinary 控制测量费 / 复合费 |
| --- | ---: | ---: | ---: |
| O1 | 50 / 53.225 | 60 / 66.225 | **50 / 55.075** |
| O2 | 50 / 53.225 | 60 / 71.225 | **50 / 59.075** |

两题仍严格通过；控制后的 RMSE 分别为 .000574、.001468，与 Hegel 相同。原 10 单位测量差在此 seed 消失，剩余候选枚举计算差仍然存在。[D4]

这是单 seed 的归因诊断，足以说明额外两点并非这两题所必需，不替代四 seed 新实验。最小改法是所有臂获得同一 pilot 和相同可用工具后，先允许使用自己的全部历史；只有实际未达标才决定追加观测。历史结构是否减少搜索、是否减少新实验，应分别计量。

## 4. M3 的核心科学问题：PPR 随机游走贡献尚未识别

### 4.1 数学传播实现成立，算法包还包含另一项关键机制

PPR 的转移矩阵按出边权归一化，执行 `p_next = (1−α)r + αPᵀp`。在当前合法非负参数下，它是概率守恒的传播。共现特征只用于 reset，没有变成游走中的特征顶点，这确实避免了 arity 枢纽作为中间跨树捷径。[S2]

但是，PPR 最后会把同 schema、由 live wormhole 连接的成员分数求和，再按 schema 去重。Flat 对每个节点打 BM25 分，按 schema 留最高分节点。原比较同时包含特异性种子、随机游走、跨实例汇总和代表节点选择。等信息条件基本成立，因果归因的对象是整套检索算法。[S2]

跨实例汇总很符合本项目“多种历史支持同一抽象”的方向，值得保留；它的收益应与 walk 的收益分开。

### 4.2 对已提交冻结图的因子控制

范围严格限定为唯一持久化的 eval04 图、23 个实际可检索查询。T14 在检索前被 ScaleMap 阻断。重建付费 pilot 后，原 PPR 的签名、raw 和 authorized 排名与原证据 **0 mismatch**。truth 只用于事后评分。[D5]

| 同一图与查询上的变体 | 授权 R@1 | 授权 R@2 |
| --- | ---: | ---: |
| 原 PPR，α=.8，原分量汇总 | 16 / 23 | 23 / 23 |
| **α=0，保留原分量汇总** | **21 / 23** | **23 / 23** |
| 原 walk，取消分量汇总 | 15 / 23 | 21 / 23 |
| 关闭横向 walk，保留原分量汇总 | 17 / 23 | 22 / 23 |
| 原 Flat BM25 | 18 / 23 | 21 / 23 |
| Flat BM25，对同一分量求和 | 17 / 23 | **23 / 23** |

完全关闭随机游走后，Top-2 保持全中，Top-1 反而更好；Flat 加同分量汇总也达到 Top-2 全中。因此，“牺牲 Top-1 是为获得 Top-2 所必需的图扩散权衡”尚未被这批数据支持。

这不证明图或虫洞无用，也不证明 α=0 在未测任务上普遍更优。它说明当前 perfect Top-2 不需要依赖 walk；应先加入这几个廉价控制，再讨论复杂自适应阻尼。

此外，节点 ID 的字典序参与平分排序。保持全部内容和拓扑，仅反向重新标识节点后，Flat R@2 从 21/23 变成 22/23，T13 的正确 schema 从第三升到第二。这属于标识符平局敏感性，没有证据据此推测命名意图。候选窗口很小时，应采用内容稳定的 tie-break，或明确处理同分层。[D5]

### 4.3 对 T13 和 T20 的实际消费者控制

共执行九次目标调用。原冻结 gate 的完整数值状态未持久化，本控制明确固定两题原来已经发生的 Stage-1 admission=true，其余使用原 solve_target。对两题原 PPR/Flat 的模型 SHA、schema、RMSE、质量和全部逻辑成本核对，**0 mismatch**。[D6]

| T13 变体 | 测量费 | 复合费 | 质量 |
| --- | ---: | ---: | --- |
| 原 PPR | 30 | 31.4650 | 严格通过 |
| α=0，同分量汇总 | 30 | 31.3915 | 与原 PPR 同模型 SHA、同 RMSE |
| 原 Flat，试两个候选 | 45 | 49.5940 | 严格通过，转全搜 |
| Flat，允许试三个候选 | 30 | 32.4690 | 与原 PPR 同模型 SHA、同 RMSE |

Flat 第三个候选原本就在检索结果里。前两次廉价拟合失败也会用完两个槽，随后转去高测量费用的全量探索。多容纳一次候选拟合，就追回 T13 全部 15 单位测量差，仍保留两次无效拟合的计算劣势。统一两槽是一个有效的固定策略比较，但它不足以证明 PPR 是省观测的唯一可行机制。

更合适的下一步是分别预算候选拟合和新观测：训练残差立即失败时，可以在计算预算内试下一个候选，而非自动转去买数据。所有臂共享这一策略后重新比较。

T20 的情况不同：原 Flat 的真值 rational schema 排第三，但第二名 rollover 已在容差内严格通过。这里 schema Recall miss 没有导致任务失败或额外观测。必须并列报告“命中生成家族”和“得到容差内有效模型”，不能把二者等同。[D6]

### 4.4 与 Flat 的全簇净收益主要由 T13 决定

原每 24 题簇的 H−Flat 目标复合差为 −12.487875；T13 单题贡献 −18.129。其余 23 题合计是 +5.6411 复合费、+5 测量费；加索引差为 +7.8386。[D7]

这不是删掉 T13 的理由。它说明当前方法确实擅长这道困难题，也说明四个 seed 的同号结果主要重复同一困难结构。下一轮应保留 T13，并加入结构不同的桥及干扰强度控制，不能用四个参数/噪声种子代替四种独立机制。

## 5. 历史方法、两跳与尺度证书的实际边界

### 5.1 M3 又绕过了 program_steps

M2.5 删除校准/确认步骤会改变执行与回退，原装饰性字段问题确有修复。但新 M3 消费者只读节点的 schema_tag，调用固定合成器，再走固定确认路径；它没有解释 program_steps。[S1] [S3]

本次删除全部 36 个节点的 program_steps，再运行 T13，模型 SHA、观测、严格质量和全部逻辑成本与原 PPR 完全相同。[D6]

现有 M3 证明了历史检索能够缩小函数族候选，尚未证明历史中形成的方法程序被抽象、迁移并执行。最小修复是让 M3 复用已有受限执行器，规定允许的步骤确实决定校准、角色绑定、拟合与确认；不需要为这一点创建新的通用 agent。更一般的程序顺序/控制流能力，可留给明确需要它的里程碑。

### 5.2 当前“多跳”更接近指定桥的预算筛选

eval04 原 PPR 在 23 次检索中返回的 129 个候选全部属于 signature 的直接命中集合，没有候选需要非直接命中的 BFS 路径发现。代码仍真实执行 PPR 传播，但这些案例没有演示必须沿隐含关系才能发现的新节点。[S2] [D5]

T13 对 high_load_rollover_detected 且 schema=rational_monod 的情形有专门分支，寻找指定的 rational→rollover 桥。rollover 标志来自观测，这比旧目标专用提示好；桥关系仍是该机制对的手工适配。[S1]

同 AST 边使用两源模型 RMSE 的最大值、Lipschitz=1；第二跳用 monod 拟合 rollover 已付数据的训练 RMSE、Lipschitz=1。拟合真实发生，但这些数值没有自动成为实际 observation/parameter map 的传递误差界。

组合公式形如 `ε_total ≤ L₂ ε₁ + ε₂` 可以保留；前提是每个 ε 对应已定义变换、范数、作用域及误差保证。目前各源残差、近似族拟合残差、映射误差混用了。超预算表示不能继承该保证，也不直接证明候选物理上无效。[S1] [S6]

阶段性最小做法是称其为经验路由风险估计，仍保留目标新点确认。若要主张多跳保证，则补每边独立验证、真实映射组合与相应误差/不确定性记录，而非仅给合成 helper 输入一个数。

### 5.3 ScaleMap 的经验认证输入目前是常数

`m3_scale_universe.py:1376–1379` 的 scale_probe_fn 先收费 observation=2、measurement=10、compute=4，然后直接返回 mean_flux_diff=.0012、peak_crest_diff=.84。没有实际采样、微观轨迹、窗口平均或差异计算。后续据此写入 verified_discarded_in_scope，并按 window_average 名称复用。[S1]

这是一项预置测试 fixture 被当成经验认证。五个森林臂共享它，因此不单独偏袒 PPR；它不能支持已经学习了跨尺度可遗忘变量证书的主张。

可以选择一个很小的真实微观/宏观校准实验，把证书绑定到协议、窗口、变量范围、目标、容差和数据摘要。若当前不想扩展实验，直接标注为 fixture，把“经验认证已完成”从完成清单移出即可。ScaleMap 的设计本身仍可保留。

### 5.4 反例与一般化关系也需分开记录

M3 对 incumbent 确实做了 witness 拟合，但修订准入没有要求 witness 超容差才注册；部分排除谓词由新 schema 名选择，zero_limit_reduction_to_parent 也直接声明，缺少对应极限变换检验。[S1]

当前四个受控近失可以作为正例保留。扩大到开放作者前，应分开“新候选有效”“旧候选被反驳”“新公式保留或一般化旧契约”三种证据。它们不应由同一个 schema 标签自动推出。

## 6. AST 骨架还有误合并和误拒，修复应分两边

原三类 AST 反例已经修好。新的受支持子集仍有两个小碰撞：

| 代码差别 | canonicalizer 输出 | 真实预测差别 |
| --- | --- | --- |
| 自定义 exp(x)=x² 与 exp(x)=x³，predict 都调用 exp(x) | 都 VERIFIED_PURE_AST、相同 digest | x=2 时 4 对 8 |
| round(x, ndigits=1) 与 round(x, ndigits=3) | 都 VERIFIED_PURE_AST、相同 digest | x=1.2345 时 1.2 对 1.234 |

根因是仅按函数文本名识别调用、未检查实际绑定，并丢弃调用 keywords。这是明确的受支持语法合同问题，无需解决任意程序等价：不支持的绑定/关键词可以返回 UNKNOWN，结构性参数应保留。[S3] [D3]

另一方面，提交图中 9/36 节点带 mismatch digest 但仍为 SUPPORTED_IN_SCOPE，包括 linear、exp、cobb 三对源及三个单源。M3 准入只检查 digest 非空，没有检查 skeleton_status 是否为 VERIFIED_PURE_AST。[S1]

不能把这九个节点写成九个错误科学模型。本次按原 seed 重放 S01/S02，严格通过、digest 与提交一致；正文和旁注表达同一公式，却被规范化器不对称处理。predict 侧递归剥 affine，把 core 内的比例/截距也剥掉；annotation 侧保留，导致 ROLE_0 与 POS_NUM + POS_NUM*ROLE_0 被判不一致。同族身份因此被拆开，潜在横边缺失。[D8]

**最低充分修复：** 先修正文/旁注的归一规则和 lens 边界，再统一消费者的显式身份状态。行为已验证、骨架未验证的模型可以作为独立 local 求解器保留；它不应被当成已证明抽象同一性的依据。不能只删除九个节点。图分组会改变，修后要重新形成森林并比较相应检索消融。

## 7. 基线消费和计价还有两项影响结论的问题

### 7.1 HExA 的新错误经验在默认窗口里不可达

M3 先拼接全部源 skill，再拼接 mistake_bank，最后只试前两个。按四个原评测簇收据重建：

| 元数 | 基础技能数 | 新 mistake 的实际位置 |
| --- | ---: | --- |
| 2D | 9 | capped、ringing、haldane 排 10、11、12 |
| 3D | 2 | state_memory 排 3 |

删除 mistake_bank 后，所有实际试用前缀不变。最后全量搜索仍能解题，因此 100% 成功率掩盖了消费缺口。当前臂应称“arity/频率优先的两技能基线”，不能据 −296.893375 的端到端差评判充分消费反例的 HExA 类方法。[S1] [D10]

最低充分改法是让适用条件参与排序，使与当前观测相容的恢复技能能进入窗口，并用保留/删除相关错误经验的行为对照验证。ordinary 全历史臂也确实保留了候选，但只按 arity 全量拟合；它适合测量条件化调度的整体价值，不能单独证明图结构不可替代。

### 7.2 相同候选搜索可被收取 120 或 48 计算单位

合成器对 allowed_schemas=None 收固定 48，显式列表收 10×列表长度。同一 T01、六个已付 pilot、相同读出校准，显式 12 个已验证 2D 候选与 None 执行后，选择同一 schema、train/val RMSE 同为 .003128、model.py 字节及 SHA 完全相同，收费却为 120 对 48。[S1] [D11]

仅把 ordinary 首轮历史搜索的计算费封顶为已有全搜价 48，不改变任何科学动作，每簇减少 1152 计算单位，即 57.6 复合单位：

| 每 24 目标簇的四簇均值，H−ordinary | 原计价 | 此次一致计价敏感性 |
| --- | ---: | ---: |
| 目标测量费 | −26.250000 | −26.250000 |
| 目标复合费 | −130.594625 | **−72.994625** |
| 给定 gate 的源期加目标端到端费 | −104.637125 | **−47.037125** |

优势仍在，原幅度有相当部分来自代理价目。这是固定行为的事后计价敏感性，不是重新校准门控后的新主实验。

另实际捕获到每次源形成的桥拟合收 10 compute=0.5 composite，却没进入 source 收据；两个 DEV、四个 EVAL 各一次。补齐后，计价归一化、给定 gate 的 H−ordinary 端到端差为 −46.537125，DEV 加 1。H 与 Flat 共享形成成本，差额抵消。这项是小额完整性问题，不会翻转当前结论。[D11]

正式修复应按实际执行候选、角色排列和拟合工作统一计费；等价候选集合应同价。修价后重新校准 gate 并冻结评测。本次没有重算新价目的 N*，不能沿用原 7/13。

### 7.3 真实墙钟与代理计算优势需要分别报告

报告 12 墙钟列取自 eval01，其他主要成本列是四簇均值。实际四簇平均的“24 目标 Python 子进程时间之和”：H=1.851000 秒，Flat=1.804875 秒；本次重放为 2.197325 对 2.198025 秒，几乎持平。这个指标也不包含整个 solver 的全部检索与调度时间。[S8] [S10] [D10]

因此现有结果支持测量和代理成本收益，尚未建立对 Flat 的实际运行时间加速。只需统一聚合单位，并在确实讨论性能时独立测量检索、拟合、确认和总墙钟；不必为了本次审计重复大量计时。

## 8. 规模、Recall 与摊销应怎样解释

### 8.1 当前不是隔离 K 的规模曲线

提交图为 36 节点、32 棵树、23 schema、4 纵边、11 横边。32 个 source 建树，4 个 near-miss 增加原树子节点；K≥32 的树数门槛已达到，36 更准确指工作区/节点数。[D5]

K=7 与 K=36 两版同时改变目标数量、机制库、pilot、合成费率和检索器。四个评测 seed 主要改变噪声和有限参数，不是四套独立机理宇宙。因此前后“由亏转盈”可以如实报告，不能单独归因为 K 增长。

最有判别力的规模实验是冻结目标、候选语法、费用和正确知识覆盖，只增加 distractor、重复来源或拓扑复杂度；分别测量这几种增长。无需立即扩到几千个节点，也不宜把同公式换名数量当成新的独立机制。

### 8.2 Recall 是授权后的 schema Hit@k

原 H/Flat 的分母是 92，因四个 T14 在检索前阻断；Unchecked 为 96。同一 92 题上 Unchecked 的 R@2 为 84/92=.9130，原表 84/96=.8750 混入了分母差异。H 仍是 92/92，但应并列报告原始排名、授权后排名、scope 阻断率和语义任务成功率。[S8] [D7]

### 8.3 N*=7、13 是原顺序和原价目下的结果

原曲线在首次穿零后确实持续为负，未发现“短暂回本又反弹”。但 Flat 的第 13 题恰好是 T13。对已存逐题成本做 20000 次顺序置换，Flat 回本位置范围为 1–24，5%–95% 为 2–23；ordinary 为 1–11，5%–95% 为 2–8。ordinary 的一次历史重建费已移至 upfront，再计算前缀。[D2]

这些只是代数顺序敏感性，不是在线门控重新运行的概率分布。实际顺序变化可能影响 gate 状态，因此未来若研究持续使用，要随机化顺序并保存完整状态。

### 8.4 DEV 已单列，但主表仍是给定 gate 的部署口径

原 DEV=6931.8425 已披露。若假设全部 DEV 只分给 H，每个新簇重复源期和 24 题费用，则按原价目摊销需要：

| 参照臂 | 给定 gate 每簇节约 | 摊平全部 DEV 所需同规模簇数 |
| --- | ---: | ---: |
| ordinary | 104.637125 | 67 |
| Flat | 10.290375 | 674 |

67 只对应 ordinary。若多个臂共享同一校准 gate，或同一源库服务长目标流，应该换用相应的分摊公式，不能把全部 DEV 任意分给一方。报告的 λ 网格则重加权原固定行为；它证明该策略费用分量的敏感性，不代表不同 λ 下的门控均重新训练和实测。[S8] [D2]

## 9. 对交接书五个问题的直接回答

| 问题 | 本次判断 | 最小下一步 |
| --- | --- | --- |
| F1–F11 是否全部闭合 | 原 F5、F6、F9 等已修；F11 文件通道、F2 语义身份、开放作者、收据全绑定及 M3 程序消费仍有缺口 | 先修有实际失败见证的接口，不扩大到无限安全/语法要求 |
| M3 是否有标签泄漏、是否公平 | 未发现目标真值进入实际排序；五个森林臂等信息比较显著改善。七臂整体不是只改一项内部机制 | 保留原对照，增加 seed/walk/pooling 因子分解及合理预算控制 |
| 是否应上自适应 α 或 passage reset | 当前 α=0 已保住全部 Top-2，复杂化优先级低 | 先在 DEV 比较少量静态 α 与 pooling，冻结后评新结构；再决定是否需要 query-dependent α |
| 召回与授权、多尺度是否稳妥 | 分层设计有价值，实际数值授权证据还不足以支撑一般保证 | 固定 fixture 显式标注；映射/尺度证书绑定真实数据和作用域，保留 fresh confirmation |
| M4 如何合流 | 先做一个小而真实的开放作者闭环，再增加结构与数据异质性 | 作者反馈进入候选生成，所有臂同工具同历史同预算，固定外部评估和累计成本 |

HippoRAG 2 原文确实把 phrase 与 passage 节点同时用于 PPR reset，并在开发数据上调 passage 权重；附录的 damping 为 .5。这不是本项目应照搬 .5 或加入 passage 层的证据。本项目节点是科学方法/机制对象，应先识别当前收益来自哪里。论文最适合借鉴的是把 reset、节点层与检索效果分别做消融，以及在固定任务上增加语料来测扩展行为。[S13]

## 10. 建议的 M3.1 与 M4 验收

### 10.1 M3.1 用三组小工作收口

**正确性接线。** 冻结完整确认上下文；修 AST 调用绑定、keywords 和两侧规范化；新入口强制完整 receipt；M3 接通已有步骤执行器。每项以本报告现成失败见证作为验收，正常模板应维持通过。

**归因与基线。** 保留原历史结果，补 α=0 同 pooling、walk 去 pooling、Flat 同 pooling，以及计算/测量分开的候选策略。让 HExA 的相容错误经验进入实际窗口。开放链先试历史再决定是否购测。无需调大参数网格或重写森林。

**证据与费用。** 等价搜索同价，桥拟合进入 ledger；统一 Recall 分母和时间单位；保存每簇冻结 gate、图、实际作者模型/事件/训练前缀。原研究脚本的临时开放工作区会被清理，摘要中的事件数不能替代完整作者轨迹。复用已有归档与版本系统即可。

### 10.2 M4 的第一道门应是反馈确实生成了新的方法

建议先用两三条结构不同的开放修订链。每条链在冻结时没有完整候选答案；允许作者使用通用工具与已学代码，但新的结构选择必须依赖实际 witness 反馈。保存何时提出了哪个表达式、依据哪些观测、做了什么实验、哪次确认使其取得何种权限。

最少对照可分为：持续使用全部历史的同工具普通作者；同内容的强 Flat 方法库；森林/虫洞方法库；冻结记忆版本。LLM、上下文、工具、数据权限、角色诊断工具、质量门和费用预算应一致。普通组不能被强制先多买数据，也不能把 shared 目录存在等同于它已经能实际消费历史。

针对项目的哲学框架，建议把“历史总结与一般化”转成以下可反驳的工程要求：

1. **历史产物有实际作用。** 后继执行读取并使用真实方法/代码对象；删除该对象改变后继行为，恢复对象恢复该能力。不能只改 schema 标签或额外查询脚本。
2. **反例驱动结构变化。** 改变 witness 的关键结构，作者提出的修订应相应改变；诊断 stdout 的反事实控制应能影响下一候选。
3. **保留旧知识的层级明确。** 区分固定已拟合模型、可重拟合机制族、适用条件与原实例集合。要求旧作用域验证集继续达标，并允许新证据收窄原契约。
4. **改进在未见任务上体现。** 固定外部 heldout，按机制族、实验协议、变量范围和时间段切分，避免只随机分同一轨迹的相邻行。
5. **持续改进支付全部必要费用。** 目标期、源形成/索引维护、DEV 与作者试错分别记账；画相同质量门下的累计费用与遗忘曲线。
6. **图收益超越同内容强基线。** 特异性与多来源汇总先作为强基线，只有在其上仍稳定改善，才能把剩余收益归给传播或多跳方法组合。

这些条件允许有限、可审计的开放语法，不要求一次做到所有科学领域。它们能回答“历史组织是否使研究者变得更会研究”，而不是仅证明它能从更大的已知公式表里挑一个候选。

### 10.3 大规模真实数据应放在作者与契约共同闭环之后

真实表格和动力学数据里，需要先定义哪些数据可获得、是否真的能主动查询、measurement 单位如何确定。若是离线数据，购测应对应揭示一个预先隐藏的数据块，并让各臂遵守同一揭示协议；若是模拟器或实验设备，应记录真实查询调用与费用。否则“少测一次”可能只是接口动作更少。

只有在小规模合流中同时看到新程序形成、有效历史消费、独立确认和后继收益，再逐步扩大机理数、观测协议和数据来源。若 α=0 或 pooled Flat 仍胜过 PPR，可以保留它们作为当前检索器；森林中的版本、作用域、反例与许可结构仍然有研究价值。

## 11. 阅读覆盖、复现范围与交付文件

相对 9115 本轮有 2 个提交、186 个变化路径。对新增/修改的核心代码、脚本、测试与工程报告进行逐段审查，核对新增两套证据全部账目，完整重放两项研究。核心差异为 16 个文件、7405 行增加、525 行删除；额外大量路径是上一轮审计及工作区证据归档。

| 范围 | 本轮覆盖 |
| --- | --- |
| 新检索与 M3 实现 | hipporag_wormhole_index.py、m3_scale_universe.py 全文，实际形成与消费路径 |
| 开放作者入口 | open_python_revision.py 全文，CodeAgent adapter、fit/confirm、后继消费 |
| 既有关键模块 | open_workspace_wormhole.py、open_workspace_labs.py、wormhole_forest.py 全部变更和关联调用 |
| 脚本和测试 | V16.9/M3 两研究脚本、新测试全文，相关旧回归测试 |
| 其他源码变动 | effective_history 目录归档增量及其测试 |
| 文档 | 用户交接书、报告 11/12、索引更新、上一轮审计及相关版本设计 |
| 原始证据 | V16.9 与 M3 全部 source/target/ledger、图、诊断及汇总 |
| 新控制 | 23-query 检索分解、9 次 target 控制、单 seed 开放链、AST/确认边界、同工作计价与建桥捕获 |
| 文献 | 本轮重新核对 HippoRAG 2 主文/消融/实现附录，与项目实现区分 |

固定提交的树清单有 12235 个文件，其中 Hegel Machine 为 7687。清单存在不等于这些历史归档每一字都重新审读。上一轮的版本谱系和文献覆盖边界继续保留；本次没有声称重新读完所有历史参考文献全文。此处明确覆盖范围，避免把“已索引”“已读”“已运行”混在一起。

工件包包含主报告、三个专项报告、所有诊断脚本及 JSON、两套完整重放证据、相关小实验模型/观测/确认工作区、固定代码快照、变化清单、运行环境与 SHA256 清单。复现入口与 HEGEL_REPO/PYTHONPATH 设置见 README。完整固定仓库仍以给定 Git commit 为准。

### 专项报告

- [开放作者与 F1–F11 闭环](closure/closure_audit.md)
- [M3 检索与契约机制](mechanism/MECHANISM_AUDIT_M3.md)
- [实验结果与成本核对](results/results_audit.md)

### 证据索引

| 编号 | 文件 | 用途 |
| --- | --- | --- |
| D1 | results/replay_verification.json | 两完整重放、成对比较与 DEV 一致性 |
| D2 | results/receipt_recomputation.json | 两版全账目、Recall、N 与 DEV 分摊 |
| D3 | closure/closure_probe_results.json | 文件通道假确认、AST 两新碰撞、provider 反事实、收据 |
| D4 | closure/open_forced_queries_results.json | 开放链 ordinary 删除两点控制 |
| D5 | mechanism/retrieval_factorization_result.json | 23-query 因子控制、图计数、直接命中与平局 |
| D6 | mechanism/target_controls_result.json | T13/T20 九次实际消费者控制 |
| D7 | results/contribution_and_recall_scope.json | T13 贡献与同分母 Recall |
| D8 | closure/m3_ast_rejection_results.json | 原源 S01/S02 误拒及九节点清单 |
| D9 | pytest.log | 20 里程碑与 9 存储测试的环境结果 |
| D10 | results/baseline_consumption_and_timing.json | HExA 可达性、正确墙钟单位 |
| D11 | results/cost_accounting_probe.json | 等工作不同费、桥漏账、费用敏感性 |

### 固定源码与原始报告

所有源码定位固定到 612d8eea5，避免后续分支移动改变被审计对象。

[S1]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/src/hegel_machine/cognition/m3_scale_universe.py
[S2]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/src/hegel_machine/cognition/hipporag_wormhole_index.py
[S3]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py
[S4]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_labs.py
[S5]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/src/hegel_machine/cognition/code_agent.py
[S6]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py
[S7]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/src/hegel_machine/cognition/open_python_revision.py
[S8]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/scripts/run_v17_0_m3_hipporag_scale_study.py
[S9]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/11_V16_9_M2_Audit_Closure_and_Open_Author_Report.md
[S10]: https://github.com/erzhu419/Assumption-Agent/blob/612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/12_V17_0_M3_HippoRAG_Scale_Wormhole_Report.md
[S13]: https://arxiv.org/html/2502.14802
[D1]: results/replay_verification.json
[D2]: results/receipt_recomputation.json
[D3]: closure/closure_probe_results.json
[D4]: closure/open_forced_queries_results.json
[D5]: mechanism/retrieval_factorization_result.json
[D6]: mechanism/target_controls_result.json
[D7]: results/contribution_and_recall_scope.json
[D8]: closure/m3_ast_rejection_results.json
[D9]: pytest.log
[D10]: results/baseline_consumption_and_timing.json
[D11]: results/cost_accounting_probe.json

