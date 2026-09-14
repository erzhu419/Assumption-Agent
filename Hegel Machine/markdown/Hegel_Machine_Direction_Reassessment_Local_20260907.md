**黑格尔机方向调整：基于本地代码与运行记录的复评**

日期：2026-09-07。本文保留当时本地复评；后续目标状态见[2026-09-09 核对](../docs/hegel_machine_advice_goal_status_20260909.md)，当前改动见[剩余功能整合](../docs/hegel_machine_completion_20260909.md)。

**判断：网页建议的研究重心调整仍然成立，但具体方案应修订后采用。已经完成的 Q1/Q2/Q6 应作为有限形式结果保存；下一项研究应直接检验冻结后的 Agent 能否取得新的观测、构造有用的表示，并让它降低后续任务的学习成本。最需要防止的第二次偏移，是把“自主知识生长”缩成“从几个手写修复中选一个”。**

本次将八份 Markdown 中的历史问答、施工指令和网页建议视为审阅材料，结合本地源码及小型终态记录重新评价。

**1. 本地版本差异改变了哪些判断**

当前文档目录所在工作树是 `codex/reconstruction-v2-paper`，HEAD 为 `0af65964235390ce2bebefea7379eaa9c50eda24`。后续运行工作树位于 `/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree`，HEAD 为 `20fd401121842c4755826d3b19f0b944b97542e2`。网页评估的 `3ce991932eb24aa3c8a99bd9fa5695ca39ac3f21` 是后者的直接前序。

这两个提交之间只有一次 C1 prediction/scoring replay v2 工程提交，涉及 9 个文件、4503 行新增。`diagnostics.py`、`bounded_relation_invention_v1.py`、`meta_prior_shadow_v1.py`、`engines.py`、`probes.py`、`schema.py` 没有变化。因此，网页对这些认知机制的评价不会因为 HEAD 更新而自动失效。

更重要的变化来自 Git 之外的运行记录：冻结配置中的初始状态不会随执行自动更新，HEAD 也不能单独表示项目运行进度。

| 项目 | 本轮查到的本地记录 | 对方向评价的影响 |
|---|---|---|
| Q1 | `Q1_COMPLETE_20_OF_20`。[终态](</home/erzhu419/.local/state/hegel-machine/q1-completion-successor-v8c-ledger/completion.terminal.json:1>) | 有限商闭包已完成，属于实质成果。 |
| Q2 | `Q2_ROLE_MEMBERSHIP_PASS`；odd 为 418 类、0 匹配；sink 为 89 类、1 匹配。[终态](</home/erzhu419/.local/state/hegel-machine/q2-role-membership-v1-ledger/campaign.terminal.json:1>) | 网页的数值判断成立，本轮进一步找到本地终态支持。 |
| Q6 | `Q6_PURPOSE5_THREE_OF_THREE_FORMAL_QUOTIENT_CERTIFICATE_ISSUED`；记录 3 份有效签名及 campaign complete。[终态](</home/erzhu419/.local/state/hegel-machine/q6-purpose5-formal-quotient-certificate-issuance-v1-ledger/campaign.terminal.json:1>) | “实现但未执行”“接下来封口 Q6”已经过时，应改为“签发已完成”。 |
| Q0.5b v2r20 | `CAMPAIGN_SHADOW_Q05B_SUCCESS_NONADOPTING`；slot 1 exit 0、qualification 20/20、trajectory COMPLETE_OK。[终态](</home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r20-ledger/campaign.terminal.json:1>)、[slot 结果](</home/erzhu419/.local/state/hegel-machine/q05b-shadow-retry-campaign-v2r20-slot-01-evidence/slot-result.json:1>) | 隔离运行与回放资格链已有成功记录；这是技术执行成果。 |
| 新增 C1 replay v2 | capsule ready；trajectory 只有 `CAPSULE_PREPARED`，consumption 与 campaign 均为 0。[capsule](</home/erzhu419/.local/state/hegel-machine/phase2b-c1-actual-unsealed-960-prediction-scoring-replay-v2-run/capsule.json:1>)、[轨迹](</home/erzhu419/.local/state/hegel-machine/phase2b-c1-actual-unsealed-960-prediction-scoring-replay-v2-run/trajectory.jsonl:1>) | 增加了计分回放工程，尚无这一阶段的实际运行与 C1 效果。 |

新 C1 文档也明确记录 [Q6 已 issued](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/docs/Hegel_Machine_Phase2B_C1_Actual_Unsealed_960_Case_Prediction_Scoring_Replay_v2.md:122>)，而该 C1 capsule 即使成功，其上限仍是 [mechanics complete](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/docs/Hegel_Machine_Phase2B_C1_Actual_Unsealed_960_Case_Prediction_Scoring_Replay_v2.md:117>)。

Q0.5b 的技术 shadow 与 robustification/idealization 的认知 shadow 是两项不同工作。后者仍从 fixture rows 和指定分支生成候选；两者的证据边界在[现有文档](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/docs/Hegel_Machine_Meta_Prior_Bounded_Dual_Branch_Shadow_v1.md:75>)中已有明确区分。

所以，项目实际进展比网页部分描述更靠前，核心研究缺口仍然存在：目前的证据尚未连成“自主改变观测或表示—形成可执行知识—在未见任务复用—遇到反例修订”的能力链。

**2. 网页建议哪些值得保留，哪些应重新定位**

最值得保留的是三点：把科研进展落在未见任务收益上；让候选生成摆脱“先穷尽所有失败原因”的通用前置条件；把认知实验与已有正式资格链分开推进。这些调整直接对应当前缺口。

不过，网页列出的部分思想批评，已经在旧文后半段得到原则性修正。[《黑格尔机》](</home/erzhu419/mine_code/Asumption Agent/Hegel Machine/markdown/黑格尔机.md:4315>)明确说本项目不能证明人机本质差异；同文[初始 22 条的定位](</home/erzhu419/mine_code/Asumption Agent/Hegel Machine/markdown/黑格尔机.md:3088>)已经允许扩展；[泛函分析讨论](</home/erzhu419/mine_code/Asumption Agent/Hegel Machine/markdown/黑格尔机和泛函分析.md:1929>)已经区分“组织可观测性”与“发明新语言”。

因此更准确的评价是：这些原则需要统一到现行研究定义和实验中，部分历史施工规则仍与它们冲突。不能把历史对话最初的一句话当成项目至今唯一有效的立场。

真正需要动的规则，是把所有认知变化都要求为“扩大表达能力、保留全部旧结论、退化为旧理论、排除全部便宜修复”。已有 [C1–C4 计划](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/docs/Hegel_Machine_C1_C4_Authoritative_Progression_Plan_v1.md:279>)把这些要求集中到 C2。它适用于其特定冻结主张，却不适合作为所有知识学习的定义。

新概念至少有三种不同价值：在受限旧语言之外表达新行为；把旧语言中的长程序变成容易搜索和复用的抽象；获得改变预测或决策的新经验知识。它们应分别测量。可展开为旧程序的抽象仍可能有价值；[DreamCoder](https://arxiv.org/abs/2006.08381)正是通过组合式符号抽象与搜索学习建立可迁移程序库。固定底层 primitives 本身不取消自主发现，关键在于新知识的具体内容是否由系统从经验形成，以及它后来是否有用。

**3. 构想本身还需要补上的四个区分**

**动作选择与内容形成。** 判断“这次要 robustify”，调用人写好的完整补丁，只证明策略选择。原始[用户构想](</home/erzhu419/mine_code/Asumption Agent/Hegel Machine/markdown/黑格尔机.md:1>)还要求生成新的概念、关系或元先验。首轮可限定动作类型，但至少应有一种历史摘要、关系程序或抽象体没有被预先写成答案；形成以后，后续任务真的调用它。

**世界模型与风险偏好。** 发现某种失误有非零概率、存在隐藏状态，或原不确定集覆盖不足，是世界模型变化；决定愿意为安全牺牲多少效率，是用户的任务取舍。[《Meta prior》](</home/erzhu419/mine_code/Asumption Agent/Hegel Machine/markdown/Meta prior.md:539>)把平均损失换成 CVaR 等指标，容易混合这两者。实验应先固定外部损失、风险约束和行动代价，让 Agent 学习机制、分布或不确定集。只提高惩罚权重、换指标或变得更悲观，不能算发现了新知识。也不必要求鲁棒决策一定改善点预测；应测它在同一任务目标下带来的决策收益与代价。

**失配修正与主动简化。** 网页新命题仍以“旧模型出现可复现失配”为入口，但预测正确、计算昂贵或每次重复搜索的模型，也值得改进。这正是用户关于质点、刚体等简化假设的动机。[《Meta prior》](</home/erzhu419/mine_code/Asumption Agent/Hegel Machine/markdown/Meta prior.md:612>)已经把扩张和压缩并列。触发条件应包括失配、重要解释尚不可区分，以及重复求解成本过高。

**单步修补与长期价值。** “最小有效修订”可以是成本偏好，不能变成只能走立即获益的最短一步。新观测与新表示可能要一起变化；一个抽象本题较贵，却能降低后续十个任务的成本。评价应覆盖一段任务序列，并计入建立知识的前期成本。否则算法会系统性拒绝原构想最重视的积累。

关于元先验，网页“不能保证小样本不过拟合”的判断正确，但这里也无需走向“先验没有价值”。需要证明的是，在明确任务族中，这些先验比同资源对照更有效。它们应允许竞争、调整作用域和被撤销。[Locatello 等人的结果](https://proceedings.mlr.press/v97/locatello19a.html)针对无监督解耦表示的可识别性及实证表现，支持明确归纳偏置和下游收益的重要性；它不能直接作为所有表示学习或本项目的普遍不可能性定理。

广义对位关系仍可以保留为生成候选、约束组合和提出反例的工具。它提供的结构是否值得使用，应由任务收益决定。已有 [SCAR 负结果](</home/erzhu419/mine_code/Asumption Agent/Hegel Machine/docs/v2_scar_negative_impact_assessment.md:49>)显示特定硬结构选择流程发生严重覆盖下降；同一评估也正确指出，这没有逐条否定全部 22 条先验。因此不能以“结构更严格”推断“推理必然更好”，也不能由这一 operationalization 的失败否定结构关系整个方向。

**4. 新方案落地前，四处具体接口不能直接沿用**

| 现有部件 | 已经能做什么 | 新认知实验需要调整什么 |
|---|---|---|
| bounded relation synthesis | 搜索出真实可执行关系；结果不是单纯存一个名称。 | [枚举原子](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/src/hegel_machine/bounded_relation_invention_v1.py:202>)已包含 `prior_on`。新实验给原始时序与通用历史访问能力，让系统决定取什么、如何构造特征。 |
| diagnostics 与 language patch | 按冻结阶梯稳定分类和检查报告。 | [便宜修复收益](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/src/hegel_machine/bounded_relation_invention_v1.py:313>)部分是 fixture assertions。新实验中的收益由实际执行产生；未尝试保持未知。候选生成无需伪造一个已通过的 ontology report。 |
| probe 选择及行为去重 | 在各自固定定义下选择 probe、压缩等行为程序。 | 多候选主动学习需要不同的选择目标，并保留未来实验可能分开的程序。详见下文。 |
| compiler 与版本图 | 支持当前冻结阶段的范围扩展和治理限制。 | [compiler](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/src/hegel_machine/governance.py:707>)只支持 `add_scope`，[版本图](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/src/hegel_machine/governance.py:1335>)只允许一个 genesis。新实验先用实验内可调用程序库测复用；直接接入现有图会把研究重新绑回正式 admission 工程。 |

probe 有一个与本方案直接相关的适用性问题。[`choose_discriminating_probe`](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/src/hegel_machine/probes.py:124>)计算所有候选对之间的最小 total variation，再除以 probe 成本。对于三个以上确定性候选，若每次 probe 只有二值结果，至少两个候选必然预测相同，最小距离便是 0。所有这类 probe 都得 0 分，最终按 `probe_id` 的字典序打破平局。

这个实现符合它声明的 maximin 目标，但不适合直接承担新方案中的多候选二值实验选择。第一轮可以只保留两个竞争解释；需要更多候选时，改用预测分区带来的预期候选削减量或明确任务收益，并计入成本。采用哪一种应由小实验决定，无需先训练通用实验策略网络。[DAD](https://proceedings.mlr.press/v139/foster21a.html)和 [BAX](https://proceedings.mlr.press/v139/neiswanger21a.html)可作方法参照；引用它们不会自动解决当前候选和观测接口的设计。

另一个使用边界是：当前 synthesis 会按 discovery 与 validation 上的行为向量只留最短代表，见[去重实现](</home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/src/hegel_machine/bounded_relation_invention_v1.py:446>)。这适合现有固定数据样例。主动学习时，当前观测相同的两个程序可能对新 probe 预测不同，应保留这些成员，或能够重新生成并分组。现有 Q1 的有限完整 universe 结果继续成立；变化发生在新实验的观测集合。

**5. 建议把首个实验拆成三段，最后再扩大认知动作范围**

**第一段：检验“会不会花预算取得有用证据”。** 从一个小型时序机制族开始：一类只需要重新拟合当前输入的参数；另一类需要历史状态。初始观测应允许两种解释竞争，某些合法输入序列能区分它们。系统可重拟合、请求额外原始历史或新实验、构造小型关系程序，也可以停止。

例如，对某个相同的当前输入，历史上先升后降与先降后升是否导致不同输出，能区分无记忆阈值与迟滞机制。Agent 应自行选实验、从返回的时序中构造 lag/state 特征；环境接口不直接给好 `prior_on`，也不告诉它这次应调用哪类修复。

这里预先提供通用历史访问、阈值和组合算子是合理的。第一段支持的主张就是“预算下选择观测与组合表示”，不要求它从零发明编程原语。先用知道机制的可执行参照确认这些权限足以解决任务；如果参照也无法区分，优先修正任务可识别性和预算。

**第二段：立即检验“形成的内容是不是知识”。** 在第一段能够构造可执行关系后，安排一小段相关新任务、无关任务和使旧关系失效的任务。记忆中保留程序或抽象体、输入要求、适用范围和可调用状态。观察系统是否主动复用、是否减少重新搜索，以及何时限制或撤销它。

对照是同样底层语言与工具、但每题重新搜索的系统；再移除学得条目观察收益变化。删除实验要同时处理依赖该条目的缓存结果，避免名义删除而能力仍由缓存保留。如果优势与该知识条目无关，就不能把收益归因于知识积累。

第一段成功后应尽快进入这一段，而非持续完善动作分类器。它是防止新路线再次缩回固定 treatment selection 的关键。

**第三段：加入 robustification 和 idealization。** 复用已有两分支的候选评价思路，让候选收益来自环境实际执行。固定外部损失与风险要求，再比较重拟合、扩大不确定集和使用更简单模型的得失。至少包含无需扩张的任务，以及预测已经足够准确、但值得简化的任务。

首个小实验选时序历史机制，是因为已有可执行 Boolean 程序便于复用、失败原因较容易辨别。它是走向 robust 与简化的桥梁。后续如果长期只在同一个迟滞例子里调参，仍然没有回答用户最初的模型边界演化问题。

**6. 对照、终点和停止条件应怎样具体化**

主要比较采用同一测量权限、底层算子资源和预算。预算分别约束环境调用与计算量；记录失败候选和前期概念构建成本。不要先用未经论证的权重把这些都合成一个“认知分数”。

| 要检验的贡献 | 最直接的对照 | 支持该贡献的结果 |
|---|---|---|
| 主动实验选择 | 同候选语言、同预算的固定或随机探测 | 未见任务损失更低，或达到相同性能所需测量更少。 |
| 何时改变表示 | 同底层可表达能力的普通程序搜索；另保留始终开放历史的参照 | 在需要历史的任务取得收益，同时避免在简单任务持续支付无用成本。 |
| 知识复用 | 同语言但不保留学得抽象、每题重搜；移除条目实验 | 后续任务实际调用新抽象，学习成本下降，且收益随条目移除而下降。 |
| 修订与适用范围 | 固定保留旧抽象的版本 | 在冲突或无关任务中减少错误迁移，限制或撤销后恢复表现。 |

旧 3 节点语言对新 5 节点语言的 0.75 对 1.0，可保留为现有容量扩张样例；它不单独证明新控制策略优于同容量搜索。普通 LLM/代码修改对照，在主张开始涉及“优于通用代码 Agent”时必须补齐；首轮纯机制实验可先用普通程序搜索，不必立刻建设开放代码 Agent 评测平台。

数据划分以任务机制和迁移关系为单位。只改 seed 或重采相同公式的行，支持同分布泛化；要声称更强迁移，就需要事先留出的参数区间、历史长度或机制组合，并说明具体是哪一种。各系统按相同任务序列运行，在相同节点评估知识状态。生成器内部机制标签帮助分析，主要终点由未见任务表现与成本决定，因为同一任务可能存在多种有效修订。

首轮明确一个主要终点，例如固定环境与计算预算下的整段任务累计损失；风险约束作为固定条件，其余指标解释机制。知识建立成本必须计入，同时报告后续单任务成本，才能看出复用从第几个任务开始抵偿。具体样本量、预算和最小值得继续的效应应根据开发阶段的成本与波动确定，再冻结确认设计。

如果同容量普通搜索更好，应接受当前动作选择没有净收益；如果主动探测不优于同预算对照，应报告主动实验贡献未获支持；如果形成条目却无跨任务节省，应报告尚未证明知识积累。确认结果不支持时，后续修订在开发数据上进行，并使用新的确认任务，不能反复调整后在同一确认集上宣布成功。

**7. 建议替换的核心研究命题与近期优先级**

建议将网页最后的核心命题改为：

> 面对可复现失配、尚未区分的重要解释或重复求解成本，Agent 能否在有限预算下主动选择实验，构造并修订可执行的表示、关系或模型假设，将有用部分积累为可撤销的知识，并在后续未见任务中，比拥有相同工具与先验资源的固定库或普通程序搜索方法降低累计损失和学习成本？

这保留了递归假设、表示与观测共同变化、robust 起点和主动简化，同时把“有没有知识生长”落到一个可检验的因果比较上。整体架构有研究价值；其中哪一项构成独立方法贡献，要由相应消融和对照决定，不能从模块数量或哲学名称推断。

近期主线建议是：先完成上述第一、第二段的小实验；C1 scoring replay 作为已有正式路线的独立工程处理；等有了认知收益，再决定需要接入哪些正式理论状态与确认协议。已有 C1–C4 的冻结顺序继续描述原来那组正式主张，新实验使用自己的开发与确认设计。这样保留已有成果，也不会让每个新概念试验都重新穿过整套签发链。

**局限与本次交付范围**

本次读取了指定八份文档的相关论证、关键源码和小型本地终态，没有重新运行 campaign、容器、签名校验或完整证据图回放；上述运行状态指本地终态记录所报告的结果。本次没有新实验数据，因此新路线的效果仍待检验，具体预算、样本量与效应阈值尚未冻结。外部文献只核对了方案涉及的少数原始来源，未开展完整的新颖性检索。

运行工作树存在 8 个未跟踪 Q1–Q6 CBOR 产物，当前文档工作树也已有未跟踪文件；本次仅新增这份复评，没有改动原建议、代码、冻结协议或运行状态，没有提交或推送。
