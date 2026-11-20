# Hegel M5 多尺度时间与共同子树计划评估

评估对象：[固定提交 e2b0fa97a](https://github.com/erzhu419/Assumption-Agent/commit/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b)，重点为 `hegel_v16_5_multi_precision_plan` 的新总蓝图 20 及同步更新的 03、04、06、07。日期：2026 年 11 月 20 日。

**总体判断：两项核心方向值得完整实现；当前文字还不能作为无歧义的最终施工合同。需要修改的是从经验走向确信、从局部验证走向使用权限的推断规则，不需要削减跨时刻、多尺度或跨领域部分共同结构。**

时间与尺度结合，可以让系统研究同一对象在不同窗口中呈现的不同规律；共同关系块可以让目标继承一部分已经研究过的结构，同时继续发现没有匹配上的部分。它们确实可以互相增强。但“慢就稳定可信”“匹配越深就必然更可信”“两个异质源消去巧合”“最大预测方差探针最快消除不确定性”“深层失败永远只改叶”都不能作为一般保证。

本次建议是有限的：修订四组推断与时间合同，把现有强能力接入新表示，再用保留 A/B/C 的一次联合评估确认收益和边界。没有理由为此回退到静态公式或单节点类比，也没有必要另起无限补丁路线。

# 一 方案真正有价值的部分

## 1 时间与多尺度相结合改变了可迁移知识的内容

过去的静态公式迁移主要回答“某个输入对应什么输出”。新方案允许迁移的是一段过程：受到什么扰动，经历什么暂态，哪些历史状态继续影响未来，以及观察窗口改变后什么被保留、什么被隐藏。

这使两个表面不同的领域可能在一个尺度上共享动态骨架，在另一个尺度上表现出不同机制。同一系统也可以同时拥有快响应、慢适应和长期结构改变；目标不必被永久归到一个单一领域节点。新 20 的 P/N/F 和时间折叠意图有明确的研究价值。[总蓝图时间部分](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L22-L71)

P/N/F 应理解为时间角色，而非只有三个取值的有限状态。历史摘要可以是轨迹、向量、事件序列、潜在状态或状态分布；将来可以是整段预测轨迹。没有必要为了人工可读性把它们压成三个标量，也不需要要求每个系统都存在冲击前平衡与冲击后平台。

## 2 部分共同结构保留了联合约束和开放差异

单点相似容易遗漏节点之间的关系。一个共享关系块可以同时约束变量角色、前提、子程序和后继候选；未匹配的部分仍然可以在目标中学习。这比复制整套源模型灵活，也比把若干独立相似度相加更丰富。

从一两个已经深度研究过的源系统提取这种先验，本身合理。一个源系统可能包含多次干预、许多轨迹和已确认的内部机制，不能把它等同于一个普通标量标签。应检验的是目标能否利用这些源经历减少新的研究成本。[总蓝图共同子树部分](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L75-L108)

已有的 `retrieve_candidate → bounded_trial → local_solver_guidance → validated_pruning` 分层也应该保留：大胆提出结构，花有限预算试验，依据目标证据逐步增加使用权限。新 20 明确要求多源收缩后还要目标确认，因此不能把它批评成“两个来源就自动硬剪枝”。[总蓝图权限说明](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L110-L123) [原四级权限合同](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/07_V16_6_Review_Audit_and_Horizontal_Wormhole_Architecture.md#L111-L115)

## 3 两者需要真正结合但不互相替代

时间尺度可以帮助寻找适合共享的关系，部分共同结构也可以帮助选择要观察的时间窗口。然而，多尺度本身不提供独立样本，不自动消除跨域偏差；多个共享节点本身也不补足尚未观测到的慢状态。

二者的强结合是：在不同尺度和历史条件下识别可共享的部分动态结构，再让目标中的反例决定保留、展开或分叉哪些部分。这个目标比“慢变量放树干，快变量放树叶”的固定划分更符合方案的原始雄心。

# 二 优先处理的问题及性质

| 项目 | 当前表述的风险 | 判断 | 保留机制的修订 |
|---|---|---|---|
| 慢尺度树干准入 | 用时间类别代替机制稳定性和可迁移性 | 若落实成标签硬拒，会排除可共享的快动态规律 | 慢尺度作为线索，准入依据条件化结构支持 |
| 共同基底永不降级 | 从末端失败直接假定只错最后一条边 | 缺归因前提，与原五因修订不一致 | 保留仍独立成立的部分，修改最小受影响依赖子图 |
| D_eff 触发收缩 | 表面异质分数没有校准到参数异质性 | 有用启发式尚不是收缩保证 | 分开来源依赖、结构支持和参数分布 |
| 层数增加先验必增强 | 重复、相关或反向证据被当成更多支持 | 兼容集合缩小不等于置信度必增 | 按新增的条件证据更新 |
| 方差热点一两针 | 裸方差可能偏好与目标无关的差异 | 可用候选策略，不是一般最优价值器 | 接入已有任务风险价值与停止规则 |
| 时间折叠成常数 | 把成立的特例当作一般粗粒化语义 | 一般消元可能保留记忆和隐状态 | 允许常数、低阶动力学、记忆核等多种产物 |
| 过去现在将来 | 相对时间角色未定义信息截止点 | 需要补齐在线与回顾的操作语义 | 区分历史压缩、状态估计、未来预测 |
| 树层与时间视野对应 | 可能把精度、历史需求、尺度混成同一轴 | 应明确为一种研究路线而非普遍层序 | 继承四契约和有类型的局部关系 |
| 终局实验 | 静态或盲搬参数对照不足以代表强普通方法 | A/B/C 值得保留，归因要更精确 | 同信息动态强基线加能力消融 |
| 全面击败与零错误 | 有限构造流的结果被写成普遍结论 | 需收窄证据外延 | 报告任务族、覆盖、成本、观察到的失败数 |

上表不是十项新工程。多数可归入时间语义、共同结构与不确定性、探针与反例、联合验收四组；其中多项已有实现基础。

# 三 时间语义需要区分的四个量

新 20 用同一个尺度 τ 承担了若干不同含义。至少应区分以下四个量，并保留事件锚点和信息截止时刻：[总蓝图第 24 至 71 行](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L24-L71)

| 量 | 含义 | 为什么不能混用 |
|---|---|---|
| Δt | 采样间隔 | 改采样率不等于改变系统动力学 |
| τᵢ | 系统自身的松弛或适应时间 | 一个系统可同时具有多个 τᵢ |
| w | 聚合或状态提取窗口 | 长窗口不保证可以丢弃历史 |
| H | 预测时距 | 短期通过不自动授权长期结论 |

还应知道事件何时发生、截至何时允许使用观测、是否复位以及初始历史来自哪里。这些内容可以进入现有 `ScaleMap`、`ObservationMap` 和 `protocol_spec`，不需要另写平行状态机。

## 1 不规则与连续时间值得支持但离散下标并非原则错误

新 20 说采样率变化会使离散公式直接失效，过强。对连续模型：

```math
\dot x=-\lambda x
```

精确采样关系是：

```math
x_{k+1}=e^{-\lambda\Delta t}x_k.
```

只要实际间隔保留并正确改变转移参数，离散表示仍然成立。真正要避免的是丢失时间单位、间隔和归一化条件后硬搬参数。P/N/F 可以作为跨领域提问接口，底层仍可使用差分、微分方程、事件系统或学习到的状态模型。

快慢也具有相对性。毫秒响应和百秒响应，按各自 τ 归一后可能有同样的过程结构。若 τ 需要从目标数据估计，它的成本与不确定性应随模型传播，不能由隐藏真值无偿提供。

## 2 时间折叠不能改变信息何时可获得

已经完整发生的源经历，可以将其快过程前中后阶段折叠成慢尺度的历史知识。目标在线预测时，尚未发生的未来仍然是待预测或待购买的实验结果。

建议明确三种操作：

1. **回顾压缩**：对已完成轨迹提取有效关系。
2. **当前状态估计**：只使用截止时刻以前的信息估计状态。
3. **未来预测**：根据当前信息输出未来轨迹或其分布。

可写为：

```math
(\mathcal P_\tau,\mathcal N_\tau)=G_\tau(\mathcal H_{\le t_0}),
\qquad
\mathcal F_{\tau,H}\sim p(\cdot\mid\mathcal H_{\le t_0},q).
```

如果用包含 t₀ 以后样本的居中窗口定义“宏观现在”，随后预测同一窗口末端，就把回顾分析误当了在线预测。若分区算法需要未来峰值才能知道当前处于哪一相，也存在同类问题。

这是新计划尚待明确的操作语义，**并非本轮发现当前代码已经发生未来泄漏**。同理，时间评估应按完整轨迹或独立源簇划分，避免相邻时间片被随机分到训练和测试后造成假泛化。

# 四 时间折叠应允许保留记忆和动态闭合

## 1 折成常数是重要特例但不是一般结论

新 20 的快弛豫等效刚度例子有依据；把它提升成“向上粗粒化等于时间折叠成常数”的一般规则则不成立。[总蓝图第 55 至 60 行](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L55-L60)

一个可直接推导的例子是：

```math
\dot x=-ax+bz+u,\qquad \dot z=cx-dz.
```

消去 z 得到：

```math
\dot x(t)=-ax(t)
+bc\int_0^t e^{-d(t-s)}x(s)\,ds
+be^{-dt}z(0)+u(t).
```

保留下来的是记忆卷积和初始状态项。取 a=d=2、b=c=1，系统稳定；同样的 x(0)=0 配上 z(0)=+1 或 −1，会产生相反的初始变化。若只保留一个静态 x 值，就丢掉了计划希望表达的“同现在、不同前史、不同将来”。

因此 `Π_fold` 的产物应可包括有效常数、代数约束、低阶动力学、带记忆核的动力学或扩充隐状态。只有满足相应稳定性、快慢分离、驱动频段和初始层条件，才采用无记忆的近似。

Mori–Zwanzig 的一般投影表示包含记忆项；忽略它们需要额外条件。van der Schaft 的快慢消元也明确把快变量近似达到稳态作为假设。这些来源支持条件化简化，不要求本项目实现一个通用 Mori–Zwanzig 求解器。[Chorin 等 2000，式 2.7 与第 3 节](https://math.huji.ac.il/~razk/Publications/PDF/CHK00.pdf) [van der Schaft 2015，第 11.4 节](https://pure.rug.nl/ws/portalfiles/portal/64854097/Schaft2015_Chapter_PhysicalNetworkSystemsAndModel.pdf)

仓库原设计其实已经更完整：07 明确聚合不自动闭合，并保留 CEGAR 补状态；M4.1 的实际消元结果是一维慢 ODE。[原闭合合同](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/07_V16_6_Review_Audit_and_Horizontal_Wormhole_Architecture.md#L104-L109) [原状态细化规则](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/07_V16_6_Review_Audit_and_Horizontal_Wormhole_Architecture.md#L124-L132) [M4.1 实现](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L86-L187)

## 2 局部刚度软化不等于所有长期反转

对于内部平衡状态满足 F_z=0 且 F_zz 可逆的局部问题，有 Schur 形式：

```math
K_{\mathrm{rel}}=F_{uu}-F_{uz}F_{zz}^{-1}F_{zu}.
```

进一步得到非负和软化不等式，还需要相应凸性、稳定性与共轭读出等条件。这不自动推出经济刺激、耐药或所有适应系统都具有同一个被动松弛机制。这里无需把应用类比删除；可以让它们共享快慢响应骨架，再在反馈符号、读出与适用前提上分叉。

例如：

```math
y(t)=a(1-e^{-t/\tau_f})-b(1-e^{-t/\tau_s}),\qquad \tau_f\ll\tau_s.
```

当 a/τ_f>b/τ_s 且 b>a，初期正向，长期趋于负值。这是两种对读出作用相反的模态，不应直接归入所有被动共轭松弛命题。它正适合检验“共享部分结构，但更深层分叉”。原 07 已经区分局部 Schur 刚度和有限大步长单调性，应继续以该边界为准。[原数学修正](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/07_V16_6_Review_Audit_and_Horizontal_Wormhole_Architecture.md#L96-L102)

## 3 显式时间标签不能识别尚未被观察到的慢机制

考虑两个候选：

```math
y_1(t)=1-e^{-t},\qquad
y_2(t)=y_1(t)-2(1-e^{-t/T}).
```

在 0≤t≤H 内，两者差不超过 2H/T。T 足够大时，固定短窗与噪声水平下无法分辨；长期极限却分别是 +1 与 −1。

所以绑定 τ 和 H 可以防止语义上把短期结论当作长期结论，不能保证统计上“再也不会误判”。正确能力是保留慢机制的不确定性，寻找可观察的慢状态、延长观察或改变干预；无法区分时给出与证据相称的预测范围。

# 五 时间信息深度与树的承诺层次不应合并成唯一轴

新 20 的“静态截面 → 变迁对比 → 历史条件化”是一条有意义的研究路线，但不应被解释为所有哲学树的普遍层序。[总蓝图第 34 至 52 行](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L34-L52)

需要历史多少、保留哪些物理尺度、承诺多精确，是三个不同问题。一个粗略的迟滞符号命题可以依赖很长历史；一个高精度几何命题可以是静态的。加入隐状态后，原先在观测变量上非马尔可夫的过程，又可以在扩充状态上成为马尔可夫过程。

因此局部树仍可保留为研究与导航结构，但应继承原有四契约和有类型关系：逻辑特化、状态扩充、粗粒化、历史摘要、前提依赖、证据支持分别处理。空间聚合、拓扑约化或群体统计也不必全被解释成时间折叠。[原四对象及边类型](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/04_Hegel_V16_5_Architecture_and_Execution_Master_Plan.md#L42-L50)

这不是要求取消树或强改全局大图。原内核已有 typed DAG/森林；需要的是防止说明性层级图覆盖原本更宽的语义。

# 六 部分共同子树还需定义联合映射和生成过程

## 1 连通关系块需要一致的角色绑定

假设源结构是 u→z→y。第一条边映到目标 a→b，第二条边映到目标 c→d。若 b 与 c 不是同一个状态，也没有受支持的连接，两个单边匹配不能直接拼成一个共同子树。

所以共同结构至少要在共享节点、输入输出、时间锚点、读出和尺度关系上有一套可以共同成立的局部映射。每条边独立选择最容易匹配的解释，可能得到一个并不存在的联合机制。

这不要求完整同构，也不要求每棵树从根开始匹配。共同片段可以在中间层，可以跨不同绝对深度；未匹配处可以是自由槽或目标专属分支。潜在表示与神经模型可以参与提议映射，只需其执行输入输出和所声称的联合关系能够检验。

## 2 背景条件叠加不等于因果链

城市例子可以保留为类比说明，但“气候相似”和“同处内陆、邻近经济中心”不是自动的因果先后或逻辑蕴含关系。若想表达条件逐步增加，应是 C₁ 与 C₂ 联合限制下游预测，而不是默认 C₁ 推出 C₂。

原 04 已经有 `requires_premise_and` 与 `supported_by_or`，新子树直接继承即可。这里的要求是明确每条关系的含义，不是用地理或社会机制的复杂性否定跨领域联想。[原多前提与多源关系](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/04_Hegel_V16_5_Architecture_and_Execution_Master_Plan.md#L45-L50)

## 3 新数据类不是共同结构提取算法

目前计划描述了 `SharedBaseSubtreePrior` 的使用方式，但还需说明：共同片段怎样从源研究中抽出、目标角色怎样由已购观察提出、候选映射如何进入检索与后续实验，以及错误映射如何被修订。

仓库 `branch_reconciliation.py` 的 typed AST anti-unification、参数槽、逐源重构和 fresh 支持检查可以作为基础。它们不等于已经完成全部跨层共同子树语义；这属于集成任务。[现有 anti-unification](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py#L1844-L2008) [现有支持与确认检查](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py#L2072-L2200)

同样，当前 M5.0 的写回是结果驱动的，但跨域候选仍有预定 cue→tree/node/target-cue 表。“零步骤号硬编码”可以成立，却不是开放世界中任意共同结构已经能自主形成的证明。[M5.0 候选表](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/m5_0_closure_and_continual_sublation.py#L168-L199) [写回候选形成](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/m5_0_closure_and_continual_sublation.py#L471-L615)

# 七 共同基底的推断与方差治理

## 1 慢尺度做树干应成为偏好而不是标签充分条件

原文有合理目的：不要把一次偶然热潮当作持久机制，也明确给已验证守恒和拓扑不变性留了例外。问题是 `temporal_horizon_class` 本身不足以区分稳定生成规律与偶然观察取值。[定律一](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L112-L114)

一个背景在某域里百年不变，仍可能在另一个域完全不同；一个稳定混杂因素可以制造持久伪相关。反之，一个变化极快的响应，其生成规律可能高度稳定并可跨域复用。两个一阶系统按 t/τ 对齐后，可以共享同一个快速动力学核。

若把 `fast_transient` 直接作为硬拒标签，就会误伤方案最希望保留的跨时间尺度动态联想。应改成：**共同基底候选可来自任何尺度；被用于迁移的关系须具有相应条件、角色与范围支持。慢尺度只是稳定性和搜索价值的一项线索。**

要分开时间上的波动、测量噪声、估计不确定性、跨环境机制异质性。低通滤波主要改变第一项，不会自动解决其余三项。跨环境不变推断文献的前提是条件机制等假设成立，而非领域名称相差很远。[Peters 等 2016，假设与识别条件](https://arxiv.org/pdf/1501.01332)

## 2 D_eff 可以衡量多样性但尚不足以决定参数收缩

原文称其“有效异质支撑度”，可善意理解为启发式评分，不必强行要求它等于某个经典有效样本量。真正的缺口是：为何该分数可以支持数值先验收缩，为什么阈值 2 有特殊保证。[定律二](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L116-L123)

### 顺序和来源依赖

原式尚未定义一个源与“此前所有源”的集合相似度。若采用常见的最大相似度，并设 s(A,B)=0.9、s(A,C)=0.1、s(B,C)=0.2，则：

- 顺序 A,B,C：D_eff=1+0.1+0.8=1.9。
- 顺序 A,C,B：D_eff=1+0.9+0.1=2.0。

同一证据集合换入库顺序就越过阈值。这是针对一种可能实现的条件反例，**不是当前尚未实现算法的实测 bug**。它提示该分数应有集合级、可复现的语义。

更关键的是，把同一轨迹换成流体、经济、队列三种外壳，不会创造独立信息；同域中两个独立且充分干预的系统，反而可能提供更多支持。来源依赖不能由表面相似度替代。

### 结构相同与参数可池化是不同结论

两个领域都满足同一种方程，并不意味着无量纲参数来自同一窄分布。在一个简单层级模型中，设：

```math
\theta_e\sim N(\mu,\sigma^2_{\mathrm{between}}).
```

则该模型下新领域参数的预测方差为：

```math
\operatorname{Var}(\theta_{\mathrm{new}}\mid E)
=E[\sigma^2_{\mathrm{between}}\mid E]
+\operatorname{Var}(\mu\mid E).
```

源增加可以改善共同均值估计，却不会自动消掉域间异质性。只有少数源时，方差参数的不确定性还可能很大；“宽泛无信息先验”也不是自动稳健的默认答案。[Gelman 2006，层级方差先验](https://sites.stat.columbia.edu/gelman/bayescomputation/Gelman2006.pdf)

建议保留多样性分数作为检索、探索或候选优先级，实际共享权重依据源内证据质量、来源独立性、机制支持和参数异质性决定。可用层级或混合先验，也可用更简单的加权候选机制；不必指定唯一统计框架。

“单源先只借结构”是合理默认，但不是绝对定理。若有单位转换、标定或不变量推导，单源也可能合法支持数值转移；反之，两个外壳不同的源也可能不应共享参数。

## 3 多匹配一层可以缩小空间但不能直接增加置信度

若新匹配层提供真实的额外约束，兼容的补全集合可能变小：

```math
\mathcal H_{k+1}\subseteq\mathcal H_k.
```

但这不等于后验置信度必然增强。如果第二层只是同一条源证据的重编码，再加一次支持就是重复计票；如果它更支持另一种机制，正确做法反而是降低原假设的权重。

因此应写成：**每多匹配一段受到支持且与目标相关的联合关系，就有机会减少补全歧义；先验强度由新增条件证据决定。** 这保留逐层利用的能力，也要求系统真正表示关系与证据依赖，而非退回“每层加一分”。[原逐层表述](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L78-L104)

## 4 最大预测方差不等于最大实验价值

方差热点非常适合作为便宜的候选探针策略，但不能保证是“如果先验猜错，最可能出错的地方”，也不保证一两针最快坍缩目标不确定性。它还取决于候选权重、观测噪声、目标损失和真机制是否在候选集合里。[定律三](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L125-L128)

仓库原有测试已经给出一个直接例子。四个候选对目标的预测为 −1、+1、−1、+1：

| 探针 | 四个候选对探针的预测 | 裸预测方差 | 当前实现的预期目标风险下降 |
|---|---|---:|---:|
| A | −100、−100、+100、+100 | 10000 | 0 |
| B | −0.1、+0.1、−0.1、+0.1 | 0.01 | 2 |

A 的幅度分歧大，但不帮助选择目标预测；B 的数值差小，却能分开真正影响目标的两组候选。在既定低噪声设置中，已有 `value_forecast_experiments` 正确选择 B。本轮重跑该原始测试文件，4 项通过。[原测试](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/tests/test_forecast_experiment_value.py#L11-L35) [已有任务价值器](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/probing.py#L641-L719)

因此优先复用已有价值器：

```math
q^*=\arg\max_q
\frac{R(D)-E_{o\mid D,q}[R(D\cup\{q,o\})]}{C(q)}.
```

需要辨识机制时可用信息增益；需要选择行动时用目标风险。原始方差可留作低成本筛选。信息增益本身比较的是期待后验变化，也不等于裸预测幅度。[Houlsby 等 2011，第 2 节](https://mlg.eng.cam.ac.uk/pub/pdf/HouHusGha11a.pdf)

一两针可以是某些任务的优秀实测结果或预算目标，不能成为通用保证。若八种候选需要八种不同动作，每个探针最多产生一个二值结果，两针至多四条结果路径，不可能全面区分。若所有候选共享一个错误基底，内部方差还可能为零。

时间任务的 C(q) 也应包含初始化、复位、运行和等待成本。原决策区域停止规则可以继续使用：只要剩余候选对当前行动等价，就不必继续追求唯一微观解释。[原停止规则](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/04_Hegel_V16_5_Architecture_and_Execution_Master_Plan.md#L55-L60)

## 5 局部劈叶需要归因前提而不是祖先免疫

如果已独立确认前 k 层在当前目标范围内仍成立，并且反例确实只否定第 k+1 层，保留共同基底完全正确。需要补齐的正是这个“如果”。[定律四](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L130-L131)

仅有末端失败时：

```math
B\land C\Rightarrow Y,\qquad
\neg Y\Rightarrow \neg B\lor\neg C.
```

不能推出必定是 C 错。错误也可能来自目标角色映射、丢失的慢状态、上游守恒前提或粗尺度闭合。

例如基底声称当前子系统闭合守恒，叶分支声称交换率正比于差值；实测总量持续流失时，仅改交换系数无法挽救目标的闭合前提。可撤回目标中的绑定、加入外部汇或重查读出，同时保留源域原先有效的守恒证据。

建议将规则改为：**根据归因证据修订最小受影响依赖子图；独立支持且作用域未受影响的共同部分保留。无法定位时，保留几个竞争诊断并用探针区分。**

原 07 已有五因竞争诊断、CEGAR 补状态、父节点仍成立才回退以及父节点失效时退出。新“零降级”不应覆盖这些原本正确的能力。[原双向修订](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/07_V16_6_Review_Audit_and_Horizontal_Wormhole_Architecture.md#L124-L133)

# 八 当前实现能支持什么结论

## 1 M5.0 确实有新进展

本次对固定提交的原始回执做只读复算：144 项 SHA 清单全部匹配，Part 3 有 140 条唯一主流回执，即 5 臂各 28 题；另有 3 条冷恢复回执。每臂包含 25 个域内任务与 3 个指定域外或反例任务，提交结果均达到相应质量要求，记录的假确认为 0。五臂 C、K 加总与提交 summary 一致。

| Part 3 方法 | 测量费 M | 计算费 K | 总复合成本 C | 候选评估数 |
|---|---:|---:|---:|---:|
| Hegel PPR PoolOn | 819 | 730.53 | 855.5273 | 55 |
| Compiled CueTable | 819 | 735.86 | 855.7930 | 69 |
| Hegel PPR PoolOff | 819 | 874.53 | 862.7265 | 79 |
| Dynamic Flat | 868 | 567.74 | 896.3870 | 63 |
| Frozen OpenRediscovery | 883 | 1828.45 | 974.4229 | 238 |

K 在表中按两位小数展示；C 使用提交中的汇总精度。[本次工作总结](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/19_V17_7_M4_5_Closure_and_M5_0_Continual_Sublation_Report.md#L67-L110) [原 Part 3 汇总](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/docs/evidence/v17_7_m5_0_closure_and_continual_sublation_20261120/part3_m5_0_continual_sublation_summary.json)

PoolOn 相比 PoolOff 省 7.1992 C，这是本次任务流里真实的端到端池化收益。相比 Compiled 省 0.2657 C，幅度很小；同次提交 Part 2 中 Compiled 又比 Hegel 便宜 4.9524 C。因此可以肯定进展，不能扩大为所有强普通方法被全面击败。[Part 2 的边界报告](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/19_V17_7_M4_5_Closure_and_M5_0_Continual_Sublation_Report.md#L58-L65)

## 2 Flat 对照混合了组织方式和能力授权

当前 Flat 初始化时删除横向虫洞、粗粒化边和局部蕴含边；第二域候选槽的写回也与图臂不同。因此它失去的不仅是检索组织，还包括可执行关系与粗通道授权。[Flat 初始化](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/m5_0_closure_and_continual_sublation.py#L1568-L1573) [第二域写回差异](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/m5_0_closure_and_continual_sublation.py#L540-L615)

逐题复算得到：

- 所有 49 M 测量费差都来自 S27 的 PiS Adiabatic 任务。
- 该题 PoolOn 的 M=21、C=21.765；Flat 的 M=70、C=70.71。
- 单题 C 差为 48.945，而整流 C 差为 40.8597。
- 其余 27 题合计 Flat 反而便宜 8.0853 C。

这并不否定多尺度关系的价值；恰恰说明这项价值在该流里很集中。它不能单独证明同内容、同证书、同工具下“图这种表示形式”独立节省 40.8597 C。新 M5 应把组件能力消融与同信息强基线比较分开。

## 3 新计划的核心仍处在待实现阶段

`temporal_horizon_class`、`regime_partition_op`、`SharedBaseSubtreePrior`、`D_eff` 和新热点选择器尚未在当前 `src/tests` 中形成实现。这符合总蓝图“待启动”的状态，不是本次提交漏交代码。当前 M5.0 主要提供持续写回、作用域修订、关系晋升和冷恢复的承载基础。

字段进入语义哈希可以防止身份混淆，却不会自动产生历史状态估计、共同子树抽取或先验消费行为。M5 的实际增量应落在这些消费者上。[当前 M5 定位与施工入口](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L139-L157)

# 九 一个能同时检验两项核心的联合案例

保留 A/B/C，并让至少一个案例在同一条持续流中同时需要时间、多尺度、部分共享和反例修订。

可采用以下**建议性实验结构**：

1. 源 A 具有快响应与慢反馈，经过已有研究形成局部动态关系块。
2. 源 B 与 A 共享其中的快响应骨架，但读出符号、反馈机制或慢状态并不完全相同。
3. 目标 C 的早期行为与二者都相似；仅看短窗口无法判断长期方向。
4. 系统通过已有历史与有价值的目标探针，保留共享部分，在真正不匹配的慢反馈处分叉；若反例指向旧闭合或目标映射，则允许修订该部分。
5. 后续改变快慢比或观察窗口，检查系统何时采用粗模型、何时补回状态，以及冷恢复后是否仍正确消费这些经历。

这能区分“共享一段过程”与“因为表面相似搬一个数”，也能观察时间视角如何改变共同结构的范围。

若已声明存在细、粗两种跨域映射，还可比较两条路径在保留查询上的一致性：

```math
\Phi_{\mathrm{coarse}}\circ\Pi_A
\approx
\Pi_B\circ\Phi_{\mathrm{fine}}.
```

这里仅检查**双方都已声称可用的路径**及累计误差。宏观涌现的共同性可以根本不存在微观同构；不应把这条检查变成所有跨域迁移的前置门槛。

# 十 保留 A B C 但让验收回答正确的问题

总蓝图最后已经要求相同工具、原始历史和计费标准，这是正确的。具体场景设计也需要贯彻这一要求，不能用信息不足的消融替代强普通基线。[终局实验及验收](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L159-L171) 端到端系统比较应从相同原始历史出发，计入各自形成、维护和消费知识的成本，不能把 Hegel 已经发现的结构免费交给对照后再声称比较了发现能力。只有单独测组织与检索方式时，才固定双方已经形成的关系事实和证书内容；下表中的同内容对照属于这个层次。

| 场景 | 保留的核心任务 | 能力消融的作用 | 关键强对照及判断 |
|---|---|---|---|
| A 跨时刻与路径记忆 | 同当前读数、不同历史、不同未来 | 无历史静态模型说明历史必要 | 强动态或状态空间方法也得到同样历史和工具；比较结构利用与形成成本 |
| B 多尺度折叠与反转 | 快慢分离、闭合失效、短长时域分歧 | 固定粗模型暴露过度简化 | 完整细模型与自适应粗细模型共享通道和证书事实；比较相同质量下总成本 |
| C 部分共同结构 | 一层与多层联合结构、目标分叉 | 单节点与盲搬参数说明各机制作用 | 强基线可存共同依赖、保留不确定性并主动探针；比较同源历史的形成与消费收益 |

有限的任务族应包括一些会区分机制的反例：快但可共享的规律、慢但不可迁移的背景、结构共享但参数不同、第二层只是重复证据、深层失败实际来自上游、候选内部一致但共同失配。它们可作为同一组封存任务的分支，不需要变成六套大型基准。

比较一层和两层时，还应说明增加了哪些信息和参数约束。若两层臂多拿了一个有效变量，性能提升首先说明额外信息有用；只有同等输入、预算或明确的增量归因，才能进一步支持层级组织的价值。

源系统数、源内干预数、轨迹数和目标新探针数应分开报告。一个深入研究过的案例不等于一个监督标签。重复同源参数变体也不应算成同样数量的独立领域。

## 验收可以有限但结论必须与实验相称

可以要求指定回归集全通过、指定域外集诚实返回 unknown、封存评测中观察到的假确认为零。同时约束域内覆盖、完成率与预算，防止仅靠拒答达到零错误。

有限数据不能证明未来总体错误率严格为零。即使在独立伯努利等较强条件下，n 次零失败对应的单侧 95% 上界仍是：

```math
1-0.05^{1/n}.
```

本项目还有源簇相关性和自适应选择，五臂 140 条回执更不能视为 140 个独立世界。这里无需追加一套庞大统计工程，只需准确区分测试红线与总体保证。

“同时击败四种基线”可以是预先规定的目标，不能成为不断换题、换计费或换基线直到胜出的理由。如果完整方案在某些结构族有稳定收益，在另一些场景正确回退或不划算，这仍然是有价值、可收官的 M5 结果。

# 十一 有限施工建议及已有能力复用

## 1 时间与尺度合同

在已有四契约中明确采样、动力学时间、窗口、预测时距、信息截止与复位条件。P/N/F 允许丰富摘要；Π_fold 允许多种有效模型。分类枚举负责索引，真正语义由可执行算子、参数与适用范围决定。

## 2 共同结构与不确定性

共同片段引用已有节点与有类型关系，保留联合角色绑定、来源依赖和未匹配部分。慢尺度、多样性、深度都可以参与排序；不直接等同于真值、独立支持或数值收缩。只共享受到支持的参数部分，保留局部差异。

## 3 探针与反例消费

把候选补全送入已有任务价值器，用决策区域控制停止。遇到反例复用竞争归因和依赖修订；确认只错叶时保叶外结构，确认上游有问题时允许修订上游。新候选与直接研究出口继续存在。

## 4 联合封存评估

保留 A/B/C，加入一个真正联合案例与少量关键负例；在冻结历史、工具、任务族、预算、计费口径后一次评估，报告收益、失效边界与未覆盖的范围。无需再拆成十几个补丁里程碑。

| 既有能力 | 可复用位置 | 本次需要的连接 |
|---|---|---|
| 历史递推和动态执行 | `language.py` 的历史累积、AR2、VdP 等 | 把时态与事件窗口接入实际执行 |
| Markov 充分性与隐状态前提审计 | `premise_dependency_audit.py`、`world_structure_commitments.py` | 将历史失配回流至共享结构和尺度判断 |
| 快慢消元与闭合范围 | `m4_1_unified_discovery_and_renormalization.py`、`wormhole_forest.py` | 支持时域、状态和尺度变化后的实际消费 |
| AST anti-unification 与分支修订 | `branch_reconciliation.py` | 从程序骨架走向联合的局部关系块 |
| 任务相关实验价值 | `probing.py` | 消费共同子树产生的候选及预测 |
| 来源、实例与持续写回 | 当前 M4.5/M5.0 森林与 checkpoint 路线 | 持久化并恢复新的时间与共同结构语义 |

这条路线的实质是接通已经存在的能力，并补上新的联合语义。已有模块不等于联合能力已经完成，但也没有必要把历史资产当成不存在再做一遍。

# 十二 结论

**可以继续推进完整的 M5 方向。当前最值得保留的，是按尺度和历史组织过程知识，以及只迁移共同部分、让差异继续生长的研究机制。**

现在不宜把 20 的“四条治理律”原样变成固定推断规则。将它们改成具有明确前提、可检验效果和可撤回结论的机制，会让系统更接近原本设想的开放研究者：它能够提出大胆跨域猜测，也能够发现自己共享得太多、折叠得太早，或者沿错了共同树干。

共同结构不需要永远正确才能有价值。能够根据后继经验保留有效部分、重构错误部分，并使下一轮研究更省成本，才是本计划中更有说服力的持续自我改进目标。

# 附录一 本轮核验范围

本轮完整阅读更新后的总蓝图 20、配套 00 至 07、当前工作总结 19 及 `Gemini_dialogue.md`；按照仓库 AGENTS.md 的历史复用要求定位相关历史机制，核对时间状态、部分结构、反例修订、主动探针和当前持续流实现。参考文献核验集中在与本次推断直接相关的一手论文及章节，没有把“全部历史版本、全部论文全文重新读过”作为本轮完成声明。

实际执行包括：原提交证据清单和账本复算，现有 `test_forecast_experiment_value.py` 的 4 项测试，以及两个已提交实例的只读冷载分发控制。没有重跑完整 M5.0 长基准，也没有修改仓库源代码。

附件中的 `integration_evidence_recalculation.json` 保存逐题费用归因；`evidence_readonly_check_summary.json` 为简表；`midpoint_scope_cold_control.json` 保存冷载控制及其限制；复算脚本可以在固定提交及原依赖环境中再次执行。

# 附录二 少量探针不能自动认证整段边界的具体例子

M5.0 在部分作用域修订中取“最大支持点”和“最小反例点”的中点写入前提。当前实例的支持点 1.45、反例点 2.25，得到 b≤1.85。[边界生成代码](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/m5_0_closure_and_continual_sublation.py#L420-L454)

S02 目标函数另含 `2.5*max(0,b-1.55)^2`。在 b=1.80 时，这一项是 0.15625，大于容差 0.05，但仍落在 b≤1.85。若基础项系数已经正确，这说明该中点不构成整个 S02 基础族的安全范围证书；误差达到 0.05 的界点约为 1.691421。[S02 目标定义](https://github.com/erzhu419/Assumption-Agent/blob/e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b/Hegel%20Machine/src/hegel_machine/cognition/m5_0_closure_and_continual_sublation.py#L1105-L1125)

为避免混淆，本轮还实际冷载原保存实例并调用正式 dispatcher。b=1.40 与 1.80 均获授权，且预测原源公式正确，无 fit 调用。但保存实例参数为 9.0/1.6/0.72，S02 目标参数为 7.2/1.85/0.58，二者不是同一拟合实例。

**因此这里不能报告一条新发生的 S02 假确认。实际控制证明的是授权范围；遗漏项给出的是边界认证的解析反例。** 该例支持的设计建议很具体：支持与否证之间的空白可以保留 unknown，或用明确的单调性、误差界和额外边界探针获得更大范围；不能仅因一两针成功就把整个连续区间视作已证。

# 附录三 一手理论参考

- Chorin, Hald and Kupferman. 2000. *Optimal prediction and the Mori–Zwanzig representation of irreversible processes.* PNAS 97(7):2968–2973. [作者公开全文](https://math.huji.ac.il/~razk/Publications/PDF/CHK00.pdf)。本报告使用其一般投影记忆项及忽略条件。
- van der Schaft. 2015. *Physical Network Systems and Model Reduction.* Mathematical Control Theory II:199–219. [大学仓储全文](https://pure.rug.nl/ws/portalfiles/portal/64854097/Schaft2015_Chapter_PhysicalNetworkSystemsAndModel.pdf)。本报告使用第 11.4 节的快慢分离与 Schur 消元条件。
- Peters, Bühlmann and Meinshausen. 2016. *Causal inference using invariant prediction: identification and confidence intervals.* [作者预印本](https://arxiv.org/pdf/1501.01332)。本报告使用其条件不变性假设，不把因果识别当作所有预测迁移的必要条件。
- Gelman. 2006. *Prior distributions for variance parameters in hierarchical models.* Bayesian Analysis 1(3):515–533. [作者全文](https://sites.stat.columbia.edu/gelman/bayescomputation/Gelman2006.pdf)。本报告使用其少数组下的层级方差先验问题。
- Houlsby, Huszár, Ghahramani and Lengyel. 2011. *Bayesian Active Learning for Classification and Preference Learning.* [作者全文](https://mlg.eng.cam.ac.uk/pub/pdf/HouHusGha11a.pdf)。本报告使用第 2 节的信息增益形式。

上述来源分别支持有限的技术结论，不作为“整个 Hegel 计划已被文献证明”或“跨领域类比不可行”的权威背书。报告中的线性消元、短窗长时域分歧、顺序敏感和局部反证例子为本次直接推导。
