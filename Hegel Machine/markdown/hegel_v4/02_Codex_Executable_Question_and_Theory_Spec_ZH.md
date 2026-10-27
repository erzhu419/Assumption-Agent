# 黑格尔机：可执行问题生成、假设复用与理论谱生长

## Codex 实施规格 v0.1

**日期：2026-10-20**  
**基线：`erzhu419/Assumption-Agent`，`codex/hegel-cognitive-mainline`，`468256bfc7e9d0c6077f29a0475e7e065d24c6d6`。**  
**文档性质：建议的新增算法与实验合同；不是已执行的项目事实，也不是旧结果的追溯性变更。**

数学与研究依据见 `01_Concepts_Mathematics_and_Research_ZH.md`；来源见 `03_References_and_Source_Boundaries.md`。

---

# 0. 任务边界和唯一主线

本轮要改变的不是记录数量，而是以下执行链：

```text
明确的长期目标与当前任务
→ 生成有答案语义的研究问题
→ 匹配旧理论并判断当前适用性
→ 必要时产生带父用途的子研究／候选结构
→ 真实证据更新具体承诺
→ 子答案使父计算或实验实际改变
→ 从多次研究归纳可复用对象
→ 在独立后继问题中使用、检验和修订
```

本轮不实施：通用数学证明器、通用无穷语言 closure、新签名／Merkle 治理、全量资料每轮重读、让 Agent 更改外部成功标准、或强制所有认知变化满足收缩性。

一个可验收产物必须至少改变预测、研究行动、理论复用或任务终态之一。新增字段、显示提示、工具被调用、图节点增加均不能单独作为科学成功。

## 0.1 必须保留的现有能力

当前已经有数值检查、`bind_research`、参数消费、依赖失效、前瞻冻结、真实表示碰撞检查、整轨迹 refit、历史模型与方法库。复用它们，不要以新名称复制一套。[G2–G5]

付费方法修订线是**外层作者＋内层零新增 LLM 实践**。新路由必须在 `provider=None` 时仍能完成可计算检查；生成任务只能返回外层作者，不能偷偷调用不存在的 live 子会话。[G2]

旧34/34、正式12源负比较、source006未决、各版本费用、历史runtime均保留。新实验使用新运行身份，不修改旧源码与证据来制造历史成功。[G3]

## 0.2 研究中不强求唯一机制

允许多个候选、集合答案、条件性结论和未知。只要不同解释对当前任务给出足够一致、经适当支持的行动，系统可完成当前任务，同时保留理论问题。

不能把操作成功写成机制唯一识别；不能因机制未唯一确定就拒绝比较一项有可测试后果的修订。

---

# 1. 四种不能混淆的身份

| 身份 | 固定什么 | 允许怎样变化 |
|---|---|---|
| 任务／研究使命 | 要预测、解释、控制或归纳什么；外部评价和合法接口 | 只能显式开新版本，不得静默改成功含义 |
| 问题语义 | 所求量／关系、对象角色、范围、答案类型、父用途 | 改研究算法不改问题；改所求或范围需新语义版本 |
| 假设族 | 结构、表示合同、允许参数、支持前提 | 参数拟合产生实例；结构／表示／作用域改动有显式修订 |
| 假设实例／应用 | 具体源码、参数、拟合数据、绑定、执行版本和场景 | 每次重拟合或重绑定产生新实例／应用，旧支持不自动迁移 |

现有 question ID 继续做会话定位。不要把它改成理论ID；也不要用模型路径判断科学同一性。

同一问题可以比较多个假设；同一假设可用于多个问题；竞争假设不等于支持性子假设。

---

# 2. 最小共同语义：适配旧对象，而不是重写整个系统

以下五类记录是逻辑角色，可映射到当前JSON／对象与数据库；不要求创建五个微服务。

## 2.1 ResearchMission：保护宏观目标不被“还能怎样提高”替换

必需内容：

```yaml
mission_id: stable_identifier
purpose: task_prediction | intervention | explanation | theory_learning | method_learning
target_definition: 固定的外部所求与解释
legal_actions_ref: 公开接口版本
terminal_evaluation_ref: 不由研究者静默修改的评价
scope_ref: 任务分布／来源／可访问范围
resource_contract_ref: 现有预算合同
theory_learning_allocation: none_or_explicit_budget_slice
```

`theory_learning` 可以是用户本来就提出的主要目标，不必强求即时预测增益。其评价应是所形成原则的真实解释／迁移能力，而不是节点数或语句深度。

当前任务已可完成时，理论队列可以留待独立研究阶段继续；不得隐瞒额外花费。如果目标明确只要求完成任务，没有授权理论探索预算，就不能任意扩张该运行。

## 2.2 QuestionSpec：所问内容与怎么回答分开

```yaml
question_id: 保留旧会话ID
meaning_version: immutable_version
mission_id: 所属使命
kind: value | parameter_region | distinguish | applicability | representation | structural_relation | discover_object | method_effect
subject_refs: 带类型对象／模型族／研究轨迹引用
context_ref: 仪器、单位、历史、复位、时间尺度、范围
query_semantics:
  mode: model_query | solution_relation
  query_expression_ref: 可执行查询，或允许多答案的解关系
answer_type: number | interval | distribution | boolean_claim | binding | program | relation | scoped_unknown
answer_resolution_ref: 何种证据足以回答，采用何损失／容差／判定语义
use_sites: 父输入槽、待决行动，或理论库学习目标
origin: user | diagnostic | theory_match | consolidation | proposer
```

### 对 `model_query` 的解释

它可以在候选模型上算出候选答案。例如在一个候选程序上求两个实验的预测差值。研究端不得通过它访问隐藏真实机制。

### 对 `solution_relation` 的解释

“找到能说明一组现象的关系”可能有多个合格答案。此时定义答案类型、复杂度合同和验收程序，不假定存在唯一标准程序，也不强制枚举全部解空间。

### 构想期可以不完整

自然语言想法先以 `IDEA` 保存，允许尚无完整预测。它不计为已检验假设。进入数值评价或外部实验前，要补齐能执行的具体合同。

这防止“为了严谨，必须已经知道答案形式才能允许发明”的循环。

### 哪些内容不应作为科学子问题

- 单独一句“还能如何提升分数”；
- “把报错修好”“将结果转JSON”之类执行任务；
- 与任何任务或理论学习使命无关联的命名、排名。

它们可以作为合法工程动作或开放构想存在，但不能计为已经完成一个新的科学问题。

## 2.3 ClaimSpec／HypothesisView：给现有程序一个共同科学接口

```yaml
claim_id: 稳定理论/主张身份
version: 不可变版本
kind: predictor | constraint | dynamics | method | prior
role_signature: 输入输出、单位、对象角色与时间顺序
representation_ref: 实际输入映射；未知则明确未知
body_ref: Program或Python或约束程序
family_parameters_ref: 允许参数及工作区域
initial_boundary_conditions_ref: 已知／未知／待研究条件
declared_scope_ref: 声称范围
measured_coverage_ref: 实际检验范围，不能等同声称范围
premise_refs: 支持前提，不是文件依赖的别名
evidence_refs: 当前版本对应支持、反例及未决
consumers: 调用关系与支持关系分开
```

约束可以不独立给出完整预测，例如偶对称、守恒、单调或条件独立。它们可以限制模型族、生成实验或改变先验。

## 2.4 Binding／TheoryRelation：每条理论联系也有可检查内容

```yaml
relation_kind: exact_compute | same_family | scoped_behavior_match | partial_correspondence | special_case | duality | incompatible | computational_call | empirical_support
source_ref: 对象及版本
target_ref: 对象及版本
mapping_ref: 变量、动作、输出或参数映射
scope_ref: 该关系声明的范围
required_premises: 映射或推导所需前提
check_ref: 结构／代数／预测检验
status: proposed | supported_in_scope | contradicted_in_scope | unknown
coverage_ref: 已经测试哪些条件
```

同族边、部分对应边和经验支持边不能用同一个分数替代。

## 2.5 EvidenceEvent：事实、主张和拟合状态分开

保留当前实际行索引及已付轨迹引用，补齐统一读出：

```yaml
source_kind: measured | code_execution | mathematical_derivation | reported_external_claim | author_interpretation | hypothetical
payload_ref: 原始记录或计算输出
conditioning_refs: 复用的旧条件证据
outcome_refs: 本次新结果
event_time_or_order: 真实可核对顺序
candidate_snapshot_ref: 哪个冻结候选事前预测
statistic_contract_ref: 数据范围、分组、损失、单位
result: 数值、区间、状态，不由自由文字覆盖
```

同一个原始结果被十篇笔记引用，只是一条来源，不是十次独立支持。实测仍可重复，但重复的目的、成本和相关性必须保留。

---

# 3. 问题如何自动生成：六种来源、两个队列

## 3.1 任务队列与理论队列

`task_agenda`：直接服务当前任务或必要支持条件。

`theory_agenda`：服务已声明的跨任务理论学习使命，包括共享结构、对照、尺度和先验效用。它不能因即时任务MSE不变而永远被饿死，也不能无预算无限抢占任务。

固定哪种预算分配只是研究协议超参数，不是认知定律。第一个实验可预先规定独立的离线整合阶段，避免引入任意在线百分比。

## 3.2 问题生成的六种来源

| 来源 | 输入证据 | 生成什么问题 | 必需的可执行连接 |
|---|---|---|---|
| 失败预测 | 冻结预测、新标签及相关前提 | 哪个共同前提需要辨别？ | 冲突依赖与候选答案后果 |
| 表示冲突 | 实际捕获特征、原始记录、误差下界 | 哪个被删除区别对所求重要？ | 新旧表示的候选与共同实验 |
| 参数未识别 | 可行区域或多个兼容实例 | 哪些区间内差异影响任务？ | 参数答案改变父输出或行动 |
| 旧理论迁移 | 候选结构对应、单位和范围差异 | 这个映射／适用条件成立吗？ | 映射检验与目标预测 |
| 多次研究共性 | 程序片段、错误模式、共享约束 | 能否形成共同结构并预测新来源？ | 抽象生成、实例化和迁移试验 |
| 理论冲突或冗余 | 不同理论／尺度在重叠条件上的后果 | 是矛盾、特殊情形还是任务等价？ | 共同物理问题上的对照 |

**从这些来源产生的是竞争问题，不是自动得到正确诊断。**

Reiter式冲突分析可用于已经形式化的精确前提组合；对于纯经验误差，输出可疑前提集合，不能假装已经找出完备最小诊断。[R04]

## 3.3 生成器分工

1. 确定性生成器：从实际依赖、过期绑定、表示碰撞、已知可行区域产生候选问题。
2. 结构生成器：从带类型程序和关系反统一、共享子结构、差分生成候选。
3. LLM提出者：创造新的关系、测量读出和解释；提交提案及其证据引用，而不是签发真值。

三个通道共用后面的编译、执行和评价。只有第三个不可用时，也不能让整个研究路由失效。

## 3.4 QuestionCompile：让提案产生工作，而不是多一段提示

```python
def compile_question(idea, mission, knowledge, evidence, budget):
    meaning = parse_or_accept_typed_meaning(idea)
    if meaning.is_only_score_improvement_request():
        return ActionOrIdea(idea, not_a_scientific_question=True)
    if not meaning.has_answer_type_or_solution_relation():
        return NeedsOperationalization(missing='answer semantics')
    if not meaning.has_parent_use_or_theory_goal():
        return Deferred(reason='no declared use')
    check_public_bindings_and_units(meaning)
    alternatives = obtain_declared_or_constructible_answer_alternatives(meaning)
    # 对新对象发现问题，可先返回构造任务，不强迫事先拥有全部答案。
    if meaning.is_discovery_query():
        return ReadyToConstruct(meaning, answer_evaluator=meaning.resolver)
    plan = build_or_retrieve_resolution_plan(meaning, alternatives, evidence)
    if plan.requires_missing_measurement():
        return NeedsSupportingQuestion(meaning, plan.missing_premises)
    return ReadyQuestion(meaning, alternatives, plan)
```

这段是规范性伪代码。实际函数名可调整；不能把函数体全部替换为再次问LLM“你觉得下一步是什么”。

## 3.5 提问先验与贪心的正确位置

可以使用 `informative + concise + relevant` 的生成偏好，但不要把它声称为唯一科学价值。

问题是“想知道什么”；实验是“怎样取得证据”；选择实验可以局部贪心，但系统应保留多个问题与两步支持实验。形式化问题后仍只选最大即时分数，并不自动解决短视。

与当前无关但有跨任务价值的问题必须由 `theory_agenda` 接纳，并在对应使命上评价，而不是伪造即时任务收益。

---

# 4. 问题和假设的判同、适用与复用算法

## 4.1 不建立万能相似度

实现四级匹配。结果不得只有一个0到1分数。

### Level A：精确计算身份

只对声明支持的表达式片段应用经过验证的重命名和重写。实数恒等式不自动适用于有溢出、非确定读出和IEEE浮点顺序的程序。

任意Python只能先做源码／拟合现场的精确身份与行为评价，不能从相似AST推断科学等价。

### Level B：同模型族／可重绑定结构

用带类型反统一找共同骨架，保留：变量角色、单位、参数位置、时间先后、状态与边界。产生映射候选，并计算其复杂度。

相同骨架、不同参数 → 新实例。

相同骨架但输入含义不同 → 先重绑定，不直接确认。

### Level C：有限观测行为对应

在固定实验族及共同单位下计算签名。

\[
d_Q(h,g)=\max_{e\in Q}|f_h(e)-f_g(e)|/s_e.
\]

`s_e` 和读出定义在本次比较前固定；不能为缩小差异而在观察到结果后任意重标定。

存储“在Q上相同／接近”的边，不永久合并理论。数值阈值接近一般不传递，禁止通过union-find把链式近邻并成一个真等价类。

### Level D：新问题适用性

检查实际作用域、前提、绑定、当前证据和新的预测。

行为匹配只提供候选，不自动完成Level D。有限候选池没找到旧理论，也不能声称旧知识绝对无法使用。

## 4.2 复用返回值

```text
REUSE_INSTANCE           当前实例与绑定仍有支持
REFIT_FAMILY             同族需要新参数
REBIND_THEN_TEST         类型可对齐，测量／角色映射待检验
RESUME_QUESTION          相同问题当前证据尚不足
SCOPED_REUSE             只对受支持范围复用
COMPARE_ALTERNATIVES     多个候选产生任务相关不同后果
PROPOSE_REVISION         当前对象有可执行失配，值得生成变化
UNKNOWN_MATCH            匹配计算或证据不足
NO_COMPATIBLE_CANDIDATE_FOUND  仅对搜索过的库成立
```

不要返回泛化的 `SAME/DIFFERENT` 后丢掉为什么不同。

## 4.3 问题身份缓存

缓存键至少绑定：任务所求、物理对象角色、实验解释、范围、答案语义版本。另保留被消费证据和依赖版本，用于判断能否复用旧答案。

- 同一语义的新证据：沿用问题，创建新尝试和更新答案。
- 同一ID但所求量改变：创建新问题语义版本，保留旧问题。
- 算法变化但所求不变：沿用问题，不创建新理论。
- `UNKNOWN` 不能被当作永不重开的否定；证据、计算预算或可用方法改变时可以续研。

## 4.4 问题之间的推导关系

若可证明 `answer(q1)=g(answer(q2))`，则已有q2答案可以回答q1，但仍须检查绑定与范围。只有双向可逆、解释意义保持时才可称问题等价。

例如“阈值区间宽度是否小于0.01”可由“当前阈值工作区间”推导，无须重购实验。这个关系不意味着两者要求同样的信息精度。

---

# 5. 判定不够用、开子研究和选择修订

## 5.1 不把所有失败都叫模型不够

| 观察 | 允许的结论 | 后续动作 |
|---|---|---|
| 代码错误／超时 | 本次实现未完成 | 修复执行、换可计算实现或未决 |
| 未使用新参数区域 | 被声称的候选尚未实际测试 | 重建／重拟合，不能先否定结构 |
| 单个冻结候选新预测失败 | 该候选在该处失败 | 更新支持、检查同族和前提 |
| 相同实际表示对应不相容确定性输出 | 表示与确定性前提的联合不足 | 查输入、历史、测量或噪声解释 |
| 同族仍存在不同兼容参数 | 参数尚未识别 | 值得时做区分实验；不等待族穷尽 |
| 一个有界精确约束系统不可行 | 该限定模型／前提集合不相容 | 产生候选诊断；不是所有旧语言失败 |
| 训练好、反复新确认差 | 选择／泛化或表示有问题 | 方法及数据使用审查 |
| 当前模型已完成任务但很复杂 | 可能有简化余地 | 独立理论／效率研究，不强制继续购测 |

## 5.2 子研究准入：必须有依赖用途，但不必有即时分数收益

`ChildUse` 至少指定一种：

```text
bind_parameter_region
bind_coordinate_or_instrument
supply_component
restrict_or_expand_scope
choose_parent_experiment
disambiguate_relation_mapping
supply_theory_library_candidate
```

至少有一个明确返回位置。子问题结果改变后，应执行当前已有依赖失效逻辑。

子答案可能范围很局部，但经父解释改变广泛后果；无需人为要求每个子题自己覆盖多个对象。

## 5.3 有价值的子问题：两种证据

**见证性检查**：存在两个当前仍兼容的子答案，会使父预测、行动或理论候选选择不同。这支持值得区分，但不是最优实验价值证明。

**充分性检查**：已经能在声明范围内验证所有允许子答案产生同样的任务答案，可以暂不继续该子题。若只是没搜索到反例，返回未知，不能伪造充分性。

复杂任务允许两步计划：先校准，再依新校准做响应实验。把第一步单独评分为“没有即时预测收益”会造成短视。

## 5.4 先做轻量调度，不先建立通用RL

调度按事件形成可行候选行动集合，不把全部行为压成一个贪心分数：

1. 先处理失效绑定、未评价的新结果及未完成的父返回。
2. 尝试以现有知识回答问题，避免重复计算或购测。
3. 保留任务已支持的直接行动；研究目的不是无限辨别全部机制。
4. 对真正影响任务的问题比较一到两步实验、廉价计算、参数重拟合和结构构造。
5. 对符合理论使命的共性、冲突和压缩机会进入理论队列。
6. 若无可解释的后续工作，记录未决或结束，而非原样追加同类调用。

调度的每项代价都计入已有账户。任何固定优先级或轮次上限都是可检验的设计，不是科学定律。

## 5.5 已有后验时与没有后验时要分开

有合法且经检查的模型后验时，可以估计：

\[
\operatorname{EVSI}(e)=
\mathbb E_y\left[\max_a\mathbb E[U(a,M)\mid D,y,e]\right]
-\max_a\mathbb E[U(a,M)\mid D]-c(e).
\]

或围绕问题答案使用 \(I(\psi_q(M);Y_e\mid D)\)。[R13]

没有后验时，用有限候选的任务相关差异、可行性见证和成本作启发式；必须标为启发式。候选复制不增加信息价值，有限代表个数不当作可信概率。

## 5.6 不能以唯一全局机制为子题完成条件

参数区间、部分映射或条件结论可以是合格答案。必须说明父问题如何使用剩余不确定性；不可填入一个精确数值后隐藏其他可能值。

若父预测随子区间变化很大，可以再开辨别；若所有值给出相同任务行动，可以保持区间并继续。

---

# 6. 子答案返回：从说法到实际计算

沿用现有 `bind_research`、组件绑定、refit、revision，不再创建平行调用链。

## 6.1 每个返回必须同时给出

```text
答案是什么
在哪些范围成立
依据哪些具体结果
依赖哪些仍未解决前提
哪个父输入或实验因此改变
哪些旧计算变成过期
```

`returned`、`finished`、`answered` 不等于被父模型消费，更不等于父确认通过。

## 6.2 父更新是声明的continuation

```python
def apply_child_answer(parent, answer, use_site):
    verify_answer_type_and_context(answer, use_site)
    # 支持是有范围的，不是自动变成全局真。
    new_binding = instantiate_binding(answer, use_site)
    if new_binding == parent.current_binding(use_site):
        return NoComputationalChange(reason='same effective answer')
    parent.install_binding(new_binding)
    invalidate_exact_dependents(parent, use_site)
    return schedule_existing_refit_or_experiment_rebuild(parent)
```

源代码路径不变但拟合状态改变也须更新实例身份；新数据尚未评分时不能使用旧确认。

## 6.3 支持图不是简单AND树

一个结论可能存在两条独立推导路径。某条路径前提失效，只撤销该路径；若另一条完整支持仍有效，不必全局撤销结论。[R05]

若支持互相循环，不能靠循环边彼此认证；当前具体证据／推导必须终止于实测、给定假设或独立形式证明。理论关系图可以有环，当前证成不能无凭据自支持。

## 6.4 参数域更新的专门验收

沿用本项目已识别的旧失败作为回归，不作为新自主发现结果：

- 新工作区间不含任何旧网格点时，标记“原候选未在新域测试”；
- 受管理拟合器必须从新域取得参数或明确保持参数集合；
- 自由Python须真实消费绑定；声明标志不能替代参数与行为检查；
- 区间中点是代表，不是新实证真值；
- 支持前提或门控语义改变，要重验该区间。

---

# 7. 统一证据更新与新旧假设比较

## 7.1 每批数据的顺序固定

```text
取得实际新结果
→ 核对对象／动作／行号／候选快照
→ 用购测前冻结预测评分
→ 更新该版本在该范围下的支持与反例
→ 标记依赖失效
→ 之后才允许使用这些数据重拟合
```

看过新标签后拟合成功不是原版本的前瞻成功。

## 7.2 旧条件＋新响应

允许复用已测物性和校准作为条件。新证据的身份由 `outcome_refs` 与候选冻结截点确定，条件是否仍有效由支持前提检查。当前结构研究已支持这一区分，保持它，不退回所有文件完全不相交的旧规则。[G5]

C5实验不能更新成C4实测。跨对象传播只能经过明确共享关系或映射，不能靠错配行号完成。

## 7.3 控制三种不同数值

- 历史作者断言是否满足；
- 已付记录的关系实测值；
- 当前候选对这条关系的预测残差。

它们不相互替代。采用当前已经纠正的读出助手，避免每次作者用自己的偏移重新组织证据。[G3]

## 7.4 未知的确切含义

以下至少分开记录：

```text
unknown_due_to_missing_evidence
unknown_due_to_unavailable_observation
unknown_due_to_incomplete_search
unknown_due_to_opaque_program_semantics
unknown_due_to_uncertain_mapping
```

这些不是新自然定律的真假答案，而是认识状态。只有拥有当前范围的有效不相容证据，才记录反驳。

## 7.5 库支持与统计校准

同一数据上的规则拟合、映射选择和模型排名不能重复算独立确认。概率模式必须有似然／损失标度、先验和近似状态；经验支持模式则报告实际范围、误差和未来测试，不伪造后验概率。

统计阈值沿任务合同执行，不因新形式化改写而放宽。若以后接入序贯检验，需同时实现适用数据前提，不能只采用其名词。[R10、R17]

---

# 8. 理论归纳：怎样让材料变成理论谱，而不是文档树

## 8.1 MaterialIntake：增量学习的最小含义

每份新资料被登记并至少经过一次证据分类；后续按新内容和依赖变化处理，不每轮重读全部源文件。

分类：实测、执行、外部文献主张、作者解释、构想、错误、未决、已有内容的复制。

相同来源的重复总结不增加支持。作者说某关系成立，而数值检查违反它时，保存两者和冲突，不将作者文字升级为事实。

这是一项材料处理算法，不是对整个互联网建立真值库。

## 8.2 从真实轨迹提取Episode

每次Episode至少包含：

```text
原问题语义
初始候选／表示／支持前提
导致变化的具体数据
实际改动的程序或关系
因此改变的实验或预测
独立结果与代价
仍未解决范围
```

只有行动发生过，才能记录“采用”；只有新的结果取得，才能记录“检验”。失败Episode不删除。

## 8.3 从多Episode生成抽象

首版先支持现有 `Program` 的带类型片段，不尝试任意Python的完全语义反编译。

算法：

```python
def consolidate(episodes, library, allocation):
    groups = index_by_roles_operations_and_failure_structure(episodes)
    proposals = []
    for group in select_bounded_groups(groups, allocation):
        proposals += typed_anti_unification(group.programs, group.bindings)
        proposals += extract_shared_constraint_patterns(group)
        proposals += llm_suggest_with_grounded_examples(group) if allocation.llm_available else []
    for item in deduplicate_proposals(proposals):
        compiled = compile_abstraction_and_inverse_instantiations(item)
        if not compiled.ok:
            retain_as_idea(item, compiled.reason)
            continue
        check_reconstruction_on_sources(compiled)
        challenge_with_retained_counterexamples(compiled)
        register_candidate_with_scope_and_cost(compiled)
    return independent_transfer_candidates()
```

反统一只提供公共骨架；来源重构只能证明抽象忠实表示旧程序，不证明它是世界规律。迁移与新预测必须另测。[R11–R12]

## 8.4 描述长度成本包括什么

若用MDL，至少计：

\[
L(K)+\sum_j[L(b_j\mid K)+L(\theta_j\mid K,b_j)+L(D_j\mid K,b_j,\theta_j)].
\]

编码合同应固定；新库定义、适用选择器、映射、参数精度和例外都要付费。不能只把长代码包装成一个新名字，然后用名字长度声明压缩。

已有训练数据上的压缩是候选指标，不取代未来预测和成本收益。

## 8.5 新元先验如何接入下一任务

一个候选原则必须至少改变以下之一：

- 生成哪些候选结构；
- 以什么顺序检查不确定前提；
- 组成怎样的实验；
- 何时考虑某个可逆简化；
- 如何匹配并检验旧理论。

仅保存“过去证明了要重视历史”这种文本，不算完成。

首版使用软排序或可选择的研究程序，保留基础搜索／替代解释通道。不把刚归纳出的原则直接升级成跨所有任务的硬过滤器。

## 8.6 哲学谱生长与修订

图可以新增：实例化、同族、特殊情形、局部对应、候选对偶、反驳某作用域、计算调用与支持关系。

新理论不必保留被可靠反例否定的旧全称命题。必须保留证据历史，并说明旧理论在哪些条件仍有效、哪些已撤销、哪些只是计算资源。

“理论越来越大”不是目标。合并冗余、限定作用域、保留相互竞争解释和删除活跃索引中的无用对象，也可以是进步；原证据仍可追溯。

---

# 9. 先验、正则与结构关系如何编译进实际学习

## 9.1 PriorSpec的用途必须声明

```text
parameter_prior
structure_prior
constraint_penalty
proposal_preference
research_policy_preference
```

不同类别不同评价。一个来源学得的程序可以作特征，但不能自动获得与原程序相同的统计不确定性。

## 9.2 一个对称性候选的完整路径

1. 假设：某变换 \(T\) 下输出遵守 \(\rho(T)\)。
2. 数学表达：\(f(Tx)=\rho(T)f(x)\)。
3. 经验检验：在共同物理输入语义和新成对条件上比较。
4. 有条件采用：若只是近似支持，先用软项
   \[
   R(f)=\sum_{x\in Q}\|f(Tx)-\rho(T)f(x)\|^2/s_x^2.
   \]
5. 预测学习：损失加该项，但评价依然用独立物理输出。
6. 反例：不得通过调整单位或变换使关系自动成立；降低作用域支持或撤回偏好。

如果该关系来自当前标签，不能再次把其数值检验当成独立先验；应视为同一数据的结构学习或以独立来源学得先验后迁移。

## 9.3 正式Bayesian模式的最低要求

记录：参考测度、似然、参数／结构先验、归一化可用性、近似方式、候选池覆盖限制和数据使用边界。

无法做到时输出 `heuristic_score` 或 `empirical_support`。不要用softmax包一下损失就写成“可信后验”。

---

# 10. 三个贯穿例子

## 10.1 从“还能提高吗”改成一个有答案的问题

原任务：预测装置响应。

不充分提问：再试哪个回归器更好？

有明确答案的问题：当前表示是否在这些已付记录上合并了会改变响应的区别？

过程：

```text
当前真实表示
→ 相同表示分组
→ 对应不同实测输出及经验下界
→ 父模型的确定性充分表示承诺受挑战
→ 产生表示／测量／随机性竞争解释
→ 选择可执行修订
→ 重建父模型
→ 新条件上的事前预测
```

旧零填充案例用于接口与数学回归，不能把宿主已知修复偷偷发给自主发现实验。[G4]

## 10.2 当前尾段相同，不等于历史状态相同

问题Q1：两条历史对已测共同后缀的输出是否相同？

问题Q2：它们对一类未来行动序列是否不可区分？

Q1可以已经回答，Q2仍未决。Q2的候选回答应由未来行动条件预测表达，而不是由“隐状态已消失”的文本标签表达。[R16]

子问题可以询问：什么公开探测序列，在仍兼容Q1的候选中区分Q2？找到后通过现有revision先冻结预测再购测。无区分序列可能来自候选贫乏、行动权限不足或真实任务等价，不能混成一个结论。

这直接适用于source006，但本文没有指定其隐藏机制或正确新结构。[G3]

## 10.3 勒夏特列—楞次作为理论问题

父问题：两个已描述规律在什么角色／尺度下保持共同关系，以及这个关系对当前新对象有什么后果？

不是预先回答“二者都是稳定负反馈”，而是建立候选：

```text
源对象／扰动／响应／观察量
→ 候选角色映射
→ 分别核对单位、时间和适用前提
→ 编译一项可比较关系，例如特定投影上的响应方向
→ 生成能违反此对应的对照
→ 在新系统上提出事前预测
```

若只能验证方向相反，就只保存方向关系；不能自动获得幅度减弱、渐近稳定、收缩或完整机制等价。[O2]

第一项正式实验优先用匿名、形式明确的合成结构，著名定律作远域诊断，不作为唯一金标准。

---

# 11. 与当前代码的最小接线

| 现有位置 | 保留内容 | 本轮增加的责任 |
|---|---|---|
| `online_inquiry_llm.py` | 问题ID与assignment、模型登记 | 问题语义适配；会话ID与科学问题身份分开 |
| `common_archive_inquiry.py` | 失败触发、子返回、父阶段 | 共享的可计算研究路由与continuation；live只负责生成时的会话包装 |
| `anonymous_method_learning.py` | 外层作者、零LLM实践、成功链 | 将待生成问题／候选显式交回外层；内层执行语义检查不依赖provider |
| `inquiry_revision.py` | 事前冻结、readout、数值checks | 绑定QuestionSpec与被检验主张；区分作者解释和实际答案状态 |
| `executable_evidence.py` | 参数绑定、来源与残差、失效 | 向统一路由发出具体工作义务，而不是只增加上下文文本 |
| `representation_diagnostics.py` | 实际输入碰撞下界 | 返回有限见证及其前提，不返回正确新特征 |
| `structural_inquiry.py`／`world_relation_research.py` | 类型Program、候选银行、显式加入 | 模型族／问题语义索引、候选结构实例化与匹配接口 |
| `historical_capabilities.py`与目录 | 历史能力导航 | 只作召回入口；科学适用性由新的匹配与检验决定 |
| `temporal_method_learning.py` | 发布方法和实践支持 | 记录方法回答过哪类问题、实际输入条件和可迁移语义 |

建议只新增三个薄层文件，名称可调整：

```text
question_semantics.py        问题编译、答案类型、身份和用途
theory_matching.py          类型匹配、结构映射、行为签名和适用返回
theory_consolidation.py      Episode提取、共同结构提案和跨任务索引
```

不新增另一套拟合器、实验购买、确认器或签名系统。

## 11.1 可运行的主循环规范

```python
def step_research(state, public_lab):
    new_events = state.read_new_events_once()
    score_frozen_forecasts_before_refit(state, new_events)
    update_scoped_evidence_and_staleness(state, new_events)

    pending = apply_ready_child_returns(state)
    if pending:
        return execute_existing_action(pending)

    diagnostic_questions = generate_from_actual_dependencies_and_conflicts(state)
    state.agenda.record_compilation_outcomes(diagnostic_questions)
    # IDEA、缺少操作化与待作者工作分别保留，不能静默丢弃。

    question = state.agenda.choose_under_mission_and_budget()
    reuse = match_existing_answer_and_theory(question, state.knowledge)
    if reuse.is_supported_answer:
        return apply_answer_to_declared_use(question, reuse.answer)
    if reuse.requires_rebinding:
        return schedule_or_resume_binding_question(question, reuse)

    plan = prepare_bounded_resolution_plan(question, state)
    if plan.needs_authored_object:
        # 无live provider时交回作者，不隐式调用。
        return AuthorWorkRequest(question, plan.object_type, plan.required_contract)
    if plan.has_supported_or_discriminating_action:
        return freeze_then_execute(plan.action, public_lab)
    if question.kind == 'discover_object':
        return construct_with_available_program_and_rule_generators(question, state)
    return record_scoped_unknown(question, plan.reason)


def step_consolidation(state):
    if not state.mission.has_theory_budget:
        return NoAction()
    episodes = state.new_episodes_since_consolidation_cursor()
    candidates = consolidate(episodes, state.knowledge, state.remaining_theory_budget)
    return enqueue_scoped_transfer_questions(candidates)
```

这两条循环必须共享证据与预算，但各自保留使命。不能把“离当前目标没有立即收益”写成所有理论问题的永久拒绝理由。

---

# 12. 必须实际执行的回归与反例测试

以下是本轮算法验收，不是另开22个正式科学研究。

| ID | 场景 | 必须断言的行为 |
|---|---|---|
| T01 | 同ID改变所求目标量 | 新语义版本；旧结果不覆盖 |
| T02 | 同问题换拟合算法 | 保持问题，新增尝试而非新理论 |
| T03 | 换参数但同结构 | 新实例，同族可追溯 |
| T04 | 同代码但仪器单位变 | 不直接迁移确认，先映射／重验 |
| T05 | 当前预测签名相同，新增probe可区分 | 原机制仍独立保存，新边反映区分 |
| T06 | A近B、B近C、A不近C | 不用阈值链永久合并 |
| T07 | 复制候选100份 | 不增加信息价值或支持次数 |
| T08 | 旧答案已足够当前所求 | 复用，无重复购测 |
| T09 | 支持前提改变 | 只使真实依赖路径失效 |
| T10 | 仍有另一完整支持路径 | 不无条件撤销所有支持 |
| T11 | 参数子答案更新但旧网格不含内部点 | 实际重建候选，不先否定结构 |
| T12 | 子问题返回答案无父用途 | 不记作父修订完成 |
| T13 | 子答案改变待执行父probe | 购测前计划确实改变并保留旧版 |
| T14 | 旧物性＋新响应 | 条件化新检验合法，旧校准不当作新独立证据 |
| T15 | 新响应用于重拟合后再评分 | 不计作该新版事前确认 |
| T16 | C5计划接C4行号 | 阻止错误支持更新 |
| T17 | 搜索截断或数值求解失败 | UNKNOWN，不是族反驳 |
| T18 | 缺失历史与零值历史 | 通用诊断暴露见证，不给隐藏正确窗口 |
| T19 | provider=None | 可计算路由执行；生成要求显式返回外层 |
| T20 | 理论问题无即时MSE收益但有独立使命 | 进入有预算理论队列，不被默默丢弃 |
| T21 | 相同记录被多份笔记重复引用 | 支持来源不被重复计数 |
| T22 | 纯文字规律与已保存数值冲突 | 两者并存且冲突可见，不覆盖实际值 |
| T23 | 新库函数只是查表包装 | 原体积与映射成本不因命名消失 |
| T24 | 来源抽象可重构但新域反例失败 | 源内支持保留，跨域支持不成立 |
| T25 | 对称关系只在局部证据成立 | 不自动升级全域稳定／收缩 |
| T26 | 所有候选后验权重高但池外未覆盖 | 不报告全模型空间可信度 |
| T27 | 原模型已足够任务、机制仍未知 | 可以完成任务并保留理论未决 |
| T28 | 索引／绑定／算法均未变且旧工作已完成 | 读缓存结果，继续研究必须有新依据 |

测试应包含实际源码、输入、动作轨迹和输出断言；仅检查字段非空不算通过。

---

# 13. 最小独立研究实验

## 13.1 先验证组织，再测发现

阶段A给两组相同候选和理论材料，只比较问题语义、判同与父用途路由。这避免候选内容差异掩盖组织效应。

阶段B再允许相同模型、调用和低层原语自主生成问题／假设。测能否提出新的有用对象，而不是只在固定正确模板里选标签。

阶段C从来源研究归纳一项原则／方法，冻结后迁移；所有来源失败保留在分母。

## 13.2 三组主对照

1. 强通用可执行档案：相同代码、读写、查询、回滚、重验和修复权限。
2. 明确问题与理论匹配，但理论库固定。
3. 上述组织加跨任务理论归纳。

同源配对，预算、模型、工具、终评和初始知识相同。自主实验允许获取不同数据，这是比较对象，不是混杂；额外数据和计算仍计费。

必要消融：冻结提问、删除父用途、关闭新原则移植、打乱关系映射。不要一次铺满所有组合。

## 13.3 测什么

主要：未见条件预测、任务完成、错误复用、错误接受、覆盖、全部来源与后继费用。

次要：问题回答率、跨对象后果改变、重复工作减少、库对象实际调用或语义使用、概念删除后的差异。

不以哲学节点数、子问题数、树深度或叙述深度作主分数。

## 13.4 任务族建议

- 表面／单位改变而机制不变；
- 表面相同而机制改变；
- 同族参数改变；
- 当前输出一致、后续行动能区分；
- 原表示合并必要信息；
- 真随机或不可观测，应该保持不确定性；
- 多来源可共享结构；
- 多来源不应被强行统一。

来源是独立实验单位；同一来源的多条轨迹不是多个独立领域。已消费的source006作为开发诊断，不再包装成新盲测。

## 13.5 元改进的严格移植

固定后继任务知识与起点，只替换旧／新研究方法或先验，观察相同预算下“获得新答案”的成本与质量。预测知识、研究方法和搜索偏好分别移植，不把更完整答案库归因成更强元学习。

人工给出的新原则可以检验采用与迁移，不计自主发现。系统运行内部实际产生的对象才进入相应自主主张。

---

# 14. 施工顺序与停止条件

## PR1：问题语义＋可执行父用途

交付：旧对象适配、问题身份与答案类型、实际绑定返回、provider=None路径、T01–T04／T08–T19。

先用已有历史记录重放，不启动付费科学运行。成功标准是已知证据确实改变父动作／拟合，不是多了一份JSON。

## PR2：理论匹配＋问题生成

交付：四级匹配、有限关系与问题生成器、行为签名不永久合并、任务／理论双队列、T05–T07／T20／T25–T28。

运行共同候选的机制实验；若强普通档案表现一样，保留平局，不修改基线制造效果。

## PR3：单一主题的跨任务归纳

交付：Episode抽取、类型反统一／共享关系提案、反例检验、冻结移植、T21–T24。

先只选一个主题，例如“缺失与零值不能在任务相关时混同”。若原则由开发者给定，则实验名为采用／迁移；真正发现实验必须隐藏该答案，仅给共同低层原语和实际材料。

## PR4：完整自主与统计后端选择

在前三步实际改变行为后，再决定是否接MDA式概率结构推断或更复杂实验价值。无明确增益则不扩张。

下一批付费续研必须写明交付问题、候选对象、区别实验或理论移植。只完成接口修复、文字解释或重复旧比较时，不原样续费。

---

# 15. 可直接交给Codex的首轮指令

> 以 `468256bfc7e9d0c6077f29a0475e7e065d24c6d6` 为审阅基线，先核对工作树是否有后续变更；保留所有旧实验与运行时。只实施PR1，不改科学答案、成功阈值和付费预算。
>
> 将问题的语义、会话身份、假设族、冻结实例和实际应用区分开，用适配层接回现有Common／方法实践、`bind_research`、revision、refit及确认。运行时在`provider=None`时仍能消费已有答案与更新父计算；需要生成新科学对象时显式返回作者工作，不隐式调用LLM。
>
> 在原有测试路径内加入本稿指定行为断言。用已有历史数据只到当时截点重放：已测参数域应进入下一拟合；表示冲突应成为具体未解问题；已有子答案应改变待执行父实验。不得把宿主事后公式或精确隐藏参数提供给提出者。
>
> 交付代码差异、实际回归输出、一次父用途被真实消费的短轨迹，以及仍未实现的内容。若只完成记录结构而没有行为变化，直接报告未完成，不启动新付费批次。

---

# 16. 本方案怎样允许宏观研究，而不把宏观变成空话

最上层可以问：“这些不同系统能否由一条更一般的关系组织？”

但系统接下来需要给出：比较哪些对象、允许什么映射、保持什么、在哪些条件上、需要什么新证据、怎样影响未来研究。

这不是把哲学问题缩回单次拟合，而是让它获得可检验的内容。问题可以是宏观的，回答可以是局部且有范围的；局部支持逐渐累积成更一般理论，而不是一次命名就获得普遍真理。

最终主线应是：

\[
\boxed{\text{问题明确地要求某种知识；假设明确地承担某种承诺；证据改变承诺；理论谱改变以后怎样提问和研究。}}
\]

---

# 附录A. 第一版问题程序语言：不能只存自然语言问题

本附录补齐第3节中 `query_expression_ref` 的最低实现。它是**在现有Program与实验接口上的查询表达式**，不是替换当前预测语言。

## A.1 两条分别执行的路径

- `evaluate_on_model(q, model)`：在候选模型上获得预测答案，用于比较和设计；没有读真实标签的权限。
- `evaluate_on_evidence(q, evidence)`：从声明范围内的实际记录计算答案或检验；没有数据时返回未知，不用候选输出填成实测。

两条路径的物理输入、单位、读出和角色必须一致。

## A.2 最小查询构造子

| 构造子 | 参数 | 返回 | 必要限制 |
|---|---|---|---|
| `Readout` | experiment_ref, output_role, statistic_ref | scalar/vector/distribution | 模型端预测，证据端查匹配实验；不能混用 |
| `Difference` | q_left, q_right | 同单位数值差 | 两子查询答案单位可减 |
| `RelationResidual` | law_ref, argument_queries | residual及各前提状态 | law必须可执行且注明适用域 |
| `ParameterRegion` | family_ref, parameter_role, evidence_scope | 区域或未知 | 找到代表不等于区域完备 |
| `AllOnFiniteSet` | finite_cases, predicate_query | observed_true/false/unknown | 结果只对给定集合，不叫全域forall |
| `FindDistinguishingExperiment` | claim_refs, legal_constructor_ref, target_query_ref | experiment或unknown | bounded搜索；候选可能都错 |
| `FindSharedStructure` | typed_program_refs, mapping_budget | abstract_program + substitutions或unknown | 仅结构重构通过，另需经验检验 |
| `CheckApplication` | theory_ref, target_context, permitted_mapping | scoped evidence result | 相似匹配不等于适用 |
| `MethodEffect` | two frozen methods, common start, task_contract | paired outcome/cost | 必须实际执行；未执行只可预测 |

第一版只须完整实现 `Readout`、`Difference`、`RelationResidual`、`CheckApplication` 与共同的未知传播，再逐个接现有参数域和结构查找。不能宣布九项全实现而仅保留字符串。

## A.3 一个有限偶对称检验

查询的可执行主体是：

\[
q_x(m)=m(e_{+x})-m(e_{-x}).
\]

它问的是“在已绑定的相同背景与协议下，这对输入的响应差是多少”。候选主张 \(q_x=0\) 才是偶对称的局部后果。

若查询集合只含两个幅值，回答“这两个幅值上相容”，不能回答“全函数为偶函数”。

初值／背景未知时，`RelationResidual`返回 `premise_unknown`；若实际只改变了一个条件的声明与动作不一致，返回 `invalid_experiment_binding`，不是关系被反驳。

## A.4 自然语言编译的信任范围

LLM可以把“好像正负输入反应相同”编译为一个候选 `Difference` 查询。运行时核对实际行号、单位、背景和数字，但不能声称因此证明了自然语言所有含义都已被捕获。

保存自然语言原句、编译对象和编译来源。存在歧义时，保留多个候选解释或要求外层先明确读出，不静默挑一个最容易通过的形式。

---

# 附录B. 有界结构匹配算法：明确到可施工

## B.1 受限Program片段匹配

输入：源Program、目标Program／待填模式、类型和单位合同、可展开库深度、实际范围。

遍历规则：

1. 对相同操作符和相同有序子树位置递归；
2. 变量可重命名，但同一源变量每次出现必须保持同一映射；
3. 参数常数差异可以成为待重估参数，必须记录约束；
4. `lag`、`window_sum`、`if`等的时序与分支位置不可当普通排列忽略；
5. 交换参数只在该算子当前数值语义明确允许时尝试；
6. 单位转换只来自当前允许的映射族，并记录标度／原点；
7. 可选择展开已保存库函数，但展开深度与计算计入上限；
8. 用尽上限返回 `incomplete_match_search`，不返回不可能对应。

## B.2 输出不只是一个“最像的”对象

每项候选输出：

```text
结构映射与代价
哪些字段被严格保持
哪些参数需要重估
哪些作用域／测量前提尚未验证
在共同probe面板上的预测差异
哪些证据来自映射拟合，哪些来自随后检验
建议调用：reuse / refit / bind / compare / unknown
```

多个不等价映射同样相容时都保留，直到预算限制或新证据区分。不得仅凭token相似度挑一项后将其称为唯一映射。

## B.3 缓存与语义别名

运行缓存可以合并同一输入版本下完全相同的计算任务；理论节点仍保留来源和作用域。

如果两个程序只在Q上相同，可共享Q上已计算的预测缓存，但不得共享Q之外的预测、真实作用域或确认记录。

---

# 附录C. 没有可信后验时的首版研究选择器

本附录是明确标注的**有限池启发式**，不是Bayesian最优实验设计。

## C.1 候选先按当前查询后果去重

给当前问题q和计划面板T，在同一观察语义下计算候选答案签名。相同签名在**本次评分**中仅算一次，源候选仍不删除。

只有当前仍兼容证据且产生不同任务答案的签名对进入待区分集合 \(\mathcal P_q\)。没有可比较候选时，不用任意“两句互补预测”假装拥有科学信息增益。

## C.2 实验的区分分数

对确定性候选与声明容差，一个低成本分数是：

\[
G(e)=\frac{\#\{(i,j)\in\mathcal P_q:
|\hat y_i(e)-\hat y_j(e)|>\epsilon_i(e)+\epsilon_j(e)\}}{\max(1,|\mathcal P_q|)}.
\]

它只统计当前不同答案之间，有多少对会被这个实验的预测区分。它不表示真实世界的信息量，更不保证候选正确。随机模型必须用已定义的分布／噪声检验，不可把两个均值略有不同当成可稳定区分。

## C.3 选择顺序

在同一使命和前提都满足的行动池内：

1. 去掉非法输入、未解绑定和无预算动作；
2. 重用已完成的同语义计算，除非问题本来就是重复性；
3. 对直接回答当前任务的已有支持行动，不以纯探索优势强迫推迟；
4. 对研究行动优先比较 `G>0` 的一到两步计划；
5. 在区分能力相当的计划之间，按观测、计算和模型成本的预声明偏好选择；
6. 没有区分见证时，选择有明确父用途的测量／绑定工作或生成新候选；
7. 无上述工作时记录未决，允许在已授权范围内进行标明为覆盖探索的操作。

两步计划的第二步由第一步答案条件化；未执行分支只是候选预测，不能生成实测支持。

这个选择器仍可能短视或被贫乏候选误导，必须与当前基线比较。其目的首先是避免“重复脚本、重复票数、纯互补文本”占据评分，而不是宣称已解决通用最优研究。

## C.4 研究进展的记录

分开记录：

- 新观察覆盖；
- 某冻结候选的前瞻支持／反驳；
- 一个有限族的可行域变化；
- 父问题的可行答案集合缩小；
- 新结构产生并经独立条件检验；
- 当前任务成功；
- 跨任务理论复用收益。

换一个参数代表、重跑同一OOF、候选个数减少，都不能自动成为后两种进展。

---

# 附录D. 结构归纳最低算法：带类型反统一

对两个已经带类型的Program，计算一个共同模式与两套替换：

\[
g[\sigma_1]=p_1,\qquad g[\sigma_2]=p_2.
\]

递归规则：

- 完全相同子树直接保留；
- 相同算子和相同类型递归子树；
- 不同但类型兼容的子树引入一个hole；
- 同一对子树再次出现时使用同一hole，保持共享约束；
- 类型、单位或时间解释不兼容时，不建立这个对应；
- 整棵树退化成一个任意hole的模式允许生成，但通常没有可检验结构／压缩增益，不应自动进入活跃库。

例如两个仿射程序可产生共同 `a*x+b` 模式；这只表明代码结构可共享。是否代表两个物理过程同族，要继续检查角色、测量和未来预测。

新抽象进入库前至少完成：重构、复杂度核算、来源反例检查；进入新任务后必须完成新的适用检验。大规模压缩可后续接Stitch，不要求第一版自行重现其全部搜索器。[R12]

---

# 附录E. 不是所有组件都必须先形式化才能使用

真实研究中，科学创造本来可以先是模糊想法。系统应有三个明确层次：

1. `IDEA`：允许自然语言猜想，没有数值支持身份；
2. `EXECUTABLE_CANDIDATE`：已有可执行预测、约束或检验，允许探索；
3. `SUPPORTED_APPLICATION`：具体版本与范围获得实际支持，可用于当前决定。

三层可以返回、撤销、修订。不要因为“形式化要求”阻止产生新语言，也不要因为“需要创造性”让任何流畅文字直接进入支持库。

这正是本方案希望从“全部让Luna决定”和“所有东西由宿主手写”之间取出的可实施分工。


---

# 附录F. 失败如何约束下一候选，而不是无限换代表

可借鉴ILP的generate–test–constrain思路，但不直接移植它的无噪声逻辑剪枝结论。[R20–R21]

每个失败约束须标明来源和生效范围：

| 失败 | 可合法约束的对象 | 不能约束的对象 |
|---|---|---|
| 类型不合法或公共接口不存在 | 当前类型系统下同类非法程序结构 | 其他合法变量／接口 |
| 新参数域未被实际消费 | 声称完成了当前域内试验的实例 | 整个科学模型族 |
| 可靠确定性表示碰撞 | 只依赖这个完全相同表示的确定性预测器 | 更丰富表示、适用随机模型 |
| 精确逻辑片段中的不相容 | 证明所覆盖的结构／前提组合 | 未被该证明覆盖的扩展 |
| 单个冻结实例前瞻失败 | 该实例在该范围的支持 | 同族其他参数或新范围 |
| 多次检验支持不足 | 当前经验采用资格／搜索优先级 | 无条件逻辑真值 |

后续提案器应接收这些约束并真实过滤或降权，不只是把失败附到长prompt里。约束版本／前提改变时重新判断适用，保留其原始依据。

同一失败被多次解释不重复增加约束权重；同一个被证明非法的程序变换名字后仍不能重新作为新候选评分。另一方面，明确改变被反驳前提的候选应当被允许重新测试。
