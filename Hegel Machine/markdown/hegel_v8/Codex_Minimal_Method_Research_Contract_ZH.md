# Codex施工合同：从真实研究片段形成一个可执行的方法选择/改写对象

基线：`3806d239b5c53b9b78ac23fcf7fa4c74f16c27ed`。
性质：拟议施工规范，不是已经发生的实现，也不授予新的计算、墙钟、观测或作者额度。
来源与解释：见同包《当前诊断与主线计划》及`source_index.json`。

## 1. 唯一主任务与非目标

交付一条链：真实方法冲突 → 带父用途的问题 → 有限、不同的可执行研究程序 → 匹配开发比较 → 普通Common调用 → 结果及反例回到下一选择。

不交付新的Agent、方法DSL、日志服务、存储后端、通用程序等价器或统一科学相似度。不得重复解决已修的数值、全池、恢复、反馈导入、并列探索与旧答案回调问题。原16输入变体和九项条件对照保持结束，不原样追加。

首个产品是一个有限Python研究程序，不要求发现任意科学语言。不保证优于基线；完成一个有内容的阴性比较也允许结束。

## 2. 开始前固定哪些东西

保存当前提交、运行入口、公共方法源、原成功阈值、评价范围、物理来源身份、当前候选池、当前已付评分/拟合引用、费用及未知占额。复用现有注册机制，不另建冻结系统。

本次不修改原冻结源码或历史记录。所有公共正确性修复同等提供给各臂。当前已完成case002：145行、4762入口、父稿0118、已知累计1,195,711计算、未知773,448；这些不是可自动开新实验的授权。

资料不足时，不以“恢复正常”为名重跑科学拟合。若某项实际前态无法由原引用恢复，则明确缺失，不从后态删行后伪造前态。

## 3. 两个不混淆的产物

### 3.1 研究片段视图

只读薄投影，来源是现有method_results/operator_results/native rounds/研究Episode/评分引用。不得另存整份轨迹或方法图。

建议在`research_episodes.py`中提供适配函数，函数名可按现行命名约定确定；逻辑合同固定：

```python
def extract_method_segment(agent, *, root_result_ref, decision_start_ref):
    """Return a thin, reference-grounded segment; never run a scientific tool."""
```

返回的逻辑信息必须包括：

```text
root_result_ref                  实际比较、父处理或方法结果位置
start_state_refs                 可恢复的调用前证据/候选/知识/费用范围
question_ref                    question_id + meaning_version
method_source_ref               实际程序源与调用上下文
nodes                           对已存在执行事件的引用
read_refs / produced_refs        具体产物版本及数据范围
depends_on_refs                  由实际读写确认的前序事件
execution_status                完成、资源停止、错误、行政删失或未知
cost_refs                       原账引用，不把费用复制累加
missing_dependencies            未能确认的依赖
```

这些是薄视图字段含义，不是强制另建持久schema。已有字段能表达即直接复用。

### 3.2 可执行方法对象

保存普通Python `select_actions`源码及其原有manifest/知识引用，附方法来源、条件、范围、反例和评价引用。**不要把它伪装成原生DAG的`research_program`，然后送入`replay_research_program`。**后者有明确的term/gate/basis合同，不能执行任意Python行动策略。

如果现有方法库需要区分两种执行载体，只在同一记录/路由上增加明确的载体判别：原生重放继续走原生入口；Python程序走已存在的`ProcedureConsumerAgent/execute_procedure`。具体字段名在实施时与现有manifest一致，但不能靠猜测payload的形状静默分派。

原方法@2的两源支持只属于其原内容。新版程序从一个开发片段产生，不能自动继承“两源已验证新程序”的主张。可作为有范围的实验候选，独立支持仍待取得。

## 4. 精确依赖与原ResearchMethods兼容

### 4.1 提取规则

每次读取评分、模型或参数，记录它实际引用的实例，而非相同动作名。

示例：

```text
fit:A → model:A → assessment:A → comparison:C → adopted:A
fit:B → model:B ────────────────────────┘
```

comparison:C比较两稿时依赖两份评分及共同scope；采用A不意味着B不存在，也不意味着A的每一步都有正因果贡献。

另一个示例：

```text
paid_record:7
    → boundary_update(question=q, meaning=v, round=r, proposal=p)
    → interval:new
    → 下一次实际消费该区间的拟合/实验
```

物理答案和分析完成是不同节点；已经完成的相同分析继续复用。

### 4.2 对旧序列学习器的最小扩展

Common路径使用可选的确切step_refs/依赖闭包；原controller仍使用旧trace合同。确切模式不能退回`requires_actions=['fit']`猜一个最近fit。

闭包只访问实际祖先，可以跨越最近六步；遇到外置或缺失引用，保留unknown，不假装前提不存在。重复方法调用以现有调用上下文分段，不将同method_id的不同调用拼接。

保留无即时收益但有实际下游依赖的步骤。即便某步没有被当前最终产物读取，也不能直接宣称其总体研究无益；删步先成为候选，再由配对运行验证。

## 5. 问题合同：一个method_use题，不新增宏观议程

问题：相似公开研究状态下的相反方法回报，是否来自阶段/证据差异，或者需要不同的研究段？

必须引用：实际旧方法版本、至少一份有益与一份无益/未决回报、调用前状态、当前父问题、可支付备选。缺相反回报时可研究单一失败，但不得伪造混合样本。

至少两个答案有不同执行后果：

- 旧方法在补齐某项实际区分证据后可用 → 选择具体Q后按原方法处理。
- 同资料下另一研究段更有用 → 选择具体N或替代方法。
- 资料不够/候选无差异 → 保留原普通研究或有界探索，并说明缺失区别。

引用、题义、结果绑定、依赖失效与父评价沿原v4实现。这个问题由本报告建议；只有被算法从真实记录自动实例化，才能在实验报告中称为自主提出。

## 6. 候选空间：最多两个局部候选加原程序

### 6.1 允许的候选形式

普通原程序P0始终在场。P1/P2只允许：

- 在一个真实选择点，用另一项已经存在且合法的研究段替换原段；
- 在一次实际反馈后，根据调用前/已返回的公开状态改变下一段；
- 选择一项已有区分或配对实验，并执行其原分析回调及必要消费者。

每个方法只允许一个条件分支、两个叶子，学习部分最多两个研究段。导出、全池比较、父处理和保存是共同闭包，不被当作学得新步骤，也不计作“允许删掉的第二段”。总底层调用次数仍受原执行器max_steps和资源合同约束，必须足够容纳共同闭包。

不允许任意Python编辑、任意实验扫描、不断加历史窗口、改世界阈值、无限迭代或将未来评价作为输入。

### 6.2 备选项来源

从当前`method_decision`/native返回/实际方法库中读取原普通proposal：

| 类别 | 提案来源 | 必须先验证 |
|---|---|---|
| R | 原`theory_transfer`/方法重放绑定 | 实际数据、角色、两seed规则、未重复出售已完成无益调用 |
| N | `_native_compute_continuation`实际返回的action | 同一题义、真实未完成工作、原准备和完整收尾已报价 |
| Q | `proposed_experiments.query_action` | 公开合法控制/长度/reset，实际claim或配对目标、回调、完整所需观测 |

候选必须引用原proposal并仅替换当前绑定允许变化的参数。不得沿用其他题、其他轮次的过期报价。Q的readout与其所检验的主张必须匹配。

N的报价是一次有界尝试估计，不是完成所有搜索的保证。若这段已经完成且没有新输入/新构造，不能仅凭ID变化再调用。

若N是从已付竞争程序计算实验分歧的`structure_planning`，第二段可限定为消费**该次实际返回**的`conditional_history_structure`实验。计划只保存这个返回绑定，不预填pulse／step胜者；实际Q仍须通过当前合法选项、来源匹配和完整报价。发布者的两段容量界限包括规划计算、最多12行新资料后的有界native研究、最大导出、全强池比较和父处理，并保存`followup_complete_quote`；它不是已选择Q或实际支出，首段`max_compute`仍只支付规划。Q须在实际规划返回后重新报价；无不同预测或Q不可支付时保留相应未决／负结果。

### 6.3 静态区别检查

在已曝光的调用前状态上，只执行不拟合、不预测、不购测的选择逻辑，比较普通proposal的语义内容：操作、所用证据、程序绑定、物理实验及资源范围。

路径变化、版本号变化、额外文字不构成不同内容。同样最终Program但不同实际研究成本，之后仍可能构成方法收益；静态阶段不以最终输出不同为强制要求。

仅在本次声明状态/工作单位中没有新增行动后果的候选，可以合并执行机会，但原方法和来源继续保留；有限面板上同动作不证明两个程序永远等价。如果差异在另一个合法但尚未出现的分支状态，应明确记为该分支未测，不假造见证，也不永久删除候选。

静态区别检测所需状态提取费用若在正式研究合同中计费，必须由调用者记账；“不买新观测”不等于所有计算都免费。

## 7. 调用前状态与小样本选择

### 7.1 只读状态

复用`method_profile`并按所选操作补：工作阶段、最后未完成请求、真实未消费证据、可辨实验是否存在、完整报价是否可支付、当前比较是否陈旧。

每个字段有原引用。未知值保留可用性，不自动0。覆盖条目按其真实语义定位，不能让等价集合的排列决定某条条件；不能把时间序列排序。

case名、路径、隐藏kind、生成seed、未发生的target、后验最优动作等不得进入条件。历史真值只可用于形成旧回报标签，不可穿越到该次调用前特征。

### 7.2 偏好标签

只从同一匹配起点的完整P0/P1对照得到相对标签。下面的内容不是相对动作标签：

```text
P0在另一任务上失败
P1从未运行
因此P1更好        # 禁止
```

保留`working_comparison`、`paired_future`、`process_return`的不同证据身份，不简单压成一个真假结果混训。原六次整方法使用提供提案和状态候选；没有替代动作结果的部分不伪造成有对照的监督样本。

**识别口径澄清（2026-10-27）：** 共同`data_ref`指决策前可用资料。若某臂在研究段中新增购测，比较识别的是“实验选择＋各自所得资料＋后续研究”的整体策略效果；不能据此单独归因资料或方法体。双方均零新增观测时，复用旧答案的Q仍属于同资料研究。资料效应及资料×方法交互须由固定D1、交叉程序的实际完整格识别，不能只填一个识别标签。

### 7.3 选择规则

在配对材料上，先比较原任务完成/质量/覆盖；共同新评价损失可用于未达标阶段的开发进度。相同质量下资源向量支配可以给出偏好；资源互换无外部偏好时为不可排序。

复用`learn_condition`提出最多8个单分支条件。枚举上限在看新结果前登记。只允许已有特征的阈值/等值/可用性及现有有限表达，不能根据case逐项查表。

出现以下任一种情况，不发布新效用条件：配对不足；标签在全部可见状态下仍冲突；只有同向结果无法定位边界；组外检查不支持；条件只靠未知填0分开；所有候选的普通动作相同。

允许保留有局部证据的新程序，不赋予一般条件；允许完全保留P0。没有正结果不得人为制造body或guard新版本。

## 8. 一个开发单位的精确运行合同

### 8.1 起点

优先读取case002首个真实方法调用前的已存72行状态及方法反馈所引用的父身份。它是已曝光开发材料。必须实际验证完整代码、候选池、数据、划分、问题题义和费用可以匹配。

若该前态无法恢复，终止历史配对准备；以当前完整145行状态另登记开发起点，不声称它是原72行。排除已经完成的同合同native段；只允许新且明确的候选内容。

原准备/研究已付结果可合法复用，但仅在实际合同匹配时。原历史单位的不同公共代码、暂停、缓存或比较范围，不能默认充当新对照的一臂。

### 8.2 四槽位设计

M0为固定旧方法体，M1为静态阶段预选的一个不同研究段。先不学习新guard，以免把“没执行”当成“执行更好”。

```text
00: M0(D0)
10: M1(D0)
01: M0(D0 + D1)
11: M1(D0 + D1)
```

D1是现有原语生成的一项具体区分/完整配对实验。允许由00/10已冻结预测产生实验提案，但必须在D1真值前保存提案。D1若无合法、有区别的定义，就只做00/10，不购买占位数据。

四格在相同资源上限内执行；每格的工作结果都通过原完整池比较，不能只比较新候选。所有标签可用于该格声明的拟合，Qeval不能参与任何格的拟合或候选选择。

### 8.3 评价

四个实际返回预测器先冻结，再对共同Qeval执行。建议小单位上限D1≤24行、Qeval≤24行，具体长度受实际公开接口限制；这些数值不是正式确认替代品，不改96行确认/512行终测。

**Qeval时序澄清（2026-10-27）：** 冻结前不允许Qeval影响任何格的拟合、提案、条件或D1。全部输出冻结后才揭示共同评价；揭示后的结果允许作为本开发单位的方法选择／条件研究标签，此后该评价已曝光，独立后继须另取未曝光材料。运行回执记录实际冻结与揭示阶段，未完成冻结、超活动窗或仅补固定模型读出的结果不形成策略偏好。

若某方法没返回预测器，则保留其实际未完成状态，不由宿主挑一个最好旧稿救场。若原方法合同本来就明确返回保留父稿，则照合同返回；不能临时改变。

数据购测一次共同提供适合检验数据×方法机制；全实验费用实际计一次。生产策略效用另外按每臂自己的采集决定评价，不将共享D1当作选择Q的免费收益。

### 8.4 结束后才形成新程序

保留每格完整结果、实际停止原因、预测器、动作与资源。生成器根据配对结果提出/保留有限条件，编译到普通方法程序，使用已有问题返回把它交给父研究。

在一个符合已声明用途的公开状态，实际通过Common调用一次，检验规则与普通动作是否一致。此调用可能花科学计算/观测，必须在本单位额度内提前报价；只跑选择函数不能冒充完整调用收益。

新反馈经原`method_feedback`返回方法问题。同一origin只更新原记录，不增加独立样本数。没有可支持的新程序时直接记录阴性，不执行独立后继。

## 9. 参考控制伪代码

下列名称说明拟议的连接职责，不表示这些同名函数已经存在。实现应复用表中原入口。

```python
def construct_one_method_revision(common, source_refs, contract):
    assert contract.base_commit_is_frozen
    assert contract.permissions_are_explicit

    episodes = project_existing_episode_refs(source_refs)  # 不调用拟合/购测
    conflict = find_actual_method_conflict(episodes)
    question = open_existing_method_use_question(conflict)

    snapshot = load_exact_decision_snapshot(conflict.start_refs)
    ordinary = unchanged_ordinary_policy(snapshot)
    alternatives = current_legal_complete_options(snapshot)
    candidates = propose_at_most_two_local_edits(ordinary, alternatives, episodes)
    candidates = group_current_scope_equivalent_trials_keep_originals(
        candidates, episodes.pre_use_states)
    if not candidates:
        return retain_unresolved(question, 'no_distinct_legal_alternative')

    # 只按预登记的可行性/合同差异选择待试候选；不虚构其预期成功。
    challenger = select_registered_trial_candidate(candidates, contract.tie_seed)
    trial = build_one_paired_study(snapshot, ordinary, challenger, contract)
    quote = quote_all_cells_and_evaluation(trial)
    if not admitted_under_actual_permissions(quote, contract):
        return retain_unresolved(question, 'not_admitted', quote=quote)

    outcomes = run_with_existing_Common_and_execute_procedure(trial)
    retain_all_outcomes_and_cost_refs(outcomes)
    preferences = derive_only_identified_pair_preferences(outcomes)
    revised = choose_supported_local_program_or_old_program(
        ordinary, challenger, preferences, max_conditions=8)
    if revised.has_no_executable_difference:
        return close_finite_research_with_negative_result(question, outcomes)

    artifact = install_existing_Python_method_artifact(revised)
    result = execute_one_prequoted_ordinary_use(artifact)
    return_existing_method_feedback(question, artifact, result)
    return artifact, outcomes, result
```

不能将伪代码的`choose_supported...`用人工填写正确分支代替。它必须有实际候选、有限条件合成、配对结果和可复现选择顺序；资料不足时保留旧程序。

## 10. 完整预算与授权门

计算报价必须包括：

```text
C_prepare
+ C_state_projection_and_proposals
+ Σ C_cell(search + export + complete_comparison + returned_parent)
+ C_D1_analysis
+ C_Qeval_predictions
+ C_method_compilation_and_installation
+ C_one_ordinary_use
+ C_feedback_and_preservation_as_charged_by_existing_contract
```

观测报价包括D1、Qeval以及实际普通调用/确认可能需要的新观测。只有明确授权独立确认时，才列入96行；独立后继阶段的512行按原评价协议单列。

每格先保留收尾，再分配研究额。能够由已有索引验证的已付评分/拟合应在报价中精确复用；保留完整强池并不要求重复执行全部旧预测。冷上界不能被写成必然实际支出，复用的索引/上下文开销仍按原合同计量。根据最大数据行数、最大导出和当前强池给出比较界限；研究部分是有限尝试估计还是完整有限枚举上界必须标明。

首轮不得复制旧446,856到不同数据或不同方法作为保证。旧1,137,587已证明可能漏准备开销。当前145行若池增加182，仅比较已为716,880，必须计入。

时间同时分native、外部Python、原计费P、活动墙钟、恢复和行政暂停。旧混合账不补造精确拆分。冷启动/热缓存公平但不消除真实部署成本。跨阶段/跨方法引用旧费用只记一次，未知继续给区间。

本文件`execution_authorized=false`。没有授权时允许只读提案和工程控制，不启动真实科学试验。`AnonymousResearchMethodLearner`的write/run改源码可能自动practice，未经授权不要调用该路径进行“纯写文件”。

## 11. 验收清单：应新增的行为测试，不是已经通过的测试

| ID | 应验收的行为 |
|---|---|
| T01 | 两次同名fit生成不同实例，比较依赖准确，不绑定最近一次名字 |
| T02 | 必要前提在最近六步之外仍保留；缺引用返回未决 |
| T03 | 中性准备步骤因实际依赖保留，不因即时gain=0删除 |
| T04 | 同method_id的两个调用不能串成已完成序列 |
| T05 | record7复用执行未完成回调；同题义同分析完成后不再收费 |
| T06 | 可用性未知与数值0分开，未来标签干预不改变调用前提案 |
| T07 | case/kind/path/seed改变不产生策略条件；合法时序改变可影响决策 |
| T08 | 集合语义覆盖条目重排不误改条件；时间序列不被排序 |
| T09 | 相同已完成研究仅改ID，不构成合法新N/R |
| T10 | 已有Q缺完整配对时不作提前归属；报价包括剩余所有角点 |
| T11 | 未运行替代动作无收益标签；原六份反馈不自动变成六份配对 |
| T12 | 新程序走select_actions消费者，原生DAG走replay消费者，不混派 |
| T13 | 一次普通Common调用实际改变操作/证据/实验或返回条件 |
| T14 | 同预测质量但较少观测的案例被记录为过程收益 |
| T15 | 资源一增一减无偏好时保留不可排序，不私设加权总分 |
| T16 | 合同内资源耗尽计入方法交付失败，但世界结构仍未被据此反驳 |
| T17 | 行政暂停另列删失且保留总费用；不训练成方法无益标签 |
| T18 | 新增N/K后报价失效并重算，不覆盖旧cap或自动加时 |
| T19 | Qeval标签不能改变四格已冻结输出；D1只在声明格中进入拟合 |
| T20 | 两臂可变状态隔离；公共缓存同能力；C目标产物不免费进入B |
| T21 | 同一次反馈重复摄取不新增样本和费用；后续补充覆盖保留origin |
| T22 | 没有不同可执行内容时不新版本、不独立重跑九项 |
| T23 | 实验完成无收益可正常收束，保留所有失败与原程序 |
| T24 | 旧controller原合同仍能运行，公共存储/运行/并行实现未被无故改写 |

工程测试支持接线正确，不支持科学净收益。至少一条实际生产调用链和对应回执才能确认新对象被真实消费。

## 12. 独立后继登记

当前原九项及512/144/49行材料全部保留为已曝光，不重新命名为测试。

有实际新内容后，首个建议单位为3个新物理来源×A/B/C=9项，仅检验整体差异。想分离条件和方法体，需同组织四格加A，共3×5=15项；预算不够就不作分解主张。

运行前明确：source来源、目标生成规则、冻结截止、安装成本、数据/计算/时间上限、哪个状态属于有效调用/错误迁移/过度拒绝的可检验范围。过度拒绝的真值须有实际可比的被拒方案运行，否则未测。

A/B可基于共同历史重构等价内容，不能人为削弱。若C只靠公共修复获得收益，归入公共修复，不归入学习对象。终测时不得热改方法或回灌结果。

## 13. 本次完成标准与退出条件

工程完成：现有引用能连接实际研究段，有限候选可由普通执行器运行。

研究完成：至少一个事前固定、完整且可解释的匹配试验结束，支持新候选、保留旧候选或明确有限阴性。

学习链完成：实际结果形成或修订了可执行内容/使用决定，普通调用消费它，并把后继结果交回原方法问题。

泛化成立：必须由另外冻结的未曝光后继支持；不能以本开发单位代替。

当没有不同候选、资料不足或全负时，记录具体缺失/有限否定范围即可结束。不得以“必须产生正结果”为理由重复同一个单位。
