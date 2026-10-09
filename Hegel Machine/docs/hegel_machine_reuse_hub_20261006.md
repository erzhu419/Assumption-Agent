# 黑格尔机能力总入口

0.1–0.181的181项工作统一登记在[能力目录](../src/hegel_machine/cognition/historical_capabilities.json)，完整逐版本说明见[历史索引](hegel_machine_historical_reuse_index_20261006.md)。目录包含现行源码、原生入口、报告、保留/撤回状态及替代版本；现在也关联了0.181之后的8条接入路径。代码查询、命令行和LLM工具读取同一份目录。

## 0.181之后发生了什么

**开发没有停止，版本编号停在0.181。** 当前源码包含后续未发布工作，最初0.181没有独立冻结快照。之后先修复了六类整链错误：旧覆盖压过新反例、确认预算重复扣留、近零回报误确认或撤回、合并候选丢项、同源不同程序串用回报、组合执行次序退化。修复后窄闭环12/12、48/48任务；双源闭环随后12/12通过。[原审计](hegel_machine_post_0181_audit_20261004.md)、[窄链修复](hegel_machine_narrow_cycle_repair_20261004.md)、[双源结果](hegel_machine_two_source_cycle_20261004.md)。这些已修好的问题不是当前阻塞。

**接续断在任务接口和方法采用。** 后续重点转到论文有限能力实验、真实LLM对照与普通Python档案接入。旧研究器消费受限`Program`和完整`KnowledgeLibrary`；后来的时序方法获得`instrument`与`predictor`；普通Python档案只提供`predictor`及72行已付前缀。模型、方法和反馈的前提变了，旧能力需要适配。见[实际适配代码](../src/hegel_machine/cognition/python_method_learning.py)。先前还混淆过运行时/压缩反馈、评价/新确认和编辑/提交，这些接口问题与数值算法分别处理。

**整段重写又丢掉了已有分支。** 完整程序选项曾展示24次而采用0次；一次新程序6/6成功，却删除便宜重拟合分支，测量288→864。随后作者40次调用仍用误差幅度决定修复，未发布。局部接回0.38/0.79/0.84的历史规则后，源实践5/5、目标10/10；同精度下测量少33.33%、计算少42.41%。这些结果促成了现成方法的采用与修订实验，数值拟合器继续复用。[程序采用负例](../artifacts/python_procedure_offer_luna_20261006/RUN_STATUS.md)、[重写退步](../artifacts/python_method_publication_luna_20261006/RUN_STATUS.md)、[局部接入结果](../artifacts/python_current_evidence_bridge_20261006/RUN_STATUS.md)。

**最新接入：采用后遇到新反例，模型能修订研究程序。** 直接修订实验0/3作者发布；编辑后即时返回真实付费实践的实验2/3发布，两份程序均6/6实践、各12/12目标，原程序为10/12。首份成功源码已原样保存为[响应研究重开方法](../src/hegel_machine/cognition/python_response_reopening_method.py)，关联0.117的响应反例重开；旧人工桥接程序保留。[直接修订结果](../artifacts/python_method_reopening_luna_20261006/RUN_STATUS.md)、[编辑实践结果](../artifacts/python_method_edit_practice_luna_20261006/RUN_STATUS.md)。

**父方法保留已接入提交。** `PythonArchiveMethodLearner.finish`在新稿实践或原发布门槛失败后，复用恢复、实践及原发布器，检验受支持旧稿并补其自身确认。真实三作者中1次由此发布，旧稿6/6、冻结目标12/12；另两次为服务容量错误和未提交。32项测试通过。[本轮结果](../artifacts/python_retained_method_comparison_luna_20261007/RUN_STATUS.md)。

**模型自主取回已取得重复结果。** 将已有实践支持直接呈现在`method_decision.json`，仍由作者选择候选。关闭主机自动回退的20次固定尝试中，保留组10/10主动取回、8/10发布，移除组2/10发布；7次原样发布、1次取回后编辑，2次取回后未提交。每个发布均满足原六案例要求。[本轮结果](../artifacts/python_autonomous_recall_luna_20261007/RUN_STATUS.md)与[主文5.2](../paper/manuscript.md)及[补充F.11](../paper/supplementary.md)区分取回、发布和跨任务证据。

**论文修订新增冻结迁移与归因定位。** 20作者的10次发布按字面源码去重为4稿，新24配对目标各24/24、固定修复21/24，新增120离线执行、0模型调用；3份改稿已有后继证据，但不优于成功旧稿。[迁移结果](../artifacts/python_recall_frozen_transfer_20261007/brief.json)。既有冻结Python目标实际使用普通`code_react`，强flat与recursive同60/60，conditional/unmarked同90/90且同费用；统一same-history组织优势尚未成立。[归因审计](../artifacts/executable_archive_controls_20261007/evidence.json)。另用原冻结runtime复现20个保留/删除首修失败，16关系歧义停研、4误确认，不重复修已有通用方法库。[停止定位](../artifacts/paper_retention_stopping_audit_20261007/summary.json)。

**第二轮论文意见已落实为收尾修订。** 主文区分预测程序、研究程序与研究偏好；补充Table B2标出提出者、学习对象和固定接口，保留泛函分析的组织性形式化。已完成的20例停止诊断、F.12四份源码迁移及同预算空效应不再列为新缺口。[主文](../paper/manuscript.md)补齐删除范围、历史SC-KG前缀来源与理想化边界。[表格生成器](../scripts/regenerate_manuscript_tables.py)直接消费保存JSON并进入主文构建；解压包的代表性冻结数值重跑6链通过、无模型调用。公开永久归档与统一同历史组织优势仍未完成；新DiscoveryWorld试跑用于难度开发。

此前只有长篇版本目录和按轮追加的“下一步”，最新接入没有结构化关联回旧能力，容易把已完成的算法再次列为新缺口。现将问题检索、当前输入要求、已有证据和剩余缺口放在同一入口。

**泛函分析已接回论文现有实现。** [主文3.2–3.3](../paper/manuscript.md)与[补充A.3/B.2](../paper/supplementary.md)将观测签名、可区分性、表示和实验修订对应到旧online/growth研究器：有限响应网格等价类、校准后改探针、实测换视图、方差模型及简化后返回。沿用0.70/0.76/0.77/0.84/0.92等实验知识与方法修订方向，不另建“泛函引擎”。新测量30/30对0/30、名义95%区间实际91.09%等既有结果保持；一般空间自主构造尚未实现。

### 泛函方向以前是否尝试过（2026-10-07复核）

**做过，且有限范围内已有成功。** “一般函数空间、观测族及通用研究器尚未完成”不能解释成这三条线没有实施。这里区分设计边界、历史失败和当前接入遗漏。

| 方向 | 已有尝试与原始结果 | 尚未成立的部分及原因 |
|---|---|---|
| 表示与假设空间 | 0.26学习投影并挑战零空间；0.30无家族标签的有限元先验研究，保留知识24/24、逐题18/24，3题最终采用递归子答案。[原结果](../artifacts/meta_prior_research_efficient_v030_20260910.json) | 组合、投影、残差和可调用定义扩展了有效搜索语言，但底层算子语义和候选构造族仍有人工边界；没有一般空间自主发明的完成证据。 |
| 实验及观测构造 | 0.70联合求边界，0.76/0.77重绑和改写时序步骤；0.92真实学习四步方法、修成五步并跨装置复用。[0.92结果及原限制](hegel_machine_instrument_method_revision_20260919.md) | 0.92两版均24/24，计算只少0.46%；新实验有效不等于持续收益。Python作者后来也能写具体trajectory，但没有一般观测族发明的独立证据。 |
| 统一研究 | 0.59同一循环研究参数、历史、非线性、仪器和残差，48/48；Growth方法通过原知识库与执行回调接入旧研究器。[0.59](hegel_machine_unified_research_20260912.md)、[共享方法原结果](../artifacts/workflow_bridge_20261006_main_12/results/summary.json) | 接口统一已经发生；共享方法两臂均72/72，保留反而多5.91%观测、8.14%测量和2.74%计算。不同任务接口尚非自动通用；LLM选择、知识新增价值及长期收益也不由接通本身保证。 |

还有一条更早的真实失败：[规范程序枚举的终止记录](../artifacts/phase3_m3_runtime/formal_m3_terminal_outcome_v1.json)显示3,292,439次原始算子应用后达到50,000个规范程序，触及数量上限，目标评价尚未开始。后来改向观测行为的商化去重。这是搜索规模问题，不能据此说“泛函分析被实验否定”；该旧正式轨道的局部结果也不代替认知主线证据。

后来的失败同样需要具体定位：0.111有边界实验反复不缩小歧义、计算耗尽而仍剩观测的问题；0.130累计研究67/72，逐题70/72，累计CPU为11.16倍，暴露重复搜索与投入选择。[0.111诊断](../artifacts/joint_response_discovery_20260922/fresh_replication/empty384_failure_diagnosis.json)、[0.130](hegel_machine_cold_temporal_discovery_20260927.md)。这些是当时的版本结果，后续已继续修订，不能直接列为今天仍未修复的错误。

**本轮已补仪器边界方法的积累断点。** `plan_instrument_disagreement`能生成并实测`learned_boundary`，但`observe_instrument_method`原先只接收pulse/step/random，边界求解经验没有回传。现在两条路径共用[当前切点及相邻中点构造](../src/hegel_machine/cognition/instrument_hypotheses.py)，在[原方法库](../src/hegel_machine/cognition/instrument_experiment_methods.py)保存`at`或`between_adjacent`求解策略，后继任务用当前候选重解，不保存旧点或固定长度。只有实际读数淘汰候选后才增加方法支持。

[两类迁移回归](../tests/test_cognition_instrument_experiment_methods.py)修前均复现源实验成功却未入库；修后源购测→JSON恢复→新控制名/字段/范围/阈值→当前边界重解→实购区分通过。测试还确认旧点按范围迁移已无区分力，且无购测、无预算或无分歧时不增加支持。[原UnifiedObjectLab实验](../tests/test_cognition_instrument_competition.py)也确认边界方法实际入库。共23项定向测试通过，旧序列延长和原调用路径保留。这是接入回归证据，没有新增科研benchmark或模型调用；更广未见任务收益未据此宣称。

本轮验证命令（项目根目录，结果为23项通过）：

```bash
PYTHONPATH=src python3 -m pytest -q tests/test_cognition_instrument_experiment_methods.py tests/test_cognition_instrument_competition.py tests/test_cognition_instrument_method_runtime.py tests/test_temporal_instrument_method_book.py
```

“通用”也不能以没有任何预设原语或实现无限维空间为验收条件。更具体的问题是：现有观测无法区分时，能否依据候选构造新的可执行实验，将返回证据用于同一父问题，再在改变条件的后继任务中复用这个构造步骤。有限实现的正结果、接入回归测试和未见任务的独立收益应分别解释。

**方法的未见任务收益已有独立比较，不应再次整体列为未做。** [Growth方法库删除实验](../artifacts/growth_paper_20261006_main_30/summary.json)从相同stage3实测快照出发，仅删除方法库、保留预测程序与候选搜索；30个新角色/仪器次序目标均30/30，保留方法使观测5,088→4,028、计算5,219,559→3,944,154，测量同为7,200。它支持有限任务族中整个方法库的边际收益，未隔离单个构造动作。0.113的[12源/216配对后继](../artifacts/boundary_method_transfer_20260922/fresh_replication)也已做过响应方法留删；新背景改写没有独立精度增益。这里复用原结果，不重跑两条旧路线。

**新补的仪器边界分支现也完成留删比较。** [预定协议](../artifacts/instrument_boundary_recipe_20261007/protocol.json)包含30个源，每源两个独立后继：新装置/测量改名，以及对象数4→5。目标共享同一实测前缀与拟合候选，只删继承的边界recipe；两臂均保留现成完整规划器和本地学习。30个源全部实测入库，保留组在60个目标中实际采用并区分60次；两臂仪器终测均60/60。[结果](../artifacts/instrument_boundary_recipe_20261007/summary.json)、[运行入口](../scripts/run_instrument_boundary_recipe_study.py)。

| 含每源一次学习成本 | 保留边界方法 | 只删除边界方法 |
|---|---:|---:|
| 观测行数 | 4,620 | 4,620 |
| 测量成本 | 60,120 | 60,120 |
| 计算计数 | 743,280 | 1,409,280 |

总计算减少47.26%；仅后继计算减少66.29%。终测另计、不回传，用随机点、公开轴端点和半格边界检查全部存活仪器解释；负对照确认未消歧的真实前缀会因边界误差1.0失败，评价不修改模型。全程离线，没有模型调用。

这条分支的剩余范围：此次收益来自复用现成构造策略、跳过重复完整搜索；两臂选出的实测点相同，没有精度或物理样本增益。30个源的计算差均为22,200，源聚类区间退化为同一数值，不能当作广泛结构变化下的不确定性估计。角色schema仍由环境提供，半整数成员关系家族未变，`between_adjacent`仍只有迁移回归；未比较人工固定“边界优先”的规划器。后续若推进一般构造，应转向现有候选/观测原语不足时的结构发现，先查旧对象发现与观测扩展实现，不继续调优本项重复搜索。

**递归采用不是历史空白，DW此前漏接了程序合同。** [真实Luna旧轨迹](../artifacts/python_research_advice_ablation_luna_20261006/RUN_STATUS.md)已经完成仪器子研究、区分实测和父解释修订；[持续提醒](../artifacts/python_pending_advice_luna_20261006/RUN_STATUS.md)后来仍4/6对4/6，程序选项也有0/24采用。当前已将原`execute_procedure`直接接入DW，恢复程序返回问题、采集后分析、旧证据分析与逐步反馈；旧数值回归保留。两臂共享完整stdout，子步源码/输出分开保存；外层决策课题不会覆盖程序子问题。[新24轮检查](../artifacts/discoveryworld_procedure_adoption_20261007/RUN_STATUS.md)找到对话控制并提出未确认规律，但0子问题/0分析程序，已结束。公开任务根和必填但可为空的课题声明不算自主发现。当前缺口是开放环境研究步骤的选择和生成，旧数值成功算子不能直接解释世界对象/对话记录。

## 按问题找到实现

**DW研究控制接入修复（2026-10-07）。** 0.62已有的动作JSON错误付费反馈已接回；公开对话菜单在USE入场和Exit后同步，真实官方环境2项回归通过。新增[公开数值绑定](../src/hegel_machine/cognition/world_relation_research.py)复用`synthesize`、`choose_probe`和原`execute_procedure`：模型从实际记录绑定变量/实验，系统拟合并保存候选、选择区分动作、冻结候选独立确认，子答案再驱动模型父程序。8项宿主闭环测试、9项数值研究测试、8项错误恢复测试通过；原程序回归通过。[v3](../artifacts/discoveryworld_pilot_20261007_v3/RUN_STATUS.md)与[v4](../artifacts/discoveryworld_pilot_20261007_v4/RUN_STATUS.md)均已停止。v4停止检查点Hegel为147调用、146动作，0子问题/程序/数值绑定；Code为159调用、226动作，末期执行1次根问题程序、无子回传。Hegel在修复前已通过USE于观测139打开菜单、142/145取得参考频率，不能将这次自发发现归功于后续修复。

**DW非数值操作前提现已出现真实回传。** 旧数值研究器依赖已绑定控制/测量，未找到已解决未知设备操作语义的历史研究器。现在[操作失败定位](../src/hegel_machine/cognition/world_prerequisite_research.py)进入[Agent主循环](../src/hegel_machine/cognition/discoveryworld_agent.py)：同对象重复不支持的操作触发前提课题，模型编写`select_actions`和`answer`，沿既有程序执行与采后分析回传；两臂共享[公开交互角色](../src/hegel_machine/cognition/discoveryworld_lab.py)。19项定向回归通过。[固定前缀配对检查](../artifacts/discoveryworld_prerequisite_check_20261007/RUN_STATUS.md)共用Hegel的106调用/104观测历史，各追加12调用：Hegel首轮编写一份程序、执行两个函数，取得1个受支持子答案，父任务第3/4轮读到3880.56/2239.92 Hz；Code第7轮找到交互入口、第11/12轮取得相同频率。初始子课题标为`host_execution_failure`，不是模型自主触发；Hegel末期另自主提出`microscope_test`并取得读数，但尚未分析回传。两臂任务均未完成，过程分数Hegel 0.60、Code 0.64；这支持本失败状态下的接入，未建立任务优势或数值研究自主采用。

**再次核对后的工作边界：复用已有研究链，修任务语义接入。** 候选生成、区分、冻结确认、子答案推动父修订均非新能力：[0.22](hegel_machine_discriminating_research_20260909.md)、[0.59](hegel_machine_unified_research_20260912.md)已有实现；[真实Luna原轨迹](../artifacts/python_research_advice_ablation_luna_20261006/trial_003/training.json)在第5/6/8/9/11个零基事件完成仪器拟合、区分、实测淘汰、父响应修订和新确认。当前DW的`research`已接通`bind → synthesize/choose_probe → after_observation → execute_procedure → parent_path`。最新模型未调用这条路径；显微镜子问题也没有提交分析源码或采后分析步骤。两者是未采用现成协议，不能据此断言求解器失效或子答案丢失。

实际适配差异是证据含义：旧数值夹具`READ(x)`直接返回真实目标；反应器菜单报告设定频率，另有公开激活/未校准状态。设定值不能直接当正确频率标签。旧`synthesis`已有0/1关系与布尔组合，`choose_probe`支持布尔候选区分；已有`query → after_observation evaluate → memory.answers`可分析实际干预反馈。下一步只补这层公开结果绑定及已有分析步骤的实际采用，不另造搜索器、确认器或递归执行器。原12轮检查在显微镜问题提出后两轮结束，尚不能证明模型在更长运行中永不收尾；本次核查没有新模型调用或重跑实验。

**上述分析续接现已实测。** 模型声明子问题并采得新证据后，Hegel主循环安排模型编写`answer`，直接复用旧`evaluate`；所有有效判定均释放本轮焦点，无新证据不重复审阅，科学解释由模型给出。两臂共享设定值/实际响应的接口说明，原数值与布尔研究器未改。续跑脚本也修复多次接续的费用累计、程序注册和研究状态恢复。15项新增定向测试及原相关回归通过。[完整旧现场续测](../artifacts/discoveryworld_analysis_continuation_20261007/RUN_STATUS.md)中，Hegel第一轮完成显微镜分析，无重复测量；32轮共新增3次函数执行、2个子答案，64动作，仍未完成任务（过程分数0.68）。其逆平方间隙假设停留在文字与手工干预，数值绑定仍0；公开结果并未丢失。普通Code首轮调用旧研究器但没有提供probe，正确返回歧义；完整服务/结果状态见该运行记录。当前缺口是模型将公开干预结果组织成可执行假设检验，不能再称为旧求解器或回传链缺失。

**已知操作后的父假设续研接线已修复。** 先核对[0.118](hegel_machine_instrument_response_research_20260926.md)的“仪器已支持、父响应仍失配”旧路径，确认本次不是新算法。DW原来用任何同对象supported答案永久抑制后续研究；现在按操作答案及后续分析的实际审阅边界计新失败，保留TALK答案，经旧程序执行提交父干预检验，三类有效结论均释放焦点。[13项新增回归](../tests/test_discoveryworld_intervention_followup.py)及相关旧测试通过。[真实续测](../artifacts/discoveryworld_intervention_followup_20261007/RUN_STATUS.md)两臂各12次调用、无超时，Hegel首轮编写并执行两个函数、复用旧观测返回inconclusive；新增动作19/50，两臂分数均0.68且未完成。没有频率规律的确认或任务收益。续测后统一了method_decision与执行路径的父依赖计算，22项测试通过；这处摘要修复未重跑LLM。下一处是公开物理状态如何改变科学候选及实验选择，仍先复用旧仪器—响应选路。

**公开物理状态现已进入共享行动反馈与父干预证据。** 复核0.84/0.118后，确认旧数值路由需要已绑定真实响应，不能把TALK支持当数值仪器校准。此次补齐摘要中遗漏的对象状态、容纳关系，以及焦点中的成功操作/读回；完整观测原本一直存在。新增8项回归，相关46项通过。[主续测](../artifacts/discoveryworld_public_outcomes_20261007/RUN_STATUS.md)各24调用无超时，两臂仍0.68未完成。Hegel第22轮拟合新幂律，预测1353.093/916.361 Hz，但随后未实测；新2次函数执行属于生成器操作研究。[未审阅现场3轮检查](../artifacts/discoveryworld_outcome_focus_20261007/RUN_STATUS.md)实际采用新父焦点，引用安装/读回/未校准状态并反驳具体激活预测，但分析是固定旧证据的回顾。接入有进展，候选差异尚未驱动区分实验，无任务收益。

**父问题返回后的程序入口已接通，但尚未转为科学检验。** 复核0.69/0.118及旧Luna成功轨迹后，沿原`execute_procedure`补齐无现成程序的父续研；保留自定义程序、直接分析与数值绑定入口，21项新增/83项相关回归通过。[真实同230观测续测](../artifacts/discoveryworld_parent_research_20261007/RUN_STATUS.md)各12调用：Hegel首轮生成并执行父程序，却误用全局历史长度而提前停止，新增1函数/0答案，后续未测新候选；Code实际设置1353/916 Hz并重装晶体4，仍未校准。两臂均0.68未完成。运行后已澄清按程序path读取历史，未据此宣称模型已修好；下一步聚焦已生成程序的状态分支和物理反例后的候选修订。

**停止结果已接回程序修订。** 历史`EditPracticeMethodLearner`已有实测驱动改稿；本次复用旧执行器，将DW已有父程序的实际stop/error接回一次完整动作复议，并提供按程序路径的执行历史。新增15项/相关98项测试通过。[同241观测续测](../artifacts/discoveryworld_parent_review_20261007/RUN_STATUS.md)各12调用：Hegel6次改稿、26次函数执行，真实行动从首轮开始推进；第六稿却在TALK/Exit间重复直至12步上限，新频率仍未实测。新增动作Hegel30/Code12，两臂均0.68、0新子答案。模型会修执行程序，尚未让反例改变科学研究；下一步转向操作必要性与规律失配的区分，而非增加调用次数。

**定性区分实验已接回旧选择器，实际采用仍未出现。** 先复核0.79/0.84/0.118，新增作者物理预测到`choose_probe`的适配，沿旧采集／分析回传；修复负子答案返回、整段程序步数上限复议，以及设计修订后旧支持未撤回，135项相关测试通过。[同271观测现场24调用续测](../artifacts/discoveryworld_question_choice_20261007/RUN_STATUS.md)：Hegel18次函数执行、2个自主操作子答案、92动作，Code72动作；Hegel实际试验竞争频率但未确认，均0.68未完成、`contrast`采用0。另修正父复议提示无条件要求改稿的冲突，同前缀6调用复查仍无采用或任务收益。下一步用公开成功参考检验额外条件是否必要，再让实际响应淘汰候选。

**新子答案已能重开现成父研究，未决结果已实际进入区分设计。** 旧父程序复议游标曾屏蔽后来返回的答案，现沿原焦点按答案快照接续；两臂共享原始参考状态投影。单纯接续／暴露的[可选检查](../artifacts/discoveryworld_reference_return_20261007/RUN_STATUS.md)仍0采用。随后将新未决子答案接入现有`contrast`：[同363观测检查](../artifacts/discoveryworld_reference_design_20261007/RUN_STATUS.md)Hegel21调用、7函数、1区分设计、2操作答案、64动作，Code15调用、92动作；均0.68未完成。参考反应器拒绝DEACTIVATE，作者限定了操作结论，未完成参数扰动／恢复或频率确认。157项相关测试通过。入口安排属于主机研究策略，不计自主方法选择；旧父程序保留，科学实验质量仍是当前缺口。

**反馈式区分实验已接回原执行器，分析修复保留实验源码。** 上轮原始记录证实答案补稿覆盖了`design`；另有固定动作`contrast`无法直接调用现成逐步程序的接口限制。现允许选中实验引用原`execute_procedure`，固定预测后逐步读取反馈；分析独立保存，重跑可继续使用修复稿。95项相关测试通过，含新增12项基线／扰动／恢复、失败中断与源码保留测试。[同363观测续测](../artifacts/discoveryworld_adaptive_reference_20261007/RUN_STATUS.md)Hegel23次响应后SIGTERM中断，Code24调用完成；Hegel预测全同返回歧义，仍无反馈式采用，两臂最后0.68。独立人工公开控制诊断以4操作实测参考activated→uncalibrated→activated，无ACTIVATE，证明对照可行。旧固定动作也能完成简单正负调节，因此接口限制不能当成上轮错误选择DEACTIVATE的已证实原因。

**方法供应已接回历史摘要与调用入口（2026-10-07）。** 复核旧Python方法取回路径后发现：此前DW两臂方法库为空，成功诊断脚本从未交给模型。因此旧结果支持“未自主构造对照”，不支持“已供应方法取回失败”。现在两臂共享同一人工来源程序、独立公开实践和原`procedure`完整调用入口；比较阶段也允许直接取回，不强迫重写。新增11项回归及原相关测试通过；[新程序真实实践](../artifacts/discoveryworld_supplied_reference_20261007_practice/result.json)以4动作/5函数/0LLM再次完成参考扰动恢复，问题保持open。独立档案不充作本轮观测。[配对续测](../artifacts/discoveryworld_supplied_reference_20261007/RUN_STATUS.md)已正常结束：Hegel16调用/58动作/26函数，Code11调用/51动作/0函数，均0.68；无供应方法显式执行。Hegel自行实测886设定顺序及870/916频率，取得未校准反例，但分析代码误读记录／容纳描述，正式子答案仍未正确回传。

**旧反例的分析修复已实测回传（2026-10-07）。** 先核对现成公开对象投影、evaluate与父答案快照，确认无需重建回传器。本次统一程序观测视图，保留历次冻结试验，并支持单独evaluate改稿；12项新增、合计62项相关测试通过。[同421观测指定复查](../artifacts/discoveryworld_analysis_repair_20261007/RUN_STATUS.md)两臂各8调用：Hegel前两轮不加观测，正确否定916Hz安装预测及886Hz空置设定／插入预测，父研究收到两份答案并继续；Code仅修复前者。后续新增50／56动作，均0.68未完成。此次由操作者指定问题及索引，未证明自主找错，亦未证明各接口的独立收益。下一缺口是反例改变父假设与区分实验选择。

**反例后已实际取回参考方法，科学修订仍有断点（2026-10-08）。** 复核0.79／0.84／0.117／0.137后，复用旧contrast入口处理有冻结试验的refuted返回，并保留作者改写的问题；两臂共用完整USE/READ摘要。7项新增、57项相关回归通过。[新续测](../artifacts/discoveryworld_counterexample_revision_20261007/RUN_STATUS.md)中Hegel首轮取回供应的reference_control，4动作完成扰动恢复，次轮模型限定支持参考读出；两臂各8调用，新增17／70动作，均0.68未完成。之后仍沿间隙试值，未运行新contrast。1296Hz分析发生uncalibrated子串误匹配，笔记认错但正式支持未撤回。[单独人工适配](../artifacts/discoveryworld_answer_withdrawal_20261008/brief.json)复用本轮正确1297Hz分析，经旧evaluate以0LLM／0新观测撤回，父题重开。原错误实验保留；不能将人工修复计入模型自主成功。

**已保存设计接回实际执行入口（2026-10-08）。** 复用旧method_decision呈现路径、contrast选择器和evaluate；成功写入的design/answer稿件现在返回可调用动作，并从顶层／父程序／采后步骤的真实历史恢复。保留独立修好的分析；分析焦点可直接evaluate旧答案。[同438记录续测](../artifacts/discoveryworld_saved_design_adoption_20261008/RUN_STATUS.md)两臂各8调用完成，Hegel／Code新增15／10动作，均0.68；模型修复当前分析，但旧1296错误支持仍在，contrast均0。实际引用428已滑出所有观测摘要，现以原投影在答案旁持续呈现。[同原现场复测](../artifacts/discoveryworld_cited_answer_adoption_20261008/RUN_STATUS.md)两臂各8调用完成：Hegel第7轮自主以原428撤回错误支持，第8轮父研究接收；Code未撤回。新增12／20动作，均0.68，contrast仍0。修稿曾覆盖设计函数，现复用独立分析路径，并以原模型决策完成[保留回放](../artifacts/discoveryworld_cited_answer_retention_20261008/brief.json)，0新LLM／观测；原实验不改。19项新增、79项相关测试通过。下一续接源见保留回放brief。

**普通实测反例与原位修稿接回已有比较（2026-10-08）。** 复用0.79／0.84／0.117／0.137的响应修订、输入条件诊断和比较流程，补上query/evaluate反例及保留路径write两处调用断点；Code恢复相同知识时不继承Hegel主持焦点。6项新增、80项相关测试通过。[449记录首测](../artifacts/discoveryworld_draft_counterexample_choice_20261008/RUN_STATUS.md)各12调用，Hegel提出新条件和密度变量，但旧路径改稿只保存，密度线性预测算错，均0.68。[原位修稿复测](../artifacts/discoveryworld_draft_counterexample_choice_20261008_v2/RUN_STATUS.md)各12调用完成：Hegel／Code实际contrast为8／0、新函数执行34／0、新动作29／51，均0.68。Hegel两次错误支持随后自主撤回，1723目标首次未达后也修正并实测；“空置时调频”仍未执行完整。8次比较含修复重试，不等于8次新发现。

**实际计算候选后完成任务（2026-10-08）。** 历史核对确认数值研究器和阶段执行均已接通，v2取放循环是作者分支错误；本轮只改比较复议提示，明确已有计算入口与成功阶段程序的用法，接口测试1项通过。[同478记录续测](../artifacts/discoveryworld_computed_phase_repair_20261008/RUN_STATUS.md)两臂上限24调用：Hegel实际4调用／32动作，Code24／51；环境成功为是／否，分数0.84／0.68。Hegel用普通Python正确拟合此前已提出但算错的密度线性候选，实测并分析晶体3，再用于晶体4完成任务。未提供科学变量或公式；本轮没有新采用bind／contrast／阶段程序。源链279次调用的成本保留，单现场成功不构成独立递归归因或整体性能结论。

**源方法整理与新任务迁移（2026-10-08）。** 历史冻结/重绑/编辑实践机制已核对；此次将源作者文件和独立经验接入旧方法注册器，两臂从零当前观测开始。源端20调用登记未验证程序，真实新任务见[本轮记录](../artifacts/discoveryworld_source_method_transfer_20261008/RUN_STATUS.md)。两项新种子Hegel已结束：seed1为32调用/101动作、0.52未成功，最后混用了源成功记录；seed2为42/156、0.80成功，改用当前gap关系，子实测支持回传后用于第二目标。原程序在三次采用中均零动作停止、未改写；成功来自后续研究。Code seed2首轮停研；seed1为52调用/329动作、0.52未成功，第52请求900秒超时，对照未跑满64调用；也出现源记录和对象身份混淆。19项相关测试通过。下一步复用旧编辑实践/父稿保留，将源端登记接成实际实践后修订；两目标已经打开，不再用作新确认。

**持续子研究与源端真实实践已接通（2026-10-08）。** `run {operator:'inquire',question:{...},max_model_calls:8}`复用同一Luna调用器、工具和账户，在本题笔记与执行反馈下编写、运行、改错、再分析；子层finish返回父题。程序可调用该循环并在子答案返回后继续。[真实局部修复](../artifacts/discoveryworld_local_llm_repair_20261008/RUN_STATUS.md)9调用、4次改稿及执行，取得65.020 g/cm³并回传支持；主机指定子题，仍有重复读数，整题未完成。源端复用旧`PythonArchiveMethodLearner`的编辑实践、精确源码支持与父稿比较；[首次20实践](../artifacts/discoveryworld_method_edit_practice_20261008/RUN_STATUS.md)仅准备现场成功一次，从头19次未过、0自主委派。已修正完整实践反复内联造成的输入超限及作者笔记未持久化，用原`restore_practice`保留全部候选与费用续跑。

**实际失败已接入连续子研究并完成源现场恢复（2026-10-08）。** 沿用旧前提研究、子LLM、父答案回传及精确源码实践，补齐三处断点：源实践绕过研究入口、子请求误以整题为目标、父续跑绕过再次修复。[首次16调用](../artifacts/discoveryworld_automatic_repair_20261008/RUN_STATUS.md)未返回的结果保留；[限定子题后8调用](../artifacts/discoveryworld_automatic_repair_scoped_20261008/RUN_STATUS.md)已实际修稿、分析并返回父程序。[同现场继续](../artifacts/discoveryworld_automatic_repair_continued_20261008/RUN_STATUS.md)再11调用，第二次失败自动进入子研究，两目标实际激活，环境`completedSuccessfully=true`，过程分数0.40；成功恢复链共19调用，相关46项用例通过。触发来自主机执行策略，研究与改稿由Luna完成。

**成功修复稿已进入原样实践和发布（2026-10-08）。** 同源修复经验经`prepare --repair-result`返回旧作者；源实践将成功改稿作为候选，在同案例零LLM重跑通过后保留。分析与`select_actions`程序分开保存，原候选与修复稿也分开。真实64调用作者的首份成功修复稿已原样通过初始34动作和续接新增31动作；[15条已付决策重放](../artifacts/discoveryworld_repair_retention_replay_20261008/RUN_STATUS.md)进一步接通自动确认、保留和发布，无新LLM调用。原作者未发布结果保留。

在项目目录运行；多个空格分隔的关键词需同时命中，英文忽略大小写。

```bash
PYTHONPATH=src python3 -m hegel_machine cognition historical --query refit --compact
PYTHONPATH=src python3 -m hegel_machine cognition historical --query '来源 反馈' --compact
PYTHONPATH=src python3 -m hegel_machine cognition historical --query '研究程序' --compact
PYTHONPATH=src python3 -m hegel_machine cognition historical --version 0.174
```

现有LLM算子同样可查询：`run {operator:'legacy_capabilities', query:'refit'}`。结果给出历史编号、状态、现行接入、所需输入、实际证据与剩余问题；`compact:false`可查看完整实现定位。找到版本后，原`--execute`入口继续运行该能力的现行实现及显式任务参数。

| 当前问题 | 历史与现行调用路径 | 当前边界 |
|---|---|---|
| 继续旧统一研究 | 原`UnifiedResearchSession` → `legacy_research`；传完整知识库 | 原生研究已修复；时序适配不覆盖所有原生环境 |
| SC-KG自主元先验 | `MetaPriorSession.run_deploy`、`objective_research`；`cognition prior`；`run_sc_kg_structural_prior_comparison.py` | [历史优化证据](../artifacts/kg_op_paper_integration_20261007/evidence.json)与[旧后端full配置直接对照](../artifacts/sc_kg_structural_prior_comparison_20261007/brief.json)已入论文：同prefix下Hegel对full 9胜；独立源训练及高维比较待做 |
| 表示、组合及方法步骤 | `GrowthResearcher` → `KnowledgeLibrary.research_workflows` → 旧执行回调 | 八能力已有有限实现；共享方法仍无稳定整链净收益 |
| 区分实验与方法复用 | 0.92等方法库 → `disagreement`/`assess_instrument`及响应对应算子 | 须采购新轨迹并回传实际记录；候选族仍有限 |
| 来源贡献、撤回与回访 | 0.157/0.180/0.181等 → `LegacyInquirySession`及原回报链 | 先注册真实来源会话；LLM自主回访未成立 |
| 旧Python假设返回 | 旧Code归档 → `research_choice` → 新确认 | 已有真实自主回取；全部记录比较不等同范围修订 |
| 新旧证据及修复选路 | 0.38/0.79/0.84/0.117 → `python_response_reopening_method`、`refit(records=...)` | 旧程序供应库采用3/3；受反例驱动的新程序修订2/3发布，两份各12/12目标 |
| 研究程序生成与修订 | `EditPracticeMethodLearner` → `PythonArchiveMethodLearner`父稿比较／发布 → `PreparedPythonMethodAgent.bind_method` | 保留主机实践与父稿比较；另在关闭回退的实验中，模型10/10自主取回、8/10发布，对照2/10 |

181项按原目录分为162项能力保留、9项撤回、8项被替代、2项旧正式协议。旧正式验证协议仍由`formal`入口承载；对象、延迟反馈和SC-KG等能力保留原生入口。撤回编号指向当前替代实现。八条新关联描述当前接入，不改变这181项的历史归属。

目录聚合时9项检索、Agent接入及原生执行测试通过，三个问题查询找到对应接入；这次目录聚合本身无模型调用。[聚合验证记录](../artifacts/historical_capability_aggregation_20261006/summary.json)。

调用新方法时，将模块的原样源码交给`PreparedPythonMethodAgent.bind_method(source)`，再经`execute_method`运行；已有预测器、72行已付前缀和当前`research_state.json`沿用原接口。登记源码不会自动替每个Agent选择它。

## Limitations与下一步

**当前进度（2026-10-09）：** 已复用旧Program和数值研究器，将[解释提案](../src/hegel_machine/cognition/world_llm_inquiry.py)的可选`bind`接到`comparison_action → research_update`及`research_action → research`；原`design/answer → contrast_action`保留。父研究可先让同一模型比较多条已付观测，再选是否购测。3项新增、28项相关测试通过，覆盖冲突排除、同题冻结候选续验和父层选择；提案本身不执行实验或改变科学答案。[真实Luna试跑](../artifacts/discoveryworld_executable_proposal_luna_20261009/RUN_STATUS.md)的提案阶段主动在6/8调用结束，未写`bind`，两项数值动作仍为空；父续研仍运行，接口通过尚未转为实际数值采用。

**当前边界：** 同题续研保留冻结候选，编辑`bind`不替换候选库；新候选族暂需独立关系题，新响应复用物性引用不自动算独立数值确认。新试跑由宿主启动提案阶段，不能据此归因自主选路。[原中性续研](../artifacts/discoveryworld_joint_response_luna_20261009/RUN_STATUS.md)已以`provider_failed`结束：新增29调用（27次有效决策、1次中途JSON错误后恢复、1次末尾请求超限）、151动作，仍15/29未成功；末次1,052,870字符超过1,048,576限制，不能当作完整预算科学失败。随后修复重复返回子答案的上下文副本，22项相关测试通过；失败请求体离线重放降至1,003,277字符，未重启旧扫描。局部扫描命中／不命中的0.25分歧不能独立代表解释价值。下一步核对新试跑的真实候选、父层采用及新预测收益。

**此前冻结续跑：** [冻结续跑](../artifacts/discoveryworld_multi_object_continuation_20261009/RUN_STATUS.md)两臂各追加128次程序调用（累计192），单目标/跨对象新增113/111动作、15/17个预测实测全部否定，仍13/29与15/29、无目标激活。C5=1686已否定，原32窗口累计排除仍0/16、原8系数耗尽仍0/4，追加研究未扩大上一轮的固定解释区分收益。代码未改、0新增实验LLM。历史复核确认旧持续inquire及父程序复议已接通，下一步通过既有自主续研入口把负结果交回Luna，由它决定并执行假设或方法修订；本轮未启动LLM续研。

**此前跨对象接入：** [跨对象共同研究](../artifacts/discoveryworld_multi_object_response_20261009/RUN_STATUS.md)接通既有布尔选择、按对象反馈及完成状态更新；54项相关测试通过（10新增）。同529前缀、同64调用，单目标/跨对象新增55/59动作，对同一原32窗口反驳0/16、耗尽原8固定系数0/4，均未完成。新臂四次对象5预测均否定，随后联合修订；15/29较13/29仅多安装分。另一408前缀两臂同10动作完成，全部0新增LLM。后续冻结验证见当前进度。

**此前联合参数修订：** [联合系数与响应区间](../artifacts/discoveryworld_joint_response_20261009/RUN_STATUS.md)已接通旧Program、布尔选择和物理反例回流；44项不同测试通过（7新增），0新增LLM。seed1两臂同10动作成功；seed0旧臂0动作停研，新臂76动作、11预测被否定后主动中止，未完成预定512调用，尚无新增任务收益。新可行模型的参考矩阵秩3、参数4或6，存在保持参考预测不变而改变目标预测的方向。下一步把已有完整测量、尚无响应的对象5纳入共同实验选择，复用现有逐对象比较/安装/调频，并按实际测试对象回传预测；对象5能增加设计秩。

**此前参考回访：** [参考回访与候选接入](../artifacts/discoveryworld_reference_discrimination_20261009_v3/RUN_STATUS.md)已结束。复用旧拟合/布尔区分/参考恢复，修证据缓存、实际控制网格和恢复判定；37项不同测试通过。seed0参考5试验使相容系数21→8、窗口60→32，仍13/29；原配对新增74动作，修复后条件续跑再1动作，单列保留。seed1两臂同10动作完成。全部0新增LLM，前两次中止失败保留。最终8组解释均将目标支持区间预测到控制范围外；下一步联合修订系数与响应区间，不再追加父子调度或重复扫描。历史0.71/0.124有可复用数值部件，但没有这个二元反馈联合求解接口；新研究自主采用及任务收益仍待验证。

**此前解释提案：** [解释提案与父研究交接](../artifacts/discoveryworld_explanation_proposal_20261009/RUN_STATUS.md)已结束。旧inquire/同Luna账户/contrast已接入propose_explanations，提案保留旧题和程序，返回草稿及观测建议；75项不同测试通过、6项新增。同358动作/8调用现场，2次提案调用仅给存在性备选、无实验；父层收到后自主委派并继续扫描。合计24新调用/96动作，仍13/29，父稿未改；2次分析错误自行修复，最后12观测未分析。宿主启动阶段，无运行中科学提示，无同期控制；未见机制或任务收益。下一步查已有数值候选/基函数搜索的实际输入与采用条件，不再将提案入口或父子调度当作缺失。

**此前解释指引：** [解释生成指引与同Luna对照](../artifacts/discoveryworld_explanation_adoption_20261009/RUN_STATUS.md)已结束。复用原研究/比较/持续inquire，只补Hegel解释与跨实验预测要求；实际请求确认收到。同358动作/8调用现场，两臂各56新请求至64累计上限，Hegel/Code为333/218新动作、18.97M/12.61M reported tokens，均13/29、父稿未改。Hegel除扫频还实测否定一次光谱信息性预期并开始C5准备；Code也独立细化和偏移扫描，无新机制或任务收益。3次分析文件错误自修，末次反馈仍未闭合。Code在观察到Hegel继续扫描后追加，属于开发定位比较。下一步是将候选解释形成作为明确子研究交付；旧CodexProposer→controller已有提案/检验分工，但输入为绑定数值数据/AST且账户独立，迁入世界证据/设计与同Luna共享账户仍须适配，不能称为漏接一个已有调用。

**此前父层选择：** [负实验返回父研究选择](../artifacts/discoveryworld_parent_direction_20261008/RUN_STATUS.md)配对已结束。复用旧父答案回传/程序复议，已实测并分析的负比较返回父选择，首次设计入口保留；62项不同用例通过、8项新增。同358动作/8调用现场，旧调度新增14调用110动作后主动finish，新调度24调用121动作耗尽原32总额度；两臂仍13/29、父程序均未改。旧臂改同一contrast，新臂另开query/evaluate子题，科学上仍逐段扫频；各1/2次分析错误自行修复，无运行中科学提示。该单现场未见任务收益。现有research/research_update拟合、contrast多解释/多实验接口已开放，但作者仍只给单一扫频成功/失败命题；下一步检验这些旧接口的父层有效采用，不再扩调度，也不把未提出备选归为选择器缺失。

**此前未测子题：** [未测子题接回持续研究](../artifacts/discoveryworld_pending_inquiry_20261008/RUN_STATUS.md)已结束。复用旧pending→inquire，补未测无binding子题调度；修复新实验/旧自身分析错配，以及嵌套、镜像和完整日志内联。145项相关用例通过，33项新增。普通入口1新调用即结束、0新动作；原样重放上轮模型已声明子题后，条件续跑29新请求（25有效/4超限）、1232新动作，31/32总调用时主动finish，仍13/29，父程序未改。1次自动子研究后，已有18次有限实验执行及自主分析/操作纠错，但只在同一单实验扫频内修订，没有新规律或任务收益。下一步复用父答案回传/程序复议，检验反例之后先回到父研究选择，能否减少持续同一子contrast的调度偏向；已有多实验/表示修订工具并非缺失。原负结果、原程序和全部成本保留。

**此前中性子研究：** [中性子研究续跑](../artifacts/discoveryworld_autonomous_redesign_20261008/RUN_STATUS.md)已结束。复用旧inquire，从seed0的348动作现场给Luna自主选择下一研究；先修复累计物理证据/大提案绕过历史上下文压缩的问题，50项相关测试通过。冻结后28次有效决策、新增52动作，模型在总账29/32次时自行finish；C5安装后15/29，仍未激活、整题未完成。没有新拟合、程序改写或下级研究；一次操作失败后能自行继续。现有高级工具实际已提供但未采用，下一步复用旧`parent_program_path → procedure_action / last_program_result`，把保留程序绑定到局部题；本轮缺这个具体绑定，不再重写已有求解器。提示负担、调度和模型能力的贡献尚未区分。

**此前固定对照：** [六次运行](../artifacts/discoveryworld_response_ablation_20261008/RUN_STATUS.md)全部结束，0新增LLM、无运行中源码修改。seed1从零两臂均76动作成功，未调用响应研究；同408步现场仅有响应研究臂新增10动作完成，无响应臂停在22/29。seed0从零两臂均13/29未完成，有响应研究348动作、无响应226动作，增122动作/10333计算无收益。seed0在15轮后仍有21组相容系数/60窗口假设，停止因当前可达选点耗尽，非预算不足。直接缺口是参考单点阳性与目标阴性无法排除窄窗口下的系数解释；现策略只对目标选点，不能直接断言缺新公式。

**归因边界：** [响应机制接入](../artifacts/discoveryworld_response_uncertainty_20261008/RUN_STATUS.md)由助手查看开发失败后设计，直接复用旧拟合器与布尔选择器；0.81的半径区间只提供思路。10个基函数来自此前Luna、14个来自旧有限语法。后续冻结运行自动拟合、选实验、执行，不代表研发全过程没有外部指导。本次停滞现场真实激活在7014，计划7012仍未验证；采后自动分析已保留实际响应。旧四起点72/30/78/84的既有回归保留，本轮未重复运行。

**此前近似拟合与语法接入：** 6个旧近似候选及12个新语法候选实测全部失败，新增67/107动作、0LLM，Challenge停在22/29；保留原负结果。C4两个公开成功设置2944/2945使同一公式的C5预测相差10.17Hz，定位到点标签压缩与外推。[此前结果](../artifacts/discoveryworld_predictive_trial_20261008/RUN_STATUS.md)。

**此前证据回流：** 将途中成功设置接入恢复、后继公式拟合及已知负响应筛选；C4证据进入C5的10份拟合并产生一次新物理反例，新增49动作/32调用尝试，C5未完成。[保留结果](../artifacts/discoveryworld_observed_settings_20261008/RUN_STATUS.md)。

**上一开发回合：** 将模型改稿入口收束为测量变量/基函数声明，复用旧系数拟合、安装/调频与物理分析；声明进入原源码保留流程，9项新增/36项相关回归通过。真实Luna32调用已提出10候选并实测8个反例，另2个越界；父源码读取由上一开发运行的22次变成0次，仍有1次Python计算错误但没有中断研究。C4从中间调频记录的真实激活恢复至2945Hz，C5未完成，没有支持新规律或发布。含声明改稿的旧四起点零LLM全过，动作数72/30/78/84不变。[本轮记录](../artifacts/discoveryworld_estimator_research_20261008/RUN_STATUS.md)。下一核心转为候选选择、实验区分力，以及将途中获得的新证据纳入后续假设约束；两次调用数不同，不能据此主张性能优势。

**评测主线调整（2026-10-07）：旧online留作回归，新选高难公开任务。** 已核定DiscoveryWorld全8主题Normal/Challenge，16开发、64评测；另选DiscoveryBench原始数据57query/27假设组。放宽调用和时间限制，先在固定开发集确认难度和资源，再做同模型强普通代码档案/Hegel配对。[选择依据与协议](hegel_machine_hard_benchmark_selection_20261007.md)。公开环境适配已完成；复用Codex调用/Python执行/select_actions接口，将实时子答案交给父程序，5项针对性测试通过。Reactor首批因共有上下文错误已停止，Normal两臂未完成、Challenge未启动，保留为[接入诊断](../artifacts/discoveryworld_pilot_20261007_v2/RUN_STATUS.md)。现已修复证据文件/记录路径与跨轮笔记，并将旧`method_decision`的当前支持摘要接入DiscoveryWorld；相关回归通过；[24轮采用检查](../artifacts/discoveryworld_context_repair_20261007/RUN_STATUS.md)显示可选笔记仍0次保存，故两臂统一要求每轮返回模型笔记。[同346条公开前缀续接](../artifacts/discoveryworld_persistent_notes_20261007/RUN_STATUS.md)各12轮，24/24保存笔记、新增11/7环境动作，但子研究仍0；所有检查均已结束。新增采集索引与分析引用分开，未分析的新观测不会自动变成旧答案支持。这不是完整性能比较；全部16开发题与64正式题尚未完成。沿用现成代码执行、方法和历史检索，不把旧时序/匿名对象实验重新称作首次实现。

这里完成的是统一定位和已有入口聚合，并非181套实验重跑或所有能力融为一个全能Agent。开发结果、冻结实验、LLM实际采用和人工接入分别标注；没有证据将接续问题归因于模型路由。

窄采用与受反例驱动的修订已有结果。编辑后的首次实践由主机启动，两份12目标结果使用同一组开发任务；第三个作者曾通过新案例，却继续编辑退化并在token上限时未发布。

方法保留已有历史实现：[0.77](hegel_machine_preparation_revision_20260913.md)保留父方法并单独记录失败子版本，[0.92](hegel_machine_instrument_method_revision_20260919.md)把同一机制用于仪器区分步骤。Python现已把`practice_trials`中的受支持源码接为提交时的实际比较对象。仅提供`method_candidates.json`与`restore_method`入口时的[0/3失败记录](../artifacts/python_method_recall_compact_luna_20261007/RUN_STATUS.md)保留；此前1/3成功由主机策略完成。现有直接证据摘要后的固定实验取得10/10自主取回、8/10发布；这些是同一失败状态下的重复作者，不是十种独立任务。两次仍未完成提交。后续检验不同失败状态与研究收尾，长期累积收益仍开放；先使用已完成证据收束论文。
