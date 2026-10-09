# 历史能力复用索引（2026-10-06）

当前问题从[能力总入口](hegel_machine_reuse_hub_20261006.md)进入。可用`cognition historical --query 'refit' --compact`直接定位旧实现及最新接入，不必先知道版本号；本页保留完整181项历史记录。

**已有功能从当前实现接入，已有失败作为选路依据。** 本索引已定位 **181 个次版本编号（0.1–0.181）**。0.1 的初始提交为 `17f14126`，其中 `pyproject.toml` 明确标为0.1.0。编号含补丁、研究候选和撤回实验；0.1–0.2 属于旧正式验证轨道，认知主线从 0.3 开始。当前包仍标为 0.181.0，并含后续未发布修改。

完整[可执行目录](../src/hegel_machine/cognition/historical_capabilities.json)逐项给出现存模块、调用符号和原生运行入口：162项能力留存、9项撤回、8项被后续修复替代、2项旧正式协议。每个编号均有当前入口；撤回条目指向当前采用的替代实现。这些分类说明代码去向，不是实验通过率。

```bash
PYTHONPATH=src python -m hegel_machine cognition historical --version 0.93
PYTHONPATH=src python -m hegel_machine cognition historical --version 0.93 --execute unified --request config/cognition_observation_action_session.json --checkpoint /tmp/hegel-observation-session.json --steps 1
```

LLM工具 `legacy_capabilities` 可按版本或能力组查询。`legacy_research` 直接委托现存 `UnifiedResearchAgent` 研究当前公有实验室，共用已付数据和剩余预算，并保存完整 `shared/legacy_knowledge.json`；可导出的预测器交回LLM选择和确认。`research_choice` 则复用旧动作策略及实验规划器，输出可直接采购和评估的下一步。对象、SC-KG、延迟实验保留目录列出的原生环境入口。

当前轨迹实验室也已接入原跨任务会话。用 [LegacyInquirySession](../src/hegel_machine/cognition/legacy_session_bridge.py) 注册已有 Agent、公开实验室、已付研究报告和当时可用概念，再传入完整知识库；未决任务可从已有观测初始化。绑定后的 Agent 可调用 `legacy_revisit`，由原调度器选择回访；`legacy_return_confirmation` 购买新轨迹及必要读数，复用旧来源和生产方法回报链。`finish` 的实际新证据也会回传。各题保留自身资源上限和最终确认额度，旧数据直接续用；跨任务方法更新使用同一知识库。来源贡献确认与免仪器预测的完成状态分别保留。[运行与结果](../artifacts/legacy_session_bridge_20261006/RUN_STATUS.md)。

`callable_transfer` 后续修订也已保留实际调用的模板、角色绑定及已付适配费用，研究中途找到的来源可进入原回传链。同12装置复测中，12次旧概念调用均由新观测确认来源收益并回传生产方法；原预测、观测及测量不变。[修复复测](../artifacts/callable_source_returns_20261006/RUN_STATUS.md)。

真实Luna会话已完成8次尝试：旧仪器、响应和历史长度算子被实际调用，尚未观察到自主回访或可归因的旧概念复用。三组无服务中断的配对中，保留知识2/3完成、逐题研究3/3完成；已有有效候选却未及时提交是下一决策缺口。见[实测记录](../artifacts/legacy_session_luna_20261006/RUN_STATUS.md)。

旧Code Agent的`retain_evaluated_python_model`及既有`ResearchActionPolicy`现已接入当前LLM入口。12组离线变化／返回序列验证旧数值解释直接复用；新增两组真实Luna连续试跑，两次返回均自主提交未修改的原解释，其中一次实际采用`research_choice`所选候选。原训练数据和依赖随模型归档，分支传递实际确认的知识；两臂均4/4目标正确，含来源调用30／42次。见[候选保留与返回](hegel_machine_candidate_retention_20261006.md)。

## 参数失效与结构修订：已有工作及当前衔接

本次复核确认：**核心算法在181项历史工作中已经实现，当前问题是普通Python研究程序的调用与选路。** 0.38的[证据范围修订](hegel_machine_revisable_research_scope_20260910.md)已在再次失配后重划后段证据并重新研究；0.79冻结源码先尝试系数重拟合、购买新验证，再尝试局部时间结构，均失败才采购仪器。0.84进一步区分输入与关系失配。旧实现分别面向SC-KG及受限程序语言；当前接口则是任意Python档案与会话内新旧记录。181是版本目录项数量，不是181个独立Git快照。

这条主线已有实现和实验。0.79的[控制机制／历史修订](hegel_machine_dynamic_research_replication_20260913.md)用`RetainedControlRevisions`区分原结构重新估参与历史扩展，六装置30/30通过；0.84的[输入／关系诊断](hegel_machine_instrument_relation_diagnosis_20260914.md)已有36/36结果；0.117的[响应基底重开](hegel_machine_response_basis_reopening_20260926.md)由反例触发结构候选，终测51/72→62/72。研究步骤改写也已有0.77的[准备步骤修订](hegel_machine_preparation_revision_20260913.md)和0.137的[条件响应研究](hegel_machine_condition_response_actions_20260930.md)。这些能力不能重新计作首次实现。

0.181之后还已有Luna编写研究方法、冻结执行的试验：`temporal_counterexample_learning`为10/18→18/18；`temporal_failure_selectivity`未发布的方法及20/32负结果保留；后续`temporal_evidence_scope_learning`达到32/32，`temporal_selective_preservation`两臂56/56。其运行目录均在`artifacts/`下的同名`_luna_20261006`路径。

此次窄衔接是普通Python归档的`refit`入口：沿用原程序自己的`fit()`、既有整轨迹划分和Python执行器，比较“旧数值解释失败”之后同一拟合程序能否恢复验证预测；原归档不变。后续表示研究仍调用已有`history`、`instrument`和`response`，没有新增一套参数或结构搜索器。旧版本的离线分流、方法源码学习与这次实时LLM动作选择分别记载。

接入后的固定流程4/4通过；两轮实时Luna均参数2/2、结构0/2。第二轮公开了原先遗漏的验收阈值，完整重放四题，原失败保留；见[前轮结果](../artifacts/python_refit_criteria_luna_20261006/RUN_STATUS.md)。

后续已补Python `research_choice`失败分支：原拟合程序验证失败后，给出已有仪器测量、实测评估和响应修订的可执行建议；独立轨迹验证通过的导出模型自动进入原候选比较。使用现有`ResearchActionPolicy`及算子，不新增数值拟合器。相同四题离线仅执行建议即可4/4通过；重复控制序列保持在响应训练／验证同一侧，59项相关测试通过。[离线结果](../artifacts/python_research_routing_distinct_offline_20261006/summary.json)。

前轮Luna重放4/4通过新确认及终测，结构变化两例实际调用旧仪器／响应研究；新选择器和`refit`仍未被调用。[实测及范围](../artifacts/python_research_routing_final_luna_20261006/RUN_STATUS.md)。

后续已把同一策略的具体`next_action`直接接入评价、重拟合、仪器研究及响应修订反馈，不重复评分或购买数据。新确认失败使用本次实测反例，保留旧评分的过期状态；完整反馈链无需显式调用选择器即可执行至独立确认。68项相关测试通过。相同四题展示／隐藏建议的Luna对照为3/4、2/4，均无错误确认；实际出现原源码重拟合，以及仪器区分子研究推动父响应修订。少14.15% token、多50%测量，未决子问题仍可能被后续动作放弃。[结果与范围](../artifacts/python_research_advice_ablation_luna_20261006/RUN_STATUS.md)。

现已从已有仪器／拟合记录恢复未决步骤，在普通Python、写文件及历史拟合后继续呈现；正确替代候选仍可直接申请确认。四新装置来源3/4成功，持续／事件提示均4/6已启动目标通过，服务正常配对仍持平且持续组成本更高。实测后补上空仪器模型的既有补测路径；累计78项相关测试通过。[冻结实测与后续补丁范围](../artifacts/python_pending_advice_luna_20261006/RUN_STATUS.md)。

完整流程现直接复用 [ProcedureConsumerAgent／FrozenMethodAgent](../src/hegel_machine/cognition/temporal_paired_method.py) 的执行循环，允许实时模型显式选择整段程序，并共用原资源账和新确认。离线六题全通过，63项相关测试通过；实时可用／仅逐步组3/6、5/6，服务正常配对2/4、4/4，完整程序及单次程序调用均为零。尚待把完整程序接入当前失败反馈的可选动作。[实际采用与结果](../artifacts/python_procedure_luna_20261006/RUN_STATUS.md)。

随后当前反馈的24次程序选项仍未被采用，两臂均5/6，服务正常配对均4/4且展示组更贵；提示已移出默认生产。现修复已验证候选在最后一次模型调用的直接确认，60项相关测试通过。[本轮结果](../artifacts/python_procedure_offer_luna_20261006/RUN_STATUS.md)。下一步复用 [PairedMethodLearner](../src/hegel_machine/cognition/temporal_paired_practice.py) 和[既有方法实验入口](../scripts/run_temporal_paired_method.py)，补当前Python来源与实践前缀的衔接；旧完整研究包预算机制已经实现，无需再造调度器。

## 按能力直接复用

证据范围修订继续使用上述旧算子；新增同检查点 finite2→recurrent 对照。实际脚本诊断中，同一`refit(records=[4,5])`能修复响应变化而不能修复结构变化，复用了历史范围修订机制。LLM最终稿虽实践5/5确认，仍未学得该选路，未发布；旧程序扩展目标10/10。[本轮结果](../artifacts/python_scoped_method_comparison_luna_20261006/RUN_STATUS.md)。

现已在[可执行方法](../src/hegel_machine/cognition/python_current_evidence_method.py)中局部接入历史规则：新反例确定当前记录，沿用`refit(records=..., validation_records=...)`，以独立验证决定是否转入既有仪器研究。源实践5/5、目标10/10；直接修改前的同一候选也为10/10，观测1722→1628、测量1278→852、计算50388→29020。原候选对照在桥接完成后追加，属于开发诊断；此前LLM未发布结果保留。下一步解决已有方法的可靠采用，数值拟合器无需重写。[结果](../artifacts/python_current_evidence_bridge_20261006/RUN_STATUS.md)。

后续真实采用实验已完成：三次Luna作者均自行复制库中历史修复程序，15/15新实践通过，6调用；原同源码10目标结果复用。普通`run`改变程序后现同步下一轮作者上下文。这是供应库中直接采用阶段的结果；遇到新反例后的修订见下段。[采用结果](../artifacts/python_method_adoption_luna_20261006/RUN_STATUS.md)。

响应验证之后再失配的条件已接入：直接修订作者0/3发布；主机在编辑后调用原付费实践并返回实测反馈，模型自行修订后2/3发布，两份程序均6/6实践、各12/12冻结目标，原程序10/12。首份成功源码原样保存为[python_response_reopening_method](../src/hegel_machine/cognition/python_response_reopening_method.py)，通过原`PreparedPythonMethodAgent.bind_method(source)`及`execute_method`调用；0.117的反例重开历史已关联回同一路线，旧人工桥接文件未替换。[直接修订](../artifacts/python_method_reopening_luna_20261006/RUN_STATUS.md)、[编辑实践](../artifacts/python_method_edit_practice_luna_20261006/RUN_STATUS.md)。

选择性恢复继续复用旧 [counterexample供应流程](../scripts/run_temporal_counterexample_learning.py)、[calibration_change实验室](../src/hegel_machine/cognition/temporal_inquiry.py)及`response(records=...)`：原三个条件加公开96行变化条件，学得程序实践4/4、跨来源8/8；保住无测量refit，验证后遭新反例也能修复。[实测](../artifacts/python_selective_method_completion_luna_20261006/RUN_STATUS.md)。本轮另修复`refit(records=...)`和拟合证据边界，已过真实数值测试；学得程序尚未采用这个新入口。

后续显式 `finish` 已复用原 `_practice` 与发布门槛，依次完成当前源码缺少的付费实践，首败返回实际轨迹。Luna自行改写并提交后实践3/3、冻结迁移6/6；固定程序同为6/6，新源码丢失便宜refit分支，投入更高。[发布闭环结果](../artifacts/python_method_publication_luna_20261006/RUN_STATUS.md)。下一步复用已有选择性修订与机制变化环境，保住旧候选的有效分支。

后续直接复用旧“冻结最后候选程序”的实验方式：修复作者反馈与执行状态的结构错位、补齐评分/确认契约，未人工修补学得源码。三类源实践及六个跨来源目标通过；真正执行了原 `refit`、`instrument`、`assess_instrument`、`response` 与原新确认路径。固定对照同为6/6，学得程序成本更高；作者发布仍未完成。[冻结候选结果](../artifacts/python_method_candidate_replay_20261006/RUN_STATUS.md)。

当前 Python 假设档案已通过 [PythonArchiveMethodLearner](../src/hegel_machine/cognition/python_method_learning.py) 接回上述旧作者与执行器：共享原实践记账、发布覆盖、失败恢复和冻结执行，仅适配档案、付费前缀及依赖比较。真实作者曾用同一程序通过复用与响应修订两例，仪器变化例未过；最终改写仅验证1/3，未发布，固定对照为6/6。[本轮实测](../artifacts/python_method_learning_resume_luna_20261006/RUN_STATUS.md)。

| 能力 | 现成实现与接入位置 | 历史依据与适用范围 |
|---|---|---|
| 表示与组合 | [language](../src/hegel_machine/cognition/language.py)、[continuous_state](../src/hegel_machine/cognition/continuous_state.py)、[object_induction](../src/hegel_machine/cognition/object_induction.py)、[temporal_coordinates](../src/hegel_machine/cognition/temporal_coordinates.py)、[composed_conditions](../src/hegel_machine/cognition/composed_conditions.py) | 0.10 callable、0.11 状态与测量、0.61–68 对象/关系/等价、0.105–124 局部组合、0.129–137 尺度和条件。直接消费可执行程序、输入角色与候选；对象、状态及算子族有限。 |
| 假设与递归子问题 | [controller](../src/hegel_machine/cognition/controller.py)、[research_graph](../src/hegel_machine/cognition/research_graph.py)、[evolution_research](../src/hegel_machine/cognition/evolution_research.py)、[instrument_hypotheses](../src/hegel_machine/cognition/instrument_hypotheses.py)、[meta_prior_research](../src/hegel_machine/cognition/meta_prior_research.py) | 0.3–4 问题与组合、0.12 可变依赖、0.24 多后继共享子研究、0.30–39 元先验/共同失配、0.91 竞争仪器。父任务必须实际消费子答案；SC-KG 的优化结论依原环境合同。 |
| 主动实验与测量 | [program_experiment_methods](../src/hegel_machine/cognition/program_experiment_methods.py)、[instrument_condition_probes](../src/hegel_machine/cognition/instrument_condition_probes.py)、[conditional_experiments](../src/hegel_machine/cognition/conditional_experiments.py)、[observation_research](../src/hegel_machine/cognition/observation_research.py)、[experiment_simulator](../src/hegel_machine/cognition/experiment_simulator.py) | 0.15/22/26 实验续研与模拟器、0.69–71 联合条件求解、0.90–93 长历史区分、0.135–141 同背景和条件实验。已有求解器使用当前候选与公开合法控制，实际读数由原采购接口取得。 |
| 研究程序的保存、复用与改写 | [research_methods](../src/hegel_machine/cognition/research_methods.py)、[research_programs](../src/hegel_machine/cognition/research_programs.py)、[temporal_experiment_methods](../src/hegel_machine/cognition/temporal_experiment_methods.py)、[instrument_experiment_methods](../src/hegel_machine/cognition/instrument_experiment_methods.py)、[object_dependency_methods](../src/hegel_machine/cognition/object_dependency_methods.py) | 0.12–19 方法与适用条件、0.70/76/77 步骤重绑及准备修订、0.92 观察方法闭环、0.102 归属方法改写、0.140–153 草稿与实测改写。本轮已将0.92仪器实验方法库与实测回传接入LLM共同算子；其他方法按原入口复用。固定求解器和LLM改写程序分别计功。 |
| 修订、保留与撤回 | [memory](../src/hegel_machine/cognition/memory.py)、[knowledge_evolution](../src/hegel_machine/cognition/knowledge_evolution.py)、[condition_learning](../src/hegel_machine/cognition/condition_learning.py)、[transfer_condition_research](../src/hegel_machine/cognition/transfer_condition_research.py)、[noisy_response_experiments](../src/hegel_machine/cognition/noisy_response_experiments.py) | 0.12/18 合并拆分与来源反例、0.80–86 局部关系修订、0.99 旧完成挑战、0.121–131 删项/噪声/尺度、0.155–158 噪声与撤回修复、0.169 失败绑定撤回。调用当前修正版，保留未被反驳的旧条件。 |
| 研究调度、继续与停研 | [session](../src/hegel_machine/cognition/session.py)、[unified_session](../src/hegel_machine/cognition/unified_session.py)、[research_opportunities](../src/hegel_machine/cognition/research_opportunities.py)、[research_portfolio](../src/hegel_machine/cognition/research_portfolio.py)、[condition_research_returns](../src/hegel_machine/cognition/condition_research_returns.py) | 0.19–25 机会与共享前置、0.93–98 共同动作与回访、0.125/133 路线与续研、0.157–178 完整投入及有效确认、0.181 反馈重排。优先复用一个已有会话和资源账，不再造并行调度器。 |
| 知识贡献归因与后继收益 | [knowledge_returns](../src/hegel_machine/cognition/knowledge_returns.py)、[memory](../src/hegel_machine/cognition/memory.py)、[unified_revisits](../src/hegel_machine/cognition/unified_revisits.py)、[method_revision](../src/hegel_machine/cognition/method_revision.py) | 0.16–18 产物回报、0.116/120 多代来源与增量回报、0.157/177 方法归因、0.180–181 普通来源回传。后续同源不同程序、联合来源、近零收益及撤回修复见[0.181后审计](hegel_machine_post_0181_audit_20261004.md)与[窄闭环修复](hegel_machine_narrow_cycle_repair_20261004.md)。 |
| 对照、外部消费与持久化 | [experiments](../src/hegel_machine/cognition/experiments.py)、[code_agent](../src/hegel_machine/cognition/code_agent.py)、[code_session](../src/hegel_machine/cognition/code_session.py)、[sc_kg_environment](../src/hegel_machine/cognition/sc_kg_environment.py)、[workspace](../src/hegel_machine/cognition/workspace.py)、[delayed_feedback](../src/hegel_machine/cognition/delayed_feedback.py) | 0.4 消融、0.12 延迟与Code、0.26 跨任务Code、0.30–58 SC-KG、0.62/78/79/84 真实普通Agent对照。任务、工具、预算和模型按现研究协议匹配，沿用已有运行/恢复入口。 |

当前主调用链是 [UnifiedResearchAgent](../src/hegel_machine/cognition/unified_research.py) 和 `KnowledgeLibrary`；早期 [CognitiveAgent](../src/hegel_machine/cognition/controller.py) 与 SC-KG 保留各自任务接口。近期 [Growth桥接](hegel_machine_workflow_bridge_20261006.md) 已共用旧学习器的采样、拟合、确认和来源反馈。本轮 [temporal_operator_agent](../src/hegel_machine/cognition/temporal_operator_agent.py) 的 `disagreement` 已复用0.92的 [instrument_experiment_methods](../src/hegel_machine/cognition/instrument_experiment_methods.py) 与 `ProgramExperimentMethods`，默认读写 `shared/instrument_methods.json`；实购对应轨迹后，`assess_instrument(experiment=proposal_path, records=[new_record])` 将证据回传同一方法库。三装置的4→5→5步改写/复用和三臂共同接口共7项测试通过。实际LLM调用与成本结果见下方历史方法对照。

父响应接入复用0.69–0.77以来的同一竞争闭环：[temporal_response_experiments](../src/hegel_machine/cognition/temporal_response_experiments.py) 把新接口的观测转换为旧比较器的配对记录；规划、步骤重绑、方法更新仍调用 [competing_object_experiments](../src/hegel_machine/cognition/competing_object_experiments.py) 与原方法库。`response` 返回 `hypotheses_path`，也可直接使用旧计划的 `hypotheses` JSON列表：

```text
run {operator:'response_disagreement', hypotheses:'shared/candidates/hypotheses.json'}
query {actions:[proposal.action], after_observation:[
  {operator:'assess_response', experiment:proposal.experiment, records:'acquired'}]}
```

测量坐标中的旧响应假设可附 `instrument:'instrument.json'` 提供当前校准；实际采购仍取得真实字段。原窄窗口/事件任务验证了新校准及阈值下的步骤复用；规划后仪器失配会限制旧方法条件，同时保留测量域仍正确的响应规律。[4项接入测试](../tests/test_temporal_response_experiments.py)与50项相关回归通过。当前多项式响应可直接竞争，但不据此声称其已学出新的边界研究方法。

实测LLM消费现已补上：[同模型历史方法对照](../artifacts/temporal_historical_methods_luna_20261006/RUN_STATUS.md)
直接加载0.76保存的实测方法库。保留/删除两臂各2/2通过；事件任务实际调用旧步骤、按新仪器与阈值重绑，4行实测区分5个候选，并回写跨任务支持。
首轮总体观测336→252，但计算23540→25946、token173825→283230。随后接通已有0.93动作选择，四次LLM均主动使用它：两臂各2/2正确、观测与测量一致，保留少15.2%计算而多15.0%token；[后续实测](../artifacts/temporal_historical_actions_luna_20261006/RUN_STATUS.md)。

## 版本工作索引

每行指向当时工作报告；“已有实现/实验记录”表示定位到了工作，采用与效果以报告中的实际结论为准。补丁归入同一次版本。0.131–0.154 的整链退化已有[专项复核](hegel_machine_version_and_goal_reassessment_20261001.md)，应使用现有定点修复与旧能力参照。

| 编号 | 已做工作 | 记录 | 采用说明 |
|---|---|---|---|
| 0.1 | 独立v3的有限结构律验证内核 | [归档](history/formal_track_readme_20260907.md)，`17f14126` | 旧正式轨道 |
| 0.2 | Phase 1–2 验证器与有限结构选择 | [报告](history/formal_track_readme_20260907.md) | 旧正式轨道 |
| 0.3 | 认知循环、程序假设、实验和知识库骨架 | [报告](cognitive_runtime_first_run.md) | 已有实现/实验记录 |
| 0.4 | 依赖计划、抽象、角色映射、连续实验与消融 | [报告](cognitive_runtime_v2_results.md) | 已有实现/实验记录 |
| 0.5 | 实验、修订选择与迁移研究改进 | [报告](cognitive_runtime_v3_results.md) | 已有实现/实验记录 |
| 0.6 | 联合修订、持续观测与风险学习 | [报告](cognitive_runtime_v4_results.md) | 已有实现/实验记录 |
| 0.7 | 状态转换、迁移实验与尾部学习 | [报告](cognitive_runtime_v5_results.md) | 已有实现/实验记录 |
| 0.8 | 联合状态坐标与持续研究会话 | [报告](cognitive_runtime_v6_results.md) | 已有实现/实验记录 |
| 0.9 | 研究恢复、行动经验与条件知识 | [报告](hegel_machine_functional_implementation_20260908.md) | 已有实现/实验记录 |
| 0.10 | 可调用概念、组合与共享执行 | [报告](hegel_machine_callable_concepts_20260908.md) | 已有实现/实验记录 |
| 0.11 | 测量选择、连续状态与独立进程装置 | [报告](hegel_machine_measurement_and_state_20260908.md) | 已有实现/实验记录 |
| 0.12 | 实验迁移、方法修订、递归图、合并拆分与延迟反馈 | [报告](hegel_machine_integrated_research_20260908.md) | 已有实现/实验记录 |
| 0.13 | 方法适用条件与实际续研收益 | [报告](hegel_machine_method_returns_20260908.md) | 已有实现/实验记录 |
| 0.14 | 保留成熟机制、局部修订与校准选择 | [报告](hegel_machine_local_revision_and_calibration_20260908.md) | 已有实现/实验记录 |
| 0.15 | 边界实验与实验方法迁移 | [报告](hegel_machine_boundary_and_method_transfer_20260908.md) | 已有实现/实验记录 |
| 0.16 | 知识价值、等价化简与排序 | [报告](hegel_machine_knowledge_value_20260909.md) | 已有实现/实验记录 |
| 0.17 | 知识生产方法回报与边界覆盖 | [报告](hegel_machine_knowledge_production_and_boundaries_20260909.md) | 已有实现/实验记录 |
| 0.18 | 具体知识反例修订与研究选择 | [报告](hegel_machine_transfer_repair_20260909.md) | 已有实现/实验记录 |
| 0.19 | 联合历史研究、实验续研与方法变化 | [报告](hegel_machine_joint_research_capabilities_20260909.md) | 已有实现/实验记录 |
| 0.20 | 持续仪器重选、概念演化与统一投入 | [报告](hegel_machine_continuing_research_20260909.md) | 已有实现/实验记录 |
| 0.21 | 风险决策续研与停止选择 | [报告](hegel_machine_decision_research_20260909.md) | 已有实现/实验记录 |
| 0.22 | 区分实验与冲突证据逐批续研 | [报告](hegel_machine_discriminating_research_20260909.md) | 已有实现/实验记录 |
| 0.23 | 完整访问、共享预算与动态跨能力议程 | [报告](hegel_machine_complete_research_agenda_20260909.md) | 已有实现/实验记录 |
| 0.24 | 多个后继共用前置子研究 | [报告](hegel_machine_shared_research_20260909.md) | 已有实现/实验记录 |
| 0.25 | 可修订的研究收益评价器 | [报告](hegel_machine_research_evaluator_20260909.md) | 已有实现/实验记录 |
| 0.26 | 实验模拟器修订、结构发现与跨任务Code | [报告](hegel_machine_completion_20260909.md) | 已有实现/实验记录 |
| 0.27 | 仪器与连续表示重选 | [报告](hegel_machine_measurement_revision_20260909.md) | 已有实现/实验记录 |
| 0.28 | 测量表示迁移减少后继研究 | [报告](hegel_machine_measurement_transfer_20260909.md) | 已有实现/实验记录 |
| 0.29 | 知识指导测量投入 | [报告](hegel_machine_measurement_acquisition_20260909.md) | 已有实现/实验记录 |
| 0.30 | 元先验自主研究接入SC-KG | [报告](hegel_machine_meta_prior_research_20260910.md) | 已有实现/实验记录 |
| 0.31 | 先验采用条件与真实优化 | [报告](hegel_machine_prior_adoption_20260910.md) | 已有实现/实验记录 |
| 0.32 | 两模型共同失配后重开假设研究 | [报告](hegel_machine_joint_model_revision_20260910.md) | 已有实现/实验记录 |
| 0.33 | 候选竞争、区分实验与剩余投入 | [报告](hegel_machine_revision_competition_20260910.md) | 已有实现/实验记录 |
| 0.34 | 强共同反例后的主动复测确认 | [报告](hegel_machine_active_confirmation_20260910.md) | 已有实现/实验记录 |
| 0.35 | 实验同时区分机制并服务优化目标 | [报告](hegel_machine_decision_experiments_20260910.md) | 已有实现/实验记录 |
| 0.36 | 观测修订中保留有效知识 | [报告](hegel_machine_observation_repair_20260910.md) | 已有实现/实验记录 |
| 0.37 | 独立目标研究与带噪推荐 | [报告](hegel_machine_objective_research_20260910.md) | 已有实现/实验记录 |
| 0.38 | 研究范围与目标预测器修订 | [报告](hegel_machine_revisable_research_scope_20260910.md) | 已有实现/实验记录 |
| 0.39 | 目标和约束共享取证 | [报告](hegel_machine_joint_evidence_20260910.md) | 已有实现/实验记录 |
| 0.40 | 末段观测价值与逐次重选 | [报告](hegel_machine_terminal_value_20260910.md) | 已有实现/实验记录 |
| 0.41 | 实验估值精度与证书复测 | [报告](hegel_machine_value_precision_20260910.md) | 已有实现/实验记录 |
| 0.42 | 有证据的边界探索与局部预测更新 | [报告](hegel_machine_supported_exploration_20260910.md) | 已有实现/实验记录 |
| 0.43 | 原生SC-KG与普通Code两类对照 | [报告](hegel_machine_vanilla_and_completion_20260910.md) | 已有实现/实验记录 |
| 0.44 | 按观测费用比较短研究序列 | [报告](hegel_machine_research_value_20260910.md) | 未采用 |
| 0.45 | 由重复观测修订噪声模型 | [报告](hegel_machine_noise_revision_20260910.md) | 未采用 |
| 0.46 | 噪声联合策略试验 | [报告](hegel_machine_active_noise_experiments_20260910.md) | 未默认采用 |
| 0.47 | 目标验证调度与证据共享修复 | [报告](hegel_machine_active_noise_experiments_20260910.md) | 已有实现/实验记录 |
| 0.48 | 冻结目标候选的持续验证 | [报告](hegel_machine_budgeted_validation_20260910.md) | 已有实现/实验记录 |
| 0.49 | 目标GP重建后的候选比较 | [报告](hegel_machine_current_incumbent_20260911.md) | 两版未采用 |
| 0.50 | 目标模型替换的独立研究 | [报告](hegel_machine_objective_replacement_20260911.md) | 已有实现/实验记录 |
| 0.51 | 保留约束解释、共同选择末段实验 | [报告](hegel_machine_constraint_retention_20260911.md) | 已有实现/实验记录 |
| 0.52 | 目标候选重入与表示修订 | [报告](hegel_machine_objective_reentry_20260911.md) | 已有实现/实验记录 |
| 0.53 | 完成目标研究并部署解释 | [报告](hegel_machine_goal_completion_20260911.md) | 采用完成版；早期重入/交互试验未采用 |
| 0.54 | 已部署目标解释重新竞争 | [报告](hegel_machine_goal_recompetition_20260911.md) | 已有实现/实验记录 |
| 0.55 | 共享候选验证、联合识别与末段观测 | [报告](hegel_machine_shared_tail_20260912.md) | 含.1/.2；末段专用策略未默认采用 |
| 0.56 | 复测与候选验证共同选择 | [报告](hegel_machine_dual_value_20260912.md) | 已有实现/实验记录 |
| 0.57 | 约束知识保留与程序项系数修订 | [报告](hegel_machine_constraint_knowledge_20260912.md) | 已有实现/实验记录 |
| 0.58 | 逐题研究→部署→修订回传→下一题 | [报告](architecture.md) | MetaPriorSession.run_deploy |
| 0.59 | 未知任务统一研究循环 | [报告](hegel_machine_unified_research_20260912.md) | 已有实现/实验记录 |
| 0.60 | 连续状态、风险修订与方法复用 | [报告](hegel_machine_state_and_risk_20260912.md) | 已有实现/实验记录 |
| 0.61 | 对象、关系、事件、尺度进入统一研究 | [报告](hegel_machine_unified_structure_20260913.md) | 已有实现/实验记录 |
| 0.62 | 跨结构组合与可回放研究程序 | [报告](hegel_machine_composed_research_20260913.md) | 已有实现/实验记录 |
| 0.63 | 仪器知识与校准修订 | [报告](hegel_machine_instrument_knowledge_20260913.md) | 已有实现/实验记录 |
| 0.64 | 输入依赖与机制联合修订 | [报告](hegel_machine_revisable_inputs_20260913.md) | 已有实现/实验记录 |
| 0.65 | 匿名对象发现与组合研究 | [报告](hegel_machine_object_induction_20260913.md) | 已有实现/实验记录 |
| 0.66 | 部分对象知识与角色修订 | [报告](hegel_machine_partial_objects_20260913.md) | 已有实现/实验记录 |
| 0.67 | 有歧义的对象解释共同执行 | [报告](hegel_machine_object_equivalence_20260913.md) | 已有实现/实验记录 |
| 0.68 | 竞争对象解释与实测修订 | [报告](hegel_machine_object_competition_20260913.md) | 已有实现/实验记录 |
| 0.69 | 可执行程序分歧生成定向实验 | [报告](hegel_machine_program_disagreement_20260913.md) | 已有实现/实验记录 |
| 0.70 | 联合控制求解与研究步骤迁移 | [报告](hegel_machine_joint_experiment_methods_20260913.md) | 已有实现/实验记录 |
| 0.71 | 组合条件实验与方法适用范围 | [报告](hegel_machine_compound_experiments_20260913.md) | 已有实现/实验记录 |
| 0.72 | 程序迁移条件研究 | [报告](hegel_machine_transfer_conditions_20260913.md) | 已有实现/实验记录 |
| 0.73 | 共享条件求解和实测计算 | [报告](hegel_machine_shared_conditions_20260913.md) | 已有实现/实验记录 |
| 0.74 | 多个调用版本共享拟合 | [报告](hegel_machine_shared_fitting_20260913.md) | 已有实现/实验记录 |
| 0.75 | 动态仪器学习、修订与免测量预测 | [报告](hegel_machine_dynamic_instruments_20260913.md) | 已有实现/实验记录 |
| 0.76 | 时序实验步骤跨任务积累 | [报告](hegel_machine_temporal_methods_20260913.md) | 已有实现/实验记录 |
| 0.77 | 实验准备历史局部改写 | [报告](hegel_machine_preparation_revision_20260913.md) | 已有实现/实验记录 |
| 0.78 | 动态仪器真实普通Code对照 | [报告](hegel_machine_dynamic_code_comparison_20260913.md) | 已有实现/实验记录 |
| 0.79 | 先修订控制解释，再决定仪器采购 | [报告](hegel_machine_dynamic_research_replication_20260913.md) | 已有实现/实验记录 |
| 0.80 | 组合内部关系修订与连续区间研究 | [报告](hegel_machine_object_relation_revision_20260913.md) | 含.2修复 |
| 0.81 | 关系不确定性进入后继概念调用 | [报告](hegel_machine_inherited_relation_intervals_20260914.md) | 已有实现/实验记录 |
| 0.82 | 残差驱动关系类型修订 | [报告](hegel_machine_relation_type_revision_20260914.md) | 已有实现/实验记录 |
| 0.83 | 关系条件合成与后继复用 | [报告](hegel_machine_relation_condition_synthesis_20260914.md) | 含.1实验背景修复 |
| 0.84 | 输入校准与关系失配的区分 | [报告](hegel_machine_instrument_relation_diagnosis_20260914.md) | 已有实现/实验记录 |
| 0.85 | 可撤销跨任务反例与参数域重开 | [报告](hegel_machine_relation_trajectory_evidence_20260914.md) | 已有实现/实验记录 |
| 0.86 | 紧凑关系修订和实验后缀缩减 | [报告](hegel_machine_relation_trajectory_evidence_20260914.md) | 已有实现/实验记录 |
| 0.87 | 统一研究动作选择与保留旧参数 | [报告](hegel_machine_unified_action_policy_20260919.md) | 已有实现/实验记录 |
| 0.88 | 真实确认收益与探索投入 | [报告](hegel_machine_action_confirmation_20260919.md) | 已有实现/实验记录 |
| 0.89 | 直接复用、残差研究、父贡献修订 | [报告](hegel_machine_research_direction_selection_20260919.md) | 已有实现/实验记录 |
| 0.90 | 连续状态观察模型与替换 | [报告](hegel_machine_stateful_observation_20260919.md) | 已有实现/实验记录 |
| 0.91 | 观察解释竞争、长实验与保留歧义 | [报告](hegel_machine_observation_competition_20260919.md) | 已有实现/实验记录 |
| 0.92 | 观察区分方法积累、改写和后继复用 | [报告](hegel_machine_instrument_method_revision_20260919.md) | 已有实现/实验记录 |
| 0.93 | 机制、重测和观察区分共同选择 | [报告](hegel_machine_observation_actions_20260919.md) | 已有实现/实验记录 |
| 0.94 | 研究方法适用条件与经验迁移 | [报告](hegel_machine_method_applicability_20260919.md) | 已有实现/实验记录 |
| 0.95 | 对象任务共同研究选择 | [报告](hegel_machine_object_action_selection_20260919.md) | 已有实现/实验记录 |
| 0.96 | 风险、机制与输入修订共同选择 | [报告](hegel_machine_risk_action_selection_20260919.md) | 负结果及.1流程修复保留 |
| 0.97 | 弱证据风险解释驱动追加实验 | [报告](hegel_machine_risk_experiments_20260920.md) | 已有实现/实验记录 |
| 0.98 | 新概念驱动旧未解题回访 | [报告](hegel_machine_successor_revisits_20260920.md) | 已有实现/实验记录 |
| 0.99 | 新知识挑战旧完成判断 | [报告](hegel_machine_completed_challenges_20260920.md) | 已有实现/实验记录 |
| 0.100 | 对象结构旧题续研 | [报告](hegel_machine_object_revisits_20260920.md) | 已有实现/实验记录 |
| 0.101 | 匿名对象归纳续研 | [报告](hegel_machine_anonymous_revisits_20260920.md) | 含.1衔接修复 |
| 0.102 | 按实测区分力改写归属研究步骤 | [报告](hegel_machine_dependency_method_revision_20260920.md) | 已有实现/实验记录 |
| 0.103 | 由剩余归属假设缩减实验 | [报告](hegel_machine_dependency_contrast_selection_20260920.md) | 已有实现/实验记录 |
| 0.104 | 对象归属与响应联合修订 | [报告](hegel_machine_joint_object_response_20260920.md) | 已有实现/实验记录 |
| 0.105 | 关系子程序跨组合复用 | [报告](hegel_machine_relation_kernel_transfer_20260920.md) | 已有实现/实验记录 |
| 0.106 | 已学关系提前进入响应拟合 | [报告](hegel_machine_relation_kernel_composition_20260920.md) | 已有实现/实验记录 |
| 0.107 | 研究投入与已有解释保留 | [报告](hegel_machine_research_investment_20260920.md) | 已有实现/实验记录 |
| 0.108 | 知识采用与跨仪器关系 | [报告](hegel_machine_knowledge_adoption_20260920.md) | 已有实现/实验记录 |
| 0.109 | 跨仪器关系与响应研究预算 | [报告](hegel_machine_relation_response_budget_20260920.md) | 已有实现/实验记录 |
| 0.110 | 响应研究投入顺序 | [报告](hegel_machine_response_investment_20260920.md) | 已有实现/实验记录 |
| 0.111 | 响应组合与关系范围联合发现 | [报告](hegel_machine_joint_response_discovery_20260922.md) | 已有实现/实验记录 |
| 0.112 | 有效边界证据与完整投入 | [报告](hegel_machine_adaptive_boundary_research_20260922.md) | 已有实现/实验记录 |
| 0.113 | 实验方法的响应背景迁移 | [报告](hegel_machine_response_context_transfer_20260922.md) | 已有实现/实验记录 |
| 0.114 | 保留组合、修订关系范围 | [报告](hegel_machine_composition_range_research_20260926.md) | 已有实现/实验记录 |
| 0.115 | 仪器字段局部续修 | [报告](hegel_machine_instrument_field_repair_20260926.md) | 已有实现/实验记录 |
| 0.116 | 未决研究和完整知识多代交接 | [报告](hegel_machine_knowledge_lineage_20260926.md) | 已有实现/实验记录 |
| 0.117 | 当前反例重开响应基底 | [报告](hegel_machine_response_basis_reopening_20260926.md) | 已有实现/实验记录 |
| 0.118 | 仪器修订后继续响应研究 | [报告](hegel_machine_instrument_response_research_20260926.md) | 已有实现/实验记录 |
| 0.119 | 新知识驱动旧问题区分实验 | [报告](hegel_machine_knowledge_update_revisits_20260926.md) | 已有实现/实验记录 |
| 0.120 | 追加研究相对当前解释归因 | [报告](hegel_machine_marginal_research_returns_20260926.md) | 已有实现/实验记录 |
| 0.121 | 检验并删除无用旧机制 | [报告](hegel_machine_mechanism_revision_20260926.md) | 已有实现/实验记录 |
| 0.122 | 冷启动范围与响应共同修订 | [报告](hegel_machine_calibration_scope_revision_20260926.md) | 已有实现/实验记录 |
| 0.123 | 非线性组合内部机制修订 | [报告](hegel_machine_nonlinear_mechanism_revision_20260927.md) | 已有实现/实验记录 |
| 0.124 | 非线性内部参数与空库组合发现 | [报告](hegel_machine_nonlinear_parameter_discovery_20260927.md) | 已有实现/实验记录 |
| 0.125 | 程序重放、坐标发现及修订路径选择 | [报告](hegel_machine_research_route_selection_20260927.md) | 已有实现/实验记录 |
| 0.126 | 完整轨迹重复区分系统残差与噪声 | [报告](hegel_machine_residual_noise_discrimination_20260927.md) | 已有实现/实验记录 |
| 0.127 | 含噪响应修订与独立比较 | [报告](hegel_machine_noisy_response_revision_20260927.md) | 已有实现/实验记录 |
| 0.128 | 含噪响应更新的追加投入 | [报告](hegel_machine_noisy_research_investment_20260927.md) | 已有实现/实验记录 |
| 0.129 | 时间尺度自主修订 | [报告](hegel_machine_temporal_scale_revision_20260927.md) | 已有实现/实验记录 |
| 0.130 | 空库响应与时间尺度联合发现 | [报告](hegel_machine_cold_temporal_discovery_20260927.md) | 已有实现/实验记录 |
| 0.131 | 重复实验、噪声停研与程序重放 | [报告](hegel_machine_research_reuse_cost_20260930.md) | 旧能力参照 |
| 0.132 | 含噪响应定向区分及任务确认 | [报告](hegel_machine_noisy_response_confirmation_20260930.md) | .0/.1均未整体采用 |
| 0.133 | 冻结未决响应候选、共享余款续研 | [报告](hegel_machine_response_question_investment_20260930.md) | 已有实现/实验记录 |
| 0.134 | 条件、响应、窗口及运算顺序联合发现 | [报告](hegel_machine_conditional_temporal_discovery_20260930.md) | 已有实现/实验记录 |
| 0.135 | 同背景扫点、脉冲和阶跃区分实验 | [报告](hegel_machine_conditional_experiments_20260930.md) | 已有实现/实验记录 |
| 0.136 | 多条件与时间层共同发现 | [报告](hegel_machine_composed_conditions_20260930.md) | 已有实现/实验记录 |
| 0.137 | 反例驱动方法步骤与条件响应修订 | [报告](hegel_machine_condition_response_actions_20260930.md) | 条件能力参照；不能替代0.131含噪能力 |
| 0.138 | 观测条件、窗口解码和背景改写 | [报告](hegel_machine_observed_condition_decoding_20260930.md) | 已有实现/实验记录 |
| 0.139 | 完整竞争解释、阈值区间与未决草稿 | [报告](hegel_machine_competing_observed_conditions_20260930.md) | 已有实现/实验记录 |
| 0.140 | 未决方法消费与研究投入 | [报告](hegel_machine_partial_method_choice_20260930.md) | 已有实现/实验记录 |
| 0.141 | 时间实验改写与后继方法机会 | [报告](hegel_machine_temporal_method_opportunity_20260930.md) | 已有实现/实验记录 |
| 0.142 | 未决结构续研与整轮方法回报 | [报告](hegel_machine_unresolved_temporal_research_20261001.md) | 已有实现/实验记录 |
| 0.143 | 实测支持结构优先与适用条件 | [报告](hegel_machine_supported_response_priority_20261001.md) | 已有实现/实验记录 |
| 0.144 | 实验反馈与追加投入试验 | [报告](hegel_machine_experiment_value_investment_20261001.md) | 整链退步；0.145改用0.143基底 |
| 0.145 | 平局保留旧付费反例修订 | [报告](hegel_machine_paid_counterexample_retention_20261001.md) | 已有实现/实验记录 |
| 0.146 | 当前窄区间的覆盖与投入 | [报告](hegel_machine_condition_experiment_coverage_20261001.md) | 已有实现/实验记录 |
| 0.147 | 先消费已付门槛反例再采购 | [报告](hegel_machine_paid_condition_revision_20261001.md) | 已有实现/实验记录 |
| 0.148 | 累积反例改写研究议程 | [报告](hegel_machine_paid_evidence_agenda_20261001.md) | 研究候选；整链未通过 |
| 0.149 | 局部结构持续参加完整响应竞争 | [报告](hegel_machine_paid_structure_response_20261001.md) | 研究候选；整链未通过 |
| 0.150 | 原响应检索旧方法与回报 | [报告](hegel_machine_method_revision_returns_20261001.md) | 研究候选；见退化复核 |
| 0.151 | 数值与风险研究步骤共同改写 | [报告](hegel_machine_counterexample_methods_20261001.md) | 研究候选；见退化复核 |
| 0.152 | 方法适用条件与初期投入 | [报告](hegel_machine_method_applicability_agenda_20261001.md) | 研究候选；见退化复核 |
| 0.153 | 跨实验部分结构支持与竞争 | [报告](hegel_machine_partial_structure_research_20261001.md) | 研究候选；见退化复核 |
| 0.154 | 局部线索与完整边界研究 | [报告](hegel_machine_informative_structure_research_20261001.md) | 发生旧能力退化；不恢复整版 |
| 0.155 | 噪声证据约束研究选路 | [报告](hegel_machine_noise_research_routing_20261001.md) | 修复0.131旧能力，保留0.137结构能力 |
| 0.156 | 含噪机制实验与有效确认 | [报告](hegel_machine_noisy_mechanism_confirmation_20261001.md) | 已有实现/实验记录 |
| 0.157 | 完整研究投入与知识来源归因 | [报告](hegel_machine_investment_and_attribution_20261001.md) | 已有实现/实验记录 |
| 0.158 | 确认投入和错误迁移撤回 | [报告](hegel_machine_research_confirmation_withdrawal_20261001.md) | 已有实现/实验记录 |
| 0.159 | 当前解释驱动研究投入 | [报告](hegel_machine_current_reference_research_20261002.md) | 已有实现/实验记录 |
| 0.160 | 实验命中、完整费用与确认资源 | [报告](hegel_machine_confirmation_investment_20261002.md) | 已有实现/实验记录 |
| 0.161 | 追加研究的边际收益 | [报告](hegel_machine_marginal_investment_20261002.md) | 已有实现/实验记录 |
| 0.162 | 当前响应风险与研究投入 | [报告](hegel_machine_response_risk_investment_20261002.md) | 已有实现/实验记录 |
| 0.163 | 实验背景与联合研究投入 | [报告](hegel_machine_background_research_investment_20261003.md) | 已有实现/实验记录 |
| 0.164 | 共享余额与经验适用性 | [报告](hegel_machine_shared_condition_research_20261003.md) | 已有实现/实验记录 |
| 0.165 | 共同失配与实际研究投入 | [报告](hegel_machine_joint_mismatch_investment_20261003.md) | 已有实现/实验记录 |
| 0.166 | 首次完整研究费用预测 | [报告](hegel_machine_cold_cycle_investment_20261003.md) | 已撤回，保留0.165 |
| 0.167 | 共享资源池下共同报价与选择 | [报告](hegel_machine_shared_research_priority_20261003.md) | 已有实现/实验记录 |
| 0.168 | 实测方法反馈重开旧计划 | [报告](hegel_machine_method_feedback_reopening_20261003.md) | 已有实现/实验记录 |
| 0.169 | 草案失败绑定限定撤回 | [报告](hegel_machine_draft_counterexample_withdrawal_20261003.md) | 已有实现/实验记录 |
| 0.170 | 完整回访费用进入报价 | [报告](hegel_machine_instrument_whole_visit_20261003.md) | 已有实现/实验记录 |
| 0.171 | 冻结旧、新解释的定向确认 | [报告](hegel_machine_revision_confirmation_20261003.md) | 已有实现/实验记录 |
| 0.172 | 区分收益与命中概率共用实测分布 | [报告](hegel_machine_coherent_confirmation_20261003.md) | 已有实现/实验记录 |
| 0.173 | 分批确认后的追加、重规划或暂缓 | [报告](hegel_machine_pair_confirmation_investment_20261003.md) | 已有实现/实验记录 |
| 0.174 | 全局确认和共享预算试验 | [报告](hegel_machine_shared_pair_confirmation_20261003.md) | .0/.1已撤回，恢复0.173 |
| 0.175 | 可执行修订估值试验 | [报告](hegel_machine_executable_revision_value_20261003.md) | .0/.1未采用，保留0.173 |
| 0.176 | 可执行研究组合与机会成本 | [报告](hegel_machine_research_portfolio_20261003.md) | 采用.1保留采购门槛版；放宽门槛分支撤回 |
| 0.177 | 实际方法收益归因 | [报告](hegel_machine_method_return_attribution_20261004.md) | 已有实现/实验记录 |
| 0.178 | 保留已选组合中仍可支付的后续研究 | [报告](hegel_machine_portfolio_companion_20261004.md) | 已有实现/实验记录 |
| 0.179 | 读数模型选择与确认预算 | [报告](hegel_machine_reading_model_selection_20261004.md) | .0/.1未采用，恢复0.178 |
| 0.180 | 普通successor迁移的实测来源回传 | [报告](hegel_machine_unified_source_returns_20261004.md) | 已有实现/实验记录 |
| 0.181 | 来源反馈重排待执行研究及条件限定撤回 | [报告](hegel_machine_source_feedback_selection_20261004.md) | 含后续未发布修复；不等同最初0.181快照 |

## Limitations

历史报告证明对应实现或实验已经发生；其分数不自动转为当前LLM选择、递归研究或新任务收益。撤回分支保留为可研究的机制与负对照。当前可执行能力以现存源码为准，原0.181缺少独立冻结快照的事实见后续审计。181项入口定位不代表181套历史实验重新通过，也不表示单个时序环境能承载所有研究合同。0.78普通Code对照保留独立脚本，其旧配置指向冻结源码及旧模型，运行当前对照须显式提供当前源码和获授权模型的配置。

最新方法修订中，首次编辑后实践由主机启动；两个成功作者面对同一组开发任务。第三个作者在新案例通过后又编辑退化、未发布，故保留已验证的更好程序及更广可靠性仍是当前缺口。
