# Hegel Machine

黑格尔机研究 Agent 如何主动提出、检验和修订假设，并把学到的表示、关系与模型积累为后续可调用、可修订的知识。

当前 **0.57.0** 已采用[约束知识保留与程序系数修订](docs/hegel_machine_constraint_knowledge_20260912.md)：验证后的机制重建同时保留观测对均值和不确定性的影响，程序各项可以分别调整并接受未来实测比较。79例开发、新9例目标任务和新3例机制变化的最终目标均保持；开发约束预测MSE下降18.26%，新机制三例平均下降37.71%。主要失败案例的53个乐观均值误判降至0；初始模型及部分边界误差仍需解决。

此前 **0.56.0** 接通[复测与候选验证的共同投入选择](docs/hegel_machine_dual_value_20260912.md)：校准当前目标值，比较两类实验的预计收益，并完成预算内最后一次验证。70例开发为3改善、66持平、1退化；当批新九例为1改善、8持平，平均最优值差距降低56.67%。

此前 **0.55.0** 完成[共享候选验证与联合识别实验](docs/hegel_machine_joint_identification_20260911.md)。联合方案52例开发为8改善、36持平、8退化；独立新九例平均最优值差距增加35.2%，因此未默认采用。默认选点恢复0.54，保留共享未来验证及候选身份修复。

此前 **0.54.0** 接通[目标解释重新竞争与边界研究](docs/hegel_machine_goal_recompetition_20260911.md)。43例开发对0.53为3改善、40持平、0退化，可行解42→43；当批新九例为1改善、8持平，全部可行，平均最优值差距降低40%。当批新九例对普通SC-KG为8胜、1负，全域目标预测误差上升69.4%。

此前 **0.53.0** 接通[目标研究完成预算与解释部署](docs/hegel_machine_goal_completion_20260911.md)。该版独立新九例对0.52为5胜、4平、0负，对普通SC-KG为9胜，全部可行；平均最优值差距0.039846→0.017944，目标解释已实际进入GP并改变推荐。34例开发与保留比较、九例新种子确认共完成61次优化。

此前 **0.52.0** 保留被局部比较拒绝的目标候选及正反证据，重新选择区分实验，并尝试用已验证约束概念修订目标表示。[544 例完整比较](docs/hegel_machine_objective_reentry_20260911.md)相对 0.51 终值全部持平，可行数均为 538；目标平均 MSE 下降 11.24%，新 27 例下降 32.57%。约束误差总体上升 0.063%、新例下降 1.77%。新例仅一例实际重入并改善；该批收益来自后续选点更新原 GP，未出现目标新表示的独立采用收益。其后[首版交互实验](docs/hegel_machine_objective_interactions_20260911.md)因新确认退化未采用，0.53补充完成预算与部署路径。

此前[重入比较与未知最优点试跑](docs/hegel_machine_reentry_value_20260911.md)完成两版重入策略开发，均未支持替换默认策略。随后新增三个未知内部最优点任务，与普通 SC-KG 共享初始观测、等在线预算比较：0.52 的终值三例全部更好且可行，平均最优值差距 0.03836 → 0.000533。该批共新增 1344 次物理调用；未知目标结果支持现有目标建模与实验选点组合，尚未证明递归修订或长期净收益。

此前 **0.50** 将目标 GP 替换作为独立假设，用后续实测比较后采用；约束范围修订可以先完成，待决目标替换继续共享已付观测。[490 例完整比较](docs/hegel_machine_objective_replacement_20260911.md)相对 0.48 终值全部持平，可行数均为 486；目标预测平均 MSE 下降 12.55%，预先固定的新 27 例下降 11.01%。目标解释选择已有改善，终值增益尚未出现；新例约束预测退化与下一步见该报告的 Limitations。

此前 **0.48** 的[冻结目标候选续验](docs/hegel_machine_budgeted_validation_20260910.md)完成 436 例，相对 0.47 为 2 改善、434 持平、无终值退化；一例可行任务目标降低 14.91%，另一例改善后仍不可行。共享续验结果与主动规划相同且计算更少，继续保留。[0.49 的两种重建后比较方案](docs/hegel_machine_current_incumbent_20260911.md)因预测退化未采用；[0.46 噪声联合策略](docs/hegel_machine_active_noise_experiments_20260910.md)也未默认采用。

此前原生 SC-KG 共享初始观测对照为 355 例中 339 胜、11 平、5 负，新 27 例独立冷启动为 26 胜、1 负，见[两类 Agent 对照](docs/hegel_machine_research_value_20260910.md)。文首的新任务对照另计，本轮未新增 Code Agent 对照。模型重建后的候选采用和稳定长期净收益仍需改进，见最新报告的 Limitations。

统一入口包含 25 个阶段，覆盖持续任务、仪器选择、实验迁移、方法投入、局部机制、对象与尺度、结构先验、元先验研究和延迟反馈。

- [当前能力地图](docs/hegel_machine_capability_map_20260909.md)：逐项对应 GPT6 建议，列出实际模块、入口和能力边界。
- [0.29 知识指导测量投入](docs/hegel_machine_measurement_acquisition_20260909.md)：三组各 24 题，相对旧共享版测量费减少 52.9%、计算减少 26.0%，观测增加 14.7%；平均 MSE 降至 1.50e-13，619 项认知回归通过。
- [0.28 表示跨任务复用](docs/hegel_machine_measurement_transfer_20260909.md)：三组各 24 题，相对旧共享版少用 18.86% 搜索计算，平均误差保持约 1.55e-7；611 项认知回归通过。
- [0.27 仪器与表示修订](docs/hegel_machine_measurement_revision_20260909.md)：重拟合压缩后的状态读出、比较多种表示、持续残差重开与已付测量复用；16 题实际研究完成，607 项认知回归通过。
- [0.26 剩余功能整合](docs/hegel_machine_completion_20260909.md)：模拟器、结构发现与持续代码研究；24 阶段整机、15 个结构案例和八题普通代码对照完成，600 项认知回归通过。
- [0.25 研究评价器自身修订](docs/hegel_machine_research_evaluator_20260909.md)：函数学习、后续比较与实际选择；574 项认知测试通过，构造功能例改变选择，两种子对照尚无净收益。
- [0.24 共享前置研究](docs/hegel_machine_shared_research_20260909.md)：实际后继需求、可修改依赖与共享知识调用。
- [0.23 完整决策与动态议程](docs/hegel_machine_complete_research_agenda_20260909.md)：本轮功能、实际结果与下一步；[目标核对](docs/hegel_machine_advice_goal_status_20260909.md)说明哪些建议仍未达成。
- [0.22 区分实验与冲突续研](docs/hegel_machine_discriminating_research_20260909.md)：同轮实验与拟合、分批决策比较及跨访问证据累积；537 项认知回归、三组完整配对和 13 阶段整机完成，方法臂行动损失两组下降、一组上升。
- [0.21 风险决策研究](docs/hegel_machine_decision_research_20260909.md)：完整方案保留、短比较、新知识重开；509 项认知回归与 13 阶段整机完成，四组完整序列的平均行动损失两组下降、两组上升。
- [0.20 持续研究能力与结果](docs/hegel_machine_continuing_research_20260909.md)：上一轮整批功能、完整任务序列、成本和整机运行。
- [整机目标与当前路线](docs/hegel_machine_system_completion_20260908.md)：当前生命周期和后续建设方向。
- [0.18 迁移修订](docs/hegel_machine_transfer_repair_20260909.md)、[0.17 知识生产](docs/hegel_machine_knowledge_production_and_boundaries_20260909.md)、[0.16 知识价值](docs/hegel_machine_knowledge_value_20260909.md)、[0.12 整合系统](docs/hegel_machine_integrated_research_20260908.md)：保留各阶段实际结果。
- [原始 GPT6 建议](markdown/GPT6_advice.md)、[详细审视方案](markdown/Hegel_Machine_Vision_Progress_and_Research_Critique.md)、[系统架构](docs/architecture.md)：方向依据与实现组织。

## 运行入口

模拟器与结构研究可单独运行；普通代码比较使用[可编辑资源配置](config/cognition_code_session.json)，会调用当前配置的真实模型。以下检查点用于新运行，后续命令继续同一状态。

```bash
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/prior_session.json --evaluate
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/sc_prior_session.json --request config/cognition_sc_kg_prior.json --evaluate
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/sc_prior_session.json --optimize --optimizer-config config/cognition_sc_kg_optimizer.json
PYTHONPATH=src python3 -m hegel_machine cognition simulator --checkpoint artifacts/simulator.json
PYTHONPATH=src python3 -m hegel_machine cognition mechanism --family order --variant broken --output artifacts/order.json
PYTHONPATH=src python3 -m hegel_machine cognition code-session --checkpoint artifacts/code_sequence.json --workspace artifacts/code_workspace --request config/cognition_code_session.json --steps 4
PYTHONPATH=src python3 -m hegel_machine cognition code-session --checkpoint artifacts/code_sequence.json --workspace artifacts/code_workspace --evaluate
```

`mechanism` 的 `--family` 支持 `low_rank`、`modules`、`dynamics`、`topology`、`order`，`--memory PATH` 共享程序知识。`code-session` 的后续请求可以用 `new_tasks` 追加实例；总资源继续约束整条序列。

## 核心研究问题

> 当旧模型失配、解释尚不可区分，或重复求解成本过高时，Agent 能否在有限预算下选择实验，构造有效的表示、关系或不确定性修订，将有用内容积累为知识，并在后续未见任务中降低累计损失与学习成本？

```text
固定任务目标、公开能力和整场研究预算
  → 观测、已有程序、抽象与研究经验
  → 竞争解释及依赖子问题
  → 单步或联合计划生成可执行候选
  → 固定候选后获取新轨迹、比较预测与成本
  → 构造阈值转换与任务价值 probe、检查定向反例
  → 直接复用、参数重拟合或角色映射
  → 条件作用域修订与反例撤销
  → 保存会话、解决下一任务或重访未解决问题
```

## 当前实现

控制器从残差和程序复杂度形成研究问题，执行带中间程序和依赖关系的修订计划。联合搜索保留多个变量组合，对系数共同拟合，并允许删除、替换早期选中的项；当前变量、平方项、交互项和不同历史深度分别比较。分轮预算在搜索和选实验前为后续候选比较留出计算，使新查询所得证据能够进入选择。

历史尚不足时，程序通过 `available(k)` 分支使用从观测中学到的起始模型。各历史深度的基础起始拟合先轮流获得预算，再使用余量搜索更复杂分支。后续计划可以使用此前形成的表示或谓词，计划预算会按实际新观测上的收益调整。

状态搜索从实际观测中的两个输出值提出历史状态解释。原始变量规则与数据拟合的联合坐标共同竞争，进入、退出及起始阈值分别学习。多输入实验在候选坐标附近改变控制组合，并在不同准备历史后比较相同输入，让表示变化带动观测变化。

主动观测在公布的连续控制范围内构造轨迹。候选先固定，在新参考轨迹上比较，再为剩余问题选择实验。转换 probe 从阈值区间构造相同输入、不同准备历史的测试，联合安排不同边界，并从实际响应判断准备状态是否达到；其他实验仍按任务风险下降和控制覆盖选择。定向反例会阻止提前结束，查询安排为下一轮参考观测保留额度。代表性轨迹长度随候选所需历史增长，受公开长度和剩余预算限制；新证据继续进入拟合与经验行为分组。

知识库同时保存具体程序、带参数槽和角色槽的抽象，以及可调用定义。定义可嵌套，实参按实际分支需要求值；同一观测的重复调用共享计算。学到的定义进入后续拟合与组合搜索，复杂度仍按展开语义衡量。后续任务可以直接执行程序，也可以按当前证据筛选结构、重拟合参数和映射变量角色；实际迁移结果反馈到后续结构搜索。观测支持区间与反例用于生成可执行的条件作用域；已支持情境内的新反例可以撤销程序。删除指定概念会连带删除依赖它的派生知识和抽象，供因果干预比较。

研究经验也跨任务保存。按实际观测的 binary、numeric、risk 及单输入/多输入条件组织新参考预测收益与支出，反馈到状态搜索、联合坐标和依赖计划的预算分配。候选筛选保留同等拟合但使用不同观测变量的解释；更复杂的候选若明确解释已有反例、收益超过新增复杂度且参考与新预测不退化，也能进入修订。

迁移调度共同测试保留原参数的结构和重新拟合的结构。既有证据与新参考轨迹均支持、且没有待探测转换区间的迁移可以早停；两类当前证据都较差的旧候选退出当前候选池。

任务固定预测损失和行动成本。风险任务比较直接 pinball 拟合与位置、尺度共享尾部证据的程序，使用累计参考轨迹判断候选。继续研究时，旧程序和其已部署校准作为完整方案保留；新程序及其校准需要在两条未参与拟合的后续轨迹上取得支持，才能替换旧方案。程序改变时，即使没有选中单独校准方法，也会安排这次比较。轨迹收益相互冲突且仍有有用改善的可能时，系统给下一对代表轨迹报价，追加只计算新证据；未决时保存完整方案、校准和逐轨迹结果，进入后续短研究。报告分别测量预测损失、行动损失、区间覆盖、观测成本、程序计算单位、运行时间和模型用量。

连续状态可以从已见轨迹学习并保存为可执行递推程序。可选测量环境把控制与仪器读数分开，先按预测收益和费用选择通道，再进入同一实验、修订和知识循环；后续残差可重新打开仪器问题，实际购买新读数并更新原工作区。`continuous_state` 也参与普通控制器的研究动作选择。累计证据支持的解释会跨次研究保留，可调用搜索原子只在原始拟合仍有残差时按其实际贡献参与选择；完整程序和拟合模板仍参与启动检索。

## 快速运行

在本目录先建立一场跨分区、共享知识的研究；第二条命令在新进程中加载并继续：

```bash
PYTHONPATH=src python3 -m hegel_machine cognition session \
  --partitions development confirmation forward continual stateful \
  --seed 809 --checkpoint artifacts/hegel_session.json --steps 2

PYTHONPATH=src python3 -m hegel_machine cognition session \
  --checkpoint artifacts/hegel_session.json
```

`--steps` 限制本次新增的完整任务访问数；每完成一次访问都会更新 checkpoint。`--total-observation-budget`、`--total-compute-budget` 控制整场研究资源，`--max-visits` 控制每个任务最多访问次数，默认 2。恢复时可以显式增大这些总额度，继续未解决的工作。

会话先覆盖任务，再结合未解决误差、可执行的区分实验、新知识和条件方法经验安排重访。已付费观测与候选随工作区恢复；保存实验同时预留后续比较所需观测。风险任务还可保存当前完整决策与至多两个已有挑战方案，以两条新轨迹作一次有限比较；调度器按它的实际报价与普通修订、实验和方法共同选择，执行后的收益与费用决定后续投入。非零 pinball 损失继续保留。`completed` 表示允许的访问已经结束，`unresolved_tasks` 继续列出未解决任务；`budget_exhausted` 保留待办。Python 的 `ResearchSession` 同时接受环境对象或环境工厂，外部环境只需提供 `spec` 与 `observe`。

可调用概念的独立演示用观测学习基础规律，再在后续任务中组合并测量调用收益：

```bash
PYTHONPATH=src python3 scripts/run_callable_concept_demo.py --output artifacts/callable_concept_demo.json
```

可选测量与连续状态的进程环境使用独立入口；默认四题共享知识，支持任务边界恢复：

```bash
PYTHONPATH=src python3 -m hegel_machine cognition lab \
  --checkpoint artifacts/lab_research.json --backend process --steps 1

PYTHONPATH=src python3 -m hegel_machine cognition lab \
  --checkpoint artifacts/lab_research.json --resume
```

`--strategy controls_only` 和 `--strategy all` 用各自 checkpoint 运行测量对照。实际观测、仪器费用和研究计算分别记录，独立评估计算另列。

固定任务序列与算法对照继续使用以下入口：

```bash
PYTHONPATH=src python3 -m hegel_machine cognition run \
  --output artifacts/cognition_run.json

PYTHONPATH=src python3 -m hegel_machine cognition compare \
  --output artifacts/cognition_comparison.json

PYTHONPATH=src python3 -m hegel_machine cognition research \
  --partition development --seeds 7 11 19 \
  --output artifacts/cognition_research.json
```

`run` 输出任务序列和学习轨迹；`compare` 执行配置对照及知识删除干预；`research` 汇总多个种子的配对差值和置信区间。指定 `--output` 时还会生成 `_summary.json`，日常分析可先读该小型文件。`run --memory PATH` 可以加载并保存实验知识库。

0.8.0 与保存于 `artifacts/cognition_v5_reference/` 的 0.7.0 算法包使用专门脚本比较。下面运行两个开发分区及九个对照臂：

```bash
python3 scripts/run_cognition_v6_research.py \
  --partitions continual stateful \
  --systems reference_v5 cognitive_v6 without_joint_state_coordinates \
    without_joint_state_probes without_research_memory without_active_experiments \
    without_knowledge fixed_model program_search \
  --output-dir artifacts/my_v6_research
```

脚本默认运行 continual，种子为 401、503、607，并包含上述九臂。stateful 默认种子为 809、907、1009，多分区请求中它最后运行。五个分区均已提供开发依据。`--partitions`、`--systems` 和 `--seeds` 可以缩小范围；`--resume` 继续相同设计的保存结果。各臂共享环境、初始与目录实验、最终评测和预算上限，默认每任务 96 个观测、24000 个计算单位。

输出目录保存每分区的 `*_runs.json`、`*_summary.json` 和总 `summary.json`，仅保留少量详细病例。报告保留实际查询长度，终局损失按历史长度 0–7、8–31、32 及以上分组；二值状态任务另计分类错误、转换行错误及各自分母。额外评测工作单列。该脚本使用本地程序搜索。

默认使用本地程序搜索。显式选择 `--proposer codex` 或 `--proposer hybrid` 时，通过本机已配置的 Codex 调用模型提出程序和 probe：

```bash
PYTHONPATH=src python3 -m hegel_machine cognition run \
  --proposer codex --llm-calls-per-task 1 --llm-token-budget 4000 \
  --output artifacts/cognition_codex.json
```

比较时，`--include-llm` 增加模型生成对照；`--systems` 可以选择需要的配置，例如 `cognitive_agent without_knowledge single_step_planning`。调用数、耗时和可取得的 token 使用单独报告。全部选项见 `python3 -m hegel_machine cognition --help` 及相应子命令帮助。

## 环境与测量

development 有 17 个 episode，confirmation、forward 与 continual 各有 12 个，包含参数、非线性、历史、迟滞、风险、角色映射和机制变化。四个分区均已用于当前开发，保留原定义供版本回顾比较。

stateful 有 12 个 episode，包含不同阈值、方向和控制角色的滞回、双变量组合、角色重命名、同上下文状态条件变化，以及线性、平方、交互、历史和偏斜风险任务。训练允许最长 32 步，最终独立评测使用八条 64 步轨迹。分区和机制标签只进入评测报告，控制器取得公开任务能力、训练观测与固定目标。

最终评测在每个学习 episode 结束后执行。报告保留所有尝试任务的分母，分别统计失败、预测覆盖、修订、直接复用、抽象复用和作用域变化。fresh 检查说明当时证据与决定是否一致；独立最终评测另行比较被采纳、回退的程序与其父程序，计量错误接受和错误回退。多种子的区间以种子为重采样单位；知识价值按质量、采样、计算和时间分别比较。

通用 `compare` 保留固定模型、普通程序搜索、主动实验、知识、计划、抽象、角色映射、构造 probe、候选选择等对照，以及全库和特定概念删除干预。当前版本脚本重点区分联合状态坐标、多输入实验、跨任务研究经验与知识的贡献。

通用先验已有压缩、稀疏组合、局部历史、机制复用、分段状态和残差不确定性等执行路径。更一般的守恒、对称、低秩、拓扑和潜在状态发现仍需具体表示及环境操作，详见[整机完成路线](docs/hegel_machine_system_completion_20260908.md)。

## 模块与职责

| 模块 | 职责 |
|---|---|
| `session` | 持续研究、任务重访、共享资源、保存与恢复 |
| `decision_frontier`、`decision_research`、`research_opportunities` | 保存完整决策问题、执行短比较，并按实际收益与其他续研机会竞争 |
| `models`、`diagnosis`、`planning` | 任务、预算、竞争解释、依赖步骤与中间程序 |
| `language`、`synthesis`、`sparse_search`、`state_search`、`risk_search`、`proposers` | 可执行语言、联合变量组合、状态转换、尾部拟合和模型生成接口 |
| `continuous_state` | 从轨迹学习连续递推、驱动方向和有限读出 |
| `measurement_lab`、`measurement_research`、`measurement_session` | 仪器操作、测量选择、进程装置及恢复会话 |
| `probing`、`environments` | 连续实验构造、经验行为分组、实际观测 |
| `abstractions`、`memory`、`concept_selection` | 参数与角色抽象、复用、研究经验、条件作用域和撤销 |
| `uncertainty`、`evaluation` | 残差不确定性、完整程序与校准比较、行动选择及独立最终计分 |
| `selection`、`controller` | 新轨迹比较、起始历史分支、计划分配与学习循环 |
| `experiments`、`research_protocol`、`cli` | 对照、删除干预、多种子分析和运行入口 |

代码位于 `src/hegel_machine/cognition/`。各模块的数据关系见[系统架构](docs/architecture.md)。

## 正式验证轨道与历史

已有结构定律验证、有限商闭包、membership、精确 wire 和证书工作保留为专门正式轨道，可为具体研究对象提供额外数学证据。对应冻结协议、产物和声明继续按各自身份读取。

方向调整前的[原 README](docs/history/formal_track_readme_20260907.md)和[原 architecture](docs/history/formal_track_architecture_20260907.md)保留原文。[首轮运行](docs/cognitive_runtime_first_run.md)、[0.4.0 研究](docs/cognitive_runtime_v2_results.md)、[0.5.0 研究](docs/cognitive_runtime_v3_results.md)、[0.6.0 研究](docs/cognitive_runtime_v4_results.md)、[0.7.0 研究](docs/cognitive_runtime_v5_results.md)、[0.8.0 实现与结果](docs/cognitive_runtime_v6_results.md)与[0.11.0 功能进展](docs/hegel_machine_measurement_and_state_20260908.md)保留各自的历史结论。当前能力与结果入口见文首。历史副本中的相对链接按其原始文件位置解析。

## Limitations

- 当前语言由有限数值、Boolean、历史与组合 primitive 构成；抽象和模型生成输出仍在该语言中执行。任意新传感器、开放世界元算子和通用概念发明尚未得到验证。
- 动作调度、依赖计划、probe 搜索和作用域学习采用有限启发式；代码贯通这些机制不等于它们已具备普遍认知优势。
- 五个分区均使用公开 synthetic 生成器，也均已是开发依据。公开生成器不构成独立托管的隐藏基准。
- 0.8.0 同时改变联合状态坐标、实验、候选保留及跨任务研究经验，版本总差值需要结合消融解释。会话完成与所有任务已经学会分别报告。
- 二值状态分支依据可观测的两值输出学习条件阈值，连续状态分支学习有限递推族；这些能力尚不覆盖一般潜在状态恢复；分类错误按固定 0.5 阈值统计，也不等于恢复了真实机制。
- continual 与 stateful 的 64 步评测超出最大 32 步训练轨迹，需要按历史长度解释时域外推。`worst_case_loss` 是所学区间端点损失的平均最大值，零宽名义区间会给出零，不能用于断言实际测试尾部风险更低。
- 最终评测与自适应 validation 分开；少量种子的置信区间仍可能不稳定。预测误差口径的误扩展/误拒绝不代表最佳操作真值，也不包含复杂度收益。
- 计算单位、实测时间与 token 分别报告。模型返回的未知 token 保留为未知；调用预算约束后续调用，实际模型消耗以返回用量为准。
