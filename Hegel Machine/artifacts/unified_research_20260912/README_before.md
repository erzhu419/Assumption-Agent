# Hegel Machine

黑格尔机研究 Agent 如何主动提出、检验和修订假设，并把学到的表示、关系与模型积累为后续可调用、可修订的知识。

当前 **0.58.0** 补上部署发现进入下一任务研究的路径：每题研究与实际优化结束后，先回收修订知识，再研究后继任务。三种子、三组流程对照完成36次优化；9个后继任务有6个实际调用新知识，后继搜索计算减少14.03%，关闭部署回传后该收益消失。观测数、最终目标与可行性持平。

[完成度与方向复核](docs/hegel_machine_advice_goal_status_20260909.md)按原建议重新核对：发现、检验、修订、调用均有真实执行；统一未知任务的研究选择、更一般的问题分解及稳定长期净收益尚未完成。当前整合入口的25个阶段仍含开发者编排，后续重点是测量、历史、表示和风险的共同选择，以及同一任务流上的完整对照。

- [能力地图](docs/hegel_machine_capability_map_20260909.md)与[系统架构](docs/architecture.md)：已实现的模块、入口和范围。
- [部署知识的连续研究实验](artifacts/cognitive_direction_reset_20260912/summary.json)：本次实际结果；[原建议](markdown/GPT6_advice.md)与[具体方案](markdown/Hegel_Machine_Vision_Progress_and_Research_Critique.md)为方向依据。
- [测量知识复用](docs/hegel_machine_measurement_acquisition_20260909.md)、[元先验研究](docs/hegel_machine_meta_prior_research_20260910.md)、[普通Agent比较](docs/hegel_machine_vanilla_and_completion_20260910.md)：此前跨任务与对照证据。
- [0.57约束知识保留](docs/hegel_machine_constraint_knowledge_20260912.md)、[0.56实验投入](docs/hegel_machine_dual_value_20260912.md)、[历史施工记录](markdown/GPT6_advice_Implementation_20260907.md)：保留已有策略、结果及未采用实验。

## 运行入口

模拟器与结构研究可单独运行；普通代码比较使用[可编辑资源配置](config/cognition_code_session.json)，会调用当前配置的真实模型。以下检查点用于新运行，后续命令继续同一状态。

```bash
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/prior_session.json --evaluate
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/sc_prior_session.json --request config/cognition_sc_kg_prior.json --optimize --optimizer-config config/cognition_sc_kg_optimizer.json --optimizer-steps 1
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/sc_prior_session.json --evaluate
PYTHONPATH=src python3 -m hegel_machine cognition simulator --checkpoint artifacts/simulator.json
PYTHONPATH=src python3 -m hegel_machine cognition mechanism --family order --variant broken --output artifacts/order.json
PYTHONPATH=src python3 -m hegel_machine cognition code-session --checkpoint artifacts/code_sequence.json --workspace artifacts/code_workspace --request config/cognition_code_session.json --steps 4
PYTHONPATH=src python3 -m hegel_machine cognition code-session --checkpoint artifacts/code_sequence.json --workspace artifacts/code_workspace --evaluate
```

上述SC-KG第一条命令完成一题研究与部署后暂停，第二条继承保存的配置，逐题研究、部署并回收知识，最后评价。

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
