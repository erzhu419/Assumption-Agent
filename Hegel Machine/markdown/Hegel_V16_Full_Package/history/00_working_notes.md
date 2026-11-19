# V16 历史考古工作记录

## 审查任务与边界

核对 Gemini 上传方案 `upload/hegel_v16_wormhole_synthesis_and_master_plan.md` 的历史主张：phase2_v2 42%、旧 v16 +72pp、phase three graph recall 0、SCAR 98.9% 弃权；追查结论到可执行问题/提问树的真实实现；明确可复用 Hippo/结构映射接口。

只读仓库原件及现有历史快照；不调用付费模型、不修改项目源文件。不得把搜索索引覆盖或分支清单等同于逐行读完整仓库。

## 首轮定位

- GitHub code search 索引仍落在 `71e133b4aba373839e5faf8ea9c9a45854039e24`。这些结果只用于发现文件路径，最终内容以父代理提供的本轮固定 HEAD 与必要分支提交为准。
- `phase2_redesign_notes.md`、`phase2_final_state_report.md` 明确有 phase2_v2 42% 的表；比较对象和指标待原件核对。
- 原始判断文件路径：`phase two/analysis/cache/judgments/phase2_v2_triggers_vs_baseline.json` 与 `phase2_v2_triggers_vs_phase2_triggers.json`。
- 旧 v16 路径为根 `v16_final_results.md`；相关实现：`phase one/scripts/validation/phase2_v12_framework.py`、`phase2_v15_exemplar_framework.py` 等。不能把旧 Phase2 v16 与目前 Hegel V16 混为一个版本。
- `phase three/scripts/run_formalization_v2.py` 会写 `phase three/analysis/formalization_v2_report.json`，需核对其 recall 0 对什么对象/样本、如何定义。
- Gemini 给出的 SCAR postmortem 精确路径已记录，搜索未命中并不说明文件不存在。

## 既有可沿用材料

`analysis_v15/hipporag_history.md` 和 `hipporag_sources_manifest.json` 已在 `787256bd415515905359626f6a5498174ad5db3c` 固定了早期十题结构演示、十五题 QA、三千题策略选择器及十二题官方 HippoRAG 对照。新轮必须核定源 blob 是否改变。

## 固定版本确认

父代理已枚举16个branch：`cognitive` HEAD仍为787256bd，`main`为71e133b4；无tags。当前读取的787256bd材料即本轮cognitive现状，不需要重新把旧指标当新提交结果。全branch/历史树覆盖由父代理维护，本分工聚焦可核证据，不重复声称全仓逐行审计。

## 初步确认与修正（已查源码/原始JSON）

1. `phase2_v2_triggers_vs_baseline.json` 100条judgment独立数得42胜58负。这个42%是pairwise judge win rate，不是客观任务正确率。v2同时缩成6条、全局展示、调整引用措辞；prompt还展示每条`core_insight`，不只是裸警句。
2. `phase2_v16_vs_baseline_long.json`独立数得86胜14负；+72pp是胜负差。该旧Phase2 v16为案例+二轮audit/revise，不能与当前Hegel V16同名混同。
3. 没有停在旧v16报告的“heldout下一步”：当前仓库有后续50题留出原JSON，vsbaseline_long44胜6负、vsv13_reflect41胜9负。题ID隔离与实际生成上下文仍需细查；这些正结果不能忽略。
4. `phase three/analysis/formalization_v2_report.json`在cognitive与main均404，现phase3 tree无analysis目录。已读取runner/SEG/kernel/_config；comparison不是末端公式AST，而是策略operational_steps→16类动作核/顺序图。expected pairs仅三个策略对：控制变量↔增量构建、求同↔类比、特殊到一般↔降维。Gemini将recall0归因于化学幂次公式和RL微分方程差异，没有此实验依据。
5. SCAR98.8935%是typed-incidence单步删除715/723原可选variant；full+length2只5/724有输出，最终总弃权719/724≈99.3094%。后来baseline-preserving repair恢复coverage但U1−S0=−1/724、0正1负override。结构pool仍新增38/724正确候选，不能由gate失败推导所有结构先验无效。
6. `question_semantics.py`已经有meaning版本/历史、use_sites、child_answers、parent_use_history、可执行query编译。应继续区分一般question持久化与当前跨题memory是否把研究policy作为可检索可执行对象；不能笼统说系统只有末端Program。


## 补记：已存在的可执行问题与研究 DAG

固定 cognitive HEAD 787256bd 下，`question_semantics.py` 的版本化编译契约、`question_continuation.py` 的求值与 parent use 不是仅有设计文字。更直接地，`KnowledgeLibrary.research_workflows` 被持久化；默认 `ResearchConfig.learn_research_policy=True` 时，`unified_research.py` 创建 `WorkflowEpisode`、调用 replay 和 finish。`growth_methods.py` 从成功实测研究轨迹编译有 consumes/produces/depends_on 的 DAG，抽象公共控制角色、范围、通道；`growth_procedures.py` 绑定 instrument / conditional_variance 子问题。复用失败会重开普通研究，最终必须由原生新确认检验产生收益回执。尚不能只凭这些代码证明 V15 特定实测确实复用了该机制。

因此 Gemini 的“跨任务只保存终端 Program、完全丢掉提问树”过度概括。真实欠缺更可能是：问句语义契约、已学习操作 DAG、条件化概念与拟议上层 motif 的统一身份、适用域和收益归因，而不是从零恢复一棵树。

## 补记：旧 v16 的缓存身份限制

原 100 题与后续 50 题题号完全不交；后续原始两两判决仍确实为 44:6 与 41:9，应保留为正向历史证据。当前默认 `phase2_v3_selections.json` 只覆盖原 100 题、没有 50 题题号；当前 diverse exemplar 缓存已扩至 79 个条目而原报告为 75，出现 9 个留出题号。对当前原 100 题执行 v16 原始选例算法，在 70 个非数理任务中有 16 个取到了自己的历史作答示例（17 个实例）。这说明当前缓存无法单独充当当时冻结执行输入；没有运行命令和当时快照，不能反推原运行一定泄漏这些例子，也不能宣称原结果就是严格未见迁移。复核明细在 `phase2_v16_recomputed.json`。


## 收束与覆盖边界

最终子报告已写入 `01_历史证据与可执行问题链审计.md`。本轮 41 份原始文件全部按固定 Git blob SHA 检查字节一致，详见 `source_manifest.json`。复算脚本 `recompute_history.py` 仅读已有 JSON，无模型调用。已有研究 DAG 不能等同完整诊断决策树：success=False 会被剪除，但 model_confirmed=False 的确认会转 vary_controls；必须分别记录动作执行失败和成功取得否证，不能由当前代码无条件推断所有反例丢失。已有 V15 报告中未检出通用工作流实际回放的 receipt；接口存在、默认启用条件、某实测发生消费应分别表述。全部 16 分支头没有被引用的 formalization_v2_report；历史提交未由本子任务遍历。
