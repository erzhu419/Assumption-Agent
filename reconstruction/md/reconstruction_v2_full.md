根据 2026-06-10 的 gap 文档和你 repo 里当前 `assumption_os` 的状态，我的判断是：

**Phase 0–7 不是完整路线的终点，而是“第一版可运行闭环 / v1 kernel”。**
它已经证明这件事不是纯概念：现在系统已经有 residual → hypothesis → preflight → world-model screen → ablation/judge → gate → recursive resume → apply/reject → descendants 的递归闭环；但你自己的 gap 文档也明确写了，目前更像 **75–80% 的 recursive hypothesis-validation prototype**，而不是完整长期运行的 Hypothesis OS，general long-running OS 大约还在 50–60%。

所以答案是：

> **不需要把 Phase 0–7 推倒重做；需要把 Phase 0–7 冻结成 kernel，然后对每个 phase 做 v2/v3 的“能力升级”，但升级时永远只替换一个模块，其余模块保持稳定。**

这正好符合你自己已经总结出来的“增量替换 / 控制变量法”。

---

# 0. 先回答你的核心疑问：Phase 0–7 完整么？

**不完整，但已经足够作为地基。**

现在的 Phase 0–7 更像：

```text
MVP: 能不能把递归假设生成、验证、图记忆、world model gate、daemon 跑通？
```

真正完整版本要回答的是：

```text
Scientific / AGI-level version:
1. 假设表示是否稳定？
2. 假设新增是否可回滚、可证伪、可继承？
3. world model 是否能做前瞻模拟，而不只是 cheap gate？
4. generator 是否能发现新 family，而不只是修补旧失败？
5. verifier 是否能控制 false positive？
6. graph memory 是否越用越强，而不是越用越脏？
7. agent 是否能区分“假设错了”和“没执行好”？
8. 系统是否能跨 domain 学到方法层规律？
9. 当已有哲学策略不够时，能否提出新策略并验证？
10. 这些能力是否在 frozen unseen benchmark 上赢过强 baseline？
```

你现在的系统已经有很多模块原型：schema、graph memory、metaproductivity selector、residual taxonomy、candidate overlay、recursive runner、evolution context、assumption bench、manifest logger、world model、trajectory search、verifier stack、recursive daemon、residual clusterer 等；这说明不是“缺架构”，而是缺每个模块的**研究级版本**。([GitHub][1])

---

# 1. 不要“一个 phase 一个 phase 替换”，要做“兼容升级”

你问：“如果 Phase 0–7 works well，我有必要再一个 phase 一个 phase 替换成更完整版本么？”

我的建议是：**不要替换，做 shadow upgrade。**

每个 phase 都应该有三层：

```text
v1: MVP implementation
v2: research-grade implementation
v3: autonomous / scalable implementation
```

升级方式不是：

```text
old phase remove → new phase replace
```

而是：

```text
old phase remains committed
new phase runs in shadow mode
compare old vs new
only if new passes gates, promote to default
old phase kept as fallback
```

也就是说，你的系统应该有一个 “phase compatibility contract”。

每个 phase 的升级必须满足：

```text
1. 输入格式兼容
2. 输出 manifest 兼容
3. graph mutation 可回滚
4. 能与旧模块并行跑
5. 能在同一 benchmark 上 A/B
6. promotion 需要通过 verifier stack
7. 失败时不污染 committed graph
```

所以真正的工程组织应该是：

```text
Phase 0 kernel
  ├── phase0_v1_schema
  ├── phase0_v2_schema_typed_payloads
  └── phase0_v3_schema_contract_checker

Phase 1 graph memory
  ├── phase1_v1_jsonl_graph
  ├── phase1_v2_assumption_ppr
  └── phase1_v3_memory_consolidation

Phase 2 residual/verifier
  ├── phase2_v1_audit
  ├── phase2_v2_residual_taxonomy
  └── phase2_v3_falsification_stack
```

这就是你自己说的“不要一次性搭完所有代码，而是替换一个核心模块”。你的系统现在应该把这条方法论原则用在自己身上。

---

# 2. Phase 0–7 的真正展开方式

下面我给一个更细的版本。它不是“重新做 7 个 phase”，而是：

```text
对每个 phase：
  当前 v1 做到了什么
  v2 要补什么
  v3 才算完整
  什么时候升级
  什么时候停止
  具体实验怎么做
```

---

# Phase 0：Assumption Kernel / Schema / Manifest

## 当前 v1 的意义

Phase 0 的任务不是做强 AI，而是回答：

```text
系统里的每一步行动，能不能都变成一个可记录、可验证、可回滚的 assumption？
```

你之前已经把核心定义钉住了：

```text
Assumption = 在上下文 C 下，若采用解释/策略/模型/行动 H，
则未来观测 O 应该以某种方式变化；
若不变化，则产生 residual，并触发修正。
```

这比“假设 = 图上一条边”稳得多，因为它覆盖了问题分类、检索、策略选择、world model 预测、judge 可信度、工具调用、自修改等所有 agent 行为。

repo 里 `schema.py` 已经定义 assumptions、edges、evidence、trial manifests，说明 Phase 0 v1 已经基本有了。([GitHub][1])

## v2 要补的东西

Phase 0 v2 的目标是把 schema 从“能存”升级成“能治理”。

你需要把 assumption 分成更强的类型：

```text
ObjectAssumption
  关于世界对象、事实、规律。
  例：东汉王室短寿可能和遗传病有关。

MethodAssumption
  关于如何解决问题。
  例：复杂系统应先最小原型再增量替换。

RetrievalAssumption
  关于该检索什么记忆。
  例：卖布朗尼应检索低门槛零售/街头摊位案例。

EvaluatorAssumption
  关于什么评价器可信。
  例：LLM judge 的偏好能代表真实用户偏好。

WorldModelAssumption
  关于某个 simulator 的预测可信度。
  例：cheap world model 可预测 candidate 是否值得 live ablation。

AlignmentAssumption
  关于两个过程/策略是否同构/同态/相似。
  例：勒夏特列原理和楞次定律共享负反馈过程 schema。

SelfModificationAssumption
  关于系统改动本身是否会提升 agent。
  例：启用 graph context 会提升 math/science bridge 任务。
```

每个 assumption 都必须有：

```json
{
  "claim": "...",
  "scope": "...",
  "context_conditions": [],
  "expected_effects": [],
  "expected_regressions": [],
  "graph_ops": [],
  "formal_payload_refs": [],
  "verifier_contract": {},
  "rollback_condition": "...",
  "evidence_refs": [],
  "residual_refs": [],
  "status": "candidate | active | rejected | deprecated",
  "confidence": 0.0,
  "metaproductivity": null
}
```

这和 reconstruction 里建议的 AssumptionNode 基本一致：它不只是 claim，还要有 formal form、context conditions、predicted effects、risk predictions、verifier、evidence、residuals、confidence、metaproductivity、status。

## v3 才算完整的地方

Phase 0 v3 要做 **schema contract checker**。

也就是说，每个候选假设在进入系统前必须通过：

```text
1. scope 是否明确？
2. 预期收益是否可测？
3. 预期风险是否写出？
4. verifier 是否存在？
5. rollback condition 是否存在？
6. graph diff 是否可回滚？
7. 是否污染 committed graph？
8. 是否和已有 active assumptions 冲突？
9. 是否只是旧假设改写？
10. 是否缺少 negative control？
```

没有通过 checker 的假设不能进入 candidate overlay，只能进入 `draft_hypothesis_pool`。

## Phase 0 的验收标准

不要用 answer win-rate 验收 Phase 0。Phase 0 的指标应该是：

```text
manifest_coverage:       关键行动中有多少生成 manifest
schema_validity:         manifest 有多少字段完整
rollback_coverage:       graph mutation 是否都有 rollback ref
scope_precision:         assumption 适用条件是否足够明确
verifier_presence:       每条 candidate 是否有 verifier contract
conflict_detection:      是否能发现和旧假设冲突
duplicate_detection:     是否能发现旧假设改写
```

`assumption_bench.py` 现在已经朝这个方向走了，它不把所有进步压成 answer win-rate，而是拆成 explicitness、selection、execution、attribution、transfer、metaproductivity、verifier reliability、world-model quality、harness governance。([GitHub][1])

---

# Phase 1：Assumption Graph Memory

## 当前 v1 的意义

Phase 1 不是普通 RAG，而是：

```text
problem/residual → activated assumption subgraph
```

你之前的 v16 是：

```text
problem → wisdom selection → cases → draft → audit
```

更完整的版本应该是：

```text
problem / residual
  → seed assumption graph
  → activate assumptions / cases / residuals / verifiers
  → give executor structured context
```

这也是 reconstruction 里对 HippoRAG 的吸收：不要把 KG 直接换成范畴论，而要升级成 Assumption Graph；节点不是实体，而是假设、方法、残差、验证器、案例。

外部文献上，HippoRAG 的核心确实是 LLM + knowledge graph + Personalized PageRank，用图式长期记忆支持多跳知识整合；HippoRAG 2 也强调从 RAG 到 memory，增强 associativity 和 sense-making。([arXiv][2])

## v2 要补的东西

Phase 1 v2 要从“能检索相关节点”升级到“能检索正确假设子图”。

Graph retrieval 不能只按相似度排。应该综合：

```text
semantic_relevance
+ graph_centrality
+ confidence
+ domain_match
+ residual_match
+ verifier_availability
+ metaproductivity
- known_regression_risk
- stale_evidence_penalty
- context_pollution_risk
```

所以 retrieval score 可以是：

```text
score(h, q) =
  α * semantic_sim(q, h)
+ β * PPR_score(q_seed, h)
+ γ * confidence(h)
+ δ * ACP(h)
+ ε * residual_match(q, h)
- ρ * regression_risk(h, domain)
- κ * context_cost(h)
```

其中 ACP 是 Assumption Clade Productivity，也就是不要只看这条 assumption 单次赢没赢，而要看它的后代是否能带来长期可用的新假设；这个方向在 reconstruction 里已经被明确为选择函数升级的关键。

## v3 才算完整的地方

Phase 1 v3 要有 **memory consolidation**。

你的 Claude.md 里说得对：长期学习不是把所有经验都写入 memory；经验会变长、变碎、互相污染，所以需要定期压缩、合并、删除、冲突检测和质量门控。

所以 Assumption Graph 要有“睡眠阶段”：

```text
每 N 次 trial:
  1. 找到重复 assumptions
  2. 合并同 family 的 evidence
  3. 提炼 scope condition
  4. 删除低质量/未验证/过期 evidence
  5. 发现互相冲突的 active assumptions
  6. 把局部经验压缩成 method refinement
  7. 给每个 family 更新 ACP
```

这一步非常重要，因为没有 memory consolidation，Assumption Graph 会变成“错误经验垃圾场”。

## Phase 1 的验收标准

```text
retrieval_precision:
  取出的 assumption 是否真的 relevant？

retrieval_transfer:
  一个 domain 学到的 assumption 是否能帮另一个 domain？

negative_transfer_rate:
  graph context 是否伤害某些 domain？

context_efficiency:
  每 token 带来的收益是多少？

assumption_activation_accuracy:
  该激活的策略有没有激活？

residual_retrieval_accuracy:
  失败后能否找回相似 residual？
```

你的 repo README 里已经有很好的警告：graph context 在 software engineering 上出现过 negative transfer，所以默认 gated off；这说明 Phase 1 的目标不是“更多图上下文”，而是“条件化地注入正确图上下文”。([GitHub][1])

---

# Phase 2：Residual Analyzer + Verifier Stack

## 当前 v1 的意义

Phase 2 的核心不是“判断答案对错”，而是回答：

```text
失败说明什么？
```

这一步非常关键，因为如果你把所有失败都当成“缺新假设”，系统会疯狂发明垃圾 wisdom。

你需要至少区分：

```text
ExecutionLapse:
  假设是对的，但 agent 没有真正执行它。

Optimization:
  假设方向对，但执行方式不够好。

AssumptionDefect:
  假设本身错了、缺条件、过度泛化。

Discovery:
  现有 KB 缺少新假设。

EvaluatorDefect:
  judge/verifier 不可信。

RetrievalDefect:
  检索错了，或者注入了干扰上下文。

WorldModelDefect:
  simulator 预测错了。
```

这和 reconstruction 里从 EmbodiSkill 借来的 residual taxonomy 一致：失败不一定是 skill/assumption 错，也可能是 executor 没 follow valid skill。

## v2 要补的东西

Phase 2 v2 要把 verifier 从 single judge 升级成分层协议：

```text
V0: schema / scope / duplicate / conflict check
V1: cheap self-check or programmatic test
V2: world-model predicted value/risk
V3: matched ablation
V4: placebo / length-matched context control
V5: cross-judge / cross-solver
V6: fresh heldout
V7: human / objective benchmark
```

POPPER 值得借鉴，因为它不是让 LLM judge 直接判断假设，而是让 agent 设计并执行 falsification experiments，并用 sequential testing 控制 Type-I error；但你的系统要把 POPPER 从“自然语言科学假设验证”扩展到 retrieval、strategy、evaluator、world model、自修改等 agentic assumptions。([arXiv][3])

## v3 才算完整的地方

Phase 2 v3 要有 **verifier synthesis**。

现在你可能是人为指定 verifier。完整系统应该能对 candidate assumption 自动生成：

```text
1. positive tests
2. negative controls
3. placebo controls
4. regression tests
5. minimal falsification case
6. scope-boundary tests
7. fresh-distribution tests
```

例如一个新 assumption：

```text
“工程调试任务中，增量替换优于一次性重写。”
```

系统应该自动生成：

```text
positive:
  模块边界清晰、baseline 可运行、目标模块可隔离的任务。

negative:
  强耦合、多模块必须同时改变的任务。

placebo:
  给同样长度但无控制变量逻辑的提示。

regression:
  简单一次性任务，检查增量策略是否过度拖慢。

fresh:
  新代码库/新任务类型。
```

## Phase 2 的验收标准

```text
residual_classification_accuracy
false_positive_rate_of_acceptance
regression_detection_recall
placebo_sensitivity
cross_judge_stability
fresh_split_generalization
falsification_power
```

注意：不要让 answer win-rate 掩盖 verifier 的失败。reconstruction 里已经明确警告，评价必须拆模块；cheap world model prompt 和直接让 LLM 发明 new prior 都可能在小样本里看似有效但实际失效。

---

# Phase 3：World Model / Simulator

## 当前 v1 的意义

你现在的 world model 主要是：

```text
cheap verifier / budget gate
```

它预测：

```text
proposal acceptance probability
regression risk
next action: ablate / collect evidence / repair / reject
low-benefit tie
```

gap 文档说得很清楚：当前 world model 不是替代 judge，也不是替代 final answer，而是为了减少弱 descendant 的验证成本；`pre_live_tie_screen_20260609` 已经显示 live call 从 7 降到 3，live call reduction 57.14%，accepted positive blocked 为 0。

这已经很有价值，但不是最终目标。

## v2 要补的东西

Phase 3 v2 要把 world model 从“是否值得测”升级成“预测执行轨迹”。

也就是说，它要从：

```text
candidate h → accept_prob / regression_prob
```

升级为：

```text
state S_t + action A_t → predicted next state S_{t+1}
```

其中 state 不是物理世界，而是 agent world：

```json
{
  "task_features": {},
  "active_assumption_subgraph": [],
  "retrieved_cases": [],
  "residual_state": [],
  "verifier_state": {},
  "budget_state": {},
  "recent_failures": [],
  "candidate_overlay": {}
}
```

action 是：

```text
add_assumption
add_alignment
run_ablation
mask_edge
collect_evidence
repair_scope
reject_candidate
promote_candidate
switch_strategy
change_retrieval_policy
```

world model 输出：

```text
predicted_score_delta
predicted_regression_risk
predicted_residual_type
predicted_cost
predicted_information_gain
predicted_transfer_potential
predicted_graph_pollution_risk
```

这和 reconstruction 里说的“先用 v16/v20 expensive simulator 收集数据，再蒸馏 cheap predictor”一致。目标不是生成最终答案，而是预测这条 assumption 会不会帮、最可能失败在哪里、是否值得真实执行。

## v3 才算完整的地方

Phase 3 v3 才是你说的“奇异博士”版本：

```text
multi-step rollout over assumption graph futures
```

也就是：

```text
当前 residual cluster
  → 生成 10 个 candidate hypotheses
  → world model rollout 每个 candidate 的 3-step 后果
  → 预测哪些 branch 会产生 productive descendants
  → 只 live test 高信息增益分支
```

这时 world model 就不只是省钱，而是 search control。

AEvo 的新方向也可以并入这里：它把 agentic evolution 看成 interactive environment，让 meta-agent 编辑控制未来演化的 procedure / context，而不是只提出下一个 candidate；这和你要做的“让调度假设生成过程本身也成为可编辑假设”非常贴近。([arXiv][4])

## Phase 3 的验收标准

```text
accept_AUROC
accept_Brier
regression_AUROC
failure_type_F1
expected_value_calibration
cost_saved
true_positive_block_rate
multi_step_rollout_accuracy
information_gain_correlation
```

其中最重要的是：

```text
cost_saved high
true_positive_block_rate low
regression_recall high
calibration good
```

如果 world model 省了很多钱但挡掉了真正有价值的新假设，那它是失败的。

---

# Phase 4：Hypothesis Generator

## 当前 v1 的意义

你现在的 generator 已经能从 residual、trace、graph state 生成 proposal，但 gap 文档也指出它仍然偏 template-driven / repair-oriented，需要更强的 residual clustering、LLM synthesis、multi-trajectory search，才能发现真正新的 hypothesis families，而不是局部 patch。

所以 Phase 4 的目标不是“生成更多假设”，而是：

```text
从系统性 residual 中生成更高层的新 assumption family
```

## v2 要补的东西

Phase 4 v2 要改成：

```text
residual-driven hypothesis generation
```

流程：

```text
1. 收集最近 N 个失败 trials
2. 过滤掉 ExecutionLapse
3. 聚类 AssumptionDefect / Discovery / EvaluatorDefect / WorldModelDefect
4. 找出 systematic residual cluster
5. 让 LLM 解释共同缺口
6. 生成 candidate assumption family
7. 查重 / 查冲突 / 查 scope
8. 生成 verifier contract
9. 进入 candidate overlay
10. world model screen
11. fresh ablation
12. accept/reject/defer
```

你 reconstruction 里已经写过：不要再 “LLM，请发明一条新 wisdom”；要从最近 500 个 trials 中筛出已有 assumption 激活但仍失败的样本，做 residual embedding cluster，再生成 candidate assumption，通过 verifier stack 才入库。

## v3 才算完整的地方

Phase 4 v3 要能生成多层假设：

```text
Object-level:
  关于具体世界的假设。

Method-level:
  关于如何解决问题的假设。

Evaluator-level:
  关于评价器是否可信的假设。

Memory-level:
  关于什么应该被检索/写入/删除的假设。

World-model-level:
  关于 simulator 何时可信的假设。

Meta-evolution-level:
  关于未来假设生成过程如何调整的假设。
```

最终系统不能只是生成“新方法”，还要能生成：

```text
“当前 verifier 太弱。”
“当前 graph retrieval 在 SE domain 有 negative transfer。”
“当前 world model 在 regression risk 上低估。”
“当前 residual cluster 其实来自 judge 偏好，而不是 assumption 错误。”
```

这才是 “everything is an assumption”。

## Phase 4 的验收标准

```text
novel_family_rate
duplicate_rate
conflict_rate
fresh_validation_success_rate
cross_domain_transfer_rate
descendant_productivity
false_discovery_rate
residual_explained_fraction
```

尤其要看：

```text
新 assumption 是否解释了旧 residual，同时不破坏旧成功案例？
```

这和科学史里的范式转换非常像：新理论不是只解释 anomalies，还必须继承旧理论已解释的成功区域。

---

# Phase 5：Metaproductivity Selector / Philosophy Scheduler

## 当前 v1 的意义

你最核心的博士命题之一在这里：

```text
AI 不是重新发明所有哲学原则，
而是把人类已有哲学方法论作为可调用策略，
然后学习什么时候调用哪条。
```

Claude.md 里已经说得很对：问题不是 LLM 不知道“控制变量法”，而是它不会主动把这些原则当作操作指令执行；所以你要做的不是 Cyc 式穷举规则，而是建一个调度层，把隐式知识转成显式策略选择。

## v2 要补的东西

Phase 5 v2 要建立 **Philosophy Strategy Library**，但不要一下子追求“所有哲学”。

先做 20–30 条核心方法层假设：

```text
controlled intervention / 控制变量
divide and conquer / 分而治之
abduction / 溯因
deduction / 演绎
induction / 归纳
analogy / 类比
reductio / 归谬
proof by contradiction / 反证
Occam / 简洁性
Bayesian update / 先验-证据-后验
minimal prototype / 最小可运行原型
incremental replacement / 增量替换
counterexample-guided refinement / 反例驱动修正
boundary case analysis / 边界条件分析
negative control / 负对照
model comparison / 模型比较
error decomposition / 误差分解
invariant seeking / 寻找不变量
causal intervention / 因果干预
feedback stabilization / 负反馈稳定
```

每条策略不是一段 wisdom，而是一个 assumption family：

```json
{
  "name": "incremental_replacement",
  "parent": "controlled_intervention",
  "claim": "When a system has a working baseline and isolatable modules, replacing one component at a time is more likely to converge than rewriting all components at once.",
  "scope_conditions": [
    "working baseline exists",
    "module boundary exists",
    "new component can be tested independently"
  ],
  "failure_conditions": [
    "components are strongly coupled",
    "global redesign is required",
    "interfaces are wrong"
  ],
  "canonical_cases": [],
  "negative_cases": [],
  "verifier": {},
  "descendants": []
}
```

## v3 才算完整的地方

Phase 5 v3 是 RL / bandit / Bayesian scheduler。

调度器面对任务时选择：

```text
which strategy family?
which specific descendant?
which verifier?
which world model?
how much budget?
```

可以先不用复杂 RL，先用 contextual bandit：

```text
context = task features + residual features + graph state
action = strategy family
reward = solve success + residual reduction + cost penalty + regression penalty + descendant productivity
```

然后升级成 model-based RL：

```text
world model predicts action outcomes
selector optimizes expected utility + ACP - cost - risk
```

这和你现在的 metaproductivity selector 对齐：一个方法不只是因为单次 A/B 赢才有价值，也可能因为它能产生高质量后代。repo 里 `selector.py` 已经有 HGM-inspired metaproductivity-aware scoring。([GitHub][1])

## Phase 5 的验收标准

```text
strategy_selection_accuracy_against_experts
success_rate_improvement
time_to_solution_reduction
cross_domain_transfer
method_family_ACP
strategy_boundary_learning
negative_transfer_reduction
```

最关键的实验不是：

```text
LLM 是否能解释控制变量法？
```

而是：

```text
LLM agent 是否能在新任务中主动选择控制变量法，并因此更快收敛？
```

---

# Phase 6：Formal Alignment Layer / Category + Information Geometry

## 当前 v1 的意义

这一层不是总框架。它是插件。

它处理的是：

```text
已经被形式化的 assumption / process / strategy
```

而不是所有假设。你之前已经把范畴论 + 信息几何从“统一所有假设”降级成 Formal Alignment Layer，这是正确的；reconstruction 也明确说它不能覆盖“卖布朗尼”“控制变量法”“该不该先做最小原型”这些概念性/方法层假设。

repo 目前也已经把它做成 bounded structural morphism layer：不是 category-theory theorem prover，而是检查 typed diagrams、object roles、morphism roles、composition hints、invariants、negative controls。([GitHub][1])

## v2 要补的东西

Phase 6 v2 不应该先追求大而全，而应该做 **ProcessModel + AlignmentHypothesis**。

比如勒夏特列和楞次：

```text
LeChatelier_Process:
  perturbation
  response
  state variables
  potential / equilibrium constraint
  local counteraction invariant
  failure cases

Lenz_Process:
  flux change
  induced current
  induced magnetic field
  sign opposition invariant
  failure cases

AlignmentHypothesis:
  maps perturbation → flux change
  maps compensatory response → induced opposing field
  preserves negative-feedback schema
  does not preserve thermodynamic formalism
  does not preserve electromagnetic equations
```

这时范畴论检查的是：

```text
这个 mapping 是否保留关键 invariants？
```

信息几何 / metric layer 检查的是：

```text
两个过程在映射后的 trajectory distribution 有多近？
```

## v3 才算完整的地方

Phase 6 v3 是 formal transfer evaluator：

```text
formal alignment quality 是否预测 downstream transfer success？
```

也就是说，不要只证明两个结构“看起来同构”。要验证：

```text
当 formal alignment score 更高时，
agent 是否更能把 A domain 的方法迁移到 B domain？
```

repo 里现在已经有 row KL、total variation、Frobenius distance、Blackwell-style dominance proxy、formal-equivalence signature、formal-transfer-eval 等方向，说明这一层已经开始进入可测量状态。([GitHub][1])

## Phase 6 的验收标准

```text
alignment_precision_against_expert
negative_control_rejection
formal_equivalence_dedup_accuracy
formal_score_transfer_correlation
top1_formal_mapping_hit_rate
unsafe_mapping_block_rate
```

这一层的底线：

```text
宁可少发现，也不能乱合并。
```

因为 formal alignment 一旦错，会把错误的跨域迁移注入 Assumption Graph，污染长期记忆。

---

# Phase 7：Autonomous Daemon / Harness / Benchmark

## 当前 v1 的意义

Phase 7 不是“让它自动跑起来”这么简单，而是：

```text
让长期自我演化过程可控、可审计、可回滚。
```

gap 文档指出 daemon 目前仍是 bounded/gated；完整版本需要 persistent scheduling、parallel execution、cost/rate-limit control、failure recovery、continuous learning。

这和 2026 年 AI Harness Engineering 的观点非常一致：自主软件 agent 的能力不只是模型能力，而是 model–harness–environment 系统能力；harness 要负责 task specification、context selection、tool access、project memory、task state、observability、failure attribution、verification、permissions、entropy auditing、intervention recording 等。([arXiv][5])

## v2 要补的东西

Phase 7 v2 要做 harness governance：

```text
1. 所有 graph mutation 默认 dry-run
2. 只有通过 acceptance gate 才能 apply
3. apply 数量受预算和权限限制
4. 每次 apply 都有 rollback plan
5. evaluator 不可被 candidate prompt 污染
6. world model 只能建议，不能直接 promote
7. daemon 失败后能恢复 frontier
8. 每个 run 生成 episode package
```

repo 里 `evolution_context.py` 已经朝这个方向走：把 self-evolution procedure 本身当成 harness assumption，带 task、context、observability、verification、permission、rollback、intervention-recording responsibilities。([GitHub][1])

## v3 才算完整的地方

Phase 7 v3 是 frozen benchmark + long-running evaluation。

你需要两个 benchmark：

```text
AssumptionBench:
  测 assumption lifecycle 能力。

DownstreamBench:
  测最终任务能力是否真的提升。
```

AssumptionBench 测：

```text
explicitness
selection
execution
residual attribution
transfer
metaproductivity
verifier reliability
world-model calibration
harness governance
```

DownstreamBench 比较：

```text
ordinary RAG
HippoRAG-style graph retrieval
v16/v20 case reflection
no world model
no recursive runner
no formal layer
full Assumption OS
```

gap 文档也明确说，paper-level claim 需要 frozen end-to-end benchmark on unseen tasks，并和 HippoRAG、ordinary RAG、one-shot self-improve、no world model、no recursive runner 对比。

## Phase 7 的验收标准

```text
long_run_stability
graph_pollution_rate
rollback_success_rate
cost_per_accepted_assumption
accepted_assumption_survival_rate
downstream_win_rate_on_unseen
capability_score_improvement
daemon_recovery_success
evaluator_integrity
```

---

# 3. 更完整的全局路线：不是 7 个 phase，而是 4 条主线并行升级

你之所以觉得 Phase 0–7 “好像简单”，是因为它们被写成线性 phase 了。但真正复杂度不是线性 7 步，而是四条主线互相咬合：

```text
A. Representation:
   假设如何表示、落图、形式化、回滚？

B. Search:
   如何从 residual 中生成多个候选，并用 world model 控制分支爆炸？

C. Validation:
   如何证伪、消融、控制 false positive、避免 judge/style/exemplar bias？

D. Learning:
   如何把经验蒸馏成长期方法层知识，而不是污染 memory？
```

对应你的系统：

```text
Representation:
  Manifest + Assumption Graph + ProcessModel + AlignmentHypothesis

Search:
  Residual Clusterer + Generator + Trajectory Search + World Model

Validation:
  Verifier Stack + POPPER-style falsification + Fresh Ablation + Negative Controls

Learning:
  Memory Consolidation + ACP + Philosophy Scheduler + World Model Calibration
```

这四条主线都做到 v2，系统才从“能跑”变成“可信”。
这四条主线都做到 v3，才接近你说的“碰到已有哲学框架解决不了的问题时，提出新哲学/新领域假设并自我验证”。

---

# 4. “Phase 0–7 works well” 后应该怎么做？

我的建议是做一个 **Capability Ladder**，而不是继续叫 Phase 8、Phase 9。

每个能力都有 5 级：

```text
L0: absent
L1: hand-coded / template
L2: logged + auditable
L3: learned / calibrated
L4: self-improving but gated
L5: autonomous but rollback-safe
```

然后把当前系统打分：

```text
Assumption Manifest:
  L2-L3

Assumption Graph:
  L2-L3

Graph Retrieval:
  L2，部分 domain 到 L3

Residual Attribution:
  L2

World Model:
  L2，cheap gate；还不是 L3 simulator

Verifier Stack:
  L2

Generator:
  L1-L2，偏 repair/template

Formal Alignment:
  L2 bounded checker

Daemon:
  L2 bounded/gated

Memory Consolidation:
  L1-L2

Philosophy Scheduler:
  L1，概念清楚但需要实验化

Benchmark:
  L1-L2，AssumptionBench 有雏形，frozen downstream 还需要加强
```

这样你不会被“Phase 0–7 做完了下一步是什么”困住。下一步就是把最短板从 L1/L2 拉到 L3。

我认为当前最短板排序是：

```text
1. Generator strength
2. World model as prospective simulator
3. Frozen downstream benchmark
4. Memory consolidation / graph pollution control
5. Philosophy scheduler experiments
6. Formal alignment transfer evaluation
7. Fully autonomous daemon
```

这和 gap 文档列出的缺口基本一致：generator、world model、daemon、downstream benchmark、formal boundary、observability。

---

# 5. 我建议你现在先不要全做，先做一个“v2 slice”

最强的下一步不是扩展所有 phase，而是做一个完整 vertical slice：

```text
Residual cluster
  → generate 5 competing hypotheses
  → put each into candidate overlay
  → world model predicts value/risk/failure type
  → verifier stack selects 2 for fresh ablation
  → run ablation
  → update graph + world model calibration
  → measure downstream and AssumptionBench deltas
  → repeat 3–5 generations
```

这正好是 gap 文档给出的 near-term priority：让 generator 和 world model 变成 prospective，围绕新 residual cluster 生成多个竞争假设、分类、screen、fresh ablation、记录 accept/reject、更新 graph 和 world-model calibration，并在 frozen heldout benchmark 上重复 3–5 代。

这个 slice 足够小，符合增量替换；又足够完整，可以证明你的系统不是普通 prompt engineering。

---

# 6. 我会把下一步具体改成这个实验

## 实验名

```text
Prospective Recursive Assumption Evolution on Frozen Residual Clusters
```

## 核心问题

```text
给定一个 frozen residual cluster，
Assumption OS 是否能比 one-shot self-improve / ordinary RAG / graph retrieval only
更有效地产生、筛选、验证并保留有用假设？
```

## 固定输入

```text
1. frozen task set
2. frozen baseline answers
3. frozen judge protocol
4. frozen existing Assumption Graph
5. frozen residual clusters
```

## 每一代循环

```text
for generation in 1..5:
    cluster = select_residual_cluster()
    candidates = generator.propose_k(cluster, k=5~10)
    overlays = build_candidate_overlays(candidates)
    wm_predictions = world_model.score(overlays)
    selected = select_top_by(EV + info_gain - risk - cost)
    ablation_results = run_fresh_ablation(selected)
    verifier_results = verifier_stack.decide(ablation_results)
    graph.update_only_accepted(verifier_results)
    world_model.calibrate(wm_predictions, ablation_results)
    assumption_bench.score()
```

## 对照组

```text
A. no evolution
B. one-shot LLM new wisdom
C. graph retrieval only
D. generator without world model
E. generator + world model but no verifier stack
F. full recursive assumption loop
```

## 成功条件

```text
1. full loop accepted assumptions survive fresh validation
2. full loop saves live calls without blocking true positives
3. residual cluster explained fraction increases
4. downstream heldout score improves
5. graph pollution remains low
6. world model calibration improves across generations
```

这就是从 prototype 到 paper claim 的最短路。

---

# 7. 关于“完整版本”的正确心态

你的问题确实很大，因为它的最终目标接近：

```text
能利用人类已有哲学；
能把哲学策略变成可调用操作；
能从失败中学习适用边界；
能在已有策略不足时提出新假设；
能验证新假设不和旧体系冲突；
能长期运行而不污染自己。
```

但它不能靠一次性大设计完成。它应该像操作系统一样演化：

```text
kernel first
capability modules second
governance third
autonomy last
```

所以：

```text
Phase 0–7 = kernel v1
现在要做 = v2 vertical slice
后面再做 = 每个能力升 L3/L4
最终目标 = autonomous but rollback-safe Hypothesis OS
```

你现在最不应该做的是“写一个更宏大的 Phase 0–20”。
最应该做的是拿一个 frozen residual cluster，把这条链跑通 3–5 代：

```text
systematic residual
  → competing hypotheses
  → world-model prospective screening
  → fresh verifier
  → graph update
  → calibration update
  → next-generation descendants
```

如果这条链成立，你就已经从“我有一个递归式自我论证的想法”进入了“我有一个可实验验证的递归假设操作系统”。

[1]: https://raw.githubusercontent.com/erzhu419/Assumption-Agent/main/assumption_os/README.md "raw.githubusercontent.com"
[2]: https://arxiv.org/abs/2405.14831?utm_source=chatgpt.com "HippoRAG: Neurobiologically Inspired Long-Term Memory ..."
[3]: https://arxiv.org/abs/2502.09858?utm_source=chatgpt.com "Automated Hypothesis Validation with Agentic Sequential Falsifications"
[4]: https://arxiv.org/abs/2605.13821?utm_source=chatgpt.com "Harnessing Agentic Evolution"
[5]: https://arxiv.org/abs/2605.13357?utm_source=chatgpt.com "AI Harness Engineering: A Runtime Substrate for Foundation-Model Software Agents"
