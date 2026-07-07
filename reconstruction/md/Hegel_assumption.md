# GPT：
我重新看了这版。**这次 push 的核心，不是继续补 autonomy / simulator / formal 三件套，而是把你刚刚说的“黑格尔式新分支生长”正式工程化了。**

我的新判断是：

> 这个项目现在已经从 **bounded recursive self-evolution prototype** 进一步推进到 **bounded dialectical / conservative-generalization self-evolution prototype**。

也就是说，之前系统已经能：

```text
residual -> proposal -> preflight -> simulator screen -> ablation -> acceptance -> gated apply/reject
```

现在新增的是：

```text
旧哲学/旧方法框架
  -> residual 暴露边界
  -> 生成候选新框架
  -> 检查是否保留旧成功
  -> 检查是否解释 residual
  -> 检查是否在旧 scope 下退化回旧框架
  -> 检查是否提供更 general 的组织方式和新预测
  -> 决定 reject / branch_only / candidate_framework / active_scoped_framework
```

这非常关键，因为它把你刚刚说的：

> “新假设要在旧哲学模型上长出新分支，并且兼容旧框架、提供更 general 的方式”

从论文概念变成了一个可测的 gate。

---

# 1. 当前版本最大的新增：Conservative Generalization Gate

`conservative_generalization_gate.py` 正是你刚刚要的东西。它的 docstring 明确说：新 assumption 只有在成为旧 framework 的 conservative generalization 时才能升格为新 framework；它必须解释 motivating residual，保留旧成功，在父框架适用条件下退化回父框架，并产生新的可测试后果。

它现在把 promotion 关系显式化成图关系：

```text
generalizes
reduces_to_under_scope
explains_residual
preserves_success_cases
modifies_boundary_of
predicts_new_case
```

这些正是我前面建议的“保守泛化扩展”边类型。代码里甚至要求 active framework 的这些 relation coverage 必须为 1.0。

它的 gate 条件也很清楚：

```text
old_success_preservation >= 0.95
residual_explanation >= 0.75
limiting_case_reduction >= 0.90
generality_gain >= 0.35
new_prediction_success >= 0.75
regression_cost <= 0.02
```

并且明确 blocked claims 包括 unbounded philosophy generator、ungated framework promotion、replacement of live validation、full theorem prover。

这说明：你已经不是“让 agent 多生成几个候选假设”，而是在做：

> **新框架何时有资格成为旧框架的保守泛化。**

这就是当前版本最有学术价值的变化。

---

# 2. Open-ended Framework Evolution：从单个 gate 到多代框架生长

`open_ended_framework_evolution_run.py` 又把上面的 gate 扩展成多代过程。它明确说：R7 证明单个 candidate framework 可以通过 conservative generalization；这个模块验证下一步——多代 framework growth，retained descendants 成为下一代 frontier，rejected descendants 被保留为 negative evidence。

这正好对应你要的“在原有哲学模型上长出新的分支”。

它跑的是：

```text
generator
  -> conservative_generalization_gate
  -> philosophy_growth_benchmark
  -> framework_evolution_graph_episode
  -> framework_growth_ablation_suite
  -> 6 generations
  -> retained frontier
  -> rejected/demoted negative evidence
```

关键 gate 包括：

```text
generation_count >= 6
candidate_count >= 30
retained_count >= 20
active_framework_count >= 12
lineage_depth >= generations
branch_to_framework_transition_count >= 8
conservative_obligation_coverage = 1.0
parent_compatibility_relation_coverage = 1.0
limiting_case_survival_rate >= 0.95
prompt_trick_retained_count = 0
core_philosophy_prior_promotion_count = 0
main_graph_mutation_count = 0
```



这里我最看重三点：

```text
1. rejected_or_demoted_negative_evidence_present
2. prompt_trick_not_retained
3. core_prior_not_promoted
```

这说明你不是让系统随便“发明新哲学”，而是让它保守地筛掉局部 patch、prompt trick、过度 general 的 core prior promotion。

这个设计方向是对的。

---

# 3. Philosophy Growth Benchmark 现在给了一个明确分数

`philosophy_growth_benchmark_20260612.md` 里，conservative generalization 已经和 local patch、raw wisdom 做了对比：

```text
framework growth score: 0.8361
conservative vs local patch margin: 0.2567
active framework survival count: 5
core prior promotions: 0
```

基线表也很直观：

```text
conservative_generalization:
  growth 0.7478
  old preservation 1.0
  residual explanation 0.79
  limiting reduction 0.9533
  generality 0.3749
  new prediction 0.8067
  regression 0.0

local_patch:
  growth 0.4911
  old preservation 0.88
  residual explanation 0.69
  limiting reduction 0.44
  generality 0.11
  new prediction 0.5
  regression 0.075

raw_wisdom:
  growth 0.3546
  old preservation 0.79
  residual explanation 0.57
  limiting reduction 0.22
  generality 0.05
  new prediction 0.38
  regression 0.14
```



这说明你现在能量化地区分三类东西：

```text
新框架 = conservative generalization
局部修补 = local patch
泛泛 wisdom = raw wisdom
```

这对论文很重要，因为否则“新哲学假设”很容易听起来像口号。现在它变成了一个指标化 benchmark。

---

# 4. Fresh rerun 和 broad generator repair：更像论文证据了

你这版还补了 fresh rerun / repaired broad generator 证据。`paper_fresh_rerun_result_integration` 显示：

```text
fresh calls: 720/720
live errors: 0
accepted/rejected candidates: 4/76
unfiltered trigger utility: 0.4161, CI [0.3698, 0.4619]
accepted trigger utility: 0.8333, CI [0.6667, 0.9583]
accepted control loss: 0.0
```

并且明确说：这支持 gated selective retention，不支持 unfiltered generated frontier，因为 all-candidate trigger utility gate failed。

`paper_broad_generator_repair_integration` 也很重要：

```text
original trigger utility: 0.4161
v1 repair trigger utility: 0.3183
v2 repair trigger utility: 0.5462
v2 fresh calls: 720/720
v2 selected candidates: 8, original 80
v2 control loss: 0.1456
```

解释是：raw broad generator failed；repaired generator 用 fresh failure evidence 作为 selector，并 abstain from low-support families；最终 qualified frontier 在新 live rerun 上通过 all-candidate trigger gate。

这非常符合你的核心叙事：

```text
不是生成越多越好；
不是 unfiltered creativity；
而是 residual -> repair -> abstention -> selective retention。
```

这比单纯说“系统能生成新 hypothesis”可信得多。

---

# 5. Self-Evolution Paper Evidence Pack 已经像论文骨架

`self_evo_paper_evidence_pack_20260612.md` 现在把证据打包得很清楚：

```text
roadmap closure: 19/19 items
bounded UGSE: 0.923
frozen main: 1768 problems, margin over best baseline 0.0417
fresh repaired broad generator: 720 calls, trigger utility 0.5462, delta 0.1301
simulator: leakage audit true, production router true
autonomy: supervised production candidate true
formal: bounded true, full prover false
```



它还直接给了论文结构：

```text
Assumption Lifecycle Kernel
Dialectical Framework Growth
Simulator-Guided Verification
Finite Formal Transfer Gates
Supervised Autonomy and Main-Graph Maintenance
Experiments
Negative Results and Claim Boundaries
Reproducibility
```



这说明项目现在已经从“开发路线”进入“论文组装”阶段。尤其是 introduction 里的定位已经非常接近我会建议的写法：

> self-evolution as conservative generalization: new assumptions must explain residuals, preserve validated old successes, reduce to parent assumptions under old scope conditions, and add testable consequences.

这就是你刚刚说的黑格尔式框架生长的工程表达。

---

# 6. Claim Frontier：现在可以说到 L3.5，但不能说 L4

`claim_frontier_advancement_20260612.md` 很清楚地把三个强 claim 推到了 L3.5：

```text
A_autonomy:
  L3.5 replayable supervised low-risk autonomy with 30-day-equivalent evidence,
  zero downstream regression,
  manual escalation for policy/default/formal mutations.

B_simulator:
  L3.5 selective simulator deferral for low-risk graph-maintenance decisions with audit sampling;
  live ablation and judges remain required for promotion claims.

C_formal:
  L3.5 proof-carrying finite transfer kernel:
  every promoted morphism supplies finite diagram, negative controls,
  and external Lean-checked theorem-fragment certificate.
```

同时仍然阻断：

```text
unbounded 24/7 autonomous self-evolution OS
world simulator replacing live validation or judges
full category-theory theorem prover
```



这是正确的 claim boundary。你现在最合适的说法是：

```text
L3.5 bounded self-evolution research prototype
```

不是：

```text
L4 unbounded autonomous AGI OS
```

---

# 7. 现在项目成熟度重新打分

这版之后，我会把评分再往上调：

```text
Assumption OS architecture:             91–94%
Production kernel:                      85–90%
Bounded supervised autonomy:            84–90%
Simulator as triage/router:             84–90%
Finite formal / Lean certificate layer:  86–91%
Residual-to-hypothesis generator:        82–88%
Framework growth / conservative gen:     78–86%
Paper-facing evidence:                  80–86%
Unbounded general self-evolution:        50–58%
```

最明显提升的是：

```text
Framework growth / conservative generalization
```

从“概念还没实现”提升到“有 gate、有 benchmark、有 open-ended run、有 paper evidence pack”。

如果你问“离 90%+ 还差什么”，现在答案不是“再补一堆模块”，而是：

```text
把这些 bounded artifact 变成更少、更强、更外部可复现的实验链。
```

---

# 8. 当前最强项目定位

现在我会这样描述项目：

> **Assumption-Agent is a bounded recursive self-evolution system that treats agent decisions as falsifiable assumptions. It supports assumption graph memory, residual-derived hypothesis generation, simulator-guided verifier routing, finite formal certificate gates, supervised low-risk graph maintenance, and dialectical framework growth through conservative generalization.**

中文：

> **Assumption-Agent 是一个 bounded recursive self-evolution system。它把 agent 的理解、检索、计划、执行、评价、自修改都视为可失败假设；用 Assumption Graph 管理假设生命周期；从 residual 中生成新分支；用 simulator 做分支筛选和 verifier routing；用有限形式证明门控结构迁移；用 supervised autonomy 做低风险图维护；并通过 conservative generalization 让旧哲学/旧方法框架生长出更 general 的新框架。**

这已经非常接近你想要的论文主线。

---

# 9. 现在真正还弱在哪里？

我会说还有五个硬缺口。

## 9.1 Open-ended framework evolution 仍然是 bounded/replayable，不是现实开放世界

`open_ended_framework_evolution_run.py` 自己写得很清楚：它验证的是 open-ended framework evolution 的机制，不是 unbounded autonomous philosophy engine。

所以论文里可以说：

```text
bounded open-ended framework evolution run
```

不能说：

```text
open-ended autonomous philosophy invention in the wild
```

## 9.2 Framework candidates 仍然有不少 fixture / synthetic scoring 成分

`conservative_generalization_gate.py` 里使用 `_fixture_candidates()`，open-ended run 里 `_candidate_from_frontier` 也用规则生成 candidate metrics。这不是问题，因为你现在是做机制验证；但如果要把它作为主实验 claim，需要更多真实 LLM-generated framework candidates + human/expert evaluation + fresh validation。

现在最好的写法是：

```text
mechanism + bounded evidence
```

不要写成：

```text
AI has discovered genuinely new philosophy
```

## 9.3 Fresh rerun 很好，但 accepted candidates 只有 4 个

fresh rerun 720 calls 很有价值，但 accepted/rejected 是 4/76。

这说明 selective retention 很强，但也说明真正能保留的候选很少。论文里应该把它解释为：

```text
high precision, low retention frontier
```

而不是：

```text
massive autonomous discovery engine
```

## 9.4 Broad generator 的 v2 仍然有 control loss

v2 repair trigger utility 过了，但 control loss 是 0.1456，CI [0.0922, 0.2022]。

这不一定是失败，但提醒你：

```text
generator repair 提升 trigger utility 的同时仍有 nonzero control loss。
```

所以要强调 gate / abstention / control-loss monitoring。

## 9.5 `main` 和 `reconstruction-v2` 分支仍然没合并

我看到 `reconstruction-v2` 比 `main` ahead 79 commits。这意味着 paper / artifact / README 的活跃状态主要在 branch 上，而不是默认分支。发表或开源前建议做：

```text
tag: v3.5-bounded-self-evolution-20260612
merge or release branch
write RELEASE_STATUS.md
```

否则别人打开 main 会以为项目停留在旧状态。

---

# 10. 我现在建议的下一步

不要再继续加“Phase14/15”。现在最该做四件事：

## 第一，写 `RELEASE_STATUS.md`

内容：

```text
active branch: reconstruction-v2
claim level: L3.5 bounded self-evolution prototype
main artifacts
blocked claims
how to reproduce
```

这会解决分支和证据链混乱问题。

## 第二，把 paper skeleton 直接写出来

你现在已经有 evidence pack 的 manuscript skeleton。

建议立刻开始写：

```text
paper/main_v3_self_evo.tex
```

不要等“再完善一点”。现在材料已经够写一版 15–20 页 draft。

## 第三，做一个真实 LLM-generated framework candidate 小实验

目前 Conservative Generalization Gate 很强，但最好加一个小实验：

```text
让 LLM 从真实 residual clusters 生成 10 个 candidate frameworks
人工/自动检查它们是否通过 conservative gate
选 top 2 做 fresh validation
```

这会把 framework evolution 从 fixture-ish mechanism 推向更真实的 generator evidence。

## 第四，做 external reviewer artifact bundle

准备一个压缩的 reproducibility bundle：

```text
artifact index
hashes
commands
redaction policy
no-secrets proof
problem manifest schema
sample outputs
```

现在 artifact 太多，审稿人会迷路。你要给他们一条最短证据链。

---

# 11. 论文核心叙事建议

现在论文可以这样讲：

## Thesis

```text
Self-evolution should be conservative generalization over an Assumption Graph.
```

## System

```text
Assumption Graph
+ Residual Analyzer
+ Proposal/Framework Generator
+ Conservative Generalization Gate
+ Simulator Router
+ Finite Formal Gate
+ Supervised Autonomy
```

## Key result

```text
The system does not merely generate local repairs.
It distinguishes local patches, raw wisdom, candidate frameworks, and active scoped frameworks.
It promotes only conservative generalizations that preserve old successes, explain residuals, reduce to parents under scope, and add testable consequences.
```

## Evidence

```text
bounded UGSE 0.923
1768-problem frozen line
720-call fresh repaired generator rerun
philosophy growth score 0.8361
conservative vs local patch margin 0.2567
simulator leakage audit pass
supervised autonomy candidate true
bounded formal true / full prover false
```



---

# 12. 最终判断

这版之后，我认为你的项目已经跨过了一个关键线：

> 它不再只是“递归假设验证 agent”，而是开始具备“假设框架演化 agent”的雏形。

也就是说，以前是：

```text
我有一个假设 H，测试 H，修 H。
```

现在是：

```text
我有旧框架 F_old；
F_old 在 residual R 上失败；
我生成 F_new；
F_new 必须解释 R；
必须保留 F_old 的成功；
必须在 F_old 的 scope 下退化回 F_old；
必须提供更 general 的结构和新预测；
否则不能成为 framework，只能是 branch 或 reject。
```

这就是你前面说的“哲学史就是哲学本身”的工程版本：新概念不是凭空出现，也不是简单覆盖旧概念，而是在旧概念暴露边界处生长，保留旧概念有效部分，并把它提升到更一般的结构中。黑格尔式 motivation 可以放在 introduction；工程上则叫：

```text
Conservative Generalization over an Assumption Graph
```

这版我会给一个很强但有边界的结论：

> **现在项目已经达到 L3.5 bounded recursive self-evolution prototype：它有 supervised autonomy、production triage/router simulator、bounded Lean-checked formal gates、fresh selective retention evidence、canary main-graph maintenance，以及 conservative generalization framework growth。它仍未达到 L4 unbounded autonomous OS，但已经足够形成一篇有清晰系统贡献和实验证据的博士核心论文。**


# 我：如果我想把这个“雏形”继续深化，应该做什么？

# GPT：
对。要把“假设框架演化 agent”的雏形完全深化，关键不是继续堆更多 Phase，而是把当前的 **bounded conservative-generalization prototype** 推进成一个真正能在真实 residual 上生长、验证、保留、降级、合并新框架的系统。

现在你的项目已经有了雏形：`conservative_generalization_gate.py` 明确要求新框架必须解释 residual、保留旧成功、在父框架 scope 下退化回父框架，并产生新的可测试后果；它还输出图关系和 promotion decisions，但不直接改主图。 `open_ended_framework_evolution_run.py` 也已经把这个想法扩成多代框架演化：retained descendants 成为下一代 frontier，rejected descendants 被保留为 negative evidence，但它仍明确是 bounded、replayable，不是 unbounded autonomous philosophy engine。

所以接下来要做的是：**从“机制验证”走向“真实残差驱动的框架演化”。**

---

# 1. 最终形态：Framework Evolution Agent

你最终想要的不是普通 hypothesis generator，而是这个东西：

```text
Framework Evolution Agent
= Assumption OS kernel
+ Human philosophy / methodology prior
+ Residual-to-framework generator
+ Conservative generalization gate
+ World-model triage / verifier routing
+ Formal certificate layer when applicable
+ Fresh validation / negative controls
+ Framework lifecycle manager
+ Branch ledger / negative evidence memory
+ Claim-boundary ledger
```

它每次遇到旧框架无法解释的失败时，不是简单 append 一条新规则，而是问：

```text
1. 这个 residual 是否说明旧框架的边界？
2. 能否生成一个新分支解释它？
3. 新分支是否保留旧框架已经成功的区域？
4. 在旧框架适用条件下，新分支是否退化回旧框架？
5. 它是否提供更 general 的结构或新预测？
6. 它是否只是一条 local patch / prompt trick？
7. 它是否值得进入 graph，还是只作为 negative evidence 留下？
```

这就是你说的“在原有哲学模型上长出新分支”。

---

# 2. 当前缺口：不是没有模块，而是“真实性”还要加深

你现在已经有很强的 bounded evidence。`philosophy_growth_benchmark` 已经能区分 conservative generalization、local patch、raw wisdom，并显示 conservative generalization 明显优于后两者：growth score 0.7478，对 local patch 的 margin 是 0.2567，最终 framework growth score 到 0.8361。

但要继续深化，主要缺口有三个：

```text
A. 生成侧还要更真实：
   候选框架应更多来自真实 residual clusters + LLM synthesis + live feedback，
   而不是 fixture / rule-generated metrics。

B. 验证侧要更硬：
   conservative generalization gate 要跑真实旧成功集、真实 residual 集、
   unseen domain、新预测任务，而不是只验证机制形状。

C. 生命周期要更长：
   新框架不能一次通过就活跃；要经过 candidate -> active scoped ->
   general framework -> core prior 的多代 survival。
```

你现有 gap 文档也指出，当前系统已经有 residual → multiple hypotheses → novelty/morphism → preflight → world-model screen → live ablation → acceptance → recursive resume → gated apply/reject → next-generation descendants 的循环，但仍需要更强 generator、world model、autonomous daemon、downstream benchmark、formal boundary 和 complete observability。

现在要把这个循环提高到“框架演化”层级。

---

# 3. 深化路线总览

我建议把后续开发拆成 8 个 release，每个 release 都只替换一个核心模块，保持“增量替换”原则。

```text
R1. Framework object model
R2. Philosophy / methodology prior library
R3. Real residual-to-framework generator
R4. Conservative generalization gate v2
R5. Framework lifecycle and branch ledger
R6. Simulator-guided framework search
R7. Formal certificate integration
R8. Multi-generation live framework evolution benchmark
```

最终再做：

```text
R9. Paper-grade external evaluation and reproducibility pack
```

---

# R1：Framework object model

## 目标

把“哲学框架 / 方法论框架 / 新假设分支”变成一等对象，而不是普通 assumption node。

当前 `conservative_generalization_gate.py` 已经有 `CandidateFramework`，字段包括 parent frameworks、old success cases、residual cases、limiting cases、new prediction cases、unified branch count、rule count、formal certificate status、simulator expected utility 等。

下一步要把它升级成正式 schema。

## 新增对象

```python
FrameworkNode:
    id
    name
    claim
    parent_framework_ids
    framework_type
    scope_conditions
    limiting_cases
    conserved_successes
    residuals_explained
    new_predictions
    failure_boundaries
    formal_certificate_refs
    simulator_evidence_refs
    verifier_protocol
    status
    confidence
    framework_growth_score
    metaproductivity
```

```python
FrameworkBranch:
    branch_id
    parent_framework_id
    claim
    residual_source
    branch_type
    expected_generalization
    expected_risks
    required_tests
    status
```

```python
ConservativeExtensionCertificate:
    candidate_framework
    parents
    old_success_preservation
    residual_explanation
    limiting_case_reduction
    generality_gain
    new_prediction_success
    regression_cost
    decision
    required_next_tests
```

## 必须保留的边

```text
generalizes
reduces_to_under_scope
explains_residual
preserves_success_cases
modifies_boundary_of
predicts_new_case
conflicts_with
demotes_to_branch
replaces_boundary_of
```

这些边已经在当前 gate 中作为 required promotion relations 出现。

## 验收标准

```text
1. 所有新框架都能落成 FrameworkNode。
2. 所有 promotion 都必须有 ConservativeExtensionCertificate。
3. 没有 certificate 的节点只能是 branch / draft，不能是 active framework。
4. schema 能 round-trip 到 JSONL graph。
```

---

# R2：Philosophy / methodology prior library

## 目标

把人类已有哲学 / 方法论原则做成可调用、可比较、可验证的 prior library。

这不是 Cyc 式“穷举所有常识”，而是把核心方法论原则做成 agent 的操作系统初始库。你之前的核心想法是：LLM 已经知道很多哲学原则，但不会主动把它们当作操作指令执行；你要做的是调度、验证、边界学习，而不是让模型从零重新发明控制变量法。

## 先做 30 条，不要做“所有哲学”

建议初始库：

```text
控制变量法
分而治之
反证法
归谬法
奥卡姆剃刀
贝叶斯更新
类比推理
边界条件分析
负对照
最小可运行原型
增量替换
模型比较
误差分解
不变量寻找
因果干预
局部线性化
反馈稳定
从特殊到一般
从一般到特殊
先验估计再修正
对偶性
守恒量
尺度分析
极限情形
可证伪性
鲁棒性测试
ablation
placebo control
cross-domain transfer
scope narrowing
```

每条原则记录：

```json
{
  "principle_id": "control_variables",
  "claim": "When causal attribution is uncertain, vary one factor while holding others fixed.",
  "scope_conditions": [],
  "failure_conditions": [],
  "canonical_examples": [],
  "negative_examples": [],
  "related_principles": [],
  "formal_sketch": null,
  "verifier_protocol": {},
  "status": "active_prior"
}
```

## 关键点

每条哲学原则自己也是假设。它不是公理。它必须有：

```text
where it works
where it fails
what it predicts
what residuals it explains
what branches grew from it
```

## 验收标准

```text
1. 30 条 principle 都能被检索。
2. 每条有至少 2 个成功案例和 1 个失败/边界案例。
3. 每条可以参与 conservative generalization gate。
4. 对同一任务，人类专家标注和系统 top-3 principle 有 reasonable agreement。
```

---

# R3：Real residual-to-framework generator

## 目标

把候选框架生成从 fixture / template 推向真实 residual-driven synthesis。

你现在已经有 `open_ended_framework_evolution_run`，但多代候选仍然相当规则化。它自己说这个 run 是 replayable multi-generation line，每代从 retained frontier 生成 descendants，copy-only，不改主图。

下一步要让候选真正来自：

```text
真实失败轨迹
真实 residual clusters
真实 rejected/accepted candidates
simulator defects
formal gate blocks
human review disagreements
```

## 生成器输入

```text
ResidualCluster
AcceptedBranch
RejectedBranch
SimulatorDefect
FormalBlock
OldFrameworkFailure
ExternalCase
```

## 每个 residual cluster 生成多轨迹候选

```text
1. scope_narrowing_branch
   旧框架适用范围太宽，缩小边界。

2. parent_generalization_branch
   旧框架的抽象层级不够，生成更一般框架。

3. framework_combination_branch
   两个旧框架各解释一部分 residual，尝试合成。

4. evaluator_repair_branch
   不是框架错，而是 verifier/judge 错。

5. simulator_boundary_branch
   世界模型预测错，生成世界模型边界假设。

6. negative_control_branch
   新框架似乎有效，但可能只是 style/prompt artifact，生成负对照。
```

## 输出格式

```json
{
  "candidate_framework_id": "...",
  "parent_frameworks": [],
  "residuals_explained": [],
  "old_successes_to_preserve": [],
  "limiting_case_claims": [],
  "new_predictions": [],
  "proposed_scope_conditions": [],
  "risk_predictions": [],
  "required_tests": [],
  "generation_trace": {}
}
```

## 验收标准

```text
1. 至少 50 个 candidate frameworks 来自真实 residual clusters。
2. 每个 candidate 自动生成完整 conservative obligations。
3. 至少 20% candidate 不是 scope narrowing，而是 genuine generalization / framework combination。
4. 生成器必须保留 negative evidence，不允许只输出看起来会赢的候选。
```

---

# R4：Conservative Generalization Gate v2

## 目标

把当前 gate 从“机制正确”升级成“真实任务集验证”。

当前 gate 的核心条件已经很好：old success preservation、residual explanation、limiting-case reduction、generality gain、new prediction success、regression bounded、required relation coverage、non-promoted next tests、no main graph mutation。

v2 要让每个分数都来自真实 test suite。

## 五类测试

### 1. Old success preservation

```text
旧框架过去成功过的任务集合 D_old_success。
候选框架必须不显著变差。
```

指标：

```text
old_success_preservation >= 0.95
old_success_regression <= 0.02
```

### 2. Residual explanation

```text
旧框架失败的 residual cluster D_residual。
候选框架必须显著改善。
```

指标：

```text
residual_utility_delta > 0
residual_explanation >= 0.75
```

### 3. Limiting-case reduction

```text
在父框架 scope condition 下，
候选框架输出的策略应与父框架一致或兼容。
```

例子：

```text
dependency-aware controlled intervention
在低耦合组件场景下应退化为普通控制变量法。
```

指标：

```text
limiting_case_reduction >= 0.90
```

### 4. Generality gain

候选框架必须至少满足一项：

```text
覆盖更多 residual families
压缩多个旧规则
统一多个 branch
跨域迁移
产生更强预测
```

### 5. New testable consequence

不能只是解释旧数据。必须产生新预测：

```text
在某类未见任务上，候选框架应该优于父框架。
```

## 决策

```text
reject:
  破坏旧成功或无法解释 residual。

branch_only:
  局部有用，但不够 general。

candidate_framework:
  通过核心测试，但需要更多 fresh evidence。

active_scoped_framework:
  通过 old success + residual + limiting case + new prediction。

general_framework:
  多域、多代 survival 后才允许。

core_philosophy_prior:
  很久以后再说，不能自动 promotion。
```

## 验收标准

```text
1. 每个 active framework 有完整证书。
2. 每个 rejected framework 留下 negative evidence。
3. 至少一个 candidate 从 branch_only 升到 active_scoped_framework。
4. 至少一个 candidate 因破坏旧成功被 reject。
```

---

# R5：Framework lifecycle and branch ledger

## 目标

框架必须有生命周期，不是 pass/fail 一次性判断。

## 状态机

```text
draft_branch
candidate_branch
branch_only
candidate_framework
active_scoped_framework
general_framework
deprecated
demoted_to_branch
rejected_boundary_only
contradicted
```

## Branch Ledger 记录

```json
{
  "branch_id": "...",
  "parent_framework": "...",
  "origin_residual": "...",
  "current_status": "...",
  "promotion_history": [],
  "demotion_history": [],
  "negative_evidence": [],
  "descendants": [],
  "framework_growth_score_history": []
}
```

## 关键机制

```text
1. 不删除失败分支，保留为 negative evidence。
2. 不允许 prompt trick promotion。
3. 不允许一次成功就进入 core prior。
4. 每个 active framework 定期 recheck。
5. 被新框架 generalize 的旧框架不删除，而是变成 limiting case。
```

## 验收标准

```text
1. 100% promoted framework 有 ledger。
2. 100% rejected framework 有 rejection reason。
3. demotion 可以 rollback。
4. active framework survival rate 在 fresh recheck 上可测。
```

---

# R6：Simulator-guided framework search

## 目标

让 world model 不只是 proposal triage，而是框架演化中的分支预算控制器。

你现在已经有 production graph-action simulator for triage/routing 的证据。Evidence pack 里写明 simulator leakage audit pass，production router true。 当前 claim frontier 也把 simulator 推到 L3.5：selective simulator deferral for low-risk graph-maintenance decisions with audit sampling，但 live ablation 和 judges 仍然 required。

现在要把它接到 framework evolution：

```text
candidate framework -> simulator predicts:
  old_success_regression_risk
  residual_improvement_prob
  limiting_case_failure_prob
  new_prediction_success_prob
  expected_test_cost
  expected_information_gain
```

## Simulator 只能做三件事

```text
1. 排序候选分支。
2. 选择哪些值得 fresh validation。
3. 决定 verifier tier。
```

不能做：

```text
1. 直接 promotion。
2. 替代 live validation。
3. 替代 human/expert review。
```

## 验收标准

```text
1. simulator 能减少至少 40% fresh tests。
2. 不阻断 true positive framework。
3. 对 rejected framework 的 high-risk prediction 有校准。
4. simulator 错误产生 SimulatorDefect residual，进入下一轮。
```

---

# R7：Formal certificate integration

## 目标

范畴论 / 信息几何 / Lean 证明层只在适用时工作：当候选框架可以形式化为有限结构时，要求 proof-carrying certificate。

你现在 formal 层已经很强。Claim frontier 里 C_formal 已经到 L3.5：每个 promoted morphism 要有 finite diagram、negative controls 和 external Lean-checked theorem-fragment certificate；full category-theory theorem prover 仍被阻断。

## 对 framework evolution 的作用

```text
1. 检查新框架是否保留父框架结构。
2. 检查 limiting-case reduction 是否成立。
3. 阻止 unsafe morphism / false analogy。
4. 给结构迁移提供 proof-carrying evidence。
```

## 三档处理

```text
formal_applicable:
  必须有 certificate。

semi_formal:
  有 process graph / testable invariants / negative controls。

not_formalizable:
  只走 empirical + semantic gate，不强求范畴论。
```

## 验收标准

```text
1. formal applicable framework 的 certificate coverage = 1.0。
2. 至少一个 unsafe mapping 被 formal gate block。
3. 不把 non-formalizable 方法论假设错误地塞进 formal theorem prover。
4. Lean artifact 可复现。
```

---

# R8：Multi-generation live framework evolution benchmark

## 目标

这是最重要的下一步：把所有机制放到真实多代闭环里。

你现在 self-evo paper evidence pack 已经有 bounded UGSE 0.923、1768-problem frozen main、720-call fresh repaired generator、simulator leakage audit、autonomy supervised production candidate、formal bounded true。

下一步要做一个专门针对 framework evolution 的 live benchmark。

## 实验设计

```text
Input:
  10 real residual clusters
  30 parent frameworks / philosophy principles
  old success set
  residual set
  unseen domain set

Loop:
  for generation in 1..5:
      generate candidate frameworks
      run conservative generalization gate
      simulator pre-screen
      formal gate if applicable
      run selected fresh validation
      update branch ledger
      promote / demote / reject
      generate next frontier
```

## 对照组

```text
A. no framework evolution
B. local patch only
C. raw wisdom generation
D. simulator without conservative gate
E. conservative gate without simulator
F. full framework evolution agent
```

Philosophy Growth Benchmark 已经给了这个对比雏形：conservative generalization 明显优于 local patch 和 raw wisdom。 现在要把它放到真实 live residual 任务上。

## 指标

```text
framework_growth_score
old_success_preservation
residual_explanation
limiting_case_reduction
generality_gain
new_prediction_success
regression_cost
active_framework_survival_rate
negative_evidence_retention
branch_to_framework_transition_count
prompt_trick_retained_count
core_prior_overpromotion_count
```

## 通过标准

```text
1. full agent 显著优于 local patch / raw wisdom。
2. 至少 3 个 active_scoped_framework 跨代存活。
3. old_success_preservation >= 0.95。
4. residual_explanation >= 0.75。
5. prompt_trick_retained_count = 0。
6. core_philosophy_prior_promotion_count = 0。
7. fresh validation 中有 accepted 和 rejected，不是全收或全拒。
```

---

# R9：External evaluation and paper-grade packaging

## 目标

让审稿人相信这不是自嗨 artifact。

## 需要三类外部验证

### 1. Human/expert framework evaluation

给专家看：

```text
parent framework
residual cluster
candidate framework
old success tests
new prediction tests
system decision
```

专家标注：

```text
is this a conservative generalization?
is it only local patch?
does it preserve old framework?
is the new prediction meaningful?
```

目标：

```text
expert agreement with system >= 0.65
```

### 2. Fresh rerun

你已经有 fresh rerun：720/720 calls，无 live errors，accepted candidates trigger utility 0.8333，accepted control loss 0。

下一步做 framework-specific fresh rerun：

```text
fresh tasks generated after framework promotion
no reuse of old evidence
same baselines
problem-level CI
```

### 3. Reproducibility pack

必须提供：

```text
artifact index
hashes
commands
redaction policy
no-secret audit
frozen manifests
test scripts
claim ledger
```

现在 evidence pack 已经开始做这个了，并列出了 paper sections 和 repro commands。

---

# 4. 最重要的新增概念：Framework Growth Score

你现在已经有 framework growth score，但建议把它正式写成论文公式。

```text
FrameworkGrowth(F_new | F_old) =
  w1 * OldSuccessPreservation
+ w2 * ResidualExplanation
+ w3 * LimitingCaseReduction
+ w4 * GeneralityGain
+ w5 * NewPredictionSuccess
+ w6 * SimulatorExpectedUtility
- w7 * RegressionCost
- w8 * ComplexityPenalty
```

对应你当前 gate 里的实际实现：它已经用 old preservation、residual explanation、limiting reduction、generality gain、new prediction success、simulator utility、regression cost、complexity penalty 计算 framework_growth_score。

论文中可以解释：

```text
这不是普通 answer score。
它测的是一个新框架是否值得成为旧框架的保守泛化。
```

---

# 5. 90%+ 的真正定义

你现在 bounded UGSE 已经到 0.923，但要注意这不是“unbounded AGI 90%”。Evidence pack 写的是 roadmap closure 19/19、bounded UGSE 0.923。

我建议把 90%+ 定义成：

```text
90%+ Bounded Framework Evolution Agent =
  能在真实 residual 上生成新框架；
  能证明新框架保留旧成功；
  能证明新框架解释 residual；
  能证明新框架在旧 scope 下退化回旧框架；
  能产生新预测并 fresh validate；
  能把失败分支保留为 negative evidence；
  能多代递归演化；
  能阻止 prompt trick / raw wisdom / overgeneralization；
  能用 simulator 节省测试成本但不替代验证；
  能在主图中 canary-scope apply 并 rollback。
```

不是：

```text
通用 24/7 自主哲学发明机。
```

---

# 6. 现在该停止做什么

非常重要。

## 不要继续加很多 Phase 名字

现在 artifact 已经很多了。继续加 Phase 会让论文读者迷失。

应该合并成 4 条主线：

```text
1. Assumption Lifecycle Kernel
2. Dialectical Framework Growth
3. Simulator/Formal Verifier Routing
4. Supervised Autonomy and Evidence
```

## 不要 claim “发现新哲学”

更稳的 claim 是：

```text
the system performs bounded framework evolution through conservative generalization.
```

也就是：

```text
系统可以在旧框架失败处生成、验证和保留更 general 的 scoped framework。
```

## 不要让 simulator 或 formal layer 越权

Simulator 是 triage/router。
Formal 是 finite certificate gate。
Fresh validation 和 negative controls 仍是最终依据。

---

# 7. 论文现在可以怎么写

我建议题目类似：

```text
Everything Is an Assumption:
Conservative Generalization over Assumption Graphs
for Bounded Recursive Self-Evolution
```

核心贡献：

```text
1. Assumption Graph:
   把 agent 决策变成可失败假设的生命周期对象。

2. Conservative Generalization:
   新假设/框架必须解释 residual、保留旧成功、退化回父框架，并产生新预测。

3. Framework Evolution Agent:
   多代 residual-to-framework generation + branch ledger + negative evidence retention。

4. Simulator/Formal Gates:
   world model 做 proposal triage/verifier routing，finite certificate 阻止结构迁移幻觉。

5. Evidence:
   bounded UGSE 0.923，720-call fresh rerun，philosophy growth benchmark，framework evolution run，simulator leakage audit，supervised autonomy candidate。
```

---

# 8. 最终判断

这版之后，我认为你的项目已经不是“假设框架演化 agent 的雏形”那么简单了。更准确说：

> **它已经有了一个 bounded dialectical framework-evolution agent 的工程原型。**

现在它能做的已经包括：

```text
1. 把新假设分成 branch / candidate framework / active scoped framework。
2. 用 conservative generalization gate 阻止 local patch 冒充新框架。
3. 用 open-ended framework evolution run 做多代 retained frontier。
4. 用 Philosophy Growth Benchmark 量化新框架优于 local patch 和 raw wisdom。
5. 用 fresh rerun 证明 unfiltered generator 不行，gated selective retention 才行。
6. 用 claim frontier 明确 L3.5 能 claim 什么，L4 仍阻断什么。
```

如果想“完全深化”，现在不是从头再设计，而是把这套机制从 bounded artifact 推向真实外部实验：

```text
真实 residual clusters
真实 LLM-generated framework candidates
真实 old-success / residual / unseen-domain validation
真实专家评审
真实 fresh rerun
真实 canary graph apply
```

做到这些之后，你的项目就可以非常有底气地说：

> **我们不是让 AI 随机提出新假设，而是让 AI 在已有方法论框架的边界处，通过保守泛化生长新框架，并用递归验证决定它们能否进入 Assumption Graph。**

这就是“哲学史是哲学自身展开”的工程版本，也是你这个项目最独特的论文贡献。
