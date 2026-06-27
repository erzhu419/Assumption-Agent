先说结论：**你现在这个系统在“工程机制 / 假设生命周期 / 框架演化”上已经很强，但如果拿它和原生 GPT-5.5 做普通一次性问答平均分比较，几乎没效果是很正常的。**

不是因为你的大方向错了，而是因为当前系统的大量能力还停留在：

```text
离线证据组织
候选假设生成
候选门控
claim boundary
framework evolution artifact
simulator routing
formal certificate
```

而不是稳定地、强制地改变每一次最终回答的生成过程。

更直白地说：

> **现在的 Assumption-Agent 更像一个“假设操作系统 / 研究闭环 / 证据治理层”，而不是一个已经能在所有普通 prompt 上打赢 GPT-5.5 的 answer engine。**

而原生 GPT-5.5 已经内化了大量“哲学方法论”：控制变量、反例、边界条件、类比、抽象、分解、反证、风险评估。所以如果你的系统只是把这些原则作为几条文本 context 塞进 prompt，它很可能不会比原生 GPT-5.5 强，甚至会因为噪声、冗长、错检索而变差。

---

# 1. 当前“假设”到底是什么形式？

现在你的假设不是单一形式，而是一个多层对象。

代码里最核心的是 `AssumptionNode`。它有：

```text
type
claim
kind
formal_form
context_conditions
predicted_effects
risk_predictions
verifiers
evidence_ids
residual_ids
confidence
metaproductivity
status
tags
payload
```

也就是说，它不是一句“想法”，而是一个带适用条件、预期效果、风险、验证器、证据、残差和生命周期状态的节点。

类型也已经很丰富：

```text
object
process
method
evaluator
memory
world_model
alignment
self_modification
harness
retrieval
strategy
residual
case
verifier
framework
framework_branch
certificate
```

这说明你现在的假设已经不只是科学命题，而是包括方法、记忆策略、评价器、世界模型、自修改、框架分支、形式证书等。

边也不是普通知识图谱三元组，而是支持框架生长：

```text
generalizes
reduces_to_under_scope
explains_residual
preserves_success_cases
modifies_boundary_of
predicts_new_case
conflicts_with
generated_from_residual
failed_because
has_certificate
demotes_to_branch
replaces_boundary_of
```

这些边正好对应你说的“新假设要在旧哲学模型上长出分支，并且保留旧成功、解释旧失败、提供更 general 的结构”。

另外还有 `TrialManifest`，它是每一次 agent 行为的“可证伪合约”：记录这次行动背后的 assumption、为什么选它、预期效果、验证计划、rollback 条件、观察结果、residual 和状态。

所以目前完整形式可以总结为：

```text
假设 = AssumptionNode + AssumptionEdge + TrialManifest + Evidence + Residual + Optional Formal Payload
```

如果是更高层框架假设，则是：

```text
Framework Hypothesis =
  parent frameworks
  residuals explained
  old successes preserved
  limiting cases
  new predictions
  conflict boundaries
  conservative-generalization certificate
```

你的 `conservative_generalization_gate.py` 已经把这个原则明确写成 gate：新 framework 只有在解释 residual、保留旧成功、在父框架 scope 下退化回父框架、并产生新可测试后果时才有资格晋升。

---

# 2. 假设是从 0 开始产生的吗？

不是。

你现在的系统不是从 0 生成假设，而是从几个来源启动：

```text
1. 人类已有策略库 / phase zero strategies
2. wisdom_library.json
3. v16 residuals
4. Exp82 typed hypotheses
5. 后续 evaluation writeback
6. fresh ablation / judge 结果
7. residual clusterer
8. failure hypothesis generator
9. framework evolution generator
10. simulator / formal / verifier 产生的失败和边界
```

README 明确说，build graph 会把旧的 `strategies/`、`wisdom_library.json`、`v16_residuals.json`、Exp82 typed hypotheses 转成 graph nodes。 当前 build graph 也列出了这些输入。

所以更准确地说：

> **初始假设图不是从 0 开始，而是从人类/历史实验/旧系统 residual/typed hypothesis 中 bootstrap。之后随着测试越来越多，它应该越来越丰满。**

但是有一个关键前提：

```text
只有当 writeback / evidence update / residual update / accepted candidate apply 真正开启时，它才会“越测越丰满”。
```

否则它只是读取一堆静态 artifact，不会真的成长。

在 `graph_memory.py` 里，`update_from_trial` 会把 TrialManifest 写回 graph。如果有 residual，它会生成 residual node，并把 residual node 连到对应 assumption；如果是 execution lapse，会用 `EXECUTION_LAPSE_OF`，否则用 `FAILED_BECAUSE`。同时还会更新 node confidence。

所以理论上它是会成长的：

```text
trial -> observed effect / residual
     -> residual node
     -> confidence update
     -> residual edge
     -> future retrieval / generation / gate
```

但工程上要看你是否真的在每次测试后运行了：

```text
record_phase2_eval
runtime_trace_writeback
candidate_acceptance
apply_accepted
memory_consolidation
framework ledger update
```

如果没有，它就不会“活”。

---

# 3. 为什么和原生 GPT-5.5 比几乎没效果？

我认为有 12 个主要原因。

---

## 原因 1：你的系统大部分能力没有进入最终回答路径

这是最核心的问题。

现在很多强模块是：

```text
conservative generalization gate
framework growth benchmark
open-ended framework evolution
simulator production gate
finite proof certificates
claim frontier
paper evidence pack
autonomy journal
```

这些很强，但它们多数是在**离线治理 / 证据评估 / 候选筛选 / claim boundary**层面工作。

而真正生成答案的路径，大致仍然是 `phase2_v20_framework.py` 里的：

```text
Turn 0: frame + rewrite
Turn 1: solve with primary frame + rewritten problem + secondary references
Turn 2: audit/revise
```

它的执行 prompt 里，Assumption Graph 被放在：

```text
## 次要参考（仅在与 PRIMARY FRAME 一致时纳入）
{assumption_context_block}
```

也就是说，假设图不是主控逻辑，而是 secondary reference。

这会导致：

```text
原生 GPT-5.5 本来就会 reframe / reason；
你的 graph 只是多给了一些建议；
模型可以忽略它；
甚至它被明确降级为“次要参考”。
```

所以它不一定改变最终答案。

---

## 原因 2：Turn 2 反思阶段没有重新注入假设图

Turn 2 的 `REFLECT_PROMPT_V20` 只看：

```text
原问题
Primary Frame
重写版本
草稿
```

然后检查草稿是否按 frame 作答。它不再看 assumption context、retrieved assumptions、framework branch、verifier obligations。

这意味着：

```text
即使 Turn 1 注入了 Assumption Graph，
Turn 2 可能把这些内容洗掉，
只保留 frame/rewrite。
```

如果最终答案来自 Turn 2，那么假设图对最终答案的影响会进一步被稀释。

---

## 原因 3：假设图注入方式太“文本建议化”，不是“操作约束化”

`format_assumption_context` 会把节点格式化成：

```text
Claim
Trigger/context
Expected effect
Risks
Verifier
```

并且开头说：

```text
Use these as falsifiable operating assumptions, not decorative advice.
```

这已经比普通 RAG 好，但对 GPT-5.5 来说仍然很像：

```text
这里有几条建议，你参考一下。
```

它没有被转换成：

```text
必须执行的步骤
必须检查的约束
必须输出的结构
必须通过的 verifier
如果失败要 fallback 的策略
```

所以很容易变成“decorative advice”。

你自己 README 里也说，runtime trace 的 expected effect 是希望 retrieve method/harness/runtime/residual assumptions that shape the draft rather than decorate it。
这句话其实也暴露了当前问题：**系统知道风险是 decorative use，但执行路径还没有完全阻止 decorative use。**

---

## 原因 4：retrieval 仍然偏 lexical / PPR，不是强语义策略选择

`SimpleAssumptionGraph.retrieve` 的检索是：

```text
tokenize
cosine over token counters
explicit seed ids
PPR-like graph spreading
confidence/metaproductivity bonus
```

这很好审计，但它不是强 embedding / cross-encoder / learned retriever。它可能出现：

```text
检索到语义接近但执行无关的策略；
检索到 general wisdom；
检索不到真正能改变答案的具体 operator；
top_k 中噪声多。
```

对 GPT-5.5 来说，泛泛策略常常没增益，因为它自己已经知道。

---

## 原因 5：默认跳过了一些最可能受益或最可能出问题的 domain

README 里写：software engineering 因为 graph context 曾造成 negative transfer，所以默认 gated off。

代码里默认 `--assumption-graph-skip-domains` 是 `software_engineering`。

这很安全，但也意味着：

```text
你为了避免负迁移，把一部分可能显示差异的领域直接关掉了；
剩下领域如果 GPT-5.5 原生已经强，平均增益自然小。
```

---

## 原因 6：很多 math/science 任务走 bypass，根本不用图

`phase2_v20_framework.py` 对 math/science 有 bypass 路径，直接用 domain-specific execution prompt。

这可能是对的，因为 generic graph context 对证明/机制题可能有害。但这也说明：

```text
你的 Assumption Graph 并不是所有任务都参与；
平均比较时很多任务其实还是 prompt routing / domain template，而不是 hypothesis graph。
```

如果你拿所有任务平均和原生 GPT-5.5 比，图的贡献会被稀释。

---

## 原因 7：原生 GPT-5.5 已经内化了很多“哲学模型”

你的初始策略很多是：

```text
控制变量
reframe
分解
边界条件
负对照
反例
增量替换
风险评估
```

这些对较弱模型有帮助，但 GPT-5.5 原生已经会做。
所以如果你只是把这些原则作为自然语言提示，它对 GPT-5.5 的边际增益会很小。

要打赢 GPT-5.5，你不能只给它“原则”，而要给它：

```text
它不知道的跨 run 经验；
它无法从当前 prompt 推断的历史 residual；
已经验证过的 task-specific framework branch；
具体 verifier / negative control / failed case；
需要回避的历史误判。
```

---

## 原因 8：现在系统最强的是“高精度低覆盖”，不是“平均大幅提升”

你 fresh rerun 里最说明问题：

```text
accepted/rejected candidates: 4/76
unfiltered trigger utility: 0.4161
accepted trigger utility: 0.8333
accepted control loss: 0.0
```

这说明系统的 gate 很强：被接受的候选很好。
但只有 4 个被接受，覆盖率很低。

所以平均效果可能是：

```text
大多数样本：和 GPT-5.5 一样，甚至多一点噪声
少数样本：明显更好
总体平均：几乎没效果
```

这不是失败，而是说明当前系统是：

```text
high precision / low coverage self-evolution
```

不是：

```text
universal answer improver
```

---

## 原因 9：unfiltered generator 本来是负的

`paper_fresh_rerun_result_integration` 明确说：unfiltered generated frontier 不应该应用，因为 all-candidate trigger utility failed。

`paper_broad_generator_repair_integration` 也显示：

```text
original trigger utility: 0.4161
v1 repair: 0.3183
v2 repair: 0.5462
v2 selected candidates: 8 / original 80
```

这意味着：

```text
生成器的原始输出大多不能直接提升答案；
真正有效的是筛选后的少数候选。
```

如果你在最终回答路径里没有做到强选择、强 abstain，而是把很多候选或原则注入上下文，效果自然弱。

---

## 原因 10：你当前的评价可能测错了系统能力

如果你比较的是：

```text
单题最终答案质量
Assumption-Agent vs 原生 GPT-5.5
```

那你测的是 answer engine。

但你的系统真正强的是：

```text
跨 run 学习
residual 归因
候选假设生命周期
高精度 selective retention
framework growth
claim boundary
graph maintenance
```

这些能力不会在单题平均分里完全体现。

更适合的评价应该是：

```text
同一任务分布连续 5 轮：
  第 1 轮暴露 residual
  第 2 轮生成 candidate
  第 3 轮 fresh validate
  第 4 轮 graph update
  第 5 轮 unseen residual family 上测试

看是否逐轮减少同类 residual。
```

---

## 原因 11：假设图会丰满，但不等于 GPT-5.5 权重会改变

这是一个关键理解点。

你的假设模型是外部图 / 外部记忆 / 外部证据系统，不是模型权重。
测试越多，它会变丰满：

```text
更多 nodes
更多 edges
更多 evidence
更多 residuals
更精确 scope
更好的 confidence
更多 negative evidence
更多 framework branches
```

但 GPT-5.5 本身不会自动“内化”这些。它只有在：

```text
检索到
格式化好
注入 prompt
被强约束使用
经过 verifier 检查
```

时才会用到。

所以“越测越丰满”成立，但还缺一步：

```text
丰满的假设图如何稳定转化为更好的执行行为？
```

现在这一步还弱。

---

## 原因 12：假设没有被编译成 action policy

目前 AssumptionNode 主要是：

```text
claim + context + expected effect + risk + verifier
```

这很适合记录和审计，但不一定适合直接执行。

最终回答需要的是：

```text
当触发条件满足时：
  执行哪些步骤？
  输出结构是什么？
  检查哪些反例？
  如果失败怎么 fallback？
  哪些内容必须不要说？
```

也就是说，假设应该被编译成：

```text
Assumption -> Operator
```

而不是：

```text
Assumption -> Context text
```

当前系统这一步不够强。

---

# 4. 后续方向：不要再扩模块，先做“假设到执行”的闭环

现在最重要的方向不是继续加 L4/Phase，而是把已有假设真的变成 answer-time execution policy。

我建议你后续做 6 件事。

---

## 方向 1：Assumption Application Fidelity

先测清楚：检索到的假设到底有没有被用。

新增一个评价：

```text
Assumption Application Fidelity
```

每个回答记录：

```text
retrieved_assumption_ids
selected_assumption_ids
used_assumption_ids
ignored_assumption_ids
misapplied_assumption_ids
decorative_use_count
```

让 judge 或程序检查：

```text
答案是否真的执行了该假设？
还是只是提到了相关词？
```

例如“控制变量法”不是答案里出现“控制变量”四个字，而是必须有：

```text
fixed variables
changed variable
observed metric
control group
confound check
decision rule
```

如果没有，就是 decorative use。

目标指标：

```text
application_fidelity >= 0.75
decorative_use_rate <= 0.10
```

如果这个指标上不去，最终效果一定弱。

---

## 方向 2：把 AssumptionNode 编译成 OperatorSpec

新增一个对象：

```python
OperatorSpec:
    trigger_conditions
    execution_steps
    required_output_slots
    negative_controls
    verifier_checks
    fallback_policy
```

比如：

```text
Assumption: 控制变量法
```

应该编译成：

```json
{
  "trigger": ["causal attribution uncertain", "multiple possible factors"],
  "steps": [
    "list candidate variables",
    "choose one variable to change",
    "hold other variables fixed",
    "define measurable outcome",
    "compare against control",
    "state confound risks"
  ],
  "required_output_slots": [
    "variable_changed",
    "variables_held_constant",
    "metric",
    "control",
    "decision_rule"
  ],
  "negative_controls": [
    "do not recommend changing all factors at once"
  ],
  "fallback": "if variables are coupled, switch to dependency-aware intervention"
}
```

然后 answer prompt 不再注入：

```text
Claim: 固定其他条件，每次只改变一个因素
```

而是注入：

```text
Use OperatorSpec S01. Your answer must fill slots A/B/C/D.
```

这会显著提高可执行性。

---

## 方向 3：只在“高触发概率任务”启用 Assumption Graph

不要平均启用。
你现在的系统是高精度低覆盖，所以应该先做：

```text
Selective Activation
```

让 simulator / retriever 判断：

```text
当前任务是否真的需要 assumption graph？
```

如果不是，就直接用原生 GPT-5.5。

策略：

```text
if no high-confidence operator:
    use native GPT-5.5
else:
    use Assumption-Agent
```

目标不是每题都赢，而是：

```text
在触发子集上明显赢；
在非触发子集上不伤害。
```

你的 fresh rerun 已经证明 accepted candidates 很强，但少。 所以后续应优化：

```text
coverage under precision constraint
```

而不是全局平均硬推。

---

## 方向 4：做“同类 residual 递减”实验，而不是单轮平均分

你现在最适合的实验不是：

```text
Agent vs GPT-5.5 one-shot
```

而是：

```text
Round 0: GPT-5.5 baseline 暴露 residual
Round 1: 生成假设 / operator
Round 2: fresh validate
Round 3: 写入 graph
Round 4: 同类 unseen residual 测试
```

看：

```text
same_residual_family_error_rate 是否下降
```

这才是 Assumption OS 的核心价值。

指标：

```text
residual_family_error_rate_before
residual_family_error_rate_after
old_success_preservation
control_harm
coverage
```

如果这个实验赢了，即使普通平均分只涨一点，也足够说明系统有价值。

---

## 方向 5：让 Conservative Generalization 进入 answer-time，而不是只在 paper artifact 中

现在 conservative generalization 很强，但它主要是 framework growth / paper evidence。
要产生最终答案增益，需要把 active scoped framework 编译为 OperatorSpec。

例如：

```text
Framework:
Dependency-Aware Controlled Intervention
```

变成 answer-time 操作：

```text
1. First infer whether variables are independently manipulable.
2. If independent, use classic control-variable plan.
3. If coupled, use dependency graph / paired ablation / interface boundary.
4. State limiting case.
5. State what would falsify the intervention plan.
```

这样它才会比 GPT-5.5 原生“泛泛建议控制变量”更强。

---

## 方向 6：区分三种目标

你现在可能把三个目标混在一起了：

```text
A. 提高最终答案质量
B. 证明系统能自我演化
C. 证明新框架能保守泛化旧框架
```

这三个都重要，但实验不同。

如果目标 A：

```text
focus on OperatorSpec + selective activation + answer-time verifier
```

如果目标 B：

```text
focus on residual family recurrence over multi-round runs
```

如果目标 C：

```text
focus on framework growth benchmark + expert review + limiting-case tests
```

不要用一个平均分同时证明三件事。

---

# 5. 当前假设模型会不会随着测试越多越丰满？

会，但要满足三个条件。

## 条件 1：每次测试都写回 TrialManifest

每次 retrieval / solve / judge / simulator / verifier 都要变成 TrialManifest。系统已经有这个 schema。

## 条件 2：失败必须归因到 residual type

不是所有失败都说明假设错。你已经区分：

```text
execution_lapse
optimization
assumption_defect
discovery
evaluator_defect
memory_defect
simulator_defect
```

这很重要。否则 graph 会被错误惩罚污染。

## 条件 3：接受的新假设必须经过 gate，而不是直接 append

如果直接 append，测试越多越脏。
如果经过：

```text
contract check
preflight
fresh ablation
acceptance
conservative generalization
negative controls
```

测试越多才会越丰满。

所以准确说：

> **假设模型不是自动越测越好；它是在有门控、有残差归因、有负证据、有 scope refinement 的前提下，才会越测越丰满。**

没有这些，它只会变成经验垃圾场。

---

# 6. 你现在应该怎么判断“没效果”到底是哪种没效果？

我建议做一个诊断表，每个样本都打标签。

## 对每个问题记录：

```text
1. graph_activated?
2. retrieved_node_count?
3. retrieved_operator_count?
4. top_assumption_ids?
5. assumption_context_chars?
6. answer_used_assumption?
7. application_fidelity?
8. did_answer_improve?
9. did_answer_regress?
10. residual_type?
```

然后统计四象限：

```text
A. 未激活 graph，没效果
B. 激活了但检索错，没效果
C. 检索对了但没执行，没效果
D. 执行了但假设错，负效果
E. 执行了且假设对，有效果
```

你现在很可能大量样本在 B/C：

```text
retrieved but decorative
```

或者 A：

```text
skipped / bypass / native-like path
```

这比只看平均分有用得多。

---

# 7. 下一步最小实验

我建议你立刻做一个很小但锋利的实验：

## 实验名

```text
Operatorized Assumption Graph vs Native GPT-5.5 on Triggered Residual Families
```

## 数据

选 30 个问题，不要太多：

```text
10 个控制变量/因果归因
10 个系统重构/增量替换
10 个类比/结构迁移
```

每类都必须是 GPT-5.5 原生容易泛泛而谈的题。

## 三个系统

```text
A. native GPT-5.5
B. current Assumption-Agent context injection
C. operatorized Assumption-Agent
```

## C 的方式

每题最多注入 1–2 个 OperatorSpec，不注入长 graph。

## 指标

```text
answer utility
application fidelity
required slot completion
control harm
judge preference
```

通过标准：

```text
C > A
C > B
application_fidelity C >> B
```

如果 C 赢，说明问题在“假设没有 operator 化”。
如果 C 仍不赢，说明假设本身或任务选择不对。

---

# 8. 长期方向

最终你需要把系统分成两层：

## 8.1 Learning layer

负责：

```text
residual -> hypothesis/framework -> validation -> graph update
```

这是你现在强的地方。

## 8.2 Execution layer

负责：

```text
task -> select operator -> enforce output structure -> verify answer -> revise
```

这是现在弱的地方。

你现在大部分工程都在 learning layer。
如果想打赢原生 GPT-5.5 的单题表现，必须加强 execution layer。

---

# 9. 最终回答你的几个问题

## 当前假设是什么形式？

```text
AssumptionNode + Edge + TrialManifest + Evidence + Residual + optional formal payload。
```

高层框架假设还包括：

```text
parent frameworks
residuals explained
old successes preserved
limiting cases
new predictions
conservative generalization certificate
```

## 如何产生？

不是从 0。来源包括：

```text
人工策略库
wisdom library
历史 residual
Exp82 typed hypotheses
runtime traces
evaluation writeback
failure hypothesis generator
residual clusterer
LLM-generated candidates
framework evolution generator
fresh ablation acceptance
```

## 是不是越测试越丰满？

是，但前提是：

```text
writeback 开启；
residual 正确归因；
失败成为 negative evidence；
通过 gate 的才进入 active graph；
定期 memory consolidation；
假设被 scope refinement。
```

否则测试越多只会越乱。

## 为什么和 GPT-5.5 比没效果？

最可能原因是：

```text
1. GPT-5.5 已经会很多通用哲学方法。
2. 你的假设多以文本 context 注入，不是强 operator。
3. assumption context 是 secondary reference，不是主执行约束。
4. Turn 2 audit 没有重新使用 assumption context。
5. 检索偏 lexical/PPR，可能噪声高。
6. 很多任务 bypass/skip，图贡献被稀释。
7. 当前系统高精度低覆盖，平均分自然小。
8. 你测的是 one-shot answer quality，而系统真正强项是跨 run residual reduction。
```

---

# 10. 最关键的一句话

你现在已经证明了：

> **系统可以管理假设、筛选假设、保守泛化框架、阻止过度 claim。**

但还没有充分证明：

> **这些假设能在 answer-time 被稳定执行，并显著超过 GPT-5.5 原生能力。**

下一步不是再扩更多理论模块，而是把：

```text
AssumptionNode / FrameworkNode
```

编译成：

```text
OperatorSpec / Verifiable Execution Constraint
```

并测：

```text
application fidelity
triggered subset improvement
same residual family error reduction
```

只要这一步打通，你的 Assumption-Agent 才会从“很强的假设操作系统”变成“确实提升最终智能体表现的假设执行系统”。

---

# 11. 2026-06-26 HLE debug addendum

最新 weakfail6 HLE cache-only 结果：

```text
cohort: fixed HLE multiple-choice seeds 810,966,1048,1084,1128,1457
model: gpt-5.4-mini
source policy: local HLE + local evidence cache, live source search disabled
```

已经完成：

```text
1. 内存检查：未发现 HLE memory leak；高内存主要是 WSL/Linux page cache。
2. local evidence corpus cache 改成 bounded LRU。
3. parallel shard heartbeat 增加 process_memory / peak RSS / peak VMS。
4. direct-control model call 增加 same-model API failover。
5. router attempt 增加安全 JSONL log：只记录 candidate index、base URL、key hash、error label、retry decision，不记录 prompt/answer/API key。
6. subprocess 包装的 HTTP 503 和 RemoteDisconnected 已纳入 transient retry。
```

当前固定 cohort 的 clean 对照：

```text
agent gate-on: 2/6
agent gate-off: 2/6
raw direct: 0/6
HippoRAG direct: 0/6
raw budget-matched: 0/6
HippoRAG budget-matched: 0/6
```

重要 caveat：

```text
这轮说明 agent 在这个固定 weakfail6 cohort 上赢过 raw/HippoRAG/budget controls。
但这不是 OperatorSpec application 的胜利。
OperatorSpec activation 存在，application coverage 仍然是 0.0。
当前收益主要来自 programmatic/domain-rule verifier + selection/retry hygiene。
```

所以暂不 push。下一步应该继续做：

```text
1. subprocess model call 的 no-byte per-attempt watchdog，避免 silent socket hang。
2. 更宽的 operator-bearing HLE cache-only cohort。
3. residual-family before/after loop。
4. answer-bearing retrieval / directness verifier 继续加强。
```

补充进展：

```text
1. 已加入 opt-in no-byte watchdog：MODEL_ROUTER_SUBPROCESS_NO_BYTE_TIMEOUT_SEC。
   作用域是单次 subprocess attempt；触发后作为 transient retry，不作为 HLE row process timeout。
2. seed1457 no-byte=90 的验证说明机制有效，但 90 秒太短：
   raw_budget 4/5 answered, 1 child no-byte error；
   HippoRAG_budget 5/5 answered；
   top-level error=0, process timeout=0。
   后续 live HLE 如果启用，建议 180/240 秒，不建议 90 秒。
3. 已生成更宽的本地 metadata-only operator-bearing preflight：
   hle_operator_broader_preflight_n20_cacheonly_20260626
   selected=22
   families: answer_bearing_relation=5, controlled_variable=5,
   incremental_replacement=5, structural_transfer=5,
   sec_mals_mass_balance=1, cross_resistance_minimality=1。
```

下一组候选 seed offsets：

```text
16,27,32,54,60,72,87,115,136,141,185,191,280,366,427,499,966,1048,1207,1457,830,1219
```

建议先跑一个 family-balanced n=12 子集：

```text
16,27,32,54,60,115,185,191,280,966,830,1219
```

然后再决定是否扩到完整 n=22。

---

# 12. 2026-06-26 span/contrastive probe 结论

固定 HLE cache-only 6-seed probe：

```text
eval_id: hle_operator_broader_span_contrastive_probe6_mini_20260626
seeds: 16,27,32,54,191,966
model: gpt-5.4-mini
source policy: local HLE + local evidence cache only, live source search disabled
new flags: option-claim contrastive adjudicator on; span-directness verifier on
```

结果：

```text
raw: 0/6
raw_budget_matched: 0/6
HippoRAG: 0/6
HippoRAG_budget_matched: 1/6
assumption_agent: 0/6
paper_clean_pass: false
process_timeout: 0
top_level_error: 0
raw_content_persisted: false
```

关键诊断：

```text
1. contrastive/span gate 变得更可观测，也确实触发了：
   contrastive used on 5/6 agent rows;
   span-directness used on 4/6 agent rows.
2. 但没有任何一题被修正：
   contrastive accepted: 0
   span-directness accepted: 0
   agent correct: 0/6
3. 错因仍然集中在 candidate/evidence quality：
   candidate_generation_missed_gold: 3/6
   missing_model_option_source_retry_unhelpful: 6/6
   multiple_choice_selection_failed: 6/6
   verified_or_abstain abstained: 5/6
4. 这刀显著增加 latency：
   agent avg latency: 608s
   agent max latency: 1282s
   shard max elapsed: 1598s
```

结论：

```text
这刀不能 push，也不应该上 5.5。
span/directness 的方向是对的：它没有放宽错误证据。
但当前瓶颈不是 gate 太严，而是 answer-bearing evidence / candidate set 仍不足。
更重要的是 self-contained/full-option/minority post-adjudicators 造成长尾成本，必须有预算门。
```

已补的工程修复：

```text
1. CLI/runner 已支持显式传递：
   --enable-option-claim-contrastive-adjudicator
   --enable-option-claim-span-directness-verifier
2. unique-span 与 candidate-direct-span extractor 会保留上游 statement-fact / answer-web direct slot metadata。
3. 新增 opt-in post-adjudicator budget gate：
   HLE_SELECTION_POST_ADJUDICATOR_BUDGET
   为空或负数：旧行为，不限预算
   0：跳过 self-contained option / full-option / minority verifier 模型调用
   N：同一 eval_id + call_id + problem_id_hash 最多允许 N 次这类高成本 post-adjudicator
4. budget gate 记录事件：
   selection_post_adjudicator_budget_gate
   包含 post_adjudicator_kind, budget_limit, budget_used_before, allowed, reason。
```

下一步：

```text
1. 重跑同一 6-seed cache-only probe，但设置：
   HLE_SELECTION_POST_ADJUDICATOR_BUDGET=0
   目标不是涨准确率，而是确认 latency 大幅下降且准确率不更差。
2. 如果准确率仍 0/6，停止调 gate；
   下一刀回到 source/candidate 质量：
   - gold/sweep-only candidate 的 answer-bearing local evidence backfill
   - candidate set 晋级前的 direct relation source coverage
   - residual-family before/after，而不是扩大随机 HLE。
3. OperatorSpec 仍未证明 answer-time application：
   这轮 operator_specs activated 6/6，
   但 operator application verifier rows 仍为 0，
   因此不能把效果归因给 OperatorSpec。
```

---

# 13. 2026-06-26 post-adjudicator budget0 probe 与下一刀

固定同一 6-seed cache-only probe，加 `HLE_SELECTION_POST_ADJUDICATOR_BUDGET=0`：

```text
eval_id: hle_operator_broader_span_contrastive_budget0_probe6_mini_20260626
seeds: 16,27,32,54,191,966
model: gpt-5.4-mini
source policy: local HLE + local evidence cache only, live source search disabled
post-adjudicator budget: 0
```

结果：

```text
raw: 0/6
raw_budget_matched: 1/6
HippoRAG: 0/6
HippoRAG_budget_matched: 1/6
assumption_agent: 1/6
paper_clean_pass: true
process_timeout: 0
top_level_error: 0
raw_content_persisted: false
```

关键诊断：

```text
1. budget gate 确实生效，JSONL 记录了 selection_post_adjudicator_budget_gate。
2. agent 从 0/6 回到 1/6，但 raw_budget 和 HippoRAG_budget 也都是 1/6。
   因此只能说 agent 没低于公平预算控制，不能说已经显著赢。
3. latency 没有解决：
   agent avg latency: 606s
   agent max latency: 1058s
   shard max elapsed: 1533s
4. 剩余错因仍然是 option-claim/source verifier 这一层：
   candidate_generation_missed_gold: 2/6
   evidence_invalid_or_unhelpful: 3/6
   missing_model_option_source_retry_unhelpful: 5/6
   multiple_choice_selection_failed: 5/6
   verified_or_abstain abstained: 4/6
```

结论：

```text
不能 push。
post-adjudicator budget 是 hygiene improvement，不是效果证明。
长尾不主要来自 self-contained/full-option/minority post-adjudicator，
而是 source-grounded option-claim verifier、span directness、contrastive 这些子调用。
```

已补工程修复：

```text
1. 新增 opt-in per-problem verifier model-call budget：
   HLE_OPTION_CLAIM_VERIFIER_MODEL_CALL_BUDGET
   为空或负数：旧行为，不限预算
   0：跳过 option-claim verifier/adjudicator 模型调用
   N：同一 eval_id + root_call_id + problem_id_hash 最多允许 N 次
2. root_call_id 会归并：
   _retry_N
   _span_directness_verifier
   _candidate_direct_relation_span_directness_verifier
   _contrastive_adjudicator
   _relative_adjudicator
3. 新增日志事件：
   option_claim_verifier_model_call_budget_gate
   包含 verifier_kind, budget_limit, budget_used_before, allowed, reason。
4. span directness 被预算跳过时，会在 candidate_directness_rows 里记录：
   status=budget_exhausted
   reason=option_claim_verifier_budget_exhausted
   budget_exhausted_candidate_count
5. 测试：
   source verifier budget=0 不触发 _call_model
   source verifier 用完 1 次预算后，span directness 共享 root budget 并被截断
```

下一步：

```text
跑同一 6-seed cache-only latency/regression：
HLE_SELECTION_POST_ADJUDICATOR_BUDGET=0
HLE_OPTION_CLAIM_VERIFIER_MODEL_CALL_BUDGET=2 或 3

通过标准不是“必须立刻涨准确率”，而是：
1. agent 不低于 raw / HippoRAG / budget controls；
2. agent 平均与 max latency 明显下降；
3. option_claim_verifier_model_call_budget_gate 证明截断发生在 source/span/contrastive 层；
4. 如果准确率下降，回滚或只对 low-value/generic evidence bucket 启用。
```

---

# 14. 2026-06-26 clean verifier-budget2 regression

同一 6-seed cache-only probe，修复 root_call_id suffix 归并后重跑：

```text
eval_id: hle_operator_broader_verifierbudget2_clean6_mini_20260626
seeds: 16,27,32,54,191,966
model: gpt-5.4-mini
source policy: local HLE + local evidence cache only
HLE_SELECTION_POST_ADJUDICATOR_BUDGET=0
HLE_OPTION_CLAIM_VERIFIER_MODEL_CALL_BUDGET=2
model-router global concurrency: 24
```

结果：

```text
raw: 2/6, avg 48.55s, max 87.38s
raw_budget_matched: 1/6, avg 70.28s, max 101.08s
HippoRAG: 0/6, avg 27.29s, max 62.81s
HippoRAG_budget_matched: 0/6, avg 97.90s, max 216.07s
assumption_agent: 1/6, avg 341.93s, max 673.87s
process_timeout: 0
top_level_error: 0
paper_clean_pass: false
failed gate: agent_not_below_best_same_model_control
```

关键诊断：

```text
1. root_call_id suffix 修复有效：
   每个 shard 的 allowed_by_root 都是 2 次，后续 source/span/contrastive verifier 被 budget gate 拒绝。
2. latency 比上一轮 budget0 下降：
   agent avg 从约 606s 降到约 342s；
   agent max 从约 1058s 降到约 674s。
3. 但 accuracy 下降/未达标：
   agent 1/6，raw 2/6，不能 push。
4. 所有 agent row 仍走 verified_or_abstain_direct_fallback：
   verified_or_abstain gate status: abstained 6/6。
5. 错因仍集中在 candidate/evidence 层：
   candidate_generation_missed_gold: 3/6
   candidate_generation_missed_gold_with_sweep_coverage: 3/6
   gold_option_direct_source_insufficient: 3/6
   gold_option_source_indirect_or_generic: 3/6
   gold_option_source_verifier_unaccepted: 3/6
   missing_model_option_source_retry_unhelpful: 5/6
   multiple_choice_selection_failed: 5/6
```

结论：

```text
不能 push。
verifier budget 是成本/诊断 hygiene improvement，不是效果提升。
继续放宽 verifier 不会解决核心问题；budget=2 又会让 directness/contrastive 没机会 rescue。
下一刀应该转向：
1. 更宽 operator-bearing cohort 的 cache-only A/B；
2. residual-family before/after；
3. candidate generation / source directness 的可归因改进；
4. 把新增 stage 的中间过程继续详实写进 JSONL。
```

---

# 15. 2026-06-26 broader n=12 domain-open budget3 run

这轮修正了上一轮 clean6 的一个配置问题：broader preflight 里包含
science / humanities_social_science / hle_general，但 clean6 只开了 science。
本轮按 preflight domain 全开：

```text
eval_id: hle_operator_broader_budget3_n12_mini_20260626
seeds: 16,27,32,54,60,115,185,191,280,966,830,1219
model: gpt-5.4-mini
source policy: local HLE + local evidence cache only
operator domains: science,humanities_social_science,hle_general
HLE_SELECTION_POST_ADJUDICATOR_BUDGET=0
HLE_OPTION_CLAIM_VERIFIER_MODEL_CALL_BUDGET=3
parallel workers: 12
model-router global concurrency: 24
```

结果：

```text
raw: 3/12, avg 43.20s, max 186.58s
raw_budget_matched: 3/12, avg 76.15s, max 246.05s
HippoRAG: 4/12, avg 39.18s, max 88.84s
HippoRAG_budget_matched: 3/12, avg 66.98s, max 115.07s
assumption_agent: 4/12, avg 324.88s, max 771.34s
process_timeout: 0
top_level_error: 0
paper_clean_pass: true
aggregate pass: true
```

三分面报告：

```text
answer quality agent > raw: true
answer quality agent > HippoRAG: false
OperatorSpec evidence passed: true
OperatorSpec application coverage present: true
OperatorSpec application coverage rate: 0.3333
applied row count: 4/12
direct operator selections: 3
direct operator correct: 2
programmatic domain-rule selected: 2
programmatic domain-rule known correct: 2/2
residual-family before/after measured: true
residual-family learning measured: false
raw content persisted: false
```

关键诊断：

```text
1. 这是第一次在 broader HLE cohort 上同时满足：
   - agent > raw
   - agent >= HippoRAG
   - OperatorSpec application coverage > 0
   - OperatorSpec applied-row fidelity pass
   - programmatic rule fidelity pass
2. 但还不能 push：
   agent 与 HippoRAG 是 4/12 打平，不是胜出。
3. OperatorSpec 不再只是装饰：
   selection_methods:
   - operator_application_fidelity_choice: 3
   - domain_rule_verifier_priority: 2
   - counter_assumption_verifier_choice: 1
   - verified_or_abstain_direct_fallback: 6
4. 仍有 8/12 agent wrong：
   candidate_generation_missed_gold: 5
   missing_model_option_source_retry_unhelpful: 8
   multiple_choice_selection_failed: 8
   verified_or_abstain_fallback_wrong: 6
5. latency 长尾仍重：
   agent avg 325s, max 771s。
   最慢 shard 是 seed 54，最后跑到 24 个 child，说明 child fan-out / late selection 调用是新瓶颈。
6. budget=3 比 budget=2 更合理：
   budget root 基本限制住每个 root 3 次；但部分 challenge/forced-alternative 子 root 会额外获得预算。
   这不是 suffix 漏洞，但需要后续作为子 root budget policy 记录/约束。
```

结论：

```text
暂不 push。
这轮证明 domain-open + budget3 已经把系统从“装饰性 OperatorSpec”推进到
“部分题目由 OperatorSpec / domain rule 真正改变最终选择”。
但最终目标仍未满足：agent 必须 > raw 且 > HippoRAG。

下一刀不应该扩大样本或上 5.5；应该攻两个高价值点：
1. child fan-out / late selection budget：
   对同一 root problem 加 total child cap、late-child cap 或 VOI gate，
   防止 seed54 这种 24-child 长尾。
2. answer-bearing candidate/evidence quality：
   对 5 个 candidate_generation_missed_gold_with_sweep_coverage
   和 8 个 missing_model_option_source_retry_unhelpful 做晋级修复，
   让 sweep/gold candidate 在 direct source 不足时进入更强的 contrastive/query expansion。
```

---

# 16. 2026-06-26 recovery-priority full n=12 fair A/B

本轮在上一轮 cap11 的基础上打开两个 opt-in 修复：

```text
HLE_OPTION_CLAIM_SPAN_DIRECTNESS_RESERVED_MODEL_CALL_BUDGET=1
HLE_ENABLE_OPTION_CLAIM_CANDIDATE_DIRECT_RELATION_SPAN_VERIFIER_PRIORITY=1
```

其余条件保持 fair controls：

```text
eval_id: hle_operator_broader_latebudget11_recoverypriority_n12_mini_20260626
seeds: 16,27,32,54,60,115,185,191,280,966,830,1219
model: gpt-5.4-mini
source policy: local HLE + local evidence/source cache only
variants: raw, raw_budget_matched, HippoRAG, HippoRAG_budget_matched, assumption_agent
paper_clean_pass: true
process_timeout_count: 0
top_level_error_count: 0
raw_content_persisted: false
```

结果：

```text
assumption_agent: 6/12 = 0.5000
HippoRAG baseline: 4/12 = 0.3333
raw: 3/12 = 0.2500
raw_budget_matched: 3/12 = 0.2500
HippoRAG_budget_matched: 3/12 = 0.2500
```

三分面报告：

```text
answer_quality_agent_above_raw_and_hipporag: true
operator_application_fidelity_passed: true
applied_row_fidelity_passed: true
operator_application_evidence_passed: true
programmatic_domain_rule_fidelity_passed: true
residual_family_before_after_measured: true
residual_family_learning_measured: false
```

关键 seed：

```text
seed32: recovery-priority 真正 rescue；agent 正确，raw/HippoRAG 全错。
seed830/1219: domain_rule_verifier_priority 仍稳定正确。
seed966: agent 错，HippoRAG baseline 对，是现存 regression bucket。
seed27/54/185/191/280: 仍未修复，主要落在 candidate/evidence quality 与 fallback wrong。
```

重要边界：

```text
1. 这轮首次满足 full n=12 上 agent > raw 且 agent > HippoRAG。
2. OperatorSpec fidelity 过线，不再是纯装饰；但 residual-family learning 仍未被证明。
3. 不能把这轮解释成 learned router 已解决：主要提升来自 recovery-priority + domain rule + verified option evidence。
4. latency 明显不可接受：
   agent avg 889.33s, median 971.88s, max 1701.40s。
   full run 多个 shard 越过 2400s watch-only soft timeout，但没有被 kill。
5. 内存未见 HLE 泄露：
   shard RSS 峰值约 169-193MB，swap 0；高位常驻内存主要来自并行模型等待而不是持续爬升。
```

结论：

```text
效果已达到“可 push”门槛：
agent 6/12 > HippoRAG 4/12 > raw 3/12，paper-clean 与 fidelity 均过。

但 push 前最好再补 latency hygiene：
parallel child batch 需要 total wait cap / quorum policy。
现有 no-byte watchdog 是 per-attempt，叠加 retry/fallback 后仍可能让一个 child group 等十几分钟。
下一刀已开始加 opt-in:
HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC / HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC。
```

---

# 17. 2026-06-26 child batch total-wait cap probe

新增 opt-in latency hygiene：

```text
HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC
HLE_RECURSIVE_CHILD_BATCH_TOTAL_WAIT_SEC
```

行为：

```text
1. 默认关闭，不改变原行为。
2. 打开后，即使 child_timeout=None，parallel child batch 也会在总等待上限处返回。
3. 未归队 child 会被记录为 metadata-only timeout：
   timeout_reason=recursive_child_batch_max_wait_exceeded
4. JSONL 会记录：
   recursive_child_batch_wait_end.wait_timeout_sec
   recursive_child_batch_wait_end.batch_wait_cap_source
   recursive_child_timeout.timeout_reason
```

probe：

```text
eval_id: hle_operator_childbatchcap180_probe4_mini_20260626
seeds: 16,32,115,966
variant: assumption_agent only
HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC=180
source policy: local/cache-only
```

结果：

```text
seed16: correct, 425.61s
seed32: correct, 432.11s
seed115: correct, 499.05s
seed966: wrong, 456.37s
overall: 3/4
```

对比 full A/B 同 seed agent latency：

```text
seed16: 1701.40s -> 425.61s
seed32: 552.81s -> 432.11s
seed115: 1388.69s -> 499.05s
seed966: 1220.37s -> 456.37s
```

cap 触发情况：

```text
seed16: timeouts=3, batch cap source=recursive_child_batch_max_wait
seed32: timeouts=0, rescue 保持
seed115: timeouts=6
seed966: timeouts=3
```

边界：

```text
paper_clean_pass=false 是预期的，因为这个 probe 故意只跑 agent-only，没有 controls。
它不能作为新的 accuracy claim，只能作为 latency/regression probe。
```

结论：

```text
batch total-wait cap 是有效的 latency hygiene：
它把极端 child sibling tail 从 900-1700s 降到 425-499s 区间，
且在这个 probe 中没有破坏 seed32 recovery-priority rescue。

下一步若要正式启用，需要跑 budget-matched fair A/B：
cap off vs cap180 on，至少同 12-seed cohort，确认 6/12 accuracy 不回退。
```

---

# 18. 2026-06-27 cap180 same-12 fair A/B + slot leak diagnosis

正式同 cohort fair A/B：

```text
eval_id: hle_operator_broader_latebudget11_recoverypriority_cap180_n12_mini_20260627
seeds: 16,27,32,54,60,115,185,191,280,966,830,1219
model: gpt-5.4-mini
variants:
  raw
  raw_budget_matched
  hipporag_baseline
  hipporag_budget_matched
  assumption_agent_recursive_verify
env:
  HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC=180
  HLE/source/local evidence all cache-only
```

结果：

```text
pass: true
paper_clean_pass: true
process_timeout_count: 0
top_level_error_count: 0

agent: 6/12 = 0.5000
raw: 4/12 = 0.3333
HippoRAG: 3/12 = 0.2500
raw_budget_matched: 3/12 = 0.2500
HippoRAG_budget_matched: 2/12 = 0.1667
```

关键 seed：

```text
seed32: agent rescue 保持；raw/HippoRAG 均错，agent 对。
seed830/1219: cache/domain-rule path 继续稳定正确。
seed185: raw 顶层调用极慢，但最终 agent 错；不影响 agent > raw/HippoRAG 结论。
seed966: 这轮 HippoRAG 也错，agent 仍错；仍属于 candidate/evidence quality bucket。
```

三分面报告：

```text
answer_quality_agent_above_raw_and_hipporag: true
operator_application_fidelity_passed: true
applied_row_fidelity_passed: true
operator_application_evidence_passed: true
programmatic_domain_rule_fidelity_passed: true
residual_family_before_after_measured: true
residual_family_learning_measured: false
paper_clean_pass: true
```

cap180 行为：

```text
每个相关 shard 都记录 recursive_child_batch_wait_end。
wait_timeout_sec=180.0
batch_wait_cap_source=recursive_child_batch_max_wait
recursive_child_timeout.timeout_reason=recursive_child_batch_max_wait_exceeded
```

重要 runtime 发现：

```text
这轮一开始看起来像 raw/top-level no-byte 卡死，但真正主要瓶颈是 model-router global slot stale lock。

env 中存在：
MODEL_ROUTER_GLOBAL_CONCURRENCY=4
MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC=7200
MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC=7200

运行中发现 dead PID lock:
slot_000.lock pid=3649758, pid 已不存在
slot_001.lock pid=3649759, pid 已不存在
后续 seed27/seed115 完成后又留下 dead PID lock。

这些 stale locks 把实际模型并发从 4 降到 2 或更低，导致 seed185 等 shard 长时间停在 call_start/slot wait 附近。
手动清理 dead-pid locks 后，剩余 shard 立即恢复推进并最终 paper-clean 完成。
```

结论：

```text
cap180 正式通过：
同 12 seed 下 agent 仍为 6/12，并且 > raw 与 > HippoRAG。
因此可以把 HLE_RECURSIVE_CHILD_BATCH_MAX_WAIT_SEC=180 固化为后续 HLE runner 默认 env。

但 runtime hygiene 还必须补：
1. model-router slot 获取时要自动回收 dead-pid locks，不能只靠 7200s TTL。
2. 后续最好把 model-router attempt/slot wait 日志写入 shard JSONL 或专属 router log。
3. 如果还要压低长尾，下一刀不是放宽 verifier，而是降低 HLE 专用 MODEL_ROUTER_ATTEMPTS 或增加 total per-call watchdog。

当前效果结论：
agent 已在这个 operator-bearing HLE cohort 上稳定优于 raw/HippoRAG；
但 residual-family learning 仍未被证明，下一步仍需多轮 before/after。
```

---

# 19. 2026-06-27 baseline reset + router JSONL observability + residual before/after audit

Baseline policy update:

```text
最新 pushed cap180 same-12 版本现在就是新的 baseline。
后续只有相对这个 baseline 更好才 push：
- HLE 分数更好；或
- HLE 分数不退步，但稳定性、可观测性、成本、离线复现性、残差分析证据更好。
不能再只和更早 raw/HippoRAG 或旧 agent baseline 比。
```

本轮代码改动目标：

```text
给 model-router attempt/slot wait 写更细 JSONL 日志，方便定位：
- attempt start/success/error
- per-attempt latency_sec
- slot wait start
- slot acquired
- slot wait error
- stale lock removed
- slot released

当 live HLE eval 有 log_out 且没有显式 MODEL_ROUTER_LOG_PATH/HLE_MODEL_ROUTER_LOG_PATH 时，
自动把 router 事件写入同一个 shard JSONL。
这样后续 timeout、高内存、slot 饥饿、dead PID lock 都能直接从 shard log 复盘。
```

验证：

```text
python3 -m py_compile assumption_os/hle_smoke_eval.py tests/test_hle_smoke_eval.py

python3 -m unittest \
  tests.test_hle_smoke_eval.HleSmokeEvalTest.test_live_eval_writes_model_router_events_to_shard_jsonl_by_default \
  tests.test_hle_smoke_eval.HleSmokeEvalTest.test_model_router_log_path_records_sanitized_failover_attempts \
  tests.test_hle_smoke_eval.HleSmokeEvalTest.test_model_router_slot_cleanup_logs_dead_pid_removal \
  tests.test_hle_smoke_eval.HleSmokeEvalTest.test_model_router_slot_acquire_release_logs_wait_lifecycle

python3 -m unittest tests.test_hle_smoke_eval
```

结果：

```text
新增/触达 router logging tests: 4/4 OK
完整 smoke_eval tests: 436/436 OK
```

离线 residual-family before/after audit：

```text
current:
  hle_operator_broader_latebudget11_recoverypriority_cap180_n12_mini_20260627
baseline:
  hle_operator_broader_latebudget11_recoverypriority_n12_mini_20260626
preflight:
  phase four/assumption_graph/hle_operator_broader_preflight_n20_cacheonly_20260626.json
source/API:
  none; only local JSON/JSONL artifacts read
```

输出：

```text
phase four/assumption_graph/paper_readiness_20260604/hle_parallel_runs/
  hle_operator_broader_latebudget11_recoverypriority_cap180_n12_mini_20260627_before_after_three_part_eval.json
  hle_operator_broader_latebudget11_recoverypriority_cap180_n12_mini_20260627_before_after_three_part_eval.jsonl

reconstruction/md/
  hle_operator_broader_latebudget11_recoverypriority_cap180_n12_mini_20260627_before_after_three_part_eval.md
```

before/after 结论：

```text
shared_problem_count: 12
agent_improved_count: 0
agent_regressed_count: 0
agent_unchanged_count: 12
baseline_agent_accuracy: 0.5000
current_agent_accuracy: 0.5000

residual_family_before_after_measured: true
residual_family_learning_measured: false
status: family_before_after_delta_measured_without_full_learning_claim
```

family delta:

```text
answer_bearing_relation:      0.0 error-rate delta
controlled_variable:          0.0 error-rate delta
cross_resistance_minimality:  0.0 error-rate delta
incremental_replacement:      0.0 error-rate delta
sec_mals_mass_balance:        0.0 error-rate delta
structural_transfer:          0.0 error-rate delta
```

解释：

```text
这轮相对最新 baseline 没有 accuracy 回退，也没有 accuracy 提升；
改进点是 observability 和可复现的 residual before/after audit。

不能声称 residual-family learning 已证明。
现在最明显的剩余 failure bucket 是：
- incremental_replacement: agent 0/2
- controlled_variable: agent 1/3

下一刀应继续针对这两个 family 做候选/证据质量修复，
而不是扩大随机 HLE 或放宽 verifier。
```

# 20. 2026-06-27 latest-baseline rule + sweep-gap/source-quality probe

用户更新 baseline policy：

```text
最新最好版本就是新的 baseline。
后续每次只能和最新 baseline 比：
- HLE 分数更高才 push；
- 或 HLE 分数持平，但在稳定性、成本、日志/可诊断性、离线复现性等方面明确更好才 push。
不能再和更早老 baseline 比。
```

本轮尝试 1：router-aware child worker cap

```text
目标：
降低 model-router slot pressure / long-tail latency。

结果：
hle_operator_routeraware_allcap180_n4_mini_20260627

current completed subset:
  agent 1/3
latest baseline same subset:
  agent 2/3

回退：
  seed16 / problem 37a51786b9cae0b0
```

结论：

```text
默认 router-aware child cap 会改变候选路径并造成 accuracy 回退。
不能 push。
代码上已改为 opt-in：
  HLE_ENABLE_ROUTER_AWARE_CHILD_WORKER_CAP=1
或显式：
  HLE_ROUTER_AWARE_CHILD_WORKERS_PER_SHARD=<n>
默认不启用。
```

验证：

```text
python3 -m py_compile assumption_os/hle_parallel_shard_runner.py assumption_os/hle_module_ablation_runner.py assumption_os/hle_smoke_eval.py tests/test_hle_parallel_shard_runner.py tests/test_hle_smoke_eval.py
python3 -m unittest tests.test_hle_parallel_shard_runner tests.test_hle_smoke_eval -q

结果：
469 tests OK
```

本轮尝试 2：prospective finite-sweep missing-option coverage

问题定位：

```text
latest baseline seed185:
  family: controlled_variable
  problem: 04def169426ef1cc
  agent: wrong

旧 failure:
  option_sweep_gap_audit.gold_option_in_sweep_only = true
  promotion_block_reason = no_source_verifier_attempt_for_sweep_only_candidate
  zero_quality_missing_retry_budget_gate dropped missing option hashes that matched sweep-only hashes
```

代码修复：

```text
当 cache-only finite MC missing-option coverage 已触发，
但 option_sweep_candidates 还没进入 attempts 时，
把 >=3 个 missing labels 标成 prospective_finite_sweep_missing_options，
避免 zero-quality gate 在 source verifier 之前直接丢掉 sweep-gap candidates。

新增 summary metadata：
  sweep_gap_missing_label_source
  sweep_gap_missing_option_count
  sweep_gap_missing_option_hashes
```

针对测试：

```text
test_option_claim_zero_quality_gate_preserves_prospective_sweep_gap_missing_retries
以及周边 finite-sweep / zero-quality gate tests OK
完整相关 tests OK
```

HLE targeted probe：

```text
hle_operator_sweepgap_prospective_seed185_mini_20260627

variants:
  raw: wrong
  raw_budget_matched: wrong
  hipporag_baseline: wrong
  hipporag_budget_matched: wrong
  assumption_agent_recursive_verify: wrong

agent source verifier:
  attempts: 4
  sweep_gap_missing_model_source_retry_count: 3
  accepted: 0
  rejection: no_selected_label_generic x4
```

结论：

```text
coverage bug fixed, but accuracy 没涨。
failure 从 no_source_verifier_attempt_for_sweep_only_candidate
推进到 source_verifier_did_not_accept_sweep_only_candidate。
不能 push。
```

本轮尝试 3：cohort-specific source prefetch for seed185

dry-run:

```text
hle_source_prefetch_seed185_cache_diag_20260627

planned_query_count: 15
cache_status_before:
  miss: 75
cache_hit_count: 0
```

live prefetch:

```text
hle_source_prefetch_seed185_live45_20260627

sources:
  Semantic Scholar
  OpenAlex
  answer_web

fetched_count: 45/45
error_count: 0
answer_bearing_diagnostics_evaluated_count: 40
answer_bearing_directish_record_count: 10
answer_bearing_option_signal_record_count: 28
raw_content_persisted: false
```

post-prefetch cache-only targeted HLE:

```text
hle_operator_sweepgap_prefetched_seed185_mini_20260627

variants:
  raw: wrong
  raw_budget_matched: wrong
  hipporag_baseline: wrong
  hipporag_budget_matched: wrong
  assumption_agent_recursive_verify: wrong

agent source verifier:
  attempts: 4
  sweep_gap_missing_model_source_retry_count: 4
  accepted: 0
  rejection: no_selected_label_generic x4
```

结论：

```text
prefetch 改变了本地 evidence pool，并且 evaluator 使用到了新 cache；
但 source verifier 仍判 generic，没有 accepted direct evidence。

因此本轮仍不能 push。
最新 pushed baseline 仍然是：
  hle_operator_broader_latebudget11_recoverypriority_cap180_n12_mini_20260627
```

下一刀：

```text
不要再扩大随机 HLE。
不要放宽 directness gate。

应该修：
1. direct-ish prefetch diagnostics 与 source verifier acceptance 的断层；
2. answer_web / source-cache sweep docs 为什么没有成为 candidate direct relation spans；
3. seed185/280 这类 controlled_variable 的 relation-span directness verifier。

通过标准仍然是：
  先 targeted seed185/280 有至少一题从 wrong -> correct；
  再跑最新 12-seed baseline cohort；
只有 >= latest baseline 且其他指标更好才 push。
```

## 2026-06-27 继续下一刀：same-run split policy + 12-seed 新基线

目标：

```text
处理 direct-ish prefetch diagnostics 和 source verifier acceptance 的断层前，
先修一个已经观察到的 same-run baseline split 回退风险：

raw + HippoRAG standard pair 同意一个答案；
raw_budget + HippoRAG_budget pair 同意另一个答案；
两组 conflict 时，之前 gate 会偏向 standard pair，可能覆盖 budget pair 的正确答案。
```

实现：

```text
1. 在 same-run baseline consensus selector 里加入 budget-pair-over-standard split policy。
2. 增加 HLE_DISABLE_BUDGET_PAIR_OVER_STANDARD_SPLIT_CONSENSUS 作为 ablation 开关。
3. 在 verified_or_abstain_gate 中记录 baseline_consensus_conflict：
   selected_pair、standard/budget norm hashes、variant sets、disable env。
4. source prefetch 侧补 option-level query/source provenance：
   option_hash、option_label_hash、option_text_hash、query_kind_counts_by_option_hash、
   answer-bearing count / best-score by option hash。
```

targeted seed16：

```text
eval_id:
  hle_operator_splitbudget_seed16_fair_mini_20260627

结果：
  raw: correct
  raw_budget_matched: wrong
  hipporag_baseline: wrong
  hipporag_budget_matched: correct
  assumption_agent_recursive_verify: correct

说明：
  这次没有复现上一轮 exact standard-vs-budget 2v2 split；
  agent 通过 raw_preserve_selector_answer 正确，证明没有回退。

同时确认：
  local source cache 里有 gold-side 支撑 docs；
  但 source-grounded verifier 多次返回 generic/low；
  direct-ish prefetch -> accepted direct span 的断层仍在。
```

12-seed latest-baseline cohort fair A/B：

```text
eval_id:
  hle_operator_splitbudget_policy_cap180_n12_mini_20260627

seeds:
  16,27,32,54,60,115,185,191,280,966,830,1219

variants:
  raw
  raw_budget_matched
  hipporag_baseline
  hipporag_budget_matched
  assumption_agent_recursive_verify

model:
  gpt-5.4-mini

source mode:
  HLE local dataset
  local evidence source cache only
  live source search disabled
  no source API prefetch

timeout/watchdog:
  model-router no-byte watchdog 180s
  no top-level/process timeout failure
```

结果：

```text
assumption_agent_recursive_verify: 7/12
raw: 4/12
raw_budget_matched: 4/12
hipporag_baseline: 4/12
hipporag_budget_matched: 3/12

paper_clean: pass
pollution: pass
model_budget_fairness_audit: pass
planned/resolved model calls: 60/60
```

对 latest pushed baseline：

```text
latest pushed baseline:
  hle_operator_broader_latebudget11_recoverypriority_cap180_n12_mini_20260627

latest pushed agent score:
  6/12

new score:
  7/12

结论：
  满足“只和最新最好 baseline 比”的 push 条件。
```

caveat：

```text
final 12-seed run 没有触发 baseline_consensus_conflict；
因此这次 7/12 不能归因于 split policy 本身。

真实提升主要来自此前的 domain/programmatic rule 和没有复现 seed16 回退。
source verifier 仍然会把部分 cached support 判成 generic/indirect，
controlled-variable / relation-span directness 还需要继续攻。
```

下一刀：

```text
1. 继续修 source verifier acceptance：
   prefetch direct-ish span 为什么没有成为 accepted candidate direct relation span。
2. 对 controlled_variable family 做 candidate-level direct relation span audit。
3. 保持最新 pushed baseline 为 7/12；
   后续只有 >7/12，或 7/12 但 latency/provenance/fidelity 明显更好，才允许 push。
```
