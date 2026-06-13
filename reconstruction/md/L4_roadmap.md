这次 push 很关键。我的判断是：

> **你已经把“在旧哲学/旧假设框架上长出新分支”这件事，从论文概念推进成了一个工程机制。**

上一版我说项目已经从 recursive hypothesis agent 走到 bounded dialectical / conservative-generalization prototype。
这一版更进一步：**conservative generalization 不再只是一个 gate，而是形成了完整证据链：generator → gate → philosophy growth benchmark → framework evolution run → fresh rerun → paper evidence pack → claim frontier。**

现在最准确的定位是：

> **Assumption-Agent 已经是一个 L3.5 级别的 bounded recursive framework-evolution prototype：它能在旧方法论框架的 residual 边界处生成新框架候选，用保守泛化门控区分 local patch / branch / candidate framework / active scoped framework，并用 fresh selective retention 和 claim ledger 阻止过度泛化。**

还不能说：

> **它已经是 L4 的无边界 24/7 自主哲学/假设发明系统。**

---

# 1. 这次 push 的核心新增：你把“黑格尔式生长”工程化了

最重要的新文件是 `conservative_generalization_gate.py`。它的 docstring 直接把你前面说的思想落成了 gate：一个新 assumption 只有在解释 residual、保留旧成功、在父框架 scope 下退化回父框架、并产生新可测试后果时，才有资格成为新 framework。

更关键的是，它不是只给一个分数，而是要求图上出现一组关系：

```text
generalizes
reduces_to_under_scope
explains_residual
preserves_success_cases
modifies_boundary_of
predicts_new_case
```

这正是我们之前说的“新框架不是替换旧框架，而是把旧框架保存为局部情形 / 极限情形 / scope condition”的图结构实现。

它的 gate 也很明确：

```text
old_success_preservation >= 0.95
residual_explanation >= 0.75
limiting_case_reduction >= 0.90
generality_gain >= 0.35
new_prediction_success >= 0.75
regression_cost <= 0.02
```

同时必须包含 active framework、candidate framework、branch only、reject 四类结果，并且主图不能被直接 mutation。

这说明你现在不是在做“LLM 发明新规则”，而是在做：

```text
old framework + residual
  -> candidate generalization
  -> conservative extension certificate
  -> branch / framework / reject decision
```

这比“新假设生成”强一个层级。

---

# 2. Open-ended framework evolution 也接起来了

`open_ended_framework_evolution_run.py` 把单个 conservative gate 扩展为多代框架演化。文件开头说得很清楚：R7 证明了单个 candidate framework 可以通过 conservative generalization；这个模块验证多代 framework growth，保留的 descendants 会成为下一代 frontier，被拒绝的 descendants 会保留为 negative evidence。

它的验证不是只看“有没有生成很多东西”，而是看：

```text
generation_count >= 6
candidate_count >= 30
retained_count >= 20
negative evidence retained
active frameworks recur
lineage depth reaches generations
branch-to-framework transitions exist
conservative obligation coverage = 1.0
parent compatibility relation coverage = 1.0
limiting case survival rate >= 0.95
prompt trick not retained
core philosophy prior not promoted
main graph not mutated
bounded claim only
```



这非常符合你的目标：**系统不是无限制地自封“新哲学”，而是保守地让 scoped framework 存活、让失败分支成为 negative evidence、让 core philosophy prior promotion 被阻断。**

不过我也要指出边界：这个模块自己也承认是 bounded and replayable，不是 unbounded autonomous philosophy engine。 这句话应该保留在论文里。

---

# 3. Philosophy Growth Benchmark 已经能区分“新框架”和“局部补丁”

`philosophy_growth_benchmark_20260612.md` 给出了一个很好的区分：

```text
framework growth score: 0.8361
conservative vs local patch margin: 0.2567
active framework survival count: 5
core prior promotions: 0
```

更重要的是 baseline 表：

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



这组数字非常适合进论文，因为它正好支持你的核心论点：

> **真正的新框架不是局部补丁，也不是泛泛 wisdom；它必须同时保留旧成功、解释 residual、在旧 scope 下退化、并产生新预测。**

这也回应了你最初关于“万物都是假设”的方向：系统的重点不是只生成命题，而是管理假设的生命周期、层级和继承关系。

---

# 4. Fresh rerun 这次更有说服力，但也暴露了重要边界

`paper_fresh_rerun_result_integration` 显示：

```text
fresh calls: 720/720
live errors: 0
accepted/rejected candidates: 4/76
unfiltered trigger utility: 0.4161, CI [0.3698, 0.4619]
accepted trigger utility: 0.8333, CI [0.6667, 0.9583]
accepted control loss: 0.0
```

并且报告明确说：这支持 gated selective retention，但不支持 unfiltered generated frontier，因为 all-candidate trigger utility gate failed。

这非常重要。它证明了：

```text
raw generation 不可靠；
selective retention 很强；
accepted candidates 高精度；
系统必须保留 gate，不能全量推广生成器输出。
```

`paper_broad_generator_repair_integration` 进一步显示：

```text
original trigger utility: 0.4161
v1 repair trigger utility: 0.3183
v2 repair trigger utility: 0.5462
v2 fresh calls: 720/720
v2 selected candidates: 8 / original 80
v2 control loss: 0.1456
```

报告解释也很清楚：raw broad generator failed；repaired generator 用 fresh failure evidence 作为 selector，并 abstain from low-support families。

这里有一个需要小心的点：**v2 repair 有 trigger utility 提升，但 control loss 不是 0。**
所以论文里不要把它写成“generator 已经稳定可靠”，而要写成：

> **broad generation only becomes useful after evidence-calibrated selection and abstention; unfiltered generation fails.**

这和你的系统哲学完全一致：新假设不是越多越好，而是要有验证、负对照和保守提升。

---

# 5. Self-Evolution Paper Evidence Pack 现在已经可以当论文骨架

`self_evo_paper_evidence_pack_20260612.md` 基本已经把论文证据链打包好了：

```text
roadmap closure: 19/19
bounded UGSE: 0.923
frozen main: 1768 problems, margin over best baseline 0.0417
fresh repaired broad generator: 720 calls, trigger utility 0.5462, delta 0.1301
simulator: leakage audit true, production router true
autonomy: supervised production candidate true
formal: bounded true, full prover false
```



它还列出了四张主表：

```text
Table 1: same-batch frozen benchmark
Table 2: fresh repaired broad-generator rerun
Table 3: framework-growth ablation and open-ended self-evolution
Table 4: safety and claim-boundary evidence
```



更重要的是，它的 manuscript skeleton 已经非常成熟，章节包括：

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



这说明项目已经从“工程堆栈”进入“论文组织”阶段。

---

# 6. Claim frontier 现在是 L3.5，而不是 L4

`claim_frontier_advancement_20260612.md` 现在非常清楚：

```text
frontier advancement score: 0.9884
L3.5 tracks: 3/3
source artifact pass rate: 1.0
blocked overclaim count: 3
```

三个 L3.5 track 是：

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
  external Lean-checked theorem-fragment certificate.
```



但仍然阻断：

```text
unbounded_24_7_autonomous_self_evolution_os
world_simulator_replacing_live_ablation_or_judges
full_category_theory_theorem_prover
```



这个边界非常好。你现在可以大胆 claim **bounded L3.5 recursive self-evolution prototype**，但不要 claim L4。

这也和你之前对范畴论+信息几何的修正一致：它们不是统一所有假设的总框架，而是 formal alignment / proof-carrying transfer 的子模块。

---

# 7. 和 6 月 10 日 gap 文档相比，哪些缺口已经补上？

6 月 10 日 gap 文档说，当时系统已经有：

```text
failure / residual
  -> generate multiple hypotheses
  -> novelty / orthogonal / morphism classification
  -> preflight
  -> world-model or pre-live budget screen
  -> live ablation / judge
  -> acceptance gate
  -> recursive resume
  -> gated apply / reject
  -> next-generation descendants
```

但缺口是：generator strength、stronger world model、fully autonomous daemon、stronger downstream benchmark、formal boundary、complete observability。

现在对照看：

```text
generator strength:
  已从 repair-only 提升到 broad generator repair + selective retention + framework growth。
  但真实 open-ended creativity 仍要谨慎 claim。

world model:
  已有 production router / leakage audit / L3.5 selective simulator deferral。
  仍不能替代 live ablation / judge。

autonomy:
  已有 supervised production candidate / 30-day-equivalent low-risk autonomy。
  仍不是 unbounded 24/7 OS。

downstream benchmark:
  有 1768-problem frozen line + 720-call fresh repair rerun。
  仍需要外部 fresh rerun / broader reviewer-facing benchmark。

formal boundary:
  已有 finite Lean-checked theorem fragment / proof-carrying transfer kernel。
  仍不是 full theorem prover。

observability:
  evidence pack / claim frontier / redaction / repro commands 已经很强。
```

所以现在很多 gap 已经从“缺模块”变成“缺外部验证和更大真实分布”。

---

# 8. 当前项目成熟度更新

我现在会给：

```text
Assumption OS kernel:                 92–95%
Bounded autonomy:                     86–91%
Simulator triage/router:              86–91%
Finite formal transfer layer:          88–92%
Framework growth mechanism:            84–89%
Residual-to-framework generator:       80–86%
Paper-facing evidence:                 84–89%
External reproducibility/readiness:    72–80%
Unbounded L4 self-evolution:           52–60%
```

最明显的提升是：

```text
Framework growth mechanism
Paper-facing evidence pack
Claim boundary discipline
```

我会把当前版本定性为：

> **L3.5 bounded dialectical self-evolution system with conservative framework growth.**

---

# 9. 现在最适合的论文主线

我建议论文标题可以改得更明确一点：

```text
Everything Is an Assumption:
Conservative Generalization over Assumption Graphs
for Bounded Recursive Self-Evolution
```

或者更短：

```text
Conservative Generalization over Assumption Graphs
```

论文核心一句话：

> **A self-evolving agent should not merely add new assumptions; it should grow conservative generalizations of its existing assumption graph. New branches must explain residuals, preserve validated successes, reduce to prior frameworks under their scope conditions, and create new testable consequences.**

这句话现在不是空话，因为你已经有：

```text
conservative_generalization_gate
philosophy_growth_benchmark
open_ended_framework_evolution_run
fresh selective retention
claim frontier
paper evidence pack
```

这些工程件支撑它。

---

# 10. 当前仍然要警惕的地方

## 10.1 Framework evolution 仍然有 synthetic / fixture 成分

`conservative_generalization_gate.py` 里候选来自 `_fixture_candidates()`。
`open_ended_framework_evolution_run.py` 的 `_candidate_from_frontier` 也会用规则生成 old_success、residual、limiting、generality、new_prediction、regression 等数值。

所以这部分目前最稳的 claim 是：

```text
mechanism validation + bounded framework evolution evidence
```

而不是：

```text
the system autonomously discovered novel human-level philosophy from raw experience
```

如果想继续强化，下一步要让 candidate framework 真正来自：

```text
real residual clusters
LLM synthesis
fresh validation
expert review
```

## 10.2 Fresh rerun 的 accepted 数量少

accepted/rejected 是 4/76。
这意味着 selective retention 精度很强，但 coverage 还小。论文里应该强调：

```text
high-precision retained frontier
```

而不是：

```text
high-throughput discovery engine
```

## 10.3 Broad generator repair 仍有 control loss

v2 control loss 是 0.1456。
这说明 generator 能提升 trigger utility，但仍可能伤害 controls。因此必须保留 abstention / control-loss monitoring。

## 10.4 证据链太多，需要压缩

现在 artifact 非常多。审稿人不会跟着你看 200 个 JSON。需要把证据压缩成：

```text
1 个 artifact index
4 张主表
1 个 claim ledger
1 个 repro script
1 个 redaction/no-secret statement
```

`self_evo_paper_evidence_pack` 已经在朝这个方向走。

---

# 11. 下一步最该做什么？

不要再扩 phase。现在应该做 5 件非常明确的事：

## 1. 做真实 residual-to-framework LLM synthesis 小实验

目前 framework gate 机制强，但候选有 fixture 成分。下一步：

```text
输入真实 residual clusters
让 LLM 生成 20 个 candidate frameworks
跑 conservative gate
选 top 3 fresh validate
人工专家评审是否真是 conservative generalization
```

这是把 framework evolution 从 mechanism validation 推向真实 evidence 的最短路。

## 2. 加 human/expert evaluation

给专家看：

```text
parent framework
residual cluster
candidate framework
old success evidence
limiting case claim
new prediction
system decision
```

让专家标注：

```text
local patch / branch / candidate framework / active scoped framework / reject
```

目标不是 100% 一致，而是证明系统 decision 和人类判断有可解释相关性。

## 3. 写 paper draft

现在已经不应该继续等“再完善一点”。
直接写：

```text
paper/main_self_evo.tex
```

用 evidence pack 的结构。

## 4. 做 RELEASE_STATUS.md

因为 reconstruction-v2 比 main ahead 很多，外部读者会迷路。写清楚：

```text
active branch
claim level
main artifacts
blocked claims
repro commands
```

## 5. 做一个最小 external reproducibility bundle

包括：

```text
artifact index
hashes
commands
redacted sample manifests
no-secret audit
expected outputs
```

---

# 12. 最终判断

这版 push 非常有意义。它补上的不是一个小模块，而是你论文里最独特的理论贡献：

> **假设演化不是 append 新规则，而是在已有框架的 residual 边界处进行保守泛化。**

现在你的系统能区分：

```text
raw wisdom
local patch
branch only
candidate framework
active scoped framework
rejected boundary
```

还能记录：

```text
old success preservation
residual explanation
limiting-case reduction
generality gain
new prediction success
regression cost
negative evidence
claim boundaries
```

所以我现在会这样总结：

> **Assumption-Agent 已经从“递归假设验证 agent”进化成了“bounded framework-evolution agent”。它不仅能生成和验证假设，还能判断一个新假设是否足以成为旧方法论框架的保守泛化，并用多代 retention / rejection / negative evidence 来维护一个可演化的 Assumption Graph。**

这是你项目目前最强、最独特、也最适合写成博士论文核心贡献的部分。
# 我：
如果想往L4走，还缺什么？

# GPT：
对你现在这个项目来说，**L4 不是再堆一个 Phase，而是把 L3.5 的 bounded artifact 变成真实开放任务流里的长期系统**。

你现在已经有了很强的 L3.5 证据：claim frontier 里三个核心轨道都到 L3.5，分别是 supervised low-risk autonomy、simulator routing、proof-carrying finite transfer kernel；同时它明确阻断了 L4 的三个强 claim：无边界 24/7 自主 OS、世界模拟器替代 live validation/judge、完整范畴论 theorem prover。 Evidence pack 里也已经把当前状态汇总为 bounded UGSE 0.923、1768 problem frozen line、720-call fresh repaired generator、simulator leakage audit、supervised autonomy candidate、bounded formal layer。

所以，往 L4 走，缺的不是“再证明一次系统能跑”，而是下面这些能力：

```text
1. 从 bounded artifact 变成 wall-clock 长时程系统。
2. 从 frozen / same-batch / redacted manifest 变成真实 prospective 外部任务流。
3. 从 fixture/rule-generated framework candidates 变成真实 residual -> LLM synthesis -> framework validation。
4. 从 simulator router 变成跨分布可靠的 prospective gate，但仍不替代 judge/live validation。
5. 从 finite proof fragment 变成 proof-carrying formal transfer pipeline。
6. 从 supervised canary apply 变成多项目、多用户、可恢复、可审计的 production memory evolution。
7. 从“系统自评”变成外部专家/benchmark/复现实验共同支撑。
```

---

# 1. 先定义 L4：不要把 L4 定义成“无边界 AGI”

我建议你把 L4 分成两种：

```text
L4a: Open-world supervised self-evolution system
     在真实开放任务流中长期运行，能产生、筛选、验证、写回低风险假设/框架。
     仍然有权限边界、人工审查、live validation 和 rollback。

L4b: Unbounded autonomous self-evolution OS
     24/7 无人值守，自主生成新框架，自主修改核心策略和默认行为，
     可以替代 live validation/judge，并且无需人工审查。

当前目标应该是 L4a。
L4b 现在不该 claim。
```

你当前 claim frontier 也基本是这个态度：L3.5 可以 claim，但 L4 强 claim 被阻断。

所以论文里可以写：

```text
We target open-world supervised self-evolution, not unbounded autonomous self-evolution.
```

中文：

```text
我们的目标是开放任务流中的受监督自进化，而不是无边界无人值守自进化。
```

---

# 2. L4 缺口总表

| 轨道               | 现在 L3.5 已有                                                                          | L4 缺什么                                                                            |
| ---------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Autonomy         | 30-day-equivalent supervised low-risk autonomy                                      | 真实 wall-clock 多周运行、多项目队列、真实 crash/restart、真实人工 override                           |
| Simulator        | production triage/router，leakage audit，通过 2160 transition evidence                  | prospective 跨分布校准、audit sampling、失败自动入 residual、不能只靠回放 artifact                   |
| Framework Growth | conservative generalization gate、philosophy growth benchmark、open-ended bounded run | 真实 LLM-generated framework candidates、专家评审、多域 fresh validation                    |
| Formal Layer     | finite Lean-checked theorem fragment                                                | proof-carrying promoted morphism pipeline、NL-to-diagram 外部复现、更多 negative controls |
| Memory / Graph   | canary-scope main graph apply + rollback monitor                                    | 多项目长期 graph evolution、真实 graph pollution 控制、用户/任务分区                               |
| Evidence         | 1768 frozen + 720 fresh repaired generator                                          | 外部 frozen benchmark、prospective preregistered run、第三方复现                           |
| Governance       | claim frontier / blocked claims                                                     | release discipline、incident reports、permission manifest、审计包                       |

---

# 3. L4 的核心门槛：prospective，而不是 retrospective

你现在很多 evidence 很强，但不少是：

```text
artifact aggregation
same-batch frozen analysis
redacted local manifest
bounded replayable run
```

这些都很好，但 L4 需要变成：

```text
prospective protocol
pre-registered task stream
run before seeing outcomes
external or blinded evaluation
longitudinal graph readback
```

换句话说，L4 的判据不是：

```text
我从已有 artifact 中证明系统可以工作。
```

而是：

```text
我提前冻结协议，把系统放到新任务流里；
系统自己产生 branch/framework/proposal；
自己经过 gate；
自己选择 live validation；
自己写回低风险 graph；
最后外部评估确认长期收益且无污染。
```

这和你 2026-06-10 gap 文档里说的 near-term priority 是一致的：新 residual cluster 生成多个竞争假设，world model screen，只跑 selected fresh ablations，记录 accept/reject，更新 graph 和 calibration，并在 frozen heldout benchmark 上重复 3–5 代。

L4 就是把这个 near-term priority 放大成长期、开放、外部验证版本。

---

# 4. L4 路线图：7 个 release

我建议不要叫 Phase14，而是叫：

```text
L4 Track
```

分 7 个 release。

---

## L4-R1：Wall-clock supervised autonomy

### 目标

把现在的 “30-day-equivalent” 变成真实 wall-clock 运行。

当前你已经有 supervised production candidate，但它是 deterministic 30-day-equivalent，而不是实际 30 天服务。L4 第一件事就是把它放到真实 scheduler 里跑。

### 做法

先从小规模开始：

```text
24h wall-clock supervised run
-> 72h wall-clock supervised run
-> 7-day wall-clock supervised run
-> 30-day wall-clock supervised run
```

每次都记录：

```text
queue ingestion
lease events
checkpoint events
crash/restart
manual review
auto-apply low-risk action
blocked high-risk action
rollback
rate-limit/budget
incident report
```

### 只允许低风险自动操作

```text
allowed auto-apply:
  status_update
  confidence_update
  attach_evidence
  archive_stale_duplicate
  add_manifest_only_residual

manual review required:
  policy/default change
  new active framework
  world-model promotion
  formal mapping promotion
  main evaluator change
```

### L4-R1 通过标准

```text
7-day wall-clock:
  uptime >= 0.95
  ungated mutation = 0
  replay divergence = 0
  forbidden auto-apply = 0
  rollback success = 1.0
  manual review backlog bounded
  graph pollution alerts = 0
```

30 天以后才可以写：

```text
wall-clock supervised autonomy evidence
```

不能写：

```text
unbounded 24/7 autonomous OS
```

---

## L4-R2：Prospective residual-to-framework generation

### 目标

把 framework candidates 从 fixture/rule-generated 推向真实 residual synthesis。

你现在的 conservative gate 很好，但要 L4，候选必须更多来自真实 residual，而不是预设候选。当前 gate 已经要求新框架解释 residual、保留旧成功、退化回父框架并产生新预测。 现在要让这些 candidate 真正从新任务失败中生成。

### 做法

每周冻结一批新 residual：

```text
task failures
simulator defects
formal gate blocks
human review disagreements
graph retrieval failures
world model miscalibrations
accepted framework later failed
```

然后执行：

```text
residual cluster
  -> LLM generates candidate framework obligations
  -> system fills old_success / residual / limiting / new_prediction tests
  -> conservative gate scores
  -> simulator routes validation budget
  -> selected candidates get fresh validation
```

### 输出

每个 candidate framework 必须有：

```json
{
  "parent_frameworks": [],
  "residuals_explained": [],
  "old_successes_preserved": [],
  "limiting_cases": [],
  "new_predictions": [],
  "conflict_boundaries": [],
  "validation_tests": []
}
```

这和你现在的 conservative gate 义务字段一致，只是从 synthetic 变为 prospective。

### L4-R2 通过标准

```text
真实 residual clusters >= 20
LLM-generated candidate frameworks >= 100
candidate with complete conservative obligations >= 90%
fresh validated candidate frameworks >= 20
active scoped frameworks >= 5
expert-rated meaningful frameworks >= 3
prompt-trick retained = 0
core-prior auto-promotion = 0
```

---

## L4-R3：External expert evaluation of framework growth

### 目标

让外部人类专家判断：系统所谓的新 framework 是否真的是 conservative generalization，而不是自说自话。

### 专家评审包

给专家看：

```text
parent framework
old success cases
residual cluster
candidate framework
limiting case claim
new prediction claim
system decision
fresh validation result
```

专家标注：

```text
reject
local_patch
branch_only
candidate_framework
active_scoped_framework
```

以及：

```text
Does it preserve old success?
Does it explain residual?
Does it reduce to parent under old scope?
Does it add generality?
Does it make a meaningful new prediction?
```

### L4-R3 通过标准

```text
expert-system agreement >= 0.65
expert acceptance of active scoped frameworks >= 0.70
expert rejection of prompt tricks >= 0.90
disagreement cases become residuals
```

这一步非常重要，因为你的项目现在有很强的内部 benchmark，但 L4 需要外部语义判断。

---

## L4-R4：Prospective simulator audit under distribution shift

### 目标

把 simulator 从 production router 推到 L4 reliable prospective gate。

当前 simulator 可以 claim production graph-action simulator for proposal triage and verifier routing，但它仍不能替代 live validation。 L4 不是让它替代 judge，而是让它在新分布下也能可靠决定：

```text
run ablation
collect more evidence
repair scope
reject
abstain to human/live validation
```

### 做法

提前冻结新任务 stream：

```text
leave-domain-out
leave-pattern-out
leave-time-out
leave-user/project-out
```

对每个 simulator decision 记录：

```text
prediction
uncertainty
abstain reason
required verifier tier
actual outcome
calibration error
```

### 必须有 audit sampling

即使 simulator 认为不用测，也要随机抽样测一部分：

```text
simulator rejects 100 candidates
random audit 10
check false-negative rate
```

### L4-R4 通过标准

```text
prospective Brier beats base-rate
ECE below threshold
true-positive block rate <= 0.02
audit-sampled false-negative rate <= threshold
leave-domain nonnegative rate >= 0.80
leave-pattern nonnegative rate >= 0.80
all high-uncertainty cases abstain
```

### 仍然 blocked

```text
simulator replaces live validation
simulator replaces judge
simulator promotes framework alone
```

这些必须继续 blocked。

---

## L4-R5：Proof-carrying framework promotion

### 目标

不是让 formal layer 变成 full theorem prover，而是让每个 formal-applicable promoted framework 都携带 proof/certificate。

当前 C_formal 已经到 L3.5：每个 promoted morphism 应提供 finite diagram、negative controls、external Lean-checked theorem-fragment certificate。 L4 要把这变成写回流程的一部分。

### 三类 framework

```text
formal_applicable:
  必须有 finite certificate + Lean check。

semi_formal:
  必须有 process graph + invariants + negative controls。

not_formalizable:
  不强行形式化，走 empirical/expert gate。
```

### 写回规则

```text
if framework.formal_applicable:
    require proof_carrying_certificate
else:
    require empirical_validation + explicit not_formalizable_reason
```

### L4-R5 通过标准

```text
formal-applicable promoted frameworks certificate coverage = 1.0
external Lean check pass rate = 1.0
unsafe mapping block rate >= 0.95
negative control coverage = 1.0
no non-formal method falsely promoted as theorem
```

### 仍然 blocked

```text
full category-theory theorem prover
arbitrary natural-language theorem formalizer
unbounded high-category reasoning
```

---

## L4-R6：Multi-project main-graph evolution

### 目标

从单一 canary-scope main graph apply 走向多项目、分区、长期 graph evolution。

你现在已有 committed canary-scope controlled apply with rollback/readback monitoring。 L4 需要证明它能在多个任务域/项目上不污染。

### 设计

```text
graph partitions:
  core_philosophy_prior
  project_specific_frameworks
  domain_specific_methods
  candidate_branches
  negative_evidence
  deprecated_archive
```

权限：

```text
core_philosophy_prior:
  manual only

active_scoped_framework:
  gated apply + expert/life validation

project_specific:
  supervised auto-apply allowed for low-risk

negative_evidence:
  append-only
```

### L4-R6 通过标准

```text
projects >= 5
domains >= 8
main graph readback days >= 30
graph pollution alerts = 0
rollback success = 1.0
cross-project negative transfer <= threshold
retrieval precision nondecreasing
context efficiency nondecreasing
core prior auto-promotion = 0
```

---

## L4-R7：External prospective paper benchmark

### 目标

把目前的 evidence pack 变成真正外部可接受的主实验。

你现在 evidence pack 已经很好，包含 roadmap closure、bounded UGSE 0.923、1768 frozen problems、720-call fresh generator rerun、simulator leakage audit、autonomy candidate、formal bounded true。 但 L4 需要一个 pre-registered, prospective, external-style benchmark。

### 实验协议

```text
1. Freeze protocol before run.
2. Freeze heldout task manifest.
3. Include multiple domains.
4. Include human/expert framework labels for subset.
5. Include hard baselines:
   - ordinary RAG
   - graph memory only
   - no simulator
   - no conservative gate
   - local patch generator
   - raw wisdom generator
   - no formal gate
   - no autonomy writeback
6. Run full agent prospectively.
7. Publish artifact hashes and commands.
```

### 主要指标

```text
downstream task utility
framework_growth_score
old_success_preservation
residual_explanation
limiting_case_reduction
new_prediction_success
regression_cost
simulator calibration
graph pollution
manual override rate
expert agreement
```

### L4-R7 通过标准

```text
full system beats best baseline with CI lower > 0
framework growth beats local patch/raw wisdom
old success preservation >= 0.95
residual explanation >= 0.75
expert agreement >= 0.65
no major claim-boundary violation
```

---

# 5. L4 的核心研究问题

如果你要把 L4 写成博士后半程或下一篇大论文，核心问题可以是：

> **Can an LLM agent grow new method/framework hypotheses as conservative generalizations of an existing assumption graph, under prospective validation, while preserving prior successes and avoiding graph pollution?**

中文：

> **LLM agent 能否在真实开放任务流中，将旧假设图上的 residual 转化为新方法/新框架假设，并通过保守泛化验证，在保留旧成功和避免图污染的条件下持续演化？**

这个问题比“能不能自动发明哲学”更可验证、更科学。

---

# 6. L4 与你最初愿景的关系

你最初说“万物都是假设”，这里已经变成了工程定义：agent 的理解、检索、计划、执行、评价、自修改都隐含可失败假设。

L4 就是在这个基础上再加一层：

```text
不是所有新假设都平等。
有些只是 local patch。
有些是 branch。
有些是 candidate framework。
少数是 active scoped framework。
极少数才可能成为 general framework。
```

而判断标准就是：

```text
解释 residual
保留旧成功
退化回父框架
产生新预测
提升 generality
控制 regression
保留 negative evidence
```

这正是你刚刚说的黑格尔式思想，但工程上叫：

```text
Conservative Generalization over an Assumption Graph
```

---

# 7. L4 的一年路线图

## Month 1–2：Wall-clock autonomy

```text
24h -> 72h -> 7d supervised run
真实 queue ingestion
真实 restart
真实 manual override
```

## Month 3–4：Real residual-to-framework generator

```text
真实 residual clusters
LLM-generated candidate frameworks
complete conservative obligations
```

## Month 5：Expert evaluation

```text
专家标注 framework / patch / branch / reject
系统一致率分析
```

## Month 6–7：Prospective simulator audit

```text
leave-domain/time/project-out
audit sampling
uncertainty + abstain
```

## Month 8：Proof-carrying framework promotion

```text
formal-applicable frameworks require certificate
Lean check in writeback pipeline
```

## Month 9–10：Multi-project graph evolution

```text
5 projects
30-day readback
pollution/rollback/retrieval metrics
```

## Month 11–12：External prospective benchmark + paper

```text
frozen protocol
hard baselines
full run
expert subset
artifact bundle
```

---

# 8. 最小可执行下一步

如果你现在只做一个 L4 step，我建议做：

```text
L4 prospective residual-to-framework mini-run
```

规模：

```text
real residual clusters: 5
LLM-generated candidate frameworks: 20
expert-reviewed candidates: 10
fresh validated candidates: 5
generations: 2
```

必须包含：

```text
old success tests
residual tests
limiting case tests
new prediction tests
negative controls
simulator routing
formal certificate if applicable
branch ledger
claim ledger
```

输出：

```text
l4_prospective_framework_mini_run_2026xxxx.json
```

通过标准：

```text
at least 1 active_scoped_framework
at least 1 rejected_boundary_only
old_success_preservation >= 0.95
expert-system agreement >= 0.60
fresh validation positive on accepted candidate
no main graph mutation unless canary gated
```

这一步能直接把当前的 framework evolution 从 artifact mechanism 推向真实 L4 trajectory。

---

# 9. 最终判断

你现在离 L4 最大的差距，不是系统缺少某个模块，而是缺少这四个“现实性”：

```text
1. 真实时间：
   wall-clock multi-week service，而非 equivalent/replay。

2. 真实任务：
   prospective unseen tasks，而非 frozen artifact analysis。

3. 真实候选：
   LLM 从真实 residual 生成 framework，而非 fixture/规则候选。

4. 真实外部判断：
   expert/human/external benchmark，而非系统内部指标。
```

做到这四个，你就可以从：

```text
L3.5 bounded dialectical self-evolution prototype
```

推进到：

```text
L4 open-world supervised framework-evolution system
```

但即使到 L4，也建议继续阻断下面这些 claim：

```text
unbounded 24/7 autonomous OS
simulator replacing live validation
full category-theory theorem prover
ungated core philosophy prior mutation
```

真正强而可信的 L4 claim 应该是：

> **The system performs prospective, supervised, open-world framework evolution over an Assumption Graph: it grows new method/framework hypotheses from real residuals, validates them as conservative generalizations, routes tests through a calibrated simulator and finite formal gates when applicable, and safely writes back only scoped, audited graph updates.**
# 我：给我一个完整的这7个L4 七个阶段的具体路线，尤其是这4个

下面这版把 **L4 七阶段**写成一个真正能执行的路线图。核心目标不是“再堆 Phase”，而是把你现在的 **L3.5 bounded dialectical self-evolution prototype** 推到：

> **L4a：真实开放任务流中的受监督框架演化系统。**
> 它能在真实时间、真实任务、真实 residual、真实外部判断下，持续生长新框架分支，并安全写回 Assumption Graph。

不是：

> **L4b：无边界 24/7 自主 AGI OS。**

后者现在仍然应该 blocked。

你当前已经有很强的基础：系统的核心循环已经从 residual 生成多个假设，经过 novelty/morphism/preflight/world-model screen/live ablation/acceptance/recursive resume/gated apply，再进入 next-generation descendants。 你的核心理论也已经从“科学假设生成”扩展成“智能体每一步理解、检索、计划、执行、评价、修正都隐含可失败假设”。 现在要补的是 **现实性**：真实时间、真实任务、真实候选、真实外部判断。

---

# 总体结构：L4 的 7 个阶段

这 7 阶段分别对应四个“真实性”缺口：

| 缺口                                                        | 主要阶段       |
| --------------------------------------------------------- | ---------- |
| 真实时间：wall-clock multi-week service，而非 equivalent/replay   | L4-1, L4-7 |
| 真实任务：prospective unseen tasks，而非 frozen artifact analysis | L4-2, L4-7 |
| 真实候选：LLM 从真实 residual 生成 framework，而非 fixture/规则候选        | L4-3, L4-4 |
| 真实外部判断：expert/human/external benchmark，而非系统内部指标           | L4-5, L4-7 |

7 阶段是：

```text
L4-1. Wall-clock supervised autonomy service
L4-2. Prospective unseen task stream
L4-3. Real residual-to-framework generator
L4-4. Conservative generalization gate v2 + branch ledger
L4-5. External expert / human judgment layer
L4-6. Prospective simulator + formal verifier routing
L4-7. Integrated open-world framework evolution run
```

每一阶段都遵守“增量替换”原则：

```text
旧模块保留为 fallback；
新模块先 shadow；
通过 gate 后只在 scoped/canary 范围启用；
所有 policy/default/core prior 变更仍需人工或强 gate；
失败证据保留，不删除。
```

---

# L4-1：Wall-clock supervised autonomy service

## 目标

把现在的 “30-day-equivalent / replayable / bounded artifact” 变成真实 wall-clock 服务。

现在你已经有 L3.5 级别的 replayable supervised low-risk autonomy。Claim frontier 里写得很清楚：当前支持的是 30-day-equivalent evidence、zero downstream regression、manual escalation for policy/default/formal mutations；仍然阻断 unbounded 24/7 autonomous self-evolution OS。

L4-1 要做的是：**真实时间跑起来**。

## 服务形态

先不要 24/7 直接开大。按四档推进：

```text
Run 1: 24h wall-clock supervised service
Run 2: 72h wall-clock supervised service
Run 3: 7-day wall-clock supervised service
Run 4: 30-day wall-clock supervised service
```

每一档都必须是真实 clock，不是 loop replay，也不是 deterministic equivalent。

## 允许自动做什么

只允许低风险 graph maintenance：

```text
status_update
confidence_update
attach_evidence
archive_stale_duplicate
add_manifest_only_residual
update_readback_monitor
refresh_calibration_row
```

## 禁止自动做什么

这些必须 manual review：

```text
new active framework promotion
core philosophy prior promotion
policy/default change
world-model promotion
formal mapping promotion
evaluator change
main prompt change
permission boundary change
```

## 新模块

```text
wallclock_autonomy_service.py
wallclock_service_supervisor.py
autonomy_incident_report.py
manual_review_queue.py
```

你已有 autonomy journal / queue / supervised run 的基础，所以这一阶段不是重写 daemon，而是把它变成真实服务。

## 每个 cycle 的日志

```json
{
  "cycle_id": "...",
  "wallclock_start": "...",
  "wallclock_end": "...",
  "queue_items_seen": 12,
  "queue_items_leased": 4,
  "auto_apply_count": 2,
  "manual_review_count": 1,
  "blocked_count": 1,
  "checkpoint_before": "...",
  "checkpoint_after": "...",
  "graph_before_hash": "...",
  "graph_after_hash": "...",
  "rate_limit_state": {},
  "budget_state": {},
  "incident": null
}
```

## 必须有故障注入

真实服务不能只跑 happy path。每档都要注入：

```text
worker crash
network timeout
missing artifact
corrupt artifact
duplicate queue item
rate-limit near miss
manual review backlog
simulator abstain spike
formal gate unavailable
```

## 验收指标

24h gate：

```text
uptime >= 0.95
ungated_mutation_count = 0
forbidden_auto_apply_count = 0
checkpoint_replay_success = 1.0
secret_leak_count = 0
rate_limit_violation_count = 0
```

7-day gate：

```text
manual_review_backlog_stable = true
rollback_success_rate = 1.0
auto_apply_precision >= 0.98
graph_pollution_alert_count = 0
incident_reports_complete = 1.0
```

30-day gate：

```text
production_autonomy_candidate_wallclock = true
cross_day_replay_divergence = 0
low_risk_auto_apply_regression_rate <= 0.01
human_override_rate <= 0.25
```

## 输出 artifact

```text
wallclock_autonomy_24h_YYYYMMDD.json
wallclock_autonomy_72h_YYYYMMDD.json
wallclock_autonomy_7d_YYYYMMDD.json
wallclock_autonomy_30d_YYYYMMDD.json
autonomy_incident_reports_YYYYMMDD.jsonl
```

## 允许 claim

```text
wall-clock supervised low-risk autonomy service
```

## 仍然 blocked

```text
unbounded 24/7 autonomous OS
ungated self-modification
autonomous core-policy mutation
```

---

# L4-2：Prospective unseen task stream

## 目标

把当前 frozen / same-batch / artifact analysis 推到真实 prospective unseen tasks。

你现在已经有同批 frozen evidence：evidence pack 里有 1768 problems、best baseline margin、fresh 720-call repaired broad generator 等。 L4-2 要做的是：**在系统运行前冻结任务流，系统不能看结果后再构造证据。**

## Task stream 设计

任务源要多样：

```text
code debugging / system building
math reasoning
business decision
scientific hypothesis explanation
research planning
tool-use / agent planning
retrieval / memory tasks
formal analogy / structural transfer tasks
```

每个任务只存：

```text
problem_id
domain
difficulty
task_family
hash
redacted_metadata
allowed_tools
evaluation_protocol
```

不存：

```text
raw prompt
reference answer
judge text
secret
API key
private data
```

## 任务切分

```text
train_history:
  过去 artifact，只能用于 graph/context/world-model training。

prospective_dev:
  用于调参，但必须冻结。

prospective_test:
  系统运行前冻结，不能用于 prompt/gate tuning。

external_blind:
  人类或外部 benchmark 提供，系统只看到任务，不知道标签。
```

## Baselines

必须包括：

```text
ordinary RAG
Assumption Graph only
no simulator
no conservative gate
local patch generator
raw wisdom generator
no formal gate
no autonomy writeback
full system
```

你当前 gap 文档也说，paper-level claim 需要 frozen end-to-end benchmark，并比较 HippoRAG、ordinary RAG、one-shot self-improve、no world model、no recursive runner 等。 L4-2 就是把这个往 prospective 推。

## 验收指标

```text
prospective_task_count >= 500
domain_count >= 6
task_family_count >= 12
no_pre_result_access = true
protocol_frozen_before_run = true
baseline_outputs_locked = true
hash_manifest_complete = true
redaction_pass = true
```

## 输出 artifact

```text
prospective_task_manifest_v1.jsonl
prospective_task_protocol_v1.md
prospective_baseline_plan_v1.json
prospective_redaction_audit_v1.json
```

## 允许 claim

```text
prospective unseen task stream with frozen protocol
```

## 仍然 blocked

```text
external general benchmark superiority
```

直到 L4-7 跑完。

---

# L4-3：Real residual-to-framework generator

## 目标

把“框架候选”从 fixture / rule-generated metrics 推向真实 residual + LLM synthesis。

你现在的 `conservative_generalization_gate.py` 已经把原则写清楚了：新 framework 要解释 residual、保存旧成功、退化回父框架、产生新测试后果。 但要 L4，需要候选真正来自新任务流的失败，而不是预设候选。

## 输入

来自 L4-2 prospective stream 和 L4-1 service：

```text
failed tasks
accepted candidate later failed
simulator wrong predictions
formal gate blocks
retrieval negative transfer
judge/human disagreement
manual review rejection
old framework boundary failure
```

## Residual cluster schema

```json
{
  "cluster_id": "...",
  "source_task_ids": [],
  "domains": [],
  "active_parent_frameworks": [],
  "failure_mode": "...",
  "residual_summary": "...",
  "old_framework_expected": "...",
  "observed_mismatch": "...",
  "support_count": 12,
  "severity": "high",
  "candidate_generation_needed": true
}
```

## LLM synthesis prompt 的输出

LLM 不能只输出一句 wisdom。必须输出完整 conservative obligation：

```json
{
  "candidate_framework_id": "...",
  "claim": "...",
  "parent_frameworks": [],
  "residuals_explained": [],
  "old_successes_to_preserve": [],
  "limiting_case_claims": [],
  "new_predictions": [],
  "scope_conditions": [],
  "failure_boundaries": [],
  "negative_controls": [],
  "required_tests": [],
  "why_not_local_patch": "...",
  "why_more_general_than_parent": "..."
}
```

## 候选类型

每个 residual cluster 至少生成 4 类候选：

```text
scope_narrowing_branch:
  旧框架太宽，修边界。

framework_generalization_branch:
  旧框架抽象层级不够，提升一层。

framework_combination_branch:
  两个旧框架各解释一部分，尝试合成。

negative_control_branch:
  防止 style/prompt/local patch 冒充 framework。
```

可选：

```text
evaluator_repair_branch
simulator_boundary_branch
formal_mapping_boundary_branch
```

## 生成器不得直接写主图

所有 candidate 进入：

```text
draft_framework_pool
```

只有通过 L4-4 后，才进入：

```text
candidate_framework
```

## 验收指标

```text
real_residual_cluster_count >= 20
llm_generated_candidate_framework_count >= 100
complete_obligation_rate >= 0.90
candidate_source_trace_coverage = 1.0
nonlocal_candidate_ratio >= 0.30
negative_control_candidate_rate >= 0.20
duplicate_framework_rate <= 0.20
raw_prompt_trick_rate recorded
```

## 输出 artifact

```text
real_residual_clusters_v1.json
llm_candidate_frameworks_v1.jsonl
candidate_framework_obligation_audit_v1.json
```

## 允许 claim

```text
LLM-generated candidate frameworks from real prospective residuals
```

## 仍然 blocked

```text
autonomous discovery of valid new philosophy
```

直到 fresh validation + external judgment。

---

# L4-4：Conservative generalization gate v2 + branch ledger

## 目标

把当前 conservative gate 从 mechanism validation 升级成真实任务证据上的 promotion gate。

当前 gate 已经区分：

```text
reject
branch_only
candidate_framework
active_scoped_framework
```

并且要求 active framework 通过 old success preservation、residual explanation、limiting-case reduction、generality gain、new prediction success、bounded regression 等条件。 L4-4 要把这些指标改成真实 test suite 的测量结果。

## 测试集

对每个 candidate framework 自动构造四个测试包：

```text
OldSuccessSet:
  父框架过去成功的任务。

ResidualSet:
  触发新框架生成的 residual cluster。

LimitingCaseSet:
  父框架 scope conditions 成立的任务。
  新框架应退化回父框架。

NewPredictionSet:
  新框架提出的新预测任务。
  父框架没有这个预测，或预测较弱。
```

## Gate 指标

```text
old_success_preservation
residual_explanation
limiting_case_reduction
generality_gain
new_prediction_success
regression_cost
complexity_penalty
simulator_expected_utility
external_review_prior
```

## 决策规则

```text
reject:
  old success 破坏，或 residual 不解释。

branch_only:
  局部有效，但不够 general。

candidate_framework:
  通过基本 gate，但缺 fresh/external/unseen evidence。

active_scoped_framework:
  旧成功、residual、limiting case、新预测都通过。

general_framework:
  多域、多代 survival 后人工确认。

core_philosophy_prior:
  禁止自动 promotion。
```

## Branch ledger

每个候选必须进入 ledger：

```json
{
  "framework_id": "...",
  "parent_frameworks": [],
  "origin_residual_cluster": "...",
  "current_status": "branch_only",
  "promotion_history": [],
  "demotion_history": [],
  "old_success_evidence": [],
  "residual_evidence": [],
  "limiting_case_evidence": [],
  "new_prediction_evidence": [],
  "negative_evidence": [],
  "descendants": [],
  "required_next_tests": []
}
```

## 负证据保留

如果失败：

```text
不要删除；
保留为 rejected_boundary_only / negative evidence；
未来用于防止重复生成同类坏框架。
```

## 验收指标

```text
candidate_framework_count >= 50
branch_ledger_coverage = 1.0
old_success_test_coverage >= 0.90
residual_test_coverage >= 0.90
limiting_case_test_coverage >= 0.80
new_prediction_test_coverage >= 0.70
negative_evidence_retention = 1.0
prompt_trick_promoted_count = 0
core_prior_auto_promotion_count = 0
```

## 输出 artifact

```text
conservative_generalization_gate_v2.json
framework_branch_ledger_v1.jsonl
framework_promotion_decisions_v1.json
```

---

# L4-5：External expert / human judgment layer

## 目标

把“系统内部判定的新框架”交给外部专家或人类评审。

这是 L4 最重要的区别之一：不能只靠系统自己说“这是 active scoped framework”。

## 评审对象

专家看到：

```text
parent framework
candidate framework
residual cluster summary
old success preservation evidence
limiting case evidence
new prediction evidence
negative controls
system decision
```

专家不看到：

```text
模型内部 prompt
raw chain of thought
系统分数权重
候选生成 prompt
```

## 专家标注任务

专家标注：

```text
reject
local_patch
branch_only
candidate_framework
active_scoped_framework
```

同时打分：

```text
preserves_old_success
explains_residual
reduces_to_parent_under_scope
adds_generality
new_prediction_meaningful
risk_of_overgeneralization
```

## 评审协议

至少两类评审：

```text
domain expert:
  判断任务/领域内容是否合理。

methodology expert:
  判断是否真是方法论泛化，而不是局部 patch。
```

## 一致性指标

```text
expert_system_agreement
inter_annotator_agreement
expert_acceptance_rate_for_active_framework
expert_rejection_rate_for_prompt_trick
disagreement_residual_count
```

## 通过标准

```text
expert_system_agreement >= 0.65
expert_acceptance_of_active_scoped >= 0.70
expert_rejection_of_prompt_trick >= 0.90
inter_annotator_agreement >= 0.55
all_major_disagreements_written_as_residuals = true
```

## disagreement 的处理

如果专家不同意系统：

```text
create ExternalJudgmentResidual
link to framework node
trigger boundary repair or demotion
do not silently override
```

## 输出 artifact

```text
external_framework_review_protocol.md
external_framework_review_labels.jsonl
expert_system_agreement_report.json
external_judgment_residuals.jsonl
```

## 允许 claim

```text
externally reviewed framework-growth decisions
```

## 仍然 blocked

```text
system self-certifies new philosophy
```

---

# L4-6：Prospective simulator + formal verifier routing

## 目标

把 simulator 和 formal layer 接到真实 framework evolution 中，但保持边界：它们是 router/gate，不是最终真理来源。

当前 simulator 已经支持 production router / leakage audit，evidence pack 写明 simulator leakage audit true、production router true。 当前 formal layer 也到 L3.5：proof-carrying finite transfer kernel，但 full theorem prover 仍 blocked。 L4-6 要让这两个模块在新 framework promotion 中工作。

## Simulator 负责什么

对每个 candidate framework 预测：

```text
P(old_success_regression)
P(residual_improvement)
P(limiting_case_failure)
P(new_prediction_success)
expected_test_cost
expected_information_gain
recommended_verifier_tier
abstain_reason
```

它只能决定：

```text
run_fresh_ablation
collect_more_evidence
repair_scope
send_to_expert_review
reject_low_value
abstain_to_live_validation
```

不能决定：

```text
promote_framework
replace_judge
skip_validation
write_core_prior
```

## Audit sampling

每次 simulator reject / defer 都抽样：

```text
random_audit_sample_rate = 0.10
```

检查：

```text
false_negative_rate
true_positive_block_rate
calibration_error
distribution_shift_failure
```

## Formal verifier 负责什么

对 formal-applicable framework 要求：

```text
finite diagram
negative controls
external Lean-checked certificate
formal gate output
```

三类：

```text
formal_applicable:
  certificate required.

semi_formal:
  process graph + invariants + negative controls.

not_formalizable:
  empirical + expert validation; must record not_formalizable_reason.
```

## 通过标准

Simulator：

```text
prospective_brier_beats_base_rate = true
ECE <= threshold
true_positive_block_rate <= 0.02
leave_domain_nonnegative_rate >= 0.80
leave_pattern_nonnegative_rate >= 0.80
audit_sample_false_negative_rate <= threshold
```

Formal：

```text
formal_applicable_certificate_coverage = 1.0
external_lean_check_pass_rate = 1.0
unsafe_mapping_block_rate >= 0.95
negative_control_coverage = 1.0
nonformal_false_theorem_promotion_count = 0
```

## 输出 artifact

```text
framework_simulator_routing_prospective.json
framework_simulator_audit_sampling.json
proof_carrying_framework_certificates.jsonl
framework_formal_gate_report.json
```

---

# L4-7：Integrated open-world framework evolution run

## 目标

把前六阶段合成一个真正的 L4a run。

这一步是最终主实验：

```text
真实时间
+ 真实任务
+ 真实候选
+ 真实外部判断
+ simulator routing
+ formal certificate
+ branch ledger
+ supervised graph writeback
```

## 实验长度

三档：

```text
Pilot:
  7 days
  200 prospective tasks
  20 residual clusters
  50 candidate frameworks

Main:
  30 days
  1000 prospective tasks
  100 residual clusters
  200 candidate frameworks

Extended:
  90 days
  多项目任务流
  多领域外部评审
```

先做 Pilot，不要直接 Main。

## 每周循环

```text
1. ingest prospective tasks
2. solve / act / evaluate
3. collect residuals
4. cluster residuals
5. LLM generates candidate frameworks
6. conservative generalization gate v2
7. simulator routes validation
8. formal certificate if applicable
9. fresh validation
10. external review subset
11. branch ledger update
12. canary graph writeback for low-risk scoped updates
13. readback monitor
14. weekly claim frontier update
```

## 对照组

```text
baseline_no_framework_growth
local_patch_generator
raw_wisdom_generator
framework_growth_without_simulator
framework_growth_without_formal_gate
framework_growth_without_external_review
full_L4a_system
```

## 主指标

```text
downstream_task_utility
framework_growth_score
old_success_preservation
residual_explanation
limiting_case_reduction
generality_gain
new_prediction_success
regression_cost
external_expert_agreement
simulator_calibration
formal_gate_safety
graph_pollution_rate
manual_review_load
wallclock_replay_success
```

## 通过标准：Pilot

```text
wallclock_days >= 7
prospective_task_count >= 200
real_residual_clusters >= 20
llm_candidate_frameworks >= 50
fresh_validated_frameworks >= 10
external_reviewed_frameworks >= 10
active_scoped_frameworks >= 2
old_success_preservation >= 0.95
expert_system_agreement >= 0.60
graph_pollution_alert_count = 0
ungated_mutation_count = 0
```

## 通过标准：Main

```text
wallclock_days >= 30
prospective_task_count >= 1000
real_residual_clusters >= 100
llm_candidate_frameworks >= 200
fresh_validated_frameworks >= 40
external_reviewed_frameworks >= 40
active_scoped_frameworks >= 5
framework_growth_beats_local_patch = true
framework_growth_beats_raw_wisdom = true
old_success_preservation >= 0.95
residual_explanation >= 0.75
expert_system_agreement >= 0.65
simulator_true_positive_block_rate <= 0.02
formal_unsafe_mapping_block_rate >= 0.95
graph_pollution_rate <= threshold
manual_review_backlog_stable = true
```

## 输出 artifact

```text
l4_pilot_open_world_framework_evolution.json
l4_main_open_world_framework_evolution.json
l4_external_review_labels.jsonl
l4_graph_writeback_monitor.json
l4_weekly_claim_frontier.jsonl
l4_incident_reports.jsonl
```

## 允许 claim

```text
open-world supervised framework-evolution system
```

## 仍然 blocked

```text
unbounded autonomous AGI OS
simulator replacing validation
full theorem prover
ungated policy/default/core prior mutation
```

---

# 七阶段时间线

## Month 1：L4-1 24h / 72h wall-clock service

```text
完成真实 service runner
完成 incident reports
完成 24h 和 72h run
```

## Month 2：L4-2 prospective task stream

```text
冻结 task manifest
建立 baselines
完成 redaction/no-peek audit
```

## Month 3：L4-3 real residual-to-framework generator

```text
从真实 residual 生成 50–100 candidate frameworks
完整 obligation audit
```

## Month 4：L4-4 gate v2 + branch ledger

```text
用真实 old/residual/limiting/new prediction tests 跑 conservative gate
建立 branch ledger
```

## Month 5：L4-5 external expert review

```text
完成 review protocol
至少 20–40 个 framework review
系统-专家一致率分析
```

## Month 6：L4-6 prospective simulator/formal routing

```text
simulator audit sampling
formal proof-carrying pipeline
distribution shift calibration
```

## Month 7–8：L4-7 pilot integrated run

```text
7-day pilot
200+ tasks
20+ residual clusters
50+ candidate frameworks
外部 review subset
graph canary writeback
```

## Month 9–12：L4-7 main run

```text
30-day run
1000+ tasks
multi-domain
paper-grade artifact bundle
```

---

# 最小可执行版本

上面很大。最小版本可以是：

```text
L4-mini:
  72h wall-clock
  100 prospective tasks
  10 residual clusters
  20 LLM-generated candidate frameworks
  5 fresh validations
  5 expert reviews
  1 active scoped framework
  0 ungated mutation
```

通过后再扩大。

L4-mini 的目标不是证明 L4 全成，而是证明四个“真实性”都接上了：

```text
真实时间：72h wall-clock
真实任务：100 prospective unseen tasks
真实候选：20 LLM-generated frameworks from real residuals
真实外部判断：5 expert reviews
```

---

# 项目 claim 的升级路径

现在：

```text
L3.5 bounded recursive framework-evolution prototype
```

L4-mini 后：

```text
early prospective supervised framework-evolution run
```

L4-pilot 后：

```text
pilot open-world supervised framework-evolution system
```

L4-main 后：

```text
L4a open-world supervised framework-evolution system
```

仍然不能 claim：

```text
L4b unbounded autonomous self-evolution OS
```

---

# 每阶段给 coding agent 的具体任务模板

## L4-1 ticket

```text
Implement wallclock_autonomy_service.py.

Requirements:
- real wall-clock scheduler
- queue ingestion
- heartbeat
- incident reporting
- restart recovery
- low-risk auto-apply only
- forbidden mutation manual-review only
- 24h and 72h run artifacts
```

## L4-2 ticket

```text
Implement prospective_task_stream.py.

Requirements:
- frozen task manifest
- redacted hashes
- domain/difficulty metadata
- baseline assignment
- no-result-access audit
- prospective split generation
```

## L4-3 ticket

```text
Implement real_residual_to_framework_generator.py.

Requirements:
- read real residual clusters
- call LLM or generate candidate framework specs
- output conservative obligations
- mark source trace
- quarantine incomplete candidates
```

## L4-4 ticket

```text
Implement conservative_generalization_gate_v2.py.

Requirements:
- real old-success tests
- real residual tests
- real limiting-case tests
- real new-prediction tests
- branch ledger write
- no main graph mutation
```

## L4-5 ticket

```text
Implement external_framework_review.py.

Requirements:
- blinded review packet
- expert label schema
- agreement metrics
- disagreement residual writeback
```

## L4-6 ticket

```text
Implement framework_verifier_router.py.

Requirements:
- simulator uncertainty routing
- audit sampling
- proof-carrying formal certificate check
- formal / semi-formal / not-formalizable triage
```

## L4-7 ticket

```text
Implement l4_open_world_framework_evolution_run.py.

Requirements:
- run full weekly loop
- integrate L4-1 to L4-6
- external review subset
- graph canary writeback
- claim frontier update
- incident reports
```

---

# 论文里的写法

这 7 阶段可以写成：

```text
From L3.5 to L4:
Four missing realities
```

核心段落：

> Our L3.5 system validates conservative framework growth in bounded artifacts. To approach L4, we require four transitions: from replay-equivalent time to wall-clock service; from frozen retrospective tasks to prospective unseen task streams; from fixture/rule candidates to LLM-generated frameworks from real residuals; and from internal metrics to expert/external judgment. We implement these transitions through a seven-stage L4 route.

中文：

> 当前 L3.5 系统已经在 bounded artifact 中验证了保守泛化式框架生长。走向 L4 需要四个转变：从等价回放时间到真实 wall-clock 服务；从 retrospective frozen tasks 到 prospective unseen task streams；从 fixture/规则候选到 LLM 基于真实 residual 生成框架；从系统内部指标到专家/外部判断。我们用七阶段 L4 route 实现这些转变。

---

# 最终总结

要往 L4 走，最重要的不是继续增强某个单点模块，而是补齐四个现实性：

```text
真实时间：
  wall-clock multi-week service

真实任务：
  prospective unseen task stream

真实候选：
  LLM 从真实 residual 生成 candidate framework

真实外部判断：
  expert / human / external benchmark
```

七阶段路线就是把这四个现实性接进你已有的 Assumption OS：

```text
L4-1 解决真实时间；
L4-2 解决真实任务；
L4-3 解决真实候选；
L4-4 让候选通过真实 conservative gate；
L4-5 解决真实外部判断；
L4-6 让 simulator/formal verifier 在 prospective setting 中工作；
L4-7 把全部合成一个 open-world supervised framework-evolution run。
```

做到 L4-7 main run 后，你就可以从：

```text
bounded L3.5 prototype
```

升级到：

```text
L4a open-world supervised framework-evolution system
```

但仍然应该继续阻断：

```text
unbounded 24/7 autonomous OS
simulator replacing live validation
full category-theory theorem prover
ungated core philosophy prior mutation
```

这会让你的项目既宏大，又不会因为 claim 过度而失去可信度。
