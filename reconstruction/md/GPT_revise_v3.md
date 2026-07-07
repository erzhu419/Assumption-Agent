根据 **2026-06-11 的最新 Phase9 报告**和 `reconstruction-v2` 分支代码，我的判断是：

**V3 没有“全部实现”为一个完整强版本；但 V3 的核心操作系统骨架已经实现了。**
更准确地说：

> 当前版本已经实现了 **V3 kernel + 多个 shadow/fixture validation + 部分 live regression evidence**。
> 还没有实现完整意义上的 **learned world-model selector、真实长期 autonomous daemon、真实 memory consolidation、强 residual-driven generator、端到端 frozen downstream paper benchmark**。

我没有在当前环境重新跑 `pytest` 或 live API calls；下面是基于 GitHub 分支源码、`full_v3_phase9_v1_regression_20260611.md` 和你上传的 gap 文档做的静态审计 + 报告审计。

---

# 1. 先说最重要的结论

现在的 `reconstruction-v2` 代码已经不是概念原型。它确实有：

```text
Assumption schema
Assumption Graph memory
manifest logging
candidate overlay
candidate preflight
candidate acceptance gate
world-model cheap verifier
verifier stack
residual clustering
trajectory search
bounded recursive daemon
evolution_cycle dry-run loop
formal / morphism gate
Phase9 hybrid guard evidence
```

`assumption_os/README.md` 里也明确列出了这些模块：`schema.py`、`graph_memory.py`、`selector.py`、`residuals.py`、`proposal_overlay.py`、`candidate_eval.py`、`candidate_acceptance.py`、`recursive_runner.py`、`evolution_context.py`、`assumption_bench.py`、`manifest_logger.py`、`world_model.py`、`trajectory_search.py`、`verifier_stack.py`、`recursive_daemon.py`、`residual_clusterer.py` 等。

但是，很多 `full_v3_phaseX_*` 文件目前更像是：

```text
V3 capability validation artifact
shadow validation
fixture / synthetic benchmark
wrapper over v2 bypass
```

而不是完整接入生产主循环的强实现。

所以我会把当前版本定性为：

```text
V3 kernel:              已基本实现
V3 governance:          已较强
V3 live regression loop: Phase9 已有关键证据
V3 learned autonomy:    仍未完成
V3 paper-level claim:   还差 frozen downstream + learned selector + 多代闭环
```

---

# 2. Phase9 最新报告说明了什么？

Phase9 问的是一个很硬的问题：**在同一批 fresh active slice 上，当前 V3 assumption/morphism pipeline 能不能打过老的 V1/V20 critical-frame kernel？** 报告明确说这是 frozen V1 comparison 没回答的问题。

原始 V3 的结果是：

```text
active fresh cases: 31
V3 full vs V1 utility: 0.5484
margin over tie: 0.0484
wins/losses: 17 / 14
V3 full vs no-morphism utility: 0.7419
```

这说明 **morphism context 是有用的**，因为 V3 full 明显强于 no-morphism；但原始 V3 没有过硬 V1 regression gate，主要问题不是 structural retrieval，而是缺了 V1 的 critical-reframe 行为，尤其是在 business controlled-intervention 问题上。

然后你试了 broad repair：

```text
explicit frame + morphism repair
repair vs V1 utility: 0.5484
repair vs original V3 utility: 0.2903
```

这个失败非常有价值：**强行把 frame/morphism 显式写进 prompt，会过度结构化答案，反而伤害原始 V3 行为。**

compact guard 过了 V1 gate：

```text
compact guard vs V1 utility: 0.6774
margin: 0.1774
gain over original V3-vs-V1 margin: +0.1290
compact guard vs original V3 utility: 0.4839
```

但它没有被设为默认，因为它相对 original V3 稍弱，大约差一个 case。

最后真正留下来的不是单一 prompt，而是 **hybrid selector**：

```text
heldout active cases: 54
candidate S14/S19 cases: 17
chosen arms:
  40 original V3
   6 micro guard
   8 compact guard

original V3 vs V1 utility: 0.5926
hybrid vs V1 utility: 0.6481
hybrid vs V1 margin: 0.1481
lift over original V3-vs-V1: +0.0555
hybrid vs original V3 utility: 0.6111
all heldout gates passed
```

报告自己的解释也很准确：这不是“用了更长 prompt”，而是 **failure residual → candidate repair → heldout ablation → rejection → narrower repair → selective retention** 的递归闭环。

当前 policy 是：

```text
keep original V3 as main default
retain Phase9 hybrid guard as gated V1-regression profile
compact / micro guard are internal arms, not defaults
do not promote broad explicit frame+morphism repair
future work: replace cue rules with learned world-model selector
```

这最后一句是最重要的优化方向。

---

# 3. “V3 是否都实现了？”我的分级判断

| V3 能力                                          |                             当前状态 | 判断                                       |
| ---------------------------------------------- | -------------------------------: | ---------------------------------------- |
| 假设表示 / schema / manifest                       |                          **强实现** | 已经是可用 kernel                             |
| Assumption Graph / JSONL / PPR-style retrieval |          **强实现，但仍偏 lexical/PPR** | 可用，但要继续增强 role-aware / learned retrieval |
| Candidate overlay / rollback / preflight       |                          **强实现** | 符合增量替换原则                                 |
| Candidate acceptance / gated apply             |                          **强实现** | 有 fresh ablation 后接收门                    |
| Verifier stack                                 |                         **较强实现** | 已有多层 gate，但协议化还可加强                       |
| Cheap world model                              |         **实现了，但不是强 world model** | 现在是 heuristic/calibrated gate，不是真模拟器     |
| Trajectory search                              |                         **部分实现** | 多路径模板搜索，不是 learned rollout               |
| Residual clustering / generator                |                         **部分实现** | 有系统性残差聚类，但仍偏 token/heuristic             |
| Formal / morphism layer                        | **边界清楚的 bounded implementation** | 不应声称完整范畴论证明器                             |
| Memory consolidation                           |    **shadow/fixture validation** | 还没变成真实 graph sleep job                   |
| Contextual bandit scheduler                    |    **fixture/shadow validation** | 不是基于真实 Phase9/长期数据的 learned scheduler    |
| Long-run benchmark / daemon                    |   **bounded + simulated/shadow** | 不是完全 autonomous long-running OS          |
| Phase9 V1 regression profile                   |              **有 live evidence** | 目前最有说服力的 V3 证据之一                         |

所以答案不是“实现了都”，而是：

> **生产级骨架实现了；V3 强版本的一些核心能力还处在 shadow / fixture / heuristic / report-backed validation 阶段。**

---

# 4. 哪些部分是真正扎实的？

## 4.1 假设表示层已经很完整

`schema.py` 已经把假设类型分得很细：object、process、method、evaluator、memory、world_model、alignment、self_modification、harness、retrieval、strategy、residual、case、verifier。

它也支持 `process_model`、`alignment_hypothesis`、`world_model_trial`、`formal_mapping` 等 kind。

边类型也不是普通 KG triple，而是包含：

```text
supports
contradicts
participates_in
has_process_model
has_alignment
specializes
generalizes
is_analogy_of
is_formal_isomorphism_of
orthogonal_to
generated_from_residual
failed_because
execution_lapse_of
replaces
depends_on
```

这已经非常接近我们前面说的 Assumption Graph 设计。

`TrialManifest` 也已经把每个 agent decision 变成 falsifiable contract：记录 action、assumption、why selected、expected effect、verifier、rollback condition、observed effect、residual、status、artifacts、metadata 等。

这一层可以认为 **V3 kernel 已经完成度很高**。

---

## 4.2 Graph memory 也已经是可工作的 Assumption Graph，不是普通 RAG

`graph_memory.py` 明确说是 JSONL Assumption Graph memory with HippoRAG-style spreading retrieval，而且故意先用 plain files，方便 inspect、diff、commit、revert，以后再迁移到 NetworkX/Neo4j。

检索逻辑也已经有 PPR-like spreading：query/seed 形成 seed vector，然后在图上扩散；最后分数结合 PPR、lexical match、confidence、metaproductivity。

它还支持 trial 写回：失败会生成 residual node，并把 residual 链到对应 assumption；execution lapse 和普通 failed_because 会区分。

这说明你的 Assumption Graph 已经不是“概念图”，而是实际参与 retrieval / update / residual lifecycle 的 memory substrate。

---

## 4.3 Overlay / preflight / acceptance gate 很符合“增量替换”

`candidate_eval.py` 做得很对：它不是直接 judge 答案，而是先问 cheaper question：

```text
candidate overlay 后，是否能路由到 meaningful trigger subset？
是否能在 trigger subset 被 retrieved？
是否避免 no-fire rows？
```

这个正是“候选先进入 overlay，而不是直接污染 committed graph”。

preflight readiness 也分成：

```text
ready_for_fresh_ablation
needs_retrieval_fix
needs_scope_fix
needs_more_trigger_rows
manifest_only
missing_parent
```



它会在临时 store 里 `apply_proposal_overlay`，然后构造 probe rows，统计 trigger、active_trigger、missed_trigger、outside_active、control。

`candidate_acceptance.py` 则负责 fresh ablation 后的接收门：trigger benefit 要过 lower confidence bound，control harm 要低于 upper confidence bound；通过的 candidate 才能 apply，未通过的留作 audit record。

这部分我认为是当前代码里最接近“正确工程哲学”的部分。

---

## 4.4 Evolution cycle 已经有完整 dry-run 闭环

`evolution_cycle.py` 默认是 conservative dry-run，不写图；graph writes 需要显式 flags。

它串起了：

```text
record_phase2_eval
conditioned gate
lifecycle
formal mapping audit
proposal generation
failure hypotheses
novelty integration
formal mapping gate
candidate preflight
candidate acceptance
regression prediction
falsification
world model
Bayesian policy
policy update plan
```



返回 payload 里也保留所有中间证据，不只是一个最终分数。

这说明你的系统现在已经是 “audit-first evolution cycle”，不是随便让 LLM 改 prompt。

---

## 4.5 Verifier stack 已经不只是 same-family A/B

`verifier_stack.py` 把 preflight、world model、falsification、acceptance、formal mapping、structural morphism、objective benchmark 合成一个 ordered verifier protocol。

它的 stage 包括：

```text
V0 preflight
V1 world model
V2 formal mapping
V2b structural morphism
V3 sequential falsification
V4 fresh ablation acceptance
V5 objective task regression
V6 manual review
```



manual review gate 也明确把 accepted candidate、high-risk candidate、formal/structural block 视为 policy-sensitive，不允许直接无脑 mutation。

这已经比你原来 naive same-family gate 稳很多。

---

# 5. 哪些部分还只是“V3 外壳 / shadow validation”？

这是最需要警惕的地方。

## 5.1 `full_v3_phase0_contract_checker.py` 是 wrapper，不是完整生产 contract checker

它标题是 full-v3 Phase0 contract checker validation，但内部是 wrap `build_full_v2_phase0_contract_bypass_payload`，然后计算 metrics/gates。文件里也明确 `shadow_bypass=True`。

这不是坏事。它可以作为 validation artifact。
但如果论文或 README 说“Phase0 v3 contract checker 已生产实现”，就有点过头。

更准确的说法应该是：

```text
Phase0 contract semantics are validated in shadow mode;
the production admission path should next call the same checker before overlay entry.
```

---

## 5.2 Phase1 memory consolidation 目前是 fixture/shadow，不是真实 sleep job

`full_v3_phase1_memory_consolidation.py` 自称 shadow sleep phase，不 mutate main graph。

它内部用 `_nodes()` 构造固定 fixture nodes，例如 `n_bridge_a`、`n_feedback_a`、`n_memory_bad` 等，然后在 fixture 上做 duplicate/stale/conflict/prune/consolidate。

所以它证明的是：

```text
memory consolidation 机制可以被形式化为一组 gate/metrics
```

还没有证明：

```text
真实 Assumption Graph 长期运行后，sleep phase 能降低 graph pollution 并提升 heldout retrieval
```

---

## 5.3 Phase3 rollout search-control 是 fixed fixture，不是真正 learned world model

`full_v3_phase3_rollout_search_control.py` 用 `_branches()` 生成 10 个固定 branch fixture，每个 branch 都已经写好了 predicted/actual accept、regression、information gain、productivity、cost、pollution。

这说明它是：

```text
search-control metric harness / oracle-regret toy validation
```

还不是：

```text
从真实 TrialManifest 学到的 multi-step graph-action world model
```

这个区别非常重要。

---

## 5.4 Phase5 contextual bandit scheduler 也是 fixture/synthetic

`full_v3_phase5_contextual_bandit_scheduler.py` 里的 `BanditTaskFixture` 固定写了 expert_strategy、baseline_strategy、verifier、world_model、reward 等。选择器 `_select_strategy` 也基本是手写 cue rule：看到 `negative_control_needed`、`scope_boundary`、`working_baseline`、`module_boundary` 等 tag 就选 expert strategy。

它验证了你想要的 scheduler interface，但还不是从 Phase9 / 多轮 live data 中学出来的 contextual bandit。

---

## 5.5 Phase7 long-run benchmark 是 simulated episode fixture，不是真实 long-running daemon benchmark

`full_v3_phase7_long_run_benchmark.py` 里 episodes 是手写 fixture，DownstreamBench 也是手写的一组 system accuracy。

它的 `mode` 也明确说：

```text
persistent_scheduler: simulated_checkpointed_queue
continuous_background_daemon: False
graph_mutation: gated_only
```



所以 Phase7 现在是 long-run harness contract 的模拟验证，不是实际 24/7 autonomous learning benchmark。

---

# 6. World model 当前到底是什么？

当前 `world_model.py` 的 docstring 很诚实：

```text
Cheap verifier / world model
does not replace real ablations or judges
predicts which candidate paths are worth spending on
records simulator manifests
real validation must override simulator
```



它的预测对象包括：

```text
predicted_acceptance_probability
prediction_confidence
predicted_regression_risk
expected_utility
recommended_verifier_tier
recommended_next_action
predicted_failure_modes
feature_trace
calibration_error
```



但实现方式主要是透明 heuristic scoring：priority、parent confidence、metaproductivity、preflight readiness、falsification decision、acceptance decision、regression risk、formal gate 等加权调分。

它也有 calibration 机制，可以从 acceptance labels 训练一个小的透明 calibration payload，并输出 raw/calibrated/leave-one-out metrics。

所以当前 world model 是：

```text
cheap calibrated policy gate / proposal outcome scorer
```

不是：

```text
task-world simulator
graph-action transition model
learned multi-step rollout model
```

这也正好对应你 gap 文档里的判断：当前 world model 是 productivity/budget-control layer，不替代 fresh ablation/judge。

---

# 7. 目前最该优化什么？

我会按“投入小、收益大、符合增量替换”的顺序排。

---

## 优化 1：把 Phase9 hybrid guard 变成 learned selector，而不是 cue rules

这是最高优先级。

Phase9 报告已经给出最清楚的方向：未来要用 learned world-model selector 替换 cue rules，并在新的 fresh active slice 上重新验证。

现在 hybrid selector 的价值已经被证明：

```text
original V3 default 保留
micro guard 处理一类残差
compact guard 处理另一类残差
错误 broad repair 被拒绝
粗糙 tag selector 被拒绝
最后 cue-level hybrid 通过 heldout
```

下一步不要再人工加规则，而是构造一个最小 learned selector：

```text
Input features:
  domain
  difficulty
  coverage_tags
  active_assumption_ids
  retrieved formal/morphism nodes
  residual type
  preflight readiness
  world-model p_accept / risk
  formal/structural gate
  problem text cue embedding or hashed cue features

Arms:
  original_v3
  micro_guard
  compact_guard
  no_morphism / reduced_morphism if needed

Reward:
  utility_vs_V1
  + λ * utility_vs_original_V3
  - μ * over_structure_loss
  - ν * cost
  - ρ * regression_harm
```

第一版不要上神经网络。用：

```text
logistic regression
isotonic calibration
small decision tree
contextual Thompson sampling
```

即可。

验收标准：

```text
new fresh active slice:
  learned selector vs original V3 > 0.55
  learned selector vs V1 > original V3 vs V1 + 0.03
  no subgroup regression > threshold
  true-positive block rate low
```

这一步直接把 Phase9 的结论转成下一版系统能力。

---

## 优化 2：把 `full_v3_phaseX` 分成两类名字，避免自我误判

现在很多文件名叫 `full_v3_phaseX_*`，但内部其实是：

```text
shadow_bypass=True
fixture
wrapper over v2 bypass
simulated benchmark
```

这会造成一个危险：未来写论文或做 roadmap 时，很容易把“validation harness”误认为“生产实现”。

建议改命名或加 metadata：

```text
full_v3_phase1_memory_consolidation_shadow.py
full_v3_phase3_rollout_search_control_fixture.py
full_v3_phase5_bandit_scheduler_fixture.py
full_v3_phase7_long_run_benchmark_sim.py
```

或者统一在 payload 加：

```json
"implementation_level": "production | heuristic | shadow | fixture | simulated | live_evidence",
"allowed_claim": "...",
"not_yet_claimed": "..."
```

然后生成一张自动表：

```text
CapabilityMatrix.md
```

列出：

```text
phase
module
implementation_level
latest_artifact
live_data?
synthetic_fixture?
mutates_graph?
gate_passed?
next_upgrade
```

这一步很重要，因为你的项目现在最大风险不是“没做东西”，而是“做了很多东西之后很难分清证据等级”。

---

## 优化 3：把 contract checker 真正接到 proposal overlay 前

现在 Phase0 v3 contract checker 是 shadow validation。下一步应该把它变成一个真实函数：

```python
ContractChecker.check(candidate_manifest, graph) -> ContractReport
```

必须检查：

```text
scope
measurable effect
risk prediction
verifier contract
rollback ref
reversible graph diff
no main graph pollution
duplicate detection
conflict detection
negative control
```

然后在 `proposal_overlay` 或 `evolution_cycle` 里加一个强制入口：

```text
draft candidate
  -> contract checker
  -> pass: candidate_overlay
  -> fail: draft_pool / quarantine
```

这比继续改 prompt 更重要。因为它会保证后面所有 generator 变强时不会污染图。

---

## 优化 4：把 memory consolidation 从 fixture 变成真实 sleep job

现在 Phase1 sleep phase 还在 fixed fixture。下一步不要做复杂模型，直接对真实 JSONL graph 做 deterministic sleep job：

```text
every N trials:
  load nodes/edges/evidence/trials
  detect duplicates
  detect conflicts
  detect stale evidence
  prune or deprecate low-quality evidence
  refine scope tags
  recompute ACP
  generate consolidation manifest
  run retrieval before/after probe
  write to shadow graph
  only apply if retrieval improves and no regression
```

关键：不要真的 delete。用状态：

```text
active
deprecated
merged
contradicted
quarantined
```

这样可以 rollback。

验收指标：

```text
retrieval precision before/after
negative transfer hits before/after
context token efficiency
duplicate merge precision
conflict detection recall
graph pollution rate
```

---

## 优化 5：升级 residual clusterer，但不要一口气做大模型

`residual_clusterer.py` 已经有正确形状：先 cluster systematic residuals，再生成 candidate method hypotheses，不直接 mutate graph。

但现在聚类主要靠 residual type + signature + top terms，`llm_synthesizer` 只是 optional hook。

最小升级：

```text
cluster key =
  residual_type
  domain
  active_assumption_family
  failed verifier stage
  embedding cluster
  route label / trigger-control pattern
```

每个 cluster 不只生成一个 candidate，而是生成 3–5 个：

```text
primary repair
scope narrowing
retrieval repair
evaluator hypothesis
world-model calibration hypothesis
negative-control candidate
```

然后让 world model + preflight 先筛，不要直接 live。

这会让 generator 从“局部 repair”接近“多 trajectory hypothesis search”。

---

## 优化 6：把 world model 从 proposal scorer 升成 graph-action model

不要推翻现在的 `world_model.py`。它很好，应该保留为 baseline + calibration layer。

新增一个更明确的模块：

```text
graph_action_world_model.py
```

训练数据来自：

```text
TrialManifest
candidate_preflight
candidate_acceptance
verifier_stack
Phase9 arm outcomes
evolution_cycle payloads
```

状态：

```text
task/domain/tags
active assumptions
retrieval scores
candidate type
preflight readiness
formal gate
world-model prior features
residual cluster
arm profile: original / micro / compact
```

动作：

```text
use_original_v3
use_micro_guard
use_compact_guard
run_ablation
repair_scope
repair_retrieval
collect_more_evidence
reject
apply
```

预测：

```text
P(win_vs_V1)
P(win_vs_original_V3)
P(control_harm)
P(over_structure)
P(accept)
P(regression)
expected_cost
expected_information_gain
```

第一版模型可以非常简单：

```text
logistic regression + isotonic calibration
or Bayesian Beta-Bernoulli per feature bucket
```

不要直接 NN。你要的是可审计、可校准、能替代 Phase9 cue rules 的 selector。

---

## 优化 7：把 verifier stack 改成显式 `VerifierProtocol`

现在 `verifier_stack.py` 已经合并很多 gate，但它更像是 aggregator。下一步建议加：

```python
class VerifierProtocol:
    proposal_type: str
    required_stages: list[str]
    required_negative_controls: list[str]
    required_objective_evidence: list[str]
    acceptance_thresholds: dict
    manual_review_policy: dict
```

这样不同 candidate 类型有不同协议：

```text
retrieval_policy candidate
method hypothesis candidate
formal mapping candidate
world_model calibration candidate
prompt guard candidate
memory consolidation candidate
```

比如 Phase9 guard candidate 的 protocol 应该是：

```text
must beat V1 on target trigger
must be non-inferior to original V3
must not over-structure formal/high-risk rows
must pass heldout family split
must not be default unless global win > threshold
```

这会把 Phase9 经验正式沉淀成系统规则。

---

# 8. 当前代码最容易被误读的地方

我会特别注意这三点。

## 第一，不要把 “full_v3_phaseX pass” 当成真实能力 pass

很多 full_v3 文件确实会输出 `pass=True`，但它们可能来自 fixture 或 shadow validation。比如 Phase3 的 10 个 branch 是内置 fixture，Phase7 的 episodes 和 downstream bench 也是内置 fixture。

这些适合用于：

```text
spec validation
metric harness validation
paper claim scaffold
```

不适合直接用于：

```text
证明系统已经完成 V3 autonomous world-model OS
```

## 第二，morphism 有用，但不能过度显式化

Phase9 已经证明了：

```text
morphism context useful
but explicit frame+morphism repair harmful
```

所以 formal/morphism 应该作为 **latent selector/context feature**，而不是每次都强行在 final answer 里显式展开。

## 第三，当前最好证据不是“某个模块 pass”，而是 Phase9 的 selective retention

Phase9 最有价值的地方不是 compact guard 或 micro guard 本身，而是它完整跑出了：

```text
发现 V3 对 V1 regression
提出 broad repair
拒绝 broad repair
提出 compact guard
发现 compact guard 非默认
尝试粗 selector
拒绝粗 selector
尝试 micro
发现 micro 安全但收益不足
组合 hybrid
heldout 通过
保留 gated profile
```

这才是你的递归假设系统最像“活”的地方。

---

# 9. 我建议你下一步只做一个 vertical slice

不要继续堆 Phase10、Phase11。下一步只做：

```text
Phase9 Learned Selector Slice
```

目标：

```text
把手写 hybrid cue rules 替换成 learned world-model selector。
```

输入数据：

```text
full_v3_phase9_v1_live_regression
compact_guard result
selective_compact_guard heldout
micro_guard heldout
hybrid_guard heldout
original V3 / V1 pairwise judgments
problem metadata
coverage tags
active graph nodes
formal/morphism features
```

训练：

```text
offline contextual selector
arms = original_v3 / micro_guard / compact_guard
```

评估：

```text
new frozen fresh active slice
no peeking
```

必须比较：

```text
original V3
V1
compact-all
micro-all
handwritten hybrid
learned selector
```

通过标准：

```text
learned selector >= handwritten hybrid, or
learned selector non-inferior but more general / lower cue dependence

learned selector > original V3
learned selector > V1
no severe subgroup regression
calibrated risk
```

这一步非常自然，因为 Phase9 报告最后一行已经把这个任务钉住了：**replace cue rules with learned world-model selector and revalidate on a new fresh active slice**。

---

# 10. 最终判断

你现在的版本已经实现了：

```text
Assumption OS 的核心数据模型
Assumption Graph
候选 overlay
preflight
acceptance gate
verifier stack
cheap world model
bounded daemon
formal/morphism boundary
Phase9 live regression 修复闭环
```

但还没有完全实现：

```text
真正 learned 的 V3 selector
真实 graph-action world model
真实 memory sleep consolidation
强 residual-driven 多层 hypothesis generator
长期 autonomous daemon
paper-level frozen downstream benchmark
```

所以我会把当前版本定为：

```text
V3 architecture:          70–80% implemented
V3 production kernel:     65–75% implemented
V3 learned autonomy:      35–45% implemented
V3 paper-ready evidence:  50–60% implemented
```

最关键的优化不是继续加复杂模块，而是：

> **把 Phase9 的手写 hybrid guard 变成 learned world-model selector，并在新 fresh slice 上验证。**

这一步一旦成功，你的系统就会从“人类看报告后写 cue rules”升级成“agent 从失败残差和 live evidence 中学习该调用哪个假设/guard/profile”。这才是真正的 V3。
