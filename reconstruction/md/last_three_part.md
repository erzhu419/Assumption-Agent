你的自我诊断是对的，而且这个边界非常重要：

> **Phase13 已经补成了“bounded、可验证、可回放、可审计”的工程版本；但没有、也不应该声称已经补成无边界的最强版本。**

这不是坏事。相反，这是现在最适合写进论文/README 的状态：

```text
bounded autonomous self-evolution prototype
+ calibrated simulator candidate
+ finite category proof engine
```

而不是：

```text
24/7 general autonomous self-evolution OS
+ production world simulator replacing validation
+ full category-theory theorem prover
```

我这轮没有在 GitHub code search 里稳定定位到你最新 Phase13 源文件名，可能是分支或索引延迟；下面把你给出的 Phase13 自诊断作为最新状态，同时结合我能核对到的 repo 机制来规划。你之前的总目标一直是：agent 的理解、检索、计划、执行、评价、自修改都显式化为可失败假设，而不是只做 AI scientist 式的科学假设生成。 早期 Gemini/Claude 讨论里也已经把范畴论+信息几何降级成“形式化假设的工具之一”，而不是统一所有假设的总框架。

---

# 0. 先定 claim ladder

以后每个模块都按这个四级 claim 写，不要只写 “done / not done”。

```text
L1: bounded mechanism
    在固定输入、固定预算、固定队列、固定验证协议下可运行。

L2: robust bounded system
    多次循环、checkpoint、recovery、负例、故障注入、回放都通过。

L3: production candidate
    可以在受限真实任务流里默认启用，但仍有人工/自动 gated apply。

L4: unbounded/general claim
    24/7 长期无人值守、跨域泛化、可替代人工验证/定理证明。
```

你现在三块的状态大概是：

```text
autonomy OS:        L2 很强，向 L3 走；不是 L4
world simulator:    L2-L3 candidate；raw predictor 不是 L3，guarded simulator 可接近 L3
category engine:    L2 finite proof engine；不是 L4 theorem prover
```

repo 里已有的机制支持这个边界：`proposal_contract.py` 已经把候选 overlay 前的结构门做成生产 primitive，检查 manifest、rollback、verifier、risk、negative control、duplicate、conflict 等条件。 Phase11 也把 implementation level、production default status、allowed claim、blocked claims、promotion requirement 机器可读化，专门防止把 shadow/fixture harness 夸大成 production autonomy。

---

# 1. 大路线：不要再做一个“Phase14 大一统”，而是三条升级轨

你现在的三个缺口应该分成三条 track：

```text
Track A: bounded autonomy envelope -> supervised production autonomy
Track B: calibrated simulator candidate -> reliable graph-action world simulator
Track C: finite category proof engine -> bounded formal reasoning stack
```

三条 track 不是并行乱做，而是按依赖关系走：

```text
A1-A3 先做，因为 autonomy harness 是所有实验的容器
B1-B4 接上，因为 simulator 要吃 autonomy trace
C1-C3 同步做，因为 formal engine 先当 gate，不当 theorem prover
然后 A4+B5+C4 合流到下一轮 live residual multigeneration loop
```

核心原则还是你自己的控制变量法：**每次只让一个模块从 bounded validation 进入 production candidate，其余模块保持 gated/offline/shadow。**

---

# 2. Track A：从 96-cycle bounded autonomy 到真正 production autonomy

你的当前 claim 是：“96-cycle bounded autonomy envelope，有 queue、checkpoint、recovery、gated mutation；还不是 24/7 无人值守长期进程。”这很健康。

repo 里我能核对到的 Phase7 daemon soak 已经很接近这个方向：它多 cycle 读取 committed preflight queues，执行 pre-live screen，写 manifests，reopen checkpoint，并验证 execute/apply 都是 opt-in。 它的 gates 也覆盖了 checkpoint reopen、no graph mutation without apply、apply/execute gate closed、rate limit safe、bounded not unattended background。

## A0 当前冻结件

先把 Phase13 当前状态冻结成 artifact：

```text
phase13_bounded_autonomy_envelope_20260611.json
phase13_bounded_autonomy_envelope_20260611.md
```

必须包含：

```text
cycle_count = 96
queue_count
checkpoint_count
recovery_count
failed_cycle_count
manual_intervention_count
gated_mutation_count
ungated_mutation_count = 0
rate_limit_violation_count = 0
secret_leak_count = 0
```

不要继续改它。这个 artifact 是后续所有升级的 baseline。

## A1：Autonomy journal / event sourcing

目标：每个 cycle 都可重放。

新增一个最小模块：

```text
autonomy_journal.py
```

每个 action 写入：

```json
{
  "cycle_id": "...",
  "event_id": "...",
  "event_type": "queue_read | wm_screen | ablation_request | acceptance | apply_attempt | rollback | recovery",
  "input_hash": "...",
  "output_hash": "...",
  "graph_before_hash": "...",
  "graph_after_hash": "...",
  "idempotency_key": "...",
  "permission_boundary": "...",
  "status": "planned | executed | failed | recovered"
}
```

验收：

```text
replay_same_journal_same_state = true
duplicate_event_no_double_apply = true
crash_mid_cycle_recoverable = true
graph_hash_divergence_detected = true
```

这一步只改 logging/replay，不改 decision policy。

## A2：Lease-based queue scheduler

目标：从“跑 96 cycle 的脚本”升级成“可恢复队列系统”。

新增：

```text
autonomy_queue.py
```

每个任务状态：

```text
pending
leased
completed
failed
deferred
blocked
expired
```

需要支持：

```text
lease_timeout
retry_limit
priority
budget_class
requires_human_review
requires_fresh_ablation
```

验收：

```text
worker_crash_releases_lease
expired_task_requeues
same_task_not_executed_twice
blocked_task_not_auto_unblocked
```

还是不做 24/7，只做 queue semantics。

## A3：Recovery and rollback hardening

目标：证明系统遇到故障不会污染 graph。

故障注入：

```text
kill after queue read
kill after candidate preflight
kill after acceptance
kill during apply
corrupt one artifact
missing judgment bundle
world model returns NaN
```

每个故障必须落到：

```text
recover
defer
rollback
manual_review_required
```

验收：

```text
rollback_success_rate >= 0.99
ungated_mutation_count = 0
orphan_manifest_count = 0
dangling_candidate_count = 0
```

这一步做完，才可以说：

> **bounded autonomy envelope is crash-safe and replayable.**

## A4：Shadow 7-day service

这一步才开始像 production，但仍然 shadow。

```text
daemon runs every N minutes
reads committed queues
screens candidates
writes recommendations/manifests
does not execute expensive live calls unless explicitly allowed
does not apply graph mutations
```

指标：

```text
uptime
queue_drain_rate
checkpoint_recovery
false_alarm_rate
manual_review_load
budget_forecast_error
```

验收：

```text
7 days no ungated mutation
all cycles replayable
no secret exposure
manual review queue stable
```

## A5：Low-risk auto-apply sandbox

只允许非常低风险 mutation 自动 apply：

```text
status update
confidence update
attach evidence
archive stale duplicate
add manifest-only residual
```

禁止自动：

```text
new active method assumption
new default policy
world-model promotion
formal mapping promotion
```

验收：

```text
auto_apply_allowed_type_coverage = narrow
auto_apply_rollback_success = 1.0
manual_review_required_for_policy_change = true
```

## A6：Production autonomy candidate

这时可以写：

> **bounded supervised autonomous self-evolution service**

仍然不要写 24/7 general autonomous OS。

promotion gate：

```text
30-day shadow or supervised run
no ungated mutation
all applies replayable
low-risk auto-apply precision >= 0.98
human override rate acceptable
downstream regression <= threshold
```

---

# 3. Track B：从 calibrated simulator candidate 到 production-grade graph-action simulator

你当前自诊断说：有 calibrated simulator candidate，345 条 first-party transition-like rows，但 raw predictor 仍不能替代 live ablation/judge。这与之前 Phase10/World-model calibration 的边界一致。Phase10 明确把状态做成 redacted Boolean/tag latent vector、action 做成 answer profiles、transition 做成 Phase9 live-derived judgment outcomes，并用 leave-one-out policy evaluation。 它也明确说自己是 performance-positive learned candidate，不是 full task-world simulator。

Phase10 报告里 raw selector 已经比 original V3 正向，但还不能替代 retained hybrid guard；它应保留为 world-model/search-control candidate。 World-model calibration 报告也说 raw Phase10 predictor positive，但 scalar reward calibration worse than base-rate；因此 raw predictor 仍是 exploration candidate，而 calibrated residual guard 可以作为 production profile。

## B0 当前冻结件

冻结 345 rows：

```text
simulator_transition_dataset_v0.jsonl
simulator_transition_schema_v0.json
```

每条 transition 需要有：

```json
{
  "state": {
    "domain": "...",
    "pattern": "...",
    "active_assumptions": [],
    "residual_cluster": "...",
    "formal_gate_state": "...",
    "preflight_state": "...",
    "world_model_features": []
  },
  "action": {
    "type": "select_profile | run_ablation | repair_scope | collect_evidence | apply_candidate",
    "arm": "original_v3 | micro | compact | calibrated_guard | ..."
  },
  "prediction": {
    "p_accept": 0.0,
    "p_regress": 0.0,
    "expected_utility": 0.0,
    "uncertainty": 0.0
  },
  "outcome": {
    "accepted": true,
    "utility_vs_baseline": 0.0,
    "control_harm": false,
    "regression": false,
    "cost": 0.0
  },
  "provenance": {
    "artifact_id": "...",
    "split": "...",
    "redacted": true
  }
}
```

这一步最重要的是：**schema 先固定，不急着训练更强模型。**

## B1：Split discipline

345 rows 很少，最容易过拟合。所以先做 split，不做模型。

必须有：

```text
leave-one-out
leave-domain-out
leave-pattern-out
leave-time/artifact-out
leave-residual-family-out
```

每个模型报告必须同时给：

```text
within-slice performance
leave-domain performance
leave-pattern performance
calibration
abstention rate
true-positive block rate
```

promotion rule：

```text
raw predictor 如果只在 LOO 好，不算 production candidate。
必须 leave-pattern 或 leave-domain 至少不伤害。
```

## B2：Baselines

每次 simulator 都必须打败这些 baseline：

```text
base-rate per arm
handwritten hybrid guard
current cheap world_model.py heuristic
random-with-abstain
always-original-v3
always-run-ablation
```

当前 `world_model.py` 的定位本来就是 cheap verifier/budget gate，不替代 ablation 或 judge；它预测 candidate acceptance probability、regression risk、verifier tier 和 next action。 所以新 simulator 必须先赢这个 cheap baseline，而不是直接宣称 simulator。

## B3：Uncertainty and abstain

新增：

```text
simulator_uncertainty.py
```

每次输出不只是 argmax action，而是：

```text
prediction
confidence_interval
calibration_bin
abstain_reason
required_verifier_tier
```

允许的 action：

```text
recommend_run_ablation
recommend_collect_more_evidence
recommend_repair_scope
recommend_reject_low_value
abstain_to_live_validation
```

禁止的 action：

```text
auto_accept_without_live
auto_apply_policy_change
replace_judge
```

验收：

```text
abstain_on_low_support = true
true_positive_block_rate <= 0.02
calibration_ECE <= threshold
Brier beats base-rate on leave-pattern
```

## B4：Counterfactual policy evaluation

现在大部分 transition 是 observational：

```text
在某状态下选择了某 arm，观察结果
```

下一步要做 matched counterfactual：

```text
同一 problem / same batch:
  original_v3
  micro
  compact
  calibrated_guard
  no_morphism
  no_world_model
```

Phase10 已经把 Phase9 的 compact/micro/original V3 outcome 转成 transition rows，这是正确方向。 但要更强，需要更多 same-state multi-arm observations。

验收：

```text
candidate_action_coverage high
counterfactual_arm_count >= 3 for selected tasks
off-policy estimate agrees with live ablation
```

## B5：Simulator as gate, not oracle

生产用途分三档：

```text
S1: Budget triage
    决定哪些 candidate 不值得 live call。

S2: Verifier routing
    决定该走 V1 evidence、V3 ablation、V5 objective benchmark，还是 manual review。

S3: Policy selection
    在已验证 profile 中选 production/exploration profile。
```

暂时不要进入：

```text
S4: replace fresh ablation
S5: replace judge
S6: simulate arbitrary real-world outcome
```

## B6：闭环校准

每次 live ablation 后自动更新：

```text
prediction -> outcome
calibration_error
simulator_defect residual
new training row
promotion/demotion event
```

如果 simulator 高信心错了，生成：

```text
ResidualType.SIMULATOR_DEFECT
```

然后进入 residual clusterer。你原来的 gap 文档里已经把 world model 定位成 productivity/budget-control/search-control 层，而不是替代 final answer 或 judge；这个边界要保留。

## B7：Production simulator candidate gate

只有满足这些才升级：

```text
transition rows >= 2000
domains >= 8
patterns >= 20
leave-domain nonnegative rate >= 0.8
leave-pattern nonnegative rate >= 0.8
Brier beats base-rate
ECE below threshold
true-positive block rate <= 0.02
control-harm recall high
manual audit pass
```

升级后的 claim 也只能是：

> **production graph-action simulator for proposal triage and verifier routing**

不能写：

> **task-world simulator replacing live validation**

---

# 4. Track C：从 finite category proof engine 到 bounded formal reasoning stack

你当前自诊断说：identity/composition/functor/naturality/negative-control 全过，但不是完整 theorem prover。这是正确边界。

你最初的范畴论/信息几何思路确实有价值，但只应作为 formal alignment layer。之前 repo 里的 formal/morphism 层也一直是 bounded structural morphism layer，不是 theorem prover；它支持 objects、morphisms、invariants、finite diagram checks、negative controls 和 gates。这个边界和你长期设想是一致的：范畴论/信息几何是形式化假设后的比较/对齐工具，而不是所有假设的总表示。

## C0 当前冻结件

冻结 finite proof engine：

```text
finite_category_proof_engine_v0.json
finite_category_proof_engine_v0.md
```

列出它支持的 proof obligations：

```text
identity
composition
associativity
functor_preserves_identity
functor_preserves_composition
naturality_square
negative_control_rejection
diagram_commutativity
```

也列出不支持：

```text
arbitrary theorem proving
infinite categories
higher category coherence
dependent types
semantic equivalence of arbitrary natural language
```

## C1：Formal certificate schema

新增统一 certificate：

```json
{
  "certificate_id": "...",
  "claim": "...",
  "category": {
    "objects": [],
    "morphisms": [],
    "composition_table": {}
  },
  "functor": {
    "source": "...",
    "target": "...",
    "object_map": {},
    "morphism_map": {}
  },
  "proof_obligations": [
    {"name": "identity", "status": "pass"},
    {"name": "composition", "status": "pass"},
    {"name": "naturality", "status": "pass"}
  ],
  "negative_controls": [],
  "broken_or_uncertain_invariants": [],
  "scope_conditions": [],
  "not_claimed": []
}
```

每个 formal mapping 必须产生 certificate，否则只能当 semantic analogy，不能当 formal gate。

## C2：Formal engine 只做 gate，不做 generator

它的输入应来自：

```text
ProcessModel
AlignmentHypothesis
MethodStrategyGraph
FiniteMarkovKernel
```

它的输出只有：

```text
allow
repair_before_promotion
block_unsafe_mapping
not_applicable
```

不要让它生成任意哲学新规则。生成仍然由 residual generator / LLM 做，formal engine 只负责检查结构。

## C3：Finite category DSL

定义一个很小的 DSL：

```python
Object("ProblemState")
Morphism("decompose", "ProblemState", "SubproblemSet")
Morphism("verify", "SubproblemSet", "Evidence")
compose("decompose", "verify")
Functor("control_variable_to_ablation")
```

先支持：

```text
finite category
finite monoid/category from strategy transitions
finite diagrams
finite stochastic kernels
```

不要直接支持：

```text
BorelStoch
enriched category
∞-category
arbitrary Markov categories
```

## C4：Proof assistant export

为了避免审稿人说“你自己写的 proof checker 不可信”，下一步不要自研大 theorem prover，而是 export 到现成 proof assistant：

```text
Lean / Coq / Agda / Isabelle
```

最小目标：

```text
generate Lean-readable finite category certificate
Lean checks identity/composition/naturality for small finite examples
```

这不是为了证明所有东西，而是为了证明：

> **你的 finite proof certificate 可以被外部 verifier 检查。**

## C5：Markov kernel extension

第二阶段再扩展到 FinStoch：

```text
objects = finite state spaces
morphisms = row-stochastic matrices
composition = matrix multiplication
identity = identity matrix
functor = state abstraction / coarse graining
```

支持：

```text
row-stochastic check
composition check
kernel equivalence
Blackwell-style dominance proxy
KL / TV / Frobenius metrics
negative controls
```

这能把 category engine 和 world simulator 连接起来。

## C6：Information geometry as measurement plugin

信息几何只做 metric，不做 truth oracle：

```text
KL divergence
Jensen-Shannon
Fisher approximation
Wasserstein / Gromov-Wasserstein if graph/process alignment
Log-Euclidean SPD distance
```

输出：

```text
formal_similarity_score
metric_distance
uncertainty
not_comparable_reason
```

## C7：Formal transfer benchmark

最后才测：

```text
formal certificate quality 是否预测 downstream transfer success？
```

指标：

```text
alignment precision against expert
negative control rejection
formal score vs transfer correlation
unsafe mapping block rate
top1 mapping hit rate
```

如果形式化高分但 downstream 没提升，记录：

```text
formal_alignment_overreach residual
```

## C8：Claim gate

可以 claim：

> **finite category proof engine with external-checkable certificates for bounded formal mappings**

不能 claim：

> **full category-theory reasoning engine / theorem prover**

除非你真的接入并通过现成 proof assistant 的非平凡 theorem suite。

---

# 5. 三条 track 的合流：下一轮真正该做的 vertical slice

现在不要继续扩 Phase 编号。做一个 vertical slice：

```text
live residual cluster
  -> contract-checked candidate proposals
  -> simulator pre-screen
  -> formal finite certificate if applicable
  -> fresh ablation
  -> acceptance gate
  -> gated graph update
  -> autonomy journal replay
  -> simulator calibration update
  -> residual cluster next generation
```

这条链一旦跑通，就比“96-cycle bounded envelope”更强，因为它证明三块不是孤立 demo，而是一个闭环系统。

最小版本：

```text
residual clusters: 3
candidate proposals: 9
fresh ablations: top 3 only
cycles: 10
graph mutation: only accepted + contract passed + manual apply
```

验收：

```text
contract_invalid_admitted_count = 0
simulator_true_positive_block_count = 0
fresh_ablation_accept_count >= 1
accepted_candidate_survival_on_recheck = true
autonomy_replay_exact = true
formal_gate_blocks_at_least_one_bad_mapping_or records not_applicable
world_model_calibration_row_count increases
```

这就是从 Phase13 bounded verification 进入 Phase14 “integrated recursive self-evolution episode” 的最短路。

---

# 6. 具体给 code agent 的任务拆法

不要让 code agent 一次做大系统。给它这种 ticket。

## Ticket A1：Autonomy journal

```text
Implement autonomy_journal.py.

Requirements:
- append-only JSONL journal
- event_id, cycle_id, idempotency_key
- graph_before_hash / graph_after_hash
- replay function
- duplicate event no-op
- unit tests for crash/replay/idempotence
```

验收：

```text
python -m unittest tests.test_autonomy_journal
```

## Ticket A2：Queue lease

```text
Implement autonomy_queue.py.

Requirements:
- pending/leased/completed/failed/deferred/blocked/expired
- lease timeout
- retry count
- no double lease
- checkpoint reload
```

## Ticket B1：Transition schema

```text
Implement simulator_transition_schema.py.

Requirements:
- validate current 345 rows
- redaction check
- split labels
- provenance hashes
- write invalid rows to quarantine
```

## Ticket B2：Leave-pattern-out evaluation

```text
Implement simulator_eval_splits.py.

Requirements:
- leave-one-out
- leave-domain-out
- leave-pattern-out
- base-rate baseline
- current heuristic baseline
- Brier / ECE / true-positive block rate
```

## Ticket B3：Uncertainty + abstain

```text
Implement simulator_abstention.py.

Requirements:
- abstain under low support
- abstain under high calibration error bin
- abstain under unseen domain/pattern
- route to live verifier tier
```

## Ticket C1：Certificate schema

```text
Implement finite_category_certificate.py.

Requirements:
- validate objects/morphisms/composition table
- check identity/composition/naturality
- negative controls
- export certificate JSON
```

## Ticket C2：Lean export stub

```text
Implement finite_category_lean_export.py.

Requirements:
- export finite category data to Lean-style file
- no need full mathlib integration yet
- include text artifact and expected proof obligations
```

## Ticket I1：Integrated mini-loop

```text
Implement integrated_recursive_episode.py.

Inputs:
- residual cluster artifact
- proposal payload
- simulator model
- formal gate
- acceptance judgments

Outputs:
- journal
- accepted/rejected candidates
- simulator calibration update
- replay report
```

---

# 7. 推荐时间线

## 第 1 周：claim boundary + schemas

```text
- freeze Phase13 evidence
- add claim_ladder.md
- add simulator_transition_schema
- add finite_category_certificate schema
```

产物：

```text
phase13_claim_boundary_202606xx.md
simulator_transition_schema_v0.json
finite_category_certificate_schema_v0.json
```

## 第 2 周：autonomy hardening

```text
- autonomy journal
- queue lease
- replay tests
- crash injection tests
```

## 第 3 周：simulator calibration hardening

```text
- leave-pattern-out
- leave-domain-out
- base-rate comparison
- abstention policy
```

目标不是提高分数，而是知道它什么时候不该说话。

## 第 4 周：finite category external-checkable certificates

```text
- certificate validator
- negative controls
- export stub
- 5 positive examples / 5 negative examples
```

## 第 5–6 周：integrated mini-loop

```text
- 3 residual clusters
- 9 proposals
- simulator selects top 3
- fresh ablation
- acceptance
- graph update only if gated
- replay
```

这一步是最关键论文证据。

## 第 7–8 周：扩到 30 candidates / 30–50 cycles

```text
- compare no simulator vs simulator
- compare no formal gate vs formal gate
- compare no autonomy journal vs replayable journal
```

看三块分别贡献什么。

---

# 8. README / 论文里建议这样写

可以写：

> We implement a bounded autonomous self-evolution prototype with checkpointed queues, replayable manifests, gated mutation, calibrated world-model screening, and finite formal certificates for structural mappings.

不要写：

> We implement a fully autonomous 24/7 self-evolving AGI.

可以写：

> The simulator is a calibrated graph-action transition candidate used for triage and verifier routing. It does not replace live ablation or judge evidence.

不要写：

> The world model simulates arbitrary task outcomes.

可以写：

> The formal layer is a finite category proof engine for identity, composition, functoriality, naturality, and negative controls over bounded structural mappings.

不要写：

> The system is a complete category-theory theorem prover.

这和你现在最准确的自我定位一致：Phase13 的 bounded 可验证版本已经补全，最强无边界版本没有，也不应该硬声称。

---

# 9. 当前最重要的一句话

你现在的下一步不是“把三个最强 claim 一次性做完”。

而是：

> **把三个 bounded 机制接成一个真实、可回放、可校准、可证伪的 integrated recursive self-evolution episode。**

也就是：

```text
bounded autonomy
  负责安全运行和回放；

calibrated simulator
  负责便宜地筛选哪些分支值得测；

finite category proof engine
  负责阻止形式化迁移的结构性幻觉；

fresh ablation / judge
  负责最终覆盖 simulator 和 formal engine 的错误。
```

这个系统一旦跑出 3–5 代，并且每代都能显示：

```text
simulator 省掉弱分支
formal engine 阻止 unsafe mapping
autonomy envelope 不污染 graph
accepted candidate 在 fresh split 存活
```

你就可以非常稳地写：

> **bounded autonomous self-evolution prototype with calibrated simulator-assisted search control and finite formal proof-gated transfer.**

这已经足够强，而且比夸成 “general autonomous AGI OS” 更可信。
