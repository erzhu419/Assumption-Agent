# 我：
# HLE Assumption-Agent Next-Step Question: Source Coverage vs Self-Contained Chemistry Rule

We are debugging an HLE multiple-choice assumption-agent. The accepted historical fixed-cohort baseline is still `f577d1a7` at 6/12, but that fixed cohort is now treated as a regression/debug set, not a generalization claim. We should only promote/push a new baseline if it beats the latest baseline on cleaner evidence, or ties accuracy with clear stability/latency/fidelity gains.

Constraints:
- HLE data is local.
- Source APIs are used only for bounded prefetch; final evaluations are cache-only.
- Model is `gpt-5.4-mini` until the agent is clearly better.
- No gold answer is available to decision code. Some debug seeds have been inspected, so improvements there are diagnostic only.

Recent attempted fix:
1. Added answer-time source-cache required-term/fulltext query expansion.
2. Added PubMed as a supported source prefetch backend.
3. Fixed an over-specific `Biology/Medicine` local source domain filter that treated broad Biology/Medicine as clinical medicine and rejected biochemistry/proteomics/chemical-nomenclature docs.
4. Added a conservative self-contained click-chemistry alkyne-probe domain rule:
   - triggers only on probe + click/azide + fluorescence/SDS gel + light/photochemistry + changed/second-probe cues;
   - selects the unique alkyne/propargyl/propargyl-like option;
   - excludes common azide dye/reagent and generic radical/carbene/photoinitiator options;
   - no source or gold answer use.

Diagnostics:

Source/prefetch path on inspected seed 1298:
- Before the self-contained rule, cache-only replay stayed 0/1.
- PubMed/fulltext targeted prefetch found more rows, but did not produce answer-bearing candidate-specific evidence for the gold option.
- The source verifier still ended in `source_grounded_verifier_blocked_no_fallback`.
- Source-cache backfill found docs for distractor options but not for the correct alkyne photoproduct option.
- This suggests this seed is partly self-contained chemistry reasoning rather than retrievable source evidence.

After adding the click-chemistry rule:
- `hle_click_rule_seed1298_cacheonly_mini_20260706`: 1/1, selected by `domain_rule_verifier_priority`.
- This is not acceptance evidence because seed 1298 was inspected.

Diagnostic holdout6, same previously inspected clean-negative cohort:
- Eval: `hle_click_rule_diagnostic_holdout6_cacheonly_mini_20260706`
- Seeds: 1298, 1292, 1285, 1159, 1127, 1108
- Agent-only cache-only result: 1/6.
- The only fixed item is the inspected click-chemistry/proteomics item.
- Remaining buckets:
  - `candidate_generation_missed_gold`: 2
  - `missing_model_option_source_retry_unhelpful`: 5
  - `multiple_choice_selection_failed`: 5
  - `source_verifier_no_candidate_emitted`: 5
  - `span_directness_lexical_unique_but_generic`: 4
  - `source_verifier_rejection_reason`: `no_selected_label_generic`: 16
  - `verified_or_abstain`: `no_fallback`: 4

Fresh unseen n=4 generalization probe:
- Eval: `hle_click_rule_unseen4_generalization_cacheonly_mini_20260706`
- Remapped seed offsets: 1511, 1856, 1863, 2187
- Agent-only cache-only result: 1/4.
- Same seeds raw/Hippo controls:
  - Eval: `hle_click_rule_unseen4_controls_cacheonly_mini_20260706`
  - raw: 0/4
  - HippoRAG: 0/4
- This is mildly positive but too small and lacks budget-matched controls. It is not enough to promote.
- Remaining fresh-unseen buckets:
  - `candidate_generation_missed_gold`: 3
  - `source_verifier_no_candidate_emitted`: 3
  - `source_verifier_rejection_reason`: `no_selected_label_generic`: 17, `no_selected_label_indirect`: 2
  - `source_quality_directness_promotion_blocked`: 3
  - `verified_or_abstain`: `no_fallback`: 4

Question:

Given these results, what is the highest-leverage next architecture change?

Candidate directions:
1. Continue building self-contained domain/operator solvers for chemistry/physics/math cases, but make them general option-matrix solvers rather than one-off rules.
2. Implement the previously recommended pair-binding / option-witness lattice:
   - freeze 2-3 candidate-specific spans per option;
   - compare whether each span establishes the exact question relation;
   - select only direct positive-margin pair-bound candidates;
   - otherwise fall back to raw/Hippo/baseline rather than `no_fallback`.
3. Improve source coverage further with trusted full-text/local snippets, but only for options whose span bundle lacks option/relation/required-term coverage.
4. Add conservative raw/Hippo fallback when source binding fails, especially for `source_verifier_no_candidate_emitted` / `verified_or_abstain no_fallback`.
5. Stop adding narrow rules and first run a larger unseen n=12/24 fair evaluation with raw/Hippo/budget-matched controls to quantify whether current changes are already a small but real improvement.

What minimal experiment would prove the chosen change without overfitting?

Desired acceptance criteria:
- New unseen HLE operator/source-bearing cohort, not the debug seeds.
- Include raw, HippoRAG, agent, and budget-matched controls.
- Report accuracy, source/operator fidelity, and latency/error stability.
- Do not count the inspected seed 1298 click-rule fix as generalization evidence.

# GPT：
这次问题的核心不是“继续加 source prefetch”或“继续堆一个 click chemistry 规则”，而是你现在遇到了一个更清楚的结构性分裂：

```text
一部分 HLE 题是 source-bearing：
  需要从文献/来源里找到 option-specific answer-bearing span。

另一部分 HLE 题是 self-contained domain reasoning：
  正确答案来自题干内部的化学/物理/数学结构推理，source path 可能反而抓到 distractor 文档。
```

所以最高杠杆的下一步不是单选 1/2/3/4/5，而是：

> **建立一个 option-centered candidate matrix + lane router。**
> 每个选项同时走两条候选通道：
> **Source-bound witness lane** 和 **Self-contained operator/solver lane**。
> 如果两条都弱，就回退 raw/Hippo/baseline，而不是 `no_fallback`。

这比继续修 source verifier 或继续加窄规则更重要。

---

# 1. 当前结果说明了什么？

你现在有三个重要信号。

第一，1298 被 click-chemistry rule 修对了，但这是 inspected seed，不能算泛化证据。它说明某些题确实不是“source 不够”，而是需要领域内推理。source-cache 找到了 distractor options 的 docs，但找不到 correct alkyne photoproduct option 的 answer-bearing evidence，这很像 self-contained chemistry reasoning，而不是 retrievable source evidence。

第二，诊断 holdout6 还是 1/6，唯一修的是 inspected click-chemistry item。剩余 bucket 仍是：

```text
candidate_generation_missed_gold
missing_model_option_source_retry_unhelpful
multiple_choice_selection_failed
source_verifier_no_candidate_emitted
span_directness_lexical_unique_but_generic
no_selected_label_generic
verified_or_abstain no_fallback
```

这说明 source path 仍然不能稳定把“文档/摘要/全文片段”转成“某个选项的直接关系证据”。

第三，新 unseen n=4 是 agent 1/4，raw/Hippo 都 0/4。这个是轻微正信号，但太小，而且不是 budget-matched；不能 promote。它说明现在方向也许有一点真实收益，但还远远不足以说明泛化。

所以现在的问题不是：

```text
source coverage 还不够，所以继续 fetch 更多。
```

也不是：

```text
click rule 有效，所以继续写更多窄规则。
```

真正的问题是：

```text
系统没有先判断这道题到底应该走 source-bound evidence，
还是 self-contained domain reasoning，
还是 fallback baseline。
```

---

# 2. 我会如何选择 1–5？

## 方向 1：继续做 self-contained solvers

**要做，但不能继续一题一规则。**

1298 的 click-chemistry rule 是有启发的，但不能再写成：

```text
if probe + click/azide + fluorescence/SDS + light -> choose unique alkyne option
```

这种规则会过拟合 inspected seed。

它应该升级成：

```text
self-contained option-matrix solver
```

也就是对每个选项抽取结构特征，而不是写死某个题型。

例如 chemistry lane 应该输出：

```json
{
  "option_label": "B",
  "functional_groups": ["alkyne", "propargyl"],
  "probe_role": "click-handle",
  "excluded_roles": ["azide dye", "radical initiator", "generic carbene"],
  "question_cues": ["probe", "azide", "fluorescence", "SDS gel", "photochemistry"],
  "solver_score": 0.82,
  "reason": "Alkyne-like handle is uniquely compatible with azide click labeling."
}
```

这才是泛化，不是一题一 rule。

---

## 方向 2：pair-binding / option-witness lattice

**必须做，是 source lane 的主线。**

你现在 source verifier 的失败很像：

```text
source exists
但不是 option-specific
或者是 lexical unique but generic
或者 comparator 没拿到 direct candidate
```

所以 source path 必须改成：

```text
for each option:
    freeze 2–3 candidate-specific spans
    classify span type
    compute option/anchor/relation/required-term coverage
    ask comparator whether this exact span establishes this exact option relation
```

不是让 verifier 在 broad source context 里自己找关系。

---

## 方向 3：继续 fulltext/local snippet coverage

**只在 span bundle 缺覆盖时做。**

现在 source backfill 对 1298 找到了 distractor docs，但没找到 correct option 证据。盲目 fulltext 只会带来更多 generic/ambiguous docs。

正确做法是：

```text
if option has no span with option_overlap + relation_overlap + required_term coverage:
    fetch targeted fulltext/local snippets
else:
    do not expand
```

---

## 方向 4：raw/Hippo fallback

**必须加，作为 selector safety。**

你现在多次出现：

```text
verified_or_abstain: no_fallback
source_verifier_no_candidate_emitted
```

这会让系统在 source path 失败时直接放弃 baseline 信号。对于 multiple-choice HLE，这很危险。

在 source path 没有 strong direct candidate 时，应该：

```text
fallback to raw/Hippo/baseline
```

不是 abstain/no_fallback。

---

## 方向 5：先跑更大的 unseen n=12/24 公平评估

**需要，但不要在修 architecture 前空跑。**

现在你已经知道 failure bucket 主要在哪里。直接跑更大 unseen 只会更稳定地证明系统不行。

正确顺序是：

```text
先实现最小 lane-router 修复；
再跑 n=12/24 unseen。
```

但 n=12/24 必须是 promotion gate，不可缺。

---

# 3. 最高杠杆架构：Option-Centered Evidence/Solver Matrix

我建议下一步的核心结构是：

```text
Question + Options
    |
    v
Option Matrix
    |
    +-- Source-bound witness lane
    |
    +-- Self-contained operator/solver lane
    |
    +-- Baseline/raw/Hippo lane
    |
    v
Lane Router / Selector
```

每个 option 都有一行：

```json
{
  "option_label": "C",
  "option_text": "...",
  "source_lane": {
    "top_span_bundles": [],
    "best_direct_score": 0.0,
    "pair_binding_verdict": "direct|indirect|generic|none",
    "source_confidence": 0.0
  },
  "self_contained_lane": {
    "operator_family": "chem_click_probe | physics_formula | math_symbolic | none",
    "solver_features": {},
    "solver_score": 0.0,
    "solver_confidence": 0.0
  },
  "baseline_lane": {
    "raw_label_score": 0.0,
    "hippo_label_score": 0.0
  }
}
```

然后 selector 决定：

```text
if source lane has direct pair-bound positive margin:
    use source label

elif self-contained lane has high-confidence unique option:
    use solver label

elif source and solver agree:
    use agreed label even with moderate confidence

elif source weak/generic/no-candidate:
    fallback raw/Hippo/baseline

else:
    fallback raw/Hippo/baseline
```

这解决你的两个主要问题：

```text
1. source-bound 题：用 pair-binding lattice 找 direct span。
2. self-contained 题：不要强行 source verification，用 option-matrix solver。
3. source/solver 都弱：不要 no_fallback，保留 baseline。
```

---

# 4. 具体模块设计

## 4.1 `hle_option_matrix.py`

输入：

```text
question
options
source records
raw/Hippo predictions
cache metadata
```

输出：

```json
{
  "question_id": "...",
  "question_hash": "...",
  "option_rows": [
    {
      "label": "A",
      "option_text": "...",
      "aliases": [],
      "source_features": {},
      "solver_features": {},
      "baseline_features": {}
    }
  ],
  "matrix_hash": "..."
}
```

要求：

```text
deterministic
cache-only repeated run hash stable
no gold
```

---

## 4.2 `hle_option_span_bundle.py`

每个 option 保留 top 2–3 spans，不要过早 dedupe 成 1 个。

每个 span bundle 记录：

```json
{
  "option_label": "B",
  "source_id": "...",
  "span_hash": "...",
  "span_text": "...",
  "option_overlap_terms": [],
  "anchor_overlap_terms": [],
  "relation_overlap_terms": [],
  "required_terms_present": [],
  "required_terms_missing": [],
  "shared_doc_option_count": 2,
  "shared_doc_penalty": 0.15,
  "generic_penalty": 0.20,
  "bundle_type": "direct_relation | definition | formula | table_row | indirect | generic | contradiction",
  "directness_score": 0.0
}
```

重点是让 source verifier 不再面对 broad context，而是面对固定 span bundle。

---

## 4.3 `hle_pair_binding_comparator.py`

只回答一个问题：

```text
这个 span 是否建立了该 option 与题目所问 relation 的直接关系？
```

输入不是一堆 source docs，而是：

```text
question_relation
option
candidate span bundle
runner-up span bundle
```

输出：

```json
{
  "option_label": "B",
  "relation_established": true,
  "binding_strength": "direct",
  "beats_runner_up": true,
  "missing_relation_terms": [],
  "verdict": "accept_direct | reject_generic | reject_indirect | reject_ambiguous"
}
```

---

## 4.4 `hle_self_contained_operator_matrix.py`

不要再写 one-off click rule。写 operator families。

第一批只做 3 类，别贪多：

```text
chem_probe_click_matrix
physics_formula_matrix
math_symbolic_matrix
```

### chem_probe_click_matrix

检测：

```text
probe / chemical probe
click / azide / alkyne
fluorescence / SDS gel / labeling
photochemistry / light / irradiation
changed probe / second probe / photoproduct
```

对每个 option 抽特征：

```text
is_alkyne_like
is_azide_dye
is_generic_radical
is_carbene_or_photoinitiator
is_reagent_not_probe_handle
```

输出：

```text
unique alkyne/propargyl-like option with exclusions
```

注意它必须是 option-matrix：

```text
每个 option 都评分，而不是直接匹配题干后选某类词。
```

### physics_formula_matrix

针对 self-contained formula 题：

```text
extract variables
identify formula family
evaluate options by dimensional/limiting consistency
```

### math_symbolic_matrix

针对纯数学/符号：

```text
symbolic transform
check option equality / counterexample
```

---

## 4.5 `hle_lane_router.py`

这是关键。

输入 option matrix，输出 label。

伪代码：

```python
if source.best_direct_margin >= STRONG and source.pair_binding_accept:
    return source.label, "source_direct_override"

if solver.unique_margin >= STRONG and solver.confidence >= THRESH:
    return solver.label, "self_contained_solver_override"

if source.label == solver.label and min(source.conf, solver.conf) >= MODERATE:
    return source.label, "source_solver_agreement"

if source.is_generic_or_no_candidate:
    return baseline.label, "fallback_weak_source"

if source.conflict or source.low_margin:
    return baseline.label, "fallback_conflict_or_low_margin"

return baseline.label, "fallback_default"
```

一定要消灭：

```text
verified_or_abstain: no_fallback
```

---

# 5. 为什么这个方案比单独 1/2/3/4/5 更好？

## 比只做 1 好

只做 chemistry/physics/math solvers，会把项目拖进一堆 narrow rules。你需要的是：

```text
self-contained lane
```

而不是：

```text
一题一规则
```

## 比只做 2 好

pair-binding lattice 适合 source-bearing 题，但 1298 说明有些题找不到正确 source，靠 source path 会错过。

## 比只做 3 好

更多 source 可能增加 distractor docs。只有在 option span bundle 缺覆盖时，targeted expansion 才值得做。

## 比只做 4 好

fallback 可以防止倒退，但不能创造提升。它必须和 source/solver lane 配合。

## 比只做 5 好

大评估必要，但现在 failure mechanism 已经很清楚。先修最小架构，再用 n=12/24 证明。

---

# 6. 最小实验：不要过拟合 1298

你需要一个 **two-lane router proof**。

## Cohorts

```text
D0: inspected debug holdout6
    seeds: 1298,1292,1285,1159,1127,1108
    只做诊断，不算 promotion。

G1: fresh unseen n=12
    operator/source-bearing mix
    不检查 gold 后调规则。

G2: fresh unseen n=24
    如果预算允许，作为最终 acceptance。
```

G1/G2 要分层抽样：

```text
self-contained chem/physics/math: 1/3
source-bearing biomedical/science/history: 1/3
mixed/ambiguous: 1/3
```

---

## Systems

至少比较 6 个：

```text
S0 raw
S1 HippoRAG
S2 f577d1a7 accepted baseline
S3 current click-rule/source branch
S4 pair-binding source lattice + fallback
S5 self-contained option-matrix solver + fallback
S6 full option-matrix lane router
```

如果预算不够，最少：

```text
raw
Hippo
current
source lattice + fallback
full router
```

---

## Metrics

### Accuracy

```text
overall accuracy
source-bearing subset accuracy
self-contained subset accuracy
mixed subset accuracy
```

### Source path fidelity

```text
option_with_span_bundle_rate
gold_option_has_bundle_rate
gold_option_direct_bundle_rate
pair_binding_candidate_rate
pair_binding_accept_rate
source_generic_rate
source_no_candidate_rate
```

这些 gold metrics 只能在评估后算，不能进入决策路径。

### Solver path fidelity

```text
solver_activation_rate
solver_correct_activation_rate
solver_false_positive_rate
unique_option_margin
operator_family_distribution
self_contained_lane_accuracy
```

### Fallback behavior

```text
no_fallback_count
fallback_to_raw_count
fallback_to_hippo_count
fallback_correct_rate
source_override_correct_rate
solver_override_correct_rate
```

### Stability

cache-only 重跑 3–5 次：

```text
option_matrix_hash_stability
span_bundle_hash_stability
solver_feature_hash_stability
router_payload_hash_stability
selected_label_stability
```

### Cost/latency

```text
prefetch_calls
model_calls
cache_hit_rate
mean_latency
p95_latency
```

---

# 7. Acceptance criteria

我会只在满足下面条件时 promote：

```text
1. D0 debug holdout6:
   不低于当前 1/6。
   不能只靠 1298 一题改善。

2. G1 unseen n=12:
   full router > max(raw, Hippo, current, f577)
   或至少持平 best baseline 且 no_fallback=0、稳定性/延迟明显更好。

3. 如果 G2 unseen n=24 跑了：
   full router 至少比 best baseline 多 2 题正确，
   或者非劣且 source/solver fidelity 明显改善。

4. no_fallback_count = 0。

5. source-bound subset:
   pair_binding_candidate_rate 提升；
   source_verifier_no_candidate_emitted 降低；
   no_selected_label_generic 降低。

6. self-contained subset:
   solver false-positive rate 低；
   solver override accuracy > baseline。

7. cache-only stability:
   selected_label_stability >= 0.95。
```

如果只在 1298 或 inspected holdout6 上有效，不 promote。

---

# 8. 当前最可能的失败模式和如何诊断

## 8.1 Full router 没提升，但 fallback 保住 baseline

说明：

```text
router 安全，但 source/solver 没有足够强的 override。
```

下一步看：

```text
source bundle gold coverage
solver activation precision
```

## 8.2 Solver lane 提升 self-contained，但伤害 source-bearing

说明：

```text
self-contained detector false-positive。
```

修：

```text
raise solver activation threshold
add anti-trigger
require unique option margin
```

## 8.3 Source lattice 提升 candidate coverage，但不提升 accuracy

说明：

```text
pair-binding comparator 或 selector 弱。
```

修：

```text
stronger fixed-span comparator
programmatic source margin
runner-up comparison
```

## 8.4 Source lattice 仍然没有 gold bundle

说明：

```text
query expansion / fulltext snippets / alias expansion 仍不足。
```

修方向是 3，但 targeted 做。

## 8.5 Hippo/raw fallback 经常救回来

说明：

```text
source/solver override 太激进。
```

修：

```text
increase override margin
use fallback more often
```

---

# 9. 具体给 Codex 的任务

你可以这样下任务：

```text
Implement HLE option-centered two-lane router v1.

Files:
1. hle_option_matrix.py
2. hle_option_span_bundle.py
3. hle_pair_binding_comparator.py
4. hle_self_contained_operator_matrix.py
5. hle_lane_router.py
6. tests/test_hle_option_matrix_router.py

Requirements:
- Build deterministic option rows for each MC option.
- Source lane keeps top 2–3 span bundles per option.
- Span bundles expose option_overlap, anchor_overlap, relation_overlap, required_terms, shared_doc_penalty, generic_penalty, bundle_type, directness_score.
- Self-contained lane supports at least chem_probe_click_matrix, physics_formula_matrix, math_symbolic_matrix.
- Lane router uses source direct override, solver unique override, source+solver agreement, otherwise raw/Hippo/baseline fallback.
- No `verified_or_abstain no_fallback`.
- Emit option_matrix_hash, span_bundle_hash, solver_feature_hash, router_payload_hash.
- Add cache-only repeated stability test.
- Add eval script comparing raw, Hippo, current, source-lattice, solver-lane, full-router on debug holdout6 and unseen n=12.
```

---

# 10. 最终建议

我的排序是：

```text
第一优先级：
  Option-centered two-lane router:
    source pair-binding lattice
    + self-contained option-matrix solver
    + raw/Hippo fallback

第二优先级：
  targeted fulltext/local snippets only when span bundles lack coverage

第三优先级：
  larger unseen n=12/24 fair evaluation

暂时不要：
  更多 one-off chemistry rules
  更严格 generic skip
  consensus verifier calls
  盲目扩大 source prefetch
```

一句话：

> **你现在需要的不是“source 路线”或“domain rule 路线”二选一，而是让每个 option 同时进入 source-bound witness lane 和 self-contained solver lane，再由一个保守 router 决定是否 override baseline。**

这样才能解释 1298 这种 source 找不到 gold 但化学结构能推的题，也能继续修 source-bearing 题里 direct span 进不了 comparator 的问题，同时避免 `no_fallback` 把 baseline 信号丢掉。
