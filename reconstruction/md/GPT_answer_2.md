# 我：
# HLE assumption-agent source lattice debugging question, 2026-07-05

Context:
- Current acceptance baseline remains commit `f577d1a7`, 6/12 on the fixed baseline cohort.
- New work should only be pushed if it beats that baseline, or ties accuracy with clear latency/stability gains.
- Runs use `gpt-5.4-mini`, cache-only HLE scoring unless explicitly doing bounded source prefetch.
- Gold answers are not available to prompt/selection/verifier stages.

Recent implementation:
1. Wired source-verifier private option-witness context into source-grounded MC selection.
2. Persisted only safe public bridge metadata: hashes/counts/option hashes/record summaries, no raw source text in summary.
3. Annotated source-grounded verifier candidates with source-verifier context signals.
4. Removed `support_count=` from the source-grounded verifier prompt so branch majority is not directly shown as a vote weight.
5. Added detailed start-event metadata for source-verifier context candidates.
6. Added audited/admitted distinction:
   - every option with a source-verifier row can be `source_verifier_context_audited`
   - only accepted/direct or strong indirect source-quality rows become `source_verifier_context_candidate`
7. Tried a strict bridge-only no-admitted-candidate skip, but it hurt accuracy, so the skip is now behind `HLE_ENABLE_SOURCE_VERIFIER_BRIDGE_NO_ADMITTED_SKIP` and default off.

Focused tests passed:
- source-verifier bridge keeps private context out of summaries
- private source context is popped before persistence
- source-grounded MC selection can compare sweep candidates
- source-grounded MC selection uses source-verifier context metadata
- generic bridge-only skip is only active when explicitly enabled
- source-grounded verifier prompt hides branch support counts
- deterministic cache-only relation query planner tests
- witness-level semantic top-k context test
- recursive selection budget reserve test

Probe results on seeds `971,145,427,831`:

```text
hle_selectionctx_candidate_signal_probe4_cacheonly_mini_20260705
  accuracy: 1/4
  source-grounded selector ran on all 4
  source_verifier_context_candidate_count was all options, so signal was non-discriminative

hle_selectionctx_admission_probe4_cacheonly_mini_20260705
  accuracy: 0/4
  strict no-admitted skip hurt the only previously-correct shard

hle_selectionctx_admission_noskip_probe4_cacheonly_mini_20260705
  accuracy: 1/4
  default no-skip restored prior behavior

bounded source prefetch:
  dry-run planned 92 queries, 224 source misses
  live80 fetched 48 records
  answer_bearing_directish_record_count improved 8 -> 18
  answer_bearing_diagnostics_evaluated_count improved 21 -> 41

hle_option_lattice_postprefetch_probe4_cacheonly_mini_20260705
  accuracy: 1/4
  source_verifier_generic improved slightly but accuracy did not
```

Important diagnostics:
- Candidate/source metrics improved after prefetch, but final accuracy did not move.
- Failure buckets remain dominated by:
  - `candidate_generation_missed_gold`
  - `candidate_generation_missed_gold_with_sweep_coverage`
  - `gold_option_direct_source_insufficient`
  - `gold_option_source_indirect_or_generic`
  - `gold_option_source_verifier_unaccepted`
  - `source_verifier_generic`
  - `candidate_direct_relation_span_directness_rejected`
  - `relation_span_comparator no direct candidate`
  - `verified_or_abstain no_fallback`
- Post-prefetch source diagnostics improved a bit:
  - `source_verifier_generic` count dropped
  - `missing_required_relation_terms` dropped
  - but `candidate_generation_missed_gold` did not improve.

Observed mechanism:
- Source-verifier bridge now reaches source-grounded selection.
- The source-grounded verifier can sometimes pick the correct sweep candidate even when source verifier labels are generic/indirect.
- However, most source rows still do not become accepted direct relation spans.
- Strictly refusing generic/indirect source paths caused a regression.
- More source prefetch improved direct-ish evidence counts but did not produce accepted direct candidates.

Current question:
Given that:
1. source evidence coverage improved but accuracy stayed 1/4;
2. strict generic-source abstention hurt;
3. source-grounded verifier is sometimes useful even when source verifier labels are weak;
4. the recurring gap is still:

```text
source/preferred/shared docs
  -> candidate-specific direct relation span
  -> comparator accepted direct candidate
  -> final selection
```

What is the next highest-leverage architecture change?

Options I see:

A. Build a true deterministic option-witness lattice that freezes top 2-3 witnesses per option before any model comparator, then compare fixed witnesses only.

B. Add a programmatic source-witness scorer/selector before model selection:
   - if one option has a large source-quality/direct-ish margin, select or promote it;
   - otherwise let source-grounded verifier run as an override candidate but do not block baseline.

C. Improve relation-span extraction/directness:
   - stop asking verifier to find the relation in broad source context;
   - construct explicit candidate-specific span bundles with option overlap, anchor overlap, relation overlap, required-term coverage, and shared-doc penalties.

D. Add baseline fallback for source-verifier-generic no-fallback cases:
   - source path is an override only;
   - when source path remains generic/indirect, fall back to same-run raw/Hippo/budget baseline rather than leaving `no_fallback`.
   Prior weak-source consensus fallback tests were not good, so this needs a safer policy.

E. Stop tuning these debug seeds and run a fresh unseen cohort with the diagnostic bridge only, to see whether the source-lattice problem generalizes before more code changes.

What minimal experiment would distinguish these?

Candidate proof:
- old fixed 12 only as regression
- unseen operator/source-bearing 12 or 24 as actual generalization
- compare:
  - current accepted baseline commit `f577d1a7`
  - current diagnostic bridge
  - deterministic option-witness lattice
  - lattice + fixed-witness comparator
  - lattice + fallback-to-baseline
- require:
  - accuracy beats baseline or is non-inferior with lower latency/variance
  - path hashes stable across repeated cache-only runs
  - gold option admitted witness rate improves
  - generic/no-direct source failure decreases
  - no increase in model calls unless accuracy improves



# GPT:
According to a document from 2026-07-05, this latest issue is **not primarily a comparator problem anymore**. The source bridge and prefetch are now reaching the selector, and some source diagnostics improved, but final accuracy stayed at **1/4**; the persistent failure is still the path from source documents to a candidate-specific direct relation span to an accepted comparator candidate to final label selection.

My recommendation is:

> **Do A + C first, with B as a deterministic scoring layer, and D only as a conservative fallback policy. Do E in parallel as a guard against overfitting, but do not use E as a reason to stop fixing the lattice.**

More concretely:

```text
Next architecture change =
deterministic option-witness lattice
+ explicit candidate-specific span bundles
+ programmatic source-quality scorer
+ fixed-witness comparator
+ baseline fallback when source path is weak.
```

Not:

```text
more source prefetch alone
more abstention alone
more model calls / consensus
stricter generic-source blocking
```

---

## 1. What the latest results actually say

The latest run tells a pretty clear story.

You improved source coverage:

```text
dry-run planned 92 queries, 224 source misses
live80 fetched 48 records
answer_bearing_directish_record_count improved 8 -> 18
answer_bearing_diagnostics_evaluated_count improved 21 -> 41
```

But accuracy did not move:

```text
hle_option_lattice_postprefetch_probe4_cacheonly_mini_20260705
accuracy: 1/4
```

And `candidate_generation_missed_gold` did not improve, even though `source_verifier_generic` and `missing_required_relation_terms` dropped a bit.

That means:

> **More source material is entering the system, but it is not being transformed into the right option-specific direct candidate.**

So B alone, “more source prefetch/cache coverage,” is insufficient. You do have more evidence; it is just not landing in the final decision path.

The strict no-admitted skip also failed:

```text
hle_selectionctx_admission_probe4_cacheonly_mini_20260705
accuracy: 0/4
```

while no-skip restored 1/4.

That means:

> **Generic/indirect source paths are weak, but sometimes still contain useful signal. Blocking them outright throws away the only useful path on some shards.**

So D as “hard abstain/skip” is also wrong. D should become fallback logic, not rejection logic.

---

## 2. Why A is the backbone fix

The recurring gap in your own document is:

```text
source/preferred/shared docs
  -> candidate-specific direct relation span
  -> comparator accepted direct candidate
  -> final selection
```



That is exactly an **evidence topology problem**. Right now, too much of the topology is still implicit:

```text
source row exists
source verifier has some metadata
source-grounded selector sees it
model tries to reason over weak/generic context
```

You need to replace that with an explicit, frozen lattice:

```text
Question
  -> option A witnesses
  -> option B witnesses
  -> option C witnesses
  -> option D witnesses
  -> fixed comparator payload
  -> deterministic selector / fallback
```

For every option, before any model comparator runs, the system should freeze:

```json
{
  "option_label": "B",
  "option_text_hash": "...",
  "query_specs": [],
  "witnesses": [
    {
      "witness_id": "...",
      "source_doc_id": "...",
      "span_hash": "...",
      "option_overlap": 2,
      "anchor_overlap": 1,
      "relation_overlap": 1,
      "required_term_coverage": 0.75,
      "directness_score": 0.68,
      "shared_doc_penalty": 0.10,
      "witness_type": "direct_relation | indirect_relation | definition | generic | contradiction"
    }
  ],
  "top_witnesses_kept": 3
}
```

Then the comparator no longer decides where to look. It only decides:

```text
Given these fixed witnesses, which option is most directly supported?
```

That is the highest-leverage fix because it attacks:

```text
candidate_generation_missed_gold
relation_span_comparator no direct candidate
candidate_direct_relation_span_directness_rejected
source_verifier_generic
gold_option_source_indirect_or_generic
```

at the actual source.

---

## 3. Why C should be part of A, not a separate later task

C says:

> stop asking verifier to find the relation in broad source context; construct explicit candidate-specific span bundles with option overlap, anchor overlap, relation overlap, required-term coverage, and shared-doc penalties.

Yes. This is the concrete implementation of A.

The current failure mode is not merely that the final selector is bad. It is that the final selector often does not receive a well-formed object like:

```text
option B has this direct relation span
option C has only generic source overlap
option D has indirect support but no relation term
```

Instead, it sees something broader/noisier.

So implement C inside the lattice as a **witness bundle builder**.

Each witness should be classified before model comparison:

```text
direct_relation:
  option_overlap > 0
  anchor_overlap > 0
  relation_overlap > 0
  required_term_coverage sufficient

definition:
  option_overlap > 0
  definitional cue present
  question asks identity/definition

indirect_relation:
  option_overlap > 0
  anchor/relation weak but source is relevant

generic:
  option or anchor appears, but no answer-bearing relation

shared_doc_generic:
  document is relevant to many options, but span is not discriminative
```

Then the selector can treat these differently.

---

## 4. What B should do: deterministic scoring, not final truth

B is also useful, but it should not be the only fix.

Use B to produce a **programmatic source-quality score** before the model comparator:

```text
source_quality_score =
  + option_overlap
  + anchor_overlap
  + relation_overlap
  + required_term_coverage
  + directness_score
  - generic_penalty
  - shared_doc_penalty
  - contradiction_penalty
```

Then use it for routing:

```text
if one option has a large direct-witness margin:
    promote as source-preferred candidate
elif multiple options have close direct witnesses:
    send fixed top witnesses to comparator
elif only generic/indirect evidence exists:
    source path may contribute as weak override candidate, but cannot block baseline
else:
    fallback baseline
```

This is important because your source-grounded verifier is “sometimes useful even when source verifier labels are weak,” but strict generic blocking hurts.

So the right policy is not:

```text
generic source => reject/skip
```

It is:

```text
generic source => weak evidence, cannot override unless model comparator plus margin confirms it
```

---

## 5. D should be fallback, not abstain

`verified_or_abstain no_fallback` remains a failure bucket.

That is a design smell.

For HLE multiple choice, source verification should be an **override path**, not the only path. Until the source lattice beats the 6/12 baseline, source evidence should not be allowed to destroy baseline signal.

Use this selector policy:

```python
if direct_source_margin >= strong_margin:
    return source_label, "strong_source_override"

if fixed_witness_comparator_confident and source_quality_margin >= weak_margin:
    return comparator_label, "comparator_source_override"

if source_path_generic_or_indirect:
    return baseline_label, "fallback_weak_source"

if no_direct_candidate:
    return baseline_label, "fallback_no_direct_candidate"

if source_conflict:
    return baseline_label, "fallback_source_conflict"
```

Do **not** use:

```python
if no admitted source:
    abstain/no_fallback
```

The strict no-admitted skip already hurt accuracy.

---

## 6. E is necessary, but not instead of the fix

You should absolutely run a fresh unseen cohort. Your own proposed proof design is right: old fixed 12 only as regression, unseen 12/24 as generalization, compare baseline, diagnostic bridge, deterministic lattice, lattice+comparator, lattice+fallback.

But E alone will not fix anything. It will only tell you whether the same pathology generalizes.

Given the current diagnostics, the pathology is already specific enough:

```text
candidate/source metrics improved
accuracy did not
candidate_generation_missed_gold did not improve
direct candidates still not admitted
strict skip regressed
```

So proceed with A+C+B+D, and use E as the acceptance test.

---

## 7. Minimal architecture I would implement next

### Module 1: `hle_option_witness_lattice.py`

Inputs:

```text
question
options
source_verifier_rows
prefetched_source_records
relation_query_specs
```

Outputs:

```json
{
  "question_hash": "...",
  "options": {
    "A": {
      "option_hash": "...",
      "query_specs": [],
      "witnesses": [],
      "top_witnesses": []
    },
    "B": {}
  },
  "lattice_hash": "..."
}
```

Hard rule:

```text
The same cache-only input must produce the same lattice_hash.
```

Keep **top 2–3 witnesses per option**, not one. Your earlier one-witness dedupe is too brittle.

---

### Module 2: `hle_candidate_span_bundle.py`

Build explicit span bundles:

```json
{
  "span_bundle_id": "...",
  "option_label": "C",
  "source_doc_id": "...",
  "span_hash": "...",
  "witness_type": "direct_relation",
  "option_overlap_terms": [],
  "anchor_overlap_terms": [],
  "relation_overlap_terms": [],
  "required_terms_present": [],
  "required_terms_missing": [],
  "shared_doc_option_count": 3,
  "shared_doc_penalty": 0.25,
  "generic_penalty": 0.0,
  "directness_score": 0.82
}
```

This makes C testable.

---

### Module 3: `hle_source_quality_scorer.py`

Scores each witness and option.

Per witness:

```text
witness_score
directness_score
generic_penalty
shared_doc_penalty
```

Per option:

```text
best_direct_score
best_indirect_score
top_k_score_sum
direct_witness_count
generic_witness_count
source_quality_margin
```

---

### Module 4: `hle_fixed_witness_comparator.py`

Comparator receives only fixed top witnesses, not broad source context.

Prompt shape:

```text
Question
Options
For each option:
  top 2 fixed witness summaries with span hashes and public metadata
Task:
  choose which option has the strongest direct answer-bearing witness.
  If no direct witness, return no_direct_candidate.
```

Cache key:

```text
question_hash
option_hashes
top_witness_hashes
comparator_prompt_version
model
temperature
```

---

### Module 5: `hle_source_override_selector.py`

Combines:

```text
baseline label
programmatic source-quality margin
fixed-witness comparator label
source conflict / generic status
```

Returns:

```text
source_override
comparator_override
fallback_baseline_weak_source
fallback_baseline_no_direct
fallback_baseline_conflict
```

---

## 8. The minimal experiment that distinguishes A/B/C/D/E

Use exactly your proposed structure, but add an ablation that isolates each mechanism.

### Cohorts

```text
C0: old fixed 12
C1: unseen source-bearing 12
C2: unseen source-bearing 24 if budget allows
```

Old fixed 12 is only a regression guard. Unseen 12/24 is the actual acceptance set.

---

### Systems

```text
S0: f577d1a7 accepted baseline
S1: current diagnostic bridge
S2: A+C deterministic option-witness lattice
S3: S2 + B programmatic source-quality scorer
S4: S3 + fixed-witness comparator
S5: S4 + D fallback-to-baseline selector
```

Do not add consensus yet.

---

### Required repeated runs

For cache-only stability:

```text
run each system 5 times on C0 and C1
```

Report:

```text
lattice_hash_stability
witness_set_hash_stability
comparator_payload_hash_stability
selected_label_stability
```

Pass condition:

```text
lattice_hash_stability = 1.0
witness_set_hash_stability = 1.0
comparator_payload_hash_stability = 1.0
selected_label_stability >= 0.95
```

If S2 does not achieve path stability, do not evaluate accuracy yet. Fix determinism first.

---

## 9. Metrics that matter

### Accuracy

```text
accuracy_old12
accuracy_unseen12
accuracy_unseen24
```

Accept only if:

```text
old12 >= 6/12
and unseen improves over S0
```

or:

```text
old12 == 6/12
unseen non-inferior
latency/stability clearly better
```

---

### Source path metrics

```text
gold_option_has_any_witness_rate
gold_option_has_direct_witness_rate
gold_option_top3_witness_rate
gold_option_admitted_witness_rate
gold_option_comparator_candidate_rate
```

The main proof of A+C is:

```text
gold_option_top3_witness_rate improves
gold_option_admitted_witness_rate improves
relation_span_comparator no direct candidate decreases
```

---

### Failure bucket metrics

Track these before/after:

```text
candidate_generation_missed_gold
candidate_generation_missed_gold_with_sweep_coverage
gold_option_direct_source_insufficient
gold_option_source_indirect_or_generic
gold_option_source_verifier_unaccepted
source_verifier_generic
candidate_direct_relation_span_directness_rejected
relation_span_comparator no direct candidate
verified_or_abstain no_fallback
```

Pass condition:

```text
candidate_generation_missed_gold decreases
relation_span_comparator no direct candidate decreases
verified_or_abstain no_fallback -> 0 after fallback selector
```

---

### Generic-source behavior

You do not want generic evidence to dominate, but you also know hard skipping hurts.

Measure:

```text
generic_witness_count
generic_witness_selected_count
generic_path_fallback_count
generic_path_correct_when_fallback
generic_path_wrong_when_override
```

Healthy behavior:

```text
generic paths mostly fallback,
not hard skip,
not strong source override.
```

---

### Cost/latency

```text
unique_model_calls
comparator_cache_hit_rate
mean_latency
p95_latency
source_prefetch_count
```

A good lattice should reduce model variance, not increase calls wildly.

---

## 10. Expected interpretation of ablations

### If S2 improves witness coverage but not accuracy

Then A+C fixed candidate generation, but final selector/comparator is weak.

Next fix:

```text
S4 comparator / S5 fallback policy
```

---

### If S2 does not improve witness coverage

Then relation templates/source prefetch are still wrong.

Next fix:

```text
query template expansion
alias expansion
required term extraction
source index coverage
```

---

### If S3 improves accuracy without S4

Then programmatic source quality is enough for some cases. Keep comparator only for low-margin conflicts.

---

### If S4 hurts

The comparator is overreading fixed witnesses. Add stricter prompt or use comparator only when programmatic margin is inconclusive.

---

### If S5 restores baseline but does not improve

The source path is not good enough yet, but fallback prevents regressions. Keep S5 as safety; continue improving A+C.

---

## 11. Concrete acceptance gate

I would only accept the next change if it satisfies:

```text
1. old12 accuracy >= 6/12
2. unseen12 or unseen24 accuracy > baseline, or non-inferior with lower variance/latency
3. path hashes stable across 5 cache-only reruns
4. gold_option_admitted_witness_rate improves
5. relation_span_comparator no direct candidate decreases
6. verified_or_abstain no_fallback becomes 0
7. generic hard-skip remains off by default
8. model calls increase <= 20%, unless accuracy gain is clear
```

If it fails any of 1, 3, or 6, do not push.

---

## 12. My direct answer to A/B/C/D/E

```text
Primary architecture:
  A + C.

Add immediately:
  B as deterministic scoring/routing inside the lattice.

Safety policy:
  D as fallback-to-baseline, not abstain/skip.

Evaluation guard:
  E as the acceptance experiment on unseen data.

Do not do yet:
  repeated/consensus verifier calls.
```

The one-line next direction:

> **Build a deterministic option-witness lattice that turns broad source/preferred/shared docs into fixed candidate-specific span bundles; score those bundles programmatically; compare only fixed top witnesses; and use source evidence only as a high-confidence override with baseline fallback.**

That is the smallest architecture that directly attacks the current recurring gap without overfitting the four debug seeds.
