# 我：
# HLE generalization debug question: source/directness still failing

Context:
- We are debugging an HLE multiple-choice assumption-agent.
- Historical accepted fixed-cohort baseline is `f577d1a7` at 6/12, but that cohort is now considered a regression/debug set, not generalization evidence.
- Current branch has added conservative domain/programmatic rules and source-verifier plumbing. We only want to accept/push as a promoted baseline if it improves the latest real baseline or ties with clear stability/latency gains.
- HLE data is local. Source API usage is only for bounded prefetch; final evaluation is cache-only.
- Model for these runs is `gpt-5.4-mini`, temperature 0, no gold in decision path.

Recent changes:
1. Added a syntactic garden-path POS domain rule.
2. Added an hourglass running-weight formula rule.
3. Added a self-contained experimental table rule that selects a broad behavior-effect option when the data support a perturbation effect and the competing options add unsupported specific habitat/settlement premises.
4. Kept the stricter source/directness path: candidate-specific relation spans must pass directness/source-verifier gates; no broad relaxing of source gates.

Positive but non-general result:
- On the inspected freshheldout6 cohort `1196,2062,1301,2212,1383,2318`, the new rules move agent from 1/6 to 3/6.
- Same-seed raw/Hippo controls were raw 0/6, Hippo 0/6.
- This is not clean generalization evidence because the cohort was inspected and then rules were added.

Clean negative result:
- A new untouched generalization holdout n=6 was sampled by `--generalization-holdout` from seed offset 9000 / stride 191.
- Remapped seed offsets: `1298,1292,1285,1159,1127,1108`.
- Topic-level sanitized descriptions:
  - 1298: biochemistry/proteomics chemical probe workflow.
  - 1292: endohedral fullerene + disilirane reaction, effect on Ce atoms.
  - 1285: aryl halide lithiation/boronic acid/NMR B-signal issue.
  - 1159: Death Valley sulfate deposit isotope/paleoclimate interpretation.
  - 1127: dielectric/conducting sphere in uniform electric field, steady-state physics formula.
  - 1108: Fabry-Perot cavity optical communication formula/condition.
- Agent cache-only result: 0/6.
- Raw/Hippo controls on same seeds: raw 0/6, Hippo 1/6.
- Targeted live source prefetch was run for the same seeds using Semantic Scholar/OpenAlex with failure-focused candidate-specific queries.
- Prefetch diagnostics found some answer-bearing/direct-ish records:
  - fetched_count 54, source_record_count 124
  - answer_bearing_option_signal_record_count 19
  - answer_bearing_directish_record_count 4
- Cache-only replay after prefetch was still 0/6.

Dominant failure buckets after prefetch replay:
- agent_wrong_or_error: 6
- candidate_generation_missed_gold: 4
- evidence_invalid_or_unhelpful: 4
- missing_model_option_source_retry_unhelpful: 6
- multiple_choice_selection_failed: 6
- source_verifier_no_candidate_emitted: 6
- span_directness_lexical_unique_but_generic: 6
- span_directness_verifier_rejected: 5
- relation_span_comparator_rejected: 5
- source_quality_directness_promotion_blocked: 6
- verified_or_abstain: no_fallback 4, abstained 2

Important diagnostic:
- Prefetch is no longer simply empty. The system can fetch relation-ish/direct-ish records.
- But answer-time source verifier/directness still rejects or cannot emit a selected candidate:
  - source_verifier_rejection_reason: no_selected_label_generic 15, ambiguous 3, indirect 2
  - source_quality_directness_rejection includes too_many_ambiguous_docs, not_span_direct, relation_span_shared_doc, missing_required_relation_terms
- Therefore the bottleneck looks like:

`source/preferred/shared docs -> candidate-specific direct relation span -> source verifier selected label -> accepted comparator candidate -> final selection`

is still broken, even when source prefetch adds records.

Question:
Given this evidence, what is the highest-leverage next architecture change?

Candidate directions:
1. Build deterministic/programmatic solvers for self-contained math/physics/chemistry cases, instead of relying on source directness. This may help formulas but risks becoming a pile of narrow rules.
2. Replace the current source verifier with a pair-binding comparator that takes, for each option, one candidate-specific source span and asks: "does this span establish the exact relation requested by the question for this option, and does it refute/beat the runner-up?" This would focus on semantic binding rather than generic source quality.
3. Make source prefetch stronger by switching from paper-title/abstract APIs to local full-text snippets or trusted web cache, because abstract-level sources are too generic for HLE option-level decisions.
4. Add a conservative HippoRAG-preserve fallback when the agent has no verified candidate and Hippo has an answer-bearing route, since Hippo beat agent by 1/6 on this clean cohort. Risk: this may only preserve lucky guesses.
5. Add consensus/repeated verifier calls for high-variance source/directness stages. Risk: budget and latency go up without fixing missing direct evidence.

What minimal experiment would prove the chosen change?
- It should use a new unseen HLE cohort, not the debug seeds above.
- It should report three axes: accuracy vs raw/Hippo, source/operator fidelity, and latency/error stability.
- It should not count improvements on inspected/tuned cohorts as generalization.

# GPT:
According to a document from **2026-07-05**, this latest issue is **not solved by simply adding more prefetch, stricter directness gates, or more verifier calls**. Your prefetch is no longer empty, but the answer-time path still cannot convert retrieved records into a candidate-specific, accepted relation span and final label. The clean untouched holdout was **0/6** for the agent after prefetch, while raw was **0/6** and Hippo was **1/6**; prefetch found **54 fetched records / 124 source records / 19 option-signal records / 4 direct-ish records**, but cache-only replay remained **0/6**. The dominant failures include `candidate_generation_missed_gold`, `source_verifier_no_candidate_emitted`, `span_directness_lexical_unique_but_generic`, `relation_span_comparator_rejected`, `source_quality_directness_promotion_blocked`, and `verified_or_abstain: no_fallback`.

My recommendation:

> **Primary next change: Direction 2, but implemented as a pair-binding / option-witness lattice, not as another generic source verifier.**
> Add Direction 3 only where abstracts are provably too generic, and add Direction 4 as a fallback safety layer. Do **not** prioritize Direction 1 or 5 yet.

In concrete terms:

```text
Replace:
  source verifier asks "is this source good / direct enough?"

With:
  for each option, freeze one or more candidate-specific spans;
  ask whether this span establishes the exact question relation for this option;
  compare it against the runner-up span;
  if no option has a source-bound relation, fall back to baseline/Hippo/raw rather than no_fallback.
```

---

## 1. Why the current failure is not “source empty” anymore

The latest evidence says source prefetch is doing something. It produced relation-ish/direct-ish records. But those records are not becoming selected answer evidence. The document explicitly says prefetch is no longer simply empty, yet the answer-time source verifier/directness stage still rejects or fails to emit selected candidates. The rejection reasons include `no_selected_label_generic`, `ambiguous`, `indirect`, `too_many_ambiguous_docs`, `not_span_direct`, `relation_span_shared_doc`, and `missing_required_relation_terms`.

That means the bottleneck is here:

```text
source/preferred/shared docs
  -> candidate-specific direct relation span
  -> source verifier selected label
  -> accepted comparator candidate
  -> final selection
```

The document identifies exactly this chain as still broken even after prefetch.

So the problem is **semantic binding**, not just retrieval coverage.

---

## 2. Do not make Direction 1 the main fix yet

Direction 1 says to build deterministic/programmatic solvers for self-contained math/physics/chemistry cases.

That will help some of the clean holdout topics:

```text
1127: dielectric/conducting sphere in uniform electric field
1108: Fabry-Perot cavity optical communication formula/condition
```

But it risks producing a pile of narrow rules. The clean holdout also includes proteomics chemical probe workflow, endohedral fullerene reaction effects, aryl halide lithiation/NMR B-signal, and isotope/paleoclimate interpretation.  These are not all formula-solvable from a tiny deterministic physics library.

So Direction 1 should become a **fallback lane**:

```text
if problem is clearly self-contained formula / table / symbolic derivation:
    use deterministic solver lane
else:
    use source-binding lattice
```

It should not be the main architecture change.

---

## 3. Direction 2 is the highest-leverage fix, but phrase it more precisely

The current source verifier appears to judge generic source quality/directness. That fails when a source is topically relevant but not answer-binding.

The next verifier should be **pair-binding**:

```text
For option X:
  Does span S establish the exact relation R asked by the question for option X?
  Does span S beat or refute the best span for runner-up option Y?
```

The key is that the comparator must not search broad context. It should only see frozen, candidate-specific spans.

### Pair-binding comparator input

```json
{
  "question_relation": "what exact relation must be established?",
  "option": {
    "label": "B",
    "text": "...",
    "aliases": [...]
  },
  "candidate_span": {
    "source_id": "...",
    "span_text": "...",
    "option_overlap_terms": [],
    "anchor_overlap_terms": [],
    "relation_overlap_terms": [],
    "required_terms_present": [],
    "required_terms_missing": [],
    "span_hash": "..."
  },
  "runner_up": {
    "label": "C",
    "span_hash": "...",
    "quality_score": 0.62
  }
}
```

### Pair-binding comparator output

```json
{
  "option_label": "B",
  "relation_established": true,
  "binding_strength": "direct|indirect|generic|contradicted",
  "beats_runner_up": true,
  "missing_relation_terms": [],
  "reason": "..."
}
```

This directly attacks the current failure buckets:

```text
source_verifier_no_candidate_emitted
span_directness_lexical_unique_but_generic
span_directness_verifier_rejected
relation_span_comparator_rejected
source_quality_directness_promotion_blocked
```

---

## 4. Direction 3 is useful, but only after you separate “abstract too generic” from “binding failed”

Switching from titles/abstracts to full text or trusted snippets can help. But the current diagnostic already shows that even when source records exist, they are rejected as generic/ambiguous/indirect. More source may simply create more ambiguous docs unless the span-binding step is fixed.

So Direction 3 should be gated by a diagnostic:

```text
if option has no candidate-specific span after abstract-level lattice:
    request full-text/local snippet expansion
else:
    do not expand source
```

Do not do:

```text
always fetch more full text
```

Do:

```text
fetch full text only for options whose witness bundle lacks:
  option_overlap > 0
  anchor_overlap > 0
  relation_overlap > 0
  required_term_coverage >= threshold
```

This keeps latency and budget controlled.

---

## 5. Direction 4 should be added as fallback, not as source truth

The clean holdout had Hippo at **1/6** while agent/raw were **0/6**, so preserving Hippo can prevent some regressions. But the document correctly notes this may just preserve lucky guesses.

Use Hippo fallback only when source binding fails:

```text
if no bound source candidate:
    fallback to same-run Hippo/raw/budget baseline
```

Do **not** let Hippo override a strong pair-bound source candidate.

Policy:

```python
if pair_bound_source_label has direct relation and beats runner-up:
    choose pair_bound_source_label

elif source evidence is generic/ambiguous/indirect:
    fallback to baseline_or_hippo

elif no source candidate:
    fallback to baseline_or_hippo

else:
    fallback to baseline_or_hippo
```

This also fixes the persistent `verified_or_abstain: no_fallback` failure bucket.

---

## 6. Direction 5 should wait

Consensus/repeated calls are not the right first fix. Your failures are not mainly “the same fixed span gets judged inconsistently.” They are:

```text
no selected candidate
generic source
directness rejected
relation comparator rejected
source quality promotion blocked
```

Consensus over weak/generic spans will be expensive and probably still wrong.

Only use consensus after you have:

```text
fixed question/options
fixed span bundles
fixed comparator payload hash
fixed runner-up span
```

Then, if that fixed pair-binding comparator still flips, use narrow 3-call consensus only at that stage.

---

## 7. Minimal architecture change

Build three small modules, not a big rewrite.

### Module A: `hle_option_span_bundle.py`

For each option, construct 2–3 frozen span bundles.

Each bundle should include:

```text
option_label
option_text / aliases
source_id
span_hash
option_overlap_terms
anchor_overlap_terms
relation_overlap_terms
required_terms_present
required_terms_missing
shared_doc_option_count
generic_penalty
directness_score
bundle_type:
  direct_relation | definition | formula | table_row | indirect | generic | contradiction
```

Keep top **2–3**, not just 1. One top span per option is too brittle.

---

### Module B: `hle_pair_binding_comparator.py`

Only compares frozen bundles.

It should answer:

```text
Does this span establish the exact question relation for this option?
Does it beat or refute the runner-up?
```

Not:

```text
Is this source relevant?
What is the answer?
Can you search the context?
```

---

### Module C: `hle_source_binding_selector.py`

Selector policy:

```text
strong direct pair-bound label -> choose source label
weak/generic/ambiguous source -> fallback to baseline/Hippo/raw
conflict between direct candidates -> fallback or use comparator margin
no source candidate -> fallback
```

This is the safe version of Direction 4.

---

## 8. Minimal experiment that would prove this change

The document says improvements on inspected/tuned cohorts must not count as generalization and asks for a new unseen HLE cohort reporting accuracy, fidelity, and stability.  I agree.

### Cohorts

Use three sets:

```text
R0: old fixed 12
    regression only, not generalization.

D0: latest debug holdout 6
    diagnostic only, not acceptance.

G1: new unseen holdout 12 or 24
    actual acceptance set.
```

Do not tune on G1 after seeing labels.

---

### Systems

Compare:

```text
S0 = f577d1a7 accepted baseline
S1 = current branch
S2 = span-bundle lattice only
S3 = span-bundle lattice + pair-binding comparator
S4 = S3 + source-binding fallback to baseline/Hippo/raw
S5 = S4 + full-text/local-snippet expansion only for missing-span options
```

Do not include consensus yet.

---

### Required metrics

#### Accuracy

```text
accuracy_vs_gold
delta_vs_f577d1a7
delta_vs_raw
delta_vs_Hippo
```

Acceptance:

```text
G1 accuracy must beat f577d1a7/Hippo, or tie with clear stability/latency gains.
```

#### Source fidelity

```text
option_with_span_bundle_rate
gold_option_has_bundle_rate
gold_option_direct_bundle_rate
pair_binding_candidate_rate
pair_binding_accept_rate
source_generic_rate
source_ambiguous_rate
source_no_candidate_rate
```

The key proof is:

```text
gold_option_has_bundle_rate ↑
pair_binding_candidate_rate ↑
source_verifier_no_candidate_emitted ↓
relation_span_comparator_rejected ↓
verified_or_abstain no_fallback = 0
```

#### Stability / latency

Run cache-only 3–5 times.

Report:

```text
query_spec_hash_stability
span_bundle_hash_stability
pair_binding_payload_hash_stability
selected_label_stability
unique_model_calls
cache_hit_rate
mean_latency
p95_latency
```

Acceptance:

```text
span_bundle_hash_stability = 1.0
pair_binding_payload_hash_stability = 1.0
selected_label_stability >= 0.95
model calls not > +20% unless accuracy improves
```

---

## 9. Acceptance gate

I would only promote if:

```text
1. Old fixed 12:
   accuracy >= 6/12, or equal with lower variance/latency.

2. New unseen G1:
   accuracy > current accepted baseline or Hippo,
   or equal with clear latency/stability gain.

3. Source path:
   source_verifier_no_candidate_emitted decreases.
   relation_span_comparator_rejected decreases.
   source_quality_directness_promotion_blocked decreases.
   verified_or_abstain no_fallback becomes 0.

4. Stability:
   cache-only path hashes stable across repeated runs.

5. Budget:
   source/model call increase bounded.
```

If it only improves the inspected `1196,2062,1301,2212,1383,2318` cohort, do not promote. The document itself says that result is positive but not clean generalization evidence because rules were added after inspection.

---

## 10. Concrete next ticket for Codex

Give the coding agent this:

```text
Implement pair-bound source selection v1.

Files:
- hle_option_span_bundle.py
- hle_pair_binding_comparator.py
- hle_source_binding_selector.py
- tests/test_hle_pair_bound_source_selection.py

Requirements:
1. For each MC option, build top 2–3 deterministic span bundles from prefetched source records.
2. Each bundle must expose option_overlap, anchor_overlap, relation_overlap, required_term_coverage, shared_doc_penalty, and bundle_type.
3. Comparator sees only fixed bundles, not broad source context.
4. Selector treats pair-bound source as an override only when direct and margin-positive.
5. If source is generic/ambiguous/no-candidate, fallback to baseline/Hippo/raw rather than no_fallback.
6. Emit query_spec_hash, span_bundle_hash, comparator_payload_hash, selector_payload_hash.
7. Add cache-only repeated-run stability test.
8. Add unseen holdout evaluation script with S0–S4 comparisons.
```

---

## 11. Final recommendation among the five options

My ranking:

```text
1. Direction 2 — pair-binding comparator over frozen candidate-specific spans.
2. Direction 3 — targeted full-text/local snippet expansion only when span bundles are missing.
3. Direction 4 — conservative baseline/Hippo fallback when source remains generic/indirect/no-candidate.
4. Direction 1 — deterministic solvers only as a separate formula/table lane.
5. Direction 5 — consensus only after fixed-payload comparator variance is proven.
```

The highest-leverage single change is:

> **Replace generic source directness verification with pair-bound option-span comparison over deterministic span bundles.**

That is the narrowest change that directly attacks the latest observed bottleneck:

```text
source/preferred/shared docs
  -> candidate-specific direct relation span
  -> selected label
  -> accepted comparator candidate
  -> final selection
```
