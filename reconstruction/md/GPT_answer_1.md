# 我：
We are debugging an HLE multiple-choice assumption-agent system. Current baseline is 6/12 at commit f577d1a7, so new changes should only be accepted if they beat that or match accuracy with clear stability/latency gains.

Recent change:
1. Added witness-level semantic comparator for candidate-specific source spans.
2. Wired source verifier audit rows into the comparator.
3. Added cache for duplicate witness comparator calls.
4. Deduped witness comparator candidates to one top witness per option.
5. Tightened source-verifier audit witness admission: required_overlap must be > 0. Relation-only overlap is no longer enough.
6. Added model-router prompt/payload/response hash logging.

Cache-only HLE mini results:
- sourceaudit_probe2: 0/2
- strictcache_probe2: 0/2
- strictcache_dedupe_probe2: 0/2
- requiredgate_probe2: 1/2
- requiredgate_probe4 with seeds 971,145,427,831: 0/4

Important instability:
The same seed 427 was correct in requiredgate_probe2 but wrong in requiredgate_probe4. Both used cache-only source flags, gpt-5.4-mini, temperature 0, no gold in decision path. The stage trace is broadly similar, but upstream planner/verifier response hashes differ. This suggests model-output variance or query-planner/candidate-generation instability, not a stable source/comparator improvement.

Failure buckets remain:
- candidate_generation_missed_gold
- source_verifier_generic / no_selected_label_generic
- candidate_direct_relation_span_directness_rejected
- relation_span_comparator no direct candidate
- verified_or_abstain no_fallback

Question:
Given this, should the next fix target:
A. deterministic/programmatic relation query planning and candidate generation, reducing dependence on model planner variance;
B. stronger answer-bearing local source prefetch/cache coverage;
C. repeated/consensus verifier calls for high-variance stages, despite budget cost;
D. a conservative abstain/selection policy that refuses unstable source paths;
or something else?

The goal is not to overfit the debug seeds, but to improve unseen HLE operator/source-bearing cohorts. What is the highest-leverage next architecture change and what minimal experiment would prove it?

# GPT：
Given these results, **I would not target the comparator first**. The evidence points upstream.

The highest-leverage next architecture change is:

> **A+ B: deterministic, option-conditioned relation-query planning and candidate/witness generation, with stronger local answer-bearing source prefetch.**
> In other words: make the source path deterministic and option-specific before the model ever judges anything.

C, repeated/consensus verifier calls, should be a last resort. D, conservative abstain/selection, is necessary as a safety layer, but it will not create accuracy unless the candidate/witness path actually contains the answer-bearing evidence.

---

# 1. What the current failures imply

Your latest results are not a stable improvement over the 6/12 baseline:

```text
sourceaudit_probe2:          0/2
strictcache_probe2:          0/2
strictcache_dedupe_probe2:   0/2
requiredgate_probe2:         1/2
requiredgate_probe4:         0/4
```

The most important signal is this:

> **Seed 427 was correct in requiredgate_probe2 but wrong in requiredgate_probe4, despite cache-only source flags, GPT-5.4-mini, temperature 0, and no gold in the decision path.**

If the same seed flips when the broader probe changes, and upstream planner/verifier response hashes differ, then the system is not yet testing a stable source-verification mechanism. It is testing a partly model-generated candidate path whose upstream structure can move.

That means the current issue is probably **not**:

```text
semantic comparator is slightly too weak
```

It is more likely:

```text
the gold path is not deterministically generated,
or the option-specific answer-bearing witness is not deterministically retrieved,
or small upstream prompt/candidate differences cause a different verifier path.
```

So the next fix should not be “ask the model harder.” It should be “stop letting the model decide the evidence topology.”

---

# 2. Diagnose the four proposed directions

## A. Deterministic/programmatic relation query planning and candidate generation

**Yes. This is the highest-leverage direction.**

Your failure buckets include:

```text
candidate_generation_missed_gold
source_verifier_generic / no_selected_label_generic
relation_span_comparator no direct candidate
verified_or_abstain no_fallback
```

These are upstream path-construction failures. If the gold candidate is not generated, or if the verifier sees generic source text instead of option-specific answer-bearing spans, no semantic comparator can reliably recover.

The model should not be responsible for deciding:

```text
which option to generate evidence for
which relation query to try
which source span counts as candidate-specific
which witness gets admitted
```

Those should be deterministic, programmatic, and option-conditioned.

---

## B. Stronger answer-bearing local source prefetch/cache coverage

**Yes, but as part of A, not instead of A.**

B alone does not solve the problem if the query planner misses the gold relation or never asks for the right option-specific evidence.

The right version is:

```text
For every answer option:
  deterministically generate relation queries
  prefetch candidate-specific local spans
  score answer-bearingness programmatically
  only then ask the model/comparator to judge fixed witnesses
```

So B should be embedded inside an option-conditioned candidate lattice.

---

## C. Repeated/consensus verifier calls

**Not yet.**

Consensus helps only after the candidate/witness set is frozen. If you run repeated verifier calls while the planner path is unstable, you are just averaging over upstream chaos.

Use C only later, and only for a narrow stage:

```text
same fixed option
same fixed source span
same fixed comparator prompt
same payload hash
different model calls
```

If that fixed comparator still has high variance, then 3-call consensus may be worth it. But do not apply consensus to the planner/candidate-generation stage. That will increase cost and may hide the real bug.

---

## D. Conservative abstain/selection policy

**Necessary, but not sufficient.**

D protects you from regressions. It does not create new correct answers unless the evidence path is good.

For multiple-choice HLE, I would avoid a pure “verified_or_abstain no_fallback” policy if the task requires an answer. Instead:

```text
if verified source path is strong:
    use source-backed label
elif source path is unstable or generic:
    fall back to native/baseline answer
else:
    abstain only if abstention is allowed by evaluation
```

Right now `verified_or_abstain no_fallback` is one of the failure buckets. That suggests the system is throwing away baseline signal when source verification fails. Since your baseline is 6/12, source verification should be an **override**, not a replacement, until it proves itself.

---

# 3. The core architectural fix: deterministic option-conditioned evidence lattice

The next architecture should be:

```text
Question + answer options
  -> deterministic option-conditioned query planner
  -> local source prefetch per option
  -> programmatic witness admission
  -> fixed witness comparator
  -> conservative selector / fallback
```

Not:

```text
model planner
  -> source verifier
  -> comparator
  -> unstable selected label
```

## Stage 1: Parse options into canonical entities

For each option:

```json
{
  "label": "A",
  "surface": "...",
  "aliases": [],
  "normalized_entity": "...",
  "entity_tokens": [],
  "disambiguation_terms": []
}
```

Add aliases aggressively. Your new `required_overlap > 0` rule is good, but only if overlap is computed against:

```text
surface form
aliases
abbreviations
canonical entity name
known synonyms
```

Otherwise you will falsely reject good spans where the source uses an alias.

---

## Stage 2: Deterministic relation-query planning

For each option, generate a fixed set of query specs:

```json
{
  "query_id": "q_A_relation_03",
  "option_label": "A",
  "relation_template": "X causes Y",
  "anchor_terms": [],
  "option_terms": [],
  "required_overlap_terms": [],
  "source_scope": "...",
  "fallback_query": false
}
```

This should be programmatic. The model can help build the template library offline, but during the HLE run the planner should be deterministic.

Example query families:

```text
direct_entity_relation:
  question anchor + option entity + relation words

definition_or_identity:
  option entity + definition cues

cause_effect:
  option entity + causes / leads to / associated with

temporal_or_origin:
  option entity + date / origin / introduced / first

comparison:
  option entity + compared with anchor entity

exclusion:
  option entity + not / except / incompatible
```

Each query should have a stable ID and stable string. Sort all query specs deterministically.

---

## Stage 3: Answer-bearing source prefetch per option

For each option, retrieve local spans. Do not retrieve generic source chunks and later hope the verifier maps them.

Each witness should be:

```json
{
  "witness_id": "...",
  "option_label": "A",
  "source_id": "...",
  "span_start": 1234,
  "span_end": 1480,
  "span_text_hash": "...",
  "anchor_overlap": 2,
  "option_overlap": 1,
  "relation_overlap": 1,
  "directness_score": 0.74,
  "generic_penalty": 0.10
}
```

Admission should require:

```text
option_overlap > 0
anchor_overlap > 0 or strong question-anchor alias
relation_overlap > 0 for relation questions
directness_score >= threshold
```

Your `required_overlap > 0` is directionally right, but make it **option-specific overlap**, not just any overlap.

---

## Stage 4: Do not dedupe to one top witness too early

Your recent “one top witness per option” dedupe may reduce cost, but it is brittle. If the top witness is generic or slightly wrong, the gold evidence disappears.

Better:

```text
keep top 2–3 programmatic witnesses per option
only call LLM comparator on the best admitted witness after programmatic filtering
```

Or:

```text
keep top 2 direct witnesses per option
collapse only after directness and answer-bearingness checks
```

For n=4 multiple choice, top 2 per option is only 8 witnesses. That is affordable if most of the scoring is programmatic and cached.

---

## Stage 5: Comparator only sees frozen witnesses

The semantic comparator should receive:

```text
option A fixed witness
option B fixed witness
...
question relation
```

It should not receive open-ended raw source chunks or be asked to invent the candidate path.

The comparator’s job should be narrow:

```text
Does this witness directly support option A as the answer to this question?
```

not:

```text
Find the answer in this source context.
```

---

## Stage 6: Conservative selector with baseline fallback

Selection policy:

```python
if one option has verified direct witness and margin >= m:
    choose that option
elif multiple options have conflicting verified witnesses:
    fallback_to_baseline_or_abstain
elif no direct candidate:
    fallback_to_baseline_or_abstain
elif all witnesses are generic:
    fallback_to_baseline_or_abstain
```

For HLE multiple-choice, unless abstention is part of scoring, I would use:

```text
fallback_to_native_baseline_answer
```

not:

```text
no_fallback
```

Your source verifier should improve baseline when it has a strong path, not replace baseline everywhere.

---

# 4. Why this addresses each failure bucket

## `candidate_generation_missed_gold`

Fix:

```text
option-conditioned deterministic query templates
top-k witnesses per option
alias expansion
```

Measure:

```text
gold_option_has_admitted_witness_rate
```

---

## `source_verifier_generic / no_selected_label_generic`

Fix:

```text
every verifier row must carry option_label + source_span + relation_template
generic source chunks are not admissible
```

Measure:

```text
generic_witness_admission_rate
```

This should go down.

---

## `candidate_direct_relation_span_directness_rejected`

This could mean two different things:

```text
1. The directness gate is correctly rejecting generic evidence.
2. The gate is too literal and rejects valid alias/paraphrase evidence.
```

Fix:

```text
add alias/paraphrase relation matching
add bridge witness class for non-direct but valid two-hop evidence
```

Do not simply loosen directness. Add structured witness types:

```text
direct_relation_witness
definition_witness
bridge_witness
negative_witness
```

---

## `relation_span_comparator no direct candidate`

Fix:

```text
if no direct candidate:
    run deterministic fallback query family before comparator
```

Fallback query families:

```text
option entity + definition
option entity + question anchor
option entity + relation synonym
question anchor + relation synonym
```

If still none:

```text
source path unavailable -> fallback baseline
```

---

## `verified_or_abstain no_fallback`

Fix:

```text
verified source path is an override, not mandatory path.
```

Use:

```text
source_verified_label if strong
else baseline_label
```

until source verification beats baseline on a decent holdout.

---

# 5. Why not consensus now?

Consensus is attractive because you observed hash instability. But the instability is upstream:

```text
planner/verifier response hashes differ
stage trace broadly similar
same seed correct in one probe, wrong in another
```

That means consensus would likely average over different candidate worlds.

Before consensus, enforce:

```text
same input question
same options
same relation query specs
same source span IDs
same comparator prompt hash
same payload hash
same cache key
```

Then test variance.

If the **fixed comparator** still flips, then use C:

```text
3-call comparator consensus only for high-variance fixed witness pairs
```

But not before.

---

# 6. The minimal experiment that would prove the fix

You need a small but clean experiment. Do not tune on the debug seeds.

## Experiment name

```text
deterministic_option_witness_lattice_v1
```

## Dataset

Use three sets:

```text
A. old debug 12
B. new unseen cache-only 24
C. new unseen live or cache+live 24
```

If budget is tight:

```text
A. old 12
B. unseen 12
```

But do not judge success only on the old seeds.

## Systems compared

```text
S0: f577d1a7 baseline
S1: current requiredgate
S2: deterministic option-witness lattice + baseline fallback
S3: S2 + fixed-witness comparator
S4: S2 + fixed-witness comparator + narrow consensus only if fixed comparator variance > threshold
```

S4 is optional. The key comparison is S0/S1/S2/S3.

---

## Metrics

### Accuracy

```text
accuracy_overall
accuracy_source_bearing_subset
accuracy_no_source_subset
```

Acceptance:

```text
S2/S3 must beat 6/12 on old 12,
and beat baseline on unseen set,
or match accuracy with lower variance and lower latency.
```

For n=12:

```text
>= 7/12 is the minimum interesting win.
```

But for unseen 24, require something like:

```text
+2 correct over baseline
or non-inferior accuracy with significant stability/latency gain.
```

---

### Candidate coverage

```text
gold_option_candidate_generated_rate
gold_option_admitted_witness_rate
gold_option_direct_witness_rate
```

This is the most important diagnostic.

If accuracy does not improve but gold witness coverage improves, the next bug is selector/comparator.

If gold witness coverage does not improve, the next bug is query/source prefetch.

---

### Stability

Run the exact same cache-only set 5 times.

Require:

```text
pre_comparator_candidate_path_hash_identical_rate = 1.0
source_witness_set_hash_identical_rate = 1.0
selected_label_identical_rate >= 0.95
```

If model comparator is cached, selected label should be deterministic. If not, you have a cache-key/payload canonicalization bug.

---

### Generic-source reduction

```text
generic_witness_admission_rate
no_selected_label_generic_count
source_verifier_generic_count
```

These should drop.

---

### Fallback behavior

```text
no_fallback_count = 0
fallback_to_baseline_count
fallback_correct_rate
```

If fallback_to_baseline saves errors, that proves source verification should be an override.

---

### Latency/cost

```text
unique_model_calls
cache_hit_rate
mean_latency
p95_latency
```

Programmatic planning should reduce planner calls. If accuracy matches baseline but variance and latency improve, that may be acceptable under your rule.

---

# 7. Concrete acceptance gate

I would accept the next change only if it passes all of these:

```text
1. On old 12:
   accuracy >= 7/12
   OR accuracy = 6/12 with selected_label_identical_rate >= 0.95 and lower latency.

2. On unseen 24:
   accuracy >= baseline + 2 correct
   OR accuracy non-inferior with strong stability and latency gains.

3. Candidate path stability:
   deterministic candidate/witness path hash identical across 5 cache-only reruns.

4. Gold witness coverage:
   gold_option_admitted_witness_rate improves materially over current.

5. No fallback hole:
   verified_or_abstain no_fallback count = 0.

6. Generic evidence:
   source_verifier_generic and no_selected_label_generic decrease.

7. Budget:
   unique model calls do not increase by more than 20% unless accuracy gain is clear.
```

---

# 8. Implementation sketch

## `hle_deterministic_relation_planner.py`

```python
def build_option_query_specs(question, options, relation_templates):
    specs = []
    for option in options:
        aliases = expand_aliases(option.text)
        for template in relation_templates:
            specs.append(QuerySpec(
                option_label=option.label,
                option_aliases=aliases,
                relation_template_id=template.id,
                query=template.render(question, option, aliases),
                required_anchor_terms=extract_question_anchors(question),
                required_option_terms=aliases,
                required_relation_terms=template.relation_terms,
            ))
    return sorted(specs, key=lambda s: s.query_id)
```

---

## `hle_option_witness_prefetch.py`

```python
def prefetch_witnesses(query_specs, source_cache):
    witnesses = []
    for spec in query_specs:
        spans = source_cache.retrieve(spec.query, top_k=K)
        for span in spans:
            witness = score_witness(spec, span)
            if witness.option_overlap > 0 and witness.directness_score >= threshold:
                witnesses.append(witness)
    return stable_top_n_per_option(witnesses, n=2)
```

---

## `hle_witness_selector.py`

```python
def select_label(option_witnesses, baseline_label):
    verified = [w for w in option_witnesses if w.verified and w.directness_score >= threshold]
    if not verified:
        return baseline_label, "fallback_no_verified_witness"

    by_option = best_by_option(verified)
    best, second = top_two(by_option)

    if best.score - second.score >= margin:
        return best.option_label, "source_verified_override"

    return baseline_label, "fallback_low_margin_or_conflict"
```

---

## `hle_path_hashing.py`

For every row log:

```json
{
  "question_hash": "...",
  "option_hashes": {},
  "query_spec_hash": "...",
  "source_witness_hash": "...",
  "comparator_payload_hash": "...",
  "selector_payload_hash": "...",
  "selected_label": "C"
}
```

If seed 427 flips again, this will show exactly where.

---

# 9. Important caution: do not over-tighten directness

You tightened witness admission to require `required_overlap > 0`. Good. But if this uses only surface overlap, it will reject valid witnesses with alias/paraphrase.

So add:

```text
canonical aliases
abbreviations
normalized entity forms
relation paraphrases
```

Also, do not collapse to one witness too early. Keep top 2–3 per option until programmatic directness and comparator have run.

---

# 10. Summary recommendation

My answer to your A/B/C/D question:

```text
Primary: A
But implement it as A+B:
  deterministic option-conditioned query planning
  + stronger answer-bearing local source prefetch/cache coverage.

Secondary: D
  conservative selector with fallback-to-baseline,
  not no_fallback.

Defer: C
  repeated/consensus calls only after the candidate/witness payload is frozen
  and fixed-comparator variance is proven.
```

The highest-leverage architecture change is:

> **Replace model-planned source paths with a deterministic option-conditioned evidence lattice. Let the model compare only fixed, candidate-specific witnesses. Use verified evidence only as a high-confidence override; otherwise fall back to baseline.**

The minimal proof is:

> **On old 12 plus an unseen HLE source-bearing cohort, the new lattice must improve gold-witness admission, eliminate path hash instability, reduce generic verifier failures, and beat or non-inferiorly match the 6/12 baseline with lower variance/latency.**
