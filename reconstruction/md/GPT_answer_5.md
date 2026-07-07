# 我：
We fixed several false-positive paths in an HLE MC assumption-agent: model-only span directness, pre-directness comparator skipping candidate directness, source-quality promotion overriding source-verifier lane guard, and numeric option anchors in source prefetch. On seed1079, the system now avoids wrong promotion but still gets 0/1 and ends in no_fallback. Source prefetch now retrieves rows for numeric option “-78 C”, but answer-bearing score remains low and no direct candidate is accepted.

Should the next architecture change be:
A. stronger full-text/source cache retrieval for numeric/scientific threshold questions;
B. a general numeric-threshold ordered-option solver backed by source spans;
C. a stricter source-verifier/comparator redesign;
D. something else?

What minimal unseen-cohort experiment would prove it without overfitting seed1079?

# GPT：
I would choose **B as the primary next architecture change**, with **D as a mandatory safety policy**, and **A only as a targeted support layer**.

Not C first.

The current signal is:

```text
false-positive promotions were fixed
numeric option “-78 C” now retrieves rows
answer-bearing score remains low
no direct candidate is accepted
system ends in no_fallback
```

That means the problem has shifted from:

```text
wrong source path gets over-promoted
```

to:

```text
the system has no robust path for numeric / ordered threshold options when direct source evidence is weak or not phrased as a normal relation sentence.
```

So the next architecture should be:

> **A general numeric-threshold ordered-option solver backed by source spans, plus fallback-to-baseline when the numeric/source path is inconclusive.**

---

# 1. Why B is higher leverage than A or C

## A alone: stronger full-text/source cache retrieval

A may help, but it does not solve the core issue.

You already said source prefetch retrieves rows for the numeric option “-78 C”. The problem is that the system still gives the source low answer-bearing score and accepts no direct candidate. So more retrieval may just add more weak numeric mentions unless the system knows how to interpret numeric options.

A should be used only when the numeric solver says:

```text
no source span contains a parseable numeric value / unit / relation
```

Then fetch more local/full-text snippets.

## C: stricter source-verifier/comparator redesign

You just fixed several false-positive paths:

```text
model-only span directness
pre-directness comparator skipping candidate directness
source-quality promotion overriding source-verifier lane guard
numeric option anchors in source prefetch
```

So stricter C would likely increase false negatives. Seed1079 already avoids wrong promotion; the failure is now no accepted candidate and no fallback. Tightening verifier logic further will probably worsen `no_fallback`.

## B: numeric-threshold ordered-option solver

Numeric MC options are special. They are not like normal entity options.

For normal options, the source verifier asks:

```text
Does this source span support option B?
```

For numeric options, the better question is:

```text
What numeric value / threshold / interval does the source or problem imply?
Which option is closest, equal, within range, above, below, earliest/latest, min/max, or satisfies the direction?
```

A span can be answer-bearing even if it does not “semantically argue” for the option in prose. It may simply state:

```text
temperature = -78 °C
threshold below -78 °C
at dry-ice acetone temperature
transition occurs around 195 K
```

The system needs a numeric relation solver that can bind these to ordered options.

---

# 2. Recommended architecture

Add a new lane:

```text
numeric_threshold_lane
```

It should sit beside the source verifier lane and the baseline/raw/Hippo fallback lane.

Current rough flow:

```text
question + options
  -> source prefetch
  -> source verifier
  -> comparator
  -> selection or no_fallback
```

New flow:

```text
question + options
  -> option matrix
      -> source-bound witness lane
      -> numeric-threshold solver lane
      -> baseline/raw/Hippo lane
  -> conservative selector
```

---

# 3. Numeric-threshold solver design

## 3.1 Parse numeric options

For each answer option, extract:

```json
{
  "label": "C",
  "raw_text": "-78 C",
  "value": -78.0,
  "unit": "degC",
  "normalized_value": 195.15,
  "normalized_unit": "K",
  "value_type": "temperature",
  "is_range": false,
  "inequality": null
}
```

It should handle:

```text
-78 C
−78 °C
195 K
0.1 M
5%
10^6
1.2 × 10^-3
pH 7.4
< 5 mm
between 3 and 5
```

For temperatures, normalize:

```text
°C <-> K <-> °F
```

For other units, start with simple families:

```text
temperature
time
length
mass
concentration
percentage
energy
frequency
pressure
```

---

## 3.2 Infer question relation type

The question should be classified into numeric relation families:

```text
exact_value
threshold_minimum
threshold_maximum
below_threshold
above_threshold
closest_value
ordered_extreme_lowest
ordered_extreme_highest
range_membership
unit_conversion
formula_derived_value
```

Also infer direction:

```text
higher_is_correct
lower_is_correct
closest_is_correct
within_range_is_correct
```

Example:

```text
“at what temperature...”
  -> exact_value / closest_value

“below which temperature...”
  -> threshold_maximum or below_threshold

“minimum temperature required...”
  -> threshold_minimum

“which value is closest...”
  -> closest_value
```

This relation type should be deterministic or at least cached with a payload hash.

---

## 3.3 Extract numeric witnesses from source spans

For each source span, parse:

```json
{
  "source_id": "...",
  "span_hash": "...",
  "numbers": [
    {
      "raw": "-78 °C",
      "value": -78,
      "unit": "degC",
      "normalized_value": 195.15,
      "normalized_unit": "K"
    }
  ],
  "relation_terms": ["temperature", "threshold", "reaction", "observed"],
  "anchor_overlap": 2,
  "option_value_match": true,
  "question_relation_match": true,
  "generic_penalty": 0.1
}
```

This should be programmatic before asking an LLM.

The key improvement over the current source verifier:

> A numeric source span can be direct because it states the numeric value, even if it does not contain an option label in natural-language form.

---

## 3.4 Score each option numerically

For each option:

```text
score =
  value_match_score
+ unit_match_score
+ relation_match_score
+ anchor_overlap_score
+ source_quality_score
- generic_penalty
- conflicting_span_penalty
```

Cases:

```text
source says -78 °C, option is -78 C:
  exact match

source says 195 K, option is -78 C:
  unit-converted match

source says below -78 °C, options are -40, -78, -120:
  relation direction matters

source says between -80 and -70 °C:
  range membership
```

---

## 3.5 Conservative selector

Do not let numeric solver always override. Use it only when margin is strong.

```python
if numeric_solver.has_unique_high_confidence_label:
    choose numeric_label

elif source_pair_binding.has_direct_high_margin_label:
    choose source_label

elif numeric_solver and source_pair_binding agree:
    choose agreed_label

else:
    fallback baseline/raw/Hippo
```

Most importantly:

```text
no_fallback_count must become 0
```

For MC HLE, no_fallback is usually worse than a baseline guess.

---

# 4. Where A fits

A should be targeted, not global.

Use stronger source/full-text retrieval only when:

```text
numeric options are detected
and no span has a parseable numeric value with relevant unit/relation
```

Targeted queries should include:

```text
question anchor terms
option numeric values
unit variants
relation terms
entity names
```

For “-78 C”:

```text
"-78 °C"
"-78 C"
"195 K"
"dry ice acetone"
"temperature"
question anchor/entity terms
```

Do not fetch more text for every option blindly.

---

# 5. Where C fits

C should become **numeric-aware**, not merely stricter.

The source verifier should accept a new witness type:

```text
numeric_direct_witness
```

This witness does not need prose like “therefore option C is correct.” It needs:

```text
numeric value match
unit normalization match
question relation match
anchor/context overlap
```

So C should be:

```text
source verifier redesign to include numeric witness semantics
```

not:

```text
stricter comparator blocking
```

---

# 6. Minimal unseen-cohort experiment

Do not promote based on seed1079. Treat seed1079 as diagnostic only.

## Cohorts

Use three sets:

```text
D0: seed1079
    diagnostic only; not promotion evidence.

R0: old fixed 12
    regression guard.

G1: fresh unseen numeric/scientific threshold cohort, n=24
    acceptance set.
```

If budget is very tight:

```text
G1 n=12
```

But 24 is much better.

---

## G1 composition

Stratify the unseen cohort:

```text
6 exact numeric value questions
6 threshold / above-below questions
6 unit conversion or equivalent value questions
6 numeric distractor / source-bearing scientific questions
```

Examples of categories:

```text
temperature threshold
concentration / dose
wavelength / energy / frequency
percentage / probability
time / age / date
pH / pKa / log-scale
physics formula numeric
chemistry numeric condition
```

Do not include seed1079-like wording only. You want category coverage, not clone coverage.

---

## Systems to compare

```text
S0: f577d1a7 baseline
S1: current false-positive-fixed branch
S2: current + stronger source retrieval only
S3: numeric-threshold solver only + fallback
S4: numeric-threshold solver + source spans + fallback
S5: S4 + targeted full-text/source expansion when numeric evidence missing
```

Optional:

```text
S6: S5 + pair-binding comparator
```

But do not add consensus yet.

---

## Metrics

### Accuracy

```text
accuracy_R0
accuracy_G1
delta_vs_f577
delta_vs_current
delta_vs_raw
delta_vs_Hippo
```

Promotion gate:

```text
R0 >= f577 baseline
G1 > current and >= f577 baseline
```

If baseline is 6/12, then on G1 n=24 I would want:

```text
+2 correct over current
```

or non-inferior accuracy with clear stability/latency gains.

---

### Numeric path fidelity

```text
numeric_option_parse_rate
unit_normalization_success_rate
question_relation_classification_rate
gold_option_value_match_rate
gold_option_numeric_witness_rate
numeric_solver_activation_rate
numeric_solver_override_accuracy
numeric_solver_false_positive_rate
```

Important:

```text
numeric_solver_false_positive_rate
```

must stay low. Otherwise the solver will overfit seed1079-like cases.

---

### Source path metrics

```text
numeric_source_span_found_rate
numeric_direct_witness_accept_rate
generic_numeric_span_rejected_rate
source_verifier_no_candidate_emitted_count
source_quality_directness_promotion_blocked_count
relation_span_comparator_rejected_count
```

You want:

```text
source_verifier_no_candidate_emitted ↓
numeric_direct_witness_accept_rate ↑
```

---

### Fallback metrics

```text
no_fallback_count
fallback_to_baseline_count
fallback_correct_rate
solver_override_count
solver_override_correct_rate
source_override_count
source_override_correct_rate
```

Acceptance:

```text
no_fallback_count = 0
```

---

### Stability

Run cache-only 3–5 times.

Require:

```text
numeric_parse_hash_stability = 1.0
relation_classification_hash_stability = 1.0
numeric_witness_hash_stability = 1.0
selected_label_stability >= 0.95
```

The seed427 issue showed why this matters.

---

# 7. Ablation interpretation

## If S2 helps but S3 does not

Then retrieval was the bottleneck, not numeric reasoning.

## If S3 helps but S2 does not

Then source retrieval was not the issue; ordered numeric reasoning was missing.

## If S4 helps more than S3

Then source spans are useful when interpreted through numeric solver semantics.

## If S5 helps only slightly but costs a lot

Keep targeted expansion, not global full-text.

## If S3/S4 hurt non-numeric questions

Your numeric detector is too broad. Add anti-triggers.

---

# 8. Concrete ticket for implementation

Give the coding agent this:

```text
Implement HLE numeric-threshold option solver v1.

Files:
- hle_numeric_option_parser.py
- hle_numeric_relation_classifier.py
- hle_numeric_source_witness.py
- hle_numeric_threshold_solver.py
- hle_numeric_lane_router.py
- tests/test_hle_numeric_threshold_solver.py

Requirements:
1. Parse numeric answer options, including negative values, units, ranges, inequalities, and unit variants.
2. Normalize temperature units at minimum: C, °C, K, F.
3. Classify question relation: exact_value, closest_value, threshold_minimum, threshold_maximum, above/below, range_membership, formula_derived.
4. Extract numeric witnesses from source spans with value/unit/relation/anchor overlap.
5. Score options by numeric match, relation direction, unit compatibility, and source quality.
6. Add `numeric_direct_witness` to source verifier semantics.
7. Selector may override only on unique high-confidence numeric/source margin.
8. If no high-confidence numeric/source label, fallback to baseline/raw/Hippo; never no_fallback.
9. Emit parse_hash, relation_hash, numeric_witness_hash, router_payload_hash.
10. Add unseen n=12/24 numeric-threshold evaluation script.
```

---

# 9. Final answer

I would choose:

```text
Primary: B
  general numeric-threshold ordered-option solver backed by source spans.

Secondary: A
  targeted stronger full-text/source cache retrieval only when numeric witness coverage is missing.

Also required: D-like fallback
  no more no_fallback when numeric/source path is weak.

Not primary: C
  do not merely make source verifier stricter;
  make it numeric-aware by accepting numeric_direct_witness.

Not yet:
  consensus/repeated calls.
```

The minimal proof is:

> On a fresh unseen numeric/scientific threshold cohort, the numeric solver must improve accuracy or match baseline with clear stability gains, reduce `source_verifier_no_candidate_emitted`, create accepted `numeric_direct_witness` rows, and eliminate `no_fallback`, without regressing the old fixed 12 baseline.

Seed1079 should only be used as a regression diagnostic:

```text
it should no longer wrongly promote;
it should either solve via numeric threshold evidence
or safely fallback instead of no_fallback.
```
