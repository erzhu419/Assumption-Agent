# Phase2 v20 GPT-5.5 Heldout 21-50: Learned Graph + SE Template vs Raw Baseline

Date: 2026-05-31

## Setup

- Baseline: `baseline`
- Variant: `phase2_v20_ag_learned_se_template_gpt55`
- Sample: `phase two/analysis/cache/sample_21_50.json`
- Judge model: `gpt-5.5` through environment variables only

`baseline` is the plain cached framework baseline prompt, without Phase2 v20
frame rewrite, wisdom/cases, Assumption Graph, or SE execution template.

## Result

Forward judge (`variant` vs `baseline`):

- Variant: 28
- Baseline: 2
- Tie: 0
- Decisive win rate: 93.3%
- Mean delta: +1.50

Reverse judge (`baseline` vs `variant`), normalized:

- Variant: 27
- Baseline: 3
- Tie: 0
- Decisive win rate: 90.0%
- Mean delta from baseline-as-A report: -1.43

Combined bidirectional result:

- Variant: 55
- Baseline: 5
- Tie: 0
- Decisive win rate: 91.7%

## Domain Breakdown, Combined

- `business`: variant 4, baseline 0, tie 0, decisive win rate 100.0%.
- `daily_life`: variant 8, baseline 0, tie 0, decisive win rate 100.0%.
- `engineering`: variant 10, baseline 0, tie 0, decisive win rate 100.0%.
- `mathematics`: variant 4, baseline 4, tie 0, decisive win rate 50.0%.
- `science`: variant 9, baseline 1, tie 0, decisive win rate 90.0%.
- `software_engineering`: variant 20, baseline 0, tie 0, decisive win rate 100.0%.

## Loss Pattern

The five baseline wins are concentrated in bypass domains:

- `mathematics_0251`, both directions: baseline gave a more specific
  Chowla-Selberg / CM-theory bridge; the variant was judged more rigorous but
  too generic.
- `mathematics_0192`, both directions: baseline addressed the research decision
  tradeoff and counterexample pressure more concretely; the variant leaned more
  formal.
- `science_0273`, one direction: baseline gave more actionable publication,
  graduation, and communication steps.

## Interpretation

The current policy is clearly above the no-method raw baseline on this heldout
slice. Remaining losses are not Assumption Graph failures: math/science use the
v20 hygiene bypass and do not receive graph context. The next improvement should
therefore separate math/science stabilization from graph-memory evolution.
