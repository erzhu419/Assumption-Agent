# Phase2 v20 GPT-5.5 Heldout 21-50: Learned Graph + SE Template

Date: 2026-05-31

## Setup

- Baseline: `phase2_v20_gpt55`
- Variant: `phase2_v20_ag_learned_se_template_gpt55`
- Sample: `phase two/analysis/cache/sample_21_50.json`
- Model for generated answers: `gpt-5.5` through environment variables only

The variant is a synthesized answer set:

- non-`software_engineering` answers come from `phase2_v20_ag_learned_gpt55`
- `software_engineering` answers come from `phase2_v20_ag_se_template_gpt55`

This isolates the tested policy decision: keep learned graph context for the
domains where it helped, but replace SE graph injection with the prompt-side SE
execution template.

## Result

Forward judge (`variant` vs `baseline`):

- Variant: 14
- Baseline: 9
- Tie: 7
- Decisive win rate: 60.9%
- Mean delta: +0.10

Reverse judge (`baseline` vs `variant`), normalized:

- Variant: 14
- Baseline: 10
- Tie: 6
- Decisive win rate: 58.3%
- Mean delta from baseline-as-A report: -0.17

Combined bidirectional result:

- Variant: 28
- Baseline: 19
- Tie: 13
- Decisive win rate: 59.6%

For comparison, the previous learned-graph heldout result on the same 30
problems was:

- Learned graph: 21
- Baseline: 25
- Tie: 14
- Decisive win rate: 45.7%

## Domain Breakdown, Combined

- `business`: variant 1, baseline 1, tie 2, decisive win rate 50.0%.
- `daily_life`: variant 5, baseline 1, tie 2, decisive win rate 83.3%.
- `engineering`: variant 7, baseline 0, tie 3, decisive win rate 100.0%.
- `mathematics`: variant 1, baseline 7, tie 0, decisive win rate 12.5%.
- `science`: variant 4, baseline 4, tie 2, decisive win rate 50.0%.
- `software_engineering`: variant 10, baseline 6, tie 4, decisive win rate 62.5%.

## Decision

The best current heldout policy is:

1. use learned graph context for the non-gated domains,
2. keep `software_engineering` graph context skipped by default,
3. inject the SE domain execution template for graph experiment variants,
4. leave the SE reranker as opt-in until it shows positive answer-quality lift.

The remaining weakness is math/science stochasticity. Those rows are hygiene
bypass rows rather than graph-context rows, so they should be evaluated and
stabilized separately from Assumption Graph writeback.
