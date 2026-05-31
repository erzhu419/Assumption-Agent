# Route-Gated Proposal Policy Ablation

Date: 2026-05-31

## Setup

- Mini baseline: `phase2_v20_gpt54mini_prop_union`
- Mini proposal variant: `proposal_ready_combo_se_mh_gpt54mini`
- Mini routed policy: `proposal_ready_combo_se_mh_policy_gpt54mini`
- GPT-5.5 baseline: `phase2_v20_gpt55`
- GPT-5.5 proposal variant: `proposal_ready_combo_se_mh_gpt55_confirm`
- GPT-5.5 hard-only policy: `proposal_ready_combo_se_hard_policy_gpt55`
- Proposal source: `proposals_phase2_v20_gpt55_21_50.json`
- No candidate was applied to the persistent graph.

## Mini Screening

The first combo test forced the six ready proposals only on their own routed
`should_fire` rows and beat the same-model baseline 26-12, but regressions were
concentrated in `daily_life` and `easy` rows.

Adding explicit route filters to proposal forcing produced:

- `proposal_ready_combo_se_mh_gpt54mini`: 25 wins, 12 losses, 1 tie
- Decisive win rate: 67.6%
- Mean score delta: +0.37
- `software_engineering`: 13 wins, 5 losses
- `daily_life`: 5 wins, 2 losses, 1 tie
- `easy`: 2 wins, 4 losses

The safer routed policy uses proposal answers only for
`software_engineering` + `medium|hard` and falls back to baseline elsewhere:

- `proposal_ready_combo_se_mh_policy_gpt54mini`: 13 wins, 5 losses, 20 ties
- Decisive win rate: 72.2%
- Mean score delta: +0.21
- Non-target routes: 20 ties, 0 losses

## Scoped Retrieval Check

`--assumption-route-scope-proposals` removes proposal candidate nodes outside
their routed `should_fire` subset. It improved non-SE/easy safety but suppressed
the software-engineering gain:

- `proposal_ready_combo_se_mh_scoped_gpt54mini`: 25 wins, 13 losses
- Decisive win rate: 65.8%
- `software_engineering`: 9 wins, 9 losses

This flag remains available as an experimental guard, but it is not the current
recommended policy.

## GPT-5.5 Confirmation

On the nine software-engineering `medium|hard` rows, GPT-5.5 confirmed a smaller
positive effect:

- `proposal_ready_combo_se_mh_gpt55_confirm`: 10 wins, 7 losses, 1 tie
- Decisive win rate: 58.8%
- Mean score delta: +0.22

The effect split by difficulty:

- `hard`: 6 wins, 2 losses
- `medium`: 4 wins, 5 losses, 1 tie

Because medium rows disagreed between mini screening and GPT-5.5 confirmation,
the current high-confidence gate is hard-only:

- `proposal_ready_combo_se_hard_policy_gpt55`: 6 wins, 2 losses, 10 ties
- Decisive win rate: 75.0%
- Mean score delta: +0.22
- `medium`: 10 ties by fallback
- `hard`: 6 wins, 2 losses

## Current Decision

Use proposal forcing only under this gate:

- domain: `software_engineering`
- difficulty: `hard`
- route label: proposal target is `should_fire`

Keep `software_engineering` + `medium` as experimental pending more evidence.
Do not apply any candidate to the graph until the acceptance gate passes both
trigger benefit and control-harm thresholds.
