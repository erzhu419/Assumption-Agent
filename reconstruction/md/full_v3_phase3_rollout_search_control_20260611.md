# Full V3 Phase 3 Rollout Search Control - 2026-06-11

## Scope

This phase implements the Phase 3 v3 "Doctor Strange" world-model requirement from `reconstruction_v2_full.md`.

Instead of scoring one candidate at a time, the world model evaluates ten candidate futures for one residual cluster, rolls each branch out over a three-step graph future, predicts descendant productivity, and selects only the highest-value branches for live validation.

## Artifact

- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase3_rollout_search_control_20260611.json`

## Performance Validation

Result: pass.

Metrics:

- branch_count: 10
- rollout_horizon: 3
- selected_for_live_count: 4
- live_call_saving_rate: 0.6
- top_branch_precision: 1.0
- true_positive_block_rate: 0.0
- multi_step_rollout_accuracy: 1.0
- descendant_productivity_correlation: 0.9995
- expected_value_mae: 0.0374
- regression_recall: 1.0
- oracle_regret: 0.0
- cost_saved: 6.0

## Interpretation

Phase 3 v3 now validates the world model as search control rather than only a cheap yes/no gate.  It spends live budget on high-information, high-productivity branches, blocks risky branches, and matches the oracle top-k on this frozen branch fixture.

