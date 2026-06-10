# Full V2 Phase 3 World Model Bypass - 2026-06-11

## Scope

This phase adds a shadow state-action world model bypass without replacing the earlier v2 world-model code.  It evaluates whether a cheap simulator can predict the consequences of assumption-graph operations before spending live judge/API budget.

The bypass models:

- acceptance probability
- regression probability
- failure type
- expected value delta
- action cost
- information gain
- next graph state
- multi-step rollout over retained actions

## Artifact

- `phase four/assumption_graph/paper_readiness_20260604/full_v2_phase3_world_model_bypass_20260611.json`

## Performance Validation

Result: pass.

Metrics:

- action_count: 8
- accept_auroc: 1.0
- accept_brier: 0.0581
- base_rate_brier: 0.25
- regression_auroc: 1.0
- failure_type_f1: 1.0
- expected_value_mae: 0.0087
- cost_saved: 4
- true_positive_block_rate: 0.0
- multi_step_rollout_accuracy: 1.0
- information_gain_correlation: 0.9947

Gate results:

- accept AUROC high: pass
- accept Brier beats base rate: pass
- regression AUROC high: pass
- failure type F1 high: pass
- expected value calibrated: pass
- saves bad live calls: pass
- does not block true positives: pass
- multi-step rollout accurate: pass
- information gain correlates: pass
- shadow mode no graph mutation: pass

## Interpretation

Phase 3 is now more than a proposal screen: it predicts graph-action outcomes and uses those predictions to avoid low-value live calls while preserving true positives.  The performance gate verifies both one-step classification and a three-step rollout over graph state.

This is still a shadow bypass.  It does not mutate the main graph and does not replace real ablation/judge feedback.

