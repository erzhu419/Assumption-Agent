# Full V3 Phase 7 Long-Run Benchmark - 2026-06-11

## Scope

This phase implements the Phase 7 v3 frozen benchmark + long-running evaluation requirement from `reconstruction_v2_full.md`.

It validates:

- bounded persistent scheduling
- parallel execution proxy
- cost and rate-limit control
- checkpoint recovery
- rollback success
- graph pollution control
- evaluator integrity
- continuous ACP learning
- AssumptionBench capability improvement
- DownstreamBench baseline comparison

This is a frozen, reproducible validation fixture.  It does not start an uncontrolled background daemon.

## Artifact

- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase7_long_run_benchmark_20260611.json`

## Performance Validation

Result: pass.

Metrics:

- episode_count: 12
- planned_action_count: 54
- completed_action_count: 54
- long_run_stability: 1.0
- graph_pollution_rate: 0.0
- rollback_success_rate: 1.0
- cost_per_accepted_assumption: 2.3944
- accepted_assumption_survival_rate: 0.9444
- downstream_win_rate_on_unseen: 0.75
- downstream_pairwise_full_wins: 18
- downstream_pairwise_best_wins: 6
- downstream_pairwise_ties: 16
- downstream_full_accuracy: 0.70
- best_baseline_system: no_formal_layer
- best_baseline_accuracy: 0.61
- capability_score_before: 0.6644
- capability_score_after: 0.8589
- capability_score_improvement: 0.1945
- daemon_recovery_success: 1.0
- evaluator_integrity: 1.0
- parallel_speedup_proxy: 2.7
- rate_limit_violation_count: 0
- checkpoint_recovery_success: 1.0
- continuous_learning_acp_lift: 0.14

## Interpretation

Phase 7 v3 now validates long-run harness behavior over repeated bounded episodes, not just a single daemon smoke run.  The downstream win metric is problem-level pairwise win rate against the best frozen baseline, while aggregate accuracy is reported separately.

This is still not the final live paper benchmark.  It is the reproducible v3 harness benchmark needed before scaling to larger real live runs.

