# Full V2 Vertical Slice Bypass - 2026-06-11

## Scope

This artifact validates the full v2 slice described at the end of `reconstruction_v2_full.md`:

```text
frozen residual cluster
  -> competing hypotheses
  -> candidate overlays
  -> world-model prospective screen
  -> fresh ablation / verifier
  -> accepted-only graph update
  -> world-model calibration
  -> repeat for 5 generations
```

It does not replace any Phase 0-7 module.  It checks whether the phase bypasses compose into one recursive assumption-evolution loop.

## Artifact

- `phase four/assumption_graph/paper_readiness_20260604/full_v2_vertical_slice_bypass_20260611.json`

## Performance Validation

Result: pass.

Metrics:

- generation_count: 5
- candidate_count: 25
- selected_for_fresh_ablation_count: 10
- live_call_saving_rate: 0.6
- true_positive_block_rate: 0.0
- accepted_count: 5
- accepted_assumption_survival_rate: 1.0
- residual_explained_start: 0.22
- residual_explained_final: 0.91
- residual_explained_delta: 0.69
- downstream_score_start: 0.48
- downstream_score_final: 0.70
- downstream_score_delta: 0.22
- graph_pollution_rate: 0.0
- world_model_brier_start: 0.165
- world_model_brier_final: 0.064
- world_model_brier_improvement: 0.101
- full_loop_downstream_score: 0.70
- best_control_system: generator_world_model_no_verifier
- best_control_downstream_score: 0.60
- full_loop_margin_over_best_control: 0.10
- full_loop_residual_explained_margin: 0.23

## Controls

The slice compares the full loop against:

- no_evolution
- one_shot_llm_new_wisdom
- graph_retrieval_only
- generator_without_world_model
- generator_world_model_no_verifier
- full_recursive_assumption_loop

## Interpretation

This is the first full-v2 composition artifact: the individual phase bypasses now form a five-generation recursive loop with prospective screening, verifier gating, accepted-only retention, and calibration improvement.

This remains a bounded frozen-fixture validation.  The next step for paper-grade evidence is to run the same slice on larger unseen task sets with real live ablation/judging and problem-level bootstrap confidence intervals.

