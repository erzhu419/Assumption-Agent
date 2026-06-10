# Full V2 Phase 4 Hypothesis Generator Bypass - 2026-06-11

## Scope

This phase adds a shadow multi-layer hypothesis generator bypass.  It does not replace the existing v2 residual generator and does not mutate the committed graph.

The bypass validates the Phase 4 requirement from `reconstruction_v2_full.md`: generate higher-level assumption families from systematic residual clusters rather than emitting local template repairs.

Pipeline:

1. collect residual trials
2. filter execution lapses
3. cluster systematic residuals
4. generate competing trajectories per cluster
5. classify novelty / duplicate / conflict / scope
6. attach verifier contracts and reversible overlay ops
7. run world-model screen
8. run heldout and negative-control fresh validation
9. retain only validated candidates for recursive execution

## Artifact

- `phase four/assumption_graph/paper_readiness_20260604/full_v2_phase4_hypothesis_generator_bypass_20260611.json`

## Performance Validation

Result: pass.

Metrics:

- trial_count: 14
- eligible_residual_count: 12
- execution_lapse_count: 2
- execution_lapse_filtered_rate: 1.0
- cluster_count: 6
- candidate_count: 12
- accepted_candidate_count: 6
- screened_candidate_count: 6
- min_candidates_per_cluster: 2
- candidate_layer_coverage: 6
- novel_family_rate: 1.0
- duplicate_rate: 0.0833
- conflict_rate: 0.0833
- fresh_validation_success_rate: 1.0
- cross_domain_transfer_rate: 1.0
- descendant_productivity: 0.7
- false_discovery_rate: 0.0
- residual_explained_fraction: 1.0
- manifest_validation_issue_count: 0
- world_model_screen_precision: 1.0
- avg_candidate_generation_ms: 0.5129

Layer coverage:

- object
- method
- evaluator
- memory
- world_model
- meta_evolution

## Interpretation

Phase 4 now validates a stronger generator behavior: it can produce multiple candidate trajectories per systematic residual cluster, avoid execution-lapse hallucinations, filter duplicates/conflicts, and retain only candidates that survive verifier, world-model, and heldout checks.

This remains a bounded shadow bypass.  It is a functional validation of the generator contract, not yet a large live benchmark or an external QA score claim.

