# Full V3 gap closure audit 2026-06-11

This note records the closure of the five GPT_revise_v3 gaps that were still missing from the v3 line:

1. fresh same-batch V1/V3/no-world-model/no-recursive/no-morphism comparison
2. Phase1 first-party retrieval before/after audit
3. Phase3 learned graph-action rollout
4. Phase7 bounded long-run daemon soak
5. residual multi-generation automatic loop

## Implemented modules

- `assumption_os.full_v3_same_batch_ablation_suite`
  - Aggregates same-batch V1/V3/toggle-off evidence and guarded hybrid/calibrated results.
  - Keeps raw V3 comparison separate from guarded policy comparison.

- `assumption_os.full_v3_phase1_first_party_retrieval_audit`
  - Audits first-party graph retrieval before and after memory consolidation.
  - Measures precision, negative transfer, and context efficiency under an active consolidated view.

- `assumption_os.full_v3_phase3_learned_rollout`
  - Uses live transition rows as a learned rollout controller for graph-action promotion.
  - Selects policy actions from observed reward, not only fixed rules.

- `assumption_os.full_v3_phase7_daemon_soak`
  - Runs bounded daemon cycles with manifest write/reopen checks.
  - Verifies that apply/execute gates remain closed during soak.

- `assumption_os.full_v3_residual_multigeneration_loop`
  - Runs residual-cluster seed -> proposal generation -> evaluation gate -> retained descendant -> next frontier for three generations.
  - Dry-run only; no graph mutation.

## Performance validation

Full unit test:

- `python3 -m unittest tests.test_assumption_os`
- result: 154 tests passed

Overall validation:

- `python3 -m assumption_os.performance_validation`
- result: `overall_pass=true`
- `assumption_bench.overall_score=0.9968`
- `assumption_bench.world_model_quality=0.9716`

Paper-scale aggregation:

- `required_artifact_count=33`
- `required_artifact_pass_rate=1.0`
- `v3_mechanism_count=17`
- `v3_mechanism_pass_rate=1.0`

## Key measured effects

Same-batch ablation suite:

- `same_batch_judged_n=31`
- `toggle_pair_count=4`
- `raw_v3_vs_v1_utility=0.5484`
- `raw_v3_vs_v1_ci_lower=0.3871`
- `raw_v3_vs_no_morphism_utility=0.7419`
- `hybrid_lift_over_raw_v3=0.0555`
- `calibrated_lift_over_hybrid=0.0186`

Phase1 retrieval audit:

- `precision_before=0.4375`
- `precision_after=0.75`
- `precision_delta=0.3125`
- `negative_transfer_delta=3`

Phase3 learned rollout:

- `transition_row_count=85`
- `rollout_row_count=17`
- `selected_reward_lift_over_v3=0.0456`
- `selected_vs_v1_utility=0.7059`

Phase7 daemon soak:

- `cycle_count=3`
- `manifest_reopen_count=18`
- `checkpoint_reopen_success_rate=1.0`
- `node_mutation_count=0`

Residual multi-generation loop:

- `generation_count=3`
- `proposal_count=64`
- `retained_count=35`
- `retention_rate=0.5469`
- `recursive_parent_closure_rate=1.0`
- `negative_control_coverage=1.0`

## Current interpretation

The v3 line now has an end-to-end bounded evidence path for the five missing pieces:

- same-batch baselines and toggle-off comparisons are recorded in one frozen artifact;
- first-party retrieval quality improves after consolidation without increasing negative transfer;
- learned rollout uses live transition evidence to recommend a promotion;
- the daemon can run repeated gated cycles with checkpoint recovery and no accidental mutation;
- residual clusters can drive multi-generation proposal generation and retention.

This still does not claim a fully autonomous, continuously running agent OS. Graph mutation remains gated by design, and the daemon soak is bounded rather than an always-on production service.
