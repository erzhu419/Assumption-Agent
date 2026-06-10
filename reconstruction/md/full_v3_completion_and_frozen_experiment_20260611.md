# Full V3 Completion and Frozen V1 Comparison - 2026-06-11

## Scope

This pass closes the naming/contract gap in the V3 implementation:

- Phase 0 is now an explicit `full_v3_phase0_contract_checker` rather than only the v2 contract bypass.
- Phase 4 is now an explicit `full_v3_phase4_hypothesis_generator` with variation, evaluation, and selective retention metrics.
- Phase 6 is now an explicit `full_v3_phase6_formal_transfer_engine` with bounded proof-lite certificates.
- A frozen `full_v3_frozen_v1_comparison` experiment compares full V3 against the v1-style kernel and retrieval baselines.

No new API calls were made. These are cached/frozen performance validations over existing artifacts and fixtures.

## New Artifacts

- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase0_contract_checker_20260611.json`
- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase4_hypothesis_generator_20260611.json`
- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase6_formal_transfer_engine_20260611.json`
- `phase four/assumption_graph/paper_readiness_20260604/full_v3_frozen_v1_comparison_20260611.json`
- Updated `phase four/assumption_graph/paper_readiness_20260604/full_v3_paper_scale_evidence_20260611.json`

## Performance Validation

Phase 0 explicit V3 contract checker: pass.

- Contract item coverage: 1.0
- Valid candidate acceptance rate: 1.0
- Invalid draft rejection rate: 1.0
- Contract decision accuracy: 1.0
- Main graph mutation count: 0

Phase 4 explicit V3 hypothesis generator: pass.

- Layer coverage: 1.0 over object/method/evaluator/memory/world-model/meta-evolution
- Min trajectories per cluster: 2
- Novelty/integration accuracy: 1.0
- Selective retention precision: 1.0
- Recursive runner seed rate: 0.5
- False discovery rate: 0.0

Phase 6 explicit V3 formal transfer engine: pass.

- Proof-lite certificate coverage: 1.0
- Typed role mapping coverage: 1.0
- Negative control coverage: 1.0
- Formal score transfer correlation: 0.9986
- Formal margin over best baseline: 0.1875
- Category theorem-prover claim count: 0

Frozen V3 vs V1 comparison: pass.

- Full V3 downstream accuracy: 0.70
- V1 kernel system: `case_reflection_v20`
- V1 kernel accuracy: 0.58
- Full V3 margin vs V1 kernel: 0.12
- Full V3 margin vs HippoRAG-style retrieval: 0.14
- Full V3 margin vs best non-full ablation: 0.09
- Assumption capability improvement: 0.1945

Updated paper-scale evidence aggregation: pass.

- Required artifact count: 18
- Required artifact pass rate: 1.0
- V3 mechanism pass rate: 1.0
- Raw first-party live events: 6403
- Valid judge events: 2818
- Main problem-level n: 100
- Structural vs base p-value: 0.0124006
- Structural vs placebo p-value: 0.00003114

## Remaining Experiment Gap

This closes the explicit V3 mechanism gap and adds a frozen V3-vs-V1 comparison. It still does not replace the need for a fresh live 300/600/full benchmark rerun under the same frozen pipeline. That next experiment should call the current API-backed runner with parallel workers and report problem-level bootstrap confidence intervals.
