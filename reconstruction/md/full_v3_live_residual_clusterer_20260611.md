# Full V3 Live Residual Clusterer

Date: 2026-06-11

## What changed

Residual clustering is no longer only a v2 formal-alignment fixture.  A new live-derived clusterer now reads committed, redacted artifacts and unifies four residual sources:

- formal alignment residuals from `residual_hypothesis_generator_v2`
- Phase9 same-batch live V1/toggle-off residual proposals
- Phase8 creative residual families
- profile-level rejection/calibration residuals from compact, micro, coverage, and Phase10 world-model candidates

It emits unified clusters plus next-generation proposal seeds without reading raw prompts, raw answers, or raw judge text.

## Performance Validation

Validation passed.

- source artifact count: `7`
- observation count: `43`
- weighted residual count: `160`
- cluster count: `31`
- systematic weighted coverage: `0.925`
- Phase9 live residual observations: `16`
- formal residual observations: `15`
- profile residual observations: `4`
- next-generation proposal seeds: `28`
- largest live cluster: `business / pat_controlled_intervention`
- largest live cluster axis: `same_batch_v1_regression:critical_reframe_gap`
- largest live cluster support: `7`
- largest live cluster status: `resolved_by_phase9_hybrid_guard`
- blocked profile residual count: `2`
- raw prompt/answer usage: `false`

## Interpretation

This closes the residual-clusterer gap at the artifact level.  The system can now distinguish:

- residuals already repaired by a retained profile;
- profile residuals that should stay as negative evidence;
- calibration/coverage residuals that should remain exploration-only;
- unresolved live residual clusters that should seed the next generation.

The key behavior is that the largest observed live failure cluster is not blindly converted into another broad repair.  It is marked as resolved by Phase9 hybrid, while the remaining open clusters become proposal seeds for future recursive validation.

## Updated Evidence

- `full_v3_live_residual_clusterer_20260611.json`: `pass=true`
- `full_v3_phase4_hypothesis_generator_20260611.json`: now includes live residual cluster metrics
- `full_v3_phase11_capability_audit_20260611.json`: `phase4_status=validated_live_residual_clusterer_not_full_generator`
- `full_v3_paper_scale_evidence_20260611.json`: `required_artifact_count=27`, `v3_mechanism_count=11`, `pass=true`

## Boundary

This is still not a fully creative autonomous generator.  It creates structured proposal seeds from observed residuals; future work is to run those seeds through multi-generation fresh-live descendants and retain only those that pass downstream gates.
