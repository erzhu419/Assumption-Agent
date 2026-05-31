# Formal Mapping Search Eval: phase2 graph

Date: 2026-05-31

## Setup

- Graph: `phase four/assumption_graph`
- Complete formal mappings: 9
- Query count: 5
- Search layer: `assumption_os.formal_mapping.search_formal_mappings`

The search layer scores complete mappings by trigger keywords and regex hits,
then returns executable mapping applications: constraints, ordered
decomposition steps, verifier instruction, and runtime policy.

## Result

- Passed: 5 / 5
- `fragile_prod` -> `WCAND01`
- `migration_risk` -> `WCAND02`
- `irreversible_choice` -> `WCAND03`
- `evidence_first` -> `WCAND10`
- `benchmark_compare` -> `WCROSSL01`

## Integration

`assumption_os.retrieval_policy.retrieve_phase2_assumptions` now runs this
formal search alongside normal graph retrieval. `format_policy_context` injects
a `Formal Mapping Reasoning` section when a mapping trigger matches, so formal
bundles are no longer only audited after the fact; they can guide solver-time
reasoning with explicit constraints, steps, verification, and runtime hints.
