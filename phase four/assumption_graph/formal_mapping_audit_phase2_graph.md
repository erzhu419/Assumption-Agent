# Formal Mapping Audit: phase2 graph

Date: 2026-05-31

## Command

```bash
python3 -m assumption_os.formal_mapping \
  --graph-dir "phase four/assumption_graph" \
  --summary-out "phase four/assumption_graph/formal_mapping_audit_phase2_graph.json"
```

## Result

- Formal nodes considered: 45
- Formal mappings: 9
- Status counts: `complete=9`, `partial=0`, `unsafe=0`
- Role counts: `feature=9`, `constraint=9`, `decomposition=9`, `verification=9`, `hp_change=9`

Each mapping preserves the executable invariants required by the current audit:
trigger detector, answer transformation constraint, ordered decomposition,
verifier, and runtime policy.

## Interpretation

This implements the first executable GRAM/category-style layer for the graph.
The audit treats each Exp82 seed bundle as a morphism from a problem signal into
an answer transformation and verification policy. A mapping is considered safe
only when the trigger, transformation, verification, and runtime policy are all
present.

This is a bounded audit, not a full mapping generator. The evolution cycle feeds
these mapping statuses into proposal gating so incomplete or unsafe formal
bundles cannot be promoted without repair. Solver-time retrieval now also uses
complete mappings as searchable reasoning operators when their trigger signals
match the current problem.
