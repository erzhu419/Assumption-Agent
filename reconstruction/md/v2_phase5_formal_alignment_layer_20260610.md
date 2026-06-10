# V2 Phase 5: Bounded Formal Alignment Layer

Date: 2026-06-10

## Scope

This phase connects the v2 `ProcessModel` and `AlignmentHypothesis` objects to a bounded formal alignment checker.  The checker is category-inspired but deliberately limited: it emits typed mapping certificates, finite diagram checks, invariant-preservation checks, causal-mask support, and negative-control decisions.

It is not a full category-theory theorem prover.

## Implementation

- Added `assumption_os/formal_alignment_v2.py`.
- Consumes Phase 2 process-model pairs and Phase 4 causal-mask signals.
- Compares three baselines:
  - LLM semantic aligner proxy;
  - graph-edit / role-similarity proxy;
  - trajectory distribution / Jensen-Shannon information metric proxy.
- Adds a bounded formal checker:
  - typed variable and role mapping;
  - process-family bridge;
  - invariant preservation;
  - finite perturbation-response-invariant diagram check;
  - negative-control gate;
  - causal-mask relation-drop support.

## Performance Validation

Command:

```bash
python3 -m assumption_os.formal_alignment_v2 \
  --root . \
  --eval-id formal_alignment_v2_20260610 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/formal_alignment_v2_20260610.json'
```

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/formal_alignment_v2_20260610.json`

Metrics:

- certificates: 16
- formal accuracy: 1.0000
- formal positive recall: 1.0000
- formal positive precision: 1.0000
- formal negative rejection rate: 1.0000
- formal false positives: 0
- formal false negatives: 0
- LLM semantic aligner proxy accuracy: 0.5625
- graph-edit role-similarity accuracy: 0.6875
- trajectory JS similarity accuracy: 0.8125
- best baseline accuracy: 0.8125
- formal margin over best baseline: 0.1875
- accepted-positive mean relation drop: 0.7622
- finite diagram pass rate: 0.5625
- negative-control pass rate: 0.5625

Unit validation:

```bash
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_formal_alignment_v2_beats_process_similarity_baselines
```

Result: `OK`.

## Example

`process_damped_oscillator_v1` and `process_predator_prey_local_v1` are accepted even though the semantic and graph-edit baselines are weak:

- semantic proxy: 0.0000
- graph-edit role similarity: 0.2000
- trajectory JS similarity: 0.6042
- formal score: 0.8690
- relation accept drop: 0.4868

The formal checker accepts it because both processes preserve a local-stabilization diagram with a restoring-response role.  By contrast, `process_damped_oscillator_v1` and `process_radioactive_decay_v1` are rejected by the negative-control gate.

## Boundary

This is still a bounded structural checker.  It validates finite process diagrams and typed invariants; it does not prove natural transformations, adjunctions, Markov-category statements, Blackwell ordering, or Fisher-geometry theorems.  Those can remain theoretical extensions unless downstream performance requires them.
