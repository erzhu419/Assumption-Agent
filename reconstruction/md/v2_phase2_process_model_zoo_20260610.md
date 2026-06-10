# V2 Phase 2 Process Model Zoo

Date: 2026-06-10

Branch: `reconstruction-v2`

## Goal

Phase 2 implements the `reconstruction_v2.md` requirement to stop comparing
opaque principle names and instead compare typed process models.

The first process zoo contains 10 processes:

1. Le Chatelier principle
2. Lenz's law
3. thermostat negative feedback
4. predator-prey local stabilization
5. chemical first-order decay
6. radioactive decay
7. RC circuit discharge
8. logistic growth
9. damped oscillator
10. supply-demand equilibrium response

## Implementation

Added:

- `assumption_os/process_model_zoo_v2.py`

Each process has:

- state variables
- interventions
- perturbation
- response
- invariants
- failure cases
- family tags
- role schema

The benchmark tests process-family alignment on gold positive and negative
pairs.  It generates `AlignmentHypothesis` rows only for accepted positive
process-pair alignments.

## Validation Artifact

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/process_model_zoo_v2_20260610.json`

Metrics:

- process count: 10
- family count: 14
- positive pair count: 9
- negative pair count: 7
- alignment hypothesis count: 8
- validation issue count: 0
- true positive: 8
- false negative: 1
- true negative: 7
- false positive: 0
- accuracy: 0.9375
- positive recall: 0.8889
- positive precision: 1.0
- negative rejection rate: 1.0

Passed gates:

- has ten process models
- all process models validate
- has multiple process families
- positive alignment recall is high
- negative-control rejection is high
- overall pair accuracy is high
- alignment nodes are available for positives

## Boundary

This validates process-family representation and deterministic alignment
classification.  It does not yet run downstream QA or train the graph-action
world model.

The next step is Phase 3: train or evaluate a graph-action world model over
states such as process family, candidate overlay type, prior failures, predicted
acceptance, regression risk, value delta, and expected cost.
