# V2 Phase 0 Schema Freeze

Date: 2026-06-10

Branch: `reconstruction-v2`

## Goal

Start the v2 reconstruction without replacing the existing recursive runner.
The first step is to freeze the lifecycle representation of a hypothesis:

```text
AssumptionManifestV2
  + graph overlay projection
  + optional ProcessModel / AlignmentHypothesis payload
  + VerifierContract
  + WorldModelTrial over graph actions
```

This directly addresses the confusion in `reconstruction_v2.md`: a hypothesis
is not only a graph edge, not only a category-theory morphism, and not only a
natural-language claim.  It is a lifecycle object that can be projected into the
Assumption Graph.

## Implementation

Added:

- `assumption_os/hypothesis_lifecycle_v2.py`
- schema enum support for:
  - `AssumptionType.PROCESS`
  - `HypothesisKind.PROCESS_MODEL`
  - `HypothesisKind.ALIGNMENT_HYPOTHESIS`
  - `HypothesisKind.WORLD_MODEL_TRIAL`
  - `EdgeType.PARTICIPATES_IN`
  - `EdgeType.HAS_PROCESS_MODEL`
  - `EdgeType.HAS_ALIGNMENT`

The v2 fixture represents Le Chatelier's principle and Lenz's law as process
models, then represents their relation as a separate alignment-hypothesis node.

High-level display may still look like:

```text
Le Chatelier -- analogous_negative_feedback_schema --> Lenz
```

but the graph projection is:

```text
Process_LeChatelier
  -> AlignmentHypothesis_LeChatelier_Lenz
  -> Process_Lenz
```

The relation node owns:

- typed variable/role mapping
- preserved structure
- broken structure
- verifier tests
- risk predictions
- counterfactual mask actions
- world-model prediction slot

## Validation Artifact

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/hypothesis_lifecycle_v2_schema_20260610.json`

Key metrics:

- process model count: 2
- alignment relation node count: 1
- bare alignment edge count: 0
- participates-in edge count: 2
- counterfactual mask count: 3
- validation issue count: 0
- mapping score: 0.7969

Passed gates:

- manifest contract complete
- process models are typed
- alignment is relation node, not bare edge
- perturbation / response / opposition roles are preserved
- broken-structure boundaries are recorded
- world-model trial is a graph action
- counterfactual masks are explicit
- main graph is not mutated

## Boundary

This is not yet a full discrete/causal world model.  It is the schema and
projection layer needed before that model can be trained or evaluated.

The next v2 step should be:

```text
systematic residual cluster
  -> generate several candidate lifecycle objects
  -> classify duplicate / specialization / novel / orthogonal
  -> world-model graph-action prediction
  -> selected fresh ablation
  -> accepted/rejected graph overlay merge
```

This keeps v2 incremental: freeze object semantics first, then upgrade generator
and world model.
