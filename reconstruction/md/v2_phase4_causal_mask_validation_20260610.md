# V2 Phase 4: Counterfactual Causal Mask Validation

Date: 2026-06-10

## Scope

This phase adds `do(mask_assumption h)` style counterfactual validation for graph actions.  The environment can remove a candidate relation node or one of its evidence supports, and the system measures how much the graph-action world-model prediction changes.

This is a pipeline-level contribution audit, not a claim of external-world physical causality.

## Implementation

- Added `assumption_os/causal_mask_v2.py`.
- Consumes Phase 3 graph-action predictions.
- Applies four deterministic counterfactual masks:
  - `do(mask_alignment_relation_node)`;
  - `do(mask_process_family_overlap)`;
  - `do(mask_role_schema_overlap)`;
  - `do(mask_invariant_trace)`.
- Computes masked acceptance probability, regression risk, action utility, and importance ranking.

## Performance Validation

Command:

```bash
python3 -m assumption_os.causal_mask_v2 \
  --root . \
  --eval-id causal_mask_v2_20260610 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/causal_mask_v2_20260610.json'
```

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/causal_mask_v2_20260610.json`

Metrics:

- base graph actions: 16
- counterfactual mask trials: 64
- mask count: 4
- mean positive relation accept drop: 0.7622
- mean negative relation accept drop: 0.0140
- mean positive relation utility drop: 0.8859
- mean negative relation utility drop: -0.0153
- relation-drop AUROC: 1.0000
- negative-control mask false-live count: 0
- positive top relation-mask fraction: 1.0000
- masked positive block count: 9
- average mask evaluation: 0.0301 ms

Importance ranking:

1. `do(mask_alignment_relation_node)`: positive utility drop 0.8859
2. `do(mask_process_family_overlap)`: positive utility drop 0.4143
3. `do(mask_role_schema_overlap)`: positive utility drop 0.2778
4. `do(mask_invariant_trace)`: positive utility drop 0.0678

Unit validation:

```bash
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_causal_mask_v2_identifies_relation_node_contribution
```

Result: `OK`.

## Interpretation

The full relation-node mask behaves like deleting the candidate assumption edge/node.  It strongly degrades accepted actions, barely changes negative controls, and becomes the top-ranked contributor for every positive action.  This gives the v2 system an internal importance signal: if deleting a candidate assumption sharply drops predicted utility while controls remain screened, that assumption deserves live verification budget.

## Boundary

The mask audit is causal only relative to the current graph-action prediction pipeline.  It does not prove that the aligned scientific processes are causally identical, and it does not replace fresh heldout ablation.  Phase 5 should use this signal as one input to a formal alignment layer with typed mappings, preserved invariants, broken structures, and negative controls.
