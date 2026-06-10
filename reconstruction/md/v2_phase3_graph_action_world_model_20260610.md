# V2 Phase 3: Graph-Action World Model

Date: 2026-06-10

## Scope

This phase implements the first v2 world model over discrete assumption-graph actions.  The model predicts whether a candidate `add_alignment_hypothesis` graph action should be sent to live validation or screened out before spending LLM/judge budget.

This is intentionally a search-control world model, not a full simulator of the external task world.

## Implementation

- Added `assumption_os/graph_action_world_model_v2.py`.
- Consumes the Phase 2 process-model zoo pair judgments.
- Converts positive process-family alignments into accepted graph-action labels and negative controls into rejected labels.
- Predicts:
  - acceptance probability;
  - regression probability;
  - failure type;
  - expected value delta;
  - expected cost;
  - recommended action.

## Performance Validation

Command:

```bash
python3 -m assumption_os.graph_action_world_model_v2 \
  --root . \
  --eval-id graph_action_world_model_v2_20260610 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/graph_action_world_model_v2_20260610.json'
```

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/graph_action_world_model_v2_20260610.json`

Metrics:

- labeled graph actions: 16
- positive actions: 9
- negative controls: 7
- accept AUROC: 1.0000
- accept Brier: 0.0247
- base-rate Brier: 0.2461
- live actions: 9
- screened actions: 7
- accepted actions wrongly blocked: 0
- negative actions saved: 7
- screen cost reduction: 0.4375
- mean regression risk, positives: 0.1804
- mean regression risk, negatives: 0.6617
- mean value delta, positives: 0.1470
- mean value delta, negatives: -0.0855

Gates passed:

- source process-model zoo passes;
- labeled action count is sufficient for the fixture;
- accept AUROC is high;
- Brier beats base-rate calibration;
- screening blocks no accepted action;
- screening saves all seven negative controls;
- regression risk is ordered correctly.

Unit validation:

```bash
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_graph_action_world_model_v2_predicts_alignment_action_outcomes
```

Result: `OK`.

## Boundary

This phase does not claim that the world model can replace true ablation or judge feedback.  It provides a cheap pre-live verifier and budget gate for graph actions.  Phase 4 should add explicit counterfactual masks so the system can estimate which graph elements causally support or harm the predicted action outcome.
