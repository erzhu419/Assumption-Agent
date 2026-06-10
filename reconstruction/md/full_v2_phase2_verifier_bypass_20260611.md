# Full V2 Phase 2: Shadow Residual Analyzer + Verifier Stack

Date: 2026-06-11

## Scope

This phase adds a shadow V0-V7 verifier stack without replacing existing verifier modules.  The purpose is diagnostic: failures must be classified before the system generates or promotes any new hypothesis.

## Implementation

- Added `assumption_os/full_v2_phase2_verifier_bypass.py`.
- Uses an 8-case frozen verifier fixture covering:
  - valid candidate / optimization;
  - execution lapse;
  - assumption defect;
  - discovery;
  - evaluator defect;
  - retrieval defect;
  - world-model defect;
  - placebo trap.
- Implements V0-V7 layers:
  - V0 schema/scope/duplicate/conflict;
  - V1 cheap programmatic/self check;
  - V2 world-model value/risk;
  - V3 matched ablation;
  - V4 placebo/length-matched control;
  - V5 cross-judge/cross-solver;
  - V6 fresh heldout;
  - V7 objective/human review.

## Performance Validation

Command:

```bash
python3 -m assumption_os.full_v2_phase2_verifier_bypass \
  --root . \
  --eval-id full_v2_phase2_verifier_bypass_20260611 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/full_v2_phase2_verifier_bypass_20260611.json'
```

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v2_phase2_verifier_bypass_20260611.json`

Metrics:

- case count: 8
- residual classification accuracy: 1.0000
- decision accuracy: 1.0000
- false-positive acceptance rate: 0.0000
- regression detection recall: 1.0000
- placebo sensitivity: 1.0000
- cross-judge stability: 1.0000
- fresh split generalization: 1.0000
- falsification power: 1.0000
- execution-lapse new-hypothesis count: 0

Unit validation:

```bash
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_full_v2_phase2_verifier_bypass_classifies_residual_causes
```

Result: `OK`.

## Interpretation

The shadow verifier prevents a common self-evolution failure: treating every failed answer as evidence that a new assumption is needed.  Execution lapses, retrieval defects, evaluator defects, and world-model defects are routed to repair/calibration paths instead of hypothesis generation.

## Boundary

This phase uses a controlled fixture.  It does not yet synthesize new verifier tests automatically or measure cross-judge stability on live model outputs.
