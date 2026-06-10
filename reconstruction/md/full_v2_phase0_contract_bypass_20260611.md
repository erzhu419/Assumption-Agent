# Full V2 Phase 0: Shadow Contract Bypass

Date: 2026-06-11

## Scope

This phase starts the full-v2 route without replacing the existing v2 Phase 0-7 kernel.  It adds a shadow contract checker that decides whether a candidate manifest may enter candidate overlay or must stay in `draft_hypothesis_pool`.

The checker does not mutate the committed graph.

## Implementation

- Added `assumption_os/full_v2_phase0_contract_bypass.py`.
- Consumes valid candidate manifests from `residual_hypothesis_generator_v2`.
- Adds known-bad draft fixtures for:
  - duplicate claim;
  - missing scope;
  - missing verifier;
  - missing rollback;
  - governance conflict;
  - missing negative control.
- Checks:
  - scope presence;
  - measurable expected effects;
  - risk predictions;
  - layered verifier contract;
  - rollback refs for every graph op;
  - duplicate claim;
  - governance conflict;
  - negative-control presence.

## Performance Validation

Command:

```bash
python3 -m assumption_os.full_v2_phase0_contract_bypass \
  --root . \
  --eval-id full_v2_phase0_contract_bypass_20260611 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/full_v2_phase0_contract_bypass_20260611.json'
```

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v2_phase0_contract_bypass_20260611.json`

Metrics:

- manifest count: 9
- source valid manifest count: 3
- known-bad manifest count: 6
- candidate overlay count: 3
- draft pool count: 6
- valid candidate acceptance rate: 1.0000
- invalid draft rejection rate: 1.0000
- duplicate detection recall: 1.0000
- conflict detection recall: 1.0000
- valid rollback coverage: 1.0000
- valid verifier presence: 1.0000
- valid negative-control presence: 1.0000
- main graph mutation count: 0
- average contract check: 0.074 ms
- total elapsed: 12.1071 ms

Unit validation:

```bash
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_full_v2_phase0_contract_bypass_routes_invalid_drafts
```

Result: `OK`.

## Interpretation

Full-v2 Phase 0 upgrades the kernel from "schema can store a hypothesis" to "schema can govern candidate admission".  Existing v2 manifests remain unchanged; this bypass adds a stricter gate before overlay admission.

## Boundary

This phase validates contract governance on a controlled fixture.  It does not yet learn schema rules, calibrate scope precision from large traces, or replace the existing v2 schema.
