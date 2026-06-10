# V2 Phase 1 Overlay Diff and Rollback

Date: 2026-06-10

Branch: `reconstruction-v2`

## Goal

Phase 1 implements the second step in `reconstruction_v2.md`: a new hypothesis
must enter as a candidate overlay diff with explicit rollback references, not as
an immediate committed graph mutation.

## Implementation

Added:

- `assumption_os/hypothesis_overlay_v2.py`

The module consumes the v2 lifecycle fixture and validates:

- dry apply of candidate graph ops
- explicit rollback refs for every op
- exact rollback to the previous graph signature
- idempotent re-apply without duplicate nodes/edges
- a small apply/rollback performance loop

## Validation Artifact

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/hypothesis_overlay_v2_20260610.json`

Key result:

- nodes added by overlay: 3
- edges added by overlay: 2
- rollback failure count: 0
- performance loop iterations: 200
- average apply/rollback time: about 0.016 ms
- main graph mutation: false

Passed gates:

- overlay has explicit rollback refs
- overlay adds expected relation subgraph
- rollback restores exact signature
- idempotent reapply does not duplicate edges
- idempotent reapply does not duplicate nodes
- performance loop has no rollback failures
- performance loop completes requested iterations
- average apply/rollback stays under budget

## Boundary

This phase validates the graph mutation substrate.  It does not yet train a
graph-action world model and does not claim downstream task improvement.

Next step is Phase 2: build a 10-process model zoo and process-family alignment
benchmark so future generator/world-model work has structured states to act on.
