# Hegel Machine Current-State Reconciliation v3

## 1. Status and precedence

This document is the repository-level current-state narration after the first
full-node6 outer campaign reached a terminal transport failure.  It supersedes
v2 only as the current dashboard.  The v2 reconciliation remains the immutable
historical snapshot after Q0.5b attempt 2, and every track not changed below
retains its exact v2 state and claim boundary.

This document is not a scientific result, execution authorization, retry
authorization, gate receipt, adoption, promotion, current-pointer write or
`ACTIVE` transition.  Bound artifacts, ledgers and trajectories remain
authoritative for their narrow claims; if they disagree with this narration,
they win and a new reconciliation version is required.

## 2. Reconciled delta

| Track | Current state | Exact claim boundary |
|---|---|---|
| Q0.5b actual attempt 2 | **TERMINAL SUCCESS; 20/20; Q1 not run** | Unchanged from v2: `ATTEMPT2_TARGET_D_V2_QUALIFIED_Q1_NOT_RUN`, qualification mask `0x0fffff`, no retry/resume. |
| Full-node6 transport v1 | **CONSUMED; TERMINAL FAILURE; no endpoint measurement start** | Capsule `sha256:85fc53352b957995162254d45bf32c75cf1f1444a01f2676cce1e773ea357c90` invoked the outer supervisor once and stopped at `SOURCE_SNAPSHOTS`.  It cannot retry or resume. |
| Full-node6 successor v2 | **IMPLEMENTED; not executed; exact capsule not yet authorized** | Distinct outer contract and authorization identity, fresh v2 state roots, frozen technical action v1.  Its only success claim remains `PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`. |
| Formal Q1 | **`NOT_RUN`; 0/20** | `q1_gate_count = 0`, `q1_gate_mask = 0`, all eight formal Q1 output roots null; target/split/role inputs remain unopened by this diagnostic path. |

The current ordering is:

```text
Q0.5b attempt 2 = terminal success / 20/20 / Q1 NOT_RUN
  -> full-node6 transport v1 = consumed / SOURCE_SNAPSHOTS failure
       / Python measurement starts 0 / Rust measurement starts 0
  -> full-node6 successor v2 = implemented / not executed / not authorized
  -> saturated diagnostic, if obtained = formal-resource amendment boundary
  -> Q1 = 0/20 / NOT_RUN
```

## 3. Superseded transport evidence

The failed transport is bound by:

- source commit `473b39da5b95670a74f347b4e6933e507c4e7c57`;
- consumed record
  `sha256:f0d178038ca3ebe6fcdcad1647dd530ebb2f38e18703915fff37397d6c919a0b`;
- terminal record
  `sha256:9ca2f5eafae098992414973a7bf80a79ab2a2aaebb63f5d69b4b32c49c4e53a5`;
- outcome
  `sha256:6de54c48cfc36f5ee1e51a75092653cb0a99c2c1fbbf05969e63befcb74e98bf`;
- terminal status `FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME`;
- error code `FAIL_FULL_NODE6_SUPERVISOR_SOURCE`.

The exact eight-event trajectory is limited to supervisor start, source replay,
local-image admission, and the failed source-snapshot phase.  It contains no
`RUST_OFFLINE_BUILD`, `MEASUREMENT_AND_REPLAY`, `PYTHON_ENDPOINT` or
`RUST_ENDPOINT` event.  The run root is empty and the artifact is null.  Earlier
pre-consumption Python recipe and Cargo qualification containers are engineering
checks, not quotient measurements.

The reachable defect was a repository-subdirectory Git object path:
`COMMIT:tools/...` was interpreted from the repository tree root, while the
object is `COMMIT:Hegel Machine/tools/...`.

## 4. Successor-v2 boundary

The successor retains the scientific/technical full-node6 action v1 because no
actual endpoint measurement began.  Its outer contract and authorization are
distinct successor-v2 identities.  It uses only fresh
`q1-full-node6-capacity-v2-*` lock, ledger, evidence, trajectory, run and probe
roots, plus `artifacts/phase3_q1_full_node6_capacity_preflight_v2.json`; the v1
roots are read-only predecessor evidence.

Before authorization consumption or supervisor Popen, preflight must replay the
Q0.5b predecessor, the superseded consumed/terminal/trajectory evidence, the
frozen technical contract, and the production supervisor's exact prefix-aware
commit-blob materializer over 34 Python, 21 Rust release-build and one host
replay source row.  Repairable path, permission, dependency, image/daemon,
CPU-affinity, memory or storage failures remain pre-consumption `PENDING` with
zero authorization consumption and zero supervisor Popen.

The successor-v2 preflight does not authorize measurement.  A clean committed
source must first derive one exact new capsule, and that capsule requires a new
explicit owner authorization.  Formal Q1 start remains a later, separate
`phase3-q1-start` action after a saturated diagnostic and its mechanical
formal-resource amendment.
