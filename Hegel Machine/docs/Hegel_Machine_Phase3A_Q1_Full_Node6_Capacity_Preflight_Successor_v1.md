# Hegel Machine Phase-3A-Q1 Full Node-6 Capacity-Preflight Successor v2

## 1. Position and claim

This is the additive execution successor after the Q0.5b Target-D campaign
qualified its bounded projection, isolation and replay mechanics at `20/20`.
It does not amend the historical preregistration or the bounded Q0.5a/Q0.5b
artifacts.

The successor runs the target-blind full depth-3/node-6 quotient capacity
diagnostic in independent Python and Rust endpoints and then performs trusted
host replay.  Its only successful claim is:

```text
PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY
```

Q1 remains `NOT_RUN`, with `0/20`, mask zero, no Q1 receipt, and eight null
output slots.  A diagnostic stream commitment, fixed-point preimage or
projection result is not a formal Q1 root.

## 2. Bound predecessor

The predecessor is the terminal Q0.5b attempt-2 result preserved by
`Hegel_Machine_Current_State_Reconciliation_v2.md`; the current transport state
is reconciled by `Hegel_Machine_Current_State_Reconciliation_v3.md`:

- source commit `33dc9cbb7d753f83650169d6d6b9caa508927764`;
- capsule `sha256:dd35f2e135452b67d0472efc720bee1454d8ebb03bed73743434135d9ecf3256`;
- terminal ledger record
  `sha256:c5494513991a1c94e4d43116f62d4b829f0deeb643f992d6daf49168271166b1`;
- artifact SHA-256
  `7bbbe13efbc43a128c7310fa3a3a32c0545fe0d55acf03d6212742ebdef3c700`;
- terminal status `ATTEMPT2_TARGET_D_V2_QUALIFIED_Q1_NOT_RUN`.

The predecessor authorization is consumed and cannot authorize this run.

### 2.1 Superseded transport campaign

The first full-node6 outer campaign reached a terminal transport failure before
either measurement endpoint started:

- source commit `473b39da5b95670a74f347b4e6933e507c4e7c57`;
- capsule
  `sha256:85fc53352b957995162254d45bf32c75cf1f1444a01f2676cce1e773ea357c90`;
- consumed record
  `sha256:f0d178038ca3ebe6fcdcad1647dd530ebb2f38e18703915fff37397d6c919a0b`;
- terminal record
  `sha256:9ca2f5eafae098992414973a7bf80a79ab2a2aaebb63f5d69b4b32c49c4e53a5`;
- outcome
  `sha256:6de54c48cfc36f5ee1e51a75092653cb0a99c2c1fbbf05969e63befcb74e98bf`;
- terminal status `FULL_NODE6_DIAGNOSTIC_FAILED_TERMINAL_NO_RESUME`.

Its exact eight-event trajectory stops at `SOURCE_SNAPSHOTS`.  It contains no
`RUST_OFFLINE_BUILD`, `MEASUREMENT_AND_REPLAY`, `PYTHON_ENDPOINT` or
`RUST_ENDPOINT` event.  Thus the outer supervisor Popen count is one, while the
actual Python and Rust measurement start counts are both zero.  Q1 remains
`NOT_RUN`, `0/20`, mask zero and eight null output slots.  That capsule is
terminal and is neither retried nor resumed.

The reachable defect was Git commit-path interpretation from a repository
subdirectory: `COMMIT:tools/...` was evaluated from the repository tree root,
where the committed object is `COMMIT:Hegel Machine/tools/...`.  The successor
derives the repository-tree prefix and uses the exact latter object path.

## 3. Same-run measurement and projection

Each endpoint measures each production signature at most once.  Odd runs
before Sink within an endpoint.  Python runs first and its container must exit
before Rust starts; Rust must exit before trusted host replay starts.  The same
quotient-state return supplies final-bank material to the frozen four-stream
program/cohort/class/coverage counting-discard projection.

Projection may replay target-blind application identities and strict AST
canonicalization from the same final bank.  It must not instantiate a second
quotient state, reevaluate a second behavior closure, or call the historical
public snapshot validator whose bounded prototype contract deliberately
performs a fresh engine replay.

The four ordered streams are program, continuation cohort, quotient class and
semantic coverage.  Their counting-discard descriptors, chunk manifests,
external-sort projections and scratch ledgers use the already frozen v1 wire.
For each signature, the diagnostic also strict-encodes the frozen fixed-point
record, signature manifest and partition row.  The aggregate strict assembler
then encodes the closure bundle and archive-projection result.  Python and Rust
independently reproduce the same node-3 wire KAT before the production path is
eligible.  The measured run counts these exact formal frames without assigning
their roots to Q1.  No formal archive is materialized during this diagnostic.

## 4. Execution boundary

The frozen scientific/technical contract remains
`config/phase3_q1_full_node6_capacity_preflight_v1.json`; neither endpoint began
measurement under the superseded transport.  The outer campaign contract is
now `hegel-phase3a-q1-full-node6-campaign-successor-v2`, with a distinct owner
authorization digest and fresh v2 state roots.  Before owner authorization can
be consumed, the campaign must replay:

1. the predecessor terminal ledger and artifact;
2. the superseded transport's consumed/terminal ledgers and exact eight-event
   trajectory, proving zero endpoint measurement starts;
3. a clean committed source and its exact Git object closure;
4. the actual supervisor's prefix-aware commit-blob materializer over all 34
   Python, 21 Rust release-build and one host-replay rows;
5. the Python, Rust and trusted-host source recipes;
6. the locked offline Cargo closure and pinned local images;
7. the frozen semantic/projection roots;
8. effective CPU and exact affinity admission;
9. sufficient memory and storage reservation;
10. absence of fresh v2 permanent, probe, formal-Q1 and pointer roots.

The successor uses `q1-full-node6-capacity-v2-*` lock, ledger, evidence,
trajectory, run and probe roots, plus
`artifacts/phase3_q1_full_node6_capacity_preflight_v2.json`.  The v1 roots are
read-only predecessor evidence and are never an absence target, cleanup target
or resume target.

Memory admission is `16 GiB`: one `14 GiB` active-container ceiling plus a
frozen `2 GiB` host control-plane reserve.  At most one measurement container
is live, so Python, Rust and host replay memory ceilings are not added.  A
failure to observe `MemAvailable >= 16 GiB` remains pre-consumption `PENDING`.

Storage admission retains both endpoint envelopes plus 4 GiB for
the two committed source snapshots, offline Rust build state and final
publication.  Preparation is outside the measurement interval but is not
pretended to be free.

The endpoint execution snapshot is an exact target-blind source closure:
34 Python rows plus 21 Rust release-build rows, or 55 distinct rows in total.
The Rust closure includes the projection oracle manifest and library because
Cargo resolves that local dev-dependency even for the release-binary recipe.
Trusted-host replay receives its one source row separately.  Target, split and
role files are not mounted into any of these actors.

Each endpoint receives a distinct 48 GiB host scratch root at `/scratch` and
a distinct 32 GiB sidecar root at `/output`; projection spill is forbidden
from the 1 GiB container `/tmp`.  The frozen per-signature wall guard remains
172800 seconds.  Two ordered signatures plus one bounded projection interval
give each endpoint a 518400-second deadline.  The two independently enforced
process budgets therefore sum to 1036800 seconds; the outer supervisor adds the
172800-second trusted-host replay ceiling, offline build and publication
allowances for an exact 1212000-second timeout.
The Python endpoint and the later trusted-host replay use the pinned CPython
3.11.15 image; the outer campaign and dual supervisor remain isolated host
Python 3.10 tools.  Both endpoints and the fresh post-endpoint replay container
run as `65534:65534`, with a read-only root/source snapshot, network none, pull
never, private IPC, pids 128, nofile 256, the frozen seccomp profile, all
capabilities dropped, no-new-privileges and only their declared scratch/output
mounts writable.  Endpoint outputs are never mounted into the peer endpoint;
the replay container receives both completed endpoint trees read-only.

Endpoint elapsed time begins immediately before Docker start and ends only at
container exit, including language import and projection encoding.  Peak memory
is the completed container cgroup-v2 `memory.peak`, including descendants.
Actual diagnostic output is the exact raw canonical stdout plus the four
declared sidecars and manifest, including every trailing LF.  Actual diagnostic
scratch is the frozen allocator's logical high-water and must return to zero.
The trusted-host replay records its wall time and cgroup peak separately and
emits the frozen projected replay-output and replay-scratch quantities required
by the later mechanical resource amendment.  Formal output projection includes
the partition payloads, closure-bundle frame and the frozen 1 MiB endpoint
metadata allowance; formal scratch is the maximum partition projection scratch.

Repairable commit-blob path, permission, local dependency, image/daemon, CPU-affinity,
memory/storage-admission and read-only source-check failures are
pre-consumption `PENDING`.  They resume only under the same run root and exact
capsule digest and do not count as a campaign execution, authorization
consumption or supervisor invocation.

An unknown measurement outcome, interruption after consumption, registered
capacity guard, semantic disagreement or isolation failure is terminal.  It
cannot be converted to `PENDING` and cannot resume the measurement.

## 5. Terminal routing

`PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY` authorizes only a mechanically derived,
byte-exact formal-resource amendment.  That amendment must apply the frozen
formulas and can qualify Q1 Gates 1--14 while still leaving Q1 unstarted.

A resource guard hit is inconclusive.  Python/Rust disagreement, host replay
failure or isolation failure is a failed diagnostic.  None of these outcomes
opens target truth, changes the DSL, authorizes Path B by itself, or advances
M3/ACTIVE.

Direct observed endpoint limits route to `PREFLIGHT_CAPACITY_GUARD_HIT` using
the lowest registered guard identifier.  If all measurements and replay are
complete but the mechanically derived formal output, scratch, memory or wall
envelope exceeds its frozen hard ceiling, the distinct terminal status is
`PREFLIGHT_RESOURCE_ENVELOPE_INFEASIBLE`.  That status preserves the complete
diagnostic artifact and replay evidence, but it is not success, is not PENDING,
cannot resume, and does not authorize the formal-resource amendment.

Formal Q1 execution remains a separate later `phase3-q1-start` action after an
amendment has qualified Gates 1--14.  This successor provides no such action.
