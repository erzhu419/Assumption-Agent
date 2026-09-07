# Hegel Machine Phase-2B-C1 actual-unsealed-960 case prediction/scoring replay v2

## Position

This slice freezes the local engineering path for the next Phase-2B C1
construction step.  It binds the already checked-in input/replay contracts,
readiness DAG, R01--R16 authoritative evidence schema catalogue, attempt-chain
and ledger mechanics, external-custodian checkpoint candidate mechanics,
recognizer and scorer execution surfaces, three-baseline profile, and formal
scoring/gate contracts.

The engineering freeze ends at:

```text
PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2_CAPSULE_READY_NOT_AUTHORIZED
```

The stage name is not evidence that an actual cohort exists or has been read,
that a campaign or attempt identity was allocated, that any readiness node was
admitted, or that prediction, reveal, scoring, replay, a formal gate, effect, or
C1 PASS occurred.

## Producer-free preflight

Capsule preparation reads only the twenty public dependency files listed in
the preregistration and the new runtime source closure.  For every dependency
it checks the exact relative path, byte count where recorded, raw SHA-256, and
the public identity fields selected by the preregistration.  Engineering
preflight does not stat or read either future actual input and does not open an
actual cohort, target, split, answer, gate-label, prediction, or custody input.

The future inputs have fixed repository-external paths.  The upstream admission
manifest must use
`hegel-phase2b-c1-actual-unsealed-960-case-prediction-scoring-replay-upstream-admission-manifest/2`
and is bounded to 1,048,576 bytes.  The scorer bundle is bounded to 536,870,912
bytes.  Neither bound authorizes either input to be produced or imported during
this engineering freeze.

Preflight action counts are exactly zero for:

- logical campaign and authorization consumption;
- campaign-ID and attempt-ID allocation;
- R01--R16 admission;
- actual cohort generation or read;
- internal Popen, Docker, live recognizer, baseline, scorer, reveal, and
  measurement invocation;
- Q6 read, replay, signing, or issuance;
- network access.

The existing readiness DAG is replayed only as a frozen specification.  Its
engineering input remains `0/16`, with all nodes unsatisfied.  Likewise, the
historical owner authorization and external-custodian checkpoint objects are
bound inputs, not evidence that authority, custody, time, durability, or a
launch admission currently holds.

The preregistration also records
`MISSING_EXACT_SCALE_BRIDGE_REQUIRES_UPSTREAM_ADMISSION`.  The existing S01/S02
labels are not silently aliased to S0/S1.  A future upstream admission must
explicitly establish the exact scale bridge before authorization consumption.

## Capsule and source closure

The controller publishes the capsule as canonical ASCII JSON with sorted keys,
compact separators, and one trailing LF.  Raw file SHA-256 values cover exact
bytes.  The source-closure root is a domain-separated SHA-256 over the exact
ordered rows, each containing only `relative_path`, `byte_count`, and
`raw_sha256`.  The capsule digest is independently domain-separated and covers
the complete capsule object before its `capsule_digest` field is added.

The source commit is supplied only after all runtime files are committed.  A
prepared capsule must be replayed against that immutable commit and exact
source rows; a source or dependency drift cannot be repaired by silently
updating an existing capsule.

Capsule preparation may create only the bound capsule and its zero-action
trajectory.  It creates no authorization, consumption, checkpoint, ledger
event, actual result artifact, or terminal record.

## Future actual authorization boundary

A later exact authorization may allow one logical campaign and one
authorization consumption.  It must bind the frozen commit, source-closure
root, preregistration and actual-contract raw digests, capsule digest, and the
capsule's dependency receipt.  No authorization is inferred from this document
or from capsule preparation.

Before consumption, a non-core local validation failure may enter
same-capsule `PENDING_NOT_CONSUMED`; that replay consumes no authorization,
counts no campaign, allocates no identity, and invokes no Docker or endpoint.
During an authorized run, the controller reads and validates the bounded
canonical upstream admission manifest, but for the scorer bundle performs only
a regular-file stat/size admission before consumption.  It does not read scorer
bundle bytes or any underlying actual cohort content before consumption.

After consumption, no PENDING transition is allowed.  The controller performs
exactly one scorer-bundle read, one decode, one scoring-mechanics evaluation,
and one producer-free completion replay.  A valid sealed artifact may be used
only for artifact-to-terminal partial finalization; otherwise a
post-consumption failure is terminal-no-resume.

The replay-only import order is:

```text
sealed R01--R16 authoritative bundle admission
-> sealed recognizer result admission
-> three sealed baseline result admissions in frozen baseline order
-> sealed prediction archive admission
-> sealed answer manifest admission (no reveal producer)
-> scorer bundle stat/size admission
-> authorization consumption
-> one scorer bundle read and decode
-> one scoring-mechanics evaluation
-> one producer-free completion replay
-> sealed mechanics-only artifact and terminal
```

This slice's maximum successful disposition is
`PHASE2B_C1_ACTUAL_UNSEALED_960_CASE_PREDICTION_SCORING_REPLAY_V2_MECHANICS_COMPLETE_NOT_FORMAL_GATE_NOT_C1`.
Formal 720+240 audit, independently generated labels, sealed execution, and a
C1 PASS transition remain outside this capsule and require later authority.

## Q6 and downstream lock

Q6 remains in its already issued immutable state.  This slice supplies no Q6
input path and permits zero Q6 read/replay, private-key or custody access,
signature, re-signing, or re-issuance.

Network, external scientific claims, C1 PASS, C2/C3/C4 start, synthesis, MDL,
adoption, promotion, project-wide ACTIVE, production, and current-pointer
mutation remain forbidden by this engineering freeze.
