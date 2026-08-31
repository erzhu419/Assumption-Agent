# Hegel Machine Phase 3A Q1 completion-only successor v8c

## Purpose

The terminal v7r1 campaign completed its Python, Rust, and trusted-host actors,
but the outer completion adapter compared the trusted-host peak memory against
the immutable predecessor's 64 MiB Guard-11 value instead of the v7r1
successor's authorized 2 GiB value.  The measured peak was 111,919,104 bytes,
all memory-event OOM counters were zero, and every other resource and semantic
check passed.

V8c is a distinct, owner-authorizable completion-only action.  It does not
resume v7r1 and does not repeat any measurement.  It replays the exact sealed
v7r1 run specification, three clean checkpoints, supervisor outcome, actor
outputs, source manifests, runtime evidence, and terminal lineage as read-only
inputs.  It then constructs the completion receipt with the corrected
successor Guard-11 mapping.

## Authority boundary

The v7r1 terminal remains immutable and terminal-no-resume.  V8c has:

- zero internal `Popen` calls;
- zero supervisor, actor, Docker, startup-smoke, build, and measurement calls;
- no new actor intent or checkpoint;
- no mutation of any v7r1 run, ledger, output, source, or scratch path;
- no network, target-truth, split, or role access;
- no adoption, promotion, `ACTIVE`, production, certificate, or current-pointer
  authority.

The previous actor counts remain historical facts (`python=1`, `rust=1`,
`host=1`).  The v8c actor counts are separately and always zero.

Before authorization consumption, missing or changed source/evidence bytes, an
unavailable local path, or an inability to prove the fresh output paths
produces same-capsule `PENDING_NOT_CONSUMED`.  It does not consume authorization
or execute the completion action.  After consumption, only deterministic
partial-finalization is permitted: the controller may replay the capsule-bound
completion bytes and finish publishing the v8c artifact, outcome, and terminal.
It may never invoke a supervisor or actor.  A post-consumption mismatch or
unknown outcome is terminal-no-resume.

An interrupted durable publication is reported only as
`Q1_COMPLETION_REPLAY_V8C_PARTIAL_FINALIZATION_RECOVERY_REQUIRED` until its
terminal can be read back from disk.  If a consumption record exists but its
capsule or authorization can no longer be replayed, the controller reports
`Q1_COMPLETION_REPLAY_V8C_UNSEALED_TERMINAL_NO_RESUME`; it never reports that
state as pre-consumption `PENDING` or as a sealed terminal.

## Success projection

Only an exact completion receipt may publish `Q1_COMPLETE_20_OF_20`:

- Q1 `COMPLETE`, 20/20, mask `1048575`;
- eight non-null Q1 output roots;
- Q2 `NOT_RUN`;
- null/false M3, role, certificate, adoption, promotion, and current-pointer
  state.

The receipt continues to identify the v7r1 run genesis, prestart receipt,
actor implementations, and measurements.  The outer v8c capsule supplies the
new authority for the cross-terminal read-only replay; it does not rewrite the
measurements as v8c actor executions.

## State-machine entries

The local controller exposes three entries:

```text
--preflight --run-id HEX32 --created-at-unix-seconds INTEGER
--record-authorization --capsule-digest sha256:...
--run --capsule-digest sha256:...
```

Preflight builds the expected completion receipt entirely in memory, binds its
raw digest, receipt root, and eight projected slots into the capsule, and
publishes only the fresh capsule.  Recording authorization and running both
replay the current source closure and all v7r1 evidence before any new ledger
publication.  The run publishes exactly one consumption record and one
logical completion-replay outcome.

## Files

- Contract: `config/phase3_q1_completion_successor_v8c.json`
- Pure state wire:
  `src/hegel_machine/phase3_q1_completion_replay_successor_v1.py`
- Read-only receipt adapter:
  `src/hegel_machine/phase3_q1_start_completion_adapter_v1.py`
- Controller: `tools/phase3_q1_completion_successor_v8c.py`

The implementation alone is `IMPLEMENTED_NOT_EXECUTED`.  It does not authorize
or perform the v8c completion replay.
