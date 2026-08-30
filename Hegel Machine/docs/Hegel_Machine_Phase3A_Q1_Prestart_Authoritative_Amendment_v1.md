# Hegel Machine Phase 3A Q1 prestart authoritative amendment v1

## Result boundary

This amendment is the local owner-authorizable transition from the pure Q1
prestart candidate to an authoritative `NOT_RUN` artifact with Gates 1--14
admitted.  It does not start Q1, run either formal materializer, invoke the
trusted host replay, access target/split/role data, use the network, adopt or
promote a result, update a current pointer, or enter `ACTIVE`.

The resulting state is exactly:

- `q1_state = NOT_RUN`;
- `q1_gate_count = 14`, `q1_gate_total = 20`, and
  `q1_gate_mask = 0x003fff`;
- eight null Q1 output slots;
- `future_q1_start_requires_distinct_authorization = true`.

## Post-clean-commit capsule

The committed contract intentionally contains no source commit, source tree,
source-closure digest, actor implementation roots, run ID, or timestamp.  Those
values are supplied only after the implementation is in a clean commit through
one canonical `--capsule-inputs` JSON object.  That object binds:

- the exact clean commit, tree, and replayed source-closure rows;
- the Python, Rust, and host formal actor source-manifest rows and entrypoint
  paths as exact ordered registries within the contract source closure;
- closed dependency-identity, runtime-identity, and offline-isolation objects;
- closed mount-manifest and formal-input-key-absence objects;
- a fresh 16-byte run ID and integer creation timestamp;
- the seven small predecessor evidence files used by the pure candidate.

The amendment recomputes every actor source-manifest, dependency, runtime,
offline-isolation, entrypoint, mount, and input-absence root from those
objects.  It also re-reads the exact source bytes before admission.  A root string that merely
has the right syntax is therefore insufficient.  The capsule digest is
computed after all inputs are present.  Preflight reads and replays the small
inputs, performs only local free-space and reversible write probes, and
publishes the capsule by a fsynced atomic no-replace link.

## State machine

The three mutually exclusive CLI entries are:

```text
--preflight --capsule-inputs PATH
--record-authorization --capsule-digest sha256:...
--run --capsule-digest sha256:...
```

`--record-authorization` replays the published capsule and all pre-consumption
gates before publishing one exact owner-authorization record.  It does not
consume that authorization.  `--run` requires the exact authorization, replays
the gates, publishes exactly one consumption record, and then publishes the
single final amendment artifact by atomic no-replace publication.

Missing or changed small files, unavailable publication paths, insufficient
space, or a failed reversible probe return
`Q1_PRESTART_AMENDMENT_PENDING_NOT_CONSUMED`.  Re-entry uses the same capsule
digest, has zero campaign executions, zero authorization consumptions, zero
Popen calls, and zero network access.

A missing or malformed contract also returns a structured pre-consumption
PENDING result with a null contract digest instead of a traceback. Contract
booleans use exact JSON boolean types; integer aliases are rejected.

The committed Q0 input bound is 2,097,152 bytes; it covers the observed
1,443,257-byte Q0 artifact.  Each of the other six predecessor files remains
bounded at 1 MiB.

Once the consumption record exists, the state cannot return to PENDING.  An
exact consumption record plus its exact hash-bound final artifact is a
read-only success replay.  A missing or different artifact, or a different
consumption record, is `Q1_PRESTART_AMENDMENT_TERMINAL_NO_RESUME`.  The artifact
hash in the consumption record exists specifically so this replay does not
need to re-read all predecessor evidence.

This consumed-state rule applies to all three entries, including preflight. If
artifact publication reports an error after the exact target already exists,
the state is classified immediately by consumed read-only replay: exact bytes
are success, while absence or mismatch is terminal. It never reports terminal
and later reclassifies the same exact bytes as success.

## Files

- Contract: `config/phase3_q1_prestart_authoritative_amendment_v1.json`
- State core:
  `src/hegel_machine/phase3_q1_prestart_authoritative_amendment_v1.py`
- CLI: `tools/phase3_q1_prestart_authoritative_amendment_v1.py`
- Synthetic state-machine tests:
  `tests/test_phase3_q1_prestart_authoritative_amendment_v1.py`

The implementation contract is `IMPLEMENTED_NOT_EXECUTED`.  Adding these files
does not create a real capsule, record an authorization, consume an
authorization, or publish a Q1 artifact.
