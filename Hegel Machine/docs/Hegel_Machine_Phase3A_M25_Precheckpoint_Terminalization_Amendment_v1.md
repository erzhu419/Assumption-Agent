# Hegel Machine Phase-3A M2.5 Precheckpoint Terminalization Amendment v1

Status: implementation amendment for a fail-closed, pre-seed transaction. This
document does not freeze a new formal wire object and does not promote any M3
gate.

## 1. Incident class

The affected transaction reached durable `RESERVED` after its run, ledger and
three public-output paths had been reserved. It then failed before the formal
actor set started and before either `actor-trust-checkpoint.json` or
`actor-trust-checkpoint.json.next` was created.

The concrete A5 basis is:

`de6402172fe589efb2c1c3707b034be32b30e468`

The observed transaction identifiers are diagnostic, not reusable authority:

- run ID: `49529f8b5edade3c0dbf571df2051a1d`
- ledger ID: `21cf82bb31061a224dbab271e7a2161c`

The pre-terminalization audit established all of the following:

- transaction journal state was exactly `RESERVED`;
- the stage contained exactly the five reservation-bootstrap files;
- no actor-trust checkpoint or checkpoint-next inode existed;
- no PENDING or COMPLETE marker existed;
- no seed intent, seed completion receipt or raw seed existed;
- no final formal output existed;
- no run-labelled actor container, key volume, mount or process remained.

No raw seed or private key was opened or hashed during that audit.

## 2. Root cause

`static_replay_control_plane_v1()` correctly prepared the local Docker control
plane before formal actor launch. The old `start()` guard incorrectly treated
the resulting local temporary runtime as proof that the four role actors had
already started. The first formal `start()` therefore failed with
`Docker ceremony actors may be started only once`.

Runtime preparation and actor start are distinct lifecycle states.

## 3. Frozen implementation correction

`DockerCeremonyActorsV1` now has an irreversible
`_actor_start_attempted` boundary.

- Preparing or reclaiming the local runtime does not consume the boundary.
- The first call to `start()` consumes it before any start-side effect.
- A successful start, a failed start and a later cleanup all leave it consumed.
- A second call on the same backend is rejected before cleanup logic, so it
  cannot accidentally remove a live first actor set.
- Qualification actors and formal actors remain distinct backend instances;
  no qualification key, volume or container is reused.

## 4. Exact early-precheckpoint abort

The terminal abort accepts exactly three checkpoint shapes while the journal
is `RESERVED`:

1. checkpoint absent;
2. only the durable checkpoint file present;
3. only the checkpoint-next file present.

It rejects both checkpoint variants together and rejects every unknown stage
entry. In every accepted shape it still requires:

- the exact five required bootstrap files;
- no marker or seed-continuity state;
- no final public output;
- exact run-labelled Docker actor and key-volume absence;
- an immutable deletion plan binding every actual inode and payload digest;
- exact absent-prefix crash recovery;
- a durable terminal tombstone and one retirement marker for each of the three
  physical output paths before deletion starts.

Checkpoint absence does not prove that key generation was never attempted. It
proves only that no durable trust checkpoint exists. The authoritative cleanup
claim is narrower: after the exact run-label audit, actor containers and key
volumes are absent, seed continuity is absent, and the transaction is terminal.

## 5. Verification matrix

The checkpoint-present plan has 15 deletion rows and 76 unique injected crash
points. The checkpoint-absent plan has 14 deletion rows and 73 unique injected
crash points. Both matrices cover plan durability, every unlink boundary,
parent-directory fsync and terminal-lock recovery. Separate tests cover
checkpoint-final, checkpoint-next, both-variant rejection, unknown-entry
rejection, runtime-prepared first start, failed-start lockout and live-actor
second-start non-cleanup.

## 6. Authority and continuation

This amendment authorizes only terminalization of the stranded A5 reservation.
It does not resume A5, create a seed or key, generate a formal root, satisfy a
gate, sign a certificate or start M3.

After exact abort:

- the A5 tombstone and three retirement markers remain permanently in their
  original public parent;
- the old physical output paths are never reused;
- the next ceremony uses a fresh custody root, a fresh qualification-custody
  root and the frozen fresh public parent
  `Hegel Machine/artifacts/phase3_m25_external/formal_genesis_v2/`;
- inside that parent, the two fixed allowlisted basenames remain
  `phase3_m25_formal_gate_evidence_v1.json` and
  `phase3_m25_gate_promotion_v1.json`;
- the next valid endpoint remains exactly `24/24 + NOT_RUN`;
- `phase3-m3-start` remains a separate, uninvoked action.
