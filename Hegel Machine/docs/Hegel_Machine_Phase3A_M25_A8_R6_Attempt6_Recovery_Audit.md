# Hegel Machine Phase-3A M2.5 — A8 R6 Attempt-6 Recovery Audit

Status: **IMPLEMENTATION QUALIFIED / FORMAL PRE-ATTEMPT AUTHORIZATION NOT MATERIALIZED / ATTEMPT-6 NOT RUN**
Formal child state: **NOT_RUN**
Formal recovery state at qualification: **PENDING / RESERVED**

## Scope

This amendment is the sole-child continuation of terminal R5 commit
`0024f8117f6ad20bd004f1a6024987d923f2b7ad`. It authorizes only recovery
ordinal 6 for the already reserved run
`e4af9f57c38fb298462ec628c4ed8a03` and ledger
`ec849e2f1e2e1163cfc450370b25b484`.

It does not authorize ordinary execution, identity redraw, abort/recreate,
post-stage substitution, M3 start, or any read/hash of the retained raw seed.

## Frozen R5 terminal evidence

The eight canonical R5 records are fixed in order from `preflight.json`
through `failure.json`. Their implemented canonical chain root is:

```text
bcbe5e09f843b71e7448159307a02f698ace61fdccdff80767f3c826b6fb245b
```

R5 canonically retained only the sanitized primary type
`FormalStaticBasisError`, the formal failure-evidence digest, and the outer
close failure. The more specific missing-explicit-Rust-path diagnosis is
therefore recorded only as a non-canonical, read-only forensic inference.

The deterministic reproduction found two independent defects:

1. Recovery rebuilt the static basis with the fixed main Rust binary but did
   not pass that bound path into Rust replay, allowing an unintended worktree
   default to trigger `FAIL_M25_STATIC_RUST_REPLAY_POLICY`.
2. Successful inner actor cleanup was followed by a second non-idempotent
   `close()`, which obscured the primary failure with the exact detail
   `actor identities remain without their bound Docker control plane`.

## R6 engineering changes

R6 makes the following boundaries executable before attempt consumption:

- The formal Rust replay path and digest must equal the fixed main artifact:
  `d38eabce2be158326fe16a7185ffc2c9be1262ce8d5098afed25eff431465093`.
- The existing frozen Docker daemon binding is checked before replay.
- Python and Rust static receipts are fully replayed and passed through the
  same dual-receipt validator, producing the six ordered Gate-19 roots.
- Parent-absence evidence is generated and replayed while no purpose actor is
  running.
- The actual 47-field R6 source admission enters the same executor validator,
  commit-context check, and isolated fixed-A8 capability path used by the
  formal core.
- The static qualification and source-capability qualification are separate
  records, avoiding a hash dependency cycle. Owner authorization binds both.
- The executor carries a private same-process capability sealed to the exact
  recovery and actor objects. It is one-shot and is consumed before any actor
  operation. R6 core reuses it instead of rebuilding basis/static/source
  qualification after `attempt-start.json` becomes visible.
- `FormalStaticBasisError` is translated with its exact code/detail at every
  static boundary and is also recognized by formal failure serialization.
- The R6 actor close latch is set only after inherited close succeeds; a
  failed first close is never suppressed.
- Stored audit authority is exact canonical bytes, not Python object equality,
  preventing `False == 0` / `True == 1` type confusion.

## One-shot and concurrency boundary

The recovery flock remains the only execution lock. Immediately before
attempt consumption, R6 rechecks the external audit directory identity and
descriptor-reads all seven prepared/authorized records. Their canonical
bytes, self hashes, inode identities, modes, owners, and directory namespace
must equal the pre-lock snapshot.

Only then may `attempt-start.json` be atomically installed and fsynced while
the recovery lock remains held. That visible file is the attempt-6 consumption
token. A second process cannot pass the held-lock recheck, and any later call
must reject the visible token even if no terminal record exists yet.

## Implementation qualification evidence before commit

This evidence qualifies the R6 implementation and its seed-free replay path.
It does not constitute formal pre-attempt qualification: the commit-bound
six-record preparation prefix and owner authorization do not yet exist.

The affected R3, R4, R5, R6 recovery tests and the complete formal-container
executor regression passed after the sealed-prefix integration. Four-way
project test sharding is required again after the final manifest is committed.

A real read-only static-prefix replay against the retained transaction passed
without creating an attempt or starting a purpose actor:

```text
dual Gate-19 roots: 6/6
root-row digest: 43859ea51d50afd013d152dc4015eefb38249344394ad8beec4cb45e6920127e
Python full receipt: 593a91fce33ea2dae6f5c1ee55e48e36aee49544edcf1f46c635bc61b2896898
Rust full receipt: c56c015ed0acf0c6bbebd8d9e77703f9d5bfe26315f3e4d46ed8b81a014d99ee
frozen daemon equals static daemon: true
purpose actor start attempted: false
raw seed bytes read by R6 orchestrator: false
M3 start invoked: false
```

The immediate post-replay state remained:

```text
marker=PENDING
journal=RESERVED
run-labelled containers=0
retained fixed role volumes=4
raw seed metadata=regular, mode 0600, size 32
raw seed bytes read=false
raw seed SHA-256 computed=false
```

## Execution gate

Attempt-6 may run exactly once only after all of the following are true:

1. This amendment is one clean committed sole child of terminal R5.
2. The canonical manifest exactly matches changed paths and source hashes.
3. The branch is pushed and the committed source preflight passes.
4. Four-way project regression passes from the committed tree.
5. `prepare-authorization` regenerates both locked qualifications and writes
   the exact six-record preparation prefix.
6. Owner authorization binds that exact prefix.
7. The execution command requalifies the same prefix under the recovery lock.

Passing Attempt-6 may produce **24/24 READY BUT NOT_RUN**. It must not invoke
`phase3-m3-start`; M3 remains a separate explicit action.
