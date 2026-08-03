# Phase-3A M2.5 Ceremony Transaction Hardening v1

**status:** deterministic implementation hardening; no external genesis was run
**child state:** `NOT_RUN`
**formal gate count:** unchanged at `14/24`
**authority disclosure:** `same_admin_controller=true`,
`organizational_independence=false`, `independent_human_actors=false`,
`technical_role_independence=true`, `owner_accepted_threat_model=true`,
`remote_attestation=false`, `hardware_key_nonexportability=false`

This amendment records the engineering controls added after the first
adversarial review of the container ceremony.  It changes no formal CBOR tag,
hash domain, DSL, target, split quota, or M3 state.  Docker remains strictly
offline (`--pull=never`, `--network=none`) and uses only locally present,
digest-pinned images.

## 1. Transaction order

The accepted order is now:

1. complete read-only Commit-A admission;
2. validate the bound Rust replay executable by both resolved path and SHA-256;
3. create a persistent `O_EXCL` ceremony lock;
4. create `O_EXCL` run-ID, ledger-ID, and hidden output reservations;
5. start the four purpose containers;
6. construct a **prospective** `COMPLETE` marker snapshot without mutating the
   real marker;
7. construct all evidence, serialize it, write it to a public-only staging
   directory, fsync it, reload it, and replay it to the same prospective
   promotion bytes;
8. only then atomically transition the real marker from `PENDING` to
   `COMPLETE` and require equality with the staged snapshot;
9. stop all actor containers, delete the four actor-key volumes, and prove
   that neither a labelled actor container nor a labelled private volume
   remains; retain the raw seed/intent/completion/marker in durable custody
   under the separate continuity policy;
10. reload and replay the durable stage once more;
11. create each final public output with `O_EXCL`, publish the evidence,
    promotion and publication receipt, then fsync the directory;
12. journal `PUBLISHED`.

`COMPLETE` is therefore unreachable before a durable, reloaded, prospectively
replayed evidence set exists. A crash between individual final-file writes is
fail closed and leaves the unchanged public stage and journal. Fresh-process
recovery is now implemented by
`FormalCeremonyTransactionV1.rehydrate_post_stage_v1` followed by
`continue_post_stage_transaction_recovery_v1`. The first operation acquires the
same persistent lock and derives one phase from exact durable artifacts; the
second can only continue that phase. Neither operation allocates an opaque ID,
calls a seed/key CSPRNG, signs an object, or reconstructs a replacement evidence
bundle.

The persistent lock schema is now
`hegel-phase3-m25-persistent-ceremony-lock/2`. It binds the absolute evidence,
promotion and receipt paths plus the exact stage-directory name. Each output
reservation uses `hegel-phase3-m25-public-output-reservation/2` and separately
binds its output kind and exact path. This makes a caller-supplied alternate
output tree or a second pair of run/ledger IDs a hard recovery failure rather
than another place to publish the same-looking bytes.

The journal states are:

```text
NEW
  -> RESERVED
  -> STAGED_PROSPECTIVE_REPLAY_PASSED
  -> MARKER_COMPLETE
  -> ACTORS_ABSENT
  -> PUBLISHED
```

All lock, reservation, stage and journal artifacts are excluded from Commit B.
Only the exact final evidence, promotion, publication receipt, actor
qualification/errata reports, and purpose-specific public receipts are
allowlisted by `.gitignore`.

## 2. PENDING and no-redraw rule

Purpose-private key state now uses four distinct local Docker volumes instead
of tmpfs.  Each volume name and Docker labels bind the run ID, purpose ID,
Commit-A SHA-1, actor-profile SHA-256 and pinned image SHA-256.  Exactly one
purpose container mounts each volume; no actor receives the Docker socket.
The common administrator remains explicitly outside the technical-role threat
boundary.

The Docker client boundary is exact: `/usr/bin/docker` always receives the
explicit local Unix-socket host argument, an empty private client config and a
sanitized environment with no ambient Docker/proxy variables. A live daemon
identity receipt is bound into container and volume labels. Every volume must
inspect as daemon-managed `Driver=local`, `Scope=local`, empty options and the
expected local mountpoint; bind/NFS/plugin-backed substitutes are rejected.

Before the sole permitted split-seed CSPRNG call, purpose 1 durably writes:

```text
split_seed_generation.intent
state = CSPRNG_CALL_COMMITTED_NO_REDRAW
```

It then pre-creates and fsyncs a zero-length `split_master_seed.bin`, calls
`getrandom(32)` once, writes exactly 32 bytes and fsyncs the file and custody
directory.  Only after that durable boundary does it create and fsync
`split_seed_generation.complete`, binding the exact intent digest, attempt 1,
seed length and domain-separated seed commitment. Subsequent behavior is exact:

| PENDING custody state | Action |
|---|---|
| no intent, no seed file, no completion receipt | ordinary first execution, or an **explicit** PENDING recovery after a crash before worker entry, may durably create intent/zero inode and make the first and only CSPRNG call (`REAL_FIRST_GENESIS_AFTER_PENDING_NO_INTENT` for the latter) |
| exact intent; regular mode-0600 32-byte seed; exact mode-0600 completion receipt | explicit recovery resumes both calculators with zero new CSPRNG calls |
| intent present with missing/invalid completion receipt, or absent/zero/partial/oversized/symlink/wrong-mode/wrong-content seed | terminal `FAIL_M25_SPLIT_SEED_UNRECOVERABLE_NO_REDRAW` |
| seed without intent | terminal no-redraw failure |

Before durable stage plus verified `COMPLETE`, exception cleanup stops and
verifies actor-container absence but retains all four private volumes.  Volume
destruction has a separate API and is denied until the executor explicitly
authorizes it after the marker/stage equality check.

The fresh-process recovery boundary is now an explicit API, separate from
ordinary `execute`. It reopens and locks the existing canonical mode-0600
ceremony lock, validates the exact run/ledger and output reservations, the
PENDING marker and transaction journal, reopens only the four exactly labelled
purpose volumes, requires the existing purpose-1 key, and invokes the dedicated
`seed-split-resume` worker operation. Ordinary execution never auto-selects
this path. The retained volumes and seed are recovery material, not permission
to improvise a retry or redraw. The prior
`FAIL_M25_PENDING_CUSTODIAN_RECOVERY_NOT_FROZEN` blocker is therefore resolved
at the no-redraw calculator boundary; this does not turn a PENDING recovery
call into Gate 24 publication or M3 start.

Once a public stage exists, the narrower post-stage PENDING path requires the
intent, exact-32 seed inode and completion receipt all to exist before the
worker is entered. It invokes only `REAL_PENDING_RESUME`; the earlier
`REAL_FIRST_GENESIS_AFTER_PENDING_NO_INTENT` edge is categorically unavailable
after staging. The recovered purpose-1 public-key ID and both recovered FD-5
frames must equal the staged marker and staged Python/Rust frames byte for byte
before `PENDING -> COMPLETE` is permitted.

## 3. Purpose-private snapshots

The conservative host admission check binds every top-level Python package
file because `python -m hegel_machine...` executes the package initializer.
That does **not** authorize broad visibility inside actors.  Actor input
snapshots use explicit minimal allowlists:

- purpose 1: custodian worker, wire/CBOR, and the two split calculators;
- purpose 2: Python bridge-only worker and wire/CBOR;
- purpose 3: Rust bridge-only worker and the exact path-and-digest-bound Rust
  replay binary;
- purpose 4: parent-auditor-only worker, parent-audit module, wire/CBOR.

The shared Python worker was split so purpose 2 cannot see seed/custody
operations and purpose 4 cannot see seed or bridge-sign operations.  The
formal wire registry necessarily names all frozen schemas, but unrelated DSL,
target, secret-audit and split-calculator implementations are not copied.

## 4. Filesystem and cleanup corrections

- Ephemeral container snapshots, build outputs and client config live only
  below a validated Linux-local `/tmp` private parent. The persistent custody
  directory is deliberately outside `/tmp` and the repository, must be
  caller-owned mode `0700` on an allowlisted durable local filesystem, and
  passes file fsync, atomic rename and directory fsync probes.
- Rust build/output directories receive explicit mode `0777` after creation;
  the host umask cannot silently make UID 65534 builds unwritable.
- Each new Docker volume is initialized once by an offline, pinned helper with
  root UID and exactly `CAP_CHOWN`, then exercised by a second all-capabilities-
  zero UID/GID 65534 stat/write/remove probe. The helper receipt is retained as
  public run metadata. Its incorporation into the actor qualification root is
  still covered by the live-probe blocker below.
- Custody ownership transfer uses a short-lived administrative container with
  exactly `CAP_CHOWN`, root UID, no network, read-only root, no-new-privileges,
  the frozen seccomp profile and a digest-pinned image.  It is not a formal
  actor and never receives a key or seed through argv/environment/stdin.
- Actor containers continue with `cap-drop=ALL`, UID/GID `65534:65534`.
- The four long-lived actors run `qualify-only` together before keygen. The
  same worker process (purpose 3: its parent shell) then produces a fresh
  strict receipt before every sensitive operation, binding monotonic sequence,
  nonce, request digest and exact environment; host Docker inspection and
  receipt replay are repeated for each operation.
- A nonzero container removal, surviving labelled container, failed custody
  reclaim, failed volume deletion, or surviving labelled volume is fatal.
- COMPLETE cleanup erases exactly the four actor-key volumes. Raw seed,
  intent, completion and marker remain in custody; no API or receipt may claim
  that all private state was destroyed.

## 5. Deterministic implementation admission versus external execution

The former post-stage recovery, purpose-4 detached Git replay, bridge full-DAG
replay and live-probe implementation blockers are now executable and covered by
their qualification/fault suites. Pre-stage continuation and exact pre-seed
abort additionally use a host recovery anchor, safe UID-65534 Docker reclaim,
an immutable exact-prefix deletion plan and a durable terminal tombstone.

This means the implementation is eligible for a fresh admission replay; it
does **not** mean external genesis has occurred. No real external seed, key,
signature or formal root has been generated by these tests. Until the explicit
external workflow is actually run and its evidence validates, the observed
state remains `14/24 / NOT_RUN`. A future `24/24` still remains `NOT_RUN` until
the separate M3 start action.

## 6. Crash-point matrix

| Last durable point | Marker | Private-volume action | Current restart result |
|---|---|---|---|
| exact fully reserved transaction, before marker/intent/seed | absent | destroy exact run-labelled state and verify absent | explicit pre-seed abort writes immutable plan + absence receipt + terminal tombstone; exact-prefix deletion is resumable and permanently retires the output path |
| PENDING marker, before seed intent | PENDING | retain and verify | explicit recovery may perform `REAL_FIRST_GENESIS_AFTER_PENDING_NO_INTENT` with one CSPRNG call |
| intent or seed inode exists, completion receipt absent/invalid | PENDING | retain and verify | terminal no-redraw failure; no byte repair or replacement is permitted |
| valid intent + exact seed + valid completion receipt, before public stage | PENDING | retain and verify | explicit calculator recovery is available, but there is no public stage to publish yet |
| public stage reload/replay passed, marker still PENDING | PENDING | retain and verify | rehydrate as `STAGED_PENDING`; resume existing key/seed only, compare exact frames, complete marker, destroy/verify, publish exact stage |
| fsynced exact `split_seed_instantiation.marker.complete.tmp`, before atomic replace | PENDING + exact next COMPLETE | retain until all stage/custody checks pass | validate the one exact transition, atomically promote it, then use COMPLETE cleanup recovery |
| marker COMPLETE, journal still STAGED or MARKER_COMPLETE, all/some/no private volumes remain | COMPLETE | validate labels, remove exact surviving actors/volumes, prove none remain | rehydrate as `MARKER_COMPLETE_CLEANUP_STATUS_UNKNOWN`; cleanup is idempotent, then journal `ACTORS_ABSENT` |
| actor/volume cleanup passed, before final publication | COMPLETE | re-verify absence | rehydrate as `ACTORS_ABSENT`, then publish the exact three staged files |
| evidence only or evidence+promotion published | COMPLETE | re-verify absence | rehydrate as `PARTIAL_PUBLICATION`; require the write-order prefix and exact bytes, create only missing files with `O_EXCL` |
| all three files written, reservations and/or PUBLISHED journal incomplete | COMPLETE | re-verify absence | rehydrate as `ALL_PUBLIC_OUTPUTS_UNJOURNALED`; verify all exact bytes, remove any exact remaining reservations, journal `PUBLISHED` |
| publication complete | COMPLETE | re-verify absence | rehydrate as `PUBLISHED`; replay and return the same bytes idempotently; M3 remains `NOT_RUN` |

The unit fault matrix exercises the ordering guard, exact v4 lock/reservations,
post-stage PENDING, COMPLETE before and after private-volume deletion,
`ACTORS_ABSENT`, every prefix of three-file publication, reservation deletion,
already-`PUBLISHED` replay, staged/public byte tampering, marker tampering,
opaque-ID tampering, malformed-marker retention, seed-intent completion receipt,
the no-intent pre-stage recovery edge, pre-journal stage replay, and every
fsynced one-step `transaction-journal.next` transition. Ordinary execute remains
non-recovering. The pre-seed abort matrix additionally covers plan/absence/tombstone
installation, every before-unlink, after-unlink-before-fsync and after-parent-fsync
boundary, plan-gone/lock-present and lock-gone terminal recovery, non-prefix gaps,
live-holder refusal and output-path retirement. The
local Docker volume initializer has an opt-in live test gated by
`HEGEL_RUN_M25_VOLUME_LIVE_TEST=1`; it always uses local images,
`--pull=never` and `--network=none`. The complete four-long-lived-actor,
zero-secret qualification test is separately gated by
`HEGEL_RUN_FORMAL_SIGNER_LIVE_PROBE=1`; it performs `qualify-only` twice for
purpose 1 to prove fresh sequence/nonce/request bindings, then verifies no
container or key volume remains.
