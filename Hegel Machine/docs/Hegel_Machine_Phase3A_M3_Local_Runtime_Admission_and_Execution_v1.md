# Hegel Machine Phase-3A M3 Local Runtime Admission and Execution v1

Status: engineering implementation contract for the first canonical M3 run

Claim boundary: local owner admission and offline container isolation only

Formal run ID: `e4af9f57c38fb298462ec628c4ed8a03`

## 1. Purpose

This contract closes the host-side gap between the already published Commit-B
M2.5 evidence and the first canonical M3 enumeration.  It qualifies the exact
runtime source, persists the unique explicit start, runs the Python and Rust
enumerators concurrently in network-disabled containers, and publishes one
terminal outcome without replacement.

It does not create a new external-person attestation.  The admission claim is
exactly `LOCAL_TWO_COMMIT_ADMISSION`: the project owner accepts the runtime in
Commit C through the single-purpose artifact added by Commit D.  The receipt
records `external_actor_attestation=false` and an empty external-signature set.

## 2. Frozen authority and scope

- Commit B remains the public M2.5 evidence authority.
- Commit A remains the implementation-qualification basis bound by Commit B.
- Commit C contains the complete executable runtime, its source-closure list,
  tests, and this engineering contract.
- Commit D must be the single child of C and add exactly one file:
  `Hegel Machine/artifacts/phase3_m3_runtime/local_two_commit_admission_v1.json`.
- Admission is limited to `CANONICAL_ENUMERATION_ONLY`.
- Role evaluation, target synthesis, ACTIVE promotion, network fetch, and
  Docker pull remain forbidden.
- All project Git commits in this stage are local only; publication is a later,
  separately authorized action.

## 3. Unique host custody

The formal host custody parent is the literal path
`/home/erzhu419/.local/state/hegel-machine`.  It is intentionally not derived
from `HOME` or `Path.home()`: environment redirection must not create a second
"canonical" state for the same run ID.

The unique run root is:

`/home/erzhu419/.local/state/hegel-machine/phase3-m3-e4af9f57c38fb298462ec628c4ed8a03`

The parent and run root must be real, caller-owned, mode-0700 directories.
State, sidecar, lock, and terminal files use no-follow, pinned-directory,
owned-file checks and mode 0600.

## 4. Commit C to Commit D admission

At C, the artifact builder verifies:

1. C is the sole child of the frozen Commit B;
2. every allowlisted runtime source is a regular, symlink-free file;
3. Git blob, C tree, index, and working-tree bytes and modes agree;
4. the approval path is absent; and
5. the scoped status is clean before and after the snapshot.

The builder returns canonical bytes and never writes them.  Those exact bytes
are added as the only D change.  At D, live admission verifies `B -> C -> D`,
the one-file diff, the artifact self-hash, the full source manifest, and the
C/D/index/worktree equality.  Returned evidence is recursively immutable.

Repository-local hooks, fsmonitor, untracked cache, external diff, replacement
objects, lazy fetch, interactive credentials, and user/global Git
configuration are disabled for admission queries.

## 5. Direct entrypoints and dependency closure

Formal start and execution are accepted only through their committed direct
entrypoint files under exact `/usr/bin/python3.10 -I -S -B`.  The interpreter
version and binary digest are checked, and bytecode is redirected to a fresh,
owned, empty mode-0700 directory.

Python 3.10 TOML parsing uses the vendored pure-Python `tomli` 2.4.1 source and
license.  Formal signature replay uses the Commit-C hash-pinned offline OpenSSL
boundary rather than importing an uncommitted site-package dependency.  The
runtime source closure contains no more than 64 explicitly committed paths.

## 6. Exact-once state transitions

The start action alone may publish index-zero
`RUNNING/CANONICAL_ENUMERATION`.  The canonical state and its publication
sidecar bind the Commit-D admission artifact, source manifest, and receipt.
The state is never overwritten.  An identical retry is a replay; a different
occupant fails closed.

Formal execution holds an attempt-wide advisory lease across terminal replay,
container execution, archive validation, and terminal publication.  A waiter
replays both the start publication and live Commit-D admission after acquiring
the lease.  If another process has already terminalized, the waiter validates
that terminal record and never enters Docker.

If container non-running status cannot be proven, terminalization is forbidden
and the persisted state remains `RUNNING`; recovery must first establish safe
containment.  Ordinary execution and semantic failures publish their distinct
fail-closed terminal forms.

## 7. Offline dual enumeration

The Python and Rust roles run concurrently, each with:

- an immutable locally present image selected by digest;
- `--pull=never`;
- `--network=none`;
- `--restart=no`;
- a deterministic container name;
- a read-only committed source/input mount and bounded private output; and
- intent, start, completion, stdout, and containment journals.

The supervisor treats either one-sided report, archive, replay, or I/O failure
as an execution failure.  Only two independently valid results that disagree
become a dual semantic failure.  Unsafe terminalization has absolute
precedence.

## 8. Current budget semantics

The frozen wire registry retains `INCONCLUSIVE_BUDGET`, but the currently
qualified enumerator and frozen inputs have a preregistered raw application
count of 3,292,439, below the 5,000,000 cap.  The accepted live result is
therefore the exact `DSL_TOO_LARGE` witness at canonical program 50,001 after
the 50,000-program archive.

The present host archive shape cannot truthfully encode a live partial-budget
run.  Consequently a live `INCONCLUSIVE_BUDGET` report from this qualified
runtime is treated as runner/runtime mismatch and requires requalification.
The registry value remains reserved; it is not deleted or silently redefined.

## 9. Admission and execution gates

Formal start remains forbidden until all of the following hold:

1. the complete runtime and tests pass from the clean C snapshot;
2. C is committed locally and its runtime source manifest is reproducible;
3. D adds only the exact builder-produced admission artifact;
4. live D admission passes through both direct entrypoints;
5. no named M3 container exists or is running;
6. the local images and qualified Rust binary match their frozen identities;
7. a non-formal offline Docker preflight succeeds without pulling; and
8. the canonical state and terminal paths are absent.

Only an explicit `phase3-m3-start` action may then create the state.  Only a
separate explicit formal-execution action may start the concurrent enumeration.
Passing admission does not itself change M3 state.

## 10. Permitted claims after execution

Before a terminal record exists, the strongest statement is that the runtime
is locally admitted and the run is either `NOT_RUN` or explicitly `RUNNING`.
After a terminal record exists, claims are limited to the exact closure status
validated by that record.  No result from this stage alone establishes
language-external odd-cardinality invention, sink-mechanism discovery, role
evaluation success, Phase-3 exit, or ACTIVE eligibility.
