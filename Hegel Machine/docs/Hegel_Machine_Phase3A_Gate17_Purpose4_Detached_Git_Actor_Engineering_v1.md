# Phase-3A Gate-17 Purpose-4 Detached Git Actor Engineering v1

Status: **IMPLEMENTED AS A NO-KEY ADAPTER; NOT YET A FORMAL GATE-17 EXIT**

This amendment records the engineering boundary for the owner-accepted
purpose-4 technical actor.  Docker isolation is accepted here as a technical
role boundary.  It does not claim different people, administrators,
organizations, or resistance to a malicious host/Docker daemon.

The implementation is:

- `src/hegel_machine/phase3_m25_purpose4_detached_audit_v1.py`;
- `tools/phase3_m25_purpose4_detached_audit_worker_v1.py`;
- `tests/test_phase3_m25_purpose4_detached_audit_v1.py`.

It does not generate or load a split seed, signing key, signature, marker,
formal root publication, or M3 execution identity.

## 1. Independent evidence generation

The host creates a temporary Git repository whose worktree contains only
`.git`.  Its object database contains exactly the objects reachable from the
frozen audited parent
`fb3a3ee4865a140c558821017ddd3e9a6a99de48`; no other Git object is admitted.
The snapshot has:

- `HEAD` and `refs/hegel/audited-parent` fixed to that parent;
- SHA-1 object format and a complete, strict `git fsck` replay;
- no shallow boundary, alternates, HTTP alternates, grafts, replacement refs,
  promisor packs, partial-clone configuration, or remotes;
- a complete reachable-object type/size inventory;
- a complete `.git` path/length/SHA-256 inventory;
- no checked-out host payload and no host-generated audit rows.

The purpose-4 worker receives this snapshot read-only and independently calls
the existing full parent-absence generator.  It recomputes commit, touched
path, unique blob, path-predicate, content-predicate, diagnostic receipt, and
formal audit-bundle identities from the detached object database.  The host
acceptor replays the complete compact receipt and reconstructs the canonical
CBOR, attestation root, and exact purpose-4 signature preimage.

The temporary snapshot and runtime are outside the repository by default,
are sealed read-only before container use, are unsealed only for cleanup, and
are removed on success or failure.

## 2. Commit-A and executable binding

Every run has an explicit lowercase 40-hex `basis_commit_sha1`.  The value is
bound independently into the snapshot manifest, actor request, runtime-source
binding receipt, and actor response.

Before actor launch, every Python source and both container-control JSON files
(the actor profile and seccomp policy) are resolved as blobs in that exact
basis commit.  Their current worktree bytes must equal both the committed blob
bytes and Git blob SHA-1.  The committed control JSON copies, rather than live
repository paths, are used for the actor runtime and Docker seccomp input.  A
modified, staged-only, untracked, missing, or otherwise non-Commit-A input fails with
`FAIL_GATE17_PURPOSE4_RUNTIME_BASIS`.  This intentionally means that this
adapter cannot perform a qualifying run until its implementation has first
been committed as Commit A.

The Git executable is an external dependency rather than a repository blob.
Its exact resolved executable bytes, SHA-256, length, and `git --version`
output are bound into the snapshot and runtime receipts.  The same executable
is mounted as `/runtime/bin/git` and revalidated inside the actor.  The Python
image is separately required to be an already-local digest-pinned profile
image.

## 3. Offline container profile and live evidence

The adapter uses only:

- `--pull=never` and `--network=none`;
- a digest-pinned `policy_auditor` image already present locally;
- read-only root, snapshot, runtime, and request mounts;
- non-root UID/GID `65534:65534`;
- all capabilities dropped, no-new-privileges, and the frozen custom seccomp
  profile;
- `pids-limit=64`, memory `512m`, memory-swap `512m`, private IPC, exact
  nofile limit, and a 64 MiB purpose-private `/tmp`.

No later network access is needed.  Image acquisition, if ever required, is a
separate one-time provisioning action outside this adapter; this adapter will
not pull it.

The worker emits measured live-probe evidence, not merely requested flags.  It
requires:

- UID/GID, zero capability sets, `NoNewPrivs=1`, and `Seccomp=2`;
- only loopback in the network namespace and distinct namespace identities;
- six frozen forbidden syscalls returning `-1/EPERM`;
- writes to root, snapshot, runtime, and request denied while private `/tmp`
  remains writable;
- forbidden host/cross-purpose paths absent and inherited FDs exactly 0/1/2;
- cgroup-v2 memory max 512 MiB, swap max zero, and pids max 64;
- the exact environment allowlist.

The receipt is self-digested and is included in the response.  The host
revalidates every measurement.  Recomputing only the outer JSON digest cannot
make a tampered resource, seccomp, path, basis, or parent receipt acceptable.

## 4. Signature-chain integration boundary

The current worker is intentionally a no-key generator.  It outputs:

- the complete public parent-absence receipt;
- exact canonical `ParentManifestAbsenceAttestationV2` CBOR;
- its formal content root;
- the exact purpose-4/epoch-0 Ed25519 signature preimage;
- Commit-A, image, Git, runtime, snapshot, and live-probe bindings;
- explicit `signature_present=false` and no secret material.

Formal Gate 17 must not be closed by handing this response to a host-side
signature oracle.  Actual ceremony integration must run the same detached
snapshot replay and response validation inside the same purpose-4 key-bearing
container that performs the single purpose-4 signature.  The key-bearing
container must fail before signing if any Commit-A source, Git/image binding,
snapshot inventory, complete audit receipt, canonical CBOR/root/preimage, or
live-probe check differs.  Only the signed envelope and public replay receipt
may cross back to the host.  The purpose-4 container must not expose purpose-1
seed or key state.

The frozen attestation schema must not be silently extended to carry the
live-probe receipt.  Ceremony integration must bind that receipt through the
existing actor-qualification/transaction provenance path, or make an explicit
future wire amendment before claiming that it is cryptographically covered.
Until that integration is implemented and replayed, the honest status is:

> `PURPOSE4_DETACHED_REPLAY_QUALIFIED_NO_KEY`; Gate 17 remains open.

## 5. Admission sequence for the later executor adapter

1. Commit this implementation and select that exact commit as Commit A.
2. Re-run source/blob binding; reject any dirty actor-source byte.
3. Confirm the digest-pinned image is already local; never pull during the
   ceremony.
4. Build and validate the exact detached parent snapshot outside the Git
   worktree.
5. Start the purpose-4 key-bearing container under the frozen 512 MiB profile.
6. Inside that same container, run the live probes and full independent audit,
   construct and revalidate CBOR/root/preimage, and only then sign.
7. Return and replay the public receipt and envelope; durably bind the live
   probe through the ceremony provenance record.
8. Clean all temporary snapshot/runtime state and prove the purpose-4 private
   state lifecycle required by the ceremony executor.

The adapter API `prepare_detached_parent_snapshot_v1(...)` plus
`run_purpose4_detached_audit_v1(...)` supplies the no-key qualification half
for that later executor integration.  It deliberately does not mutate the
current ceremony executor or transaction implementation.
