# Phase-3A Gate-17 Purpose-4 Key-Bearing Detached Actor Engineering v1

Status: **IMPLEMENTED AND SYNTHETICALLY QUALIFIED; NOT EXECUTOR-INTEGRATED; NOT A FORMAL GATE-17 EXIT**

This amendment implements the missing key-bearing half of the owner-accepted
Docker technical-actor model.  It treats one long-lived, purpose-private,
offline Docker container as the purpose-4 actor.  That is a real technical
isolation boundary for this project, but it is not a claim that a different
human, organization, host administrator, or Docker daemon controls the role.

The implementation is:

- `src/hegel_machine/phase3_m25_purpose4_keybearing_detached_v1.py`;
- `tools/phase3_m25_purpose4_keybearing_detached_worker_v1.py`;
- `tests/test_phase3_m25_purpose4_keybearing_detached_v1.py`.

The earlier no-key layer remains unchanged:

- `src/hegel_machine/phase3_m25_purpose4_detached_audit_v1.py`;
- `tools/phase3_m25_purpose4_detached_audit_worker_v1.py`.

The no-key layer remains useful for snapshot/runtime qualification and for
diagnosis without key access.  It must not be used as a host-side signature
oracle.  The new key-bearing worker repeats the complete detached replay in
the same process that holds the purpose-4 key and performs the signature.

## 1. Security boundary

The host request schema is
`hegel-gate17-purpose4-keybearing-detached-request/1`.  Its exact information
classes are:

1. Commit-A identity: `basis_commit_sha1`;
2. digest-pinned actor image identity;
3. the complete detached-snapshot manifest and its self-digest;
4. the immutable runtime inventory and Commit-A source bindings;
5. one audit timestamp;
6. the expected key ID already derived from the purpose-4 container's local
   public key;
7. three mandatory false provenance markers.

The request has no formal audit row, attestation CBOR, attestation root,
signature preimage, signature, private key, seed, or custody marker.  Unknown
fields fail closed.  A forbidden field fails specifically with
`FAIL_GATE17_PURPOSE4_HOST_SIGNATURE_ORACLE`, including when nested or when an
outer digest has been recomputed after insertion.

The worker reads only:

- `/input/purpose4-keybearing-request.json`;
- `/input/runtime`, an immutable Commit-A runtime bundle;
- `/input/detached-parent-snapshot`, an immutable detached Git object store;
- `/state/ed25519-private.pem`, an existing purpose-private mode-0600 key;
- its private `/tmp` and writable `/output`.

It has no purpose-1 custody mount and never reads the raw split seed.  The
worker contains no key-generation path.  Key creation/recovery remains the
separate earlier operation in the same long-lived purpose-4 container.

## 2. Same-process operation order

The only accepted operation is `purpose4-parent-sign`.  One invocation does,
in order:

1. validate UID/GID 65534, purpose/profile identity, and the exact request;
2. run `phase3_m25_actor_operation_probe_v1.qualify_operation_v1(...)` in the
   same worker process, with the operation environment bound to the request
   digest, Commit A, image, and purpose;
3. recompute the complete runtime inventory and validate every Commit-A
   source/dependency binding;
4. validate the detached snapshot, its exact object/file inventories, the
   frozen parent, and the bound Git executable;
5. resolve every audit Git command through the immutable runtime Git bytes;
6. generate every parent-absence audit row from the detached object database;
7. replay the formal bundle and independently regenerate it again from the
   same detached Git objects;
8. build the public replay receipt and canonical
   `ParentAbsenceAuditBundleV1` CBOR/root;
9. build canonical `ParentManifestAbsenceAttestationV2` CBOR/root using the
   local key ID and requested timestamp;
10. derive
   `external_signature_preimage_v1(tag=ParentManifestAbsenceAttestationV2,
   root, purpose=4, epoch=0)` inside the worker;
11. derive the public key from `/state`, verify the expected key ID, sign the
   internally derived preimage, and verify that signature before response;
12. emit one self-digested public response.

The frozen detached adapter records the Git executable's logical container
path as `/runtime/bin/git`.  Executor integration may physically nest the
same immutable runtime bytes at `/input/runtime`; the worker preserves the
frozen logical binding and separately verifies the actual
`/input/runtime/bin/git` length and SHA-256 before use.  This does not rewrite
the prior snapshot manifest identity.

## 3. Private-key and temporary-file contract

`/state` must be a non-symlink directory owned by the actor with mode 0700.
`ed25519-private.pem` must be a one-link, non-symlink regular file owned by the
actor with mode 0600.  The worker never exports it and never invokes
`openssl genpkey`.

OpenSSL is called by the fixed absolute path `/usr/bin/openssl`, without a
shell and with a minimal fixed environment.  The internally generated
preimage, derived public DER, and signature used for self-verification are
placed only in a newly created mode-0700 directory below the purpose-private
`/tmp`; files are created with `O_EXCL|O_NOFOLLOW`, mode 0600, and fsync.  The
message and signature are never command-line values.  Temporary material is
removed and the temporary parent is fsynced on both success and failure.

The Ed25519 public-key DER must be the exact 44-byte SPKI form with prefix
`302a300506032b6570032100`.  The local key ID is
`SHA256(raw_public_key_32_bytes)[0:16]`.

## 4. Public response and host replay

The response schema is
`hegel-gate17-purpose4-keybearing-detached-response/1`.  It contains:

- Commit-A, image, request, snapshot, runtime, and source-binding digests;
- the exact operation-bound live-probe receipt;
- the strict parent-absence public replay receipt;
- canonical audit-bundle CBOR/root;
- canonical attestation CBOR/root;
- raw public key, key ID, epoch 0, and the 64-byte Ed25519 signature;
- explicit true/false provenance declarations;
- a canonical response self-digest.

It does not return the signing preimage, any private key bytes, raw split seed,
or host-supplied formal rows.  The host validator independently checks exact
field sets and self-digests, canonical CBOR round trips, audit/attestation root
links, the full public-receipt policy, operation-probe bindings, key identity,
and the signature through an explicit verification callback.

Successful validation returns `authoritative=false`.  A synthetic response or
a locally verified component cannot grant Gate 17 by itself.  Authority
requires the later executor integration and a live full ceremony audit.

## 5. Executor integration contract

The current executor is deliberately not modified by this amendment.  Its
later integration must:

1. include every new worker/module dependency as exact blobs in Commit A;
2. prepare the frozen detached snapshot and full runtime before starting the
   long-lived purpose-4 container, under the already read-only `/input` mount;
3. start and qualify that container offline with `--pull=never` and
   `--network=none`;
4. generate or resume the purpose-4 key using the existing separate operation;
5. derive only the public key ID on the host and write the identity-only
   request;
6. set `HEGEL_OPERATION_REQUEST_SHA256` to that exact request digest and run
   the new worker in the same live container;
7. validate the returned response and signature, then use its attestation
   CBOR/root/signature in the formal envelope path;
8. bind the operation probe, runtime, snapshot, and host before/after container
   identity through the existing transaction provenance records;
9. never write `parent-audit-replay.json`, `signing-preimage.bin`, host-built
   audit rows, or host-built attestation CBOR into purpose 4;
10. preserve fail-closed recovery and cleanup semantics.

The integrated executor adopts the builder's mode-0555 snapshot with one
Linux-specific, descriptor-bound rename transaction.  A directory moved
between different parents needs owner-write while Linux updates its `..`
entry, even though rename ordinarily depends on parent permissions.  The
executor therefore opens and pins the source root and both parents, proves
their inode identities, ownership and common filesystem, checks that the
destination is absent, and grants owner-write only to the held snapshot-root
inode.  Descendant directories and files remain read-only.  It then performs
one dirfd-relative Linux `renameat2(RENAME_NOREPLACE)`, restores the held root
to 0555, fsyncs the root and both parents, and replays the complete detached
manifest and file/object inventories at the destination.  An `EEXIST` race
fails closed and the raced target is not registered for cleanup.  No ordinary
rename, replacement, or copy fallback is permitted.  Any failed identity,
permission, filesystem, durability, or replay check unwinds only the exact
adopted inode before the container can start.

The executor persists the adopted `(st_dev, st_ino)` and the detached
temporary-directory owner before validation begins.  Every unwind and later
local-runtime cleanup first uses nofollow descriptors and atomically moves the
bound inode, with no replacement, to a private quarantine name.  Permission
changes and traversal are descriptor-relative; symlinks and identity changes
fail closed.  Every vacated canonical or quarantine name remains registered
until full cleanup, and any object reappearing there becomes a persistent
foreign-entry blocker before either detached-owner or ceremony-wide temporary
cleanup.  A failed detached-owner or temporary cleanup retains the exact
identity, current quarantine path, ceremony root, and retry handles.

This is an owner-accepted trusted-host boundary with the complete disclosure:
`same_admin_controller=true`, `organizational_independence=false`,
`independent_human_actors=false`, `technical_role_independence=true`,
`owner_accepted_threat_model=true`, `remote_attestation=false`, and
`hardware_key_nonexportability=false`.  The host orchestrator and same-UID
administrator are therefore trusted not to run a continuously adversarial
rename/unlink loop inside the private ceremony directory.  The implementation
prevents actor-originated traversal, ordinary filesystem races, symlink
following, stale-name cleanup, and deletion of a replacement observed at the
guard/action boundary.  It does not claim a POSIX unlink-by-handle primitive
or protection from a malicious same-UID controller that keeps mutating names
after the quarantine has been successfully claimed.

In particular, replacing the old `sign_parent(...)` transport is mandatory:
the host may consume the new actor's result but may not choose or supply what
the purpose-4 private key signs.

## 6. Current state

This amendment generated no real seed, key, signature, snapshot root, formal
root, or ceremony identity.  It did not remove the dedicated purpose-4
blocker, advance any gate, start M3, or alter `NOT_RUN`.

The engineering result is:

> `PURPOSE4_KEYBEARING_DETACHED_ACTOR_SYNTHETICALLY_QUALIFIED`

The next admissible action is executor integration followed by a live,
full-history, offline Docker replay in the same long-lived purpose-4 actor.
Only that execution can supply evidence for the Gate-17 transition.
