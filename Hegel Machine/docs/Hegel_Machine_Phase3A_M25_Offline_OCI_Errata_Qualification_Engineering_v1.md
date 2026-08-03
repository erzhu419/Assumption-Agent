# Phase-3A M2.5 Offline OCI Errata Qualification Engineering v1

## Status

This document records the engineering replacement for the obsolete host-Rust
qualification path. It is diagnostic implementation evidence only. It is not
an external attestation, does not satisfy Gates 15–24, leaves the gate count at
14/24 and leaves M3 in `NOT_RUN`.

No step in this qualification generates or accepts a real split seed, private
key, signature, formal root, run ID, ledger ID, marker or M3 transition.

## Closed blocker

The prior runner required a particular host `rustup`, `cargo`, `rustc` and
complete host toolchain directory. That made a missing host Rust installation
an unrelated hard blocker. The runner now requires only the shared Linux-local
Docker control plane and a locally present, digest-pinned image:

```text
rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89
```

The approved policy binds the OCI manifest/image ID, Linux/amd64 platform,
container-internal cargo/rustc paths and binary hashes, cargo version, full
`rustc -vV` output hash, exact runtime/build environment hashes and separate
runtime/build seccomp hashes. Image `Config.Env` is ignored by launching the
container through `/usr/bin/env -i` with an exact allowlist.

Every Docker execution uses the absolute `/usr/bin/docker`, the explicit local
Unix socket, an empty private client configuration and sanitized host
environment. Runtime and build both use `--pull=never --network=none`, a
read-only root filesystem, all capabilities dropped and no-new-privileges.

## Dependency snapshot

The host Cargo cache is never mounted into a container. The runner reads it
only to locate `.crate` archives, verifies every archive against the exact
checksum in detached Commit-A `Cargo.lock`, rejects unsafe archive members and
extracts a run-private vendor tree on the Linux-local filesystem.

The frozen formal-bridge snapshot is:

```text
domain: HEGEL/M25/CARGO_DEPENDENCY_SNAPSHOT/V1
Cargo.lock SHA-256: 605c610af13de12a7374a15410ed3cda6ce3091b089e1b6680a0c474bc6602b0
registry packages: 23
snapshot files: 1088
snapshot root: 4e5d54ced90acddbc17de82aba21baa331d26eb3178738bc319ffd269c5d7abd
```

Cargo receives only detached source (`/input:ro`), this vendor snapshot
(`/vendor:ro`) and one fresh target (`/output:rw`). The normalized build is
`cargo build --release --locked --offline --jobs=1` with explicit source
replacement to `/vendor`.

## Validation and persistence order

1. Verify all bound worktree bytes equal Commit A and extract them with
   `git archive` into a private snapshot.
2. Qualify the local Docker daemon and OCI toolchain probes.
3. Build the exact checksum-verified vendor snapshot.
4. Run the detached Python endpoint and the freshly built Rust endpoint.
5. Require exact Python/Rust/checked-golden equality before publication.
6. Atomically replace only
   `rust/formal_bridge_m25/target/debug/hegel-formal-bridge-m25` with the
   already validated bytes, mode `0755`.
7. Execute that persisted binary again in the same offline OCI boundary and
   require identical digest and report.
8. Recheck detached-source, vendor, seccomp and Commit-A stability.

The path remains the historical `DEFAULT_RUST_BINARY` path so formal-static
basis and ceremony code consume the qualified binary without a second,
unbound compiler path. The release binary stored at that path is identified by
its receipt digest; the path name does not claim a Cargo debug profile.

## Verified local evidence

An offline engineering replay on 2026-08-02 completed without registry access:

```text
fresh binary SHA-256: d38eabce2be158326fe16a7185ffc2c9be1262ce8d5098afed25eff431465093
Rust report equals checked errata golden: true
persisted binary replay equals fresh replay: true
```

The errata unit suite passes its pre-Commit-A tests and explicitly skips live
qualification, CLI and checked-artifact assertions while the new bound files
are not yet all present in `HEAD`. This is the required fail-closed state, not a
guard relaxation. After Commit A, those tests must run rather than skip, the
fresh qualification report must be regenerated under schema
`hegel-phase3-m25-exact-wire-errata-qualification/2`, and the checked artifact
must validate before any later ceremony step uses it.

## Post-commit qualification corrections

The fresh report is a Commit-B public-evidence object. Its only in-repository
output path is
`artifacts/phase3_m25_external/phase3_m25_errata_qualification_v1.json`; the
direct `artifacts/phase3_m25_errata_qualification_v1.json` file is retained as
a historical pre-Commit-A diagnostic and is not overwritten.

The history-complete secret-absence policy is version 2. Private-key magic
headers match only at the start of a blob or a CR/LF-delimited record, after
optional horizontal ASCII whitespace. Complete PEM/OpenPGP header-to-footer
blocks are additionally findings at any offsets, including JSON/YAML/Markdown
escaped strings; age, PuTTY and binary OpenSSH magic are findings at any
offset. Only an isolated inline PEM/OpenPGP header example without a footer is
outside these two rules unless the filename/JSON-key rules independently
match.

History enumeration does not use `rev-list` parent semantics. It traverses
the raw parent headers returned by `cat-file commit`, requires every parent,
tree and blob locally, and scans every unique raw ancestor tree. Shallow,
promisor/partial-clone, alternate-object, graft and replace-ref metadata fail
closed. This keeps a local graft or shallow boundary from turning a deleted
ancestor finding into a false absence receipt. The exact match and history
rules are included in the public policy payload and report identity.

Output publication is guarded against the entire Git toplevel, not only the
`Hegel Machine` subtree. The only allowed in-repository lexical path is the
Commit-B evidence path above. Every parent is opened by a dirfd walk with
`O_DIRECTORY|O_NOFOLLOW`; the validation dirfd is the same held parent used
for `O_EXCL` creation. The created fd remains open through fsync and replay.
A second lexical walk must reach the same parent inode and the reopened file
must retain the same regular-file inode, owner, mode, link count, size and
exact bytes; the original fd is reread once more before success. Failure
cleanup unlinks only the inode created through the held parent. The tracked
`artifacts/phase3_m25_external/README.md` makes the unique parent available in
a clean clone without runtime directory creation. The publisher never
overwrites the historical diagnostic or another repository path.
