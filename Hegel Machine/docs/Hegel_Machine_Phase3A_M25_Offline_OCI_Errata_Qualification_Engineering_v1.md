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
