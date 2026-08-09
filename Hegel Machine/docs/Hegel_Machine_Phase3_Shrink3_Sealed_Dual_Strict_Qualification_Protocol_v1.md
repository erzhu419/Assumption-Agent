# Hegel Machine Phase-3 shrink-3 sealed dual strict qualification protocol v1

Status: **ENGINEERING QUALIFICATION PROTOCOL; NON-FORMAL AND NOT RUN**

This protocol closes the gap between equal built-in category counts and replay
of one sealed input identity.  It does not execute closure, generate a formal
root, instantiate a seed, evaluate a target role, sign a certificate, or
change ACTIVE governance.

## 1. Frozen evidence basis

The evidence generator is the ordered 36-vector manifest in
`phase3_shrink3_golden_vectors_v1.py`.  Its exact commitments are:

```text
manifest root  sha256:e091e08f33be8bbfa579b6d333f618326b4ed2ebae6d2830d3adc0df7a6333b5
outcome root   sha256:b37fcb96c78d53f7da3271513e0cae128ab7e2538288b8aa723254a0f98fde74
vector count  36
```

Each recognizer receives the exact source JSON bytes or formal CBOR bytes from
that manifest.  Category counts alone are insufficient.  An accepted outcome
is normalized as:

```text
"ACCEPT" || 0x00 || u64be(cbor_length) || canonical_ast_cbor || raw_sha256_digest
```

A rejected outcome is normalized as:

```text
"REJECT" || 0x00 || ASCII(error_code)
```

The supervisor independently recomputes
`SHA256("HEGEL/AST/V1" || 0x00 || canonical_ast_cbor)` before accepting an AST
hash.  Python error detail and Rust error message text are deliberately
excluded from the normalized identity.

## 2. Commit and snapshot binding

Qualification must use a full 40-hex commit that already contains the
supervisor, tests, manifest, both direct Python entrypoints, their complete
local dependency closure, and all four Rust strict-canonicalizer crates.

The supervisor requires the corresponding worktree bytes to equal every Git
blob selected from that commit, records an ordered source-file root, then uses
`git archive` to create a temporary committed snapshot.  Recognizers never run
against the mutable worktree. After extraction, every executed file is hashed
again and must reproduce its Git blob, which also fails closed on archive
attribute transformations. The result records the commit, parent, subject,
Hegel Machine tree OID, archive SHA-256, every source blob row, and the framed
source-file-set root.

Protocol code and result evidence therefore use separate local commits:

1. commit K freezes the supervisor and its tests;
2. qualification runs from the commit-K archive;
3. commit L adds only the commit-bound evidence artifact.

## 3. Hard-isolated recognizer roles

The host supervisor is only the evidence generator and comparator.  The two
untrusted recognizers run in distinct digest-pinned images from the committed
container profile:

```text
Python  python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3
Rust    rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89
```

Every container uses `--pull never`, `--network none`, `--cap-drop ALL`,
`no-new-privileges`, a read-only root filesystem, the committed seccomp
profile, `--pids-limit=64`, `--memory=512m`, and the profile's exact private
tmpfs. Runtime recognizers use uid/gid 65534. Python additionally uses
`-I -S -B` and a read-only committed snapshot mount.

The host control plane is also frozen: `/usr/bin/docker` is invoked with the
explicit `--host=unix:///var/run/docker.sock` argument under an exact six-key
environment and a fresh empty private Docker client configuration. Ambient
proxy, context, credential-helper, and alternate-daemon variables are absent.
Before execution the supervisor records and hashes the live local Linux daemon
and Unix-socket identity. Each container has a unique exact name so a timeout
can force-remove and verify only that container before volume cleanup.

Rust is built once with `--release --locked --offline` from the same read-only
snapshot and a read-only host Cargo registry.  Its target uses a fresh,
commit-named Docker volume.  Runtime containers mount that volume read-only;
the supervisor hashes the executable and removes the dedicated volume after
the run. The volume is explicitly created with the local driver and must
inspect as local scope with no options. The supervisor refuses to reuse or
delete any pre-existing volume.

Build ownership is frozen separately in
`phase3_shrink3_offline_build_profile_v1.json`. The build-only container uses
uid/gid 0 with all capabilities dropped and a root-owned private tmpfs; this is
required because rustc must create temporary files while the fresh target
volume is still root-owned. It has no target/split/seed/key/formal-root input.
This exception does not change the two recognizer runtimes, which remain
uid/gid 65534 with the actor-profile tmpfs.

These roles provide technical process/container isolation under the
owner-accepted threat model. They remain under the same administrative
controller; they are not independent organizations or independent humans, and
the evidence must say so explicitly.

## 4. Required agreement

For every vector, Python and Rust must agree on:

- acceptance versus rejection;
- exact error code for rejection;
- exact canonical CBOR and AST hash for acceptance;
- root operator, output sort, depth, and node count;
- normalized outcome bytes and their ordered outcome root.

Python rejection exits zero because it returns a diagnostic object.  Rust
acceptance exits zero, expected rejection exits one, and exit two is always a
transport/protocol failure.

Both built-in 36-vector controls must also pass.  The independent 2,160-source
survivor replays must agree on every common field, including:

```text
accepted-set commitment
sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e
```

That capacity replay remains `SURVIVOR_SUBSET_ONLY_NOT_COMPLETE`; it is never a
closure claim.

## 5. Authority guard

Even a successful report is limited to:

```text
status       SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS
claim level  NON_FORMAL_DUAL_STRICT_QUALIFICATION
```

It must state all of the following:

```text
execution_state                  NOT_RUN
closure_executed                 false
formal_roots_generated           false
formal_roots                     null
certificate/signature/seed       absent
target_roles_evaluated           false
ACTIVE governance changed        false
formal state transition allowed  false
```

This qualification permits development of independent complete enumerators;
it does not qualify either complete enumerator and cannot start M3.

## 6. Invocation

After committing this protocol and supervisor locally:

```bash
python3 'Hegel Machine/tools/phase3_shrink3_dual_strict_qualification_v1.py' \
  --basis-commit FULL_40_HEX_COMMIT \
  --workers 8
```

The command emits one canonical diagnostic JSON object to stdout.  A failure
emits one stable fail-closed object to stderr and no evidence artifact.
