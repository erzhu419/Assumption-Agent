# Phase-3A M3 Implementation-Binding Qualification — Engineering v1

Status: implemented and diagnostically replayed; Commit-A roots not yet
instantiated.  M3 remains `NOT_RUN`.

## 1. Scope and authority boundary

This stage qualifies two runnable, bit-exact bounded closure enumerators as
inputs to `ImplementationBindingV1`:

1. `hegel-python-m3-bounded-closure-enumerator-v1`;
2. `hegel-rust-m3-bounded-closure-enumerator-v1`.

It does **not** create a split seed, private/public key, signature, run ID,
ledger ID, role output archive, match receipt, closure certificate, formal M3
output root, or M3 state transition.  Even an exact-budget CLI result has:

```text
claim_level = FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY
authoritative_claim_allowed = false
```

Only later formal M3 receipt assembly may promote a bound result.  A bare CLI
JSON document is rejected if it claims `FORMAL_ENUMERATION_OUTPUT` or sets
`authoritative_claim_allowed=true`.

## 2. Hard technical-role isolation

The owner-accepted Docker isolation rules are used as a hard technical
boundary. This is technical independence under one administrator, not a claim
of organizational or independent-human custody. The enumerators do not reuse
the formal actor profile identity: their runtime profile ID is
`hegel-m3-enumerator-runtime-docker-v1`, the non-authoritative Rust compiler
profile is `hegel-m3-rust-offline-build-docker-v1`, and the composite dual
qualification policy is `hegel-m3-dual-offline-qualification-docker-v1`.

Both execution images are digest pinned:

- Python: `python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3`
- Rust: `rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`

Every build/replay uses `--pull=never` and `--network=none`. Cargo additionally
uses `--locked --offline` and `CARGO_NET_OFFLINE=true`. The build container is
never given `~/.cargo`, a registry index, or the host Cargo cache. The host
parses the committed `Cargo.lock`, verifies every selected `.crate` SHA-256,
extracts the exact 21-package set into a run-private mode-0700 vendor snapshot,
and mounts only that snapshot read-only.

The one-time bootstrap of the sole previously absent archive (`libc 0.2.189`)
is frozen in `artifacts/phase3_m3_cargo_offline_bootstrap_record_v1.json`.
Qualification requires its lock hash, package count, post-bootstrap policy,
snapshot root and 1,054-file count to match a fresh reconstruction. The exact
snapshot root is
`56e25972693d7ef40b3acb16e9efcc538a5e810fbbd07a7f647c4f4e75084ada`.
No later registry or package-network access is permitted.

Runtime replay uses
`config/phase3_internal_actor_seccomp_v1.json`.  Cargo compilation requires
local `socketpair`-style process IPC, so it uses the separately committed
`config/phase3_m3_offline_build_seccomp_v1.json`: `socketpair` is allowed, but
the `socket` syscall and all network access remain blocked.  Both seccomp file
digests are included in the typed qualification receipt.

All temporary state is created through the validated Linux-local `/tmp`
boundary. Docker is invoked only as absolute `/usr/bin/docker`, with explicit
`--host=unix:///var/run/docker.sock`, an empty private client configuration,
and a sanitized environment containing no proxy variables. The live local
Linux daemon identity is validated and its 32-byte receipt binding is included
in the qualification receipt.

The build container is a non-authoritative compiler actor.  It is not one of
the formal Purpose 1--4 actors and has no seed, custody, signing, ledger, gate,
or state-transition permission.  It runs as the invoking numeric uid/gid, so
its private `0700` target directory never needs world-write permission. The
runtime enumerators likewise use the invoking non-root numeric uid/gid so the
two private output trees remain reclaimable by the host; read-only root,
capability-drop, no-new-privileges, seccomp and network isolation remain exact.

The Python and Rust source trees are materialized from exact Git blobs into
different Linux-local temporary directories and mounted read-only.  Neither
container receives the repository, benchmark data, split material, custody
directory, or another actor's source snapshot.

## 3. Target-free source closure

Merely setting `target_roles_evaluated=false` was not sufficient: the original
Python import closure could see `phase3_dsl_v1.py`, and the original Rust crate
depended on the target/ceremony-aware `formal_bridge_m25` crate.  Both leaks
were removed.

### 3.1 Python closure

The Python snapshot contains exactly these nine files:

```text
phase3_m3_isolated_entrypoint_v1.py
phase3_m3_bounded_enumerator_cli_v1.py
phase3_m3_bounded_enumerator_v1.py
phase3_m3_dsl_core_v1.py
phase3_m3_shrink1_core_v1.py
phase3_m3_record_wire_v1.py
strict_ast_v1.py
strict_ast_shrink1_v1.py
strict_cbor_v1.py
```

The direct entrypoint creates only a minimal in-memory package shell and does
not execute `hegel_machine/__init__.py`.  The frozen AST constants and three M3
record schemas were projected into target-free modules.  A mechanical AST
walk proves that every relative import is present and that there are no
unreachable extra files.

### 3.2 Rust closure

The Rust closure contains the root enumerator crate and only its two reachable
path crates:

```text
rust/m3_closure_enumerator
rust/strict_canonicalizer
rust/strict_canonicalizer_shrink1
```

For each crate the closure binds `Cargo.toml`, `Cargo.lock`, `src/lib.rs`, and
`src/main.rs`; the root crate also binds `src/formal_core.rs`.  Cargo path
dependencies are resolved mechanically and must equal this exact crate set.
The enumerator now owns a small target-free deterministic-CBOR/RFC6962 core and
has no dependency on `formal_bridge_m25`.

Both closure validators reject benchmark universes, odd/sink target IDs,
split-seed material, ceremony imports, forbidden source paths, omitted local
imports, unexpected path crates, and unreachable source extras.

## 4. Typed golden vector

`golden_vectors/phase3_m3_bounded_dual_agreement_v1.json` is strict JSON with
an exact field set.  Its formal identity is not the JSON byte hash.  The
validator converts its ordered fields into a numeric/byte-string-only
deterministic-CBOR array and computes:

```text
ContentHash(
  "HEGEL/M3_ENUMERATOR_DUAL_GOLDEN/V1",
  typed_golden_array
)
```

Current typed golden root:

```text
2b055cc13ba13791cd0f3267217fe3734fa3c9371ab6e7f561b79b64fe18dfea
```

The `11/22/33` input roots are synthetic qualification bindings only.  They
are never published as the Phase-3 formal DSL/operator/registry roots and
never become M3 output roots.

## 5. Full dual agreement

After the target-free refactor, the complete frozen profile was replayed by
both implementations.  They agree exactly:

| Field | Exact value |
|---|---|
| status | `DSL_TOO_LARGE` |
| raw operator applications | `3,292,439` |
| retained canonical programs | `50,000` |
| program archive root | `a23151e07f77edcbebe5b7e2e382e1a81b36c6b15c8997899f7f43dcbda874d1` |
| chunk manifest root | `98c8deb02a62630f5813717a28c3b9deb5a3845e663b9af5a78fe7f9427f453d` |
| bucket accounting root | `5dd13e5d284785dab7fbe3c16fbb1f1bcba3a44466ab0fb258f75f36ee9661ec` |
| first out-of-budget AST hash | `96200a6a131204315ffcd1efd0aa2dcfe2ce665a2c06516461772c9812f0ec71` |
| first out-of-budget AST CBOR | `820184020383010083000103860003050300828201f58203f5` |

This proves an exact bounded-prefix candidate result, not formal closure
authority and not an `OUTSIDE_FROZEN_CLOSURE` certificate.

Agreement is no longer inferred from self-reported JSON roots. Each actor
writes all 50,000 framed program records, 13 chunk manifests and 175 bucket
records. The host strictly decodes every frame, recomputes the three RFC6962
roots and every chunk blob hash, checks contiguous indices, canonical AST
metadata, semantic bindings, traversal order, bucket partitions and the
program-50,001 witness, then requires the complete Python and Rust streams to
be byte-for-byte identical. The boundary bucket's 33,727 additional canonical
programs are explicitly validated as outside-budget residuals, not silently
counted as archived acceptance.

## 6. ImplementationBindingV1 construction

After Commit A exists, qualification constructs each binding with the exact
frozen schema fields:

```text
implementation_id
source_root
binary_digest
execution_environment_spec_root
compiler_or_interpreter_id_digest
compiler_or_interpreter_version_digest
dependency_lock_root
build_profile_id_digest
entrypoint_id_digest
golden_vector_root
repository_commit_id
```

Python's `binary_digest` is the SHA-256 of the resolved interpreter binary
inside the pinned OCI image.  Rust's is the SHA-256 of the release enumerator
built from the Commit-A snapshot.  The entrypoint IDs are distinct:

```text
entrypoint:python-m3-isolated-enumerator-v1
entrypoint:rust-m3-bounded-enumerator-v1
```

The source roots are RFC6962 roots over `SourceFileRecordV1`.  The Rust lock
root is the RFC6962 root over exact `DependencyLockRecordV1` rows from the root
`Cargo.lock`; Python uses the formal empty dependency-lock root.  Environment
roots are `ExecutionEnvironmentSpecV1` content roots bound to the exact OCI
manifest digest and the dedicated enumerator runtime policy ID. They never
bind the Purpose-actor profile ID. Rust's `build_profile_id_digest` binds the
separate non-authoritative offline-build policy; the typed receipt binds that
policy to its distinct build seccomp digest.

The repository commit is a binding field, so the two exact
`ImplementationBindingV1` roots cannot truthfully be computed before Commit A.
No placeholder root is allowed.

## 7. Qualification receipt and substitution guard

The full replay creates a strict, machine-readable receipt with:

- exact Commit-A ID, golden root, source roots and file counts;
- exact dependency-lock and environment roots;
- interpreter/enumerator binary digests;
- compiler/interpreter version digests;
- distinct entrypoint digests;
- both `ImplementationBindingV1` roots;
- canonical report and raw stdout digests;
- SHA-256 of all three independently emitted framed archive streams;
- exact archive-file-set, strict host replay and witness-adjacency flags;
- Rust build stdout/stderr digests;
- checksum-exact Cargo snapshot root/file count and committed bootstrap-record
  digest;
- both seccomp digests and offline Docker policy;
- validated local Docker daemon/control-plane binding;
- the exact count/root/witness agreement;
- explicit target/split/secret absence flags;
- `formal_m3_output_roots_generated=false` and `m3_state=NOT_RUN`.

The receipt itself is a typed deterministic-CBOR array under
`HEGEL/M3_IMPLEMENTATION_QUALIFICATION/V1`.  Extra keys, missing keys, type
drift, a changed golden, or authority promotion all fail closed.

The ceremony preflight additionally requires the exact generated Rust path
for Commit A and rehashes that binary. It recomputes the formal source,
dependency, environment and implementation-binding preimages; revalidates the
committed bootstrap record against `Cargo.lock`; replays the stored daemon
identity receipt; and, in live mode, requires the same daemon binding before
re-probing Python. It also checks both entrypoint IDs, source path sets, image
references, seccomp digests, and receipt root. The old static replayers and an
arbitrary executable at another path are therefore not valid substitutes.

## 8. Commit-A operation

After all implementation files are committed locally, run from `Hegel
Machine`:

```bash
PYTHONPATH=src python3 -m \
  hegel_machine.phase3_m3_implementation_qualification_cli_v1 \
  --basis-commit "$(git rev-parse HEAD)" \
  --output /an/external/public/path/phase3_m3_implementation_qualification_v1.json
```

The output path is exclusive-create.  This command performs only offline
build/replay qualification.  It does not invoke the ceremony executor.  Once
the receipt validates, the qualified in-memory static basis sets
`m3_execution_implementation_bindings_ready=true` and removes only
`M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY`.

## 9. State after this engineering stage

Before Commit A qualification:

```text
m3_execution_implementation_bindings_ready = false
M3 state = NOT_RUN
formal ceremony data = absent
```

After a successful Commit-A qualification:

```text
m3_execution_implementation_bindings_ready = true
M3 state = NOT_RUN
formal ceremony data = absent
formal M3 output roots = absent
```

The remaining external-genesis/custody/signature gates and the separate
explicit `phase3-m3-start` action remain independent later steps.
