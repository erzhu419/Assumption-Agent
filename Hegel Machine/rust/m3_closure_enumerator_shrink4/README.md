# Rust shrink-4 complete-closure diagnostic

This crate independently enumerates the target-free canonical closure of
`hegel-old-dsl-v1.4.0`. It inherits the frozen shrink-3
depth/node/sort/operator/CBOR traversal and archive wires. The sole generator
change is that normalized top-level conjunction construction admits exactly
two distinct non-AND children. AND3 is not generated and consumes no raw
operator application. Every candidate is admitted through the independent
Rust shrink-4 strict canonicalizer.

The executable stops only after it has exposed canonical program 50,001 or
after every typed frontier bucket is closed. Its JSON remains
`NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`, `execution_state=NOT_RUN`, and
`formal_roots=null`; it cannot publish a formal M3 transition or evaluate a
target role. The CLI rejects roots other than its domain-separated
`NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1` tuple.

The three diagnostic root preimages are exact UTF-8 byte strings with one NUL
byte at each `\x00` marker:

```text
HEGEL/M3/SHRINK4/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00hegel-old-dsl-v1.4.0\x00hegel-freeze-p2b-p3-v1.4.0\x00shrink-step4\x00maximum-top-level-clauses:2
HEGEL/M3/SHRINK4/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00hegel-old-dsl-v1.4.0\x00hegel-canonical-ast-v1\x00hegel-mdl-prefix-v1.0.0\x00binary-active-formal:1,2,3,5,6\x00binary-tombstones:0\x00binary-source-alias:4\x00maximum-top-level-clauses:2
HEGEL/M3/SHRINK4/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00hegel-old-dsl-v1.4.0\x00aggregate-active:0,1,5\x00aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00rational-tombstones:0,2,4,6\x00rational-reserved:7\x00binary-source-active:1,2,3,4,5,6\x00binary-formal-active:1,2,3,5,6\x00binary-source-alias:4\x00binary-tombstones:0\x00binary-reserved:7
```

Their SHA-256 digests are respectively
`736c9cf98749d9a9d2d98596d15a5b09329e1d6eb74d4bee172837fdd34e876f`,
`45fe7c575759b6955eb6b52ad954a9ca6561083dbdb67155f9731e795c6fe050`,
and `1f9b886480ace19440469267abd06f24c65ef61fb9c734c5ef5ff8ae7e981fd3`.
These are non-formal diagnostic bindings, not formal child roots.

The unchanged AST and deterministic-CBOR wire profiles are rebound only at
the diagnostic-profile layer. Their exact preimages and SHA-256 values are:

```text
HEGEL/M3/SHRINK4/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array\x00maximum-top-level-clauses:2
f9b02ddad69f04f1f9137501dccfdcefa111d0402570197b68b98c11ebcb4eda
HEGEL/M3/SHRINK4/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00hegel-cbor-det-v1\x00RFC8949-deterministic-no-map-text-float-tag-indefinite
b7fd10722f31d780d53b2f490c92491872ffc749b4cb5cdfccc3eebd5f18837f
```

```bash
cargo run --release --locked --offline -- \
  --enumerate-diagnostic \
  --child-dsl-spec-root 736c9cf98749d9a9d2d98596d15a5b09329e1d6eb74d4bee172837fdd34e876f \
  --operator-semantics-root 45fe7c575759b6955eb6b52ad954a9ca6561083dbdb67155f9731e795c6fe050 \
  --identifier-registry-root 1f9b886480ace19440469267abd06f24c65ef61fb9c734c5ef5ff8ae7e981fd3 \
  --output-directory /tmp/hegel-rust-m3-shrink4
```

The output directory must not already exist. It is created exclusively and
every file uses create-new semantics. The archive contains length-framed
program records, chunk manifests, all 175 bucket-accounting records, and the
same diagnostic JSON printed to stdout. There are no target, split, seed,
role, formal-root, or network inputs.

The JSON report binds Source O and Evidence P through the exact six
`strict_qualification_*` fields and carries
`maximum_top_level_clauses=2`, `and3_generator_attempts_allowed=false`, and
`and3_raw_operator_application_count=0`. Historical registry state remains
`NOT_RUN`; the Evidence-P PASS is prior admission authority, not M3 execution.

The crate freezes only reduced-budget structural qualification tests. It does
not commit or claim an observed 50,000-program result; full Python/Rust
execution and its commit-bound evidence belong to a later diagnostic stage.

Offline qualification is supervisor-owned and follows
`config/phase3_shrink4_enumerator_offline_build_profile_v1.json`; this README
does not define an alternate hand-written Docker invocation. The supervisor
must use the digest-pinned image with `--pull=never --network=none`, mount only
the exact reverified Cargo `registry/cache` and `registry/index` snapshots
read-only, and exclude every pre-unpacked `registry/src` tree. It then copies
those two seeds into a fresh tmpfs `CARGO_HOME`, where Cargo verifies the
`Cargo.lock` checksums while unpacking dependencies. Source is read-only and
the build target is a fresh local-driver volume (read-write only during build,
read-only during runtime). No later qualification or enumeration step may
contact the network or reuse a mutable Cargo home.
