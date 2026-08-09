# Rust shrink-5 complete-closure diagnostic

This crate independently enumerates the target-free canonical closure of
`hegel-old-dsl-v1.5.0`. It inherits the frozen shrink-4
depth/sort/operator/CBOR traversal, AND2-only construction, and archive wires.
The sole generator change is that the maximum normalized AST node count drops
from seven to six. No node-seven bucket is constructed or counted. Every
candidate is admitted through the independent Rust shrink-5 strict
canonicalizer.

The executable stops only after it has exposed canonical program 50,001 or
after every typed frontier bucket is closed. Its JSON remains
`NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`, `execution_state=NOT_RUN`, and
`formal_roots=null`; it cannot publish a formal M3 transition or evaluate a
target role. The CLI rejects roots other than its domain-separated
`NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1` tuple.

The three diagnostic root preimages are exact UTF-8 byte strings with one NUL
byte at each `\x00` marker:

```text
HEGEL/M3/SHRINK5/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00hegel-old-dsl-v1.5.0\x00hegel-freeze-p2b-p3-v1.5.0\x00shrink-step5\x00maximum-total-node-count:6\x00maximum-top-level-clauses:2
HEGEL/M3/SHRINK5/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00hegel-old-dsl-v1.5.0\x00hegel-canonical-ast-v1\x00hegel-mdl-prefix-v1.0.0\x00binary-active-formal:1,2,3,5,6\x00binary-tombstones:0\x00binary-source-alias:4\x00maximum-total-node-count:6\x00maximum-top-level-clauses:2
HEGEL/M3/SHRINK5/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00hegel-old-dsl-v1.5.0\x00aggregate-active:0,1,5\x00aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00rational-tombstones:0,2,4,6\x00rational-reserved:7\x00binary-source-active:1,2,3,4,5,6\x00binary-formal-active:1,2,3,5,6\x00binary-source-alias:4\x00binary-tombstones:0\x00binary-reserved:7
```

Their SHA-256 digests are respectively
`3340b3278caf562b560cc30cd14d3cd5f1d628e222b43d29d9d1e41b379f5675`,
`5d2700884ae7125b9712a2bd06aa929feaf2fad1d4bfcd4fa5953c157a720ee1`,
and `1b0c141126b278778009d3ebbbf49f5de231ad0166a88a8a9caf367b35bff8ef`.
These are non-formal diagnostic bindings, not formal child roots.

The unchanged AST and deterministic-CBOR wire profiles are rebound only at
the diagnostic-profile layer. Their exact preimages and SHA-256 values are:

```text
HEGEL/M3/SHRINK5/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array\x00maximum-total-node-count:6\x00maximum-top-level-clauses:2
828fdcc9f16ebd590702ff4297cac6f6ffa19b01299ea7a93753a4fced0961c5
HEGEL/M3/SHRINK5/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00hegel-cbor-det-v1\x00RFC8949-deterministic-no-map-text-float-tag-indefinite
0ccbd740c0b1f6a39fb8151ea56e114561093ee4fccb228bf83a9294e0bae783
```

```bash
cargo run --release --locked --offline -- \
  --enumerate-diagnostic \
  --child-dsl-spec-root 3340b3278caf562b560cc30cd14d3cd5f1d628e222b43d29d9d1e41b379f5675 \
  --operator-semantics-root 5d2700884ae7125b9712a2bd06aa929feaf2fad1d4bfcd4fa5953c157a720ee1 \
  --identifier-registry-root 1b0c141126b278778009d3ebbbf49f5de231ad0166a88a8a9caf367b35bff8ef \
  --output-directory /tmp/hegel-rust-m3-shrink5
```

The output directory must not already exist. It is created exclusively and
every file uses create-new semantics. The archive contains length-framed
program records, chunk manifests, all 150 bucket-accounting records, and the
same diagnostic JSON printed to stdout. There are no target, split, seed,
role, formal-root, or network inputs.

The JSON report binds Source S and Evidence T through the exact six
`strict_qualification_*` fields and carries
`maximum_ast_node_count=6`, `maximum_top_level_clauses=2`,
`and3_generator_attempts_allowed=false`, and
`and3_raw_operator_application_count=0`. Historical registry state remains
`NOT_RUN`; the Evidence-T PASS is prior admission authority, not M3 execution.

The crate freezes only reduced-budget structural qualification tests. It does
not commit or claim an observed 50,000-program result; full Python/Rust
execution and its commit-bound evidence belong to a later diagnostic stage.

Offline qualification is supervisor-owned and follows
`config/phase3_shrink5_enumerator_offline_build_profile_v1.json`; this README
does not define an alternate hand-written Docker invocation. The supervisor
must use the digest-pinned image with `--pull=never --network=none`, mount only
the exact reverified Cargo `registry/cache` and `registry/index` snapshots
read-only, and exclude every pre-unpacked `registry/src` tree. It then copies
those two seeds into a fresh tmpfs `CARGO_HOME`, where Cargo verifies the
`Cargo.lock` checksums while unpacking dependencies. Source is read-only and
the build target is a fresh local-driver volume (read-write only during build,
read-only during runtime). No later qualification or enumeration step may
contact the network or reuse a mutable Cargo home.
