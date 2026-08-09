# Rust shrink-6 complete-closure diagnostic

This crate independently enumerates the target-free canonical closure of
`hegel-old-dsl-v1.6.0`. It inherits the frozen shrink-5 typed source surface,
operator registry, node-count bound, top-level-clause bound, deterministic
CBOR wires, traversal, and AND2-only construction. The sole generator change
is that the maximum normalized AST depth drops from four to three. No
depth-four bucket is constructed or counted. Every candidate is admitted
through the independent Rust shrink-6 strict canonicalizer.

The executable stops only after it has exposed canonical program 50,001 or
after every typed frontier bucket is closed. Its JSON remains
`NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`, `execution_state=NOT_RUN`, and
`formal_roots=null`; it cannot publish a formal M3 transition or evaluate a
target role. The CLI rejects roots other than its domain-separated
`NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1` tuple.

The three diagnostic root preimages are exact UTF-8 byte strings with one NUL
byte at each `\x00` marker:

```text
HEGEL/M3/SHRINK6/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00hegel-old-dsl-v1.6.0\x00hegel-freeze-p2b-p3-v1.6.0\x00shrink-step6\x00maximum-total-ast-depth:3\x00maximum-total-node-count:6\x00maximum-top-level-clauses:2
HEGEL/M3/SHRINK6/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00hegel-old-dsl-v1.6.0\x00hegel-canonical-ast-v1\x00hegel-mdl-prefix-v1.0.0\x00binary-active-formal:1,2,3,5,6\x00binary-tombstones:0\x00binary-source-alias:4\x00maximum-total-ast-depth:3\x00maximum-total-node-count:6\x00maximum-top-level-clauses:2
HEGEL/M3/SHRINK6/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00hegel-old-dsl-v1.6.0\x00aggregate-active:0,1,5\x00aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00rational-tombstones:0,2,4,6\x00rational-reserved:7\x00binary-source-active:1,2,3,4,5,6\x00binary-formal-active:1,2,3,5,6\x00binary-source-alias:4\x00binary-tombstones:0\x00binary-reserved:7
```

Their SHA-256 digests are respectively
`da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae`,
`922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03`,
and `64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1`.
These are non-formal diagnostic bindings, not formal child roots.

The AST and deterministic-CBOR wire profiles are rebound only at the
diagnostic-profile layer. Their exact preimages and SHA-256 values are:

```text
HEGEL/M3/SHRINK6/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array\x00maximum-total-ast-depth:3\x00maximum-total-node-count:6\x00maximum-top-level-clauses:2
5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd
HEGEL/M3/SHRINK6/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00hegel-cbor-det-v1\x00RFC8949-deterministic-no-map-text-float-tag-indefinite
ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab
```

```bash
cargo run --release --locked --offline -- \
  --enumerate-diagnostic \
  --child-dsl-spec-root da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae \
  --operator-semantics-root 922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03 \
  --identifier-registry-root 64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1 \
  --output-directory /tmp/hegel-rust-m3-shrink6
```

The output directory must not already exist. It is created exclusively and
every file uses create-new semantics. The archive contains length-framed
program records, chunk manifests, all 120 bucket-accounting records, and the
same diagnostic JSON printed to stdout. There are no target, split, seed,
role, formal-root, or network inputs.

The JSON report binds Source W
`a69bf6d9746e302a07019f122047ac0bc74aa1c1` and Evidence X
`f9218e28740953c9ac15a2ada70a8616e92c378b` through the exact six
`strict_qualification_*` fields. It carries `maximum_ast_depth=3`,
`maximum_ast_node_count=6`, `maximum_top_level_clauses=2`,
`and3_generator_attempts_allowed=false`, and
`and3_raw_operator_application_count=0`. Historical registry state remains
`NOT_RUN`; the Evidence-X PASS is prior engineering admission authority, not
M3 execution.

The crate freezes only reduced-budget structural qualification tests. It does
not commit or claim an observed program count, archive root, witness, closure
verdict, or status. Full Python/Rust execution and its commit-bound evidence
belong to a later diagnostic stage.

Offline qualification is supervisor-owned and follows
`config/phase3_shrink6_enumerator_offline_build_profile_v1.json`; this README
does not define an alternate hand-written Docker invocation. The supervisor
must use the digest-pinned image with `--pull=never --network=none`, mount only
the exact reverified Cargo `registry/cache` and `registry/index` snapshots
read-only, and exclude every pre-unpacked `registry/src` tree. It then copies
those two seeds into a fresh tmpfs `CARGO_HOME`, where Cargo verifies the
`Cargo.lock` checksums while unpacking dependencies. Source is read-only and
the build target is a fresh local-driver volume (read-write only during build,
read-only during runtime). No later qualification or enumeration step may
contact the network or reuse a mutable Cargo home.
