# Rust shrink-3 complete-closure diagnostic

This crate independently enumerates the target-free canonical closure of
`hegel-old-dsl-v1.3.0`. It inherits the frozen shrink-2
depth/node/sort/operator/CBOR traversal and archive wires. The sole generator
semantic change is that permanent tombstone `BinaryOperatorId/v1 = 0` (`add`)
is absent, while ID `1` (`difference`) retains its ordered-pair traversal.
Every candidate is admitted through the independent Rust shrink-3 strict
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
HEGEL/M3/SHRINK3/DIAGNOSTIC_BINDING/V1\x00CHILD_DSL_SPEC\x00hegel-old-dsl-v1.3.0\x00hegel-freeze-p2b-p3-v1.3.0\x00shrink-step3
HEGEL/M3/SHRINK3/DIAGNOSTIC_BINDING/V1\x00OPERATOR_SEMANTICS\x00hegel-old-dsl-v1.3.0\x00hegel-canonical-ast-v1\x00hegel-mdl-prefix-v1.0.0\x00binary-active-formal:1,2,3,5,6\x00binary-tombstones:0\x00binary-source-alias:4
HEGEL/M3/SHRINK3/DIAGNOSTIC_BINDING/V1\x00IDENTIFIER_REGISTRY\x00hegel-old-dsl-v1.3.0\x00aggregate-active:0,1,5\x00aggregate-tombstones:2,3,4\x00rational-active:1,3,5\x00rational-tombstones:0,2,4,6\x00rational-reserved:7\x00binary-source-active:1,2,3,4,5,6\x00binary-formal-active:1,2,3,5,6\x00binary-source-alias:4\x00binary-tombstones:0\x00binary-reserved:7
```

Their SHA-256 digests are respectively
`64aaf01392ca89a1ade3a3766d756b53e9b0e7ec6ab4ca2b4fb74ec658490677`,
`e3337cc67974c8fbbfa6d8f89301184c1658a98b80c0a1fac11251ede9aa15f1`,
and `9dd80c452334db8afd9fbb56f1c74f365f63db61ec4c5667bddbb88e57ec05c8`.
These are non-formal diagnostic bindings, not formal child roots.

The unchanged AST and deterministic-CBOR wire profiles are rebound only at
the diagnostic-profile layer. Their exact preimages and SHA-256 values are:

```text
HEGEL/M3/SHRINK3/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_AST_SCHEMA\x00hegel-canonical-ast-v1\x00strict-numeric-tag-cbor-array
bc426bb2dbb519533cfe79a33e5d677d23a56b6f73a4e781f327fc303946f92c
HEGEL/M3/SHRINK3/DIAGNOSTIC_PROFILE/V1\x00CANONICAL_CBOR_PROFILE\x00hegel-cbor-det-v1\x00RFC8949-deterministic-no-map-text-float-tag-indefinite
072d40b7db3b91c283ff401b77766149f9b3ae7283cf8fc865090052100ccc96
```

```bash
cargo run --release --locked --offline -- \
  --enumerate-diagnostic \
  --child-dsl-spec-root 64aaf01392ca89a1ade3a3766d756b53e9b0e7ec6ab4ca2b4fb74ec658490677 \
  --operator-semantics-root e3337cc67974c8fbbfa6d8f89301184c1658a98b80c0a1fac11251ede9aa15f1 \
  --identifier-registry-root 9dd80c452334db8afd9fbb56f1c74f365f63db61ec4c5667bddbb88e57ec05c8 \
  --output-directory /tmp/hegel-rust-m3-shrink3
```

The output directory must not already exist. It is created exclusively and
every file uses create-new semantics. The archive contains length-framed
program records, chunk manifests, all 175 bucket-accounting records, and the
same diagnostic JSON printed to stdout. There are no target, split, seed,
role, formal-root, or network inputs.

The crate freezes only reduced-budget structural qualification tests. It does
not commit or claim an observed 50,000-program result; full Python/Rust
execution and its commit-bound evidence belong to a later diagnostic stage.

An offline test run uses the already cached pinned image and registry. It does
not pull or contact the network:

```bash
docker run --rm --pull never --network none \
  --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -e HOME=/tmp -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/cargo-target \
  -e RUSTC=/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin/rustc \
  -e RUSTDOC=/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin/rustdoc \
  -v /home/erzhu419/.local/state/hegel-machine/rust-cargo-cache/registry:/cargo-home/registry:ro \
  -v "$(pwd)/..:/workspace:ro" \
  -v hegel-shrink3-enumerator-target:/cargo-target:rw \
  -w /workspace/m3_closure_enumerator_shrink3 \
  rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89 \
  /usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin/cargo \
  test --release --locked --offline
```
