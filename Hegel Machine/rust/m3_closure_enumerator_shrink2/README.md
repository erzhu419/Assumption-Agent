# Rust shrink-2 complete-closure diagnostic

This crate independently enumerates the target-free canonical closure of
`hegel-old-dsl-v1.2.0`. It replays the frozen depth/node/sort/operator/CBOR
traversal, but admits every candidate through the Rust shrink-2 strict
canonicalizer and generates rational constants only from sparse active IDs
`1`, `3`, and `5`.

The executable stops only after it has exposed canonical program 50,001 or
after every typed frontier bucket is closed. Its JSON is always
`NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`, `execution_state=NOT_RUN`, and
`formal_roots=null`; it cannot publish a formal M3 transition or evaluate a
target role. The CLI rejects roots other than the preregistered
`NON_FORMAL_SYNTHETIC_CHILD_BINDINGS_V1` tuple.

```bash
cargo run --release --locked --offline -- \
  --enumerate-diagnostic \
  --child-dsl-spec-root HEX64 \
  --operator-semantics-root HEX64 \
  --identifier-registry-root HEX64 \
  --output-directory /tmp/hegel-rust-m3-shrink2
```

The output directory must not already exist. It is created exclusively and
every output file uses create-new semantics, so reruns cannot overwrite prior
evidence. It contains length-framed program records, chunk manifests,
bucket-accounting records, and the same diagnostic JSON printed to stdout.
There are no target, split, seed, role, or network inputs.

The crate freezes only small structural qualification vectors. It deliberately
does not commit any observed 50,000-program result: full Python/Rust execution
and its roots belong to a later, commit-bound diagnostic evidence artifact.
