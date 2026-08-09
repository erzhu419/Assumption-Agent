# Hegel strict canonicalizer — shrink step 3

This standalone Rust crate implements strict admission for
`hegel-old-dsl-v1.3.0` / `hegel-freeze-p2b-p3-v1.3.0`.

The only new DSL delta is sparse removal of `BinaryOperatorId/v1` ID 0
(`add`). ID 1 (`difference`) remains active and keeps its existing canonical
AST CBOR bytes, AST hash, root-operator ID, and MDL identity. Removed Add is
still parsed and returns the exact error `REJECT_REMOVED_BINARY_OPERATOR`; it
is never treated as unknown, renumbered, or migrated to Difference.

## Acceptance order

Both source and formal paths validate syntax, registry bounds, and types
before applying these full-tree sparse-admission passes:

1. inherited AggregateMap tombstones;
2. inherited RationalParameter tombstones;
3. the Add tombstone;
4. normalization, canonical-form checks, and whole-AST limits.

The source parser and the formal numeric-CBOR parser are independently owned
by this crate. In particular, formal input is not sent to the shrink-2 decoder
before Add inspection, so foldable and nested Add programs cannot be reported
as merely noncanonical. Source parsing likewise does not normalize before the
new tombstone pass, so an oversized source containing Add is still rejected as
removed after its syntax and types have been validated.

## Replays

The single-vector CLI emits the exact boundary (`SOURCE_JSON` or
`FORMAL_CBOR`) and `target_or_split_modules_loaded=false` on both accepted and
rejected outcomes. These fields let the external sealed supervisor reject a
boundary substitution or target/split visibility claim before normalizing an
outcome.

`--golden-replay` runs the shared Python/Rust 36-vector acceptance profile:

- 8 surviving byte/hash identity checks;
- 4 source Add rejections;
- 6 source priority checks, including rejection of a numeric source operator
  token before inspecting its children;
- 3 formal Add rejections;
- 6 formal priority checks;
- 6 formal shape-priority checks for invalid AND arity and aggregate scope
  extensions that must reject before hidden tombstones;
- 3 formal alias/reserved/registry checks, including out-of-range
  `BinaryOperatorId/v1` ID 8.

`--capacity-replay` reconstructs the inherited add-free 2,160-source,
target-free subset and passes every source through shrink-3 admission. Its
accepted-set commitment must remain:

```text
sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e
```

Success is explicitly `SURVIVOR_SUBSET_ONLY_NOT_COMPLETE`; it is not a closure
claim and leaves execution at `NOT_RUN` with formal roots null.

## Offline verification

The qualified environment uses the pinned local image
`rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`,
a read-only host Cargo registry, a Docker target volume, and both
`--network none` and Cargo `--offline`:

```bash
docker volume create hegel-shrink3-strict-target
docker run --rm --pull never --network none \
  --cap-drop ALL --security-opt no-new-privileges \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -e HOME=/tmp \
  -e CARGO_HOME=/cargo-home \
  -e CARGO_TARGET_DIR=/cargo-target \
  -e RUSTC=/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin/rustc \
  -e RUSTDOC=/usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin/rustdoc \
  -v /home/erzhu419/.local/state/hegel-machine/rust-cargo-cache/registry:/cargo-home/registry:ro \
  -v "$(pwd)/..:/workspace:ro" \
  -v hegel-shrink3-strict-target:/cargo-target:rw \
  -w /workspace/strict_canonicalizer_shrink3 \
  rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89 \
  /usr/local/rustup/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin/cargo \
  test --release --locked --offline
```

The same command prefix can run:

```text
cargo run --release --locked --offline -- --golden-replay
cargo run --release --locked --offline -- --capacity-replay
```
