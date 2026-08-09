# Hegel strict canonicalizer — shrink step 5

This standalone Rust crate implements strict admission for
`hegel-old-dsl-v1.5.0` / `hegel-freeze-p2b-p3-v1.5.0`.

The sole DSL delta is the pre-registered structural shrink
`max_total_node_count: 7 -> 6`. The machine report field for that bound is
`maximum_ast_node_count: 6`. The inherited shrink-4 limit remains
`maximum_top_level_clauses: 2`.

The complete shrink-4 strict path runs before the new gate. Source/formal
syntax, typing, sparse operator IDs and tombstones, normalization, strict CBOR,
registry rules, and rejection priority are unchanged. A surviving program
retains its exact canonical CBOR, AST hash, sort, operator ID, depth, node count,
and scalar-parameter occurrence count.

The single-vector CLI supports `--ast-json` and `--decode-cbor-hex`. The sealed
`--golden-replay` uses the inherited 22 vector IDs and category layout, with
the step-5 limit vectors replaced by source/formal seven-node programs. The
`--capacity-replay` binds two complete constructive sets: 175 exact programs
at or below six nodes must preserve source/formal parent identity, while all
2,160 inherited shrink-4 AND2 boundary programs have seven nodes and must
return `REJECT_STRUCTURAL_LIMIT` at both source and formal boundaries. These
are target-free controls, not closure enumeration.

The sealed replay commitments are:

- golden manifest: `sha256:156f7e20407437bb753b097a87932f469701d1de6d1d577b0fa1b7a98f47e52e`;
- golden outcomes: `sha256:8f82178c0f33d5295601d2e112b0b6e25ef18d73e5fc35d8d601024c1f0ddf94`;
- 175-program survivor set: `sha256:f5ab7f079ad943d65a74881eb59c7bb46385e1c437ca8ab036bb071dfa3874ac`;
- 2,160-program parent-only set: `sha256:7e0e8780149f03ce85723408f7e3eff2cd684e8938896125cf8e34be9ac70b5e`;
- source rejection outcomes: `sha256:8617b56bdfa347f11f2c68b6a41f0992652f1e23e6d651017b17eb50169a9f39`;
- formal rejection outcomes: `sha256:9a6b489ed90960008aebbecdbcf0bc5cf1595b7a8206d179bbe898540dabf617`.

Every replay remains `NOT_RUN`: no target/split modules, seed, key, signature,
formal root, or M3 state transition is accessed or produced.

## Offline verification

Use the pinned local Rust image and dependency cache with networking disabled.
The qualified supervisor is responsible for exact mounts; a local offline
replay is:

```bash
cargo test --release --locked --offline
cargo run --release --locked --offline -- --golden-replay
cargo run --release --locked --offline -- --capacity-replay
```
