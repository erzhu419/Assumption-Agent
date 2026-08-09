# Hegel strict canonicalizer — shrink step 4

This standalone Rust crate implements strict admission for
`hegel-old-dsl-v1.4.0` / `hegel-freeze-p2b-p3-v1.4.0`.

The sole DSL delta is the pre-registered structural shrink:
`max_top_level_clauses` changes from 3 to 2. The inherited shrink-3 strict
path remains authoritative for source/formal syntax, typing, sparse operator
IDs and tombstones, normalization, strict CBOR, and noncanonical rejection.
Only a fully accepted normalized parent program reaches the new gate.

Therefore:

- AND normalization still flattens, sorts, and deduplicates first;
- two distinct normalized clauses are accepted;
- three distinct normalized clauses return `REJECT_STRUCTURAL_LIMIT`;
- three raw clauses that deduplicate to two are accepted;
- malformed/type/registry/tombstone/noncanonical errors retain priority;
- every survivor retains exact CBOR bytes, AST hash, operator IDs, node/depth
  metadata, and consequently the unchanged inputs to the frozen MDL identity.

The single-vector CLI supports `--ast-json` and `--decode-cbor-hex` for sealed
dual replay. Every serialized outcome carries the exact field
`maximum_top_level_clauses: 2`. `--golden-replay` checks the shared 22-vector
Python/Rust contract (manifest root
`sha256:f84035e632bf5a655a9ebd636a0cafe7ab1097c45be87d4db944a0012f52aa90`,
outcome root
`sha256:c19341f08ac5f5759c2cdcb3681a37d958de362b81d02c184f7e2413dca18d7c`), and
`--capacity-replay` replays the inherited 2,160-source target-free survivor
subset. Capacity replay remains
`FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE`; it does not start M3, generate
formal roots, or establish closure.

## Offline verification

Use the pinned local Rust image and cache with networking disabled. The
qualified supervisor is responsible for exact mounts; a local host replay is:

```bash
cargo test --release --locked --offline
cargo run --release --locked --offline -- --golden-replay
cargo run --release --locked --offline -- --capacity-replay
```
