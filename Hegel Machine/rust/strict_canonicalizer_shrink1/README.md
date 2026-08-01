# Hegel strict canonicalizer: shrink step 1

Independent Rust admission layer for `hegel-old-dsl-v1.1.0`. It preserves the
parent numeric AST/CBOR implementation and rejects AggregateMapId tombstones
2, 3, and 4 before child type checking or normalization.

The `--capacity-replay` mode constructs exactly 25,872 preregistered source
programs. A successful replay is only a within-budget subset qualification; it
does not mean the full language closure is complete.
