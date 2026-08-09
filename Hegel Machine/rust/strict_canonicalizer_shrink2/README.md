# Hegel strict canonicalizer: shrink step 2

Independent Rust admission layer for `hegel-old-dsl-v1.2.0`. It inherits the
strict AST/CBOR mechanics from the parent implementation and the sparse
AggregateMap registry from shrink step 1. RationalParameterId/v1 remains a
three-bit sparse registry:

- active IDs: `1, 3, 5` (`-1, 0, 1`);
- permanent tombstones: `0, 2, 4, 6`;
- reserved/out-of-range code point: `7`.

Both source and formal admission return
`REJECT_REMOVED_RATIONAL_PARAMETER` for a tombstoned parameter. Numeric IDs
are never compacted or reused.

Source rational aliases require a positive denominator and use exact decimal
integer arithmetic, including JSON integers wider than the host word size.
Source parsing is direct and left-to-right: an unvisited sibling is never
pre-transformed in a way that could change failure priority. Formal admission
performs an exact deterministic-CBOR preflight and validates child subtrees
before later siblings or tolerance fields.
The shared Python/Rust golden profile has 59 vectors and freezes malformed,
registry-range, formal-schema, canonical-shape, and aggregate-before-rational
tombstone priorities in addition to accepted-program identity.

Constant folding is child-admission-aware. A fold is performed only when its
result is active. Consequently `add(1, 1)` and `difference(-1, 1)` remain
operator ASTs instead of manufacturing a removed constant or rejecting the
whole program. Every surviving parent AST retains exactly the same canonical
CBOR bytes and `HEGEL/AST/V1` hash.

The CLI supports source admission, formal-CBOR admission, and a deterministic
built-in golden replay:

```text
hegel-strict-canonicalizer-shrink2 --ast-json JSON [--pretty]
hegel-strict-canonicalizer-shrink2 --decode-cbor-hex HEX [--pretty]
hegel-strict-canonicalizer-shrink2 --capacity-replay [--pretty]
hegel-strict-canonicalizer-shrink2 --golden-replay [--pretty]
```

`--capacity-replay` independently constructs the preregistered Cartesian
subset: 15 constant atoms times 144 one-aggregate mixed atoms, for exactly
2,160 sources. Its report is explicitly `SUBSET_ONLY_NOT_COMPLETE`; it does
not enumerate the shrink-2 closure, create formal roots, or advance an M3
execution state.

The qualification build is locked and offline in
`rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`;
the source mount is read-only and the container uses `network=none` and
`pull=never`.
