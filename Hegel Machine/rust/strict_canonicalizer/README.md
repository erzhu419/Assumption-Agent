# Hegel Machine strict canonicalizer (Rust)

This crate is the independent Rust side of the
`hegel-freeze-p2b-p3-v1.0.2` strict-acceptance replay. It implements the
project-minimal deterministic CBOR subset, the numeric-tag canonical AST,
source type checking, the frozen rewrite set, structural-limit checks, exact
CBOR re-encode validation, and the domain-separated `HEGEL/AST/V1` program
hash.

It does **not** enumerate the old DSL, evaluate a target, produce formal
Merkle roots, or issue a closure/certificate verdict.

## CLI

From the `Hegel Machine` repository root:

```text
cargo run --manifest-path rust/strict_canonicalizer/Cargo.toml -- \
  --vectors golden_vectors/strict_ast_cbor_v1.json --pretty
```

With no arguments the CLI looks for
`golden_vectors/strict_ast_cbor_v1.json` relative to the current directory,
then relative to this crate. A deterministic one-off source AST can also be
processed:

```text
cargo run --manifest-path rust/strict_canonicalizer/Cargo.toml -- \
  --ast-json '["difference",["scalar_const",1,1],["scalar_const",0,1]]'
```

An already serialized strict AST can be checked with:

```text
cargo run --manifest-path rust/strict_canonicalizer/Cargo.toml -- \
  --decode-cbor-hex '<hex>'
```

The preregistered 64,680-candidate M2 witness subset is generated internally
and replayed with:

```text
cargo run --release --manifest-path rust/strict_canonicalizer/Cargo.toml -- \
  --capacity-replay --pretty
```

The capacity-set commitment is
`SHA256("HEGEL/STRICT_CAPACITY_SET/V1" || 0x00 ||
concat(u64_be(length) || canonical_ast_cbor))` over the deduplicated canonical
AST bytes in lexicographic order. This command deliberately reports executed
closure as `NOT_RUN` and keeps `dsl_too_large_claim_allowed=false`; the Rust
result must first compare equal to the independent Python replay.

The vectors file may be a JSON array, or an object containing `vectors`,
`cases`, or `golden_vectors`. Each case accepts `source_ast` (aliases: `ast`,
`input`) or `input_cbor_hex`. Expected fields may be flat or nested under
`expected`:

- `canonical_cbor_hex`
- `canonical_ast_hash` (raw lowercase hex or `sha256:<hex>`)
- `root_operator_id`
- `node_count`
- `depth`
- `error_code`

The process exits `0` when all supplied expectations match, `1` for a replay
mismatch, and `2` for CLI/fixture I/O errors. The output is a JSON summary;
an accepted case contains canonical CBOR hex and the `sha256:` program hash,
while a rejected case contains a stable rejection code.

## Source JSON vocabulary

Source values use named arrays. Examples:

```json
["scalar_const", 0, 1]
["bit_at", 0]
["bit_to_scalar", ["bit_at", 0]]
["aggregate", "sum_v1", "scope_primary_only_v1", "q0", []]
["difference", ["scalar_const", 1, 1], ["scalar_const", 0, 1]]
["top_level_AND", ["context_flag", "c0"], ["task_flag", "t0"]]
```

Registry values other than an entity slot may use their frozen names or
numeric indices; `bit_at` requires the numeric `0..7` slot index. An aggregate
always carries an explicit scope-extension array, including `[]` for no
extension. A `scalar_const` may use a frozen parameter index or a
`numerator, denominator` pair that reduces onto the frozen grid.
`approx_equal` likewise accepts a tolerance index or a frozen rational pair. The deprecated
`control_volume_primary_only_v1` scope alias is deliberately rejected here;
legacy migration is outside the formal strict canonicalizer.

## Verification status

The checked-in lockfile was verified with `rustc 1.85.1` / `cargo 1.85.1`;
`Cargo.toml` requires Rust 1.85 or newer.  The replay artifact additionally
binds the exact Rust source-set and optimized binary SHA-256 values used in
this run.

The source includes Rust unit tests covering CBOR boundaries/rejections,
typing, all frozen rewrites, canonical ordering, structural accounting, and
fixture replay. Compilation and test execution require `cargo`/`rustc`; a
source tree alone is not evidence that the dual-implementation gate passed.
