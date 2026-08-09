# Hegel strict canonicalizer — shrink step 6

This standalone Rust crate is the independent strict implementation for
`hegel-old-dsl-v1.6.0` / `hegel-freeze-p2b-p3-v1.6.0`.

The only DSL delta from v1.5.0 is the frozen structural shrink
`max_total_ast_depth: 4 -> 3`. The inherited limits remain six total AST nodes
and two top-level clauses. The complete shrink-5 source/formal path runs before
the new gate, so syntax, typing, explicit coercions, normalization, sparse
operator registries and tombstones, strict CBOR, and rejection priorities are
unchanged. Every accepted survivor preserves canonical AST CBOR bytes, AST
hash, behavior-independent structural identity, and exact
`hegel-mdl-prefix-v1.0.0` integer/Q32 length across versions. Every otherwise
valid depth-four source or formal program returns `REJECT_STRUCTURAL_LIMIT`.

The CLI supports `--ast-json`, `--decode-cbor-hex`, `--golden-replay`, and
`--capacity-replay`. Without `--pretty`, it emits exactly one canonical compact
JSON report to stdout and writes no artifacts. The sealed 25-vector replay has
the exact category partition `3/2/3/5/1/3/8`: source survivors,
source-normalization controls, source depth-limit controls, source-priority
controls, formal survivor, formal depth-limit controls, and formal-priority
controls.

The capacity replay is a frozen finite challenge lattice, not closure:

- 1,266 ordered challenge sources in families `A=486`, `B_abs=390`, and
  `B_sign=390` produce 1,249 unique parent programs;
- 1,199 unique depth-four/six-node parent-only programs reject at both source
  and formal shrink-6 boundaries;
- 67 normalization-survivor sources collapse to 50 unique programs;
- 175 inherited survivor sources remain 175 unique programs;
- the combined survivor replay contains 242 accepted sources and 225 unique
  canonical programs with exact parent identity.

The sealed commitments are:

- golden manifest: `sha256:2690413926d15db52dbd5a502ebe3fdfb1dc74d5ee3c82b2ed868cd16ab34a42`;
- golden outcomes: `sha256:e5fd0885f95669dc6d369d0d3274778425fabb7e8c6286a27237a1b2bc8d3960`;
- challenge source lattice: `sha256:a8cfb37278000933c2c51a2797e5bc0f4e7aad6970b37e178fc681f9358574d0`;
- challenge parent canonical set: `sha256:8f125763d3098d087dd7e9eb484b93097295ebd765b6f079795e8009623fb13e`;
- normalized survivor set: `sha256:dcbb5562fc754fdef932188b189dbcdc0f7c500d3fc49651ee4dbb0f271afd29`;
- inherited survivor set: `sha256:477a5abe659a7a7e7d2d50b2a5bda61b0dae1019c44fe84950c4a05036258619`;
- full survivor set: `sha256:6787cd6c0782fda149e1ee93b37ca8d425f5ac78850c610e21cebf9da13a16d1`;
- depth-four parent-only set: `sha256:d3eb2b2d9caf1eece5a709d8113540e4709d579cdfbe3194f1cf176c9100b20d`;
- source rejection outcomes: `sha256:9b0b766a4139db6297aea8b6032ad49147c1a26bf9b56291444a83681428cb0e`;
- formal rejection outcomes: `sha256:97d50c34f51683a2502157961acc79d3b4e108b28bdaa266cf3721ffda8b3a96`.

Every replay remains fail-closed at `NOT_RUN`: it neither imports target/split
roles nor accesses or produces seeds, keys, signatures, formal roots, or an M3
state transition.

## Offline verification

Use the locally pinned Rust image and dependency cache with networking disabled:

```bash
cargo test --release --locked --offline
cargo run --release --locked --offline -- --golden-replay
cargo run --release --locked --offline -- --capacity-replay
```
