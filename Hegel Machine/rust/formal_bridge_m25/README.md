# Hegel Machine M2.5 formal bridge (Rust)

This crate is an independent Rust implementation of the primitive formal-wire
operations frozen for Phase-3A M2.5:

- deterministic CBOR over arrays, integers, byte strings, booleans, and null;
- exact decode/re-encode validation with forbidden CBOR classes rejected;
- domain-separated `ContentHash` and RFC 6962 Merkle roots;
- HKDF-SHA256 role-key derivation, split-row ranking, and seed commitment.

It deliberately contains no schema constructors, M3 state enums, random-number
generation, key generation, signing, or secret persistence. In particular, this
crate cannot instantiate the real split seed or a custodian key.

## Replay CLI

The binary reads one JSON request from stdin and writes one JSON response. JSON
is only a diagnostic transport: byte strings must use
`{"bytes_hex":"..."}`, and formal CBOR still contains no maps or text.

Examples:

```sh
printf '%s\n' '{"op":"encode","value":[1,12547,{"bytes_hex":"686567656c"},true,null]}' \
  | cargo run --quiet

printf '%s\n' '{"op":"content_hash","domain":"HEGEL/EXAMPLE/V1","value":[1,{"bytes_hex":"00"}]}' \
  | cargo run --quiet

printf '%s\n' '{"op":"decode","cbor_hex":"83014100f6"}' \
  | cargo run --quiet
```

Supported operations are `encode`, `decode`, `content_hash`, `rfc6962_root`,
`derive_role_key`, `row_rank`, and `seed_commitment`. Cryptographic inputs are
hexadecimal and exact lengths are checked.
