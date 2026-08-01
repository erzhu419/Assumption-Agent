# Hegel Machine M2.5 formal bridge (Rust)

This crate is an independent Rust implementation of the primitive formal-wire
operations frozen for Phase-3A M2.5:

- deterministic CBOR over arrays, integers, byte strings, booleans, and null;
- exact decode/re-encode validation with forbidden CBOR classes rejected;
- domain-separated `ContentHash` and RFC 6962 Merkle roots;
- HKDF-SHA256 role-key derivation, split-row ranking, and seed commitment;
- v1.1.2 `IdDigestV1`, strict `OddInputV1`/`SinkInputV1`, and independent
  generation of the 480/85 universe and truth-row archives.

It deliberately contains no manifest/certificate constructors, M3 state enums,
random-number generation, key generation, signing, or secret persistence. In
particular, this crate cannot instantiate the real split seed or a custodian
key. The typed-row roots are golden qualification outputs, not authoritative
formal-root publication.

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

printf '%s\n' '{"op":"typed_rows","role_id":1}' \
  | cargo run --quiet
```

Supported operations are `encode`, `decode`, `content_hash`, `rfc6962_root`,
`derive_role_key`, `row_rank`, `seed_commitment`, `id_digest`,
`validate_typed_input`, and `typed_rows`. `typed_rows` accepts only role ID 1
(odd) or 2 (sink) and generates every row inside Rust; it never accepts a leaf
archive from Python. Cryptographic inputs are hexadecimal and exact lengths are
checked.
