# Phase-3A M2.5 deterministic Cargo transcript amendment v1

## Status and scope

This amendment closes a pre-genesis implementation-qualification defect. It
does not change the DSL, target universe, canonical AST, CBOR schema, hash
domain, gate registry, split, custody wire, formal root DAG, or M3 state
machine.

The affected implementation basis produced only non-formal qualification
artifacts. The live protocol qualification used ephemeral synthetic-protocol
keys and signatures, then destroyed their custody as required. External
genesis was never executed: no real seed, formal actor key, real-seed-bound or
formal signature, marker, formal root, gate promotion, or M3 transition exists
for that basis.

## Observed defect

Two independent pinned-image, offline Cargo builds produced the same qualified
Rust binary and the same enumeration output, but Cargo's successful stderr
included elapsed wall time. The raw stderr digests therefore differed, which
changed the typed qualification CBOR and receipt root. Standalone and live
qualification each validated in isolation, while their combined Commit-B role
set could not satisfy exact receipt equality.

## Frozen correction

The authoritative successful build invocation adds exactly one `--quiet` flag:

```text
cargo ... build --quiet --release --locked --offline --jobs=1 \
  --manifest-path Cargo.toml
```

For return code zero, stdout and stderr must each be exactly zero bytes. Any
non-empty successful stream returns `FAIL_M3_IMPLEMENTATION_BUILD`. The receipt
continues to bind SHA-256 of the raw streams, so both successful build-stream
digests must equal SHA-256 of the empty byte string. No normalization, regex
redaction, elapsed-time deletion, or transcript projection is allowed.

For nonzero return code, the executor still fails closed and preserves the
bounded raw stderr diagnostic. The quiet flag suppresses Cargo's progress log;
it does not convert a failed compiler invocation into success or hide the
failure from the operator.

## Qualification and publication consequence

All qualification artifacts bound to the superseded implementation commit are
pre-publication diagnostics only and must be archived outside the repository.
A new implementation-basis commit and new commit-keyed private state root are
required. Actor, errata, M3, bridge, live-protocol, execution-status, and
readiness artifacts must all be regenerated.

The standalone M3 receipt and the independently generated live-protocol
embedded receipt must then have identical canonical objects, typed CBOR, and
receipt roots; their outer JSON container framing is not compared. Copying or extracting one into
the other's publication slot is expressly not an independent replay. Formal
execution remains forbidden until this equality and all other pre-genesis
guards pass. The allowed endpoint remains `24/24 + NOT_RUN`; the separate
`phase3-m3-start` action is not authorized by this amendment.

The amendment path itself is a required Commit-A qualification input. Formal
readiness compares the standalone receipt object with both the fresh basis and
the archived live-protocol embedded receipt. Formal execute reads the fixed
standalone slot through a symlink-free anchored reader, compares the
same-process live admission, and reopens the fixed slot at the final
pre-irreversible boundary before allocating a formal ceremony run or ledger
ID, reserving a transaction, starting formal actors, generating a real seed,
or creating a formal key. Any mismatch returns
`FAIL_M25_M3_QUALIFICATION_RECEIPT_MISMATCH` and leaves the ceremony unstarted.
