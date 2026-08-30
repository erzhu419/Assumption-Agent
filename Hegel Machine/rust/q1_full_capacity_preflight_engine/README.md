# Q1 full capacity-preflight Rust engine

This crate contains two distinct target-blind Rust endpoints. The original
capacity-preflight binary remains diagnostic-only and does not start Q1. The
formal archive materializer is a separate full-node6 action that retains the
already frozen record, frame, chunk, manifest, fixed-point, and closure-bundle
bytes. Neither endpoint reads target truth, split, role, or Hegel-invocation
material.

The production CLI admits one action only:

```text
hegel-q1-full-capacity-preflight-engine \
  --action full-node6-capacity-preflight-v1 \
  --capsule-digest sha256:<64-lowercase-hex> \
  --source-commit <40-lowercase-hex> \
  --output-dir <absolute-empty-directory> \
  --scratch-dir <distinct-absolute-empty-directory>
```

It processes input signatures 1 and 2 sequentially and writes one canonical
JSON line to stdout. A successful closed traversal is labelled only
`PREFLIGHT_SATURATED_DIAGNOSTIC_ONLY`; every report keeps Q1 at
`NOT_RUN / 0/20 / null`. A registered guard returns the nonempty measured
signature prefix, leaves coverage and both projections null, and does not
measure a later signature. Aggregate traversal is decided before projection,
so a guard in signature 2 does not create and then discard signature 1 scratch.

The output directory is reserved for exactly five small canonical JSON sidecars:
`diagnostic-summary-v1.json`, `projection-summary-v1.json`,
`coverage-summary-v1.json`, `fixed-point-summary-v1.json`, and
`endpoint-manifest-v1.json`. The manifest binds the other four payloads; none
is a formal archive or formal root. All spill runs and external-sort transient
files are confined to the separate scratch directory. The scratch directory is
empty again before any sidecar is published.

The top-level `actual_preflight_peak_scratch_bytes` is the logical-byte
high-water of all simultaneously live regular spill files across the whole
diagnostic. The shared tracker reconciles each newly flushed run against the
filesystem and requires `actual_preflight_terminal_live_scratch_bytes == 0`
before a success or guard report is returned. These actual diagnostic metrics
remain separate from `projected_peak_formal_scratch_bytes`. Exact diagnostic
output bytes are measured by the host as raw stdout bytes plus the four
manifest row `byte_count` values plus the actual manifest payload length; every
payload includes its trailing LF.

Every saturated partition contains two deliberately separate projections:

- `diagnostic_state_projection` is a cross-language state-agreement KAT. Its
  kind is `DIAGNOSTIC_STATE_AGREEMENT_KAT_NOT_FORMAL_ARCHIVE`, and none of its
  byte counts is a formal-resource claim.
- `formal_wire_projection` is an exact counting/discard replay of the frozen
  v1 PROGRAM/COHORT/CLASS/COVERAGE CBOR records, u32 framing, chunk manifests,
  external-sort projection, and scratch ledger. It retains no framed archive
  blob and is still diagnostic-only; it does not populate a Q1 output slot.

Each partition formal projection separately reports record-stream,
chunk-manifest-stream, fixed-point-frame, and signature-manifest-frame byte
counts. `projected_partition_payload_bytes` is their exact sum. A closed
two-signature report additionally counts the singleton closure-bundle frame;
top-level `projected_formal_archive_payload_bytes` is the two partition
payloads plus that frame, and `projected_formal_total_output_bytes` adds the
frozen 1,048,576-byte endpoint metadata reservation. These top-level formal
resource fields, including the endpoint peak projected scratch value, are null
for any aggregate guard. None is an actual diagnostic-output measurement or a
Q1 formal root.

The library additionally exposes bounded node limits for executable golden
tests. Those limits are not accepted by the production CLI. Candidate
applications are visited in frozen order through a callback and are never
collected into a depth-sized application vector. Each construction depth keeps
an exact application identity consisting of the depth, parameterized operator
code, and ordered canonical child AST bytes. It repeatedly reads the latest
prior-depth continuation-bank snapshot and processes only identities not seen
at that depth. A rewrite that changes a lower-depth bank is retained and makes
its newly eligible tuples visible to the next bounded saturation batch; an
empty final scan does not consume a saturation round. Unary and ordered binary
children use the frozen global program order; commutative children use the
canonical-node hash order, with the same per-pair operator interleaving as the
Python endpoint. Projection-only coverage replay sorts application keys,
strict-admission rows, and canonical AST rows in bounded 1 MiB runs, merges
them with an online RFC6962 frontier, and removes its transient scratch
directory before sidecars are written. Every nonempty final buffer is flushed
as a tracked run even when it is smaller than 1 MiB; only an empty stream takes
the no-scratch empty-root path. Final frozen-v1 records are encoded through
counting/discard: record counts, framed byte
counts, chunk and stream commitments, and projected scratch high-water are
retained, while formal archive payloads are not written.

## Formal archive materializer

The distinct formal CLI admits only this action and requires the nonnull run
genesis plus the 14/20 prestart admission receipt that authorized the formal
boundary:

```text
hegel-q1-formal-archive-materializer \
  --action materialize-full-node6-formal-archive-v1 \
  --capsule-digest sha256:<64-lowercase-hex> \
  --source-commit <40-lowercase-hex> \
  --run-genesis-root sha256:<64-lowercase-hex> \
  --prestart-admission-receipt-root sha256:<64-lowercase-hex> \
  --output-dir <absolute-empty-directory> \
  --scratch-dir <distinct-absolute-empty-directory>
```

Its implementation identity is
`hegel-rust-q1-formal-archive-materializer-v1`. The result binds the admission
snapshot as `NOT_RUN / 14/20 / mask 0x003fff / eight null output slots`; it
does not admit the diagnostic `0/20` action. One invocation performs both
ordered Odd/Sink measurements and retains the formal archive in this layout:

```text
closure-bundle.framed
endpoint-result.json
signature-01/
  chunk-manifests.framed
  fixed-point.framed
  signature-manifest.framed
  stream-01/chunk-000000.framed ... stream-04/chunk-NNNNNN.framed
signature-02/
  chunk-manifests.framed
  fixed-point.framed
  signature-manifest.framed
  stream-01/chunk-000000.framed ... stream-04/chunk-NNNNNN.framed
```

Every `.framed` payload is the frozen four-byte big-endian length followed by
the existing strict-CBOR record. Chunk files are bound by the already frozen
chunk-manifest framed-blob hash; the materializer adds no second file-hash
scheme. The library's bounded-node3 materialization entrypoint exists only for
golden tests and is not admitted by either production CLI.

Build and test from this directory with the sealed local Cargo cache:

```text
cargo test --locked --offline
cargo build --release --locked --offline
```
