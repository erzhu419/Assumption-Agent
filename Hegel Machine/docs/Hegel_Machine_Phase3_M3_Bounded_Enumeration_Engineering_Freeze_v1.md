# Phase-3 M3 bounded canonical enumeration engineering freeze v1

Status: implementation rule frozen and independently replayable. This document
does not start M3, evaluate either target role, create a secret, or issue a
certificate.

## 1. Scope

The implementation enumerates the target-independent syntactic closure of
`hegel-old-dsl-v1.1.0`. It inherits `hegel-canonical-ast-v1`, the frozen
normalization and resource limits, sparse `AggregateMapId/v1`, and
`hegel-mdl-prefix-v1.0.0` without modification.

Active aggregate IDs are exactly `0, 1, 5`. IDs `2, 3, 4` remain tombstones.
Each of the four scopes and two quantities is crossed with exactly 33 canonical
scope extensions:

- one empty extension;
- eight one-clause extensions;
- twenty-four two-clause extensions.

No fifth scope or removed-map migration is generated.

## 2. Canonical construction frontier

Traversal order is:

```text
(ast_depth,
 ast_node_count,
 output_sort_id,
 root_operator_id,
 canonical_ast_cbor_bytes)
```

Indices are zero-based. A traversal bucket is fully constructed and sorted
before its programs enter the global prefix. The enumerator stops immediately
after the first completely closed bucket that contains ordinal 50,001.
This applies to leaves as well: depth-zero Bool, Bit, Sign, BoundedInt, and
RationalValue buckets are generated separately in formal sort order.

Raw expansion traverses only the canonical source-token surface:

- commutative child tuples are generated once in nondecreasing
  `(SHA256(child canonical-node CBOR), child canonical-node CBOR)` order;
- conjunction atoms are distinct and strictly increasing by canonical-node
  CBOR;
- the deprecated scope alias is never generated;
- source-only `greater_equal` is not generated;
- `approx_equal` tolerance zero is not generated;
- every remaining operator token plus typing-legal canonical child tuple counts
  as one raw application;
- structural rejection, syntactic duplicate, and rewrite collapse are recorded
  separately; type rejection is zero by construction.

The strict canonicalizer remains the acceptance authority. A candidate that
still needs a frozen rewrite is counted as a rewrite collapse and is not added
through that non-normal construction. Its direct normal-form construction is
visited independently in its canonical bucket.

## 3. Bucket accounting

The accounting archive always emits all 175 rows in:

```text
(output_sort_id, ast_depth, ast_node_count)
```

order, including zero rows, for:

```text
output_sort_id = 1..5
ast_depth      = 0..4
ast_node_count = 1..7
```

For `DSL_TOO_LARGE`, `accepted_canonical_programs` and program-index ranges
refer only to archived indices `0..49,999`; the 50,001st program is the
out-of-budget witness and is not inserted into the archive.

## 4. Program MDL and archives

`program_mdl_length_q32` is the exact old-program prefix-code bit length times
`2^32`. No floating-point or logarithmic approximation is involved. Registry
references use the frozen one-based Elias-delta length; aggregate IDs, scope,
quantity, clauses, RationalParameter, and Tolerance use their frozen fixed
codes.

`CanonicalProgramRecordV2` binds the three supplied formal roots. The program
archive is RFC6962 over 50,000 records. Identity chunks contain 4,096 records
except the final chunk. Each blob is:

```text
uint32_be(record_cbor_length) || record_cbor_bytes
```

and uses `SHA256(UTF8("HEGEL/CHUNK_BLOB/V1") || 0x00 || blob)`. Program,
chunk-manifest, and bucket roots are distinct.

## 5. Independent replay golden

With test bindings `11*32`, `22*32`, and `33*32` for child DSL, operator
semantics, and identifier registry roots respectively, independent Python and
Rust implementations agree on:

```text
status                  DSL_TOO_LARGE
raw applications        3,292,439
archived programs       50,000
program archive root    a23151e07f77edcbebe5b7e2e382e1a81b36c6b15c8997899f7f43dcbda874d1
chunk manifest root     98c8deb02a62630f5813717a28c3b9deb5a3845e663b9af5a78fe7f9427f453d
bucket accounting root  5dd13e5d284785dab7fbe3c16fbb1f1bcba3a44466ab0fb258f75f36ee9661ec
witness AST hash        96200a6a131204315ffcd1efd0aa2dcfe2ce665a2c06516461772c9812f0ec71
witness AST CBOR        820184020383010083000103860003050300828201f58203f5
```

The Python reference runtime observed for this replay was approximately 52
seconds on the development WSL host. Runtime is diagnostic and is not part of
the formal identity.

## 6. Claim boundary

`DSL_TOO_LARGE` is available only when both frozen bounds are used, the current
bucket has closed, and ordinal 50,001 exists before 5,000,000 raw applications.
Any raw-cap interruption before that point is `INCONCLUSIVE_BUDGET`. A bare
fixed-budget report is still `FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY` with
`authoritative_claim_allowed=false`; exact budgets alone do not supply the
Commit-A implementation binding or run identity. This pass produces no role output archive,
match set, outside-language verdict, MDL invention certificate, or M3 state
transition by itself.

## 7. Machine-readable Python entrypoint

The package CLI is shown below for development use. The qualified Docker
snapshot invokes the committed direct entrypoint instead, so package
`__init__` and target definitions are neither mounted nor imported.

```text
python3 -m hegel_machine.phase3_m3_bounded_enumerator_cli_v1 \
  --enumerate-prefix \
  --child-dsl-spec-root HEX64 \
  --operator-semantics-root HEX64 \
  --identifier-registry-root HEX64 \
  --output-directory PATH
```

It writes an exclusive new directory containing `report.json` and three
uint32-big-endian length-framed streams: canonical program records, program
chunk manifests, and bucket accounting records. It also prints the identical
summary to stdout. `--binding-material` emits public implementation metadata.
Paired `--diagnostic-canonical-budget` and
`--diagnostic-raw-application-cap` options provide a fast non-formal test
profile; such a report explicitly carries
`claim_level=NON_FORMAL_DIAGNOSTIC_TEST_PROFILE`.
