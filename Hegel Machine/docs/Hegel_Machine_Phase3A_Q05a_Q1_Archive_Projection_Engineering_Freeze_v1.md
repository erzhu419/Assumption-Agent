# Hegel Machine Phase-3A-Q0.5a — Q1 Archive/Resource Projection Engineering Freeze v1

## 0. Status and authority

This document freezes the engineering shape of the target-blind Q1 formal
archive and the resource projection which must be qualified before the full
node-six capacity preflight.  It does not start Q1 and it is not a Q1 closure
result.

```yaml
freeze_id: hegel-phase3a-q05a-q1-archive-projection-freeze-v1
engineering_status: SPECIFICATION_FROZEN_IMPLEMENTATION_NOT_QUALIFIED
dsl_version: hegel-old-dsl-v1.6.0
closure_semantics_version: hegel-quotient-closure-v1.0.1
archive_wire_version: hegel-q1-archive-wire-v1.0.0
projection_freeze_version: hegel-freeze-p3a-q05a-q1-projection-v1.0.0
projection_profile_id: hegel-q1-archive-projection-profile-v1

q1_state: NOT_RUN
q1_gate_count: 0
q1_gate_mask: 0x000000
q1_formal_roots: null
q1_receipt: null
q2_state: NOT_RUN
role_evaluation_performed: false
m3_formal_roots: null
outside_certificate_issued: false
```

The machine-readable companion is
`config/phase3_q1_archive_projection_freeze_v1.json`.

The specification and a local Python prototype now exist. Selected core
modules pass an empty-package-bootstrap smoke test, so the source closure is
isolation-capable; normal in-repository pytest imports are not themselves
runtime-isolation evidence. The prototype is not yet a commit-bound,
filesystem-isolated qualification endpoint. The Rust implementation, trusted
host replay, dual supervisor,
cross-implementation golden evidence, implementation roots and source commit
do not yet exist. Consequently Gate 10 is still unpassed and full node-six
execution remains forbidden.

## 1. Claim boundary

Q0.5a distinguishes three things which must not be conflated:

1. a **projection profile commitment**, which describes how future archive
   bytes and scratch demand will be counted;
2. a **diagnostic projection result**, which may report exact lengths and a
   simulated allocator high-water while Q1 remains `NOT_RUN`;
3. a **formal archive root**, which may only be produced after a separately
   admitted Q1 run.

Neither the projection-profile root nor a projection-result ID is a Q1 formal
output root.  The capacity preflight may compute domain-separated content IDs
needed for canonical ordering, but it may not materialize the production
formal archive or publish its archive roots.

Target truth, split state, role bindings, matcher outputs and synthesis traces
are outside the source and data boundary of all three actors.

## 2. Object tags and strict CBOR

The reserved Q1 range is:

| Tag | Object |
|---:|---|
| `0x3700` | `Q1SemanticBindingManifestV1` |
| `0x3701` | `Q1BehaviorBlobV1` |
| `0x3702` | `Q1ConstructionSignatureV1` |
| `0x3703` | `Q1RepresentativeProgramRecordV1` |
| `0x3704` | `Q1ContinuationCohortRecordV1` |
| `0x3705` | `Q1QuotientClassRecordV1` |
| `0x3706` | `Q1SemanticCoverageRecordV1` |
| `0x3707` | `Q1FixedPointRecordV1` |
| `0x3708` | `Q1ArchiveChunkManifestV1` |
| `0x3709` | `Q1SignatureArchiveManifestV1` |
| `0x370A` | `Q1ClosureBundleV1` |
| `0x370B` | `Q1ArchiveProjectionProfileV1` |
| `0x370C` | `Q1ArchiveProjectionResultV1` |

Every object is a numeric-tag CBOR array. Formal hashed cores reject maps,
text keys, floats, CBOR tags, indefinite encodings and non-minimal integers.
All strings in the schemas below are ASCII byte strings.

### 2.1 Semantic binding

```text
[1, 0x3700, b"hegel-q1-semantic-binding-manifest/1",
 dsl_version, freeze_version, closure_semantics_version,
 child_dsl_root, operator_semantics_root, identifier_registry_root,
 canonical_ast_root, canonical_cbor_root, mdl_profile_id,
 q0_receipt_root, full_v16_leaf_manifest_root,
 [[1, odd_universe_root, 480], [2, sink_universe_root, 85]],
 q1_preregistration_document_sha256,
 post_shrink6_normative_document_sha256]
```

The Q0.5a document and config hashes are deliberately absent from this
preimage. Their committed bytes belong to later source-qualification evidence,
otherwise the profile would contain its own source hash recursively.

### 2.2 Behavior and construction signature

```text
Q1BehaviorBlobV1 =
[1, 0x3701, b"hegel-q1-behavior-blob/1",
 input_signature_id, universe_root, output_sort_id,
 row_count, [typed_cell...]]

bottom cell  = [0]
defined cell = [1, exact_sort_typed_payload]
```

Bool, Bit, Sign, BoundedInt and RationalValue remain type-distinct even if
their apparent numeric payload is equal. RationalValue is an exact reduced
numerator/positive-denominator pair on the frozen value grid.

```text
Q1ConstructionSignatureV1 =
[1, 0x3702, b"hegel-q1-construction-signature/1",
 output_sort_id, ast_depth, ast_node_count,
 scalar_occurrence_count, aggregate_leaf_count,
 distinct_bit_slot_bitmap, scope_clause_count,
 top_level_clause_count, old_law_depth,
 normalization_profile_id, mdl_q32]
```

`old_law_depth` is exactly zero for every admitted v1.6 AST. MDL participates
in dominance. Bool and RationalValue cohorts retain two distinct real
witnesses; the other sorts retain one.

### 2.3 Program, cohort/bank and class records

```text
Q1RepresentativeProgramRecordV1 =
[1, 0x3703, b"hegel-q1-representative-program/1",
 input_signature_id, universe_root, program_index,
 program_id, class_id,
 canonical_ast_cbor, canonical_ast_hash,
 construction_signature_object, construction_signature_id]
```

`program_id` excludes `program_index`; it binds signature, universe, AST bytes
and strict AST hash. The program archive contains exactly the real admitted
representatives in the complete continuation bank, not all syntactically
canonical programs.

```text
Q1ContinuationCohortRecordV1 =
[1, 0x3704, b"hegel-q1-continuation-cohort/1",
 input_signature_id, universe_root, cohort_index,
 cohort_id, class_id,
 construction_signature_object, construction_signature_id,
 witness_capacity, witness_count,
 [[rank, program_id, canonical_ast_hash]...],
 visible_frontier_cohort]
```

Every exact-signature cohort remains in this archive even when dominated.
Ranks are contiguous from zero and representatives are ordered by AST CBOR.

```text
Q1QuotientClassRecordV1 =
[1, 0x3705, b"hegel-q1-quotient-class/1",
 input_signature_id, universe_root, class_index,
 behavior_blob_object, class_id,
 first_cohort_index, cohort_count, class_cohort_subtree_root,
 bank_point_count,
 visible_cohort_count, visible_frontier_point_count,
 visible_frontier_subtree_root, minimum_mdl_q32]
```

Class identity is the exact typed behavior ID. The class record binds both the
complete latent cohort bank and the visible Pareto view. `minimum_mdl_q32` is
recomputed from every exact-signature cohort, not from an arbitrary visible
representative.

## 3. Coverage and fixed point

The full leaf manifest reuses the existing capacity-seed authority
`phase3_q1_capacity_preflight_v1._frozen_leaf_asts_v1`, sorted by
`(output_sort_id, root_operator_id, canonical_ast_cbor)`, and gets coverage
codes `0..809`. Operator codes are:

```text
unary: 0x1000, 0x1001, 0x1002, 0x1003
binary: 0x2001, 0x2002, 0x2003, 0x2005, 0x2006
approx: 0x3001, 0x3002
AND2: 0x4002
```

Each signature has exactly:

```text
810 leaf rows at depth 0
+ 12 operator rows at each of depths 1, 2 and 3
= 846 coverage records
```

That is exactly 846 coverage records per signature, in the canonical order
`(construction_depth, coverage_code)`. The RFC6962 root of the 846 canonical
two-field arrays is
`4ff6ef274f4a1122f286a64a155350dab35c5ebedb198977895df97aa402d9c8`.

```text
Q1SemanticCoverageRecordV1 =
[1, 0x3706, b"hegel-q1-semantic-coverage/1",
 input_signature_id, universe_root, depth, coverage_code,
 eligible_count, eligible_root,
 processed_count, processed_root,
 strict_admitted_count, strict_admission_root,
 unique_canonical_ast_count, rewrite_collapse_count]
```

An eligible application key is:

```text
[1, b"hegel-q1-semantic-application-key/1",
 input_signature_id, universe_root, depth, coverage_code,
 operator_parameters, ordered_child_program_ids]
```

For a leaf row, `depth=0` and `coverage_code` itself is the full-v1.6 leaf
manifest index `0..809`; it is not repeated in another field.
`operator_parameters=()` and `ordered_child_program_ids=()`. The corresponding
strict-admission archive row is exactly
`(application_id, canonical_ast_hash)`, which binds the admitted leaf AST
without putting its CBOR bytes into the application key.

For PASS, eligible and processed counts and RFC6962 roots are equal. The strict
admission root binds `(application_id, canonical_ast_hash)`. This makes the
claim “every eligible bank tuple was processed exactly once” replayable rather
than inferring it from a terminal class count.

The fixed-point `raw_application_count`, `strict_admitted_count` and
`rewrite_collapse_count` are sums over all 846 coverage rows. Thus raw and
strict totals include the 810 leaf admissions; every leaf contributes zero to
rewrite collapse. Historical fields named `raw_operator_application_count`
use this same inclusive counter and must not trigger an erroneous 810-row
subtraction.

Both endpoints use the same depth-barrier direct-full-bank traversal. Therefore
the raw eligible application set and raw count must be exactly equal across
Python and Rust. A raw-count difference is an implementation disagreement, not
a legal scheduling variation. Only the work-queue high-water may use the larger
endpoint value.

```text
Q1FixedPointRecordV1 =
[1, 0x3707, b"hegel-q1-fixed-point/1",
 input_signature_id, universe_root, projection_profile_root,
 raw_count, strict_count, rewrite_count,
 maximum_depth, maximum_nodes, structural_boundary_depth,
 work_queue_empty, zero_delta_full_boundary, all_eligible_covered,
 final_class_delta, final_cohort_delta,
 final_frontier_delta, final_bank_delta,
 program_count, class_count, cohort_count,
 bank_count, frontier_count,
 max_bank_per_class, max_frontier_per_class,
 program_root, bank_root, class_root, coverage_root]
```

Intermediate round mutation histories and queue allocation histories are
diagnostic only and are excluded from semantic state identity.

The fixed-point record fails closed unless all cardinality constraints hold:

```text
raw_application_count >= 810
strict_admitted_count <= raw_application_count
rewrite_collapse_count <= strict_admitted_count
program_count == bank_point_count
1 <= class_count <= cohort_count <= bank_point_count <= 2 * cohort_count
class_count <= frontier_point_count <= bank_point_count
1 <= maximum_frontier_points_per_class
  <= maximum_bank_points_per_class <= bank_point_count
```

## 4. Canonical archives and root DAG

Record ordering is:

```text
program:  (depth, nodes, sort, root operator, AST CBOR)
cohort:   (class ID, construction-signature ID, signature CBOR)
class:    (class ID, behavior CBOR)
coverage: (depth, coverage code)
chunk:    (stream kind, chunk index)
signature partitions: Odd(1), then Sink(2)
```

All record archives use RFC6962 without an extra domain wrapper. Content IDs
and singleton roots use:

```text
SHA256(UTF8(domain) || 0x00 || canonical_cbor(preimage))
```

For every registered formal semantic/content identifier, framed-blob or
chunk-manifest identity, and RFC6962 root, the collision guarantee applies
**only when its complete preimage is present in this local materializer**. Such
same-digest/different-preimage cases fail with
`FAIL_SHA256_PREIMAGE_COLLISION`. This source-freeze does not extend that claim
to identities or roots whose preimages require the still-pending strict
partition/manifest/bundle assembler, nor to implementation-local raw-SHA
diagnostics.

The locally covered formal domains are behavior, construction-signature,
program, program-record, cohort, cohort-record, class-record, application,
coverage-record, framed-blob and chunk-manifest-record identities. The covered
non-domain roles are strict canonical-AST SHA-256 plus the eligible,
processed, strict-admission, chunk-record, full-stream, chunk-manifest,
class-cohort and visible-frontier RFC6962 roots. The machine-readable companion
lists these exact domains and roles; anything absent from those lists remains
outside this local collision qualification.
An exact repeated semantic-identity preimage is not a hash collision: record-set
replay rejects it with `REJECT_Q1_RECORD_SET_DUPLICATE_PROGRAM`,
`REJECT_Q1_RECORD_SET_DUPLICATE_CLASS`, or
`REJECT_Q1_RECORD_SET_DUPLICATE_COHORT`, respectively.

The exact formal content-hash domains are:

```text
HEGEL/Q1/SEMANTIC_BINDING/V1
HEGEL/Q1/BEHAVIOR_ID/V1
HEGEL/Q1/CONSTRUCTION_SIGNATURE_ID/V1
HEGEL/Q1/PROGRAM_ID/V1
HEGEL/Q1/PROGRAM_RECORD_ID/V1
HEGEL/Q1/COHORT_ID/V1
HEGEL/Q1/COHORT_RECORD_ID/V1
HEGEL/Q1/CLASS_RECORD_ID/V1
HEGEL/Q1/APPLICATION_ID/V1
HEGEL/Q1/COVERAGE_RECORD_ID/V1
HEGEL/Q1/FIXED_POINT_RECORD/V1
HEGEL/Q1/FRAMED_BLOB/V1
HEGEL/Q1/CHUNK_MANIFEST_RECORD_ID/V1
HEGEL/Q1/SIGNATURE_SATURATION_STATE/V1
HEGEL/Q1/SIGNATURE_ARCHIVE_MANIFEST/V1
HEGEL/Q1/CLOSURE_BUNDLE/V1
HEGEL/Q1/ARCHIVE_PROJECTION_PROFILE/V1
```

The exact diagnostic projection domains are:

```text
HEGEL/Q1/PREFLIGHT/SORTED_STREAM/V1
HEGEL/Q1/PREFLIGHT/SCRATCH_LEDGER/V1
HEGEL/Q1/PREFLIGHT/EXTERNAL_SORT_PROJECTION/V1
HEGEL/Q1/PREFLIGHT/PROJECTED_STREAM/V1
HEGEL/Q1/PREFLIGHT/SNAPSHOT_RECORD_SET/V1
HEGEL/Q1/PREFLIGHT/PARTITION_STREAM_COMMITMENT/V1
HEGEL/Q1/PREFLIGHT/PARTITION_EXTERNAL_SORT/V1
HEGEL/Q1/PREFLIGHT/PROJECTION_RESULT/V1
```

None is a formal archive root domain. Eligible, processed and
strict-admission sets are RFC6962 archives and do not acquire invented domain
wrappers.

The semantic state preimage is:

```text
[1, b"hegel-q1-signature-saturation-state/1",
 input_signature_id, universe_root,
 semantic_binding_root, projection_profile_root,
 program_archive_root, continuation_bank_archive_root,
 class_archive_root, coverage_archive_root,
 fixed_point_record_root]
```

The per-signature manifest is:

```text
stream descriptor =
[1, b"hegel-q1-stream-descriptor/1",
 stream_kind, record_count, archive_root,
 framed_stream_bytes, chunk_count, chunk_manifest_subtree_root]

Q1SignatureArchiveManifestV1 =
[1, 0x3709, b"hegel-q1-signature-archive-manifest/1",
 input_signature_id, universe_root, universe_row_count,
 semantic_binding_root, projection_profile_root,
 4096, 16777216,
 [program, cohort, class, coverage descriptors],
 fixed_point_record_root, saturation_state_root,
 chunk_manifest_count, chunk_manifest_archive_root]
```

The ordered bundle is:

```text
[1, 0x370A, b"hegel-q1-closure-bundle/1",
 semantic_binding_root, projection_profile_root, 2,
 [[1, odd_universe_root, odd_manifest_root, odd_state_root],
  [2, sink_universe_root, sink_manifest_root, sink_state_root]]]
```

The acyclic DAG is:

```text
behavior → class ID
AST/signature → program record
class/signature/program IDs → cohort-bank record
behavior + per-class cohort/frontier roots → class record
application keys → coverage record
four archive roots → fixed point → signature state
state + transport roots → signature manifest
Odd/Sink manifests → closure bundle
```

Transport chunking is bound by the signature manifest but does not alter class
or saturation-state identity.

## 5. Framing and chunks

Every record stream uses:

```text
u32be(canonical_cbor_length) || canonical_cbor
```

A chunk closes before adding the next record would exceed either:

```text
maximum_records_per_chunk = 4096
maximum_chunk_framed_bytes = 16,777,216
```

Compression is forbidden. A single oversized record fails closed. Each chunk
manifest binds stream/signature, contiguous record range, first/last record
IDs, RFC6962 subtree root, domain-separated framed-blob hash and byte length.
The chunk-manifest stream is not recursively chunked.

The framed-blob hash preimage is the canonical CBOR singleton array
`(framed_blob_bytes,)`, not the raw blob by itself. Under
`HEGEL/Q1/FRAMED_BLOB/V1`, blob `h'0001'` hashes to
`d96c9ac65a7f376c3d2c4b062cafe59e72a0b3b1201a01da5f70ef22f02b1889`.

## 6. Counting/discard projection

The qualified full-node-six projection path will invoke the same object
constructors and strict CBOR encoder as the future formal materializer, then
discard archive bytes after counting them. Run-produced 32-byte roots use a
32-byte placeholder; this changes neither CBOR length nor sort key nor chunk
boundary.

The current Python module is deliberately a bounded node-three materialized
golden/tamper-replay prototype: it retains framed blobs and typed preimages so
adversarial tests can alter and independently replay them. Its commitments are
partial diagnostic test values, never Q1 formal roots or output slots. It is
not the production counting/discard endpoint and cannot satisfy Gate 14 or
authorize full node-six execution. The later qualified endpoint must implement
the frozen streaming counting/discard policy without weakening these replay
checks.

The projected archive payload includes:

- four record streams for each signature;
- one chunk-manifest stream for each signature;
- two fixed-point singleton frames;
- two signature-manifest singleton frames;
- one closure-bundle singleton frame.

The diagnostic projection result is not included in that payload and may not
carry production archive roots. It must report Q1 `NOT_RUN`, `0/20`, formal
roots `null`, and eight output slots `null`.

### 6.1 Exact projection-profile wire

`Q1ArchiveProjectionProfileV1` is the following 42-field canonical array. The
profile does not contain its own root or the SHA-256 of this document/config.
Its `Q1_TAG_REGISTRY` field is exactly the following ordered two-field machine
rows; the three-field CamelCase registry in the JSON is descriptive only and
must never replace this hashed preimage:

```text
(0x3700, b"Q1_SEMANTIC_BINDING_MANIFEST")
(0x3701, b"Q1_BEHAVIOR_BLOB")
(0x3702, b"Q1_CONSTRUCTION_SIGNATURE")
(0x3703, b"Q1_REPRESENTATIVE_PROGRAM_RECORD")
(0x3704, b"Q1_CONTINUATION_COHORT_RECORD")
(0x3705, b"Q1_QUOTIENT_CLASS_RECORD")
(0x3706, b"Q1_SEMANTIC_COVERAGE_RECORD")
(0x3707, b"Q1_FIXED_POINT_RECORD")
(0x3708, b"Q1_ARCHIVE_CHUNK_MANIFEST")
(0x3709, b"Q1_SIGNATURE_ARCHIVE_MANIFEST")
(0x370A, b"Q1_CLOSURE_BUNDLE")
(0x370B, b"Q1_ARCHIVE_PROJECTION_PROFILE")
(0x370C, b"Q1_ARCHIVE_PROJECTION_RESULT")
```

The numeric semantic registries are also exact. Output sorts are
`1=Bool`, `2=Bit`, `3=Sign`, `4=BoundedInt`, and `5=RationalValue`.
Normalization profiles are `0=GENERAL`, `1=ABSOLUTE_ROOT`,
`2=CONST_NEGATIVE_ONE`, `3=CONST_ZERO`, `4=CONST_POSITIVE_ONE`, and
`5=TOP_LEVEL_AND2`. Q1 and Q2 state ID `0` means `NOT_RUN`.

A behavior cell is either `[0]` for bottom or `[1, payload]`. Bool payload is
an exact CBOR Boolean; Bit is an exact non-Boolean integer in `{0,1}`; Sign is
an exact non-Boolean integer in `{-1,0,1}`; BoundedInt is an exact non-Boolean
integer in `[-8,8]`; RationalValue is the canonical reduced
`(numerator, denominator)` pair with positive denominator and a value in the
frozen 663-element Fraction grid.

```text
[1, 0x370B, b"hegel-q1-archive-projection-profile/1",
 b"hegel-q1-archive-wire-v1.0.0",
 b"hegel-freeze-p3a-q05a-q1-projection-v1.0.0",
 b"hegel-q1-archive-projection-profile-v1",
 semantic_binding_root,
 Q1_TAG_REGISTRY_0x3700_through_0x370C,
 expected_coverage_registry_846_rows,
 h'4ff6ef274f4a1122f286a64a155350dab35c5ebedb198977895df97aa402d9c8',
 (1, 2, 3, 4),
 4096, 16777216, 4, 0,
 b"FRAME_U32BE_LENGTH_PLUS_CANONICAL_CBOR",
 b"CHUNK_CLOSE_BEFORE_NEXT_RECORD_EXCEEDS_RECORD_OR_FRAMED_BYTE_LIMIT",
 (b"PROGRAM_U8_DEPTH_U16_NODES_U8_SORT_U16_ROOT_OPERATOR_AST_CBOR",
  b"COHORT_CLASS_ID_SIGNATURE_ID_SIGNATURE_CBOR",
  b"CLASS_ID_BEHAVIOR_CBOR",
  b"COVERAGE_U8_DEPTH_U16_COVERAGE_CODE"),
 1048576, 1048576, 64, 16384,
 268435456, 16, b"HGQ1RUN1", 68, 4,
 4096, 4096, true,
 (1, 2, 3, 4), (1, 2),
 b"STABLE_K_WAY_MERGE_CONTIGUOUS_RUN_INDEX_GROUPS",
 b"SEAL_HASH_REOPEN_VERIFY_THEN_FREE_INPUT_GROUP",
 b"NO_RANDOM_OR_TIME_COMPONENT_IN_RUN_FILE_NAME",
 b"RUN_ROW_U32BE_KEY_LENGTH_KEY_U32BE_RECORD_LENGTH_CANONICAL_RECORD",
 b"SCRATCH_CHARGE_CEIL_FILE_SIZE_TO_4096_PLUS_4096_PER_LIVE_FILE",
 RESOURCE_GUARD_REGISTRY_1_THROUGH_12,
 ORDERED_EIGHT_Q1_OUTPUT_SLOT_NAMES,
 b"DEPTH_BARRIER_DIRECT_FULL_BANK",
 b"RAW_AND_SEMANTIC_COVERAGE_EXACT_EQUAL_WORK_QUEUE_HIGH_WATER_MAX",
 b"COUNTING_DISCARD_USES_IDENTICAL_ENCODER_AND_FIXED_ROOT_PLACEHOLDERS"]
```

The coverage root in this array is recomputed from the 846 rows and must equal
the shown value; accepting an arbitrary 32-byte root is forbidden. The
resource-guard registry is the existing ordered registry
`(1, RAW_OPERATOR_APPLICATIONS)` through `(12, WALL_TIME)`, encoded with ASCII
byte-string names.

### 6.2 Exact diagnostic projection-result wire

Before aggregation, each stream commitment hashes this exact preimage:

```text
ContentHash("HEGEL/Q1/PREFLIGHT/PROJECTED_STREAM/V1",
 [1, b"hegel-q1-projected-record-stream/1",
  input_signature_id, universe_root, stream_kind_id,
  stream_descriptor,
  ordered_chunk_manifest_objects,
  external_sort_projection])
```

Its diagnostic object appends the resulting commitment as the ninth field;
the commitment is excluded from its own preimage. The snapshot-to-wire
conversion replay is separately bound as:

```text
ContentHash("HEGEL/Q1/PREFLIGHT/SNAPSHOT_RECORD_SET/V1",
 [1, b"hegel-q1-snapshot-record-set/1",
  input_signature_id, universe_root,
  ordered_program_records, ordered_cohort_records, ordered_class_records])
```

The latter commits the converted output records only; it does not contain a
source-snapshot identity and therefore cannot by itself prove conversion
provenance or completeness. Qualification must fresh-run the complete capacity
engine and require exact equality of both the immutable snapshot and the
derived record set. It is diagnostic conversion evidence only and never
occupies a Q1 formal output slot.

Each ordered partition row is exactly:

```text
[1, b"hegel-q1-archive-projection-partition-row/1",
 input_signature_id, universe_root,
 raw_application_count, behavior_class_count, cohort_count,
 bank_point_count, frontier_point_count,
 maximum_bank_points_per_class, maximum_frontier_points_per_class,
 peak_work_queue_points, program_record_count, 846,
 projected_record_stream_bytes,
 projected_chunk_manifest_stream_bytes,
 projected_fixed_point_frame_bytes,
 projected_signature_manifest_frame_bytes,
 projected_partition_payload_bytes,
 projected_peak_scratch_bytes,
 (program_stream_commitment, cohort_stream_commitment,
  class_stream_commitment, coverage_stream_commitment),
 diagnostic_stream_commitment,
 (program_external_sort_root, cohort_external_sort_root,
  class_external_sort_root, coverage_external_sort_root),
 external_sort_projection_root]
```

It has exactly 24 fields and fails closed unless:

```text
raw_application_count >= 810
program_record_count == bank_point_count
1 <= behavior_class_count <= cohort_count
  <= bank_point_count <= 2 * cohort_count
behavior_class_count <= frontier_point_count <= bank_point_count
1 <= maximum_frontier_points_per_class
  <= maximum_bank_points_per_class <= bank_point_count
```

Program count equals bank count because each complete-bank witness has exactly
one program record.
`projected_partition_payload_bytes` is the exact sum
of its preceding four payload components. Odd is row 1 and Sink is row 2.

The aggregate stream commitment is:

```text
ContentHash("HEGEL/Q1/PREFLIGHT/PARTITION_STREAM_COMMITMENT/V1",
 [1, b"hegel-q1-partition-stream-commitment/1",
  input_signature_id, universe_root,
  raw_application_count, behavior_class_count, cohort_count,
  bank_point_count, frontier_point_count,
  maximum_bank_points_per_class, maximum_frontier_points_per_class,
  program_record_count, coverage_record_count,
  projected_record_stream_bytes, projected_chunk_manifest_stream_bytes,
  ordered_four_stream_diagnostic_commitments])
```

The aggregate external-sort projection is:

```text
ContentHash("HEGEL/Q1/PREFLIGHT/PARTITION_EXTERNAL_SORT/V1",
 [1, b"hegel-q1-partition-external-sort-projection/1",
  input_signature_id, universe_root,
  ordered_four_external_sort_stream_roots,
  projected_peak_scratch_bytes])
```

Both four-root tuples are ordered program, cohort, class, coverage. In the
future strict assembler, the first tuple is derived from each per-stream
`Q1ProjectedStreamV1.diagnostic_commitment`; the second is derived from each
per-stream `Q1ExternalSortProjectionV1.diagnostic_root`. Each external-sort
projection already embeds its own `sorted_stream_root`; that root does not
occupy a partition-row tuple slot directly.

The current `Q1ProjectionPartitionRowV1`, signature-manifest and closure-bundle
dataclasses are raw canonical-wire shape/local-self-hash validators. Their
32-byte child fields are precommitted inputs, not independently replayed DAG
preimages. Consequently an arbitrary-root constructor test is only a wire
golden and never completion evidence. A strict partition/manifest/bundle DAG
assembler taking the actual four streams, fixed-point record, chunk manifests
and two signature manifests remains an explicit Gate-12/host implementation
item; until it exists, opaque-root rejection must not be claimed.

```text
Q1ArchiveProjectionResultV1 =
[1, 0x370C, b"hegel-q1-archive-projection-result/1",
 projection_profile_root, semantic_binding_root,
 (odd_partition_row, sink_partition_row),
 projected_closure_bundle_frame_bytes,
 projected_archive_payload_bytes_per_endpoint,
 projected_endpoint_total_output_bytes,
 projected_endpoint_peak_scratch_bytes,
 projected_host_replay_output_bytes,
 projected_host_replay_peak_scratch_bytes,
 0, 0, 0, null, (null, null, null, null, null, null, null, null),
 0, false, null, false]
```

The final nine fields are respectively Q1 state, Q1 gate count, gate mask,
formal roots, eight output slots, Q2 state, role-evaluation flag, M3 formal
roots and certificate flag. The diagnostic ID uses
`HEGEL/Q1/PREFLIGHT/PROJECTION_RESULT/V1`; it is not a formal archive root.
The complete result has 21 fields. Its endpoint archive payload equals the two
partition payloads plus the closure-bundle frame; endpoint total output then
adds exactly 1,048,576 bytes. Host replay output is exactly its 1,048,576-byte
metadata reservation and its regular-file scratch high-water is zero. Endpoint
peak scratch is exactly the maximum of the Odd and Sink partition peaks under
the frozen sequential schedule.

## 7. External sort and scratch allocator

The frozen external-sort profile is:

```text
signature order: Odd, Sink
stream order: program, cohort, class, coverage
level-zero run payload maximum: 268,435,456 bytes
merge fan-in: 16
maximum open-file profile: 256
```

Run files have a 68-byte header:

```text
magic[8] = "HGQ1RUN1"
version:u16be
input_signature_id:u16be
stream_kind_id:u16be
merge_level:u16be
run_index:u32be
record_count:u64be
payload_bytes:u64be
payload_sha256[32]
```

Rows are:

```text
u32be(key_length) || key || u32be(record_length) || record
```

The order-preserving key bytes are exact and are not canonical CBOR encodings
of the logical tuple:

```text
program = u8(depth) || u16be(nodes) || u8(sort_id)
          || u16be(root_operator_id) || canonical_ast_cbor
cohort  = class_id[32] || construction_signature_id[32]
          || canonical_signature_cbor
class   = class_id[32] || canonical_behavior_cbor
coverage = u8(construction_depth) || u16be(coverage_code)
```

These encodings preserve the formal tuple order under unsigned bytewise
comparison. Encoding the tuple itself as CBOR and sorting the resulting bytes
is forbidden because CBOR byte-string length prefixes can change the intended
AST byte order.

Consecutive run indices form each merge group. Inputs remain live until the
output is sealed, hashed, reopened and replayed. Only then are the inputs
freed. Random and timestamp-based filenames are forbidden.
The total order is `(key_bytes, record_bytes)` and duplicate key bytes fail
with `REJECT_Q1_SORT_INPUT`. File IDs are exactly
`level-{level:04d}-run-{run_index:08d}`. Run manifests are archived by
increasing merge level and then run index. Every created run emits
`ALLOC, GROW, SEAL`; merged inputs are then freed by increasing child-run index,
and the final run is freed after projection.

Scratch accounting uses exact-integer numeric actions `ALLOC=1`, `GROW=2`,
`SEAL=3`, and `FREE=4`; Boolean aliases are rejected. Two different quantities
are retained after every event. `logical_live_file_bytes` is exactly
`sum(st_size)` over live regular scratch files. `charged_live_scratch_bytes` is
exactly `sum(ceil(st_size / 4096) * 4096 + 4096)`; the second term is the
deterministic per-file metadata reservation. The two high-waters are the maxima
of those respective quantities and must not be renamed or interchanged. Guard
10 is evaluated only against `charged_scratch_high_water_bytes`. Sparse
allocation, host-filesystem block counts and post-failure partial commit are
forbidden. The next allocation which would exceed Guard 10 fails before
acceptance.

The host streams already-sorted endpoint archives and requires no regular-file
scratch; its projected scratch value is exactly zero, while its buffers remain
part of RSS measurement.

The three untagged diagnostic schemas are exact arrays:

```text
ScratchEventV1 =
[1, b"hegel-q1-scratch-event/1",
 sequence, action_id, file_id, prior_size, new_size,
 live_logical_bytes_after, live_charged_bytes_after]

ExternalSortRunManifestV1 =
[1, b"hegel-q1-external-sort-run/1",
 input_signature_id, stream_kind_id, merge_level, run_index,
 record_count, payload_bytes, payload_sha256]

Q1ExternalSortProjectionV1 =
[1, b"hegel-q1-external-sort-projection/1",
 input_signature_id, stream_kind_id, record_count, input_payload_bytes,
 initial_run_count, merge_level_count, final_run_bytes,
 logical_scratch_high_water_bytes, charged_scratch_high_water_bytes,
 sorted_stream_root, run_manifest_archive_root,
 scratch_event_ledger_root, scratch_event_count]
```

The sorted stream, scratch-event ledger and complete per-stream projection use
`HEGEL/Q1/PREFLIGHT/SORTED_STREAM/V1`,
`HEGEL/Q1/PREFLIGHT/SCRATCH_LEDGER/V1` and
`HEGEL/Q1/PREFLIGHT/EXTERNAL_SORT_PROJECTION/V1`, respectively. Run manifests
form an RFC6962 archive without an added wrapper. These are diagnostic
projection commitments and never substitute for program/class/formal archive
roots.

## 8. Deterministic metadata reservation

Per endpoint and for host replay, reserve:

```text
maximum metadata frames = 64
maximum framed bytes per object = 16,384
reservation = 1,048,576 bytes
```

Thus:

```text
projected formal total output
  = exact projected archive payload + 1,048,576
```

The reservation cannot be borrowed by archive payload. Exceeding either bound
requires a new wire version and a complete preflight rerun.

## 9. Three isolated actors

Q0.5a qualification uses three Docker containers:

1. independent Python endpoint;
2. independent Rust endpoint;
3. trusted host replay container.

The host is deliberately **not** a third independent endpoint and is not an
external signer. It independently decodes, recomputes and compares both
endpoint artifacts as the trusted issuer.

All three actors use `--pull=never`, `--network=none`, a read-only root and
committed source snapshot, `cap-drop ALL`, no-new-privileges, no Docker socket,
private IPC and only explicitly frozen scratch/output writable mounts. Python
uses an empty-package bootstrap. Rust uses a prebuilt locked/offline binary.
The host inherits every one of those container controls; endpoint artifacts
are mounted read-only into it. It does not inherit endpoint independence.
Pinned image digests, seccomp digest and the three invocation identities remain
null until projection source-freeze, so Gate 19 is necessarily false today.
Wall time is measured from immediately before Docker create through container
wait/exit, including startup, import and projection rather than only the inner
algorithm call.

Truth, split and role modules must be physically absent from all snapshots;
“not imported during this run” is insufficient. The endpoints cannot exchange
outputs before both terminate.

## 10. Golden qualification

Python and Rust independently generate every golden byte. The host strictly
decodes, re-encodes and recomputes every ID, RFC6962 root, payload length and
scratch ledger. No manually chosen root becomes normative.

The required corpus covers:

- typed bottom and Bool/Bit/Rational distinctions;
- Odd/Sink non-merge;
- the two-witness context/task cohort and AND2 counterexample
  `82018204828300040083000500`;
- dominated latent cohorts and MDL/structural Pareto tradeoffs;
- all 846 coverage rows and application roots;
- chunk boundaries `1`, `4096`, `4097`, exact 16 MiB and plus one;
- merge-run boundaries `1`, `16`, `17`, `256`, `257`;
- shuffled-input canonical invariance;
- allocator ledger/high-water replay;
- root-DAG, collision, rank, sort and single-byte tamper controls;
- target-blind node-three observations:
  Odd `(1048,40,59,110)` and Sink `(1101,28,84,144)` for
  `(raw,class,frontier,bank)`.

Until the Python, Rust and host golden manifests are byte-identical, their
roots remain `null` and no Gate 10 evidence exists.

## 11. Gate 10 predicates

`Q1_ARCHIVE_WIRE_ROOT_DAG_RESOURCE_PROJECTION_AND_GOLDENS_PASS` requires all
twenty predicates:

1. tag/schema/domain registry unique;
2. strict CBOR roundtrip byte identity;
3. typed behavior/bottom/sort codec;
4. construction signature equals the qualified Q0 policy;
5. strict AST/program identity and behavior replay;
6. cohort multiplicity, rank, capacity and latent bank;
7. Pareto, class MDL and per-class roots;
8. exact 846-row coverage registry;
9. eligible/processed counts and roots equal;
10. Python/Rust raw application counts exact-equal;
11. fixed-point zero delta, empty queue and complete coverage;
12. program/bank/class/coverage/state DAG host replay;
13. chunk framing, boundaries, subtree and blob roots;
14. counting/discard lengths equal materialized golden lengths;
15. external sort equals in-memory canonical sort;
16. three-actor allocator ledger and high-water replay;
17. metadata reservation and output formula;
18. adversarial collision and tamper controls fail closed;
19. Python/Rust/host source, runtime and isolation identities bound;
20. projection result remains `NOT_RUN / 0/20 / null`.

Historical pytest counts and the Q0 receipt cannot substitute for any item.
One false predicate leaves the entire gate absent.

## 12. Null output slots and next transition

The ordered future output slots are:

```text
1 odd_signature_archive_manifest_root = null
2 odd_signature_saturation_state_root = null
3 sink_signature_archive_manifest_root = null
4 sink_signature_saturation_state_root = null
5 q1_closure_bundle_root = null
6 q1_dual_replay_agreement_root = null
7 q1_target_blind_access_ledger_root = null
8 q1_completion_receipt_root = null
```

The next authorized action, after the current source-freeze commit, is to
implement the independent Python/Rust projection encoders, host replay,
external-sort/scratch planner, Docker supervisor and golden corpus. Full
node-six capacity preflight remains forbidden until those sources are cleanly
committed, all three implementation roots are non-null, and all twenty Gate 10
predicates replay successfully. Even then Q1 remains `NOT_RUN`; a separate
resource/genesis amendment and explicit start action are still required.
