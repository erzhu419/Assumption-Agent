# Hegel Machine Phase-3A Q0.5b Node3 Dual Projection Qualification — Engineering Freeze v1

Status: `ACTUAL_IMPLEMENTED_CONDITIONALLY_ADMITTED_NOT_EXECUTED`

This document freezes the machine wire needed to qualify the target-blind,
bounded-node3 Python/Rust/host projection path. It does **not** start Q1, run
the full node6 closure, create a formal Q1 fixed point, populate a formal
output slot, evaluate a role, or issue an outside-language certificate.

The distinction that must survive every implementation and report is:

| track | current count | total | state | roots |
|---|---:|---:|---|---|
| Q0.5b qualification predicates | `0` | `20` | actual evidence predicates `1..20` pending | candidate/final receipts `null` |
| formal Q1 execution gates | `0` | `20` | `NOT_RUN` | all eight output slots `null` |

Even a future Q0.5b `20/20` receipt is a non-Q1 qualification receipt. It
does not change Q1 from `0/20 / NOT_RUN / eight null output roots`.

## 1. Authority boundary

The exact closed authority array is:

```text
[
  q1_state_id=0,
  q1_gate_count=0,
  q1_gate_mask=0,
  q1_gate_total=20,
  q1_output_slot_count=8,
  [[1,name_1,null], ... [8,name_8,null]],
  q1_receipt=null,
  q2_state_id=0,
  m3_formal_roots=null,
  formal_fixed_point_claimed=false,
  formal_fixed_point_tag=null,
  target_truth_accessed=false,
  split_accessed=false,
  role_evaluation_performed=false,
  outside_certificate_issued=false,
  active_transition_allowed=false
]
```

No Q0.5b object may substitute the formal `Q1FixedPointRecordV1` tag
`0x3707`. The bounded terminal state has its own qualification-only tag.

## 2. Qualification-only numeric tags

The ordered registry is exact and disjoint from formal Q1:

| decimal | hex | object | schema | fields |
|---:|---:|---|---|---:|
| 14848 | `0x3A00` | `Q05BFullLeafManifestRowV1` | `hegel-q05b-full-leaf-row/1` | 8 |
| 14849 | `0x3A01` | `Q05BFullLeafManifestV1` | `hegel-q05b-full-leaf-manifest/1` | 8 |
| 14850 | `0x3A02` | `Q05BNode3PartitionEvidenceV1` | `hegel-q05b-node3-partition-evidence/1` | 10 |
| 14851 | `0x3A03` | `Q05BSidecarManifestV1` | `hegel-q05b-sidecar-manifest/1` | 5 |
| 14852 | `0x3A04` | `Q05BNode3GoldenManifestV1` | `hegel-q05b-node3-golden-manifest/1` | 21 |
| 14853 | `0x3A05` | `Q05BQualificationCandidateReceiptV1` | `hegel-q05b-qualification-candidate-receipt/1` | 25 |
| 14854 | `0x3A06` | `Q05BQualificationReceiptV1` | `hegel-q05b-qualification-receipt/1` | 12 |
| 14855 | `0x3A07` | `Q05BBoundedNode3StateV1` | `hegel-q05b-bounded-node3-state/1` | 26 |

Formal Q1 continues to own exactly `0x3700..0x370C`. The qualification
registry must never be appended to `Q1_TAG_REGISTRY`.

```text
qualification_tag_registry_root =
  ContentHash("HEGEL/Q05B/QUALIFICATION/TAG_REGISTRY/V1",
              ordered_0x3A00_through_0x3A07_rows)
  = 7daf75e861dacd3f3bda5ba6a0f7952e82b0109009bf306b23ba5db346c51d10
```

## 3. Full 810-leaf identity

Each `0x3A00` row is:

```text
[1, 0x3A00, "hegel-q05b-full-leaf-row/1",
 leaf_index, output_sort_id, root_operator_id,
 canonical_ast_cbor, canonical_ast_hash]
```

Rows are ordered by `(output_sort_id, root_operator_id, canonical_ast_cbor)`;
`leaf_index` is exactly `0..809`. Every AST is strict-decoded, must have depth
zero and node count one, and its digest/sort/operator fields are replayed.

The identity root is deliberately narrow:

```text
full_v16_leaf_manifest_root =
  RFC6962_ROOT(ordered 810 Q05BFullLeafManifestRowV1 canonical objects)
  = 3fefacd3db59294f2b6d44a5d0b813e73af3ec84742a24ab846bbdacae6c1f1b
```

It does not contain Q0 roots or the Q0 receipt. The `0x3A01` sidecar wrapper
may carry DSL version, freeze version, row count, row root, and all rows; its
current canonical length is 70,244 bytes. The formal `0x3700` semantic
binding consumes the RFC6962 leaf root and separately binds semantic
provenance.

## 4. Formal input roots, not formal output roots

The `0x3700 Q1SemanticBindingManifestV1` input binds:

- DSL `hegel-old-dsl-v1.6.0` and freeze `hegel-freeze-p2b-p3-v1.6.0`;
- closure semantics `hegel-quotient-closure-v1.0.1`;
- child DSL, operator, identifier, AST and CBOR roots inherited from the
  sealed Q0/Q1 preregistration sources;
- MDL profile `hegel-mdl-prefix-v1.0.0`;
- Q0 receipt `ee198614...8ad2`;
- the narrow 810-leaf RFC6962 root;
- preregistration document SHA-256 `2fbbba86...15f42`;
- post-shrink6 decision SHA-256 `1df8d3ff...3b95`.

The deterministic roots are:

```text
q1_semantic_binding_root =
  e3b3df3e81b7632c7c713ef5ec84913f990ad8232a25b851f20c46ac7416bfcb

q1_projection_profile_root =
  aa441cdc49ab60324483b9aa44e9fdfc324a6ad49a6bff50af6daa775209816d
```

These are prospective Q1 run inputs. They are not any of the eight Q1 run
output roots and do not imply that Q1 has started.

## 5. Exact bounded-node3 state

The bounded scope is exactly:

```text
maximum_ast_depth        = 3
maximum_ast_node_count   = 3
structural_boundary_depth = 4
terminal_status = LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED
```

The `0x3A07` object binds the exact limits, terminal status, empty bounded
work queue, zero depth-three delta, zero structural-boundary delta, complete
846-row bounded coverage, eligible/processed equality, primary counts,
resource high-water values, coverage-record root, partition-evidence root,
`formal_fixed_point_claimed=false`, `formal_fixed_point_tag=null`, and the
closed authority array.

Exact goldens:

| signature | universe rows | raw | strict | rewrites | classes | cohorts | bank | frontier | max bank/class | max frontier/class | peak work | rounds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| odd 1 | 480 | 1048 | 1048 | 22 | 40 | 86 | 110 | 59 | 25 | 5 | 810 | 5 |
| sink 2 | 85 | 1101 | 1101 | 26 | 28 | 112 | 144 | 84 | 40 | 20 | 810 | 5 |

```text
odd coverage-record root = fe056f876706971d5bd15959325e4fdcc164a5ad66303d0a61bfbff88fae929c
odd partition evidence   = 99357fc3a5f48e8a63e6a87f4b182153c5cdae52bd911676f7b2ecc1058aa097
odd bounded state root   = a7460841bcd36797fa9d5d9987fafe5b5efd91f96e4e49b73a78c6406a20db37

sink coverage-record root = bdc7821d3a96087c1f0b97d7d6e0317e953d00dbf748d2e774eb3040ff0dc6ea
sink partition evidence   = 51d017cd9d7e452198d9d12c53e16728c1e220e56d47f43ce3954c4e92c9ef67
sink bounded state root   = 1788df25b4cd6b8830db28d8622e2fe146f3a3c454404e5e7eafe51315acab8f
```

These bounded zero deltas are not a depth3/node6 language fixed point.

## 6. Partition evidence, external-sort trace and counting/discard evidence

The exact `0x3A02` partition evidence object is:

```text
[1, 0x3A02, "hegel-q05b-node3-partition-evidence/1",
 input_signature_id,
 universe_root,
 snapshot_record_set_canonical_object,
 846,
 [[coverage_record_object,
   eligible_application_keys,
   processed_application_keys,
   strict_admission_preimages] x 846],
 4,
 [[stream_kind_id,
   projected_stream_canonical_object,
   framed_blobs,
   external_sort_trace_object,
   counting_discard_stream_object] x 4]]
```

Its domain is `HEGEL/Q05B/NODE3/PARTITION_EVIDENCE/V1`.

The shared external-sort trace is explicitly registered as the exact
six-field array:

```text
[1, "hegel-q1-external-sort-trace/1",
 external_sort_projection_object,
 ordered_key_record_rows,
 ordered_run_manifest_objects,
 ordered_scratch_event_objects]
```

Run manifests and scratch events are preimages, not only terminal roots. Host
replay must reconstruct run shape, merge order, file lifecycle, logical and
charged high-water values, release every live scratch file, and compare the
projection object exactly.

The nested counting/discard object is the exact 15-field array:

```text
[1, "hegel-q05b-counting-discard-record-stream/1",
 input_signature_id, universe_root, stream_kind_id,
 record_count, canonical_record_payload_bytes,
 framed_stream_bytes, chunk_count,
 descriptor_object, chunk_manifest_objects,
 external_sort_projection_object, diagnostic_commitment,
 retained_framed_blob_count=0, retained_framed_blob_bytes=0]
```

The source-level counting sink independently re-encodes the ordered formal
records while retaining no framed blobs. The `0x3A02` constructor and
candidate decoder require its record/framed/chunk counters, descriptor,
chunk manifests, external-sort projection and diagnostic commitment to equal
the materialized projection, with an exact `(0,0)` retained tail. This freezes
the Predicate 14 capability in source; it does not claim an isolated
Predicate 14 pass.

The present decoder validates strict CBOR, coverage preimage roots, exact
five-field stream-row bindings and the materialized/counting equality fields.
Full semantic replay of all formal records, framed blobs, run/scratch traces,
partition/manifest assembly and tamper behavior remains host evidence for
Predicates 12, 15 and 18. The sidecar therefore remains a raw qualification
candidate until those predicates are independently evidenced.

## 7. Deterministic sidecar layout

The actor receives an explicit empty output directory. It writes exactly:

```text
preimages/000-full-v16-leaf-manifest-v1.cbor
preimages/001-odd-node3-partition-evidence-v1.cbor
preimages/002-sink-node3-partition-evidence-v1.cbor
neutral/q05b-node3-sidecar-manifest-v1.cbor
neutral/q05b-node3-golden-manifest-v1.cbor
```

All final files have mode `0444`. Each untagged sidecar file row is:

```text
[file_index, relative_path, content_kind_id, mode=0444,
 raw_length, raw_sha256, content_hash_domain_ascii, content_root]
```

The `0x3A03` manifest contains only the three preimage rows. It excludes
itself and the `0x3A04` neutral manifest to avoid a self-hash cycle. Actors
must use exclusive no-follow creation in the caller-provided empty directory;
the host opens the completed tree read-only only after both actors exit.

The only implementation-neutral comparison object is the exact `0x3A04`
canonical CBOR, with the sidecar manifest and all sidecar raw hashes/lengths
bound below it. Actor/source/runtime identities remain outside it.

The independently replayed B0.1 candidate bytes are:

```text
odd 0x3A02:  1,244,549 bytes
  raw SHA-256 = 0b2b41acce572e05cd2f201f78a5911782b1559ed31c68625eef984bbf4b39de
  content root = 99357fc3a5f48e8a63e6a87f4b182153c5cdae52bd911676f7b2ecc1058aa097
sink 0x3A02: 1,078,063 bytes
  raw SHA-256 = 2d708648b948ac984a7632c06a71d88a6d03388ee00373c6abaf47ef8bff8756
  content root = 51d017cd9d7e452198d9d12c53e16728c1e220e56d47f43ce3954c4e92c9ef67
0x3A03: 552 bytes
  raw SHA-256 = 318b8fb9e9ba3ce881057742d59bf43314c89891cbc37e4824349ac3f72d4ba3
  content root = 1d68a6fe330f3bfe581ef37933f64d2258e1043079dae15c85607836d99ea59d
0x3A04: 4,134 bytes
  raw SHA-256 = 7fd529708a068e2fa1a8d17f5cc81a41420db944120f4f1591f73e1c67f4cc05
  content root = cbc22f6a9dc91589f77aa1564eb40d688c45ee3aa6af5a66d777ffe08a086b15
```

These are target-blind deterministic candidate roots, not qualification
receipt roots and not formal Q1 output roots.

## 8. Actor envelope

Successful stdout is one canonical JSON line plus one final LF. Required
identity fields include:

```text
schema_version = hegel-q05b-actor-envelope/1
action_id      = bounded-node3-golden-v1
status         = BOUNDED_NODE3_CANDIDATE_EMITTED_NOT_QUALIFIED
file_count     = 5
```

Exact actor/implementation pairs are:

```text
PYTHON_ENDPOINT     / HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1
RUST_ENDPOINT       / HEGEL_Q1_BOUNDED_NODE3_PROJECTION_RUST_V1
TRUSTED_HOST_REPLAY / HEGEL_Q1_BOUNDED_NODE3_PROJECTION_HOST_REPLAY_V1
```

The envelope binds actor-specific source and runtime SHA-256 identities plus
neutral/sidecar relative paths, lengths, raw SHA-256 values and content roots.
It repeats `q1_state=NOT_RUN`, gate count/mask zero, `q1_formal_roots=null`,
and eight null output slots.

## 9. Chunk boundary

No large allocation is needed to test the exact boundary:

```text
raw CBOR bstr payload  = 16,777,207
CBOR bstr bytes        = 16,777,212
u32 frame bytes        = 16,777,216  -> accepted

raw payload + 1        = 16,777,208
u32 frame bytes        = 16,777,217  -> reject/close before record
```

Records are never split across chunks.

## 10. Two-stage non-Q1 qualification receipt

The candidate `0x3A05` binds all of the following before predicate rows:

- raw 20-byte Git SHA-1 source-freeze commit wire;
- all version rows;
- qualification tag-registry, predicate-registry and wire-profile roots;
- formal Q1 semantic-binding and projection-profile input roots;
- Q0 receipt and narrow leaf root;
- three actor-specific implementation roots;
- three exactly equal Python/Rust/host neutral-manifest roots;
- odd/sink bounded-state roots;
- bundle, isolation and resource evidence roots;
- the domain-separated pre-receipt evidence root and pre-receipt root.

The candidate has exactly 19 rows:

```text
[predicate_id:int-not-bool,
 predicate_name:bytes,
 passed:true-as-exact-bool,
 evidence_root:32-bytes]

passed_count = 19
mask         = 0x7FFFF
```

The pre-receipt evidence root is exactly:

```text
ContentHash("HEGEL/Q05B/QUALIFICATION/PRE_RECEIPT_EVIDENCE/V1",
            [source_commit_wire, 19, 0x7FFFF, ordered_19_rows])
```

The final `0x3A06` strict-replays the entire candidate. Predicate 20 is:

```text
[20,
 "CANDIDATE_RECEIPT_VALIDATED_WHILE_Q1_REMAINS_NOT_RUN",
 true,
 ContentHash("HEGEL/Q05B/QUALIFICATION/PREDICATE20_EVIDENCE/V1",
             [candidate_receipt_root, true, closed_authority_array])]
```

Only then are the qualification count and mask `20` and `0xFFFFF`. The Q1
gate count/mask remain `0` and `0`, with Q1 `NOT_RUN`.

## 11. Predicate registry and pending actual evidence

The 20 predicate names are frozen in
`config/phase3_q05b_node3_dual_projection_qualification_v1.json`. The actual
supervisor entrypoint, strict replay path, and atomic publication path are
implemented, so the implementation-blocker registry is empty. This is not a
predicate-pass claim: at the source-freeze commit, actual execution has not
occurred and evidence for predicates `1..20` remains pending.

An actual attempt is conditionally admitted only when one explicit full
40-hex requested commit equals a completely clean `HEAD`, the raw Commit-A
configuration matches the runtime configuration, pinned images and sealed
source/Cargo/runtime inputs replay, and the frozen artifact target is absent.
The supervisor creates the private work root, nonce, admission boundary, and
single-use markers; callers cannot provide them. Any failed runtime
precondition produces no qualification receipt or artifact.

The shared Python wire, isolated Python entrypoint, independent Rust snapshot
validator, both Q0.5b configurations, and actual-admission replayer bind the
same exact Commit-A precondition object. Its Docker ownership additions are
four exact booleans, all `true`:

```text
attempt_unique_docker_execution_authority_required
initial_and_precreate_name_absence_required
docker_cleanup_owned_cid_only_required
foreign_or_unknown_docker_state_zero_mutation_required
```

They add execution-ownership prerequisites only. They do not change the
source-freeze authority: predicates `1..20` remain pending, Q1 remains
`NOT_RUN` at `0/20`, and all eight Q1 output slots remain `null`.

### 11.1 Attempt-unique Docker ownership and cleanup boundary

Before Stage 1, the supervisor samples the same 32-byte attempt nonce that a
later admission decision must spend. It does not accept a nonce from the CLI.
The Docker ownership namespace is

```text
SHA256("HEGEL/Q05B/DOCKER/OWNERSHIP_NAMESPACE/V1" || NUL ||
       attempt_nonce_32 || source_commit_ascii_40)
```

and the five ordered slots are `RUST_TEST`, `RUST_RELEASE`,
`PYTHON_ENDPOINT`, `RUST_ENDPOINT`, and `TRUSTED_HOST_REPLAY`. Their names are
`hegel-q05b-{full64_namespace}-{slot_suffix}`; a fixed cross-attempt container
name is forbidden. Every launch command carries exactly the three reserved
labels `org.hegel.q05b.execution_namespace`, `org.hegel.q05b.slot`, and
`org.hegel.q05b.source_commit`, in that order. Live/post-exit inspect evidence
checks `Config.Labels` against the pinned image's base labels **union** those
three labels. The pinned Rust image contributes
`org.opencontainers.image.source=https://github.com/rust-lang/docker-rust`;
the pinned Python image contributes no base label. The validator therefore
does not incorrectly require `Config.Labels` to contain only three keys.

Stage 1 contains two target-bound authoritative-not-found samples for every
one of the five unique names. Immediately before each corresponding Docker
creation, two new samples are collected under the distinct
`HEGEL/Q05B/DOCKER/PRECREATE_ABSENCE/V1` domain. An inspect transport error,
daemon error, foreign collision, mismatched ownership label, or otherwise
unknown state is not absence: it fails closed with zero destructive Docker
mutation.

Container names are read-only discovery targets only. A name is never passed
to `docker kill` or `docker rm`. Destructive cleanup may target only a full
64-lowercase-hex container ID whose name, namespace/slot/source labels, pinned
image, and exact command were ownership-validated and then re-inspected by
that ID. Name replacement after discovery therefore cannot redirect cleanup
to the replacement. A missing or unprovable ID leaves a reported residual
and fails the qualification; it does not authorize name-based deletion.

This is an application-level accidental/concurrent-run ownership boundary,
not a privilege boundary against a malicious peer that has the same UID and
access to the Docker socket. Such a peer can inspect or forge labels and
already has daemon-wide mutation authority. Covering that actor requires an
independent OS UID and Docker-socket ACL (or an isolated daemon); it is outside
this qualification's threat model.

The 18-container-ID baseline used by an operational validation run is only a
run-audit snapshot. It may be recorded and compared before/after to detect
unintended daemon changes, but it is not a qualification predicate,
admission preimage, artifact claim root, or authority to remove any baseline
container. Claim-critical cleanup evidence remains ownership validation,
ID-only mutation, and target-bound absence of that owned ID.

Consequently the dry-run authority and the Commit-A configuration remain
qualification `0/20`, candidate/final receipt `null`, Q1 `0/20 / NOT_RUN`, and
all eight formal output slots `null`.

## 12. Wire-profile root

The qualification wire profile preimage contains the ordered tag registry,
schema registry, hash-domain registry, failure-code registry, sidecar
path/mode rows, node3 limits, 16 MiB boundary row, authority field registry
and value, predicate registry, exact external-sort trace schema, the exact
15-field counting/discard schema and equality-rule registry, and actor
envelope schema/registry.

```text
ContentHash("HEGEL/Q05B/QUALIFICATION/WIRE_PROFILE/V1", profile_object)
  = bd85abed6feb4b4e9fd6102f43c5db3bbaf9733f0ec42ab5b5363e14a86d350e
```

The final root is machine-derived and is asserted by tests/config; it must be
recomputed after any wire edit before the source freeze is committed.

## 13. Two-commit protocol

Q0.5b uses two commits to avoid a source identity cycle:

1. `SOURCE_FREEZE`: commit the contract, tests, this document and the
   machine-readable configuration. It contains no actual qualification
   receipt and claims no predicate pass.
2. `ACTUAL_QUALIFICATION`: build offline actor images from commit 1, execute
   isolated Python/Rust actors, then execute read-only trusted-host replay.
   Every implementation/evidence object binds commit 1 as raw 20-byte Git
   SHA-1. Only after all 20 qualification predicates pass may commit 2 carry
   the non-Q1 final receipt.

Commit 2 still leaves Q1 at `0/20 / NOT_RUN / eight null output roots`. A
separate later admission action is required before any full node6 execution.
