# Hegel Machine Phase-3A M2.5 Formal Static Basis Engineering Freeze v1

Status: `ENGINEERING_FREEZE / PUBLIC STATIC PREIMAGES ONLY`

Machine scope:

- DSL: `hegel-old-dsl-v1.1.0`
- effective freeze: `hegel-freeze-p2b-p3-v1.1.2`
- generator: `src/hegel_machine/phase3_m25_formal_static_basis_v1.py`
- no new formal object tag, formal schema, ContentHash domain, enum value, or
  signature domain is introduced here.

This engineering freeze closes the previously explicit
`FAIL_M25_FORMAL_PUBLIC_BASIS_REGISTRY_PREIMAGE_UNFROZEN` gap. It does not
instantiate a split seed, key, signature, marker, run ID, ledger ID, Gate pass,
or M3 state transition.

## 1. Exact ordinary-digest preimage

Every `semantics_digest_or_null` descriptor is:

```text
SHA256(canonical_cbor([
  1,
  b"hegel-formal-static-descriptor/1",
  UTF8(kind),
  numeric_id,
  recursively_UTF8_encoded_value_array
]))
```

This is ordinary SHA-256 over the displayed bytes, not a new ContentHash
domain. Textual descriptor atoms are UTF-8 byte strings before CBOR encoding;
formal CBOR therefore still contains no text item. The generator retains every
descriptor preimage and rejects any digest without a retained preimage.

`canonical_name_digest`, DSL/freeze IDs, rules, profiles, codecs, errors, and
entrypoints continue to use the already frozen `IdDigestV1` preimage. Git source
records use the exact Git blob identity:

```text
SHA1(b"blob " || ASCII(decimal_byte_length) || 0x00 || exact_blob_bytes)
```

## 2. Identifier registry

Numeric IDs are zero-based positions within each `RegistryKindId`; no numeric
ID is reused. The complete child registry has 55 rows:

| kind | value | exact names / IDs | child state |
|---:|---|---|---|
| 1 | `ENTITY_SLOT` | `e0..e7` = `0..7` | active |
| 2 | `QUANTITY` | `q0,q1` = `0,1` | active |
| 3 | `CONTEXT` | `c0..c3` = `0..3` | active |
| 4 | `ROLE` | `r0..r3` = `0..3` | active |
| 5 | `SCALE` | `s0,s1` = `0,1` | active |
| 6 | `TASK` | `t0,t1` = `0,1` | active |
| 7 | `SCOPE` | catalog order = `0..3` | active |
| 8 | `AGGREGATE_MAP` | table below | sparse |
| 9 | `TRANSFORM` | catalog order = `0..3` | active; adapter-only semantics are separate |
| 10 | `OPERATOR` | expression order below = `0..18` | active |
| 11 | `NEW_SYMBOL` | no child DSL entries | empty |

Scope IDs are exactly:

```text
0 scope_all_observed_v1
1 scope_primary_only_v1
2 scope_boundary_only_v1
3 control_volume_all_observed_v1
```

Aggregate-map history is exactly:

| ID | name | parent | child | removed version |
|---:|---|---|---|---|
| 0 | `sum_v1` | active | active | null |
| 1 | `count_nonzero_v1` | active | active | null |
| 2 | `mean_v1` | active | tombstone | `hegel-old-dsl-v1.1.0` |
| 3 | `min_v1` | active | tombstone | `hegel-old-dsl-v1.1.0` |
| 4 | `max_v1` | active | tombstone | `hegel-old-dsl-v1.1.0` |
| 5 | `signed_balance_v1` | active | active | null |

Tombstones 2, 3, and 4 remain in the full registry tree, are the complete
`removed_registry_entry_root` subset, and are excluded from the
`surviving_registry_entry_root` subset. They are never renumbered or reused.

Operator identifier IDs use this exact AST-expression order:

```text
0 scalar_const              10 add
1 bit_at                    11 difference
2 set_size                  12 equal_exact
3 aggregate                 13 less_equal
4 context_flag              14 greater_equal
5 task_flag                 15 same_sign
6 bit_to_scalar             16 opposite_sign
7 int_to_scalar             17 approx_equal
8 absolute                  18 top_level_AND
9 sign
```

`aggregate` ID 3 is an AST dispatcher. Its six typed result cases are encoded
as `OperatorClassId.AGGREGATE_MAP` rows, so the wire never invents a false
single output sort for the dispatcher.

## 3. Operator-semantics rows

The child tree has 28 rows, ordered by `(operator_class_id, operator_id)`:

| class | IDs | names | admission |
|---:|---|---|---|
| 1 leaf | 0,1,2,4,5 | `scalar_const`, `bit_at`, `set_size`, `context_flag`, `task_flag` | active DSL |
| 2 unary | 0..3 | `bit_to_scalar`, `int_to_scalar`, `absolute`, `sign` | active DSL |
| 3 binary | 0..6 | `add`, `difference`, `equal_exact`, `less_equal`, `greater_equal`, `same_sign`, `opposite_sign` | active DSL |
| 4 ternary | 0 | `approx_equal` | active DSL |
| 5 conjunction | 0 | `top_level_AND` | active DSL |
| 6 aggregate | 0..5 | aggregate table order | 0,1,5 active; 2,3,4 tombstone removed |
| 7 adapter | 0..3 | transform catalog order | adapter-only |

Input/output sorts are taken directly from the committed typed catalogs.
Aggregate class inputs are `(ENTITY_SET=16, QUANTITY_ID=11)` and retain their
per-map output sort. Adapter inputs/outputs are `RATIONAL_VALUE=5`.

The primary undefined-semantics IDs are:

- leaf totals `1`; `bit_at` `3`;
- ordinary scalar propagation/comparisons/conjunction `2`;
- `add`, `difference`, and scaling adapters `6`;
- aggregate measurement failures `5`; removed mean/min/max rows retain `4`.

`executable_semantics_root` is the `NormativeDocumentBlobV1` ContentHash of the
exact committed `phase3_dsl_v1.py` Git blob. Per-row descriptor preimages bind
the row-to-source mapping. `normalization_rule_root_or_null` is null because
normalization remains inside that complete committed executable source; no
untyped root-shaped placeholder is minted.

## 4. Root DAG and lineage anchor

The three already frozen normative documents form the exact
`NormativeDocumentBundleV1` roles 1, 2, and 3. AST, CBOR, Phase-2B, MDL, and
hidden-scope profile objects use the frozen source-section shape and bind the
entire base-amendment Git blob without newline normalization.

The v1.0.0 parent predates a complete `DslSpecV1` genesis wire. Its mandatory
`parent_dsl_spec_root` field is therefore anchored to the
`NormativeDocumentBlobV1` ContentHash of the exact committed
`phase3_dsl_v1.py` source. This is a disclosed typed-lineage anchor with a
replayable formal preimage, not an arbitrary 32-byte filler. The constructed
parent `DslSpecV1` then supplies the actual parent root referenced by the child
`DslSpecV1`. Parent `FreezeSpecV1.parent_freeze_root_or_null` is null; the child
freeze references the constructed parent freeze root.

The child DSL binds the full 55-row identifier tree, 28-row operator tree,
exact extensional equivalence, structural limits `4/7/3/4/1/2/2/3/8`, canonical
program budget 50,000, raw application cap 5,000,000, and shrink step
`SHRINK_STEP_1_REMOVE_MEAN_MIN_MAX`.

The public pre-seed DAG also contains no root-shaped fillers. The
`parent_normative_decision_root` is a `NormativeDocumentBlobV1` over the exact
committed shrink-step decision bytes. `parent_execution_evidence_root` binds
the exact committed parent dual-strict capacity replay JSON, and
`shrink1_subset_replay_root` binds the exact committed child subset replay
JSON. Their repository paths and Commit-A Git ID are inside each formal
preimage.

`ApprovalEvidenceBundleV1`, `NormativeApprovalManifestV1`, and
`SplitSpecFreezeV1` are fully constructed. Their recording/freeze timestamp is
Commit A's positive Git committer Unix timestamp: this means “recorded in
Commit A”, not a cryptographic assertion about the earlier chat instant.
Approval method 1 remains the non-cryptographic pseudonym
`project-owner:erzhu419`; `SplitSpecFreezeV1.seed_state_id` remains 1.

For later assembly the generator publishes named partial field sets, not
partial formal objects. `SeedContinuityManifestV1` still requires the current
commitment, parent-absence attestation, ledger genesis, custodian core, and
actual instantiation time. `DslShrinkTransitionFormalV1` still requires both
role bindings, split binding, custodian binding, continuity root, and actual
creation time. No root is computed until every required dynamic field exists.
The frozen split algorithm machine ID is
`hegel-split-algorithm-hkdf-hmac-sha256-rank-v1`. The same adapter publishes
both role bindings' exact legacy diagnostic IdDigests from the committed
`content_id`, universe-content-ID, and truth-table-content-ID generators;
these IDs are not inferred by the ceremony runner.

## 5. Traversal, accounting, archive, output and state contracts

- traversal bucket keys: `(OUTPUT_SORT_ID, AST_DEPTH, AST_NODE_COUNT) = 1,2,3`;
- canonical order keys: all `TraversalFieldId` values `1..5`;
- accounting counters: all values `1..6`; invariants: all values `1..3`;
- program/output chunks: 4,096 records, codec `IDENTITY_V1 = 0`;
- program archive is target-independent and uses `CanonicalProgramRecordV2`;
- output archive is role-specific and uses `ProgramOutputRecordV2`;
- state/phase/reason registries are RFC6962 trees of
  `IdentifierRegistryEntryV1` rows over the frozen numeric enum tables;
- the state contract contains exactly the nine `LEGAL_M3_TRANSITIONS`, with
  terminal states 2..6 and `reopen_allowed=false`.

Python and Rust **static-replay** implementation bindings use exact committed
`SourceFileRecordV1` trees, exact binary SHA-256 bytes, exact
execution-environment objects, the Cargo.lock `DependencyLockRecordV1` tree (or
the empty Python dependency tree), the digest-pinned OCI image reference,
entrypoint/build-profile IdDigests, and Commit A. A compiled Rust binary digest
is an execution-instance identity, not a reproducible-build claim.

These two roots are named
`python_static_replay_implementation_binding_root` and
`rust_static_replay_implementation_binding_root`. They must **not** populate
`M3ExecutionCandidateV1.python_implementation_binding_root` or its Rust peer.
The repository still says both complete enumerators are absent, so the static
basis publishes the fail-closed gap
`M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY`. Only bindings for executable,
complete closure runners may close that gap and enter an M3 execution candidate.

## 6. Diagnostic bridge

The bridge contains exactly 12 `DiagnosticFormalBridgeRecordV1` rows for:

1. odd target spec; 2. odd universe; 3. odd truth; 4. sink spec; 5. sink
universe; 6. sink truth; 7. child DSL; 8. operator semantics; 9. identifier
registry; 10. AST profile; 11. CBOR profile; 12. split contract.

Each diagnostic digest and `source_artifact_digest` is SHA-256 over retained
exact `HEGEL_LEGACY_STABLE_JSON_V1` bytes. Each record binds a separately
constructed `RowTransformSpecV1`. Rows are RFC6962-ordered by the already
frozen `(artifact_role_id, diagnostic_namespace_id, diagnostic_digest)` key.

The sink witness is the strict child AST for:

```text
equal_exact(
  aggregate(signed_balance_v1, control_volume_all_observed_v1, q0, []),
  scalar_const(0/1)
)
```

Its exact canonical AST hash is bound identically by `TargetSpecFormalV1` and
`TargetBundleV1`; this supports only the frozen false-invention null claim.

## 7. Gate-19 dual replay

The Gate-19 plan has exactly six ordered entries:

```text
child_dsl_spec_root
child_freeze_root
operator_semantics_root
identifier_registry_root
canonical_ast_schema_root
canonical_cbor_profile_root
```

The Python receipt retains each exact canonical CBOR preimage and truthfully
labels itself as a committed host generator; it makes no OCI or enforced
network-isolation claim. The Rust receipt passes the same public preimages to
the existing `formal_bridge_m25` public
`content_hash` or `rfc6962_root` operation inside the pinned Rust image with:

```text
--pull=never --network=none --read-only --cap-drop=ALL
--security-opt=no-new-privileges --security-opt=seccomp=<committed profile>
```

Gate 19 must rebuild and compare both receipts, every preimage byte, request
digest, echoed canonical CBOR, row count, root, binary digest, Commit-A ID, and
the Rust endpoint's exact Docker command plus stdout/stderr byte evidence. The
Rust binary, image, and committed seccomp profile are digest-bound; its command
must include the frozen offline/read-only/capability restrictions. Two
caller-supplied equal root maps are not evidence.

Passing this replay does not alone report `24/24`; formal state remains
`NOT_RUN` until all other gate evidence is independently replayed.
