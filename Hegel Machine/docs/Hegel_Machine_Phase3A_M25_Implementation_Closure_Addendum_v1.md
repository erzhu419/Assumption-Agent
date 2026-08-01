# Hegel Machine Phase-3A M2.5 Implementation Closure Addendum v1

**Document type**: owner-authorized exact implementation decision

**document ID**: `hegel-m25-implementation-closure-addendum-v1`

**machine freeze ID**: `hegel-freeze-p2b-p3-v1.1.2`

**child DSL ID**: `hegel-old-dsl-v1.1.0`

**status**: `DETERMINISTIC_WIRE_AUTHORIZED_EXTERNAL_AUTHORITY_NOT_CREATED`

This addendum resolves the remaining places where
`Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md` names a root or
field but leaves two incompatible byte interpretations, an unbound nested
preimage, or a construction cycle. The project owner explicitly delegated
these implementation-level choices to the repository implementation. No
authoritative root, seed, key, signature, marker, gate transition, or M3 run
is created by this document.

## 1. Normative document set

Tag `0x3018` is allocated to `NormativeDocumentBundleV1`:

```text
[
  1,
  0x3018,
  b"hegel-normative-document-bundle/1",
  bundle_id_digest,
  [
    [1, base_v1_1_2_amendment_document_root],
    [2, exact_wire_errata_resolution_document_root],
    [3, implementation_closure_addendum_document_root]
  ],
  repository_commit_id
]
```

The ContentHash domain is `HEGEL/NORMATIVE_DOCUMENT_BUNDLE/V1`. Document-role
IDs are exactly `1=BASE_AMENDMENT`, `2=ERRATA_RESOLUTION`, and
`3=IMPLEMENTATION_CLOSURE_ADDENDUM`; entries are unique and ordered by role.
Every v1.1.2 field named `amendment_document_root` binds this bundle root.
The `repository_commit_id` is the deterministic implementation basis commit
(Commit A), never the later publication commit that contains external roots.

## 2. Bridge and all external signatures

The YAML field list in the errata resolution is authoritative over the
shortened prose formula. A bridge signature preimage is exactly:

```text
UTF8("HEGEL/BRIDGE_ATTESTATION_SIGNATURE/V1")
|| 0x00
|| bridge_replay_statement_root_32_bytes
|| uint16_be(signer_purpose_id)
|| uint64_be(signer_key_epoch)
```

This prevents an otherwise valid signature from being transplanted between
purposes or key epochs. The same suffix rule is used for other external
signatures:

```text
UTF8(domain_for_object_tag)
|| 0x00
|| enclosed_object_root_32_bytes
|| uint16_be(signer_purpose_id)
|| uint64_be(signer_key_epoch)
```

Custodian domains are the four E12 domains. The parent-auditor domain is
`HEGEL/PARENT_ABSENCE_AUDITOR_SIGNATURE/V2`. Each
`SignedManifestEnvelopeV1` contains exactly one signature. Its historical
wire field `custodian_key_epoch` has the sole v1 semantics
`signer_key_epoch`; the old name is only an API/documentation alias.

The pre-M3 external-input `AttestationBundleV1` bound by
`M3ExecutionCandidateV1.custodian_attestation_bundle_root` contains exactly
five entries:

1. purpose 1 / `0x3103` seed commitment;
2. purpose 1 / `0x3105` custodian binding;
3. purpose 1 / `0x3106` seed continuity;
4. purpose 1 / `0x3108` ledger genesis;
5. purpose 4 / `0x3114` parent-absence attestation V2.

The inherited candidate field name is retained for byte stability; its exact
meaning is the external-input attestation bundle above. The later bridge
bundle is separate and contains exactly purposes `1,2,3`, each signing the
same `BridgeReplayStatementV1` root.

## 3. Trust policy

`ActorTrustGenesisV1.purpose_key_policy_root` is exactly the ContentHash root
of the frozen `ReplacementPolicyV1` object. Before M4 its entries are exactly
purposes `1,2,3,4`, ordered by purpose ID. Purpose 5 is reserved but absent.
Across dereferenced `ActorKeyManifestV1` objects, key IDs and raw public keys
must both be pairwise distinct. Purpose 1 may sign all purpose-1 objects under
their distinct domains; this is same-purpose use, not cross-purpose reuse.

## 4. Formal profile preimages

The following tags have one common source-section binding shape, with the
first field name specialized as shown:

| tag | object | first field | ContentHash domain |
|---:|---|---|---|
| `0x3019` | `CanonicalAstProfileSpecV1` | `profile_id_digest` | `HEGEL/CANONICAL_AST_PROFILE/V1` |
| `0x301A` | `CanonicalCborProfileSpecV1` | `profile_id_digest` | `HEGEL/CANONICAL_CBOR_PROFILE/V1` |
| `0x301B` | `Phase2BContractSpecV1` | `contract_id_digest` | `HEGEL/PHASE2B_CONTRACT/V1` |
| `0x301C` | `MdlCodeTableSpecV1` | `table_id_digest` | `HEGEL/MDL_CODE_TABLE/V1` |
| `0x301E` | `HiddenArtifactScopeV1` | `policy_id_digest` | `HEGEL/HIDDEN_ARTIFACT_SCOPE/V1` |

Each exact array is:

```text
[
  1,
  object_tag,
  object_schema_id,
  specialized_id_digest,
  governing_normative_document_root,
  section_selector_id_digest,
  section_blob_sha256,
  section_byte_length,
  repository_commit_id
]
```

Schema IDs are respectively:

```text
hegel-canonical-ast-profile/1
hegel-canonical-cbor-profile/1
hegel-phase2b-contract/1
hegel-mdl-code-table/1
hegel-hidden-artifact-scope/1
```

A section selector names one exact Markdown heading or
`section:entire-document`. For a heading, bytes start at the first UTF-8 byte
of its heading line and end immediately before the next heading of equal or
higher level. Bytes come from the Git blob without newline normalization.
The section SHA-256 and byte length are independently recomputed.

This source-section binding is the full preimage: implementations must not
invent a second partial field-level encoding of the same profile.

## 5. Static role metadata

`StaticRoleMetadataV1` remains the exact `0x301D` schema in the errata. Numeric
identifier IDs are zero-based positions in their frozen identifier-registry
kind. Profiles are exactly:

```text
odd  = [input_signature_id=1, [], [], [], [], metadata_rule_id_digest]
sink = [input_signature_id=2, [0,1,2,3], [0], [3], [1,1,-1,-1], metadata_rule_id_digest]
```

Here quantity `0` is `q0`, scope `3` is
`control_volume_all_observed_v1`, and role IDs `0..3` are `r0..r3`.
Arrays are not per-row latent metadata; they are the public static role
contract for the typed input signature.

## 6. Parent-absence audit

All Git object algorithm fields in v1 are `1=SHA1`; their digest is exactly 20
bytes. Algorithm `2=SHA256` is reserved and rejected by this freeze.
Repository commit IDs retain wire form `[1, 20-byte SHA1]`.

`AuditedHistoryRowV1.ordered_parent_commit_ids` contains Git commit IDs in the
order stored by the commit object. `commit_generation` is `0` for a root
commit and `1 + max(parent_generation)` otherwise. The audited history tree
contains every commit reachable from the frozen audited parent commit.

For each history row, `touched_path_set_root` is the RFC6962 root of the
commit's `AuditedPathBlobRecordV1` rows, ordered by raw path bytes. The top
level audited path tree is the deduplicated union of those rows, ordered by
`(raw_repository_path_utf8_bytes, repository_path_alias_id_digest,
git_blob_digest)`.

`ParentManifestAbsenceAttestationV2` is exactly:

```text
[
  1,
  0x3114,
  b"hegel-parent-manifest-absence-attestation/2",
  parent_dsl_version_digest,
  parent_freeze_version_digest,
  parent_repository_commit_id,
  audit_bundle_root,
  absence_reason_bitmask,
  auditor_key_id,
  audited_at_unix_seconds
]
```

The bitmask is exactly `0b1111`. The audit bundle has exactly two legacy
source rows, one for each target role and exact source ID listed in E7. The
purpose-4 envelope over this attestation is required in the external-input
attestation bundle.

Repository path alias rows are ordered by
`(path_alias_id_digest, raw_repository_path_utf8_bytes)` and both columns are
unique. `SourceFileRecordV1` also requires Git algorithm 1 and a 20-byte blob
digest. `DependencyEcosystemId` is
`0=INVALID, 1=PYTHON, 2=RUST, 3=SYSTEM`; value 0 is rejected.

## 7. Opaque-ID cycle removal

Tag `0x3115` is allocated to `OpaqueIdRegistrationIntentV1`:

```text
[
  1,
  0x3115,
  b"hegel-opaque-id-registration-intent/1",
  opaque_id_kind_id,
  opaque_id_16_bytes,
  registration_context_root,
  created_at_unix_seconds,
  repository_commit_id
]
```

Its ContentHash domain is `HEGEL/OPAQUE_ID_REGISTRATION_INTENT/V1`.
`OpaqueIdRegistryRecordV1.first_seen_object_root` is this intent root. Thus
the ID is committed before registry insertion without referring to an object
whose root itself depends on the registry snapshot.

One snapshot appends exactly one record. `added_record_root` is the RFC6962
root of the singleton canonical record. `registry_tree_root` is the RFC6962
root of all records through that sequence. Record sequence numbers are
contiguous from zero. Raw 16-byte ID values, not merely `(kind,id)` pairs, are
globally unique across RUN_ID and LEDGER_ID.

The genesis snapshot has null `previous_snapshot_root_or_null` and record
count 1. Later snapshots bind the previous snapshot root and increase the
count by exactly one. The registration intent, record, O_EXCL reservation,
and snapshot are completed before any object that consumes the new ID.

## 8. M3 run state and Gate 24

The old extra integer in the historical run-state example is deleted. The
exact `M3RunStateRecordV1` fields after the standard three-element prefix are:

```text
run_id
transition_index
previous_state_record_root_or_null
from_state_id
from_phase_id
to_state_id
to_phase_id
transition_reason_id
execution_manifest_root
triggering_receipt_root_or_null
recorded_at_unix_seconds
```

`allowed_reason_ids` in `LegalTransitionRowV1` is a unique ascending array of
`M3TransitionReasonId` values. State and phase IDs use their independent
registries. The only start record is transition index 0, null previous root,
`NOT_RUN/NONE -> RUNNING/CANONICAL_ENUMERATION`, reason
`ENTRY_GATES_24_OF_24`, and null triggering receipt.

The authoritative Gate 24 name is
`M3_EXECUTION_MANIFEST_ROOT_NON_NULL_AND_15_OUTPUT_ROOTS_NULL`. Its success
qualifies entry but leaves the child `NOT_RUN`. Only a separate
`phase3-m3-start` action may construct the start record, and only after it
replays the complete 24/24 evidence and bound opaque-ID snapshot.

## 9. Repository commit fields and authority boundary

Every external object constructed between Commit A and publication Commit B
uses Commit A's Git ID in `repository_commit_id`. A formal object cannot bind
the hash of the commit that contains its own bytes. Commit B publication is
bound separately by the publication receipt and Git history.

Deterministic golden vectors may use fixed synthetic IDs, timestamps, keys,
signatures, roots, and Git IDs. They must be labeled non-authoritative and
must never satisfy an external-actor gate. Real external genesis remains
prohibited until Python and Rust reproduce every errata vector and negative
error code from committed Commit-A sources.

## 10. Commit-A build qualification and secret-absence evidence

The ten-field external-genesis start guard may not be populated from prose or
from a caller-supplied executable. `rust_errata_vectors_pass=true` requires a
fresh Rust build whose executable inputs are byte-equal to blobs read back
from Commit A. The qualification runner must:

1. prove Commit A is an ancestor of the current repository state;
2. compare every bound source, test, golden, and normative-document byte with
   `git show CommitA:path`, then materialize those blobs into a private detached
   Git-archive snapshot;
3. run Python from that detached snapshot and build Rust there in a newly
   created empty target directory; neither endpoint may execute source from
   the live worktree;
4. use an isolated Cargo home containing only the offline registry index and
   the exact `.crate` archives whose SHA-256 values match Commit A's
   `Cargo.lock`; do not copy ambient already-unpacked dependency sources, and
   require Cargo to unpack them inside the private replay directory; also use
   a whitelist environment, locked dependency resolution, verified absence of
   every visible ancestor/Cargo-home config, no wrappers, and disabled
   incremental compilation;
5. reject caller-supplied Cargo/Rust paths; require the committed local
   toolchain policy to match the rustup launcher, the actual cargo and rustc
   binaries, their versions, and the content manifest of the complete selected
   toolchain directory; record those bindings together with the source/registry
   manifests, output-binary SHA-256, interpreter, normalized command, and
   Commit A ID;
6. hash and execute the same open Rust binary inode, verify the detached source
   and tool inputs remained stable, and repeat the Commit-A blob comparison
   before granting the guard.

This local build receipt establishes which committed source bytes were built.
It is not an external build attestation and cannot substitute for the Python,
Rust, custodian, or auditor signatures required later in M2.5.
Because ephemeral debug build paths may change binary bytes, the stored binary
digest is an execution-instance identity rather than a cross-build
reproducibility claim. A checked JSON artifact is archival evidence only; an
external custodian must require a fresh dual replay (or later external
attestation) and may not authorize genesis from public fields plus a
self-generated diagnostic hash alone.

`secrets_absent_from_repository=true` requires a separate, reproducible
`RepositoryGenesisSecretAbsenceReceiptV1`. Its exact claim is limited to:

> No artifact matching the frozen Phase-3A external-genesis secret policy was
> found in the `Hegel Machine` blobs reachable from Commit A.

The audit covers Commit A and all of its ancestors, deduplicates Git blobs,
and freezes path, private-key-header, and structured secret-field detection.
It records only object/path metadata and findings, never matching secret
bytes. Deterministic public vector material is not a secret merely because it
has key-, signature-, seed-, or root-shaped length, but it receives no
exemption from private-key headers, forbidden secret-state paths, or non-null
structured secret fields. The receipt passes only with zero findings and must
replay identically after the dual report.

This is an exact project-policy audit, not a general proof that arbitrary
bytes cannot conceal a secret. A caller-supplied binary plus an informal
repository inspection is at most a candidate dual replay and must leave the
external-genesis guard below 10/10.
