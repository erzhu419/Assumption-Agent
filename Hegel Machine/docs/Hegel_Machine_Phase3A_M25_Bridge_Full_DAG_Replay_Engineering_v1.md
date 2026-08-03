# Hegel Machine Phase-3A M2.5 Bridge Full-DAG Replay Engineering Freeze v1

```yaml
document_id: hegel-m25-bridge-full-dag-replay-engineering-v1
status: IMPLEMENTED_REPLAY_AND_ACTOR_WORKERS_PENDING_EXECUTOR_BINDING
normative_precedence: engineering_implementation_below_v1_1_2_wire_freeze
authoritative_use_allowed: false
formal_root_published: false
real_seed_generated: false
real_key_generated: false
real_signature_generated: false
m3_state_changed: false
blocker_removed: false
```

## 1. Scope

This freeze defines an exact, canonical-CBOR replay package for the roots
addressed by `M3ExecutionCandidateV1`, its typed universe/truth preimages, the
candidate-to-bridge projection, the split-binding edges, the two role-binding
edges, and the purpose-1 trust/signature edge.

The package is replayed by:

1. the Python implementation for purposes 1 and 2; and
2. the independent Rust implementation for purpose 3.

The replay libraries never sign.  Purpose-private actor workers may sign only
after their in-process operation probe and a successful authoritative replay;
the bridge root and signature preimage are derived inside the actor from the
package.  No replay success by itself creates a formal root or changes M3
state.

## 2. Exact package wire

```text
tag       = 0x3501
schema_id = b"hegel-m25-bridge-full-dag-replay-package/1"
digest    = SHA256(
              b"HEGEL/M25/BRIDGE_FULL_DAG_REPLAY_PACKAGE/V1" ||
              0x00 ||
              CanonicalCBOR(package)
            )
```

The package is the 12-element numeric CBOR array:

```text
[
  1,
  0x3501,
  b"hegel-m25-bridge-full-dag-replay-package/1",
  authority_boolean,
  replay_purpose_id,
  m3_execution_candidate_cbor_bytes,
  bridge_replay_statement_cbor_bytes,
  replay_nodes,
  purpose1_actor_key_manifest_cbor_bytes,
  purpose1_bridge_signature_or_null,
  created_at_unix_seconds,
  repository_commit_id
]
```

`authority_boolean=true` is rejected unless the actor uses the explicit
runtime-only authority opt-in.  Library qualification fixtures carry `false`;
actor integration fixtures exercise `true` without producing a real key,
signature, seed, or formal root.

Each replay node is:

```text
[
  role_id,
  operation_id,           # 1=ContentHash, 2=RFC6962, 3=sealed split
  formal_object_tag,
  formal_schema_id,
  content_hash_domain_or_null,
  exact_preimage_cbor_byte_strings,
  exact_record_count,
  sealed_split_root_or_null
]
```

Nodes are mandatory, unique, ascending, and exactly the 37 registered roles.
Schema, operation, hash domain, record count, and candidate field position are
locally frozen in both implementations; none is accepted from a package as an
untrusted policy choice.

## 3. Registered roles

| Role | Candidate field | Rule | Exact count |
|---:|---|---|---:|
| 1 | `child_dsl_spec_root` | `DslSpecV1` ContentHash | 1 |
| 2 | `child_freeze_root` | `FreezeSpecV1` ContentHash | 1 |
| 3 | `approval_manifest_root` | `NormativeApprovalManifestV1` ContentHash | 1 |
| 4 | `shrink_transition_root` | `DslShrinkTransitionFormalV1` ContentHash | 1 |
| 5 | `operator_semantics_root` | `OperatorSemanticsEntryV1` RFC6962 | 28 |
| 6 | `identifier_registry_root` | `IdentifierRegistryEntryV1` RFC6962 | 55 |
| 7 | `canonical_ast_schema_root` | `CanonicalAstProfileSpecV1` ContentHash | 1 |
| 8 | `canonical_cbor_profile_root` | `CanonicalCborProfileSpecV1` ContentHash | 1 |
| 9 | `diagnostic_formal_bridge_root` | `DiagnosticFormalBridgeRecordV1` RFC6962 | 12 |
| 10 | `outside_target_binding_manifest_root` | `DslRoleBindingManifestV1` ContentHash | 1 |
| 11 | `null_control_binding_manifest_root` | `DslRoleBindingManifestV1` ContentHash | 1 |
| 12 | `split_binding_manifest_root` | `SplitBindingManifestV1` ContentHash | 1 |
| 13 | `custodian_binding_manifest_root` | `CustodianBindingManifestV1` ContentHash | 1 |
| 14 | `seed_continuity_manifest_root` | `SeedContinuityManifestV1` ContentHash | 1 |
| 15 | `custodian_attestation_bundle_root` | `AttestationBundleV1` ContentHash | 1 |
| 16 | `parent_absence_attestation_root` | `ParentManifestAbsenceAttestationV2` ContentHash | 1 |
| 17 | `hidden_access_ledger_genesis_root` | `HiddenAccessLedgerRecordV1` ContentHash | 1 |
| 18 | `hidden_access_ledger_head_root` | `HiddenAccessLedgerRecordV1` ContentHash | 1 |
| 19 | `opaque_id_registry_snapshot_root` | `OpaqueIdRegistrySnapshotV1` ContentHash | 1 |
| 20 | `actor_trust_genesis_root` | `ActorTrustGenesisV1` ContentHash | 1 |
| 21 | `outside_target_universe_root` | `BoundedUniverseRowV1` RFC6962 | 480 |
| 22 | `outside_target_truth_root` | `TargetTruthRowV1` RFC6962 | 480 |
| 23 | `null_control_universe_root` | `BoundedUniverseRowV1` RFC6962 | 85 |
| 24 | `null_control_truth_root` | `TargetTruthRowV1` RFC6962 | 85 |
| 25 | `outside_discovery_split_root` | sealed | 192 |
| 26 | `outside_validation_split_root` | sealed | 96 |
| 27 | `outside_sealed_split_root` | sealed | 192 |
| 28 | `null_discovery_split_root` | sealed | 39 |
| 29 | `null_validation_split_root` | sealed | 20 |
| 30 | `null_sealed_split_root` | sealed | 26 |
| 31 | `python_implementation_binding_root` | `ImplementationBindingV1` ContentHash | 1 |
| 32 | `rust_implementation_binding_root` | `ImplementationBindingV1` ContentHash | 1 |
| 33 | `traversal_contract_root` | `TraversalContractV1` ContentHash | 1 |
| 34 | `bucket_accounting_contract_root` | `BucketAccountingContractV1` ContentHash | 1 |
| 35 | `program_archive_contract_root` | `ProgramArchiveContractV1` ContentHash | 1 |
| 36 | `output_archive_contract_root` | `OutputArchiveContractV1` ContentHash | 1 |
| 37 | `state_machine_contract_root` | `StateMachineContractV1` ContentHash | 1 |

## 4. Mandatory replay checks

Both implementations perform these checks without trusting self-reported
candidate roots:

```yaml
package:
  - strict deterministic CBOR
  - exact numeric prefix and field count
  - exact purpose and authority guard
nodes:
  - exact 37-role set and order
  - exact operation/tag/schema/domain/count per role
  - strict CBOR formal prefix and exact schema length for every disclosed row
  - ContentHash or RFC6962 recomputation from disclosed preimages
  - recomputed root equality at the frozen candidate field position
typed_rows:
  - exact 480/480 outside and 85/85 null cardinalities
  - contiguous paired universe indices
  - outside InputSignatureId=1 and OddInputV1
  - null InputSignatureId=2 and SinkInputV1
  - truth canonical_input_hash recomputation
role_binding:
  - role IDs cannot cross
  - DSL/freeze/operator/registry/AST/CBOR roots bind back to candidate
  - universe/truth roots bind to the correct target role
  - split/custodian/continuity roots bind back to candidate
split_binding:
  - all six partition roots bind to their exact candidate positions
  - ledger genesis/head bind to the candidate
candidate_bridge:
  - candidate ContentHash recomputation
  - exact seven-field BridgeReplayStatement projection
  - bridge ContentHash recomputation
trust_signature:
  - purpose-1 ActorKeyManifest ContentHash recomputation
  - manifest is the purpose-1 entry in ActorTrustGenesisV1
  - key ID equals first 16 bytes of SHA256(raw Ed25519 public key)
  - commit and validity interval checks
  - purposes 2/3 verify purpose-1 Ed25519 signature over the exact frozen preimage
```

## 5. Hidden split claim boundary

Split membership rows and the raw seed remain undisclosed.  The only allowed
positive statement is:

```text
SEALED_ROOT_COUNT_AND_PURPOSE1_BINDING_ONLY
```

The replay result always contains:

```yaml
split_membership_recomputed: false
```

Purpose 1 receives an unsigned package inside its custody boundary.  Its
existing formal worker replays the full package, derives the bridge root and
purpose-1 preimage internally, then signs with its `/state` key.  Purpose 1
rejects a package that already contains its own signature; this prevents a
host-controlled signature-oracle workflow.

Purposes 2 and 3 require the purpose-1 signature.  Thus they verify that the
six exact split roots are included in the candidate signed by the custodian,
without falsely claiming to have reconstructed hidden allocation rows.

## 6. Dependency-free signature adapter

The Python replay core accepts an explicit verifier callback and has no import
dependency on `cryptography`.  The container adapter uses exactly
`/usr/bin/openssl` and requires an explicit, non-symlink, mode-`0700` private
temporary directory.  It uses:

```yaml
network: none
shell: false
stdin: DEVNULL
environment:
  LC_ALL: C
  LANG: C
file_creation:
  flags: O_EXCL plus O_NOFOLLOW where available
  mode: "0600"
  fsync_file: true
  fsync_directory: true
cleanup: finally
message_or_signature_in_argv: false
```

The Rust implementation uses the same fixed SPKI DER profile and OpenSSL
verification contract.  Cargo resolution and compilation were executed in the
pinned Rust image with `--network=none --offline --locked`; no dependency was
downloaded.

## 7. Actor-side contract and strict receipt

All three workers accept exactly this bridge input:

```text
/input/bridge-dag-package.cbor
```

They do not accept a host-created bridge statement, expected bridge root, or
bridge signing preimage.  The Python purpose-1 and purpose-2 workers call
`replay_bridge_dag_package_v1(..., allow_authoritative=True,
signature_verifier=OpenSSL)` in the same process that ran the operation-bound
probe.  Purpose 2 must observe `purpose1_signature_verified=true` before its
key can be used.  Purpose 1 must observe an unsigned purpose-1 package.

The purpose-3 wrapper runs the same-process Rust operation probe, then invokes:

```text
rust-bridge-dag-replay \
  --authoritative-runtime \
  --expected-purpose 3 \
  --signature-preimage-out /tmp/<purpose-private>/signing-preimage.bin \
  /input/bridge-dag-package.cbor \
  /tmp/<purpose-private>
```

The Rust replayer verifies the purpose-1 signature, derives the purpose-3
epoch-0 preimage, and commits it mode `0600` under the actor-private `/tmp`.
The wrapper signs that internally derived file and deletes it.  It never reads
`/input/signing-preimage.bin` or `/input/expected-root.bin`.

Each successful worker emits:

```text
/output/ed25519-signature.bin
/output/bridge-dag-replay-receipt.json
```

The receipt schema is
`hegel-phase3-m25-bridge-dag-actor-replay-receipt/1`.  It is one canonical
ASCII JSON line and binds `authoritative`, bridge/candidate roots, package
ContentHash, exact purpose, purpose-1 verification state, implementation,
epoch 0, the sealed-split claim boundary, and a SHA-256 self-hash.  The
signature is validated independently when the formal envelope is assembled.

## 8. Failure codes

```text
FAIL_M25_BRIDGE_REPLAY_PACKAGE_SCHEMA
FAIL_M25_BRIDGE_REPLAY_AUTHORITY_GUARD
FAIL_M25_BRIDGE_REPLAY_PURPOSE
FAIL_M25_BRIDGE_REPLAY_NODE_SET
FAIL_M25_BRIDGE_REPLAY_NODE_SCHEMA
FAIL_M25_BRIDGE_REPLAY_NODE_PREIMAGE
FAIL_M25_BRIDGE_REPLAY_NODE_COUNT
FAIL_M25_BRIDGE_REPLAY_ROOT_BINDING
FAIL_M25_BRIDGE_REPLAY_CANDIDATE
FAIL_M25_BRIDGE_REPLAY_BRIDGE
FAIL_M25_BRIDGE_REPLAY_CROSS_ROLE
FAIL_M25_BRIDGE_REPLAY_TYPED_BINDING
FAIL_M25_BRIDGE_REPLAY_SEALED_SPLIT_BINDING
FAIL_M25_BRIDGE_REPLAY_PURPOSE1_TRUST_BINDING
FAIL_M25_BRIDGE_REPLAY_SIGNATURE_PHASE
FAIL_M25_BRIDGE_REPLAY_PURPOSE1_SIGNATURE
```

## 9. Qualification and adversarial coverage

The non-authority suite covers:

```yaml
python_purpose1_unsigned_replay: pass
python_purpose2_or_3_signature_replay: pass
python_explicit_openssl_adapter: pass
rust_purpose3_replay: pass
substitution: rejected
omission: rejected
cross_role_swap: rejected
candidate_root_splice: rejected
wrong_dsl: rejected
wrong_operator_semantics: rejected
wrong_identifier_registry: rejected
wrong_universe: rejected
signature_substitution: rejected
authority_promotion_without_opt_in: rejected
fixture_authority: false
python_purpose1_authoritative_actor_fixture: pass
python_purpose2_authoritative_actor_fixture: pass
purpose2_wrong_or_missing_purpose1_signature: rejected_before_signing
purpose2_wrong_purpose: rejected_before_signing
purpose2_non_authoritative_package: rejected_before_signing
rust_authoritative_flag_required: pass
rust_purpose3_internal_preimage_derivation: pass
host_bridge_root_or_preimage_input: absent
```

## 10. Integration hook and remaining blocker

The runtime integration must construct `ReplayNodeV1` values directly from
the already-typed `GateEvidenceInputsV1` and `FormalStaticBasisV1` preimages,
not from serialized root summaries.  The exact hook points are:

```text
after _build_gate_inputs_and_sign_v1 has built the candidate DAG
before actors.sign_bridge(purpose, bridge_fields)
```

Required order:

```text
purpose 1: build unsigned package -> Python replay -> isolated sign
purpose 2: inject purpose-1 signature -> Python replay -> isolated sign
purpose 3: inject purpose-1 signature -> Rust replay -> isolated sign
```

Runtime integration must now build the purpose-specific packages, install the
Rust DAG-replay binary in the purpose-3 snapshot, replay the strict receipts,
and bind the returned signatures into the three envelopes.  It must use the
purpose-private `/tmp` tmpfs from the actor profile, run with `--pull=never
--network=none`, and persist only public replay receipts.  It must add live
fail-closed tests in the four isolated containers.

Until that integration and its live tests pass, the authoritative blocker
`FAIL_M25_BRIDGE_FULL_DAG_REPLAY_NOT_IMPLEMENTED` remains present and Gate 24,
formal roots, real genesis, signatures, and M3 remain unchanged.
