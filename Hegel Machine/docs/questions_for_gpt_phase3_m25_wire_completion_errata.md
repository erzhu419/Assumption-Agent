# Questions for GPT — Phase-3A M2.5 exact-wire errata before external genesis

**Document type**: fail-closed normative decision request
**machine freeze under review**: `hegel-freeze-p2b-p3-v1.1.2`
**child DSL**: `hegel-old-dsl-v1.1.0`
**current evidence state**: `14/24`, `NOT_RUN`
**current transition**: `EXACT_ERRATA_REQUIRED_EXTERNAL_GENESIS_BLOCKED`

This document does not reopen the already reproducible portions of the
v1.1.2 amendment. The published `IdDigestV1` vector, odd 480-row roots, sink
85-row roots, and both split quota tables reproduce exactly. The questions
below concern only choices that still change formal bytes, a root DAG, signer
authority, or M3 state.

Until all twelve decisions are returned in machine-readable form:

```yaml
external_genesis_start_allowed: false
os_csprng_call_allowed: false
split_seed_marker_creation_allowed: false
real_key_or_seed_generation_allowed: false
formal_root_claim_allowed: false
m3_gates_satisfied: 14
m3_gates_total: 24
child_state: NOT_RUN
```

The repository may implement deterministic schemas, public row/root replay,
synthetic split/signature tests, and read-only storage/FD-3/marker validators.
It may not interpret a recommendation below as an approval.

---

## E1. M3RunGenesis has 15 named output slots, but Gate 24 says 16

Evidence:

- `M3RunGenesisV1` names 15 run-produced root fields between
  `canonical_program_archive_root_or_null` and
  `final_state_record_root_or_null`.
- Section 14.2 says “all 16 run-produced output slots”.
- The current source and regression test also enumerate exactly 15.

Choose exactly one:

1. `E1_A_EXACTLY_15` — the listed 15 fields are authoritative; correct the
   prose count from 16 to 15.
2. `E1_B_DEFINE_16TH_SLOT` — provide the missing field name, type, exact
   position, semantics, nullability, and schema-version consequence.

Recommendation: `E1_A_EXACTLY_15`, because it changes no listed wire.

Required response:

```yaml
run_output_slot_count: 0
ordered_run_output_slot_names: []
m3_run_genesis_schema_id: ""
```

---

## E2. Bridge signatures are ordered before the execution root they sign

Evidence:

- The bridge signature message includes
  `execution_manifest_candidate_root`.
- The topological order creates bridge attestations at step 25 and the M3
  execution manifest at step 30.
- The execution manifest also needs the role bindings, transition/contracts,
  and a registered `run_id`.

Choose exactly one:

1. `E2_A_REORDER_EXECUTION_BEFORE_SIGNATURES` — finish role/transition/
   implementation contracts, generate and register `run_id`, construct the
   execution candidate, and only then collect bridge signatures.
2. `E2_B_REMOVE_EXECUTION_ROOT_FROM_SIGNATURE` — change the signature message
   to bind only the bridge root.

Recommendation: `E2_A_REORDER_EXECUTION_BEFORE_SIGNATURES`; it retains
execution-specific replay protection.

Required response:

```yaml
ordered_root_dag_steps: []
run_id_registration_step: 0
bridge_signature_preimage_fields: []
```

---

## E3. The 3/3 bridge statement has no unique envelope identity

Evidence:

- `SignedManifestEnvelopeV1` encloses one object/root, while the bridge
  signature covers both bridge and execution roots.
- The text does not decide between one three-signature envelope and three
  purpose-specific one-signature envelopes.
- `AttestationBundleV1` can list envelopes, but its root is not bound by the
  current execution/genesis wire.

Choose exactly one:

1. `E3_A_STATEMENT_PLUS_THREE_ENVELOPES` — define a formal bridge-statement
   object containing both roots; produce three single-signature envelopes and
   bind their attestation-bundle root into run identity.
2. `E3_B_EXTEND_EXISTING_ENVELOPE` — version the signed envelope to carry both
   roots and exactly three signatures.

Recommendation: `E3_A_STATEMENT_PLUS_THREE_ENVELOPES`; it preserves distinct
purposes and permits independent replay.

Required response:

```yaml
bridge_statement:
  tag: 0
  schema_id: ""
  fields: []
  hash_domain: ""
bridge_envelope_cardinality: 0
bridge_attestation_bundle_binding_location: ""
```

---

## E4. Signed-envelope and dual-agreement root domains are absent

Evidence:

- `signed_envelope_root` is required, but no root formula/domain is assigned
  to `SignedManifestEnvelopeV1`.
- `M3DualReplayAgreementV1` has a schema but no ContentHash domain.

Choose exactly one:

1. `E4_A_DOMAIN_SEPARATED_CONTENT_HASH` — assign explicit ContentHash domains
   to both objects.
2. `E4_B_UNDOMAINED_SHA256` — explicitly create exceptions using bare SHA-256
   over canonical CBOR.

Recommendation: `E4_A_DOMAIN_SEPARATED_CONTENT_HASH`, consistent with the
formal identity policy.

Required response:

```yaml
signed_manifest_envelope:
  root_formula: ""
  hash_domain: ""
m3_dual_replay_agreement:
  root_formula: ""
  hash_domain: ""
```

---

## E5. Actor-key trust roots and signer purposes are not anchored

Evidence:

- Actor key manifests are not transitively bound by execution/genesis
  identity.
- A verifier has no exact source for the “pinned” custodian/attester keys.
- The text does not state whether the custodian manifest signer is the same
  purpose-1 key used for the bridge.
- `SignedManifestEnvelopeV1.custodian_key_epoch` is semantically unclear for
  auditor and implementation-attester envelopes.

Choose exactly one:

1. `E5_A_PURPOSE1_IS_CUSTODIAN_IDENTITY` — purpose 1 is the same custodian
   identity across domain-separated manifest/bridge messages; bind all actor
   key roots in a trust-genesis bundle.
2. `E5_B_SEPARATE_CUSTODY_SIGNER_PURPOSE` — add a distinct purpose/key for
   custodian manifests and bind it beside the bridge key.

Recommendation: `E5_A_PURPOSE1_IS_CUSTODIAN_IDENTITY`, provided the amendment
explicitly says this is one numeric purpose rather than prohibited
cross-purpose reuse.

Required response:

```yaml
actor_key_root_binding_object: ""
pinned_genesis_trust_anchor: ""
custodian_manifest_signer_purpose_id: 0
noncustodian_envelope_epoch_semantics: ""
```

---

## E6. FD-3 calculator replay conflicts with a genesis-only ledger

Evidence:

- Gate 16 requires one ledger record, its genesis, and no access/reveal event.
- The custodian sends the raw seed to Python and Rust calculators over FD 3.
- Hidden-artifact event type 2 includes raw-seed access.

Choose exactly one:

1. `E6_A_CALCULATORS_INSIDE_CUSTODIAN_BOUNDARY` — both constrained calculator
   processes are inside the custodian actor boundary; FD-3 delivery is not a
   separate ledger access event.
2. `E6_B_RECORD_CALCULATOR_ACCESS` — append authorized access records and
   change Gate 16, ledger count/head, split binding, and execution inputs.

Recommendation: `E6_A_CALCULATORS_INSIDE_CUSTODIAN_BOUNDARY`; it preserves the
frozen Gate-16 meaning while keeping two implementations.

Required response:

```yaml
calculator_actor_boundary: ""
fd3_access_event_required: false
gate16_required_ledger_count: 0
gate16_required_head_rule: ""
```

---

## E7. Parent-absence history and two legacy sources lack a complete wire

Evidence:

- `AuditedPathBlobRecordV1 = 0x3210` is defined inline but omitted from the
  numeric tag registry.
- No complete schema/order/root is provided for the audited path set or
  reachable-history records.
- The described history root has no field in
  `ParentManifestAbsenceAttestationV1`.
- Current odd and sink bindings contain distinct legacy source IDs, but the
  absence attestation has one singular source digest.

Choose exactly one:

1. `E7_A_VERSIONED_AUDIT_BUNDLE` — define path/history/legacy-source rows and a
   versioned audit-bundle root referenced by a revised absence attestation.
2. `E7_B_COMPOSITE_EXISTING_ROOTS` — define the two existing audit roots as
   heterogeneous tagged trees that jointly cover history and both sources.

Recommendation: `E7_A_VERSIONED_AUDIT_BUNDLE`; it avoids silently changing
the meaning of existing fields.

Required response:

```yaml
audited_path_blob_tag_registry_entry: 0
audited_path_set_row_schema: []
audited_history_row_schema: []
audit_tree_ordering: []
audit_root_binding_field: ""
legacy_parent_payload_source_ids: []
```

---

## E8. Target-role and initial-state numeric bytes are ambiguous

Evidence:

- `M3StateId.NOT_RUN = 0` and `ChildInitialStateId.NOT_RUN = 1`.
- `M3RunGenesisV1.initial_state_id` does not select one registry.
- `role_id` appears in target, split, binding, evaluation, and agreement wires,
  but there is no `TargetRoleId` registry.
- `ArtifactRoleId` names artifact classes, not target execution roles.

Choose exactly one:

1. `E8_A_M3STATE0_NEW_TARGET_ROLE_ENUM` — initial state uses M3StateId 0; add
   `TargetRoleId {OUTSIDE_TARGET=1, IN_LANGUAGE_NULL=2}` and map every role
   field explicitly.
2. `E8_B_CHILDSTATE1_ARTIFACT_ROLE_REUSE` — initial state uses
   ChildInitialStateId 1 and selected ArtifactRoleId values are reused.

Recommendation: `E8_A_M3STATE0_NEW_TARGET_ROLE_ENUM`; it avoids namespace
coupling.

Required response:

```yaml
m3_run_genesis_initial_state_registry: ""
m3_run_genesis_initial_state_value: 0
target_role_enum_registry: []
role_id_field_registry_map: []
```

---

## E9. Root preimages, instance IDs, and the repository-path alias are incomplete

Evidence:

- Hidden-artifact scope is prose/YAML without a formal object/domain.
- Canonical AST/CBOR profile, Phase-2B contract, and MDL table roots lack
  unique preimage objects.
- Implementation source/dependency roots and StateMachineContract nested
  tables lack row schemas/field-ID registries; `legal_transition_table` has no
  nested row schema.
- The inherited `M3RunStateRecordV1` example has an extra prefix integer that
  conflicts with the amendment's common `[1, tag, schema_id, ...]` prefix.
- `InputSignatureSpecV1.static_role_metadata` has no payload schema.
- `TraversalContractV1` and `BucketAccountingContractV1` refer to field IDs
  and accounting invariants without numeric registries or nested schemas.
- `TargetSpecFormalV1.claim_level_id` and
  `MismatchRecordV1.mismatch_kind_id` have no enum registries.
- `SplitContractV1.assignment_ordering_rule_id` and
  `fallback_split_policy_id` have no numeric registry.
- Several order/profile/fallback `*_id_digest` values have no exact machine-ID
  catalog.
- `NormativeDocumentBlobV1.repository_relative_path_id_digest` cannot digest
  the literal path
  `Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md`,
  because `IdDigestV1` forbids spaces.

Choose exactly one:

1. `E9_A_FORMAL_PREIMAGE_AND_ALIAS_REGISTRY` — define all root-preimage rows
   and one ASCII machine-ID/path-alias registry whose records bind a legal
   alias to exact raw repository path bytes.
2. `E9_B_PATH_BYTES_AND_DOCUMENT_ROOTS` — version affected schemas to use raw
   path bytes and explicitly reuse normative-document roots for named profile
   roles.

Recommendation: `E9_A_FORMAL_PREIMAGE_AND_ALIAS_REGISTRY`; it preserves
IdDigestV1 and existing field widths.

Required response:

```yaml
formal_root_preimage_registry: []
instance_machine_id_catalog: []
repository_path_alias_rule: ""
amendment_document_path_alias: ""
source_and_dependency_root_row_schemas: []
state_machine_nested_row_schemas: []
m3_run_state_exact_prefix: []
input_signature_static_role_metadata_schema: []
target_claim_level_enum_registry: []
mismatch_kind_enum_registry: []
split_contract_numeric_rule_registry: []
traversal_and_bucket_field_id_registries: []
```

---

## E10. Fresh run/ledger IDs have no replayable registry evidence

Evidence:

- `AppendOnlyOpaqueIdRegistryV1` is named without a record schema, ordering,
  root, or persistence wire.
- Gate 24 requires `run_id` to be fresh and registered.
- No formal input currently binds the registry evidence covering all reachable
  project runs/ledgers.

Choose exactly one:

1. `E10_A_FORMAL_APPEND_ONLY_REGISTRY_ROOT` — define formal registry rows/root
   and bind the current root into M3 genesis qualification.
2. `E10_B_OPERATIONAL_OEXCL_REGISTRY` — freeze one-file-per-ID `O_EXCL`
   persistence plus a public/repository scan and an exact Gate-24 evidence
   receipt.

Recommendation: `E10_A_FORMAL_APPEND_ONLY_REGISTRY_ROOT`; it is independently
replayable.

Required response:

```yaml
opaque_id_registry_record_schema: []
opaque_id_registry_ordering: []
opaque_id_registry_root_formula: ""
opaque_id_registry_binding_location: ""
duplicate_scope_verification_rule: ""
```

---

## E11. The designated sink witness is assigned to a nonexistent field

Evidence:

- Section 4.6 requires
  `DslRoleBindingManifestV1.required_witness_ast_hash`.
- That schema has no such field.
- `TargetSpecFormalV1.required_witness_ast_hash_or_null` and
  `TargetBundleV1.null_control_required_witness_ast_hash_or_null` already
  exist.

Choose exactly one:

1. `E11_A_USE_TARGET_SPEC_AND_BUNDLE_FIELDS` — correct the prose and require
   both existing fields to contain the same designated witness hash.
2. `E11_B_ADD_ROLE_BINDING_FIELD` — add a field in a new role-binding schema
   version and update all downstream roots.

Recommendation: `E11_A_USE_TARGET_SPEC_AND_BUNDLE_FIELDS`; it is the only
option with no wire change.

Required response:

```yaml
authoritative_witness_binding_fields: []
dsl_role_binding_schema_change_required: false
```

---

## E12. The DAG and exact guard disagree on custodian envelope coverage

Evidence:

- The root DAG says final `CustodianBindingManifestV1` receives a signature
  envelope.
- The inherited one-signature guard lists only seed commitment (`0x3103`),
  seed continuity (`0x3106`), and ledger genesis (`0x3108`).
- The destination that binds each signed-envelope root is not complete.

Choose exactly one:

1. `E12_A_SIGN_FOUR_CUSTODIAN_OBJECTS` — require one pinned custodian
   signature for `0x3103`, `0x3105`, `0x3106`, and `0x3108`, and bind all four
   envelope roots.
2. `E12_B_REMOVE_FINAL_BINDING_SIGNATURE` — retain only the inherited three
   signatures and correct the DAG.

Recommendation: `E12_A_SIGN_FOUR_CUSTODIAN_OBJECTS`; it matches the published
topology.

Required response:

```yaml
custodian_signed_object_tags: []
custodian_signature_domain_by_tag: []
custodian_envelope_root_binding_locations: []
required_signature_count_by_tag: []
```

---

# Required machine-readable answer envelope

Please return all twelve decisions in one Markdown code block. Do not answer
only in prose.

```yaml
document_id: "hegel-m25-wire-completion-errata-answer-v1"
machine_freeze_id: "hegel-freeze-p2b-p3-v1.1.2"
decision_status: "RESOLVED"

decisions:
  E1_M3_RUN_GENESIS_SLOT_CARDINALITY:
    selected_option_id: ""
    run_output_slot_count: 0
    ordered_run_output_slot_names: []
    m3_run_genesis_schema_id: ""

  E2_BRIDGE_TOPOLOGY_ORDER:
    selected_option_id: ""
    ordered_root_dag_steps: []
    run_id_registration_step: 0
    bridge_signature_preimage_fields: []

  E3_BRIDGE_ENVELOPE_AND_IDENTITY_BINDING:
    selected_option_id: ""
    bridge_statement:
      tag: 0
      schema_id: ""
      fields: []
      hash_domain: ""
    bridge_envelope_cardinality: 0
    bridge_attestation_bundle_binding_location: ""

  E4_MISSING_HASH_DOMAINS:
    selected_option_id: ""
    signed_manifest_envelope:
      root_formula: ""
      hash_domain: ""
    m3_dual_replay_agreement:
      root_formula: ""
      hash_domain: ""

  E5_ACTOR_KEY_TRUST_AND_PURPOSE:
    selected_option_id: ""
    actor_key_root_binding_object: ""
    pinned_genesis_trust_anchor: ""
    custodian_manifest_signer_purpose_id: 0
    noncustodian_envelope_epoch_semantics: ""

  E6_LEDGER_CALCULATOR_ACTOR_BOUNDARY:
    selected_option_id: ""
    calculator_actor_boundary: ""
    fd3_access_event_required: false
    gate16_required_ledger_count: 0
    gate16_required_head_rule: ""

  E7_PARENT_ABSENCE_AUDIT_WIRE:
    selected_option_id: ""
    audited_path_blob_tag_registry_entry: 0
    audited_path_set_row_schema: []
    audited_history_row_schema: []
    audit_tree_ordering: []
    audit_root_binding_field: ""
    legacy_parent_payload_source_ids: []

  E8_ROLE_AND_INITIAL_STATE_ENUMS:
    selected_option_id: ""
    m3_run_genesis_initial_state_registry: ""
    m3_run_genesis_initial_state_value: 0
    target_role_enum_registry: []
    role_id_field_registry_map: []

  E9_ROOT_PREIMAGES_INSTANCE_IDS_AND_PATH_ALIAS:
    selected_option_id: ""
    formal_root_preimage_registry: []
    instance_machine_id_catalog: []
    repository_path_alias_rule: ""
    amendment_document_path_alias: ""
    source_and_dependency_root_row_schemas: []
    state_machine_nested_row_schemas: []
    m3_run_state_exact_prefix: []
    input_signature_static_role_metadata_schema: []
    target_claim_level_enum_registry: []
    mismatch_kind_enum_registry: []
    split_contract_numeric_rule_registry: []
    traversal_and_bucket_field_id_registries: []

  E10_OPAQUE_ID_REGISTRY_EVIDENCE:
    selected_option_id: ""
    opaque_id_registry_record_schema: []
    opaque_id_registry_ordering: []
    opaque_id_registry_root_formula: ""
    opaque_id_registry_binding_location: ""
    duplicate_scope_verification_rule: ""

  E11_NULL_WITNESS_BINDING_FIELD:
    selected_option_id: ""
    authoritative_witness_binding_fields: []
    dsl_role_binding_schema_change_required: false

  E12_CUSTODIAN_ENVELOPE_COVERAGE:
    selected_option_id: ""
    custodian_signed_object_tags: []
    custodian_signature_domain_by_tag: []
    custodian_envelope_root_binding_locations: []
    required_signature_count_by_tag: []

post_decision_state:
  exact_errata_resolved: true
  deterministic_implementation_update_allowed: true
  external_genesis_start_allowed_after_dual_golden_verification: true
  m3_gates_satisfied_before_external_actors: 14
  child_state_before_external_actors: "NOT_RUN"
```

If any required field remains unknown, set:

```yaml
decision_status: "INCOMPLETE"
external_genesis_start_allowed_after_dual_golden_verification: false
```

Do not use placeholder roots, synthetic keys/seeds, or implementation defaults
to convert an incomplete answer into a formal M2.5 qualification.
