# Hegel Machine Phase-3A M2.5 Exact-Wire Errata Resolution

**Document type**: fail-closed normative errata amendment
**document ID**: `hegel-m25-wire-completion-errata-answer-v1`
**machine freeze ID**: `hegel-freeze-p2b-p3-v1.1.2`
**child DSL**: `hegel-old-dsl-v1.1.0`
**decision status**: `RESOLVED`

This errata finalizes `hegel-freeze-p2b-p3-v1.1.2` before external genesis.
No freeze-version bump is required because no authoritative roots, real
seed/key, signature, marker, or M3 execution identity existed before these
decisions. The child remains `14/24 / NOT_RUN`.

The errata resolves only the twelve wire, DAG, signer-authority, and state
ambiguities. It does not reopen the already reproduced `IdDigestV1` vector,
odd/sink row roots, split quotas, shrink-1 DSL, or closure budgets.

## 0. Required machine-readable answer

```yaml
document_id: hegel-m25-wire-completion-errata-answer-v1
machine_freeze_id: hegel-freeze-p2b-p3-v1.1.2
decision_status: RESOLVED
decisions:
  E1_M3_RUN_GENESIS_SLOT_CARDINALITY:
    selected_option_id: E1_A_EXACTLY_15
    run_output_slot_count: 15
    ordered_run_output_slot_names:
    - canonical_program_archive_root_or_null
    - program_chunk_manifest_root_or_null
    - bucket_accounting_root_or_null
    - outside_program_output_archive_root_or_null
    - outside_output_chunk_manifest_root_or_null
    - outside_match_set_root_or_null
    - outside_role_evaluation_receipt_root_or_null
    - null_program_output_archive_root_or_null
    - null_output_chunk_manifest_root_or_null
    - null_match_set_root_or_null
    - null_role_evaluation_receipt_root_or_null
    - python_enumeration_receipt_root_or_null
    - rust_enumeration_receipt_root_or_null
    - dual_replay_agreement_root_or_null
    - final_state_record_root_or_null
    m3_run_genesis_schema_id: hegel-m3-run-genesis/1
  E2_BRIDGE_TOPOLOGY_ORDER:
    selected_option_id: E2_A_REORDER_EXECUTION_BEFORE_SIGNATURES
    ordered_root_dag_steps:
    - 01_STATIC_NORMATIVE_AND_PROFILE_ROOTS
    - 02_IDENTIFIER_AND_OPERATOR_ROOTS
    - 03_CHILD_DSL_AND_FREEZE_ROOTS
    - 04_TARGET_CONTROL_UNIVERSE_TRUTH_ROOTS
    - 05_SPLIT_CONTRACT_AND_TARGET_BUNDLE_ROOTS
    - 06_ACTOR_TRUST_GENESIS_AND_KEY_MANIFEST_ROOTS
    - 07_CUSTODIAN_SEED_LEDGER_CONTINUITY_AND_ATTESTATION_ROOTS
    - 08_SPLIT_PARTITION_AND_ROLE_BINDING_ROOTS
    - 09_DIAGNOSTIC_FORMAL_BRIDGE_ROOT
    - 10_SHRINK_TRANSITION_AND_IMPLEMENTATION_CONTRACT_ROOTS
    - 11_GENERATE_AND_REGISTER_FRESH_RUN_ID
    - 12_BUILD_M3_EXECUTION_CANDIDATE_ROOT
    - 13_BUILD_BRIDGE_REPLAY_STATEMENT_ROOT
    - 14_COLLECT_THREE_PURPOSE_SPECIFIC_BRIDGE_ENVELOPES
    - 15_BUILD_BRIDGE_ATTESTATION_BUNDLE_ROOT
    - 16_BUILD_FINAL_M3_EXECUTION_MANIFEST_V2_ROOT
    - 17_BUILD_M3_RUN_GENESIS_ROOT_WITH_15_NULL_OUTPUT_SLOTS
    - 18_EVALUATE_GATE_24_WITHOUT_STARTING_M3
    - 19_OPTIONAL_EXPLICIT_M3_START_ACTION
    run_id_registration_step: 11
    bridge_signature_preimage_fields:
    - bridge_replay_statement_root
    - signer_purpose_id
    - signer_key_epoch
  E3_BRIDGE_ENVELOPE_AND_IDENTITY_BINDING:
    selected_option_id: E3_A_STATEMENT_PLUS_THREE_ENVELOPES
    bridge_statement:
      tag: 12558
      tag_hex: '0x310E'
      schema_id: hegel-bridge-replay-statement/1
      fields:
      - version=1
      - tag=0x310E
      - schema_id_bytes
      - run_id_16_bytes
      - diagnostic_formal_bridge_root
      - m3_execution_candidate_root
      - child_dsl_spec_root
      - child_freeze_root
      - actor_trust_genesis_root
      - opaque_id_registry_snapshot_root
      hash_domain: HEGEL/BRIDGE_REPLAY_STATEMENT/V1
    bridge_envelope_cardinality: 3
    bridge_envelope_signer_purposes:
    - 1
    - 2
    - 3
    bridge_attestation_bundle_binding_location: M3ExecutionManifestV2.bridge_attestation_bundle_root
    execution_candidate:
      tag: 12559
      tag_hex: '0x310F'
      schema_id: hegel-m3-execution-candidate/1
      hash_domain: HEGEL/M3_EXECUTION_CANDIDATE/V1
    final_execution_manifest:
      tag: 12560
      tag_hex: '0x3110'
      schema_id: hegel-m3-execution-manifest/2
      hash_domain: HEGEL/M3_EXECUTION_MANIFEST/V2
      required_fields:
      - m3_execution_candidate_root
      - bridge_replay_statement_root
      - bridge_attestation_bundle_root
  E4_MISSING_HASH_DOMAINS:
    selected_option_id: E4_A_DOMAIN_SEPARATED_CONTENT_HASH
    signed_manifest_envelope:
      root_formula: ContentHash('HEGEL/SIGNED_MANIFEST_ENVELOPE/V1', canonical_cbor(SignedManifestEnvelopeV1))
      hash_domain: HEGEL/SIGNED_MANIFEST_ENVELOPE/V1
    m3_dual_replay_agreement:
      root_formula: ContentHash('HEGEL/M3_DUAL_REPLAY_AGREEMENT/V1', canonical_cbor(M3DualReplayAgreementV1))
      hash_domain: HEGEL/M3_DUAL_REPLAY_AGREEMENT/V1
  E5_ACTOR_KEY_TRUST_AND_PURPOSE:
    selected_option_id: E5_A_PURPOSE1_IS_CUSTODIAN_IDENTITY
    actor_key_root_binding_object: ActorTrustGenesisV1
    actor_trust_genesis:
      tag: 12561
      tag_hex: '0x3111'
      schema_id: hegel-actor-trust-genesis/1
      hash_domain: HEGEL/ACTOR_TRUST_GENESIS/V1
      required_purpose_ids:
      - 1
      - 2
      - 3
      - 4
    pinned_genesis_trust_anchor: ActorTrustGenesisV1 root supplied explicitly to the verifier, published in commit B, and
      independently retained outside the repository by the project owner
    custodian_manifest_signer_purpose_id: 1
    purpose_registry:
      1: CUSTODIAN_IDENTITY_AND_BRIDGE_ATTESTER
      2: PYTHON_BRIDGE_ATTESTER
      3: RUST_BRIDGE_ATTESTER
      4: PARENT_ABSENCE_AUDITOR
      5: FINAL_CERTIFICATE_SIGNER_RESERVED_FOR_M4
    noncustodian_envelope_epoch_semantics: The existing epoch field is signer_key_epoch and is resolved against the ActorKeyManifest
      for the envelope signer purpose; custodian_key_epoch is a deprecated documentation alias
    cross_purpose_key_reuse_allowed: false
  E6_LEDGER_CALCULATOR_ACTOR_BOUNDARY:
    selected_option_id: E6_A_CALCULATORS_INSIDE_CUSTODIAN_BOUNDARY
    calculator_actor_boundary: Python and Rust split calculators are constrained child processes of the custodian one-shot
      process, receive the seed only through inherited FD 3, possess no independent actor key, and cannot persist or emit
      the seed
    fd3_access_event_required: false
    gate16_required_ledger_count: 1
    gate16_required_head_rule: ledger_head_root == ledger_genesis_root AND sequence_number == 0 AND no access-granted or revealed
      record exists
    calculator_process_requirements:
    - no network
    - no argv or environment secret
    - no writable repository
    - no secret output or logging
    - close FD 3 after exactly 32 bytes
    - zeroize input buffer before exit
  E7_PARENT_ABSENCE_AUDIT_WIRE:
    selected_option_id: E7_A_VERSIONED_AUDIT_BUNDLE
    audited_path_blob_tag_registry_entry: 12816
    audited_path_blob_tag_hex: '0x3210'
    audited_path_set_row_schema:
      tag: 12816
      schema_id: hegel-audited-path-blob-record/1
      fields:
      - version=1
      - tag=0x3210
      - schema_id_bytes
      - repository_path_alias_id_digest
      - raw_repository_path_utf8_bytes
      - git_object_algorithm_id
      - git_blob_digest
      - file_mode
      - byte_length
      ordering: raw_repository_path_utf8_bytes ascending
      root_rule: RFC6962 over canonical CBOR records
    audited_history_row_schema:
      tag: 12817
      tag_hex: '0x3211'
      schema_id: hegel-audited-history-row/1
      fields:
      - version=1
      - tag=0x3211
      - schema_id_bytes
      - commit_generation
      - repository_commit_id
      - ordered_parent_commit_ids
      - touched_path_set_root
      ordering: commit_generation ascending, then raw repository_commit_id bytes ascending
      root_rule: RFC6962 over canonical CBOR records
    legacy_source_row_schema:
      tag: 12818
      tag_hex: '0x3212'
      schema_id: hegel-legacy-parent-source-row/1
      fields:
      - version=1
      - tag=0x3212
      - schema_id_bytes
      - target_role_id
      - legacy_parent_payload_source_id_digest
      - diagnostic_namespace_id
      - diagnostic_digest
      - source_repository_commit_id
      ordering: target_role_id ascending
      root_rule: RFC6962 over canonical CBOR records
    audit_bundle:
      tag: 12563
      tag_hex: '0x3113'
      schema_id: hegel-parent-absence-audit-bundle/1
      fields:
      - version=1
      - tag=0x3113
      - schema_id_bytes
      - audited_parent_repository_commit_id
      - audited_path_tree_root
      - audited_history_tree_root
      - legacy_source_tree_root
      - audited_path_count
      - audited_history_row_count
      - legacy_source_count
      hash_domain: HEGEL/PARENT_ABSENCE_AUDIT_BUNDLE/V1
    revised_attestation:
      tag: 12564
      tag_hex: '0x3114'
      schema_id: hegel-parent-manifest-absence-attestation/2
      audit_root_binding_field: audit_bundle_root
      hash_domain: HEGEL/PARENT_MANIFEST_ABSENCE_ATTESTATION/V2
    audit_tree_ordering:
    - path rows by raw path bytes
    - history rows by generation then commit bytes
    - legacy source rows by TargetRoleId
    audit_root_binding_field: ParentManifestAbsenceAttestationV2.audit_bundle_root
    legacy_parent_payload_source_ids:
    - target_spec_b491c0a9719fb0279fe02798ede026e440c17a539965514145a7818b15387ac3
    - sink_control_spec_7fd6f9a6e2b4c6eda0c7e1545ad42cb19666743ede8ed87f40d82c0ef46198a0
    audited_parent_commit: fb3a3ee4865a140c558821017ddd3e9a6a99de48
    absence_reason_bitmask: 15
  E8_ROLE_AND_INITIAL_STATE_ENUMS:
    selected_option_id: E8_A_M3STATE0_NEW_TARGET_ROLE_ENUM
    m3_run_genesis_initial_state_registry: M3StateId
    m3_run_genesis_initial_state_value: 0
    target_role_enum_registry:
    - value: 0
      name: INVALID
    - value: 1
      name: OUTSIDE_TARGET
    - value: 2
      name: IN_LANGUAGE_NULL
    - range: 3..32767
      name: RESERVED
    - range: 32768..65535
      name: PRIVATE_EXTENSION_PROHIBITED_IN_AUTHORITATIVE_WIRE
    role_id_field_registry_map:
      DslRoleBindingManifestV1.role_id: TargetRoleId
      SplitAssignmentRowV1.role_id: TargetRoleId
      TargetSpecFormalV1.role_id: TargetRoleId
      ProgramOutputRecordV2.role_id: TargetRoleId
      RoleOutputChunkManifestV2.role_id: TargetRoleId
      MatchRecordV2.role_id: TargetRoleId
      M3RoleEvaluationReceiptV1.role_id: TargetRoleId
      M3RoleAgreementEntryV1.role_id: TargetRoleId
      ArtifactRoleId_fields: remain ArtifactRoleId and must not carry target execution roles
  E10_OPAQUE_ID_REGISTRY_EVIDENCE:
    selected_option_id: E10_A_FORMAL_APPEND_ONLY_REGISTRY_ROOT
    opaque_id_registry_record_schema:
      tag: 12824
      tag_hex: '0x3218'
      schema_id: hegel-opaque-id-registry-record/1
      fields:
      - version=1
      - tag=0x3218
      - schema_id_bytes
      - registry_sequence_number
      - opaque_id_kind_id
      - opaque_id_16_bytes
      - first_seen_object_root
      - first_seen_repository_commit_id
      - created_at_unix_seconds
    opaque_id_kind_registry:
      0: INVALID
      1: RUN_ID
      2: LEDGER_ID
      3..32767: RESERVED
    opaque_id_registry_ordering:
    - registry_sequence_number ascending
    - sequence numbers contiguous from 0
    - opaque_id_kind_id plus opaque_id bytes globally unique
    opaque_id_registry_root_formula: RFC6962 over canonical CBOR OpaqueIdRegistryRecordV1 records in registry-sequence order
    opaque_id_registry_snapshot:
      tag: 12562
      tag_hex: '0x3112'
      schema_id: hegel-opaque-id-registry-snapshot/1
      fields:
      - version=1
      - tag=0x3112
      - schema_id_bytes
      - previous_snapshot_root_or_null
      - registry_tree_root
      - record_count
      - added_record_root
      - repository_commit_id
      hash_domain: HEGEL/OPAQUE_ID_REGISTRY_SNAPSHOT/V1
    opaque_id_registry_binding_location: M3ExecutionCandidateV1.opaque_id_registry_snapshot_root
    duplicate_scope_verification_rule: Replay the snapshot chain from the pinned registry genesis; verify every reachable
      public M3RunGenesis and HiddenAccessLedger ID is represented once; reject a new ID before insertion if it occurs anywhere;
      append with O_EXCL before constructing the execution candidate
  E11_NULL_WITNESS_BINDING_FIELD:
    selected_option_id: E11_A_USE_TARGET_SPEC_AND_BUNDLE_FIELDS
    authoritative_witness_binding_fields:
    - TargetSpecFormalV1.required_witness_ast_hash_or_null
    - TargetBundleV1.null_control_required_witness_ast_hash_or_null
    equality_guard: Both fields must be non-null and byte-identical for TargetRoleId.IN_LANGUAGE_NULL
    dsl_role_binding_schema_change_required: false
  E12_CUSTODIAN_ENVELOPE_COVERAGE:
    selected_option_id: E12_A_SIGN_FOUR_CUSTODIAN_OBJECTS
    custodian_signed_object_tags:
    - 12547
    - 12549
    - 12550
    - 12552
    custodian_signed_object_tags_hex:
    - '0x3103'
    - '0x3105'
    - '0x3106'
    - '0x3108'
    custodian_signature_domain_by_tag:
      '0x3103': HEGEL/CUSTODIAN_SPLIT_SEED_COMMITMENT_SIGNATURE/V1
      '0x3105': HEGEL/CUSTODIAN_BINDING_SIGNATURE/V1
      '0x3106': HEGEL/CUSTODIAN_SEED_CONTINUITY_SIGNATURE/V1
      '0x3108': HEGEL/CUSTODIAN_LEDGER_GENESIS_SIGNATURE/V1
    custodian_envelope_root_binding_locations:
    - AttestationBundleV1.entries[purpose_id=1,object_tag=0x3103]
    - AttestationBundleV1.entries[purpose_id=1,object_tag=0x3105]
    - AttestationBundleV1.entries[purpose_id=1,object_tag=0x3106]
    - AttestationBundleV1.entries[purpose_id=1,object_tag=0x3108]
    - M3ExecutionCandidateV1.custodian_attestation_bundle_root
    required_signature_count_by_tag:
      '0x3103': 1
      '0x3105': 1
      '0x3106': 1
      '0x3108': 1
    signer_purpose_id: 1
  E9_ROOT_PREIMAGES_INSTANCE_IDS_AND_PATH_ALIAS:
    selected_option_id: E9_A_FORMAL_PREIMAGE_AND_ALIAS_REGISTRY
    formal_root_preimage_registry:
    - root: amendment_document_root
      object: NormativeDocumentBlobV1
      tag: '0x3001'
      domain: HEGEL/NORMATIVE_DOCUMENT/V1
    - root: parent_freeze_root
      object: FreezeSpecV1
      tag: '0x3002'
      domain: HEGEL/FREEZE_SPEC/V1
    - root: child_freeze_root
      object: FreezeSpecV1
      tag: '0x3002'
      domain: HEGEL/FREEZE_SPEC/V1
    - root: child_dsl_spec_root
      object: DslSpecV1
      tag: '0x3003'
      domain: HEGEL/DSL_SPEC/V1
    - root: operator_semantics_root
      object: OperatorSemanticsEntryV1[]
      tag: '0x3205'
      domain: RFC6962
    - root: identifier_registry_root
      object: IdentifierRegistryEntryV1[]
      tag: '0x3204'
      domain: RFC6962
    - root: canonical_ast_schema_root
      object: CanonicalAstProfileSpecV1
      tag: '0x3019'
      domain: HEGEL/CANONICAL_AST_PROFILE/V1
    - root: canonical_cbor_profile_root
      object: CanonicalCborProfileSpecV1
      tag: '0x301A'
      domain: HEGEL/CANONICAL_CBOR_PROFILE/V1
    - root: phase2b_contract_root
      object: Phase2BContractSpecV1
      tag: '0x301B'
      domain: HEGEL/PHASE2B_CONTRACT/V1
    - root: mdl_code_table_root
      object: MdlCodeTableSpecV1
      tag: '0x301C'
      domain: HEGEL/MDL_CODE_TABLE/V1
    - root: split_contract_root
      object: SplitContractV1
      tag: '0x3004'
      domain: HEGEL/SPLIT_CONTRACT/V1
    - root: target_bundle_root
      object: TargetBundleV1
      tag: '0x3005'
      domain: HEGEL/TARGET_BUNDLE/V1
    - root: approval_evidence_root
      object: ApprovalEvidenceBundleV1
      tag: '0x3006'
      domain: HEGEL/APPROVAL_EVIDENCE_BUNDLE/V1
    - root: replacement_policy_root
      object: ReplacementPolicyV1
      tag: '0x3007'
      domain: HEGEL/REPLACEMENT_POLICY/V1
    - root: split_spec_freeze_root
      object: SplitSpecFreezeV1
      tag: '0x3008'
      domain: HEGEL/SPLIT_SPEC_FREEZE/V1
    - root: removed_registry_entry_root
      object: IdentifierRegistryEntryV1[tombstone subset]
      tag: '0x3204'
      domain: RFC6962
    - root: surviving_registry_entry_root
      object: IdentifierRegistryEntryV1[active subset]
      tag: '0x3204'
      domain: RFC6962
    - root: tombstone_policy_root
      object: TombstonePolicyV1
      tag: '0x3009'
      domain: HEGEL/TOMBSTONE_POLICY/V1
    - root: cross_dsl_hash_policy_root
      object: CrossDslHashPolicyV1
      tag: '0x300A'
      domain: HEGEL/CROSS_DSL_HASH_POLICY/V1
    - root: fallback_registry_root
      object: FallbackRegistryV1
      tag: '0x300B'
      domain: HEGEL/FALLBACK_REGISTRY/V1
    - root: python_implementation_binding_root
      object: ImplementationBindingV1
      tag: '0x300C'
      domain: HEGEL/IMPLEMENTATION_BINDING/V1
    - root: rust_implementation_binding_root
      object: ImplementationBindingV1
      tag: '0x300C'
      domain: HEGEL/IMPLEMENTATION_BINDING/V1
    - root: traversal_contract_root
      object: TraversalContractV1
      tag: '0x300D'
      domain: HEGEL/TRAVERSAL_CONTRACT/V1
    - root: bucket_accounting_contract_root
      object: BucketAccountingContractV1
      tag: '0x300E'
      domain: HEGEL/BUCKET_ACCOUNTING_CONTRACT/V1
    - root: program_archive_contract_root
      object: ProgramArchiveContractV1
      tag: '0x300F'
      domain: HEGEL/PROGRAM_ARCHIVE_CONTRACT/V1
    - root: output_archive_contract_root
      object: OutputArchiveContractV1
      tag: '0x3010'
      domain: HEGEL/OUTPUT_ARCHIVE_CONTRACT/V1
    - root: state_machine_contract_root
      object: StateMachineContractV1
      tag: '0x3011'
      domain: HEGEL/M3_STATE_MACHINE_CONTRACT/V1
    - root: row_transform_spec_root
      object: RowTransformSpecV1
      tag: '0x3012'
      domain: HEGEL/ROW_TRANSFORM_SPEC/V1
    - root: hidden_artifact_scope_root
      object: HiddenArtifactScopeV1
      tag: '0x301E'
      domain: HEGEL/HIDDEN_ARTIFACT_SCOPE/V1
    - root: actor_trust_genesis_root
      object: ActorTrustGenesisV1
      tag: '0x3111'
      domain: HEGEL/ACTOR_TRUST_GENESIS/V1
    - root: opaque_id_registry_snapshot_root
      object: OpaqueIdRegistrySnapshotV1
      tag: '0x3112'
      domain: HEGEL/OPAQUE_ID_REGISTRY_SNAPSHOT/V1
    instance_machine_id_catalog:
    - repo-path:m25-wire-completion-amendment-v1
    - profile:hegel-canonical-ast-v1
    - profile:hegel-cbor-det-v1
    - contract:phase2b-v1
    - table:hegel-mdl-prefix-v1.0.0
    - rule:per-stratum-rank-then-quota-v1
    - rule:universe-index-within-partition-v1
    - rule:new-target-new-split-first-instantiation-v1
    - rule:rank-digest-then-input-hash-v1
    - rule:canonical-program-order-v1
    - rule:bucket-order-v1
    - profile:hegel-rfc6962-v1
    - codec:identity-length-framed-v1
    - profile:hegel-undefined-bitmap-v1
    - policy:hidden-artifact-split-only-v1
    - policy:opaque-id-global-project-v1
    - state-machine:m3-v1
    repository_path_alias_rule: RepositoryPathAliasRecordV1 binds IdDigestV1(legal ASCII alias) to exact raw UTF-8 repository
      path bytes; records sort by alias digest and the registry root is RFC6962
    repository_path_alias_record:
      tag: 12819
      tag_hex: '0x3213'
      schema_id: hegel-repository-path-alias-record/1
      fields:
      - version=1
      - tag=0x3213
      - schema_id_bytes
      - path_alias_id_digest
      - raw_repository_path_utf8_bytes
      - repository_commit_id
    amendment_document_path_alias: repo-path:m25-wire-completion-amendment-v1
    amendment_document_raw_path: Hegel Machine/docs/Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md
    source_and_dependency_root_row_schemas:
    - name: SourceFileRecordV1
      tag: '0x3215'
      schema: '[1,0x3215,b''hegel-source-file-record/1'',path_alias_id_digest,raw_path_bytes,git_blob_algorithm_id,git_blob_digest,file_mode,byte_length]'
      ordering: raw_path_bytes ascending
      root: RFC6962
    - name: DependencyLockRecordV1
      tag: '0x3216'
      schema: '[1,0x3216,b''hegel-dependency-lock-record/1'',ecosystem_id,package_name_id_digest,version_id_digest,source_id_digest,lock_entry_digest]'
      ordering: (ecosystem_id,package_name_id_digest,version_id_digest) ascending
      root: RFC6962
    state_machine_nested_row_schemas:
    - name: LegalTransitionRowV1
      tag: '0x3217'
      schema: '[1,0x3217,b''hegel-legal-transition-row/1'',from_state_id,from_phase_id,to_state_id,to_phase_id,allowed_reason_ids]'
      ordering: (from_state_id,from_phase_id,to_state_id,to_phase_id) ascending
    - name: StaticRoleMetadataV1
      tag: '0x301D'
      schema: '[1,0x301D,b''hegel-static-role-metadata/1'',input_signature_id,role_ids,quantity_ids,scope_ids,signed_orientations,metadata_rule_id_digest]'
      hash_domain: HEGEL/STATIC_ROLE_METADATA/V1
    m3_run_state_exact_prefix:
    - 1
    - '0x3301'
    - b'hegel-m3-run-state-record/1'
    input_signature_static_role_metadata_schema:
      object: StaticRoleMetadataV1
      odd_profile: empty role, quantity, scope, and orientation arrays
      sink_profile: four observed channels with signed orientations [1,1,-1,-1], frozen quantity q0, and all-observed scope
    target_claim_level_enum_registry:
      0: INVALID
      1: FALSE_INVENTION_NULL_ONLY
      2: MECHANISM_SPECIFIC_RECOVERY
      3: OUTSIDE_TARGET_CANDIDATE
      4..32767: RESERVED
    mismatch_kind_enum_registry:
      0: INVALID
      1: CANONICAL_PROGRAM_COUNT
      2: CANONICAL_PROGRAM_ARCHIVE_ROOT
      3: PROGRAM_OUTPUT_ARCHIVE_ROOT
      4: BUCKET_ACCOUNTING_ROOT
      5: FIRST_OUT_OF_BUDGET_WITNESS
      6: ROLE_MATCH_SET
      7: RECEIPT_FIELD_PRESENCE
      8: EXECUTION_ENVIRONMENT_BINDING
      9..32767: RESERVED
    split_contract_numeric_rule_registry:
      AssignmentOrderingRuleId:
        0: INVALID
        1: PER_STRATUM_RANK_THEN_QUOTA
        2: UNIVERSE_INDEX_WITHIN_PARTITION
      FallbackSplitPolicyId:
        0: INVALID
        1: NEW_TARGET_NEW_SPLIT_FIRST_INSTANTIATION
      RankTieBreakRuleId:
        0: INVALID
        1: RANK_DIGEST_THEN_CANONICAL_INPUT_HASH
    traversal_and_bucket_field_id_registries:
      TraversalFieldId:
        1: AST_DEPTH
        2: AST_NODE_COUNT
        3: OUTPUT_SORT_ID
        4: ROOT_OPERATOR_ID
        5: CANONICAL_AST_CBOR_BYTES
      BucketFieldId:
        1: OUTPUT_SORT_ID
        2: AST_DEPTH
        3: AST_NODE_COUNT
      AccountingCounterFieldId:
        1: RAW_OPERATOR_APPLICATIONS
        2: ACCEPTED_CANONICAL_PROGRAMS
        3: SYNTACTIC_DUPLICATES
        4: TYPE_REJECTIONS
        5: STRUCTURAL_LIMIT_REJECTIONS
        6: REWRITE_COLLAPSES
      AccountingInvariantId:
        1: SUM_ACCEPTED_EQUALS_RECEIPT_CANONICAL_PROGRAM_COUNT
        2: SUM_RAW_APPLICATIONS_EQUALS_RECEIPT_RAW_APPLICATION_COUNT
        3: NONNULL_PROGRAM_INDEX_RANGE_SIZE_EQUALS_ACCEPTED_COUNT
post_decision_state:
  exact_errata_resolved: true
  deterministic_implementation_update_allowed: true
  external_genesis_start_allowed_after_dual_golden_verification: true
  m3_gates_satisfied_before_external_actors: 14
  child_state_before_external_actors: NOT_RUN
  authoritative_formal_root_generation_before_external_genesis: false
  os_csprng_call_before_dual_errata_vectors_pass: false
  marker_creation_before_dual_errata_vectors_pass: false

```


## 1. Normative notes resolving the cyclic dependencies

### 1.1 `M3ExecutionCandidateV1`

```text
[
  1,
  0x310F,
  b"hegel-m3-execution-candidate/1",
  run_id_16_bytes,
  child_dsl_spec_root,
  child_freeze_root,
  approval_manifest_root,
  shrink_transition_root,
  operator_semantics_root,
  identifier_registry_root,
  canonical_ast_schema_root,
  canonical_cbor_profile_root,
  diagnostic_formal_bridge_root,
  outside_target_binding_manifest_root,
  null_control_binding_manifest_root,
  split_binding_manifest_root,
  custodian_binding_manifest_root,
  seed_continuity_manifest_root,
  custodian_attestation_bundle_root,
  parent_absence_attestation_root,
  hidden_access_ledger_genesis_root,
  hidden_access_ledger_head_root,
  opaque_id_registry_snapshot_root,
  actor_trust_genesis_root,
  outside_target_universe_root,
  outside_target_truth_root,
  null_control_universe_root,
  null_control_truth_root,
  outside_discovery_split_root,
  outside_validation_split_root,
  outside_sealed_split_root,
  null_discovery_split_root,
  null_validation_split_root,
  null_sealed_split_root,
  canonical_program_budget,
  raw_operator_application_cap,
  records_per_chunk,
  equivalence_mode_id,
  python_implementation_binding_root,
  rust_implementation_binding_root,
  traversal_contract_root,
  bucket_accounting_contract_root,
  program_archive_contract_root,
  output_archive_contract_root,
  state_machine_contract_root,
  created_at_unix_seconds,
  repository_commit_id
]
```

```text
candidate_root =
ContentHash(
  "HEGEL/M3_EXECUTION_CANDIDATE/V1",
  canonical_cbor(M3ExecutionCandidateV1)
)
```

### 1.2 `BridgeReplayStatementV1`

```text
[
  1,
  0x310E,
  b"hegel-bridge-replay-statement/1",
  run_id_16_bytes,
  diagnostic_formal_bridge_root,
  m3_execution_candidate_root,
  child_dsl_spec_root,
  child_freeze_root,
  actor_trust_genesis_root,
  opaque_id_registry_snapshot_root
]
```

```text
statement_root =
ContentHash(
  "HEGEL/BRIDGE_REPLAY_STATEMENT/V1",
  canonical_cbor(BridgeReplayStatementV1)
)
```

Each bridge attester signs:

```text
UTF8("HEGEL/BRIDGE_ATTESTATION_SIGNATURE/V1")
|| 0x00
|| statement_root
```

The execution candidate therefore exists before signatures, while the final
execution manifest exists after signatures. This removes the original cycle.

### 1.3 `M3ExecutionManifestV2`

```text
[
  1,
  0x3110,
  b"hegel-m3-execution-manifest/2",
  run_id_16_bytes,
  m3_execution_candidate_root,
  bridge_replay_statement_root,
  bridge_attestation_bundle_root,
  actor_trust_genesis_root,
  opaque_id_registry_snapshot_root,
  created_at_unix_seconds,
  repository_commit_id
]
```

```text
execution_manifest_root =
ContentHash(
  "HEGEL/M3_EXECUTION_MANIFEST/V2",
  canonical_cbor(M3ExecutionManifestV2)
)
```

`M3RunGenesisV1.execution_manifest_root` binds this V2 root.

### 1.4 `ActorTrustGenesisV1`

```text
[
  1,
  0x3111,
  b"hegel-actor-trust-genesis/1",
  trust_genesis_id_16_bytes,
  [
    [purpose_id, actor_key_manifest_root],
    ...
  ],
  purpose_key_policy_root,
  created_at_unix_seconds,
  repository_commit_id
]
```

Entries are ordered by `purpose_id`. Its root is the trust anchor. A file name,
key label, repository URL, or implementation source root is not a trust anchor.

---

## 2. Gate 24 correction

The authoritative gate name becomes:

```text
M3_EXECUTION_MANIFEST_ROOT_NON_NULL_AND_15_OUTPUT_ROOTS_NULL
```

Pass requires:

```yaml
m3_execution_manifest_v2_root_non_null: true
m3_run_genesis_v1_root_non_null: true
m3_run_genesis_initial_state: "M3StateId.NOT_RUN = 0"
run_output_slot_count: 15
all_run_output_slots_null: true
run_id_registered_in_bound_opaque_id_snapshot: true
bridge_envelope_count: 3
bridge_signer_purposes_exactly: [1, 2, 3]
```

Passing Gate 24 means:

```yaml
m3_entry_qualified: true
m3_entry_allowed: true
m3_run_started: false
child_state: "NOT_RUN"
```

It does not create the `NOT_RUN -> RUNNING` transition.

---

## 3. External-genesis start guard

The external one-shot process may create a marker or call the OS CSPRNG only
after all of the following are true:

```yaml
external_genesis_start_guard:
  errata_document_in_commit_A: true
  python_errata_vectors_pass: true
  rust_errata_vectors_pass: true
  python_rust_canonical_bytes_equal: true
  python_rust_error_codes_equal: true
  actor_trust_genesis_schema_frozen: true
  append_only_id_registry_schema_frozen: true
  parent_audit_bundle_schema_frozen: true
  bridge_statement_and_execution_v2_schema_frozen: true
  secrets_absent_from_repository: true
```

Otherwise the process must return before any side effect:

```text
FAIL_M25_EXACT_ERRATA_REQUIRED
```

---

## 4. Claim boundary

After deterministic implementation of this errata, but before external actors,
the strongest allowed statement is:

> Python and Rust implement the same completed M2.5 wire contract, including
> the acyclic bridge/execution topology, actor trust anchor, versioned parent
> audit bundle, formal append-only opaque-ID registry, and exactly fifteen M3
> output slots.

After external genesis and 24/24 gates, the strongest allowed statement is:

> The child DSL has a replay-equal formal input identity, externally
> instantiated split custody, independently attested parent-manifest absence,
> and a qualified but not yet started M3 execution identity.

Neither state permits:

- `COMPLETE`;
- `DSL_TOO_LARGE` for the child DSL;
- odd-target outside verdict;
- sink-control in-language verdict;
- target synthesis;
- outside/MDL certificate;
- ACTIVE promotion.

The next explicit action is a separate operator command:

```text
phase3-m3-start
```

which alone may create:

```text
NOT_RUN/NONE
-> RUNNING/CANONICAL_ENUMERATION
```
