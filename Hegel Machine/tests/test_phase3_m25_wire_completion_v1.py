from __future__ import annotations

from itertools import product

import pytest

from hegel_machine.phase3_m25_wire_v1 import (
    AUTHORITATIVE_BLOCKING_GAPS,
    AUTHORITATIVE_MIN_TIMESTAMP,
    FAIL_M25_NORMATIVE_GAP,
    FORMAL_SCHEMA_REGISTRY,
    M25WireError,
    M3_RUN_OUTPUT_ROOTS,
    NUMERIC_ENUM_REGISTRIES,
    OBJECT_TAGS,
    NumericEnumRegistry,
    build_formal_object,
    candidate_content_root,
    candidate_record_tree_root,
    decode_formal_object,
    encode_formal_object,
    formal_content_root,
    git_sha1_commit_id,
    id_digest_v1,
    validate_opaque_id128_v1,
    validate_timestamp_ordering_v1,
    validate_timestamp_v1,
)


ROOT = bytes(range(32))
OTHER_ROOT = bytes(reversed(range(32)))
RUN_ID = bytes(range(16))
COMMIT = git_sha1_commit_id(bytes(range(20)))


def _error_code(callable_object: object, *args: object, **kwargs: object) -> str:
    assert callable(callable_object)
    with pytest.raises(M25WireError) as error:
        callable_object(*args, **kwargs)
    return error.value.code


def _split_contract_fields() -> dict[str, object]:
    return {
        "split_contract_version_id_digest": ROOT,
        "split_algorithm_spec_root": ROOT,
        "hkdf_profile_id_digest": ROOT,
        "rank_hmac_profile_id_digest": ROOT,
        "exhaustive_partition_required": True,
        "odd_stratum_quota_table": (
            (1, 16, 6, 3, 7),
            (2, 16, 6, 3, 7),
            (3, 32, 13, 6, 13),
            (4, 32, 13, 6, 13),
            (5, 64, 26, 13, 25),
            (6, 64, 26, 13, 25),
            (7, 128, 51, 26, 51),
            (8, 128, 51, 26, 51),
        ),
        "sink_stratum_quota_table": (
            (9, 15, 7, 4, 4),
            (10, 18, 8, 4, 6),
            (11, 19, 9, 4, 6),
            (12, 18, 8, 4, 6),
            (13, 15, 7, 4, 4),
        ),
        "assignment_ordering_rule_id": 1,
        "fallback_split_policy_id": 1,
        "hidden_artifact_scope_root": ROOT,
    }


def _m3_genesis_fields() -> dict[str, object]:
    fields: dict[str, object] = {
        "run_id": RUN_ID,
        "execution_manifest_root": ROOT,
        "initial_state_id": 0,
        "created_at_unix_seconds": 1,
        "repository_commit_id": COMMIT,
    }
    for output_name in M3_RUN_OUTPUT_ROOTS:
        fields[f"{output_name}_or_null"] = None
    return fields


def test_id_digest_is_bit_exact_and_never_normalizes() -> None:
    assert id_digest_v1("hegel-old-dsl-v1.1.0").hex() == (
        "49022ed9fa53522e10dd60ce5da983a4ac0be2d7bc8c7737f6d5ae1dc88c4703"
    )
    assert id_digest_v1("A/B") != id_digest_v1("a/b")
    assert _error_code(id_digest_v1, "") == "REJECT_MACHINE_ID_LENGTH"
    assert _error_code(id_digest_v1, "a" * 257) == "REJECT_MACHINE_ID_LENGTH"
    assert _error_code(id_digest_v1, "Hegel Machine/docs/x") == "REJECT_MACHINE_ID_SYNTAX"
    assert _error_code(id_digest_v1, "黑格尔") == "REJECT_MACHINE_ID_NON_ASCII"
    assert "NormativeDocumentPathMachineId" in AUTHORITATIVE_BLOCKING_GAPS


def test_timestamp_and_opaque_id_profiles_fail_closed() -> None:
    assert validate_timestamp_v1(0) == 0
    assert validate_timestamp_v1(AUTHORITATIVE_MIN_TIMESTAMP, authoritative=True, verifier_unix_seconds=AUTHORITATIVE_MIN_TIMESTAMP) == AUTHORITATIVE_MIN_TIMESTAMP
    assert _error_code(validate_timestamp_v1, -1) == "REJECT_TIMESTAMP_OUT_OF_RANGE"
    assert _error_code(
        validate_timestamp_v1,
        0,
        authoritative=True,
        verifier_unix_seconds=AUTHORITATIVE_MIN_TIMESTAMP,
    ) == "FAIL_AUTHORITATIVE_TIMESTAMP_ZERO"
    assert _error_code(
        validate_timestamp_v1,
        AUTHORITATIVE_MIN_TIMESTAMP + 301,
        authoritative=True,
        verifier_unix_seconds=AUTHORITATIVE_MIN_TIMESTAMP,
    ) == "FAIL_TIMESTAMP_EXCESSIVELY_FUTURE"
    validate_timestamp_ordering_v1(1, 2)
    assert _error_code(validate_timestamp_ordering_v1, 2, 1) == "FAIL_TIMESTAMP_ORDERING"

    assert validate_opaque_id128_v1(RUN_ID) == RUN_ID
    assert _error_code(validate_opaque_id128_v1, bytes(16)) == "FAIL_OPAQUE_ID_ALL_ZERO"
    assert _error_code(validate_opaque_id128_v1, RUN_ID, seen={RUN_ID}) == "FAIL_OPAQUE_ID_ALREADY_USED"


def test_numeric_enum_registry_is_complete_and_context_strict() -> None:
    required = {
        "InputSignatureId",
        "SortId",
        "RegistryKindId",
        "RegistryEntryStateId",
        "OperatorClassId",
        "OperatorAdmissionStateId",
        "UndefinedSemanticsId",
        "ArtifactRoleId",
        "DiagnosticNamespaceId",
        "FormalObjectKindId",
        "DiagnosticProfileId",
        "FormalProfileId",
        "StratumId",
        "PartitionId",
        "EquivalenceModeId",
        "ImplementationId",
        "ParentStatusId",
        "ChildInitialStateId",
        "M3TransitionReasonId",
        "M3ClosureStatusId",
        "M3StateId",
        "M3RunningPhaseId",
        "RoleAgreementStatusId",
    }
    assert required <= set(NUMERIC_ENUM_REGISTRIES)
    assert NUMERIC_ENUM_REGISTRIES["M3StateId"].validate(0, field="state") == 0
    assert _error_code(
        NUMERIC_ENUM_REGISTRIES["SortId"].validate, 0, field="sort"
    ) == "REJECT_UNKNOWN_ENUM_VALUE"
    assert _error_code(
        NUMERIC_ENUM_REGISTRIES["SortId"].validate, 22, field="sort"
    ) == "REJECT_UNKNOWN_ENUM_VALUE"
    assert _error_code(
        NUMERIC_ENUM_REGISTRIES["SortId"].validate, 32768, field="sort"
    ) == "REJECT_UNKNOWN_ENUM_VALUE"
    tombstoned = NumericEnumRegistry(
        "test", {1: "ACTIVE", 2: "REMOVED"}, tombstones=frozenset({2})
    )
    assert _error_code(tombstoned.validate, 2, field="test") == "REJECT_TOMBSTONED_ENUM_VALUE"


def test_v112_tag_and_schema_registry_contains_every_addition() -> None:
    assert len(OBJECT_TAGS) == 58
    expected = {
        "NormativeDocumentBlobV1": 0x3001,
        "ExecutionEnvironmentSpecV1": 0x3016,
        "CustodianBindingCoreV1": 0x310B,
        "ActorKeyManifestV1": 0x310C,
        "AttestationBundleV1": 0x310D,
        "MismatchRecordV1": 0x320D,
        "PartialDiagnosticBundleV1": 0x320E,
        "M3RunGenesisV1": 0x3300,
        "OddInputV1": 0x3401,
        "SinkInputV1": 0x3402,
    }
    for name, tag in expected.items():
        assert OBJECT_TAGS[name] == tag
        assert FORMAL_SCHEMA_REGISTRY[name].tag == tag
    assert FORMAL_SCHEMA_REGISTRY["FreezeSpecV1"].hash_domain == "HEGEL/FREEZE_SPEC/V1"
    assert FORMAL_SCHEMA_REGISTRY["BucketAccountingRecordV1"].ordering_fields == ("bucket_index",)
    assert FORMAL_SCHEMA_REGISTRY["MismatchRecordV1"].ordering_fields == ("mismatch_index",)


def test_core_content_schema_round_trip_and_candidate_root() -> None:
    fields = {
        "freeze_version_id_digest": id_digest_v1("hegel-freeze-p2b-p3-v1.1.2"),
        "parent_freeze_root_or_null": ROOT,
        "child_dsl_spec_root": ROOT,
        "phase2b_contract_root": ROOT,
        "canonical_ast_schema_root": ROOT,
        "canonical_cbor_profile_root": ROOT,
        "mdl_code_table_root": ROOT,
        "amendment_document_root": OTHER_ROOT,
        "effective_repository_commit_id": COMMIT,
    }
    payload = encode_formal_object("FreezeSpecV1", fields)
    decoded = decode_formal_object(payload, expected_name="FreezeSpecV1")
    assert decoded.value[:3] == (1, 0x3002, b"hegel-freeze-spec/1")
    assert encode_formal_object(decoded.schema.name, decoded.fields) == payload
    assert len(candidate_content_root("FreezeSpecV1", fields)) == 32
    assert _error_code(formal_content_root, "FreezeSpecV1", fields) == FAIL_M25_NORMATIVE_GAP


def test_split_contract_requires_the_exact_exhaustive_quota_tables() -> None:
    fields = _split_contract_fields()
    build_formal_object("SplitContractV1", fields)

    fields["exhaustive_partition_required"] = False
    assert _error_code(build_formal_object, "SplitContractV1", fields) == "REJECT_M25_FIELD_VALUE"

    fields = _split_contract_fields()
    odd = list(fields["odd_stratum_quota_table"])
    odd[0] = (1, 16, 7, 3, 6)
    fields["odd_stratum_quota_table"] = tuple(odd)
    assert _error_code(build_formal_object, "SplitContractV1", fields) == "REJECT_M25_FIELD_VALUE"


def test_attestation_bundle_has_exact_rows_enums_and_order() -> None:
    fields = {"attestations": ((1, ROOT, ROOT), (2, ROOT, OTHER_ROOT), (3, OTHER_ROOT, ROOT))}
    build_formal_object("AttestationBundleV1", fields)
    assert len(candidate_content_root("AttestationBundleV1", fields)) == 32

    fields["attestations"] = ((2, ROOT, OTHER_ROOT), (1, ROOT, ROOT))
    assert _error_code(build_formal_object, "AttestationBundleV1", fields) == "REJECT_M25_FIELD_VALUE"
    fields["attestations"] = ((0, ROOT, ROOT),)
    assert _error_code(build_formal_object, "AttestationBundleV1", fields) == "REJECT_UNKNOWN_ENUM_VALUE"


def test_bucket_mismatch_and_partial_wires_are_no_longer_normative_gaps() -> None:
    bucket = {
        "bucket_index": 0,
        "output_sort_id": 2,
        "ast_depth": 0,
        "ast_node_count": 1,
        "raw_operator_applications": 0,
        "accepted_canonical_programs": 1,
        "syntactic_duplicates": 0,
        "type_rejections": 0,
        "structural_limit_rejections": 0,
        "rewrite_collapses": 0,
        "first_program_index_or_null": 0,
        "last_program_index_or_null": 0,
    }
    assert len(candidate_record_tree_root("BucketAccountingRecordV1", [bucket])) == 32

    mismatch = {
        "mismatch_index": 0,
        "mismatch_kind_id": 1,
        "python_object_root_or_null": ROOT,
        "rust_object_root_or_null": OTHER_ROOT,
        "affected_program_index_or_null": 0,
        "diagnostic_detail_digest": ROOT,
    }
    assert len(candidate_record_tree_root("MismatchRecordV1", [mismatch])) == 32

    partial = {
        "run_id": RUN_ID,
        "implementation_id": 1,
        "terminal_failure_code_id_digest": ROOT,
        "completed_bucket_count": 0,
        "partial_bucket_accounting_root_or_null": None,
        "partial_log_digest": ROOT,
        "authoritative_claim_allowed": False,
    }
    build_formal_object("PartialDiagnosticBundleV1", partial)
    partial["authoritative_claim_allowed"] = True
    assert _error_code(build_formal_object, "PartialDiagnosticBundleV1", partial) == "REJECT_M25_FIELD_VALUE"


def test_odd_and_sink_formal_rows_match_all_normative_roots() -> None:
    odd_universe: list[dict[str, object]] = []
    odd_truth: list[dict[str, object]] = []
    index = 0
    for set_size in range(5, 9):
        for bits in product((0, 1), repeat=set_size):
            input_fields = {"set_size": set_size, "bits": bits}
            input_object = build_formal_object("OddInputV1", input_fields)
            input_hash = candidate_content_root("OddInputV1", input_fields)
            odd_universe.append(
                {"universe_index": index, "input_signature_id": 1, "canonical_input_object": input_object}
            )
            odd_truth.append(
                {"universe_index": index, "canonical_input_hash": input_hash, "target_output": sum(bits) % 2}
            )
            index += 1
    assert encode_formal_object("OddInputV1", {"set_size": 5, "bits": (0, 0, 0, 0, 0)}).hex() == (
        "850119340151686567656c2d6f64642d696e7075742f3105850000000000"
    )
    assert candidate_record_tree_root("BoundedUniverseRowV1", odd_universe).hex() == (
        "b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05"
    )
    assert candidate_record_tree_root("TargetTruthRowV1", odd_truth).hex() == (
        "f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506"
    )

    sink_universe: list[dict[str, object]] = []
    sink_truth: list[dict[str, object]] = []
    index = 0
    for a in range(5):
        for b in range(5):
            for c in range(5):
                d = a + b - c
                if 0 <= d <= 4:
                    input_fields = {"a": a, "b": b, "c": c, "d": d}
                    input_object = build_formal_object("SinkInputV1", input_fields)
                    input_hash = candidate_content_root("SinkInputV1", input_fields)
                    sink_universe.append(
                        {"universe_index": index, "input_signature_id": 2, "canonical_input_object": input_object}
                    )
                    sink_truth.append(
                        {"universe_index": index, "canonical_input_hash": input_hash, "target_output": 1}
                    )
                    index += 1
    assert index == 85
    assert candidate_record_tree_root("BoundedUniverseRowV1", sink_universe).hex() == (
        "1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5"
    )
    assert candidate_record_tree_root("TargetTruthRowV1", sink_truth).hex() == (
        "9c0f5d75ea3c31f6cb1ea9917346a7a3f480ae9ce0ac0cb3bb21aac9d3bd7808"
    )


@pytest.mark.parametrize("name", ("BoundedUniverseRowV1", "TargetTruthRowV1"))
def test_universe_and_truth_candidate_roots_reject_duplicate_or_gapped_indices(
    name: str,
) -> None:
    if name == "BoundedUniverseRowV1":
        def row(index: int) -> dict[str, object]:
            input_fields = {"set_size": 5, "bits": (0, 0, 0, 0, index % 2)}
            return {
                "universe_index": index,
                "input_signature_id": 1,
                "canonical_input_object": build_formal_object("OddInputV1", input_fields),
            }
    else:
        def row(index: int) -> dict[str, object]:
            return {
                "universe_index": index,
                "canonical_input_hash": bytes([index % 256]) * 32,
                "target_output": index % 2,
            }

    assert _error_code(candidate_record_tree_root, name, [row(0), row(0)]) == (
        "FAIL_UNIVERSE_INDEX_DUPLICATE"
    )
    assert _error_code(candidate_record_tree_root, name, [row(0), row(2)]) == (
        "FAIL_UNIVERSE_INDEX_GAP"
    )
    assert _error_code(candidate_record_tree_root, name, [row(1), row(0)]) == (
        "FAIL_ROW_ORDERING"
    )


def test_single_tree_candidate_api_does_not_claim_cross_tree_hash_binding() -> None:
    input_fields = {"set_size": 5, "bits": (0, 0, 0, 0, 0)}
    universe = [{
        "universe_index": 0,
        "input_signature_id": 1,
        "canonical_input_object": build_formal_object("OddInputV1", input_fields),
    }]
    truth_with_unpaired_hash = [{
        "universe_index": 0,
        "canonical_input_hash": OTHER_ROOT,
        "target_output": 0,
    }]
    assert len(candidate_record_tree_root("BoundedUniverseRowV1", universe)) == 32
    assert len(candidate_record_tree_root("TargetTruthRowV1", truth_with_unpaired_hash)) == 32
    assert "does not claim to validate" in candidate_record_tree_root.__doc__


def test_input_and_row_guards_reject_wrong_types_or_role_bindings() -> None:
    assert _error_code(
        build_formal_object, "OddInputV1", {"set_size": 5, "bits": (0, 1)}
    ) == "REJECT_M25_FIELD_VALUE"
    assert _error_code(
        build_formal_object, "SinkInputV1", {"a": 0, "b": 0, "c": 0, "d": 1}
    ) == "REJECT_M25_FIELD_VALUE"
    odd = build_formal_object("OddInputV1", {"set_size": 5, "bits": (0, 0, 0, 0, 0)})
    assert _error_code(
        build_formal_object,
        "BoundedUniverseRowV1",
        {"universe_index": 0, "input_signature_id": 2, "canonical_input_object": odd},
    ) == "REJECT_M25_FIELD_VALUE"
    assert _error_code(
        build_formal_object,
        "TargetTruthRowV1",
        {"universe_index": 0, "canonical_input_hash": ROOT, "target_output": True},
    ) == "FAIL_TARGET_OUTPUT_TYPE"


def test_parent_mask_and_approval_child_root_follow_v112() -> None:
    absence = {
        "parent_dsl_version_digest": ROOT,
        "parent_freeze_version_digest": ROOT,
        "parent_repository_commit_id": COMMIT,
        "audited_source_tree_root": ROOT,
        "audited_path_set_root": ROOT,
        "legacy_parent_payload_source_id_digest": ROOT,
        "absence_reason_bitmask": 0b1111,
        "auditor_key_id": bytes(range(16)),
        "audited_at_unix_seconds": 1,
    }
    build_formal_object("ParentManifestAbsenceAttestationV1", absence)
    absence["absence_reason_bitmask"] = 0b11
    assert _error_code(
        build_formal_object, "ParentManifestAbsenceAttestationV1", absence
    ) == "REJECT_M25_FIELD_VALUE"

    approval = {
        "amendment_document_root": ROOT,
        "parent_freeze_root": ROOT,
        "child_freeze_root": ROOT,
        "child_dsl_spec_root_or_null": None,
        "approval_status_id": 1,
        "approval_method_id": 1,
        "approval_evidence_root": ROOT,
        "approving_actor_id_digest": ROOT,
        "recorded_at_unix_seconds": 1,
        "repository_commit_id": COMMIT,
    }
    assert _error_code(
        build_formal_object, "NormativeApprovalManifestV1", approval
    ) == "REJECT_M25_FIELD_NULL"
    approval["child_dsl_spec_root_or_null"] = ROOT
    build_formal_object("NormativeApprovalManifestV1", approval)


def test_m3_genesis_carries_exactly_15_null_slots_but_gate24_remains_blocked() -> None:
    assert len(M3_RUN_OUTPUT_ROOTS) == 15
    fields = _m3_genesis_fields()
    payload = encode_formal_object("M3RunGenesisV1", fields)
    assert decode_formal_object(payload).schema.name == "M3RunGenesisV1"
    assert len(candidate_content_root("M3RunGenesisV1", fields)) == 32
    fields["canonical_program_archive_root_or_null"] = ROOT
    assert _error_code(build_formal_object, "M3RunGenesisV1", fields) == "FAIL_M3_OUTPUT_ROOT_PREPOPULATED"
    assert "M3RunGenesisOutputSlotCount" in AUTHORITATIVE_BLOCKING_GAPS


def test_unresolved_state_dual_domain_and_witness_conflicts_stay_explicit() -> None:
    assert _error_code(build_formal_object, "M3RunStateRecordV1", {}) == FAIL_M25_NORMATIVE_GAP
    assert _error_code(build_formal_object, "M3DualReplayAgreementV1", {}) == FAIL_M25_NORMATIVE_GAP
    assert {
        "NormativeDocumentPathMachineId",
        "OutsideTargetClaimLevel",
        "SplitContractRuleIds",
        "ContractFieldIdRegistries",
        "StateMachineLegalTransitionRow",
        "InputSignatureStaticRoleMetadata",
        "MismatchKindId",
        "M3RunStateRecordV1",
        "M3DualReplayAgreementV1",
        "SinkWitnessBinding",
        "M3RunGenesisOutputSlotCount",
        "ExternalActorEvidence",
    } == set(AUTHORITATIVE_BLOCKING_GAPS)
    assert "required_witness_ast_hash" not in FORMAL_SCHEMA_REGISTRY["DslRoleBindingManifestV1"].fields


def test_false_invention_target_bundle_requires_its_witness() -> None:
    fields = {
        "outside_target_spec_root": ROOT,
        "outside_target_universe_root": ROOT,
        "outside_target_truth_root": ROOT,
        "null_control_spec_root": ROOT,
        "null_control_universe_root": ROOT,
        "null_control_truth_root": ROOT,
        "fallback_registry_root": ROOT,
        "null_control_required_witness_ast_hash_or_null": None,
        "null_control_claim_level_id": 1,
    }
    assert _error_code(build_formal_object, "TargetBundleV1", fields) == "REJECT_M25_FIELD_NULL"
    fields["null_control_required_witness_ast_hash_or_null"] = OTHER_ROOT
    build_formal_object("TargetBundleV1", fields)
