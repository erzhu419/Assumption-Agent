from __future__ import annotations

import hashlib

import pytest

from hegel_machine.phase3_m25_split_v1 import (
    CUSTODIAN_SIGNATURE_PREFIX,
    SplitRankInput,
    assert_authoritative_seed_genesis_available,
    custodian_signature_preimage,
    derive_role_key,
    ed25519_key_id,
    hkdf_expand_sha256,
    hkdf_extract_sha256,
    rank_split_rows,
    split_hkdf_prk,
    split_rank,
    split_seed_commitment,
    uint16_be,
)
from hegel_machine.phase3_m25_wire_v1 import (
    AUTHORITATIVE_BLOCKING_GAPS,
    FAIL_M25_NORMATIVE_GAP,
    FORMAL_SCHEMA_REGISTRY,
    LEGAL_M3_TRANSITIONS,
    M25WireError,
    M3_REQUIRED_INPUT_ROOTS,
    M3_RUN_OUTPUT_ROOTS,
    M3RunningPhase,
    M3State,
    OBJECT_TAGS,
    assert_authoritative_m25_ready,
    build_formal_object,
    decode_formal_object,
    encode_formal_object,
    formal_content_root,
    formal_record_tree_root,
    git_sha1_commit_id,
    synthetic_content_root,
    synthetic_record_tree_root,
    validate_m3_input_roots,
    validate_hidden_access_ledger_genesis,
    validate_m3_output_roots_null,
    validate_m3_prerun_snapshot,
    validate_m3_state_chain_link,
    validate_m3_state_transition,
)


ROOT = bytes(range(32))
OTHER_ROOT = bytes(reversed(range(32)))
COMMIT = git_sha1_commit_id(bytes(range(20)))


def _approval_fields() -> dict[str, object]:
    return {
        "amendment_document_root": ROOT,
        "parent_freeze_root": OTHER_ROOT,
        "child_freeze_root": ROOT,
        "child_dsl_spec_root_or_null": None,
        "approval_status_id": 1,
        "approval_method_id": 1,
        "approval_evidence_root": OTHER_ROOT,
        "approving_actor_id_digest": ROOT,
        "recorded_at_unix_seconds": 123456789,
        "repository_commit_id": COMMIT,
    }


def _role_binding_fields() -> dict[str, object]:
    return {
        "role_id": 1,
        "child_dsl_spec_root": ROOT,
        "child_freeze_root": ROOT,
        "operator_semantics_root": ROOT,
        "identifier_registry_root": ROOT,
        "canonical_ast_schema_root": ROOT,
        "canonical_cbor_profile_root": ROOT,
        "semantic_spec_diagnostic_id_digest": ROOT,
        "semantic_spec_formal_root": ROOT,
        "universe_diagnostic_id_digest": ROOT,
        "truth_diagnostic_id_digest": ROOT,
        "formal_universe_root": ROOT,
        "formal_truth_root": ROOT,
        "split_binding_manifest_root": ROOT,
        "custodian_binding_manifest_root": ROOT,
        "seed_continuity_manifest_root": ROOT,
        "parent_binding_manifest_root_or_null": None,
        "legacy_parent_payload_source_id_digest_or_null": ROOT,
        "parent_manifest_absence_attestation_root_or_null": ROOT,
        "fallback_registry_root_or_null": None,
        "created_at_unix_seconds": 1,
        "repository_commit_id": COMMIT,
    }


def _all_input_roots() -> dict[str, bytes]:
    roots = {
        name: hashlib.sha256(name.encode("ascii")).digest()
        for name in M3_REQUIRED_INPUT_ROOTS
    }
    genesis_root = synthetic_content_root(
        "HiddenAccessLedgerRecordV1",
        _ledger_genesis_fields(),
    )
    roots["split_seed_commitment_manifest_root"] = ROOT
    roots["hidden_access_ledger_genesis_root"] = genesis_root
    roots["hidden_access_ledger_head_root"] = genesis_root
    return roots


def _all_null_output_roots() -> dict[str, None]:
    return {name: None for name in M3_RUN_OUTPUT_ROOTS}


def _ledger_genesis_fields() -> dict[str, object]:
    return {
        "ledger_id": bytes(range(16)),
        "sequence_number": 0,
        "previous_record_root_or_null": None,
        "event_type_id": 1,
        "actor_key_id": bytes(range(16)),
        "subject_manifest_root": ROOT,
        "revealed_artifact_root_or_null": None,
        "authorization_root_or_null": None,
        "created_at_unix_seconds": 1,
        "repository_commit_id": COMMIT,
    }


def _enumeration_receipt_fields() -> dict[str, object]:
    return {
        "implementation_id": 1,
        "run_id": bytes(range(16)),
        "execution_manifest_root": ROOT,
        "implementation_source_root": ROOT,
        "implementation_binary_digest": ROOT,
        "environment_image_digest": ROOT,
        "child_dsl_spec_root": ROOT,
        "operator_semantics_root": ROOT,
        "identifier_registry_root": ROOT,
        "canonical_ast_schema_root": ROOT,
        "canonical_cbor_profile_root": ROOT,
        "closure_status_id": 0,
        "raw_operator_application_count": 0,
        "canonical_program_count": 0,
        "closure_cardinality_or_null": None,
        "frontier_exhausted": False,
        "all_type_buckets_closed": False,
        "raw_expansion_limit_hit": False,
        "wall_clock_abort_hit": False,
        "canonical_program_archive_root_or_null": None,
        "program_chunk_manifest_root_or_null": None,
        "bucket_accounting_root_or_null": None,
        "first_out_of_budget_program_hash_or_null": None,
        "partial_diagnostic_bundle_root_or_null": None,
        "started_at_unix_seconds": 1,
        "finished_at_unix_seconds": 2,
        "process_exit_code": 0,
    }


def test_tag_registry_matches_all_frozen_tags_and_marks_known_gaps() -> None:
    assert len(OBJECT_TAGS) == 28
    assert OBJECT_TAGS["M3DualReplayAgreementV1"] == 0x3304
    assert FORMAL_SCHEMA_REGISTRY["DiagnosticFormalBridgeRecordV1"].ordering_fields == (
        "artifact_role_id",
        "diagnostic_namespace_id",
        "diagnostic_digest",
    )
    assert FORMAL_SCHEMA_REGISTRY["BucketAccountingRecordV1"].wire_gap is not None
    assert "CustodianBindingCoreV1" in AUTHORITATIVE_BLOCKING_GAPS


def test_run_output_slots_are_the_exact_unique_frozen_15_tuple() -> None:
    assert M3_RUN_OUTPUT_ROOTS == (
        "canonical_program_archive_root",
        "program_chunk_manifest_root",
        "bucket_accounting_root",
        "outside_program_output_archive_root",
        "outside_output_chunk_manifest_root",
        "outside_match_set_root",
        "outside_role_evaluation_receipt_root",
        "null_program_output_archive_root",
        "null_output_chunk_manifest_root",
        "null_match_set_root",
        "null_role_evaluation_receipt_root",
        "python_enumeration_receipt_root",
        "rust_enumeration_receipt_root",
        "dual_replay_agreement_root",
        "final_state_record_root",
    )
    assert len(M3_RUN_OUTPUT_ROOTS) == 15
    assert len(set(M3_RUN_OUTPUT_ROOTS)) == 15


def test_numeric_array_round_trip_preserves_field_order_and_exact_bytes() -> None:
    fields = _approval_fields()
    value = build_formal_object("NormativeApprovalManifestV1", fields)
    assert value[:3] == (
        1,
        0x3101,
        b"hegel-normative-approval-manifest/1",
    )
    assert value[3] == fields["amendment_document_root"]
    assert value[-1] == COMMIT

    encoded = encode_formal_object("NormativeApprovalManifestV1", fields)
    decoded = decode_formal_object(
        encoded,
        expected_name="NormativeApprovalManifestV1",
    )
    assert decoded.value == value
    assert encode_formal_object(decoded.schema.name, decoded.fields) == encoded


def test_formal_content_root_uses_exact_domain_zero_separator_and_cbor() -> None:
    fields = _approval_fields()
    encoded = encode_formal_object("NormativeApprovalManifestV1", fields)
    expected = hashlib.sha256(
        b"HEGEL/NORMATIVE_APPROVAL_MANIFEST/V1\x00" + encoded
    ).digest()
    assert synthetic_content_root("NormativeApprovalManifestV1", fields) == expected


def test_synthetic_record_has_stable_cbor_and_rfc6962_root() -> None:
    row = {
        "universe_index": 7,
        "input_signature_id": 3,
        "canonical_input_object": (1, (False, True)),
    }
    assert encode_formal_object("BoundedUniverseRowV1", row).hex() == (
        "8601193201581c686567656c2d626f756e6465642d756e6976657273652d"
        "726f772f310703820182f4f5"
    )
    assert synthetic_record_tree_root("BoundedUniverseRowV1", [row]).hex() == (
        "b141278a3d8c00115177f08514fd9eb003cfacb63af00d5b7b9598a902da3aad"
    )


def test_record_tree_rejects_wrong_order() -> None:
    rows = [
        {
            "universe_index": index,
            "input_signature_id": 3,
            "canonical_input_object": (index,),
        }
        for index in (2, 1)
    ]
    with pytest.raises(M25WireError, match="REJECT_M25_RECORD_ORDER") as error:
        synthetic_record_tree_root("BoundedUniverseRowV1", rows)
    assert error.value.code == "REJECT_M25_RECORD_ORDER"


def test_formal_object_rejects_maps_text_float_and_wrong_field_set() -> None:
    fields = _approval_fields()
    fields["extra"] = 1
    with pytest.raises(M25WireError) as error:
        build_formal_object("NormativeApprovalManifestV1", fields)
    assert error.value.code == "REJECT_M25_FIELD_SET"

    row = {
        "universe_index": 0,
        "input_signature_id": 1,
        "canonical_input_object": {"forbidden": "map"},
    }
    with pytest.raises(ValueError, match="REJECT_CBOR_MAP"):
        encode_formal_object("BoundedUniverseRowV1", row)


def test_decode_rejects_trailing_and_wrong_expected_schema() -> None:
    payload = encode_formal_object("NormativeApprovalManifestV1", _approval_fields())
    with pytest.raises(ValueError, match="REJECT_TRAILING_CBOR"):
        decode_formal_object(payload + b"\x00")
    with pytest.raises(M25WireError) as error:
        decode_formal_object(payload, expected_name="DslRoleBindingManifestV1")
    assert error.value.code == "REJECT_M25_SCHEMA_MISMATCH"


def test_parent_provenance_xor_is_fail_closed() -> None:
    fields = _role_binding_fields()
    build_formal_object("DslRoleBindingManifestV1", fields)

    fields["parent_binding_manifest_root_or_null"] = ROOT
    with pytest.raises(M25WireError) as error:
        build_formal_object("DslRoleBindingManifestV1", fields)
    assert error.value.code == "FAIL_PARENT_BINDING_PROVENANCE_INCOMPLETE"

    fields["legacy_parent_payload_source_id_digest_or_null"] = None
    fields["parent_manifest_absence_attestation_root_or_null"] = None
    build_formal_object("DslRoleBindingManifestV1", fields)


@pytest.mark.parametrize(
    "object_name",
    [
        "BucketAccountingRecordV1",
        "M3RunStateRecordV1",
        "M3DualReplayAgreementV1",
        "CustodianBindingCoreV1",
    ],
)
def test_unfrozen_wire_or_domain_paths_raise_one_normative_gap_code(
    object_name: str,
) -> None:
    with pytest.raises(M25WireError) as error:
        build_formal_object(object_name, {})
    assert error.value.code == FAIL_M25_NORMATIVE_GAP


def test_no_content_hash_domain_is_not_guessed_for_record_objects() -> None:
    row = {
        "universe_index": 0,
        "input_signature_id": 1,
        "canonical_input_object": (0,),
    }
    with pytest.raises(M25WireError) as error:
        synthetic_content_root("BoundedUniverseRowV1", row)
    assert error.value.code == FAIL_M25_NORMATIVE_GAP


def test_authoritative_root_apis_cannot_bypass_open_row_and_root_preimages() -> None:
    row = {
        "universe_index": 0,
        "input_signature_id": 1,
        "canonical_input_object": (0,),
    }
    with pytest.raises(M25WireError) as error:
        formal_record_tree_root("BoundedUniverseRowV1", [row])
    assert error.value.code == FAIL_M25_NORMATIVE_GAP

    with pytest.raises(M25WireError) as error:
        formal_content_root("NormativeApprovalManifestV1", _approval_fields())
    assert error.value.code == FAIL_M25_NORMATIVE_GAP


def test_signed_custodian_objects_require_one_sorted_signature() -> None:
    fields = {
        "enclosed_object_tag": 0x3103,
        "enclosed_manifest_root": ROOT,
        "created_at_unix_seconds": 1,
        "custodian_key_epoch": 0,
        "signatures": ((bytes(16), bytes(64)),),
    }
    build_formal_object("SignedManifestEnvelopeV1", fields)
    fields["signatures"] = ()
    with pytest.raises(M25WireError) as error:
        build_formal_object("SignedManifestEnvelopeV1", fields)
    assert error.value.code == "REJECT_M25_SIGNATURE_COUNT"


def test_nullable_integer_fields_still_require_nonnegative_exact_integers() -> None:
    fields = _enumeration_receipt_fields()
    build_formal_object("M3ImplementationEnumerationReceiptV1", fields)
    for invalid in ("0", -1, True):
        fields["closure_cardinality_or_null"] = invalid
        with pytest.raises(M25WireError) as error:
            build_formal_object("M3ImplementationEnumerationReceiptV1", fields)
        assert error.value.code == "REJECT_M25_FIELD_TYPE"


def test_envelope_rejects_unallocated_or_noninteger_object_tag() -> None:
    fields = {
        "enclosed_object_tag": 0x3103,
        "enclosed_manifest_root": ROOT,
        "created_at_unix_seconds": 1,
        "custodian_key_epoch": 0,
        "signatures": ((bytes(16), bytes(64)),),
    }
    for invalid in (0x9999, "0x3103", True):
        fields["enclosed_object_tag"] = invalid
        with pytest.raises(M25WireError):
            build_formal_object("SignedManifestEnvelopeV1", fields)


def test_parent_absence_attestation_requires_exact_frozen_reason_mask() -> None:
    fields = {
        "parent_dsl_version_digest": ROOT,
        "parent_freeze_version_digest": ROOT,
        "parent_repository_commit_id": COMMIT,
        "audited_source_tree_root": ROOT,
        "audited_path_set_root": ROOT,
        "legacy_parent_payload_source_id_digest": ROOT,
        "absence_reason_bitmask": 0b11,
        "auditor_key_id": bytes(16),
        "audited_at_unix_seconds": 1,
    }
    build_formal_object("ParentManifestAbsenceAttestationV1", fields)
    fields["absence_reason_bitmask"] = 1
    with pytest.raises(M25WireError) as error:
        build_formal_object("ParentManifestAbsenceAttestationV1", fields)
    assert error.value.code == "REJECT_M25_FIELD_VALUE"


def test_complete_prerun_diagnostic_guards_pass_only_with_genesis_ledger() -> None:
    inputs = _all_input_roots()
    outputs = _all_null_output_roots()
    validate_m3_input_roots(inputs)
    validate_m3_output_roots_null(outputs)
    validate_m3_prerun_snapshot(
        inputs,
        outputs,
        ledger_record_count=1,
        ledger_genesis_fields=_ledger_genesis_fields(),
    )


def test_prerun_guard_rejects_null_input_and_prepopulated_output() -> None:
    inputs = _all_input_roots()
    inputs["child_dsl_spec_root"] = None  # type: ignore[assignment]
    with pytest.raises(M25WireError) as error:
        validate_m3_input_roots(inputs)
    assert error.value.code == "FAIL_M3_INPUT_ROOT_NULL"

    outputs = _all_null_output_roots()
    outputs["canonical_program_archive_root"] = ROOT  # type: ignore[assignment]
    with pytest.raises(M25WireError) as error:
        validate_m3_output_roots_null(outputs)
    assert error.value.code == "FAIL_M3_OUTPUT_ROOT_PREPOPULATED"


def test_output_slot_absence_and_non_genesis_ledger_fail_closed() -> None:
    outputs = _all_null_output_roots()
    del outputs["final_state_record_root"]
    with pytest.raises(M25WireError) as error:
        validate_m3_output_roots_null(outputs)
    assert error.value.code == FAIL_M25_NORMATIVE_GAP

    inputs = _all_input_roots()
    inputs["hidden_access_ledger_head_root"] = OTHER_ROOT
    with pytest.raises(M25WireError) as error:
        validate_m3_prerun_snapshot(
            inputs,
            _all_null_output_roots(),
            ledger_record_count=1,
            ledger_genesis_fields=_ledger_genesis_fields(),
        )
    assert error.value.code == "FAIL_M3_LEDGER_HEAD_NOT_GENESIS"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequence_number", 1),
        ("previous_record_root_or_null", ROOT),
        ("event_type_id", 2),
        ("event_type_id", 4),
        ("revealed_artifact_root_or_null", ROOT),
        ("authorization_root_or_null", ROOT),
    ],
)
def test_ledger_genesis_rejects_access_reveal_or_non_genesis_fields(
    field: str,
    value: object,
) -> None:
    fields = _ledger_genesis_fields()
    fields[field] = value
    with pytest.raises(M25WireError) as error:
        validate_hidden_access_ledger_genesis(fields)
    assert error.value.code == "FAIL_M3_LEDGER_HEAD_NOT_GENESIS"


def test_prerun_ledger_subject_must_bind_split_seed_commitment() -> None:
    inputs = _all_input_roots()
    inputs["split_seed_commitment_manifest_root"] = OTHER_ROOT
    with pytest.raises(M25WireError) as error:
        validate_m3_prerun_snapshot(
            inputs,
            _all_null_output_roots(),
            ledger_record_count=1,
            ledger_genesis_fields=_ledger_genesis_fields(),
        )
    assert error.value.code == "FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH"


def test_all_and_only_frozen_state_transitions_are_accepted() -> None:
    for transition in LEGAL_M3_TRANSITIONS:
        assert validate_m3_state_transition(*transition) == transition

    with pytest.raises(M25WireError) as error:
        validate_m3_state_transition(
            M3State.NOT_RUN,
            M3RunningPhase.NONE,
            M3State.COMPLETE,
            M3RunningPhase.NONE,
        )
    assert error.value.code == "FAIL_ILLEGAL_M3_STATE_TRANSITION"


def test_terminal_state_reopen_and_chain_break_have_exact_codes() -> None:
    with pytest.raises(M25WireError) as error:
        validate_m3_state_transition(
            M3State.COMPLETE,
            M3RunningPhase.NONE,
            M3State.RUNNING,
            M3RunningPhase.CANONICAL_ENUMERATION,
        )
    assert error.value.code == "FAIL_M3_TERMINAL_STATE_REOPEN"

    validate_m3_state_chain_link(ROOT, ROOT)
    with pytest.raises(M25WireError) as error:
        validate_m3_state_chain_link(ROOT, OTHER_ROOT)
    assert error.value.code == "FAIL_M3_STATE_CHAIN_BREAK"


def test_rfc5869_sha256_known_answer_vector() -> None:
    ikm = bytes.fromhex("0b" * 22)
    salt = bytes.fromhex("000102030405060708090a0b0c")
    info = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9")
    prk = hkdf_extract_sha256(ikm, salt)
    assert prk.hex() == (
        "077709362c2e32df0ddc3f0dc47bba63"
        "90b6c73bb50f9c3122ec844ad7c2b3e5"
    )
    assert hkdf_expand_sha256(prk, info, 42).hex() == (
        "3cb25f25faacd57a90434f64d0362f2a"
        "2d2d0a90cf1a5a4c5db02d56ecc4c5bf"
        "34007208d5b887185865"
    )


def test_frozen_split_primitives_have_stable_synthetic_vectors() -> None:
    seed = bytes(range(32))
    assert split_hkdf_prk(seed).hex() == (
        "5b2193562439873dce160cf179127e6a029973a9e545752a38837bddd47908ed"
    )
    role_key = derive_role_key(seed, 1)
    assert role_key.hex() == (
        "d036e9447f84fa91bdc3f7a805375884527d85b25a300218d5b27e686fda3044"
    )
    assert split_rank(role_key, 1, 0x1234, OTHER_ROOT).hex() == (
        "4c1ec0b1d849e65f5fd6d90eeaa88fd0caf4d73ae92185cafce0ceb4578bbc5e"
    )
    assert split_seed_commitment(seed).hex() == (
        "3126668b3227a5e6ab711bcaa66f9d573a7e8bf8b1d1c6cabbb07a96ccf566ba"
    )


def test_key_id_and_custodian_signature_preimage_are_exact() -> None:
    assert ed25519_key_id(ROOT).hex() == "630dcd2966c4336691125448bbb25b4f"
    assert custodian_signature_preimage(ROOT) == CUSTODIAN_SIGNATURE_PREFIX + b"\x00" + ROOT


def test_ranked_rows_are_sorted_by_rank_then_input_hash() -> None:
    role_key = derive_role_key(ROOT, 2)
    rows = (
        SplitRankInput(ROOT, b"row-a", 0),
        SplitRankInput(OTHER_ROOT, b"row-b", 0),
    )
    ranked = rank_split_rows(role_key, 2, rows)
    assert tuple((item.rank_digest, item.canonical_input_hash) for item in ranked) == tuple(
        sorted((item.rank_digest, item.canonical_input_hash) for item in ranked)
    )


def test_rank_identity_collision_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from hegel_machine import phase3_m25_split_v1 as split_module

    monkeypatch.setattr(split_module, "split_rank", lambda *args: bytes(32))
    rows = (
        SplitRankInput(ROOT, b"different-a", 0),
        SplitRankInput(ROOT, b"different-b", 0),
    )
    with pytest.raises(M25WireError) as error:
        rank_split_rows(ROOT, 1, rows)
    assert error.value.code == "FAIL_SPLIT_RANK_IDENTITY_COLLISION"


@pytest.mark.parametrize("value", [-1, 65536, True, "1"])
def test_uint16_rejects_out_of_range_or_non_exact_integer(value: object) -> None:
    with pytest.raises(M25WireError) as error:
        uint16_be(value)  # type: ignore[arg-type]
    assert error.value.code == "REJECT_M25_UINT16"


def test_authoritative_paths_never_generate_seed_key_signature_or_gate_claim() -> None:
    with pytest.raises(M25WireError) as error:
        assert_authoritative_seed_genesis_available()
    assert error.value.code == FAIL_M25_NORMATIVE_GAP

    with pytest.raises(M25WireError) as error:
        assert_authoritative_m25_ready()
    assert error.value.code == FAIL_M25_NORMATIVE_GAP
