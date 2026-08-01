from __future__ import annotations

import json

from hegel_machine.phase3_m25_errata_vectors_v1 import (
    VECTOR_SCHEMA,
    generate_errata_vector_report_v1,
    vector_id_v1,
    vector_key_v1,
    vector_root_v1,
    vector_signature_v1,
)
from hegel_machine.phase3_m25_wire_v1 import (
    MACHINE_FREEZE_ID,
    OBJECT_TAGS,
    candidate_content_root,
    decode_formal_object,
)


OBJECT_NAMES = [
    "ActorTrustGenesisV1",
    "BridgeAttestationSignaturePreimageV1",
    "BridgeReplayStatementV1",
    "CanonicalAstProfileSpecV1",
    "CanonicalCborProfileSpecV1",
    "HiddenArtifactScopeV1",
    "M3DualReplayAgreementV1",
    "M3ExecutionCandidateV1",
    "M3ExecutionManifestV2",
    "M3RunGenesisV1",
    "M3RunStateRecordV1.synthetic_start_shape",
    "MdlCodeTableSpecV1",
    "NormativeDocumentBundleV1",
    "OpaqueIdRegistrationIntentV1.run",
    "OpaqueIdRegistrySnapshotV1.genesis",
    "ParentAbsenceAuditBundleV1",
    "ParentManifestAbsenceAttestationV2",
    "Phase2BContractSpecV1",
    "SignedManifestEnvelopeV1.parent_absence",
    "StaticRoleMetadataV1.odd",
    "StaticRoleMetadataV1.sink",
]

TREE_NAMES = [
    "AuditedHistoryRowV1",
    "AuditedPathBlobRecordV1",
    "DependencyLockRecordV1",
    "LegacyParentSourceRowV1",
    "LegalTransitionRowV1",
    "OpaqueIdRegistryRecordV1",
    "RepositoryPathAliasRecordV1",
    "SourceFileRecordV1",
]

GUARD_ERRORS = {
    "actor_public_key_reused_across_purposes": "FAIL_ACTOR_KEY_CROSS_PURPOSE_REUSE",
    "actor_trust_missing_purpose": "FAIL_ACTOR_TRUST_PURPOSE_SET",
    "actor_trust_reused_manifest_root": "FAIL_ACTOR_TRUST_KEY_ROOT_MISMATCH",
    "audited_path_wrong_order": "REJECT_M25_RECORD_ORDER",
    "bridge_attester_purpose_order": "FAIL_BRIDGE_ATTESTATION_PURPOSE_SET",
    "document_roles_wrong_order": "REJECT_M25_FIELD_VALUE",
    "external_attestation_missing_auditor": "FAIL_EXTERNAL_INPUT_ATTESTATION_COVERAGE",
    "m3_genesis_output_prepopulated": "FAIL_M3_OUTPUT_ROOT_PREPOPULATED",
    "m3_start_wrong_reason": "FAIL_ILLEGAL_M3_STATE_TRANSITION",
    "null_witness_mismatch": "FAIL_NULL_WITNESS_BINDING_MISMATCH",
    "opaque_registry_raw_id_reuse_across_kinds": "FAIL_OPAQUE_ID_ALREADY_USED",
    "opaque_registry_sequence_gap": "FAIL_OPAQUE_ID_REGISTRY_SEQUENCE",
    "opaque_snapshot_tree_root_mismatch": "FAIL_OPAQUE_ID_REGISTRY_SNAPSHOT",
    "parent_absence_bitmask_not_15": "REJECT_M25_FIELD_VALUE",
    "sink_static_orientation_mismatch": "REJECT_M25_FIELD_VALUE",
}


def _objects_by_name(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {item["name"]: item for item in report["objects"]}  # type: ignore[index,union-attr]


def test_report_has_exact_public_contract_counts_and_order() -> None:
    report = generate_errata_vector_report_v1()
    assert tuple(report) == (
        "ok",
        "op",
        "machine_freeze_id",
        "vector_schema",
        "objects",
        "record_trees",
        "guard_errors",
    )
    assert report["ok"] is True
    assert report["op"] == "errata_vectors"
    assert report["machine_freeze_id"] == MACHINE_FREEZE_ID
    assert report["vector_schema"] == VECTOR_SCHEMA == (
        "hegel-phase3-m25-exact-wire-errata-vectors/1"
    )

    objects = report["objects"]
    trees = report["record_trees"]
    guards = report["guard_errors"]
    assert isinstance(objects, list) and isinstance(trees, list) and isinstance(guards, list)
    assert len(objects) == 21
    assert len(trees) == 8
    assert len(guards) == 15
    assert [item["name"] for item in objects] == OBJECT_NAMES
    assert [item["name"] for item in trees] == TREE_NAMES
    assert [item["vector_id"] for item in guards] == sorted(GUARD_ERRORS)


def test_object_and_tree_item_shapes_are_exact_and_candidate_only() -> None:
    report = generate_errata_vector_report_v1()
    for item in report["objects"]:  # type: ignore[union-attr]
        assert tuple(item) == (
            "name",
            "schema_name",
            "tag",
            "status",
            "bytes_hex",
            "candidate_root_hex",
            "error_code",
        )
        assert item["status"] == "PASS_CANDIDATE_NON_AUTHORITATIVE"
        assert item["error_code"] is None
        bytes.fromhex(item["bytes_hex"])
        if item["name"] == "BridgeAttestationSignaturePreimageV1":
            assert item["schema_name"] == "raw-signature-preimage"
            assert item["tag"] == 0
            assert item["candidate_root_hex"] is None
        else:
            assert item["tag"] == OBJECT_TAGS[item["schema_name"]]
            assert len(bytes.fromhex(item["candidate_root_hex"])) == 32

    for item in report["record_trees"]:  # type: ignore[union-attr]
        assert tuple(item) == (
            "name",
            "schema_name",
            "tag",
            "status",
            "record_count",
            "first_record_cbor_hex",
            "root_hex",
            "error_code",
        )
        assert item["name"] == item["schema_name"]
        assert item["tag"] == OBJECT_TAGS[item["schema_name"]]
        assert item["status"] == "PASS_CANDIDATE_NON_AUTHORITATIVE"
        assert item["record_count"] == 2
        assert item["error_code"] is None
        assert len(bytes.fromhex(item["root_hex"])) == 32


def test_every_formal_object_and_first_tree_record_strictly_decodes() -> None:
    report = generate_errata_vector_report_v1()
    for item in report["objects"]:  # type: ignore[union-attr]
        if item["tag"] == 0:
            continue
        payload = bytes.fromhex(item["bytes_hex"])
        decoded = decode_formal_object(payload, expected_name=item["schema_name"])
        assert candidate_content_root(item["schema_name"], decoded.fields).hex() == item[
            "candidate_root_hex"
        ]

    for item in report["record_trees"]:  # type: ignore[union-attr]
        decode_formal_object(
            bytes.fromhex(item["first_record_cbor_hex"]),
            expected_name=item["schema_name"],
        )


def test_derivation_primitives_and_selected_roots_are_bit_exact() -> None:
    assert vector_root_v1("normative_document_bundle_id").hex() == (
        "2ebadddfb9c64f5ec52325b2675b79940627c0d356aa6eab4449a564c24ff3f8"
    )
    assert vector_id_v1("run_id").hex() == "90423d805c9dec51c2f2972542beb83c"
    assert vector_key_v1("actor-1").hex() == (
        "c5c621f94a41bd36b654ff1054aef3f09ca437111b78caa4288522b4caf3a2c7"
    )
    assert vector_signature_v1("parent_absence_auditor").hex() == (
        "a537c2e69b03c6409038f149e7a44d623f3bc10819725dbe61671bf33fa1d331"
        "96a97d64e0d65d2e347cab3a5bc0728b05887085673df6ac0280b5c3db546401"
    )

    report = generate_errata_vector_report_v1()
    objects = _objects_by_name(report)
    assert objects["NormativeDocumentBundleV1"]["candidate_root_hex"] == (
        "7304779d570f6a005f48d41becc59de4822e41da57d4fbea2bb449cc195b80d9"
    )
    assert objects["M3ExecutionCandidateV1"]["candidate_root_hex"] == (
        "586d5321f8f712e127a434baccf2aacb7de73acde4f2d02e4e9b53bbe8d45a58"
    )
    assert objects["BridgeAttestationSignaturePreimageV1"]["bytes_hex"] == (
        "484547454c2f4252494447455f4154544553544154494f4e5f5349474e41545552452f563100"
        "95f689405ccad7d7967cd8acfd205b868b4b5410aee9d3280152481f80e3566d"
        "00020000000000000007"
    )


def test_guard_report_is_captured_from_production_validator_codes() -> None:
    report = generate_errata_vector_report_v1()
    assert {
        item["vector_id"]: item["error_code"] for item in report["guard_errors"]  # type: ignore[union-attr]
    } == GUARD_ERRORS


def test_report_is_reproducible_json_and_contains_no_authority_claim() -> None:
    first = generate_errata_vector_report_v1()
    second = generate_errata_vector_report_v1()
    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True, separators=(",", ":"))) == first
    serialized = json.dumps(first, sort_keys=True)
    assert "PASS_CANDIDATE_NON_AUTHORITATIVE" in serialized
    assert "formal_content_root" not in serialized
    assert "formal_record_tree_root" not in serialized
    assert "private_key" not in serialized
    assert "seed_bytes" not in serialized
