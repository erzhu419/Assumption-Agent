from __future__ import annotations

import hashlib

import pytest

from hegel_machine.phase3_m25_wire_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    AUTHORITATIVE_BLOCKING_GAPS,
    BRIDGE_ATTESTATION_SIGNATURE_DOMAIN,
    FAIL_M25_NORMATIVE_GAP,
    FORMAL_SCHEMA_REGISTRY,
    LEGACY_PARENT_SOURCE_IDS,
    M25WireError,
    M3_RUN_OUTPUT_ROOTS,
    M3_RUN_OUTPUT_SLOT_COUNT,
    NUMERIC_ENUM_REGISTRIES,
    OBJECT_TAGS,
    bridge_attestation_signature_preimage_v1,
    build_formal_object,
    candidate_content_root,
    candidate_record_tree_root,
    encode_formal_object,
    external_signature_preimage_v1,
    formal_content_root,
    git_sha1_commit_id,
    id_digest_v1,
    validate_actor_trust_bindings_v1,
    validate_bridge_attestation_bundle_v1,
    validate_execution_identity_linkage_v1,
    validate_external_input_attestation_bundle_v1,
    validate_null_witness_binding_v1,
    validate_opaque_id_registry_append_v1,
    validate_parent_absence_audit_bundle_v1,
    validate_source_section_profile_v1,
)


DOMAIN_ROOT = b"HEGEL/M25/ERRATA/VECTOR/ROOT/V1\x00"
DOMAIN_ID = b"HEGEL/M25/ERRATA/VECTOR/ID/V1\x00"
DOMAIN_KEY = b"HEGEL/M25/ERRATA/VECTOR/KEY/V1\x00"
DOMAIN_SIG_A = b"HEGEL/M25/ERRATA/VECTOR/SIG/A/V1\x00"
DOMAIN_SIG_B = b"HEGEL/M25/ERRATA/VECTOR/SIG/B/V1\x00"
BASE_TIME = 1_704_067_200
COMMIT = git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1)


def vroot(label: str) -> bytes:
    return hashlib.sha256(DOMAIN_ROOT + label.encode("ascii")).digest()


def vid(label: str) -> bytes:
    return hashlib.sha256(DOMAIN_ID + label.encode("ascii")).digest()[:16]


def vkey(label: str) -> bytes:
    return hashlib.sha256(DOMAIN_KEY + label.encode("ascii")).digest()


def vsig(label: str) -> bytes:
    raw = label.encode("ascii")
    return hashlib.sha256(DOMAIN_SIG_A + raw).digest() + hashlib.sha256(DOMAIN_SIG_B + raw).digest()


def error_code(callable_object: object, *args: object, **kwargs: object) -> str:
    assert callable(callable_object)
    with pytest.raises(M25WireError) as error:
        callable_object(*args, **kwargs)
    return error.value.code


def replacement_policy() -> dict[str, object]:
    return {
        "key_rotation_threshold": 2,
        "key_revocation_threshold": 2,
        "custodian_replacement_requires_new_seed_version": True,
        "actor_key_reuse_across_purposes_allowed": False,
        "secret_material_export_allowed": False,
    }


def actor_key(purpose: int) -> dict[str, object]:
    return {
        "purpose_id": purpose,
        "key_id": vid(f"actor-{purpose}"),
        "public_key_32_bytes": vkey(f"actor-{purpose}"),
        "key_epoch": 0,
        "valid_from_unix_seconds": BASE_TIME,
        "valid_until_unix_seconds_or_null": None,
        "repository_commit_id": COMMIT,
    }


def envelope(tag: int, label: str, *, key_epoch: int = 0) -> dict[str, object]:
    return {
        "enclosed_object_tag": tag,
        "enclosed_manifest_root": vroot(label),
        "created_at_unix_seconds": BASE_TIME,
        "signer_key_epoch": key_epoch,
        "signatures": ((vid(f"signer-{label}"), vsig(label)),),
    }


def execution_candidate() -> dict[str, object]:
    result: dict[str, object] = {}
    for field in FORMAL_SCHEMA_REGISTRY["M3ExecutionCandidateV1"].fields:
        if field == "run_id":
            result[field] = vid("m3-run")
        elif field == "repository_commit_id":
            result[field] = COMMIT
        elif field == "created_at_unix_seconds":
            result[field] = BASE_TIME
        elif field == "equivalence_mode_id":
            result[field] = 1
        elif field in {
            "canonical_program_budget",
            "raw_operator_application_cap",
            "records_per_chunk",
        }:
            result[field] = 50_000 if field == "canonical_program_budget" else 1_000
        else:
            result[field] = vroot(field)
    result["hidden_access_ledger_head_root"] = result["hidden_access_ledger_genesis_root"]
    return result


def run_genesis(run_id: bytes, manifest_root: bytes) -> dict[str, object]:
    fields: dict[str, object] = {
        "run_id": run_id,
        "execution_manifest_root": manifest_root,
        "initial_state_id": 0,
        "created_at_unix_seconds": BASE_TIME + 2,
        "repository_commit_id": COMMIT,
    }
    fields.update({f"{name}_or_null": None for name in M3_RUN_OUTPUT_ROOTS})
    return fields


def test_errata_tag_schema_domain_and_enum_registry_is_complete() -> None:
    assert len(OBJECT_TAGS) == len(FORMAL_SCHEMA_REGISTRY) == 81
    assert len(set(OBJECT_TAGS.values())) == 81
    expected_tags = {
        "NormativeDocumentBundleV1": 0x3018,
        "CanonicalAstProfileSpecV1": 0x3019,
        "StaticRoleMetadataV1": 0x301D,
        "BridgeReplayStatementV1": 0x310E,
        "M3ExecutionCandidateV1": 0x310F,
        "M3ExecutionManifestV2": 0x3110,
        "ActorTrustGenesisV1": 0x3111,
        "OpaqueIdRegistrySnapshotV1": 0x3112,
        "ParentAbsenceAuditBundleV1": 0x3113,
        "ParentManifestAbsenceAttestationV2": 0x3114,
        "OpaqueIdRegistrationIntentV1": 0x3115,
        "AuditedPathBlobRecordV1": 0x3210,
        "OpaqueIdRegistryRecordV1": 0x3218,
    }
    assert all(OBJECT_TAGS[name] == tag for name, tag in expected_tags.items())
    assert FORMAL_SCHEMA_REGISTRY["SignedManifestEnvelopeV1"].hash_domain == (
        "HEGEL/SIGNED_MANIFEST_ENVELOPE/V1"
    )
    assert FORMAL_SCHEMA_REGISTRY["M3DualReplayAgreementV1"].hash_domain == (
        "HEGEL/M3_DUAL_REPLAY_AGREEMENT/V1"
    )
    for enum_name in (
        "TargetRoleId",
        "MismatchKindId",
        "AssignmentOrderingRuleId",
        "FallbackSplitPolicyId",
        "RankTieBreakRuleId",
        "TraversalFieldId",
        "BucketFieldId",
        "AccountingCounterFieldId",
        "AccountingInvariantId",
        "OpaqueIdKindId",
        "DependencyEcosystemId",
    ):
        assert enum_name in NUMERIC_ENUM_REGISTRIES
    assert NUMERIC_ENUM_REGISTRIES["ClaimLevelId"].entries[3] == "OUTSIDE_TARGET_CANDIDATE"
    assert set(AUTHORITATIVE_BLOCKING_GAPS) == {"ExternalActorEvidence"}


def test_normative_document_bundle_and_source_section_profiles_are_exact() -> None:
    bundle = {
        "bundle_id_digest": id_digest_v1("bundle:m25-v1.1.2"),
        "document_entries": ((1, vroot("base")), (2, vroot("errata")), (3, vroot("addendum"))),
        "repository_commit_id": COMMIT,
    }
    assert len(candidate_content_root("NormativeDocumentBundleV1", bundle)) == 32
    bad = dict(bundle, document_entries=((2, vroot("errata")), (1, vroot("base")), (3, vroot("addendum"))))
    assert error_code(build_formal_object, "NormativeDocumentBundleV1", bad) == "REJECT_M25_FIELD_VALUE"

    section = b"## exact section\nraw bytes\n"
    profile = {
        "profile_id_digest": id_digest_v1("profile:hegel-canonical-ast-v1"),
        "governing_normative_document_root": candidate_content_root(
            "NormativeDocumentBundleV1", bundle
        ),
        "section_selector_id_digest": id_digest_v1("section:entire-document"),
        "section_blob_sha256": hashlib.sha256(section).digest(),
        "section_byte_length": len(section),
        "repository_commit_id": COMMIT,
    }
    assert len(validate_source_section_profile_v1("CanonicalAstProfileSpecV1", profile, section)) == 32
    assert error_code(
        validate_source_section_profile_v1,
        "CanonicalAstProfileSpecV1",
        dict(profile, section_byte_length=len(section) + 1),
        section,
    ) == "FAIL_SOURCE_SECTION_LENGTH_MISMATCH"


def test_static_role_metadata_is_exact_and_nested_in_input_signature() -> None:
    odd = {
        "input_signature_id": 1,
        "role_ids": (),
        "quantity_ids": (),
        "scope_ids": (),
        "signed_orientations": (),
        "metadata_rule_id_digest": id_digest_v1("rule:odd-static-role-v1"),
    }
    sink = {
        "input_signature_id": 2,
        "role_ids": (0, 1, 2, 3),
        "quantity_ids": (0,),
        "scope_ids": (3,),
        "signed_orientations": (1, 1, -1, -1),
        "metadata_rule_id_digest": id_digest_v1("rule:sink-static-role-v1"),
    }
    build_formal_object("StaticRoleMetadataV1", odd)
    sink_object = build_formal_object("StaticRoleMetadataV1", sink)
    signature = {
        "input_signature_id": 2,
        "input_object_tag": 0x3402,
        "field_sort_ids": (4, 4, 4, 4),
        "static_role_metadata": sink_object,
        "canonical_ordering_rule_id_digest": id_digest_v1("rule:sink-input-order-v1"),
    }
    build_formal_object("InputSignatureSpecV1", signature)
    assert error_code(
        build_formal_object,
        "StaticRoleMetadataV1",
        dict(sink, signed_orientations=(1, 1, 1, -1)),
    ) == "REJECT_M25_FIELD_VALUE"


def test_bridge_signature_preimage_binds_purpose_and_epoch_exactly() -> None:
    statement_root = vroot("bridge-statement")
    expected = (
        BRIDGE_ATTESTATION_SIGNATURE_DOMAIN.encode("utf-8")
        + b"\x00"
        + statement_root
        + (2).to_bytes(2, "big")
        + (7).to_bytes(8, "big")
    )
    assert bridge_attestation_signature_preimage_v1(statement_root, 2, 7) == expected
    assert external_signature_preimage_v1(0x3103, vroot("seed"), 1, 0).endswith(
        (1).to_bytes(2, "big") + bytes(8)
    )
    assert error_code(bridge_attestation_signature_preimage_v1, statement_root, 4, 7) == (
        "REJECT_M25_FIELD_VALUE"
    )

    fields = envelope(0x310E, "bridge-statement", key_epoch=7)
    build_formal_object("SignedManifestEnvelopeV1", fields)
    legacy_label = dict(fields)
    legacy_label["custodian_key_epoch"] = legacy_label.pop("signer_key_epoch")
    assert error_code(build_formal_object, "SignedManifestEnvelopeV1", legacy_label) == (
        "REJECT_M25_FIELD_SET"
    )


def test_actor_trust_anchor_replays_policy_and_distinct_keys() -> None:
    keys = [actor_key(purpose) for purpose in range(1, 5)]
    policy = replacement_policy()
    trust = {
        "trust_genesis_id_16_bytes": vid("trust-genesis"),
        "purpose_key_entries": tuple(
            (key["purpose_id"], candidate_content_root("ActorKeyManifestV1", key)) for key in keys
        ),
        "purpose_key_policy_root": candidate_content_root("ReplacementPolicyV1", policy),
        "created_at_unix_seconds": BASE_TIME,
        "repository_commit_id": COMMIT,
    }
    assert len(validate_actor_trust_bindings_v1(trust, keys, policy)) == 32
    reused = [dict(key) for key in keys]
    reused[3]["public_key_32_bytes"] = reused[0]["public_key_32_bytes"]
    assert error_code(validate_actor_trust_bindings_v1, trust, reused, policy) == (
        "FAIL_ACTOR_KEY_CROSS_PURPOSE_REUSE"
    )


def test_external_input_and_bridge_attestation_bundle_coverage() -> None:
    external = [
        (purpose, envelope(tag, f"external-{tag:x}"))
        for purpose, tag in ((1, 0x3103), (1, 0x3105), (1, 0x3106), (1, 0x3108), (4, 0x3114))
    ]
    external_rows = sorted(
        (
            purpose,
            item["enclosed_manifest_root"],
            candidate_content_root("SignedManifestEnvelopeV1", item),
        )
        for purpose, item in external
    )
    bundle = {"attestations": tuple(external_rows)}
    assert len(validate_external_input_attestation_bundle_v1(bundle, external)) == 32

    statement_root = vroot("statement-shared")
    bridge = []
    for purpose in (1, 2, 3):
        item = envelope(0x310E, "statement-shared", key_epoch=purpose)
        item["signatures"] = ((vid(f"bridge-{purpose}"), vsig(f"bridge-{purpose}")),)
        bridge.append((purpose, item))
    bridge_rows = sorted(
        (
            purpose,
            statement_root,
            candidate_content_root("SignedManifestEnvelopeV1", item),
        )
        for purpose, item in bridge
    )
    bridge_bundle = {"attestations": tuple(bridge_rows)}
    assert len(
        validate_bridge_attestation_bundle_v1(statement_root, bridge_bundle, bridge)
    ) == 32
    assert error_code(
        validate_bridge_attestation_bundle_v1,
        statement_root,
        bridge_bundle,
        bridge[:2],
    ) == "FAIL_BRIDGE_ATTESTATION_PURPOSE_SET"


def test_opaque_id_intent_record_and_snapshot_remove_the_cycle() -> None:
    opaque_id = vid("run-0")
    intent = {
        "opaque_id_kind_id": 1,
        "opaque_id_16_bytes": opaque_id,
        "registration_context_root": vroot("registration-context"),
        "created_at_unix_seconds": BASE_TIME,
        "repository_commit_id": COMMIT,
    }
    record = {
        "registry_sequence_number": 0,
        "opaque_id_kind_id": 1,
        "opaque_id_16_bytes": opaque_id,
        "first_seen_object_root": candidate_content_root("OpaqueIdRegistrationIntentV1", intent),
        "first_seen_repository_commit_id": COMMIT,
        "created_at_unix_seconds": BASE_TIME,
    }
    registry_root = candidate_record_tree_root("OpaqueIdRegistryRecordV1", [record])
    snapshot = {
        "previous_snapshot_root_or_null": None,
        "registry_tree_root": registry_root,
        "record_count": 1,
        "added_record_root": registry_root,
        "repository_commit_id": COMMIT,
    }
    assert len(validate_opaque_id_registry_append_v1([intent], [record], snapshot)) == 32
    duplicate = dict(record, registry_sequence_number=1, opaque_id_kind_id=2)
    assert error_code(
        candidate_record_tree_root,
        "OpaqueIdRegistryRecordV1",
        [record, duplicate],
    ) == "FAIL_OPAQUE_ID_ALREADY_USED"


def test_parent_absence_rows_bundle_and_v2_attestation_replay() -> None:
    path_row = {
        "repository_path_alias_id_digest": id_digest_v1("repo-path:parent-source"),
        "raw_repository_path_utf8_bytes": b"parent/source.md",
        "git_object_algorithm_id": 1,
        "git_blob_digest": bytes(range(20)),
        "file_mode": 0o100644,
        "byte_length": 123,
    }
    touched_root = candidate_record_tree_root("AuditedPathBlobRecordV1", [path_row])
    history_row = {
        "commit_generation": 0,
        "repository_commit_id": COMMIT,
        "ordered_parent_commit_ids": (),
        "touched_path_set_root": touched_root,
    }
    legacy_rows = []
    for role_id, full_id in enumerate(LEGACY_PARENT_SOURCE_IDS, start=1):
        suffix = full_id.rsplit("_", 1)[-1]
        legacy_rows.append(
            {
                "target_role_id": role_id,
                "legacy_parent_payload_source_id_digest": id_digest_v1(full_id),
                "diagnostic_namespace_id": role_id,
                "diagnostic_digest": bytes.fromhex(suffix),
                "source_repository_commit_id": COMMIT,
            }
        )
    bundle = {
        "audited_parent_repository_commit_id": COMMIT,
        "audited_path_tree_root": touched_root,
        "audited_history_tree_root": candidate_record_tree_root(
            "AuditedHistoryRowV1", [history_row]
        ),
        "legacy_source_tree_root": candidate_record_tree_root(
            "LegacyParentSourceRowV1", legacy_rows
        ),
        "audited_path_count": 1,
        "audited_history_row_count": 1,
        "legacy_source_count": 2,
    }
    assert len(
        validate_parent_absence_audit_bundle_v1(
            [path_row], [history_row], [[path_row]], legacy_rows, bundle
        )
    ) == 32
    attestation = {
        "parent_dsl_version_digest": id_digest_v1("hegel-old-dsl-v1.0.0"),
        "parent_freeze_version_digest": id_digest_v1("hegel-freeze-p2b-p3-v1.0.0"),
        "parent_repository_commit_id": COMMIT,
        "audit_bundle_root": candidate_content_root("ParentAbsenceAuditBundleV1", bundle),
        "absence_reason_bitmask": 0b1111,
        "auditor_key_id": vid("auditor"),
        "audited_at_unix_seconds": BASE_TIME,
    }
    build_formal_object("ParentManifestAbsenceAttestationV2", attestation)
    assert error_code(
        build_formal_object,
        "ParentManifestAbsenceAttestationV2",
        dict(attestation, absence_reason_bitmask=0b11),
    ) == "REJECT_M25_FIELD_VALUE"


def test_execution_candidate_statement_manifest_and_genesis_are_acyclically_linked() -> None:
    candidate = execution_candidate()
    candidate_root = candidate_content_root("M3ExecutionCandidateV1", candidate)
    statement = {
        "run_id": candidate["run_id"],
        "diagnostic_formal_bridge_root": candidate["diagnostic_formal_bridge_root"],
        "m3_execution_candidate_root": candidate_root,
        "child_dsl_spec_root": candidate["child_dsl_spec_root"],
        "child_freeze_root": candidate["child_freeze_root"],
        "actor_trust_genesis_root": candidate["actor_trust_genesis_root"],
        "opaque_id_registry_snapshot_root": candidate["opaque_id_registry_snapshot_root"],
    }
    statement_root = candidate_content_root("BridgeReplayStatementV1", statement)
    bridge_bundle = {"attestations": ()}
    manifest = {
        "run_id": candidate["run_id"],
        "m3_execution_candidate_root": candidate_root,
        "bridge_replay_statement_root": statement_root,
        "bridge_attestation_bundle_root": candidate_content_root(
            "AttestationBundleV1", bridge_bundle
        ),
        "actor_trust_genesis_root": candidate["actor_trust_genesis_root"],
        "opaque_id_registry_snapshot_root": candidate["opaque_id_registry_snapshot_root"],
        "created_at_unix_seconds": BASE_TIME + 1,
        "repository_commit_id": COMMIT,
    }
    manifest_root = candidate_content_root("M3ExecutionManifestV2", manifest)
    genesis = run_genesis(candidate["run_id"], manifest_root)  # type: ignore[arg-type]
    assert validate_execution_identity_linkage_v1(
        candidate, statement, bridge_bundle, manifest, genesis
    ) == (candidate_root, statement_root, manifest_root)
    assert M3_RUN_OUTPUT_SLOT_COUNT == len(M3_RUN_OUTPUT_ROOTS) == 15
    assert error_code(
        validate_execution_identity_linkage_v1,
        candidate,
        dict(statement, child_freeze_root=vroot("wrong")),
        bridge_bundle,
        manifest,
        genesis,
    ) == "FAIL_M3_EXECUTION_IDENTITY_LINKAGE"


def test_m3_run_state_prefix_fields_and_exact_start_guard() -> None:
    fields = {
        "run_id": vid("state-run"),
        "transition_index": 0,
        "previous_state_record_root_or_null": None,
        "from_state_id": 0,
        "from_phase_id": 0,
        "to_state_id": 1,
        "to_phase_id": 1,
        "transition_reason_id": 1,
        "execution_manifest_root": vroot("manifest"),
        "triggering_receipt_root_or_null": None,
        "recorded_at_unix_seconds": BASE_TIME,
    }
    encoded = encode_formal_object("M3RunStateRecordV1", fields)
    assert encoded.startswith(bytes.fromhex("8e01193301"))
    assert len(candidate_content_root("M3RunStateRecordV1", fields)) == 32
    assert error_code(
        build_formal_object,
        "M3RunStateRecordV1",
        dict(fields, transition_reason_id=2),
    ) == "FAIL_ILLEGAL_M3_STATE_TRANSITION"


def test_legal_transition_and_field_id_registries_are_strict() -> None:
    transition = {
        "from_state_id": 0,
        "from_phase_id": 0,
        "to_state_id": 1,
        "to_phase_id": 1,
        "allowed_reason_ids": (1,),
    }
    transition_object = build_formal_object("LegalTransitionRowV1", transition)
    state_contract = {
        "m3_state_registry_root": vroot("states"),
        "m3_phase_registry_root": vroot("phases"),
        "m3_transition_reason_registry_root": vroot("reasons"),
        "legal_transition_table": (transition_object,),
        "terminal_state_ids": (2, 3, 4, 5, 6),
        "reopen_allowed": False,
    }
    build_formal_object("StateMachineContractV1", state_contract)
    traversal = {
        "bucket_key_field_ids": (1, 2, 3),
        "canonical_sort_key_field_ids": (1, 2, 3, 4, 5),
        "commutative_child_ordering_rule_id_digest": id_digest_v1("rule:children-v1"),
        "maximum_canonical_programs": 50_000,
        "maximum_raw_operator_applications": 1_000_000,
        "frontier_exhaustion_definition_id_digest": id_digest_v1("rule:frontier-v1"),
    }
    build_formal_object("TraversalContractV1", traversal)
    assert error_code(
        build_formal_object,
        "TraversalContractV1",
        dict(traversal, canonical_sort_key_field_ids=(1, 1)),
    ) == "REJECT_M25_FIELD_VALUE"


def test_sink_witness_fields_must_match_and_authority_remains_closed() -> None:
    witness = vroot("sink-witness")
    target = {
        "role_id": 2,
        "target_machine_id_digest": id_digest_v1("target:sink-null-v1"),
        "input_signature_spec_root": vroot("sink-signature"),
        "output_sort_id": 2,
        "target_rule_id_digest": id_digest_v1("rule:sink-null-v1"),
        "universe_row_count": 85,
        "target_output_cardinality": 1,
        "required_witness_ast_hash_or_null": witness,
        "claim_level_id": 1,
    }
    bundle = {
        "outside_target_spec_root": vroot("outside-spec"),
        "outside_target_universe_root": vroot("outside-universe"),
        "outside_target_truth_root": vroot("outside-truth"),
        "null_control_spec_root": vroot("null-spec"),
        "null_control_universe_root": vroot("null-universe"),
        "null_control_truth_root": vroot("null-truth"),
        "fallback_registry_root": vroot("fallback"),
        "null_control_required_witness_ast_hash_or_null": witness,
        "null_control_claim_level_id": 1,
    }
    validate_null_witness_binding_v1(target, bundle)
    assert error_code(
        validate_null_witness_binding_v1,
        target,
        dict(bundle, null_control_required_witness_ast_hash_or_null=vroot("other")),
    ) == "FAIL_NULL_WITNESS_BINDING_MISMATCH"
    assert error_code(formal_content_root, "BridgeReplayStatementV1", {}) == (
        FAIL_M25_NORMATIVE_GAP
    )
