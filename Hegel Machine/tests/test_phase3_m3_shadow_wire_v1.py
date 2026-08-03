from __future__ import annotations

from hashlib import sha256

import pytest

from hegel_machine.phase3_m25_wire_v1 import M25WireError, decode_formal_object
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode, rfc6962_root
from hegel_machine.phase3_m3_shadow_wire_v1 import (
    FORMAL_CHILD_DSL_ID,
    FORMAL_MACHINE_FREEZE_ID,
    FORMAL_TRACK_SNAPSHOT,
    LEGAL_SHADOW_TRANSITIONS,
    SHADOW_ALL_GATES_BITSET,
    SHADOW_ARTIFACT_KIND,
    SHADOW_ARTIFACT_KIND_ID,
    SHADOW_HASH_DOMAINS,
    SHADOW_ISOLATION_INVARIANT_BITSET,
    SHADOW_OBJECT_TAGS,
    SHADOW_SCHEMA_REGISTRY,
    SHADOW_TRACK_ID,
    ShadowAdmissionGateId,
    ShadowLandlockStatusId,
    ShadowPurposeId,
    ShadowStateId,
    ShadowTransitionReasonId,
    ShadowWireError,
    build_shadow_object,
    decode_shadow_object,
    encode_shadow_object,
    git_sha1_commit_id,
    require_shadow_admission,
    shadow_digest_v1,
    shadow_gate_bitset,
    shadow_object_digest,
    shadow_purpose_worker_digest_set_v1,
    shadow_security_probe_set_digest_v1,
    shadow_signature_preimage_v1,
    shadow_tree_digest_v1,
    validate_formal_track_snapshot,
    validate_shadow_artifact_header,
    validate_shadow_state_chain_link,
    validate_shadow_state_transition,
)


DIGEST = bytes(range(32))
OTHER_DIGEST = bytes(reversed(range(32)))
RUN_ID = bytes(range(1, 17))
WORKER_ID = bytes(range(17, 33))
KEY_ID = bytes(range(33, 49))
COMMIT = git_sha1_commit_id(bytes(range(20)))


def _policy_fields() -> dict[str, object]:
    return {
        "artifact_kind_id": 1,
        "shadow_track_id": SHADOW_TRACK_ID.encode("ascii"),
        "formal_machine_freeze_id": FORMAL_MACHINE_FREEZE_ID.encode("ascii"),
        "formal_child_dsl_id": FORMAL_CHILD_DSL_ID.encode("ascii"),
        "amendment_git_blob_sha256": DIGEST,
        "basis_commit_id": COMMIT,
    }


def _worker_fields(purpose: int = 1) -> dict[str, object]:
    return {
        "artifact_kind_id": 1,
        "shadow_run_id": RUN_ID,
        "shadow_purpose_id": purpose,
        "worker_instance_id": WORKER_ID,
        "isolation_profile_id": 1,
        "basis_commit_id": COMMIT,
        "snapshot_manifest_digest": DIGEST,
        "executable_manifest_digest": OTHER_DIGEST,
        "environment_manifest_digest": sha256(b"env").digest(),
        "namespace_manifest_digest": sha256(b"ns").digest(),
        "ephemeral_key_id": KEY_ID,
        "ephemeral_public_key": sha256(b"public").digest(),
        "key_epoch": 0,
        "external_independence_claim": False,
    }


def _security_probe_fields(
    *,
    purpose: int = 1,
    phase: int = 1,
    landlock: int = 1,
    incident_count: int = 0,
) -> dict[str, object]:
    return {
        "artifact_kind_id": 1,
        "shadow_run_id": RUN_ID,
        "shadow_purpose_id": purpose,
        "probe_phase_id": phase,
        "worker_instance_id": WORKER_ID,
        "basis_commit_id": COMMIT,
        "proc_status_seccomp_value": 2,
        "proc_status_no_new_privs_value": 1,
        "attack_syscall_errno_rows": [[index, 1] for index in range(1, 7)],
        "landlock_status_id": int(landlock),
        "landlock_nonblocking_gap_disclosed": landlock != 1,
        "transient_capability_probe_incident_count": incident_count,
        "transient_capability_probe_incident_digest_or_null": (
            None if incident_count == 0 else sha256(b"incident").digest()
        ),
        "observed_at_unix_seconds": 1_700_000_000,
        "external_security_attestation_claim": False,
    }


def _plan_fields() -> dict[str, object]:
    return {
        "artifact_kind_id": 1,
        "shadow_run_id": RUN_ID,
        "basis_commit_id": COMMIT,
        "snapshot_manifest_digest": DIGEST,
        "purpose_ids": [1, 2, 3, 4],
        "isolation_profile_id": 1,
        "worker_launch_plan_digest": sha256(b"launch").digest(),
        "required_security_probe_digest": sha256(b"seccomp probes").digest(),
        "fd_policy_digest": sha256(b"fd").digest(),
        "output_allowlist_digest": sha256(b"allow").digest(),
        "secret_lint_policy_digest": sha256(b"lint").digest(),
        "external_independence_claim": False,
    }


def _isolation_fields() -> dict[str, object]:
    return {
        "artifact_kind_id": 1,
        "shadow_run_id": RUN_ID,
        "basis_commit_id": COMMIT,
        "snapshot_manifest_digest": DIGEST,
        "purpose_worker_digests": [sha256(bytes([i])).digest() for i in range(1, 5)],
        "isolation_invariant_bitset": SHADOW_ISOLATION_INVARIANT_BITSET,
        "required_security_probe_digest": sha256(b"runtime probes").digest(),
        "fd_policy_digest": sha256(b"fd").digest(),
        "output_allowlist_digest": sha256(b"allow").digest(),
        "secret_lint_policy_digest": sha256(b"lint").digest(),
        "created_at_unix_seconds": 1_700_000_001,
        "external_independence_claim": False,
    }


def _admission_fields() -> dict[str, object]:
    return {
        "artifact_kind_id": 1,
        "shadow_run_id": RUN_ID,
        "policy_binding_digest": sha256(b"policy").digest(),
        "isolation_plan_digest": sha256(b"plan").digest(),
        "basis_commit_id": COMMIT,
        "shadow_gate_bitset": 0x0FFF,
        "shadow_gate_count": 12,
        "formal_gates_satisfied": 14,
        "formal_gates_total": 24,
        "formal_m3_state_id": 0,
        "formal_roots_all_null": True,
        "external_actor_evidence": False,
        "admitted_at_unix_seconds": 1_700_000_002,
    }


def _genesis_fields() -> dict[str, object]:
    return {
        "artifact_kind_id": 1,
        "shadow_run_id": RUN_ID,
        "policy_binding_digest": sha256(b"policy").digest(),
        "admission_receipt_digest": sha256(b"admission").digest(),
        "isolation_manifest_digest": sha256(b"isolation").digest(),
        "basis_commit_id": COMMIT,
        "initial_shadow_state_id": 1,
        "canonical_program_archive_digest_or_null": None,
        "program_chunk_manifest_digest_or_null": None,
        "bucket_accounting_digest_or_null": None,
        "odd_output_archive_digest_or_null": None,
        "odd_match_set_digest_or_null": None,
        "odd_role_receipt_digest_or_null": None,
        "sink_output_archive_digest_or_null": None,
        "sink_match_set_digest_or_null": None,
        "sink_role_receipt_digest_or_null": None,
        "dual_replay_agreement_digest_or_null": None,
        "created_at_unix_seconds": 1_700_000_003,
        "formal_run_genesis_claim": False,
    }


def _state_fields(
    *,
    index: int = 0,
    previous: bytes | None = None,
    from_state: int = 0,
    to_state: int = 1,
    reason: int = 1,
) -> dict[str, object]:
    return {
        "artifact_kind_id": 1,
        "shadow_run_id": RUN_ID,
        "transition_index": index,
        "previous_state_record_digest_or_null": previous,
        "from_shadow_state_id": from_state,
        "to_shadow_state_id": to_state,
        "transition_reason_id": reason,
        "triggering_shadow_receipt_digest_or_null": sha256(b"trigger").digest(),
        "recorded_at_unix_seconds": 1_700_000_004,
        "formal_gates_satisfied": 14,
        "formal_gates_total": 24,
        "formal_m3_state_id": 0,
    }


def _all_gate_results(value: bool = True) -> dict[str, bool]:
    return {gate.name: value for gate in ShadowAdmissionGateId}


def test_tag_and_domain_registries_are_complete_and_disjoint() -> None:
    assert len(SHADOW_OBJECT_TAGS) == 14
    assert len(set(SHADOW_OBJECT_TAGS.values())) == 14
    assert all(0x7A00 <= tag <= 0x7AFF for tag in SHADOW_OBJECT_TAGS.values())
    assert set(SHADOW_SCHEMA_REGISTRY) == {
        "ShadowPolicyBindingV1", "ShadowPurposeWorkerManifestV1",
        "ShadowIsolationManifestV1", "ShadowAdmissionReceiptV1",
        "ShadowEnvelopeV1", "ShadowRunGenesisV1", "ShadowStateRecordV1",
        "ShadowIsolationPlanV1", "ShadowSecurityProbeReceiptV1",
    }
    assert len(SHADOW_HASH_DOMAINS) == 14


def test_policy_binding_round_trip_and_digest() -> None:
    fields = _policy_fields()
    value = build_shadow_object("ShadowPolicyBindingV1", fields)
    payload = encode_shadow_object("ShadowPolicyBindingV1", fields)
    decoded = decode_shadow_object(payload, expected_name="ShadowPolicyBindingV1")
    assert decoded.value == value
    assert decoded.fields["basis_commit_id"] == COMMIT
    expected = sha256(
        b"HEGEL/INTERNAL_SHADOW/NON_AUTHORITATIVE/POLICY_BINDING/V1\x00"
        + canonical_cbor_encode(value)
    ).digest()
    assert shadow_object_digest("ShadowPolicyBindingV1", fields) == expected


def test_formal_decoder_rejects_shadow_and_shadow_decoder_rejects_formal() -> None:
    with pytest.raises(M25WireError) as formal_rejection:
        decode_formal_object(encode_shadow_object("ShadowPolicyBindingV1", _policy_fields()))
    assert formal_rejection.value.code == "REJECT_UNKNOWN_M25_SCHEMA"

    formal_shaped = canonical_cbor_encode((1, 0x3001, b"formal-object"))
    with pytest.raises(ShadowWireError) as shadow_rejection:
        decode_shadow_object(formal_shaped)
    assert shadow_rejection.value.code == "REJECT_SHADOW_FORMAL_TAG_NAMESPACE"


def test_shadow_decoder_rejects_tag_schema_aliasing() -> None:
    value = list(build_shadow_object("ShadowPolicyBindingV1", _policy_fields()))
    value[2] = b"hegel-internal-shadow-isolation-plan/1"
    with pytest.raises(ShadowWireError) as exc:
        decode_shadow_object(canonical_cbor_encode(value))
    assert exc.value.code == "REJECT_UNKNOWN_SHADOW_SCHEMA"


@pytest.mark.parametrize(
    "domain",
    ["policy_binding", "POLICY-BINDING", "HEGEL/POLICY_BINDING/V1", "FORMAL_ROOT", "策略"],
)
def test_shadow_digest_rejects_nonexact_or_colliding_domains(domain: str) -> None:
    with pytest.raises(ShadowWireError) as exc:
        shadow_digest_v1(domain, (1, 2, 3))
    assert exc.value.code == "FAIL_SHADOW_DOMAIN_COLLISION"


def test_shadow_tree_has_separate_empty_leaf_node_preimages() -> None:
    prefix = b"HEGEL/INTERNAL_SHADOW/NON_AUTHORITATIVE/TREE/SECURITY_PROBE_SET/V1\x00"
    rows = [(1, b"a"), (2, b"b"), (3, b"c")]
    assert shadow_tree_digest_v1("SECURITY_PROBE_SET", []) == sha256(prefix + b"\x02").digest()
    leaf = sha256(prefix + b"\x00" + canonical_cbor_encode(rows[0])).digest()
    assert shadow_tree_digest_v1("SECURITY_PROBE_SET", [rows[0]]) == leaf
    assert shadow_tree_digest_v1("SECURITY_PROBE_SET", rows) != rfc6962_root(rows)
    with pytest.raises(ShadowWireError) as exc:
        shadow_tree_digest_v1("FORMAL_PROGRAM_ARCHIVE", rows)
    assert exc.value.code == "FAIL_SHADOW_DOMAIN_COLLISION"


def test_worker_plan_and_isolation_round_trip() -> None:
    for name, fields in (
        ("ShadowPurposeWorkerManifestV1", _worker_fields()),
        ("ShadowIsolationPlanV1", _plan_fields()),
        ("ShadowIsolationManifestV1", _isolation_fields()),
    ):
        decoded = decode_shadow_object(encode_shadow_object(name, fields))
        assert decoded.schema.name == name


def test_worker_set_requires_exact_order_common_basis_and_unique_identity() -> None:
    workers = []
    for purpose in range(1, 5):
        fields = _worker_fields(purpose)
        fields["worker_instance_id"] = bytes([purpose]) * 16
        fields["ephemeral_key_id"] = bytes([purpose + 4]) * 16
        fields["ephemeral_public_key"] = bytes([purpose + 8]) * 32
        workers.append(fields)
    digests = shadow_purpose_worker_digest_set_v1(workers)
    assert len(digests) == 4
    assert len(set(digests)) == 4

    workers[3]["ephemeral_key_id"] = workers[0]["ephemeral_key_id"]
    with pytest.raises(ShadowWireError) as exc:
        shadow_purpose_worker_digest_set_v1(workers)
    assert exc.value.code == "FAIL_SHADOW_KEY_REUSE"


def test_worker_rejects_key_epoch_and_external_claim() -> None:
    fields = _worker_fields()
    fields["key_epoch"] = 1
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowPurposeWorkerManifestV1", fields)
    assert exc.value.code == "FAIL_SHADOW_KEY_REUSE"

    fields = _worker_fields()
    fields["external_independence_claim"] = True
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowPurposeWorkerManifestV1", fields)
    assert exc.value.code == "FAIL_SHADOW_FORBIDDEN_CLAIM"


def test_plan_requires_exact_purpose_set_and_security_probe_digest() -> None:
    fields = _plan_fields()
    fields["purpose_ids"] = [1, 2, 4, 3]
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowIsolationPlanV1", fields)
    assert exc.value.code == "FAIL_SHADOW_PURPOSE_SET"

    fields = _plan_fields()
    fields["purpose_ids"] = [True, 2, 3, 4]
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowIsolationPlanV1", fields)
    assert exc.value.code == "FAIL_SHADOW_PURPOSE_SET"

    fields = _plan_fields()
    fields.pop("required_security_probe_digest")
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowIsolationPlanV1", fields)
    assert exc.value.code == "REJECT_SHADOW_FIELD_SET"


def test_isolation_manifest_requires_exact_18_bit_profile_and_distinct_workers() -> None:
    fields = _isolation_fields()
    fields["isolation_invariant_bitset"] = SHADOW_ISOLATION_INVARIANT_BITSET | (1 << 18)
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowIsolationManifestV1", fields)
    assert exc.value.code == "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE"

    fields = _isolation_fields()
    fields["purpose_worker_digests"] = [DIGEST] * 4
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowIsolationManifestV1", fields)
    assert exc.value.code == "FAIL_SHADOW_PROCESS_REUSE"


@pytest.mark.parametrize("landlock", list(ShadowLandlockStatusId))
@pytest.mark.parametrize("phase", [1, 2])
def test_security_probe_exact_happy_paths(landlock: ShadowLandlockStatusId, phase: int) -> None:
    fields = _security_probe_fields(landlock=landlock, phase=phase)
    payload = encode_shadow_object("ShadowSecurityProbeReceiptV1", fields)
    decoded = decode_shadow_object(payload)
    assert decoded.fields["proc_status_seccomp_value"] == 2
    assert decoded.fields["proc_status_no_new_privs_value"] == 1


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("proc_status_seccomp_value", 0), ("proc_status_no_new_privs_value", 0)],
)
def test_security_probe_rejects_missing_live_seccomp_invariant(field: str, bad_value: int) -> None:
    fields = _security_probe_fields()
    fields[field] = bad_value
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowSecurityProbeReceiptV1", fields)
    assert exc.value.code == "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE"


def test_security_probe_requires_six_real_eperm_results() -> None:
    fields = _security_probe_fields()
    fields["attack_syscall_errno_rows"] = [[index, 1] for index in range(1, 6)]
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowSecurityProbeReceiptV1", fields)
    assert exc.value.code == "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE"

    fields = _security_probe_fields()
    fields["attack_syscall_errno_rows"][0] = [True, True]  # type: ignore[index]
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowSecurityProbeReceiptV1", fields)
    assert exc.value.code == "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE"

    fields = _security_probe_fields()
    fields["attack_syscall_errno_rows"][5][1] = 22  # type: ignore[index]
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowSecurityProbeReceiptV1", fields)
    assert exc.value.code == "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE"


def test_security_probe_landlock_gap_and_incident_binding_are_exact() -> None:
    fields = _security_probe_fields(landlock=ShadowLandlockStatusId.UNAVAILABLE)
    fields["landlock_nonblocking_gap_disclosed"] = False
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowSecurityProbeReceiptV1", fields)
    assert exc.value.code == "FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED"


def test_security_probe_set_binds_exact_purpose_order_phase_and_incident() -> None:
    probes = []
    for purpose in range(1, 5):
        fields = _security_probe_fields(purpose=purpose, phase=1, incident_count=1)
        fields["worker_instance_id"] = bytes([purpose]) * 16
        probes.append(fields)
    digest = shadow_security_probe_set_digest_v1(probes, expected_phase=1)
    rows = [build_shadow_object("ShadowSecurityProbeReceiptV1", fields) for fields in probes]
    assert digest == shadow_tree_digest_v1("SECURITY_PROBE_SET", rows)

    probes[3]["probe_phase_id"] = 2
    with pytest.raises(ShadowWireError) as exc:
        shadow_security_probe_set_digest_v1(probes, expected_phase=1)
    assert exc.value.code == "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE"

    fields = _security_probe_fields(incident_count=1)
    fields["transient_capability_probe_incident_digest_or_null"] = None
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowSecurityProbeReceiptV1", fields)
    assert exc.value.code == "FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED"


def test_admission_gate_registry_is_exact_12_bit_and_fail_closed() -> None:
    passing = _all_gate_results()
    assert shadow_gate_bitset(passing) == SHADOW_ALL_GATES_BITSET
    assert require_shadow_admission(passing) == SHADOW_ALL_GATES_BITSET

    failing = _all_gate_results()
    failing[ShadowAdmissionGateId.LOCAL_NAMESPACE_AND_SECCOMP_ISOLATION_AVAILABLE.name] = False
    assert shadow_gate_bitset(failing) == SHADOW_ALL_GATES_BITSET & ~(1 << 7)
    with pytest.raises(ShadowWireError) as exc:
        require_shadow_admission(failing)
    assert exc.value.code == "FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE"

    incomplete = _all_gate_results()
    incomplete.pop(ShadowAdmissionGateId.SHADOW_OWNER_POLICY_BOUND.name)
    with pytest.raises(ShadowWireError) as exc:
        shadow_gate_bitset(incomplete)
    assert exc.value.code == "FAIL_SHADOW_ADMISSION_INCOMPLETE"


def test_admission_receipt_reasserts_formal_snapshot() -> None:
    decoded = decode_shadow_object(
        encode_shadow_object("ShadowAdmissionReceiptV1", _admission_fields())
    )
    assert decoded.fields["shadow_gate_bitset"] == 0x0FFF

    fields = _admission_fields()
    fields["formal_gates_satisfied"] = 15
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowAdmissionReceiptV1", fields)
    assert exc.value.code == "FAIL_SHADOW_FORMAL_STATE_MUTATION"

    fields = _admission_fields()
    fields["shadow_gate_bitset"] = 0x07FF
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowAdmissionReceiptV1", fields)
    assert exc.value.code == "FAIL_SHADOW_ADMISSION_INCOMPLETE"


def test_genesis_has_exactly_ten_null_candidate_slots() -> None:
    decoded = decode_shadow_object(
        encode_shadow_object("ShadowRunGenesisV1", _genesis_fields())
    )
    null_slots = [
        value for field, value in decoded.fields.items() if field.endswith("_digest_or_null")
    ]
    assert null_slots == [None] * 10

    fields = _genesis_fields()
    fields["odd_match_set_digest_or_null"] = DIGEST
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowRunGenesisV1", fields)
    assert exc.value.code == "FAIL_SHADOW_FORBIDDEN_CLAIM"


def test_all_and_only_frozen_shadow_transitions_validate() -> None:
    assert len(LEGAL_SHADOW_TRANSITIONS) == 17
    for transition in LEGAL_SHADOW_TRANSITIONS:
        assert validate_shadow_state_transition(*transition) == transition

    with pytest.raises(ShadowWireError) as exc:
        validate_shadow_state_transition(1, 3, 2)
    assert exc.value.code == "FAIL_SHADOW_INVALID_TRANSITION"
    with pytest.raises(ShadowWireError) as exc:
        validate_shadow_state_transition(4, 2, 2)
    assert exc.value.code == "FAIL_SHADOW_INVALID_TRANSITION"
    with pytest.raises(ShadowWireError) as exc:
        validate_shadow_state_transition(True, 1, 1)
    assert exc.value.code == "FAIL_SHADOW_INVALID_TRANSITION"


def test_state_record_admission_and_start_guards() -> None:
    decode_shadow_object(encode_shadow_object("ShadowStateRecordV1", _state_fields()))
    start = _state_fields(
        index=1,
        previous=sha256(b"state0").digest(),
        from_state=1,
        to_state=2,
        reason=2,
    )
    decode_shadow_object(encode_shadow_object("ShadowStateRecordV1", start))

    start["previous_state_record_digest_or_null"] = None
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowStateRecordV1", start)
    assert exc.value.code == "FAIL_SHADOW_INVALID_TRANSITION"

    wrong_index_zero = _state_fields(from_state=1, to_state=2, reason=2)
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowStateRecordV1", wrong_index_zero)
    assert exc.value.code == "FAIL_SHADOW_INVALID_TRANSITION"


def test_formal_snapshot_is_complete_exact_and_immutable_by_validation() -> None:
    validate_formal_track_snapshot(dict(FORMAL_TRACK_SNAPSHOT))
    missing = dict(FORMAL_TRACK_SNAPSHOT)
    missing.pop("formal_roots")
    with pytest.raises(ShadowWireError) as exc:
        validate_formal_track_snapshot(missing)
    assert exc.value.code == "FAIL_SHADOW_FORMAL_STATUS_OMITTED"

    changed = dict(FORMAL_TRACK_SNAPSHOT)
    changed["m3_entry_allowed"] = True
    with pytest.raises(ShadowWireError) as exc:
        validate_formal_track_snapshot(changed)
    assert exc.value.code == "FAIL_SHADOW_FORMAL_STATE_MUTATION"


def test_public_artifact_header_cannot_claim_external_or_formal_authority() -> None:
    validate_shadow_artifact_header(
        artifact_kind_id=SHADOW_ARTIFACT_KIND_ID,
        artifact_kind=SHADOW_ARTIFACT_KIND,
        external_independence_claim=False,
        formal_evidence_claim=False,
    )
    with pytest.raises(ShadowWireError) as exc:
        validate_shadow_artifact_header(
            artifact_kind_id=1,
            artifact_kind=SHADOW_ARTIFACT_KIND,
            external_independence_claim=True,
            formal_evidence_claim=False,
        )
    assert exc.value.code == "FAIL_SHADOW_FORBIDDEN_CLAIM"


def test_signature_preimage_is_exact_and_epoch_zero_only() -> None:
    expected = (
        b"HEGEL/INTERNAL_SHADOW/NON_AUTHORITATIVE/SIGNATURE/V1\x00"
        + DIGEST
        + (2).to_bytes(2, "big")
        + (0).to_bytes(8, "big")
        + RUN_ID
    )
    assert shadow_signature_preimage_v1(DIGEST, ShadowPurposeId(2), 0, RUN_ID) == expected
    with pytest.raises(ShadowWireError) as exc:
        shadow_signature_preimage_v1(DIGEST, 2, 1, RUN_ID)
    assert exc.value.code == "FAIL_SHADOW_KEY_REUSE"


def test_shadow_state_chain_link_is_exact() -> None:
    validate_shadow_state_chain_link(DIGEST, DIGEST)
    validate_shadow_state_chain_link(None, None)
    with pytest.raises(ShadowWireError) as exc:
        validate_shadow_state_chain_link(DIGEST, OTHER_DIGEST)
    assert exc.value.code == "FAIL_SHADOW_INVALID_TRANSITION"


def test_policy_rejects_wrong_machine_identity_and_noncanonical_field_types() -> None:
    fields = _policy_fields()
    fields["formal_machine_freeze_id"] = b"different"
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowPolicyBindingV1", fields)
    assert exc.value.code == "FAIL_SHADOW_POLICY_NOT_BOUND"

    fields = _policy_fields()
    fields["artifact_kind_id"] = True
    with pytest.raises(ShadowWireError) as exc:
        build_shadow_object("ShadowPolicyBindingV1", fields)
    assert exc.value.code == "FAIL_SHADOW_ARTIFACT_KIND"
