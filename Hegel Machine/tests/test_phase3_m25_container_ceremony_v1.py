from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess

import pytest

import hegel_machine.phase3_m25_container_ceremony_v1 as ceremony
from hegel_machine.phase3_m25_container_ceremony_v1 import (
    FAIL_FORMAL_PROMOTION_CONTEXT,
    FAIL_MARKER_ALREADY_EXISTS,
    FAIL_MARKER_RECOVERY_REQUIRED,
    FAIL_PUBLIC_BASIS_REGISTRY_PREIMAGE,
    FAIL_SIGNATURE_INVALID,
    FAIL_SPLIT_FULL_ENDPOINT_REQUIRED,
    FAIL_SPLIT_RESPONSE_FRAMING,
    GATE_NAMES,
    M25ContainerCeremonyError,
    ODD_TRUTH_ROOT,
    ODD_UNIVERSE_ROOT,
    QualifiedGateEvidenceV1,
    SINK_TRUTH_ROOT,
    SINK_UNIVERSE_ROOT,
    SPLIT_RESPONSE_ROWS,
    TECHNICAL_ACTOR_DISCLOSURE_V1,
    SplitCalculatorPublicResponseV2,
    SplitRootCommitment,
    build_committed_public_basis_candidates_v1,
    build_actor_key_manifest_fields_v1,
    build_offline_actor_invocation_v1,
    complete_marker_v1,
    create_pending_marker_v1,
    decode_split_calculator_public_frame_v2,
    encode_split_calculator_public_frame_v2,
    promote_gate_evidence_v1,
    parse_ed25519_spki_der_v1,
    read_marker_snapshot_v1,
    require_full_split_response_agreement_v2,
    validate_single_signature_envelope_v1,
    validate_no_secret_transport_v1,
)
from hegel_machine.phase3_m25_split_v1 import ed25519_key_id
from hegel_machine.phase3_m25_wire_v1 import (
    OBJECT_TAGS,
    candidate_content_root,
    external_signature_preimage_v1,
)


def _code(action, *args, **kwargs) -> str:
    with pytest.raises(M25ContainerCeremonyError) as captured:
        action(*args, **kwargs)
    return captured.value.code


def _split_response() -> SplitCalculatorPublicResponseV2:
    return SplitCalculatorPublicResponseV2(
        seed_commitment=bytes(range(32)),
        partitions=tuple(
            SplitRootCommitment(role, partition, count, bytes((index,)) * 32)
            for index, (role, partition, count) in enumerate(
                SPLIT_RESPONSE_ROWS, start=1
            )
        ),
    )


def test_full_split_fd5_response_round_trips_exactly() -> None:
    response = _split_response()
    frame = encode_split_calculator_public_frame_v2(response)
    assert decode_split_calculator_public_frame_v2(frame) == response
    assert require_full_split_response_agreement_v2(frame, frame) == response
    assert tuple(response.roots) == (
        "outside_discovery_split_root",
        "outside_validation_split_root",
        "outside_sealed_split_root",
        "null_discovery_split_root",
        "null_validation_split_root",
        "null_sealed_split_root",
    )


def test_split_response_rejects_legacy_commitment_only_and_bad_framing() -> None:
    legacy_payload = bytes.fromhex(
        "8301582e686567656c2d7068617365332d73706c69742d63616c63756c61746f722d"
        "6664332d726573706f6e73652f315820"
    ) + bytes(32)
    legacy_frame = len(legacy_payload).to_bytes(8, "big") + legacy_payload
    assert _code(
        require_full_split_response_agreement_v2, legacy_frame, legacy_frame
    ) == FAIL_SPLIT_FULL_ENDPOINT_REQUIRED

    frame = encode_split_calculator_public_frame_v2(_split_response())
    assert _code(
        decode_split_calculator_public_frame_v2, frame[:8] + frame[8:-1]
    ) == FAIL_SPLIT_RESPONSE_FRAMING


def test_split_response_rejects_quota_count_drift() -> None:
    response = _split_response()
    first = replace(response.partitions[0], row_count=191)
    drifted = replace(response, partitions=(first, *response.partitions[1:]))
    assert _code(encode_split_calculator_public_frame_v2, drifted) == (
        "FAIL_M25_SPLIT_FD5_RESPONSE_SCHEMA"
    )


def test_committed_public_basis_uses_real_docs_and_exact_typed_rows() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repository_root = project_root.parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    basis = build_committed_public_basis_candidates_v1(commit)
    assert basis["complete"] is False
    assert basis["formal_promotion_allowed"] is False
    assert basis["failure_code"] == FAIL_PUBLIC_BASIS_REGISTRY_PREIMAGE
    assert basis["typed_role_roots"] == {
        "outside_target_universe_root": ODD_UNIVERSE_ROOT,
        "outside_target_truth_root": ODD_TRUTH_ROOT,
        "null_control_universe_root": SINK_UNIVERSE_ROOT,
        "null_control_truth_root": SINK_TRUTH_ROOT,
    }
    assert len(basis["normative_document_objects"]) == 3
    assert len(basis["source_profile_objects"]) == 5


def test_offline_container_invocation_has_no_pull_network_or_secret_transport(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    output_directory.mkdir()
    invocation = build_offline_actor_invocation_v1(
        purpose_id=3,
        operation="sign",
        read_only_input_directory=input_directory,
        private_state_volume="hegel-m25-purpose-3-" + "ab" * 16,
        public_output_directory=output_directory,
        entrypoint="/input/actor-worker",
    )
    assert "--pull=never" in invocation.command
    assert "--network=none" in invocation.command
    assert "--read-only" in invocation.command
    assert "--cap-drop=ALL" in invocation.command
    assert invocation.image_ref.startswith("rust@sha256:")
    assert invocation.stdin_payload is None
    assert not any("seed" in item.lower() for item in invocation.command)
    assert _code(
        validate_no_secret_transport_v1,
        argv=("worker",),
        environment={"MASTER_SEED_HEX": "00"},
        stdin_payload=None,
    ) == "FAIL_M25_CEREMONY_SECRET_TRANSPORT"


def test_pending_marker_is_o_excl_and_complete_is_atomic(tmp_path: Path) -> None:
    secret_state = tmp_path / "outside-secret-state"
    secret_state.mkdir(mode=0o700)
    os.chmod(secret_state, 0o700)
    marker, pending = create_pending_marker_v1(
        secret_state_directory=secret_state,
        split_version_digest=bytes.fromhex("12" * 32),
        custodian_key_id=bytes.fromhex("34" * 16),
        created_at_unix_seconds=1_800_000_000,
    )
    assert pending.state == "PENDING"
    assert marker.stat().st_mode & 0o777 == 0o600
    assert _code(
        create_pending_marker_v1,
        secret_state_directory=secret_state,
        split_version_digest=bytes.fromhex("12" * 32),
        custodian_key_id=bytes.fromhex("34" * 16),
        created_at_unix_seconds=1_800_000_000,
    ) == FAIL_MARKER_RECOVERY_REQUIRED

    complete = complete_marker_v1(
        marker_path=marker, seed_commitment_manifest_root=bytes.fromhex("56" * 32)
    )
    assert complete.state == "COMPLETE"
    assert read_marker_snapshot_v1(marker) == complete
    assert _code(
        create_pending_marker_v1,
        secret_state_directory=secret_state,
        split_version_digest=bytes.fromhex("12" * 32),
        custodian_key_id=bytes.fromhex("34" * 16),
        created_at_unix_seconds=1_800_000_000,
    ) == FAIL_MARKER_ALREADY_EXISTS


def test_real_ed25519_envelope_signature_replays() -> None:
    # Fixed signature from the private seed 0x42 * 32.  The private seed is not
    # needed at replay time, so this test also proves the runtime path has no
    # Python ``cryptography`` dependency.
    public = bytes.fromhex(
        "2152f8d19b791d24453242e15f2eab6c"
        "b7cffa7b6a5ed30097960e069881db12"
    )
    signature = bytes.fromhex(
        "6faf8cbf3d6f2cbcdf9ecd10b0424de6"
        "3b8c70242a695f13b3b83a3ab993b0ea"
        "0d86db52d80185e1b75a102861ad8faa"
        "0a067dde9134dc82784832850273c80c"
    )
    key_id = ed25519_key_id(public)
    enclosed_root = bytes.fromhex("ab" * 32)
    tag = OBJECT_TAGS["SplitSeedCommitmentManifestV1"]
    assert external_signature_preimage_v1(tag, enclosed_root, 1, 0).hex() == (
        "484547454c2f435553544f4449414e5f53504c49545f534545445f434f4d4d"
        "49544d454e545f5349474e41545552452f563100"
        + "ab" * 32
        + "00010000000000000000"
    )
    envelope = {
        "enclosed_object_tag": tag,
        "enclosed_manifest_root": enclosed_root,
        "created_at_unix_seconds": 1_800_000_000,
        "signer_key_epoch": 0,
        "signatures": ((key_id, signature),),
    }
    assert validate_single_signature_envelope_v1(
        envelope_fields=envelope,
        signer_purpose_id=1,
        signer_key_id=key_id,
        signer_public_key=public,
    ) == candidate_content_root("SignedManifestEnvelopeV1", envelope)
    bad = dict(envelope, signatures=((key_id, bytes(64)),))
    assert _code(
        validate_single_signature_envelope_v1,
        envelope_fields=bad,
        signer_purpose_id=1,
        signer_key_id=key_id,
        signer_public_key=public,
    ) == FAIL_SIGNATURE_INVALID


def test_exact_ed25519_spki_and_key_manifest_identity() -> None:
    raw = bytes.fromhex("91" * 32)
    der = bytes.fromhex("302a300506032b6570032100") + raw
    public, key_id = parse_ed25519_spki_der_v1(der)
    assert public == raw
    assert key_id == ed25519_key_id(raw)
    manifest = build_actor_key_manifest_fields_v1(
        purpose_id=2,
        public_key=public,
        created_at_unix_seconds=1_800_000_000,
        basis_commit="12" * 20,
    )
    assert manifest["purpose_id"] == 2
    assert manifest["key_epoch"] == 0
    assert manifest["key_id"] == key_id
    assert _code(parse_ed25519_spki_der_v1, der[:-1]) == FAIL_SIGNATURE_INVALID


def test_caller_cannot_forge_formal_promotion_with_a_lookalike_result() -> None:
    fake = QualifiedGateEvidenceV1(
        basis_commit="0" * 40,
        gate_report={
            "all_gates_15_24_passed": True,
            "gates_after": 24,
            "child_state": "NOT_RUN",
            "m3_run_started": False,
        },
        formal_roots={"x": bytes(32)},
        _seal=object(),
    )
    assert _code(promote_gate_evidence_v1, fake) == FAIL_FORMAL_PROMOTION_CONTEXT
    assert list(GATE_NAMES) == list(range(15, 25))


def test_final_promotion_emits_exact_seven_field_technical_actor_disclosure() -> None:
    expected = {
        "same_admin_controller": True,
        "organizational_independence": False,
        "independent_human_actors": False,
        "technical_role_independence": True,
        "owner_accepted_threat_model": True,
        "remote_attestation": False,
        "hardware_key_nonexportability": False,
    }
    assert dict(TECHNICAL_ACTOR_DISCLOSURE_V1) == expected
    assert len(TECHNICAL_ACTOR_DISCLOSURE_V1) == 7
    with pytest.raises(TypeError):
        TECHNICAL_ACTOR_DISCLOSURE_V1["same_admin_controller"] = False  # type: ignore[index]

    evidence = QualifiedGateEvidenceV1(
        basis_commit="0" * 40,
        gate_report={
            "all_gates_15_24_passed": True,
            "gates_after": 24,
            "child_state": "NOT_RUN",
            "m3_run_started": False,
        },
        formal_roots={"test_formal_root": bytes(32)},
        _seal=ceremony._PROMOTION_SEAL,
    )
    promoted = promote_gate_evidence_v1(evidence)
    assert promoted["authority_disclosure"] == expected
    assert set(promoted["authority_disclosure"]) == set(expected)
