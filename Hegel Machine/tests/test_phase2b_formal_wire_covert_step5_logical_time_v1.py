from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import phase2b_formal_wire_covert_step5_logical_time_v1 as logical_time
from hegel_machine.rfc8785_v1 import canonicalize_rfc8785_v1


_REQUEST_ID_DOMAIN = b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/REQUEST_ID/V1\x00"
_TOKEN_ID_DOMAIN = b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/TOKEN_ID/V1\x00"
_TOKEN_SIGNATURE_DOMAIN = b"HEGEL/PHASE2B/STEP5/SIGN/LOGICAL_TIME_TOKEN/V1\x00"
_HEAD_ID_DOMAIN = b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/LEDGER_HEAD_ID/V1\x00"
_CHECKPOINT_ID_DOMAIN = (
    b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/LEDGER_CHECKPOINT_ID/V1\x00"
)


@dataclass(frozen=True)
class Material:
    private_key: Ed25519PrivateKey
    public_key: bytes
    key_id: str
    authority_profile_id: str
    signer_registry_id: str
    attempt_context_id: str


@dataclass(frozen=True)
class Issued:
    material: Material
    root: Path
    request_payload: bytes
    previous_head_payload: bytes
    result: logical_time.LogicalTimeServiceIssuanceV1


def _raw_public(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _material(*, seed: bytes = bytes(range(32)), attempt_tag: str = "c") -> Material:
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    public_key = _raw_public(private_key)
    return Material(
        private_key=private_key,
        public_key=public_key,
        key_id=logical_time.derive_step5_logical_time_role_key_id_v1(public_key),
        authority_profile_id=(
            logical_time.AUTHORITY_PROFILE_PREFIX_V2 + ("a" * 64)
        ),
        signer_registry_id=(logical_time.SIGNER_REGISTRY_PREFIX_V2 + ("b" * 64)),
        attempt_context_id=f"phase2b_step5_attempt_{attempt_tag * 64}",
    )


def _request(
    material: Material,
    *,
    nonce: str = "d" * 64,
    purpose: str = "R01_STATUS_OBSERVED",
    artifact_tag: str = "e",
    artifact_sha_tag: str = "f",
    prior_token_id: str = logical_time.GENESIS_TOKEN_ID_V1,
    next_sequence: int = 0,
) -> bytes:
    return logical_time.build_logical_time_request_v1(
        authority_profile_id=material.authority_profile_id,
        signer_registry_id=material.signer_registry_id,
        logical_time_key_id=material.key_id,
        logical_time_key_epoch=1,
        attempt_context_id=material.attempt_context_id,
        request_nonce=nonce,
        purpose=purpose,
        bound_artifact_id=f"artifact_{artifact_tag * 64}",
        bound_artifact_sha256=f"sha256:{artifact_sha_tag * 64}",
        prior_token_id=prior_token_id,
        next_sequence=next_sequence,
    )


def _ledger(root: Path, material: Material, *, create: bool) -> logical_time.LogicalTimeLedgerV1:
    return logical_time.LogicalTimeLedgerV1(
        root_directory=str(root),
        authority_profile_id=material.authority_profile_id,
        signer_registry_id=material.signer_registry_id,
        logical_time_key_id=material.key_id,
        logical_time_key_epoch=1,
        attempt_context_id=material.attempt_context_id,
        logical_time_public_key=material.public_key,
        create=create,
    )


def _json(payload: bytes) -> dict[str, object]:
    value = json.loads(payload.decode("utf-8"))
    assert type(value) is dict
    return value


def _canonical(value: object) -> bytes:
    return canonicalize_rfc8785_v1(value)  # type: ignore[arg-type]


def _content_id(prefix: str, domain: bytes, body: object) -> str:
    return prefix + hashlib.sha256(domain + _canonical(body)).hexdigest()


def _readdress_request(body: dict[str, object]) -> bytes:
    subject = {key: value for key, value in body.items() if key != "request_id"}
    body["request_id"] = _content_id(
        logical_time.REQUEST_ID_PREFIX_V1, _REQUEST_ID_DOMAIN, subject
    )
    return _canonical(body)


def _readdress_token(body: dict[str, object]) -> bytes:
    subject = {key: value for key, value in body.items() if key != "token_id"}
    body["token_id"] = _content_id(
        logical_time.TOKEN_ID_PREFIX_V1, _TOKEN_ID_DOMAIN, subject
    )
    return _canonical(body)


def _readdress_head(body: dict[str, object]) -> bytes:
    subject = {key: value for key, value in body.items() if key != "head_id"}
    body["head_id"] = _content_id(
        logical_time.LEDGER_HEAD_ID_PREFIX_V1, _HEAD_ID_DOMAIN, subject
    )
    return _canonical(body)


def _readdress_checkpoint(body: dict[str, object]) -> bytes:
    subject = {key: value for key, value in body.items() if key != "checkpoint_id"}
    body["checkpoint_id"] = _content_id(
        logical_time.LEDGER_CHECKPOINT_ID_PREFIX_V1,
        _CHECKPOINT_ID_DOMAIN,
        subject,
    )
    return _canonical(body)


def _verify_token(
    material: Material, request_payload: bytes, token_payload: bytes
) -> logical_time.SignedLogicalTimeTokenV1:
    return logical_time.verify_signed_logical_time_token_v1(
        token_payload,
        request_payload=request_payload,
        logical_time_public_key=material.public_key,
        expected_authority_profile_id=material.authority_profile_id,
        expected_signer_registry_id=material.signer_registry_id,
        expected_logical_time_key_id=material.key_id,
        expected_logical_time_key_epoch=1,
        expected_attempt_context_id=material.attempt_context_id,
    )


def _verify_transition(
    material: Material,
    previous_head_payload: bytes,
    checkpoint_payload: bytes,
    next_head_payload: bytes,
) -> logical_time.LogicalTimeLedgerHeadV1:
    return logical_time.verify_logical_time_ledger_transition_v1(
        previous_head_payload,
        checkpoint_payload,
        next_head_payload,
        logical_time_public_key=material.public_key,
        expected_authority_profile_id=material.authority_profile_id,
        expected_signer_registry_id=material.signer_registry_id,
        expected_logical_time_key_id=material.key_id,
        expected_logical_time_key_epoch=1,
        expected_attempt_context_id=material.attempt_context_id,
    )


def _expect_code(code: str):
    return pytest.raises(
        logical_time.Phase2BStep5LogicalTimeErrorV1,
        match=f"^{code}:",
    )


@pytest.fixture
def issued(tmp_path: Path) -> Issued:
    material = _material()
    root = tmp_path / "ledger"
    with _ledger(root, material, create=True) as ledger:
        previous_head = ledger.current_head_bytes()
        request_payload = _request(material)
        result = ledger.issue_service_side_v1(
            request_payload=request_payload,
            expected_previous_head_payload=previous_head,
            private_key=material.private_key,
        )
    return Issued(
        material=material,
        root=root,
        request_payload=request_payload,
        previous_head_payload=previous_head,
        result=result,
    )


def test_claim_boundary_is_explicitly_mechanics_only() -> None:
    assert "MECHANICS_ONLY_NOT_AUTHORITY" in logical_time.AUTHORITY_CLAIM_BOUNDARY_V1
    assert "NOT_TRUSTED_UTC" in logical_time.AUTHORITY_CLAIM_BOUNDARY_V1
    assert "NOT_EVENT_ORDER_POLICY" in logical_time.AUTHORITY_CLAIM_BOUNDARY_V1
    assert "NOT_CAS_CHECKPOINT" in logical_time.AUTHORITY_CLAIM_BOUNDARY_V1
    assert logical_time.REQUIRED_EXTERNAL_AUTHORITY_CHECKS_V1 == (
        "OUT_OF_BAND_AUTHORITY_PROFILE_PIN_VERIFIED",
        "OUT_OF_BAND_SIGNER_REGISTRY_PIN_AND_LOGICAL_TIME_ROLE_POLICY_VERIFIED",
        "FRESH_STEP5_ATTEMPT_CONTEXT_VERIFIED",
        "PINNED_LOGICAL_TIME_SERVICE_IMAGE_AND_RUNTIME_CLOSURE_VERIFIED",
        "EXPECTED_DISTINCT_SERVICE_UID_VERIFIED",
        "STEP5_LOGICAL_TIME_CONTROL_DOMAIN_VERIFIED",
        "PRIVATE_KEY_CUSTODY_VERIFIED",
        "FROZEN_EVENT_ORDER_AND_TERMINAL_POLICY_VERIFIED",
        "EXTERNAL_DURABLE_CAS_CHECKPOINT_OF_COMPLETE_LEDGER_VERIFIED",
    )


def test_unfrozen_dependency_replacement_is_v2_only_without_schema_shape_bump() -> None:
    assert logical_time.AUTHORITY_PROFILE_PREFIX_V2 == (
        authority.AUTHORITY_PROFILE_PREFIX_V2
    )
    assert logical_time.SIGNER_REGISTRY_PREFIX_V2 == (
        authority.SIGNER_REGISTRY_PREFIX_V2
    )
    assert not hasattr(logical_time, "AUTHORITY_PROFILE_PREFIX_V1")
    assert not hasattr(logical_time, "SIGNER_REGISTRY_PREFIX_V1")
    assert "UNCOMMITTED_UNFROZEN_CANDIDATE" in (
        logical_time.DEPENDENCY_REPLACEMENT_V1
    )
    assert "NO_V1_OR_HYBRID_COMPATIBILITY" in (
        logical_time.DEPENDENCY_REPLACEMENT_V1
    )
    assert logical_time.LOGICAL_TIME_REQUEST_SCHEMA_VERSION.endswith("/1")
    assert logical_time.SIGNED_LOGICAL_TIME_TOKEN_SCHEMA_VERSION.endswith("/1")
    assert logical_time.LOGICAL_TIME_LEDGER_HEAD_SCHEMA_VERSION.endswith("/1")
    assert logical_time.LOGICAL_TIME_LEDGER_CHECKPOINT_SCHEMA_VERSION.endswith("/1")
    assert logical_time.ROLE_KEY_PREFIX_V1 == authority.ROLE_KEY_PREFIX_V1
    assert logical_time.SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1 == (
        "SIGN_LOGICAL_TIME_TOKEN"
    )


@pytest.mark.parametrize(
    ("profile_id", "registry_id"),
    [
        (
            "phase2b_step5_authority_profile_v1_" + "1" * 64,
            "phase2b_step5_signer_registry_v1_" + "2" * 64,
        ),
        (
            "phase2b_step5_authority_profile_v1_" + "1" * 64,
            logical_time.SIGNER_REGISTRY_PREFIX_V2 + "2" * 64,
        ),
        (
            logical_time.AUTHORITY_PROFILE_PREFIX_V2 + "1" * 64,
            "phase2b_step5_signer_registry_v1_" + "2" * 64,
        ),
    ],
)
def test_request_rejects_v1_and_v1_v2_hybrid_dependencies(
    profile_id: str, registry_id: str
) -> None:
    material = _material()
    with _expect_code(logical_time.FAIL_SCHEMA_V1):
        logical_time.build_logical_time_request_v1(
            authority_profile_id=profile_id,
            signer_registry_id=registry_id,
            logical_time_key_id=material.key_id,
            logical_time_key_epoch=1,
            attempt_context_id=material.attempt_context_id,
            request_nonce="d" * 64,
            purpose="R01_STATUS_OBSERVED",
            bound_artifact_id="artifact_" + "e" * 64,
            bound_artifact_sha256="sha256:" + "f" * 64,
            prior_token_id=logical_time.GENESIS_TOKEN_ID_V1,
            next_sequence=0,
        )


def test_role_key_id_exactly_matches_authority_registry_projection() -> None:
    material = _material()
    row = {
        "schema_version": "hegel-machine-phase2b-step5-role-key/1",
        "role": "LOGICAL_TIME",
        "control_domain": "STEP5_LOGICAL_TIME_CONTROL",
        "authorized_signature_purposes": ["SIGN_LOGICAL_TIME_TOKEN"],
        "algorithm": "Ed25519",
        "key_epoch": 1,
        "public_key_hex": material.public_key.hex(),
    }
    assert authority.phase2b_step5_role_key_id_v2(row) == material.key_id


def test_deterministic_wire_known_answer_vector(issued: Issued) -> None:
    assert issued.material.public_key.hex() == (
        "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8"
    )
    assert issued.material.key_id == (
        "phase2b_step5_role_key_v1_"
        "dad055e76e7f5b0d86735b6648e85d9fe867b3829dfb1767ac1ec24916a3f4f3"
    )
    assert hashlib.sha256(issued.request_payload).hexdigest() == (
        "ccea0b039fd2165ef19f7b1a09bb7f38cb246ed09260e9bd4e61d11349718eb3"
    )
    assert hashlib.sha256(issued.previous_head_payload).hexdigest() == (
        "f4b4bba5cdda70bdbc7e6b36773628c5f20dc5221d1da7863f4962b330cb8529"
    )
    assert hashlib.sha256(issued.result.signed_token_payload).hexdigest() == (
        "bba96c0bbf0fa58b36c7379b919ebf41f564a9200a4dc306bd645b19201128fc"
    )
    assert hashlib.sha256(issued.result.checkpoint_payload).hexdigest() == (
        "b92488cb797b35c59333febef94d9022ca8c7ad2a42a4c9deae6b23f952cab9b"
    )
    assert hashlib.sha256(issued.result.head_payload).hexdigest() == (
        "63f0d9435dc4a9430f26830cf5a9ef0f7f0fadbe1fbfe08f1172c61161fcdf06"
    )


def test_request_roundtrip_and_domain_separated_id_preimage() -> None:
    payload = _request(_material())
    request = logical_time.load_logical_time_request_v1(payload)
    assert request.canonical_bytes() == payload
    assert logical_time.phase2b_step5_logical_time_request_id_preimage_v1(payload) == (
        _REQUEST_ID_DOMAIN + _canonical(request._id_subject())
    )
    assert request.request_id == (
        "phase2b_step5_logical_time_request_v1_"
        "89ae67d6e10506a976a9587a6eb9bd198d45949217758085fc7b21bdc8c14403"
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: bytearray(payload),
        lambda payload: payload + b"\n",
        lambda payload: b" " + payload,
        lambda payload: payload.replace(b"{", b"{ ", 1),
        lambda payload: payload.replace(
            b'"attempt_context_id":', b'"attempt_context_id":"duplicate","attempt_context_id":', 1
        ),
    ],
)
def test_request_rejects_nonexact_or_noncanonical_transport(mutator) -> None:
    payload = _request(_material())
    with pytest.raises(logical_time.Phase2BStep5LogicalTimeErrorV1):
        logical_time.load_logical_time_request_v1(mutator(payload))


@pytest.mark.parametrize("field", ["purpose", "bound_artifact_id", "request_id"])
def test_request_rejects_missing_fields(field: str) -> None:
    body = _json(_request(_material()))
    del body[field]
    with _expect_code(logical_time.FAIL_SCHEMA_V1):
        logical_time.load_logical_time_request_v1(_canonical(body))


def test_request_rejects_unknown_field() -> None:
    body = _json(_request(_material()))
    body["trusted_utc"] = "2026-08-22T00:00:00Z"
    with _expect_code(logical_time.FAIL_SCHEMA_V1):
        logical_time.load_logical_time_request_v1(_canonical(body))


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("next_sequence", True, logical_time.FAIL_SCHEMA_V1),
        ("next_sequence", 1.5, logical_time.FAIL_SCHEMA_V1),
        ("next_sequence", -1, logical_time.FAIL_SCHEMA_V1),
        ("next_sequence", 1 << 53, logical_time.FAIL_SCHEMA_V1),
        ("logical_time_key_epoch", True, logical_time.FAIL_SCHEMA_V1),
        ("logical_time_key_epoch", 2, logical_time.FAIL_BINDING_V1),
        ("signature_purpose", "SIGN_CAS_IF_ABSENT_RECEIPT", logical_time.FAIL_BINDING_V1),
        ("purpose", "lowercase", logical_time.FAIL_SCHEMA_V1),
        ("request_nonce", "A" * 64, logical_time.FAIL_SCHEMA_V1),
    ],
)
def test_request_rejects_type_and_policy_confusion(
    field: str, value: object, expected_code: str
) -> None:
    body = _json(_request(_material()))
    body[field] = value
    payload = _readdress_request(body)
    with _expect_code(expected_code):
        logical_time.load_logical_time_request_v1(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authority_profile_id", logical_time.AUTHORITY_PROFILE_PREFIX_V2 + "1" * 64),
        ("signer_registry_id", logical_time.SIGNER_REGISTRY_PREFIX_V2 + "2" * 64),
        ("logical_time_key_id", logical_time.ROLE_KEY_PREFIX_V1 + "3" * 64),
        ("attempt_context_id", "other_attempt_" + "4" * 64),
        ("prior_token_id", logical_time.TOKEN_ID_PREFIX_V1 + "5" * 64),
        ("bound_artifact_sha256", "sha256:" + "6" * 64),
    ],
)
def test_request_content_id_commits_every_binding(field: str, value: object) -> None:
    payload = _request(_material())
    body = _json(payload)
    body[field] = value
    with _expect_code(logical_time.FAIL_CONTENT_ID_V1):
        logical_time.load_logical_time_request_v1(_canonical(body))


def test_request_exact_dto_rejects_subclass() -> None:
    class RequestSubclass(logical_time.LogicalTimeRequestV1):
        pass

    body = _json(_request(_material()))
    with _expect_code(logical_time.FAIL_INPUT_V1):
        RequestSubclass(**body)  # type: ignore[arg-type]


def test_real_signature_request_projection_and_transition_verify(issued: Issued) -> None:
    token = _verify_token(
        issued.material, issued.request_payload, issued.result.signed_token_payload
    )
    head = _verify_transition(
        issued.material,
        issued.previous_head_payload,
        issued.result.checkpoint_payload,
        issued.result.head_payload,
    )
    assert token.sequence == 0
    assert token.prior_token_id == logical_time.GENESIS_TOKEN_ID_V1
    assert head.checkpoint_count == head.next_sequence == 1
    assert head.last_token_id == token.token_id


def test_signature_and_content_id_preimages_are_distinct(issued: Issued) -> None:
    signature_preimage = (
        logical_time.phase2b_step5_logical_time_token_signature_preimage_v1(
            issued.result.signed_token_payload
        )
    )
    token_id_preimage = logical_time.phase2b_step5_logical_time_token_id_preimage_v1(
        issued.result.signed_token_payload
    )
    head_id_preimage = logical_time.phase2b_step5_logical_time_head_id_preimage_v1(
        issued.result.head_payload
    )
    checkpoint_id_preimage = (
        logical_time.phase2b_step5_logical_time_checkpoint_id_preimage_v1(
            issued.result.checkpoint_payload
        )
    )
    assert signature_preimage.startswith(
        b"HEGEL/PHASE2B/STEP5/SIGN/LOGICAL_TIME_TOKEN/V1\x00"
    )
    assert token_id_preimage.startswith(_TOKEN_ID_DOMAIN)
    assert head_id_preimage.startswith(_HEAD_ID_DOMAIN)
    assert checkpoint_id_preimage.startswith(_CHECKPOINT_ID_DOMAIN)
    assert len({signature_preimage, token_id_preimage, head_id_preimage, checkpoint_id_preimage}) == 4


def test_structural_token_loader_is_not_signature_verifier(issued: Issued) -> None:
    body = _json(issued.result.signed_token_payload)
    signature = str(body["signature_hex"])
    body["signature_hex"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    forged_payload = _readdress_token(body)
    loaded = logical_time.load_signed_logical_time_token_v1(forged_payload)
    assert loaded.signature_hex == body["signature_hex"]
    with _expect_code(logical_time.FAIL_SIGNATURE_V1):
        _verify_token(issued.material, issued.request_payload, forged_payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("purpose", "R02_BRANCH_OBSERVED"),
        ("bound_artifact_id", "artifact_" + "1" * 64),
        ("bound_artifact_sha256", "sha256:" + "2" * 64),
        ("request_nonce", "3" * 64),
        ("prior_token_id", logical_time.TOKEN_ID_PREFIX_V1 + "4" * 64),
        ("sequence", 1),
    ],
)
def test_signature_cannot_be_reused_after_token_body_mutation(
    issued: Issued, field: str, value: object
) -> None:
    body = _json(issued.result.signed_token_payload)
    body[field] = value
    request_body = {
        "schema_version": logical_time.LOGICAL_TIME_REQUEST_SCHEMA_VERSION,
        "authority_profile_id": body["authority_profile_id"],
        "signer_registry_id": body["signer_registry_id"],
        "logical_time_key_id": body["logical_time_key_id"],
        "logical_time_key_epoch": body["logical_time_key_epoch"],
        "attempt_context_id": body["attempt_context_id"],
        "request_nonce": body["request_nonce"],
        "signature_purpose": body["signature_purpose"],
        "purpose": body["purpose"],
        "bound_artifact_id": body["bound_artifact_id"],
        "bound_artifact_sha256": body["bound_artifact_sha256"],
        "prior_token_id": body["prior_token_id"],
        "next_sequence": body["sequence"],
    }
    request_payload = _readdress_request(request_body)
    body["request_id"] = logical_time.load_logical_time_request_v1(
        request_payload
    ).request_id
    forged_payload = _readdress_token(body)
    with _expect_code(logical_time.FAIL_SIGNATURE_V1):
        logical_time.verify_signed_logical_time_token_v1(
            forged_payload,
            request_payload=request_payload,
            logical_time_public_key=issued.material.public_key,
            expected_authority_profile_id=issued.material.authority_profile_id,
            expected_signer_registry_id=issued.material.signer_registry_id,
            expected_logical_time_key_id=issued.material.key_id,
            expected_logical_time_key_epoch=1,
            expected_attempt_context_id=issued.material.attempt_context_id,
        )


def test_token_verifier_rejects_different_request(issued: Issued) -> None:
    other_request = _request(issued.material, nonce="1" * 64)
    with _expect_code(logical_time.FAIL_BINDING_V1):
        _verify_token(
            issued.material, other_request, issued.result.signed_token_payload
        )


def test_token_verifier_rejects_wrong_public_key_lineage(issued: Issued) -> None:
    wrong_public = _raw_public(Ed25519PrivateKey.from_private_bytes(b"\x99" * 32))
    with _expect_code(logical_time.FAIL_BINDING_V1):
        logical_time.verify_signed_logical_time_token_v1(
            issued.result.signed_token_payload,
            request_payload=issued.request_payload,
            logical_time_public_key=wrong_public,
            expected_authority_profile_id=issued.material.authority_profile_id,
            expected_signer_registry_id=issued.material.signer_registry_id,
            expected_logical_time_key_id=issued.material.key_id,
            expected_logical_time_key_epoch=1,
            expected_attempt_context_id=issued.material.attempt_context_id,
        )


@pytest.mark.parametrize("private_key", [bytes(range(32)), object(), lambda _: b"x" * 64])
def test_service_rejects_seed_bytes_callbacks_and_protocol_objects(
    tmp_path: Path, private_key: object
) -> None:
    material = _material()
    with _ledger(tmp_path / "ledger", material, create=True) as ledger:
        head = ledger.current_head_bytes()
        with _expect_code(logical_time.FAIL_INPUT_V1):
            ledger.issue_service_side_v1(
                request_payload=_request(material),
                expected_previous_head_payload=head,
                private_key=private_key,
            )


def test_service_rejects_wrong_real_private_key(tmp_path: Path) -> None:
    material = _material()
    wrong_key = Ed25519PrivateKey.from_private_bytes(b"\x98" * 32)
    with _ledger(tmp_path / "ledger", material, create=True) as ledger:
        head = ledger.current_head_bytes()
        with _expect_code(logical_time.FAIL_BINDING_V1):
            ledger.issue_service_side_v1(
                request_payload=_request(material),
                expected_previous_head_payload=head,
                private_key=wrong_key,
            )


def test_private_key_is_not_retained_on_ledger_instance(tmp_path: Path) -> None:
    material = _material()
    with _ledger(tmp_path / "ledger", material, create=True) as ledger:
        assert "private_key" not in logical_time.LogicalTimeLedgerV1.__slots__
        assert not hasattr(ledger, "private_key")


def test_two_issuances_form_exact_contiguous_chain(tmp_path: Path) -> None:
    material = _material()
    with _ledger(tmp_path / "ledger", material, create=True) as ledger:
        h0 = ledger.current_head_bytes()
        first_request = _request(material)
        first = ledger.issue_service_side_v1(
            request_payload=first_request,
            expected_previous_head_payload=h0,
            private_key=material.private_key,
        )
        first_token = logical_time.load_signed_logical_time_token_v1(
            first.signed_token_payload
        )
        second_request = _request(
            material,
            nonce="1" * 64,
            purpose="R02_BRANCH_OBSERVED",
            artifact_tag="2",
            artifact_sha_tag="3",
            prior_token_id=first_token.token_id,
            next_sequence=1,
        )
        second = ledger.issue_service_side_v1(
            request_payload=second_request,
            expected_previous_head_payload=first.head_payload,
            private_key=material.private_key,
        )
        final_head = _verify_transition(
            material,
            first.head_payload,
            second.checkpoint_payload,
            second.head_payload,
        )
        assert final_head.checkpoint_count == final_head.next_sequence == 2
        assert sorted((tmp_path / "ledger" / "heads").iterdir()) == [
            tmp_path / "ledger" / "heads" / "00000000000000000000.json",
            tmp_path / "ledger" / "heads" / "00000000000000000001.json",
            tmp_path / "ledger" / "heads" / "00000000000000000002.json",
        ]


def test_stale_head_and_exact_request_replay_fail_closed(issued: Issued) -> None:
    with _ledger(issued.root, issued.material, create=False) as ledger:
        with _expect_code(logical_time.FAIL_LEDGER_V1):
            ledger.issue_service_side_v1(
                request_payload=issued.request_payload,
                expected_previous_head_payload=issued.previous_head_payload,
                private_key=issued.material.private_key,
            )


@pytest.mark.parametrize(
    ("prior_token_id", "next_sequence"),
    [
        (logical_time.GENESIS_TOKEN_ID_V1, 1),
        (logical_time.TOKEN_ID_PREFIX_V1 + "1" * 64, 0),
    ],
)
def test_gap_or_wrong_prior_token_fails_before_signing(
    tmp_path: Path, prior_token_id: str, next_sequence: int
) -> None:
    material = _material()
    with _ledger(tmp_path / "ledger", material, create=True) as ledger:
        with _expect_code(logical_time.FAIL_LEDGER_V1):
            ledger.issue_service_side_v1(
                request_payload=_request(
                    material,
                    prior_token_id=prior_token_id,
                    next_sequence=next_sequence,
                ),
                expected_previous_head_payload=ledger.current_head_bytes(),
                private_key=material.private_key,
            )


def test_duplicate_nonce_is_rejected_even_when_request_id_differs(tmp_path: Path) -> None:
    material = _material()
    with _ledger(tmp_path / "ledger", material, create=True) as ledger:
        first = ledger.issue_service_side_v1(
            request_payload=_request(material),
            expected_previous_head_payload=ledger.current_head_bytes(),
            private_key=material.private_key,
        )
        token = logical_time.load_signed_logical_time_token_v1(
            first.signed_token_payload
        )
        second_request = _request(
            material,
            nonce="d" * 64,
            purpose="R02_BRANCH_OBSERVED",
            artifact_tag="1",
            artifact_sha_tag="2",
            prior_token_id=token.token_id,
            next_sequence=1,
        )
        assert logical_time.load_logical_time_request_v1(second_request).request_id != (
            logical_time.load_logical_time_request_v1(_request(material)).request_id
        )
        with _expect_code(logical_time.FAIL_LEDGER_V1):
            ledger.issue_service_side_v1(
                request_payload=second_request,
                expected_previous_head_payload=first.head_payload,
                private_key=material.private_key,
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_attempt_context_id", "phase2b_step5_attempt_" + "1" * 64),
        (
            "expected_authority_profile_id",
            logical_time.AUTHORITY_PROFILE_PREFIX_V2 + "2" * 64,
        ),
        (
            "expected_signer_registry_id",
            logical_time.SIGNER_REGISTRY_PREFIX_V2 + "3" * 64,
        ),
        (
            "expected_logical_time_key_id",
            logical_time.ROLE_KEY_PREFIX_V1 + "4" * 64,
        ),
    ],
)
def test_transition_rejects_cross_graph_expected_pins(
    issued: Issued, field: str, value: object
) -> None:
    arguments: dict[str, object] = {
        "logical_time_public_key": issued.material.public_key,
        "expected_authority_profile_id": issued.material.authority_profile_id,
        "expected_signer_registry_id": issued.material.signer_registry_id,
        "expected_logical_time_key_id": issued.material.key_id,
        "expected_logical_time_key_epoch": 1,
        "expected_attempt_context_id": issued.material.attempt_context_id,
    }
    arguments[field] = value
    with _expect_code(logical_time.FAIL_BINDING_V1):
        logical_time.verify_logical_time_ledger_transition_v1(
            issued.previous_head_payload,
            issued.result.checkpoint_payload,
            issued.result.head_payload,
            **arguments,  # type: ignore[arg-type]
        )


def test_checkpoint_outer_binding_cannot_splice_signed_token(issued: Issued) -> None:
    body = _json(issued.result.checkpoint_payload)
    body["attempt_context_id"] = "phase2b_step5_attempt_" + "1" * 64
    forged_checkpoint = _readdress_checkpoint(body)
    logical_time.load_logical_time_ledger_checkpoint_v1(forged_checkpoint)
    with _expect_code(logical_time.FAIL_BINDING_V1):
        _verify_transition(
            issued.material,
            issued.previous_head_payload,
            forged_checkpoint,
            issued.result.head_payload,
        )


def test_next_head_cannot_skip_or_repoint_checkpoint(issued: Issued) -> None:
    body = _json(issued.result.head_payload)
    body["checkpoint_count"] = 2
    body["next_sequence"] = 2
    forged_head = _readdress_head(body)
    logical_time.load_logical_time_ledger_head_v1(forged_head)
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _verify_transition(
            issued.material,
            issued.previous_head_payload,
            issued.result.checkpoint_payload,
            forged_head,
        )


def test_checkpoint_token_digest_is_verified(issued: Issued) -> None:
    body = _json(issued.result.checkpoint_payload)
    body["signed_token_sha256"] = "sha256:" + "1" * 64
    forged_checkpoint = _readdress_checkpoint(body)
    with _expect_code(logical_time.FAIL_BINDING_V1):
        logical_time.load_logical_time_ledger_checkpoint_v1(forged_checkpoint)


def test_ledger_files_and_directories_have_frozen_modes(issued: Issued) -> None:
    assert stat.S_IMODE(issued.root.stat().st_mode) == 0o700
    assert stat.S_IMODE((issued.root / "heads").stat().st_mode) == 0o700
    assert stat.S_IMODE((issued.root / "checkpoints").stat().st_mode) == 0o700
    assert stat.S_IMODE((issued.root / "ledger.lock").stat().st_mode) == 0o600
    for path in (issued.root / "heads").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_nlink == 1
    for path in (issued.root / "checkpoints").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_nlink == 1


@pytest.mark.parametrize(
    "relative_path",
    ["unexpected", "heads/unexpected", "checkpoints/unexpected"],
)
def test_extra_ledger_entry_fails_closed(issued: Issued, relative_path: str) -> None:
    (issued.root / relative_path).write_bytes(b"poison")
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _ledger(issued.root, issued.material, create=False)


def test_checkpoint_without_next_head_is_crash_poison(issued: Issued) -> None:
    os.unlink(issued.root / "heads" / "00000000000000000001.json")
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _ledger(issued.root, issued.material, create=False)


def test_head_without_checkpoint_is_crash_poison(issued: Issued) -> None:
    os.unlink(issued.root / "checkpoints" / "00000000000000000000.json")
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _ledger(issued.root, issued.material, create=False)


def test_partial_final_record_is_not_repaired_or_overwritten(issued: Issued) -> None:
    checkpoint = issued.root / "checkpoints" / "00000000000000000000.json"
    checkpoint.write_bytes(issued.result.checkpoint_payload[:37])
    os.chmod(checkpoint, 0o600)
    with pytest.raises(logical_time.Phase2BStep5LogicalTimeErrorV1):
        _ledger(issued.root, issued.material, create=False)
    assert checkpoint.read_bytes() == issued.result.checkpoint_payload[:37]


def test_hardlinked_record_is_rejected(issued: Issued, tmp_path: Path) -> None:
    head = issued.root / "heads" / "00000000000000000001.json"
    os.link(head, tmp_path / "second-link")
    with _expect_code(logical_time.FAIL_FILESYSTEM_V1):
        _ledger(issued.root, issued.material, create=False)


def test_complete_local_pair_truncation_is_not_claimed_as_rollback_proof(
    issued: Issued,
) -> None:
    os.unlink(issued.root / "heads" / "00000000000000000001.json")
    os.unlink(issued.root / "checkpoints" / "00000000000000000000.json")
    with _ledger(issued.root, issued.material, create=False) as ledger:
        restored = logical_time.load_logical_time_ledger_head_v1(
            ledger.current_head_bytes()
        )
    assert restored.checkpoint_count == 0
    assert (
        "EXTERNAL_DURABLE_CAS_CHECKPOINT_OF_COMPLETE_LEDGER_VERIFIED"
        in logical_time.REQUIRED_EXTERNAL_AUTHORITY_CHECKS_V1
    )
    assert "NOT_CAS_CHECKPOINT" in logical_time.AUTHORITY_CLAIM_BOUNDARY_V1


def test_cloned_local_ledger_can_fork_and_is_not_external_notarization(
    issued: Issued, tmp_path: Path
) -> None:
    clone = tmp_path / "clone"
    shutil.copytree(issued.root, clone)
    token = logical_time.load_signed_logical_time_token_v1(
        issued.result.signed_token_payload
    )
    with _ledger(issued.root, issued.material, create=False) as original:
        original_request = _request(
            issued.material,
            nonce="1" * 64,
            purpose="ORIGINAL_BRANCH",
            artifact_tag="2",
            artifact_sha_tag="3",
            prior_token_id=token.token_id,
            next_sequence=1,
        )
        original_result = original.issue_service_side_v1(
            request_payload=original_request,
            expected_previous_head_payload=issued.result.head_payload,
            private_key=issued.material.private_key,
        )
    with _ledger(clone, issued.material, create=False) as fork:
        fork_request = _request(
            issued.material,
            nonce="4" * 64,
            purpose="FORK_BRANCH",
            artifact_tag="5",
            artifact_sha_tag="6",
            prior_token_id=token.token_id,
            next_sequence=1,
        )
        fork_result = fork.issue_service_side_v1(
            request_payload=fork_request,
            expected_previous_head_payload=issued.result.head_payload,
            private_key=issued.material.private_key,
        )
    assert original_result.head_id != fork_result.head_id
    assert "NOT_EXTERNAL_NOTARIZATION" in logical_time.AUTHORITY_CLAIM_BOUNDARY_V1


def test_closed_ledger_fails_closed(tmp_path: Path) -> None:
    material = _material()
    ledger = _ledger(tmp_path / "ledger", material, create=True)
    ledger.close()
    ledger.close()
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        ledger.current_head_bytes()


def test_ledger_rejects_relative_and_lexically_noncanonical_paths(tmp_path: Path) -> None:
    material = _material()
    with _expect_code(logical_time.FAIL_INPUT_V1):
        logical_time.LogicalTimeLedgerV1(
            root_directory="relative/ledger",
            authority_profile_id=material.authority_profile_id,
            signer_registry_id=material.signer_registry_id,
            logical_time_key_id=material.key_id,
            logical_time_key_epoch=1,
            attempt_context_id=material.attempt_context_id,
            logical_time_public_key=material.public_key,
            create=True,
        )
    noncanonical = str(tmp_path / "unused" / ".." / "ledger")
    with _expect_code(logical_time.FAIL_INPUT_V1):
        logical_time.LogicalTimeLedgerV1(
            root_directory=noncanonical,
            authority_profile_id=material.authority_profile_id,
            signer_registry_id=material.signer_registry_id,
            logical_time_key_id=material.key_id,
            logical_time_key_epoch=1,
            attempt_context_id=material.attempt_context_id,
            logical_time_public_key=material.public_key,
            create=True,
        )


def test_ledger_rejects_symlink_root(issued: Issued, tmp_path: Path) -> None:
    alias = tmp_path / "ledger-alias"
    alias.symlink_to(issued.root, target_is_directory=True)
    with _expect_code(logical_time.FAIL_FILESYSTEM_V1):
        _ledger(alias, issued.material, create=False)


def test_create_does_not_silently_adopt_preexisting_empty_directory(
    tmp_path: Path,
) -> None:
    material = _material()
    root = tmp_path / "ledger"
    root.mkdir(mode=0o700)
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _ledger(root, material, create=True)


def test_service_result_booleans_are_exact_and_nonauthoritative(issued: Issued) -> None:
    assert issued.result.mechanics_only is True
    assert issued.result.authority_evidence is False
    body = {
        field: getattr(issued.result, field)
        for field in (
            "request_id",
            "token_id",
            "checkpoint_id",
            "head_id",
            "signed_token_payload",
            "checkpoint_payload",
            "head_payload",
        )
    }
    with _expect_code(logical_time.FAIL_BINDING_V1):
        logical_time.LogicalTimeServiceIssuanceV1(
            **body,
            mechanics_only=1,  # type: ignore[arg-type]
            authority_evidence=False,
        )
    with _expect_code(logical_time.FAIL_BINDING_V1):
        logical_time.LogicalTimeServiceIssuanceV1(
            **body,
            mechanics_only=True,
            authority_evidence=0,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ReplayChain:
    material: Material
    genesis_head_payload: bytes
    checkpoint_payloads: tuple[bytes, ...]
    subsequent_head_payloads: tuple[bytes, ...]
    expected_event_plan: tuple[logical_time.ExpectedLogicalEventV1, ...]


def _build_replay_chain(
    root: Path,
    material: Material | None = None,
    purposes: tuple[str, str, str] = (
        "R01_STATUS_OBSERVED",
        "R02_BRANCH_OBSERVED",
        "R03_TERMINAL_OBSERVED",
    ),
) -> ReplayChain:
    selected = material if material is not None else _material()
    checkpoints: list[bytes] = []
    heads: list[bytes] = []
    events: list[logical_time.ExpectedLogicalEventV1] = []
    with _ledger(root, selected, create=True) as ledger:
        genesis = ledger.current_head_bytes()
        prior_head = genesis
        prior_token_id = logical_time.GENESIS_TOKEN_ID_V1
        for index, purpose in enumerate(purposes):
            nonce_tag = format(index + 1, "x")
            artifact_tag = format(index + 4, "x")
            artifact_sha_tag = format(index + 8, "x")
            request_payload = _request(
                selected,
                nonce=nonce_tag * 64,
                purpose=purpose,
                artifact_tag=artifact_tag,
                artifact_sha_tag=artifact_sha_tag,
                prior_token_id=prior_token_id,
                next_sequence=index,
            )
            issuance = ledger.issue_service_side_v1(
                request_payload=request_payload,
                expected_previous_head_payload=prior_head,
                private_key=selected.private_key,
            )
            token = logical_time.load_signed_logical_time_token_v1(
                issuance.signed_token_payload
            )
            checkpoints.append(issuance.checkpoint_payload)
            heads.append(issuance.head_payload)
            events.append(
                logical_time.ExpectedLogicalEventV1(
                    purpose=purpose,
                    bound_artifact_id=f"artifact_{artifact_tag * 64}",
                    bound_artifact_sha256=f"sha256:{artifact_sha_tag * 64}",
                    request_nonce=nonce_tag * 64,
                )
            )
            prior_head = issuance.head_payload
            prior_token_id = token.token_id
    return ReplayChain(
        material=selected,
        genesis_head_payload=genesis,
        checkpoint_payloads=tuple(checkpoints),
        subsequent_head_payloads=tuple(heads),
        expected_event_plan=tuple(events),
    )


@pytest.fixture
def replay_chain(tmp_path: Path) -> ReplayChain:
    return _build_replay_chain(tmp_path / "replay-ledger")


def _replay(
    chain: ReplayChain,
    *,
    genesis_head_payload: object | None = None,
    checkpoint_payloads: object | None = None,
    subsequent_head_payloads: object | None = None,
    expected_event_plan: object | None = None,
    **pin_overrides: object,
) -> logical_time.VerifiedLogicalTimeChainV1:
    arguments: dict[str, object] = {
        "logical_time_public_key": chain.material.public_key,
        "expected_authority_profile_id": chain.material.authority_profile_id,
        "expected_signer_registry_id": chain.material.signer_registry_id,
        "expected_logical_time_key_id": chain.material.key_id,
        "expected_logical_time_key_epoch": 1,
        "expected_attempt_context_id": chain.material.attempt_context_id,
        "expected_event_plan": (
            chain.expected_event_plan
            if expected_event_plan is None
            else expected_event_plan
        ),
    }
    arguments.update(pin_overrides)
    return logical_time.verify_logical_time_ledger_replay_v1(
        (
            chain.genesis_head_payload
            if genesis_head_payload is None
            else genesis_head_payload
        ),
        (
            chain.checkpoint_payloads
            if checkpoint_payloads is None
            else checkpoint_payloads
        ),
        (
            chain.subsequent_head_payloads
            if subsequent_head_payloads is None
            else subsequent_head_payloads
        ),
        **arguments,  # type: ignore[arg-type]
    )


def _resign_token_mapping(
    body: dict[str, object], private_key: Ed25519PrivateKey
) -> bytes:
    unsigned = {
        key: value
        for key, value in body.items()
        if key not in {"signature_hex", "token_id"}
    }
    wrapper = {
        "schema_version": "hegel-machine-phase2b-step5-signature-preimage/1",
        "artifact_kind": "logical_time",
        "role": logical_time.LOGICAL_TIME_ROLE_V1,
        "purpose": logical_time.SIGN_LOGICAL_TIME_TOKEN_PURPOSE_V1,
        "key_epoch": body["logical_time_key_epoch"],
        "signer_key_id": body["logical_time_key_id"],
        "body": unsigned,
    }
    body["signature_hex"] = private_key.sign(
        _TOKEN_SIGNATURE_DOMAIN + _canonical(wrapper)
    ).hex()
    return _readdress_token(body)


def _readdress_transition_around_token(
    checkpoint_payload: bytes,
    head_payload: bytes,
    token_payload: bytes,
) -> tuple[bytes, bytes]:
    token = _json(token_payload)
    checkpoint = _json(checkpoint_payload)
    checkpoint["request_id"] = token["request_id"]
    checkpoint["request_nonce"] = token["request_nonce"]
    checkpoint["token_id"] = token["token_id"]
    checkpoint["sequence"] = token["sequence"]
    checkpoint["checkpoint_index"] = token["sequence"]
    checkpoint["signed_token"] = token
    checkpoint["signed_token_sha256"] = (
        "sha256:" + hashlib.sha256(token_payload).hexdigest()
    )
    readdressed_checkpoint = _readdress_checkpoint(checkpoint)
    checkpoint = _json(readdressed_checkpoint)
    head = _json(head_payload)
    head["last_token_id"] = token["token_id"]
    head["last_checkpoint_id"] = checkpoint["checkpoint_id"]
    readdressed_head = _readdress_head(head)
    return readdressed_checkpoint, readdressed_head


def _invalid_signature_transition(chain: ReplayChain) -> tuple[bytes, bytes]:
    checkpoint = _json(chain.checkpoint_payloads[0])
    token = checkpoint["signed_token"]
    assert type(token) is dict
    signature = token["signature_hex"]
    assert type(signature) is str
    token["signature_hex"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    forged_token_payload = _readdress_token(token)
    return _readdress_transition_around_token(
        chain.checkpoint_payloads[0],
        chain.subsequent_head_payloads[0],
        forged_token_payload,
    )


def test_valid_v2_whole_chain_replay_exact_positive_and_nonauthority(
    replay_chain: ReplayChain,
) -> None:
    result = _replay(replay_chain)
    assert type(result) is logical_time.VerifiedLogicalTimeChainV1
    assert logical_time.require_verified_logical_time_chain_v1(result) is result
    assert result.verified_event_count == 3
    assert result.verified_signature_count == 3
    assert result.authority_profile_id.startswith(
        logical_time.AUTHORITY_PROFILE_PREFIX_V2
    )
    assert result.signer_registry_id.startswith(
        logical_time.SIGNER_REGISTRY_PREFIX_V2
    )
    assert result.verified_event_plan == tuple(
        event.as_tuple() for event in replay_chain.expected_event_plan
    )
    assert len(result.request_ids) == len(result.token_ids) == 3
    assert len(result.checkpoint_ids) == 3
    assert len(result.head_ids) == 4
    assert result.final_head_id == result.head_ids[-1]
    assert result.final_head_payload_sha256 == (
        "sha256:"
        + hashlib.sha256(replay_chain.subsequent_head_payloads[-1]).hexdigest()
    )
    assert result.zero_event_genesis_only is False
    assert result.mechanics_only is True
    assert result.authority_evidence is False
    assert result.external_anti_fork_verified is False
    assert result.event_authorization_verified is False
    assert result.attempt_context_freshness_verified is False
    assert result.service_image_verified is False
    assert result.service_uid_verified is False
    assert result.control_domain_verified is False
    assert result.key_custody_verified is False
    assert result.external_cas_checkpoint_verified is False
    assert result.trusted_utc_verified is False


def test_zero_event_replay_accepts_only_exact_genesis(tmp_path: Path) -> None:
    material = _material(attempt_tag="0")
    root = tmp_path / "zero-ledger"
    with _ledger(root, material, create=True) as ledger:
        genesis = ledger.current_head_bytes()
    empty = ReplayChain(
        material=material,
        genesis_head_payload=genesis,
        checkpoint_payloads=(),
        subsequent_head_payloads=(),
        expected_event_plan=(),
    )
    result = _replay(empty)
    assert logical_time.require_verified_logical_time_chain_v1(result) is result
    assert result.verified_event_count == result.verified_signature_count == 0
    assert result.verified_event_plan == ()
    assert result.request_ids == result.token_ids == result.checkpoint_ids == ()
    assert result.head_ids == (
        logical_time.load_logical_time_ledger_head_v1(genesis).head_id,
    )
    assert result.zero_event_genesis_only is True
    assert result.mechanics_only is True
    assert result.authority_evidence is False
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _replay(
            empty,
            checkpoint_payloads=(b"{}",),
            subsequent_head_payloads=(genesis,),
        )
    non_genesis = _build_replay_chain(tmp_path / "non-genesis", material)
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _replay(
            empty,
            genesis_head_payload=non_genesis.subsequent_head_payloads[0],
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("checkpoint_payloads", []),
        ("subsequent_head_payloads", []),
        ("expected_event_plan", []),
        ("checkpoint_payloads", (bytearray(b"{}"),) * 3),
        ("subsequent_head_payloads", (bytearray(b"{}"),) * 3),
    ],
)
def test_replay_requires_exact_tuple_and_bytes_transport(
    replay_chain: ReplayChain, field: str, replacement: object
) -> None:
    with _expect_code(logical_time.FAIL_INPUT_V1):
        _replay(replay_chain, **{field: replacement})


def test_replay_rejects_expected_event_subclass_even_with_copied_slots(
    replay_chain: ReplayChain,
) -> None:
    class EventSubclass(logical_time.ExpectedLogicalEventV1):
        pass

    forged = object.__new__(EventSubclass)
    source = replay_chain.expected_event_plan[0]
    for name in (
        "purpose",
        "bound_artifact_id",
        "bound_artifact_sha256",
        "request_nonce",
    ):
        object.__setattr__(forged, name, getattr(source, name))
    plan = (forged, *replay_chain.expected_event_plan[1:])
    with _expect_code(logical_time.FAIL_INPUT_V1):
        _replay(replay_chain, expected_event_plan=plan)


class _EqAll:
    def __eq__(self, other: object) -> bool:
        return True


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("expected_authority_profile_id", _EqAll()),
        ("expected_signer_registry_id", _EqAll()),
        ("expected_logical_time_key_id", _EqAll()),
        ("expected_attempt_context_id", _EqAll()),
        ("expected_logical_time_key_epoch", True),
        ("expected_logical_time_key_epoch", "1"),
        ("logical_time_public_key", bytearray(range(32))),
    ],
)
def test_replay_rejects_eqall_bool_and_bytes_subclass_confusion(
    replay_chain: ReplayChain, field: str, replacement: object
) -> None:
    with pytest.raises(logical_time.Phase2BStep5LogicalTimeErrorV1):
        _replay(replay_chain, **{field: replacement})


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "expected_authority_profile_id",
            logical_time.AUTHORITY_PROFILE_PREFIX_V2 + "1" * 64,
        ),
        (
            "expected_signer_registry_id",
            logical_time.SIGNER_REGISTRY_PREFIX_V2 + "2" * 64,
        ),
        (
            "expected_logical_time_key_id",
            logical_time.ROLE_KEY_PREFIX_V1 + "3" * 64,
        ),
        (
            "expected_attempt_context_id",
            "phase2b_step5_attempt_" + "4" * 64,
        ),
        ("expected_logical_time_key_epoch", 2),
    ],
)
def test_replay_rejects_cross_graph_attempt_and_key_pins(
    replay_chain: ReplayChain, field: str, replacement: object
) -> None:
    with pytest.raises(logical_time.Phase2BStep5LogicalTimeErrorV1):
        _replay(replay_chain, **{field: replacement})


def test_replay_rejects_complete_foreign_v2_profile_registry_graph(
    replay_chain: ReplayChain,
) -> None:
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _replay(
            replay_chain,
            expected_authority_profile_id=(
                logical_time.AUTHORITY_PROFILE_PREFIX_V2 + "c" * 64
            ),
            expected_signer_registry_id=(
                logical_time.SIGNER_REGISTRY_PREFIX_V2 + "d" * 64
            ),
        )


def test_replay_rejects_wrong_public_key_lineage(replay_chain: ReplayChain) -> None:
    wrong_public = _raw_public(Ed25519PrivateKey.from_private_bytes(b"\x97" * 32))
    with _expect_code(logical_time.FAIL_BINDING_V1):
        _replay(replay_chain, logical_time_public_key=wrong_public)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("purpose", "WRONG_EVENT_PURPOSE"),
        ("bound_artifact_id", "artifact_" + "a" * 64),
        ("bound_artifact_sha256", "sha256:" + "b" * 64),
        ("request_nonce", "c" * 64),
    ],
)
def test_replay_exact_event_plan_binds_every_field(
    replay_chain: ReplayChain, field: str, replacement: str
) -> None:
    source = replay_chain.expected_event_plan[0]
    values = {
        "purpose": source.purpose,
        "bound_artifact_id": source.bound_artifact_id,
        "bound_artifact_sha256": source.bound_artifact_sha256,
        "request_nonce": source.request_nonce,
    }
    values[field] = replacement
    mutated = logical_time.ExpectedLogicalEventV1(**values)
    with _expect_code(logical_time.FAIL_BINDING_V1):
        _replay(
            replay_chain,
            expected_event_plan=(mutated, *replay_chain.expected_event_plan[1:]),
        )


def test_replay_rejects_event_plan_order_change(replay_chain: ReplayChain) -> None:
    plan = (
        replay_chain.expected_event_plan[1],
        replay_chain.expected_event_plan[0],
        replay_chain.expected_event_plan[2],
    )
    with _expect_code(logical_time.FAIL_BINDING_V1):
        _replay(replay_chain, expected_event_plan=plan)


def test_replay_revalidates_mutated_exact_event_dto(replay_chain: ReplayChain) -> None:
    event = replay_chain.expected_event_plan[0]
    object.__setattr__(event, "purpose", _StringSubclass(event.purpose))
    with _expect_code(logical_time.FAIL_SCHEMA_V1):
        _replay(replay_chain)


@pytest.mark.parametrize("duplicate_kind", ["row", "nonce", "event_binding"])
def test_replay_rejects_duplicate_expected_events(
    replay_chain: ReplayChain, duplicate_kind: str
) -> None:
    first = replay_chain.expected_event_plan[0]
    if duplicate_kind == "row":
        duplicate = first
    elif duplicate_kind == "nonce":
        duplicate = logical_time.ExpectedLogicalEventV1(
            purpose="DISTINCT_PURPOSE",
            bound_artifact_id="artifact_" + "a" * 64,
            bound_artifact_sha256="sha256:" + "b" * 64,
            request_nonce=first.request_nonce,
        )
    else:
        duplicate = logical_time.ExpectedLogicalEventV1(
            purpose=first.purpose,
            bound_artifact_id=first.bound_artifact_id,
            bound_artifact_sha256=first.bound_artifact_sha256,
            request_nonce="f" * 64,
        )
    plan = (first, duplicate, replay_chain.expected_event_plan[2])
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _replay(replay_chain, expected_event_plan=plan)


@pytest.mark.parametrize(
    ("checkpoint_slice", "head_slice", "plan_slice"),
    [
        (slice(None, -1), slice(None, -1), slice(None)),
        (slice(None), slice(None, -1), slice(None)),
        (slice(None), slice(None), slice(None, -1)),
        (slice(1, None), slice(1, None), slice(1, None)),
    ],
)
def test_replay_rejects_missing_extra_or_gap_cardinality(
    replay_chain: ReplayChain,
    checkpoint_slice: slice,
    head_slice: slice,
    plan_slice: slice,
) -> None:
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _replay(
            replay_chain,
            checkpoint_payloads=replay_chain.checkpoint_payloads[checkpoint_slice],
            subsequent_head_payloads=replay_chain.subsequent_head_payloads[head_slice],
            expected_event_plan=replay_chain.expected_event_plan[plan_slice],
        )


def test_replay_rejects_swapped_stale_and_fork_splices(
    replay_chain: ReplayChain,
) -> None:
    swapped_checkpoints = (
        replay_chain.checkpoint_payloads[1],
        replay_chain.checkpoint_payloads[0],
        replay_chain.checkpoint_payloads[2],
    )
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _replay(replay_chain, checkpoint_payloads=swapped_checkpoints)

    stale_checkpoints = (
        replay_chain.checkpoint_payloads[0],
        replay_chain.checkpoint_payloads[0],
        replay_chain.checkpoint_payloads[2],
    )
    stale_heads = (
        replay_chain.subsequent_head_payloads[0],
        replay_chain.subsequent_head_payloads[0],
        replay_chain.subsequent_head_payloads[2],
    )
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _replay(
            replay_chain,
            checkpoint_payloads=stale_checkpoints,
            subsequent_head_payloads=stale_heads,
        )

    fork_heads = (
        replay_chain.subsequent_head_payloads[0],
        replay_chain.subsequent_head_payloads[2],
        replay_chain.subsequent_head_payloads[2],
    )
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _replay(replay_chain, subsequent_head_payloads=fork_heads)


def test_replay_rejects_fully_readdressed_final_count_tamper(
    replay_chain: ReplayChain,
) -> None:
    head = _json(replay_chain.subsequent_head_payloads[-1])
    head["checkpoint_count"] = 4
    head["next_sequence"] = 4
    tampered = _readdress_head(head)
    heads = (*replay_chain.subsequent_head_payloads[:-1], tampered)
    with _expect_code(logical_time.FAIL_LEDGER_V1):
        _replay(replay_chain, subsequent_head_payloads=heads)


def test_consistent_local_forks_remain_mechanics_not_external_antifork(
    tmp_path: Path,
) -> None:
    material = _material(attempt_tag="f")
    first = _build_replay_chain(tmp_path / "branch-a", material)
    second = _build_replay_chain(
        tmp_path / "branch-b",
        material,
        (
            "R01_STATUS_OBSERVED",
            "R02_BRANCH_OBSERVED",
            "R03_ALTERNATE_OBSERVED",
        ),
    )
    first_result = _replay(first)
    second_result = _replay(second)
    assert first_result.head_ids[:-1] == second_result.head_ids[:-1]
    assert first_result.final_head_id != second_result.final_head_id
    assert first_result.external_anti_fork_verified is False
    assert second_result.external_anti_fork_verified is False
    assert first_result.authority_evidence is False
    assert second_result.authority_evidence is False


def test_replay_rejects_fully_readdressed_invalid_signature(
    replay_chain: ReplayChain,
) -> None:
    checkpoint, head = _invalid_signature_transition(replay_chain)
    checkpoints = (checkpoint, *replay_chain.checkpoint_payloads[1:])
    heads = (head, *replay_chain.subsequent_head_payloads[1:])
    with _expect_code(logical_time.FAIL_SIGNATURE_V1):
        _replay(
            replay_chain,
            checkpoint_payloads=checkpoints,
            subsequent_head_payloads=heads,
        )


def test_replay_rejects_validly_resigned_noncontent_request_id(
    replay_chain: ReplayChain,
) -> None:
    checkpoint = _json(replay_chain.checkpoint_payloads[0])
    token = checkpoint["signed_token"]
    assert type(token) is dict
    token["request_id"] = logical_time.REQUEST_ID_PREFIX_V1 + "9" * 64
    forged_token_payload = _resign_token_mapping(
        token, replay_chain.material.private_key
    )
    forged_checkpoint, forged_head = _readdress_transition_around_token(
        replay_chain.checkpoint_payloads[0],
        replay_chain.subsequent_head_payloads[0],
        forged_token_payload,
    )
    with _expect_code(logical_time.FAIL_CONTENT_ID_V1):
        _replay(
            replay_chain,
            checkpoint_payloads=(
                forged_checkpoint,
                *replay_chain.checkpoint_payloads[1:],
            ),
            subsequent_head_payloads=(
                forged_head,
                *replay_chain.subsequent_head_payloads[1:],
            ),
        )


def test_verified_result_has_no_trivial_public_issuance_token(
    replay_chain: ReplayChain,
) -> None:
    assert not hasattr(logical_time, "_VERIFIED_CHAIN_ISSUANCE_TOKEN")
    assert not hasattr(logical_time, "_hidden_issue_verified_logical_time_chain_v1")
    with pytest.raises(TypeError, match="privately issued"):
        logical_time.VerifiedLogicalTimeChainV1()
    with pytest.raises(TypeError, match="privately issued"):
        logical_time.VerifiedLogicalTimeChainV1(object())

    valid = _replay(replay_chain)
    assert logical_time.require_verified_logical_time_chain_v1(valid) is valid
    forged = object.__new__(logical_time.VerifiedLogicalTimeChainV1)
    for field_name in (
        "authority_profile_id",
        "signer_registry_id",
        "logical_time_key_id",
        "logical_time_key_epoch",
        "attempt_context_id",
        "ledger_id",
        "verified_event_count",
        "verified_signature_count",
        "verified_event_plan",
        "request_ids",
        "token_ids",
        "checkpoint_ids",
        "head_ids",
        "final_head_id",
        "final_head_payload_sha256",
        "zero_event_genesis_only",
        "mechanics_only",
        "authority_evidence",
        "external_anti_fork_verified",
        "event_authorization_verified",
        "attempt_context_freshness_verified",
        "service_image_verified",
        "service_uid_verified",
        "control_domain_verified",
        "key_custody_verified",
        "external_cas_checkpoint_verified",
        "trusted_utc_verified",
    ):
        object.__setattr__(forged, field_name, getattr(valid, field_name))
    with _expect_code(logical_time.FAIL_INPUT_V1):
        logical_time.require_verified_logical_time_chain_v1(forged)


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("verified_event_count", True),
        ("verified_signature_count", "3"),
        ("verified_event_plan", []),
        ("zero_event_genesis_only", 0),
        ("mechanics_only", 1),
        ("authority_evidence", True),
        ("external_anti_fork_verified", True),
        ("event_authorization_verified", True),
        ("attempt_context_freshness_verified", True),
        ("service_image_verified", True),
        ("service_uid_verified", True),
        ("control_domain_verified", True),
        ("key_custody_verified", True),
        ("external_cas_checkpoint_verified", True),
        ("trusted_utc_verified", True),
    ],
)
def test_verified_result_require_rejects_type_confusion_and_overclaim(
    replay_chain: ReplayChain, field_name: str, forged_value: object
) -> None:
    result = _replay(replay_chain)
    object.__setattr__(result, field_name, forged_value)
    with pytest.raises(logical_time.Phase2BStep5LogicalTimeErrorV1):
        logical_time.require_verified_logical_time_chain_v1(result)


def _rebound_helper_bomb(*args: object, **kwargs: object) -> object:
    raise AssertionError("rebound module helper must not enter captured replay")


@pytest.mark.parametrize(
    "helper_name",
    [
        "_verify_token_signature",
        "_request_from_token",
        "_require_public_key_binding",
        "_validate_common_bindings",
        "_common_tuple",
        "_verify_transition",
        "_ledger_id",
        "_head",
        "_token_from_mapping",
        "_expected_logical_event_tuple_v1",
        "_decode",
        "_exact_object",
        "_sha256_bytes",
    ],
)
def test_replay_isolated_from_each_rebound_module_helper(
    replay_chain: ReplayChain,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    monkeypatch.setattr(logical_time, helper_name, _rebound_helper_bomb)
    result = _replay(replay_chain)
    assert result.verified_event_count == result.verified_signature_count == 3
    assert result.mechanics_only is True
    assert result.authority_evidence is False
    assert logical_time.require_verified_logical_time_chain_v1(result) is result


@pytest.mark.parametrize("parser_kind", ["head", "checkpoint"])
def test_replay_isolated_from_rebound_parser_classmethod(
    replay_chain: ReplayChain,
    monkeypatch: pytest.MonkeyPatch,
    parser_kind: str,
) -> None:
    target = (
        logical_time.LogicalTimeLedgerHeadV1
        if parser_kind == "head"
        else logical_time.LogicalTimeLedgerCheckpointV1
    )
    monkeypatch.setattr(
        target,
        "from_canonical_bytes",
        classmethod(_rebound_helper_bomb),
    )
    result = _replay(replay_chain)
    assert result.verified_signature_count == 3


@pytest.mark.parametrize(
    "helper_name", ["_verify_token_signature", "_verify_transition"]
)
def test_rebound_signature_or_transition_helper_cannot_accept_invalid_signature(
    replay_chain: ReplayChain,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    forged_checkpoint, forged_head = _invalid_signature_transition(replay_chain)
    monkeypatch.setattr(logical_time, helper_name, lambda *args, **kwargs: None)
    with _expect_code(logical_time.FAIL_SIGNATURE_V1):
        _replay(
            replay_chain,
            checkpoint_payloads=(
                forged_checkpoint,
                *replay_chain.checkpoint_payloads[1:],
            ),
            subsequent_head_payloads=(
                forged_head,
                *replay_chain.subsequent_head_payloads[1:],
            ),
        )


def test_rebound_request_projection_cannot_accept_resigned_fake_request_id(
    replay_chain: ReplayChain,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _json(replay_chain.checkpoint_payloads[0])
    token = checkpoint["signed_token"]
    assert type(token) is dict
    token["request_id"] = logical_time.REQUEST_ID_PREFIX_V1 + "9" * 64
    token_payload = _resign_token_mapping(token, replay_chain.material.private_key)
    forged_checkpoint, forged_head = _readdress_transition_around_token(
        replay_chain.checkpoint_payloads[0],
        replay_chain.subsequent_head_payloads[0],
        token_payload,
    )
    original_request = logical_time.load_logical_time_request_v1(
        _request(
            replay_chain.material,
            nonce="1" * 64,
            purpose="R01_STATUS_OBSERVED",
            artifact_tag="4",
            artifact_sha_tag="8",
        )
    )
    monkeypatch.setattr(
        logical_time,
        "_request_from_token",
        lambda token: original_request,
    )
    with _expect_code(logical_time.FAIL_CONTENT_ID_V1):
        _replay(
            replay_chain,
            checkpoint_payloads=(
                forged_checkpoint,
                *replay_chain.checkpoint_payloads[1:],
            ),
            subsequent_head_payloads=(
                forged_head,
                *replay_chain.subsequent_head_payloads[1:],
            ),
        )


@pytest.mark.parametrize(
    "helper_name",
    [
        "_verify_transition",
        "_validate_common_bindings",
        "_common_tuple",
        "_require_public_key_binding",
    ],
)
def test_transition_entry_isolated_from_rebound_direct_helpers(
    replay_chain: ReplayChain,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    monkeypatch.setattr(logical_time, helper_name, _rebound_helper_bomb)
    head = logical_time.verify_logical_time_ledger_transition_v1(
        replay_chain.genesis_head_payload,
        replay_chain.checkpoint_payloads[0],
        replay_chain.subsequent_head_payloads[0],
        logical_time_public_key=replay_chain.material.public_key,
        expected_authority_profile_id=replay_chain.material.authority_profile_id,
        expected_signer_registry_id=replay_chain.material.signer_registry_id,
        expected_logical_time_key_id=replay_chain.material.key_id,
        expected_logical_time_key_epoch=1,
        expected_attempt_context_id=replay_chain.material.attempt_context_id,
    )
    assert head.checkpoint_count == 1
