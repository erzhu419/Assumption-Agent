from __future__ import annotations

import ast
import copy
from dataclasses import fields, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import phase2b_formal_wire_covert_step5_store_replay_core_v2 as core
from hegel_machine import phase2b_formal_wire_covert_step5_store_v2 as store

# The service test module owns the deterministic candidate-authority graph and
# actual CasStoreV2 corpus producer.  Reuse it so the differential tests cannot
# accidentally substitute a hand-built approximation of the service's wire.
import test_phase2b_formal_wire_covert_step5_store_v2 as service_fixture
import test_phase2b_formal_wire_covert_step5_authority_v2 as authority_fixture


EXPECTED_CORE_SOURCE_SHA256 = (
    "e33bb8ae1f4c70668de3bfbdbd803a953513db94cb0a9c1ceb1082e97141de02"
)
EXPECTED_STORE_V2_SOURCE_SHA256 = (
    "6c545144dd584244d9405c92120e09c8ecb1d115c90e964a75fb3239b0b670b1"
)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return authority.canonical_step5_json_bytes_v2(value)


def _json(payload: bytes) -> dict[str, object]:
    value = json.loads(payload)
    assert type(value) is dict
    return value


def _core_verify(
    evidence: service_fixture._Evidence,
    *,
    graph: service_fixture._AuthorityGraph = service_fixture.DEFAULT_GRAPH,
    **overrides: object,
) -> core.VerifiedCasStoreCompleteSessionReplayCoreV2:
    arguments: dict[str, object] = {
        "stage_b_lock_payload": graph.stage_b_lock_payload,
        "authority_profile_payload": graph.authority_profile_payload,
        "signer_registry_payload": graph.signer_registry_payload,
        "out_of_band_expected_lock_id_bytes": (
            graph.out_of_band_expected_lock_id_bytes
        ),
        "out_of_band_expected_lock_payload_sha256_digest": (
            graph.out_of_band_expected_lock_payload_sha256_digest
        ),
        "retention_plan_payload": evidence.plan,
        "request_payloads_in_exact_order": evidence.requests,
        "object_payloads_in_exact_order": evidence.objects,
        "signed_receipt_payloads_in_exact_order": evidence.receipts,
        "checkpoint_payloads_in_exact_order": evidence.checkpoints,
        "head_payloads_in_exact_order": evidence.heads,
        "response_payloads_in_exact_order": evidence.responses,
    }
    arguments.update(overrides)
    return core.verify_cas_store_complete_session_replay_core_v2(**arguments)  # type: ignore[arg-type]


def _service_verify(
    evidence: service_fixture._Evidence,
) -> store.VerifiedCasStoreCompleteSessionV2:
    return service_fixture._verify(evidence)


def _expect_core_error(
    operation: Callable[[], object],
    *accepted_codes: str,
) -> core.CasStoreReplayCoreV2Error:
    with pytest.raises(core.CasStoreReplayCoreV2Error) as caught:
        operation()
    if accepted_codes:
        assert caught.value.code in accepted_codes
    return caught.value


@pytest.fixture(scope="module")
def corpora(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, service_fixture._Evidence]:
    root = tmp_path_factory.mktemp("step5-store-replay-core-v2")
    duplicate_payload = b"same-content-at-two-distinct-ordinals"
    return {
        "single": service_fixture._complete(
            root,
            rows=(("single-object-kind", 1_024),),
            attempt="attempt-context-core-single-0001",
            session="retention-session-core-single-0001",
            chain="phase2b-step5-core-single-chain-0001",
            payloads=(b"one-object-produced-by-final-store-v2",),
        ),
        "duplicate": service_fixture._complete(
            root,
            rows=(
                ("duplicate-first-kind", 1_024),
                ("duplicate-second-kind", 1_024),
            ),
            attempt="attempt-context-core-duplicate-0001",
            session="retention-session-core-duplicate-0001",
            chain="phase2b-step5-core-duplicate-chain-0001",
            payloads=(duplicate_payload, duplicate_payload),
        ),
        "multi": service_fixture._complete(
            root,
            rows=(
                ("multi-alpha-kind", 1_024),
                ("multi-beta-kind", 2_048),
                ("multi-gamma-kind", 4_096),
            ),
            attempt="attempt-context-core-multi-0001",
            session="retention-session-core-multi-0001",
            chain="phase2b-step5-core-multi-chain-0001",
            payloads=(b"multi-object-alpha", b"multi-object-beta", b"multi-object-gamma"),
        ),
        "foreign": service_fixture._complete(
            root,
            rows=(
                ("multi-alpha-kind", 1_024),
                ("multi-beta-kind", 2_048),
                ("multi-gamma-kind", 4_096),
            ),
            attempt="attempt-context-core-foreign-0001",
            session="retention-session-core-foreign-0001",
            chain="phase2b-step5-core-foreign-chain-0001",
            payloads=(b"foreign-alpha", b"foreign-beta", b"foreign-gamma"),
        ),
    }


def _common_result_snapshot(result: object) -> tuple[object, ...]:
    names = (
        "authority_profile_id",
        "signer_registry_id",
        "signer_role",
        "signer_key_id",
        "key_epoch",
        "attempt_context_id",
        "retention_session_id",
        "chain_id",
        "plan_id",
        "plan_kind",
        "expected_object_count",
        "complete_head_state_id",
        "complete_head_sha256",
        "complete_checkpoint_sha256",
        "complete_response_sha256",
        "object_bindings_in_exact_order",
        "rollback_disclosure",
    )
    return tuple(getattr(result, name) for name in names)


def test_source_pin_public_surface_and_no_live_or_authority_claims(
    corpora: dict[str, service_fixture._Evidence],
) -> None:
    source = Path(core.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == EXPECTED_CORE_SOURCE_SHA256
    assert core.INSPECTED_STORE_V2_SOURCE_SHA256 == EXPECTED_STORE_V2_SOURCE_SHA256
    store_source = Path(store.__file__).read_bytes()
    assert hashlib.sha256(store_source).hexdigest() == EXPECTED_STORE_V2_SOURCE_SHA256

    signature = inspect.signature(
        core.verify_cas_store_complete_session_replay_core_v2
    )
    assert tuple(signature.parameters) == (
        "stage_b_lock_payload",
        "authority_profile_payload",
        "signer_registry_payload",
        "out_of_band_expected_lock_id_bytes",
        "out_of_band_expected_lock_payload_sha256_digest",
        "retention_plan_payload",
        "request_payloads_in_exact_order",
        "object_payloads_in_exact_order",
        "signed_receipt_payloads_in_exact_order",
        "checkpoint_payloads_in_exact_order",
        "head_payloads_in_exact_order",
        "response_payloads_in_exact_order",
    )
    assert not {
        "path",
        "fd",
        "socket",
        "service",
        "callback",
        "verifier",
    }.intersection(signature.parameters)

    result = _core_verify(corpora["single"])
    assert result.verification_status.endswith("MECHANICS_ONLY_NOT_AUTHORITY")
    assert result.claim_boundary == core.PURE_REPLAY_CLAIM_BOUNDARY
    assert result.rollback_disclosure == core.FAIL_CLOSED_ROLLBACK_DISCLOSURE
    for field_name in (
        "durability_verified",
        "live_cas_store_verified",
        "service_image_verified",
        "runtime_closure_verified",
        "live_uid_gid_verified",
        "live_control_domain_verified",
        "private_key_custody_verified",
        "external_anti_rollback_verified",
        "authority_evidence",
        "readiness_evidence",
        "execution_authorized",
    ):
        assert type(getattr(result, field_name)) is bool
        assert getattr(result, field_name) is False
    for field_name in (
        "canonical_bytes_verified",
        "content_addresses_verified",
        "ed25519_signatures_verified",
        "append_chain_verified",
        "if_absent_idempotent_complete_semantics_verified",
    ):
        assert getattr(result, field_name) is True


@pytest.mark.parametrize("corpus_name", ("single", "duplicate", "multi"))
def test_final_store_v2_raw_corpora_differentially_match_service_verifier(
    corpora: dict[str, service_fixture._Evidence], corpus_name: str
) -> None:
    evidence = corpora[corpus_name]
    service_result = _service_verify(evidence)
    core_result = _core_verify(evidence)
    assert core.require_verified_cas_store_complete_session_replay_core_v2(
        core_result
    ) is core_result
    assert _common_result_snapshot(core_result) == _common_result_snapshot(
        service_result
    )
    expected_dispositions = tuple(
        _json(payload)["signed_body"]["object_disposition"]  # type: ignore[index]
        for payload in evidence.receipts
    )
    assert core_result.object_dispositions_in_exact_order == expected_dispositions
    if corpus_name == "single":
        assert expected_dispositions == ("CREATED_IF_ABSENT",)
    elif corpus_name == "duplicate":
        assert expected_dispositions == (
            "CREATED_IF_ABSENT",
            "PRESENT_VERIFIED",
        )
        assert (
            core_result.object_bindings_in_exact_order[0][4]
            == core_result.object_bindings_in_exact_order[1][4]
        )
    else:
        assert expected_dispositions == (
            "CREATED_IF_ABSENT",
            "CREATED_IF_ABSENT",
            "CREATED_IF_ABSENT",
        )


def test_original_raw_graph_contract_survives_authority_export_rebinding(
    corpora: dict[str, service_fixture._Evidence],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consumer inherits the authority loader's definition-time contract."""

    evidence = corpora["single"]
    expected = _common_result_snapshot(_core_verify(evidence))
    scalar_exports: dict[str, object] = {
        "AUTHORITY_PROFILE_SCHEMA_V2": "evil-authority-profile/999",
        "STAGE_B_TRUST_LOCK_SCHEMA_V2": "evil-stage-b-lock/999",
        "SIGNER_REGISTRY_SCHEMA_V2": "evil-signer-registry/999",
        "DETACHED_SIGNATURE_SCHEMA_V2": "evil-detached-signature/999",
        "SERVICE_STATIC_BINDING_SCHEMA_V2": "evil-service-binding/999",
        "ROLE_KEY_SCHEMA_V1": "evil-role-key/999",
        "AUTHORITY_PROFILE_PREFIX_V2": "evil_authority_profile_",
        "STAGE_B_TRUST_LOCK_PREFIX_V2": "evil_stage_b_lock_",
        "SIGNER_REGISTRY_PREFIX_V2": "evil_signer_registry_",
        "ROLE_KEY_PREFIX_V1": "evil_role_key_",
        "QV2_PROFILE_ID_V1": "evil-qv2-profile",
        "QV2_PROFILE_PAYLOAD_SHA256_V1": "e" * 64,
        "QV2_SOURCE_SHA256_V1": "e" * 64,
        "QV2_DEPENDENCY_LOCK_SHA256_V1": "e" * 64,
    }
    for name, value in scalar_exports.items():
        monkeypatch.setattr(authority, name, value)
    monkeypatch.setattr(authority, "EXACT_QV2_BINDING_V1", {"evil": True})
    monkeypatch.setattr(authority, "ROLE_ORDER_V2", ("EVIL_ROLE",))
    monkeypatch.setattr(authority, "ROLE_CONTROL_DOMAINS_V2", {"EVIL_ROLE": "EVIL"})
    monkeypatch.setattr(
        authority,
        "ROLE_SIGNATURE_PURPOSES_V2",
        {"EVIL_ROLE": ("SIGN_EVIL",)},
    )
    monkeypatch.setattr(authority, "SERVICE_ORDER_V2", ("evil_service",))
    monkeypatch.setattr(authority, "SERVICE_ROLE_V2", {"evil_service": "EVIL_ROLE"})
    monkeypatch.setattr(authority, "SERVICE_TYPES_V2", {"evil_service": "EVIL"})
    monkeypatch.setattr(authority, "SERVICE_UID_GID_V2", {"evil_service": (1, 1)})
    monkeypatch.setattr(authority, "READINESS_NODE_ORDER_V1", ("EVIL_NODE",))

    assert _common_result_snapshot(_core_verify(evidence)) == expected


def test_fully_readdressed_resigned_replacement_schema_graph_is_rejected_transitively(
    corpora: dict[str, service_fixture._Evidence],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-consistent replacement graph cannot redefine this consumer's trust contract."""

    graph = authority_fixture._build_candidate_graph()
    evil_profile_schema = "evil-authority-profile/999"
    evil_registry_schema = "evil-registry/999"
    evil_lock_schema = "evil-stage-b-lock/999"
    evil_qv2 = {
        "schema_version": "evil-qv2/999",
        "profile_id": "evil_qv2_" + "a" * 64,
        "profile_payload_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "dependency_lock_sha256": "d" * 64,
    }
    monkeypatch.setattr(authority, "AUTHORITY_PROFILE_SCHEMA_V2", evil_profile_schema)
    monkeypatch.setattr(authority, "SIGNER_REGISTRY_SCHEMA_V2", evil_registry_schema)
    monkeypatch.setattr(authority, "STAGE_B_TRUST_LOCK_SCHEMA_V2", evil_lock_schema)
    monkeypatch.setattr(authority, "EXACT_QV2_BINDING_V1", evil_qv2)

    graph.profile_body["schema_version"] = evil_profile_schema
    graph.profile_body["qv2_binding"] = dict(evil_qv2)
    graph.registry_body["schema_version"] = evil_registry_schema
    graph.lock_body["schema_version"] = evil_lock_schema
    graph.lock_body["qv2_binding"] = dict(evil_qv2)
    evil = authority_fixture._readdress_and_resign_graph(
        graph,
        governance_private_key=Ed25519PrivateKey.from_private_bytes(b"\x01" * 32),
    )

    evidence = corpora["single"]
    error = _expect_core_error(
        lambda: core.verify_cas_store_complete_session_replay_core_v2(
            stage_b_lock_payload=evil.lock_payload,
            authority_profile_payload=evil.profile_payload,
            signer_registry_payload=evil.registry_payload,
            out_of_band_expected_lock_id_bytes=evil.lock_id.encode("ascii"),
            out_of_band_expected_lock_payload_sha256_digest=hashlib.sha256(
                evil.lock_payload
            ).digest(),
            retention_plan_payload=evidence.plan,
            request_payloads_in_exact_order=evidence.requests,
            object_payloads_in_exact_order=evidence.objects,
            signed_receipt_payloads_in_exact_order=evidence.receipts,
            checkpoint_payloads_in_exact_order=evidence.checkpoints,
            head_payloads_in_exact_order=evidence.heads,
            response_payloads_in_exact_order=evidence.responses,
        ),
        core.FAIL_AUTHORITY,
    )
    assert error.detail == "raw Stage-B/profile/registry V2 graph rejected"


_SEQUENCE_ARGUMENTS = (
    ("requests", "request_payloads_in_exact_order"),
    ("objects", "object_payloads_in_exact_order"),
    ("receipts", "signed_receipt_payloads_in_exact_order"),
    ("checkpoints", "checkpoint_payloads_in_exact_order"),
    ("heads", "head_payloads_in_exact_order"),
    ("responses", "response_payloads_in_exact_order"),
)


@pytest.mark.parametrize(("field_name", "argument_name"), _SEQUENCE_ARGUMENTS)
@pytest.mark.parametrize("cardinality_case", ("missing", "extra"))
def test_each_raw_sequence_rejects_missing_or_extra_items(
    corpora: dict[str, service_fixture._Evidence],
    field_name: str,
    argument_name: str,
    cardinality_case: str,
) -> None:
    evidence = corpora["multi"]
    sequence = getattr(evidence, field_name)
    replacement = (
        sequence[:-1]
        if cardinality_case == "missing"
        else (*sequence, sequence[-1])
    )
    error = _expect_core_error(
        lambda: _core_verify(evidence, **{argument_name: replacement}),
        core.FAIL_INCOMPLETE,
    )
    assert "cardinality" in error.detail


@pytest.mark.parametrize(("field_name", "argument_name"), _SEQUENCE_ARGUMENTS)
def test_each_raw_sequence_rejects_out_of_order_items(
    corpora: dict[str, service_fixture._Evidence],
    field_name: str,
    argument_name: str,
) -> None:
    evidence = corpora["multi"]
    sequence = getattr(evidence, field_name)
    swapped = (sequence[1], sequence[0], *sequence[2:])
    _expect_core_error(
        lambda: _core_verify(evidence, **{argument_name: swapped})
    )


def test_raw_plan_missing_extended_and_foreign_exchange_fail_closed(
    corpora: dict[str, service_fixture._Evidence],
) -> None:
    local = corpora["multi"]
    foreign = corpora["foreign"]
    _expect_core_error(
        lambda: _core_verify(local, retention_plan_payload=b""),
        core.FAIL_INPUT,
    )
    _expect_core_error(
        lambda: _core_verify(
            local, retention_plan_payload=local.plan + b"\n"
        ),
        core.FAIL_CANONICAL,
    )
    extended = _json(local.plan)
    extended["forbidden_extra_field"] = "not-in-plan-schema"
    _expect_core_error(
        lambda: _core_verify(
            local, retention_plan_payload=_canonical(extended)
        ),
        core.FAIL_SCHEMA,
    )
    _expect_core_error(
        lambda: _core_verify(
            local, retention_plan_payload=foreign.plan
        )
    )


def _mutate_wire_content_id(
    evidence: service_fixture._Evidence,
    field_name: str,
) -> service_fixture._Evidence:
    if field_name == "plan":
        body = _json(evidence.plan)
        body["plan_id"] = "phase2b_step5_retention_plan_v1_" + ("f" * 64)
        return replace(evidence, plan=_canonical(body))
    if field_name == "objects":
        objects = list(evidence.objects)
        objects[0] = b"content-address-drift-without-wire-readdress"
        return replace(evidence, objects=tuple(objects))
    sequence_name, key = {
        "requests": ("requests", "request_id"),
        "receipts": ("receipts", "signed_body_sha256"),
        "checkpoints": ("checkpoints", "signed_receipt_sha256"),
        "heads": ("heads", "checkpoint_sha256"),
        "responses": ("responses", "head_sha256"),
    }[field_name]
    sequence = list(getattr(evidence, sequence_name))
    body = _json(sequence[0])
    body[key] = (
        "phase2b_step5_cas_request_v1_" + ("f" * 64)
        if key == "request_id"
        else "sha256:" + ("f" * 64)
    )
    sequence[0] = _canonical(body)
    return replace(evidence, **{sequence_name: tuple(sequence)})


@pytest.mark.parametrize(
    "field_name",
    (
        "plan",
        "requests",
        "objects",
        "receipts",
        "checkpoints",
        "heads",
        "responses",
    ),
)
def test_all_seven_raw_artifact_classes_bind_content_addresses(
    corpora: dict[str, service_fixture._Evidence], field_name: str
) -> None:
    corrupted = _mutate_wire_content_id(corpora["single"], field_name)
    _expect_core_error(lambda: _core_verify(corrupted))


def test_foreign_attempt_session_chain_and_plan_cannot_be_spliced(
    corpora: dict[str, service_fixture._Evidence],
) -> None:
    local = corpora["multi"]
    foreign = corpora["foreign"]
    for field_name, argument_name in _SEQUENCE_ARGUMENTS:
        local_sequence = getattr(local, field_name)
        foreign_sequence = getattr(foreign, field_name)
        mixed = (foreign_sequence[0], *local_sequence[1:])
        _expect_core_error(
            lambda argument_name=argument_name, mixed=mixed: _core_verify(
                local, **{argument_name: mixed}
            )
        )

    _expect_core_error(
        lambda: _core_verify(
            local, retention_plan_payload=foreign.plan
        )
    )
    _expect_core_error(
        lambda: _core_verify(
            local,
            retention_plan_payload=foreign.plan,
            request_payloads_in_exact_order=foreign.requests,
            object_payloads_in_exact_order=foreign.objects,
            signed_receipt_payloads_in_exact_order=foreign.receipts,
            checkpoint_payloads_in_exact_order=foreign.checkpoints,
            head_payloads_in_exact_order=foreign.heads,
            response_payloads_in_exact_order=foreign.responses,
            out_of_band_expected_lock_id_bytes=b"phase2b_step5_stage_b_trust_lock_v2_"
            + (b"f" * 64),
        ),
        core.FAIL_AUTHORITY,
    )


def _replace_receipt_and_embedded_graph(
    evidence: service_fixture._Evidence,
    index: int,
    receipt: dict[str, object],
) -> service_fixture._Evidence:
    receipt_payload = _canonical(receipt)
    checkpoint = _json(evidence.checkpoints[index])
    checkpoint["signed_receipt"] = receipt
    checkpoint["signed_receipt_sha256"] = _sha256(receipt_payload)
    checkpoint_payload = _canonical(checkpoint)

    head = _json(evidence.heads[index])
    head["checkpoint_sha256"] = _sha256(checkpoint_payload)
    head["signed_receipt_sha256"] = _sha256(receipt_payload)
    head_payload = _canonical(head)

    response = _json(evidence.responses[index])
    response["checkpoint_sha256"] = _sha256(checkpoint_payload)
    response["checkpoint_record"] = checkpoint
    response["head_sha256"] = _sha256(head_payload)
    response["head_record"] = head
    response_payload = _canonical(response)

    receipts = list(evidence.receipts)
    checkpoints = list(evidence.checkpoints)
    heads = list(evidence.heads)
    responses = list(evidence.responses)
    receipts[index] = receipt_payload
    checkpoints[index] = checkpoint_payload
    heads[index] = head_payload
    responses[index] = response_payload
    return replace(
        evidence,
        receipts=tuple(receipts),
        checkpoints=tuple(checkpoints),
        heads=tuple(heads),
        responses=tuple(responses),
    )


def _resigned_disposition(
    evidence: service_fixture._Evidence,
    index: int,
    disposition: str,
) -> service_fixture._Evidence:
    receipt = _json(evidence.receipts[index])
    signed_body = receipt["signed_body"]
    assert type(signed_body) is dict
    signed_body["object_disposition"] = disposition
    receipt["signed_body_sha256"] = _sha256(_canonical(signed_body))
    signature_preimage = (
        store.CAS_STORE_SIGNATURE_DOMAIN.encode("ascii")
        + b"\x00"
        + _canonical(signed_body)
    )
    receipt["ed25519_signature_hex"] = Ed25519PrivateKey.from_private_bytes(
        service_fixture.SEED
    ).sign(signature_preimage).hex()
    return _replace_receipt_and_embedded_graph(evidence, index, receipt)


@pytest.mark.parametrize(
    ("corpus_name", "index", "wrong_disposition"),
    (
        ("single", 0, "PRESENT_VERIFIED"),
        ("duplicate", 1, "CREATED_IF_ABSENT"),
    ),
)
def test_cryptographically_valid_created_vs_present_lie_fails_closed(
    corpora: dict[str, service_fixture._Evidence],
    corpus_name: str,
    index: int,
    wrong_disposition: str,
) -> None:
    corrupted = _resigned_disposition(
        corpora[corpus_name], index, wrong_disposition
    )
    _expect_core_error(
        lambda: _core_verify(corrupted),
        core.FAIL_BINDING,
    )


@pytest.mark.parametrize(("field_name", "argument_name"), _SEQUENCE_ARGUMENTS)
def test_raw_sequence_container_and_item_types_are_exact(
    corpora: dict[str, service_fixture._Evidence],
    field_name: str,
    argument_name: str,
) -> None:
    evidence = corpora["single"]
    sequence = getattr(evidence, field_name)
    _expect_core_error(
        lambda: _core_verify(evidence, **{argument_name: list(sequence)}),
        core.FAIL_INPUT,
    )
    changed = list(sequence)
    changed[0] = bytearray(changed[0])
    _expect_core_error(
        lambda: _core_verify(evidence, **{argument_name: tuple(changed)}),
        core.FAIL_INPUT,
    )


@pytest.mark.parametrize(
    ("argument_name", "field_name"),
    (
        ("retention_plan_payload", "plan"),
        ("request_payloads_in_exact_order", "requests"),
        ("signed_receipt_payloads_in_exact_order", "receipts"),
        ("checkpoint_payloads_in_exact_order", "checkpoints"),
        ("head_payloads_in_exact_order", "heads"),
        ("response_payloads_in_exact_order", "responses"),
    ),
)
def test_every_json_wire_requires_exact_rfc8785_bytes(
    corpora: dict[str, service_fixture._Evidence],
    argument_name: str,
    field_name: str,
) -> None:
    evidence = corpora["single"]
    if field_name == "plan":
        replacement: object = evidence.plan + b"\n"
    else:
        sequence = getattr(evidence, field_name)
        replacement = (sequence[0] + b"\n",)
    _expect_core_error(
        lambda: _core_verify(evidence, **{argument_name: replacement}),
        core.FAIL_CANONICAL,
    )


def test_numeric_spelling_extra_fields_and_oob_pin_types_fail_closed(
    corpora: dict[str, service_fixture._Evidence],
) -> None:
    evidence = corpora["single"]
    noncanonical_number = evidence.plan.replace(
        b'"expected_object_count":1',
        b'"expected_object_count":1.0',
    )
    assert noncanonical_number != evidence.plan
    _expect_core_error(
        lambda: _core_verify(
            evidence, retention_plan_payload=noncanonical_number
        ),
        core.FAIL_CANONICAL,
    )

    request = _json(evidence.requests[0])
    request["forbidden_extra_field"] = "not-in-request-schema"
    _expect_core_error(
        lambda: _core_verify(
            evidence,
            request_payloads_in_exact_order=(_canonical(request),),
        ),
        core.FAIL_SCHEMA,
    )
    _expect_core_error(
        lambda: _core_verify(
            evidence,
            out_of_band_expected_lock_id_bytes=bytearray(
                service_fixture.DEFAULT_GRAPH.out_of_band_expected_lock_id_bytes
            ),
        ),
        core.FAIL_INPUT,
    )
    _expect_core_error(
        lambda: _core_verify(
            evidence,
            out_of_band_expected_lock_payload_sha256_digest=b"\x00" * 32,
        ),
        core.FAIL_AUTHORITY,
    )


class _EqAll:
    def __eq__(self, other: object) -> bool:
        return True


def test_exact_python_types_do_not_accept_bool_float_subclass_or_eqall(
    corpora: dict[str, service_fixture._Evidence],
) -> None:
    evidence = corpora["single"]
    for bad in (True, 1.0, _EqAll()):
        _expect_core_error(
            lambda bad=bad: _core_verify(
                evidence,
                retention_plan_payload=bad,
            ),
            core.FAIL_INPUT,
        )

    class BytesSubclass(bytes):
        pass

    _expect_core_error(
        lambda: _core_verify(
            evidence,
            retention_plan_payload=BytesSubclass(evidence.plan),
        ),
        core.FAIL_INPUT,
    )


def _fully_readdressed_invalid_signature(
    evidence: service_fixture._Evidence,
) -> service_fixture._Evidence:
    assert len(evidence.receipts) == 1
    receipt = _json(evidence.receipts[0])
    signature = receipt["ed25519_signature_hex"]
    assert type(signature) is str
    receipt["ed25519_signature_hex"] = (
        ("0" if signature[0] != "0" else "1") + signature[1:]
    )
    return _replace_receipt_and_embedded_graph(evidence, 0, receipt)


def _bomb(*args: object, **kwargs: object) -> object:
    raise AssertionError("module-global helper rebinding entered captured core")


def test_bad_signature_survives_module_symbol_rebinding(
    corpora: dict[str, service_fixture._Evidence],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = corpora["single"]
    invalid = _fully_readdressed_invalid_signature(evidence)

    class AcceptEverySignature:
        @classmethod
        def from_public_bytes(cls, payload: bytes) -> "AcceptEverySignature":
            return cls()

        def verify(self, signature: bytes, preimage: bytes) -> None:
            return None

    monkeypatch.setattr(core, "_Ed25519PublicKey", AcceptEverySignature)
    monkeypatch.setattr(core, "_InvalidSignature", Exception)
    monkeypatch.setattr(core, "_canonicalize_rfc8785_v1", _bomb)
    monkeypatch.setattr(core, "_decode_canonical_rfc8785_v1", _bomb)
    monkeypatch.setattr(core, "_authority_load_lock_v2", _bomb)
    monkeypatch.setattr(core, "_authority_load_profile_v2", _bomb)
    monkeypatch.setattr(core, "_authority_load_registry_v2", _bomb)
    monkeypatch.setattr(core, "_authority_require_graph_v2", _bomb)
    monkeypatch.setattr(core, "_authority_role_key_id_v2", _bomb)
    monkeypatch.setattr(
        core, "_verify_signed_store_receipt", lambda *args, **kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        core, "_issue_verified_result", lambda *args, **kwargs: object(),
        raising=False,
    )
    for name in (
        "load_step5_stage_b_trust_lock_v2",
        "load_step5_authority_profile_v2",
        "load_step5_signer_registry_v2",
        "require_locked_step5_authority_graph_v2",
        "phase2b_step5_role_key_id_v2",
        "canonical_step5_json_bytes_v2",
    ):
        monkeypatch.setattr(authority, name, _bomb)

    valid_result = _core_verify(evidence)
    assert (
        valid_result.object_bindings_in_exact_order
        == _service_verify(evidence).object_bindings_in_exact_order
    )
    error = _expect_core_error(
        lambda: _core_verify(invalid),
        core.FAIL_SIGNATURE,
    )
    assert "Ed25519" in error.detail


def test_private_result_direct_construction_copy_forgery_and_mutation_fail(
    corpora: dict[str, service_fixture._Evidence],
) -> None:
    evidence = corpora["single"]
    genuine = _core_verify(evidence)
    assert not hasattr(core, "_VERIFIED_RESULT_ISSUANCE_TOKEN")
    assert not hasattr(core, "_issue_result")
    assert not hasattr(core, "_build_store_replay_core_v2")
    with pytest.raises(TypeError, match="privately issued"):
        core.VerifiedCasStoreCompleteSessionReplayCoreV2()
    with pytest.raises(TypeError, match="privately issued"):
        core.VerifiedCasStoreCompleteSessionReplayCoreV2(object())
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(genuine)
    with pytest.raises(TypeError, match="cannot be deep-copied"):
        copy.deepcopy(genuine)

    forged = object.__new__(core.VerifiedCasStoreCompleteSessionReplayCoreV2)
    for field in fields(type(genuine)):
        object.__setattr__(forged, field.name, getattr(genuine, field.name))
    _expect_core_error(
        lambda: core.require_verified_cas_store_complete_session_replay_core_v2(
            forged
        ),
        core.FAIL_INPUT,
    )

    mutated = _core_verify(evidence)
    object.__setattr__(mutated, "authority_evidence", True)
    _expect_core_error(
        lambda: core.require_verified_cas_store_complete_session_replay_core_v2(
            mutated
        ),
        core.FAIL_BINDING,
    )
    with pytest.raises((AttributeError, TypeError)):
        genuine.execution_authorized = True  # type: ignore[misc]


def test_clean_import_excludes_service_socket_filesystem_owner_time_runtime_auditor_qv2() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    source = Path(core.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    direct_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            direct_imports.update(
                alias.name.split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            direct_imports.add(node.module.split(".")[0])
    assert not {"os", "pathlib", "socket", "tempfile", "shutil"}.intersection(
        direct_imports
    )

    script = """
import json
import sys
import hegel_machine.phase2b_formal_wire_covert_step5_store_replay_core_v2
forbidden_exact = {
    'hegel_machine.phase2b_formal_wire_covert_step5_store_v1',
    'hegel_machine.phase2b_formal_wire_covert_step5_store_v2',
    'hegel_machine.phase2b_formal_wire_covert_step5_logical_time_v1',
    'hegel_machine.phase2b_formal_wire_covert_step5_logical_time_replay_core_v1',
    'hegel_machine.phase2b_formal_wire_covert_step5_owner_status_v1',
    'hegel_machine.phase2b_formal_wire_covert_step5_runtime_v1',
    'hegel_machine.phase2b_formal_wire_covert_step5_runtime_closure_v1',
}
forbidden_markers = ('auditor', 'qualification_v2', '.qv2')
loaded = sorted(
    name for name in sys.modules
    if name in forbidden_exact
    or (
        name.startswith('hegel_machine.')
        and any(marker in name.lower() for marker in forbidden_markers)
    )
    or name == 'socket'
)
print(json.dumps(loaded))
raise SystemExit(1 if loaded else 0)
"""
    environment = dict(os.environ)
    src = str(repository_root / "src")
    environment["PYTHONPATH"] = (
        src
        if not environment.get("PYTHONPATH")
        else src + os.pathsep + environment["PYTHONPATH"]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout.strip() == "[]"
