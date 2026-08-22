from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import phase2b_formal_wire_covert_step5_logical_time_replay_core_v1 as core
from hegel_machine import phase2b_formal_wire_covert_step5_logical_time_v1 as logical_time
from hegel_machine.rfc8785_v1 import canonicalize_rfc8785_v1


_TOKEN_ID_DOMAIN = b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/TOKEN_ID/V1\x00"
_HEAD_ID_DOMAIN = b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/LEDGER_HEAD_ID/V1\x00"
_CHECKPOINT_ID_DOMAIN = (
    b"HEGEL/PHASE2B/STEP5/LOGICAL_TIME/LEDGER_CHECKPOINT_ID/V1\x00"
)


@dataclass(frozen=True)
class CandidateGraph:
    logical_time_private_key: Ed25519PrivateKey
    logical_time_public_key: bytes
    logical_time_key_id: str
    profile_id: str
    registry_id: str
    profile_payload: bytes
    registry_payload: bytes
    lock_id: str
    lock_payload: bytes


@dataclass(frozen=True)
class RawChain:
    graph: CandidateGraph
    attempt_context_id: str
    requests: tuple[bytes, ...]
    tokens: tuple[bytes, ...]
    checkpoints: tuple[bytes, ...]
    heads: tuple[bytes, ...]
    event_rows: tuple[tuple[str, str, str, str], ...]

    def core_plan(self) -> tuple[core.ExpectedLogicalEventCoreV1, ...]:
        return tuple(core.ExpectedLogicalEventCoreV1(*row) for row in self.event_rows)

    def service_plan(self) -> tuple[logical_time.ExpectedLogicalEventV1, ...]:
        return tuple(logical_time.ExpectedLogicalEventV1(*row) for row in self.event_rows)


def _canonical(value: object) -> bytes:
    return canonicalize_rfc8785_v1(value)  # type: ignore[arg-type]


def _json(payload: bytes) -> dict[str, object]:
    value = json.loads(payload.decode("utf-8"))
    assert type(value) is dict
    return value


def _content_id(prefix: str, domain: bytes, body: object) -> str:
    return prefix + hashlib.sha256(domain + _canonical(body)).hexdigest()


def _all_false_readiness() -> dict[str, object]:
    return {
        "node_states_in_canonical_topological_order": [
            {"node_id": node_id, "satisfied": False}
            for node_id in authority.READINESS_NODE_ORDER_V1
        ],
        "satisfied_node_count": 0,
        "highest_contiguous_satisfied_node": None,
        "all_nodes_satisfied": False,
        "readiness_evidence_overlay_issued": False,
    }


def _build_candidate_graph(*, salt: int = 0) -> CandidateGraph:
    private_keys: dict[str, Ed25519PrivateKey] = {}
    role_keys: list[dict[str, object]] = []
    for index, role in enumerate(authority.ROLE_ORDER_V2, start=1):
        private_key = Ed25519PrivateKey.from_private_bytes(
            bytes([index + salt]) * 32
        )
        private_keys[role] = private_key
        public_key = private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        row: dict[str, object] = {
            "schema_version": authority.ROLE_KEY_SCHEMA_V1,
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "authorized_signature_purposes": list(
                authority.ROLE_SIGNATURE_PURPOSES_V2[role]
            ),
            "algorithm": "Ed25519",
            "key_epoch": 1,
            "public_key_hex": public_key.hex(),
        }
        row["key_id"] = authority.phase2b_step5_role_key_id_v2(row)
        role_keys.append(row)

    service_bindings: list[dict[str, object]] = []
    for index, service_name in enumerate(authority.SERVICE_ORDER_V2, start=1):
        role = authority.SERVICE_ROLE_V2[service_name]
        role_key = role_keys[authority.ROLE_ORDER_V2.index(role)]
        uid, gid = authority.SERVICE_UID_GID_V2[service_name]

        def oci(offset: int) -> str:
            return "sha256:" + f"{salt * 1000 + index * 10 + offset:064x}"

        binding: dict[str, object] = {
            "schema_version": authority.SERVICE_STATIC_BINDING_SCHEMA_V2,
            "service_name": service_name,
            "service_type": authority.SERVICE_TYPES_V2[service_name],
            "binding_status": "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE",
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "uid": uid,
            "gid": gid,
            "role_key_id": role_key["key_id"],
            "key_epoch": 1,
            "authorized_signature_purposes": list(
                authority.ROLE_SIGNATURE_PURPOSES_V2[role]
            ),
            "oci_index_digest": oci(1),
            "oci_manifest_digest": oci(2),
            "oci_config_digest": oci(3),
            "ordered_layer_digests": [oci(4), oci(5)],
            "static_runtime_closure_policy_sha256": (
                f"{100 + salt * 10 + index:064x}"
            ),
        }
        binding["binding_id"] = (
            authority.phase2b_step5_service_static_binding_id_v2(binding)
        )
        service_bindings.append(binding)

    profile_body: dict[str, object] = {
        "schema_version": authority.AUTHORITY_PROFILE_SCHEMA_V2,
        "status": "CANDIDATE_STATIC_SERVICE_AUTHORITY_PROFILE_NOT_ADMITTED",
        "claim_level": (
            "STATIC_CANDIDATE_FOUNDATION_NOT_IMAGE_EXISTENCE_NOT_LIVE_"
            "RUNTIME_CLOSURE_NOT_AUTHORITY_NOT_READINESS"
        ),
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "profile_epoch": 2,
        "owner_authorization_binding": dict(authority._EXACT_OWNER_BINDING_V2),
        "readiness_dag_binding": {
            "schema_version": (
                "hegel-machine-phase2b-actual-unsealed-960-readiness-dag/1"
            ),
            "readiness_dag_id": authority.READINESS_DAG_ID_V1,
        },
        "readiness_state": _all_false_readiness(),
        "qv2_binding": dict(authority.EXACT_QV2_BINDING_V1),
        "source_binding": {
            "source_commit_sha": f"{salt + 1:040x}",
            "source_archive_sha256": f"{salt + 2:064x}",
            "dependency_lock_sha256": f"{salt + 3:064x}",
        },
        "service_static_bindings_in_exact_order": service_bindings,
        "role_policy": {
            "roles_in_exact_order": list(authority.ROLE_ORDER_V2),
            "role_count": 8,
            "distinct_public_keys": True,
            "distinct_key_ids": True,
            "distinct_control_domains": True,
            "key_epoch": 1,
            "purpose_policy_version": 2,
            "services_in_exact_order": list(authority.SERVICE_ORDER_V2),
            "service_binding_count": 5,
            "distinct_service_uids_gids": True,
            "static_service_identity_only": True,
        },
        "governance_root": role_keys[0],
        "threat_model": dict(authority._THREAT_MODEL_V2),
        "permanent_nonclaims": dict(authority._PERMANENT_NONCLAIMS_V2),
    }
    profile_id = authority.phase2b_step5_content_id_v2(
        "authority_profile", profile_body
    )
    profile_payload = authority.canonical_step5_json_bytes_v2(
        {"body": profile_body, "id": profile_id}
    )

    registry_body: dict[str, object] = {
        "schema_version": authority.SIGNER_REGISTRY_SCHEMA_V2,
        "authority_profile_id": profile_id,
        "registry_epoch": 2,
        "status": "CANDIDATE_NOT_ADMITTED",
        "claim_level": (
            "SIGNED_STATIC_KEY_REGISTRY_CANDIDATE_NOT_AUTHORITY_NOT_READINESS"
        ),
        "revocation_epoch": 0,
        "role_keys_in_exact_order": role_keys,
        "readiness_state": _all_false_readiness(),
    }
    registry_id = authority.phase2b_step5_content_id_v2(
        "signer_registry", registry_body
    )
    governance_key = private_keys["GOVERNANCE_ANCHOR"]
    signature_preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry",
        body=registry_body,
        role="GOVERNANCE_ANCHOR",
        purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=role_keys[0]["key_id"],  # type: ignore[arg-type]
    )
    registry_payload = authority.canonical_step5_json_bytes_v2(
        {
            "body": registry_body,
            "id": registry_id,
            "signatures": [
                {
                    "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
                    "algorithm": "Ed25519",
                    "role": "GOVERNANCE_ANCHOR",
                    "purpose": "SIGN_SIGNER_REGISTRY",
                    "key_epoch": 1,
                    "signer_key_id": role_keys[0]["key_id"],
                    "signature_hex": governance_key.sign(
                        signature_preimage
                    ).hex(),
                }
            ],
        }
    )

    lock_body: dict[str, object] = {
        "schema_version": authority.STAGE_B_TRUST_LOCK_SCHEMA_V2,
        "status": "CANDIDATE_OUT_OF_BAND_STAGE_B_TRUST_LOCK",
        "claim_level": "CANDIDATE_GRAPH_PIN_ONLY_NOT_AUTHORITY_NOT_READINESS",
        "canonicalization": "RFC8785_EXACT_BYTES_NO_TRAILING_LF",
        "authority_profile_schema": authority.AUTHORITY_PROFILE_SCHEMA_V2,
        "authority_profile_id": profile_id,
        "authority_profile_payload_sha256": hashlib.sha256(
            profile_payload
        ).hexdigest(),
        "signer_registry_schema": authority.SIGNER_REGISTRY_SCHEMA_V2,
        "signer_registry_id": registry_id,
        "signer_registry_payload_sha256": hashlib.sha256(
            registry_payload
        ).hexdigest(),
        "qv2_binding": dict(authority.EXACT_QV2_BINDING_V1),
    }
    lock_id = authority.phase2b_step5_content_id_v2(
        "stage_b_trust_lock", lock_body
    )
    lock_payload = authority.canonical_step5_json_bytes_v2(
        {"body": lock_body, "id": lock_id}
    )
    logical_role_index = authority.ROLE_ORDER_V2.index("LOGICAL_TIME")
    logical_private = private_keys["LOGICAL_TIME"]
    logical_public = logical_private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return CandidateGraph(
        logical_time_private_key=logical_private,
        logical_time_public_key=logical_public,
        logical_time_key_id=role_keys[logical_role_index]["key_id"],  # type: ignore[arg-type]
        profile_id=profile_id,
        registry_id=registry_id,
        profile_payload=profile_payload,
        registry_payload=registry_payload,
        lock_id=lock_id,
        lock_payload=lock_payload,
    )


def _build_raw_chain(
    root: Path,
    graph: CandidateGraph,
    event_count: int,
    *,
    attempt_context_id: str | None = None,
) -> RawChain:
    attempt = (
        attempt_context_id
        if attempt_context_id is not None
        else "phase2b_step5_attempt_" + "a" * 64
    )
    purposes = (
        "R01_STATUS_OBSERVED",
        "R02_BRANCH_OBSERVED",
        "R03_TERMINAL_OBSERVED",
    )
    assert event_count in (0, 1, 3)
    requests: list[bytes] = []
    tokens: list[bytes] = []
    checkpoints: list[bytes] = []
    heads: list[bytes] = []
    event_rows: list[tuple[str, str, str, str]] = []
    with logical_time.LogicalTimeLedgerV1(
        root_directory=str(root),
        authority_profile_id=graph.profile_id,
        signer_registry_id=graph.registry_id,
        logical_time_key_id=graph.logical_time_key_id,
        logical_time_key_epoch=1,
        attempt_context_id=attempt,
        logical_time_public_key=graph.logical_time_public_key,
        create=True,
    ) as ledger:
        previous_head = ledger.current_head_bytes()
        heads.append(previous_head)
        prior_token_id = logical_time.GENESIS_TOKEN_ID_V1
        for index, purpose in enumerate(purposes[:event_count]):
            nonce = format(index + 1, "x") * 64
            artifact_id = "artifact_" + format(index + 4, "x") * 64
            artifact_sha256 = "sha256:" + format(index + 8, "x") * 64
            request_payload = logical_time.build_logical_time_request_v1(
                authority_profile_id=graph.profile_id,
                signer_registry_id=graph.registry_id,
                logical_time_key_id=graph.logical_time_key_id,
                logical_time_key_epoch=1,
                attempt_context_id=attempt,
                request_nonce=nonce,
                purpose=purpose,
                bound_artifact_id=artifact_id,
                bound_artifact_sha256=artifact_sha256,
                prior_token_id=prior_token_id,
                next_sequence=index,
            )
            issuance = ledger.issue_service_side_v1(
                request_payload=request_payload,
                expected_previous_head_payload=previous_head,
                private_key=graph.logical_time_private_key,
            )
            token = logical_time.load_signed_logical_time_token_v1(
                issuance.signed_token_payload
            )
            requests.append(request_payload)
            tokens.append(issuance.signed_token_payload)
            checkpoints.append(issuance.checkpoint_payload)
            heads.append(issuance.head_payload)
            event_rows.append((purpose, artifact_id, artifact_sha256, nonce))
            previous_head = issuance.head_payload
            prior_token_id = token.token_id
    return RawChain(
        graph=graph,
        attempt_context_id=attempt,
        requests=tuple(requests),
        tokens=tuple(tokens),
        checkpoints=tuple(checkpoints),
        heads=tuple(heads),
        event_rows=tuple(event_rows),
    )


def _replay(
    chain: RawChain,
    *,
    requests: object | None = None,
    tokens: object | None = None,
    checkpoints: object | None = None,
    heads: object | None = None,
    event_plan: object | None = None,
    expected_attempt_context_id: object | None = None,
    lock_id_pin: object | None = None,
    lock_sha_pin: object | None = None,
) -> core.VerifiedLogicalTimeReplayCoreV1:
    return core.verify_logical_time_replay_core_v1(
        chain.graph.lock_payload,
        chain.graph.profile_payload,
        chain.graph.registry_payload,
        chain.requests if requests is None else requests,  # type: ignore[arg-type]
        chain.tokens if tokens is None else tokens,  # type: ignore[arg-type]
        chain.checkpoints if checkpoints is None else checkpoints,  # type: ignore[arg-type]
        chain.heads if heads is None else heads,  # type: ignore[arg-type]
        out_of_band_expected_lock_id_bytes=(
            chain.graph.lock_id.encode("ascii")
            if lock_id_pin is None
            else lock_id_pin  # type: ignore[arg-type]
        ),
        out_of_band_expected_payload_sha256_digest=(
            hashlib.sha256(chain.graph.lock_payload).digest()
            if lock_sha_pin is None
            else lock_sha_pin  # type: ignore[arg-type]
        ),
        expected_attempt_context_id=(
            chain.attempt_context_id
            if expected_attempt_context_id is None
            else expected_attempt_context_id  # type: ignore[arg-type]
        ),
        expected_event_plan=(
            chain.core_plan()
            if event_plan is None
            else event_plan  # type: ignore[arg-type]
        ),
    )


def _expect_code(code: str, operation: Callable[[], object]) -> None:
    with pytest.raises(core.Phase2BStep5LogicalTimeReplayCoreErrorV1) as caught:
        operation()
    assert caught.value.code == code


@pytest.mark.parametrize("event_count", [0, 1, 3])
def test_zero_one_and_three_event_chains_match_logical_time_v1(
    tmp_path: Path, event_count: int
) -> None:
    graph = _build_candidate_graph()
    chain = _build_raw_chain(tmp_path / f"ledger-{event_count}", graph, event_count)

    for index in range(event_count):
        verified_token = logical_time.verify_signed_logical_time_token_v1(
            chain.tokens[index],
            request_payload=chain.requests[index],
            logical_time_public_key=graph.logical_time_public_key,
            expected_authority_profile_id=graph.profile_id,
            expected_signer_registry_id=graph.registry_id,
            expected_logical_time_key_id=graph.logical_time_key_id,
            expected_logical_time_key_epoch=1,
            expected_attempt_context_id=chain.attempt_context_id,
        )
        verified_head = logical_time.verify_logical_time_ledger_transition_v1(
            chain.heads[index],
            chain.checkpoints[index],
            chain.heads[index + 1],
            logical_time_public_key=graph.logical_time_public_key,
            expected_authority_profile_id=graph.profile_id,
            expected_signer_registry_id=graph.registry_id,
            expected_logical_time_key_id=graph.logical_time_key_id,
            expected_logical_time_key_epoch=1,
            expected_attempt_context_id=chain.attempt_context_id,
        )
        assert verified_token.token_id == _json(chain.tokens[index])["token_id"]
        assert verified_head.head_id == _json(chain.heads[index + 1])["head_id"]

    service_result = logical_time.verify_logical_time_ledger_replay_v1(
        chain.heads[0],
        chain.checkpoints,
        chain.heads[1:],
        logical_time_public_key=graph.logical_time_public_key,
        expected_authority_profile_id=graph.profile_id,
        expected_signer_registry_id=graph.registry_id,
        expected_logical_time_key_id=graph.logical_time_key_id,
        expected_logical_time_key_epoch=1,
        expected_attempt_context_id=chain.attempt_context_id,
        expected_event_plan=chain.service_plan(),
    )
    replay_result = _replay(chain)
    assert core.require_verified_logical_time_replay_core_v1(replay_result) is (
        replay_result
    )
    assert replay_result.verified_event_count == event_count
    assert replay_result.verified_request_count == event_count
    assert replay_result.verified_signature_count == event_count
    assert replay_result.verified_checkpoint_count == event_count
    assert replay_result.verified_head_count == event_count + 1
    assert replay_result.verified_event_plan == chain.event_rows
    assert replay_result.request_ids == tuple(
        logical_time.load_logical_time_request_v1(payload).request_id
        for payload in chain.requests
    )
    assert replay_result.token_ids == service_result.token_ids
    assert replay_result.checkpoint_ids == service_result.checkpoint_ids
    assert replay_result.head_ids == service_result.head_ids
    assert replay_result.final_head_id == service_result.final_head_id
    assert replay_result.final_head_payload_sha256 == (
        service_result.final_head_payload_sha256
    )
    assert replay_result.zero_event_genesis_only is (event_count == 0)


def test_result_is_explicitly_supplied_chain_only_and_nonauthoritative(
    tmp_path: Path,
) -> None:
    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 1)
    result = _replay(chain)
    assert result.supplied_chain_final_head_verified is True
    assert result.verifier_only is True
    for field_name in (
        "private_key_present",
        "issuer_path_present",
        "filesystem_observed",
        "service_observed",
        "external_currentness_verified",
        "external_anti_fork_verified",
        "trusted_utc_verified",
        "external_cas_checkpoint_verified",
        "runtime_closure_verified",
        "authority_evidence",
        "readiness_evidence",
        "r01_satisfied",
        "execution_authorized",
    ):
        assert type(getattr(result, field_name)) is bool
        assert getattr(result, field_name) is False
    _expect_code(core.FAIL_CHAIN_V1, lambda: core.verify_external_currentness_v1())


@pytest.mark.parametrize(
    ("field_name", "replacement", "expected_code"),
    [
        ("requests", lambda chain: chain.requests[:-1], core.FAIL_CHAIN_V1),
        (
            "tokens",
            lambda chain: (*chain.tokens, chain.tokens[-1]),
            core.FAIL_CHAIN_V1,
        ),
        (
            "checkpoints",
            lambda chain: (
                chain.checkpoints[1],
                chain.checkpoints[0],
                chain.checkpoints[2],
            ),
            core.FAIL_BINDING_V1,
        ),
        (
            "heads",
            lambda chain: (
                chain.heads[0],
                chain.heads[2],
                chain.heads[1],
                chain.heads[3],
            ),
            core.FAIL_CHAIN_V1,
        ),
        (
            "event_plan",
            lambda chain: (
                chain.core_plan()[1],
                chain.core_plan()[0],
                chain.core_plan()[2],
            ),
            core.FAIL_BINDING_V1,
        ),
    ],
)
def test_missing_extra_or_out_of_order_raw_evidence_fails_closed(
    tmp_path: Path,
    field_name: str,
    replacement: Callable[[RawChain], object],
    expected_code: str,
) -> None:
    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 3)
    _expect_code(
        expected_code,
        lambda: _replay(chain, **{field_name: replacement(chain)}),
    )


def _readdress_token(body: dict[str, object]) -> bytes:
    subject = {key: value for key, value in body.items() if key != "token_id"}
    body["token_id"] = _content_id(
        logical_time.TOKEN_ID_PREFIX_V1, _TOKEN_ID_DOMAIN, subject
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


def _readdress_head(body: dict[str, object]) -> bytes:
    subject = {key: value for key, value in body.items() if key != "head_id"}
    body["head_id"] = _content_id(
        logical_time.LEDGER_HEAD_ID_PREFIX_V1, _HEAD_ID_DOMAIN, subject
    )
    return _canonical(body)


def _invalid_signature_evidence(
    chain: RawChain,
) -> tuple[bytes, bytes, bytes]:
    token = _json(chain.tokens[0])
    signature = token["signature_hex"]
    assert type(signature) is str
    token["signature_hex"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    token_payload = _readdress_token(token)

    checkpoint = _json(chain.checkpoints[0])
    checkpoint["token_id"] = token["token_id"]
    checkpoint["signed_token"] = token
    checkpoint["signed_token_sha256"] = (
        "sha256:" + hashlib.sha256(token_payload).hexdigest()
    )
    checkpoint_payload = _readdress_checkpoint(checkpoint)

    head = _json(chain.heads[1])
    head["last_token_id"] = token["token_id"]
    head["last_checkpoint_id"] = _json(checkpoint_payload)["checkpoint_id"]
    head_payload = _readdress_head(head)
    return token_payload, checkpoint_payload, head_payload


def test_fully_readdressed_invalid_signature_is_rejected(tmp_path: Path) -> None:
    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 1)
    token_payload, checkpoint_payload, head_payload = (
        _invalid_signature_evidence(chain)
    )
    _expect_code(
        core.FAIL_SIGNATURE_V1,
        lambda: _replay(
            chain,
            tokens=(token_payload,),
            checkpoints=(checkpoint_payload,),
            heads=(chain.heads[0], head_payload),
        ),
    )


def test_rebound_signature_symbols_cannot_accept_invalid_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 1)
    token_payload, checkpoint_payload, head_payload = (
        _invalid_signature_evidence(chain)
    )
    monkeypatch.setattr(core, "_verify_token_signature", lambda *args, **kwargs: None)
    monkeypatch.setattr(core, "_Ed25519PublicKey", None)
    monkeypatch.setattr(core, "_InvalidSignature", None)
    _expect_code(
        core.FAIL_SIGNATURE_V1,
        lambda: _replay(
            chain,
            tokens=(token_payload,),
            checkpoints=(checkpoint_payload,),
            heads=(chain.heads[0], head_payload),
        ),
    )


@pytest.mark.parametrize("mismatch", ["attempt", "nonce", "purpose"])
def test_attempt_nonce_and_purpose_are_independently_bound(
    tmp_path: Path, mismatch: str
) -> None:
    graph = _build_candidate_graph()
    chain = _build_raw_chain(tmp_path / "ledger", graph, 1)
    if mismatch == "attempt":
        _expect_code(
            core.FAIL_BINDING_V1,
            lambda: _replay(
                chain,
                expected_attempt_context_id=(
                    "phase2b_step5_attempt_" + "b" * 64
                ),
            ),
        )
        return
    row = chain.event_rows[0]
    event = core.ExpectedLogicalEventCoreV1(
        "WRONG_PURPOSE" if mismatch == "purpose" else row[0],
        row[1],
        row[2],
        "f" * 64 if mismatch == "nonce" else row[3],
    )
    _expect_code(
        core.FAIL_BINDING_V1,
        lambda: _replay(chain, event_plan=(event,)),
    )


def test_complete_foreign_attempt_chain_cannot_be_spliced(
    tmp_path: Path,
) -> None:
    graph = _build_candidate_graph()
    local = _build_raw_chain(
        tmp_path / "local",
        graph,
        1,
        attempt_context_id="phase2b_step5_attempt_" + "a" * 64,
    )
    foreign = _build_raw_chain(
        tmp_path / "foreign",
        graph,
        1,
        attempt_context_id="phase2b_step5_attempt_" + "b" * 64,
    )
    _expect_code(
        core.FAIL_BINDING_V1,
        lambda: _replay(
            local,
            requests=foreign.requests,
            tokens=foreign.tokens,
            checkpoints=foreign.checkpoints,
            heads=foreign.heads,
        ),
    )


class _EqAll:
    def __eq__(self, other: object) -> bool:
        return True


class _EventSubclass(core.ExpectedLogicalEventCoreV1):
    pass


@pytest.mark.parametrize("case", ["list", "bytearray", "dto-subclass", "eqall-pin"])
def test_transport_and_python_types_are_exact(tmp_path: Path, case: str) -> None:
    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 1)
    if case == "list":
        operation = lambda: _replay(chain, requests=list(chain.requests))
    elif case == "bytearray":
        operation = lambda: _replay(
            chain, tokens=(bytearray(chain.tokens[0]),)
        )
    elif case == "dto-subclass":
        source = chain.core_plan()[0]
        forged = object.__new__(_EventSubclass)
        for name in ("purpose", "bound_artifact_id", "bound_artifact_sha256", "request_nonce"):
            object.__setattr__(forged, name, getattr(source, name))
        operation = lambda: _replay(chain, event_plan=(forged,))
    else:
        operation = lambda: _replay(chain, expected_attempt_context_id=_EqAll())
    _expect_code(core.FAIL_INPUT_V1 if case != "eqall-pin" else core.FAIL_SCHEMA_V1, operation)


@pytest.mark.parametrize("case", ["leading-space", "trailing-lf", "numeric-spelling", "extra-field"])
def test_noncanonical_or_extended_request_wire_is_rejected(
    tmp_path: Path, case: str
) -> None:
    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 1)
    request = chain.requests[0]
    if case == "leading-space":
        malformed = b" " + request
    elif case == "trailing-lf":
        malformed = request + b"\n"
    elif case == "numeric-spelling":
        malformed = request.replace(b'"next_sequence":0', b'"next_sequence":0.0')
        assert malformed != request
    else:
        body = _json(request)
        body["extra"] = "forbidden"
        malformed = _canonical(body)
    expected_code = (
        core.FAIL_SCHEMA_V1 if case == "extra-field" else core.FAIL_CANONICAL_V1
    )
    if case == "trailing-lf":
        expected_code = core.FAIL_INPUT_V1
    _expect_code(
        expected_code,
        lambda: _replay(chain, requests=(malformed,)),
    )


def test_out_of_band_pins_require_exact_bytes_and_exact_values(
    tmp_path: Path,
) -> None:
    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 0)
    _expect_code(
        core.FAIL_INPUT_V1,
        lambda: _replay(chain, lock_id_pin=bytearray(chain.graph.lock_id.encode())),
    )
    _expect_code(
        core.FAIL_AUTHORITY_V1,
        lambda: _replay(chain, lock_sha_pin=b"\x00" * 32),
    )


def _result_snapshot(result: core.VerifiedLogicalTimeReplayCoreV1) -> tuple[object, ...]:
    return tuple(getattr(result, field.name) for field in fields(type(result)))


def _bomb(*args: object, **kwargs: object) -> object:
    raise AssertionError("rebound module symbol entered the captured verifier")


def test_public_verifier_result_does_not_drift_after_module_helper_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 3)
    baseline = _replay(chain)
    baseline_snapshot = _result_snapshot(baseline)
    original_result_type = core.VerifiedLogicalTimeReplayCoreV1
    for helper_name in (
        "_extract_logical_time_authority",
        "_parse_request",
        "_parse_token",
        "_parse_checkpoint",
        "_parse_head",
        "_request_projection_from_token",
        "_verify_token_signature",
        "_ledger_id",
        "_expected_head_mapping",
        "_exact_mapping_equal",
        "_event_row",
        "_canonical",
        "_sha256_bytes",
        "_fail",
    ):
        monkeypatch.setattr(core, helper_name, _bomb)
    for helper_name in (
        "load_step5_stage_b_trust_lock_v2",
        "load_step5_authority_profile_v2",
        "load_step5_signer_registry_v2",
        "require_locked_step5_authority_graph_v2",
        "canonical_step5_json_bytes_v2",
    ):
        monkeypatch.setattr(authority, helper_name, _bomb)
    rebound = _replay(chain)
    assert type(rebound) is original_result_type
    assert _result_snapshot(rebound) == baseline_snapshot
    assert core.require_verified_logical_time_replay_core_v1(rebound) is rebound


def test_core_authority_loader_contract_survives_exported_contract_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core must retain the authority contract that existed at import."""

    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 3)
    baseline = _result_snapshot(_replay(chain))

    scalar_replacements: dict[str, object] = {
        "AUTHORITY_PROFILE_SCHEMA_V2": "evil-authority-profile/999",
        "STAGE_B_TRUST_LOCK_SCHEMA_V2": "evil-stage-b-lock/999",
        "SIGNER_REGISTRY_SCHEMA_V2": "evil-registry/999",
        "DETACHED_SIGNATURE_SCHEMA_V2": "evil-signature/999",
        "SERVICE_STATIC_BINDING_SCHEMA_V2": "evil-service/999",
        "READINESS_CANDIDATE_SNAPSHOT_SCHEMA_V2": "evil-readiness/999",
        "ROLE_KEY_SCHEMA_V1": "evil-role-key/999",
        "AUTHORITY_PROFILE_PREFIX_V2": "evil_profile_",
        "STAGE_B_TRUST_LOCK_PREFIX_V2": "evil_lock_",
        "SIGNER_REGISTRY_PREFIX_V2": "evil_registry_",
        "SERVICE_STATIC_BINDING_PREFIX_V2": "evil_service_",
        "READINESS_CANDIDATE_PREFIX_V2": "evil_readiness_",
        "ROLE_KEY_PREFIX_V1": "evil_role_",
        "QV2_PROFILE_ID_V1": "evil_qv2_" + "f" * 64,
        "QV2_PROFILE_PAYLOAD_SHA256_V1": "f" * 64,
        "QV2_SOURCE_SHA256_V1": "e" * 64,
        "QV2_DEPENDENCY_LOCK_SHA256_V1": "d" * 64,
        "_MAX_CANONICAL_BYTES": 1,
        "_MAX_DEPTH": 0,
        "_MAX_NODES": 1,
        "_MAX_STRING_UTF8_BYTES": 1,
        "_SAFE_INTEGER": 0,
        "_PUBLIC_KEY_HEX_LENGTH": 1,
        "_SIGNATURE_HEX_LENGTH": 1,
        "_MAX_LAYERS": 0,
    }
    for name, replacement in scalar_replacements.items():
        monkeypatch.setattr(authority, name, replacement)

    monkeypatch.setattr(authority, "EXACT_QV2_BINDING_V1", {"evil": True})
    monkeypatch.setattr(authority, "ROLE_ORDER_V2", ("EVIL_ROLE",))
    monkeypatch.setattr(
        authority, "ROLE_CONTROL_DOMAINS_V2", {"EVIL_ROLE": "EVIL"}
    )
    monkeypatch.setattr(
        authority,
        "ROLE_SIGNATURE_PURPOSES_V2",
        {"EVIL_ROLE": ("SIGN_EVIL",)},
    )
    monkeypatch.setattr(authority, "SERVICE_ORDER_V2", ("evil_service",))
    monkeypatch.setattr(
        authority, "SERVICE_ROLE_V2", {"evil_service": "EVIL_ROLE"}
    )
    monkeypatch.setattr(
        authority, "SERVICE_TYPES_V2", {"evil_service": "EVIL"}
    )
    monkeypatch.setattr(
        authority, "SERVICE_UID_GID_V2", {"evil_service": (1, 1)}
    )
    monkeypatch.setattr(authority, "READINESS_NODE_ORDER_V1", ("EVIL_NODE",))
    monkeypatch.setattr(authority, "READINESS_DEPENDENCIES_V1", ((),))
    monkeypatch.setattr(authority, "_DOMAINS", {"authority_profile": b"EVIL\x00"})
    monkeypatch.setattr(authority, "_PREFIXES", {"authority_profile": "evil_"})
    monkeypatch.setattr(
        authority, "_SIGNATURE_DOMAINS", {"SIGN_EVIL": b"EVIL\x00"}
    )
    monkeypatch.setattr(authority, "_HEX64_RE", re.compile(r".+"))
    monkeypatch.setattr(authority, "_COMMIT_RE", re.compile(r".+"))
    monkeypatch.setattr(authority, "_OCI_DIGEST_RE", re.compile(r".+"))
    monkeypatch.setattr(authority, "_ID_RE", re.compile(r".+"))
    monkeypatch.setattr(authority, "canonicalize_rfc8785_v1", lambda value: b"{}")
    monkeypatch.setattr(
        authority,
        "decode_and_require_canonical_rfc8785_v1",
        lambda payload: {},
    )
    monkeypatch.setattr(authority, "_Ed25519PublicKey", object)
    monkeypatch.setattr(authority, "_InvalidSignature", RuntimeError)
    for name, value in tuple(vars(authority).items()):
        if (
            inspect.isfunction(value)
            and value.__module__ == authority.__name__
        ):
            monkeypatch.setattr(authority, name, _bomb)

    rebound = _replay(chain)
    assert _result_snapshot(rebound) == baseline
    assert core.require_verified_logical_time_replay_core_v1(rebound) is rebound


def test_core_rejects_fully_readdressed_resigned_replacement_authority_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Self-consistency under a substituted authority contract is insufficient."""

    graph = _build_candidate_graph()
    evil_profile_schema = "evil-authority-profile/999"
    evil_registry_schema = "evil-registry/999"
    evil_lock_schema = "evil-stage-b-lock/999"
    evil_signature_schema = "evil-signature/999"
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
    monkeypatch.setattr(authority, "DETACHED_SIGNATURE_SCHEMA_V2", evil_signature_schema)
    monkeypatch.setattr(authority, "EXACT_QV2_BINDING_V1", evil_qv2)

    profile_body = _json(graph.profile_payload)["body"]
    assert type(profile_body) is dict
    profile_body["schema_version"] = authority.AUTHORITY_PROFILE_SCHEMA_V2
    profile_body["qv2_binding"] = dict(authority.EXACT_QV2_BINDING_V1)
    profile_id = authority.phase2b_step5_content_id_v2(
        "authority_profile", profile_body
    )
    profile_payload = authority.canonical_step5_json_bytes_v2(
        {"body": profile_body, "id": profile_id}
    )

    registry_body = _json(graph.registry_payload)["body"]
    assert type(registry_body) is dict
    registry_body["schema_version"] = authority.SIGNER_REGISTRY_SCHEMA_V2
    registry_body["authority_profile_id"] = profile_id
    registry_id = authority.phase2b_step5_content_id_v2(
        "signer_registry", registry_body
    )
    role_keys = registry_body["role_keys_in_exact_order"]
    assert type(role_keys) is list and type(role_keys[0]) is dict
    governance_key_id = role_keys[0]["key_id"]
    assert type(governance_key_id) is str
    signature_preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry",
        body=registry_body,
        role="GOVERNANCE_ANCHOR",
        purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=governance_key_id,
    )
    governance_private_key = Ed25519PrivateKey.from_private_bytes(b"\x01" * 32)
    registry_payload = authority.canonical_step5_json_bytes_v2(
        {
            "body": registry_body,
            "id": registry_id,
            "signatures": [
                {
                    "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
                    "algorithm": "Ed25519",
                    "role": "GOVERNANCE_ANCHOR",
                    "purpose": "SIGN_SIGNER_REGISTRY",
                    "key_epoch": 1,
                    "signer_key_id": governance_key_id,
                    "signature_hex": governance_private_key.sign(
                        signature_preimage
                    ).hex(),
                }
            ],
        }
    )

    lock_body = _json(graph.lock_payload)["body"]
    assert type(lock_body) is dict
    lock_body["schema_version"] = authority.STAGE_B_TRUST_LOCK_SCHEMA_V2
    lock_body["authority_profile_schema"] = authority.AUTHORITY_PROFILE_SCHEMA_V2
    lock_body["authority_profile_id"] = profile_id
    lock_body["authority_profile_payload_sha256"] = hashlib.sha256(
        profile_payload
    ).hexdigest()
    lock_body["signer_registry_schema"] = authority.SIGNER_REGISTRY_SCHEMA_V2
    lock_body["signer_registry_id"] = registry_id
    lock_body["signer_registry_payload_sha256"] = hashlib.sha256(
        registry_payload
    ).hexdigest()
    lock_body["qv2_binding"] = dict(authority.EXACT_QV2_BINDING_V1)
    lock_id = authority.phase2b_step5_content_id_v2(
        "stage_b_trust_lock", lock_body
    )
    lock_payload = authority.canonical_step5_json_bytes_v2(
        {"body": lock_body, "id": lock_id}
    )
    evil_graph = CandidateGraph(
        logical_time_private_key=graph.logical_time_private_key,
        logical_time_public_key=graph.logical_time_public_key,
        logical_time_key_id=graph.logical_time_key_id,
        profile_id=profile_id,
        registry_id=registry_id,
        profile_payload=profile_payload,
        registry_payload=registry_payload,
        lock_id=lock_id,
        lock_payload=lock_payload,
    )
    evil_chain = _build_raw_chain(tmp_path / "evil-ledger", evil_graph, 1)

    _expect_code(core.FAIL_AUTHORITY_V1, lambda: _replay(evil_chain))


def test_private_result_constructor_clone_and_mutation_are_rejected(
    tmp_path: Path,
) -> None:
    chain = _build_raw_chain(tmp_path / "ledger", _build_candidate_graph(), 1)
    result = _replay(chain)
    assert not hasattr(core, "_hidden_issue_verified_logical_time_replay_core_v1")
    with pytest.raises(TypeError, match="privately issued"):
        core.VerifiedLogicalTimeReplayCoreV1()
    with pytest.raises(TypeError, match="privately issued"):
        core.VerifiedLogicalTimeReplayCoreV1(object())

    forged = object.__new__(core.VerifiedLogicalTimeReplayCoreV1)
    for field in fields(type(result)):
        object.__setattr__(forged, field.name, getattr(result, field.name))
    _expect_code(
        core.FAIL_RESULT_V1,
        lambda: core.require_verified_logical_time_replay_core_v1(forged),
    )

    object.__setattr__(result, "authority_evidence", True)
    _expect_code(
        core.FAIL_RESULT_V1,
        lambda: core.require_verified_logical_time_replay_core_v1(result),
    )


def test_clean_import_excludes_service_store_owner_runtime_auditor_and_qv2_modules() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    script = """
import json
import sys
import hegel_machine.phase2b_formal_wire_covert_step5_logical_time_replay_core_v1
forbidden = (
    'phase2b_formal_wire_covert_step5_logical_time_v1',
    'phase2b_formal_wire_covert_step5_store',
    'phase2b_formal_wire_covert_step5_owner',
    'phase2b_formal_wire_covert_step5_runtime',
    'auditor',
    'qualification_v2',
    '.qv2',
)
loaded = sorted(
    name for name in sys.modules
    if name.startswith('hegel_machine.')
    and any(marker in name.lower() for marker in forbidden)
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
