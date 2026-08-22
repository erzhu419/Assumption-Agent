from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import inspect
import re
from typing import Callable

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import hegel_machine.phase2b_formal_wire_covert_step5_authority_v2 as authority


@dataclass(frozen=True)
class _CandidateGraph:
    role_keys: list[dict[str, object]]
    service_bindings: list[dict[str, object]]
    profile_body: dict[str, object]
    profile_payload: bytes
    registry_body: dict[str, object]
    registry_payload: bytes
    lock_body: dict[str, object]
    lock_id: str
    lock_payload: bytes


def _all_false_readiness() -> dict[str, object]:
    return {
        "node_states_in_canonical_topological_order": [
            {"node_id": node, "satisfied": False}
            for node in authority.READINESS_NODE_ORDER_V1
        ],
        "satisfied_node_count": 0,
        "highest_contiguous_satisfied_node": None,
        "all_nodes_satisfied": False,
        "readiness_evidence_overlay_issued": False,
    }


def _refresh_service_binding_id(binding: dict[str, object]) -> None:
    binding.pop("binding_id", None)
    binding["binding_id"] = (
        authority.phase2b_step5_service_static_binding_id_v2(binding)
    )


def _build_candidate_graph(
    *,
    salt: int = 0,
    profile_mutator: Callable[[dict[str, object]], None] | None = None,
    registry_mutator: Callable[[dict[str, object]], None] | None = None,
    signature_hex_override: str | None = None,
) -> _CandidateGraph:
    private_keys: list[Ed25519PrivateKey] = []
    role_keys: list[dict[str, object]] = []
    for index, role in enumerate(authority.ROLE_ORDER_V2, start=1):
        private_key = Ed25519PrivateKey.from_private_bytes(
            bytes([index + salt]) * 32
        )
        private_keys.append(private_key)
        row: dict[str, object] = {
            "schema_version": authority.ROLE_KEY_SCHEMA_V1,
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "authorized_signature_purposes": list(
                authority.ROLE_SIGNATURE_PURPOSES_V2[role]
            ),
            "algorithm": "Ed25519",
            "key_epoch": 1,
            "public_key_hex": private_key.public_key()
            .public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
            .hex(),
        }
        row["key_id"] = authority.phase2b_step5_role_key_id_v2(row)
        role_keys.append(row)

    service_bindings: list[dict[str, object]] = []
    for index, service in enumerate(authority.SERVICE_ORDER_V2, start=1):
        role = authority.SERVICE_ROLE_V2[service]
        key = role_keys[authority.ROLE_ORDER_V2.index(role)]
        uid, gid = authority.SERVICE_UID_GID_V2[service]

        def oci(offset: int) -> str:
            return "sha256:" + f"{salt * 1000 + index * 10 + offset:064x}"

        binding: dict[str, object] = {
            "schema_version": authority.SERVICE_STATIC_BINDING_SCHEMA_V2,
            "service_name": service,
            "service_type": authority.SERVICE_TYPES_V2[service],
            "binding_status": "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE",
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "uid": uid,
            "gid": gid,
            "role_key_id": key["key_id"],
            "key_epoch": 1,
            "authorized_signature_purposes": list(
                authority.ROLE_SIGNATURE_PURPOSES_V2[role]
            ),
            "oci_index_digest": oci(1),
            "oci_manifest_digest": oci(2),
            "oci_config_digest": oci(3),
            "ordered_layer_digests": [oci(4), oci(5)],
            "static_runtime_closure_policy_sha256": f"{100 + salt * 10 + index:064x}",
        }
        _refresh_service_binding_id(binding)
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
    if profile_mutator is not None:
        profile_mutator(profile_body)
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
    if registry_mutator is not None:
        registry_mutator(registry_body)
    registry_id = authority.phase2b_step5_content_id_v2(
        "signer_registry", registry_body
    )
    signature_preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry",
        body=registry_body,
        role="GOVERNANCE_ANCHOR",
        purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=role_keys[0]["key_id"],  # type: ignore[arg-type]
    )
    signature = {
        "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
        "algorithm": "Ed25519",
        "role": "GOVERNANCE_ANCHOR",
        "purpose": "SIGN_SIGNER_REGISTRY",
        "key_epoch": 1,
        "signer_key_id": role_keys[0]["key_id"],
        "signature_hex": (
            signature_hex_override
            if signature_hex_override is not None
            else private_keys[0].sign(signature_preimage).hex()
        ),
    }
    registry_payload = authority.canonical_step5_json_bytes_v2(
        {
            "body": registry_body,
            "id": registry_id,
            "signatures": [signature],
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
    return _CandidateGraph(
        role_keys=role_keys,
        service_bindings=service_bindings,
        profile_body=profile_body,
        profile_payload=profile_payload,
        registry_body=registry_body,
        registry_payload=registry_payload,
        lock_body=lock_body,
        lock_id=lock_id,
        lock_payload=lock_payload,
    )


def _load_lock(graph: _CandidateGraph):
    return authority.load_step5_stage_b_trust_lock_v2(
        graph.lock_payload,
        out_of_band_expected_lock_id_bytes=graph.lock_id.encode("ascii"),
        out_of_band_expected_payload_sha256_digest=hashlib.sha256(
            graph.lock_payload
        ).digest(),
    )


def _load_complete(graph: _CandidateGraph):
    lock = _load_lock(graph)
    profile = authority.load_step5_authority_profile_v2(
        graph.profile_payload, lock
    )
    registry = authority.load_step5_signer_registry_v2(
        graph.registry_payload, lock, profile
    )
    return lock, profile, registry


def _readdress_and_resign_graph(
    graph: _CandidateGraph,
    *,
    governance_private_key: Ed25519PrivateKey,
) -> _CandidateGraph:
    """Rebuild every dependent address, signature, digest, and OOB lock pin."""

    profile_body = copy.deepcopy(graph.profile_body)
    profile_id = authority.phase2b_step5_content_id_v2(
        "authority_profile", profile_body
    )
    profile_payload = authority.canonical_step5_json_bytes_v2(
        {"body": profile_body, "id": profile_id}
    )

    registry_body = copy.deepcopy(graph.registry_body)
    registry_body["authority_profile_id"] = profile_id
    registry_id = authority.phase2b_step5_content_id_v2(
        "signer_registry", registry_body
    )
    governance_key = registry_body["role_keys_in_exact_order"][0]
    signature_preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry",
        body=registry_body,
        role="GOVERNANCE_ANCHOR",
        purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=governance_key["key_id"],
    )
    signature = {
        "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
        "algorithm": "Ed25519",
        "role": "GOVERNANCE_ANCHOR",
        "purpose": "SIGN_SIGNER_REGISTRY",
        "key_epoch": 1,
        "signer_key_id": governance_key["key_id"],
        "signature_hex": governance_private_key.sign(signature_preimage).hex(),
    }
    registry_payload = authority.canonical_step5_json_bytes_v2(
        {
            "body": registry_body,
            "id": registry_id,
            "signatures": [signature],
        }
    )

    lock_body = copy.deepcopy(graph.lock_body)
    lock_body["authority_profile_schema"] = profile_body["schema_version"]
    lock_body["authority_profile_id"] = profile_id
    lock_body["authority_profile_payload_sha256"] = hashlib.sha256(
        profile_payload
    ).hexdigest()
    lock_body["signer_registry_schema"] = registry_body["schema_version"]
    lock_body["signer_registry_id"] = registry_id
    lock_body["signer_registry_payload_sha256"] = hashlib.sha256(
        registry_payload
    ).hexdigest()
    lock_id = authority.phase2b_step5_content_id_v2(
        "stage_b_trust_lock", lock_body
    )
    lock_payload = authority.canonical_step5_json_bytes_v2(
        {"body": lock_body, "id": lock_id}
    )
    return _CandidateGraph(
        role_keys=registry_body["role_keys_in_exact_order"],
        service_bindings=profile_body[
            "service_static_bindings_in_exact_order"
        ],
        profile_body=profile_body,
        profile_payload=profile_payload,
        registry_body=registry_body,
        registry_payload=registry_payload,
        lock_body=lock_body,
        lock_id=lock_id,
        lock_payload=lock_payload,
    )


def _assert_error_code(expected: str, operation: Callable[[], object]) -> None:
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV2) as caught:
        operation()
    assert caught.value.code == expected


def test_valid_exact_bytes_graph_closes_with_five_typed_services() -> None:
    graph = _build_candidate_graph()
    lock, profile, registry = _load_complete(graph)
    closed = authority.require_locked_step5_authority_graph_v2(
        lock, profile, registry
    )
    assert closed == (lock, profile, registry)
    assert registry.signature_backends == ("CRYPTOGRAPHY_ED25519",)
    services = profile.body["service_static_bindings_in_exact_order"]
    assert type(services) is tuple
    assert tuple(row["service_name"] for row in services) == authority.SERVICE_ORDER_V2
    assert tuple(row["service_type"] for row in services) == tuple(
        authority.SERVICE_TYPES_V2[name] for name in authority.SERVICE_ORDER_V2
    )


def test_exact_eight_roles_and_purpose_cascade_are_frozen() -> None:
    assert len(authority.ROLE_ORDER_V2) == 8
    assert len(set(authority.ROLE_ORDER_V2)) == 8
    assert authority.ROLE_SIGNATURE_PURPOSES_V2["GOVERNANCE_ANCHOR"] == (
        "SIGN_SIGNER_REGISTRY",
        "SIGN_OWNER_AUTHORIZATION_STATUS_EVENT",
    )
    assert authority.ROLE_SIGNATURE_PURPOSES_V2[
        "SOURCE_RUNTIME_CUSTODIAN"
    ] == (
        "SIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
        "SIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
        "SIGN_RUNTIME_CLOSURE",
        "SIGN_R01_AUTHORIZATION_STATUS_RECEIPT",
    )
    assert authority.ROLE_SIGNATURE_PURPOSES_V2["PURE_REPLAYER"] == (
        "COUNTERSIGN_R01_AUTHORIZATION_STATUS_CANDIDATE",
        "COUNTERSIGN_R02_SYNTHETIC_BRANCH_CANDIDATE",
        "SIGN_PURE_REPLAY_RESULT",
        "COUNTERSIGN_R01_AUTHORIZATION_STATUS_RECEIPT",
        "SIGN_OWNER_STATUS_ANTI_FORK_ANCHOR",
    )


def test_five_service_identity_dimensions_are_globally_distinct() -> None:
    graph = _build_candidate_graph()
    bindings = graph.service_bindings
    for field in (
        "binding_id",
        "role_key_id",
        "uid",
        "gid",
        "oci_index_digest",
        "oci_manifest_digest",
        "oci_config_digest",
    ):
        assert len({row[field] for row in bindings}) == 5
    all_unix_ids = [row["uid"] for row in bindings] + [
        row["gid"] for row in bindings
    ]
    assert len(set(all_unix_ids)) == 10
    assert len(
        {
            tuple(row["authorized_signature_purposes"])  # type: ignore[arg-type]
            for row in bindings
        }
    ) == 5


@pytest.mark.parametrize(
    ("lock_id_pin", "payload_sha_pin"),
    [
        (lambda graph: graph.lock_id, lambda graph: hashlib.sha256(graph.lock_payload).digest()),
        (lambda graph: bytearray(graph.lock_id.encode()), lambda graph: hashlib.sha256(graph.lock_payload).digest()),
        (lambda graph: graph.lock_id.encode(), lambda graph: hashlib.sha256(graph.lock_payload).hexdigest()),
        (lambda graph: graph.lock_id.encode(), lambda graph: b"\x00" * 32),
        (
            lambda graph: (
                graph.lock_id[:-1] + ("0" if graph.lock_id[-1] != "0" else "1")
            ).encode(),
            lambda graph: hashlib.sha256(graph.lock_payload).digest(),
        ),
    ],
)
def test_out_of_band_pins_require_exact_independent_raw_bytes(
    lock_id_pin: Callable[[_CandidateGraph], object],
    payload_sha_pin: Callable[[_CandidateGraph], object],
) -> None:
    graph = _build_candidate_graph()
    _assert_error_code(
        authority.FAIL_INPUT_V2
        if not isinstance(lock_id_pin(graph), bytes)
        or not isinstance(payload_sha_pin(graph), bytes)
        or payload_sha_pin(graph) == b"\x00" * 32
        else authority.FAIL_TRUST_ANCHOR_V2,
        lambda: authority.load_step5_stage_b_trust_lock_v2(
            graph.lock_payload,
            out_of_band_expected_lock_id_bytes=lock_id_pin(graph),  # type: ignore[arg-type]
            out_of_band_expected_payload_sha256_digest=payload_sha_pin(graph),  # type: ignore[arg-type]
        ),
    )


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1}\n',
        b'{"a":1} ',
        b'{ "a":1}',
        b'{"a":1,"a":1}',
        b'{"a":1.5}',
    ],
)
def test_canonical_decoder_rejects_trailing_duplicate_and_float_forms(
    payload: bytes,
) -> None:
    with pytest.raises(authority.Phase2BStep5AuthorityErrorV2):
        authority.decode_canonical_step5_json_v2(payload)


def test_qv2_binding_is_exact_qualification_slash_nine() -> None:
    assert authority.EXACT_QV2_BINDING_V1["schema_version"] == (
        "hegel-machine-phase2b-formal-wire-covert-qualification/9"
    )
    graph = _build_candidate_graph(
        profile_mutator=lambda profile: profile["qv2_binding"].__setitem__(  # type: ignore[union-attr]
            "source_sha256", "f" * 64
        )
    )
    lock = _load_lock(graph)
    _assert_error_code(
        authority.FAIL_BINDING_V2,
        lambda: authority.load_step5_authority_profile_v2(
            graph.profile_payload, lock
        ),
    )


@pytest.mark.parametrize("bad_uid", [True, 41001.0, "41001"])
def test_service_uid_rejects_bool_float_and_text(bad_uid: object) -> None:
    graph = _build_candidate_graph()
    binding = copy.deepcopy(graph.service_bindings[0])
    binding["uid"] = bad_uid
    _assert_error_code(
        authority.FAIL_SCHEMA_V2,
        lambda: authority._validate_service_static_binding(binding, "runner"),
    )


def test_eqall_and_str_subclass_cannot_launder_role_key_schema() -> None:
    class EqAll:
        def __eq__(self, other: object) -> bool:
            return True

    class StrSubclass(str):
        pass

    graph = _build_candidate_graph()
    for field, value in (
        ("key_epoch", EqAll()),
        ("role", StrSubclass("GOVERNANCE_ANCHOR")),
    ):
        row = copy.deepcopy(graph.role_keys[0])
        row[field] = value
        with pytest.raises(authority.Phase2BStep5AuthorityErrorV2):
            authority._validate_role_key(row, "GOVERNANCE_ANCHOR")


def test_ordered_oci_layers_are_content_bound_and_duplicates_rejected() -> None:
    graph = _build_candidate_graph()
    original = copy.deepcopy(graph.service_bindings[0])
    reordered = copy.deepcopy(original)
    reordered["ordered_layer_digests"].reverse()  # type: ignore[union-attr]
    assert reordered["binding_id"] != authority.phase2b_step5_service_static_binding_id_v2(
        {key: value for key, value in reordered.items() if key != "binding_id"}
    )
    duplicate = copy.deepcopy(original)
    layers = duplicate["ordered_layer_digests"]
    assert isinstance(layers, list)
    layers[1] = layers[0]
    _refresh_service_binding_id(duplicate)
    _assert_error_code(
        authority.FAIL_SCHEMA_V2,
        lambda: authority._validate_service_static_binding(duplicate, "runner"),
    )


def test_cross_service_oci_graph_collision_is_rejected() -> None:
    graph = _build_candidate_graph()
    bindings = copy.deepcopy(graph.service_bindings)
    bindings[1]["oci_index_digest"] = bindings[0]["oci_index_digest"]
    _refresh_service_binding_id(bindings[1])
    _assert_error_code(
        authority.FAIL_SCHEMA_V2,
        lambda: authority._validate_service_static_bindings(bindings),
    )


def test_service_to_registry_role_key_lineage_is_complete() -> None:
    def mutate(profile: dict[str, object]) -> None:
        services = profile["service_static_bindings_in_exact_order"]
        assert isinstance(services, list)
        services[0]["role_key_id"] = authority.ROLE_KEY_PREFIX_V1 + "f" * 64
        _refresh_service_binding_id(services[0])

    graph = _build_candidate_graph(profile_mutator=mutate)
    lock = _load_lock(graph)
    profile = authority.load_step5_authority_profile_v2(
        graph.profile_payload, lock
    )
    _assert_error_code(
        authority.FAIL_BINDING_V2,
        lambda: authority.load_step5_signer_registry_v2(
            graph.registry_payload, lock, profile
        ),
    )


def test_invalid_governance_signature_is_rejected() -> None:
    graph = _build_candidate_graph(signature_hex_override="00" * 64)
    lock = _load_lock(graph)
    profile = authority.load_step5_authority_profile_v2(
        graph.profile_payload, lock
    )
    _assert_error_code(
        authority.FAIL_SIGNATURE_V2,
        lambda: authority.load_step5_signer_registry_v2(
            graph.registry_payload, lock, profile
        ),
    )


def test_invalid_signature_survives_no_global_verifier_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_candidate_graph(signature_hex_override="00" * 64)
    lock = _load_lock(graph)
    profile = authority.load_step5_authority_profile_v2(
        graph.profile_payload, lock
    )
    monkeypatch.setattr(
        authority,
        "_verify_detached_signature",
        lambda **kwargs: "CRYPTOGRAPHY_ED25519",
    )
    monkeypatch.setattr(authority, "_Ed25519PublicKey", object)
    monkeypatch.setattr(
        authority,
        "_validate_detached_signature",
        lambda value, role, purpose, key: value,
    )
    _assert_error_code(
        authority.FAIL_SIGNATURE_V2,
        lambda: authority.load_step5_signer_registry_v2(
            graph.registry_payload, lock, profile
        ),
    )


def test_malformed_profile_survives_no_validator_global_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mutate(profile: dict[str, object]) -> None:
        services = profile["service_static_bindings_in_exact_order"]
        assert isinstance(services, list)
        services[0]["uid"] = True
        _refresh_service_binding_id(services[0])

    graph = _build_candidate_graph(profile_mutator=mutate)
    monkeypatch.setattr(
        authority, "_validate_service_static_bindings", lambda value: value
    )
    monkeypatch.setattr(
        authority, "_validate_authority_profile_body", lambda value: value
    )
    monkeypatch.setattr(authority, "_snapshot", lambda value, **kwargs: value)
    monkeypatch.setattr(authority, "_object", lambda value, fields, name: value)
    monkeypatch.setattr(authority, "_exact_equal", lambda left, right: True)
    monkeypatch.setattr(
        authority, "canonical_step5_json_bytes_v2", lambda value: b"{}"
    )
    monkeypatch.setattr(
        authority, "decode_canonical_step5_json_v2", lambda payload: {}
    )
    lock = _load_lock(graph)
    _assert_error_code(
        authority.FAIL_SCHEMA_V2,
        lambda: authority.load_step5_authority_profile_v2(
            graph.profile_payload, lock
        ),
    )


def test_raw_loader_contract_survives_all_exported_contract_and_helper_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_candidate_graph()
    expected_profile_id = authority.phase2b_step5_content_id_v2(
        "authority_profile", graph.profile_body
    )
    expected_preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry",
        body=graph.registry_body,
        role="GOVERNANCE_ANCHOR",
        purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=graph.role_keys[0]["key_id"],  # type: ignore[arg-type]
    )

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
    monkeypatch.setattr(authority, "READINESS_DEPENDENCIES_V1", ((),))
    monkeypatch.setattr(authority, "_DOMAINS", {"authority_profile": b"EVIL\x00"})
    monkeypatch.setattr(authority, "_PREFIXES", {"authority_profile": "evil_"})
    monkeypatch.setattr(authority, "_SIGNATURE_DOMAINS", {"SIGN_EVIL": b"EVIL\x00"})
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
            name.startswith("_")
            and inspect.isfunction(value)
            and value.__module__ == authority.__name__
        ):
            monkeypatch.setattr(authority, name, lambda *args, **kwargs: None)

    lock, profile, registry = _load_complete(graph)
    assert profile.artifact_id == expected_profile_id
    assert registry.signature_backends == ("CRYPTOGRAPHY_ED25519",)
    assert authority.phase2b_step5_content_id_v2(
        "authority_profile", graph.profile_body
    ) == expected_profile_id
    assert authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry",
        body=graph.registry_body,
        role="GOVERNANCE_ANCHOR",
        purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=graph.role_keys[0]["key_id"],  # type: ignore[arg-type]
    ) == expected_preimage
    assert authority.require_locked_step5_authority_graph_v2(
        lock, profile, registry
    ) == (lock, profile, registry)


def test_fully_readdressed_resigned_replacement_schema_graph_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _build_candidate_graph()
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

    graph.profile_body["schema_version"] = authority.AUTHORITY_PROFILE_SCHEMA_V2
    graph.profile_body["qv2_binding"] = dict(authority.EXACT_QV2_BINDING_V1)
    graph.registry_body["schema_version"] = authority.SIGNER_REGISTRY_SCHEMA_V2
    graph.lock_body["schema_version"] = authority.STAGE_B_TRUST_LOCK_SCHEMA_V2
    graph.lock_body["qv2_binding"] = dict(authority.EXACT_QV2_BINDING_V1)
    evil = _readdress_and_resign_graph(
        graph,
        governance_private_key=Ed25519PrivateKey.from_private_bytes(b"\x01" * 32),
    )

    _assert_error_code(authority.FAIL_TRUST_ANCHOR_V2, lambda: _load_lock(evil))


def test_no_module_global_issuance_capability_and_constructor_always_rejects() -> None:
    assert not hasattr(authority, "_RESULT_TOKEN")
    assert not hasattr(authority, "_issue")
    assert not hasattr(authority, "_hidden_issue_verified_artifact_v2")
    with pytest.raises(TypeError):
        authority.Phase2BStep5VerifiedArtifactV2()
    with pytest.raises(TypeError):
        authority.Phase2BStep5VerifiedArtifactV2(object())


def test_object_new_field_copy_cannot_forge_verified_lineage() -> None:
    lock, profile, registry = _load_complete(_build_candidate_graph())
    forged = object.__new__(authority.Phase2BStep5VerifiedArtifactV2)
    for name in ("kind", "artifact_id", "body", "signature_backends"):
        object.__setattr__(forged, name, getattr(lock, name))
    _assert_error_code(
        authority.FAIL_INPUT_V2,
        lambda: authority.require_locked_step5_authority_graph_v2(
            forged, profile, registry
        ),
    )


def test_mutating_genuine_token_is_detected_by_private_identity_record() -> None:
    lock, profile, registry = _load_complete(_build_candidate_graph())
    object.__setattr__(lock, "artifact_id", authority.STAGE_B_TRUST_LOCK_PREFIX_V2 + "f" * 64)
    _assert_error_code(
        authority.FAIL_INPUT_V2,
        lambda: authority.require_locked_step5_authority_graph_v2(
            lock, profile, registry
        ),
    )


def test_cross_lock_profile_registry_tokens_cannot_be_mixed() -> None:
    first = _load_complete(_build_candidate_graph(salt=0))
    second = _load_complete(_build_candidate_graph(salt=10))
    _assert_error_code(
        authority.FAIL_TRUST_ANCHOR_V2,
        lambda: authority.require_locked_step5_authority_graph_v2(
            first[0], second[1], second[2]
        ),
    )


def test_public_entrypoints_expose_no_injected_verifier_parameters() -> None:
    for function in (
        authority.load_step5_stage_b_trust_lock_v2,
        authority.load_step5_authority_profile_v2,
        authority.load_step5_signer_registry_v2,
        authority.require_locked_step5_authority_graph_v2,
        authority.phase2b_step5_content_id_v2,
        authority.phase2b_step5_role_key_id_v2,
        authority.phase2b_step5_service_static_binding_id_v2,
        authority.phase2b_step5_signature_preimage_v2,
    ):
        assert all(
            not name.startswith("_")
            for name in inspect.signature(function).parameters
        )


def test_profile_and_registry_decoded_object_verifiers_fail_closed() -> None:
    for operation in (
        lambda: authority.verify_step5_authority_profile_v2({}),
        lambda: authority.verify_step5_signer_registry_v2({}),
    ):
        _assert_error_code(authority.FAIL_TRUST_ANCHOR_V2, operation)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: authority.verify_step5_runtime_closure_v2(),
        lambda: authority.verify_step5_r01_authorization_status_receipt_v2(),
        lambda: authority.derive_step5_r01_readiness_overlay_v2(),
    ],
)
def test_real_runtime_r01_and_overlay_entrypoints_remain_not_implemented(
    operation: Callable[[], object],
) -> None:
    _assert_error_code(authority.FAIL_NOT_IMPLEMENTED_V2, operation)


def test_profile_registry_and_all_sixteen_rows_remain_false() -> None:
    _, profile, registry = _load_complete(_build_candidate_graph())
    for artifact in (profile, registry):
        state = artifact.body["readiness_state"]
        assert state["satisfied_node_count"] == 0  # type: ignore[index]
        assert state["all_nodes_satisfied"] is False  # type: ignore[index]
        rows = state["node_states_in_canonical_topological_order"]  # type: ignore[index]
        assert len(rows) == 16
        assert all(row["satisfied"] is False for row in rows)
    assert all(value is False for value in profile.body["permanent_nonclaims"].values())  # type: ignore[union-attr]


def test_signature_helper_rejects_wrong_role_purpose_and_type_confusion() -> None:
    graph = _build_candidate_graph()
    for role, purpose, epoch in (
        ("GOVERNANCE_ANCHOR", "SIGN_RUNTIME_CLOSURE", 1),
        ("GOVERNANCE_ANCHOR", "SIGN_SIGNER_REGISTRY", True),
        ("GOVERNANCE_ANCHOR", "SIGN_SIGNER_REGISTRY", 1.0),
    ):
        _assert_error_code(
            authority.FAIL_INPUT_V2,
            lambda role=role, purpose=purpose, epoch=epoch: authority.phase2b_step5_signature_preimage_v2(
                artifact_kind="signer_registry",
                body=graph.registry_body,
                role=role,
                purpose=purpose,
                signer_key_id=graph.role_keys[0]["key_id"],  # type: ignore[arg-type]
                key_epoch=epoch,  # type: ignore[arg-type]
            ),
        )
