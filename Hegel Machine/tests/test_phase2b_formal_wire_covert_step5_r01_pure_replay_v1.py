from __future__ import annotations

import ast
from dataclasses import dataclass, fields, replace
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import phase2b_formal_wire_covert_step5_logical_time_replay_core_v1 as time_core
from hegel_machine import phase2b_formal_wire_covert_step5_logical_time_v1 as logical_time
from hegel_machine import phase2b_formal_wire_covert_step5_owner_status_replay_core_v1 as owner_core
from hegel_machine import phase2b_formal_wire_covert_step5_owner_status_v1 as owner_status
from hegel_machine import phase2b_formal_wire_covert_step5_r01_pure_replay_v1 as r01
from hegel_machine import phase2b_formal_wire_covert_step5_runtime_closure_v1 as closure
from hegel_machine import phase2b_formal_wire_covert_step5_store_replay_core_v2 as store_core
from hegel_machine import phase2b_formal_wire_covert_step5_store_v2 as store
from hegel_machine.rfc8785_v1 import canonicalize_rfc8785_v1

# Producer fixtures are used only to make real wire bytes.  The separate clean
# import KAT below imports only the replay composer and its verifier-only graph.
import test_phase2b_formal_wire_covert_step5_runtime_closure_v1 as runtime_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OWNER_CONFIG = PROJECT_ROOT / "config" / "phase2b_c1_owner_actual_unsealed_960_authorization_v1.json"
REPORT_DOMAIN = b"HEGEL/PHASE2B/STEP5/RUNTIME_CLOSURE_REPORT/V1\x00"
BUNDLE_DOMAIN = b"HEGEL/PHASE2B/STEP5/RUNTIME_CLOSURE_REPORT_BUNDLE/V1\x00"
DIGEST_ROOT_DOMAIN = b"HEGEL/PHASE2B/STEP5/RUNTIME_CLOSURE_REPORT/ORDERED_DIGEST_ROOT/V1\x00"


def _json(payload: bytes) -> dict[str, object]:
    value = json.loads(payload)
    assert type(value) is dict
    return value


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _address(payload: bytes) -> str:
    return "sha256:" + _sha(payload)


def _content_id(prefix: str, domain: bytes, body: object) -> str:
    return prefix + hashlib.sha256(domain + canonicalize_rfc8785_v1(body)).hexdigest()


def _root(values: tuple[str, ...]) -> str:
    return hashlib.sha256(DIGEST_ROOT_DOMAIN + canonicalize_rfc8785_v1(list(values))).hexdigest()


@dataclass(frozen=True)
class RuntimeRaw:
    service_name: str
    policy: bytes
    evidence: bytes
    index: bytes
    manifest: bytes
    config: bytes
    layers: tuple[tuple[bytes, ...], ...]
    supports: tuple[tuple[bytes, ...], ...]


@dataclass(frozen=True)
class AuthorityRuntimeGraph:
    lock: bytes
    profile: bytes
    registry: bytes
    lock_id: str
    profile_id: str
    registry_id: str
    oob_id: bytes
    oob_digest: bytes
    runtime_raw: tuple[RuntimeRaw, ...]
    role_rows: tuple[dict[str, object], ...]


def _build_authority_and_five_runtime_graphs(*, salt: int = 0) -> AuthorityRuntimeGraph:
    bases = tuple(
        runtime_fixture._build_runtime_graph(
            config_mutator=lambda config, ordinal=ordinal: config.update(
                {"created": f"2026-08-22T00:{salt:02d}:0{ordinal}Z"}
            )
        )
        for ordinal in range(len(authority.SERVICE_ORDER_V2))
    )
    base = bases[0]
    args = base.verify_arguments
    registry0 = _json(args["signer_registry_payload"])  # type: ignore[arg-type]
    role_rows = tuple(copy.deepcopy(registry0["body"]["role_keys_in_exact_order"]))  # type: ignore[index]

    policies: list[bytes] = []
    policy_ids: list[str] = []
    for ordinal, service_name in enumerate(authority.SERVICE_ORDER_V2):
        role = authority.SERVICE_ROLE_V2[service_name]
        row = role_rows[authority.ROLE_ORDER_V2.index(role)]
        uid, gid = authority.SERVICE_UID_GID_V2[service_name]
        body = copy.deepcopy(bases[ordinal].policy_body)
        body.update(
            service_name=service_name,
            service_type=authority.SERVICE_TYPES_V2[service_name],
            role=role,
            control_domain=authority.ROLE_CONTROL_DOMAINS_V2[role],
            uid=uid,
            gid=gid,
            role_key_id=row["key_id"],
        )
        policy_id = closure.phase2b_step5_runtime_closure_policy_id_v1(body)
        policies.append(closure.canonical_runtime_closure_json_bytes_v1({"body": body, "id": policy_id}))
        policy_ids.append(policy_id)

    service_bindings: list[dict[str, object]] = []
    for ordinal, service_name in enumerate(authority.SERVICE_ORDER_V2):
        oci = bases[ordinal].evidence_body["oci_image"]
        assert type(oci) is dict
        role = authority.SERVICE_ROLE_V2[service_name]
        row = role_rows[authority.ROLE_ORDER_V2.index(role)]
        uid, gid = authority.SERVICE_UID_GID_V2[service_name]
        binding: dict[str, object] = {
            "schema_version": authority.SERVICE_STATIC_BINDING_SCHEMA_V2,
            "service_name": service_name,
            "service_type": authority.SERVICE_TYPES_V2[service_name],
            "binding_status": "STATIC_REQUESTED_BINDING_NOT_OBSERVED_LIVE",
            "role": role,
            "control_domain": authority.ROLE_CONTROL_DOMAINS_V2[role],
            "uid": uid,
            "gid": gid,
            "role_key_id": row["key_id"],
            "key_epoch": 1,
            "authorized_signature_purposes": list(authority.ROLE_SIGNATURE_PURPOSES_V2[role]),
            "oci_index_digest": oci["index_digest"],
            "oci_manifest_digest": oci["manifest_digest"],
            "oci_config_digest": oci["config_digest"],
            "ordered_layer_digests": [x["digest"] for x in oci["ordered_layer_descriptors"]],  # type: ignore[index]
            "static_runtime_closure_policy_sha256": _sha(policies[ordinal]),
        }
        binding["binding_id"] = authority.phase2b_step5_service_static_binding_id_v2(binding)
        service_bindings.append(binding)

    profile0 = _json(args["authority_profile_payload"])  # type: ignore[arg-type]
    profile_body = copy.deepcopy(profile0["body"])
    profile_body["service_static_bindings_in_exact_order"] = service_bindings  # type: ignore[index]
    profile_id = authority.phase2b_step5_content_id_v2("authority_profile", profile_body)
    profile = authority.canonical_step5_json_bytes_v2({"body": profile_body, "id": profile_id})

    registry_body = copy.deepcopy(registry0["body"])
    registry_body["authority_profile_id"] = profile_id  # type: ignore[index]
    registry_id = authority.phase2b_step5_content_id_v2("signer_registry", registry_body)
    governance = role_rows[0]
    preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind="signer_registry", body=registry_body,
        role="GOVERNANCE_ANCHOR", purpose="SIGN_SIGNER_REGISTRY",
        signer_key_id=governance["key_id"],
    )
    registry = authority.canonical_step5_json_bytes_v2({
        "body": registry_body, "id": registry_id,
        "signatures": [{
            "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
            "algorithm": "Ed25519", "role": "GOVERNANCE_ANCHOR",
            "purpose": "SIGN_SIGNER_REGISTRY", "key_epoch": 1,
            "signer_key_id": governance["key_id"],
            "signature_hex": Ed25519PrivateKey.from_private_bytes(b"\x01" * 32).sign(preimage).hex(),
        }],
    })
    lock0 = _json(args["stage_b_trust_lock_payload"])  # type: ignore[arg-type]
    lock_body = copy.deepcopy(lock0["body"])
    lock_body.update({  # type: ignore[union-attr]
        "authority_profile_id": profile_id,
        "authority_profile_payload_sha256": _sha(profile),
        "signer_registry_id": registry_id,
        "signer_registry_payload_sha256": _sha(registry),
    })
    lock_id = authority.phase2b_step5_content_id_v2("stage_b_trust_lock", lock_body)
    lock = authority.canonical_step5_json_bytes_v2({"body": lock_body, "id": lock_id})

    runtime_raw: list[RuntimeRaw] = []
    for ordinal, service_name in enumerate(authority.SERVICE_ORDER_V2):
        service_args = bases[ordinal].verify_arguments
        evidence_body = copy.deepcopy(bases[ordinal].evidence_body)
        evidence_body.update({
            "stage_b_trust_lock_id": lock_id,
            "authority_profile_id": profile_id,
            "signer_registry_id": registry_id,
            "service_name": service_name,
            "service_binding_id": service_bindings[ordinal]["binding_id"],
            "closure_policy_id": policy_ids[ordinal],
            "closure_policy_payload_sha256": _sha(policies[ordinal]),
        })
        evidence_id = closure.phase2b_step5_image_closure_evidence_id_v1(evidence_body)
        evidence = closure.canonical_runtime_closure_json_bytes_v1({"body": evidence_body, "id": evidence_id})
        runtime_raw.append(RuntimeRaw(
            service_name, policies[ordinal], evidence,
            service_args["raw_oci_index_payload"], service_args["raw_oci_manifest_payload"], service_args["raw_oci_config_payload"],  # type: ignore[arg-type]
            service_args["ordered_raw_layer_chunk_tuples"], service_args["ordered_support_artifact_chunk_tuples"],  # type: ignore[arg-type]
        ))
    return AuthorityRuntimeGraph(
        lock, profile, registry, lock_id, profile_id, registry_id,
        lock_id.encode("ascii"), hashlib.sha256(lock).digest(), tuple(runtime_raw), role_rows,
    )


@dataclass(frozen=True)
class OwnerChain:
    attempt: str
    owner: bytes
    decisions: tuple[bytes, ...]
    events: tuple[bytes, ...]
    owner_checkpoints: tuple[bytes, ...]
    owner_heads: tuple[bytes, ...]
    projection: bytes
    time_requests: tuple[bytes, ...]
    time_tokens: tuple[bytes, ...]
    time_checkpoints: tuple[bytes, ...]
    time_heads: tuple[bytes, ...]
    event_plan: tuple[time_core.ExpectedLogicalEventCoreV1, ...]


def _build_owner_chain(root: Path, graph: AuthorityRuntimeGraph, count: int) -> OwnerChain:
    attempt = "phase2b_step5_attempt_" + "a" * 64
    owner_payload = OWNER_CONFIG.read_bytes()
    gov = graph.role_rows[authority.ROLE_ORDER_V2.index("GOVERNANCE_ANCHOR")]
    cas = graph.role_rows[authority.ROLE_ORDER_V2.index("CAS_STORE")]
    logical = graph.role_rows[authority.ROLE_ORDER_V2.index("LOGICAL_TIME")]
    owner_root, time_root = root / "owner", root / "time"
    owner_root.mkdir(mode=0o700)
    owner_root.chmod(0o700)
    values: dict[str, list[bytes]] = {k: [] for k in ("d", "e", "oc", "oh", "p", "tr", "tt", "tc", "th")}
    plans: list[time_core.ExpectedLogicalEventCoreV1] = []
    with owner_status.OwnerStatusLedgerV1(
        str(owner_root), authority_profile_id=graph.profile_id,
        signer_registry_id=graph.registry_id, governance_key_id=gov["key_id"],
        governance_key_epoch=1, attempt_context_id=attempt,
        governance_public_key=bytes.fromhex(gov["public_key_hex"]),
        logical_time_public_key=bytes.fromhex(logical["public_key_hex"]),
        owner_authorization_payload=owner_payload, initialize_empty=True,
    ) as ol, logical_time.LogicalTimeLedgerV1(
        root_directory=str(time_root), authority_profile_id=graph.profile_id,
        signer_registry_id=graph.registry_id, logical_time_key_id=logical["key_id"],
        logical_time_key_epoch=1, attempt_context_id=attempt,
        logical_time_public_key=bytes.fromhex(logical["public_key_hex"]), create=True,
    ) as tl:
        values["oh"].append(ol.current_head_bytes())
        prior_time_head = tl.current_head_bytes(); values["th"].append(prior_time_head)
        prior_token = logical_time.GENESIS_TOKEN_ID_V1
        for index in range(count):
            decision = owner_status.build_owner_status_decision_v1(
                ol.current_head_bytes(), owner_authorization_payload=owner_payload,
                reason_code="INITIAL_OWNER_STATUS" if index == 0 else "OWNER_AUTHORIZATION_REVOKED",
                decision_nonce=("1" if index == 0 else "3") * 64,
            )
            decision_id = _json(decision)["decision_id"]
            nonce = ("2" if index == 0 else "4") * 64
            request = logical_time.build_logical_time_request_v1(
                authority_profile_id=graph.profile_id, signer_registry_id=graph.registry_id,
                logical_time_key_id=logical["key_id"], logical_time_key_epoch=1,
                attempt_context_id=attempt, request_nonce=nonce,
                purpose=owner_core.LOGICAL_TIME_EVENT_PURPOSE_V1,
                bound_artifact_id=decision_id, bound_artifact_sha256=_address(decision),
                prior_token_id=prior_token, next_sequence=index,
            )
            issuance = tl.issue_service_side_v1(
                request_payload=request, expected_previous_head_payload=prior_time_head,
                private_key=Ed25519PrivateKey.from_private_bytes(bytes([authority.ROLE_ORDER_V2.index("LOGICAL_TIME") + 1]) * 32),
            )
            logical_ref = owner_status.build_logical_time_dependency_reference_v1(
                decision_payload=decision, logical_time_token_payload=issuance.signed_token_payload,
                logical_time_public_key=bytes.fromhex(logical["public_key_hex"]),
            )
            marker = _address(f"legacy-cas-{index}".encode())
            cas_ref = owner_status.build_cas_store_dependency_reference_v1(
                decision_payload=decision, retention_session_id="legacy-owner-session",
                chain_id="legacy-owner-chain", signer_key_id=cas["key_id"],
                signed_receipt_schema_version=owner_status.CAS_SIGNED_RECEIPT_SCHEMA_V1,
                signed_receipt_id=marker, signed_receipt_canonical_sha256=marker,
                checkpoint_schema_version=owner_status.CAS_CHECKPOINT_SCHEMA_V1,
                checkpoint_id=marker, checkpoint_canonical_sha256=marker,
                checkpoint_index=index, head_schema_version=owner_status.CAS_HEAD_SCHEMA_V1,
                head_id=marker, head_canonical_sha256=marker,
                head_last_checkpoint_id=marker, head_checkpoint_count=index + 1,
            )
            appended = ol.append_service_side_v1(
                decision_payload=decision, cas_store_dependency_reference_payload=cas_ref,
                logical_time_dependency_reference_payload=logical_ref,
                logical_time_token_payload=issuance.signed_token_payload,
                expected_previous_head_payload=ol.current_head_bytes(),
                governance_private_key=Ed25519PrivateKey.from_private_bytes(b"\x01" * 32),
            )
            for key, payload in (("d", decision), ("e", appended.signed_event_payload),
                ("oc", appended.checkpoint_payload), ("oh", appended.head_payload),
                ("p", appended.projection_payload), ("tr", request),
                ("tt", issuance.signed_token_payload), ("tc", issuance.checkpoint_payload),
                ("th", issuance.head_payload)):
                values[key].append(payload)
            plans.append(time_core.ExpectedLogicalEventCoreV1(
                owner_core.LOGICAL_TIME_EVENT_PURPOSE_V1, decision_id, _address(decision), nonce
            ))
            prior_time_head = issuance.head_payload
            prior_token = _json(issuance.signed_token_payload)["token_id"]
    return OwnerChain(attempt, owner_payload, tuple(values["d"]), tuple(values["e"]),
        tuple(values["oc"]), tuple(values["oh"]), values["p"][-1],
        tuple(values["tr"]), tuple(values["tt"]), tuple(values["tc"]),
        tuple(values["th"]), tuple(plans))


def _runtime_result(graph: AuthorityRuntimeGraph, raw: RuntimeRaw):
    return closure.require_verified_offline_step5_image_closure_mechanics_v1(
        closure.verify_offline_step5_image_closure_mechanics_v1(
            stage_b_trust_lock_payload=graph.lock, authority_profile_payload=graph.profile,
            signer_registry_payload=graph.registry, out_of_band_expected_lock_id_bytes=graph.oob_id,
            out_of_band_expected_lock_payload_sha256_digest=graph.oob_digest,
            service_name=raw.service_name, runtime_closure_policy_payload=raw.policy,
            image_closure_evidence_payload=raw.evidence, raw_oci_index_payload=raw.index,
            raw_oci_manifest_payload=raw.manifest, raw_oci_config_payload=raw.config,
            ordered_raw_layer_chunk_tuples=raw.layers,
            ordered_support_artifact_chunk_tuples=raw.supports,
        )
    )


REPORT_FALSE = {
    "runtime_closure_inputs_durably_retained": False, "signed_runtime_attestation_verified": False,
    "service_image_exists": False, "service_image_pulled": False, "service_image_executed": False,
    "live_runtime_closure_observed": False, "entry_exit_closure_equal": False,
    "live_uid_gid_verified": False, "live_control_domain_verified": False,
    "key_custody_verified": False, "runtime_closure_signature_verified": False,
    "remote_attestation_verified": False, "external_cas_verified": False,
    "authority_evidence": False, "r01_satisfied": False, "readiness_evidence": False,
    "execution_authorized": False,
}


def _report(graph: AuthorityRuntimeGraph, chain: OwnerChain, ordinal: int, raw: RuntimeRaw) -> bytes:
    result = _runtime_result(graph, raw)
    binding = _json(graph.profile)["body"]["service_static_bindings_in_exact_order"][ordinal]  # type: ignore[index]
    layers, supports, manifests = tuple(result.ordered_layer_digests), tuple(result.support_artifact_digests), tuple(result.static_manifest_digests)
    body = {
        "schema_version": r01.RUNTIME_CLOSURE_REPORT_SCHEMA_V1,
        "claim_boundary": r01.RUNTIME_REPORT_CLAIM_BOUNDARY_V1,
        "stage_b_trust_lock_id": graph.lock_id, "authority_profile_id": graph.profile_id,
        "signer_registry_id": graph.registry_id, "attempt_context_id": chain.attempt,
        "service_ordinal": ordinal, "service_name": raw.service_name,
        "service_binding_id": binding["binding_id"], "runtime_closure_policy_id": result.closure_policy_id,
        "runtime_closure_policy_payload_sha256": _sha(raw.policy),
        "image_closure_evidence_id": result.evidence_id,
        "image_closure_evidence_payload_sha256": _sha(raw.evidence),
        "raw_oci_index_sha256": result.oci_index_digest,
        "raw_oci_manifest_sha256": result.oci_manifest_digest,
        "raw_oci_config_sha256": result.oci_config_digest,
        "ordered_layer_sha256s": list(layers), "ordered_layer_sha256_root": _root(layers),
        "ordered_support_artifact_sha256s": list(supports), "ordered_support_artifact_sha256_root": _root(supports),
        "ordered_static_manifest_sha256s": list(manifests), "ordered_static_manifest_sha256_root": _root(manifests),
        "offline_static_byte_graph_mechanics_verified": True, "claims": dict(REPORT_FALSE),
    }
    rid = _content_id(r01.RUNTIME_CLOSURE_REPORT_PREFIX_V1, REPORT_DOMAIN, body)
    return canonicalize_rfc8785_v1({"body": body, "id": rid})


BUNDLE_FALSE = {
    "reports_signed": False, "reports_durably_retained": False, "live_runtime_closure_verified": False,
    "live_uid_gid_verified": False, "live_control_domain_verified": False, "key_custody_verified": False,
    "external_currentness_verified": False, "external_anti_fork_anchor_verified": False,
    "authority_evidence": False, "real_r01_receipt_verified": False, "r01_satisfied": False,
    "readiness_evidence": False, "execution_authorized": False,
}


def _bundle(graph: AuthorityRuntimeGraph, chain: OwnerChain, reports: tuple[bytes, ...]) -> bytes:
    ids = [_json(x)["id"] for x in reports]
    body = {
        "schema_version": r01.RUNTIME_CLOSURE_REPORT_BUNDLE_SCHEMA_V1,
        "claim_boundary": r01.RUNTIME_REPORT_BUNDLE_CLAIM_BOUNDARY_V1,
        "stage_b_trust_lock_id": graph.lock_id, "authority_profile_id": graph.profile_id,
        "signer_registry_id": graph.registry_id, "attempt_context_id": chain.attempt,
        "service_names_in_exact_order": list(authority.SERVICE_ORDER_V2),
        "runtime_closure_report_ids_in_exact_order": ids,
        "runtime_closure_report_payload_sha256s_in_exact_order": [_address(x) for x in reports],
        "exact_five_reports_presented": True, "claims": dict(BUNDLE_FALSE),
    }
    bid = _content_id(r01.RUNTIME_CLOSURE_REPORT_BUNDLE_PREFIX_V1, BUNDLE_DOMAIN, body)
    return canonicalize_rfc8785_v1({"body": body, "id": bid})


def _owner_and_time_results(graph: AuthorityRuntimeGraph, chain: OwnerChain):
    common = dict(
        stage_b_trust_lock_payload=graph.lock, authority_profile_payload=graph.profile,
        signer_registry_payload=graph.registry, out_of_band_expected_lock_id_bytes=graph.oob_id,
        out_of_band_expected_payload_sha256_digest=graph.oob_digest,
        expected_attempt_context_id=chain.attempt,
    )
    time_result = time_core.require_verified_logical_time_replay_core_v1(
        time_core.verify_logical_time_replay_core_v1(
            graph.lock, graph.profile, graph.registry, chain.time_requests, chain.time_tokens,
            chain.time_checkpoints, chain.time_heads,
            out_of_band_expected_lock_id_bytes=graph.oob_id,
            out_of_band_expected_payload_sha256_digest=graph.oob_digest,
            expected_attempt_context_id=chain.attempt, expected_event_plan=chain.event_plan,
        )
    )
    owner_result = owner_core.require_verified_owner_status_replay_core_v1(
        owner_core.verify_owner_status_replay_core_v1(
            graph.lock, graph.profile, graph.registry, chain.owner, chain.decisions, chain.events,
            chain.owner_checkpoints, chain.owner_heads, chain.owner_heads[-1], chain.projection,
            chain.time_requests, chain.time_tokens, chain.time_checkpoints, chain.time_heads,
            out_of_band_expected_lock_id_bytes=graph.oob_id,
            out_of_band_expected_payload_sha256_digest=graph.oob_digest,
            expected_attempt_context_id=chain.attempt, expected_logical_time_event_plan=chain.event_plan,
        )
    )
    return owner_result, time_result


def _expected_objects(graph: AuthorityRuntimeGraph, chain: OwnerChain, reports: tuple[bytes, ...], bundle: bytes):
    owner_result, time_result = _owner_and_time_results(graph, chain)
    rows = [
        (r01.CAS_ARTIFACT_KINDS_BASE_V1[0], graph.lock_id, graph.lock),
        (r01.CAS_ARTIFACT_KINDS_BASE_V1[1], graph.profile_id, graph.profile),
        (r01.CAS_ARTIFACT_KINDS_BASE_V1[2], graph.registry_id, graph.registry),
        (r01.CAS_ARTIFACT_KINDS_BASE_V1[3], owner_result.owner_authorization_id, chain.owner),
    ]
    rows += [(r01.CAS_ARTIFACT_KINDS_BASE_V1[4+i], _json(p)["id"], p) for i, p in enumerate(reports)]
    rows += [
        (r01.CAS_ARTIFACT_KINDS_BASE_V1[9], _json(bundle)["id"], bundle),
        (r01.CAS_ARTIFACT_KINDS_BASE_V1[10], owner_result.projection_id, chain.projection),
        (r01.CAS_ARTIFACT_KINDS_BASE_V1[11], owner_result.head_ids[0], chain.owner_heads[0]),
        (r01.CAS_ARTIFACT_KINDS_BASE_V1[12], time_result.head_ids[0], chain.time_heads[0]),
    ]
    for i in range(len(chain.events)):
        payloads = (chain.decisions[i], chain.events[i], chain.owner_checkpoints[i], chain.owner_heads[i+1],
            chain.time_requests[i], chain.time_tokens[i], chain.time_checkpoints[i], chain.time_heads[i+1])
        ids = (owner_result.decision_ids[i], owner_result.event_ids[i], owner_result.checkpoint_ids[i], owner_result.head_ids[i+1],
            time_result.request_ids[i], time_result.token_ids[i], time_result.checkpoint_ids[i], time_result.head_ids[i+1])
        rows += list(zip(r01.CAS_ARTIFACT_KINDS_PER_EVENT_V1, ids, payloads, strict=True))
    return tuple(rows)


def _build_cas(root: Path, graph: AuthorityRuntimeGraph, chain: OwnerChain, rows):
    plan = store.build_retention_plan_v1(
        authority_profile_id=graph.profile_id, signer_registry_id=graph.registry_id,
        attempt_context_id=chain.attempt, retention_session_id="r01-candidate-session",
        chain_id="r01-candidate-chain", plan_kind="R01_PURE_REPLAY_CANDIDATE_EXACT_GRAPH",
        objects_in_exact_order=tuple((kind, max(1024, len(payload))) for kind, _, payload in rows),
    )
    cas_root = root / "cas"
    seed = bytes([authority.ROLE_ORDER_V2.index("CAS_STORE") + 1]) * 32
    out = {k: [] for k in ("q", "r", "c", "h", "s")}
    with store.CasStoreV2(
        root_directory=str(cas_root), retention_plan_payload=plan, private_key_seed=seed,
        allowed_peer_uids=frozenset({os.geteuid()}), stage_b_lock_payload=graph.lock,
        authority_profile_payload=graph.profile, signer_registry_payload=graph.registry,
        out_of_band_expected_lock_id_bytes=graph.oob_id,
        out_of_band_expected_lock_payload_sha256_digest=graph.oob_digest, create=True,
    ) as cas:
        for ordinal, (_, artifact_id, payload) in enumerate(rows):
            request_payload = store.build_generic_store_request_v1(
                retention_plan_payload=plan, ordinal=ordinal, bound_artifact_id=artifact_id,
                bound_artifact_sha256=_address(payload), reported_size_bytes=len(payload),
            )
            request = store.load_generic_store_request_v1(request_payload)
            path = root / f"object-{ordinal}.bin"; path.write_bytes(payload)
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                response = cas.ingest_open_fd(request, fd, store.PeerCredentialsV2(os.getpid(), os.geteuid(), os.getegid()))
            finally:
                os.close(fd)
            out["q"].append(request_payload)
            out["r"].append(response.checkpoint_record.signed_receipt.canonical_bytes())
            out["c"].append(response.checkpoint_record.canonical_bytes())
            out["h"].append(response.head_record.canonical_bytes())
            out["s"].append(response.canonical_bytes())
    return r01.CasStoreCompleteSessionRawInputsV2(plan, tuple(out["q"]), tuple(x[2] for x in rows),
        tuple(out["r"]), tuple(out["c"]), tuple(out["h"]), tuple(out["s"]))


@dataclass(frozen=True)
class Corpus:
    graph: AuthorityRuntimeGraph
    chain: OwnerChain
    reports: tuple[bytes, ...]
    bundle: bytes
    owner_dto: r01.OwnerStatusRawReplayInputsV1
    runtime_dtos: tuple[r01.OfflineRuntimeClosureRawInputsV1, ...]
    cas_dto: r01.CasStoreCompleteSessionRawInputsV2


def _build_corpus(root: Path, count: int) -> Corpus:
    graph = _build_authority_and_five_runtime_graphs()
    chain = _build_owner_chain(root, graph, count)
    reports = tuple(_report(graph, chain, i, raw) for i, raw in enumerate(graph.runtime_raw))
    bundle = _bundle(graph, chain, reports)
    owner_dto = r01.OwnerStatusRawReplayInputsV1(
        chain.owner, chain.decisions, chain.events, chain.owner_checkpoints, chain.owner_heads,
        chain.owner_heads[-1], chain.projection, chain.time_requests, chain.time_tokens,
        chain.time_checkpoints, chain.time_heads,
    )
    runtime_dtos = tuple(r01.OfflineRuntimeClosureRawInputsV1(
        raw.service_name, raw.policy, raw.evidence, raw.index, raw.manifest, raw.config,
        raw.layers, raw.supports, reports[i],
    ) for i, raw in enumerate(graph.runtime_raw))
    rows = _expected_objects(graph, chain, reports, bundle)
    cas_dto = _build_cas(root, graph, chain, rows)
    return Corpus(graph, chain, reports, bundle, owner_dto, runtime_dtos, cas_dto)


def _verify(c: Corpus, **changes: object):
    args = dict(
        stage_b_trust_lock_payload=c.graph.lock, authority_profile_payload=c.graph.profile,
        signer_registry_payload=c.graph.registry, out_of_band_expected_lock_id_bytes=c.graph.oob_id,
        out_of_band_expected_lock_payload_sha256_digest=c.graph.oob_digest,
        expected_attempt_context_id=c.chain.attempt, expected_logical_time_event_plan=c.chain.event_plan,
        owner_status_inputs=c.owner_dto, cas_store_inputs=c.cas_dto,
        runtime_closure_inputs_in_exact_service_order=c.runtime_dtos,
        runtime_closure_report_bundle_payload=c.bundle,
    )
    args.update(changes)
    return r01.verify_r01_pure_replay_candidate_v1(**args)


@pytest.fixture(params=(1, 2), scope="module")
def corpus(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    return _build_corpus(tmp_path_factory.mktemp(f"r01-pure-{request.param}"), request.param)


def _fails(operation: Callable[[], object], *codes: str):
    with pytest.raises(r01.Phase2BStep5R01PureReplayErrorV1) as caught:
        operation()
    if codes:
        assert caught.value.code in codes
    return caught.value


def test_real_active_and_revoked_full_graph_differential(corpus: Corpus) -> None:
    result = _verify(corpus)
    assert r01.require_verified_r01_pure_replay_candidate_v1(result) is result
    assert result.owner_event_count == len(corpus.chain.events)
    assert result.exact_cas_object_count == 13 + 8 * len(corpus.chain.events)
    assert result.owner_latest_effective_status == ("ACTIVE" if len(corpus.chain.events) == 1 else "REVOKED")
    # Independent verifier differential over the exact raw transport.
    owner_result, time_result = _owner_and_time_results(corpus.graph, corpus.chain)
    store_result = store_core.require_verified_cas_store_complete_session_replay_core_v2(
        store_core.verify_cas_store_complete_session_replay_core_v2(
            stage_b_lock_payload=corpus.graph.lock, authority_profile_payload=corpus.graph.profile,
            signer_registry_payload=corpus.graph.registry, out_of_band_expected_lock_id_bytes=corpus.graph.oob_id,
            out_of_band_expected_lock_payload_sha256_digest=corpus.graph.oob_digest,
            retention_plan_payload=corpus.cas_dto.retention_plan_payload,
            request_payloads_in_exact_order=corpus.cas_dto.request_payloads_in_exact_order,
            object_payloads_in_exact_order=corpus.cas_dto.object_payloads_in_exact_order,
            signed_receipt_payloads_in_exact_order=corpus.cas_dto.signed_receipt_payloads_in_exact_order,
            checkpoint_payloads_in_exact_order=corpus.cas_dto.checkpoint_payloads_in_exact_order,
            head_payloads_in_exact_order=corpus.cas_dto.head_payloads_in_exact_order,
            response_payloads_in_exact_order=corpus.cas_dto.response_payloads_in_exact_order,
        )
    )
    assert result.owner_current_head_id == owner_result.current_head_id
    assert result.logical_time_final_head_id == time_result.final_head_id
    assert result.cas_complete_head_state_id == store_result.complete_head_state_id
    runtime_results = tuple(_runtime_result(corpus.graph, raw) for raw in corpus.graph.runtime_raw)
    assert all(x.offline_static_byte_graph_mechanics_verified for x in runtime_results)
    assert owner_result.external_currentness_verified is False
    assert owner_result.authority_evidence is False and owner_result.r01_satisfied is False
    assert time_result.external_currentness_verified is False
    assert time_result.authority_evidence is False and time_result.execution_authorized is False
    assert store_result.durability_verified is False and store_result.live_cas_store_verified is False
    assert store_result.authority_evidence is False and store_result.execution_authorized is False
    assert all(x.service_image_exists is False and x.live_runtime_closure_observed is False for x in runtime_results)
    assert all(x.authority_evidence is False and x.r01_satisfied is False for x in runtime_results)
    assert store_result.object_dispositions_in_exact_order == ("CREATED_IF_ABSENT",) * result.exact_cas_object_count
    assert tuple(x[0] for x in store_result.object_bindings_in_exact_order) == tuple(range(result.exact_cas_object_count))
    assert all(_json(report)["body"]["claims"]["runtime_closure_inputs_durably_retained"] is False for report in corpus.reports)  # type: ignore[index]


def test_exact_dtos_sequence_and_plan_binding_fail_closed(corpus: Corpus) -> None:
    class OwnerSubclass(r01.OwnerStatusRawReplayInputsV1):
        pass
    class CasSubclass(r01.CasStoreCompleteSessionRawInputsV2):
        pass
    class RuntimeSubclass(r01.OfflineRuntimeClosureRawInputsV1):
        pass
    _fails(lambda: _verify(corpus, owner_status_inputs=object()), r01.FAIL_INPUT_V1)
    _fails(lambda: _verify(corpus, cas_store_inputs=object()), r01.FAIL_INPUT_V1)
    _fails(lambda: _verify(corpus, runtime_closure_inputs_in_exact_service_order=tuple(object() for _ in range(5))), r01.FAIL_INPUT_V1)
    owner_sub = OwnerSubclass(*(getattr(corpus.owner_dto, f.name) for f in fields(corpus.owner_dto)))
    cas_sub = CasSubclass(*(getattr(corpus.cas_dto, f.name) for f in fields(corpus.cas_dto)))
    runtime_sub = RuntimeSubclass(*(getattr(corpus.runtime_dtos[0], f.name) for f in fields(corpus.runtime_dtos[0])))
    _fails(lambda: _verify(corpus, owner_status_inputs=owner_sub), r01.FAIL_INPUT_V1)
    _fails(lambda: _verify(corpus, cas_store_inputs=cas_sub), r01.FAIL_INPUT_V1)
    _fails(lambda: _verify(corpus, runtime_closure_inputs_in_exact_service_order=(runtime_sub,) + corpus.runtime_dtos[1:]), r01.FAIL_INPUT_V1)
    objects = corpus.cas_dto.object_payloads_in_exact_order
    variants = (objects[:-1], objects + (b"extra",), objects[:1] + objects[2:3] + objects[1:2] + objects[3:],
        (objects[0] + b"x",) + objects[1:])
    for changed in variants:
        _fails(lambda changed=changed: _verify(corpus, cas_store_inputs=replace(corpus.cas_dto, object_payloads_in_exact_order=changed)))
    for ordinal in range(len(objects)):
        missing = objects[:ordinal] + objects[ordinal + 1:]
        corrupt = objects[:ordinal] + (objects[ordinal] + b"!",) + objects[ordinal + 1:]
        _fails(lambda missing=missing: _verify(corpus, cas_store_inputs=replace(corpus.cas_dto, object_payloads_in_exact_order=missing)))
        _fails(lambda corrupt=corrupt: _verify(corpus, cas_store_inputs=replace(corpus.cas_dto, object_payloads_in_exact_order=corrupt)))
    for ordinal in range(len(objects) - 1):
        swapped = list(objects); swapped[ordinal], swapped[ordinal + 1] = swapped[ordinal + 1], swapped[ordinal]
        _fails(lambda swapped=tuple(swapped): _verify(corpus, cas_store_inputs=replace(corpus.cas_dto, object_payloads_in_exact_order=swapped)))
    _fails(lambda: _verify(corpus, cas_store_inputs=replace(corpus.cas_dto, retention_plan_payload=corpus.cas_dto.retention_plan_payload + b"\n")))


def test_runtime_service_report_bundle_and_cross_splices_fail(corpus: Corpus, tmp_path: Path) -> None:
    swapped = list(corpus.runtime_dtos); swapped[0], swapped[1] = swapped[1], swapped[0]
    _fails(lambda: _verify(corpus, runtime_closure_inputs_in_exact_service_order=tuple(swapped)), r01.FAIL_BINDING_V1)
    changed_report = _json(corpus.reports[0]); changed_report["body"]["raw_oci_index_sha256"] = "sha256:" + "f" * 64  # type: ignore[index]
    changed_report["id"] = _content_id(r01.RUNTIME_CLOSURE_REPORT_PREFIX_V1, REPORT_DOMAIN, changed_report["body"])
    dto0 = replace(corpus.runtime_dtos[0], runtime_closure_report_payload=canonicalize_rfc8785_v1(changed_report))
    _fails(lambda: _verify(corpus, runtime_closure_inputs_in_exact_service_order=(dto0,) + corpus.runtime_dtos[1:]), r01.FAIL_BINDING_V1)
    reports_swapped = list(corpus.runtime_dtos)
    reports_swapped[0] = replace(reports_swapped[0], runtime_closure_report_payload=corpus.reports[1])
    reports_swapped[1] = replace(reports_swapped[1], runtime_closure_report_payload=corpus.reports[0])
    _fails(lambda: _verify(corpus, runtime_closure_inputs_in_exact_service_order=tuple(reports_swapped)), r01.FAIL_BINDING_V1)
    wrong_root = _json(corpus.reports[0]); wrong_root["body"]["ordered_layer_sha256_root"] = "f" * 64  # type: ignore[index]
    wrong_root["id"] = _content_id(r01.RUNTIME_CLOSURE_REPORT_PREFIX_V1, REPORT_DOMAIN, wrong_root["body"])
    root_dto = replace(corpus.runtime_dtos[0], runtime_closure_report_payload=canonicalize_rfc8785_v1(wrong_root))
    _fails(lambda: _verify(corpus, runtime_closure_inputs_in_exact_service_order=(root_dto,) + corpus.runtime_dtos[1:]), r01.FAIL_BINDING_V1)
    changed_bundle = _json(corpus.bundle); ids = changed_bundle["body"]["runtime_closure_report_ids_in_exact_order"]  # type: ignore[index]
    ids[0], ids[1] = ids[1], ids[0]
    changed_bundle["id"] = _content_id(r01.RUNTIME_CLOSURE_REPORT_BUNDLE_PREFIX_V1, BUNDLE_DOMAIN, changed_bundle["body"])
    _fails(lambda: _verify(corpus, runtime_closure_report_bundle_payload=canonicalize_rfc8785_v1(changed_bundle)), r01.FAIL_BINDING_V1)
    foreign = _build_authority_and_five_runtime_graphs(salt=1)
    _fails(lambda: _verify(corpus, expected_attempt_context_id=corpus.chain.attempt + "x"))
    _fails(lambda: _verify(corpus, authority_profile_payload=foreign.profile))
    _fails(lambda: _verify(corpus, signer_registry_payload=foreign.registry))


def test_all_claims_private_lineage_copy_and_mutation(corpus: Corpus) -> None:
    result = _verify(corpus)
    true_fields = ("raw_authority_graph_verified", "owner_status_replay_core_verified",
        "logical_time_replay_core_verified", "cas_complete_session_replay_core_verified",
        "five_offline_runtime_closure_mechanics_verified", "runtime_reports_and_bundle_verified",
        "exact_cas_13_plus_8n_sequence_verified", "all_presented_mechanics_cross_bound", "verifier_only")
    false_fields = tuple(f.name for f in fields(result) if f.type == "bool" and f.name not in true_fields)
    assert all(getattr(result, x) is True for x in true_fields)
    assert all(getattr(result, x) is False for x in false_fields)
    with pytest.raises(TypeError):
        r01.VerifiedR01PureReplayCandidateV1()  # type: ignore[call-arg]
    forged = object.__new__(r01.VerifiedR01PureReplayCandidateV1)
    _fails(lambda: r01.require_verified_r01_pure_replay_candidate_v1(forged), r01.FAIL_RESULT_V1)
    with pytest.raises(TypeError):
        copy.copy(result)
    copied = object.__new__(r01.VerifiedR01PureReplayCandidateV1)
    for field in fields(result):
        object.__setattr__(copied, field.name, getattr(result, field.name))
    _fails(lambda: r01.require_verified_r01_pure_replay_candidate_v1(copied), r01.FAIL_RESULT_V1)
    object.__setattr__(result, "r01_satisfied", True)
    _fails(lambda: r01.require_verified_r01_pure_replay_candidate_v1(result), r01.FAIL_RESULT_V1)


def test_constants_helpers_and_upstream_export_rebinding_do_not_change_contract(corpus: Corpus, monkeypatch: pytest.MonkeyPatch) -> None:
    verify = r01.verify_r01_pure_replay_candidate_v1
    original_error = r01.Phase2BStep5R01PureReplayErrorV1
    expected_binding_code = r01.FAIL_BINDING_V1
    for name in tuple(r01.__all__) + ("_RFC8785Error", "_canonicalize_rfc8785_v1", "_decode_canonical_rfc8785_v1",
        "_load_stage_b_trust_lock_v2", "_load_authority_profile_v2", "_load_signer_registry_v2",
        "_require_authority_graph_v2", "_verify_owner_core_v1", "_require_owner_core_v1",
        "_verify_time_core_v1", "_require_time_core_v1", "_verify_store_core_v2", "_require_store_core_v2",
        "_verify_runtime_closure_v1", "_require_runtime_closure_v1"):
        if hasattr(r01, name):
            monkeypatch.setattr(r01, name, object())
    assert original_error is not r01.Phase2BStep5R01PureReplayErrorV1
    result = verify(
        stage_b_trust_lock_payload=corpus.graph.lock, authority_profile_payload=corpus.graph.profile,
        signer_registry_payload=corpus.graph.registry, out_of_band_expected_lock_id_bytes=corpus.graph.oob_id,
        out_of_band_expected_lock_payload_sha256_digest=corpus.graph.oob_digest,
        expected_attempt_context_id=corpus.chain.attempt, expected_logical_time_event_plan=corpus.chain.event_plan,
        owner_status_inputs=corpus.owner_dto, cas_store_inputs=corpus.cas_dto,
        runtime_closure_inputs_in_exact_service_order=corpus.runtime_dtos,
        runtime_closure_report_bundle_payload=corpus.bundle,
    )
    assert type(result).__name__ == "VerifiedR01PureReplayCandidateV1"
    bad_bundle = corpus.bundle[:-1] + (b"0" if corpus.bundle[-1:] != b"0" else b"1")
    with pytest.raises(original_error) as caught:
        verify(
            stage_b_trust_lock_payload=corpus.graph.lock, authority_profile_payload=corpus.graph.profile,
            signer_registry_payload=corpus.graph.registry, out_of_band_expected_lock_id_bytes=corpus.graph.oob_id,
            out_of_band_expected_lock_payload_sha256_digest=corpus.graph.oob_digest,
            expected_attempt_context_id=corpus.chain.attempt, expected_logical_time_event_plan=corpus.chain.event_plan,
            owner_status_inputs=corpus.owner_dto, cas_store_inputs=corpus.cas_dto,
            runtime_closure_inputs_in_exact_service_order=corpus.runtime_dtos,
            runtime_closure_report_bundle_payload=bad_bundle,
        )
    assert caught.value.code in (expected_binding_code, "FAIL_STEP5_R01_PURE_REPLAY_V1_CANONICAL")


def test_not_implemented_entry_points_capture_error_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    verify_real = r01.verify_real_r01_authorization_status_receipt_v1
    derive = r01.derive_r01_readiness_overlay_v1
    error_type = r01.Phase2BStep5R01PureReplayErrorV1
    code = r01.FAIL_NOT_IMPLEMENTED_V1
    monkeypatch.setattr(r01, "Phase2BStep5R01PureReplayErrorV1", object())
    monkeypatch.setattr(r01, "FAIL_NOT_IMPLEMENTED_V1", "EVIL")
    for operation in (verify_real, derive):
        with pytest.raises(error_type) as caught:
            operation()
        assert caught.value.code == code


def test_not_implemented_and_producer_free_clean_import() -> None:
    _fails(lambda: r01.verify_real_r01_authorization_status_receipt_v1(), r01.FAIL_NOT_IMPLEMENTED_V1)
    _fails(lambda: r01.derive_r01_readiness_overlay_v1(), r01.FAIL_NOT_IMPLEMENTED_V1)
    tree = ast.parse(Path(r01.__file__).read_text())
    forbidden = ("phase2b_formal_wire_covert_step5_store_v2", "phase2b_formal_wire_covert_step5_owner_status_v1",
        "phase2b_formal_wire_covert_step5_logical_time_v1", "socket", "systemd", "subprocess")
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(token in name for token in forbidden for name in imported)
    command = [sys.executable, "-c", (
        "import sys; import hegel_machine.phase2b_formal_wire_covert_step5_r01_pure_replay_v1; "
        "bad=[x for x in sys.modules if x.endswith(('step5_store_v2','step5_owner_status_v1','step5_logical_time_v1'))]; "
        "assert not bad, bad"
    )]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
