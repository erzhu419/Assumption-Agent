from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Callable, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import phase2b_formal_wire_covert_step5_logical_time_v1 as logical_time
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1
    as admission,
)
from test_phase2b_formal_wire_covert_step5_r01_evidence_contract_v1 import (
    _EvidenceGraph,
    _build_evidence,
)
from test_phase2b_formal_wire_covert_step5_r01_external_witness_v1 import (
    _issue as _issue_external_candidate,
    _private_spool,
)


_SIGNATURE_SCHEMA = (
    "hegel-machine-phase2b-step5-external-admission-signature/1"
)


@dataclass(frozen=True)
class _PositiveCorpus:
    graph: _EvidenceGraph
    call: dict[str, object]
    payloads: Mapping[str, bytes]
    roots: Mapping[str, Mapping[str, object]]
    candidate_mapping: Mapping[str, object]
    genesis_mapping: Mapping[str, object]
    online_private_key: Ed25519PrivateKey
    custody_private_key: Ed25519PrivateKey
    durable_private_key: Ed25519PrivateKey


@dataclass(frozen=True)
class _LogicalPrefix:
    root: Path
    requests: tuple[bytes, ...]
    tokens: tuple[bytes, ...]
    checkpoints: tuple[bytes, ...]
    heads: tuple[bytes, ...]


def _oid(label: str, value: int) -> str:
    return f"{label}_{value:064x}"


def _hex(value: int) -> str:
    return f"{value:064x}"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _public_hex(private_key: Ed25519PrivateKey) -> str:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()


def _signed_admission_object(
    *,
    artifact_kind: str,
    schema: str,
    id_field: str,
    body: Mapping[str, object],
    private_key: Ed25519PrivateKey,
    signer_key_id: str,
    purpose: str,
    signature_field: str = "signature",
) -> tuple[bytes, dict[str, object]]:
    exact_body = copy.deepcopy(dict(body))
    artifact_id = admission.phase2b_step5_external_witness_admission_content_id_v1(
        artifact_kind, exact_body
    )
    preimage = (
        admission.phase2b_step5_external_witness_admission_signature_preimage_v1(
            artifact_kind=artifact_kind,
            body=exact_body,
            artifact_id=artifact_id,
            signer_key_id=signer_key_id,
            key_epoch=1,
            purpose=purpose,
        )
    )
    signature = {
        "schema_version": _SIGNATURE_SCHEMA,
        "algorithm": admission.ALGORITHM_V1,
        "signer_key_id": signer_key_id,
        "key_epoch": 1,
        "purpose": purpose,
        "signature_hex": private_key.sign(preimage).hex(),
    }
    root = {
        "schema_version": schema,
        "body": exact_body,
        id_field: artifact_id,
        signature_field: signature,
    }
    return admission.canonical_external_witness_admission_json_bytes_v1(root), root


def _external_candidate_from_producer(
    tmp_path: Path, graph: _EvidenceGraph
) -> dict[str, bytes]:
    spool = _private_spool(tmp_path, "bootstrap-external-candidate")
    issued = _issue_external_candidate(graph, spool)
    names = issued.spool_file_basenames_in_commit_order
    assert len(names) == 5
    return {
        "candidate_anchor": graph.payloads["anchor"],
        "candidate_receipt": (spool / names[1]).read_bytes(),
        "candidate_ledger": (spool / names[2]).read_bytes(),
        "candidate_checkpoint": (spool / names[3]).read_bytes(),
        "candidate_head": (spool / names[4]).read_bytes(),
    }


def _candidate_mapping_from_raw(
    payloads: Mapping[str, bytes], attempt_context_id: str
) -> dict[str, object]:
    anchor = json.loads(payloads["candidate_anchor"])
    receipt = json.loads(payloads["candidate_receipt"])
    ledger = json.loads(payloads["candidate_ledger"])
    checkpoint = json.loads(payloads["candidate_checkpoint"])
    head = json.loads(payloads["candidate_head"])
    mapping: dict[str, object] = {
        "external_anchor_id": anchor["anchor_id"],
        "external_anchor_payload_sha256": _sha(payloads["candidate_anchor"]),
        "external_receipt_id": receipt["receipt_id"],
        "external_receipt_payload_sha256": _sha(payloads["candidate_receipt"]),
        "external_ledger_id": ledger["ledger_id"],
        "external_ledger_payload_sha256": _sha(payloads["candidate_ledger"]),
        "external_checkpoint_id": checkpoint["checkpoint_id"],
        "external_checkpoint_payload_sha256": _sha(
            payloads["candidate_checkpoint"]
        ),
        "external_head_id": head["head_id"],
        "external_head_payload_sha256": _sha(payloads["candidate_head"]),
        "witness_sequence": receipt["witness_sequence"],
        "tree_size": receipt["tree_size"],
        "signed_tree_root_sha256": receipt["signed_tree_root_sha256"],
        "attempt_context_id": attempt_context_id,
        "anchor_sequence": anchor["body"]["anchor_sequence"],
        "effective_state": "CURRENT_UNSUPERSEDED",
        "superseded": False,
        "revoked": False,
        "map_leaf_index": 0,
        "map_tree_size": 1,
        "map_root_sha256": None,
        "map_inclusion_path_sha256_in_exact_order": [],
        "presented_to_live_consistency_path_sha256_in_exact_order": [],
    }
    mapping["map_root_sha256"] = admission._current_map_leaf_root(mapping)
    return mapping


def _common_body(
    graph: _EvidenceGraph,
    *,
    bootstrap_policy_id: str,
    online_policy_id: str,
    durable_policy_id: str,
) -> dict[str, object]:
    return {
        "status": admission.CANDIDATE_STATUS_V1,
        "claim_level": admission.CANDIDATE_CLAIM_LEVEL_V1,
        "stage_b_trust_lock_id": graph.authority.lock_id,
        "authority_profile_id": json.loads(graph.authority.profile_payload)["id"],
        "signer_registry_id": json.loads(graph.authority.registry_payload)["id"],
        "attempt_context_id": graph.attempt_id,
        "bootstrap_external_witness_policy_id": bootstrap_policy_id,
        "online_witness_policy_id": online_policy_id,
        "durable_anchor_policy_id": durable_policy_id,
    }


def _runtime_custody_binding(
    online_policy_body: Mapping[str, object], *, salt: int
) -> dict[str, object]:
    return {
        "service_binding_id": online_policy_body["service_binding_id"],
        "runtime_instance_id": _oid("external_runtime_instance", 4100 + salt),
        "peer_pid": 4200 + salt,
        "peer_uid": online_policy_body["service_uid"],
        "peer_gid": online_policy_body["service_gid"],
        "user_namespace_inode": online_policy_body["service_user_namespace_inode"],
        "process_start_ticks": 4300 + salt,
        "control_domain_id": online_policy_body["witness_control_domain_id"],
        "runtime_closure_policy_id": online_policy_body[
            "runtime_closure_policy_id"
        ],
        "runtime_closure_report_id": _oid(
            "external_runtime_closure_report", 4400 + salt
        ),
        "runtime_closure_report_payload_sha256": _hex(4500 + salt),
        "custody_attestation_policy_id": online_policy_body[
            "custody_attestation_policy_id"
        ],
        "custody_attestation_id": _oid(
            "external_custody_attestation", 4600 + salt
        ),
        "custody_attestation_payload_sha256": _hex(4700 + salt),
        "online_key_nonexportable": True,
    }


def _build_pre_cut_chain(
    root: Path,
    graph: _EvidenceGraph,
    *,
    request_nonce: str = "d" * 64,
    purpose: str = "R01_STATUS_OBSERVED",
    bound_artifact_id: str | None = None,
    bound_artifact_sha256: str | None = None,
) -> _LogicalPrefix:
    logical_index = authority.ROLE_ORDER_V2.index("LOGICAL_TIME")
    logical_row = graph.authority.role_keys[logical_index]
    private_key = graph.authority.private_keys["LOGICAL_TIME"]
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    profile_id = json.loads(graph.authority.profile_payload)["id"]
    registry_id = json.loads(graph.authority.registry_payload)["id"]
    artifact_id = (
        graph.values["statement"]["statement_id"]
        if bound_artifact_id is None
        else bound_artifact_id
    )
    artifact_sha256 = (
        "sha256:" + _sha(graph.payloads["statement"])
        if bound_artifact_sha256 is None
        else bound_artifact_sha256
    )
    with logical_time.LogicalTimeLedgerV1(
        root_directory=str(root),
        authority_profile_id=profile_id,
        signer_registry_id=registry_id,
        logical_time_key_id=logical_row["key_id"],
        logical_time_key_epoch=1,
        attempt_context_id=graph.attempt_id,
        logical_time_public_key=public_key,
        create=True,
    ) as ledger:
        genesis_head = ledger.current_head_bytes()
        request_payload = logical_time.build_logical_time_request_v1(
            authority_profile_id=profile_id,
            signer_registry_id=registry_id,
            logical_time_key_id=logical_row["key_id"],
            logical_time_key_epoch=1,
            attempt_context_id=graph.attempt_id,
            request_nonce=request_nonce,
            purpose=purpose,
            bound_artifact_id=artifact_id,
            bound_artifact_sha256=artifact_sha256,
            prior_token_id=logical_time.GENESIS_TOKEN_ID_V1,
            next_sequence=0,
        )
        issuance = ledger.issue_service_side_v1(
            request_payload=request_payload,
            expected_previous_head_payload=genesis_head,
            private_key=private_key,
        )
    return _LogicalPrefix(
        root=root,
        requests=(request_payload,),
        tokens=(issuance.signed_token_payload,),
        checkpoints=(issuance.checkpoint_payload,),
        heads=(genesis_head, issuance.head_payload),
    )


def _append_post_q2_chain(
    prefix: _LogicalPrefix,
    graph: _EvidenceGraph,
    *,
    transcript_id: str,
    transcript_payload: bytes,
    request_nonce: str,
) -> tuple[tuple[bytes, ...], tuple[bytes, ...], tuple[bytes, ...], tuple[bytes, ...]]:
    logical_index = authority.ROLE_ORDER_V2.index("LOGICAL_TIME")
    logical_row = graph.authority.role_keys[logical_index]
    private_key = graph.authority.private_keys["LOGICAL_TIME"]
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    profile_id = json.loads(graph.authority.profile_payload)["id"]
    registry_id = json.loads(graph.authority.registry_payload)["id"]
    prior_token_id = json.loads(prefix.tokens[-1])["token_id"]
    with logical_time.LogicalTimeLedgerV1(
        root_directory=str(prefix.root),
        authority_profile_id=profile_id,
        signer_registry_id=registry_id,
        logical_time_key_id=logical_row["key_id"],
        logical_time_key_epoch=1,
        attempt_context_id=graph.attempt_id,
        logical_time_public_key=public_key,
        create=False,
    ) as ledger:
        previous_head = ledger.current_head_bytes()
        assert previous_head == prefix.heads[-1]
        request_payload = logical_time.build_logical_time_request_v1(
            authority_profile_id=profile_id,
            signer_registry_id=registry_id,
            logical_time_key_id=logical_row["key_id"],
            logical_time_key_epoch=1,
            attempt_context_id=graph.attempt_id,
            request_nonce=request_nonce,
            purpose=admission.POST_Q2_LOGICAL_TIME_PURPOSE_V1,
            bound_artifact_id=transcript_id,
            bound_artifact_sha256="sha256:" + _sha(transcript_payload),
            prior_token_id=prior_token_id,
            next_sequence=1,
        )
        issuance = ledger.issue_service_side_v1(
            request_payload=request_payload,
            expected_previous_head_payload=previous_head,
            private_key=private_key,
        )
    return (
        prefix.requests + (request_payload,),
        prefix.tokens + (issuance.signed_token_payload,),
        prefix.checkpoints + (issuance.checkpoint_payload,),
        prefix.heads + (issuance.head_payload,),
    )


def _positive_corpus(tmp_path: Path, *, salt: int = 0) -> _PositiveCorpus:
    base_graph = _build_evidence(salt=salt)
    logical_prefix = _build_pre_cut_chain(
        tmp_path / "complete-logical-time-chain", base_graph
    )
    pre_cut_token = json.loads(logical_prefix.tokens[-1])
    pre_cut_checkpoint = json.loads(logical_prefix.checkpoints[-1])
    pre_cut_head = json.loads(logical_prefix.heads[-1])

    def bind_anchor_to_pre_cut(body: dict[str, object]) -> None:
        body.update(
            {
                "current_cut_token_id": pre_cut_token["token_id"],
                "current_cut_token_payload_sha256": (
                    "sha256:" + _sha(logical_prefix.tokens[-1])
                ),
                "current_cut_checkpoint_id": pre_cut_checkpoint["checkpoint_id"],
                "current_cut_checkpoint_payload_sha256": (
                    "sha256:" + _sha(logical_prefix.checkpoints[-1])
                ),
                "current_cut_head_id": pre_cut_head["head_id"],
                "current_cut_head_payload_sha256": (
                    "sha256:" + _sha(logical_prefix.heads[-1])
                ),
            }
        )

    graph = _build_evidence(
        salt=salt,
        authority_graph=base_graph.authority,
        mutators={"anchor": bind_anchor_to_pre_cut},
    )
    assert graph.attempt_id == base_graph.attempt_id
    assert graph.payloads["statement"] == base_graph.payloads["statement"]
    payloads: dict[str, bytes] = _external_candidate_from_producer(tmp_path, graph)
    roots: dict[str, Mapping[str, object]] = {
        name: json.loads(payload) for name, payload in payloads.items()
    }
    candidate_mapping = _candidate_mapping_from_raw(payloads, graph.attempt_id)
    bootstrap_policy = json.loads(graph.payloads["policy"])
    payloads["bootstrap_policy"] = graph.payloads["policy"]
    roots["bootstrap_policy"] = bootstrap_policy

    online_private = Ed25519PrivateKey.from_private_bytes(bytes([90 + salt]) * 32)
    custody_private = Ed25519PrivateKey.from_private_bytes(bytes([100 + salt]) * 32)
    durable_private = Ed25519PrivateKey.from_private_bytes(bytes([110 + salt]) * 32)
    online_key_id = _oid("external_online_witness_key", 3000 + salt)
    custody_key_id = _oid("external_custody_attestor_key", 3001 + salt)
    durable_key_id = _oid("external_durable_anchor_key", 3002 + salt)

    online_body: dict[str, object] = {
        "schema_version": admission.ONLINE_WITNESS_POLICY_SCHEMA_V1,
        "status": admission.CANDIDATE_STATUS_V1,
        "claim_level": admission.CANDIDATE_CLAIM_LEVEL_V1,
        "policy_epoch": 1,
        "witness_namespace": bootstrap_policy["witness_namespace"],
        "witness_control_domain_id": _oid(
            "external_online_control_domain", 3010 + salt
        ),
        "bootstrap_external_witness_policy_id": bootstrap_policy[
            "external_witness_policy_id"
        ],
        "bootstrap_external_witness_policy_payload_sha256": _sha(
            graph.payloads["policy"]
        ),
        "bootstrap_key_id": bootstrap_policy["witness_key_id"],
        "bootstrap_key_epoch": bootstrap_policy["witness_key_epoch"],
        "bootstrap_public_key_hex": bootstrap_policy["public_key_hex"],
        "bootstrap_authorized_signature_purposes_in_exact_order": list(
            admission.BOOTSTRAP_PURPOSES_IN_EXACT_ORDER_V1
        ),
        "online_key_id": online_key_id,
        "online_key_epoch": 1,
        "online_public_key_hex": _public_hex(online_private),
        "online_authorized_signature_purposes_in_exact_order": list(
            admission.ONLINE_PURPOSES_IN_EXACT_ORDER_V1
        ),
        "service_binding_id": _oid("external_online_service_binding", 3020 + salt),
        "service_uid": 62000 + salt,
        "service_gid": 63000 + salt,
        "service_user_namespace_inode": 64000 + salt,
        "runtime_closure_policy_id": _oid(
            "external_runtime_closure_policy", 3030 + salt
        ),
        "runtime_closure_policy_payload_sha256": _hex(3040 + salt),
        "runtime_identity_profile": "LINUX_AF_UNIX_SEQPACKET_PROCFS_ENTRY_EXIT_V1",
        "custody_attestation_policy_id": _oid(
            "external_custody_attestation_policy", 3050 + salt
        ),
        "custody_attestation_policy_payload_sha256": _hex(3060 + salt),
        "custody_attestor_key_id": custody_key_id,
        "custody_attestor_key_epoch": 1,
        "custody_attestor_public_key_hex": _public_hex(custody_private),
        "online_key_nonexportable_required": True,
        "algorithm": admission.ALGORITHM_V1,
    }
    online_payload, online_root = _signed_admission_object(
        artifact_kind="online_witness_policy",
        schema=admission.ONLINE_WITNESS_POLICY_SCHEMA_V1,
        id_field="online_witness_policy_id",
        body=online_body,
        private_key=graph.witness_private_key,
        signer_key_id=bootstrap_policy["witness_key_id"],
        purpose=admission.SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1,
        signature_field="bootstrap_signature",
    )
    payloads["online_policy"] = online_payload
    roots["online_policy"] = online_root

    genesis_mapping = admission._genesis_mapping(graph.attempt_id)
    durable_body: dict[str, object] = {
        "schema_version": admission.DURABLE_ANCHOR_POLICY_SCHEMA_V1,
        "status": admission.CANDIDATE_STATUS_V1,
        "claim_level": admission.CANDIDATE_CLAIM_LEVEL_V1,
        "policy_epoch": 1,
        "bootstrap_external_witness_policy_id": bootstrap_policy[
            "external_witness_policy_id"
        ],
        "bootstrap_external_witness_policy_payload_sha256": _sha(
            graph.payloads["policy"]
        ),
        "online_witness_policy_id": online_root["online_witness_policy_id"],
        "anchor_namespace": f"durable-anchor-namespace-{salt}",
        "anchor_control_domain_id": _oid(
            "external_durable_control_domain", 3070 + salt
        ),
        "anchor_key_id": durable_key_id,
        "anchor_key_epoch": 1,
        "anchor_public_key_hex": _public_hex(durable_private),
        "algorithm": admission.ALGORITHM_V1,
        "authorized_signature_purposes_in_exact_order": list(
            admission.A_PURPOSES_IN_EXACT_ORDER_V1
        ),
        "genesis_checkpoint_id": admission._A_GENESIS_CHECKPOINT_ID,
        "genesis_head_id": admission._A_GENESIS_HEAD_ID,
        "genesis_tree_root_sha256": "0" * 64,
        "oob_high_water_checkpoint_id": admission._A_GENESIS_CHECKPOINT_ID,
        "oob_high_water_checkpoint_payload_sha256": "0" * 64,
        "oob_high_water_checkpoint_sequence": 0,
        "oob_high_water_head_id": admission._A_GENESIS_HEAD_ID,
        "oob_high_water_head_payload_sha256": "0" * 64,
        "oob_high_water_head_sequence": 0,
        "oob_high_water_tree_root_sha256": "0" * 64,
        "oob_high_water_mapping": copy.deepcopy(genesis_mapping),
        "admission_slot_key_profile": (
            "UNIQUE_BY_ATTEMPT_CONTEXT_ID_AND_ANCHOR_SEQUENCE"
        ),
        "compare_and_append_semantics": (
            "ATOMIC_COMPARE_EXACT_OOB_HIGH_WATER_THEN_APPEND_ONCE"
        ),
        "durable_before_return_required": True,
    }
    durable_payload, durable_root = _signed_admission_object(
        artifact_kind="durable_anchor_policy",
        schema=admission.DURABLE_ANCHOR_POLICY_SCHEMA_V1,
        id_field="durable_anchor_policy_id",
        body=durable_body,
        private_key=durable_private,
        signer_key_id=durable_key_id,
        purpose=admission.SIGN_DURABLE_ANCHOR_POLICY_PURPOSE_V1,
    )
    payloads["durable_policy"] = durable_payload
    roots["durable_policy"] = durable_root

    common = _common_body(
        graph,
        bootstrap_policy_id=bootstrap_policy["external_witness_policy_id"],
        online_policy_id=online_root["online_witness_policy_id"],
        durable_policy_id=durable_root["durable_anchor_policy_id"],
    )
    prepare_nonce = "a" * 64
    confirmation_nonce = "b" * 64
    post_q2_nonce = "c" * 64
    runtime_binding = _runtime_custody_binding(online_body, salt=salt)

    q1_request_body = {
        **common,
        "schema_version": admission.Q1_PREPARE_REQUEST_SCHEMA_V1,
        "prepare_nonce": prepare_nonce,
        "candidate_mapping": copy.deepcopy(candidate_mapping),
        "expected_previous_durable_checkpoint_id": durable_body[
            "oob_high_water_checkpoint_id"
        ],
        "expected_previous_durable_head_id": durable_body[
            "oob_high_water_head_id"
        ],
        "expected_previous_durable_checkpoint_payload_sha256": durable_body[
            "oob_high_water_checkpoint_payload_sha256"
        ],
        "expected_previous_durable_head_payload_sha256": durable_body[
            "oob_high_water_head_payload_sha256"
        ],
        "expected_previous_durable_checkpoint_sequence": durable_body[
            "oob_high_water_checkpoint_sequence"
        ],
        "expected_previous_durable_head_sequence": durable_body[
            "oob_high_water_head_sequence"
        ],
        "expected_previous_durable_tree_root_sha256": durable_body[
            "oob_high_water_tree_root_sha256"
        ],
    }
    q1_request_payload, q1_request = _signed_admission_object(
        artifact_kind="q1_prepare_request",
        schema=admission.Q1_PREPARE_REQUEST_SCHEMA_V1,
        id_field="q1_request_id",
        body=q1_request_body,
        private_key=graph.witness_private_key,
        signer_key_id=bootstrap_policy["witness_key_id"],
        purpose=admission.SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1,
    )
    payloads["q1_request"] = q1_request_payload
    roots["q1_request"] = q1_request

    q1_response_body = {
        **common,
        "schema_version": admission.Q1_PREPARE_RESPONSE_SCHEMA_V1,
        "q1_request_id": q1_request["q1_request_id"],
        "q1_request_payload_sha256": _sha(q1_request_payload),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "observed_current_mapping": copy.deepcopy(candidate_mapping),
        "prepared_candidate_mapping": copy.deepcopy(candidate_mapping),
        "runtime_custody_binding": copy.deepcopy(runtime_binding),
        "prepare_outcome": admission.PREPARE_OUTCOME_V1,
    }
    q1_response_payload, q1_response = _signed_admission_object(
        artifact_kind="q1_prepare_response",
        schema=admission.Q1_PREPARE_RESPONSE_SCHEMA_V1,
        id_field="q1_response_id",
        body=q1_response_body,
        private_key=online_private,
        signer_key_id=online_key_id,
        purpose=admission.SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1,
    )
    payloads["q1_response"] = q1_response_payload
    roots["q1_response"] = q1_response

    a_request_body = {
        **common,
        "schema_version": admission.A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1,
        "q1_request_id": q1_request["q1_request_id"],
        "q1_request_payload_sha256": _sha(q1_request_payload),
        "q1_response_id": q1_response["q1_response_id"],
        "q1_response_payload_sha256": _sha(q1_response_payload),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "expected_current_mapping": copy.deepcopy(genesis_mapping),
        "proposed_current_mapping": copy.deepcopy(candidate_mapping),
        "expected_previous_durable_checkpoint_id": durable_body[
            "oob_high_water_checkpoint_id"
        ],
        "expected_previous_durable_checkpoint_payload_sha256": durable_body[
            "oob_high_water_checkpoint_payload_sha256"
        ],
        "expected_previous_durable_head_id": durable_body[
            "oob_high_water_head_id"
        ],
        "expected_previous_durable_head_payload_sha256": durable_body[
            "oob_high_water_head_payload_sha256"
        ],
        "expected_previous_durable_checkpoint_sequence": durable_body[
            "oob_high_water_checkpoint_sequence"
        ],
        "expected_previous_durable_head_sequence": durable_body[
            "oob_high_water_head_sequence"
        ],
        "expected_previous_durable_tree_root_sha256": durable_body[
            "oob_high_water_tree_root_sha256"
        ],
        "compare_and_append_mode": (
            "ATOMIC_COMPARE_EXACT_OOB_HIGH_WATER_THEN_APPEND_ONCE"
        ),
    }
    a_request_payload, a_request = _signed_admission_object(
        artifact_kind="a_compare_and_append_request",
        schema=admission.A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1,
        id_field="a_request_id",
        body=a_request_body,
        private_key=online_private,
        signer_key_id=online_key_id,
        purpose=admission.SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1,
    )
    payloads["a_request"] = a_request_payload
    roots["a_request"] = a_request

    a_checkpoint_body = {
        **common,
        "schema_version": admission.A_DURABLE_CHECKPOINT_SCHEMA_V1,
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(a_request_payload),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "previous_mapping": copy.deepcopy(genesis_mapping),
        "committed_mapping": copy.deepcopy(candidate_mapping),
        "cas_outcome": admission.CAS_OUTCOME_V1,
        "checkpoint_sequence": 1,
        "previous_a_checkpoint_id": durable_body["oob_high_water_checkpoint_id"],
    }
    a_checkpoint_payload, a_checkpoint = _signed_admission_object(
        artifact_kind="a_durable_checkpoint",
        schema=admission.A_DURABLE_CHECKPOINT_SCHEMA_V1,
        id_field="a_checkpoint_id",
        body=a_checkpoint_body,
        private_key=durable_private,
        signer_key_id=durable_key_id,
        purpose=admission.SIGN_A_DURABLE_CHECKPOINT_PURPOSE_V1,
    )
    payloads["a_checkpoint"] = a_checkpoint_payload
    roots["a_checkpoint"] = a_checkpoint

    a_head_body = {
        **common,
        "schema_version": admission.A_DURABLE_HEAD_SCHEMA_V1,
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(a_request_payload),
        "a_checkpoint_id": a_checkpoint["a_checkpoint_id"],
        "a_checkpoint_payload_sha256": _sha(a_checkpoint_payload),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "committed_mapping": copy.deepcopy(candidate_mapping),
        "cas_outcome": admission.CAS_OUTCOME_V1,
        "head_sequence": 1,
        "previous_a_head_id": durable_body["oob_high_water_head_id"],
    }
    a_head_payload, a_head = _signed_admission_object(
        artifact_kind="a_durable_head",
        schema=admission.A_DURABLE_HEAD_SCHEMA_V1,
        id_field="a_head_id",
        body=a_head_body,
        private_key=durable_private,
        signer_key_id=durable_key_id,
        purpose=admission.SIGN_A_DURABLE_HEAD_PURPOSE_V1,
    )
    payloads["a_head"] = a_head_payload
    roots["a_head"] = a_head

    a_receipt_body = {
        **common,
        "schema_version": admission.A_COMPARE_AND_APPEND_RECEIPT_SCHEMA_V1,
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(a_request_payload),
        "a_checkpoint_id": a_checkpoint["a_checkpoint_id"],
        "a_checkpoint_payload_sha256": _sha(a_checkpoint_payload),
        "a_head_id": a_head["a_head_id"],
        "a_head_payload_sha256": _sha(a_head_payload),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "previous_mapping": copy.deepcopy(genesis_mapping),
        "committed_mapping": copy.deepcopy(candidate_mapping),
        "cas_outcome": admission.CAS_OUTCOME_V1,
        "compare_matched": True,
        "append_applied": True,
        "oob_high_water_checkpoint_id": durable_body[
            "oob_high_water_checkpoint_id"
        ],
        "oob_high_water_head_id": durable_body["oob_high_water_head_id"],
        "durable_before_return": True,
    }
    a_receipt_payload, a_receipt = _signed_admission_object(
        artifact_kind="a_compare_and_append_receipt",
        schema=admission.A_COMPARE_AND_APPEND_RECEIPT_SCHEMA_V1,
        id_field="a_receipt_id",
        body=a_receipt_body,
        private_key=durable_private,
        signer_key_id=durable_key_id,
        purpose=admission.SIGN_A_COMPARE_AND_APPEND_RECEIPT_PURPOSE_V1,
    )
    payloads["a_receipt"] = a_receipt_payload
    roots["a_receipt"] = a_receipt

    q2_request_body = {
        **common,
        "schema_version": admission.Q2_CONFIRM_REQUEST_SCHEMA_V1,
        "q1_response_id": q1_response["q1_response_id"],
        "q1_response_payload_sha256": _sha(q1_response_payload),
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(a_request_payload),
        "a_receipt_id": a_receipt["a_receipt_id"],
        "a_receipt_payload_sha256": _sha(a_receipt_payload),
        "a_checkpoint_id": a_checkpoint["a_checkpoint_id"],
        "a_checkpoint_payload_sha256": _sha(a_checkpoint_payload),
        "a_head_id": a_head["a_head_id"],
        "a_head_payload_sha256": _sha(a_head_payload),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "expected_current_mapping": copy.deepcopy(candidate_mapping),
        "expected_cas_outcome": admission.CAS_OUTCOME_V1,
    }
    q2_request_payload, q2_request = _signed_admission_object(
        artifact_kind="q2_confirm_request",
        schema=admission.Q2_CONFIRM_REQUEST_SCHEMA_V1,
        id_field="q2_request_id",
        body=q2_request_body,
        private_key=graph.witness_private_key,
        signer_key_id=bootstrap_policy["witness_key_id"],
        purpose=admission.SIGN_Q2_CONFIRM_REQUEST_PURPOSE_V1,
    )
    payloads["q2_request"] = q2_request_payload
    roots["q2_request"] = q2_request

    q2_response_body = {
        **common,
        "schema_version": admission.Q2_CONFIRM_RESPONSE_SCHEMA_V1,
        "q2_request_id": q2_request["q2_request_id"],
        "q2_request_payload_sha256": _sha(q2_request_payload),
        "a_receipt_id": a_receipt["a_receipt_id"],
        "a_receipt_payload_sha256": _sha(a_receipt_payload),
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "observed_current_mapping": copy.deepcopy(candidate_mapping),
        "runtime_custody_binding": copy.deepcopy(runtime_binding),
        "observed_cas_outcome": admission.CAS_OUTCOME_V1,
        "confirm_outcome": admission.CONFIRM_OUTCOME_V1,
    }
    q2_response_payload, q2_response = _signed_admission_object(
        artifact_kind="q2_confirm_response",
        schema=admission.Q2_CONFIRM_RESPONSE_SCHEMA_V1,
        id_field="q2_response_id",
        body=q2_response_body,
        private_key=online_private,
        signer_key_id=online_key_id,
        purpose=admission.SIGN_Q2_CONFIRM_RESPONSE_PURPOSE_V1,
    )
    payloads["q2_response"] = q2_response_payload
    roots["q2_response"] = q2_response

    transcript_body = {
        **common,
        "schema_version": admission.ADMISSION_TRANSCRIPT_SCHEMA_V1,
        "prepare_nonce": prepare_nonce,
        "confirmation_nonce": confirmation_nonce,
        "candidate_mapping": copy.deepcopy(candidate_mapping),
        "q1_request_id": q1_request["q1_request_id"],
        "q1_request_payload_sha256": _sha(q1_request_payload),
        "q1_response_id": q1_response["q1_response_id"],
        "q1_response_payload_sha256": _sha(q1_response_payload),
        "a_request_id": a_request["a_request_id"],
        "a_request_payload_sha256": _sha(a_request_payload),
        "a_checkpoint_id": a_checkpoint["a_checkpoint_id"],
        "a_checkpoint_payload_sha256": _sha(a_checkpoint_payload),
        "a_head_id": a_head["a_head_id"],
        "a_head_payload_sha256": _sha(a_head_payload),
        "a_receipt_id": a_receipt["a_receipt_id"],
        "a_receipt_payload_sha256": _sha(a_receipt_payload),
        "q2_request_id": q2_request["q2_request_id"],
        "q2_request_payload_sha256": _sha(q2_request_payload),
        "q2_response_id": q2_response["q2_response_id"],
        "q2_response_payload_sha256": _sha(q2_response_payload),
        "prepare_outcome": admission.PREPARE_OUTCOME_V1,
        "cas_outcome": admission.CAS_OUTCOME_V1,
        "confirm_outcome": admission.CONFIRM_OUTCOME_V1,
        "runtime_custody_binding": copy.deepcopy(runtime_binding),
        "post_q2_logical_time_request_nonce": post_q2_nonce,
        "presented_q1_a_q2_chain_complete": True,
        "runtime_closure_verified": False,
        "key_custody_verified": False,
        "distinct_live_uid_gid_verified": False,
        "live_channel_verified": False,
        "external_currentness_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
    }
    transcript_payload, transcript = _signed_admission_object(
        artifact_kind="admission_transcript",
        schema=admission.ADMISSION_TRANSCRIPT_SCHEMA_V1,
        id_field="admission_transcript_id",
        body=transcript_body,
        private_key=graph.witness_private_key,
        signer_key_id=bootstrap_policy["witness_key_id"],
        purpose=admission.SIGN_ADMISSION_TRANSCRIPT_PURPOSE_V1,
    )
    payloads["transcript"] = transcript_payload
    roots["transcript"] = transcript

    requests, tokens, checkpoints, heads = _append_post_q2_chain(
        logical_prefix,
        graph,
        transcript_id=transcript["admission_transcript_id"],
        transcript_payload=transcript_payload,
        request_nonce=post_q2_nonce,
    )
    request = json.loads(requests[-1])
    token = json.loads(tokens[-1])
    checkpoint = json.loads(checkpoints[-1])
    head = json.loads(heads[-1])
    result_body = {
        **common,
        "schema_version": admission.ADMISSION_CANDIDATE_RESULT_SCHEMA_V1,
        "admission_transcript_id": transcript["admission_transcript_id"],
        "admission_transcript_payload_sha256": _sha(transcript_payload),
        "logical_time_ledger_id": head["ledger_id"],
        "post_commit_logical_time_request_id": request["request_id"],
        "post_commit_logical_time_token_id": token["token_id"],
        "post_commit_logical_time_checkpoint_id": checkpoint["checkpoint_id"],
        "post_commit_logical_time_head_id": head["head_id"],
        "post_commit_logical_time_head_payload_sha256": "sha256:" + _sha(heads[-1]),
        "verified_signature_count": (
            admission.NON_LOGICAL_VERIFIED_SIGNATURE_COUNT_V1 + len(tokens)
        ),
        "verified_signature_scope": admission.VERIFIED_SIGNATURE_SCOPE_V1,
        "presented_transcript_mechanics_verified": True,
        "post_q2_supplied_logical_time_chain_verified": True,
        "runtime_closure_verified": False,
        "key_custody_verified": False,
        "distinct_live_uid_gid_verified": False,
        "live_channel_verified": False,
        "external_currentness_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
        "authority_evidence": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
    }
    result_payload, result_root = _signed_admission_object(
        artifact_kind="admission_candidate_result",
        schema=admission.ADMISSION_CANDIDATE_RESULT_SCHEMA_V1,
        id_field="candidate_result_id",
        body=result_body,
        private_key=graph.witness_private_key,
        signer_key_id=bootstrap_policy["witness_key_id"],
        purpose=admission.SIGN_ADMISSION_CANDIDATE_RESULT_PURPOSE_V1,
    )
    payloads["result"] = result_payload
    roots["result"] = result_root

    call: dict[str, object] = {
        "stage_b_trust_lock_payload": graph.authority.lock_payload,
        "authority_profile_payload": graph.authority.profile_payload,
        "signer_registry_payload": graph.authority.registry_payload,
        "bootstrap_external_witness_policy_payload": payloads["bootstrap_policy"],
        "online_witness_policy_payload": payloads["online_policy"],
        "durable_anchor_policy_payload": payloads["durable_policy"],
        "candidate_anchor_payload": payloads["candidate_anchor"],
        "candidate_receipt_payload": payloads["candidate_receipt"],
        "candidate_ledger_payload": payloads["candidate_ledger"],
        "candidate_checkpoint_payload": payloads["candidate_checkpoint"],
        "candidate_head_payload": payloads["candidate_head"],
        "q1_prepare_request_payload": payloads["q1_request"],
        "q1_prepare_response_payload": payloads["q1_response"],
        "a_compare_and_append_request_payload": payloads["a_request"],
        "a_durable_checkpoint_payload": payloads["a_checkpoint"],
        "a_durable_head_payload": payloads["a_head"],
        "a_compare_and_append_receipt_payload": payloads["a_receipt"],
        "q2_confirm_request_payload": payloads["q2_request"],
        "q2_confirm_response_payload": payloads["q2_response"],
        "admission_transcript_payload": payloads["transcript"],
        "post_commit_logical_time_request_payloads_in_exact_order": requests,
        "post_commit_signed_logical_time_token_payloads_in_exact_order": tokens,
        "post_commit_logical_time_checkpoint_payloads_in_exact_order": checkpoints,
        "post_commit_logical_time_head_payloads_in_exact_order": heads,
        "admission_candidate_result_payload": payloads["result"],
        "out_of_band_expected_lock_id_bytes": graph.authority.lock_id.encode("ascii"),
        "out_of_band_expected_lock_payload_sha256_digest": hashlib.sha256(
            graph.authority.lock_payload
        ).digest(),
        "out_of_band_expected_bootstrap_external_witness_policy_id_bytes": bootstrap_policy[
            "external_witness_policy_id"
        ].encode("ascii"),
        "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest": hashlib.sha256(
            payloads["bootstrap_policy"]
        ).digest(),
        "out_of_band_expected_online_witness_policy_id_bytes": online_root[
            "online_witness_policy_id"
        ].encode("ascii"),
        "out_of_band_expected_online_witness_policy_payload_sha256_digest": hashlib.sha256(
            payloads["online_policy"]
        ).digest(),
        "out_of_band_expected_durable_anchor_policy_id_bytes": durable_root[
            "durable_anchor_policy_id"
        ].encode("ascii"),
        "out_of_band_expected_durable_anchor_policy_payload_sha256_digest": hashlib.sha256(
            payloads["durable_policy"]
        ).digest(),
        "expected_attempt_context_id_bytes": graph.attempt_id.encode("ascii"),
    }
    return _PositiveCorpus(
        graph=graph,
        call=call,
        payloads=payloads,
        roots=roots,
        candidate_mapping=candidate_mapping,
        genesis_mapping=genesis_mapping,
        online_private_key=online_private,
        custody_private_key=custody_private,
        durable_private_key=durable_private,
    )


_SIGNED_PROTOCOL_SPECS = {
    "online_policy": (
        "online_witness_policy",
        admission.ONLINE_WITNESS_POLICY_SCHEMA_V1,
        "online_witness_policy_id",
        admission.SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1,
        "bootstrap",
        "bootstrap_signature",
        "online_witness_policy_payload",
    ),
    "durable_policy": (
        "durable_anchor_policy",
        admission.DURABLE_ANCHOR_POLICY_SCHEMA_V1,
        "durable_anchor_policy_id",
        admission.SIGN_DURABLE_ANCHOR_POLICY_PURPOSE_V1,
        "durable",
        "signature",
        "durable_anchor_policy_payload",
    ),
    "q1_request": (
        "q1_prepare_request",
        admission.Q1_PREPARE_REQUEST_SCHEMA_V1,
        "q1_request_id",
        admission.SIGN_Q1_PREPARE_REQUEST_PURPOSE_V1,
        "bootstrap",
        "signature",
        "q1_prepare_request_payload",
    ),
    "q1_response": (
        "q1_prepare_response",
        admission.Q1_PREPARE_RESPONSE_SCHEMA_V1,
        "q1_response_id",
        admission.SIGN_Q1_PREPARE_RESPONSE_PURPOSE_V1,
        "online",
        "signature",
        "q1_prepare_response_payload",
    ),
    "a_request": (
        "a_compare_and_append_request",
        admission.A_COMPARE_AND_APPEND_REQUEST_SCHEMA_V1,
        "a_request_id",
        admission.SIGN_A_COMPARE_AND_APPEND_REQUEST_PURPOSE_V1,
        "online",
        "signature",
        "a_compare_and_append_request_payload",
    ),
    "a_checkpoint": (
        "a_durable_checkpoint",
        admission.A_DURABLE_CHECKPOINT_SCHEMA_V1,
        "a_checkpoint_id",
        admission.SIGN_A_DURABLE_CHECKPOINT_PURPOSE_V1,
        "durable",
        "signature",
        "a_durable_checkpoint_payload",
    ),
    "a_head": (
        "a_durable_head",
        admission.A_DURABLE_HEAD_SCHEMA_V1,
        "a_head_id",
        admission.SIGN_A_DURABLE_HEAD_PURPOSE_V1,
        "durable",
        "signature",
        "a_durable_head_payload",
    ),
    "a_receipt": (
        "a_compare_and_append_receipt",
        admission.A_COMPARE_AND_APPEND_RECEIPT_SCHEMA_V1,
        "a_receipt_id",
        admission.SIGN_A_COMPARE_AND_APPEND_RECEIPT_PURPOSE_V1,
        "durable",
        "signature",
        "a_compare_and_append_receipt_payload",
    ),
    "q2_request": (
        "q2_confirm_request",
        admission.Q2_CONFIRM_REQUEST_SCHEMA_V1,
        "q2_request_id",
        admission.SIGN_Q2_CONFIRM_REQUEST_PURPOSE_V1,
        "bootstrap",
        "signature",
        "q2_confirm_request_payload",
    ),
    "q2_response": (
        "q2_confirm_response",
        admission.Q2_CONFIRM_RESPONSE_SCHEMA_V1,
        "q2_response_id",
        admission.SIGN_Q2_CONFIRM_RESPONSE_PURPOSE_V1,
        "online",
        "signature",
        "q2_confirm_response_payload",
    ),
    "transcript": (
        "admission_transcript",
        admission.ADMISSION_TRANSCRIPT_SCHEMA_V1,
        "admission_transcript_id",
        admission.SIGN_ADMISSION_TRANSCRIPT_PURPOSE_V1,
        "bootstrap",
        "signature",
        "admission_transcript_payload",
    ),
    "result": (
        "admission_candidate_result",
        admission.ADMISSION_CANDIDATE_RESULT_SCHEMA_V1,
        "candidate_result_id",
        admission.SIGN_ADMISSION_CANDIDATE_RESULT_PURPOSE_V1,
        "bootstrap",
        "signature",
        "admission_candidate_result_payload",
    ),
}


def _signer(
    corpus: _PositiveCorpus, name: str, body: Mapping[str, object]
) -> tuple[Ed25519PrivateKey, str]:
    signer = _SIGNED_PROTOCOL_SPECS[name][4]
    online_body = corpus.roots["online_policy"]["body"]
    durable_body = corpus.roots["durable_policy"]["body"]
    bootstrap = corpus.roots["bootstrap_policy"]
    if signer == "bootstrap":
        return corpus.graph.witness_private_key, bootstrap["witness_key_id"]
    if signer == "online":
        return corpus.online_private_key, online_body["online_key_id"]
    if name == "durable_policy":
        return corpus.durable_private_key, body["anchor_key_id"]
    return corpus.durable_private_key, durable_body["anchor_key_id"]


def _resigned_call(
    corpus: _PositiveCorpus,
    name: str,
    mutate: Callable[[dict[str, object]], None],
    *,
    private_key_override: Ed25519PrivateKey | None = None,
) -> dict[str, object]:
    root = copy.deepcopy(dict(corpus.roots[name]))
    body = root["body"]
    mutate(body)
    private_key, signer_key_id = _signer(corpus, name, body)
    if private_key_override is not None:
        private_key = private_key_override
    kind, schema, id_field, purpose, _, signature_field, call_key = (
        _SIGNED_PROTOCOL_SPECS[name]
    )
    payload, replacement = _signed_admission_object(
        artifact_kind=kind,
        schema=schema,
        id_field=id_field,
        body=body,
        private_key=private_key,
        signer_key_id=signer_key_id,
        purpose=purpose,
        signature_field=signature_field,
    )
    call = dict(corpus.call)
    call[call_key] = payload
    if name == "online_policy":
        call["out_of_band_expected_online_witness_policy_id_bytes"] = replacement[
            "online_witness_policy_id"
        ].encode("ascii")
        call[
            "out_of_band_expected_online_witness_policy_payload_sha256_digest"
        ] = hashlib.sha256(payload).digest()
    elif name == "durable_policy":
        call["out_of_band_expected_durable_anchor_policy_id_bytes"] = replacement[
            "durable_anchor_policy_id"
        ].encode("ascii")
        call[
            "out_of_band_expected_durable_anchor_policy_payload_sha256_digest"
        ] = hashlib.sha256(payload).digest()
    return call


def _bootstrap_policy_call(
    corpus: _PositiveCorpus, mutate: Callable[[dict[str, object]], None]
) -> dict[str, object]:
    policy = copy.deepcopy(dict(corpus.roots["bootstrap_policy"]))
    policy.pop("external_witness_policy_id")
    mutate(policy)
    policy_id = "phase2b_step5_external_witness_policy_v1_" + hashlib.sha256(
        b"HEGEL/PHASE2B/STEP5/EXTERNAL_WITNESS_POLICY/V1\0"
        + admission.canonical_external_witness_admission_json_bytes_v1(policy)
    ).hexdigest()
    policy["external_witness_policy_id"] = policy_id
    payload = admission.canonical_external_witness_admission_json_bytes_v1(policy)
    call = dict(corpus.call)
    call["bootstrap_external_witness_policy_payload"] = payload
    call["out_of_band_expected_bootstrap_external_witness_policy_id_bytes"] = (
        policy_id.encode("ascii")
    )
    call[
        "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest"
    ] = hashlib.sha256(payload).digest()
    return call


def _different_exact_bytes(value: bytes) -> bytes:
    replacement = b"0" if value[-1:] != b"0" else b"1"
    return value[:-1] + replacement


def _assert_rejected(call: Mapping[str, object]) -> None:
    with pytest.raises(
        (
            admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1,
            authority.Phase2BStep5AuthorityErrorV2,
        )
    ):
        admission.verify_external_witness_admission_transcript_candidate_v1(**call)


def test_current_mapping_and_distinct_durable_policy_contract_are_frozen() -> None:
    assert {
        "attempt_context_id",
        "anchor_sequence",
        "effective_state",
        "superseded",
        "revoked",
        "map_leaf_index",
        "map_tree_size",
        "map_root_sha256",
        "map_inclusion_path_sha256_in_exact_order",
        "presented_to_live_consistency_path_sha256_in_exact_order",
    } <= admission.DURABLE_MAPPING_FIELDS_V1
    assert admission.DURABLE_ANCHOR_POLICY_FIELDS_V1 == {
        "schema_version",
        "body",
        "durable_anchor_policy_id",
        "signature",
    }
    assert {
        "bootstrap_external_witness_policy_id",
        "genesis_checkpoint_id",
        "genesis_head_id",
        "genesis_tree_root_sha256",
        "oob_high_water_checkpoint_id",
        "oob_high_water_head_id",
        "oob_high_water_mapping",
        "admission_slot_key_profile",
        "compare_and_append_semantics",
        "durable_before_return_required",
    } <= admission.DURABLE_ANCHOR_POLICY_BODY_FIELDS_V1


def test_unified_raw_positive_corpus_is_mechanics_only(tmp_path: Path) -> None:
    corpus = _positive_corpus(tmp_path)
    result = admission.verify_external_witness_admission_transcript_candidate_v1(
        **corpus.call
    )
    assert admission.require_verified_external_witness_admission_candidate_v1(
        result
    ) is result
    assert result.verified_signature_count == (
        admission.NON_LOGICAL_VERIFIED_SIGNATURE_COUNT_V1 + 2
    )
    assert result.verified_signature_scope == admission.VERIFIED_SIGNATURE_SCOPE_V1
    assert result.verified_protocol_object_count == (
        admission.VERIFIED_PROTOCOL_OBJECT_COUNT_V1
    )
    assert result.raw_authority_graph_verified is True
    assert result.three_oob_policy_pins_verified is True
    assert result.key_sets_pairwise_disjoint is True
    assert result.presented_q1_a_q2_chain_verified is True
    assert result.supplied_post_q2_logical_time_chain_verified is True
    for field in (
        "private_key_present",
        "spool_path_present",
        "decoded_mapping_input_present",
        "callback_present",
        "caller_nonce_present",
        "runtime_closure_verified",
        "key_custody_verified",
        "distinct_live_uid_gid_verified",
        "live_channel_verified",
        "external_currentness_verified",
        "durability_verified",
        "non_equivocation_verified",
        "global_anti_fork_verified",
        "authority_evidence",
        "r01_satisfied",
        "readiness_evidence",
        "execution_authorized",
    ):
        assert getattr(result, field) is False


def test_public_verifier_is_raw_only_and_rejects_decoded_payload(tmp_path: Path) -> None:
    corpus = _positive_corpus(tmp_path)
    signature = inspect.signature(
        admission.verify_external_witness_admission_transcript_candidate_v1
    )
    assert "expected_attempt_context_id_bytes" in signature.parameters
    for forbidden in (
        "private_key",
        "spool",
        "path",
        "callback",
        "decoded",
        "caller_nonce",
    ):
        assert all(forbidden not in name for name in signature.parameters)
    call = dict(corpus.call)
    call["q1_prepare_request_payload"] = json.loads(
        corpus.payloads["q1_request"]
    )
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1
    ) as caught:
        admission.verify_external_witness_admission_transcript_candidate_v1(**call)
    assert caught.value.code == admission.FAIL_INPUT_V1


@pytest.mark.parametrize(
    "field",
    [
        "out_of_band_expected_lock_id_bytes",
        "out_of_band_expected_lock_payload_sha256_digest",
        "out_of_band_expected_bootstrap_external_witness_policy_id_bytes",
        "out_of_band_expected_bootstrap_external_witness_policy_payload_sha256_digest",
        "out_of_band_expected_online_witness_policy_id_bytes",
        "out_of_band_expected_online_witness_policy_payload_sha256_digest",
        "out_of_band_expected_durable_anchor_policy_id_bytes",
        "out_of_band_expected_durable_anchor_policy_payload_sha256_digest",
        "expected_attempt_context_id_bytes",
    ],
)
def test_every_raw_oob_pin_and_attempt_bytes_fail_closed(
    tmp_path: Path, field: str
) -> None:
    corpus = _positive_corpus(tmp_path)
    call = dict(corpus.call)
    if field == "expected_attempt_context_id_bytes":
        call[field] = corpus.graph.attempt_id
    else:
        call[field] = _different_exact_bytes(call[field])
    _assert_rejected(call)


@pytest.mark.parametrize(
    "field",
    [
        "bootstrap_external_witness_policy_payload",
        "online_witness_policy_payload",
        "durable_anchor_policy_payload",
    ],
)
def test_all_three_policy_payloads_reject_decoded_objects(
    tmp_path: Path, field: str
) -> None:
    corpus = _positive_corpus(tmp_path)
    call = dict(corpus.call)
    call[field] = json.loads(call[field])
    _assert_rejected(call)


@pytest.mark.parametrize("dimension", ["public-key", "key-id", "control-domain"])
def test_bootstrap_policy_rejects_authority_collisions(
    tmp_path: Path, dimension: str
) -> None:
    corpus = _positive_corpus(tmp_path)
    role = corpus.graph.authority.role_keys[-1]

    def collide(policy: dict[str, object]) -> None:
        if dimension == "public-key":
            policy["public_key_hex"] = role["public_key_hex"]
        elif dimension == "key-id":
            policy["witness_key_id"] = role["key_id"]
        else:
            policy["witness_control_domain_id"] = role["control_domain"]

    _assert_rejected(_bootstrap_policy_call(corpus, collide))


@pytest.mark.parametrize(
    "collision",
    [
        "online-key-id-authority",
        "online-public-authority",
        "custody-key-online",
        "custody-public-authority",
        "control-domain-authority",
        "service-binding-authority",
        "uid-authority",
        "uid-equals-gid",
        "runtime-policy-authority",
    ],
)
def test_online_and_custody_policy_reject_identity_collisions(
    tmp_path: Path, collision: str
) -> None:
    corpus = _positive_corpus(tmp_path)
    role = corpus.graph.authority.role_keys[-1]
    service = corpus.graph.authority.service_bindings[0]

    def collide(body: dict[str, object]) -> None:
        if collision == "online-key-id-authority":
            body["online_key_id"] = role["key_id"]
        elif collision == "online-public-authority":
            body["online_public_key_hex"] = role["public_key_hex"]
        elif collision == "custody-key-online":
            body["custody_attestor_key_id"] = body["online_key_id"]
        elif collision == "custody-public-authority":
            body["custody_attestor_public_key_hex"] = role["public_key_hex"]
        elif collision == "control-domain-authority":
            body["witness_control_domain_id"] = role["control_domain"]
        elif collision == "service-binding-authority":
            body["service_binding_id"] = service["binding_id"]
        elif collision == "uid-authority":
            body["service_uid"] = service["uid"]
        elif collision == "uid-equals-gid":
            body["service_uid"] = body["service_gid"]
        else:
            body["runtime_closure_policy_payload_sha256"] = service[
                "static_runtime_closure_policy_sha256"
            ]

    _assert_rejected(_resigned_call(corpus, "online_policy", collide))


@pytest.mark.parametrize(
    "collision",
    [
        "key-id-online",
        "public-online",
        "control-online",
        "namespace-bootstrap",
        "namespace-authority-role",
    ],
)
def test_durable_a_policy_rejects_online_and_authority_collisions(
    tmp_path: Path, collision: str
) -> None:
    corpus = _positive_corpus(tmp_path)
    online = corpus.roots["online_policy"]["body"]
    bootstrap = corpus.roots["bootstrap_policy"]

    def collide(body: dict[str, object]) -> None:
        if collision == "key-id-online":
            body["anchor_key_id"] = online["online_key_id"]
        elif collision == "public-online":
            body["anchor_public_key_hex"] = online["online_public_key_hex"]
        elif collision == "control-online":
            body["anchor_control_domain_id"] = online["witness_control_domain_id"]
        elif collision == "namespace-bootstrap":
            body["anchor_namespace"] = bootstrap["witness_namespace"]
        else:
            body["anchor_namespace"] = authority.ROLE_ORDER_V2[-1]

    override = corpus.online_private_key if collision == "public-online" else None
    _assert_rejected(
        _resigned_call(
            corpus,
            "durable_policy",
            collide,
            private_key_override=override,
        )
    )


@pytest.mark.parametrize(
    "case",
    [
        "q1-attempt",
        "q1-effective-state",
        "q1-superseded",
        "q1-revoked",
        "q1-map-root",
        "q1-inclusion-proof",
        "q1-consistency-proof",
        "q1-response-nonce-collision",
        "q1-runtime-uid",
        "a-current-map",
        "a-high-water",
        "a-checkpoint-sequence",
        "a-checkpoint-predecessor",
        "a-head-sequence",
        "a-head-predecessor",
        "a-receipt-cas",
        "a-receipt-append",
        "q2-expected-map",
        "q2-observed-revoked",
        "transcript-nonce-collision",
    ],
)
def test_resigned_q1_a_q2_mapping_nonce_highwater_and_cas_tampering_fails(
    tmp_path: Path, case: str
) -> None:
    corpus = _positive_corpus(tmp_path)
    object_name = {
        "q1-attempt": "q1_request",
        "q1-effective-state": "q1_request",
        "q1-superseded": "q1_request",
        "q1-revoked": "q1_request",
        "q1-map-root": "q1_request",
        "q1-inclusion-proof": "q1_request",
        "q1-consistency-proof": "q1_request",
        "q1-response-nonce-collision": "q1_response",
        "q1-runtime-uid": "q1_response",
        "a-current-map": "a_request",
        "a-high-water": "a_request",
        "a-checkpoint-sequence": "a_checkpoint",
        "a-checkpoint-predecessor": "a_checkpoint",
        "a-head-sequence": "a_head",
        "a-head-predecessor": "a_head",
        "a-receipt-cas": "a_receipt",
        "a-receipt-append": "a_receipt",
        "q2-expected-map": "q2_request",
        "q2-observed-revoked": "q2_response",
        "transcript-nonce-collision": "transcript",
    }[case]

    def tamper(body: dict[str, object]) -> None:
        if case == "q1-attempt":
            body["attempt_context_id"] = _oid("foreign_attempt", 1)
        elif case == "q1-effective-state":
            body["candidate_mapping"]["effective_state"] = "REVOKED"
        elif case == "q1-superseded":
            body["candidate_mapping"]["superseded"] = True
        elif case == "q1-revoked":
            body["candidate_mapping"]["revoked"] = True
        elif case == "q1-map-root":
            body["candidate_mapping"]["map_root_sha256"] = _hex(1)
        elif case == "q1-inclusion-proof":
            body["candidate_mapping"][
                "map_inclusion_path_sha256_in_exact_order"
            ] = [_hex(2)]
        elif case == "q1-consistency-proof":
            body["candidate_mapping"][
                "presented_to_live_consistency_path_sha256_in_exact_order"
            ] = [_hex(3)]
        elif case == "q1-response-nonce-collision":
            body["confirmation_nonce"] = body["prepare_nonce"]
        elif case == "q1-runtime-uid":
            body["runtime_custody_binding"]["peer_uid"] += 1
        elif case == "a-current-map":
            body["expected_current_mapping"] = copy.deepcopy(
                corpus.candidate_mapping
            )
        elif case == "a-high-water":
            body["expected_previous_durable_checkpoint_id"] = _oid(
                "wrong_checkpoint", 1
            )
        elif case == "a-checkpoint-sequence":
            body["checkpoint_sequence"] = 2
        elif case == "a-checkpoint-predecessor":
            body["previous_a_checkpoint_id"] = _oid("wrong_checkpoint", 2)
        elif case == "a-head-sequence":
            body["head_sequence"] = 2
        elif case == "a-head-predecessor":
            body["previous_a_head_id"] = _oid("wrong_head", 2)
        elif case == "a-receipt-cas":
            body["cas_outcome"] = "COMPARE_DID_NOT_MATCH"
        elif case == "a-receipt-append":
            body["append_applied"] = False
        elif case == "q2-expected-map":
            body["expected_current_mapping"] = copy.deepcopy(
                corpus.genesis_mapping
            )
        elif case == "q2-observed-revoked":
            body["observed_current_mapping"]["revoked"] = True
        else:
            body["post_q2_logical_time_request_nonce"] = body["prepare_nonce"]

    _assert_rejected(_resigned_call(corpus, object_name, tamper))


@pytest.mark.parametrize(
    ("name", "field", "hostile"),
    [
        ("result", "live_channel_verified", 0),
        ("result", "presented_transcript_mechanics_verified", 1),
        ("transcript", "global_anti_fork_verified", 0),
        ("a_receipt", "compare_matched", 1),
    ],
)
def test_json_zero_and_one_cannot_launder_protocol_booleans(
    tmp_path: Path, name: str, field: str, hostile: int
) -> None:
    corpus = _positive_corpus(tmp_path)
    _assert_rejected(
        _resigned_call(
            corpus,
            name,
            lambda body: body.__setitem__(field, hostile),
        )
    )


@pytest.mark.parametrize("name", ["transcript", "result"])
@pytest.mark.parametrize(
    "field",
    [
        "runtime_closure_verified",
        "key_custody_verified",
        "distinct_live_uid_gid_verified",
    ],
)
@pytest.mark.parametrize("hostile", [True, 0])
def test_live_runtime_custody_and_distinct_uid_gid_claims_are_exact_false(
    tmp_path: Path, name: str, field: str, hostile: object
) -> None:
    corpus = _positive_corpus(tmp_path)
    _assert_rejected(
        _resigned_call(
            corpus,
            name,
            lambda body: body.__setitem__(field, hostile),
        )
    )


def test_verified_signature_scope_is_exact_and_content_bound(tmp_path: Path) -> None:
    corpus = _positive_corpus(tmp_path)
    _assert_rejected(
        _resigned_call(
            corpus,
            "result",
            lambda body: body.__setitem__(
                "verified_signature_scope",
                admission.VERIFIED_SIGNATURE_SCOPE_V1 + "_DRIFT",
            ),
        )
    )


@pytest.mark.parametrize(
    "case",
    [
        "checkpoint-id-none",
        "checkpoint-sequence",
        "head-sequence",
        "tree-root",
        "mapping-not-genesis",
        "purpose-order",
    ],
)
def test_resigned_durable_policy_requires_exact_genesis_high_water(
    tmp_path: Path, case: str
) -> None:
    corpus = _positive_corpus(tmp_path)

    def tamper(body: dict[str, object]) -> None:
        if case == "checkpoint-id-none":
            body["oob_high_water_checkpoint_id"] = None
        elif case == "checkpoint-sequence":
            body["oob_high_water_checkpoint_sequence"] = 1
        elif case == "head-sequence":
            body["oob_high_water_head_sequence"] = 1
        elif case == "tree-root":
            body["oob_high_water_tree_root_sha256"] = _hex(1)
        elif case == "mapping-not-genesis":
            body["oob_high_water_mapping"] = copy.deepcopy(
                corpus.candidate_mapping
            )
        else:
            body["authorized_signature_purposes_in_exact_order"] = list(
                reversed(admission.A_PURPOSES_IN_EXACT_ORDER_V1)
            )

    _assert_rejected(_resigned_call(corpus, "durable_policy", tamper))


def test_complete_time_chain_and_anchor_penultimate_binding_are_exact(
    tmp_path: Path,
) -> None:
    corpus = _positive_corpus(tmp_path)
    requests = corpus.call[
        "post_commit_logical_time_request_payloads_in_exact_order"
    ]
    tokens = corpus.call[
        "post_commit_signed_logical_time_token_payloads_in_exact_order"
    ]
    checkpoints = corpus.call[
        "post_commit_logical_time_checkpoint_payloads_in_exact_order"
    ]
    heads = corpus.call["post_commit_logical_time_head_payloads_in_exact_order"]
    assert tuple(map(len, (requests, tokens, checkpoints, heads))) == (2, 2, 2, 3)
    anchor_body = corpus.roots["candidate_anchor"]["body"]
    assert json.loads(tokens[-2])["token_id"] == anchor_body["current_cut_token_id"]
    assert "sha256:" + _sha(tokens[-2]) == anchor_body[
        "current_cut_token_payload_sha256"
    ]
    assert json.loads(checkpoints[-2])["checkpoint_id"] == anchor_body[
        "current_cut_checkpoint_id"
    ]
    assert "sha256:" + _sha(checkpoints[-2]) == anchor_body[
        "current_cut_checkpoint_payload_sha256"
    ]
    assert json.loads(heads[-2])["head_id"] == anchor_body["current_cut_head_id"]
    assert "sha256:" + _sha(heads[-2]) == anchor_body[
        "current_cut_head_payload_sha256"
    ]


@pytest.mark.parametrize(
    "case",
    [
        "requests-list",
        "drop-pre-cut-request",
        "drop-pre-cut-token",
        "reverse-checkpoints",
        "drop-genesis-head",
        "reverse-requests",
    ],
)
def test_logical_time_raw_collections_require_complete_exact_ordered_tuples(
    tmp_path: Path, case: str
) -> None:
    corpus = _positive_corpus(tmp_path)
    call = dict(corpus.call)
    request_key = "post_commit_logical_time_request_payloads_in_exact_order"
    token_key = "post_commit_signed_logical_time_token_payloads_in_exact_order"
    checkpoint_key = "post_commit_logical_time_checkpoint_payloads_in_exact_order"
    head_key = "post_commit_logical_time_head_payloads_in_exact_order"
    if case == "requests-list":
        call[request_key] = list(call[request_key])
    elif case == "drop-pre-cut-request":
        call[request_key] = call[request_key][1:]
    elif case == "drop-pre-cut-token":
        call[token_key] = call[token_key][1:]
    elif case == "reverse-checkpoints":
        call[checkpoint_key] = tuple(reversed(call[checkpoint_key]))
    elif case == "drop-genesis-head":
        call[head_key] = call[head_key][1:]
    else:
        call[request_key] = tuple(reversed(call[request_key]))
    _assert_rejected(call)


def test_valid_same_authority_penultimate_chain_splice_is_rejected(
    tmp_path: Path,
) -> None:
    corpus = _positive_corpus(tmp_path)
    foreign_prefix = _build_pre_cut_chain(
        tmp_path / "foreign-penultimate-prefix",
        corpus.graph,
        request_nonce="e" * 64,
    )
    transcript = corpus.roots["transcript"]
    requests, tokens, checkpoints, heads = _append_post_q2_chain(
        foreign_prefix,
        corpus.graph,
        transcript_id=transcript["admission_transcript_id"],
        transcript_payload=corpus.payloads["transcript"],
        request_nonce=transcript["body"]["post_q2_logical_time_request_nonce"],
    )
    call = dict(corpus.call)
    call["post_commit_logical_time_request_payloads_in_exact_order"] = requests
    call["post_commit_signed_logical_time_token_payloads_in_exact_order"] = tokens
    call["post_commit_logical_time_checkpoint_payloads_in_exact_order"] = checkpoints
    call["post_commit_logical_time_head_payloads_in_exact_order"] = heads
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1
    ) as caught:
        admission.verify_external_witness_admission_transcript_candidate_v1(**call)
    assert caught.value.code == admission.FAIL_LOGICAL_TIME_V1


def test_single_event_genesis_fork_cannot_replace_pre_cut_ancestry(
    tmp_path: Path,
) -> None:
    corpus = _positive_corpus(tmp_path)
    transcript = corpus.roots["transcript"]
    fork = _build_pre_cut_chain(
        tmp_path / "single-event-genesis-fork",
        corpus.graph,
        request_nonce=transcript["body"]["post_q2_logical_time_request_nonce"],
        purpose=admission.POST_Q2_LOGICAL_TIME_PURPOSE_V1,
        bound_artifact_id=transcript["admission_transcript_id"],
        bound_artifact_sha256="sha256:" + _sha(corpus.payloads["transcript"]),
    )
    call = dict(corpus.call)
    call["post_commit_logical_time_request_payloads_in_exact_order"] = fork.requests
    call["post_commit_signed_logical_time_token_payloads_in_exact_order"] = fork.tokens
    call["post_commit_logical_time_checkpoint_payloads_in_exact_order"] = (
        fork.checkpoints
    )
    call["post_commit_logical_time_head_payloads_in_exact_order"] = fork.heads
    _assert_rejected(call)


@pytest.mark.parametrize(
    "name",
    ["q1_request", "q1_response", "a_checkpoint", "q2_response", "transcript"],
)
def test_same_kind_cross_corpus_protocol_object_swaps_fail_closed(
    tmp_path: Path, name: str
) -> None:
    local_root = tmp_path / "local"
    foreign_root = tmp_path / "foreign"
    local_root.mkdir()
    foreign_root.mkdir()
    local = _positive_corpus(local_root, salt=0)
    foreign = _positive_corpus(foreign_root, salt=1)
    call_key = _SIGNED_PROTOCOL_SPECS[name][-1]
    call = dict(local.call)
    call[call_key] = foreign.payloads[name]
    _assert_rejected(call)


def test_fully_readdressed_bad_signature_survives_global_verifier_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _positive_corpus(tmp_path)
    root = copy.deepcopy(dict(corpus.roots["q1_request"]))
    root["body"]["prepare_nonce"] = "f" * 64
    root["q1_request_id"] = (
        admission.phase2b_step5_external_witness_admission_content_id_v1(
            "q1_prepare_request", root["body"]
        )
    )
    invalid_payload = admission.canonical_external_witness_admission_json_bytes_v1(
        root
    )
    call = dict(corpus.call)
    call["q1_prepare_request_payload"] = invalid_payload
    monkeypatch.setattr(admission, "_verify_ed25519", lambda **kwargs: None)
    monkeypatch.setattr(admission, "_signature", lambda *args, **kwargs: {})
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1
    ) as caught:
        admission.verify_external_witness_admission_transcript_candidate_v1(**call)
    assert caught.value.code == admission.FAIL_SIGNATURE_V1


def _bomb(*args: object, **kwargs: object) -> object:
    raise AssertionError("EVIL post-import global was consulted")


def test_positive_verifier_uses_definition_captured_function_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _positive_corpus(tmp_path)
    for name in (
        "_decode",
        "_canonical",
        "_load_raw_authority_graph",
        "_role_keys",
        "_load_bootstrap_policy",
        "_load_online_policy",
        "_load_durable_policy",
        "_load_candidate_artifacts",
        "_load_signed_protocol_object",
        "_common_body",
        "_mapping",
        "_runtime_custody_binding",
        "_verify_logical_replay",
        "_require_logical_replay",
        "_verify_ed25519",
        "_signature",
        "phase2b_step5_external_witness_admission_content_id_v1",
        "phase2b_step5_external_witness_admission_signature_preimage_v1",
    ):
        monkeypatch.setattr(admission, name, _bomb)
    result = admission.verify_external_witness_admission_transcript_candidate_v1(
        **corpus.call
    )
    assert result.presented_q1_a_q2_chain_verified is True


def test_clean_import_blocks_producer_store_owner_runtime_and_service_modules() -> None:
    source_root = str(Path(admission.__file__).resolve().parents[1])
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {
            "hegel_machine.phase2b_formal_wire_covert_step5_logical_time_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_r01_external_witness_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_owner_status_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_runtime_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_service_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_store_v1",
        }

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked:
                    raise RuntimeError("blocked producer import: " + fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        import hegel_machine.phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1
        assert blocked.isdisjoint(sys.modules)
        print("CLEAN_IMPORT_OK")
        """
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = source_root
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CLEAN_IMPORT_OK"


def test_verified_result_has_private_lineage_and_detects_mutation(
    tmp_path: Path,
) -> None:
    corpus = _positive_corpus(tmp_path)
    result = admission.verify_external_witness_admission_transcript_candidate_v1(
        **corpus.call
    )
    with pytest.raises(TypeError):
        admission.VerifiedExternalWitnessAdmissionCandidateV1()
    forged = object.__new__(admission.VerifiedExternalWitnessAdmissionCandidateV1)
    for field in result.__dataclass_fields__:
        object.__setattr__(forged, field, getattr(result, field))
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1
    ) as forged_error:
        admission.require_verified_external_witness_admission_candidate_v1(forged)
    assert forged_error.value.code == admission.FAIL_RESULT_V1
    try:
        copied = copy.copy(result)
    except TypeError:
        copied = None
    if copied is not None and copied is not result:
        with pytest.raises(
            admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1
        ) as copy_error:
            admission.require_verified_external_witness_admission_candidate_v1(
                copied
            )
        assert copy_error.value.code == admission.FAIL_RESULT_V1
    object.__setattr__(result, "authority_evidence", True)
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1
    ) as mutation_error:
        admission.require_verified_external_witness_admission_candidate_v1(result)
    assert mutation_error.value.code == admission.FAIL_RESULT_V1


def test_real_admission_remains_not_implemented() -> None:
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV1
    ) as caught:
        admission.verify_real_external_witness_admission_v1()
    assert caught.value.code == admission.FAIL_NOT_IMPLEMENTED_V1
