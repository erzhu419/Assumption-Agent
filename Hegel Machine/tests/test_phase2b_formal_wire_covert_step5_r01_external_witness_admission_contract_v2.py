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
from types import FunctionType
from typing import Callable, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hegel_machine import phase2b_formal_wire_covert_step5_authority_v2 as authority
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v1
    as admission_v1,
)
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v2
    as admission,
)
from hegel_machine import (
    phase2b_formal_wire_covert_step5_r01_live_runtime_overlay_v2 as overlay,
)
from hegel_machine import phase2b_formal_wire_covert_step5_runtime_closure_v1 as closure
from hegel_machine import phase2b_formal_wire_covert_step5_store_replay_core_v2 as replay
from hegel_machine import phase2b_formal_wire_covert_step5_store_v2 as store

import test_phase2b_formal_wire_covert_step5_r01_live_runtime_overlay_v2 as overlay_fixture


_COLLECTION_B480_VERIFY = (
    admission_v1.verify_external_witness_admission_transcript_candidate_v1
)
_COLLECTION_B480_REQUIRE = (
    admission_v1.require_verified_external_witness_admission_candidate_v1
)
_COLLECTION_CLOSURE_VERIFY = closure.verify_offline_step5_image_closure_mechanics_v1
_COLLECTION_CLOSURE_REQUIRE = (
    closure.require_verified_offline_step5_image_closure_mechanics_v1
)
_COLLECTION_REPLAY_VERIFY = replay.verify_cas_store_complete_session_replay_core_v2
_COLLECTION_REPLAY_REQUIRE = (
    replay.require_verified_cas_store_complete_session_replay_core_v2
)


@dataclass(frozen=True)
class _CasEvidence:
    plan: bytes
    requests: tuple[bytes, ...]
    objects: tuple[bytes, ...]
    receipts: tuple[bytes, ...]
    checkpoints: tuple[bytes, ...]
    heads: tuple[bytes, ...]
    responses: tuple[bytes, ...]


@dataclass(frozen=True)
class _AdmissionV2Corpus:
    call: dict[str, object]
    payloads: Mapping[str, bytes]
    roots: Mapping[str, Mapping[str, object]]
    base: object
    complete: Mapping[str, object]
    source_private: Ed25519PrivateKey
    pure_private: Ed25519PrivateKey


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _store_sha(payload: bytes) -> str:
    return "sha256:" + _sha(payload)


def _json(payload: bytes) -> dict[str, object]:
    value = json.loads(payload)
    assert type(value) is dict
    return value


def _private_seed(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )


def _authority_signature(
    *,
    kind: str,
    body: Mapping[str, object],
    row: Mapping[str, object],
    private_key: Ed25519PrivateKey,
    purpose: str,
) -> dict[str, object]:
    preimage = authority.phase2b_step5_signature_preimage_v2(
        artifact_kind=kind,
        body=dict(body),
        role=row["role"],
        purpose=purpose,
        signer_key_id=row["key_id"],
        key_epoch=row["key_epoch"],
    )
    return {
        "schema_version": authority.DETACHED_SIGNATURE_SCHEMA_V2,
        "algorithm": admission.ALGORITHM_V2,
        "role": row["role"],
        "purpose": purpose,
        "key_epoch": row["key_epoch"],
        "signer_key_id": row["key_id"],
        "signature_hex": private_key.sign(preimage).hex(),
    }


def _external_signature(
    *,
    kind: str,
    body: Mapping[str, object],
    content_id: str,
    private_key: Ed25519PrivateKey,
    key_id: str,
    key_epoch: int,
    purpose: str,
) -> dict[str, object]:
    preimage = admission.phase2b_step5_external_witness_admission_signature_preimage_v2(
        artifact_kind=kind,
        body=body,
        content_id=content_id,
        signer_key_id=key_id,
        key_epoch=key_epoch,
        purpose=purpose,
    )
    return {
        "schema_version": (
            "hegel-machine-phase2b-step5-external-witness-admission-signature/2"
        ),
        "algorithm": admission.ALGORITHM_V2,
        "signer_key_id": key_id,
        "key_epoch": key_epoch,
        "purpose": purpose,
        "signature_hex": private_key.sign(preimage).hex(),
    }


def _signed(
    kind: str,
    body: Mapping[str, object],
    signers: tuple[
        tuple[str, Ed25519PrivateKey, Mapping[str, object] | str, int, str], ...
    ],
) -> tuple[bytes, dict[str, object]]:
    exact_body = copy.deepcopy(dict(body))
    assert frozenset(exact_body) == admission._BODY_FIELDS_BY_KIND[kind]
    content_id = admission.phase2b_step5_external_witness_admission_content_id_v2(
        kind, exact_body
    )
    signatures: list[dict[str, object]] = []
    for mode, private_key, signer, key_epoch, purpose in signers:
        if mode == "authority":
            assert isinstance(signer, Mapping)
            signatures.append(
                _authority_signature(
                    kind=kind,
                    body=exact_body,
                    row=signer,
                    private_key=private_key,
                    purpose=purpose,
                )
            )
        else:
            assert type(signer) is str
            signatures.append(
                _external_signature(
                    kind=kind,
                    body=exact_body,
                    content_id=content_id,
                    private_key=private_key,
                    key_id=signer,
                    key_epoch=key_epoch,
                    purpose=purpose,
                )
            )
    root = {
        "schema_version": admission._SCHEMA_BY_KIND[kind],
        "body": exact_body,
        admission._ID_FIELD_BY_KIND[kind]: content_id,
        "signatures_in_exact_order": signatures,
    }
    assert frozenset(root) == admission._TOP_FIELDS_BY_KIND[kind]
    return admission.canonical_external_witness_admission_json_bytes_v2(root), root


def _role(
    corpus: object, role: str
) -> tuple[Mapping[str, object], Ed25519PrivateKey]:
    graph = corpus.admission_corpus.graph.authority
    return (
        graph.role_keys[authority.ROLE_ORDER_V2.index(role)],
        graph.private_keys[role],
    )


def _session_common(
    *,
    refs: Mapping[str, object],
    custody_id: str,
    live_policy_id: str,
    a2_policy_id: str,
    session_id: str,
    sequence: int,
) -> dict[str, object]:
    return {
        "status": admission.STATUS_V2,
        "claim_level": admission.CLAIM_LEVEL_V2,
        "stage_b_trust_lock_id": refs["stage_b_trust_lock_id"],
        "authority_profile_id": refs["authority_profile_id"],
        "signer_registry_id": refs["signer_registry_id"],
        "attempt_context_id": refs["attempt_context_id"],
        "predecessor_b480_admission_candidate_result_id": refs[
            "admission_candidate_result_id"
        ],
        "predecessor_b480_admission_candidate_result_payload_sha256": refs[
            "admission_candidate_result_payload_sha256"
        ],
        "bootstrap_external_witness_policy_id": refs[
            "bootstrap_external_witness_policy_id"
        ],
        "online_witness_policy_id": refs["online_witness_policy_id"],
        "durable_anchor_policy_id": refs["durable_anchor_policy_id"],
        "runtime_closure_policy_id": refs["runtime_closure_policy_id"],
        "runtime_closure_evidence_id": refs["runtime_closure_evidence_id"],
        "custody_attestation_policy_id": custody_id,
        "admission_v2_live_policy_id": live_policy_id,
        "admission_v2_a2_policy_id": a2_policy_id,
        "session_id": session_id,
        "session_sequence": sequence,
    }


def _build_cas(
    root: Path,
    base: object,
    payloads: tuple[bytes, ...],
    roots: tuple[Mapping[str, object], ...],
    context: Mapping[str, object],
) -> tuple[_CasEvidence, dict[str, object]]:
    chain_id = admission._cas_chain_id(
        attempt_id=context["attempt_id"],
        session_id=context["session_id"],
        session_sequence=context["session_sequence"],
        result_id=roots[-1]["admission_v2_candidate_result_id"],
        result_sha=_sha(payloads[-1]),
    )
    plan = store.build_retention_plan_v1(
        authority_profile_id=context["profile_id"],
        signer_registry_id=context["registry_id"],
        attempt_context_id=context["attempt_id"],
        retention_session_id=context["session_id"],
        chain_id=chain_id,
        plan_kind=admission.CAS_PLAN_KIND_V2,
        objects_in_exact_order=tuple(
            (artifact_kind, maximum_bytes)
            for _ordinal, artifact_kind, maximum_bytes
            in admission.CAS_OBJECTS_IN_EXACT_ORDER_V2
        ),
    )
    call = base.admission_corpus.call
    _cas_row, cas_private = _role(base, "CAS_STORE")
    id_fields = (
        "admission_v2_live_policy_id",
        "pre_response_challenge_id",
        "witness_response_id",
        "witness_report_id",
        "custody_attestation_id",
        "admission_v2_candidate_result_id",
    )
    requests: list[bytes] = []
    receipts: list[bytes] = []
    checkpoints: list[bytes] = []
    heads: list[bytes] = []
    responses: list[bytes] = []
    with store.CasStoreV2(
        root_directory=str(root / "admission-v2-cas"),
        retention_plan_payload=plan,
        private_key_seed=_private_seed(cas_private),
        allowed_peer_uids=frozenset({os.geteuid()}),
        stage_b_lock_payload=call["stage_b_trust_lock_payload"],
        authority_profile_payload=call["authority_profile_payload"],
        signer_registry_payload=call["signer_registry_payload"],
        out_of_band_expected_lock_id_bytes=call[
            "out_of_band_expected_lock_id_bytes"
        ],
        out_of_band_expected_lock_payload_sha256_digest=call[
            "out_of_band_expected_lock_payload_sha256_digest"
        ],
        create=True,
    ) as cas_store:
        for ordinal, (payload, object_root, id_field) in enumerate(
            zip(payloads, roots, id_fields, strict=True)
        ):
            request_payload = store.build_generic_store_request_v1(
                retention_plan_payload=plan,
                ordinal=ordinal,
                bound_artifact_id=object_root[id_field],
                bound_artifact_sha256=_store_sha(payload),
                reported_size_bytes=len(payload),
            )
            object_path = root / f"admission-v2-object-{ordinal}.json"
            object_path.write_bytes(payload)
            descriptor = os.open(
                object_path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            try:
                response = cas_store.ingest_open_fd(
                    store.load_generic_store_request_v1(request_payload),
                    descriptor,
                    store.PeerCredentialsV2(
                        os.getpid(), os.geteuid(), os.getegid()
                    ),
                )
            finally:
                os.close(descriptor)
            requests.append(request_payload)
            receipts.append(response.checkpoint_record.signed_receipt.canonical_bytes())
            checkpoints.append(response.checkpoint_record.canonical_bytes())
            heads.append(response.head_record.canonical_bytes())
            responses.append(response.canonical_bytes())
        assert len(cas_store.audit_complete_session()) == 6
    evidence = _CasEvidence(
        plan=plan,
        requests=tuple(requests),
        objects=payloads,
        receipts=tuple(receipts),
        checkpoints=tuple(checkpoints),
        heads=tuple(heads),
        responses=tuple(responses),
    )
    replayed = replay.require_verified_cas_store_complete_session_replay_core_v2(
        replay.verify_cas_store_complete_session_replay_core_v2(
            stage_b_lock_payload=call["stage_b_trust_lock_payload"],
            authority_profile_payload=call["authority_profile_payload"],
            signer_registry_payload=call["signer_registry_payload"],
            out_of_band_expected_lock_id_bytes=call[
                "out_of_band_expected_lock_id_bytes"
            ],
            out_of_band_expected_lock_payload_sha256_digest=call[
                "out_of_band_expected_lock_payload_sha256_digest"
            ],
            retention_plan_payload=evidence.plan,
            request_payloads_in_exact_order=evidence.requests,
            object_payloads_in_exact_order=evidence.objects,
            signed_receipt_payloads_in_exact_order=evidence.receipts,
            checkpoint_payloads_in_exact_order=evidence.checkpoints,
            head_payloads_in_exact_order=evidence.heads,
            response_payloads_in_exact_order=evidence.responses,
        )
    )
    final_request = _json(evidence.requests[-1])
    final_binding = replayed.object_bindings_in_exact_order[-1]
    complete = {
        "schema_version": admission.ADMISSION_V2_CAS_COMPLETE_BINDING_SCHEMA,
        "authority_profile_id": replayed.authority_profile_id,
        "signer_registry_id": replayed.signer_registry_id,
        "attempt_context_id": replayed.attempt_context_id,
        "session_id": context["session_id"],
        "session_sequence": context["session_sequence"],
        "retention_plan_id": replayed.plan_id,
        "retention_plan_payload_sha256": _sha(evidence.plan),
        "retention_session_id": replayed.retention_session_id,
        "chain_id": replayed.chain_id,
        "plan_kind": replayed.plan_kind,
        "expected_object_count": replayed.expected_object_count,
        "artifact_kinds_in_exact_order": list(
            admission.CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2
        ),
        "final_ordinal": 5,
        "final_artifact_kind": admission.CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2[-1],
        "admission_v2_candidate_result_id": roots[-1][
            "admission_v2_candidate_result_id"
        ],
        "admission_v2_candidate_result_payload_sha256": _sha(payloads[-1]),
        "final_store_request_id": final_request["request_id"],
        "final_store_request_payload_sha256": _store_sha(evidence.requests[-1]),
        "final_object_id": final_binding[4],
        "final_object_sha256": _store_sha(payloads[-1]),
        "final_object_size_bytes": len(payloads[-1]),
        "final_signed_receipt_payload_sha256": final_binding[6],
        "final_checkpoint_payload_sha256": replayed.complete_checkpoint_sha256,
        "final_head_state_id": replayed.complete_head_state_id,
        "final_head_payload_sha256": replayed.complete_head_sha256,
        "final_response_payload_sha256": replayed.complete_response_sha256,
        "observed_object_count": 6,
        "complete": True,
        "complete_head_required": True,
        "presented_durable_before_return_claim": True,
        "presented_readback_verified_claim": True,
    }
    assert frozenset(complete) == admission.ADMISSION_V2_CAS_COMPLETE_BINDING_FIELDS
    return evidence, complete


def _positive_corpus(root: Path) -> _AdmissionV2Corpus:
    overlay_root = root / "overlay-fixture"
    overlay_root.mkdir()
    monkeypatch = pytest.MonkeyPatch()
    try:
        base = overlay_fixture._positive_overlay(overlay_root, monkeypatch)
    finally:
        monkeypatch.undo()
    refs = overlay_fixture._policy_refs(base.admission_corpus, base.runtime_graph)
    source, source_private = _role(base, "SOURCE_RUNTIME_CUSTODIAN")
    pure, pure_private = _role(base, "PURE_REPLAYER")
    bootstrap = base.admission_corpus.roots["bootstrap_policy"]
    online = base.admission_corpus.roots["online_policy"]["body"]
    durable = base.admission_corpus.roots["durable_policy"]["body"]
    custody_root = base.roots["custody_policy"]

    live_body = copy.deepcopy(base.roots["live_policy"]["body"])
    live_body.update(
        {
            "schema_version": admission.ADMISSION_V2_LIVE_POLICY_SCHEMA,
            "status": admission.STATUS_V2,
            "claim_level": admission.CLAIM_LEVEL_V2,
            "predecessor_b480_admission_candidate_result_id": live_body.pop(
                "admission_candidate_result_id"
            ),
            "predecessor_b480_admission_candidate_result_payload_sha256": live_body.pop(
                "admission_candidate_result_payload_sha256"
            ),
            "cas_retention_plan_kind": admission.CAS_PLAN_KIND_V2,
            "cas_artifact_kinds_in_exact_order": list(
                admission.CAS_ARTIFACT_KINDS_IN_EXACT_ORDER_V2
            ),
            "cas_chain_id_profile": "CONTENT_DERIVED_EXACT_ADMISSION_V2_RESULT",
        }
    )
    live_payload, live_root = _signed(
        "admission_v2_live_policy",
        live_body,
        (
            (
                "external",
                base.admission_corpus.graph.witness_private_key,
                bootstrap["witness_key_id"],
                bootstrap["witness_key_epoch"],
                admission_v1.SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1,
            ),
            (
                "authority",
                source_private,
                source,
                source["key_epoch"],
                admission.SIGN_SOURCE_PURPOSE_V2,
            ),
        ),
    )

    genesis = admission._genesis_mapping(
        attempt_id=refs["attempt_context_id"],
        live_policy_id=live_root["admission_v2_live_policy_id"],
        b480_result_id=refs["admission_candidate_result_id"],
        b480_result_sha=refs["admission_candidate_result_payload_sha256"],
    )
    zeros = "0" * 64
    a2_body = copy.deepcopy(_json(base.payloads["anchor_policy"])["body"])
    a2_body.update(
        {
            "schema_version": admission.ADMISSION_V2_A2_POLICY_SCHEMA,
            "status": admission.STATUS_V2,
            "claim_level": admission.CLAIM_LEVEL_V2,
            "predecessor_b480_admission_candidate_result_id": a2_body.pop(
                "admission_candidate_result_id"
            ),
            "predecessor_b480_admission_candidate_result_payload_sha256": a2_body.pop(
                "admission_candidate_result_payload_sha256"
            ),
            "admission_v2_live_policy_id": a2_body.pop("live_runtime_policy_id"),
            "admission_v2_live_policy_payload_sha256": a2_body.pop(
                "live_runtime_policy_payload_sha256"
            ),
            "genesis_checkpoint_id": (
                admission._PREFIX_BY_KIND["admission_v2_a2_checkpoint"] + zeros
            ),
            "genesis_head_id": (
                admission._PREFIX_BY_KIND["admission_v2_a2_head"] + zeros
            ),
            "genesis_mapping": copy.deepcopy(genesis),
            "oob_high_water_checkpoint_id": (
                admission._PREFIX_BY_KIND["admission_v2_a2_checkpoint"] + zeros
            ),
            "oob_high_water_head_id": (
                admission._PREFIX_BY_KIND["admission_v2_a2_head"] + zeros
            ),
            "oob_high_water_mapping": copy.deepcopy(genesis),
            "policy_alone_nonclaims": {
                field: False for field in admission.POLICY_ALONE_NONCLAIMS_FIELDS_V2
            },
        }
    )
    a2_body["admission_v2_live_policy_id"] = live_root[
        "admission_v2_live_policy_id"
    ]
    a2_body["admission_v2_live_policy_payload_sha256"] = _sha(live_payload)
    assert a2_body["admission_v2_live_policy_id"] == live_root[
        "admission_v2_live_policy_id"
    ]
    assert a2_body["admission_v2_live_policy_payload_sha256"] == _sha(live_payload)
    a2_payload, a2_root = _signed(
        "admission_v2_a2_policy",
        a2_body,
        ((
            "external",
            base.admission_corpus.durable_private_key,
            durable["anchor_key_id"],
            durable["anchor_key_epoch"],
            admission.SIGN_A2_POLICY_PURPOSE_V2,
        ),),
    )

    sequence = 1
    challenge_nonce = "33" * 32
    response_nonce = "44" * 32
    session_id = admission._session_id(
        attempt_id=refs["attempt_context_id"],
        live_policy_id=live_root["admission_v2_live_policy_id"],
        a2_policy_id=a2_root["admission_v2_a2_policy_id"],
        online_key_id=online["online_key_id"],
        session_sequence=sequence,
        challenge_nonce=challenge_nonce,
    )
    common = _session_common(
        refs=refs,
        custody_id=custody_root["custody_attestation_policy_id"],
        live_policy_id=live_root["admission_v2_live_policy_id"],
        a2_policy_id=a2_root["admission_v2_a2_policy_id"],
        session_id=session_id,
        sequence=sequence,
    )
    entry = copy.deepcopy(base.roots["challenge"]["body"]["entry_process_snapshot"])
    exit_snapshot = copy.deepcopy(base.roots["report"]["body"]["exit_process_snapshot"])
    entry_sha = _sha(admission.canonical_external_witness_admission_json_bytes_v2(entry))
    exit_sha = _sha(
        admission.canonical_external_witness_admission_json_bytes_v2(exit_snapshot)
    )
    challenge_body: dict[str, object] = {
        "schema_version": admission.ADMISSION_V2_PRE_RESPONSE_CHALLENGE_SCHEMA,
        **common,
        "challenge_nonce": challenge_nonce,
        "challenge_nonce_size_bytes": 32,
        "challenger_role": "SOURCE_RUNTIME_CUSTODIAN",
        "challenger_key_id": source["key_id"],
        "challenger_key_epoch": source["key_epoch"],
        "expected_responder_key_id": online["online_key_id"],
        "expected_responder_key_epoch": online["online_key_epoch"],
        "entry_process_snapshot": entry,
        "entry_process_snapshot_payload_sha256": entry_sha,
        "expected_previous_a2_mapping": copy.deepcopy(genesis),
        "challenge_monotonic_nanoseconds_decimal": "2000000",
        "maximum_round_trip_nanoseconds": live_body[
            "maximum_round_trip_nanoseconds"
        ],
        "one_shot_required": True,
        "challenge_outcome": "ISSUED_BEFORE_W_RESPONSE",
    }
    challenge_payload, challenge_root = _signed(
        "pre_response_challenge",
        challenge_body,
        ((
            "authority",
            source_private,
            source,
            source["key_epoch"],
            admission.SIGN_SOURCE_PURPOSE_V2,
        ),),
    )
    response_body: dict[str, object] = {
        "schema_version": admission.ADMISSION_V2_WITNESS_RESPONSE_SCHEMA,
        **common,
        "pre_response_challenge_id": challenge_root["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(challenge_payload),
        "challenge_nonce": challenge_nonce,
        "challenge_nonce_size_bytes": 32,
        "responder_key_id": online["online_key_id"],
        "responder_key_epoch": online["online_key_epoch"],
        "response_nonce": response_nonce,
        "response_nonce_size_bytes": 32,
        "entry_process_snapshot_id": entry["process_snapshot_id"],
        "entry_process_snapshot_payload_sha256": entry_sha,
        "response_outcome": "W_RESPONDED_TO_PRE_RESPONSE_CHALLENGE",
    }
    response_payload, response_root = _signed(
        "witness_response",
        response_body,
        ((
            "external",
            base.admission_corpus.online_private_key,
            online["online_key_id"],
            online["online_key_epoch"],
            admission.SIGN_WITNESS_RESPONSE_PURPOSE_V2,
        ),),
    )
    report_body: dict[str, object] = {
        "schema_version": admission.ADMISSION_V2_WITNESS_REPORT_SCHEMA,
        **common,
        "pre_response_challenge_id": challenge_root["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(challenge_payload),
        "witness_response_id": response_root["witness_response_id"],
        "witness_response_payload_sha256": _sha(response_payload),
        "challenge_nonce": challenge_nonce,
        "response_nonce": response_nonce,
        "entry_process_snapshot": copy.deepcopy(entry),
        "entry_process_snapshot_payload_sha256": entry_sha,
        "exit_process_snapshot": exit_snapshot,
        "exit_process_snapshot_payload_sha256": exit_sha,
        "entry_exit_identity_fields_in_exact_order": list(
            admission.ENTRY_EXIT_IDENTITY_FIELDS_IN_EXACT_ORDER_V2
        ),
        "presented_entry_exit_identity_equal": True,
        "challenge_sent_monotonic_nanoseconds_decimal": "2000000",
        "response_received_monotonic_nanoseconds_decimal": "2000100",
        "round_trip_nanoseconds": 100,
        "maximum_round_trip_nanoseconds": live_body[
            "maximum_round_trip_nanoseconds"
        ],
        "presented_round_trip_within_policy": True,
        "presented_exact_one_challenge_packet_sent": True,
        "presented_exact_one_response_packet_received": True,
        "response_message_flags": 0,
        "response_packet_size_bytes": len(response_payload),
        "response_ancillary_item_count": 0,
        "presented_message_truncation_absent": True,
        "presented_control_truncation_absent": True,
        "presented_unexpected_ancillary_absent": True,
        "presented_so_peercred_check_passed": True,
        "presented_pidfd_identity_check_passed": True,
        "presented_procfs_identity_check_passed": True,
        "presented_service_binding_check_passed": True,
        "presented_runtime_closure_policy_binding_check_passed": True,
        "presented_runtime_closure_evidence_binding_check_passed": True,
        "presented_runtime_executable_binding_check_passed": True,
        "presented_distinct_uid_gid_check_passed": True,
        "presented_live_channel_observed_claim": True,
        "report_outcome": "PRESENTED_ENTRY_EXIT_EQUAL_CLAIM_ONLY",
    }
    report_payload, report_root = _signed(
        "witness_report",
        report_body,
        ((
            "authority",
            source_private,
            source,
            source["key_epoch"],
            admission.SIGN_SOURCE_PURPOSE_V2,
        ),),
    )
    custody = custody_root["body"]
    opaque = _sha(b"admission-v2-opaque-online-key-handle")
    attestation_body: dict[str, object] = {
        "schema_version": admission.ADMISSION_V2_CUSTODY_ATTESTATION_SCHEMA,
        **common,
        "pre_response_challenge_id": challenge_root["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(challenge_payload),
        "witness_response_id": response_root["witness_response_id"],
        "witness_response_payload_sha256": _sha(response_payload),
        "witness_report_id": report_root["witness_report_id"],
        "witness_report_payload_sha256": _sha(report_payload),
        "challenge_nonce": challenge_nonce,
        "response_nonce": response_nonce,
        "entry_process_snapshot_id": entry["process_snapshot_id"],
        "entry_process_snapshot_payload_sha256": entry_sha,
        "exit_process_snapshot_id": exit_snapshot["process_snapshot_id"],
        "exit_process_snapshot_payload_sha256": exit_sha,
        "attestation_sequence": sequence,
        "attestor_service_instance_id": "custody-attestor-instance-admission-v2",
        "attestor_control_domain_id": "external-custody-control-admission-v2",
        "attestor_key_id": custody["custody_attestor_key_id"],
        "attestor_key_epoch": custody["custody_attestor_key_epoch"],
        "attestation_purpose": admission.ATTEST_WITNESS_CUSTODY_PURPOSE_V2,
        "online_witness_key_id": online["online_key_id"],
        "online_witness_key_epoch": online["online_key_epoch"],
        "online_witness_public_key_hex": online["online_public_key_hex"],
        "opaque_key_handle_sha256": opaque,
        "attestation_evidence_format": custody["attestation_evidence_format"],
        "attestation_evidence_hex": "ab" * 32,
        "certificate_chain_der_hex_in_exact_order": ["30"],
        "certificate_chain_root_id": "test-root-ca-v1",
        "endorsement_material_hex_in_exact_order": ["aa"],
        "revocation_material_hex_in_exact_order": ["bb"],
        "attested_challenge_nonce": challenge_nonce,
        "attested_report_id": report_root["witness_report_id"],
        "attested_report_payload_sha256": _sha(report_payload),
        "attested_online_key_public_key_sha256": _sha(
            bytes.fromhex(online["online_public_key_hex"])
        ),
        "attested_key_handle_sha256": opaque,
        "presented_online_key_generated_in_attested_boundary_claim": True,
        "presented_online_key_nonexportable_claim": True,
        "presented_certificate_chain_check_passed": True,
        "presented_quote_check_passed": True,
        "presented_fresh_nonce_check_passed": True,
        "presented_report_binding_check_passed": True,
        "presented_online_key_binding_check_passed": True,
        "attestation_outcome": "PRESENTED_CUSTODY_EVIDENCE_CLAIM_ONLY",
    }
    attestation_payload, attestation_root = _signed(
        "custody_attestation",
        attestation_body,
        ((
            "external",
            base.admission_corpus.custody_private_key,
            custody["custody_attestor_key_id"],
            custody["custody_attestor_key_epoch"],
            admission.ATTEST_WITNESS_CUSTODY_PURPOSE_V2,
        ),),
    )
    result_body: dict[str, object] = {
        "schema_version": admission.ADMISSION_V2_CANDIDATE_RESULT_SCHEMA,
        **common,
        "pre_response_challenge_id": challenge_root["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(challenge_payload),
        "witness_response_id": response_root["witness_response_id"],
        "witness_response_payload_sha256": _sha(response_payload),
        "witness_report_id": report_root["witness_report_id"],
        "witness_report_payload_sha256": _sha(report_payload),
        "custody_attestation_id": attestation_root["custody_attestation_id"],
        "custody_attestation_payload_sha256": _sha(attestation_payload),
        "verified_signature_count": admission.PRE_CAS_VERIFIED_SIGNATURE_COUNT_V2,
        "verified_signature_scope": (
            "CUSTODY_POLICY_1_LIVE_POLICY_2_PLUS_CHALLENGE_RESPONSE_REPORT_"
            "CUSTODY_AND_RESULT_2_"
            "EXCLUDES_AUTHORITY_GRAPH_B480_CAS_AND_A2"
        ),
        "verified_protocol_object_count": 7,
        "raw_authority_graph_verified": True,
        "frozen_b480_candidate_verified": True,
        "dependency_contract_constants_verified": True,
        "nineteen_expected_pins_verified": True,
        "key_sets_pairwise_disjoint": True,
        "no_same_session_forward_references_verified": True,
        "offline_static_image_graph_verified": True,
        "presented_response_report_chain_verified": True,
        "presented_custody_attestation_bytes_verified": True,
        "presented_only": True,
        "private_key_present": False,
        "filesystem_path_present": False,
        "socket_present": False,
        "decoded_mapping_input_present": False,
        "callback_present": False,
        "caller_nonce_present": False,
        "caller_clock_present": False,
        "actual_runtime_verified": False,
        "runtime_closure_verified": False,
        "live_channel_verified": False,
        "key_custody_verified": False,
        "online_key_nonexportable_verified": False,
        "distinct_live_uid_gid_verified": False,
        "external_cas_verified": False,
        "external_anchor_verified": False,
        "external_anti_rollback_verified": False,
        "external_currentness_verified": False,
        "durability_verified": False,
        "non_equivocation_verified": False,
        "global_anti_fork_verified": False,
        "malicious_root_resistance_verified": False,
        "authority_evidence": False,
        "r01_satisfied": False,
        "readiness_evidence": False,
        "execution_authorized": False,
        "first_failure_code": "NONE",
    }
    result_payload, result_root = _signed(
        "admission_v2_result",
        result_body,
        (
            (
                "authority",
                source_private,
                source,
                source["key_epoch"],
                admission.SIGN_SOURCE_PURPOSE_V2,
            ),
            (
                "authority",
                pure_private,
                pure,
                pure["key_epoch"],
                admission.SIGN_PURE_RESULT_PURPOSE_V2,
            ),
        ),
    )
    six_payloads = (
        live_payload,
        challenge_payload,
        response_payload,
        report_payload,
        attestation_payload,
        result_payload,
    )
    six_roots = (
        live_root,
        challenge_root,
        response_root,
        report_root,
        attestation_root,
        result_root,
    )
    context = {
        "profile_id": refs["authority_profile_id"],
        "registry_id": refs["signer_registry_id"],
        "attempt_id": refs["attempt_context_id"],
        "b480_result_id": refs["admission_candidate_result_id"],
        "b480_result_sha": refs["admission_candidate_result_payload_sha256"],
        "live_policy_id": live_root["admission_v2_live_policy_id"],
        "session_id": session_id,
        "session_sequence": sequence,
        "result_id": result_root["admission_v2_candidate_result_id"],
        "result_sha": _sha(result_payload),
    }
    cas, complete = _build_cas(root, base, six_payloads, six_roots, context)
    committed = admission._committed_mapping(
        context=context, complete=complete, anchor_sequence=1
    )
    a2_common = {
        **common,
        "admission_v2_candidate_result_id": result_root[
            "admission_v2_candidate_result_id"
        ],
        "admission_v2_candidate_result_payload_sha256": _sha(result_payload),
    }
    request_body: dict[str, object] = {
        "schema_version": admission.ADMISSION_V2_A2_REQUEST_SCHEMA,
        **a2_common,
        "admission_v2_live_policy_payload_sha256": _sha(live_payload),
        "pre_response_challenge_id": challenge_root["pre_response_challenge_id"],
        "pre_response_challenge_payload_sha256": _sha(challenge_payload),
        "witness_response_id": response_root["witness_response_id"],
        "witness_response_payload_sha256": _sha(response_payload),
        "witness_report_id": report_root["witness_report_id"],
        "witness_report_payload_sha256": _sha(report_payload),
        "custody_attestation_id": attestation_root["custody_attestation_id"],
        "custody_attestation_payload_sha256": _sha(attestation_payload),
        "cas_complete_binding": copy.deepcopy(complete),
        "expected_previous_a2_checkpoint_id": a2_body[
            "oob_high_water_checkpoint_id"
        ],
        "expected_previous_a2_checkpoint_payload_sha256": a2_body[
            "oob_high_water_checkpoint_payload_sha256"
        ],
        "expected_previous_a2_checkpoint_sequence": a2_body[
            "oob_high_water_checkpoint_sequence"
        ],
        "expected_previous_a2_head_id": a2_body["oob_high_water_head_id"],
        "expected_previous_a2_head_payload_sha256": a2_body[
            "oob_high_water_head_payload_sha256"
        ],
        "expected_previous_a2_head_sequence": a2_body[
            "oob_high_water_head_sequence"
        ],
        "expected_current_mapping": copy.deepcopy(genesis),
        "proposed_current_mapping": copy.deepcopy(committed),
        "compare_and_append_mode": a2_body["compare_and_append_semantics"],
    }
    request_payload, request_root = _signed(
        "admission_v2_a2_request",
        request_body,
        ((
            "authority",
            source_private,
            source,
            source["key_epoch"],
            admission.SIGN_SOURCE_PURPOSE_V2,
        ),),
    )
    checkpoint_body: dict[str, object] = {
        "schema_version": admission.ADMISSION_V2_A2_CHECKPOINT_SCHEMA,
        **a2_common,
        "admission_v2_a2_request_id": request_root[
            "admission_v2_a2_request_id"
        ],
        "admission_v2_a2_request_payload_sha256": _sha(request_payload),
        "previous_mapping": copy.deepcopy(genesis),
        "committed_mapping": copy.deepcopy(committed),
        "cas_complete_binding": copy.deepcopy(complete),
        "cas_outcome": "APPENDED_PRESENTED_ONLY",
        "checkpoint_sequence": 1,
        "previous_admission_v2_a2_checkpoint_id": a2_body[
            "oob_high_water_checkpoint_id"
        ],
    }
    checkpoint_payload, checkpoint_root = _signed(
        "admission_v2_a2_checkpoint",
        checkpoint_body,
        ((
            "external",
            base.admission_corpus.durable_private_key,
            durable["anchor_key_id"],
            durable["anchor_key_epoch"],
            admission.SIGN_A2_CHECKPOINT_PURPOSE_V2,
        ),),
    )
    head_body: dict[str, object] = {
        "schema_version": admission.ADMISSION_V2_A2_HEAD_SCHEMA,
        **a2_common,
        "admission_v2_a2_request_id": request_root[
            "admission_v2_a2_request_id"
        ],
        "admission_v2_a2_request_payload_sha256": _sha(request_payload),
        "admission_v2_a2_checkpoint_id": checkpoint_root[
            "admission_v2_a2_checkpoint_id"
        ],
        "admission_v2_a2_checkpoint_payload_sha256": _sha(checkpoint_payload),
        "committed_mapping": copy.deepcopy(committed),
        "cas_complete_binding": copy.deepcopy(complete),
        "cas_outcome": "APPENDED_PRESENTED_ONLY",
        "head_sequence": 1,
        "previous_admission_v2_a2_head_id": a2_body[
            "oob_high_water_head_id"
        ],
    }
    head_payload, head_root = _signed(
        "admission_v2_a2_head",
        head_body,
        ((
            "external",
            base.admission_corpus.durable_private_key,
            durable["anchor_key_id"],
            durable["anchor_key_epoch"],
            admission.SIGN_A2_HEAD_PURPOSE_V2,
        ),),
    )
    receipt_body: dict[str, object] = {
        "schema_version": admission.ADMISSION_V2_A2_RECEIPT_SCHEMA,
        **a2_common,
        "admission_v2_a2_request_id": request_root[
            "admission_v2_a2_request_id"
        ],
        "admission_v2_a2_request_payload_sha256": _sha(request_payload),
        "admission_v2_a2_checkpoint_id": checkpoint_root[
            "admission_v2_a2_checkpoint_id"
        ],
        "admission_v2_a2_checkpoint_payload_sha256": _sha(checkpoint_payload),
        "admission_v2_a2_head_id": head_root["admission_v2_a2_head_id"],
        "admission_v2_a2_head_payload_sha256": _sha(head_payload),
        "previous_mapping": copy.deepcopy(genesis),
        "committed_mapping": copy.deepcopy(committed),
        "cas_complete_binding": copy.deepcopy(complete),
        "cas_outcome": "APPENDED_PRESENTED_ONLY",
        "compare_matched": True,
        "append_applied": True,
        "oob_high_water_checkpoint_id": a2_body[
            "oob_high_water_checkpoint_id"
        ],
        "oob_high_water_head_id": a2_body["oob_high_water_head_id"],
        "presented_durable_before_return_claim": True,
    }
    receipt_payload, receipt_root = _signed(
        "admission_v2_a2_receipt",
        receipt_body,
        ((
            "external",
            base.admission_corpus.durable_private_key,
            durable["anchor_key_id"],
            durable["anchor_key_epoch"],
            admission.SIGN_A2_RECEIPT_PURPOSE_V2,
        ),),
    )

    call: dict[str, object] = {}
    base_call = base.call
    for name in tuple(inspect.signature(
        admission.verify_external_witness_admission_candidate_v2
    ).parameters)[:24]:
        call[name] = base_call[name]
    call.update(
        {
            "predecessor_b480_admission_candidate_result_payload": (
                base.admission_corpus.payloads["result"]
            ),
            "runtime_closure_policy_payload": base_call[
                "runtime_closure_policy_payload"
            ],
            "runtime_closure_evidence_payload": base_call[
                "runtime_closure_evidence_payload"
            ],
            "raw_oci_index_payload": base_call["raw_oci_index_payload"],
            "raw_oci_manifest_payload": base_call["raw_oci_manifest_payload"],
            "raw_oci_config_payload": base_call["raw_oci_config_payload"],
            "ordered_raw_layer_chunk_tuples": base_call[
                "ordered_raw_layer_chunk_tuples"
            ],
            "ordered_support_artifact_chunk_tuples": base_call[
                "ordered_support_artifact_chunk_tuples"
            ],
            "custody_attestation_policy_payload": base.payloads["custody_policy"],
            "admission_v2_live_policy_payload": live_payload,
            "admission_v2_a2_policy_payload": a2_payload,
            "pre_response_challenge_payload": challenge_payload,
            "witness_response_payload": response_payload,
            "witness_report_payload": report_payload,
            "witness_custody_attestation_payload": attestation_payload,
            "admission_v2_candidate_result_payload": result_payload,
            "admission_v2_cas_retention_plan_payload": cas.plan,
            "admission_v2_cas_store_request_payloads_in_exact_order": cas.requests,
            "admission_v2_cas_object_payloads_in_exact_order": cas.objects,
            "admission_v2_cas_signed_receipt_payloads_in_exact_order": cas.receipts,
            "admission_v2_cas_checkpoint_payloads_in_exact_order": cas.checkpoints,
            "admission_v2_cas_head_payloads_in_exact_order": cas.heads,
            "admission_v2_cas_response_payloads_in_exact_order": cas.responses,
            "admission_v2_a2_request_payload": request_payload,
            "admission_v2_a2_checkpoint_payload": checkpoint_payload,
            "admission_v2_a2_head_payload": head_payload,
            "admission_v2_a2_receipt_payload": receipt_payload,
        }
    )
    for pin in admission.EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2:
        call[pin] = base_call[pin]
    call.update(
        {
            "out_of_band_expected_live_runtime_policy_id_bytes": live_root[
                "admission_v2_live_policy_id"
            ].encode("ascii"),
            "out_of_band_expected_live_runtime_policy_payload_sha256_digest": (
                hashlib.sha256(live_payload).digest()
            ),
            "out_of_band_expected_live_runtime_anchor_policy_id_bytes": a2_root[
                "admission_v2_a2_policy_id"
            ].encode("ascii"),
            "out_of_band_expected_live_runtime_anchor_policy_payload_sha256_digest": (
                hashlib.sha256(a2_payload).digest()
            ),
        }
    )
    payload_map = {
        "live_policy": live_payload,
        "a2_policy": a2_payload,
        "challenge": challenge_payload,
        "response": response_payload,
        "report": report_payload,
        "attestation": attestation_payload,
        "result": result_payload,
        "a2_request": request_payload,
        "a2_checkpoint": checkpoint_payload,
        "a2_head": head_payload,
        "a2_receipt": receipt_payload,
    }
    root_map = {
        "live_policy": live_root,
        "a2_policy": a2_root,
        "challenge": challenge_root,
        "response": response_root,
        "report": report_root,
        "attestation": attestation_root,
        "result": result_root,
        "a2_request": request_root,
        "a2_checkpoint": checkpoint_root,
        "a2_head": head_root,
        "a2_receipt": receipt_root,
    }
    return _AdmissionV2Corpus(
        call=call,
        payloads=payload_map,
        roots=root_map,
        base=base,
        complete=complete,
        source_private=source_private,
        pure_private=pure_private,
    )


@pytest.fixture(scope="module")
def positive_corpus(
    tmp_path_factory: pytest.TempPathFactory,
) -> _AdmissionV2Corpus:
    return _positive_corpus(tmp_path_factory.mktemp("external-admission-v2"))


def test_complete_raw_positive_dag_actual_six_object_store_replay_and_a2_suffix(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    value = admission.verify_external_witness_admission_candidate_v2(
        **positive_corpus.call
    )
    required = admission.require_verified_external_witness_admission_candidate_v2(
        value
    )
    assert required.verification_status == (
        "PRESENTED_RAW_ACYCLIC_EXTERNAL_WITNESS_ADMISSION_V2_"
        "VERIFIED_MECHANICS_ONLY"
    )
    assert required.exact_six_object_cas_replay_verified is True
    assert required.presented_a2_compare_and_append_verified is True
    assert required.post_a2_private_binding_verified is True
    assert required.six_object_payload_sha256_in_exact_order == tuple(
        _sha(positive_corpus.payloads[name])
        for name in (
            "live_policy",
            "challenge",
            "response",
            "report",
            "attestation",
            "result",
        )
    )
    assert positive_corpus.complete["observed_object_count"] == 6


def test_store_replay_tuple_keeps_bound_artifact_id_distinct_from_object_address(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    call = positive_corpus.call
    replayed = replay.require_verified_cas_store_complete_session_replay_core_v2(
        replay.verify_cas_store_complete_session_replay_core_v2(
            stage_b_lock_payload=call["stage_b_trust_lock_payload"],
            authority_profile_payload=call["authority_profile_payload"],
            signer_registry_payload=call["signer_registry_payload"],
            out_of_band_expected_lock_id_bytes=call[
                "out_of_band_expected_lock_id_bytes"
            ],
            out_of_band_expected_lock_payload_sha256_digest=call[
                "out_of_band_expected_lock_payload_sha256_digest"
            ],
            retention_plan_payload=call[
                "admission_v2_cas_retention_plan_payload"
            ],
            request_payloads_in_exact_order=call[
                "admission_v2_cas_store_request_payloads_in_exact_order"
            ],
            object_payloads_in_exact_order=call[
                "admission_v2_cas_object_payloads_in_exact_order"
            ],
            signed_receipt_payloads_in_exact_order=call[
                "admission_v2_cas_signed_receipt_payloads_in_exact_order"
            ],
            checkpoint_payloads_in_exact_order=call[
                "admission_v2_cas_checkpoint_payloads_in_exact_order"
            ],
            head_payloads_in_exact_order=call[
                "admission_v2_cas_head_payloads_in_exact_order"
            ],
            response_payloads_in_exact_order=call[
                "admission_v2_cas_response_payloads_in_exact_order"
            ],
        )
    )
    final_binding = replayed.object_bindings_in_exact_order[-1]
    assert len(final_binding) == 10
    assert final_binding[2] == positive_corpus.roots["result"][
        "admission_v2_candidate_result_id"
    ]
    assert final_binding[4] == _store_sha(positive_corpus.payloads["result"])
    assert final_binding[2] != final_binding[4]


_CALL_KEY_BY_OBJECT_NAME = {
    "live_policy": "admission_v2_live_policy_payload",
    "a2_policy": "admission_v2_a2_policy_payload",
    "challenge": "pre_response_challenge_payload",
    "response": "witness_response_payload",
    "report": "witness_report_payload",
    "attestation": "witness_custody_attestation_payload",
    "result": "admission_v2_candidate_result_payload",
    "a2_request": "admission_v2_a2_request_payload",
    "a2_checkpoint": "admission_v2_a2_checkpoint_payload",
    "a2_head": "admission_v2_a2_head_payload",
    "a2_receipt": "admission_v2_a2_receipt_payload",
}


_KIND_BY_OBJECT_NAME = {
    "live_policy": "admission_v2_live_policy",
    "a2_policy": "admission_v2_a2_policy",
    "challenge": "pre_response_challenge",
    "response": "witness_response",
    "report": "witness_report",
    "attestation": "custody_attestation",
    "result": "admission_v2_result",
    "a2_request": "admission_v2_a2_request",
    "a2_checkpoint": "admission_v2_a2_checkpoint",
    "a2_head": "admission_v2_a2_head",
    "a2_receipt": "admission_v2_a2_receipt",
}


def _signers_for(
    corpus: _AdmissionV2Corpus, name: str
) -> tuple[
    tuple[str, Ed25519PrivateKey, Mapping[str, object] | str, int, str], ...
]:
    source, source_private = _role(corpus.base, "SOURCE_RUNTIME_CUSTODIAN")
    pure, pure_private = _role(corpus.base, "PURE_REPLAYER")
    bootstrap = corpus.base.admission_corpus.roots["bootstrap_policy"]
    online = corpus.base.admission_corpus.roots["online_policy"]["body"]
    durable = corpus.base.admission_corpus.roots["durable_policy"]["body"]
    custody = corpus.base.roots["custody_policy"]["body"]
    authority_source = (
        "authority",
        source_private,
        source,
        source["key_epoch"],
        admission.SIGN_SOURCE_PURPOSE_V2,
    )
    external_anchor_purpose = {
        "a2_policy": admission.SIGN_A2_POLICY_PURPOSE_V2,
        "a2_checkpoint": admission.SIGN_A2_CHECKPOINT_PURPOSE_V2,
        "a2_head": admission.SIGN_A2_HEAD_PURPOSE_V2,
        "a2_receipt": admission.SIGN_A2_RECEIPT_PURPOSE_V2,
    }
    if name == "live_policy":
        return (
            (
                "external",
                corpus.base.admission_corpus.graph.witness_private_key,
                bootstrap["witness_key_id"],
                bootstrap["witness_key_epoch"],
                admission_v1.SIGN_ONLINE_WITNESS_POLICY_PURPOSE_V1,
            ),
            authority_source,
        )
    if name in external_anchor_purpose:
        return ((
            "external",
            corpus.base.admission_corpus.durable_private_key,
            durable["anchor_key_id"],
            durable["anchor_key_epoch"],
            external_anchor_purpose[name],
        ),)
    if name == "response":
        return ((
            "external",
            corpus.base.admission_corpus.online_private_key,
            online["online_key_id"],
            online["online_key_epoch"],
            admission.SIGN_WITNESS_RESPONSE_PURPOSE_V2,
        ),)
    if name == "attestation":
        return ((
            "external",
            corpus.base.admission_corpus.custody_private_key,
            custody["custody_attestor_key_id"],
            custody["custody_attestor_key_epoch"],
            admission.ATTEST_WITNESS_CUSTODY_PURPOSE_V2,
        ),)
    if name == "result":
        return (
            authority_source,
            (
                "authority",
                pure_private,
                pure,
                pure["key_epoch"],
                admission.SIGN_PURE_RESULT_PURPOSE_V2,
            ),
        )
    assert name in {"challenge", "report", "a2_request"}
    return (authority_source,)


def _resigned_payload(
    corpus: _AdmissionV2Corpus,
    name: str,
    mutator: Callable[[dict[str, object]], None],
) -> bytes:
    body = copy.deepcopy(corpus.roots[name]["body"])
    mutator(body)
    return _signed(_KIND_BY_OBJECT_NAME[name], body, _signers_for(corpus, name))[0]


def _assert_rejected(
    call: Mapping[str, object],
    *codes: str,
) -> admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2:
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2
    ) as caught:
        admission.verify_external_witness_admission_candidate_v2(**dict(call))
    if codes:
        assert caught.value.code in set(codes)
    return caught.value


def test_public_verifier_has_exact_raw_only_51_plus_19_abi_and_frozen_dependencies(
) -> None:
    positional_names = tuple(
        """
        stage_b_trust_lock_payload authority_profile_payload signer_registry_payload
        bootstrap_external_witness_policy_payload online_witness_policy_payload
        durable_anchor_policy_payload candidate_anchor_payload candidate_receipt_payload
        candidate_ledger_payload candidate_checkpoint_payload candidate_head_payload
        q1_prepare_request_payload q1_prepare_response_payload
        a_compare_and_append_request_payload a_durable_checkpoint_payload
        a_durable_head_payload a_compare_and_append_receipt_payload
        q2_confirm_request_payload q2_confirm_response_payload admission_transcript_payload
        post_commit_logical_time_request_payloads_in_exact_order
        post_commit_signed_logical_time_token_payloads_in_exact_order
        post_commit_logical_time_checkpoint_payloads_in_exact_order
        post_commit_logical_time_head_payloads_in_exact_order
        predecessor_b480_admission_candidate_result_payload
        runtime_closure_policy_payload runtime_closure_evidence_payload
        raw_oci_index_payload raw_oci_manifest_payload raw_oci_config_payload
        ordered_raw_layer_chunk_tuples ordered_support_artifact_chunk_tuples
        custody_attestation_policy_payload admission_v2_live_policy_payload
        admission_v2_a2_policy_payload pre_response_challenge_payload
        witness_response_payload witness_report_payload
        witness_custody_attestation_payload admission_v2_candidate_result_payload
        admission_v2_cas_retention_plan_payload
        admission_v2_cas_store_request_payloads_in_exact_order
        admission_v2_cas_object_payloads_in_exact_order
        admission_v2_cas_signed_receipt_payloads_in_exact_order
        admission_v2_cas_checkpoint_payloads_in_exact_order
        admission_v2_cas_head_payloads_in_exact_order
        admission_v2_cas_response_payloads_in_exact_order
        admission_v2_a2_request_payload admission_v2_a2_checkpoint_payload
        admission_v2_a2_head_payload admission_v2_a2_receipt_payload
        """.split()
    )
    signature = inspect.signature(
        admission.verify_external_witness_admission_candidate_v2
    )
    parameters = tuple(signature.parameters.values())
    assert len(positional_names) == 51
    assert len(parameters) == 70
    assert tuple(parameter.name for parameter in parameters[:51]) == positional_names
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters[:51]
    )
    assert tuple(parameter.name for parameter in parameters[51:]) == (
        admission.EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2
    )
    assert len(admission.EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2) == 19
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters[51:]
    )
    assert signature.return_annotation == "VerifiedExternalWitnessAdmissionCandidateV2"
    provenance = (
        (
            admission_v1,
            admission.INSPECTED_ADMISSION_CONTRACT_V1_SOURCE_SHA256,
            "b480b0d4b286e1b0cfb43f88846bbbf635b753fffc64b5e89e9073264360e285",
        ),
        (
            overlay,
            admission.INSPECTED_LIVE_RUNTIME_OVERLAY_V2_SOURCE_SHA256,
            "5e4b23b79c04e4a613995ddec5aafe883e1aa694b945b86bda7ba67caff3be0e",
        ),
        (
            authority,
            admission.INSPECTED_AUTHORITY_V2_SOURCE_SHA256,
            "03d689327ca422f93e5a32db0a7f813fe84b1099f1bac1b31395c975d76d8e86",
        ),
        (
            closure,
            admission.INSPECTED_RUNTIME_CLOSURE_V1_SOURCE_SHA256,
            "f836b3d00f332398214d3ccf7787109abb8f2de96b7b8281748bbb0ec695f2f9",
        ),
        (
            replay,
            admission.INSPECTED_STORE_REPLAY_CORE_V2_SOURCE_SHA256,
            "e33bb8ae1f4c70668de3bfbdbd803a953513db94cb0a9c1ceb1082e97141de02",
        ),
    )
    for module, declared, expected in provenance:
        assert declared == expected
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected


def test_decoded_transport_and_each_of_nineteen_raw_pins_fail_closed(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    decoded = dict(positive_corpus.call)
    decoded["admission_v2_live_policy_payload"] = copy.deepcopy(
        positive_corpus.roots["live_policy"]
    )
    _assert_rejected(decoded, admission.FAIL_INPUT_V2)
    for pin_name in admission.EXPECTED_PIN_NAMES_IN_EXACT_ORDER_V2:
        call = dict(positive_corpus.call)
        original = call[pin_name]
        assert type(original) is bytes and original
        if pin_name.endswith("_digest"):
            call[pin_name] = bytes([original[0] ^ 1]) + original[1:]
        else:
            call[pin_name] = original + b"x"
        _assert_rejected(
            call,
            admission.FAIL_INPUT_V2,
            admission.FAIL_BINDING_V2,
            admission.FAIL_DEPENDENCY_V2,
        )


def test_same_session_future_result_reference_is_rejected(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    root = copy.deepcopy(positive_corpus.roots["challenge"])
    body = root["body"]
    body["admission_v2_candidate_result_id"] = positive_corpus.roots["result"][
        "admission_v2_candidate_result_id"
    ]
    content_id = admission.phase2b_step5_external_witness_admission_content_id_v2(
        "pre_response_challenge", body
    )
    root["pre_response_challenge_id"] = content_id
    source, source_private = _role(positive_corpus.base, "SOURCE_RUNTIME_CUSTODIAN")
    root["signatures_in_exact_order"] = [
        _authority_signature(
            kind="pre_response_challenge",
            body=body,
            row=source,
            private_key=source_private,
            purpose=admission.SIGN_SOURCE_PURPOSE_V2,
        )
    ]
    call = dict(positive_corpus.call)
    call["pre_response_challenge_payload"] = (
        admission.canonical_external_witness_admission_json_bytes_v2(root)
    )
    _assert_rejected(call, admission.FAIL_SCHEMA_V2, admission.FAIL_ACYCLIC_V2)


def test_signed_body_drift_fails_ed25519_after_readdressing(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    root = copy.deepcopy(positive_corpus.roots["live_policy"])
    root["body"]["cas_chain_id_profile"] += "_SIGNED_DRIFT"
    root["admission_v2_live_policy_id"] = (
        admission.phase2b_step5_external_witness_admission_content_id_v2(
            "admission_v2_live_policy", root["body"]
        )
    )
    payload = admission.canonical_external_witness_admission_json_bytes_v2(root)
    call = dict(positive_corpus.call)
    call["admission_v2_live_policy_payload"] = payload
    call["out_of_band_expected_live_runtime_policy_id_bytes"] = root[
        "admission_v2_live_policy_id"
    ].encode("ascii")
    call[
        "out_of_band_expected_live_runtime_policy_payload_sha256_digest"
    ] = hashlib.sha256(payload).digest()
    _assert_rejected(call, admission.FAIL_SIGNATURE_V2)


@pytest.mark.parametrize(
    ("field", "hostile"),
    (
        ("role", "PURE_REPLAYER"),
        ("purpose", admission.SIGN_PURE_RESULT_PURPOSE_V2),
        ("key_epoch", 2),
    ),
)
def test_signature_role_purpose_and_epoch_are_exact(
    positive_corpus: _AdmissionV2Corpus,
    field: str,
    hostile: object,
) -> None:
    root = copy.deepcopy(positive_corpus.roots["challenge"])
    root["signatures_in_exact_order"][0][field] = hostile
    call = dict(positive_corpus.call)
    call["pre_response_challenge_payload"] = (
        admission.canonical_external_witness_admission_json_bytes_v2(root)
    )
    _assert_rejected(call, admission.FAIL_SIGNATURE_V2)


@pytest.mark.parametrize(
    ("field", "hostile"),
    (("execution_authorized", 0), ("verified_protocol_object_count", True)),
)
def test_json_bool_as_int_cannot_launder_claims_or_counts(
    positive_corpus: _AdmissionV2Corpus,
    field: str,
    hostile: object,
) -> None:
    call = dict(positive_corpus.call)
    call["admission_v2_candidate_result_payload"] = _resigned_payload(
        positive_corpus,
        "result",
        lambda body: body.__setitem__(field, hostile),
    )
    _assert_rejected(call, admission.FAIL_CLAIM_BOUNDARY_V2)


def test_b480_q1_and_q2_observation_transports_cannot_alias(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    call = dict(positive_corpus.call)
    call["q2_confirm_response_payload"] = call["q1_prepare_response_payload"]
    _assert_rejected(call, admission.FAIL_DEPENDENCY_V2)


def test_exact_six_object_cas_order_cannot_be_permuted(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    call = dict(positive_corpus.call)
    objects = list(call["admission_v2_cas_object_payloads_in_exact_order"])
    objects[0], objects[1] = objects[1], objects[0]
    call["admission_v2_cas_object_payloads_in_exact_order"] = tuple(objects)
    _assert_rejected(call, admission.FAIL_BINDING_V2)


def test_legally_resigned_a2_proposed_mapping_drift_is_rejected(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    def mutate(body: dict[str, object]) -> None:
        proposed = body["proposed_current_mapping"]
        assert type(proposed) is dict
        proposed["revoked"] = True

    call = dict(positive_corpus.call)
    call["admission_v2_a2_request_payload"] = _resigned_payload(
        positive_corpus, "a2_request", mutate
    )
    _assert_rejected(call, admission.FAIL_BINDING_V2)


def test_verified_result_private_forge_copy_and_mutation_are_rejected(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    value = admission.verify_external_witness_admission_candidate_v2(
        **positive_corpus.call
    )
    assert admission.require_verified_external_witness_admission_candidate_v2(
        value
    ) is value
    with pytest.raises(TypeError):
        admission.VerifiedExternalWitnessAdmissionCandidateV2()
    forged = object.__new__(admission.VerifiedExternalWitnessAdmissionCandidateV2)
    for field in value.__dataclass_fields__:
        object.__setattr__(forged, field, getattr(value, field))
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2
    ) as forged_error:
        admission.require_verified_external_witness_admission_candidate_v2(forged)
    assert forged_error.value.code == admission.FAIL_RESULT_V2
    with pytest.raises(TypeError):
        copy.copy(value)
    with pytest.raises(TypeError):
        copy.deepcopy(value)
    object.__setattr__(value, "authority_evidence", True)
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2
    ) as mutation_error:
        admission.require_verified_external_witness_admission_candidate_v2(value)
    assert mutation_error.value.code == admission.FAIL_RESULT_V2


def _captured_implementation() -> FunctionType:
    cells = admission.verify_external_witness_admission_candidate_v2.__closure__
    assert cells is not None and len(cells) == 1
    implementation = cells[0].cell_contents
    assert type(implementation) is FunctionType
    assert implementation.__name__ == (
        "_verify_external_witness_admission_candidate_implementation_v2"
    )
    return implementation


def _bomb(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("EVIL post-import global replacement was reached")


def test_definition_capture_survives_module_and_dependency_global_rebinding(
    positive_corpus: _AdmissionV2Corpus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = _captured_implementation()
    assert implementation.__globals__["_canonical"] is not admission._canonical
    kwdefaults = implementation.__kwdefaults__
    assert kwdefaults is not None
    assert kwdefaults["_verify_b480"] is _COLLECTION_B480_VERIFY
    assert kwdefaults["_require_b480"] is _COLLECTION_B480_REQUIRE
    assert kwdefaults["_verify_closure"] is _COLLECTION_CLOSURE_VERIFY
    assert kwdefaults["_require_closure"] is _COLLECTION_CLOSURE_REQUIRE
    captured_cas = implementation.__globals__["_verify_cas_session"]
    assert type(captured_cas) is FunctionType
    cas_defaults = captured_cas.__kwdefaults__
    assert cas_defaults is not None
    assert cas_defaults["_verify"] is _COLLECTION_REPLAY_VERIFY
    assert cas_defaults["_require"] is _COLLECTION_REPLAY_REQUIRE
    for name in (
        "_canonical",
        "_decode",
        "_load_root",
        "_load_challenge",
        "_load_result",
        "_verify_cas_session",
        "_load_a2_transcript",
    ):
        monkeypatch.setattr(admission, name, _bomb)
    monkeypatch.setattr(
        admission_v1,
        "verify_external_witness_admission_transcript_candidate_v1",
        _bomb,
    )
    monkeypatch.setattr(
        admission_v1,
        "require_verified_external_witness_admission_candidate_v1",
        _bomb,
    )
    monkeypatch.setattr(
        closure, "verify_offline_step5_image_closure_mechanics_v1", _bomb
    )
    monkeypatch.setattr(
        closure, "require_verified_offline_step5_image_closure_mechanics_v1", _bomb
    )
    monkeypatch.setattr(
        replay, "verify_cas_store_complete_session_replay_core_v2", _bomb
    )
    monkeypatch.setattr(
        replay, "require_verified_cas_store_complete_session_replay_core_v2", _bomb
    )
    value = admission.verify_external_witness_admission_candidate_v2(
        **positive_corpus.call
    )
    assert admission.require_verified_external_witness_admission_candidate_v2(
        value
    ) is value


def test_clean_import_excludes_live_producers_store_and_service_modules() -> None:
    source_root = str(Path(admission.__file__).resolve().parents[1])
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        sys.path.insert(0, {source_root!r})
        blocked = {{
            "hegel_machine.phase2b_formal_wire_covert_step5_logical_time_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_r01_external_witness_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_r01_online_witness_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_r01_current_cut_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_r01_durable_anchor_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_owner_status_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_runtime_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_service_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_store_v1",
            "hegel_machine.phase2b_formal_wire_covert_step5_store_v2",
        }}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked:
                    raise RuntimeError("blocked live/producer import: " + fullname)
                return None

        sys.meta_path.insert(0, Blocker())
        import hegel_machine.phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v2
        assert blocked.isdisjoint(sys.modules)
        print("CLEAN_IMPORT_OK")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "CLEAN_IMPORT_OK"


def test_all_elevated_claims_remain_false_and_real_admission_is_not_implemented(
    positive_corpus: _AdmissionV2Corpus,
) -> None:
    value = admission.require_verified_external_witness_admission_candidate_v2(
        admission.verify_external_witness_admission_candidate_v2(
            **positive_corpus.call
        )
    )
    for field in (
        "actual_runtime_verified",
        "runtime_closure_verified",
        "live_channel_verified",
        "key_custody_verified",
        "online_key_nonexportable_verified",
        "distinct_live_uid_gid_verified",
        "external_cas_verified",
        "external_anchor_verified",
        "external_anti_rollback_verified",
        "external_currentness_verified",
        "durability_verified",
        "non_equivocation_verified",
        "global_anti_fork_verified",
        "malicious_root_resistance_verified",
        "authority_evidence",
        "r01_satisfied",
        "readiness_evidence",
        "execution_authorized",
    ):
        assert getattr(value, field) is False
    with pytest.raises(
        admission.Phase2BStep5R01ExternalWitnessAdmissionContractErrorV2
    ) as not_implemented:
        admission.verify_real_external_witness_admission_v2()
    assert not_implemented.value.code == admission.FAIL_NOT_IMPLEMENTED_V2
