from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from hegel_machine.rfc8785_v1 import (
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)
import hegel_machine.rfc8785_v1 as rfc8785_module
import hegel_machine.phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1 as catalog
import hegel_machine.phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1 as mechanics


EXPECTED_CORPUS_FINGERPRINT_V1 = (
    "30e3a9a4b5ba847e07d8121d507f8650ca33dc7584bad20d760ccb43ea1c10d4"
)


def _canonical(value: object) -> bytes:
    return canonicalize_rfc8785_v1(value)  # type: ignore[arg-type]


def _raw_sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _stable_hex(label: str) -> str:
    return hashlib.sha256(("package0-kat-v1/" + label).encode("ascii")).hexdigest()


def _stable_id(prefix: str, label: str) -> str:
    return prefix + _stable_hex(label)


def _body_content_id(
    domain_ascii: str,
    prefix: str,
    body: dict[str, object],
) -> str:
    return prefix + hashlib.sha256(
        domain_ascii.encode("ascii") + _canonical(body)
    ).hexdigest()


def _public_bytes(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _kat_private_keys_by_role() -> dict[str, Ed25519PrivateKey]:
    return {
        role: Ed25519PrivateKey.from_private_bytes(bytes([index + 1]) * 32)
        for index, role in enumerate(
            catalog.SIGNER_REGISTRY_ROLES_IN_EXACT_ORDER_V1
        )
    }


def _kat_key_ids_by_role() -> dict[str, str]:
    return {
        role: _stable_id(
            "phase2b_actual_unsealed_960_signer_key_v1_",
            "registry-key/" + role,
        )
        for role in catalog.SIGNER_REGISTRY_ROLES_IN_EXACT_ORDER_V1
    }


def _generic_signature(
    *,
    private_key: Ed25519PrivateKey,
    signer_key_id: str,
    descriptor: dict[str, object],
    artifact_schema_version: str,
    content_id: str,
    body: dict[str, object],
    slot: dict[str, object],
) -> dict[str, object]:
    preimage = {
        "schema_version": descriptor["signature_preimage_schema_version"],
        "artifact_kind": descriptor["signature_preimage_artifact_kind"],
        "artifact_schema_version": artifact_schema_version,
        "content_id": content_id,
        "body": body,
        "signature_ordinal": slot["ordinal"],
        "role": slot["role"],
        "purpose": slot["purpose"],
        "key_epoch": slot["key_epoch"],
        "signer_key_id": signer_key_id,
    }
    signed = str(slot["signature_domain_ascii"]).encode("ascii") + _canonical(
        preimage
    )
    return {
        "schema_version": catalog.DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1,
        "algorithm": "Ed25519",
        "role": slot["role"],
        "purpose": slot["purpose"],
        "key_epoch": slot["key_epoch"],
        "signer_key_id": signer_key_id,
        "signature_hex": private_key.sign(signed).hex(),
    }


def _package_envelope(
    descriptor: dict[str, object],
    body: dict[str, object],
    *,
    private_keys_by_role: dict[str, Ed25519PrivateKey],
    key_ids_by_role: dict[str, str],
) -> tuple[str, bytes]:
    package_id = _body_content_id(
        str(descriptor["content_id_domain_ascii"]),
        str(descriptor["content_id_prefix"]),
        body,
    )
    signatures = [
        _generic_signature(
            private_key=private_keys_by_role[str(slot["role"])],
            signer_key_id=key_ids_by_role[str(slot["role"])],
            descriptor=descriptor,
            artifact_schema_version=str(descriptor["schema_version"]),
            content_id=package_id,
            body=body,
            slot=slot,
        )
        for slot in descriptor["signature_slots_in_exact_order"]  # type: ignore[union-attr]
    ]
    return package_id, _canonical(
        {
            "schema_version": descriptor["schema_version"],
            "body": body,
            "package_id": package_id,
            "signatures_in_exact_order": signatures,
        }
    )


def _referenced_envelope(
    contract: dict[str, object],
    body: dict[str, object],
    *,
    private_keys_by_role: dict[str, Ed25519PrivateKey],
    key_ids_by_role: dict[str, str],
) -> tuple[str, bytes]:
    artifact_id = _body_content_id(
        str(contract["record_content_id_domain_ascii"]),
        str(contract["record_content_id_prefix"]),
        body,
    )
    signature_descriptor = {
        "signature_preimage_schema_version": contract[
            "signature_preimage_schema_version"
        ],
        "signature_preimage_artifact_kind": contract[
            "signature_preimage_artifact_kind"
        ],
        "signature_slots_in_exact_order": contract[
            "signature_slots_in_exact_order"
        ],
    }
    signatures = [
        _generic_signature(
            private_key=private_keys_by_role[str(slot["role"])],
            signer_key_id=key_ids_by_role[str(slot["role"])],
            descriptor=signature_descriptor,
            artifact_schema_version=str(contract["record_schema_version"]),
            content_id=artifact_id,
            body=body,
            slot=slot,
        )
        for slot in contract["signature_slots_in_exact_order"]  # type: ignore[union-attr]
    ]
    return artifact_id, _canonical(
        {
            "schema_version": contract["record_schema_version"],
            "body": body,
            "artifact_id": artifact_id,
            "signatures_in_exact_order": signatures,
        }
    )


@dataclass(frozen=True)
class _Package0CorpusV1:
    positional: tuple[object, ...]
    keyword_only: dict[str, bytes]
    expected: dict[str, object]
    fingerprint: str


def _build_package0_corpus_v1() -> _Package0CorpusV1:
    package_schemas = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    stage_schemas = catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    gate_schemas = {
        str(item["gate_id"]): item
        for item in catalog.GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    }
    signer_schema = catalog.SIGNER_REGISTRY_SCHEMA_OBJECT_V1
    package0_schema = package_schemas[0]
    ledger_contract = package0_schema[
        "nested_record_binding_contracts_in_exact_order"
    ][0]
    referenced_contracts = {
        str(item["record_name"]): item
        for item in package0_schema[
            "nested_record_binding_contracts_in_exact_order"
        ][1:6]
    }

    roles = tuple(catalog.SIGNER_REGISTRY_ROLES_IN_EXACT_ORDER_V1)
    private_keys_by_role = _kat_private_keys_by_role()
    public_keys_by_role = {
        role: _public_bytes(private_keys_by_role[role]) for role in roles
    }
    key_ids_by_role = _kat_key_ids_by_role()
    governance_root_private = Ed25519PrivateKey.from_private_bytes(b"\xf1" * 32)
    custodian_root_private = Ed25519PrivateKey.from_private_bytes(b"\xf2" * 32)
    root_private_by_role = {
        "GOVERNANCE_ANCHOR": governance_root_private,
        "CUSTODIAN": custodian_root_private,
    }
    root_key_ids_by_role = {
        role: _stable_id(
            "phase2b_actual_unsealed_960_signer_key_v1_",
            "oob-root-key/" + role,
        )
        for role in root_private_by_role
    }
    root_public_by_role = {
        role: _public_bytes(private_key)
        for role, private_key in root_private_by_role.items()
    }

    authorization_rows = [
        dict(row)
        for row in signer_schema["required_authorization_rows_in_exact_order"]
    ]
    registry_entries: list[dict[str, object]] = []
    for index, role in enumerate(roles):
        purposes: list[str] = []
        for row in authorization_rows:
            if row["role"] == role and row["purpose"] not in purposes:
                purposes.append(str(row["purpose"]))
        registry_entries.append(
            {
                "entry_ordinal": str(index),
                "role": role,
                "control_domain_id": _stable_id(
                    "phase2b_actual_unsealed_960_control_domain_v1_",
                    "control-domain/" + role,
                ),
                "key_id": key_ids_by_role[role],
                "key_epoch": 1,
                "algorithm": "Ed25519",
                "public_key_hex": public_keys_by_role[role].hex(),
                "authorized_signature_purposes_in_exact_order": purposes,
            }
        )
    entries_root = _raw_sha(
        str(signer_schema["entry_root_domain_ascii"]).encode("ascii")
        + _canonical(registry_entries)
    )
    authorization_root = _raw_sha(
        str(signer_schema["authorization_root_domain_ascii"]).encode("ascii")
        + _canonical(authorization_rows)
    )
    registry_body = {
        "schema_version": signer_schema["schema_version"],
        "status": mechanics.CONTRACT_BODY_V1["signer_registry_required_literals"][
            "status"
        ],
        "claim_level": mechanics.CONTRACT_BODY_V1[
            "signer_registry_required_literals"
        ]["claim_level"],
        "catalog_id": catalog.CATALOG_ID_V1,
        "registry_epoch": "1",
        "entries_in_exact_order": registry_entries,
        "entries_root_sha256": entries_root,
        "authorized_signature_rows_in_exact_order": authorization_rows,
        "authorized_signature_rows_root_sha256": authorization_root,
    }
    registry_id = _body_content_id(
        str(signer_schema["content_id_domain_ascii"]),
        str(signer_schema["content_id_prefix"]),
        registry_body,
    )
    registry_signatures = [
        _generic_signature(
            private_key=root_private_by_role[str(slot["role"])],
            signer_key_id=root_key_ids_by_role[str(slot["role"])],
            descriptor=signer_schema,
            artifact_schema_version=str(signer_schema["schema_version"]),
            content_id=registry_id,
            body=registry_body,
            slot=slot,
        )
        for slot in signer_schema["signature_slots_in_exact_order"]
    ]
    raw_registry = _canonical(
        {
            "schema_version": signer_schema["schema_version"],
            "body": registry_body,
            "signer_registry_id": registry_id,
            "signatures_in_exact_order": registry_signatures,
        }
    )
    registry_sha = _raw_sha(raw_registry)

    campaign_id = _stable_id(
        mechanics.CAMPAIGN_ID_PREFIX_V1,
        "campaign",
    )
    attempt_id = _stable_id(
        mechanics.ATTEMPT_ID_PREFIX_V1,
        "attempt",
    )

    def package_common(descriptor: dict[str, object]) -> dict[str, object]:
        return {
            "schema_version": descriptor["schema_version"],
            "status": catalog.PACKAGE_OUTER_STATUS_V1,
            "claim_level": catalog.PACKAGE_OUTER_CLAIM_LEVEL_V1,
            "catalog_id": catalog.CATALOG_ID_V1,
            "signer_registry_id": registry_id,
            "signer_registry_payload_sha256": registry_sha,
            "signer_registry_entries_root_sha256": entries_root,
            "package_ordinal": descriptor["ordinal"],
            "package_name": descriptor["package_name"],
            "external_evidence_class": descriptor[
                "required_external_evidence_class"
            ],
            "external_evidence_disposition": descriptor[
                "producer_evidence_disposition"
            ],
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
        }

    def placeholder_package_pair(ordinal: int) -> tuple[str, str]:
        descriptor = package_schemas[ordinal]
        label = f"placeholder-package/{ordinal}"
        return (
            _stable_id(str(descriptor["content_id_prefix"]), label),
            _raw_sha(label.encode("ascii")),
        )

    p15_predecessors = [placeholder_package_pair(i) for i in range(10, 15)]
    gate_result_ids = [
        _stable_id(
            "phase2b_actual_unsealed_960_gate_result_v1_",
            f"gate-result/{index}",
        )
        for index in range(36)
    ]
    gate_result_shas = [
        _raw_sha(f"gate-result/{index}".encode("ascii")) for index in range(36)
    ]
    package15_body = {
        **package_common(package_schemas[15]),
        "predecessor_package_instance_ids_in_exact_order": [
            item[0] for item in p15_predecessors
        ],
        "predecessor_package_instance_payload_sha256s_in_exact_order": [
            item[1] for item in p15_predecessors
        ],
        "interleaved_dag_root": catalog.DAG_ROOT_V1,
        "scored_rows_root_sha256": _raw_sha(b"package0-kat/scored-rows"),
        "metric_counts_id": _stable_id(
            "phase2b_actual_unsealed_960_metric_counts_v1_",
            "metric-counts",
        ),
        "metric_counts_payload_sha256": _raw_sha(b"package0-kat/metric-counts"),
        "wilson_report_id": _stable_id(
            "phase2b_actual_unsealed_960_wilson_report_v1_",
            "wilson-report",
        ),
        "wilson_report_payload_sha256": _raw_sha(b"package0-kat/wilson-report"),
        "overall_gate_count": 12,
        "slice_gate_count": 24,
        "gate_result_ids_in_exact_order": gate_result_ids,
        "gate_result_payload_sha256s_in_exact_order": gate_result_shas,
        "all_gate_results_root_sha256": _raw_sha(b"package0-kat/all-gate-results"),
    }
    package15_id, raw_package15 = _package_envelope(
        package_schemas[15],
        package15_body,
        private_keys_by_role=private_keys_by_role,
        key_ids_by_role=key_ids_by_role,
    )
    package10_pair = p15_predecessors[0]
    package16_body = {
        **package_common(package_schemas[16]),
        "predecessor_package_instance_ids_in_exact_order": [
            package10_pair[0],
            package15_id,
        ],
        "predecessor_package_instance_payload_sha256s_in_exact_order": [
            package10_pair[1],
            _raw_sha(raw_package15),
        ],
        "interleaved_dag_root": catalog.DAG_ROOT_V1,
        "attempt_record_ids_in_exact_order": [
            _stable_id(
                "phase2b_actual_unsealed_960_attempt_record_v1_",
                "attempt-record/0",
            )
        ],
        "attempt_record_payload_sha256s_in_exact_order": [
            _raw_sha(b"package0-kat/attempt-record/0")
        ],
        "attempt_indices_in_exact_order": [0],
        "execution_retry_action": "NONE",
        "execution_retry_reason": "NONE",
        "upload_retry_action": "NONE",
        "upload_retry_reason": "NONE",
        "any_raw_prediction_byte_produced": True,
        "raw_output_byte_count": 1,
        "attempt_universe_head_id": _stable_id(
            "phase2b_actual_unsealed_960_attempt_universe_head_v1_",
            "attempt-universe-head",
        ),
        "attempt_universe_head_payload_sha256": _raw_sha(
            b"package0-kat/attempt-universe-head"
        ),
        "permanent_history_root_sha256": _raw_sha(
            b"package0-kat/permanent-history"
        ),
    }
    package16_id, raw_package16 = _package_envelope(
        package_schemas[16],
        package16_body,
        private_keys_by_role=private_keys_by_role,
        key_ids_by_role=key_ids_by_role,
    )

    stage_records: list[tuple[str, bytes]] = []
    for index, descriptor in enumerate(stage_schemas):
        predecessor_names = list(descriptor["predecessor_nodes_in_exact_order"])
        predecessor_ids: list[str] = []
        predecessor_shas: list[str] = []
        for node_name in predecessor_names:
            if node_name in catalog.STAGE_NAMES_IN_EXACT_ORDER_V1:
                prior_index = catalog.STAGE_NAMES_IN_EXACT_ORDER_V1.index(node_name)
                predecessor_ids.append(stage_records[prior_index][0])
                predecessor_shas.append(_raw_sha(stage_records[prior_index][1]))
            else:
                gate_descriptor = gate_schemas[str(node_name)]
                predecessor_ids.append(
                    _stable_id(
                        str(gate_descriptor["content_id_prefix"]),
                        "gate-predecessor/" + str(node_name),
                    )
                )
                predecessor_shas.append(
                    _raw_sha(
                        ("package0-kat/gate-predecessor/" + str(node_name)).encode(
                            "ascii"
                        )
                    )
                )
        if index == 9:
            subject_kinds = list(
                mechanics.TERMINAL_SUBJECT_ARTIFACT_KINDS_IN_EXACT_ORDER_V1
            )
            subject_ids = [package15_id, package16_id]
            subject_shas = [_raw_sha(raw_package15), _raw_sha(raw_package16)]
        else:
            subject_kinds = [f"package0_kat_stage_{index}_subject_v1"]
            subject_ids = [
                _stable_id(
                    "phase2b_actual_unsealed_960_stage_subject_v1_",
                    f"stage-subject/{index}",
                )
            ]
            subject_shas = [
                _raw_sha(f"package0-kat/stage-subject/{index}".encode("ascii"))
            ]
        stage_body = {
            "schema_version": descriptor["event_schema_version"],
            "status": mechanics.STAGE_EVENT_STATUS_V1,
            "claim_level": mechanics.STAGE_EVENT_CLAIM_LEVEL_V1,
            "catalog_id": catalog.CATALOG_ID_V1,
            "signer_registry_id": registry_id,
            "signer_registry_payload_sha256": registry_sha,
            "signer_registry_entries_root_sha256": entries_root,
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "stage_ordinal": index,
            "stage": descriptor["stage"],
            "predecessor_node_names_in_exact_order": predecessor_names,
            "predecessor_evidence_or_event_ids_in_exact_order": predecessor_ids,
            "predecessor_evidence_or_event_payload_sha256s_in_exact_order": predecessor_shas,
            "subject_artifact_kinds_in_exact_order": subject_kinds,
            "subject_artifact_ids_in_exact_order": subject_ids,
            "subject_artifact_payload_sha256s_in_exact_order": subject_shas,
            "event_sequence": index,
            "terminal_disposition_or_none": "COMPLETE_FAIL" if index == 9 else None,
            "stage_occurred": True,
            "first_failure_code": "NONE",
        }
        stage_id = _body_content_id(
            str(descriptor["content_id_domain_ascii"]),
            str(descriptor["content_id_prefix"]),
            stage_body,
        )
        stage_signatures = [
            _generic_signature(
                private_key=private_keys_by_role[str(slot["role"])],
                signer_key_id=key_ids_by_role[str(slot["role"])],
                descriptor=descriptor,
                artifact_schema_version=str(descriptor["event_schema_version"]),
                content_id=stage_id,
                body=stage_body,
                slot=slot,
            )
            for slot in descriptor["signature_slots_in_exact_order"]
        ]
        raw_stage = _canonical(
            {
                "schema_version": descriptor["event_schema_version"],
                "body": stage_body,
                "stage_event_id": stage_id,
                "signatures_in_exact_order": stage_signatures,
            }
        )
        stage_records.append((stage_id, raw_stage))

    ledger_records: list[tuple[str, bytes]] = []
    for index, ((stage_id, raw_stage), slot) in enumerate(
        zip(
            stage_records,
            ledger_contract["event_signature_slots_in_exact_stage_order"],
            strict=True,
        )
    ):
        ledger_body = {
            "schema_version": ledger_contract["event_body_schema_version"],
            "catalog_id": catalog.CATALOG_ID_V1,
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "event_index": index,
            "stage": catalog.STAGE_NAMES_IN_EXACT_ORDER_V1[index],
            "predecessor_event_id_or_none": (
                None if index == 0 else ledger_records[index - 1][0]
            ),
            "payload_id": stage_id,
            "payload_sha256": _raw_sha(raw_stage),
            "signer_registry_id": registry_id,
            "signer_key_id": key_ids_by_role["CUSTODIAN"],
        }
        signature_preimage = {
            "schema_version": ledger_contract[
                "event_signature_preimage_schema_version"
            ],
            "artifact_kind": f"signed_custodian_stage_event_{index}_v1",
            "artifact_schema_version": ledger_contract["record_schema_version"],
            "algorithm": "Ed25519",
            "body": ledger_body,
            "signature_ordinal": 0,
            "role": "CUSTODIAN",
            "purpose": slot["purpose"],
            "key_epoch": 1,
            "signer_key_id": key_ids_by_role["CUSTODIAN"],
        }
        ledger_signature = {
            "schema_version": catalog.DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1,
            "algorithm": "Ed25519",
            "role": "CUSTODIAN",
            "purpose": slot["purpose"],
            "key_epoch": 1,
            "signer_key_id": key_ids_by_role["CUSTODIAN"],
            "signature_hex": private_keys_by_role["CUSTODIAN"].sign(
                str(slot["signature_domain_ascii"]).encode("ascii")
                + _canonical(signature_preimage)
            ).hex(),
        }
        ledger_id_subject = {
            "schema_version": ledger_contract["record_schema_version"],
            "body": ledger_body,
            "signatures_in_exact_order": [ledger_signature],
        }
        ledger_id = _stable_id(
            str(ledger_contract["record_content_id_prefix"]),
            "unused",
        )
        ledger_id = str(ledger_contract["record_content_id_prefix"]) + hashlib.sha256(
            str(ledger_contract["record_content_id_domain_ascii"]).encode("ascii")
            + _canonical(ledger_id_subject)
        ).hexdigest()
        raw_ledger = _canonical(
            {
                **ledger_id_subject,
                "signed_stage_event_id": ledger_id,
            }
        )
        ledger_records.append((ledger_id, raw_ledger))

    chain_pairs = [
        {
            "signed_stage_event_id": event_id,
            "signed_stage_event_payload_sha256": _raw_sha(raw),
        }
        for event_id, raw in ledger_records
    ]
    chain_root = _raw_sha(
        mechanics.LEDGER_EVENT_CHAIN_ROOT_DOMAIN_V1 + _canonical(chain_pairs)
    )
    chain_id = mechanics.LEDGER_EVENT_CHAIN_ID_PREFIX_V1 + chain_root[7:]
    terminal_ledger_id, raw_terminal_ledger = ledger_records[9]

    store_contract = referenced_contracts["store_if_absent_receipt"]
    store_body = {
        "schema_version": store_contract["record_schema_version"],
        "catalog_id": catalog.CATALOG_ID_V1,
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "transition_from": "ABSENT",
        "transition_to": "PRESENT",
        "mode": "IF_ABSENT",
        "stored_artifact_id": terminal_ledger_id,
        "stored_artifact_payload_sha256": _raw_sha(raw_terminal_ledger),
        "stored_artifact_size_bytes": len(raw_terminal_ledger),
        "append_applied": True,
        "readback_verified": True,
    }
    store_id, raw_store = _referenced_envelope(
        store_contract,
        store_body,
        private_keys_by_role=private_keys_by_role,
        key_ids_by_role=key_ids_by_role,
    )
    time_contract = referenced_contracts["logical_time_token"]
    time_body = {
        "schema_version": time_contract["record_schema_version"],
        "catalog_id": catalog.CATALOG_ID_V1,
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "previous_logical_time": 40,
        "current_logical_time": 41,
        "subject_id": terminal_ledger_id,
        "subject_payload_sha256": _raw_sha(raw_terminal_ledger),
    }
    time_id, raw_time = _referenced_envelope(
        time_contract,
        time_body,
        private_keys_by_role=private_keys_by_role,
        key_ids_by_role=key_ids_by_role,
    )

    def checkpoint(name: str) -> tuple[str, bytes]:
        contract = referenced_contracts[name]
        body = {
            "schema_version": contract["record_schema_version"],
            "catalog_id": catalog.CATALOG_ID_V1,
            "campaign_id": campaign_id,
            "attempt_id": attempt_id,
            "chain_id": chain_id,
            "chain_root_sha256": chain_root,
            "store_receipt_id": store_id,
            "store_receipt_payload_sha256": _raw_sha(raw_store),
            "time_token_id": time_id,
            "time_token_payload_sha256": _raw_sha(raw_time),
            "terminal_signed_ledger_event_id": terminal_ledger_id,
            "terminal_signed_ledger_event_payload_sha256": _raw_sha(
                raw_terminal_ledger
            ),
        }
        return _referenced_envelope(
            contract,
            body,
            private_keys_by_role=private_keys_by_role,
            key_ids_by_role=key_ids_by_role,
        )

    governance_id, raw_governance = checkpoint("governance_checkpoint")
    custodian_id, raw_custodian = checkpoint("custodian_checkpoint")
    anchor_contract = referenced_contracts["external_anchor"]
    anchor_body = {
        "schema_version": anchor_contract["record_schema_version"],
        "catalog_id": catalog.CATALOG_ID_V1,
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "chain_id": chain_id,
        "chain_root_sha256": chain_root,
        "anchor_service_id": "kat_anchor_service_v1",
        "anchor_locator": "kat:anchor:0",
        "governance_checkpoint_id": governance_id,
        "governance_checkpoint_payload_sha256": _raw_sha(raw_governance),
        "custodian_checkpoint_id": custodian_id,
        "custodian_checkpoint_payload_sha256": _raw_sha(raw_custodian),
        "previous_external_anchor_id_or_none": None,
        "anchor_sequence": 0,
        "effective_state": "ACTIVE",
        "superseded": False,
        "revoked": False,
    }
    anchor_id, raw_anchor = _referenced_envelope(
        anchor_contract,
        anchor_body,
        private_keys_by_role=private_keys_by_role,
        key_ids_by_role=key_ids_by_role,
    )

    package0_body = {
        **package_common(package0_schema),
        "predecessor_package_instance_ids_in_exact_order": [],
        "predecessor_package_instance_payload_sha256s_in_exact_order": [],
        "interleaved_dag_root": catalog.DAG_ROOT_V1,
        "signed_stage_event_ids_in_exact_order": [
            item[0] for item in ledger_records
        ],
        "signed_stage_event_payload_sha256s_in_exact_order": [
            _raw_sha(item[1]) for item in ledger_records
        ],
        "store_if_absent_receipt_id": store_id,
        "store_if_absent_receipt_payload_sha256": _raw_sha(raw_store),
        "logical_time_token_id": time_id,
        "logical_time_token_payload_sha256": _raw_sha(raw_time),
        "governance_checkpoint_id": governance_id,
        "governance_checkpoint_payload_sha256": _raw_sha(raw_governance),
        "custodian_checkpoint_id": custodian_id,
        "custodian_checkpoint_payload_sha256": _raw_sha(raw_custodian),
        "external_anchor_id": anchor_id,
        "external_anchor_payload_sha256": _raw_sha(raw_anchor),
        "chain_root_sha256": chain_root,
    }
    package0_id, raw_package0 = _package_envelope(
        package0_schema,
        package0_body,
        private_keys_by_role=private_keys_by_role,
        key_ids_by_role=key_ids_by_role,
    )

    positional: tuple[object, ...] = (
        catalog.CATALOG_PAYLOAD_V1,
        raw_registry,
        raw_package0,
        tuple(raw for _, raw in ledger_records),
        tuple(raw for _, raw in stage_records),
        raw_store,
        raw_time,
        raw_governance,
        raw_custodian,
        raw_anchor,
        raw_package15,
        raw_package16,
    )
    keyword_only = {
        "out_of_band_expected_catalog_id_bytes": catalog.CATALOG_ID_V1.encode(
            "ascii"
        ),
        "out_of_band_expected_catalog_payload_sha256_digest": hashlib.sha256(
            catalog.CATALOG_PAYLOAD_V1
        ).digest(),
        "out_of_band_expected_signer_registry_id_bytes": registry_id.encode(
            "ascii"
        ),
        "out_of_band_expected_signer_registry_payload_sha256_digest": hashlib.sha256(
            raw_registry
        ).digest(),
        "out_of_band_expected_signer_registry_entries_root_sha256_digest": bytes.fromhex(
            entries_root[7:]
        ),
        "out_of_band_expected_governance_root_key_id_bytes": root_key_ids_by_role[
            "GOVERNANCE_ANCHOR"
        ].encode("ascii"),
        "out_of_band_expected_governance_root_public_key_bytes": root_public_by_role[
            "GOVERNANCE_ANCHOR"
        ],
        "out_of_band_expected_custodian_root_key_id_bytes": root_key_ids_by_role[
            "CUSTODIAN"
        ].encode("ascii"),
        "out_of_band_expected_custodian_root_public_key_bytes": root_public_by_role[
            "CUSTODIAN"
        ],
        "out_of_band_expected_readiness_dag_id_bytes": catalog.INSPECTED_READINESS_DAG_ID_V1.encode(
            "ascii"
        ),
        "out_of_band_expected_readiness_dag_payload_sha256_digest": bytes.fromhex(
            catalog.INSPECTED_READINESS_DAG_CONFIG_SHA256_V1
        ),
        "out_of_band_expected_progression_plan_payload_sha256_digest": bytes.fromhex(
            catalog.INSPECTED_PROGRESSION_PLAN_SHA256_V1
        ),
        "out_of_band_expected_gap_inventory_source_sha256_digest": bytes.fromhex(
            catalog.INSPECTED_GAP_INVENTORY_SOURCE_SHA256_V1
        ),
        "out_of_band_expected_statistical_validation_source_sha256_digest": bytes.fromhex(
            catalog.INSPECTED_STATISTICAL_VALIDATION_SOURCE_SHA256_V1
        ),
        "expected_campaign_id_bytes": campaign_id.encode("ascii"),
        "expected_attempt_id_bytes": attempt_id.encode("ascii"),
    }
    fingerprint_hasher = hashlib.sha256()
    for value in positional:
        items = value if type(value) is tuple else (value,)
        for item in items:
            assert type(item) is bytes
            fingerprint_hasher.update(len(item).to_bytes(8, "big"))
            fingerprint_hasher.update(item)
    for name, value in keyword_only.items():
        fingerprint_hasher.update(name.encode("ascii") + b"\0")
        fingerprint_hasher.update(len(value).to_bytes(8, "big"))
        fingerprint_hasher.update(value)
    expected = {
        "catalog_id": catalog.CATALOG_ID_V1,
        "package_schema_id": package0_schema["package_schema_id"],
        "package_id": package0_id,
        "package_payload_sha256": _raw_sha(raw_package0),
        "signer_registry_id": registry_id,
        "signer_registry_payload_sha256": registry_sha,
        "signer_registry_entries_root_sha256": entries_root,
        "campaign_id": campaign_id,
        "attempt_id": attempt_id,
        "chain_root_sha256": chain_root,
        "terminal_signed_stage_event_id": terminal_ledger_id,
        "terminal_signed_stage_event_payload_sha256": _raw_sha(
            raw_terminal_ledger
        ),
    }
    return _Package0CorpusV1(
        positional=positional,
        keyword_only=keyword_only,
        expected=expected,
        fingerprint=fingerprint_hasher.hexdigest(),
    )


@pytest.fixture(scope="module")
def package0_corpus() -> _Package0CorpusV1:
    return _build_package0_corpus_v1()


def _decoded_object(raw: bytes) -> dict[str, object]:
    value = decode_and_require_canonical_rfc8785_v1(raw)
    assert type(value) is dict
    return value


def _replace_positional(
    corpus: _Package0CorpusV1,
    index: int,
    replacement: object,
) -> tuple[object, ...]:
    positional = list(corpus.positional)
    positional[index] = replacement
    return tuple(positional)


def _resigned_package_raw(
    raw: bytes,
    descriptor: dict[str, object],
    *,
    mutate_body: object,
) -> bytes:
    envelope = _decoded_object(raw)
    body = copy.deepcopy(envelope["body"])
    assert type(body) is dict
    assert callable(mutate_body)
    mutate_body(body)
    return _package_envelope(
        descriptor,
        body,
        private_keys_by_role=_kat_private_keys_by_role(),
        key_ids_by_role=_kat_key_ids_by_role(),
    )[1]


def _resigned_referenced_raw(
    raw: bytes,
    contract: dict[str, object],
    *,
    mutate_body: object,
) -> bytes:
    envelope = _decoded_object(raw)
    body = copy.deepcopy(envelope["body"])
    assert type(body) is dict
    assert callable(mutate_body)
    mutate_body(body)
    return _referenced_envelope(
        contract,
        body,
        private_keys_by_role=_kat_private_keys_by_role(),
        key_ids_by_role=_kat_key_ids_by_role(),
    )[1]


def _resigned_ledger_raw(
    raw: bytes,
    index: int,
    *,
    mutate_body: object,
) -> bytes:
    package0_schema = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]
    ledger_contract = package0_schema[
        "nested_record_binding_contracts_in_exact_order"
    ][0]
    slot = ledger_contract["event_signature_slots_in_exact_stage_order"][index]
    envelope = _decoded_object(raw)
    body = copy.deepcopy(envelope["body"])
    assert type(body) is dict
    assert callable(mutate_body)
    mutate_body(body)
    custodian_key = _kat_private_keys_by_role()["CUSTODIAN"]
    custodian_key_id = _kat_key_ids_by_role()["CUSTODIAN"]
    signature_preimage = {
        "schema_version": ledger_contract[
            "event_signature_preimage_schema_version"
        ],
        "artifact_kind": f"signed_custodian_stage_event_{index}_v1",
        "artifact_schema_version": ledger_contract["record_schema_version"],
        "algorithm": "Ed25519",
        "body": body,
        "signature_ordinal": 0,
        "role": "CUSTODIAN",
        "purpose": slot["purpose"],
        "key_epoch": 1,
        "signer_key_id": custodian_key_id,
    }
    signature = {
        "schema_version": catalog.DETACHED_SIGNATURE_RECORD_SCHEMA_VERSION_V1,
        "algorithm": "Ed25519",
        "role": "CUSTODIAN",
        "purpose": slot["purpose"],
        "key_epoch": 1,
        "signer_key_id": custodian_key_id,
        "signature_hex": custodian_key.sign(
            str(slot["signature_domain_ascii"]).encode("ascii")
            + _canonical(signature_preimage)
        ).hex(),
    }
    id_subject = {
        "schema_version": ledger_contract["record_schema_version"],
        "body": body,
        "signatures_in_exact_order": [signature],
    }
    event_id = str(ledger_contract["record_content_id_prefix"]) + hashlib.sha256(
        str(ledger_contract["record_content_id_domain_ascii"]).encode("ascii")
        + _canonical(id_subject)
    ).hexdigest()
    return _canonical(
        {
            **id_subject,
            "signed_stage_event_id": event_id,
        }
    )


def _resigned_stage_raw(
    raw: bytes,
    index: int,
    *,
    mutate_body: object,
) -> bytes:
    descriptor = catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[index]
    envelope = _decoded_object(raw)
    body = copy.deepcopy(envelope["body"])
    assert type(body) is dict
    assert callable(mutate_body)
    mutate_body(body)
    stage_id = _body_content_id(
        str(descriptor["content_id_domain_ascii"]),
        str(descriptor["content_id_prefix"]),
        body,
    )
    signatures = [
        _generic_signature(
            private_key=_kat_private_keys_by_role()[str(slot["role"])],
            signer_key_id=_kat_key_ids_by_role()[str(slot["role"])],
            descriptor=descriptor,
            artifact_schema_version=str(descriptor["event_schema_version"]),
            content_id=stage_id,
            body=body,
            slot=slot,
        )
        for slot in descriptor["signature_slots_in_exact_order"]
    ]
    return _canonical(
        {
            "schema_version": descriptor["event_schema_version"],
            "body": body,
            "stage_event_id": stage_id,
            "signatures_in_exact_order": signatures,
        }
    )


def _envelope_identifier(raw: bytes, field_name: str) -> str:
    envelope = _decoded_object(raw)
    value = envelope[field_name]
    assert type(value) is str
    return value


def _cascaded_resigned_stage0_p15_forward_variant(
    corpus: _Package0CorpusV1,
) -> tuple[object, ...]:
    package0_schema = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]
    nested_contracts = package0_schema[
        "nested_record_binding_contracts_in_exact_order"
    ]
    ledger_contract = nested_contracts[0]
    referenced_contracts = {
        str(item["record_name"]): item for item in nested_contracts[1:6]
    }
    raw_p15 = corpus.positional[10]
    assert type(raw_p15) is bytes
    p15_id = _envelope_identifier(raw_p15, "package_id")

    stage_raws = list(corpus.positional[4])
    verified_stages_by_name: dict[str, tuple[str, str]] = {}
    for index, descriptor in enumerate(
        catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    ):
        if index == 0:

            def cascade_stage0(body: dict[str, object]) -> None:
                subject_kinds = body["subject_artifact_kinds_in_exact_order"]
                subject_ids = body["subject_artifact_ids_in_exact_order"]
                subject_shas = body[
                    "subject_artifact_payload_sha256s_in_exact_order"
                ]
                assert type(subject_kinds) is list
                assert type(subject_ids) is list
                assert type(subject_shas) is list
                subject_kinds[0] = (
                    mechanics.TERMINAL_SUBJECT_ARTIFACT_KINDS_IN_EXACT_ORDER_V1[
                        0
                    ]
                )
                subject_ids[0] = p15_id
                subject_shas[0] = _raw_sha(raw_p15)

            mutate_stage = cascade_stage0
        else:

            def cascade_predecessors(body: dict[str, object]) -> None:
                predecessor_names = body[
                    "predecessor_node_names_in_exact_order"
                ]
                predecessor_ids = body[
                    "predecessor_evidence_or_event_ids_in_exact_order"
                ]
                predecessor_shas = body[
                    "predecessor_evidence_or_event_payload_sha256s_in_exact_order"
                ]
                assert type(predecessor_names) is list
                assert type(predecessor_ids) is list
                assert type(predecessor_shas) is list
                for position, node_name in enumerate(predecessor_names):
                    prior = verified_stages_by_name.get(str(node_name))
                    if prior is not None:
                        predecessor_ids[position] = prior[0]
                        predecessor_shas[position] = prior[1]

            mutate_stage = cascade_predecessors
        stage_raws[index] = _resigned_stage_raw(
            stage_raws[index],
            index,
            mutate_body=mutate_stage,
        )
        stage_id = _envelope_identifier(stage_raws[index], "stage_event_id")
        verified_stages_by_name[str(descriptor["stage"])] = (
            stage_id,
            _raw_sha(stage_raws[index]),
        )

    ledger_raws: list[bytes] = []
    predecessor_ledger_id: str | None = None
    for index, original_raw in enumerate(corpus.positional[3]):
        stage_id = _envelope_identifier(stage_raws[index], "stage_event_id")
        stage_sha = _raw_sha(stage_raws[index])

        def cascade_ledger(
            body: dict[str, object],
            *,
            predecessor: str | None = predecessor_ledger_id,
            payload_id: str = stage_id,
            payload_sha: str = stage_sha,
        ) -> None:
            body["predecessor_event_id_or_none"] = predecessor
            body["payload_id"] = payload_id
            body["payload_sha256"] = payload_sha

        raw_ledger = _resigned_ledger_raw(
            original_raw,
            index,
            mutate_body=cascade_ledger,
        )
        ledger_raws.append(raw_ledger)
        predecessor_ledger_id = _envelope_identifier(
            raw_ledger,
            "signed_stage_event_id",
        )

    chain_pairs = [
        {
            "signed_stage_event_id": _envelope_identifier(
                raw,
                "signed_stage_event_id",
            ),
            "signed_stage_event_payload_sha256": _raw_sha(raw),
        }
        for raw in ledger_raws
    ]
    chain_root = _raw_sha(
        mechanics.LEDGER_EVENT_CHAIN_ROOT_DOMAIN_V1 + _canonical(chain_pairs)
    )
    chain_id = mechanics.LEDGER_EVENT_CHAIN_ID_PREFIX_V1 + chain_root[7:]
    terminal_ledger_raw = ledger_raws[9]
    terminal_ledger_id = _envelope_identifier(
        terminal_ledger_raw,
        "signed_stage_event_id",
    )
    terminal_ledger_sha = _raw_sha(terminal_ledger_raw)

    def rebuild_reference(
        original_raw: bytes,
        record_name: str,
        updates: dict[str, object],
    ) -> tuple[str, bytes]:
        def apply_updates(body: dict[str, object]) -> None:
            body.update(updates)

        rebuilt = _resigned_referenced_raw(
            original_raw,
            referenced_contracts[record_name],
            mutate_body=apply_updates,
        )
        return _envelope_identifier(rebuilt, "artifact_id"), rebuilt

    raw_store_original = corpus.positional[5]
    raw_time_original = corpus.positional[6]
    raw_governance_original = corpus.positional[7]
    raw_custodian_original = corpus.positional[8]
    raw_anchor_original = corpus.positional[9]
    assert type(raw_store_original) is bytes
    assert type(raw_time_original) is bytes
    assert type(raw_governance_original) is bytes
    assert type(raw_custodian_original) is bytes
    assert type(raw_anchor_original) is bytes
    store_id, raw_store = rebuild_reference(
        raw_store_original,
        "store_if_absent_receipt",
        {
            "stored_artifact_id": terminal_ledger_id,
            "stored_artifact_payload_sha256": terminal_ledger_sha,
            "stored_artifact_size_bytes": len(terminal_ledger_raw),
        },
    )
    time_id, raw_time = rebuild_reference(
        raw_time_original,
        "logical_time_token",
        {
            "subject_id": terminal_ledger_id,
            "subject_payload_sha256": terminal_ledger_sha,
        },
    )
    checkpoint_updates = {
        "chain_id": chain_id,
        "chain_root_sha256": chain_root,
        "store_receipt_id": store_id,
        "store_receipt_payload_sha256": _raw_sha(raw_store),
        "time_token_id": time_id,
        "time_token_payload_sha256": _raw_sha(raw_time),
        "terminal_signed_ledger_event_id": terminal_ledger_id,
        "terminal_signed_ledger_event_payload_sha256": terminal_ledger_sha,
    }
    governance_id, raw_governance = rebuild_reference(
        raw_governance_original,
        "governance_checkpoint",
        checkpoint_updates,
    )
    custodian_id, raw_custodian = rebuild_reference(
        raw_custodian_original,
        "custodian_checkpoint",
        checkpoint_updates,
    )
    anchor_id, raw_anchor = rebuild_reference(
        raw_anchor_original,
        "external_anchor",
        {
            "chain_id": chain_id,
            "chain_root_sha256": chain_root,
            "governance_checkpoint_id": governance_id,
            "governance_checkpoint_payload_sha256": _raw_sha(raw_governance),
            "custodian_checkpoint_id": custodian_id,
            "custodian_checkpoint_payload_sha256": _raw_sha(raw_custodian),
        },
    )

    ledger_ids = [
        _envelope_identifier(raw, "signed_stage_event_id")
        for raw in ledger_raws
    ]
    ledger_shas = [_raw_sha(raw) for raw in ledger_raws]

    def cascade_package0(body: dict[str, object]) -> None:
        body["signed_stage_event_ids_in_exact_order"] = ledger_ids
        body["signed_stage_event_payload_sha256s_in_exact_order"] = ledger_shas
        body["store_if_absent_receipt_id"] = store_id
        body["store_if_absent_receipt_payload_sha256"] = _raw_sha(raw_store)
        body["logical_time_token_id"] = time_id
        body["logical_time_token_payload_sha256"] = _raw_sha(raw_time)
        body["governance_checkpoint_id"] = governance_id
        body["governance_checkpoint_payload_sha256"] = _raw_sha(raw_governance)
        body["custodian_checkpoint_id"] = custodian_id
        body["custodian_checkpoint_payload_sha256"] = _raw_sha(raw_custodian)
        body["external_anchor_id"] = anchor_id
        body["external_anchor_payload_sha256"] = _raw_sha(raw_anchor)
        body["chain_root_sha256"] = chain_root

    raw_package0_original = corpus.positional[2]
    assert type(raw_package0_original) is bytes
    raw_package0 = _resigned_package_raw(
        raw_package0_original,
        package0_schema,
        mutate_body=cascade_package0,
    )
    positional = list(corpus.positional)
    positional[2] = raw_package0
    positional[3] = tuple(ledger_raws)
    positional[4] = tuple(stage_raws)
    positional[5:10] = [
        raw_store,
        raw_time,
        raw_governance,
        raw_custodian,
        raw_anchor,
    ]
    return tuple(positional)


def _assert_verify_failure(
    corpus: _Package0CorpusV1,
    expected_message: str,
    *,
    positional: tuple[object, ...] | None = None,
    keyword_only: dict[str, bytes] | None = None,
) -> None:
    with pytest.raises(
        mechanics.Phase2BActualUnsealed960Package0MechanicsErrorV1
    ) as raised:
        mechanics.verify_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1(
            *(corpus.positional if positional is None else positional),
            **(corpus.keyword_only if keyword_only is None else keyword_only),
        )
    assert str(raised.value) == expected_message


def test_01_full_deterministic_positive_graph(
    package0_corpus: _Package0CorpusV1,
) -> None:
    print("package0-corpus-fingerprint", package0_corpus.fingerprint)
    assert package0_corpus.fingerprint == EXPECTED_CORPUS_FINGERPRINT_V1
    for value in package0_corpus.positional:
        items = value if type(value) is tuple else (value,)
        for raw in items:
            assert type(raw) is bytes
            assert _canonical(decode_and_require_canonical_rfc8785_v1(raw)) == raw
    result = mechanics.verify_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1(
        *package0_corpus.positional,
        **package0_corpus.keyword_only,
    )
    assert mechanics.require_verified_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1(
        result
    ) is result
    assert result.schema_version == mechanics.SCHEMA_VERSION_V1
    assert result.status == mechanics.STATUS_V1
    assert result.claim_level == mechanics.CLAIM_LEVEL_V1
    assert (
        result.disposition
        is mechanics.Package0MechanicsDispositionV1.VERIFIED_PRESENTED_PACKAGE0_MECHANICS_NOT_R10
    )
    assert (
        result.reason
        is mechanics.Package0MechanicsReasonV1.EXACT_PACKAGE0_RAW_GRAPH_VERIFIED
    )
    assert result.catalog_payload_sha256 == _raw_sha(catalog.CATALOG_PAYLOAD_V1)
    for field_name, expected_value in package0_corpus.expected.items():
        assert getattr(result, field_name) == expected_value
    assert result.ledger_event_count == 10
    assert result.stage_subject_event_count == 10
    assert result.referenced_artifact_count == 5
    assert result.total_signature_count == 35
    for field_name in mechanics.MECHANICS_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1:
        assert getattr(result, field_name) is True
    for field_name in mechanics.NONCLAIM_BOOLEAN_FIELDS_IN_EXACT_ORDER_V1:
        assert getattr(result, field_name) is False


def test_02_public_abi_and_real_entry_remain_exact(
    package0_corpus: _Package0CorpusV1,
) -> None:
    expected_names = tuple(
        name for name, _ in mechanics.POSITIONAL_PARAMETERS_IN_EXACT_ORDER_V1
    ) + tuple(mechanics.KEYWORD_ONLY_PARAMETERS_IN_EXACT_ORDER_V1)
    for function in (
        mechanics.verify_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1,
        mechanics.verify_real_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1,
    ):
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == expected_names
        parameters = tuple(signature.parameters.values())
        assert all(
            item.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for item in parameters[:12]
        )
        assert all(
            item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters[12:]
        )
        assert all(item.default is inspect.Parameter.empty for item in parameters)
    assert mechanics.REAL_VERIFICATION_STATUS_V1 == "NOT_IMPLEMENTED"
    with pytest.raises(
        mechanics.Phase2BActualUnsealed960Package0MechanicsErrorV1,
        match=r"^FAIL_C1_PACKAGE0_MECHANICS_V1_NOT_IMPLEMENTED:",
    ):
        mechanics.verify_real_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1(
            *package0_corpus.positional,
            **package0_corpus.keyword_only,
        )


def test_03_private_lineage_copy_pickle_and_mutation_reject(
    package0_corpus: _Package0CorpusV1,
) -> None:
    result = mechanics.verify_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1(
        *package0_corpus.positional,
        **package0_corpus.keyword_only,
    )
    result_type = mechanics.VerifiedPhase2BActualUnsealed960Package0MechanicsV1
    with pytest.raises(TypeError, match="privately issued"):
        result_type()
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(result)
    with pytest.raises(TypeError, match="cannot be deep-copied"):
        copy.deepcopy(result)
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(result)
    forged = object.__new__(result_type)
    with pytest.raises(
        mechanics.Phase2BActualUnsealed960Package0MechanicsErrorV1,
        match=r"^FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE:",
    ):
        mechanics.require_verified_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1(
            forged
        )
    object.__setattr__(result, "c1_satisfied", True)
    with pytest.raises(
        mechanics.Phase2BActualUnsealed960Package0MechanicsErrorV1,
        match=r"^FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE:",
    ):
        mechanics.require_verified_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1(
            result
        )


def test_04_boundary_type_pin_canonical_and_signature_kats(
    package0_corpus: _Package0CorpusV1,
) -> None:
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_TYPE: "
            "raw_package0_ledger_event_payloads_in_exact_order must be one exact "
            "tuple of ten items"
        ),
        positional=_replace_positional(
            package0_corpus,
            3,
            list(package0_corpus.positional[3]),
        ),
    )

    flipped_pin = dict(package0_corpus.keyword_only)
    catalog_digest = bytearray(
        flipped_pin["out_of_band_expected_catalog_payload_sha256_digest"]
    )
    catalog_digest[0] ^= 1
    flipped_pin["out_of_band_expected_catalog_payload_sha256_digest"] = bytes(
        catalog_digest
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_PIN: "
            "one or more inspected catalog pins differ"
        ),
        keyword_only=flipped_pin,
    )

    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_CANONICAL: "
            "package0 is not exact RFC8785"
        ),
        positional=_replace_positional(
            package0_corpus,
            2,
            b" " + package0_corpus.positional[2],
        ),
    )

    signature_envelope = _decoded_object(package0_corpus.positional[2])
    signatures = signature_envelope["signatures_in_exact_order"]
    assert type(signatures) is list
    first_signature = signatures[0]
    assert type(first_signature) is dict
    signature_hex = first_signature["signature_hex"]
    assert type(signature_hex) is str
    first_signature["signature_hex"] = (
        ("0" if signature_hex[0] != "0" else "1") + signature_hex[1:]
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_SIGNATURE: "
            "package0 signature 0 signature rejected"
        ),
        positional=_replace_positional(
            package0_corpus,
            2,
            _canonical(signature_envelope),
        ),
    )


def test_05_validly_resigned_p16_cannot_rebind_package15(
    package0_corpus: _Package0CorpusV1,
) -> None:
    package15_schema = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[15]
    package16_schema = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[16]

    def rebind(body: dict[str, object]) -> None:
        predecessor_ids = body[
            "predecessor_package_instance_ids_in_exact_order"
        ]
        assert type(predecessor_ids) is list
        predecessor_ids[1] = _stable_id(
            str(package15_schema["content_id_prefix"]),
            "negative/p16-rebind-package15",
        )

    mutated_p16 = _resigned_package_raw(
        package0_corpus.positional[11],
        package16_schema,
        mutate_body=rebind,
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE: "
            "package16 does not shallow-bind package15"
        ),
        positional=_replace_positional(package0_corpus, 11, mutated_p16),
    )


def test_06_stage_literal_and_ledger_chain_projection_kats(
    package0_corpus: _Package0CorpusV1,
) -> None:
    stage_raws = list(package0_corpus.positional[4])
    stage_envelope = _decoded_object(stage_raws[0])
    stage_body = stage_envelope["body"]
    assert type(stage_body) is dict
    stage_body["stage_occurred"] = False
    stage_raws[0] = _canonical(stage_envelope)
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA: "
            "stage event 0 required literal differs"
        ),
        positional=_replace_positional(package0_corpus, 4, tuple(stage_raws)),
    )

    terminal_stage_raws = list(package0_corpus.positional[4])
    terminal_envelope = _decoded_object(terminal_stage_raws[9])
    terminal_body = terminal_envelope["body"]
    assert type(terminal_body) is dict
    terminal_body["terminal_disposition_or_none"] = []
    terminal_stage_raws[9] = _canonical(terminal_envelope)
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA: "
            "terminal disposition must be exact text"
        ),
        positional=_replace_positional(
            package0_corpus,
            4,
            tuple(terminal_stage_raws),
        ),
    )

    package0_schema = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]
    ledger_contract = package0_schema[
        "nested_record_binding_contracts_in_exact_order"
    ][0]

    def sever_chain(body: dict[str, object]) -> None:
        body["predecessor_event_id_or_none"] = _stable_id(
            str(ledger_contract["record_content_id_prefix"]),
            "negative/severed-ledger-predecessor",
        )

    ledger_raws = list(package0_corpus.positional[3])
    ledger_raws[1] = _resigned_ledger_raw(
        ledger_raws[1],
        1,
        mutate_body=sever_chain,
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE: "
            "ledger event 1 body projection differs"
        ),
        positional=_replace_positional(package0_corpus, 3, tuple(ledger_raws)),
    )


def test_07_validly_resigned_reference_and_package0_chain_projection_kats(
    package0_corpus: _Package0CorpusV1,
) -> None:
    package0_schema = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]
    store_contract = package0_schema[
        "nested_record_binding_contracts_in_exact_order"
    ][1]

    def alter_stored_size(body: dict[str, object]) -> None:
        stored_size = body["stored_artifact_size_bytes"]
        assert type(stored_size) in (int, float)
        body["stored_artifact_size_bytes"] = int(stored_size) + 1

    mutated_store = _resigned_referenced_raw(
        package0_corpus.positional[5],
        store_contract,
        mutate_body=alter_stored_size,
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_REFERENCED_GRAPH: "
            "terminal store receipt projection differs"
        ),
        positional=_replace_positional(package0_corpus, 5, mutated_store),
    )

    def alter_chain_root(body: dict[str, object]) -> None:
        body["chain_root_sha256"] = "sha256:" + ("00" * 32)

    mutated_package0 = _resigned_package_raw(
        package0_corpus.positional[2],
        package0_schema,
        mutate_body=alter_chain_root,
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE: "
            "package0 ledger projection differs"
        ),
        positional=_replace_positional(package0_corpus, 2, mutated_package0),
    )


def test_08_forbidden_forward_kind_and_stage_prefix_kats(
    package0_corpus: _Package0CorpusV1,
) -> None:
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE: "
            "stage event 0 has a forbidden forward subject reference"
        ),
        positional=_cascaded_resigned_stage0_p15_forward_variant(
            package0_corpus
        ),
    )

    allowed_opaque_id = _stable_id(
        "package0_kat_opaque_subject_v1_",
        "negative/allowed-opaque-subject",
    )
    assert not any(
        allowed_opaque_id.startswith(prefix)
        for prefix in mechanics.STAGE_SUBJECT_FORBIDDEN_FORWARD_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1
    )
    forbidden_kind = (
        mechanics.TERMINAL_SUBJECT_ARTIFACT_KINDS_IN_EXACT_ORDER_V1[
            0
        ]
    )

    def insert_forbidden_kind(body: dict[str, object]) -> None:
        subject_kinds = body["subject_artifact_kinds_in_exact_order"]
        subject_ids = body["subject_artifact_ids_in_exact_order"]
        assert type(subject_kinds) is list
        assert type(subject_ids) is list
        subject_kinds[0] = forbidden_kind
        subject_ids[0] = allowed_opaque_id

    stage_raws = list(package0_corpus.positional[4])
    stage_raws[0] = _resigned_stage_raw(
        stage_raws[0],
        0,
        mutate_body=insert_forbidden_kind,
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE: "
            "stage event 0 has a forbidden forward subject reference"
        ),
        positional=_replace_positional(package0_corpus, 4, tuple(stage_raws)),
    )

    p15_id = _envelope_identifier(
        package0_corpus.positional[10],
        "package_id",
    )
    assert p15_id.startswith(
        mechanics.TERMINAL_SUBJECT_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1[0]
    )

    def insert_p15_id_only(body: dict[str, object]) -> None:
        subject_ids = body["subject_artifact_ids_in_exact_order"]
        assert type(subject_ids) is list
        subject_ids[0] = p15_id

    stage_raws = list(package0_corpus.positional[4])
    stage_raws[0] = _resigned_stage_raw(
        stage_raws[0],
        0,
        mutate_body=insert_p15_id_only,
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE: "
            "stage event 0 has a forbidden forward subject reference"
        ),
        positional=_replace_positional(package0_corpus, 4, tuple(stage_raws)),
    )

    forbidden_stage_prefix = str(
        catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]["content_id_prefix"]
    )
    forbidden_stage_id = _stable_id(
        forbidden_stage_prefix,
        "negative/forbidden-stage-prefix",
    )
    assert forbidden_stage_id.startswith(forbidden_stage_prefix)

    def insert_forbidden_stage_id(body: dict[str, object]) -> None:
        subject_ids = body["subject_artifact_ids_in_exact_order"]
        assert type(subject_ids) is list
        subject_ids[0] = forbidden_stage_id

    stage_raws = list(package0_corpus.positional[4])
    stage_raws[0] = _resigned_stage_raw(
        stage_raws[0],
        0,
        mutate_body=insert_forbidden_stage_id,
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_LINEAGE: "
            "stage event 0 has a forbidden forward subject reference"
        ),
        positional=_replace_positional(package0_corpus, 4, tuple(stage_raws)),
    )


def test_09_exact_bytes_bound_and_identifier_regex_preflight_kats(
    package0_corpus: _Package0CorpusV1,
) -> None:
    oversized = b"x" * (
        mechanics.MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1 + 1
    )
    _assert_verify_failure(
        package0_corpus,
        "FAIL_C1_PACKAGE0_MECHANICS_V1_TYPE: catalog_payload exceeds byte bound",
        positional=_replace_positional(package0_corpus, 0, oversized),
    )

    malformed_generic_id = dict(package0_corpus.keyword_only)
    malformed_generic_id[
        "out_of_band_expected_governance_root_key_id_bytes"
    ] = b"not-a-generic-content-identifier"
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_SCHEMA: "
            "governance root key ID content-ID syntax differs"
        ),
        keyword_only=malformed_generic_id,
    )

    malformed_campaign = dict(package0_corpus.keyword_only)
    malformed_campaign["expected_campaign_id_bytes"] = (
        mechanics.CAMPAIGN_ID_PREFIX_V1.encode("ascii") + (b"0" * 63)
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_PIN: "
            "campaign or attempt ID syntax differs"
        ),
        keyword_only=malformed_campaign,
    )

    malformed_attempt = dict(package0_corpus.keyword_only)
    malformed_attempt["expected_attempt_id_bytes"] = (
        mechanics.ATTEMPT_ID_PREFIX_V1.encode("ascii") + (b"G" * 64)
    )
    _assert_verify_failure(
        package0_corpus,
        (
            "FAIL_C1_PACKAGE0_MECHANICS_V1_PIN: "
            "campaign or attempt ID syntax differs"
        ),
        keyword_only=malformed_attempt,
    )


def test_10_definition_time_boundary_survives_module_global_rebind(
    package0_corpus: _Package0CorpusV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = mechanics.verify_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1
    require_result = mechanics.require_verified_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1

    def rebound_dependency(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("rebound module dependency was consulted")

    monkeypatch.setattr(
        rfc8785_module,
        "canonicalize_rfc8785_v1",
        rebound_dependency,
    )
    monkeypatch.setattr(
        rfc8785_module,
        "decode_and_require_canonical_rfc8785_v1",
        rebound_dependency,
    )
    monkeypatch.setattr(mechanics, "_CAPTURED_CANONICALIZE_V1", rebound_dependency)
    monkeypatch.setattr(mechanics, "_CAPTURED_DECODE_V1", rebound_dependency)
    monkeypatch.setattr(mechanics, "_catalog", None)
    monkeypatch.setattr(mechanics, "hashlib", None)
    monkeypatch.setattr(mechanics, "_Ed25519PublicKey", None)
    monkeypatch.setattr(
        mechanics,
        "MAXIMUM_EXACT_BYTES_PARAMETER_OR_TUPLE_ITEM_BYTES_V1",
        1,
    )
    monkeypatch.setattr(
        mechanics,
        "STAGE_SUBJECT_FORBIDDEN_FORWARD_CONTENT_ID_PREFIXES_IN_EXACT_ORDER_V1",
        (),
    )
    monkeypatch.setattr(
        mechanics,
        "verify_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1",
        rebound_dependency,
    )
    monkeypatch.setattr(
        mechanics,
        "require_verified_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_mechanics_v1",
        rebound_dependency,
    )

    result = verifier(
        *package0_corpus.positional,
        **package0_corpus.keyword_only,
    )
    assert require_result(result) is result
    assert result.package_id == package0_corpus.expected["package_id"]


@pytest.mark.parametrize(
    "catalog_poisoning_statement",
    (
        (
            "p0 = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]\n"
            "p0['signature_slots_in_exact_order'].clear()\n"
            "p0['signature_cardinality'] = 0"
        ),
        (
            "abi = catalog.FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1\n"
            "abi['positional_parameters_in_exact_order'][0][0] = "
            "'poisoned_catalog_payload'"
        ),
        (
            "result_contract = "
            "catalog.FUTURE_REAL_PACKAGE0_VERIFIER_RAW_ABI_V1"
            "['verified_result_contract']\n"
            "result_contract['fields_in_exact_order'][0][0] = "
            "'poisoned_schema_version'"
        ),
        (
            "stage0 = catalog.STAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]\n"
            "stage0['stage'] = 'POISONED_PRE_IMPORT_STAGE'"
        ),
    ),
    ids=(
        "package0-signature-slots-and-cardinality",
        "raw-abi-projection",
        "verified-result-projection",
        "stage-schema-projection",
    ),
)
def test_11_fresh_subprocess_rejects_preimport_catalog_poisoning(
    catalog_poisoning_statement: str,
) -> None:
    catalog_module = (
        "hegel_machine.phase2b_actual_unsealed_960_"
        "authoritative_evidence_schema_catalog_v1"
    )
    mechanics_module = (
        "hegel_machine.phase2b_actual_unsealed_960_"
        "durable_signed_custodian_ledger_and_signer_chain_v1"
    )
    child_code = (
        "import importlib\n"
        "import sys\n"
        f"catalog = importlib.import_module({catalog_module!r})\n"
        f"assert {mechanics_module!r} not in sys.modules\n"
        f"{catalog_poisoning_statement}\n"
        "try:\n"
        f"    importlib.import_module({mechanics_module!r})\n"
        "except RuntimeError as exc:\n"
        "    expected = 'Package0 catalog public schema surface differs'\n"
        "    if str(exc) != expected:\n"
        "        raise AssertionError(repr(str(exc))) from exc\n"
        "    print('controlled-runtime-error:', str(exc))\n"
        "else:\n"
        "    raise AssertionError('poisoned catalog import unexpectedly succeeded')\n"
    )
    repository_root = Path(__file__).resolve().parents[1]
    child_environment = dict(os.environ)
    existing_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = str(repository_root / "src") + (
        "" if not existing_pythonpath else os.pathsep + existing_pythonpath
    )
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=repository_root,
        env=child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        f"child return code: {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert completed.stdout == (
        "controlled-runtime-error: "
        "Package0 catalog public schema surface differs\n"
    )


def test_12_fresh_subprocess_ignores_targeted_preimport_rfc_wrapper(
    package0_corpus: _Package0CorpusV1,
) -> None:
    rfc_module = "hegel_machine.rfc8785_v1"
    catalog_module = (
        "hegel_machine.phase2b_actual_unsealed_960_"
        "authoritative_evidence_schema_catalog_v1"
    )
    mechanics_module = (
        "hegel_machine.phase2b_actual_unsealed_960_"
        "durable_signed_custodian_ledger_and_signer_chain_v1"
    )
    child_code = (
        "import hashlib\n"
        "import importlib\n"
        "import pickle\n"
        "import sys\n"
        "payload = pickle.load(sys.stdin.buffer)\n"
        f"rfc = importlib.import_module({rfc_module!r})\n"
        f"catalog = importlib.import_module({catalog_module!r})\n"
        f"assert {mechanics_module!r} not in sys.modules\n"
        "original_canonicalize = rfc.canonicalize_rfc8785_v1\n"
        "original_decode = rfc.decode_and_require_canonical_rfc8785_v1\n"
        "package0_raw = payload['positional'][2]\n"
        "package0_envelope = original_decode(package0_raw)\n"
        "package0_body = package0_envelope['body']\n"
        "package0_body_bytes = original_canonicalize(package0_body)\n"
        "package0_descriptor = "
        "catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]\n"
        "known_package0_id = str(package0_descriptor['content_id_prefix']) + "
        "hashlib.sha256(\n"
        "    str(package0_descriptor['content_id_domain_ascii']).encode('ascii') + "
        "package0_body_bytes\n"
        ").hexdigest()\n"
        "assert known_package0_id == package0_envelope['package_id']\n"
        "assert known_package0_id == payload['expected_package_id']\n"
        "wrapper_hits = []\n"
        "def targeted_canonicalizer(value):\n"
        "    canonical = original_canonicalize(value)\n"
        "    if canonical == package0_body_bytes:\n"
        "        wrapper_hits.append(canonical)\n"
        "        return canonical + b' '\n"
        "    return canonical\n"
        "rfc.canonicalize_rfc8785_v1 = targeted_canonicalizer\n"
        "assert rfc.canonicalize_rfc8785_v1(package0_body) == "
        "package0_body_bytes + b' '\n"
        "assert len(wrapper_hits) == 1\n"
        f"mechanics = importlib.import_module({mechanics_module!r})\n"
        "result = mechanics.verify_phase2b_actual_unsealed_960_durable_signed_"
        "custodian_ledger_and_signer_chain_mechanics_v1(\n"
        "    *payload['positional'], **payload['keyword_only']\n"
        ")\n"
        "required = mechanics.require_verified_phase2b_actual_unsealed_960_"
        "durable_signed_custodian_ledger_and_signer_chain_mechanics_v1(result)\n"
        "assert required is result\n"
        "assert result.package_id == known_package0_id\n"
        "assert result.total_signature_count == 35\n"
        "assert mechanics.canonical_phase2b_actual_unsealed_960_package0_"
        "mechanics_bytes_v1(package0_body) == package0_body_bytes\n"
        "assert len(wrapper_hits) == 1\n"
        "print('private-rfc-known-package0-id:', known_package0_id)\n"
    )
    repository_root = Path(__file__).resolve().parents[1]
    child_environment = dict(os.environ)
    existing_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = str(repository_root / "src") + (
        "" if not existing_pythonpath else os.pathsep + existing_pythonpath
    )
    child_payload = pickle.dumps(
        {
            "positional": package0_corpus.positional,
            "keyword_only": package0_corpus.keyword_only,
            "expected_package_id": package0_corpus.expected["package_id"],
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    completed = subprocess.run(
        [sys.executable, "-c", child_code],
        cwd=repository_root,
        env=child_environment,
        input=child_payload,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"child return code: {completed.returncode}\n"
        f"stdout:\n{completed.stdout.decode('utf-8', errors='replace')}\n"
        f"stderr:\n{completed.stderr.decode('utf-8', errors='replace')}"
    )
    expected_package_id = package0_corpus.expected["package_id"]
    assert type(expected_package_id) is str
    assert completed.stdout == (
        f"private-rfc-known-package0-id: {expected_package_id}\n".encode(
            "ascii"
        )
    )


def test_13_canonical_config_equals_frozen_contract_payload() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    config_payload = (
        repository_root
        / "config"
        / (
            "phase2b_actual_unsealed_960_durable_signed_custodian_"
            "ledger_and_signer_chain_v1.json"
        )
    ).read_bytes()
    assert config_payload == mechanics.CONTRACT_PAYLOAD_V1
    assert not config_payload.endswith(b"\n")
    assert hashlib.sha256(config_payload).hexdigest() == (
        mechanics.CONTRACT_PAYLOAD_SHA256_V1
    )
    assert decode_and_require_canonical_rfc8785_v1(config_payload) == {
        "schema_version": mechanics.CONTRACT_SCHEMA_VERSION_V1,
        "body": mechanics.CONTRACT_BODY_V1,
        "contract_id": mechanics.CONTRACT_ID_V1,
    }
