from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from hegel_machine.rfc8785_v1 import (
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)
import hegel_machine.phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1 as catalog


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "config"
    / "phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.json"
)
SOURCE = (
    ROOT
    / "src"
    / "hegel_machine"
    / "phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.py"
)

EXPECTED_CATALOG_ID = (
    "phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1_"
    "accb3c3739394cc36f2e89fc420f58799fbe71a1f33c21ee31906a3624e23082"
)
EXPECTED_CONFIG_SHA256 = (
    "71473b38f5fd4df165884c06134f4fbef6d94bb8e307fa71703d946d3a49b0ed"
)
EXPECTED_SOURCE_SHA256 = (
    "c75bfe81e5d77d5062da2babe9adc193ca2bdf6a9943aad442ea8214f3e89dff"
)

EXPECTED_GATES = (
    "R01_AUTHORIZATION_VERIFIED",
    "R02_EXECUTION_BRANCH_BINDINGS_REPLAYED",
    "R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED",
    "R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED",
    "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
    "R06_GENERATOR_IMPLEMENTATION_QUALIFIED",
    "R07_BASELINE_REGISTRATIONS_FROZEN",
    "R08_STATISTICAL_VALIDATION_PASSED",
    "R09_SIX_ROLE_LIVE_DOCKER_QUALIFIED",
    "R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED",
    "R11_EXECUTION_FREEZE_COMMITMENTS_BOUND",
    "R12_INDEPENDENT_LAUNCH_OVERLAY_VERIFIED",
    "R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY",
    "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED",
    "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED",
    "R16_INFERENCE_ADMISSION_UNLOCKED",
)
EXPECTED_STAGES = (
    "PREREGISTERED",
    "RAW_INPUT_COMMITTED",
    "PACKAGE_COMMITMENTS_FROZEN",
    "RUN_STARTED",
    "RUN_FINISHED",
    "PREDICTIONS_COMMITTED",
    "REVEALED",
    "SCORING_STARTED",
    "REPORTS_COMMITTED",
    "TERMINAL_CONSUMED",
)
EXPECTED_PACKAGES = (
    "durable_signed_custodian_ledger_and_signer_chain",
    "prediction_reveal_scoring_chronology",
    "raw_input_and_prediction_archive_cas",
    "canonical_archive_replay_and_decoder_transcript",
    "source_image_sbom_config_runtime_execution_attestations",
    "actual_recognizer_execution_capacity_and_output_attestation",
    "allocation_source_batch_partition_answer_authorities",
    "independent_720_240_generation_and_source_disjointness_receipt",
    "audit_archive_formal_uuid_and_covert_receipts",
    "three_baseline_prediction_outputs",
    "durable_fail_closed_attempt_event_denominator",
    "preservation_496_legal_76_invalid_pair_results",
    "semantic_conflict_240_descriptor_audit_package",
    "scale_regret_oracle_loss_and_normalizer",
    "clustered_bootstrap_inputs_and_statistic",
    "metric_wilson_overall_and_slice_gate_report",
    "attempt_rerun_permanent_record",
    "formal_c1_consumed_sealed_report",
)


def _pins() -> dict[str, bytes]:
    return {
        "out_of_band_expected_readiness_dag_id_bytes": (
            catalog.INSPECTED_READINESS_DAG_ID_V1.encode("ascii")
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
    }


def _raw() -> bytes:
    return CONFIG.read_bytes()


def _real_stub_inputs() -> tuple[tuple[object, ...], dict[str, bytes]]:
    positional: tuple[object, ...] = (
        _raw(),
        b"signer-registry",
        (),
        b"preallocation-lineage",
        (),
        (),
        (),
        (),
        b"bundle-index",
        (),
        (),
        b"store-receipt",
        b"logical-time-token",
        b"governance-checkpoint",
        b"custodian-checkpoint",
        b"external-anchor",
        (),
        b"attempt-universe-head",
        b"denominator-finalization",
        (),
        (),
    )
    keyword_only = {
        name: b"required-oob-byte-pin"
        for name in (
            "out_of_band_expected_catalog_id_bytes",
            "out_of_band_expected_catalog_payload_sha256_digest",
            "out_of_band_expected_signer_registry_id_bytes",
            "out_of_band_expected_signer_registry_payload_sha256_digest",
            "out_of_band_expected_signer_registry_entries_root_sha256_digest",
            "out_of_band_expected_preallocation_lineage_id_bytes",
            "out_of_band_expected_preallocation_lineage_payload_sha256_digest",
            "out_of_band_expected_governance_root_key_id_bytes",
            "out_of_band_expected_governance_root_public_key_bytes",
            "out_of_band_expected_custodian_root_key_id_bytes",
            "out_of_band_expected_custodian_root_public_key_bytes",
            "out_of_band_expected_attempt_universe_head_id_bytes",
            "out_of_band_expected_attempt_universe_head_payload_sha256_digest",
            "out_of_band_expected_readiness_dag_id_bytes",
            "out_of_band_expected_readiness_dag_payload_sha256_digest",
            "out_of_band_expected_progression_plan_payload_sha256_digest",
            "out_of_band_expected_gap_inventory_source_sha256_digest",
            "out_of_band_expected_statistical_validation_source_sha256_digest",
            "expected_campaign_id_bytes",
            "expected_attempt_id_bytes",
        )
    }
    return positional, keyword_only


def _root(domain: bytes, prefix: str, identifiers: tuple[str, ...]) -> str:
    framed = bytearray(domain)
    framed.extend(len(identifiers).to_bytes(4, "big"))
    for identifier in identifiers:
        raw = identifier.encode("ascii")
        framed.extend(len(raw).to_bytes(2, "big"))
        framed.extend(raw)
    return prefix + hashlib.sha256(bytes(framed)).hexdigest()


def _cid(domain_ascii: str, prefix: str, body: object) -> str:
    return prefix + hashlib.sha256(
        domain_ascii.encode("ascii") + canonicalize_rfc8785_v1(body)
    ).hexdigest()


def test_config_is_exact_canonical_payload_and_positive_is_all_false() -> None:
    raw = _raw()
    assert not raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_CONFIG_SHA256
    assert raw == catalog.CATALOG_PAYLOAD_V1
    assert raw == (
        catalog.canonical_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_bytes_v1()
    )
    assert decode_and_require_canonical_rfc8785_v1(raw)
    result = catalog.verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
        raw, **_pins()
    )
    assert catalog.require_verified_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
        result
    ) is result
    assert result.catalog_id == EXPECTED_CATALOG_ID
    assert result.disposition is catalog.CatalogDispositionV1.RUNBOOK_FROZEN_NOT_ADMITTED
    assert result.gate_count == 16
    assert result.satisfied_gate_count == 0
    assert result.package_count == 18
    assert result.stage_count == 10
    body = json.loads(raw)["body"]
    assert result.bundle_index_schema_id == body["bundle_index_schema_id"]
    for field in (
        "exact_raw_bytes_verified",
        "rfc8785_canonical_bytes_verified",
        "inspected_dependency_pins_verified",
        "schema_catalog_exactly_verified",
        "content_address_roots_verified",
        "exact_preimage_profiles_verified",
        "package_schema_dag_verified",
        "bundle_index_schema_verified",
        "interleaved_dag_verified",
        "attempt_retry_policy_verified",
        "terminal_policy_verified",
        "private_lineage_verified",
    ):
        assert getattr(result, field) is True
    for field in result.__dataclass_fields__:
        if field.startswith("r") and field.endswith("_satisfied"):
            assert getattr(result, field) is False
    for field in (
        "evidence_supplied",
        "evidence_verified",
        "authority_currentness_verified",
        "owner_non_revocation_verified",
        "six_live_roles_verified",
        "external_key_custody_verified",
        "external_durability_verified",
        "external_non_equivocation_verified",
        "campaign_id_allocated",
        "attempt_id_allocated",
        "actual_960_generated",
        "actual_corpus_audit_executed",
        "inference_performed",
        "predictions_committed",
        "answers_revealed",
        "scoring_performed",
        "terminal_report_consumed",
        "execution_authorized",
        "c1_satisfied",
    ):
        assert getattr(result, field) is False


def test_public_raw_abi_is_exact_and_has_no_defaults_or_injection() -> None:
    function = catalog.verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1
    signature = inspect.signature(function)
    assert tuple(signature.parameters) == (
        "payload",
        "out_of_band_expected_readiness_dag_id_bytes",
        "out_of_band_expected_readiness_dag_payload_sha256_digest",
        "out_of_band_expected_progression_plan_payload_sha256_digest",
        "out_of_band_expected_gap_inventory_source_sha256_digest",
        "out_of_band_expected_statistical_validation_source_sha256_digest",
    )
    parameters = tuple(signature.parameters.values())
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters[1:])
    assert all(item.default is inspect.Parameter.empty for item in parameters)
    assert all(item.annotation in ("bytes", bytes) for item in parameters)
    assert not any(item.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for item in parameters)


def test_real_bundle_ni_raw_abi_is_exact_21_plus_20() -> None:
    expected_positional = (
        "catalog_payload",
        "raw_signer_registry_payload",
        "raw_gate_evidence_payloads_in_exact_order",
        "raw_preallocation_lineage_payload",
        "raw_gate_supporting_artifact_payload_tuples_in_exact_order",
        "raw_gate_source_native_payload_tuples_in_exact_order",
        "raw_stage_event_payloads_in_exact_order",
        "raw_package_envelope_payloads_in_exact_order",
        "raw_bundle_index_payload",
        "raw_package0_ledger_event_payloads_in_exact_order",
        "raw_package0_stage_subject_event_payloads_in_exact_order",
        "raw_store_if_absent_receipt_payload",
        "raw_logical_time_token_payload",
        "raw_governance_checkpoint_payload",
        "raw_custodian_checkpoint_payload",
        "raw_external_anchor_payload",
        "raw_package10_denominator_event_payloads_in_exact_order",
        "raw_attempt_universe_head_payload",
        "raw_package10_denominator_finalization_payload",
        "raw_package15_gate_result_payloads_in_exact_order",
        "raw_package16_attempt_record_payloads_in_exact_order",
    )
    _, keyword_inputs = _real_stub_inputs()
    expected_keyword_only = tuple(keyword_inputs)
    manifest = catalog.FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1
    assert tuple(row[0] for row in manifest["positional_parameters_in_exact_order"]) == (
        expected_positional
    )
    assert tuple(manifest["mandatory_keyword_only_byte_parameters_in_exact_order"]) == (
        expected_keyword_only
    )
    assert manifest["positional_parameters_in_exact_order"][4][1] == (
        "EXACT_BUILTIN_TUPLE_OF_16_EXACT_BUILTIN_TUPLES_OF_3_EXACT_BUILTIN_BYTES"
    )
    assert manifest["positional_parameters_in_exact_order"][5][1] == (
        "EXACT_BUILTIN_TUPLE_OF_16_DESCRIPTOR_FROZEN_EXACT_BUILTIN_TUPLES_"
        "OF_24_4_8_19_5_10_9_7_OR_9_11_13_15_4_6_15_11_13_EXACT_BUILTIN_BYTES"
    )
    signature = inspect.signature(
        catalog.verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1
    )
    parameters = tuple(signature.parameters.values())
    assert tuple(signature.parameters) == expected_positional + expected_keyword_only
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters[:21]
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters[21:]
    )
    assert all(parameter.default is inspect.Parameter.empty for parameter in parameters)
    assert not any(
        parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
        for parameter in parameters
    )


def test_permanent_nonclaim_oracle_is_exact_bool_false_and_matches_result_suffix() -> None:
    body = json.loads(_raw())["body"]
    nonclaims = body["permanent_nonclaims"]
    assert type(nonclaims) is dict
    assert set(nonclaims) == set(catalog.PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1)
    assert len(nonclaims) == len(catalog.PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1)
    for field_name in catalog.PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1:
        assert type(nonclaims[field_name]) is bool
        assert nonclaims[field_name] is False
    expected = {
        field_name: False
        for field_name in catalog.PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1
    }
    assert canonicalize_rfc8785_v1(nonclaims) == canonicalize_rfc8785_v1(expected)
    result_fields = tuple(
        catalog.VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1.__dataclass_fields__
    )
    start = result_fields.index("evidence_supplied")
    assert result_fields[start:] == catalog.PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1


def test_gate_catalog_and_exact_readiness_dependencies() -> None:
    specs = catalog.GATE_SPECS_IN_EXACT_ORDER_V1
    assert tuple(item.ordinal for item in specs) == tuple(range(1, 17))
    assert tuple(item.gate_id for item in specs) == EXPECTED_GATES
    assert all(item.initial_satisfied is False for item in specs)
    dependency_map = {item.gate_id: item.dependencies_in_exact_order for item in specs}
    assert dependency_map[EXPECTED_GATES[0]] == ()
    assert dependency_map[EXPECTED_GATES[5]] == (EXPECTED_GATES[4],)
    assert dependency_map[EXPECTED_GATES[6]] == (EXPECTED_GATES[4],)
    assert dependency_map[EXPECTED_GATES[7]] == (EXPECTED_GATES[5], EXPECTED_GATES[6])
    assert dependency_map[EXPECTED_GATES[-1]] == (EXPECTED_GATES[-2],)


def test_gate_source_catalog_is_literal_fail_closed_and_acyclic() -> None:
    body = json.loads(_raw())["body"]
    source_specs = catalog.GATE_SOURCE_SPECS_IN_EXACT_ORDER_V1
    sources = body["gate_source_schema_objects_in_exact_order"]
    supports = body["gate_support_schema_objects_in_exact_order"]
    gates = body["gate_schema_objects_in_exact_order"]
    assert len(source_specs) == len(sources) == len(supports) == len(gates) == 16
    assert tuple(
        tuple(len(item) for item in spec.allowed_native_source_kind_orders_in_exact_order)
        for spec in source_specs
    ) == (
        (24,),
        (4,),
        (8,),
        (19,),
        (5,),
        (10,),
        (9,),
        (7, 9),
        (11,),
        (13,),
        (15,),
        (4,),
        (6,),
        (15,),
        (11,),
        (13,),
    )
    expected_source_root_domains = tuple(
        f"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE/{ordinal}/SOURCE_ROOT/V1\x00"
        for ordinal in range(1, 17)
    )
    seen_gate_ids: list[str] = []
    forbidden_source_truth_fields = {
        "passed",
        "satisfied",
        "current",
        "non_revoked",
        "evidence_observed",
    }
    expected_source_rule_ids = (
        "EXACT_GATE_SOURCE_SCHEMA_KIND_AND_FAIL_CLOSED_AVAILABILITY",
        "CONTENT_ID_RAW_SHA256_AND_FOUR_ROLE_PROVENANCE_SIGNATURES",
        "RAW_NATIVE_SOURCE_KIND_ORDER_CARDINALITY_SHA256_AND_BYTE_COUNT",
        "SOURCE_ROOT_RECOMPUTATION",
        "PREDECESSOR_SOURCE_RECEIPT_AND_ALLOCATION_LINEAGE",
        "DISPATCH_IS_SCHEMA_ONLY_AND_CANNOT_ESTABLISH_GATE_TRUTH",
    )
    for ordinal, (spec, source, support, gate) in enumerate(
        zip(source_specs, sources, supports, gates, strict=True), start=1
    ):
        assert spec.ordinal == source["gate_ordinal"] == gate["ordinal"] == ordinal
        assert spec.gate_id == source["gate_id"] == gate["gate_id"]
        assert source["dependencies_in_exact_order"] == list(
            catalog.GATE_SPECS_IN_EXACT_ORDER_V1[ordinal - 1].dependencies_in_exact_order
        )
        assert len(source["dependencies_in_exact_order"]) == len(
            set(source["dependencies_in_exact_order"])
        )
        assert all(value in seen_gate_ids for value in source["dependencies_in_exact_order"])
        assert source["predecessor_gate_source_receipt_cardinality"] == len(
            source["dependencies_in_exact_order"]
        )
        assert source["raw_predecessor_gate_source_receipt_payloads_required"] is True
        assert source["raw_native_source_payload_tuple_required"] is True
        assert source["verification_availability"] == (
            catalog.CURRENT_GATE_VERIFICATION_AVAILABILITY_V1
        ) == "SCHEMA_FROZEN_NOT_VERIFIABLE"
        assert source["verifier_dispatch_id"] == spec.verifier_dispatch_id
        assert source["verifier_source_sha256_or_none"] is None
        assert source["allowed_native_source_kind_orders_in_exact_order"] == [
            list(item) for item in spec.allowed_native_source_kind_orders_in_exact_order
        ]
        assert source["allowed_native_source_cardinalities_in_exact_order"] == [
            len(item) for item in spec.allowed_native_source_kind_orders_in_exact_order
        ]
        assert source["source_root_domain_ascii"] == expected_source_root_domains[
            ordinal - 1
        ]
        assert source["source_root_preimage_fields_in_exact_order"] == [
            "native_source_kinds_in_exact_order",
            "native_source_payload_sha256s_in_exact_order",
            "native_source_payload_byte_counts_in_exact_order",
        ]
        assert tuple(
            rule["rule_id"] for rule in source["binding_rules_in_exact_order"]
        ) == expected_source_rule_ids
        assert source["binding_rules_in_exact_order"][0] == {
            "rule_id": "EXACT_GATE_SOURCE_SCHEMA_KIND_AND_FAIL_CLOSED_AVAILABILITY",
            "subject_fields_in_exact_order": [
                "descriptor.gate_source_schema_id",
                "descriptor.gate_source_receipt_artifact_kind",
                "gate_descriptor.gate_schema_id",
                "descriptor.verification_availability",
                "descriptor.verifier_dispatch_id",
                "raw_gate_source_receipt_payload",
            ],
            "requirement": (
                "DECODE_THE_EXACT_DESCRIPTOR_SPECIFIC_CANONICAL_GATE_SOURCE_RECEIPT_"
                "REJECT_UNKNOWN_CROSS_GATE_SCHEMA_KIND_DOMAIN_PREFIX_OR_FIELDSET_"
                "REQUIRE_BODY_GATE_SCHEMA_ID_EXACT_EQUAL_THE_MATCHED_GATE_DESCRIPTOR_"
                "AND_REQUIRE_CURRENT_V1_VERIFICATION_AVAILABILITY_EXACT_SCHEMA_FROZEN_"
                "NOT_VERIFIABLE_WITH_VERIFIER_SOURCE_SHA256_EXACT_NULL"
            ),
        }
        assert source["signature_cardinality"] == 4
        assert tuple(
            (slot["role"], slot["key_epoch"])
            for slot in source["signature_slots_in_exact_order"]
        ) == (
            ("GOVERNANCE_ANCHOR", 1),
            ("CUSTODIAN", 1),
            ("STORE_RECEIPT", 1),
            ("TIME_ATTESTOR", 1),
        )
        assert forbidden_source_truth_fields.isdisjoint(
            source["body_fields_in_exact_order"]
        )
        assert source["body_required_literals"] == {
            "status": "RAW_NATIVE_SOURCE_RECEIPT_PRESENTED",
            "claim_level": "RAW_PROVENANCE_ONLY_NOT_GATE_DECISION",
            "lineage_scope": source["lineage_scope"],
            "gate_ordinal": ordinal,
            "gate_id": source["gate_id"],
            "verification_availability": "SCHEMA_FROZEN_NOT_VERIFIABLE",
            "verifier_dispatch_id": source["verifier_dispatch_id"],
            "verifier_source_sha256_or_none": None,
            "producer_observation_sequence": ordinal,
            "first_failure_code": "GATE_VERIFIER_NOT_IMPLEMENTED",
        }
        assert support["gate_source_schema_id"] == source["gate_source_schema_id"]
        assert support["raw_gate_source_receipt_payload_required"] is True
        assert support["raw_gate_source_native_payload_tuple_required"] is True
        assert support["verification_availability"] == (
            "SCHEMA_FROZEN_NOT_VERIFIABLE"
        )
        assert support["verifier_dispatch_id"] == source["verifier_dispatch_id"]
        assert support["verifier_source_sha256_or_none"] is None
        assert all(
            support["success_required_literals"][field] is False
            for field in ("evidence_observed", "current", "non_revoked", "satisfied")
        )
        assert support["success_required_literals"]["first_failure_code"] == (
            "GATE_VERIFIER_NOT_IMPLEMENTED"
        )
        assert gate["required_supporting_artifact_kinds_in_exact_order"] == [
            "PREALLOCATION_AUTHORITY_LINEAGE_V1",
            source["gate_source_receipt_artifact_kind"],
            support["gate_support_artifact_kind"],
        ]
        assert gate["gate_source_schema_id"] == source["gate_source_schema_id"]
        assert gate["gate_support_schema_id"] == support["gate_support_schema_id"]
        assert gate["current_catalog_gate_success_allowed"] is False
        assert gate["current_catalog_gate_success_failure_code"] == (
            "GATE_VERIFIER_NOT_IMPLEMENTED"
        )
        seen_gate_ids.append(source["gate_id"])
    assert tuple(item["source_root_domain_ascii"] for item in sources) == (
        expected_source_root_domains
    )
    assert len(set(expected_source_root_domains)) == 16
    source_root_probe = {
        "native_source_kinds_in_exact_order": ["K"],
        "native_source_payload_sha256s_in_exact_order": ["0" * 64],
        "native_source_payload_byte_counts_in_exact_order": [0],
    }
    assert "sha256:" + hashlib.sha256(
        expected_source_root_domains[0].encode("ascii")
        + canonicalize_rfc8785_v1(source_root_probe)
    ).hexdigest() == (
        "sha256:28001fdec542839e3829b3747424f4e9ab62cf16bb055cca20c7878c977d2e9d"
    )
    assert len(body["signer_registry_schema_object"][
        "required_authorization_rows_in_exact_order"
    ]) == 210


def test_gate_source_static_mutation_matrix_rejects() -> None:
    script = r'''
import copy
import sys
sys.path.insert(0, "src")
import hegel_machine.phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1 as m

def must_reject(target, mutation):
    original = copy.deepcopy(target)
    mutation(target)
    try:
        try:
            m._assert_static_catalog_v1()
        except RuntimeError:
            pass
        else:
            raise AssertionError("gate-source mutation was accepted")
    finally:
        target.clear()
        target.update(original)
    m._assert_static_catalog_v1()

sources=m.GATE_SOURCE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
supports=m.GATE_SUPPORT_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
gates=m.GATE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1

def self_dependency(item):
    item["dependencies_in_exact_order"]=[item["gate_id"]]
    item["predecessor_gate_source_receipt_cardinality"]=1
must_reject(sources[0], self_dependency)

def forward_dependency(item):
    item["dependencies_in_exact_order"]=[sources[-1]["gate_id"]]
    item["predecessor_gate_source_receipt_cardinality"]=1
must_reject(sources[1], forward_dependency)

def duplicate_dependency(item):
    item["dependencies_in_exact_order"]=[sources[0]["gate_id"]] * 2
    item["predecessor_gate_source_receipt_cardinality"]=2
must_reject(sources[1], duplicate_dependency)

must_reject(
    sources[0],
    lambda item: item.__setitem__(
        "source_root_domain_ascii",
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE/1/MUTATED_SOURCE_ROOT/V1\x00",
    ),
)
must_reject(
    sources[0],
    lambda item: item["binding_rules_in_exact_order"].pop(0),
)

def inject_claim(item):
    item["body_fields_in_exact_order"].append("satisfied")
    item["body_field_types_in_exact_order"].append(
        {"field_name":"satisfied", "exact_wire_type":"EXACT_JSON_BOOLEAN"}
    )
must_reject(sources[0], inject_claim)

must_reject(
    supports[0],
    lambda item: item["success_required_literals"].__setitem__("satisfied", True),
)
must_reject(
    gates[0],
    lambda item: item.__setitem__("current_catalog_gate_success_allowed", True),
)
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_exact_stages_and_interleaved_dag_are_acyclic_and_complete() -> None:
    assert catalog.STAGE_NAMES_IN_EXACT_ORDER_V1 == EXPECTED_STAGES
    assert tuple(item.stage for item in catalog.STAGE_SPECS_IN_EXACT_ORDER_V1) == EXPECTED_STAGES
    assert tuple(item.stage for item in catalog.STAGE_SPECS_IN_EXACT_ORDER_V1 if item.terminal) == (
        "TERMINAL_CONSUMED",
    )
    edges = catalog.INTERLEAVED_DAG_EDGES_IN_EXACT_ORDER_V1
    nodes = set(EXPECTED_GATES) | set(EXPECTED_STAGES)
    indegree = {node: 0 for node in nodes}
    successors = {node: [] for node in nodes}
    for predecessor, successor in edges:
        assert predecessor in nodes and successor in nodes
        successors[predecessor].append(successor)
        indegree[successor] += 1
    ready = [node for node, count in indegree.items() if count == 0]
    seen: list[str] = []
    while ready:
        node = ready.pop()
        seen.append(node)
        for successor in successors[node]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    assert len(seen) == len(nodes) == 26
    order = {node: index for index, node in enumerate(seen)}
    assert all(order[a] < order[b] for a, b in edges)
    assert ("R13_FRESH_IDS_ALLOCATED_BY_VERIFIED_OVERLAY", "PREREGISTERED") in edges
    assert ("PREREGISTERED", "R14_ACTUAL_UNSEALED_960_COHORT_GENERATED") in edges
    assert ("R14_ACTUAL_UNSEALED_960_COHORT_GENERATED", "RAW_INPUT_COMMITTED") in edges
    assert ("RAW_INPUT_COMMITTED", "R15_ACTUAL_CORPUS_COVERT_AUDIT_PASSED") in edges
    assert ("PACKAGE_COMMITMENTS_FROZEN", "R16_INFERENCE_ADMISSION_UNLOCKED") in edges
    assert ("R16_INFERENCE_ADMISSION_UNLOCKED", "RUN_STARTED") in edges


def test_package_catalog_exact_names_backward_only_and_terminal_semantics() -> None:
    specs = catalog.PACKAGE_SPECS_IN_EXACT_ORDER_V1
    schema_objects = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1
    assert tuple(item.ordinal for item in specs) == tuple(range(18))
    assert tuple(item.package_name for item in specs) == EXPECTED_PACKAGES
    assert all(all(value < item.ordinal for value in item.predecessor_package_ordinals_in_exact_order) for item in specs)
    assert specs[0].predecessor_package_ordinals_in_exact_order == ()
    assert specs[17].predecessor_package_ordinals_in_exact_order == tuple(range(17))
    assert specs[0].finalization_stage == "TERMINAL_CONSUMED"
    assert specs[10].finalization_stage == "PREDICTIONS_COMMITTED"
    assert specs[15].predecessor_package_ordinals_in_exact_order == (
        10,
        11,
        12,
        13,
        14,
    )
    assert specs[16].predecessor_package_ordinals_in_exact_order == (10, 15)
    assert specs[16].finalization_stage == "REPORTS_COMMITTED"
    assert specs[17].finalization_stage == "TERMINAL_CONSUMED"
    assert specs[17].minimum_item_count == specs[17].maximum_item_count == 17
    stage_index = {stage: index for index, stage in enumerate(EXPECTED_STAGES)}
    required_instance_lineage_fields = {
        "external_evidence_class",
        "external_evidence_disposition",
        "predecessor_package_instance_ids_in_exact_order",
        "predecessor_package_instance_payload_sha256s_in_exact_order",
        "interleaved_dag_root",
    }
    for spec, descriptor in zip(specs, schema_objects, strict=True):
        assert required_instance_lineage_fields.issubset(spec.body_fields_in_exact_order)
        expected_predecessor_schema_ids = [
            schema_objects[ordinal]["package_schema_id"]
            for ordinal in spec.predecessor_package_ordinals_in_exact_order
        ]
        assert descriptor["predecessor_package_schema_ids_in_exact_order"] == (
            expected_predecessor_schema_ids
        )
        expected_predecessor_finalization_stages = [
            specs[ordinal].finalization_stage
            for ordinal in spec.predecessor_package_ordinals_in_exact_order
        ]
        assert descriptor[
            "predecessor_package_finalization_stages_in_exact_order"
        ] == expected_predecessor_finalization_stages
        assert all(
            stage_index[predecessor_stage]
            <= stage_index[spec.finalization_stage]
            for predecessor_stage in expected_predecessor_finalization_stages
        )
        assert descriptor["predecessor_runtime_chronology_rule"] == (
            "EVERY_COMPLETE_PREDECESSOR_PACKAGE_FINALIZATION_STAGE_MUST_NOT_"
            "FOLLOW_THE_CURRENT_PACKAGE_FINALIZATION_STAGE"
        )
        assert descriptor["predecessor_package_instance_cardinality"] == len(
            expected_predecessor_schema_ids
        )
        assert descriptor["raw_predecessor_envelope_payloads_required"] is True
        assert tuple(
            rule["rule_id"]
            for rule in descriptor[
                "predecessor_instance_binding_rules_in_exact_order"
            ]
        ) == (
            "EXACT_FOUR_TUPLE_CARDINALITY_AND_ORDER",
            "SCHEMA_ID_FROM_PREDECESSOR_ORDINAL",
            "RAW_PREDECESSOR_REQUIRED",
            "INSTANCE_CONTENT_ID_REDERIVATION",
            "RAW_PAYLOAD_SHA256_BINDING",
            "PREDECESSOR_SIGNATURE_VERIFICATION",
            "PREDECESSOR_BODY_CROSS_BINDING",
            "SELF_FORWARD_AND_DUPLICATE_INSTANCE_FORBIDDEN",
            "ATOMIC_LINEAGE_REJECTION",
        )
        assert descriptor["required_external_evidence_class"] == (
            "AUTHORITATIVE_C1_EVIDENCE_REQUIRING_EXTERNAL_LINEAGE"
        )
        assert descriptor["catalog_evidence_disposition"] == (
            "NOT_SUPPLIED_NOT_VERIFIED"
        )
        assert descriptor["producer_evidence_disposition"] == "PRESENTED_UNVERIFIED"
        assert descriptor["verifier_success_disposition"] == (
            "VERIFIED_PRESENTED_EXTERNAL_EVIDENCE_NOT_GATE_ADMISSION"
        )
        assert descriptor["verifier_failure_dispositions_in_exact_order"] == [
            "REJECTED",
            "INVALIDATED",
        ]
        assert descriptor["interleaved_dag_root"] == catalog.DAG_ROOT_V1
        expected_body_literals = {
            "external_evidence_class": (
                "AUTHORITATIVE_C1_EVIDENCE_REQUIRING_EXTERNAL_LINEAGE"
            ),
            "external_evidence_disposition": "PRESENTED_UNVERIFIED",
            "interleaved_dag_root": catalog.DAG_ROOT_V1,
        }
        if spec.ordinal in (15, 17):
            expected_body_literals["overall_gate_count"] = 12
            expected_body_literals["slice_gate_count"] = 24
        assert descriptor["body_required_literals"] == expected_body_literals
    assert schema_objects[0]["predecessor_package_schema_ids_in_exact_order"] == []
    assert len(schema_objects[17]["predecessor_package_schema_ids_in_exact_order"]) == 17
    bundle = catalog.BUNDLE_INDEX_SCHEMA_OBJECT_V1
    assert bundle["predecessor_package_ordinals_in_exact_order"] == list(range(18))
    assert bundle["predecessor_package_schema_ids_in_exact_order"] == [
        item["package_schema_id"] for item in schema_objects
    ]
    assert bundle["package_cardinality"] == 18
    assert bundle["package_instance_cardinality"] == 18
    assert bundle["raw_package_envelope_payloads_required"] is True
    assert bundle["created_after_stage"] == "TERMINAL_CONSUMED"
    assert {
        "package_schema_ids_in_exact_order",
        "package_instance_ids_in_exact_order",
        "package_instance_payload_sha256s_in_exact_order",
    }.issubset(bundle["body_fields_in_exact_order"])
    assert "package_ids_in_exact_order" not in bundle["body_fields_in_exact_order"]
    assert all(item.schema_version.endswith(f"package/{item.ordinal}/1") for item in specs)


def test_signature_slots_keysets_and_cardinalities_are_exact() -> None:
    specs = catalog.PACKAGE_SPECS_IN_EXACT_ORDER_V1
    assert len(specs[0].signature_slots_in_exact_order) == 4
    assert tuple(slot.ordinal for slot in specs[0].signature_slots_in_exact_order) == tuple(range(4))
    assert tuple(slot.role for slot in specs[0].signature_slots_in_exact_order) == (
        "STORE_RECEIPT",
        "TIME_ATTESTOR",
        "GOVERNANCE_ANCHOR",
        "CUSTODIAN",
    )
    package_zero = catalog.PACKAGE_SCHEMA_OBJECTS_IN_EXACT_ORDER_V1[0]
    assert package_zero["nested_signature_cardinality"] == 16
    assert package_zero["total_evidence_signature_cardinality"] == 20
    assert len(specs[17].signature_slots_in_exact_order) == 3
    assert tuple(slot.role for slot in specs[17].signature_slots_in_exact_order) == (
        "GOVERNANCE_ANCHOR",
        "CUSTODIAN",
        "SCORER_EVALUATOR",
    )
    assert specs[9].minimum_item_count == specs[9].maximum_item_count == 3
    assert specs[11].minimum_item_count == specs[11].maximum_item_count == 572
    assert specs[12].minimum_item_count == specs[12].maximum_item_count == 240
    assert specs[14].minimum_item_count == specs[14].maximum_item_count == 10_000
    assert specs[15].minimum_item_count == specs[15].maximum_item_count == 36
    assert specs[17].minimum_item_count == specs[17].maximum_item_count == 17
    for spec in specs:
        assert spec.envelope_fields_in_exact_order == (
            "schema_version",
            "body",
            "package_id",
            "signatures_in_exact_order",
        )
        assert spec.body_fields_in_exact_order[:16] == (
            "schema_version",
            "status",
            "claim_level",
            "catalog_id",
            "signer_registry_id",
            "signer_registry_payload_sha256",
            "signer_registry_entries_root_sha256",
            "package_ordinal",
            "package_name",
            "external_evidence_class",
            "external_evidence_disposition",
            "campaign_id",
            "attempt_id",
            "predecessor_package_instance_ids_in_exact_order",
            "predecessor_package_instance_payload_sha256s_in_exact_order",
            "interleaved_dag_root",
        )
        assert spec.content_id_domain_ascii.endswith("\x00")
        assert all(slot.key_epoch == 1 and slot.signature_domain_ascii.endswith("\x00") for slot in spec.signature_slots_in_exact_order)


def test_literal_content_id_and_length_framed_root_oracles() -> None:
    wrapper = json.loads(_raw())
    assert type(wrapper) is dict
    body = wrapper["body"]
    assert type(body) is dict
    groups = (
        ("gate_schema_objects_in_exact_order", "gate_schema_id"),
        ("gate_source_schema_objects_in_exact_order", "gate_source_schema_id"),
        ("gate_support_schema_objects_in_exact_order", "gate_support_schema_id"),
        ("stage_schema_objects_in_exact_order", "stage_schema_id"),
        ("package_schema_objects_in_exact_order", "package_schema_id"),
    )
    schema_domains: set[str] = set()
    schema_prefixes: set[str] = set()
    artifact_domains: set[str] = set()
    artifact_prefixes: set[str] = set()
    for group_name, id_name in groups:
        group = body[group_name]
        assert type(group) is list
        for item in group:
            assert type(item) is dict
            core = dict(item)
            identifier = core.pop(id_name)
            schema_domain = core["schema_content_id_domain_ascii"]
            schema_prefix = core["schema_content_id_prefix"]
            artifact_domain = core["content_id_domain_ascii"]
            artifact_prefix = core["content_id_prefix"]
            assert identifier == _cid(schema_domain, schema_prefix, core)
            assert schema_domain.endswith("\x00") and artifact_domain.endswith("\x00")
            assert schema_domain not in schema_domains
            assert schema_prefix not in schema_prefixes
            assert artifact_domain not in artifact_domains
            assert artifact_prefix not in artifact_prefixes
            schema_domains.add(schema_domain)
            schema_prefixes.add(schema_prefix)
            artifact_domains.add(artifact_domain)
            artifact_prefixes.add(artifact_prefix)
    for singleton_name, id_name in (
        ("signer_registry_schema_object", "signer_registry_schema_id"),
        ("preallocation_lineage_schema_object", "preallocation_lineage_schema_id"),
    ):
        core = dict(body[singleton_name])
        identifier = core.pop(id_name)
        schema_domain = core["schema_content_id_domain_ascii"]
        schema_prefix = core["schema_content_id_prefix"]
        artifact_domain = core["content_id_domain_ascii"]
        artifact_prefix = core["content_id_prefix"]
        assert identifier == _cid(schema_domain, schema_prefix, core)
        assert schema_domain.endswith("\x00") and artifact_domain.endswith("\x00")
        assert schema_domain not in schema_domains
        assert schema_prefix not in schema_prefixes
        assert artifact_domain not in artifact_domains
        assert artifact_prefix not in artifact_prefixes
        schema_domains.add(schema_domain)
        schema_prefixes.add(schema_prefix)
        artifact_domains.add(artifact_domain)
        artifact_prefixes.add(artifact_prefix)
    bundle_core = dict(body["bundle_index_schema_object"])
    bundle_identifier = bundle_core.pop("bundle_index_schema_id")
    assert bundle_identifier == _cid(
        bundle_core["schema_content_id_domain_ascii"],
        bundle_core["schema_content_id_prefix"],
        bundle_core,
    )
    assert body["bundle_index_schema_id"] == bundle_identifier
    assert bundle_core["schema_content_id_domain_ascii"] not in schema_domains
    assert bundle_core["schema_content_id_prefix"] not in schema_prefixes
    assert bundle_core["content_id_domain_ascii"] not in artifact_domains
    assert bundle_core["content_id_prefix"] not in artifact_prefixes
    schema_domains.add(bundle_core["schema_content_id_domain_ascii"])
    schema_prefixes.add(bundle_core["schema_content_id_prefix"])
    artifact_domains.add(bundle_core["content_id_domain_ascii"])
    artifact_prefixes.add(bundle_core["content_id_prefix"])
    assert len(schema_domains) == len(schema_prefixes) == 79
    assert len(artifact_domains) == len(artifact_prefixes) == 79
    gate_ids = tuple(item["gate_schema_id"] for item in body[groups[0][0]])
    gate_source_ids = tuple(
        item["gate_source_schema_id"] for item in body[groups[1][0]]
    )
    gate_support_ids = tuple(
        item["gate_support_schema_id"] for item in body[groups[2][0]]
    )
    stage_ids = tuple(item["stage_schema_id"] for item in body[groups[3][0]])
    package_ids = tuple(item["package_schema_id"] for item in body[groups[4][0]])
    assert body["gate_schema_ids_root"] == _root(
        b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/GATE_ROOT/V1\x00",
        "phase2b_actual_unsealed_960_authoritative_gate_schema_root_v1_",
        gate_ids,
    )
    assert body["gate_source_schema_ids_root"] == _root(
        b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/GATE_SOURCE_SCHEMA_ROOT/V1\x00",
        "phase2b_actual_unsealed_960_gate_source_schema_root_v1_",
        gate_source_ids,
    )
    assert body["gate_support_schema_ids_root"] == _root(
        b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/GATE_SUPPORT_SCHEMA_ROOT/V1\x00",
        "phase2b_actual_unsealed_960_gate_support_schema_root_v1_",
        gate_support_ids,
    )
    assert body["stage_schema_ids_root"] == _root(
        b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/STAGE_ROOT/V1\x00",
        "phase2b_actual_unsealed_960_authoritative_stage_schema_root_v1_",
        stage_ids,
    )
    assert body["package_schema_ids_root"] == _root(
        b"HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/PACKAGE_ROOT/V1\x00",
        "phase2b_actual_unsealed_960_authoritative_package_schema_root_v1_",
        package_ids,
    )
    package_dag_projection = [
        {
            "ordinal": spec.ordinal,
            "package_schema_id": descriptor["package_schema_id"],
            "predecessor_package_ordinals_in_exact_order": list(
                spec.predecessor_package_ordinals_in_exact_order
            ),
            "predecessor_package_schema_ids_in_exact_order": descriptor[
                "predecessor_package_schema_ids_in_exact_order"
            ],
        }
        for spec, descriptor in zip(
            catalog.PACKAGE_SPECS_IN_EXACT_ORDER_V1,
            body["package_schema_objects_in_exact_order"],
            strict=True,
        )
    ]
    assert body["package_schema_dag_root"] == _cid(
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/"
        "PACKAGE_SCHEMA_DAG_ROOT/V1\x00",
        "phase2b_actual_unsealed_960_authoritative_package_schema_dag_root_v1_",
        package_dag_projection,
    )
    assert body["interleaved_dag_root"] == _cid(
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/DAG_ROOT/V1\x00",
        "phase2b_actual_unsealed_960_authoritative_runbook_dag_root_v1_",
        body["interleaved_dag_edges_in_exact_order"],
    )
    catalog_core = dict(wrapper)
    identifier = catalog_core.pop("catalog_id")
    assert identifier == _cid(
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/CATALOG/ID/V1\x00",
        "phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1_",
        catalog_core["body"],
    )


def test_exact_preimage_profiles_and_field_type_manifests_are_machine_complete() -> None:
    body = json.loads(_raw())["body"]
    assert body["wire_preimage_profiles"] == {
        "content_id": {
            "profile": "BODY_ONLY_DOMAIN_BYTES_PLUS_RFC8785_CANONICAL_BODY_SHA256_V1",
            "fields_in_exact_order": ["body"],
            "formula": (
                "content_id_prefix || lowerhex(sha256(content_id_domain_ascii_bytes || "
                "rfc8785(body)))"
            ),
            "content_id_field_excluded_from_preimage": True,
            "signatures_excluded_from_preimage": True,
        },
        "detached_signature": {
            "profile": "DOMAIN_BYTES_PLUS_RFC8785_EXACT_DETACHED_SIGNATURE_PREIMAGE_V1",
            "schema_version": (
                "hegel-machine-phase2b-actual-unsealed-960-authoritative-evidence-"
                "detached-signature-preimage/1"
            ),
            "algorithm": "Ed25519",
            "fields_in_exact_order": [
                "schema_version",
                "artifact_kind",
                "artifact_schema_version",
                "content_id",
                "body",
                "signature_ordinal",
                "role",
                "purpose",
                "key_epoch",
                "signer_key_id",
            ],
            "formula": (
                "Ed25519.sign(private_key, signature_domain_ascii_bytes || "
                "rfc8785(detached_signature_preimage_object))"
            ),
            "excluded_fields_in_exact_order": [
                "signature_hex",
                "signatures_in_exact_order",
            ],
        },
    }
    profiles = {
        item["exact_wire_type"]: item
        for item in body["exact_special_wire_type_profiles_in_exact_order"]
    }
    gate_enum_type = "EXACT_ASCII_ENUM_MEMBER_OF_GATE_IDS_IN_EXACT_ORDER"
    unit_decimal_type = "EXACT_ASCII_CANONICAL_UNIT_INTERVAL_DECIMAL"
    integer_decimal_type = (
        "EXACT_ASCII_CANONICAL_NONNEGATIVE_DECIMAL_INTEGER_NO_LEADING_ZERO"
    )
    assert profiles[gate_enum_type]["reference_literals_in_exact_order"] == list(
        EXPECTED_GATES
    )
    assert profiles[unit_decimal_type]["fullmatch_ascii_regex_or_none"] == (
        r"(?:0|1|0\.[0-9]*[1-9])"
    )
    assert profiles[integer_decimal_type]["fullmatch_ascii_regex_or_none"] == (
        r"(?:0|[1-9][0-9]*)"
    )
    frozen_thresholds = body["frozen_statistical_thresholds_in_exact_order"]
    assert [item["threshold_id"] for item in frozen_thresholds] == [
        item[0] for item in catalog.FROZEN_STATISTICAL_THRESHOLDS_IN_EXACT_ORDER_V1
    ]
    assert all(
        type(item["exact_numerator"]) is int
        and type(item["exact_denominator"]) is int
        and item["exact_denominator"] > 0
        for item in frozen_thresholds
    )
    descriptor_rows: list[tuple[dict[str, object], str, str, str, str]] = []
    descriptor_rows.extend(
        (
            item,
            "evidence_envelope_fields_in_exact_order",
            "evidence_envelope_field_types_in_exact_order",
            "evidence_body_fields_in_exact_order",
            "evidence_body_field_types_in_exact_order",
        )
        for item in body["gate_schema_objects_in_exact_order"]
    )
    descriptor_rows.extend(
        (
            item,
            "envelope_fields_in_exact_order",
            "envelope_field_types_in_exact_order",
            "body_fields_in_exact_order",
            "body_field_types_in_exact_order",
        )
        for item in body["gate_source_schema_objects_in_exact_order"]
    )
    descriptor_rows.extend(
        (
            item,
            "envelope_fields_in_exact_order",
            "envelope_field_types_in_exact_order",
            "body_fields_in_exact_order",
            "body_field_types_in_exact_order",
        )
        for item in body["gate_support_schema_objects_in_exact_order"]
    )
    descriptor_rows.extend(
        (
            item,
            "event_envelope_fields_in_exact_order",
            "event_envelope_field_types_in_exact_order",
            "event_body_fields_in_exact_order",
            "event_body_field_types_in_exact_order",
        )
        for item in body["stage_schema_objects_in_exact_order"]
    )
    descriptor_rows.extend(
        (
            item,
            "envelope_fields_in_exact_order",
            "envelope_field_types_in_exact_order",
            "body_fields_in_exact_order",
            "body_field_types_in_exact_order",
        )
        for item in body["package_schema_objects_in_exact_order"]
    )
    for singleton_name in (
        "signer_registry_schema_object",
        "preallocation_lineage_schema_object",
    ):
        descriptor_rows.append(
            (
                body[singleton_name],
                "envelope_fields_in_exact_order",
                "envelope_field_types_in_exact_order",
                "body_fields_in_exact_order",
                "body_field_types_in_exact_order",
            )
        )
    descriptor_rows.append(
        (
            body["bundle_index_schema_object"],
            "envelope_fields_in_exact_order",
            "envelope_field_types_in_exact_order",
            "body_fields_in_exact_order",
            "body_field_types_in_exact_order",
        )
    )
    assert len(descriptor_rows) == 79
    signature_domains: set[str] = set()
    artifact_kinds: set[str] = set()

    def assert_manifest(fields: object, manifest: object) -> None:
        assert type(fields) is list and type(manifest) is list
        assert [row["field_name"] for row in manifest] == fields
        for row in manifest:
            assert set(row) == {"field_name", "exact_wire_type"}
            assert type(row["field_name"]) is str
            assert type(row["exact_wire_type"]) is str
            assert row["exact_wire_type"].startswith("EXACT_")

    for item, envelope_fields, envelope_types, body_fields, body_types in descriptor_rows:
        assert item["content_id_preimage_profile"] == catalog.CONTENT_ID_PREIMAGE_PROFILE_V1
        assert item["content_id_preimage_fields_in_exact_order"] == ["body"]
        assert item["signature_preimage_profile"] == catalog.DETACHED_SIGNATURE_PREIMAGE_PROFILE_V1
        assert item["signature_preimage_schema_version"] == (
            catalog.DETACHED_SIGNATURE_PREIMAGE_SCHEMA_VERSION_V1
        )
        assert item["signature_algorithm"] == "Ed25519"
        assert item["signature_preimage_fields_in_exact_order"] == list(
            catalog.DETACHED_SIGNATURE_PREIMAGE_FIELDS_IN_EXACT_ORDER_V1
        )
        artifact_kind = item["signature_preimage_artifact_kind"]
        assert type(artifact_kind) is str and artifact_kind not in artifact_kinds
        artifact_kinds.add(artifact_kind)
        assert_manifest(item[envelope_fields], item[envelope_types])
        assert_manifest(item[body_fields], item[body_types])
        for nested in item.get("nested_field_manifests_in_exact_order", []):
            assert set(nested) == {
                "record_name",
                "fields_in_exact_order",
                "field_types_in_exact_order",
            }
            assert_manifest(
                nested["fields_in_exact_order"],
                nested["field_types_in_exact_order"],
            )
        slots = item["signature_slots_in_exact_order"]
        assert type(item["signature_cardinality"]) is int
        assert item["signature_cardinality"] == len(slots)
        for ordinal, slot in enumerate(slots):
            assert type(slot["ordinal"]) is int and slot["ordinal"] == ordinal
            assert type(slot["key_epoch"]) is int and slot["key_epoch"] == 1
            domain = slot["signature_domain_ascii"]
            assert domain.endswith("\x00") and domain not in signature_domains
            signature_domains.add(domain)
    for support in body["gate_support_schema_objects_in_exact_order"]:
        assert_manifest(
            support["inline_operational_evidence_fields_in_exact_order"],
            support["inline_operational_evidence_field_types_in_exact_order"],
        )
    assert len(signature_domains) == sum(
        item[0]["signature_cardinality"] for item in descriptor_rows
    )
    assert len(signature_domains) == 187
    assert len(artifact_kinds) == 79
    for gate_descriptor in body["gate_schema_objects_in_exact_order"]:
        gate_types = {
            row["field_name"]: row["exact_wire_type"]
            for row in gate_descriptor["evidence_body_field_types_in_exact_order"]
        }
        assert gate_types["gate_id"] == gate_enum_type
        assert gate_types["gate_schema_id"] == (
            "EXACT_ASCII_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
        )
    package_zero = body["package_schema_objects_in_exact_order"][0]
    nested_types = {
        nested["record_name"]: {
            row["field_name"]: row["exact_wire_type"]
            for row in nested["field_types_in_exact_order"]
        }
        for nested in package_zero["nested_field_manifests_in_exact_order"]
    }
    exact_integer = "EXACT_JSON_SAFE_INTEGER_MIN_0_BOOL_FORBIDDEN"
    assert nested_types["store_if_absent_receipt_body"]["stored_artifact_size_bytes"] == exact_integer
    assert nested_types["declared_time_token_body"]["previous_logical_time"] == exact_integer
    assert nested_types["declared_time_token_body"]["current_logical_time"] == exact_integer
    assert nested_types["custodian_ledger_event_body"][
        "predecessor_event_id_or_none"
    ] == (
        "EXACT_JSON_NULL_OR_SIGNED_STAGE_EVENT_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
    )
    assert "predecessor_event_id" not in nested_types["custodian_ledger_event_body"]
    gate_result_types = {
        row["field_name"]: row["exact_wire_type"]
        for nested in body["package_schema_objects_in_exact_order"][15][
            "nested_field_manifests_in_exact_order"
        ]
        if nested["record_name"] == "gate_result"
        for row in nested["field_types_in_exact_order"]
    }
    assert gate_result_types["passed"] == "EXACT_JSON_BOOLEAN"
    assert gate_result_types["point_passed"] == "EXACT_JSON_BOOLEAN"
    assert gate_result_types["correct_count"] == exact_integer
    assert gate_result_types["total_count"] == exact_integer
    assert gate_result_types["point_threshold_numerator"] == exact_integer
    assert gate_result_types["point_threshold_denominator"] == exact_integer
    attempt_record_types = {
        row["field_name"]: row["exact_wire_type"]
        for nested in body["package_schema_objects_in_exact_order"][16][
            "nested_field_manifests_in_exact_order"
        ]
        if nested["record_name"] == "attempt_record"
        for row in nested["field_types_in_exact_order"]
    }
    assert attempt_record_types["attempt_index"] == exact_integer
    assert attempt_record_types["predecessor_attempt_record_id_or_none"] == (
        "EXACT_JSON_NULL_OR_ATTEMPT_RECORD_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
    )
    assert "predecessor_attempt_record_id" not in attempt_record_types
    package_fourteen_types = {
        row["field_name"]: row["exact_wire_type"]
        for row in body["package_schema_objects_in_exact_order"][14][
            "body_field_types_in_exact_order"
        ]
    }
    assert package_fourteen_types["master_seed_uint32"] == exact_integer
    assert package_fourteen_types["master_seed_decimal"] == (
        integer_decimal_type
    )
    assert gate_result_types["statistical_gate_id"] == (
        "EXACT_ASCII_ENUM_MEMBER_OF_FROZEN_VALIDATION_STATISTICAL_GATE_IDS_IN_EXACT_ORDER"
    )
    for field_name in (
        "fixed_total_count_or_none",
        "wilson_threshold_numerator_or_none",
        "wilson_threshold_denominator_or_none",
    ):
        assert gate_result_types[field_name] == (
            "EXACT_JSON_NULL_OR_SAFE_INTEGER_MIN_0_BOOL_FORBIDDEN"
        )
    assert gate_result_types["wilson_passed_or_none"] == (
        "EXACT_JSON_NULL_OR_BOOLEAN"
    )
    assert gate_result_types["wilson_lower_binary64_hex_or_none"] == (
        "EXACT_JSON_NULL_OR_ASCII_IEEE754_BINARY64_HEX"
    )
    assert not any(
        nested["record_name"] == "gate_result"
        for nested in body["package_schema_objects_in_exact_order"][17][
            "nested_field_manifests_in_exact_order"
        ]
    )
    first_stage_types = {
        row["field_name"]: row["exact_wire_type"]
        for row in body["stage_schema_objects_in_exact_order"][0][
            "event_body_field_types_in_exact_order"
        ]
    }
    assert first_stage_types["terminal_disposition_or_none"] == (
        "EXACT_JSON_NULL_OR_ENUM_MEMBER_OF_TERMINAL_DISPOSITIONS_IN_EXACT_ORDER"
    )


def _assert_independent_catalog_hard_contract(body: dict[str, object]) -> None:
    gate_enum_type = "EXACT_ASCII_ENUM_MEMBER_OF_GATE_IDS_IN_EXACT_ORDER"
    content_id_type = "EXACT_ASCII_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
    for descriptor in body["gate_schema_objects_in_exact_order"]:
        fields = descriptor["evidence_body_fields_in_exact_order"]
        manifest = {
            row["field_name"]: row["exact_wire_type"]
            for row in descriptor["evidence_body_field_types_in_exact_order"]
        }
        assert "gate_id" in fields and "gate_schema_id" in fields
        assert manifest["gate_id"] == gate_enum_type
        assert manifest["gate_schema_id"] == content_id_type
        assert descriptor["gate_id"] in EXPECTED_GATES
        assert descriptor["gate_id_wire_type"] == gate_enum_type
        assert descriptor["dependency_gate_id_wire_type"] == gate_enum_type
        assert descriptor["predecessor_gate_evidence_cardinality"] == len(
            descriptor["dependencies_in_exact_order"]
        )
        assert descriptor[
            "raw_predecessor_gate_evidence_payloads_required"
        ] is True
        assert tuple(
            rule["rule_id"]
            for rule in descriptor[
                "predecessor_gate_evidence_binding_rules_in_exact_order"
            ]
        ) == (
            "DEPENDENCY_CARDINALITY_AND_ORDER",
            "RAW_DEPENDENCY_EVIDENCE_REQUIRED",
            "DEPENDENCY_CONTENT_ID_SIGNATURE_AND_SHA256",
            "DEPENDENCY_BODY_CROSS_BINDING",
            "SELF_FORWARD_DUPLICATE_AND_PARTIAL_FORBIDDEN",
        )
    for descriptor in body["stage_schema_objects_in_exact_order"]:
        assert descriptor["predecessor_node_cardinality"] == len(
            descriptor["predecessor_nodes_in_exact_order"]
        )
        assert descriptor[
            "raw_predecessor_evidence_or_event_payloads_required"
        ] is True
        assert tuple(
            rule["rule_id"]
            for rule in descriptor["predecessor_node_binding_rules_in_exact_order"]
        ) == (
            "PREDECESSOR_NODE_CARDINALITY_AND_ORDER",
            "RAW_PREDECESSOR_NODES_REQUIRED",
            "NODE_KIND_CONTENT_ID_SIGNATURE_AND_SHA256",
            "PREDECESSOR_STATE_AND_LINEAGE",
            "SELF_FORWARD_DUPLICATE_AND_PARTIAL_FORBIDDEN",
        )
    packages = body["package_schema_objects_in_exact_order"]
    stage_index = {stage: index for index, stage in enumerate(EXPECTED_STAGES)}
    required_body_fields = {
        "external_evidence_class",
        "external_evidence_disposition",
        "predecessor_package_instance_ids_in_exact_order",
        "predecessor_package_instance_payload_sha256s_in_exact_order",
        "interleaved_dag_root",
    }
    expected_instance_rule_ids = (
        "EXACT_FOUR_TUPLE_CARDINALITY_AND_ORDER",
        "SCHEMA_ID_FROM_PREDECESSOR_ORDINAL",
        "RAW_PREDECESSOR_REQUIRED",
        "INSTANCE_CONTENT_ID_REDERIVATION",
        "RAW_PAYLOAD_SHA256_BINDING",
        "PREDECESSOR_SIGNATURE_VERIFICATION",
        "PREDECESSOR_BODY_CROSS_BINDING",
        "SELF_FORWARD_AND_DUPLICATE_INSTANCE_FORBIDDEN",
        "ATOMIC_LINEAGE_REJECTION",
    )
    for ordinal, descriptor in enumerate(packages):
        predecessors = descriptor["predecessor_package_ordinals_in_exact_order"]
        assert all(type(value) is int and value < ordinal for value in predecessors)
        assert descriptor["predecessor_package_schema_ids_in_exact_order"] == [
            packages[value]["package_schema_id"] for value in predecessors
        ]
        expected_predecessor_finalization_stages = [
            packages[value]["finalization_stage"] for value in predecessors
        ]
        assert descriptor[
            "predecessor_package_finalization_stages_in_exact_order"
        ] == expected_predecessor_finalization_stages
        assert all(
            stage_index[predecessor_stage]
            <= stage_index[descriptor["finalization_stage"]]
            for predecessor_stage in expected_predecessor_finalization_stages
        )
        assert descriptor["predecessor_runtime_chronology_rule"] == (
            "EVERY_COMPLETE_PREDECESSOR_PACKAGE_FINALIZATION_STAGE_MUST_NOT_"
            "FOLLOW_THE_CURRENT_PACKAGE_FINALIZATION_STAGE"
        )
        assert required_body_fields.issubset(descriptor["body_fields_in_exact_order"])
        assert descriptor["predecessor_package_instance_cardinality"] == len(
            predecessors
        )
        assert descriptor["raw_predecessor_envelope_payloads_required"] is True
        rules = descriptor["predecessor_instance_binding_rules_in_exact_order"]
        assert tuple(rule["rule_id"] for rule in rules) == expected_instance_rule_ids
        for rule in rules:
            assert set(rule) == {
                "rule_id",
                "subject_fields_in_exact_order",
                "requirement",
            }
            assert rule["subject_fields_in_exact_order"]
            assert type(rule["requirement"]) is str and rule["requirement"]
        assert descriptor["required_external_evidence_class"] == (
            "AUTHORITATIVE_C1_EVIDENCE_REQUIRING_EXTERNAL_LINEAGE"
        )
        assert descriptor["catalog_evidence_disposition"] == (
            "NOT_SUPPLIED_NOT_VERIFIED"
        )
        assert descriptor["producer_evidence_disposition"] == "PRESENTED_UNVERIFIED"
        assert descriptor["interleaved_dag_root"] == body["interleaved_dag_root"]
        expected_literals = {
            "external_evidence_class": (
                "AUTHORITATIVE_C1_EVIDENCE_REQUIRING_EXTERNAL_LINEAGE"
            ),
            "external_evidence_disposition": "PRESENTED_UNVERIFIED",
            "interleaved_dag_root": body["interleaved_dag_root"],
        }
        if ordinal in (15, 17):
            expected_literals["overall_gate_count"] = 12
            expected_literals["slice_gate_count"] = 24
        assert descriptor["body_required_literals"] == expected_literals
    assert packages[0]["predecessor_package_schema_ids_in_exact_order"] == []
    assert packages[0]["predecessor_package_instance_cardinality"] == 0
    assert len(packages[17]["predecessor_package_schema_ids_in_exact_order"]) == 17
    assert packages[17]["predecessor_package_instance_cardinality"] == 17
    ledger_contracts = packages[0]["nested_record_binding_contracts_in_exact_order"]
    assert len(ledger_contracts) == 1
    ledger_contract = ledger_contracts[0]
    assert ledger_contract["record_name"] == "signed_stage_event_envelope"
    assert ledger_contract["event_body_record_name"] == "custodian_ledger_event_body"
    assert ledger_contract["record_content_id_domain_ascii"].endswith("\x00")
    assert ledger_contract["record_content_id_prefix"] == (
        "phase2b_actual_unsealed_960_signed_stage_event_v1_"
    )
    assert ledger_contract["required_record_cardinality"] == 10
    assert ledger_contract["raw_record_payloads_required"] is True
    assert ledger_contract["stage_names_in_exact_order"] == list(EXPECTED_STAGES)
    assert tuple(
        rule["rule_id"] for rule in ledger_contract["binding_rules_in_exact_order"]
    ) == (
        "LEDGER_EVENT_CONTENT_ID",
        "RAW_LEDGER_EVENT_PAYLOADS_REQUIRED",
        "OUTER_LEDGER_EVENT_THREE_TUPLE_BINDING",
        "LEDGER_EVENT_ID_SHA_AND_BODY_CROSS_BINDING",
        "LEDGER_EVENT_INDEX_STAGE_AND_PREDECESSOR_CHAIN",
        "LEDGER_EVENT_LEVEL_SIGNATURE_BINDING",
        "LEDGER_EVENT_CHAIN_ROOT",
    )
    gate_result = next(
        item
        for item in packages[15]["nested_field_manifests_in_exact_order"]
        if item["record_name"] == "gate_result"
    )
    manifest = {
        row["field_name"]: row["exact_wire_type"]
        for row in gate_result["field_types_in_exact_order"]
    }
    assert manifest["statistical_gate_id"] == (
        "EXACT_ASCII_ENUM_MEMBER_OF_FROZEN_VALIDATION_STATISTICAL_GATE_IDS_IN_EXACT_ORDER"
    )
    assert manifest["fixed_total_count_or_none"] == (
        "EXACT_JSON_NULL_OR_SAFE_INTEGER_MIN_0_BOOL_FORBIDDEN"
    )
    assert manifest["wilson_passed_or_none"] == "EXACT_JSON_NULL_OR_BOOLEAN"
    assert manifest["wilson_lower_binary64_hex_or_none"] == (
        "EXACT_JSON_NULL_OR_ASCII_IEEE754_BINARY64_HEX"
    )
    assert not any(
        item["record_name"] == "gate_result"
        for item in packages[17]["nested_field_manifests_in_exact_order"]
    )
    contract = next(
        item
        for item in packages[15]["nested_record_binding_contracts_in_exact_order"]
        if item["record_name"] == "gate_result"
    )
    assert contract["record_content_id_domain_ascii"] == (
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE_RESULT/ID/V1\x00"
    )
    assert contract["record_content_id_prefix"] == (
        "phase2b_actual_unsealed_960_gate_result_v1_"
    )
    assert contract["required_record_cardinality"] == 36
    assert contract["raw_record_payloads_required"] is True
    assert contract["outer_body_reference_fields_in_exact_order"] == [
        "gate_result_ids_in_exact_order",
        "gate_result_payload_sha256s_in_exact_order",
        "all_gate_results_root_sha256",
    ]
    assert contract["frozen_gate_result_plan_in_exact_order"] == body[
        "frozen_validation_gate_result_plan_in_exact_order"
    ]
    assert contract["frozen_gate_result_plan_in_exact_order"][10][
        "minimum_passing_correct_count_or_none"
    ] == 558
    assert tuple(
        rule["rule_id"]
        for rule in contract["cross_result_binding_rules_in_exact_order"]
    ) == (
        "FAIL_CLOSED_DYNAMIC_DENOMINATOR_BINDING",
        "PRESERVATION_COMPOSITE_BINDING",
        "FAMILY_AND_SCALE_ANSWERABLE_CROSS_SUM",
        "FAMILY_AND_SCALE_CONTROL_CROSS_SUM",
        "FAMILY_AND_SCALE_ABSTENTION_CROSS_SUM",
    )
    assert tuple(
        rule["rule_id"] for rule in contract["binding_rules_in_exact_order"]
    ) == (
        "GATE_RESULT_CONTENT_ID",
        "EXACT_36_GATE_PLAN_ROW",
        "RAW_GATE_RESULT_PAYLOADS_REQUIRED",
        "OUTER_GATE_RESULT_THREE_TUPLE_BINDING",
        "OUTER_GATE_RESULT_ID_SHA_AND_ROOT_REDERIVATION",
        "COUNT_AND_POINT_THRESHOLD_EXACTNESS",
        "WILSON_THRESHOLD_AND_BINARY64_RECOMPUTATION",
        "FINAL_GATE_PASS_CONJUNCTION",
    )
    attempt_contract = packages[16]["nested_record_binding_contracts_in_exact_order"]
    assert len(attempt_contract) == 1
    attempt_contract = attempt_contract[0]
    assert attempt_contract["record_name"] == "attempt_record"
    assert attempt_contract["record_content_id_domain_ascii"].endswith("\x00")
    assert attempt_contract["record_content_id_prefix"] == (
        "phase2b_actual_unsealed_960_attempt_record_v1_"
    )
    assert attempt_contract["allowed_record_cardinalities_in_exact_order"] == [1, 2]
    assert attempt_contract["raw_record_payloads_required"] is True
    assert attempt_contract["outer_body_reference_fields_in_exact_order"] == [
        "attempt_record_ids_in_exact_order",
        "attempt_record_payload_sha256s_in_exact_order",
        "attempt_indices_in_exact_order",
    ]
    assert tuple(
        rule["rule_id"] for rule in attempt_contract["binding_rules_in_exact_order"]
    ) == (
        "ATTEMPT_RECORD_CONTENT_ID",
        "ATTEMPT_INDEX_SEQUENCE",
        "ATTEMPT_ZERO_PREDECESSOR",
        "ATTEMPT_ONE_PREDECESSOR",
        "ATTEMPT_ONE_REEXECUTE_ELIGIBILITY",
        "UPLOAD_RETRY_NOT_NEW_ATTEMPT",
        "RAW_ATTEMPT_RECORD_PAYLOADS_REQUIRED",
        "OUTER_ATTEMPT_RECORD_FOUR_TUPLE_BINDING",
        "OUTER_ATTEMPT_RECORD_ID_SHA_INDEX_REDERIVATION",
        "OUTER_ATTEMPT_TERMINAL_SUMMARY_PROJECTION",
    )
    attempt_manifest = {
        row["field_name"]: row["exact_wire_type"]
        for nested in packages[16]["nested_field_manifests_in_exact_order"]
        if nested["record_name"] == "attempt_record"
        for row in nested["field_types_in_exact_order"]
    }
    assert "predecessor_attempt_record_id" not in attempt_manifest
    assert attempt_manifest["predecessor_attempt_record_id_or_none"] == (
        "EXACT_JSON_NULL_OR_ATTEMPT_RECORD_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
    )
    bundle = body["bundle_index_schema_object"]
    assert bundle["package_instance_cardinality"] == 18
    assert bundle["raw_package_envelope_payloads_required"] is True
    assert tuple(
        rule["rule_id"]
        for rule in bundle["package_instance_binding_rules_in_exact_order"]
    ) == (
        "EXACT_FOUR_TUPLE_CARDINALITY_AND_ORDER",
        "SCHEMA_ID_FROM_PACKAGE_ORDINAL",
        "RAW_PACKAGE_REQUIRED",
        "INSTANCE_ID_AND_RAW_SHA256",
        "PACKAGE_SIGNATURE_VERIFICATION",
        "PACKAGE_BODY_CROSS_BINDING",
        "PAIRWISE_DISTINCT_AND_ATOMIC",
    )
    nonclaims = body["permanent_nonclaims"]
    assert set(nonclaims) == set(catalog.PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1)
    assert len(nonclaims) == len(catalog.PERMANENT_NONCLAIMS_IN_EXACT_ORDER_V1)
    assert all(type(value) is bool and value is False for value in nonclaims.values())


@pytest.mark.parametrize(
    "mutation",
    (
        "GATE_ENUM_AS_CONTENT_ID",
        "GATE_SCHEMA_ID_AS_ENUM",
        "GATE_PREDECESSOR_RULE_DROP",
        "STAGE_PREDECESSOR_RULE_DROP",
        "LEDGER_EVENT_RULE_DROP",
        "STATISTICAL_GATE_ENUM_AS_CONTENT_ID",
        "PREDECESSOR_SCHEMA_SPLICE",
        "PREDECESSOR_FINALIZATION_AFTER_CHILD",
        "PREDECESSOR_INSTANCE_CARDINALITY_DRIFT",
        "PREDECESSOR_INSTANCE_RULE_DROP",
        "PREDECESSOR_INSTANCE_RULE_ORDER_SWAP",
        "MISSING_INSTANCE_LINEAGE_FIELD",
        "GATE_RESULT_PLAN_DRIFT",
        "GATE_RESULT_RAW_RULE_DROP",
        "GATE_RESULT_CROSS_RULE_DROP",
        "ATTEMPT_PREDECESSOR_NONNULL_TYPE",
        "ATTEMPT_RULE_DROP",
        "BUNDLE_INSTANCE_RULE_DROP",
        "BUNDLE_MEMBER_SIGNATURE_RULE_DROP",
        "EXTERNAL_CLASS_DRIFT",
        "PACKAGE_DAG_ROOT_DRIFT",
        "NONCLAIM_BOOL_AS_INT",
        "NONCLAIM_EXTRA_KEY",
    ),
)
def test_independent_hard_contract_negative_mutations(mutation: str) -> None:
    body = copy.deepcopy(json.loads(_raw())["body"])
    if mutation == "GATE_ENUM_AS_CONTENT_ID":
        body["gate_schema_objects_in_exact_order"][0][
            "evidence_body_field_types_in_exact_order"
        ][5]["exact_wire_type"] = "EXACT_ASCII_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
    elif mutation == "GATE_SCHEMA_ID_AS_ENUM":
        body["gate_schema_objects_in_exact_order"][0][
            "evidence_body_field_types_in_exact_order"
        ][6]["exact_wire_type"] = "EXACT_ASCII_ENUM_MEMBER_OF_GATE_IDS_IN_EXACT_ORDER"
    elif mutation == "GATE_PREDECESSOR_RULE_DROP":
        body["gate_schema_objects_in_exact_order"][1][
            "predecessor_gate_evidence_binding_rules_in_exact_order"
        ].pop()
    elif mutation == "STAGE_PREDECESSOR_RULE_DROP":
        body["stage_schema_objects_in_exact_order"][0][
            "predecessor_node_binding_rules_in_exact_order"
        ].pop()
    elif mutation == "LEDGER_EVENT_RULE_DROP":
        body["package_schema_objects_in_exact_order"][0][
            "nested_record_binding_contracts_in_exact_order"
        ][0]["binding_rules_in_exact_order"].pop()
    elif mutation == "STATISTICAL_GATE_ENUM_AS_CONTENT_ID":
        gate_result = next(
            item
            for item in body["package_schema_objects_in_exact_order"][15][
                "nested_field_manifests_in_exact_order"
            ]
            if item["record_name"] == "gate_result"
        )
        next(
            row
            for row in gate_result["field_types_in_exact_order"]
            if row["field_name"] == "statistical_gate_id"
        )["exact_wire_type"] = "EXACT_ASCII_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
    elif mutation == "PREDECESSOR_SCHEMA_SPLICE":
        packages = body["package_schema_objects_in_exact_order"]
        packages[3]["predecessor_package_schema_ids_in_exact_order"][0] = packages[3][
            "package_schema_id"
        ]
    elif mutation == "PREDECESSOR_FINALIZATION_AFTER_CHILD":
        body["package_schema_objects_in_exact_order"][3][
            "predecessor_package_finalization_stages_in_exact_order"
        ][0] = "TERMINAL_CONSUMED"
    elif mutation == "PREDECESSOR_INSTANCE_CARDINALITY_DRIFT":
        body["package_schema_objects_in_exact_order"][3][
            "predecessor_package_instance_cardinality"
        ] = 0
    elif mutation == "PREDECESSOR_INSTANCE_RULE_DROP":
        body["package_schema_objects_in_exact_order"][3][
            "predecessor_instance_binding_rules_in_exact_order"
        ].pop()
    elif mutation == "PREDECESSOR_INSTANCE_RULE_ORDER_SWAP":
        rules = body["package_schema_objects_in_exact_order"][3][
            "predecessor_instance_binding_rules_in_exact_order"
        ]
        rules[0], rules[1] = rules[1], rules[0]
    elif mutation == "MISSING_INSTANCE_LINEAGE_FIELD":
        body["package_schema_objects_in_exact_order"][0][
            "body_fields_in_exact_order"
        ].remove("predecessor_package_instance_ids_in_exact_order")
    elif mutation == "GATE_RESULT_PLAN_DRIFT":
        body["package_schema_objects_in_exact_order"][15][
            "nested_record_binding_contracts_in_exact_order"
        ][0]["frozen_gate_result_plan_in_exact_order"][0][
            "point_threshold_numerator"
        ] = 91
    elif mutation == "GATE_RESULT_RAW_RULE_DROP":
        body["package_schema_objects_in_exact_order"][15][
            "nested_record_binding_contracts_in_exact_order"
        ][0]["binding_rules_in_exact_order"].pop(2)
    elif mutation == "GATE_RESULT_CROSS_RULE_DROP":
        body["package_schema_objects_in_exact_order"][15][
            "nested_record_binding_contracts_in_exact_order"
        ][0]["cross_result_binding_rules_in_exact_order"].pop()
    elif mutation == "ATTEMPT_PREDECESSOR_NONNULL_TYPE":
        attempt_record = next(
            item
            for item in body["package_schema_objects_in_exact_order"][16][
                "nested_field_manifests_in_exact_order"
            ]
            if item["record_name"] == "attempt_record"
        )
        next(
            row
            for row in attempt_record["field_types_in_exact_order"]
            if row["field_name"] == "predecessor_attempt_record_id_or_none"
        )["exact_wire_type"] = (
            "EXACT_ASCII_CONTENT_IDENTIFIER_PREFIX_PLUS_64_LOWERHEX"
        )
    elif mutation == "ATTEMPT_RULE_DROP":
        body["package_schema_objects_in_exact_order"][16][
            "nested_record_binding_contracts_in_exact_order"
        ][0]["binding_rules_in_exact_order"].pop()
    elif mutation == "BUNDLE_INSTANCE_RULE_DROP":
        body["bundle_index_schema_object"][
            "package_instance_binding_rules_in_exact_order"
        ].pop()
    elif mutation == "BUNDLE_MEMBER_SIGNATURE_RULE_DROP":
        rules = body["bundle_index_schema_object"][
            "package_instance_binding_rules_in_exact_order"
        ]
        rules[:] = [
            rule
            for rule in rules
            if rule["rule_id"] != "PACKAGE_SIGNATURE_VERIFICATION"
        ]
    elif mutation == "EXTERNAL_CLASS_DRIFT":
        body["package_schema_objects_in_exact_order"][0][
            "required_external_evidence_class"
        ] = "LOCAL_SELF_REPORT"
    elif mutation == "PACKAGE_DAG_ROOT_DRIFT":
        body["package_schema_objects_in_exact_order"][0]["interleaved_dag_root"] += "0"
    elif mutation == "NONCLAIM_BOOL_AS_INT":
        body["permanent_nonclaims"]["evidence_supplied"] = 0
    else:
        body["permanent_nonclaims"]["unexpected_claim"] = False
    with pytest.raises(AssertionError):
        _assert_independent_catalog_hard_contract(body)


def test_canonical_unit_interval_decimal_grammar_and_exact_threshold_rationals() -> None:
    pattern = re.compile(r"(?:0|1|0\.[0-9]*[1-9])", flags=re.ASCII)
    for accepted in (
        "0",
        "1",
        "0.05",
        "0.08",
        "0.1",
        "0.15",
        "0.85",
        "0.9",
        "0.94",
        "0.95",
    ):
        assert pattern.fullmatch(accepted)
    for rejected in ("0.0", "0.10", "0.90", "1.0", "+0.5", "-0.1", "5e-2", " 0.5", "0.5 "):
        assert pattern.fullmatch(rejected) is None
    thresholds = json.loads(_raw())["body"][
        "frozen_statistical_thresholds_in_exact_order"
    ]

    def exact_decimal_or_none(numerator: int, denominator: int) -> str | None:
        left, right = numerator, denominator
        while right:
            left, right = right, left % right
        numerator //= left
        denominator //= left
        if numerator == 0:
            return "0"
        if numerator == denominator:
            return "1"
        twos = fives = 0
        while denominator % 2 == 0:
            twos += 1
            denominator //= 2
        while denominator % 5 == 0:
            fives += 1
            denominator //= 5
        if denominator != 1:
            return None
        scale = max(twos, fives)
        scaled = numerator * 2 ** (scale - twos) * 5 ** (scale - fives)
        digits = str(scaled).rjust(scale + 1, "0")
        fraction = digits[-scale:].rstrip("0") if scale else ""
        return digits[:-scale] + "." + fraction if fraction else digits

    for row in thresholds:
        assert type(row["exact_numerator"]) is int
        assert type(row["exact_denominator"]) is int
        assert row["exact_numerator"] >= 0
        assert row["exact_denominator"] > 0
        decimal = row["canonical_decimal_or_none"]
        assert decimal == exact_decimal_or_none(
            row["exact_numerator"], row["exact_denominator"]
        )
        if decimal is not None:
            assert type(decimal) is str and pattern.fullmatch(decimal)
    by_id = {row["threshold_id"]: row for row in thresholds}
    assert by_id["PRESERVATION_LEGAL_POINT_MINIMUM"][
        "canonical_decimal_or_none"
    ] is None
    assert by_id["CHALLENGE_LOW_OVERLAP_POINT_MINIMUM"][
        "canonical_decimal_or_none"
    ] is None
    assert by_id["PRESERVATION_INVALID_CONTROL_EXACT"][
        "canonical_decimal_or_none"
    ] == "1"
    assert by_id["CHALLENGE_HIGH_OVERLAP_POINT_MINIMUM"][
        "canonical_decimal_or_none"
    ] == "0.95"


def test_exact_12_overall_24_slice_gate_plan_and_nested_record_id() -> None:
    body = json.loads(_raw())["body"]
    plan = body["frozen_validation_gate_result_plan_in_exact_order"]
    assert len(plan) == 36
    assert len({row["statistical_gate_id"] for row in plan}) == 36
    assert [row["scope_axis"] for row in plan[:12]] == ["overall"] * 12
    assert [row["statistical_gate_id"] for row in plan[:4]] == [
        "OVERALL_FAMILY_EXACT",
        "OVERALL_BINDING_EXACT",
        "OVERALL_SCALE_SET_ACCURACY",
        "OVERALL_JOINT_EXACT",
    ]
    assert plan[9] == {
        "statistical_gate_id": "OVERALL_FAIL_CLOSED_RATE",
        "metric": "fail_closed_rate",
        "scope_axis": "overall",
        "scope_value": "all",
        "fixed_total_count_or_none": None,
        "point_threshold_numerator": 1,
        "point_threshold_denominator": 1,
        "wilson_threshold_numerator_or_none": None,
        "wilson_threshold_denominator_or_none": None,
        "minimum_passing_correct_count_or_none": None,
    }
    family_values = (
        "F01_symmetry_equivariance",
        "F02_monotonicity_order",
        "F03_conservation_balance",
        "F04_additivity_complementarity",
        "F05_locality_composition",
        "F06_negative_feedback_stability",
    )
    slice_metrics = (
        ("answerable_joint_exact", 40, 80, 100, 70, 100, 33),
        ("all_control_rejection", 64, 88, 100, 78, 100, 57),
        ("abstention_specificity", 38, 85, 100, 75, 100, 33),
    )
    expected_family = []
    for value in family_values:
        for metric, total, point_n, point_d, wilson_n, wilson_d, minimum in slice_metrics:
            expected_family.append(
                {
                    "statistical_gate_id": f"SLICE_FAMILY_{value}_{metric}".upper(),
                    "metric": metric,
                    "scope_axis": "family",
                    "scope_value": value,
                    "fixed_total_count_or_none": total,
                    "point_threshold_numerator": point_n,
                    "point_threshold_denominator": point_d,
                    "wilson_threshold_numerator_or_none": wilson_n,
                    "wilson_threshold_denominator_or_none": wilson_d,
                    "minimum_passing_correct_count_or_none": minimum,
                }
            )
    assert plan[12:30] == expected_family
    assert [row["fixed_total_count_or_none"] for row in plan[30:]] == [
        120,
        192,
        114,
        120,
        192,
        114,
    ]
    assert [row["minimum_passing_correct_count_or_none"] for row in plan[30:]] == [
        96,
        169,
        97,
        96,
        169,
        97,
    ]
    assert plan[10]["minimum_passing_correct_count_or_none"] == 558
    descriptor = body["package_schema_objects_in_exact_order"][15]
    contract = descriptor["nested_record_binding_contracts_in_exact_order"][0]
    assert contract["frozen_gate_result_plan_in_exact_order"] == plan
    preimage = {
        "statistical_gate_id": "OVERALL_FAIL_CLOSED_RATE",
        "metric": "fail_closed_rate",
        "scope_axis": "overall",
        "scope_value": "all",
        "correct_count": 1,
        "total_count": 1,
        "fixed_total_count_or_none": None,
        "point_threshold_numerator": 1,
        "point_threshold_denominator": 1,
        "wilson_threshold_numerator_or_none": None,
        "wilson_threshold_denominator_or_none": None,
        "wilson_lower_binary64_hex_or_none": None,
        "point_passed": True,
        "wilson_passed_or_none": None,
        "passed": True,
    }
    assert tuple(preimage) == tuple(
        contract["record_content_id_preimage_fields_in_exact_order"]
    )
    record_id = _cid(
        contract["record_content_id_domain_ascii"],
        contract["record_content_id_prefix"],
        preimage,
    )
    assert re.fullmatch(
        r"phase2b_actual_unsealed_960_gate_result_v1_[0-9a-f]{64}",
        record_id,
    )


def test_attempt_zero_one_retry_and_reupload_policy_is_exact() -> None:
    policy = dict(catalog.ATTEMPT_POLICY_V1)
    assert policy["allowed_attempt_index_sequences"] == ((0,), (0, 1))
    assert policy["attempt_2_exists"] is False
    assert policy["conditional_attempt_1_action"] == "REEXECUTE"
    assert policy["conditional_attempt_1_requires_any_raw_prediction_byte_produced"] is False
    assert policy["conditional_attempt_1_requires_raw_output_byte_count"] == 0
    assert policy["conditional_attempt_1_reasons_in_exact_order"] == (
        "CONTAINER_START_FAILURE",
        "HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE",
        "HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE",
        "INPUT_TRANSFER_CHECKSUM_MISMATCH",
    )
    assert policy["attempt_1_terminal_dispositions_in_exact_order"] == (
        "COMPLETE_PASS",
        "COMPLETE_FAIL",
    )
    assert policy["upload_retry_action"] == "REUPLOAD_COMMITTED_OUTPUT"
    assert policy["upload_retry_reason"] == "OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED"
    assert policy["upload_retry_requires_byte_identical_payload"] is True
    assert policy["upload_retry_creates_new_attempt"] is False


def test_custodian_ledger_first_event_null_and_exact_ten_record_chain() -> None:
    descriptor = json.loads(_raw())["body"][
        "package_schema_objects_in_exact_order"
    ][0]
    contract = descriptor["nested_record_binding_contracts_in_exact_order"][0]
    assert contract["record_name"] == "signed_stage_event_envelope"
    assert contract["event_body_record_name"] == "custodian_ledger_event_body"
    assert contract["required_record_cardinality"] == 10
    assert contract["raw_record_payloads_required"] is True
    assert contract["stage_names_in_exact_order"] == list(EXPECTED_STAGES)
    assert contract["event_signature_preimage_fields_in_exact_order"] == [
        "schema_version",
        "artifact_kind",
        "artifact_schema_version",
        "algorithm",
        "body",
        "signature_ordinal",
        "role",
        "purpose",
        "key_epoch",
        "signer_key_id",
    ]
    event_slots = contract["event_signature_slots_in_exact_stage_order"]
    assert len(event_slots) == 10
    assert [slot["purpose"] for slot in event_slots] == [
        f"SIGN_{stage}_LEDGER_EVENT" for stage in EXPECTED_STAGES
    ]
    assert all(slot["role"] == "CUSTODIAN" for slot in event_slots)
    assert len({slot["signature_domain_ascii"] for slot in event_slots}) == 10
    assert all(slot["signature_domain_ascii"].endswith("\x00") for slot in event_slots)
    first_body = {
        "schema_version": "hegel-machine-phase2b-actual-unsealed-960-custodian-ledger-event/1",
        "catalog_id": EXPECTED_CATALOG_ID,
        "campaign_id": "phase2b_actual_unsealed_960_campaign_v1_" + "1" * 64,
        "attempt_id": "phase2b_actual_unsealed_960_attempt_v1_" + "2" * 64,
        "event_index": 0,
        "stage": "PREREGISTERED",
        "predecessor_event_id_or_none": None,
        "payload_id": "phase2b_actual_unsealed_960_stage_payload_v1_" + "3" * 64,
        "payload_sha256": "sha256:" + "4" * 64,
        "signer_registry_id": "phase2b_actual_unsealed_960_signer_registry_v1_" + "5" * 64,
        "signer_key_id": "phase2b_actual_unsealed_960_signer_key_v1_" + "6" * 64,
    }
    first_signature = {
        "schema_version": "hegel-machine-phase2b-actual-unsealed-960-detached-signature/1",
        "algorithm": "Ed25519",
        "role": "CUSTODIAN",
        "purpose": "SIGN_PREREGISTERED_LEDGER_EVENT",
        "key_epoch": 1,
        "signer_key_id": first_body["signer_key_id"],
        "signature_hex": "7" * 128,
    }
    first_preimage = {
        "schema_version": "hegel-machine-phase2b-actual-unsealed-960-signed-stage-event/1",
        "body": first_body,
        "signatures_in_exact_order": [first_signature],
    }
    assert tuple(first_preimage) == tuple(
        contract["record_content_id_preimage_fields_in_exact_order"]
    )
    first_id = _cid(
        contract["record_content_id_domain_ascii"],
        contract["record_content_id_prefix"],
        first_preimage,
    )
    assert re.fullmatch(
        r"phase2b_actual_unsealed_960_signed_stage_event_v1_[0-9a-f]{64}",
        first_id,
    )
    signature_mutated_preimage = {
        **first_preimage,
        "signatures_in_exact_order": [
            {**first_signature, "signature_hex": "9" * 128}
        ],
    }
    assert _cid(
        contract["record_content_id_domain_ascii"],
        contract["record_content_id_prefix"],
        signature_mutated_preimage,
    ) != first_id
    second_body = {
        **first_body,
        "event_index": 1,
        "stage": "RAW_INPUT_COMMITTED",
        "predecessor_event_id_or_none": first_id,
    }
    second_signature = {
        **first_signature,
        "purpose": "SIGN_RAW_INPUT_COMMITTED_LEDGER_EVENT",
        "signature_hex": "8" * 128,
    }
    second_preimage = {
        **first_preimage,
        "body": second_body,
        "signatures_in_exact_order": [second_signature],
    }
    assert _cid(
        contract["record_content_id_domain_ascii"],
        contract["record_content_id_prefix"],
        second_preimage,
    ) != first_id


def test_attempt_record_nested_content_id_and_nullable_predecessor_contract() -> None:
    descriptor = json.loads(_raw())["body"][
        "package_schema_objects_in_exact_order"
    ][16]
    contract = descriptor["nested_record_binding_contracts_in_exact_order"][0]
    assert contract["record_name"] == "attempt_record"
    assert contract["record_content_id_domain_ascii"] == (
        "HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/ATTEMPT_RECORD/ID/V1\x00"
    )
    assert contract["record_content_id_prefix"] == (
        "phase2b_actual_unsealed_960_attempt_record_v1_"
    )
    assert contract["raw_record_payloads_required"] is True
    assert contract["outer_body_reference_fields_in_exact_order"] == [
        "attempt_record_ids_in_exact_order",
        "attempt_record_payload_sha256s_in_exact_order",
        "attempt_indices_in_exact_order",
    ]
    assert "attempt_record_id" not in contract[
        "record_content_id_preimage_fields_in_exact_order"
    ]
    attempt_zero_preimage = {
        "attempt_index": 0,
        "terminal_disposition": "REJECTED",
        "execution_retry_action": "REEXECUTE",
        "execution_retry_reason": "CONTAINER_START_FAILURE",
        "upload_retry_action": "NONE",
        "upload_retry_reason": "NONE",
        "any_raw_prediction_byte_produced": False,
        "raw_output_byte_count": 0,
        "predecessor_attempt_record_id_or_none": None,
    }
    assert tuple(attempt_zero_preimage) == tuple(
        contract["record_content_id_preimage_fields_in_exact_order"]
    )
    attempt_zero_id = _cid(
        contract["record_content_id_domain_ascii"],
        contract["record_content_id_prefix"],
        attempt_zero_preimage,
    )
    assert re.fullmatch(
        r"phase2b_actual_unsealed_960_attempt_record_v1_[0-9a-f]{64}",
        attempt_zero_id,
    )
    attempt_one_preimage = {
        **attempt_zero_preimage,
        "attempt_index": 1,
        "terminal_disposition": "COMPLETE_PASS",
        "execution_retry_action": "NONE",
        "execution_retry_reason": "NONE",
        "predecessor_attempt_record_id_or_none": attempt_zero_id,
    }
    attempt_one_id = _cid(
        contract["record_content_id_domain_ascii"],
        contract["record_content_id_prefix"],
        attempt_one_preimage,
    )
    assert attempt_one_id != attempt_zero_id
    rules = {
        row["rule_id"]: row for row in contract["binding_rules_in_exact_order"]
    }
    assert rules["ATTEMPT_ZERO_PREDECESSOR"]["requirement"].startswith(
        "ATTEMPT_INDEX_0_MUST_HAVE_EXACT_JSON_NULL"
    )
    assert rules["ATTEMPT_ONE_PREDECESSOR"]["requirement"].endswith(
        "ATTEMPT_RECORD_ID_OF_INDEX_0"
    )
    assert rules["ATTEMPT_ONE_REEXECUTE_ELIGIBILITY"][
        "subject_fields_in_exact_order"
    ][0].startswith("attempt_records_in_exact_order[0]")
    assert rules["OUTER_ATTEMPT_RECORD_FOUR_TUPLE_BINDING"][
        "requirement"
    ].startswith("ALL_FOUR_SEQUENCES_MUST_HAVE_EXACT_EQUAL_CARDINALITY")


@pytest.mark.parametrize(
    "bad_payload",
    ["text", bytearray(b"{}"), memoryview(b"{}"), {}, [], True],
)
def test_payload_exact_bytes_gate(bad_payload: object) -> None:
    with pytest.raises(catalog.Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1):
        catalog.verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
            bad_payload, **_pins()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "bad_payload",
    [b"{}", b"{ }", b"\xef\xbb\xbf{}", b"[]", b"null"],
)
def test_wrong_or_noncanonical_payload_rejects_atomically(bad_payload: bytes) -> None:
    with pytest.raises(catalog.Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1):
        catalog.verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
            bad_payload, **_pins()
        )


@pytest.mark.parametrize("pin_name", tuple(_pins()))
def test_every_oob_pin_is_required_exact_bytes_and_bound(pin_name: str) -> None:
    pins = _pins()
    raw = pins[pin_name]
    pins[pin_name] = raw[:-1] + bytes([raw[-1] ^ 1])
    with pytest.raises(catalog.Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1):
        catalog.verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
            _raw(), **pins
        )
    pins = _pins()
    pins[pin_name] = bytearray(pins[pin_name])  # type: ignore[assignment]
    with pytest.raises(catalog.Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1):
        catalog.verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
            _raw(), **pins
        )


def test_private_result_forge_copy_deepcopy_and_mutation_reject() -> None:
    result = catalog.verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
        _raw(), **_pins()
    )
    forged = object.__new__(
        catalog.VerifiedPhase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogV1
    )
    with pytest.raises(TypeError):
        copy.copy(result)
    with pytest.raises(TypeError):
        copy.deepcopy(result)
    with pytest.raises(catalog.Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1):
        catalog.require_verified_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
            forged
        )
    object.__setattr__(result, "c1_satisfied", True)
    with pytest.raises(catalog.Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1):
        catalog.require_verified_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1(
            result
        )


def test_extreme_definition_time_rebind_and_real_not_implemented_are_stable() -> None:
    script = r'''
import sys
sys.path.insert(0, "src")
import hegel_machine.rfc8785_v1 as rfc
import hegel_machine.phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1 as m
raw=bytes(m.CATALOG_PAYLOAD_V1)
v=m.verify_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1
req=m.require_verified_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1
real=m.verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1
error=m.Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1
kw=dict(
 out_of_band_expected_readiness_dag_id_bytes=m.INSPECTED_READINESS_DAG_ID_V1.encode(),
 out_of_band_expected_readiness_dag_payload_sha256_digest=bytes.fromhex(m.INSPECTED_READINESS_DAG_CONFIG_SHA256_V1),
 out_of_band_expected_progression_plan_payload_sha256_digest=bytes.fromhex(m.INSPECTED_PROGRESSION_PLAN_SHA256_V1),
 out_of_band_expected_gap_inventory_source_sha256_digest=bytes.fromhex(m.INSPECTED_GAP_INVENTORY_SOURCE_SHA256_V1),
 out_of_band_expected_statistical_validation_source_sha256_digest=bytes.fromhex(m.INSPECTED_STATISTICAL_VALIDATION_SOURCE_SHA256_V1),
)
real_args=(
 raw,b"signer-registry",(),b"preallocation-lineage",(),(),(),(),
 b"bundle-index",(),(),b"store-receipt",b"logical-time-token",
 b"governance-checkpoint",b"custodian-checkpoint",b"external-anchor",(),
 b"attempt-universe-head",b"denominator-finalization",(),(),
)
real_kw={
 name:b"required-oob-byte-pin"
 for name in m.FUTURE_REAL_BUNDLE_VERIFIER_RAW_ABI_V1[
  "mandatory_keyword_only_byte_parameters_in_exact_order"
 ]
}
getter=m.canonical_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_bytes_v1
def bomb(*args,**kwargs): raise RuntimeError("REBIND_REACHED")
rfc._decode_json_bytes=bomb
rfc.canonicalize_rfc8785_v1=bomb
for name in list(vars(m)):
    if name.startswith("_") or name in {"CATALOG_PAYLOAD_V1","CATALOG_ID_V1","hashlib"}:
        try: setattr(m,name,"EVIL")
        except Exception: pass
m.__dict__["__builtins__"]={"bytes":bomb,"len":bomb,"type":bomb}
r=v(raw,**kw)
assert req(r) is r and r.c1_satisfied is False and getter()==raw
try: real(*real_args,**real_kw)
except error as exc: assert str(exc).startswith("FAIL_C1_CATALOG_V1_NOT_IMPLEMENTED:")
else: raise AssertionError("real API unexpectedly returned")
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_real_bundle_entry_is_exact_not_implemented() -> None:
    positional, keyword_only = _real_stub_inputs()
    with pytest.raises(
        catalog.Phase2BActualUnsealed960AuthoritativeEvidenceSchemaCatalogErrorV1,
        match=r"^FAIL_C1_CATALOG_V1_NOT_IMPLEMENTED:",
    ):
        catalog.verify_real_phase2b_actual_unsealed_960_authoritative_evidence_bundle_v1(
            *positional, **keyword_only
        )


def test_inspected_pin_constants_match_current_frozen_bytes() -> None:
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SOURCE_SHA256
    rows = (
        (
            ROOT / "config/phase2b_actual_unsealed_960_readiness_dag_v1.json",
            catalog.INSPECTED_READINESS_DAG_CONFIG_SHA256_V1,
        ),
        (
            ROOT / "docs/Hegel_Machine_C1_C4_Authoritative_Progression_Plan_v1.md",
            catalog.INSPECTED_PROGRESSION_PLAN_SHA256_V1,
        ),
        (
            ROOT
            / "src/hegel_machine/phase2b_actual_unsealed_960_replay_admission_evidence_contract_v2.py",
            catalog.INSPECTED_GAP_INVENTORY_SOURCE_SHA256_V1,
        ),
        (
            ROOT / "src/hegel_machine/phase2b_c1_statistical_validation_v1.py",
            catalog.INSPECTED_STATISTICAL_VALIDATION_SOURCE_SHA256_V1,
        ),
    )
    for path, expected in rows:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_source_import_surface_is_producer_free_and_has_no_runtime_side_effects() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    forbidden_modules = {
        "os",
        "pathlib",
        "socket",
        "subprocess",
        "http",
        "urllib",
        "requests",
        "docker",
        "time",
        "datetime",
        "random",
        "secrets",
        "tempfile",
        "sqlite3",
        "asyncio",
        "multiprocessing",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] in forbidden_modules for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_modules
    forbidden_calls = {
        "open",
        "exec",
        "eval",
        "compile",
        "uuid4",
        "token_bytes",
        "system",
        "popen",
        "fork",
        "sign",
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called.isdisjoint(forbidden_calls)
    source_text = SOURCE.read_text(encoding="utf-8")
    assert "0_through_at_most_2" not in source_text
    assert "attempt_consumed_terminal" not in source_text
    assert "MISSING_UNSUPPLIED_UNVERIFIED_NO_VERIFIER" not in source_text
    assert "decode_and_require_canonical_rfc8785_v1" not in source_text
    assert "content_id_preimage" in source_text
    assert "signature_preimage" in source_text
    assert "bundle_index_schema_object" in source_text
