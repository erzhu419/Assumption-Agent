from __future__ import annotations

import json
import importlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG_PATH = ROOT / "config/phase3_q1_archive_projection_freeze_v1.json"
DOC_PATH = (
    ROOT
    / "docs/Hegel_Machine_Phase3A_Q05a_Q1_Archive_Projection_Engineering_Freeze_v1.md"
)

capacity = importlib.import_module("hegel_machine.phase3_q1_capacity_preflight_v1")
contract = importlib.import_module("hegel_machine.phase3_q1_formal_archive_contract_v1")
external_sort = importlib.import_module(
    "hegel_machine.phase3_q1_external_sort_profile_v1"
)
projection = importlib.import_module("hegel_machine.phase3_q1_archive_projection_v1")
quotient = importlib.import_module("hegel_machine.phase3_q1_quotient_contract_v1")
strict_cbor = importlib.import_module("hegel_machine.strict_cbor_v1")


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key}")
        output[key] = value
    return output


def _config() -> dict[str, object]:
    value = json.loads(
        CONFIG_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    assert type(value) is dict
    return value


def test_config_is_strict_unique_json() -> None:
    config = _config()
    assert config["schema_version"] == (
        "hegel-phase3a-q05a-q1-archive-projection-freeze/1"
    )
    with pytest.raises(ValueError):
        json.loads('{"a":1,"a":2}', object_pairs_hook=_unique_object)
    with pytest.raises(ValueError):
        json.loads('{"a":NaN}', parse_constant=_reject_constant)


def test_tag_schema_and_domain_registries_match_code() -> None:
    config = _config()
    tag_rows = config["numeric_tag_registry"]
    assert type(tag_rows) is list
    assert [row[0] for row in tag_rows] == list(range(0x3700, 0x370D))
    assert [row[2] for row in tag_rows] == [f"0x{value:04X}" for value in range(0x3700, 0x370D)]
    assert [row[0] for row in tag_rows] == [row[0] for row in contract.Q1_TAG_REGISTRY]
    assert tuple(
        (row[0], row[1].encode("ascii"))
        for row in config["projection_profile_tag_registry_wire"]
    ) == contract.Q1_TAG_REGISTRY

    formal_domains = [
        contract.SEMANTIC_BINDING_ROOT_DOMAIN,
        contract.BEHAVIOR_ID_DOMAIN,
        contract.CONSTRUCTION_SIGNATURE_ID_DOMAIN,
        contract.PROGRAM_ID_DOMAIN,
        contract.PROGRAM_RECORD_ID_DOMAIN,
        contract.COHORT_ID_DOMAIN,
        contract.COHORT_RECORD_ID_DOMAIN,
        contract.CLASS_RECORD_ID_DOMAIN,
        contract.APPLICATION_ID_DOMAIN,
        contract.COVERAGE_RECORD_ID_DOMAIN,
        contract.FIXED_POINT_ROOT_DOMAIN,
        contract.FRAMED_BLOB_HASH_DOMAIN,
        contract.CHUNK_MANIFEST_RECORD_ID_DOMAIN,
        contract.SIGNATURE_SATURATION_STATE_ROOT_DOMAIN,
        contract.SIGNATURE_MANIFEST_ROOT_DOMAIN,
        contract.CLOSURE_BUNDLE_ROOT_DOMAIN,
        contract.PROJECTION_PROFILE_ROOT_DOMAIN,
    ]
    assert config["formal_archive_hash_domain_registry"] == formal_domains
    assert config["projection_diagnostic_hash_domain_registry"] == [
        external_sort.SORTED_STREAM_ROOT_DOMAIN,
        external_sort.SCRATCH_LEDGER_ROOT_DOMAIN,
        external_sort.EXTERNAL_SORT_PROJECTION_ROOT_DOMAIN,
        projection.PROJECTED_STREAM_ROOT_DOMAIN,
        projection.SNAPSHOT_RECORD_SET_ROOT_DOMAIN,
        contract.PARTITION_STREAM_COMMITMENT_DOMAIN,
        contract.PARTITION_EXTERNAL_SORT_ROOT_DOMAIN,
        contract.PROJECTION_RESULT_ROOT_DOMAIN,
    ]
    identity = config["identity_contract"]
    assert identity["locally_covered_collision_domains"] == [
        contract.BEHAVIOR_ID_DOMAIN,
        contract.CONSTRUCTION_SIGNATURE_ID_DOMAIN,
        contract.PROGRAM_ID_DOMAIN,
        contract.PROGRAM_RECORD_ID_DOMAIN,
        contract.COHORT_ID_DOMAIN,
        contract.COHORT_RECORD_ID_DOMAIN,
        contract.CLASS_RECORD_ID_DOMAIN,
        contract.APPLICATION_ID_DOMAIN,
        contract.COVERAGE_RECORD_ID_DOMAIN,
        contract.FRAMED_BLOB_HASH_DOMAIN,
        contract.CHUNK_MANIFEST_RECORD_ID_DOMAIN,
    ]
    assert identity["locally_covered_collision_digest_roles"] == [
        "STRICT_CANONICAL_AST_SHA256",
        "RFC6962_ELIGIBLE_APPLICATION_SET",
        "RFC6962_PROCESSED_APPLICATION_SET",
        "RFC6962_STRICT_ADMISSION_SET",
        "RFC6962_CHUNK_RECORD_SUBTREE",
        "RFC6962_FULL_STREAM_ARCHIVE",
        "RFC6962_CHUNK_MANIFEST_SUBTREE",
        "RFC6962_CLASS_COHORT_SUBTREE",
        "RFC6962_VISIBLE_FRONTIER_SUBTREE",
    ]
    assert identity["collision_scope"] == {
        "registered_formal_semantic_and_content_ids_with_locally_held_preimages": True,
        "framed_blob_and_chunk_manifest_ids_with_locally_held_preimages": True,
        "locally_materialized_rfc6962_preimages": True,
        "implementation_local_raw_sha_diagnostics": False,
        "strict_partition_manifest_bundle_assembler_pending": True,
        "strict_partition_manifest_bundle_assembler_collision_scope_covered": False,
    }


def test_profile_preimage_is_42_fields_and_matches_frozen_constants() -> None:
    config = _config()
    coverage_root = strict_cbor.rfc6962_root(contract.expected_coverage_registry_v1())
    profile = contract.projection_profile_object_v1(
        semantic_binding_root=b"\x11" * 32,
        coverage_registry_root=coverage_root,
        resource_guard_registry=contract.Q1_RESOURCE_GUARD_REGISTRY,
    )
    assert len(profile) == 42
    assert profile[3].decode("ascii") == config["archive_wire_version"]
    assert profile[4].decode("ascii") == config["projection_freeze_version"]
    assert profile[5].decode("ascii") == config["projection_profile_id"]
    assert profile[7] == tuple(
        (row[0], row[1].encode("ascii"))
        for row in config["projection_profile_tag_registry_wire"]
    )
    assert profile[9].hex() == config["coverage_contract"]["coverage_registry_root_hex"]
    assert profile[11:15] == (
        config["framing_and_chunking"]["maximum_records_per_chunk"],
        config["framing_and_chunking"]["maximum_chunk_framed_bytes"],
        config["framing_and_chunking"]["frame_length_prefix_bytes"],
        config["projection_profile_bound_constants"]["compression_id"],
    )
    assert profile[37] == contract.Q1_RESOURCE_GUARD_REGISTRY
    assert profile[38] == contract.Q1_OUTPUT_SLOT_NAMES
    framed_golden = config["framing_and_chunking"]["chunk_blob_hash_golden"]
    assert contract.framed_blob_hash_v1(
        bytes.fromhex(framed_golden["framed_blob_hex"])
    ).hex() == framed_golden["hash_hex"]


def test_coverage_leaf_order_and_846_registry_are_live_replayed() -> None:
    config = _config()
    coverage = config["coverage_contract"]
    leaves = capacity._frozen_leaf_asts_v1(raw_cap=capacity.LEAF_COUNT)
    keys = tuple(
        (
            capacity.OUTPUT_SORT_IDS[row.metrics.output_sort],
            row.root_operator_id,
            row.cbor_bytes,
        )
        for row in leaves
    )
    assert len(leaves) == coverage["full_v16_leaf_count"] == 810
    assert keys == tuple(sorted(keys))
    assert coverage["leaf_manifest_order"] == [
        "output_sort_id",
        "root_operator_id",
        "canonical_ast_cbor",
    ]
    assert coverage["leaf_application_key_contract"] == {
        "construction_depth": 0,
        "coverage_code": "EXACT_FULL_V16_LEAF_MANIFEST_INDEX_0_THROUGH_809",
        "operator_parameters": [],
        "ordered_child_program_ids": [],
    }
    registry = contract.expected_coverage_registry_v1()
    assert len(registry) == coverage["coverage_record_count_per_signature"] == 846
    assert strict_cbor.rfc6962_root(registry).hex() == coverage["coverage_registry_root_hex"]


def test_resource_stream_and_output_slot_registries_are_exact_typed_rows() -> None:
    config = _config()
    guards = config["resource_guard_registry"]
    assert all(
        type(row) is list
        and len(row) == 2
        and type(row[0]) is int
        and type(row[1]) is str
        for row in guards
    )
    assert tuple((row[0], row[1].encode("ascii")) for row in guards) == (
        contract.Q1_RESOURCE_GUARD_REGISTRY
    )
    streams = config["stream_kind_registry"]
    assert [row[0] for row in streams] == [int(value) for value in contract.ArchiveStreamKindId]
    assert all(type(row[2]) is bool and row[2] is True for row in streams)
    slots = config["ordered_q1_output_slots"]
    assert tuple(row[0] for row in slots) == tuple(range(1, 9))
    assert tuple(row[1].encode("ascii") for row in slots) == contract.Q1_OUTPUT_SLOT_NAMES
    assert all(row[2] is None for row in slots)
    assert tuple((row[0], row[1]) for row in config["output_sort_id_registry"]) == tuple(
        (int(value), name) for name, value in quotient.OutputSortId.__members__.items()
    )
    assert tuple(config["normalization_profile_id_registry"]) == tuple(
        [int(value), name]
        for name, value in quotient.NormalizationProfileId.__members__.items()
    )
    assert config["q1_state_id_registry"] == [[0, "NOT_RUN"]]
    assert config["q2_state_id_registry"] == [[0, "NOT_RUN"]]


def test_partition_and_result_wire_constraints_match_contract() -> None:
    config = _config()
    result_schema = config["schema_registry"]["Q1ArchiveProjectionResultV1"]
    fixed_schema = config["schema_registry"]["Q1FixedPointRecordV1"]
    assert len(result_schema["field_order"]) == 21
    assert result_schema["partition_row_length"] == 24
    assert len(result_schema["partition_row_field_order"]) == 24
    assert result_schema["ordered_stream_diagnostic_commitments_source"].startswith(
        "Q1ProjectedStreamV1.diagnostic_commitment"
    )
    assert result_schema["ordered_external_sort_stream_roots_source"].startswith(
        "Q1ExternalSortProjectionV1.diagnostic_root"
    )
    assert "strict_admitted_count <= raw_application_count" in fixed_schema[
        "cardinality_invariants"
    ]
    assert "program_count == bank_point_count" in fixed_schema["cardinality_invariants"]
    assert "program_record_count == bank_point_count" in result_schema[
        "cardinality_invariants"
    ]


def test_gate10_and_all_downstream_authority_remain_zero_null_not_run() -> None:
    config = _config()
    authority = config["authority"]
    assert authority == {
        "q1_state": "NOT_RUN",
        "q1_execution_started": False,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_gate_total": 20,
        "q1_formal_roots": None,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "role_evaluation_performed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "active_transition_allowed": False,
    }
    gate = config["gate10_qualification"]
    predicates = gate["predicates"]
    assert gate["predicate_count"] == 20
    assert gate["passed_predicate_count"] == gate["predicate_mask"] == 0
    assert len(predicates) == 20
    assert all(
        type(row) is list
        and len(row) == 3
        and type(row[0]) is int
        and type(row[1]) is str
        and type(row[2]) is bool
        and row[2] is False
        for row in predicates
    )
    assert [row[0] for row in predicates] == list(range(1, 21))
    assert gate["gate10_evidence_root"] is None
    source = config["source_freeze_requirements"]
    assert source["projection_source_commit"] is None
    assert source["local_python_projection_prototype_present"] is True
    assert source["local_python_prototype_roots_are_formal_or_q1_outputs"] is False
    assert source["qualified_isolated_python_source_available"] is False
    assert source["rust_projection_source_available"] is False
    assert source["host_replay_source_available"] is False
    assert source["dual_supervisor_source_available"] is False
    assert config["claim_boundary"]["full_node6_capacity_preflight_allowed_now"] is False
    assert config["claim_boundary"]["q1_complete_claim_allowed"] is False


def test_document_records_exact_claim_boundary_and_isolation_timing() -> None:
    config = _config()
    document = DOC_PATH.read_text(encoding="utf-8")
    for literal in (
        config["archive_wire_version"],
        config["projection_freeze_version"],
        config["projection_profile_id"],
        config["coverage_contract"]["coverage_registry_root_hex"],
        "It has exactly 24 fields",
        "The complete result has 21 fields",
        "Q1 remains `NOT_RUN`",
        "node-six capacity preflight remains forbidden",
    ):
        assert literal in document
    assert config["isolation_profile"]["wall_time_interval"] == (
        "BEFORE_DOCKER_CREATE_THROUGH_CONTAINER_WAIT_EXIT_INCLUDING_STARTUP_IMPORT_AND_PROJECTION"
    )
