from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hegel_machine import phase3_q1_formal_archive_contract_v1 as formal
from hegel_machine import phase3_q05b_actual_admission_v1 as admission
from hegel_machine import phase3_q1_qualification_wire_v1 as wire
from hegel_machine.phase3_q1_universe_v1 import production_universe_v1
from hegel_machine.strict_cbor_v1 import (
    canonical_cbor_decode,
    canonical_cbor_encode,
    rfc6962_root,
)


def _roots(start: int, count: int) -> tuple[bytes, ...]:
    return tuple(bytes((start + index,)) * 32 for index in range(count))


def _candidate(
    leaf: wire.Q05BFullLeafManifestV1,
) -> wire.Q05BQualificationCandidateReceiptV1:
    predicates = tuple(
        (predicate_id, name, True, bytes((0x40 + predicate_id,)) * 32)
        for predicate_id, name in wire.QUALIFICATION_PREDICATE_REGISTRY[:19]
    )
    semantic, projection = wire.q1_semantic_and_projection_roots_v1(leaf)
    return wire.Q05BQualificationCandidateReceiptV1(
        source_commit=bytes.fromhex("12" * 20),
        q1_semantic_binding_root=semantic,
        q1_projection_profile_root=projection,
        q0_receipt_root=wire.Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION,
        full_leaf_manifest_root=leaf.manifest_root,
        implementation_roots=_roots(0x23, 3),
        neutral_manifest_roots=(bytes.fromhex("31" * 32),) * 3,
        bounded_state_roots=_roots(0x32, 2),
        bundle_evidence_root=bytes.fromhex("34" * 32),
        isolation_evidence_root=bytes.fromhex("35" * 32),
        resource_evidence_root=bytes.fromhex("36" * 32),
        pre_receipt_evidence_root=wire.pre_receipt_evidence_root_v1(
            bytes.fromhex("12" * 20), predicates
        ),
        predicate_rows_1_through_19=predicates,
    )


def test_qualification_tag_registry_is_separate_exact_and_source_local() -> None:
    tags = [row[0] for row in wire.Q05B_QUALIFICATION_TAG_REGISTRY]
    assert tags == list(range(0x3A00, 0x3A08))
    assert not set(tags) & set(range(0x3700, 0x370D))
    assert [row[0] for row in formal.Q1_TAG_REGISTRY] == list(
        range(0x3700, 0x370D)
    )
    for path in (ROOT / "src/hegel_machine").glob("*.py"):
        if path.name in {
            "phase3_q05b_wire_qualification_contract_v1.py",
            "phase3_q1_qualification_wire_v1.py",
        }:
            continue
        source = path.read_text(encoding="utf-8")
        assert not any(f"0x{tag:04X}" in source for tag in tags)


def test_full_810_leaf_root_is_only_rfc6962_ordered_row_identity() -> None:
    manifest = wire.full_v16_leaf_manifest_v1()
    assert len(manifest.rows) == 810
    row_objects = tuple(row.canonical_object() for row in manifest.rows)
    assert manifest.manifest_root == rfc6962_root(row_objects)
    assert manifest.manifest_root.hex() == (
        "3fefacd3db59294f2b6d44a5d0b813e73af3ec84742a24ab846bbdacae6c1f1b"
    )
    decoded = canonical_cbor_decode(manifest.canonical_bytes)
    assert decoded[6] == manifest.manifest_root
    assert decoded[7] == row_objects
    assert wire.SEMANTIC_SOURCE_ROOTS not in decoded
    assert wire.Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION not in decoded
    assert (
        wire.decode_full_v16_leaf_manifest_v1(manifest.canonical_bytes).manifest_root
        == manifest.manifest_root
    )
    tampered = list(decoded)
    rows = list(tampered[7])
    rows[0], rows[1] = rows[1], rows[0]
    tampered[7] = tuple(rows)
    tampered[6] = rfc6962_root(tampered[7])
    with pytest.raises(wire.Q05BWireQualificationError) as error:
        wire.decode_full_v16_leaf_manifest_v1(canonical_cbor_encode(tuple(tampered)))
    assert error.value.code in {"REJECT_Q05B_LEAF_ORDER", "REJECT_Q05B_LEAF_MANIFEST"}


def test_formal_0x3700_semantic_binding_and_projection_profile_are_exact_inputs() -> None:
    leaf = wire.full_v16_leaf_manifest_v1()
    manifest = wire.q1_semantic_binding_manifest_v1(leaf)
    assert manifest.canonical_object()[1] == 0x3700
    assert manifest.full_v16_leaf_manifest_root == leaf.manifest_root
    assert manifest.q0_receipt_root == (
        wire.Q0_SATURATION_RECEIPT_ROOT_FROM_Q1_PREREGISTRATION
    )
    semantic, projection = wire.q1_semantic_and_projection_roots_v1(leaf)
    assert semantic == manifest.manifest_root
    assert semantic.hex() == (
        "e3b3df3e81b7632c7c713ef5ec84913f990ad8232a25b851f20c46ac7416bfcb"
    )
    assert projection.hex() == (
        "aa441cdc49ab60324483b9aa44e9fdfc324a6ad49a6bff50af6daa775209816d"
    )
    assert wire.Q1_NULL_OUTPUT_SLOTS == tuple(
        (index, name, None)
        for index, name in enumerate(formal.Q1_OUTPUT_SLOT_NAMES, start=1)
    )


@pytest.mark.parametrize(
    ("signature_id", "counts"),
    (
        (1, (1048, 1048, 22, 40, 86, 110, 59)),
        (2, (1101, 1101, 26, 28, 112, 144, 84)),
    ),
)
def test_bounded_node3_state_is_not_formal_fixed_point(
    signature_id: int,
    counts: tuple[int, ...],
) -> None:
    universe_rows = (480, 85)[signature_id - 1]
    state = wire.Q05BBoundedNode3StateV1(
        signature_id,
        production_universe_v1(signature_id).universe_root,
        universe_rows,
        b"LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED",
        counts,
        maximum_bank_points_per_class=8,
        maximum_frontier_points_per_class=4,
        peak_work_queue_points=810,
        peak_saturation_round_count=5,
        coverage_record_root=bytes.fromhex("50" * 32),
        partition_evidence_root=bytes.fromhex("51" * 32),
    )
    value = state.canonical_object()
    assert value[7:10] == (3, 3, 4)
    assert value[11:16] == (True, True, True, True, True)
    assert value[23:25] == (False, None)
    assert 0x3707 not in value
    assert len(state.state_root) == 32
    with pytest.raises(wire.Q05BWireQualificationError):
        replace(state, primary_counts=counts[:-1] + (counts[-1] + 1,))


def test_two_stage_receipt_has_19_then_20_evidence_rows_and_q1_stays_closed() -> None:
    leaf = wire.full_v16_leaf_manifest_v1()
    candidate = _candidate(leaf)
    value = candidate.canonical_object()
    assert len(value[21]) == 19
    assert all(
        row == (expected[0], expected[1], True, row[3])
        for row, expected in zip(
            value[21], wire.QUALIFICATION_PREDICATE_REGISTRY[:19], strict=True
        )
    )
    assert value[22:24] == (19, 0x7FFFF)
    assert candidate.pre_receipt_root != candidate.receipt_root
    replay = wire.decode_qualification_candidate_receipt_v1(
        candidate.canonical_bytes
    )
    assert replay.receipt_root == candidate.receipt_root

    final = wire.Q05BQualificationReceiptV1(candidate)
    final_value = final.canonical_object()
    assert len(final_value[7]) == 20
    assert final_value[7][-1] == (
        20,
        wire.QUALIFICATION_PREDICATE_REGISTRY[19][1],
        True,
        wire.predicate20_evidence_root_v1(candidate.receipt_root),
    )
    assert final_value[8:11] == (20, 0xFFFFF, True)
    assert wire.decode_qualification_receipt_v1(final.canonical_bytes).receipt_root == (
        final.receipt_root
    )

    authority = final_value[11]
    assert authority[:4] == (0, 0, 0, 20)
    assert authority[4] == 8
    assert all(row[2] is None for row in authority[5])
    assert authority[6:] == (
        None,
        0,
        None,
        False,
        None,
        False,
        False,
        False,
        False,
        False,
    )

    rows = list(candidate.predicate_rows_1_through_19)
    rows[0] = (1, rows[0][1], False, rows[0][3])
    with pytest.raises(wire.Q05BWireQualificationError) as error:
        replace(candidate, predicate_rows_1_through_19=tuple(rows))
    assert error.value.code == "REJECT_Q05B_PREDICATE"

    bool_id_rows = list(candidate.predicate_rows_1_through_19)
    bool_id_rows[0] = (True, bool_id_rows[0][1], True, bool_id_rows[0][3])
    with pytest.raises(wire.Q05BWireQualificationError) as bool_id:
        replace(candidate, predicate_rows_1_through_19=tuple(bool_id_rows))
    assert bool_id.value.code == "REJECT_Q05B_PREDICATE"

    int_pass_rows = list(candidate.predicate_rows_1_through_19)
    int_pass_rows[0] = (1, int_pass_rows[0][1], 1, int_pass_rows[0][3])
    with pytest.raises(wire.Q05BWireQualificationError) as int_pass:
        replace(candidate, predicate_rows_1_through_19=tuple(int_pass_rows))
    assert int_pass.value.code == "REJECT_Q05B_PREDICATE"

    final_tamper = list(final_value)
    final_rows = list(final_tamper[7])
    final_rows[-1] = (
        final_rows[-1][0],
        final_rows[-1][1],
        1,
        final_rows[-1][3],
    )
    final_tamper[7] = tuple(final_rows)
    with pytest.raises(wire.Q05BWireQualificationError) as final_bool_alias:
        wire.decode_qualification_receipt_v1(
            canonical_cbor_encode(tuple(final_tamper))
        )
    assert final_bool_alias.value.code == "REJECT_Q05B_PREDICATE"


def test_exact_chunk_bstr_boundary_is_frozen_without_large_allocation() -> None:
    accepted = wire.MAX_ACCEPTED_RAW_CBOR_BSTR_PAYLOAD_BYTES
    assert accepted == 16_777_207
    assert wire.cbor_bstr_encoded_length_v1(accepted) == 16_777_212
    assert wire.framed_bstr_record_length_v1(accepted) == 16_777_216
    assert wire.bstr_record_fits_frozen_chunk_v1(accepted) is True
    assert wire.framed_bstr_record_length_v1(accepted + 1) == 16_777_217
    assert wire.bstr_record_fits_frozen_chunk_v1(accepted + 1) is False


def test_actor_stdout_is_one_line_actor_specific_and_authority_closed() -> None:
    value = {
        "action_id": wire.ACTOR_ACTION_ID,
        "actor_id": "PYTHON_ENDPOINT",
        "file_count": 5,
        "implementation_id": "HEGEL_Q1_BOUNDED_NODE3_PROJECTION_PYTHON_V1",
        "neutral_manifest_length": 100,
        "neutral_manifest_raw_sha256": "11" * 32,
        "neutral_manifest_relative_path": "neutral/q05b-node3-golden-manifest-v1.cbor",
        "neutral_manifest_root": "12" * 32,
        "q1_formal_roots": None,
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_output_slots": [None] * 8,
        "q1_state": "NOT_RUN",
        "runtime_identity_sha256": "13" * 32,
        "schema_version": wire.ACTOR_ENVELOPE_SCHEMA_VERSION,
        "sidecar_manifest_length": 80,
        "sidecar_manifest_raw_sha256": "14" * 32,
        "sidecar_manifest_relative_path": "neutral/q05b-node3-sidecar-manifest-v1.cbor",
        "sidecar_manifest_root": "15" * 32,
        "source_identity_sha256": "16" * 32,
        "status": wire.ACTOR_CANDIDATE_STATUS,
    }
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    assert wire.validate_actor_stdout_envelope_v1(payload) == value
    aliased = dict(value, q1_gate_count=False)
    alias_payload = (
        json.dumps(aliased, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    with pytest.raises(wire.Q05BWireQualificationError):
        wire.validate_actor_stdout_envelope_v1(alias_payload)
    with pytest.raises(wire.Q05BWireQualificationError):
        wire.validate_actor_stdout_envelope_v1(payload.rstrip(b"\n"))


def test_actual_implementation_has_no_blockers_and_all_evidence_remains_pending(
) -> None:
    assert wire.QUALIFICATION_ENGINEERING_STATUS == (
        "ACTUAL_IMPLEMENTED_CONDITIONALLY_ADMITTED_NOT_EXECUTED"
    )
    assert wire.IMPLEMENTATION_BLOCKED_PREDICATE_IDS == ()
    assert wire.PENDING_ACTUAL_EVIDENCE_PREDICATE_IDS == tuple(range(1, 21))
    assert wire.PREDICATE14_SOURCE_CAPABILITY_FROZEN is True
    names = {row[0]: row[1] for row in wire.QUALIFICATION_PREDICATE_REGISTRY}
    assert b"ASSEMBLER" in names[12]
    assert b"CHUNK" in names[13]
    assert b"COUNTING_DISCARD" in names[14]
    assert b"ISOLATION" in names[19]
    assert (
        b"hegel-q05b-counting-discard-record-stream/1",
        15,
        wire.COUNTING_DISCARD_SCHEMA_FIELDS,
        wire.COUNTING_DISCARD_EQUALITY_RULES,
        True,
    ) in wire.qualification_wire_profile_object_v1()
    assert len(wire.QUALIFICATION_PREDICATE_REGISTRY_ROOT) == 32
    assert len(wire.QUALIFICATION_TAG_REGISTRY_ROOT) == 32
    assert len(wire.qualification_wire_profile_root_v1()) == 32


def _strict_json_object(pairs):
    value = {}
    for key, item in pairs:
        assert key not in value
        value[key] = item
    return value


def _assert_type_exact(value: object, expected: object) -> None:
    assert type(value) is type(expected)
    if type(expected) is dict:
        assert type(value) is dict
        assert set(value) == set(expected)
        for key in expected:
            _assert_type_exact(value[key], expected[key])
    elif type(expected) is list:
        assert type(value) is list
        assert len(value) == len(expected)
        for item, expected_item in zip(value, expected, strict=True):
            _assert_type_exact(item, expected_item)
    else:
        assert value == expected


def test_machine_config_replays_contract_roots_and_keeps_q1_zero_null_not_run() -> None:
    path = ROOT / "config/phase3_q05b_node3_dual_projection_qualification_v1.json"
    config = json.loads(
        path.read_bytes(),
        object_pairs_hook=_strict_json_object,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )
    assert config["engineering_status"] == (
        "ACTUAL_IMPLEMENTED_CONDITIONALLY_ADMITTED_NOT_EXECUTED"
    )
    authority = config["authority"]
    assert authority["qualification_state"] == config["engineering_status"]
    assert authority["qualification_predicate_count"] == 0
    assert authority["qualification_predicate_mask"] == 0
    assert authority["qualification_predicate_total"] == 20
    assert authority["qualification_candidate_receipt"] is None
    assert authority["qualification_final_receipt"] is None
    assert authority["q1_state"] == "NOT_RUN"
    assert authority["q1_gate_count"] == 0
    assert authority["q1_gate_mask"] == 0
    assert authority["q1_gate_total"] == 20
    assert authority["q1_formal_output_roots"] == [None] * 8
    assert authority["q1_receipt"] is None
    assert authority["q2_state"] == "NOT_RUN"
    assert authority["m3_formal_roots"] is None
    assert authority["outside_certificate_issued"] is False
    assert authority["active_transition_allowed"] is False

    config_tags = tuple(
        (row[0], row[1].encode("ascii"))
        for row in config["qualification_numeric_tag_registry"]
    )
    assert config_tags == wire.Q05B_QUALIFICATION_TAG_REGISTRY
    roots = config["registry_and_profile_roots"]
    assert roots["qualification_tag_registry_root_hex"] == (
        wire.QUALIFICATION_TAG_REGISTRY_ROOT.hex()
    )
    assert roots["qualification_predicate_registry_root_hex"] == (
        wire.QUALIFICATION_PREDICATE_REGISTRY_ROOT.hex()
    )
    assert roots["qualification_wire_profile_root_hex"] == (
        wire.qualification_wire_profile_root_v1().hex()
    )
    leaf = wire.full_v16_leaf_manifest_v1()
    semantic, projection = wire.q1_semantic_and_projection_roots_v1(leaf)
    assert config["full_v16_leaf_manifest"]["root_hex"] == leaf.manifest_root.hex()
    assert config["full_v16_leaf_manifest"]["sidecar_canonical_cbor_bytes"] == len(
        leaf.canonical_bytes
    )
    assert config["q1_formal_input_roots"]["q1_semantic_binding_root_hex"] == (
        semantic.hex()
    )
    assert config["q1_formal_input_roots"]["q1_projection_profile_root_hex"] == (
        projection.hex()
    )
    assert config["external_sort_trace_wire"] == {
        "schema_id_ascii": "hegel-q1-external-sort-trace/1",
        "field_count": 6,
        "field_order": [
            "version",
            "schema_id",
            "projection_object",
            "ordered_rows",
            "run_manifests",
            "scratch_events",
        ],
    }
    assert config["counting_discard_wire"] == {
        "schema_id_ascii": "hegel-q05b-counting-discard-record-stream/1",
        "field_count": 15,
        "field_order": [item.decode("ascii") for item in wire.COUNTING_DISCARD_SCHEMA_FIELDS],
        "equality_rules": [
            item.decode("ascii") for item in wire.COUNTING_DISCARD_EQUALITY_RULES
        ],
        "predicate14_source_capability_frozen": True,
        "predicate14_actual_qualification_passed": False,
    }
    assert config["sidecar_layout"]["file_mode_decimal"] == wire.OUTPUT_FILE_MODE
    assert config["sidecar_layout"]["output_file_count"] == 5
    actual_preconditions = config["actual_preconditions"]
    dual_actual_preconditions = json.loads(
        (ROOT / "config/phase3_q05b_dual_isolation_v1.json").read_bytes(),
        object_pairs_hook=_strict_json_object,
        parse_constant=lambda token: (_ for _ in ()).throw(AssertionError(token)),
    )["actual_preconditions"]
    for expected in (
        dual_actual_preconditions,
        wire.COMMIT_A_ACTUAL_PRECONDITIONS_V1,
        admission.COMMIT_A_ACTUAL_PRECONDITIONS_V1,
    ):
        _assert_type_exact(actual_preconditions, expected)
    assert actual_preconditions["actual_entrypoint_implemented"] is True
    assert actual_preconditions["implementation_blocked_predicate_ids"] == []
    assert actual_preconditions["pending_actual_evidence_predicate_ids"] == list(
        range(1, 21)
    )
    assert actual_preconditions["source_freeze_execution_status"] == (
        "NOT_EXECUTED_AT_COMMIT_A"
    )
    for key in (
        "attempt_unique_docker_execution_authority_required",
        "initial_and_precreate_name_absence_required",
        "docker_cleanup_owned_cid_only_required",
        "foreign_or_unknown_docker_state_zero_mutation_required",
    ):
        assert actual_preconditions[key] is True
    assert tuple(item.encode("ascii") for item in config["failure_code_registry"]) == (
        wire.FAILURE_CODE_REGISTRY
    )


def test_bounded_state_root_goldens_replay_from_config_rows() -> None:
    config = json.loads(
        (ROOT / "config/phase3_q05b_node3_dual_projection_qualification_v1.json").read_bytes()
    )["bounded_node3_contract"]
    primary = {row[0]: tuple(row[2:]) for row in config["primary_count_rows"]}
    resources = {row[0]: tuple(row[1:]) for row in config[
        "resource_rows_max_bank_max_frontier_peak_work_peak_round"
    ]}
    coverage = {row[0]: bytes.fromhex(row[1]) for row in config["coverage_record_roots"]}
    evidence = {row[0]: bytes.fromhex(row[1]) for row in config["partition_evidence_roots"]}
    expected = {row[0]: row[1] for row in config["bounded_state_roots"]}
    assert {key: value.hex() for key, value in evidence.items()} == {
        1: "99357fc3a5f48e8a63e6a87f4b182153c5cdae52bd911676f7b2ecc1058aa097",
        2: "51d017cd9d7e452198d9d12c53e16728c1e220e56d47f43ce3954c4e92c9ef67",
    }
    assert expected == {
        1: "a7460841bcd36797fa9d5d9987fafe5b5efd91f96e4e49b73a78c6406a20db37",
        2: "1788df25b4cd6b8830db28d8622e2fe146f3a3c454404e5e7eafe51315acab8f",
    }
    for signature_id in (1, 2):
        maximum_bank, maximum_frontier, peak_work, peak_round = resources[
            signature_id
        ]
        state = wire.Q05BBoundedNode3StateV1(
            signature_id,
            production_universe_v1(signature_id).universe_root,
            len(production_universe_v1(signature_id).rows),
            b"LOCAL_PROTOTYPE_SUBSET_TRAVERSAL_CLOSED",
            primary[signature_id],
            maximum_bank,
            maximum_frontier,
            peak_work,
            peak_round,
            coverage[signature_id],
            evidence[signature_id],
        )
        assert state.state_root.hex() == expected[signature_id]


def test_engineering_document_states_two_commit_and_non_q1_boundaries() -> None:
    document = (
        ROOT
        / "docs/Hegel_Machine_Phase3A_Q05b_Node3_Dual_Projection_Qualification_Engineering_v1.md"
    ).read_text(encoding="utf-8")
    required = (
        "ACTUAL_IMPLEMENTED_CONDITIONALLY_ADMITTED_NOT_EXECUTED",
        "0/20 / NOT_RUN / eight null output roots",
        "0x3A07",
        "RFC6962_ROOT(ordered 810",
        "hegel-q1-external-sort-trace/1",
        "hegel-q05b-counting-discard-record-stream/1",
        "bd85abed6feb4b4e9fd6102f43c5db3bbaf9733f0ec42ab5b5363e14a86d350e",
        "cbc22f6a9dc91589f77aa1564eb40d688c45ee3aa6af5a66d777ffe08a086b15",
        "16,777,207",
        "0x7FFFF",
        "0xFFFFF",
        "attempt_unique_docker_execution_authority_required",
        "initial_and_precreate_name_absence_required",
        "docker_cleanup_owned_cid_only_required",
        "foreign_or_unknown_docker_state_zero_mutation_required",
        "actual evidence predicates `1..20` pending",
        "implementation-blocker registry is empty",
        "evidence for predicates `1..20` remains pending",
        "Predicate 14 capability in source",
        "Predicates 12, 15 and 18",
        "SOURCE_FREEZE",
        "ACTUAL_QUALIFICATION",
    )
    assert all(item in document for item in required)
