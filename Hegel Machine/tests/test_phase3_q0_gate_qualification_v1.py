from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from hegel_machine import phase3_q0_gate_qualification_v1 as gates
from hegel_machine import phase3_q0_quotient_contract_v1 as contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "src/hegel_machine/phase3_q0_gate_qualification_v1.py"
)
EXPECTED_SOURCE_PATHS = [
    "config/phase3_q0_quotient_freeze_v1.json",
    "src/hegel_machine/phase3_m3_bounded_enumerator_v1.py",
    "src/hegel_machine/phase3_m3_dsl_core_v1.py",
    "src/hegel_machine/phase3_m3_record_wire_v1.py",
    "src/hegel_machine/phase3_m3_shrink1_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink2_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink3_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink4_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink5_core_v1.py",
    "src/hegel_machine/phase3_m3_shrink6_core_v1.py",
    "src/hegel_machine/phase3_q0_evaluator_v1.py",
    "src/hegel_machine/phase3_q0_gate_qualification_v1.py",
    "src/hegel_machine/phase3_q0_input_adapter_v1.py",
    "src/hegel_machine/phase3_q0_quotient_contract_v1.py",
    "src/hegel_machine/phase3_q0_quotient_oracle_v1.py",
    "src/hegel_machine/strict_ast_shrink1_v1.py",
    "src/hegel_machine/strict_ast_shrink2_v1.py",
    "src/hegel_machine/strict_ast_shrink3_v1.py",
    "src/hegel_machine/strict_ast_shrink4_v1.py",
    "src/hegel_machine/strict_ast_shrink5_v1.py",
    "src/hegel_machine/strict_ast_shrink6_v1.py",
    "src/hegel_machine/strict_ast_v1.py",
    "src/hegel_machine/strict_cbor_v1.py",
]


@pytest.fixture(scope="module")
def evidence() -> dict[str, object]:
    value = gates.qualify_q0_pre_dual_gates_v1(PROJECT_ROOT)
    gates.validate_pre_dual_gate_evidence_v1(value)
    return value


def _gate(evidence: dict[str, object], gate_id: int) -> dict[str, object]:
    rows = evidence["gates"]
    assert type(rows) is list
    row = rows[gate_id - 1]
    assert type(row) is dict
    return row


def test_pre_dual_registry_is_exactly_eleven_of_fourteen_and_never_receipt(
    evidence: dict[str, object],
) -> None:
    assert evidence["readiness_gate_total"] == 14
    assert evidence["readiness_gates_passed"] == 11
    assert evidence["readiness_gates_passed"] <= 11
    assert evidence["readiness_gate_mask"] == 0x0BFF
    assert evidence["receipt_created"] is False
    assert evidence["authoritative_claim_allowed"] is False
    assert evidence["q0_state"] == "PRE_DUAL_11_OF_14"

    rows = evidence["gates"]
    assert type(rows) is list and len(rows) == 14
    assert [row["gate_id"] for row in rows] == list(range(1, 15))
    assert [row["name"] for row in rows] == list(contract.Q0_READINESS_GATES)
    assert [row["gate_id"] for row in rows if row["passed"]] == list(
        gates.PRE_DUAL_PASS_GATE_IDS
    )
    assert [row["gate_id"] for row in rows if row["pending_dual"]] == [
        11,
        13,
        14,
    ]
    assert _gate(evidence, 14)["passed"] is False
    assert _gate(evidence, 14)["pending_dual"] is True
    assert all(
        value is False
        for value in _gate(evidence, 14)["predicates"].values()
    )


def test_every_internal_gate_has_live_predicates_and_nontrivial_evidence(
    evidence: dict[str, object],
) -> None:
    source_root = evidence["source_binding"]["manifest_root"]
    for gate_id in gates.PRE_DUAL_PASS_GATE_IDS:
        row = _gate(evidence, gate_id)
        assert row["passed"] is True
        assert row["pending_dual"] is False
        assert row["predicates"]
        assert all(value is True for value in row["predicates"].values())
        assert row["evidence"]["source_manifest_root"] == source_root
        # No gate is accepted on a bare True; every row carries replay data in
        # addition to the common source binding.
        assert len(row["evidence"]) >= 2


def test_direction_and_five_semantic_roots_are_bound_to_actual_sources(
    evidence: dict[str, object],
) -> None:
    gate1 = _gate(evidence, 1)
    direction_path = PROJECT_ROOT / (
        contract.NORMATIVE_DOCUMENT_PATH.removeprefix("Hegel Machine/")
    )
    assert gate1["evidence"]["sha256"] == sha256(
        direction_path.read_bytes()
    ).hexdigest()
    assert gate1["evidence"]["sha256"] == contract.NORMATIVE_DOCUMENT_SHA256

    gate2 = _gate(evidence, 2)
    assert gate2["evidence"]["five_v16_roots"] == {
        "child_dsl_spec_root": contract.Q0_CHILD_DSL_SPEC_ROOT.hex(),
        "operator_semantics_root": contract.Q0_OPERATOR_SEMANTICS_ROOT.hex(),
        "identifier_registry_root": contract.Q0_IDENTIFIER_REGISTRY_ROOT.hex(),
        "canonical_ast_schema_root": contract.Q0_CANONICAL_AST_SCHEMA_ROOT.hex(),
        "canonical_cbor_profile_root": contract.Q0_CANONICAL_CBOR_PROFILE_ROOT.hex(),
    }
    assert len(gate2["evidence"]["leaf_output_sort_ids"]) == 15
    assert gate2["evidence"]["leaf_output_sort_ids"] == [
        5,
        5,
        5,
        2,
        2,
        4,
        5,
        4,
        5,
        5,
        5,
        5,
        4,
        1,
        1,
    ]
    assert gate2["evidence"]["projection_manifest_root"] == (
        contract.q0_projection_manifest_root_v1().hex()
    )
    assert gate2["evidence"]["semantic_binding_root"] == (
        contract.q0_semantic_binding_root_v1().hex()
    )


def test_adapter_codec_and_universe_identity_vectors_are_explicit(
    evidence: dict[str, object],
) -> None:
    gate3 = _gate(evidence, 3)
    assert gate3["evidence"]["input_signature_ids"] == [1, 1, 2, 2]
    assert gate3["evidence"]["set_sizes"] == [5, 8, 4, 4]
    assert len(bytes.fromhex(gate3["evidence"]["probe_cbor_hex"])) == 172

    gate4 = _gate(evidence, 4)
    assert gate4["evidence"]["codec_cbor_hex"] == {
        "bottom_bool": "8100",
        "bool_true": "8201f5",
        "bit_one": "820101",
        "sign_negative": "820120",
        "bounded_int_negative_eight": "820127",
        "rational_negative_two_thirds": "8201822103",
    }
    assert gate4["evidence"]["bit_at_0_defined_positions"] == [
        True,
        True,
        False,
        False,
    ]

    gate5 = _gate(evidence, 5)
    assert len(set(gate5["evidence"]["identity_behavior_ids"])) == 4
    assert gate5["evidence"]["bound_input_signature_id"] == 0x7001


def test_equivalence_signature_pareto_and_cohort_counterevidence_are_replayed(
    evidence: dict[str, object],
) -> None:
    gate6 = _gate(evidence, 6)
    assert gate6["evidence"]["behavior_class_count"] == 69
    assert gate6["evidence"]["collision_failure_code"] == (
        "FAIL_SHA256_PREIMAGE_COLLISION"
    )
    assert gate6["evidence"]["output_sort_ids"] == [1, 2, 3, 4, 5]

    gate7 = _gate(evidence, 7)
    assert gate7["evidence"]["frontier_signature_count"] == 122
    assert gate7["evidence"]["tie_frontier_ranks"] == [0, 1]
    assert gate7["evidence"]["and2_counterexample_cbor_hex"] == (
        "82018204828300040083000500"
    )
    assert gate7["evidence"]["sort_witness_capacities"] == {
        "BOOL": 2,
        "BIT": 1,
        "SIGN": 1,
        "BOUNDED_INT": 1,
        "RATIONAL_VALUE": 2,
    }

    gate8 = _gate(evidence, 8)
    assert gate8["evidence"]["reservoir_visible_frontier_count"] == 2
    assert gate8["evidence"]["reservoir_continuation_bank_count"] == 3


def test_all_27_operator_rows_and_fixed_point_induction_are_evidence(
    evidence: dict[str, object],
) -> None:
    gate9 = _gate(evidence, 9)
    assert len(gate9["evidence"]["syntax_coverage_rows"]) == 27
    assert len(gate9["evidence"]["direct_coverage_rows"]) == 27
    assert gate9["evidence"]["explicit_zero_coverage_codes"] == [
        0x2005,
        0x2006,
    ]
    assert gate9["evidence"]["syntax_coverage_root"] != (
        gate9["evidence"]["direct_coverage_root"]
    )

    gate10 = _gate(evidence, 10)
    assert gate10["evidence"]["counts"] == {
        "syntax_raw_and_strict_admitted_count": 567,
        "direct_raw_and_strict_admitted_count": 545,
        "syntax_rewrite_count": 30,
        "direct_rewrite_count": 30,
        "canonical_syntax_program_count": 537,
        "behavior_class_count": 69,
        "visible_frontier_point_count": 122,
        "maximum_visible_frontier_points_per_class": 4,
        "continuation_bank_point_count_each_path": 251,
        "maximum_bank_points_per_class_each_path": 43,
        "saturation_round_count": 3,
    }
    final = gate10["evidence"]["round_deltas"][-1]
    assert final == {
        "round_index": 3,
        "queued_application_count": 0,
        "new_canonical_program_count": 0,
        "new_behavior_class_count": 0,
        "frontier_mutation_count": 0,
        "bank_mutation_count": 0,
        "complete_state_changed": False,
    }
    assert gate10["evidence"]["admission_evidence_scope"] == {
        "visible_frontier_ast_replay_count": 122,
        "all_seen_program_count_bound_by_program_archive": 537,
        "complete_continuation_bank_count_each_path_bound_by_state_roots": 251,
        "syntax_program_archive_root": (
            "bd1a59f816bd6648d0dd73b9a1622f2bb88bb9aeca1489a0d876fbc9dbf0c829"
        ),
        "syntax_state_root": (
            "7028819d133c4da6071c06a0bfca2d0b91622e106207d0b0f081148f41c0826a"
        ),
        "direct_state_root": (
            "d87ef33d9d7010ded284b55acfa71aab4d7d991e3d7703c30f1db2caf5893933"
        ),
    }
    assert (
        gate10["predicates"][
            "visible_frontier_recursive_projection_admission_replayed"
        ]
        is True
    )
    assert (
        gate10["predicates"][
            "all_seen_programs_bound_by_guarded_archive_and_state"
        ]
        is True
    )
    assert "DUAL" not in gate10["evidence"]["endpoint_status"]


def test_adversarial_sort_bottom_collision_and_ordering_vectors_pass(
    evidence: dict[str, object],
) -> None:
    gate12 = _gate(evidence, 12)
    assert all(gate12["predicates"].values())
    assert gate12["evidence"]["bool_alias_failure_code"] == (
        "REJECT_Q0_PROBE_INPUT"
    )
    assert gate12["evidence"]["behavior_cell_alias_failure_code"] == (
        "REJECT_Q0_BEHAVIOR_CELL"
    )
    assert gate12["evidence"]["collision_failure_code"] == (
        "FAIL_SHA256_PREIMAGE_COLLISION"
    )
    assert gate12["evidence"]["class_sort_failure_code"] == (
        "REJECT_Q0_QUOTIENT_ARCHIVE"
    )
    assert gate12["evidence"]["bool_bottom_behavior_id"] != (
        gate12["evidence"]["bit_bottom_behavior_id"]
    )
    assert len(set(gate12["evidence"]["bottom_position_behavior_ids"])) == 2


def test_source_manifest_root_and_canonical_json_replay(
    evidence: dict[str, object],
) -> None:
    source = evidence["source_binding"]
    assert source["entry_module"] == (
        "hegel_machine.phase3_q0_gate_qualification_v1"
    )
    assert source["closure_algorithm"] == "RECURSIVE_LOCAL_IMPORT_AST_CLOSURE_V1"
    assert source["file_count"] == 23
    assert [row["path"] for row in source["files"]] == EXPECTED_SOURCE_PATHS
    module_row = next(
        row
        for row in source["files"]
        if row["path"] == "src/hegel_machine/phase3_q0_gate_qualification_v1.py"
    )
    assert module_row["sha256"] == sha256(MODULE_PATH.read_bytes()).hexdigest()
    assert source["manifest_root"] == sha256(
        gates.SOURCE_BINDING_DOMAIN
        + gates.canonical_gate_json_bytes_v1(source["files"])
    ).hexdigest()
    encoded = gates.canonical_gate_json_bytes_v1(evidence)
    assert encoded.endswith(b"\n")
    assert json.loads(encoded) == evidence
    assert gates.canonical_gate_json_bytes_v1(json.loads(encoded)) == encoded


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("receipt_created", True),
        lambda value: value.__setitem__("readiness_gates_passed", 14),
        lambda value: value["gates"][0]["predicates"].__setitem__(
            "document_bytes_sha256_exact", False
        ),
        lambda value: value["gates"][10].__setitem__("passed", True),
        lambda value: value["gates"][13]["predicates"].__setitem__(
            "host_only_receipt_replay_verified", True
        ),
        lambda value: value["source_binding"]["files"][0].__setitem__(
            "sha256", "00" * 32
        ),
    ],
)
def test_validator_fails_closed_on_authority_or_predicate_tampering(
    evidence: dict[str, object], mutate: object
) -> None:
    changed = deepcopy(evidence)
    mutate(changed)  # type: ignore[operator]
    with pytest.raises(gates.Q0GateQualificationError):
        gates.validate_pre_dual_gate_evidence_v1(changed)


def test_live_qualifier_fails_before_oracle_on_direction_or_leaf_type_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "NORMATIVE_DOCUMENT_SHA256", "00" * 32)
    with pytest.raises(gates.Q0GateQualificationError) as caught:
        gates.qualify_q0_pre_dual_gates_v1(PROJECT_ROOT)
    assert caught.value.code == "FAIL_Q0_PRE_DUAL_GATE"
    assert "Gate 1" in caught.value.detail

    monkeypatch.undo()
    original = gates._oracle.behavior_blob_for_ast_v1

    def drift_first_leaf_sort(ast_value: object):
        behavior = original(ast_value)
        if ast_value.value[1] == contract.Q0_FROZEN_LEAF_CANONICAL_NODES[0]:
            return replace(behavior, output_sort_id=contract.OutputSortId.BOOL)
        return behavior

    monkeypatch.setattr(
        gates._oracle,
        "behavior_blob_for_ast_v1",
        drift_first_leaf_sort,
    )
    with pytest.raises(gates.Q0GateQualificationError) as leaf_caught:
        gates.qualify_q0_pre_dual_gates_v1(PROJECT_ROOT)
    assert leaf_caught.value.code == "FAIL_Q0_PRE_DUAL_GATE"
    assert "Gate 2" in leaf_caught.value.detail


def test_gate_module_has_no_import_of_target_truth_or_split_modules() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("phase3_dsl_v1", "target", "truth", "split")
    assert not any(
        token in module.lower()
        for module in imported
        for token in forbidden
    )
