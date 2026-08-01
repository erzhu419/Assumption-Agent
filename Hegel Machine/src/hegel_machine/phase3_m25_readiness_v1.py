"""Fail-closed Phase-3A M2.5 v1.1.2 readiness publication.

The deterministic typed-row and candidate-root layer is implemented, while
the exact E1--E12 errata and independent external actors remain unavailable.
This module records that distinction without turning candidate roots into
formal roots or authorizing external genesis.

This report is diagnostic JSON.  It is not a formal CBOR manifest, custodian
attestation, gate signature, or substitute for any root named by the freeze.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from .hashing import stable_hash
from .phase3_m25_external_v1 import EXACT_ERRATA_BLOCKERS
from .phase3_shrink1_publication_v1 import M3_REQUIRED_GATES as SHRINK1_GATES


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
NORMATIVE_DOCUMENT: Final = (
    PROJECT_ROOT
    / "docs"
    / "Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md"
)
OPEN_QUESTIONS_DOCUMENT: Final = (
    PROJECT_ROOT / "docs" / "questions_for_gpt_phase3_m25_wire_completion_errata.md"
)

MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
M25_PHASE_ID: Final = (
    "PHASE_3A_M2_5_FORMAL_COMMITMENT_SEED_GENESIS_BRIDGE_QUALIFICATION"
)
CURRENT_STATUS: Final = "EXACT_ERRATA_REQUIRED_EXTERNAL_GENESIS_BLOCKED"
CURRENT_CHILD_STATE: Final = "NOT_RUN"
PARENT_IMPLEMENTATION_COMMIT: Final = "d772b844e7c92b20f1e370244cc88202581fc72a"
V2_NEGATIVE_BINDING_ID: Final = (
    "v2_scar_negative_binding_"
    "3a40657c6d683da9ff74cfdade38fcced448ffe2909b13e14c684fae672b815b"
)

M25_REMAINING_GATES: Final = (
    "SPLIT_SEED_FIRST_INSTANTIATION_SIGNED",
    "HIDDEN_ACCESS_LEDGER_GENESIS_ONLY",
    "PARENT_MANIFEST_ABSENCE_ATTESTED",
    "FORMAL_BINDING_MANIFESTS_CANONICALIZED",
    "FORMAL_SPEC_AND_REGISTRY_ROOTS_DUAL_EQUAL",
    "ODD_UNIVERSE_AND_TRUTH_ROOTS_DUAL_EQUAL",
    "SINK_UNIVERSE_AND_TRUTH_ROOTS_DUAL_EQUAL",
    "SPLIT_PARTITION_ROOTS_DUAL_EQUAL",
    "M3_STATE_AND_RECEIPT_WIRE_GOLDEN_TESTS_PASS",
    "M3_EXECUTION_MANIFEST_ROOT_NON_NULL_AND_OUTPUT_ROOTS_NULL",
)

M3_GATES: Final = SHRINK1_GATES[:14] + M25_REMAINING_GATES

SPECIFICATION_BLOCKERS: Final = tuple(
    blocker.blocker_id for blocker in EXACT_ERRATA_BLOCKERS
)

EXTERNAL_ACTOR_BLOCKERS: Final = (
    "INDEPENDENT_CUSTODIAN_KEY_NOT_PROVISIONED",
    "SPLIT_SEED_FIRST_INSTANTIATION_NOT_PERFORMED",
    "CUSTODIAN_SIGNATURES_NOT_AVAILABLE",
    "INDEPENDENT_PARENT_ABSENCE_AUDIT_NOT_ATTESTED",
    "PYTHON_RUST_BRIDGE_ATTESTER_KEYS_NOT_PROVISIONED",
)

RUN_OUTPUT_ROOT_NAMES: Final = (
    "canonical_program_archive_root",
    "program_chunk_manifest_root",
    "bucket_accounting_root",
    "outside_program_output_archive_root",
    "outside_output_chunk_manifest_root",
    "outside_match_set_root",
    "outside_role_evaluation_receipt_root",
    "null_program_output_archive_root",
    "null_output_chunk_manifest_root",
    "null_match_set_root",
    "null_role_evaluation_receipt_root",
    "python_enumeration_receipt_root",
    "rust_enumeration_receipt_root",
    "dual_replay_agreement_root",
    "final_state_record_root",
)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json_type_strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_type_strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_type_strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _source_bindings() -> dict[str, str | None]:
    paths = (
        "src/hegel_machine/strict_cbor_v1.py",
        "src/hegel_machine/phase3_m25_wire_v1.py",
        "src/hegel_machine/phase3_m25_split_v1.py",
        "src/hegel_machine/phase3_m25_rows_v1.py",
        "src/hegel_machine/phase3_m25_qualification_v112.py",
        "src/hegel_machine/phase3_m25_external_v1.py",
        "src/hegel_machine/phase3_m25_replay_v1.py",
        "src/hegel_machine/phase3_m25_readiness_v1.py",
        "src/hegel_machine/cli.py",
        "rust/formal_bridge_m25/Cargo.toml",
        "rust/formal_bridge_m25/Cargo.lock",
        "rust/formal_bridge_m25/src/lib.rs",
        "rust/formal_bridge_m25/src/main.rs",
        "golden_vectors/phase3_m25_formal_wire_v1.json",
        "golden_vectors/phase3_m25_typed_rows_v1.json",
        "artifacts/phase3_m25_wire_completion_qualification_v112.json",
        "artifacts/phase3_m25_external_preflight_v1.json",
    )
    result: dict[str, str | None] = {}
    for relative in paths:
        path = PROJECT_ROOT / relative
        result[relative] = _sha256_file(path) if path.is_file() else None
    return result


def _gate_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, name in enumerate(M3_GATES, start=1):
        if index <= 14:
            status = "SATISFIED_BY_SHRINK1_QUALIFICATION"
            blockers: list[str] = []
        else:
            status = "BLOCKED_EXACT_ERRATA_AND_EXTERNAL_ACTOR"
            blockers = list(SPECIFICATION_BLOCKERS + EXTERNAL_ACTOR_BLOCKERS)
        records.append(
            {
                "gate_number": index,
                "gate_name": name,
                "status": status,
                "blockers": blockers,
            }
        )
    return records


def phase3_m25_readiness_report() -> dict[str, object]:
    """Return the diagnostic M2.5 publication without minting formal roots."""

    if len(M3_GATES) != 24 or M3_GATES[:14] != SHRINK1_GATES[:14]:
        raise AssertionError("M2.5 gate lineage drift")
    source_bindings = _source_bindings()
    foundation_files_present = all(value is not None for value in source_bindings.values())
    payload: dict[str, object] = {
        "artifact": "phase3_m25_readiness_v112_diagnostic",
        "artifact_kind": "DIAGNOSTIC_NON_AUTHORITATIVE",
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "child_dsl_id": CHILD_DSL_ID,
        "phase_id": M25_PHASE_ID,
        "status": CURRENT_STATUS,
        "child_state": CURRENT_CHILD_STATE,
        "parent_implementation_commit": PARENT_IMPLEMENTATION_COMMIT,
        "formal_repository_commit_id": None,
        "normative_document_sha256": _sha256_file(NORMATIVE_DOCUMENT),
        "open_questions_document_sha256": _sha256_file(OPEN_QUESTIONS_DOCUMENT),
        "foundation_source_bindings": source_bindings,
        "foundation_files_present": foundation_files_present,
        "foundation_scope": {
            "python_formal_wire": "DETERMINISTIC_CANDIDATE_IMPLEMENTED",
            "python_typed_rows": "CHECKED_ARTIFACT_REPORTS_480_ODD_PLUS_85_SINK",
            "python_split_crypto": "PURE_FUNCTION_QUALIFICATION_ONLY",
            "rust_formal_wire": "CHECKED_ARTIFACT_CALLER_SUPPLIED_UNATTESTED_REPLAY",
            "rust_binary_source_binding_claim": False,
            "readiness_command_reexecutes_rust_replay": False,
            "authoritative_root_generation": False,
            "real_seed_or_key_generation": False,
        },
        "m3_gates": _gate_records(),
        "m3_gates_satisfied": 14,
        "m3_gates_total": 24,
        "m3_entry_allowed": False,
        "not_run_to_running_transition_allowed": False,
        "formal_input_roots": None,
        "m3_execution_manifest_root": None,
        "run_output_roots": {name: None for name in RUN_OUTPUT_ROOT_NAMES},
        "split_seed_first_instantiated": False,
        "custodian_signature_claim": False,
        "parent_absence_attestation_claim": False,
        "hidden_access_ledger_genesis_claim": False,
        "specification_blockers": list(SPECIFICATION_BLOCKERS),
        "external_actor_blockers": list(EXTERNAL_ACTOR_BLOCKERS),
        "v2_counterevidence": {
            "binding_id": V2_NEGATIVE_BINDING_ID,
            "evidence_status": "PROTOCOL_VALID_NEGATIVE",
            "m25_formal_gate_effect": "NONE",
            "m3_closure_gate_effect": "NONE",
            "phase3b_design_risk": "HARD_STRUCTURAL_ELIGIBILITY_COVERAGE_COLLAPSE",
            "transfer_v2_thresholds_as_verified_positive_priors": False,
        },
        "claim_boundary": (
            "Checked-in M2.5 v1.1.2 diagnostic evidence reports that typed rows "
            "and candidate roots reproduce the public amendment values, but no "
            "authoritative formal root, "
            "seed genesis, actor attestation, M3 "
            "execution identity, closure verdict, outside certificate, or "
            "relation-invention claim exists."
        ),
    }
    payload["diagnostic_report_id"] = stable_hash(
        payload, prefix="phase3_m25_readiness_"
    )
    return payload


def validate_phase3_m25_readiness_report(report: dict[str, object]) -> None:
    """Reject accidental authority/state escalation in the status artifact."""

    if not isinstance(report, dict):
        raise TypeError("M2.5 readiness report must be a dictionary")
    if report.get("artifact_kind") != "DIAGNOSTIC_NON_AUTHORITATIVE":
        raise AssertionError("M2.5 status artifact must remain diagnostic")
    if report.get("child_state") != "NOT_RUN":
        raise AssertionError("underspecified M2.5 may not leave NOT_RUN")
    if report.get("m3_gates_satisfied") != 14:
        raise AssertionError("no post-shrink M3 gate is currently authoritative")
    if report.get("m3_gates_total") != 24:
        raise AssertionError("M2.5 readiness must retain the exact 24-gate registry")
    if report.get("m3_entry_allowed") is not False:
        raise AssertionError("M3 entry must remain fail-closed")
    if report.get("not_run_to_running_transition_allowed") is not False:
        raise AssertionError("NOT_RUN to RUNNING must remain prohibited")
    if report.get("formal_input_roots") is not None:
        raise AssertionError("authoritative formal roots have not been generated")
    if report.get("m3_execution_manifest_root") is not None:
        raise AssertionError("M3 execution manifest root must remain null")
    outputs = report.get("run_output_roots")
    if not isinstance(outputs, dict) or set(outputs) != set(RUN_OUTPUT_ROOT_NAMES):
        raise AssertionError("run output root registry drift")
    if any(value is not None for value in outputs.values()):
        raise AssertionError("run-produced output roots must all remain null")
    if report.get("split_seed_first_instantiated") is not False:
        raise AssertionError("diagnostic code cannot claim external seed genesis")
    if report.get("custodian_signature_claim") is not False:
        raise AssertionError("diagnostic code cannot claim custodian authority")
    if report.get("parent_absence_attestation_claim") is not False:
        raise AssertionError("diagnostic code cannot claim independent audit authority")
    if report.get("hidden_access_ledger_genesis_claim") is not False:
        raise AssertionError("diagnostic code cannot claim signed ledger genesis")
    if report.get("formal_repository_commit_id") is not None:
        raise AssertionError("no authoritative formal repository binding exists")
    if report.get("foundation_files_present") is not True:
        raise AssertionError("all synthetic foundation sources and vectors must exist")

    gates = report.get("m3_gates")
    expected_gates = _gate_records()
    if not _json_type_strict_equal(gates, expected_gates):
        raise AssertionError("M2.5 gate names, order, status, or blockers drifted")
    if report.get("specification_blockers") != list(SPECIFICATION_BLOCKERS):
        raise AssertionError("M2.5 specification blocker registry drifted")
    if report.get("external_actor_blockers") != list(EXTERNAL_ACTOR_BLOCKERS):
        raise AssertionError("M2.5 external-actor blocker registry drifted")

    provided_report_id = report.get("diagnostic_report_id")
    report_body = dict(report)
    report_body.pop("diagnostic_report_id", None)
    expected_report_id = stable_hash(
        report_body,
        prefix="phase3_m25_readiness_",
    )
    if provided_report_id != expected_report_id:
        raise AssertionError("M2.5 diagnostic report self-ID mismatch")

    expected_report = phase3_m25_readiness_report()
    if not _json_type_strict_equal(dict(report), expected_report):
        raise AssertionError("M2.5 readiness report differs from current bound sources")


__all__ = [
    "CHILD_DSL_ID",
    "CURRENT_CHILD_STATE",
    "CURRENT_STATUS",
    "EXTERNAL_ACTOR_BLOCKERS",
    "M25_PHASE_ID",
    "M25_REMAINING_GATES",
    "M3_GATES",
    "MACHINE_FREEZE_ID",
    "RUN_OUTPUT_ROOT_NAMES",
    "SPECIFICATION_BLOCKERS",
    "phase3_m25_readiness_report",
    "validate_phase3_m25_readiness_report",
]
