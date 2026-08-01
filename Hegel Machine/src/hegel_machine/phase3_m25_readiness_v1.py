"""Fail-closed Phase-3A M2.5 implementation/readiness publication.

The v1.1.1 amendment authorizes work on formal wire and custody, but a source
audit found byte-identity decisions that are still missing from the normative
text.  This module records that distinction explicitly: deterministic wire and
cryptographic *foundations* may be implemented and tested with synthetic data,
while authoritative roots, actors, seed material, signatures, and M3 state
transitions remain unavailable.

This report is diagnostic JSON.  It is not a formal CBOR manifest, custodian
attestation, gate signature, or substitute for any root named by the freeze.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from .hashing import stable_hash
from .phase3_shrink1_publication_v1 import M3_REQUIRED_GATES as SHRINK1_GATES


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
NORMATIVE_DOCUMENT: Final = (
    PROJECT_ROOT
    / "docs"
    / "Hegel_Machine_Phase3_Formal_Bridge_Seed_Genesis_M3_Wire_Freeze.md"
)
OPEN_QUESTIONS_DOCUMENT: Final = (
    PROJECT_ROOT / "docs" / "questions_for_gpt_phase3_m25_wire_completion.md"
)

MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.1"
CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
M25_PHASE_ID: Final = (
    "PHASE_3A_M2_5_FORMAL_COMMITMENT_SEED_GENESIS_BRIDGE_QUALIFICATION"
)
CURRENT_STATUS: Final = "M25_FOUNDATION_IMPLEMENTED_NORMATIVE_COMPLETION_REQUIRED"
CURRENT_CHILD_STATE: Final = "NOT_RUN"
PARENT_IMPLEMENTATION_COMMIT: Final = (
    "405ab52534c3e23eb7ab1025705310f57b217ba4"
)
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

SPECIFICATION_BLOCKERS: Final = (
    "ID_DIGEST_PROFILE_UNFROZEN",
    "NUMERIC_ENUM_REGISTRY_INCOMPLETE",
    "FORMAL_ROOT_PREIMAGE_SCHEMAS_UNFROZEN",
    "CANONICAL_ODD_SINK_INPUT_ROW_WIRE_UNFROZEN",
    "SINK_85_ROW_SPLIT_CONTRACT_UNFROZEN",
    "CUSTODIAN_BINDING_CORE_WIRE_UNDEFINED",
    "PARENT_ABSENCE_ATTESTATION_SIGNATURE_WIRE_UNDEFINED",
    "DIAGNOSTIC_BRIDGE_AGGREGATION_AND_SIGNATURE_POLICY_UNFROZEN",
    "BUCKET_ACCOUNTING_AND_IMPLEMENTATION_CONTRACT_WIRES_UNDEFINED",
    "M3_STATE_PREFIX_AND_TRANSITION_REASON_WIRE_UNFROZEN",
    "ENUMERATION_ROLE_AGREEMENT_CONSTRUCTION_CYCLE",
    "M3_OUTPUT_NULL_SLOT_CONTAINER_UNDEFINED",
    "HIDDEN_ARTIFACT_SCOPE_UNFROZEN",
    "PRE_RUN_IMPLEMENTATION_QUALIFICATION_GATES_NOT_PRESERVED",
)

EXTERNAL_ACTOR_BLOCKERS: Final = (
    "INDEPENDENT_CUSTODIAN_KEY_NOT_PROVISIONED",
    "SPLIT_SEED_FIRST_INSTANTIATION_NOT_PERFORMED",
    "CUSTODIAN_SIGNATURES_NOT_AVAILABLE",
    "INDEPENDENT_PARENT_ABSENCE_AUDIT_NOT_ATTESTED",
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


def _source_bindings() -> dict[str, str | None]:
    paths = (
        "src/hegel_machine/strict_cbor_v1.py",
        "src/hegel_machine/phase3_m25_wire_v1.py",
        "src/hegel_machine/phase3_m25_split_v1.py",
        "src/hegel_machine/phase3_m25_replay_v1.py",
        "src/hegel_machine/phase3_m25_readiness_v1.py",
        "src/hegel_machine/cli.py",
        "rust/formal_bridge_m25/Cargo.toml",
        "rust/formal_bridge_m25/Cargo.lock",
        "rust/formal_bridge_m25/src/lib.rs",
        "rust/formal_bridge_m25/src/main.rs",
        "golden_vectors/phase3_m25_formal_wire_v1.json",
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
        elif index in {15, 16, 17, 22}:
            status = "BLOCKED_SPECIFICATION_AND_EXTERNAL_ACTOR"
            blockers = list(SPECIFICATION_BLOCKERS + EXTERNAL_ACTOR_BLOCKERS)
        else:
            status = "BLOCKED_SPECIFICATION"
            blockers = list(SPECIFICATION_BLOCKERS)
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
        "artifact": "phase3_m25_readiness_v1",
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
            "python_formal_wire": "SYNTHETIC_QUALIFICATION_ONLY",
            "python_split_crypto": "PURE_FUNCTION_QUALIFICATION_ONLY",
            "rust_formal_wire": "SYNTHETIC_QUALIFICATION_ONLY",
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
            "M2.5 deterministic foundations are under qualification, but no "
            "authoritative formal root, seed genesis, actor attestation, M3 "
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
    if gates != expected_gates:
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
    if report != expected_report:
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
