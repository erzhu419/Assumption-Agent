"""Fail-closed Phase-3A M2.5 v1.1.2 readiness publication v2.

E1--E12 are resolved deterministic prerequisites.  Committed dual-golden
evidence and independent external actors remain unavailable.  This module
records that distinction without turning deterministic evidence into formal
roots, passing Gate 24, or starting M3.

This report is diagnostic JSON.  It is not a formal CBOR manifest, custodian
attestation, gate signature, or substitute for any root named by the freeze.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from .hashing import stable_hash
from .phase3_m25_external_v1 import (
    ERRATA_RESOLUTION_DOCUMENT,
    EXACT_ERRATA_PREREQUISITES,
    GATE24_NAME,
    IMPLEMENTATION_ADDENDUM_DOCUMENT,
    RUN_OUTPUT_SLOT_NAMES,
    external_genesis_preflight_report,
    external_genesis_start_guard_report,
)
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
CURRENT_STATUS: Final = "EXACT_ERRATA_RESOLVED_DUAL_GOLDEN_VERIFICATION_REQUIRED"
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
    GATE24_NAME,
)

M3_GATES: Final = SHRINK1_GATES[:14] + M25_REMAINING_GATES

RESOLVED_SPECIFICATION_PREREQUISITES: Final = tuple(
    prerequisite.decision_id for prerequisite in EXACT_ERRATA_PREREQUISITES
)
SPECIFICATION_BLOCKERS: Final = ()

DUAL_GOLDEN_QUALIFICATION_BLOCKERS: Final = (
    "COMMIT_A_NORMATIVE_BUNDLE_AND_IMPLEMENTATION_NOT_BOUND",
    "PYTHON_RUST_ERRATA_GOLDEN_VERIFICATION_NOT_BOUND",
)

EXTERNAL_ACTOR_BLOCKERS: Final = (
    "INDEPENDENT_CUSTODIAN_KEY_NOT_PROVISIONED",
    "SPLIT_SEED_FIRST_INSTANTIATION_NOT_PERFORMED",
    "CUSTODIAN_SIGNATURES_NOT_AVAILABLE",
    "INDEPENDENT_PARENT_ABSENCE_AUDIT_NOT_ATTESTED",
    "PYTHON_RUST_BRIDGE_ATTESTER_KEYS_NOT_PROVISIONED",
)

RUN_OUTPUT_ROOT_NAMES: Final = RUN_OUTPUT_SLOT_NAMES


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
        "src/hegel_machine/hashing.py",
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
        "docs/Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md",
        "docs/Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md",
        "docs/Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md",
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
            status = "BLOCKED_EXTERNAL_GENESIS_AND_FORMAL_QUALIFICATION"
            blockers = list(
                DUAL_GOLDEN_QUALIFICATION_BLOCKERS + EXTERNAL_ACTOR_BLOCKERS
            )
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
    external_preflight = external_genesis_preflight_report()
    start_guard = external_genesis_start_guard_report()
    payload: dict[str, object] = {
        "artifact": "phase3_m25_readiness_v112_diagnostic_v2",
        "schema_version": "hegel-phase3-m25-readiness/2",
        "artifact_kind": "DIAGNOSTIC_NON_AUTHORITATIVE",
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "child_dsl_id": CHILD_DSL_ID,
        "phase_id": M25_PHASE_ID,
        "status": CURRENT_STATUS,
        "child_state": CURRENT_CHILD_STATE,
        "parent_implementation_commit": PARENT_IMPLEMENTATION_COMMIT,
        "formal_repository_commit_id": None,
        "commit_A_implementation_basis_bound": False,
        "normative_document_sha256": _sha256_file(NORMATIVE_DOCUMENT),
        "errata_resolution_document_sha256": _sha256_file(
            ERRATA_RESOLUTION_DOCUMENT
        ),
        "implementation_addendum_document_sha256": _sha256_file(
            IMPLEMENTATION_ADDENDUM_DOCUMENT
        ),
        "historical_open_questions_document_sha256": _sha256_file(
            OPEN_QUESTIONS_DOCUMENT
        ),
        "foundation_source_bindings": source_bindings,
        "foundation_files_present": foundation_files_present,
        "foundation_scope": {
            "python_formal_wire": "DETERMINISTIC_CANDIDATE_IMPLEMENTED",
            "python_typed_rows": "CHECKED_ARTIFACT_REPORTS_480_ODD_PLUS_85_SINK",
            "python_split_crypto": "PURE_FUNCTION_QUALIFICATION_ONLY",
            "rust_formal_wire": "CHECKED_ARTIFACT_CALLER_SUPPLIED_UNATTESTED_REPLAY",
            "rust_binary_source_binding_claim": False,
            "readiness_command_reexecutes_rust_replay": False,
            "exact_errata_specification": "RESOLVED_DETERMINISTIC_PREREQUISITES",
            "dual_errata_golden_verification": "NOT_YET_BOUND_TO_COMMIT_A",
            "authoritative_root_generation": False,
            "real_seed_or_key_generation": False,
        },
        "exact_errata_resolved": True,
        "resolved_specification_prerequisites": list(
            RESOLVED_SPECIFICATION_PREREQUISITES
        ),
        "specification_blockers": [],
        "external_genesis_preflight_artifact": external_preflight["artifact"],
        "external_genesis_start_guard": start_guard,
        "external_genesis_start_allowed": False,
        "m3_gates": _gate_records(),
        "m3_gates_satisfied": 14,
        "m3_gates_total": 24,
        "m3_entry_qualified": False,
        "m3_entry_allowed": False,
        "m3_run_started": False,
        "not_run_to_running_transition_allowed": False,
        "formal_input_roots": None,
        "m3_execution_manifest_root": None,
        "m3_run_genesis_root": None,
        "run_output_roots": {name: None for name in RUN_OUTPUT_ROOT_NAMES},
        "gate24_contract": external_preflight["gate24_contract"],
        "gate24_qualified": False,
        "phase3_m3_start_contract": external_preflight[
            "phase3_m3_start_contract"
        ],
        "split_seed_first_instantiated": False,
        "custodian_signature_claim": False,
        "parent_absence_attestation_claim": False,
        "hidden_access_ledger_genesis_claim": False,
        "dual_golden_qualification_blockers": list(
            DUAL_GOLDEN_QUALIFICATION_BLOCKERS
        ),
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
            "E1--E12 and the implementation closure decisions are resolved at the "
            "deterministic specification layer. Commit-A dual-golden evidence and "
            "external actors are not yet bound, so the child remains 14/24 and "
            "NOT_RUN. No authoritative formal root, seed genesis, actor "
            "attestation, Gate-24 qualification, M3 execution identity, start "
            "transition, closure verdict, outside certificate, or relation-"
            "invention claim exists."
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
        raise AssertionError("pre-external M2.5 may not leave NOT_RUN")
    if report.get("m3_gates_satisfied") != 14:
        raise AssertionError("no post-shrink M3 gate is currently authoritative")
    if report.get("m3_gates_total") != 24:
        raise AssertionError("M2.5 readiness must retain the exact 24-gate registry")
    if report.get("m3_entry_allowed") is not False:
        raise AssertionError("M3 entry must remain fail-closed")
    if report.get("m3_entry_qualified") is not False:
        raise AssertionError("Gate 24 has not qualified M3 entry")
    if report.get("m3_run_started") is not False:
        raise AssertionError("readiness publication cannot start M3")
    if report.get("not_run_to_running_transition_allowed") is not False:
        raise AssertionError("NOT_RUN to RUNNING must remain prohibited")
    if report.get("formal_input_roots") is not None:
        raise AssertionError("authoritative formal roots have not been generated")
    if report.get("m3_execution_manifest_root") is not None:
        raise AssertionError("M3 execution manifest root must remain null")
    if report.get("m3_run_genesis_root") is not None:
        raise AssertionError("M3 run genesis root must remain null")
    outputs = report.get("run_output_roots")
    if not isinstance(outputs, dict) or tuple(outputs) != RUN_OUTPUT_ROOT_NAMES:
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
    if report.get("commit_A_implementation_basis_bound") is not False:
        raise AssertionError("Commit A implementation basis is not yet bound")
    if report.get("foundation_files_present") is not True:
        raise AssertionError("all synthetic foundation sources and vectors must exist")

    if report.get("exact_errata_resolved") is not True:
        raise AssertionError("E1--E12 must remain resolved prerequisites")
    if report.get("specification_blockers") != []:
        raise AssertionError("resolved E1--E12 may not remain specification blockers")
    if report.get("resolved_specification_prerequisites") != list(
        RESOLVED_SPECIFICATION_PREREQUISITES
    ):
        raise AssertionError("resolved specification prerequisite registry drifted")
    expected_guard = external_genesis_start_guard_report()
    if not _json_type_strict_equal(
        report.get("external_genesis_start_guard"), expected_guard
    ):
        raise AssertionError("external genesis dual-golden guard drifted")
    if report.get("external_genesis_start_allowed") is not False:
        raise AssertionError("external genesis lacks committed dual-golden evidence")

    gate24 = report.get("gate24_contract")
    expected_gate24 = external_genesis_preflight_report()["gate24_contract"]
    if not _json_type_strict_equal(gate24, expected_gate24):
        raise AssertionError("Gate 24 exact contract drifted")
    if report.get("gate24_qualified") is not False:
        raise AssertionError("Gate 24 cannot pass before external formal evidence")
    start_contract = report.get("phase3_m3_start_contract")
    expected_start = external_genesis_preflight_report()[
        "phase3_m3_start_contract"
    ]
    if not _json_type_strict_equal(start_contract, expected_start):
        raise AssertionError("phase3-m3-start frozen contract drifted")

    gates = report.get("m3_gates")
    expected_gates = _gate_records()
    if not _json_type_strict_equal(gates, expected_gates):
        raise AssertionError("M2.5 gate names, order, status, or blockers drifted")
    if report.get("dual_golden_qualification_blockers") != list(
        DUAL_GOLDEN_QUALIFICATION_BLOCKERS
    ):
        raise AssertionError("M2.5 dual-golden blocker registry drifted")
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
    "DUAL_GOLDEN_QUALIFICATION_BLOCKERS",
    "EXTERNAL_ACTOR_BLOCKERS",
    "M25_PHASE_ID",
    "M25_REMAINING_GATES",
    "M3_GATES",
    "MACHINE_FREEZE_ID",
    "RUN_OUTPUT_ROOT_NAMES",
    "RESOLVED_SPECIFICATION_PREREQUISITES",
    "SPECIFICATION_BLOCKERS",
    "phase3_m25_readiness_report",
    "validate_phase3_m25_readiness_report",
]
