from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.phase3_m25_readiness_v1 import (
    DUAL_GOLDEN_QUALIFICATION_BLOCKERS,
    M25_REMAINING_GATES,
    M3_GATES,
    RESOLVED_SPECIFICATION_PREREQUISITES,
    RUN_OUTPUT_ROOT_NAMES,
    SPECIFICATION_BLOCKERS,
    phase3_m25_readiness_report,
    validate_phase3_m25_readiness_report,
)
from hegel_machine.cli import main


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_CHECKED_IN = ROOT / "artifacts" / "phase3_m25_readiness_v1.json"


def test_m25_report_preserves_not_run_and_exact_gate_lineage() -> None:
    report = phase3_m25_readiness_report()
    validate_phase3_m25_readiness_report(report)

    assert len(M3_GATES) == 24
    assert M3_GATES[14:] == M25_REMAINING_GATES
    assert report["m3_gates_satisfied"] == 14
    assert report["child_state"] == "NOT_RUN"
    assert report["m3_entry_allowed"] is False
    assert report["m3_entry_qualified"] is False
    assert report["m3_run_started"] is False
    assert report["not_run_to_running_transition_allowed"] is False
    assert report["formal_input_roots"] is None
    assert report["m3_execution_manifest_root"] is None
    assert report["machine_freeze_id"] == "hegel-freeze-p2b-p3-v1.1.2"
    assert report["status"] == (
        "EXACT_ERRATA_RESOLVED_DUAL_GOLDEN_VERIFICATION_REQUIRED"
    )
    assert SPECIFICATION_BLOCKERS == ()
    assert len(RESOLVED_SPECIFICATION_PREREQUISITES) == 12
    assert report["exact_errata_resolved"] is True
    assert report["specification_blockers"] == []
    assert report["resolved_specification_prerequisites"] == list(
        RESOLVED_SPECIFICATION_PREREQUISITES
    )
    assert report["external_genesis_start_allowed"] is False

    gates = report["m3_gates"]
    assert isinstance(gates, list)
    assert [gate["gate_number"] for gate in gates] == list(range(1, 25))
    assert all(
        gate["status"] == "BLOCKED_EXTERNAL_GENESIS_AND_FORMAL_QUALIFICATION"
        for gate in gates[14:]
    )
    assert all(
        not any(
            blocker.startswith(tuple(f"E{index}_" for index in range(1, 13)))
            for blocker in gate["blockers"]
        )
        for gate in gates[14:]
    )


def test_m25_report_has_all_null_run_output_slots() -> None:
    report = phase3_m25_readiness_report()
    outputs = report["run_output_roots"]
    assert isinstance(outputs, dict)
    assert tuple(outputs) == RUN_OUTPUT_ROOT_NAMES
    assert all(root is None for root in outputs.values())


def test_checked_in_v111_readiness_remains_historical() -> None:
    historical = json.loads(HISTORICAL_CHECKED_IN.read_text(encoding="utf-8"))
    assert historical["machine_freeze_id"] == "hegel-freeze-p2b-p3-v1.1.1"
    assert phase3_m25_readiness_report()["machine_freeze_id"] == (
        "hegel-freeze-p2b-p3-v1.1.2"
    )
    with pytest.raises(AssertionError):
        validate_phase3_m25_readiness_report(historical)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("child_state", "RUNNING", "may not leave NOT_RUN"),
        ("m3_gates_satisfied", 15, "currently authoritative"),
        ("m3_gates_total", 14, "24-gate registry"),
        ("m3_entry_allowed", True, "remain fail-closed"),
        ("m3_entry_qualified", True, "has not qualified"),
        ("m3_run_started", True, "cannot start M3"),
        (
            "not_run_to_running_transition_allowed",
            True,
            "RUNNING must remain prohibited",
        ),
        ("formal_input_roots", {"forged": "sha256:x"}, "have not been generated"),
        ("split_seed_first_instantiated", True, "external seed genesis"),
        ("custodian_signature_claim", True, "custodian authority"),
        ("parent_absence_attestation_claim", True, "audit authority"),
        ("hidden_access_ledger_genesis_claim", True, "ledger genesis"),
        (
            "external_genesis_start_allowed",
            True,
            "lacks committed dual-golden evidence",
        ),
        ("gate24_qualified", True, "cannot pass"),
    ],
)
def test_m25_report_rejects_authority_escalation(
    field: str,
    value: object,
    message: str,
) -> None:
    report = deepcopy(phase3_m25_readiness_report())
    report[field] = value
    with pytest.raises(AssertionError, match=message):
        validate_phase3_m25_readiness_report(report)


def test_m25_report_rejects_prepopulated_run_output_root() -> None:
    report = deepcopy(phase3_m25_readiness_report())
    outputs = report["run_output_roots"]
    assert isinstance(outputs, dict)
    outputs[RUN_OUTPUT_ROOT_NAMES[0]] = "sha256:" + "00" * 32
    with pytest.raises(AssertionError, match="must all remain null"):
        validate_phase3_m25_readiness_report(report)


def test_m25_report_rejects_gate_status_or_self_id_tampering() -> None:
    report = deepcopy(phase3_m25_readiness_report())
    gates = report["m3_gates"]
    assert isinstance(gates, list)
    gates[14]["status"] = "SATISFIED"
    with pytest.raises(AssertionError, match="gate names, order, status"):
        validate_phase3_m25_readiness_report(report)

    report = deepcopy(phase3_m25_readiness_report())
    report["diagnostic_report_id"] = "phase3_m25_readiness_" + "00" * 32
    with pytest.raises(AssertionError, match="self-ID mismatch"):
        validate_phase3_m25_readiness_report(report)


def test_gate24_uses_15_null_slots_and_does_not_start_m3() -> None:
    report = phase3_m25_readiness_report()
    gate24 = report["gate24_contract"]
    assert gate24["gate_name"] == (
        "M3_EXECUTION_MANIFEST_ROOT_NON_NULL_AND_15_OUTPUT_ROOTS_NULL"
    )
    assert gate24["pass_predicate"]["run_output_slot_count"] == 15
    assert gate24["pass_predicate"]["bridge_signer_purposes_exactly"] == [1, 2, 3]
    assert gate24["gate24_passed"] is False
    assert report["gate24_qualified"] is False

    start = report["phase3_m3_start_contract"]
    assert start["action_id"] == "phase3-m3-start"
    assert start["requires_complete_24_of_24_replay"] is True
    assert start["requires_bound_opaque_id_snapshot"] is True
    assert start["start_record_created"] is False
    assert report["m3_run_started"] is False


def test_dual_golden_and_external_actor_blockers_are_not_specification_blockers() -> None:
    report = phase3_m25_readiness_report()
    assert report["specification_blockers"] == []
    assert report["dual_golden_qualification_blockers"] == list(
        DUAL_GOLDEN_QUALIFICATION_BLOCKERS
    )
    assert report["external_actor_blockers"]


def test_v2_negative_is_a_design_risk_not_a_formal_gate() -> None:
    counterevidence = phase3_m25_readiness_report()["v2_counterevidence"]
    assert isinstance(counterevidence, dict)
    assert counterevidence["evidence_status"] == "PROTOCOL_VALID_NEGATIVE"
    assert counterevidence["m25_formal_gate_effect"] == "NONE"
    assert counterevidence["m3_closure_gate_effect"] == "NONE"
    assert counterevidence["transfer_v2_thresholds_as_verified_positive_priors"] is False


def test_m25_cli_emits_diagnostic_readiness_without_starting_m3(
    tmp_path,
    capsys,
) -> None:
    output = tmp_path / "m25.json"
    assert main(["phase3-m25-readiness", "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert printed == written
    assert written["artifact_kind"] == "DIAGNOSTIC_NON_AUTHORITATIVE"
    assert written["child_state"] == "NOT_RUN"
    assert written["m3_entry_allowed"] is False
