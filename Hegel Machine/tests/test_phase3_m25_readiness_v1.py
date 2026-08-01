from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine.phase3_m25_readiness_v1 import (
    M25_REMAINING_GATES,
    M3_GATES,
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
    assert report["not_run_to_running_transition_allowed"] is False
    assert report["formal_input_roots"] is None
    assert report["m3_execution_manifest_root"] is None
    assert report["machine_freeze_id"] == "hegel-freeze-p2b-p3-v1.1.2"
    assert report["status"] == "EXACT_ERRATA_REQUIRED_EXTERNAL_GENESIS_BLOCKED"
    assert len(SPECIFICATION_BLOCKERS) == 12

    gates = report["m3_gates"]
    assert isinstance(gates, list)
    assert [gate["gate_number"] for gate in gates] == list(range(1, 25))
    assert all(gate["status"].startswith("BLOCKED_") for gate in gates[14:])


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
