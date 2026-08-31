from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _tool():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "phase3_q4_capability_tool_test_state",
        ROOT / "tools/phase3_q4_formal_quotient_certificate_capability_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _tool()
state = tool.state
wire = tool.wire


def _loaded():
    contract = state.load_contract(tool.DEFAULT_CONTRACT)
    prereg = state.load_preregistration(Path(contract.value["preregistration_path"]))
    rows = tool._source_closure_rows(contract)
    root = state.source_closure_root(rows)
    replay = tool._replay_q3(
        contract,
        prereg,
        q4_source_commit="a" * 40,
        q4_source_closure_root=root,
    )
    return contract, prereg, rows, root, replay


def _capsule():
    contract, prereg, rows, root, replay = _loaded()
    capsule = state.build_capsule(
        contract,
        prereg.value,
        {
            "source_commit": "a" * 40,
            "source_closure_rows": rows,
            "source_closure_root": root,
            "run_id": "b" * 32,
            "created_at_unix_seconds": 1_788_300_000,
            "preflight_receipt": replay.preflight_receipt,
        },
    )
    return contract, prereg, replay, capsule


def test_actual_q3_five_input_replay_is_exact_and_admission_only() -> None:
    _, _, _, _, replay = _loaded()
    receipt = replay.preflight_receipt
    assert [row["kind"] for row in receipt["selected_q3_inputs"]] == [
        "capsule", "authorization", "consumption", "terminal", "artifact"
    ]
    assert receipt["selected_q3_input_count"] == 5
    assert receipt["selected_q3_input_total_bytes"] == 19_924
    assert receipt["capability_disposition"] == "ADMITTED_NOT_ISSUED"
    assert receipt["q4_action_audit"] == state.ZERO_ACTION_AUDIT
    assert all(
        value == 0
        for value in state.ZERO_ACTION_AUDIT.values()
        if type(value) is int
    )
    assert all(
        value is False
        for value in state.ZERO_ACTION_AUDIT.values()
        if type(value) is bool
    )
    decoded = wire.decode_q4_capability_receipt_v1(replay.artifact_payload)
    assert decoded == replay.capability_receipt
    assert decoded.canonical_bytes == replay.artifact_payload


@pytest.mark.parametrize(
    "kind", ["capsule", "authorization", "consumption", "terminal", "artifact"]
)
def test_each_q3_input_raw_drift_is_rejected(kind: str) -> None:
    contract = state.load_contract(tool.DEFAULT_CONTRACT)
    prereg = state.load_preregistration(Path(contract.value["preregistration_path"]))
    selected = tool._read_q3_selected(contract)
    selected[kind] = bytes((selected[kind][0] ^ 1,)) + selected[kind][1:]
    rows = tool._source_closure_rows(contract)
    with pytest.raises(state.Q4CompletionReplayError, match="raw SHA-256"):
        state.replay_q3_predecessor_v1(
            prereg.value,
            q4_source_commit="a" * 40,
            q4_source_closure_root=state.source_closure_root(rows),
            capsule_payload=selected["capsule"],
            authorization_payload=selected["authorization"],
            consumption_payload=selected["consumption"],
            terminal_payload=selected["terminal"],
            artifact_payload=selected["artifact"],
        )


def test_capsule_authorization_and_consumption_are_exactly_replayed() -> None:
    contract, prereg, _, capsule = _capsule()
    assert state.validate_capsule(capsule, contract, prereg.value) == capsule
    authorization = state.build_authorization(contract, prereg.value, capsule)
    assert state.validate_authorization(
        authorization, contract, prereg.value, capsule
    ) == authorization
    assert authorization["certificate_issuance_allowed"] is False
    assert authorization["certificate_signing_allowed"] is False
    assert authorization["key_operation_allowed"] is False
    consumption = state.build_consumption(
        contract, prereg.value, capsule, authorization
    )
    assert state.validate_consumption(
        consumption, contract, prereg.value, capsule, authorization
    ) == consumption
    assert consumption["authorization_consumption_count"] == 1
    assert consumption["logical_campaign_count"] == 1
    assert consumption["outside_certificate_issued"] is False
    assert consumption["formal_certificate_signed"] is False
    assert consumption["synthesis_released"] is False


def test_complete_terminal_is_capability_only_and_digest_bound() -> None:
    contract, prereg, replay, capsule = _capsule()
    authorization = state.build_authorization(contract, prereg.value, capsule)
    consumption = state.build_consumption(
        contract, prereg.value, capsule, authorization
    )
    terminal = state.build_complete_terminal(
        contract,
        prereg.value,
        capsule,
        authorization,
        consumption,
        replay.artifact_payload,
    )
    assert state.validate_terminal(
        terminal,
        contract,
        prereg.value,
        capsule,
        authorization,
        consumption,
        artifact_payload=replay.artifact_payload,
    ) == terminal
    outcome = terminal["outcome"]
    assert terminal["status"] == wire.Q4_SUCCESS_STATUS
    assert outcome["capability_disposition"] == "ADMITTED_NOT_ISSUED"
    assert outcome["purpose5_key_installed"] is False
    assert outcome["formal_certificate_signed"] is False
    assert outcome["outside_certificate_issued"] is False
    assert outcome["synthesis_released"] is False

    forged = deepcopy(terminal)
    forged["outcome"]["outside_certificate_issued"] = True
    with pytest.raises(state.Q4CompletionReplayError, match="terminal replay"):
        state.validate_terminal(
            forged,
            contract,
            prereg.value,
            capsule,
            authorization,
            consumption,
            artifact_payload=replay.artifact_payload,
        )


def test_source_closure_forbids_legacy_certificate_module() -> None:
    _, _, rows, _, _ = _loaded()
    assert [row["relative_path"] for row in rows] == list(state.Q4_SOURCE_PATHS)
    assert "src/hegel_machine/phase3_certificate_v1.py" not in state.Q4_SOURCE_PATHS
    assert "src/hegel_machine/phase3_q3_completion_replay_v1.py" not in (
        state.Q4_SOURCE_PATHS
    )
