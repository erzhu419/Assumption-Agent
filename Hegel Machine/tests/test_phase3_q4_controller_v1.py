from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/phase3_q4_formal_quotient_certificate_capability_v1.py"
SPEC = importlib.util.spec_from_file_location("phase3_q4_tool_test_controller", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)
state = tool.state


def _temporary_contract(tmp_path: Path) -> Path:
    actual = state.load_contract(tool.DEFAULT_CONTRACT)
    value = deepcopy(actual.value)
    run_root = tmp_path / "run"
    ledger_root = tmp_path / "ledger"
    artifact_root = tmp_path / "artifacts"
    value["fixed_paths"].update(
        {
            "campaign_lock_path": str(tmp_path / "campaign.lock"),
            "run_root": str(run_root),
            "ledger_root": str(ledger_root),
            "capsule_path": str(run_root / "capsule.json"),
            "authorization_path": str(ledger_root / "owner.authorization.json"),
            "consumption_path": str(ledger_root / "campaign.consumed.json"),
            "pending_path": str(run_root / "latest.pending.json"),
            "terminal_path": str(ledger_root / "campaign.terminal.json"),
            "artifact_path": str(artifact_root / "q4-capability.cbor"),
            "trajectory_path": str(run_root / "trajectory.jsonl"),
        }
    )
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=False) + "\n",
        encoding="ascii",
    )
    return path


def _prepared(tmp_path: Path):
    contract_path = _temporary_contract(tmp_path)
    capsule = tool.prepare_capsule(
        contract_path,
        "a" * 40,
        run_id="b" * 32,
        created_at_unix_seconds=1_788_300_000,
    )
    contract = state.load_contract(contract_path)
    return contract_path, contract, capsule


def test_controller_source_has_no_process_network_key_or_signing_calls() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TOOL_PATH))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not {"subprocess", "socket", "urllib", "requests"}.intersection(imports)
    assert not {"Popen", "run", "system", "fork", "execv", "execve"}.intersection(calls)
    for forbidden in (
        "Ed25519PrivateKey", "generate_private_key", "load_pem_private_key",
        ".sign(", "phase3_certificate_v1",
    ):
        assert forbidden not in source


def test_prepare_capsule_creates_no_authorization_or_campaign_state(tmp_path: Path) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    fixed = contract.value["fixed_paths"]
    assert capsule["status"] == state.CAPSULE_READY
    assert capsule["preflight_receipt"]["capability_disposition"] == (
        "ADMITTED_NOT_ISSUED"
    )
    assert Path(fixed["capsule_path"]).exists()
    for key in ("authorization_path", "consumption_path", "artifact_path", "terminal_path"):
        assert not Path(fixed[key]).exists()
    assert capsule["execution_policy"] == state.EXECUTION_POLICY
    assert capsule["authority_boundary"] == state.AUTHORITY_BOUNDARY
    assert tool.prepare_capsule(contract_path, "f" * 40) == capsule


def test_run_without_authorization_is_zero_consumption(tmp_path: Path) -> None:
    contract_path, _, capsule = _prepared(tmp_path)
    result = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert result["status"] == state.NOT_AUTHORIZED
    assert result["authorization_consumption_count"] == 0
    assert result["logical_campaign_count"] == 0
    assert result["q4_action_audit"] == state.ZERO_ACTION_AUDIT


def test_record_authorization_then_run_completes_once_not_issued(tmp_path: Path) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    authorization = tool.record_authorization(
        contract_path, capsule["capsule_digest"]
    )
    assert authorization["status"] == state.AUTHORIZATION_RECORDED
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == tool.wire.Q4_SUCCESS_STATUS
    assert terminal["authorization_consumption_count"] == 1
    assert terminal["logical_campaign_count"] == 1
    assert terminal["outcome"]["capability_disposition"] == "ADMITTED_NOT_ISSUED"
    assert terminal["outcome"]["formal_certificate_signed"] is False
    assert terminal["outcome"]["outside_certificate_issued"] is False
    assert terminal["outcome"]["synthesis_released"] is False
    assert terminal["outcome"]["q4_action_audit"] == state.ZERO_ACTION_AUDIT
    assert tool.run_v1(contract_path, capsule["capsule_digest"]) == terminal

    fixed = contract.value["fixed_paths"]
    assert Path(fixed["artifact_path"]).exists()
    assert Path(fixed["terminal_path"]).exists()


def test_q3_drift_before_consumption_enters_pending_without_consuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    original = tool._read_q3_selected

    def drift(selected_contract: Any) -> dict[str, bytes]:
        selected = original(selected_contract)
        payload = selected["terminal"]
        selected["terminal"] = bytes((payload[0] ^ 1,)) + payload[1:]
        return selected

    monkeypatch.setattr(tool, "_read_q3_selected", drift)
    with pytest.raises(tool.ControllerPending, match="SEALED_Q3_REPLAY"):
        tool.record_authorization(contract_path, capsule["capsule_digest"])
    fixed = contract.value["fixed_paths"]
    pending_path = Path(fixed["pending_path"])
    assert pending_path.exists()
    pending = state.strict_json_bytes(pending_path.read_bytes(), "Q4 pending")
    assert pending["status"] == state.PENDING_NOT_CONSUMED
    assert pending["capsule_digest"] == capsule["capsule_digest"]
    assert pending["authorization_consumption_count"] == 0
    assert pending["logical_campaign_count"] == 0
    assert not Path(fixed["authorization_path"]).exists()
    assert not Path(fixed["consumption_path"]).exists()
    assert not Path(fixed["artifact_path"]).exists()
    assert not Path(fixed["terminal_path"]).exists()


def test_terminal_publication_failure_allows_only_artifact_to_terminal_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    tool.record_authorization(contract_path, capsule["capsule_digest"])
    terminal_path = Path(contract.value["fixed_paths"]["terminal_path"])
    original = tool._publish_json

    def fail_terminal(path: Path, value: Any, label: str) -> None:
        if path == terminal_path:
            raise OSError("synthetic terminal publication failure")
        original(path, value, label)

    monkeypatch.setattr(tool, "_publish_json", fail_terminal)
    with pytest.raises(tool.ControllerPartial):
        tool.run_v1(contract_path, capsule["capsule_digest"])
    fixed = contract.value["fixed_paths"]
    assert Path(fixed["consumption_path"]).exists()
    assert Path(fixed["artifact_path"]).exists()
    assert not terminal_path.exists()

    monkeypatch.setattr(tool, "_publish_json", original)
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == tool.wire.Q4_SUCCESS_STATUS
    assert terminal["outcome"]["outside_certificate_issued"] is False


def test_missing_artifact_after_consumption_seals_terminal_no_resume(
    tmp_path: Path,
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    authorization = tool.record_authorization(
        contract_path, capsule["capsule_digest"]
    )
    consumption = state.build_consumption(
        contract,
        state.load_preregistration(
            Path(contract.value["preregistration_path"])
        ).value,
        capsule,
        authorization,
    )
    consumption_path = Path(contract.value["fixed_paths"]["consumption_path"])
    consumption_path.parent.mkdir(parents=True, exist_ok=True)
    consumption_path.write_bytes(state.canonical_json_bytes(consumption))

    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == state.INVALIDATED_TERMINAL_NO_RESUME
    assert terminal["outcome"]["failure_code"] == (
        "Q4_CAPABILITY_ARTIFACT_MISSING_AFTER_CONSUMPTION"
    )
    assert terminal["outcome"]["terminal_no_resume"] is True
    assert terminal["outcome"]["outside_certificate_issued"] is False


def test_prepare_and_record_never_reopen_pending_after_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    authorization = tool.record_authorization(
        contract_path, capsule["capsule_digest"]
    )
    prereg = state.load_preregistration(
        Path(contract.value["preregistration_path"])
    )
    consumption = state.build_consumption(
        contract, prereg.value, capsule, authorization
    )
    consumption_path = Path(contract.value["fixed_paths"]["consumption_path"])
    consumption_path.parent.mkdir(parents=True, exist_ok=True)
    consumption_path.write_bytes(state.canonical_json_bytes(consumption))

    def forbidden_replay(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("post-consumption entry reopened Q3 replay")

    monkeypatch.setattr(tool, "_preconsumption_replay", forbidden_replay)
    assert tool.prepare_capsule(contract_path, "f" * 40) == capsule
    assert tool.record_authorization(
        contract_path, capsule["capsule_digest"]
    ) == authorization
    assert not Path(contract.value["fixed_paths"]["pending_path"]).exists()

    Path(contract.value["fixed_paths"]["capsule_path"]).write_bytes(b"not-json")
    with pytest.raises(tool.ControllerTerminal, match="CAPSULE"):
        tool.prepare_capsule(contract_path, "f" * 40)
    assert not Path(contract.value["fixed_paths"]["pending_path"]).exists()


def test_malformed_consumption_or_terminal_never_becomes_pending(
    tmp_path: Path,
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    tool.record_authorization(contract_path, capsule["capsule_digest"])
    consumption_path = Path(contract.value["fixed_paths"]["consumption_path"])
    consumption_path.parent.mkdir(parents=True, exist_ok=True)
    consumption_path.write_bytes(b"not-json")
    with pytest.raises(tool.ControllerTerminal, match="CONSUMPTION"):
        tool.run_v1(contract_path, capsule["capsule_digest"])
    assert not Path(contract.value["fixed_paths"]["pending_path"]).exists()

    consumption_path.unlink()
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == tool.wire.Q4_SUCCESS_STATUS
    terminal_path = Path(contract.value["fixed_paths"]["terminal_path"])
    terminal_path.write_bytes(b"not-json")
    with pytest.raises(tool.ControllerTerminal, match="TERMINAL"):
        tool.run_v1(contract_path, capsule["capsule_digest"])
    assert not Path(contract.value["fixed_paths"]["pending_path"]).exists()


def test_unknown_consumption_publication_outcome_never_becomes_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    tool.record_authorization(contract_path, capsule["capsule_digest"])
    consumption_path = Path(contract.value["fixed_paths"]["consumption_path"])
    original_publish = tool._publish_json
    original_read = tool._read_regular

    def publish_then_report_error(path: Path, value: Any, label: str) -> None:
        original_publish(path, value, label)
        if path == consumption_path:
            raise OSError("synthetic uncertainty after durable replace")

    def fail_durable_replay(
        path: Path, label: str, maximum_bytes: int = state.MAX_JSON_BYTES
    ) -> bytes:
        if path == consumption_path and "after publication error" in label:
            raise tool.ControllerPending(
                "LOCAL_READ_ONLY_REPLAY",
                "BOUND_LOCAL_FILE_UNAVAILABLE",
                "synthetic unreadable durable consumption",
            )
        return original_read(path, label, maximum_bytes)

    monkeypatch.setattr(tool, "_publish_json", publish_then_report_error)
    monkeypatch.setattr(tool, "_read_regular", fail_durable_replay)
    with pytest.raises(
        tool.ControllerTerminal, match="CONSUMPTION_PUBLICATION_OUTCOME_UNKNOWN"
    ):
        tool.run_v1(contract_path, capsule["capsule_digest"])
    assert consumption_path.exists()
    assert not Path(contract.value["fixed_paths"]["pending_path"]).exists()
