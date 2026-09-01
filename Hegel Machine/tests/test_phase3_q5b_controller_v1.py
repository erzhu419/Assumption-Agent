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
STATE_ROOT = Path("/home/erzhu419/.local/state/hegel-machine")
TOOL_PATH = ROOT / "tools/phase3_q5b_purpose5_trust_amendment_v1.py"
SPEC = importlib.util.spec_from_file_location("phase3_q5b_tool_test", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)
state = tool.state
wire = tool.wire


def _q5a_paths() -> dict[str, str]:
    return {
        "capsule": str(STATE_ROOT / "q5a-purpose5-signer-genesis-v1-run/capsule.json"),
        "authorization": str(
            STATE_ROOT
            / "q5a-purpose5-signer-genesis-v1-ledger/owner.authorization.json"
        ),
        "consumption": str(
            STATE_ROOT
            / "q5a-purpose5-signer-genesis-v1-ledger/campaign.consumed.json"
        ),
        "terminal": str(
            STATE_ROOT
            / "q5a-purpose5-signer-genesis-v1-ledger/campaign.terminal.json"
        ),
        "artifact": str(
            ROOT / "artifacts/phase3_q5a_purpose5_signer_genesis_candidate_v1.cbor"
        ),
        "trajectory": str(
            STATE_ROOT / "q5a-purpose5-signer-genesis-v1-run/trajectory.jsonl"
        ),
    }


def _predecessor(paths: dict[str, str]) -> dict[str, object]:
    payloads = {key: Path(path).read_bytes() for key, path in paths.items()}
    capsule = state.strict_json_bytes(payloads["capsule"], "Q5a capsule")
    authorization = state.strict_json_bytes(payloads["authorization"], "Q5a auth")
    consumption = state.strict_json_bytes(payloads["consumption"], "Q5a consumption")
    terminal = state.strict_json_bytes(payloads["terminal"], "Q5a terminal")
    candidate = wire.decode_sealed_q5a_candidate_v1(payloads["artifact"])
    return {
        "source_commit": capsule["source_commit"],
        "source_closure_root": capsule["source_closure_root"],
        "capsule": {
            "raw_sha256": state.raw_sha256(payloads["capsule"]),
            "capsule_digest": capsule["capsule_digest"],
        },
        "authorization": {
            "raw_sha256": state.raw_sha256(payloads["authorization"]),
            "authorization_digest": authorization["authorization_digest"],
        },
        "consumption": {
            "raw_sha256": state.raw_sha256(payloads["consumption"]),
            "record_digest": consumption["record_digest"],
        },
        "terminal": {
            "raw_sha256": state.raw_sha256(payloads["terminal"]),
            "record_digest": terminal["record_digest"],
        },
        "artifact": {
            "raw_sha256": state.raw_sha256(payloads["artifact"]),
            "candidate_root": "sha256:" + candidate.candidate_root.hex(),
            "unpinned_genesis_root": "sha256:" + candidate.unpinned_genesis_root.hex(),
            "genesis_only_status_root": "sha256:" + candidate.genesis_only_status_root.hex(),
            "manifest_roots": [
                "sha256:" + item.manifest_root.hex()
                for item in candidate.signer_manifests
            ],
            "key_ids": [item.key_id.hex() for item in candidate.signer_manifests],
        },
        "trajectory": {
            "raw_sha256": state.raw_sha256(payloads["trajectory"]),
            "event_count": 7,
        },
        "selected_input_total_bytes": sum(len(item) for item in payloads.values()),
    }


def _temporary_contract(tmp_path: Path) -> Path:
    q5a = _q5a_paths()
    preregistration = {
        "schema_version": state.PREREGISTRATION_SCHEMA,
        "preregistration_id": state.PREREGISTRATION_ID,
        "status": "Q5B_SPEC_FROZEN_NOT_AUTHORIZED",
        "predecessor_q5a": _predecessor(q5a),
    }
    prereg_path = tmp_path / "preregistration.json"
    prereg_raw = state.canonical_json_bytes(preregistration)
    prereg_path.write_bytes(prereg_raw)
    run_root = tmp_path / "run"
    ledger_root = tmp_path / "ledger"
    fixed = {
        "project_root": str(ROOT),
        "campaign_lock_path": str(tmp_path / "campaign.lock"),
        "run_root": str(run_root),
        "ledger_root": str(ledger_root),
        "capsule_path": str(run_root / "capsule.json"),
        "authorization_path": str(ledger_root / "owner.authorization.json"),
        "consumption_path": str(ledger_root / "campaign.consumed.json"),
        "pending_path": str(run_root / "latest.pending.json"),
        "terminal_path": str(ledger_root / "campaign.terminal.json"),
        "artifact_path": str(tmp_path / "artifacts/q5b-amendment.cbor"),
        "trajectory_path": str(run_root / "trajectory.jsonl"),
    }
    contract = {
        "schema_version": state.CONTRACT_SCHEMA,
        "contract_id": state.CONTRACT_ID,
        "status": "IMPLEMENTED_NOT_EXECUTED",
        "action_id": state.ACTION_ID,
        "preregistration_path": str(prereg_path),
        "preregistration_raw_sha256": state.raw_sha256(prereg_raw),
        "fixed_paths": fixed,
        "q5a_inputs": q5a,
        "execution_policy": deepcopy(state.EXECUTION_POLICY),
        "authority_boundary": deepcopy(state.AUTHORITY_BOUNDARY),
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="ascii")
    return path


def _prepared(tmp_path: Path):
    contract_path = _temporary_contract(tmp_path)
    capsule = tool.prepare_capsule(
        contract_path,
        "a" * 40,
        run_id="b" * 32,
        created_at_unix_seconds=1_788_600_000,
    )
    contract = state.load_contract(contract_path)
    return contract_path, contract, capsule


def _relocated_actual_contract(tmp_path: Path) -> tuple[Path, Any, dict[str, bool]]:
    actual = state.load_contract(tool.DEFAULT_CONTRACT)
    value = deepcopy(actual.value)
    original_presence = {
        key: Path(value["fixed_paths"][key]).exists()
        for key in (
            "capsule_path", "authorization_path", "consumption_path",
            "pending_path", "artifact_path", "terminal_path", "trajectory_path",
        )
    }
    run_root = tmp_path / "actual-config-run"
    ledger_root = tmp_path / "actual-config-ledger"
    value["fixed_paths"].update(
        {
            "campaign_lock_path": str(tmp_path / "actual-config.campaign.lock"),
            "run_root": str(run_root),
            "ledger_root": str(ledger_root),
            "capsule_path": str(run_root / "capsule.json"),
            "authorization_path": str(ledger_root / "owner.authorization.json"),
            "consumption_path": str(ledger_root / "campaign.consumed.json"),
            "pending_path": str(run_root / "latest.pending.json"),
            "artifact_path": str(tmp_path / "actual-config-artifacts/q5b.cbor"),
            "terminal_path": str(ledger_root / "campaign.terminal.json"),
            "trajectory_path": str(run_root / "trajectory.jsonl"),
        }
    )
    path = tmp_path / "actual-config-relocated.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    return path, actual, original_presence


def _trajectory(contract: Any) -> list[dict[str, Any]]:
    path = Path(contract.value["fixed_paths"]["trajectory_path"])
    return [
        state.strict_json_bytes(line, f"Q5b trajectory {index}")
        for index, line in enumerate(path.read_bytes().splitlines(keepends=True), 1)
    ]


def test_controller_has_no_process_private_or_external_capability() -> None:
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
    assert "ed25519-private.pem" not in source
    assert "signer_1_custody_root" not in source
    assert ".sign(" not in source


def test_actual_config_loads_and_prepares_only_in_relocated_fixture(
    tmp_path: Path,
) -> None:
    contract_path, actual, original_presence = _relocated_actual_contract(tmp_path)
    assert set(actual.value["q5a_inputs"]) == {
        "capsule", "authorization", "consumption", "terminal", "artifact",
        "trajectory",
    }
    assert "trajectory_path" in actual.value["fixed_paths"]
    assert actual.value["execution_policy"] == state.EXECUTION_POLICY
    assert actual.value["authority_boundary"] == state.AUTHORITY_BOUNDARY
    capsule = tool.prepare_capsule(
        contract_path,
        "d" * 40,
        run_id="e" * 32,
        created_at_unix_seconds=1_788_600_001,
    )
    relocated = state.load_contract(contract_path)
    fixed = relocated.value["fixed_paths"]
    assert capsule["status"] == state.CAPSULE_READY
    assert capsule["preflight_receipt"]["selected_q5a_input_count"] == 6
    assert capsule["preflight_receipt"]["custody_root_access_count"] == 0
    assert Path(fixed["capsule_path"]).exists()
    assert Path(fixed["trajectory_path"]).exists()
    for key in (
        "authorization_path", "consumption_path", "artifact_path", "terminal_path"
    ):
        assert not Path(fixed[key]).exists()
    assert {
        key: Path(actual.value["fixed_paths"][key]).exists()
        for key in original_presence
    } == original_presence


def test_prepare_authorize_run_is_three_event_success(tmp_path: Path) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    assert [row["event"] for row in _trajectory(contract)] == [
        "CAPSULE_PREPARED_NOT_AUTHORIZED"
    ]
    authorization = tool.record_authorization(
        contract_path, capsule["capsule_digest"]
    )
    assert authorization["status"] == state.AUTHORIZATION_RECORDED
    assert [row["event"] for row in _trajectory(contract)] == [
        "CAPSULE_PREPARED_NOT_AUTHORIZED",
        "AUTHORIZATION_RECORDED_NOT_CONSUMED",
    ]
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    outcome = terminal["outcome"]
    assert terminal["status"] == wire.Q5B_SUCCESS_STATUS
    assert terminal["authorization_consumption_count"] == 1
    assert terminal["logical_campaign_count"] == 1
    assert outcome["q5b_action_audit"] == state.SUCCESS_ACTION_AUDIT
    assert outcome["purpose5_active_signer_count"] == 3
    assert outcome["purpose5_trust_genesis_pinned"] is True
    assert outcome["purpose5_current_key_status_accepted"] is True
    assert outcome["global_active_transition_performed"] is False
    assert outcome["formal_certificate_signed"] is False
    assert outcome["outside_certificate_issued"] is False
    trajectory = _trajectory(contract)
    assert len(trajectory) == 3
    assert trajectory[-1]["event"] == wire.Q5B_SUCCESS_STATUS
    assert trajectory[-1]["authorization_consumption_count"] == 1
    assert trajectory[-1]["q5b_action_audit"] == state.SUCCESS_ACTION_AUDIT
    assert tool.run_v1(contract_path, capsule["capsule_digest"]) == terminal
    assert len(_trajectory(contract)) == 3


def test_existing_capsule_and_authorization_replay_live_inputs_before_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    original = tool._read_q5a_selected

    def drift(selected_contract: Any) -> dict[str, bytes]:
        rows = original(selected_contract)
        rows["trajectory"] = rows["trajectory"][:-1]
        return rows

    monkeypatch.setattr(tool, "_read_q5a_selected", drift)
    with pytest.raises(tool.ControllerPending, match="SEALED_Q5A_REPLAY"):
        tool.prepare_capsule(contract_path, "f" * 40)
    fixed = contract.value["fixed_paths"]
    pending = state.strict_json_bytes(
        Path(fixed["pending_path"]).read_bytes(), "Q5b pending"
    )
    assert pending["authorization_consumption_count"] == 0
    assert not Path(fixed["consumption_path"]).exists()

    monkeypatch.setattr(tool, "_read_q5a_selected", original)
    authorization = tool.record_authorization(
        contract_path, capsule["capsule_digest"]
    )
    monkeypatch.setattr(tool, "_read_q5a_selected", drift)
    with pytest.raises(tool.ControllerPending, match="SEALED_Q5A_REPLAY"):
        tool.record_authorization(contract_path, capsule["capsule_digest"])
    pending = state.strict_json_bytes(
        Path(fixed["pending_path"]).read_bytes(), "Q5b pending"
    )
    assert pending["authorization_digest"] == authorization["authorization_digest"]
    assert pending["authorization_consumption_count"] == 0
    assert not Path(fixed["consumption_path"]).exists()


def test_run_without_authorization_replays_live_inputs_before_not_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    original = tool._read_q5a_selected

    def drift(selected_contract: Any) -> dict[str, bytes]:
        rows = original(selected_contract)
        rows["trajectory"] = rows["trajectory"][:-1]
        return rows

    monkeypatch.setattr(tool, "_read_q5a_selected", drift)
    with pytest.raises(tool.ControllerPending, match="SEALED_Q5A_REPLAY"):
        tool.run_v1(contract_path, capsule["capsule_digest"])

    fixed = contract.value["fixed_paths"]
    pending = state.strict_json_bytes(
        Path(fixed["pending_path"]).read_bytes(), "Q5b pending"
    )
    assert pending["authorization_consumption_count"] == 0
    assert pending["logical_campaign_count"] == 0
    assert not Path(fixed["authorization_path"]).exists()
    assert not Path(fixed["consumption_path"]).exists()


def test_existing_unconsumed_entries_require_artifact_absence(
    tmp_path: Path,
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    fixed = contract.value["fixed_paths"]
    artifact_path = Path(fixed["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(b"preexisting")

    with pytest.raises(
        tool.ControllerPending,
        match="Q5B_ARTIFACT_EXISTS_BEFORE_CONSUMPTION",
    ):
        tool.prepare_capsule(contract_path, "f" * 40)
    assert not Path(fixed["consumption_path"]).exists()

    artifact_path.unlink()
    authorization = tool.record_authorization(
        contract_path, capsule["capsule_digest"]
    )
    artifact_path.write_bytes(b"preexisting-after-authorization")
    with pytest.raises(
        tool.ControllerPending,
        match="Q5B_ARTIFACT_EXISTS_BEFORE_CONSUMPTION",
    ):
        tool.record_authorization(contract_path, capsule["capsule_digest"])

    pending = state.strict_json_bytes(
        Path(fixed["pending_path"]).read_bytes(), "Q5b pending"
    )
    assert pending["authorization_digest"] == authorization["authorization_digest"]
    assert pending["authorization_consumption_count"] == 0
    assert not Path(fixed["consumption_path"]).exists()


def test_unreadable_trajectory_is_pending_before_consumption_and_partial_after_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    fixed = contract.value["fixed_paths"]
    trajectory_path = Path(fixed["trajectory_path"])
    original_read = tool._read_regular

    def unavailable(path: Path, label: str, maximum_bytes: int = state.MAX_JSON_BYTES) -> bytes:
        if path == trajectory_path:
            raise tool.ControllerPending(
                "LOCAL_READ_ONLY_REPLAY",
                "BOUND_PUBLIC_FILE_UNAVAILABLE",
                label,
            )
        return original_read(path, label, maximum_bytes)

    monkeypatch.setattr(tool, "_read_regular", unavailable)
    with pytest.raises(
        tool.ControllerPending,
        match="BOUND_PUBLIC_FILE_UNAVAILABLE",
    ):
        tool.prepare_capsule(contract_path, "f" * 40)
    pending = state.strict_json_bytes(
        Path(fixed["pending_path"]).read_bytes(), "Q5b pending"
    )
    assert pending["authorization_consumption_count"] == 0
    assert not Path(fixed["consumption_path"]).exists()

    monkeypatch.setattr(tool, "_read_regular", original_read)
    tool.record_authorization(contract_path, capsule["capsule_digest"])
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == wire.Q5B_SUCCESS_STATUS

    monkeypatch.setattr(tool, "_read_regular", unavailable)
    with pytest.raises(tool.ControllerPartial, match="trajectory replay unavailable"):
        tool.run_v1(contract_path, capsule["capsule_digest"])


def test_consumed_entries_never_reopen_preflight_or_pending(
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

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("post-consumption entry reopened preflight")

    monkeypatch.setattr(tool, "_preconsumption_replay", forbidden)
    assert tool.prepare_capsule(contract_path, "f" * 40) == capsule
    assert tool.record_authorization(
        contract_path, capsule["capsule_digest"]
    ) == authorization
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == state.INVALIDATED_TERMINAL_NO_RESUME
    assert terminal["outcome"]["failure_code"] == (
        "Q5B_AMENDMENT_ARTIFACT_MISSING_AFTER_CONSUMPTION"
    )
    assert not Path(contract.value["fixed_paths"]["pending_path"]).exists()
    trajectory = _trajectory(contract)
    assert len(trajectory) == 3
    assert trajectory[-1]["q5b_action_audit"] == state.ZERO_ACTION_AUDIT
    assert trajectory[-1]["failure_code"] == (
        "Q5B_AMENDMENT_ARTIFACT_MISSING_AFTER_CONSUMPTION"
    )


def test_artifact_to_terminal_and_terminal_to_trajectory_suffix_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract_path, contract, capsule = _prepared(tmp_path)
    tool.record_authorization(contract_path, capsule["capsule_digest"])
    terminal_path = Path(contract.value["fixed_paths"]["terminal_path"])
    original_publish_json = tool._publish_json

    def fail_terminal(path: Path, value: Any, label: str) -> None:
        if path == terminal_path:
            raise OSError("synthetic terminal failure")
        original_publish_json(path, value, label)

    monkeypatch.setattr(tool, "_publish_json", fail_terminal)
    with pytest.raises(tool.ControllerPartial):
        tool.run_v1(contract_path, capsule["capsule_digest"])
    assert Path(contract.value["fixed_paths"]["artifact_path"]).exists()
    assert not terminal_path.exists()
    assert len(_trajectory(contract)) == 2

    monkeypatch.setattr(tool, "_publish_json", original_publish_json)
    trajectory_path = Path(contract.value["fixed_paths"]["trajectory_path"])
    original_replace = tool._replace_bytes

    def fail_final_suffix(path: Path, payload: bytes, label: str) -> None:
        if path == trajectory_path and len(payload.splitlines()) == 3:
            raise OSError("synthetic trajectory suffix failure")
        original_replace(path, payload, label)

    monkeypatch.setattr(tool, "_replace_bytes", fail_final_suffix)
    with pytest.raises(tool.ControllerPartial, match="trajectory suffix"):
        tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal_path.exists()
    assert len(_trajectory(contract)) == 2

    monkeypatch.setattr(tool, "_replace_bytes", original_replace)
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == wire.Q5B_SUCCESS_STATUS
    assert len(_trajectory(contract)) == 3


def test_bootstrap_failure_never_claims_pending_zero_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = tool.main(
        [
            "--contract",
            str(tmp_path / "missing-contract.json"),
            "run",
            "--capsule-digest",
            "sha256:" + "0" * 64,
        ]
    )
    assert code == tool.TERMINAL_EXIT_CODE
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "Q5B_TERMINAL_ERROR"
    assert "NO_STATE_CLAIM" in output["code"]
    assert output["status"] != state.PENDING_NOT_CONSUMED
