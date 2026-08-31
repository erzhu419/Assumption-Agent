from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_completion_successor_v8c_tool",
    ROOT / "tools" / "phase3_q1_completion_successor_v8c.py",
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(TOOL_SPEC)
sys.modules[TOOL_SPEC.name] = tool
TOOL_SPEC.loader.exec_module(tool)


def _sha(label: str) -> str:
    return "sha256:" + sha256(label.encode("ascii")).hexdigest()


@pytest.fixture(scope="module")
def actual_v7r1_material() -> Any:
    contract = tool.state.load_contract(tool.DEFAULT_CONTRACT)
    return tool._replay_v7r1(contract)


def _temporary_contract(tmp_path: Path) -> Any:
    actual = tool.state.load_contract(tool.DEFAULT_CONTRACT)
    value = deepcopy(actual.value)
    run_root = tmp_path / "q1-completion-v8c-run"
    ledger_root = tmp_path / "q1-completion-v8c-ledger"
    value["fixed_paths"] = {
        "project_root": str(ROOT),
        "campaign_lock_path": str(tmp_path / "q1-completion-v8c.lock"),
        "run_root": str(run_root),
        "ledger_root": str(ledger_root),
        "capsule_path": str(run_root / "capsule.json"),
        "authorization_path": str(ledger_root / "owner.authorization.json"),
        "consumption_path": str(ledger_root / "completion.consumed.json"),
        "pending_path": str(run_root / "pending.json"),
        "outcome_path": str(run_root / "completion.outcome.json"),
        "terminal_path": str(ledger_root / "completion.terminal.json"),
        "artifact_path": str(tmp_path / "artifacts" / "completion-v8c.cbor"),
    }
    path = tmp_path / "completion-v8c-contract.json"
    path.write_bytes(tool.state.canonical_json_bytes(value))
    return tool.state.load_contract(path)


def _capsule(
    contract: Any,
    material: Any,
    *,
    artifact_raw_sha256: str | None = None,
) -> dict[str, Any]:
    rows = [
        {
            "relative_path": "tools/phase3_q1_completion_successor_v8c.py",
            "byte_count": 1,
            "raw_sha256": _sha("controller source"),
        }
    ]
    inputs = {
        "schema_version": tool.state.CAPSULE_INPUTS_SCHEMA,
        "source_closure_rows": rows,
        "source_closure_root": tool.state.source_closure_root(rows),
        "run_id": "8" * 32,
        "created_at_unix_seconds": 1_788_170_800,
        "expected_completion_artifact_raw_sha256": (
            artifact_raw_sha256
            or tool.state.raw_sha256(material.completion_artifact)
        ),
        "expected_completion_receipt_root": (
            "sha256:" + material.completion_receipt.receipt_root.hex()
        ),
        "expected_q1_output_slots": list(material.q1_output_slots),
    }
    return tool.state.build_capsule(contract, inputs)


def _publish_capsule(contract: Any, capsule: dict[str, Any]) -> None:
    fixed = contract.value["fixed_paths"]
    Path(fixed["run_root"]).mkdir(parents=True)
    Path(fixed["ledger_root"]).mkdir(parents=True)
    Path(fixed["artifact_path"]).parent.mkdir(parents=True)
    Path(fixed["capsule_path"]).write_bytes(
        tool.state.canonical_json_bytes(capsule)
    )


def _publish_authorization(contract: Any, capsule: dict[str, Any]) -> dict[str, Any]:
    authorization = tool.state.build_authorization(contract, capsule)
    Path(contract.value["fixed_paths"]["authorization_path"]).write_bytes(
        tool.state.canonical_json_bytes(authorization)
    )
    return authorization


def _publish_consumption(
    contract: Any,
    capsule: dict[str, Any],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    consumption = tool.state.build_consumption(
        contract, capsule, authorization
    )
    Path(contract.value["fixed_paths"]["consumption_path"]).write_bytes(
        tool.state.canonical_json_bytes(consumption)
    )
    return consumption


def _synthetic_material() -> Any:
    receipt_root = b"\x11" * 32
    return SimpleNamespace(
        completion_artifact=b"synthetic completion receipt",
        completion_receipt=SimpleNamespace(receipt_root=receipt_root),
        q1_output_slots=(
            *tuple(_sha(f"synthetic slot {index}") for index in range(7)),
            "sha256:" + receipt_root.hex(),
        ),
    )


def _isolate_replay(
    monkeypatch: pytest.MonkeyPatch,
    material: Any,
) -> None:
    monkeypatch.setattr(tool, "_replay_source_closure", lambda *_args: None)
    monkeypatch.setattr(tool, "_replay_v7r1", lambda _contract: material)


def test_actual_v7r1_read_only_replay_builds_corrected_completion_receipt(
    actual_v7r1_material: Any,
) -> None:
    material = actual_v7r1_material

    assert material.v7_terminal["record_digest"] == (
        "sha256:87c620d4ad1cf8dcd6dba4aacae51319aabbeb1189fe5e62b9b236a9b037abc8"
    )
    assert material.v7_terminal["artifact_raw_sha256"] is None
    assert material.v7_terminal["outcome"]["actor_start_count_by_actor"] == {
        "host": 1,
        "python": 1,
        "rust": 1,
    }
    assert tool.state.raw_sha256(material.completion_artifact) == (
        "sha256:623bdfa6ce60b244d3dff9d0ed274dda2ef52048b3992f7dae07ce8983c3ce18"
    )
    assert "sha256:" + material.completion_receipt.receipt_root.hex() == (
        "sha256:c75d08e038dc17c01aa4c9c6b69a2d4ef95034901a55e7982fb6a93d78f7798a"
    )
    assert len(material.q1_output_slots) == 8
    assert all(material.q1_output_slots)


def test_run_with_wrong_expected_capsule_digest_is_preconsumption_pending(
    tmp_path: Path,
    actual_v7r1_material: Any,
) -> None:
    contract = _temporary_contract(tmp_path)
    capsule = _capsule(contract, actual_v7r1_material)
    _publish_capsule(contract, capsule)

    result = tool.run_v8c(contract.path, "sha256:" + "00" * 32)

    assert result["status"] == tool.state.PENDING_NOT_CONSUMED
    assert result["condition"] == "CAPSULE_REPLAY_DIFFERS"
    assert result["authorization_consumption_count"] == 0
    assert result["logical_campaign_count"] == 0
    assert result["popen_count"] == 0
    assert result["actor_start_count"] == 0
    assert result["docker_invocation_count"] == 0
    fixed = contract.value["fixed_paths"]
    assert not Path(fixed["authorization_path"]).exists()
    assert not Path(fixed["consumption_path"]).exists()
    assert not Path(fixed["artifact_path"]).exists()


def test_run_without_authorization_is_not_consumed(
    tmp_path: Path,
    actual_v7r1_material: Any,
) -> None:
    contract = _temporary_contract(tmp_path)
    capsule = _capsule(contract, actual_v7r1_material)
    _publish_capsule(contract, capsule)

    result = tool.run_v8c(contract.path, capsule["capsule_digest"])

    assert result["status"] == (
        "Q1_COMPLETION_REPLAY_V8C_NOT_AUTHORIZED_NOT_CONSUMED"
    )
    assert result["capsule_digest"] == capsule["capsule_digest"]
    assert result["authorization_consumption_count"] == 0
    assert result["logical_campaign_count"] == 0
    assert result["q1_state"] == "NOT_RUN"
    assert result["q1_gate_count"] == 14
    assert result["q1_output_slots"] == [None] * 8
    fixed = contract.value["fixed_paths"]
    assert not Path(fixed["consumption_path"]).exists()
    assert not Path(fixed["terminal_path"]).exists()


def test_capsule_expected_completion_digest_mismatch_stays_preconsumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    actual_v7r1_material: Any,
) -> None:
    contract = _temporary_contract(tmp_path)
    capsule = _capsule(
        contract,
        actual_v7r1_material,
        artifact_raw_sha256=_sha("wrong completion artifact"),
    )
    _publish_capsule(contract, capsule)
    authorization = _publish_authorization(contract, capsule)
    _isolate_replay(monkeypatch, actual_v7r1_material)

    result = tool.run_v8c(contract.path, capsule["capsule_digest"])

    assert result["status"] == tool.state.PENDING_NOT_CONSUMED
    assert result["condition"] == "CAPSULE_EXPECTED_COMPLETION_DIFFERS"
    assert result["authorization_digest"] == authorization[
        "authorization_digest"
    ]
    assert result["authorization_consumption_count"] == 0
    assert result["logical_campaign_count"] == 0
    assert result["consumption_record_digest"] is None
    fixed = contract.value["fixed_paths"]
    assert not Path(fixed["consumption_path"]).exists()
    assert not Path(fixed["terminal_path"]).exists()
    assert not Path(fixed["artifact_path"]).exists()


@pytest.mark.parametrize("corrupt_record", ["capsule", "authorization"])
def test_preexisting_consumption_is_never_reported_as_preconsumption_pending(
    tmp_path: Path,
    corrupt_record: str,
) -> None:
    contract = _temporary_contract(tmp_path)
    material = _synthetic_material()
    capsule = _capsule(contract, material)
    _publish_capsule(contract, capsule)
    authorization = _publish_authorization(contract, capsule)
    _publish_consumption(contract, capsule, authorization)
    fixed = contract.value["fixed_paths"]

    if corrupt_record == "capsule":
        corrupted = deepcopy(capsule)
        corrupted["expected_completion_receipt_root"] = _sha(
            "corrupt capsule receipt"
        )
        corrupt_path = Path(fixed["capsule_path"])
    else:
        corrupted = deepcopy(authorization)
        corrupted["authorization_digest"] = _sha(
            "corrupt authorization digest"
        )
        corrupt_path = Path(fixed["authorization_path"])
    corrupt_path.write_bytes(tool.state.canonical_json_bytes(corrupted))

    result = tool.run_v8c(contract.path, capsule["capsule_digest"])

    assert result["status"] == (
        "Q1_COMPLETION_REPLAY_V8C_UNSEALED_TERMINAL_NO_RESUME"
    )
    assert result["status"] != tool.state.PENDING_NOT_CONSUMED
    assert result["status"] != tool.state.TERMINAL_NO_RESUME
    assert result["authorization_consumption_count"] == 1
    assert result["logical_campaign_count"] == 1
    assert Path(fixed["consumption_path"]).exists()
    assert not Path(fixed["terminal_path"]).exists()


def test_failed_terminal_publication_is_not_returned_as_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _temporary_contract(tmp_path)
    material = _synthetic_material()
    capsule = _capsule(contract, material)
    _publish_capsule(contract, capsule)
    authorization = _publish_authorization(contract, capsule)
    _publish_consumption(contract, capsule, authorization)
    _isolate_replay(monkeypatch, material)
    fixed = contract.value["fixed_paths"]
    terminal_path = Path(fixed["terminal_path"])
    publish_json = tool._publish_json

    def fail_terminal_publication(
        path: Path,
        value: dict[str, Any],
        digest: str,
    ) -> None:
        if path == terminal_path:
            raise tool.ControllerPending(
                "PATH_AND_FORMAT",
                "DURABLE_NO_REPLACE_PUBLICATION_FAILED",
                "injected terminal publication failure",
            )
        publish_json(path, value, digest)

    monkeypatch.setattr(
        tool,
        "_validate_expected_completion",
        lambda *_args: tool._terminal(
            "INJECTED_POSTCONSUMPTION_FAILURE",
            "force failed-terminal publication",
        ),
    )
    monkeypatch.setattr(tool, "_publish_json", fail_terminal_publication)

    result = tool.run_v8c(contract.path, capsule["capsule_digest"])

    assert result["status"] == (
        "Q1_COMPLETION_REPLAY_V8C_PARTIAL_FINALIZATION_RECOVERY_REQUIRED"
    )
    assert result["status"] != tool.state.PENDING_NOT_CONSUMED
    assert result["status"] != tool.state.TERMINAL_NO_RESUME
    assert result["authorization_consumption_count"] == 1
    assert result["logical_campaign_count"] == 1
    assert Path(fixed["outcome_path"]).exists()
    assert not terminal_path.exists()


def test_success_publication_lifecycle_uses_zero_execution_and_temp_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    actual_v7r1_material: Any,
) -> None:
    contract = _temporary_contract(tmp_path)
    capsule = _capsule(contract, actual_v7r1_material)
    _publish_capsule(contract, capsule)
    authorization = _publish_authorization(contract, capsule)
    _isolate_replay(monkeypatch, actual_v7r1_material)

    terminal = tool.run_v8c(contract.path, capsule["capsule_digest"])

    fixed = contract.value["fixed_paths"]
    consumption = tool.state.strict_json_bytes(
        Path(fixed["consumption_path"]).read_bytes(), "test consumption"
    )
    assert tool.state.validate_consumption(
        consumption, contract, capsule, authorization
    ) == consumption
    assert tool.state.validate_terminal(
        terminal, contract, capsule, authorization, consumption
    ) == terminal
    assert terminal["status"] == tool.state.COMPLETE
    assert terminal["outcome"]["q1_state"] == "COMPLETE"
    assert terminal["outcome"]["q1_gate_count"] == 20
    assert terminal["outcome"]["q1_gate_mask"] == 1_048_575
    assert terminal["outcome"]["q1_output_slots"] == list(
        actual_v7r1_material.q1_output_slots
    )
    assert terminal["outcome"]["q2_state"] == "NOT_RUN"
    assert terminal["outcome"]["popen_count"] == 0
    assert terminal["outcome"]["actor_start_count"] == 0
    assert terminal["outcome"]["docker_invocation_count"] == 0
    assert Path(fixed["artifact_path"]).read_bytes() == (
        actual_v7r1_material.completion_artifact
    )
    assert Path(fixed["outcome_path"]).exists()
    assert Path(fixed["terminal_path"]).exists()
