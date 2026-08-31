from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_SPEC = importlib.util.spec_from_file_location(
    "phase3_q3_bounded_quotient_certificate_tool_test_controller",
    ROOT / "tools/phase3_q3_bounded_quotient_certificate_v1.py",
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(TOOL_SPEC)
sys.modules[TOOL_SPEC.name] = tool
TOOL_SPEC.loader.exec_module(tool)
state = tool.state


def _temporary_contract(tmp_path: Path) -> Path:
    actual = state.load_contract(tool.DEFAULT_CONTRACT)
    value = deepcopy(actual.value)
    run_root = tmp_path / "q3-run"
    ledger_root = tmp_path / "q3-ledger"
    value["fixed_paths"] = {
        "project_root": str(ROOT),
        "campaign_lock_path": str(tmp_path / "q3.lock"),
        "run_root": str(run_root),
        "ledger_root": str(ledger_root),
        "capsule_path": str(run_root / "capsule.json"),
        "authorization_path": str(ledger_root / "owner.authorization.json"),
        "consumption_path": str(ledger_root / "campaign.consumed.json"),
        "pending_path": str(run_root / "latest.pending.json"),
        "terminal_path": str(ledger_root / "campaign.terminal.json"),
        "artifact_path": str(tmp_path / "artifacts" / "q3.cbor"),
        "trajectory_path": str(run_root / "trajectory.jsonl"),
    }
    q2_root = tmp_path / "q2-selected"
    q2_root.mkdir()
    selected = tool._read_q2_selected(actual)
    value["q2_inputs"] = {}
    for kind, payload in selected.items():
        suffix = ".cbor" if kind == "artifact" else ".json"
        path = q2_root / f"{kind}{suffix}"
        path.write_bytes(payload)
        value["q2_inputs"][kind] = str(path)
    path = tmp_path / "q3-contract.json"
    path.write_bytes(state.canonical_json_bytes(value))
    return path


def test_source_closure_is_exact_four_row_producer_free_graph() -> None:
    contract = state.load_contract(tool.DEFAULT_CONTRACT)
    rows = tool._source_closure_rows(contract)
    assert [row["relative_path"] for row in rows] == list(state.Q3_SOURCE_PATHS)
    assert len(rows) == 4
    assert len({row["relative_path"] for row in rows}) == 4
    assert state.source_closure_root(rows).startswith("sha256:")
    forbidden = (
        "phase3_q2_role_membership_v1",
        "phase3_q2_campaign_v1",
        "phase3_q2_archive_reader_v1",
        "phase3_m25_rows_v1",
        "phase3_m25_split_v1",
        "phase3_m3_start",
        "phase3_q1_archive",
    )
    assert not any(
        token in row["relative_path"] for row in rows for token in forbidden
    )


def test_preflight_reads_exact_five_q2_files_and_stops_not_authorized(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract_path = _temporary_contract(tmp_path)
    contract = state.load_contract(contract_path)
    q2_paths = {
        Path(path): kind for kind, path in contract.value["q2_inputs"].items()
    }
    observed: list[str] = []
    original = tool._read_regular

    def recording_read(path: Path, label: str, maximum_bytes: int = state.MAX_JSON_BYTES) -> bytes:
        if path in q2_paths:
            observed.append(q2_paths[path])
        forbidden_fragments = ("target", "split", "truth", "q1-archive")
        assert not any(fragment in str(path) for fragment in forbidden_fragments)
        return original(path, label, maximum_bytes)

    monkeypatch.setattr(tool, "_read_regular", recording_read)
    capsule = tool.prepare_capsule(
        contract_path,
        "3" * 40,
        run_id="4" * 32,
        created_at_unix_seconds=1_788_200_100,
    )
    assert observed == [
        "capsule",
        "authorization",
        "consumption",
        "terminal",
        "artifact",
    ]
    receipt = capsule["preflight_receipt"]
    assert receipt["selected_q2_input_total_bytes"] == 22_898
    assert receipt["q3_action_audit"] == state.ZERO_ACTION_AUDIT
    assert capsule["status"] == state.CAPSULE_READY
    assert capsule["source_commit"] == "3" * 40

    fixed = contract.value["fixed_paths"]
    assert Path(fixed["capsule_path"]).exists()
    assert Path(fixed["trajectory_path"]).exists()
    for key in (
        "authorization_path",
        "consumption_path",
        "artifact_path",
        "terminal_path",
        "pending_path",
    ):
        assert not Path(fixed[key]).exists()
    assert not Path(fixed["ledger_root"]).exists()


def test_same_capsule_preflight_reentry_remains_zero_campaign(tmp_path: Path) -> None:
    contract_path = _temporary_contract(tmp_path)
    first = tool.prepare_capsule(
        contract_path,
        "5" * 40,
        run_id="6" * 32,
        created_at_unix_seconds=1_788_200_200,
    )
    second = tool.prepare_capsule(
        contract_path,
        "5" * 40,
        run_id="7" * 32,
        created_at_unix_seconds=1_788_200_300,
    )
    assert second == first
    assert second["preflight_receipt"]["q3_action_audit"] == state.ZERO_ACTION_AUDIT
    contract = state.load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    assert not Path(fixed["authorization_path"]).exists()
    assert not Path(fixed["consumption_path"]).exists()


def test_authorization_builder_in_temporary_tree_still_does_not_consume(
    tmp_path: Path,
) -> None:
    contract_path = _temporary_contract(tmp_path)
    capsule = tool.prepare_capsule(
        contract_path,
        "8" * 40,
        run_id="9" * 32,
        created_at_unix_seconds=1_788_200_400,
    )
    authorization = tool.record_authorization(
        contract_path, capsule["capsule_digest"]
    )
    assert authorization["status"] == state.AUTHORIZATION_RECORDED
    contract = state.load_contract(contract_path)
    fixed = contract.value["fixed_paths"]
    assert Path(fixed["authorization_path"]).exists()
    assert not Path(fixed["consumption_path"]).exists()
    assert not Path(fixed["artifact_path"]).exists()
    assert not Path(fixed["terminal_path"]).exists()


def test_tool_has_no_process_network_or_legacy_m3_surface() -> None:
    source = (ROOT / "tools/phase3_q3_bounded_quotient_certificate_v1.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "subprocess",
        "docker",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "phase3_m3_start",
        "phase3_q2_role_membership_v1",
        "phase3_q2_campaign_v1",
        "phase3_m25_rows_v1",
    ):
        assert forbidden not in source
    assert ".Popen(" not in source


def _prepared_and_authorized(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    contract_path = _temporary_contract(tmp_path)
    capsule = tool.prepare_capsule(
        contract_path,
        "a" * 40,
        run_id="b" * 32,
        created_at_unix_seconds=1_788_200_500,
    )
    tool.record_authorization(contract_path, capsule["capsule_digest"])
    return contract_path, capsule


def test_run_completes_once_replays_terminal_and_clears_pending(
    tmp_path: Path,
) -> None:
    contract_path, capsule = _prepared_and_authorized(tmp_path)
    contract = state.load_contract(contract_path)
    pending_path = Path(contract.value["fixed_paths"]["pending_path"])
    authorization = state.build_authorization(
        contract,
        state.load_preregistration(Path(contract.value["preregistration_path"])).value,
        capsule,
    )
    pending = state.build_pending(
        contract,
        state.load_preregistration(Path(contract.value["preregistration_path"])).value,
        capsule,
        phase="LOCAL_READ_ONLY_REPLAY",
        condition="BOUND_LOCAL_FILE_UNAVAILABLE",
        detail="synthetic pre-consumption checkpoint",
        authorization=authorization,
    )
    pending_path.write_bytes(state.canonical_json_bytes(pending))

    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == tool.wire.Q3_SUCCESS_STATUS
    assert terminal["authorization_consumption_count"] == 1
    assert terminal["logical_campaign_count"] == 1
    assert terminal["outcome"]["q3_action_audit"] == state.ZERO_ACTION_AUDIT
    assert not pending_path.exists()
    assert tool.run_v1(contract_path, capsule["capsule_digest"]) == terminal


def test_valid_artifact_terminal_publication_failure_is_exactly_resumable(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract_path, capsule = _prepared_and_authorized(tmp_path)
    contract = state.load_contract(contract_path)
    terminal_path = Path(contract.value["fixed_paths"]["terminal_path"])
    original_publish = tool._publish_json

    def fail_terminal(path: Path, value: Any, label: str) -> None:
        if path == terminal_path:
            raise tool.ControllerPending(
                "LOCAL_PUBLICATION",
                "Q3_LOCAL_PUBLICATION_UNAVAILABLE",
                "synthetic terminal publication failure",
            )
        original_publish(path, value, label)

    monkeypatch.setattr(tool, "_publish_json", fail_terminal)
    partial = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert partial["status"] == state.PARTIAL_FINALIZATION_REQUIRED
    assert partial["authorization_consumption_count"] == 1
    assert partial["logical_campaign_count"] == 1
    assert partial["exact_artifact_to_terminal_reentry_allowed"] is True
    assert not terminal_path.exists()

    monkeypatch.setattr(tool, "_publish_json", original_publish)
    monkeypatch.setattr(
        tool,
        "_replay_q2",
        lambda *_args, **_kwargs: pytest.fail(
            "partial finalization must not replay Q2"
        ),
    )
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == tool.wire.Q3_SUCCESS_STATUS


def test_forged_existing_terminal_is_never_accepted_as_complete(
    tmp_path: Path,
) -> None:
    contract_path, capsule = _prepared_and_authorized(tmp_path)
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    contract = state.load_contract(contract_path)
    terminal_path = Path(contract.value["fixed_paths"]["terminal_path"])
    forged = deepcopy(terminal)
    forged["outcome"]["campaign_complete"] = False
    terminal_path.write_bytes(state.canonical_json_bytes(forged))

    result = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert result["status"] == state.PARTIAL_FINALIZATION_REQUIRED
    assert result["authorization_consumption_count"] == 1
    assert result["logical_campaign_count"] == 1


def test_malformed_existing_terminal_is_classified_as_partial_finalization(
    tmp_path: Path,
) -> None:
    contract_path, capsule = _prepared_and_authorized(tmp_path)
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == tool.wire.Q3_SUCCESS_STATUS
    contract = state.load_contract(contract_path)
    terminal_path = Path(contract.value["fixed_paths"]["terminal_path"])
    terminal_path.write_bytes(b"{\n")

    result = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert result["status"] == state.PARTIAL_FINALIZATION_REQUIRED
    assert result["authorization_consumption_count"] == 1
    assert result["logical_campaign_count"] == 1


def test_post_write_terminal_parent_fsync_failure_never_reports_zero(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract_path, capsule = _prepared_and_authorized(tmp_path)
    contract = state.load_contract(contract_path)
    terminal_path = Path(contract.value["fixed_paths"]["terminal_path"])
    original_fsync_parent = tool._fsync_parent

    def fail_terminal_parent(path: Path) -> None:
        if path == terminal_path:
            raise OSError("synthetic terminal parent fsync failure")
        original_fsync_parent(path)

    monkeypatch.setattr(tool, "_fsync_parent", fail_terminal_parent)
    result = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert result["status"] == state.PARTIAL_FINALIZATION_REQUIRED
    assert result["authorization_consumption_count"] == 1
    assert result["logical_campaign_count"] == 1
    assert terminal_path.exists()

    monkeypatch.setattr(tool, "_fsync_parent", original_fsync_parent)
    replayed = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert replayed["status"] == tool.wire.Q3_SUCCESS_STATUS


def test_consumed_state_without_authorization_never_falls_back_to_zero(
    tmp_path: Path,
) -> None:
    contract_path, capsule = _prepared_and_authorized(tmp_path)
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == tool.wire.Q3_SUCCESS_STATUS
    contract = state.load_contract(contract_path)
    Path(contract.value["fixed_paths"]["authorization_path"]).unlink()

    result = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert result["status"] == state.INVALIDATED_TERMINAL_NO_RESUME
    assert result["authorization_consumption_count"] == 1
    assert result["logical_campaign_count"] == 1
    assert result["counts_are_conservative"] is True


def test_consumed_state_with_unreadable_capsule_never_falls_back_to_zero(
    tmp_path: Path,
) -> None:
    contract_path, capsule = _prepared_and_authorized(tmp_path)
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == tool.wire.Q3_SUCCESS_STATUS
    contract = state.load_contract(contract_path)
    Path(contract.value["fixed_paths"]["capsule_path"]).write_bytes(b"{\n")

    result = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert result["status"] == state.INVALIDATED_TERMINAL_NO_RESUME
    assert result["authorization_consumption_count"] == 1
    assert result["logical_campaign_count"] == 1
    assert result["counts_are_conservative"] is True


def test_fresh_lock_acquire_failure_remains_preconsumption_pending(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract_path, capsule = _prepared_and_authorized(tmp_path)

    def fail_lock(_descriptor: int, _operation: int) -> None:
        raise OSError("synthetic lock acquire failure")

    monkeypatch.setattr(tool.fcntl, "flock", fail_lock)
    with pytest.raises(tool.ControllerPending) as caught:
        tool.run_v1(contract_path, capsule["capsule_digest"])
    assert caught.value.phase == "CAMPAIGN_LOCK"
    contract = state.load_contract(contract_path)
    assert not Path(contract.value["fixed_paths"]["consumption_path"]).exists()


def test_consumed_lock_acquire_failure_never_falls_back_to_zero(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    contract_path, capsule = _prepared_and_authorized(tmp_path)
    terminal = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == tool.wire.Q3_SUCCESS_STATUS

    def fail_lock(_descriptor: int, _operation: int) -> None:
        raise OSError("synthetic lock acquire failure")

    monkeypatch.setattr(tool.fcntl, "flock", fail_lock)
    result = tool.run_v1(contract_path, capsule["capsule_digest"])
    assert result["status"] == state.INVALIDATED_TERMINAL_NO_RESUME
    assert result["authorization_consumption_count"] == 1
    assert result["logical_campaign_count"] == 1
    assert result["counts_are_conservative"] is True
