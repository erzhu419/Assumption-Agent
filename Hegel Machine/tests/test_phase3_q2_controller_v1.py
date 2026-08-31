from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import phase3_q2_role_membership_v1 as controller


CAPSULE_DIGEST = "sha256:" + "11" * 32


@contextmanager
def _lock(_contract: object):
    yield


def _harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[object, dict, dict, dict]:
    fixed = {
        "authorization_path": str(tmp_path / "authorization.json"),
        "consumption_path": str(tmp_path / "consumption.json"),
        "artifact_path": str(tmp_path / "artifact.cbor"),
        "terminal_path": str(tmp_path / "terminal.json"),
        "pending_path": str(tmp_path / "pending.json"),
        "trajectory_path": str(tmp_path / "trajectory.jsonl"),
    }
    contract = SimpleNamespace(value={"fixed_paths": fixed})
    capsule = {
        "capsule_digest": CAPSULE_DIGEST,
        "preflight_receipt": {"receipt": "exact"},
    }
    authorization = {"authorization_digest": "sha256:" + "22" * 32}
    consumption = {"record_digest": "sha256:" + "33" * 32}
    monkeypatch.setattr(controller, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(controller, "_ensure_roots", lambda _contract: None)
    monkeypatch.setattr(controller, "_campaign_lock", _lock)
    monkeypatch.setattr(
        controller,
        "_load_preregistration",
        lambda _contract: (b"prereg", {}),
    )
    monkeypatch.setattr(
        controller,
        "_read_capsule",
        lambda _contract, _prereg, _digest: capsule,
    )
    monkeypatch.setattr(
        controller,
        "_read_authorization",
        lambda _contract, _prereg, _capsule: authorization,
    )
    monkeypatch.setattr(controller, "_fresh_absence", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_replay_source_closure", lambda _capsule: None)
    monkeypatch.setattr(controller, "_resource_gate", lambda *_args: None)
    monkeypatch.setattr(controller, "_append_trajectory", lambda *args, **kwargs: None)
    return contract, capsule, authorization, consumption


def test_measurement_is_entered_only_after_consumption_publish_and_readback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, consumption = _harness(monkeypatch, tmp_path)
    read_count = 0
    published = False

    def read_consumption(*_args):
        nonlocal read_count
        read_count += 1
        return None if read_count == 1 else consumption

    def publish_json(path: Path, value: dict, _marker: str) -> None:
        nonlocal published
        assert path == Path(contract.value["fixed_paths"]["consumption_path"])
        assert value == consumption
        published = True

    def execute(*_args):
        assert published
        assert read_count == 2
        return b"artifact", object()

    monkeypatch.setattr(controller, "_read_consumption", read_consumption)
    monkeypatch.setattr(controller, "_read_terminal", lambda *_args: None)
    monkeypatch.setattr(
        controller,
        "_target_blind_replay",
        lambda *_args: (object(), (object(), object()), capsule["preflight_receipt"]),
    )
    monkeypatch.setattr(controller.state, "build_consumption", lambda *_args: consumption)
    monkeypatch.setattr(controller, "_publish_json", publish_json)
    monkeypatch.setattr(controller, "_execute_measurement", execute)
    monkeypatch.setattr(
        controller,
        "_publish_complete",
        lambda *_args: {"status": "COMPLETE"},
    )

    assert controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST) == {
        "status": "COMPLETE"
    }


def test_consumption_publication_failure_before_link_remains_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, capsule, _, consumption = _harness(monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "_read_consumption", lambda *_args: None)
    monkeypatch.setattr(
        controller,
        "_target_blind_replay",
        lambda *_args: (object(), (object(), object()), capsule["preflight_receipt"]),
    )
    monkeypatch.setattr(controller.state, "build_consumption", lambda *_args: consumption)
    monkeypatch.setattr(
        controller,
        "_publish_json",
        lambda *_args: (_ for _ in ()).throw(
            controller.ControllerPending(
                "LOCAL_STATE", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", "before link"
            )
        ),
    )
    monkeypatch.setattr(
        controller,
        "_execute_measurement",
        lambda *_args: pytest.fail("measurement must remain unopened"),
    )
    monkeypatch.setattr(
        controller,
        "_publish_pending",
        lambda *_args, **_kwargs: {"status": controller.state.PENDING_NOT_CONSUMED},
    )

    result = controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST)

    assert result["status"] == controller.state.PENDING_NOT_CONSUMED
    assert not Path(tmp_path / "consumption.json").exists()


def test_durable_consumption_with_failed_readback_is_terminal_no_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, capsule, _, consumption = _harness(monkeypatch, tmp_path)
    read_count = 0

    def read_consumption(*_args):
        nonlocal read_count
        read_count += 1
        if read_count == 1:
            return None
        raise controller.ControllerTerminal("Q2_CONSUMPTION_REPLAY_DIFFERS", "bad")

    def publish(path: Path, _value: dict, _marker: str) -> None:
        assert path == Path(contract.value["fixed_paths"]["consumption_path"])
        path.write_text("durable-but-unreadable", encoding="ascii")

    monkeypatch.setattr(controller, "_read_consumption", read_consumption)
    monkeypatch.setattr(
        controller,
        "_target_blind_replay",
        lambda *_args: (object(), (object(), object()), capsule["preflight_receipt"]),
    )
    monkeypatch.setattr(controller.state, "build_consumption", lambda *_args: consumption)
    monkeypatch.setattr(controller, "_publish_json", publish)
    monkeypatch.setattr(
        controller,
        "_execute_measurement",
        lambda *_args: pytest.fail("measurement must remain unopened"),
    )

    result = controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST)

    assert result["status"] == controller.UNSEALED_TERMINAL
    assert result["authorization_consumption_count"] == 1
    assert result["terminal_no_resume"] is True


def test_valid_artifact_after_terminal_failure_requires_partial_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, capsule, _, consumption = _harness(monkeypatch, tmp_path)
    read_count = 0

    def read_consumption(*_args):
        nonlocal read_count
        read_count += 1
        return None if read_count == 1 else consumption

    def fail_after_artifact(*_args):
        Path(tmp_path / "artifact.cbor").write_bytes(b"sealed")
        raise RuntimeError("terminal publication failed")

    monkeypatch.setattr(controller, "_read_consumption", read_consumption)
    monkeypatch.setattr(
        controller,
        "_target_blind_replay",
        lambda *_args: (object(), (object(), object()), capsule["preflight_receipt"]),
    )
    monkeypatch.setattr(controller.state, "build_consumption", lambda *_args: consumption)
    monkeypatch.setattr(controller, "_publish_json", lambda *_args: None)
    monkeypatch.setattr(controller, "_execute_measurement", lambda *_args: (b"sealed", object()))
    monkeypatch.setattr(controller, "_publish_complete", fail_after_artifact)
    monkeypatch.setattr(controller, "_read_artifact", lambda *_args: (b"sealed", object()))

    result = controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST)

    assert result["status"] == controller.PARTIAL_FINALIZATION
    assert result["exact_artifact_to_terminal_reentry_allowed"] is True
    assert result["target_or_role_reentry_allowed"] is False


def test_preconsumption_pending_never_enters_measurement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, _, _ = _harness(monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "_read_consumption", lambda *_args: None)
    monkeypatch.setattr(
        controller,
        "_target_blind_replay",
        lambda *_args: (_ for _ in ()).throw(
            controller.ControllerPending(
                "Q1_ARCHIVE", "Q2_Q1_ARCHIVE_UNAVAILABLE", "repairable"
            )
        ),
    )
    monkeypatch.setattr(
        controller,
        "_execute_measurement",
        lambda *_args: pytest.fail("measurement must remain unopened"),
    )
    monkeypatch.setattr(
        controller,
        "_publish_pending",
        lambda *_args, **_kwargs: {"status": controller.state.PENDING_NOT_CONSUMED},
    )
    result = controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST)
    assert result["status"] == controller.state.PENDING_NOT_CONSUMED


def test_consumed_valid_artifact_finalizes_without_archive_or_measurement_reentry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, _, consumption = _harness(monkeypatch, tmp_path)
    Path(tmp_path / "consumption.json").write_text("consumed", encoding="ascii")
    monkeypatch.setattr(controller, "_read_consumption", lambda *_args: consumption)
    monkeypatch.setattr(controller, "_read_terminal", lambda *_args: None)
    monkeypatch.setattr(controller, "_read_artifact", lambda *_args: (b"sealed", object()))
    monkeypatch.setattr(
        controller,
        "_target_blind_replay",
        lambda *_args: pytest.fail("Q1 archive must not be reread"),
    )
    monkeypatch.setattr(
        controller,
        "_execute_measurement",
        lambda *_args: pytest.fail("measurement must not reenter"),
    )
    monkeypatch.setattr(
        controller,
        "_publish_complete",
        lambda *_args: {"status": "FINALIZED"},
    )
    assert controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST) == {
        "status": "FINALIZED"
    }


def test_consumed_source_drift_blocks_terminal_and_artifact_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, _, _ = _harness(monkeypatch, tmp_path)
    Path(tmp_path / "consumption.json").write_text("consumed", encoding="ascii")
    monkeypatch.setattr(
        controller,
        "_replay_source_closure",
        lambda *_args: (_ for _ in ()).throw(
            controller.ControllerPending(
                "SOURCE_CLOSURE", "Q2_LOCAL_IMPLEMENTATION_UNAVAILABLE", "drift"
            )
        ),
    )
    monkeypatch.setattr(
        controller,
        "_read_terminal",
        lambda *_args: pytest.fail("terminal must not be interpreted after source drift"),
    )
    monkeypatch.setattr(
        controller,
        "_read_artifact",
        lambda *_args: pytest.fail("artifact must not be interpreted after source drift"),
    )

    result = controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST)

    assert result["status"] == controller.UNSEALED_TERMINAL
    assert result["authorization_consumption_count"] == 1
    assert result["terminal_no_resume"] is True


def test_consumed_without_artifact_invalidates_without_measurement_reentry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, _, consumption = _harness(monkeypatch, tmp_path)
    Path(tmp_path / "consumption.json").write_text("consumed", encoding="ascii")
    monkeypatch.setattr(controller, "_read_consumption", lambda *_args: consumption)
    monkeypatch.setattr(controller, "_read_terminal", lambda *_args: None)
    monkeypatch.setattr(controller, "_read_artifact", lambda *_args: None)
    monkeypatch.setattr(
        controller,
        "_execute_measurement",
        lambda *_args: pytest.fail("measurement must not reenter"),
    )
    monkeypatch.setattr(
        controller,
        "_publish_invalidated",
        lambda *_args, **_kwargs: {"status": "INVALIDATED"},
    )
    assert controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST) == {
        "status": "INVALIDATED"
    }


def test_consumption_path_prevents_missing_authorization_from_looking_unconsumed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, _, _ = _harness(monkeypatch, tmp_path)
    Path(tmp_path / "consumption.json").write_text("consumed", encoding="ascii")
    monkeypatch.setattr(controller, "_read_authorization", lambda *_args: None)

    result = controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST)

    assert result["status"] == controller.UNSEALED_TERMINAL
    assert result["authorization_consumption_count"] == 1
    assert result["terminal_no_resume"] is True


def test_unreadable_existing_consumption_is_terminal_not_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, _, _ = _harness(monkeypatch, tmp_path)
    Path(tmp_path / "consumption.json").write_text("unreadable", encoding="ascii")

    def unreadable(*_args):
        raise controller.ControllerTerminal("Q2_CONSUMPTION_REPLAY_DIFFERS", "bad")

    monkeypatch.setattr(controller, "_read_consumption", unreadable)

    result = controller.run_v1(tmp_path / "contract.json", CAPSULE_DIGEST)

    assert result["status"] == controller.UNSEALED_TERMINAL
    assert result["authorization_consumption_count"] == 1
    assert result["terminal_no_resume"] is True


def test_pending_trajectory_failure_returns_rewritten_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, capsule, _, _ = _harness(monkeypatch, tmp_path)
    writes: list[dict] = []

    def build_pending(*_args, **kwargs):
        return {
            "status": controller.state.PENDING_NOT_CONSUMED,
            "pending_digest": "sha256:" + "44" * 32,
            "phase": kwargs["phase"],
            "condition": kwargs["condition"],
        }

    monkeypatch.setattr(controller.state, "build_pending", build_pending)
    monkeypatch.setattr(
        controller,
        "_replace_json",
        lambda _path, value, _marker: writes.append(dict(value)),
    )
    monkeypatch.setattr(
        controller,
        "_append_trajectory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            controller.ControllerPending(
                "TRAJECTORY", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", "disk"
            )
        ),
    )

    result = controller._publish_pending(
        contract,
        {},
        capsule,
        controller.ControllerPending(
            "Q1_ARCHIVE", "Q2_Q1_ARCHIVE_UNAVAILABLE", "repairable"
        ),
    )

    assert result["status"] == controller.state.PENDING_NOT_CONSUMED
    assert result["phase"] == "TRAJECTORY"
    assert len(writes) == 2


def test_pending_record_failure_returns_non_consuming_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, capsule, _, _ = _harness(monkeypatch, tmp_path)
    monkeypatch.setattr(
        controller.state,
        "build_pending",
        lambda *_args, **_kwargs: {
            "status": controller.state.PENDING_NOT_CONSUMED,
            "pending_digest": "sha256:" + "55" * 32,
        },
    )
    monkeypatch.setattr(
        controller,
        "_replace_json",
        lambda *_args: (_ for _ in ()).throw(
            controller.ControllerPending(
                "LOCAL_STATE", "Q2_LOCAL_PUBLICATION_UNAVAILABLE", "read-only"
            )
        ),
    )

    result = controller._publish_pending(
        contract,
        {},
        capsule,
        controller.ControllerPending(
            "Q1_ARCHIVE", "Q2_Q1_ARCHIVE_UNAVAILABLE", "repairable"
        ),
    )

    assert result["status"] == controller.PENDING_PUBLICATION_UNAVAILABLE
    assert result["authorization_consumption_count"] == 0
    assert result["logical_campaign_count"] == 0


def test_invalid_artifact_can_be_sealed_by_invalidated_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract, capsule, authorization, consumption = _harness(monkeypatch, tmp_path)
    Path(tmp_path / "artifact.cbor").write_bytes(b"invalid")
    terminal = {
        "status": controller.state.INVALIDATED_TERMINAL_NO_RESUME,
        "record_digest": "sha256:" + "66" * 32,
    }
    published: list[dict] = []
    monkeypatch.setattr(
        controller,
        "_read_artifact",
        lambda *_args: (_ for _ in ()).throw(
            controller.ControllerTerminal("Q2_SEALED_ARTIFACT_REPLAY_DIFFERS", "bad")
        ),
    )
    monkeypatch.setattr(
        controller.state,
        "build_invalidated_terminal",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        controller,
        "_publish_json",
        lambda _path, value, _marker: published.append(dict(value)),
    )
    monkeypatch.setattr(controller, "_read_terminal", lambda *_args: terminal)

    result = controller._publish_invalidated(
        contract,
        {},
        capsule,
        authorization,
        consumption,
        controller.ControllerTerminal("Q2_SEALED_ARTIFACT_REPLAY_DIFFERS", "bad"),
        action_audit=controller._empty_action_audit(),
    )

    assert result == terminal
    assert published == [terminal]


def test_source_closure_includes_target_generator_and_canonical_wire() -> None:
    paths = controller._source_paths()
    assert "src/hegel_machine/phase3_m25_rows_v1.py" in paths
    assert "src/hegel_machine/strict_cbor_v1.py" in paths
    assert "tools/phase3_q2_role_membership_v1.py" in paths


def test_peak_rss_uses_current_exec_vm_hwm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    status = tmp_path / "status"
    status.write_text("Name:\tpython3.10\nVmHWM:\t12345 kB\n", encoding="ascii")
    monkeypatch.setattr(controller, "PROC_SELF_STATUS", status)

    assert controller._peak_rss_bytes() == 12_641_280
