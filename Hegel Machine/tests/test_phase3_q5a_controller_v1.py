from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hegel_machine import phase3_q5a_completion_replay_v1 as state  # noqa: E402
from hegel_machine import phase3_q5a_purpose5_signer_genesis_v1 as wire  # noqa: E402


SPEC = importlib.util.spec_from_file_location(
    "phase3_q5a_controller_v1_under_test",
    ROOT / "tools/phase3_q5a_purpose5_signer_genesis_candidate_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


ACTUAL_CONTRACT = (
    ROOT / "config/phase3_q5a_purpose5_signer_genesis_candidate_actual_v1.json"
)
ACTUAL_PREREGISTRATION = (
    ROOT / "config/phase3_q5a_purpose5_signer_genesis_candidate_v1.json"
)


def _canonical(value: object) -> bytes:
    return state.canonical_json_bytes(value)


def _worker_receipt(ordinal: int, public_key: bytes) -> dict[str, object]:
    return {
        "key_epoch": 1,
        "key_generation_count": 1,
        "key_id_hex": state.sha256(public_key).digest()[:16].hex(),
        "openssl_popen_count": 2,
        "private_key_exported": False,
        "public_key_derivation_count": 1,
        "public_key_hex": public_key.hex(),
        "purpose_id": 5,
        "schema": state.WORKER_RECEIPT_SCHEMA,
        "signature_count": 0,
        "signer_ordinal": ordinal,
        "signer_role": state.SIGNER_ROLE_NAMES[ordinal - 1],
        "status": state.WORKER_SUCCESS_STATUS,
    }


def _smoke_receipt(ordinal: int) -> dict[str, object]:
    return {
        "expected_gid": 1000,
        "expected_uid": 1000,
        "key_generation_count": 0,
        "openssl_path": "/usr/bin/openssl",
        "openssl_popen_count": 0,
        "public_key_derivation_count": 0,
        "purpose_id": 5,
        "schema": (
            "hegel-phase3-q5a-purpose5-keygen-startup-smoke-receipt/1"
        ),
        "signature_count": 0,
        "signer_ordinal": ordinal,
        "signer_role": state.SIGNER_ROLE_NAMES[ordinal - 1],
        "status": "Q5A_PURPOSE5_KEYGEN_STARTUP_SMOKE_PASS",
    }


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class FakeWorker:
    """A deterministic public fixture; it never invokes OpenSSL or makes a key."""

    def __init__(self, *, fail_generate_ordinal: int | None = None) -> None:
        self.calls: list[tuple[str, int]] = []
        self.fail_generate_ordinal = fail_generate_ordinal

    def __call__(
        self, command: list[str] | tuple[str, ...]
    ) -> subprocess.CompletedProcess[bytes]:
        argv = list(command)
        ordinal = int(_option(argv, "--signer-ordinal"))
        if "--startup-smoke" in argv:
            self.calls.append(("smoke", ordinal))
            return subprocess.CompletedProcess(
                argv, 0, _canonical(_smoke_receipt(ordinal)), b""
            )

        assert "generate" in argv
        self.calls.append(("generate", ordinal))
        if self.fail_generate_ordinal == ordinal:
            return subprocess.CompletedProcess(
                argv, 70, b"", b"FAIL_FAKE_KEYGEN\n"
            )

        custody = Path(_option(argv, "--state-root"))
        receipt_root = Path(_option(argv, "--receipt-root"))
        output_root = Path(_option(argv, "--output-root"))
        for root in (custody, receipt_root, output_root):
            os.mkdir(root, 0o700)
            os.chmod(root, 0o700)

        # This is deliberately not a PEM or private key.  The controller is
        # supposed to inspect only its metadata, never these sentinel bytes.
        private_path = custody / controller.PRIVATE_KEY_NAME
        private_path.write_bytes(b"opaque-custody-fixture-not-a-key\n")
        private_path.chmod(0o600)

        public_key = bytes([ordinal]) * 32
        public_der = state.ED25519_SPKI_DER_PREFIX + public_key
        public_path = output_root / controller.PUBLIC_DER_NAME
        public_path.write_bytes(public_der)
        public_path.chmod(0o644)

        receipt = _worker_receipt(ordinal, public_key)
        receipt_path = receipt_root / controller.WORKER_RECEIPT_NAME
        receipt_path.write_bytes(_canonical(receipt))
        receipt_path.chmod(0o644)
        return subprocess.CompletedProcess(argv, 0, _canonical(receipt), b"")

    @property
    def generate_ordinals(self) -> list[int]:
        return [ordinal for kind, ordinal in self.calls if kind == "generate"]


@pytest.fixture
def isolated_contract(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    preregistration = json.loads(ACTUAL_PREREGISTRATION.read_text("utf-8"))
    preregistration_path = tmp_path / "q5a-preregistration.json"
    preregistration_raw = (
        json.dumps(preregistration, indent=2, ensure_ascii=True) + "\n"
    ).encode("ascii")
    preregistration_path.write_bytes(preregistration_raw)

    value = deepcopy(json.loads(ACTUAL_CONTRACT.read_text("utf-8")))
    run_root = tmp_path / "run"
    ledger_root = tmp_path / "ledger"
    fixed = value["fixed_paths"]
    fixed.update(
        {
            "project_root": str(ROOT),
            "campaign_lock_path": str(tmp_path / "campaign.lock"),
            "run_root": str(run_root),
            "ledger_root": str(ledger_root),
            "capsule_path": str(run_root / "capsule.json"),
            "authorization_path": str(ledger_root / "authorization.json"),
            "consumption_path": str(ledger_root / "consumption.json"),
            "pending_path": str(run_root / "pending.json"),
            "checkpoint_path": str(ledger_root / "checkpoint.json"),
            "terminal_path": str(ledger_root / "terminal.json"),
            "artifact_path": str(tmp_path / "candidate.cbor"),
            "trajectory_path": str(run_root / "trajectory.jsonl"),
        }
    )
    for ordinal in (1, 2, 3):
        fixed[f"signer_{ordinal}_intent_path"] = str(
            ledger_root / f"signer-{ordinal}.intent.json"
        )
        fixed[f"signer_{ordinal}_custody_root"] = str(
            tmp_path / f"signer-{ordinal}-custody"
        )
        fixed[f"signer_{ordinal}_receipt_root"] = str(
            ledger_root / f"signer-{ordinal}-receipt"
        )
        fixed[f"signer_{ordinal}_output_root"] = str(
            run_root / f"signer-{ordinal}-output"
        )

    value["preregistration_path"] = str(preregistration_path)
    value["preregistration_raw_sha256"] = state.raw_sha256(
        preregistration_raw
    )
    value["worker"] = {
        "python_executable": "/usr/bin/python3.10",
        "worker_path": str(
            ROOT / "tools/phase3_q5a_purpose5_keygen_worker_v1.py"
        ),
        "openssl_executable": "/usr/bin/openssl",
        "expected_uid": 1000,
        "expected_gid": 1000,
        "signer_order": [1, 2, 3],
        "signer_roles": list(state.SIGNER_ROLE_NAMES),
    }
    contract_path = tmp_path / "q5a-contract.json"
    contract_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    return contract_path, value


def _prepare_and_authorize(
    contract_path: Path, fake: FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Any]]:
    monkeypatch.setattr(controller, "_invoke_worker", fake)
    capsule = controller.prepare_capsule(
        contract_path,
        "a" * 40,
        run_id="b" * 32,
        created_at_unix_seconds=1_788_400_000,
    )
    authorization = controller.record_authorization(
        contract_path, capsule["capsule_digest"]
    )
    return capsule, authorization


def _trajectory(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("ascii").splitlines()]


def test_prepare_only_smokes_and_record_does_not_consume(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker()
    monkeypatch.setattr(controller, "_invoke_worker", fake)

    capsule = controller.prepare_capsule(
        contract_path,
        "a" * 40,
        run_id="b" * 32,
        created_at_unix_seconds=1_788_400_000,
    )

    assert fake.calls == [("smoke", 1), ("smoke", 2), ("smoke", 3)]
    assert not Path(value["fixed_paths"]["consumption_path"]).exists()
    for ordinal in (1, 2, 3):
        assert not Path(
            value["fixed_paths"][f"signer_{ordinal}_custody_root"]
        ).exists()

    authorization = controller.record_authorization(
        contract_path, capsule["capsule_digest"]
    )
    assert authorization["status"] == state.AUTHORIZATION_RECORDED
    assert not Path(value["fixed_paths"]["consumption_path"]).exists()
    assert fake.generate_ordinals == []


def test_success_is_ordered_and_controller_never_reads_private_bytes(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker()
    capsule, _authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name == controller.PRIVATE_KEY_NAME:
            raise AssertionError("controller opened private-key contents")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])

    assert terminal["status"] == wire.Q5A_SUCCESS_STATUS
    assert fake.generate_ordinals == [1, 2, 3]
    assert terminal["outcome"]["q5a_action_audit"] == state.SUCCESS_ACTION_AUDIT
    events = _trajectory(Path(value["fixed_paths"]["trajectory_path"]))
    assert [row["event"] for row in events] == [
        "CAPSULE_PREPARED_NOT_AUTHORIZED",
        "AUTHORIZATION_RECORDED_NOT_CONSUMED",
        "Q5A_AUTHORIZATION_CONSUMED",
        "SIGNER_1_PUBLIC_RECEIPT_COMPLETE",
        "SIGNER_2_PUBLIC_RECEIPT_COMPLETE",
        "SIGNER_3_PUBLIC_RECEIPT_COMPLETE",
        wire.Q5A_SUCCESS_STATUS,
    ]


def test_valid_worker_result_recovers_wrapper_without_regeneration(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, _value = isolated_contract
    fake = FakeWorker()
    capsule, _authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )
    original_publish_json = controller._publish_json
    failed = False

    def fail_first_outer(path: Path, value: Any, label: str, **kwargs: Any) -> None:
        nonlocal failed
        if path.name == controller.OUTER_RECEIPT_NAME and not failed:
            failed = True
            raise OSError("injected wrapper publication failure")
        original_publish_json(path, value, label, **kwargs)

    monkeypatch.setattr(controller, "_publish_json", fail_first_outer)
    with pytest.raises(controller.PublicResultPending):
        controller.run_v1(contract_path, capsule["capsule_digest"])
    assert fake.generate_ordinals == [1]

    monkeypatch.setattr(controller, "_publish_json", original_publish_json)
    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == wire.Q5A_SUCCESS_STATUS
    assert fake.generate_ordinals == [1, 2, 3]


def test_third_worker_wrapper_pending_preserves_complete_action_audit(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker()
    capsule, _authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )
    third_outer = (
        Path(value["fixed_paths"]["signer_3_receipt_root"])
        / controller.OUTER_RECEIPT_NAME
    )
    original_publish_json = controller._publish_json
    failed = False

    def fail_third_outer(path: Path, item: Any, label: str, **kwargs: Any) -> None:
        nonlocal failed
        if path == third_outer and not failed:
            failed = True
            raise OSError("injected third-wrapper publication failure")
        original_publish_json(path, item, label, **kwargs)

    monkeypatch.setattr(controller, "_publish_json", fail_third_outer)
    with pytest.raises(controller.PublicResultPending):
        controller.run_v1(contract_path, capsule["capsule_digest"])
    pending = state.strict_json_bytes(
        Path(value["fixed_paths"]["pending_path"]).read_bytes(),
        "test full-prefix pending",
    )
    assert pending["completed_prefix"] == [1, 2, 3]
    assert pending["q5a_action_audit"] == state.SUCCESS_ACTION_AUDIT
    assert fake.generate_ordinals == [1, 2, 3]

    monkeypatch.setattr(controller, "_publish_json", original_publish_json)
    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == wire.Q5A_SUCCESS_STATUS
    assert fake.generate_ordinals == [1, 2, 3]


def test_intent_without_verifiable_result_is_durable_terminal_no_resume(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker(fail_generate_ordinal=1)
    capsule, _authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )

    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == state.INVALIDATED_TERMINAL_NO_RESUME
    assert terminal["outcome"]["unknown_intent_digest"] is not None
    assert terminal["outcome"]["q5a_action_audit"]["worker_popen_count"] == 1
    assert Path(value["fixed_paths"]["terminal_path"]).exists()

    replay = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert replay == terminal
    assert fake.generate_ordinals == [1]
    events = _trajectory(Path(value["fixed_paths"]["trajectory_path"]))
    assert events[-1]["q5a_action_audit"]["worker_popen_count"] == 1


def test_invalidated_terminal_publication_failure_is_classified_no_resume(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker(fail_generate_ordinal=1)
    capsule, _authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )
    terminal_path = Path(value["fixed_paths"]["terminal_path"])
    original_publish_json = controller._publish_json

    def fail_invalidated_terminal(
        path: Path, item: Any, label: str, **kwargs: Any
    ) -> None:
        if path == terminal_path:
            raise OSError("injected invalidated-terminal publication failure")
        original_publish_json(path, item, label, **kwargs)

    monkeypatch.setattr(controller, "_publish_json", fail_invalidated_terminal)
    with pytest.raises(controller.ControllerTerminal) as raised:
        controller.run_v1(contract_path, capsule["capsule_digest"])
    assert raised.value.code == (
        "Q5A_INVALIDATED_TERMINAL_PUBLICATION_FAILED_NO_RESUME"
    )
    assert fake.generate_ordinals == [1]


@pytest.mark.parametrize(
    ("failed_event", "first_ordinals"),
    [
        ("SIGNER_1_PUBLIC_RECEIPT_COMPLETE", [1]),
        ("SIGNER_3_PUBLIC_RECEIPT_COMPLETE", [1, 2, 3]),
    ],
)
def test_trajectory_backfills_signer_suffix_without_duplicate_or_worker_retry(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    failed_event: str,
    first_ordinals: list[int],
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker()
    capsule, _authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )
    original_append = controller._append_trajectory
    failed = False

    def fail_signer_event(contract: Any, event: dict[str, Any]) -> None:
        nonlocal failed
        if event["event"] == failed_event and not failed:
            failed = True
            raise OSError("injected publication-before-trajectory crash")
        original_append(contract, event)

    monkeypatch.setattr(controller, "_append_trajectory", fail_signer_event)
    with pytest.raises(controller.ControllerPending):
        controller.run_v1(contract_path, capsule["capsule_digest"])
    assert fake.generate_ordinals == first_ordinals

    monkeypatch.setattr(controller, "_append_trajectory", original_append)
    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == wire.Q5A_SUCCESS_STATUS
    events = _trajectory(Path(value["fixed_paths"]["trajectory_path"]))
    names = [row["event"] for row in events]
    assert len(names) == len(set(names)) == 7
    assert names[2] == "Q5A_AUTHORIZATION_CONSUMED"
    assert fake.generate_ordinals == [1, 2, 3]


def test_trajectory_row_publication_failure_preserves_complete_prefix(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker()
    capsule, authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )
    contract = state.load_contract(contract_path)
    trajectory_path = Path(value["fixed_paths"]["trajectory_path"])
    prefix = trajectory_path.read_bytes()
    original_replace = controller.os.replace

    def fail_trajectory_replace(source: Any, destination: Any) -> None:
        if Path(destination) == trajectory_path:
            raise OSError("injected atomic trajectory replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(controller.os, "replace", fail_trajectory_replace)
    with pytest.raises(OSError):
        controller._append_trajectory(
            contract,
            controller._event(
                "TEST_SUFFIX",
                capsule,
                authorization=authorization,
            ),
        )
    assert trajectory_path.read_bytes() == prefix


def test_artifact_to_terminal_partial_finalization_never_reruns_worker(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker()
    capsule, _authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )
    terminal_path = Path(value["fixed_paths"]["terminal_path"])
    original_publish_json = controller._publish_json

    def fail_terminal(path: Path, value: Any, label: str, **kwargs: Any) -> None:
        if path == terminal_path:
            raise OSError("injected terminal publication failure")
        original_publish_json(path, value, label, **kwargs)

    monkeypatch.setattr(controller, "_publish_json", fail_terminal)
    with pytest.raises(controller.ControllerPartial) as raised:
        controller.run_v1(contract_path, capsule["capsule_digest"])
    assert raised.value.kind == controller.PARTIAL_ARTIFACT_TO_TERMINAL
    assert Path(value["fixed_paths"]["artifact_path"]).exists()
    assert fake.generate_ordinals == [1, 2, 3]

    monkeypatch.setattr(controller, "_publish_json", original_publish_json)
    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == wire.Q5A_SUCCESS_STATUS
    assert fake.generate_ordinals == [1, 2, 3]


def test_terminal_to_trajectory_suffix_partial_never_reruns_worker(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker()
    capsule, _authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )
    original_append = controller._append_trajectory
    failed = False

    def fail_terminal_event(contract: Any, event: dict[str, Any]) -> None:
        nonlocal failed
        if event["event"] == wire.Q5A_SUCCESS_STATUS and not failed:
            failed = True
            raise OSError("injected terminal-trajectory suffix failure")
        original_append(contract, event)

    monkeypatch.setattr(controller, "_append_trajectory", fail_terminal_event)
    with pytest.raises(controller.ControllerPartial) as raised:
        controller.run_v1(contract_path, capsule["capsule_digest"])
    assert (
        raised.value.kind
        == controller.PARTIAL_TERMINAL_TO_TRAJECTORY_SUFFIX
    )
    assert Path(value["fixed_paths"]["terminal_path"]).exists()
    assert fake.generate_ordinals == [1, 2, 3]

    monkeypatch.setattr(controller, "_append_trajectory", original_append)
    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == wire.Q5A_SUCCESS_STATUS
    assert len(_trajectory(Path(value["fixed_paths"]["trajectory_path"]))) == 7
    assert fake.generate_ordinals == [1, 2, 3]


def test_postconsumption_consumption_and_terminal_replay_never_leak_pending(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker()
    capsule, _authorization = _prepare_and_authorize(
        contract_path, fake, monkeypatch
    )
    original_reconcile = controller._reconcile_trajectory

    def stop_after_consumption(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("consumption") is not None:
            raise controller.ControllerPending(
                "TEST_STOP", "TEST_CLEAN_PREFIX", "stop after consumption"
            )
        original_reconcile(*args, **kwargs)

    monkeypatch.setattr(controller, "_reconcile_trajectory", stop_after_consumption)
    with pytest.raises(controller.ControllerPending):
        controller.run_v1(contract_path, capsule["capsule_digest"])
    consumption_path = Path(value["fixed_paths"]["consumption_path"])
    consumption_path.write_bytes(b"{}\n")
    monkeypatch.setattr(controller, "_reconcile_trajectory", original_reconcile)
    with pytest.raises(controller.ControllerTerminal):
        controller.run_v1(contract_path, capsule["capsule_digest"])

    # A fresh fixture is unnecessary for the terminal branch: restore the
    # exact consumption, complete once, then prove malformed terminal replay
    # is terminal rather than a resumable path/format PENDING.
    contract = state.load_contract(contract_path)
    preregistration = state.load_preregistration(
        Path(contract.value["preregistration_path"])
    )
    authorization = state.strict_json_bytes(
        Path(value["fixed_paths"]["authorization_path"]).read_bytes(),
        "test authorization",
    )
    exact_consumption = state.build_consumption(
        contract, preregistration.value, capsule, authorization
    )
    consumption_path.write_bytes(_canonical(exact_consumption))
    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == wire.Q5A_SUCCESS_STATUS
    terminal_path = Path(value["fixed_paths"]["terminal_path"])
    terminal_path.write_bytes(b"{}\n")
    with pytest.raises(controller.ControllerTerminal):
        controller.run_v1(contract_path, capsule["capsule_digest"])


def test_first_prepare_rejects_preexisting_run_root_before_any_worker(
    isolated_contract: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_path, value = isolated_contract
    fake = FakeWorker()
    monkeypatch.setattr(controller, "_invoke_worker", fake)
    Path(value["fixed_paths"]["run_root"]).mkdir(mode=0o700)

    with pytest.raises(controller.ControllerPending) as raised:
        controller.prepare_capsule(contract_path, "a" * 40)
    assert raised.value.condition == "Q5A_INITIAL_ROOT_ALREADY_EXISTS"
    assert fake.calls == []
