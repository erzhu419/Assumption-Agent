from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "phase3_q6_controller_test_support",
    ROOT / "tests/test_phase3_q6_controller_v1.py",
)
assert SUPPORT_SPEC is not None and SUPPORT_SPEC.loader is not None
support = importlib.util.module_from_spec(SUPPORT_SPEC)
SUPPORT_SPEC.loader.exec_module(support)

controller = support.controller
state = support.state

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "phase3_q6_completion_fixture_for_controller_audit",
    ROOT / "tests/test_phase3_q6_completion_replay_v1.py",
)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
fixture_module = importlib.util.module_from_spec(FIXTURE_SPEC)
sys.modules[FIXTURE_SPEC.name] = fixture_module
FIXTURE_SPEC.loader.exec_module(fixture_module)

CONTRACT_PATH = (
    ROOT / "config/phase3_q6_formal_quotient_certificate_issuance_actual_v1.json"
)


@pytest.fixture
def linux_tmp_path() -> Path:
    with tempfile.TemporaryDirectory(prefix="q6-controller-audit-", dir="/tmp") as raw:
        yield Path(raw)


def _optional_bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _fixture() -> dict[str, object]:
    return fixture_module._fixture()


def _temp_contract(data: dict[str, object], tmp_path: Path) -> state.LoadedJson:
    original = data["contract"]
    assert isinstance(original, state.LoadedJson)
    value = deepcopy(original.value)
    fixed = value["fixed_paths"]
    for key in tuple(fixed):
        if key == "project_root":
            continue
        fixed[key] = str(tmp_path / key)
    return state.LoadedJson(
        tmp_path / "contract.json",
        original.raw,
        original.raw_digest,
        value,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(state.canonical_json_bytes(value))


def _signer_paths(contract: state.LoadedJson, ordinal: int) -> dict[str, Path]:
    return controller._signer_paths(contract, ordinal)


def test_prepare_after_success_terminal_is_read_only_and_history_stable(
    linux_tmp_path: Path, monkeypatch,
) -> None:
    contract_path = support._contract(linux_tmp_path)
    fixed = json.loads(contract_path.read_text(encoding="ascii"))["fixed_paths"]
    invocation_counts = {"smoke": 0, "sign": 0}

    def tracked_worker(command: list[str]):
        operation = "smoke" if "--startup-smoke" in command else "sign"
        invocation_counts[operation] += 1
        return support._fake_worker(command)

    monkeypatch.setattr(controller, "_invoke_worker", tracked_worker)
    capsule = controller.prepare_capsule(
        contract_path,
        "c" * 40,
        run_id="d" * 32,
        created_at_unix_seconds=1_800_000_000,
        certificate_created_at_unix_seconds=1_800_000_001,
    )
    controller.record_authorization(contract_path, capsule["capsule_digest"])

    for ordinal in state.SIGNER_ORDINALS:
        custody = Path(fixed[f"signer_{ordinal}_custody_root"])
        custody.mkdir(mode=0o700)
        os.chmod(custody, 0o700)
        private_key = custody / "ed25519-private.pem"
        private_key.write_bytes(b"mock-private-metadata-only\n")
        os.chmod(private_key, 0o600)

    monkeypatch.setattr(controller.wire, "verify_ed25519_v1", lambda *_args: None)
    terminal = controller.run_v1(contract_path, capsule["capsule_digest"])
    assert terminal["status"] == controller.wire.Q6_SUCCESS_STATUS

    watched_paths = {
        name: Path(fixed[name])
        for name in (
            "pending_history_path",
            "pending_path",
            "terminal_path",
            "trajectory_path",
        )
    }
    before = {name: _optional_bytes(path) for name, path in watched_paths.items()}
    calls_before = dict(invocation_counts)
    pending_calls = 0
    original_persist_pending = controller._persist_pending

    def tracked_persist_pending(*args, **kwargs):
        nonlocal pending_calls
        pending_calls += 1
        return original_persist_pending(*args, **kwargs)

    monkeypatch.setattr(controller, "_persist_pending", tracked_persist_pending)
    replayed_capsule = controller.prepare_capsule(
        contract_path,
        "c" * 40,
        run_id="e" * 32,
        created_at_unix_seconds=1_800_000_100,
        certificate_created_at_unix_seconds=1_800_000_101,
    )

    assert replayed_capsule == capsule
    assert invocation_counts == calls_before
    assert pending_calls == 0
    assert {
        name: _optional_bytes(path) for name, path in watched_paths.items()
    } == before


def test_fixed_signer_outputs_have_three_distinct_roles_and_exact_temp_names() -> None:
    contract = state.load_contract(CONTRACT_PATH)
    fixed = contract.value["fixed_paths"]
    for ordinal in state.SIGNER_ORDINALS:
        worker = Path(fixed[f"signer_{ordinal}_worker_receipt_path"])
        verification = Path(fixed[f"signer_{ordinal}_verification_intent_path"])
        verified = Path(fixed[f"signer_{ordinal}_verified_checkpoint_path"])
        signature = Path(fixed[f"signer_{ordinal}_signature_path"])
        transient = Path(fixed[f"signer_{ordinal}_signature_temp_path"])
        assert len({worker, verification, verified}) == 3
        assert transient == signature.with_name(signature.name + ".q6-tmp")


def test_observed_action_audit_rejects_impossible_causal_counts() -> None:
    audit = state.q6_observed_action_audit(
        (),
        metadata_admission_count=1,
        signing_intent_count=1,
        worker_popen_count=1,
        openssl_sign_count=1,
        private_key_load_count=1,
        signature_output_count=1,
        worker_receipt_count=1,
        verification_intent_count=0,
        verification_count=0,
        valid_signature_count=0,
        verified_checkpoint_count=0,
        verified_checkpoint_publication_count=0,
    )
    assert state.validate_observed_action_audit(audit) == audit

    impossible = deepcopy(audit)
    impossible["signing_intent_publication_count"] = 0
    with pytest.raises(state.Q6CompletionReplayError):
        state.validate_observed_action_audit(impossible)

    impossible = deepcopy(audit)
    impossible["signature_count"] = 0
    with pytest.raises(state.Q6CompletionReplayError):
        state.validate_observed_action_audit(impossible)

    impossible = deepcopy(audit)
    impossible["certificate_issuance_count"] = 1
    with pytest.raises(state.Q6CompletionReplayError):
        state.validate_observed_action_audit(impossible)

    wrong_json_type = deepcopy(audit)
    wrong_json_type["certificate_envelope_publication_count"] = True
    with pytest.raises(state.Q6CompletionReplayError):
        state.validate_observed_action_audit(wrong_json_type)


def test_complete_prefix_pending_is_deterministic_only() -> None:
    data = _fixture()
    audit = state.q6_observed_action_audit(
        (1, 2, 3),
        metadata_admission_count=3,
        signing_intent_count=3,
        worker_popen_count=3,
        openssl_sign_count=3,
        private_key_load_count=3,
        signature_output_count=3,
        worker_receipt_count=3,
        verification_intent_count=3,
        verification_count=3,
        valid_signature_count=3,
        verified_checkpoint_count=3,
        verified_checkpoint_publication_count=3,
        envelope_publication_count=1,
        issuance_count=0,
    )
    pending = state.build_pending(
        data["contract"],
        data["capsule"],
        failure_phase="FINALIZATION",
        failure_code="Q6_FINAL_ARTIFACT_PUBLICATION_NOT_DURABLE",
        failure_detail="injected",
        authorization=data["authorization"],
        consumption=data["consumption"],
        receipts=data["receipts"],
        observed_action_audit=audit,
    )
    assert pending["status"] == state.PENDING_CLEAN_CHECKPOINT
    assert pending["completed_prefix"] == [1, 2, 3]
    assert pending["inflight_public_result"] is None
    with pytest.raises(state.Q6CompletionReplayError):
        state.build_signing_intent(
            data["contract"],
            data["prereg"].value,
            data["capsule"],
            data["authorization"],
            data["consumption"],
            data["receipts"],
            created_at_unix_seconds=1_900_000_000,
        )


def test_fresh_prepare_does_not_reject_its_own_campaign_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _fixture()
    contract = _temp_contract(data, tmp_path)

    class ReachedSourceClosure(RuntimeError):
        pass

    monkeypatch.setattr(
        controller,
        "_load_contract",
        lambda path: (contract, data["prereg"]),
    )
    monkeypatch.setattr(
        controller,
        "_source_closure_rows",
        lambda unused: (_ for _ in ()).throw(ReachedSourceClosure()),
    )
    with pytest.raises(ReachedSourceClosure):
        controller.prepare_capsule(
            tmp_path / "contract.json",
            "a" * 40,
            run_id="b" * 32,
            created_at_unix_seconds=1_900_000_001,
            certificate_created_at_unix_seconds=1_900_000_002,
        )
    assert Path(contract.value["fixed_paths"]["campaign_lock_path"]).exists()


def test_intent_without_worker_result_is_terminal_no_resume(tmp_path: Path) -> None:
    data = _fixture()
    contract = _temp_contract(data, tmp_path)
    paths = _signer_paths(contract, 1)
    _write_json(paths["intent"], data["intents"][0])

    with pytest.raises(controller.PrefixReplayTerminal) as caught:
        controller._recover_prefix(
            contract,
            data["prereg"],
            data["capsule"],
            data["authorization"],
            data["consumption"],
        )
    assert caught.value.code == "Q6_SIGNING_OUTCOME_UNKNOWN_AFTER_INTENT"
    audit = caught.value.observed_action_audit
    assert audit["completed_prefix"] == []
    assert audit["signing_intent_publication_count"] == 1
    assert audit["signing_worker_popen_count"] == 1


def test_known_worker_result_can_advance_once_but_verification_intent_cannot_reenter(
    tmp_path: Path,
) -> None:
    data = _fixture()
    contract = _temp_contract(data, tmp_path)
    paths = _signer_paths(contract, 1)
    signature = data["signatures"][0]
    intent = data["intents"][0]
    worker_receipt = state.expected_worker_receipt(
        intent=intent, signature_payload=signature
    )
    _write_json(paths["intent"], intent)
    paths["signature"].parent.mkdir(parents=True, exist_ok=True)
    paths["signature"].write_bytes(signature)
    _write_json(paths["worker_receipt"], worker_receipt)

    recovered = controller._recover_prefix(
        contract,
        data["prereg"],
        data["capsule"],
        data["authorization"],
        data["consumption"],
    )
    assert recovered[0] == []
    assert recovered[4] is not None
    assert recovered[4]["ordinal"] == 1

    _write_json(paths["verification_intent"], data["verification_intents"][0])
    with pytest.raises(controller.PrefixReplayTerminal) as caught:
        controller._recover_prefix(
            contract,
            data["prereg"],
            data["capsule"],
            data["authorization"],
            data["consumption"],
        )
    assert caught.value.code == "Q6_VERIFICATION_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME"


def test_global_checkpoint_tamper_is_terminal(tmp_path: Path) -> None:
    data = _fixture()
    contract = _temp_contract(data, tmp_path)
    checkpoint = state.build_checkpoint(
        contract,
        data["capsule"],
        data["authorization"],
        data["consumption"],
        data["receipts"][:1],
    )
    checkpoint["completed_receipts"][0]["signature_raw_sha256"] = (
        "sha256:" + "0" * 64
    )
    _write_json(
        Path(contract.value["fixed_paths"]["checkpoint_path"]), checkpoint
    )
    with pytest.raises(controller.ControllerTerminal) as caught:
        controller._publish_global_checkpoint(
            contract,
            data["capsule"],
            data["authorization"],
            data["consumption"],
            data["receipts"][:1],
        )
    assert caught.value.code == "Q6_GLOBAL_CHECKPOINT_REPLAY_DIFFERS"


def test_pending_history_rejects_noncontiguous_or_malformed_rows(
    tmp_path: Path,
) -> None:
    data = _fixture()
    contract = _temp_contract(data, tmp_path)
    path = Path(contract.value["fixed_paths"]["pending_history_path"])
    row = {
        "schema_version": "hegel-phase3a-q6-pending-history-event/1",
        "occurrence": 2,
        "pending_digest": "sha256:" + "1" * 64,
        "failure_phase": "LOCAL_GATE",
        "failure_code": "INJECTED",
        "failure_detail": "injected",
        "capsule_digest": data["capsule"]["capsule_digest"],
        "authorization_digest": data["authorization"]["authorization_digest"],
        "consumption_record_digest": data["consumption"]["record_digest"],
        "completed_prefix": [],
    }
    _write_json(path, row)
    with pytest.raises(controller.ControllerTerminal) as caught:
        controller._pending_history_rows(contract)
    assert caught.value.code == "Q6_PENDING_HISTORY_REPLAY_DIFFERS"

    row["occurrence"] = 1
    row["pending_digest"] = "sha256:not-a-digest"
    _write_json(path, row)
    with pytest.raises(controller.ControllerTerminal) as caught:
        controller._pending_history_rows(contract)
    assert caught.value.code == "Q6_PENDING_HISTORY_REPLAY_DIFFERS"


def test_terminal_trajectory_projects_complete_success_and_failure_evidence() -> None:
    data = _fixture()
    history = {
        "raw_sha256": state.raw_sha256(b""),
        "byte_count": 0,
        "event_count": 0,
    }
    issuance = state.build_issuance_artifact(
        data["capsule"],
        data["receipts"],
        data["signatures"],
        q4_artifact_payload=data["q4"],
        q5b_artifact_payload=data["q5b"]["artifact"],
        complete_checkpoint_digest=data["checkpoint"]["checkpoint_digest"],
    )
    success = state.build_complete_terminal(
        data["contract"],
        data["capsule"],
        data["authorization"],
        data["consumption"],
        issuance.canonical_bytes,
        issuance.envelope.canonical_bytes,
        issuance,
        data["receipts"],
        complete_checkpoint_digest=data["checkpoint"]["checkpoint_digest"],
        pending_history_receipt=history,
    )
    success_row = controller._trajectory_rows(
        data["capsule"],
        data["authorization"],
        data["consumption"],
        data["receipts"],
        success,
    )[-1]
    for field in (
        "artifact_raw_sha256",
        "envelope_raw_sha256",
        "complete_checkpoint_digest",
        "issuance_receipt_root",
    ):
        assert success_row[field] == success["outcome"][field]
    assert success_row["pending_history_receipt"] == history
    assert success_row["failure_phase"] is None
    assert success_row["failure_code"] is None
    assert success_row["failure_detail"] is None

    failure_audit = state.q6_observed_action_audit(
        (),
        metadata_admission_count=1,
        signing_intent_count=1,
        worker_popen_count=1,
        openssl_sign_count=1,
        private_key_load_count=1,
        signature_output_count=1,
        worker_receipt_count=1,
        verification_intent_count=0,
        verification_count=0,
        valid_signature_count=0,
        verified_checkpoint_count=0,
        verified_checkpoint_publication_count=0,
    )
    invalidated = state.build_invalidated_terminal(
        data["contract"],
        data["capsule"],
        data["authorization"],
        data["consumption"],
        (),
        failure_phase="SIGNER_1",
        failure_code="Q6_SIGNING_OUTCOME_UNKNOWN_AFTER_INTENT",
        failure_detail="injected failure",
        observed_action_audit=failure_audit,
        pending_history_receipt=history,
    )
    failure_row = controller._trajectory_rows(
        data["capsule"],
        data["authorization"],
        data["consumption"],
        (),
        invalidated,
    )[-1]
    assert failure_row["failure_phase"] == "SIGNER_1"
    assert failure_row["failure_code"] == (
        "Q6_SIGNING_OUTCOME_UNKNOWN_AFTER_INTENT"
    )
    assert failure_row["failure_detail"] == "injected failure"
    assert failure_row["pending_history_receipt"] == history


def test_existing_invalid_consumption_is_not_reported_as_unconsumed_pending(
    tmp_path: Path,
) -> None:
    data = _fixture()
    contract = _temp_contract(data, tmp_path)
    path = Path(contract.value["fixed_paths"]["consumption_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"{not-json}\n")
    with pytest.raises(controller.ControllerTerminal) as caught:
        controller._read_consumption(
            contract,
            data["prereg"],
            data["capsule"],
            data["authorization"],
        )
    assert caught.value.code == "Q6_CONSUMPTION_REPLAY_UNAVAILABLE"


def test_cli_preserves_postconsumption_pending_status_and_counts(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pending = controller.ControllerPending(
        "FINALIZATION", "Q6_FINAL_ARTIFACT_PUBLICATION_NOT_DURABLE", "injected"
    )
    pending.pending_record = {
        "schema_version": state.PENDING_SCHEMA,
        "status": state.PENDING_CLEAN_CHECKPOINT,
        "completed_prefix": [1, 2, 3],
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "failure_phase": pending.phase,
        "failure_code": pending.condition,
        "failure_detail": pending.detail,
    }

    def injected(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise pending

    monkeypatch.setattr(controller, "run_v1", injected)
    assert controller.main([
        "run", "--capsule-digest", "sha256:" + "a" * 64
    ]) == controller.PENDING_EXIT_CODE
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == state.PENDING_CLEAN_CHECKPOINT
    assert output["completed_prefix"] == [1, 2, 3]
    assert output["authorization_consumption_count"] == 1
    assert output["logical_campaign_count"] == 1


@pytest.mark.parametrize("failure_site", ("history", "snapshot"))
def test_persist_pending_write_failure_retains_original_postconsumption_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
) -> None:
    data = _fixture()
    contract = _temp_contract(data, tmp_path)
    original = controller.ControllerPending(
        "FINALIZATION",
        "Q6_FINAL_ARTIFACT_PUBLICATION_NOT_DURABLE",
        "injected original failure",
    )

    if failure_site == "history":
        def fail_history(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise controller.ControllerPending(
                "PENDING_HISTORY",
                "Q6_PENDING_HISTORY_NOT_DURABLE",
                "injected history failure",
            )

        monkeypatch.setattr(controller, "_append_pending_history", fail_history)
    else:
        monkeypatch.setattr(
            controller,
            "_append_pending_history",
            lambda *args, **kwargs: None,
        )

        def fail_snapshot(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise OSError("injected snapshot failure")

        monkeypatch.setattr(controller, "_replace_json", fail_snapshot)

    with pytest.raises(controller.ControllerPending) as caught:
        controller._persist_pending(
            contract,
            data["capsule"],
            original,
            authorization=data["authorization"],
            consumption=data["consumption"],
            receipts=(),
        )

    pending = caught.value.pending_record
    assert pending is not None
    assert pending["status"] == state.PENDING_CLEAN_CHECKPOINT
    assert pending["failure_phase"] == "FINALIZATION"
    assert pending["failure_code"] == (
        "Q6_FINAL_ARTIFACT_PUBLICATION_NOT_DURABLE"
    )
    assert pending["failure_detail"] == "injected original failure"
    assert pending["completed_prefix"] == []
    assert pending["authorization_consumption_count"] == 1
    assert pending["logical_campaign_count"] == 1
    assert pending["authorization_digest"] == (
        data["authorization"]["authorization_digest"]
    )
    assert pending["consumption_record_digest"] == (
        data["consumption"]["record_digest"]
    )
