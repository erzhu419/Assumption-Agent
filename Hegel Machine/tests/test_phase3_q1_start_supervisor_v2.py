from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_start_supervisor_v2",
    ROOT / "tools" / "phase3_q1_start_supervisor_v2.py",
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
supervisor = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = supervisor
MODULE_SPEC.loader.exec_module(supervisor)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _digest(payload: bytes) -> str:
    return "sha256:" + sha256(payload).hexdigest()


def _run_spec_payload(tmp_path: Path) -> bytes:
    run = tmp_path / "fresh-v2-run"
    run.mkdir()
    manifests: dict[str, list[dict[str, object]]] = {}
    for stage in supervisor.STAGES:
        root = run / f"{stage}-source"
        root.mkdir()
        payload = f"{stage}-v2\n".encode("ascii")
        (root / "actor.py").write_bytes(payload)
        manifests[stage] = [{
            "relative_path": "actor.py",
            "byte_count": len(payload),
            "raw_sha256": _digest(payload),
        }]
    rust_binary = tmp_path / "rust-binary"
    rust_binary.write_bytes(b"rust-v2")
    rust_binary.chmod(0o555)
    admission = tmp_path / "admission.json"
    admission.write_bytes(_canonical({"kind": "admission"}))
    seccomp = tmp_path / "seccomp.json"
    seccomp.write_bytes(_canonical({"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}))
    evidence_paths: dict[str, Path] = {}
    for name in (
        "start-capsule", "start-authorization", "successor-bundle",
        "base-prestart", "failed-v1-terminal",
    ):
        path = (run if name == "start-capsule" else tmp_path) / f"{name}.json"
        path.write_bytes(_canonical({"name": name}))
        evidence_paths[name] = path
    digest = lambda token: "sha256:" + token * 64
    limits = {
        "python": supervisor.RuntimeLimitV2("24", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "rust": supervisor.RuntimeLimitV2("25", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "host": supervisor.RuntimeLimitV2("26", 67_108_864, 2_097_152, 1, 2_097_152, 3_600).object(),
    }
    return _canonical({
        "schema_version": supervisor.RUN_SPEC_SCHEMA,
        "start_capsule_digest": digest("1"),
        "start_capsule": str(evidence_paths["start-capsule"]),
        "start_capsule_raw_sha256": _digest(evidence_paths["start-capsule"].read_bytes()),
        "start_authorization": str(evidence_paths["start-authorization"]),
        "start_authorization_raw_sha256": _digest(evidence_paths["start-authorization"].read_bytes()),
        "start_authorization_digest": digest("2"),
        "successor_prestart_bundle": str(evidence_paths["successor-bundle"]),
        "successor_prestart_bundle_raw_sha256": _digest(evidence_paths["successor-bundle"].read_bytes()),
        "successor_prestart_bundle_root": digest("3"),
        "base_authoritative_artifact": str(evidence_paths["base-prestart"]),
        "base_authoritative_artifact_raw_sha256": _digest(evidence_paths["base-prestart"].read_bytes()),
        "failed_start_v1_terminal": str(evidence_paths["failed-v1-terminal"]),
        "failed_start_v1_terminal_raw_sha256": _digest(evidence_paths["failed-v1-terminal"].read_bytes()),
        "failed_start_v1_terminal_record_digest": digest("4"),
        "failed_start_v1_capsule_digest": digest("5"),
        "corrected_resource_envelope_root": digest("6"),
        "actor_source_manifest_roots": {"python": digest("7"), "rust": digest("8"), "host": digest("9")},
        "actor_source_closure_root": digest("a"),
        "source_commit": "b" * 40,
        "run_genesis_root": digest("c"),
        "prestart_admission_receipt_root": digest("d"),
        "python_image": "python@" + digest("e"),
        "rust_image": "rust@" + digest("f"),
        "rust_binary_raw_sha256": _digest(rust_binary.read_bytes()),
        "seccomp_profile_raw_sha256": _digest(seccomp.read_bytes()),
        "python_admission_raw_sha256": _digest(admission.read_bytes()),
        "actor_implementation_roots": {"python": digest("1"), "rust": digest("2"), "host": digest("3")},
        "python_source_root": str(run / "python-source"),
        "rust_source_root": str(run / "rust-source"),
        "host_source_root": str(run / "host-source"),
        "rust_binary": str(rust_binary),
        "python_admission_json": str(admission),
        "seccomp_profile": str(seccomp),
        "run_root": str(run),
        "checkpoint_root": str(run / "checkpoints"),
        "python_output": str(run / "python-output"),
        "python_scratch": str(run / "python-scratch"),
        "rust_output": str(run / "rust-output"),
        "rust_scratch": str(run / "rust-scratch"),
        "host_output": str(run / "host-output"),
        "host_scratch": str(run / "host-scratch"),
        "runtime_limits": limits,
        "actor_source_manifests": manifests,
    })


def test_v2_run_spec_binds_lineage_and_128_mib_endpoint_scratch(tmp_path: Path) -> None:
    payload = _run_spec_payload(tmp_path)
    spec = supervisor.load_run_spec_v2(payload)
    assert spec.start_capsule_digest != spec.failed_start_v1_capsule_digest
    assert spec.runtime_limits["python"].scratch_bytes == 134_217_728
    assert set(spec.authority_binding()) == supervisor._AUTHORITY_BINDING_KEYS
    assert spec.start_capsule.parent == spec.run_root

    value = json.loads(payload)
    value["runtime_limits"]["python"]["scratch_bytes"] = 4_194_304
    with pytest.raises(supervisor.Q1StartSupervisorError, match="frozen resource envelope differs"):
        supervisor.load_run_spec_v2(_canonical(value))


def test_v2_run_spec_rejects_capsule_outside_run_root_and_binds_full_authorization_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = json.loads(_run_spec_payload(tmp_path))
    outside = tmp_path / "outside-capsule.json"
    outside.write_bytes(Path(value["start_capsule"]).read_bytes())
    value["start_capsule"] = str(outside)
    value["start_capsule_raw_sha256"] = _digest(outside.read_bytes())
    with pytest.raises(supervisor.Q1StartSupervisorError, match="direct run-root child"):
        supervisor.load_run_spec_v2(_canonical(value))

    second = tmp_path / "second"
    second.mkdir()
    value = json.loads(_run_spec_payload(second))
    spec = supervisor.load_run_spec_v2(_canonical(value))
    monkeypatch.setattr(supervisor, "_validate_successor_bundle", lambda current, raw: {})
    monkeypatch.setattr(supervisor, "_validate_start_capsule", lambda current, capsule, bundle: None)
    monkeypatch.setattr(supervisor, "_validate_base_authoritative_artifact", lambda value: None)
    monkeypatch.setattr(supervisor, "_validate_failed_v1_terminal", lambda current, value: None)
    Path(value["start_authorization"]).write_bytes(_canonical({"tampered": True}))
    with pytest.raises(supervisor.PendingCondition, match="bytes differ"):
        supervisor._replay_successor_authority(spec)


def test_every_actor_has_two_authority_replays_before_one_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v2(_run_spec_payload(tmp_path))
    authority_calls: list[str] = []
    launches: list[str] = []

    def authority(_spec: supervisor.Q1StartRunSpecV2) -> dict[str, object]:
        authority_calls.append("replay")
        return {"authority": "exact"}

    def launcher(stage: str, _command: object, _spec: object) -> supervisor.ActorProcessResultV2:
        launches.append(stage)
        return supervisor.ActorProcessResultV2(0, b"", b"", True, {})

    def validate_endpoint(stage: str, _process: object, _diagnostic: object, current: supervisor.Q1StartRunSpecV2, _replayer: object):
        result = {"stage": stage}
        (current.output_for(stage) / "endpoint-result.json").write_bytes(_canonical(result))
        return result, {"runtime_evidence": {}}, len(_canonical(result))

    def validate_host(_process: object, _diagnostic: object, current: supervisor.Q1StartRunSpecV2):
        result = {"stage": "host"}
        (current.host_output / "host-result.json").write_bytes(_canonical(result))
        return result, {"runtime_evidence": {}}, len(_canonical(result))

    monkeypatch.setattr(supervisor, "_replay_successor_authority", authority)
    monkeypatch.setattr(supervisor, "_replay_stage_bindings", lambda stage, current: {"stage": stage})
    monkeypatch.setattr(supervisor, "_validate_endpoint", validate_endpoint)
    monkeypatch.setattr(supervisor, "_validate_host", validate_host)
    monkeypatch.setattr(
        supervisor,
        "_replay_checkpoint",
        lambda current, stage, command, replayer: supervisor._read_canonical_file(
            supervisor._stage_paths(current, stage)[1], f"{stage} checkpoint"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_success_outcome",
        lambda current, checkpoints: {
            "schema_version": supervisor.OUTCOME_SCHEMA,
            "status": supervisor.SUCCESS_STATUS,
            **current.binding(),
            **current.authority_binding(),
        },
    )
    outcome = supervisor.run_supervisor_v2(
        spec,
        launcher=launcher,
        preflight=lambda stage, current: None,
        endpoint_replayer=lambda stage, output, current: {},
    )
    assert outcome["status"] == supervisor.SUCCESS_STATUS
    assert launches == ["python", "rust", "host"]
    assert authority_calls == ["replay"] * 6
    for stage in supervisor.STAGES:
        intent = supervisor._read_canonical_file(supervisor._stage_paths(spec, stage)[0], f"{stage} intent")
        assert intent["successor_prestart_bundle_root"] == spec.successor_prestart_bundle_root
        assert intent["failed_start_v1_terminal_record_digest"] == spec.failed_start_v1_terminal_record_digest


def test_second_preintent_lineage_replay_pending_never_publishes_intent_or_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v2(_run_spec_payload(tmp_path))
    calls = 0
    launches: list[str] = []

    def authority(_spec: supervisor.Q1StartRunSpecV2) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise supervisor.PendingCondition("Q1_START_BOUND_FILE_PENDING", "capsule changed")
        return {"authority": "exact"}

    monkeypatch.setattr(supervisor, "_replay_successor_authority", authority)
    monkeypatch.setattr(supervisor, "_replay_stage_bindings", lambda stage, current: {})
    outcome = supervisor.run_supervisor_v2(
        spec,
        launcher=lambda stage, command, current: launches.append(stage),
        preflight=lambda stage, current: None,
        endpoint_replayer=lambda stage, output, current: {},
    )
    assert calls == 2
    assert launches == []
    assert outcome["status"] == supervisor.PENDING_STATUS
    assert outcome["next_actor_intent_absent"] is True
    assert not supervisor._stage_paths(spec, "python")[0].exists()
    assert not spec.python_output.exists()
    assert not spec.python_scratch.exists()


def test_failed_v1_terminal_must_be_exact_no_resume_python_only_lineage(tmp_path: Path) -> None:
    spec = supervisor.load_run_spec_v2(_run_spec_payload(tmp_path))
    value = {
        "schema_version": supervisor.FAILED_V1_TERMINAL_SCHEMA,
        "status": supervisor.FAILED_V1_TERMINAL_STATUS,
        "record_digest": spec.failed_start_v1_terminal_record_digest,
        "capsule_digest": spec.failed_start_v1_capsule_digest,
        "logical_campaign_count": 1,
        "authorization_consumption_count": 1,
        "artifact_raw_sha256": None,
        "outcome": {
            "status": supervisor.FAILED_V1_TERMINAL_STATUS,
            "capsule_digest": spec.failed_start_v1_capsule_digest,
            "actor_start_count_by_actor": {"host": 0, "python": 1, "rust": 0},
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_mask": 16_383,
            "q1_output_slots": [None] * 8,
            "network_accessed": False,
            "target_truth_accessed": False,
            "split_accessed": False,
            "role_evaluation_performed": False,
        },
    }
    supervisor._validate_failed_v1_terminal(spec, value)
    value["outcome"]["actor_start_count_by_actor"]["rust"] = 1
    with pytest.raises(supervisor.Q1StartSupervisorError, match="terminal no-resume state differs"):
        supervisor._validate_failed_v1_terminal(spec, value)
