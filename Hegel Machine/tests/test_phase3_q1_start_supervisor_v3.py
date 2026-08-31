from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_start_supervisor_v3",
    ROOT / "tools" / "phase3_q1_start_supervisor_v3.py",
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
    run = tmp_path / "fresh-v3-run"
    run.mkdir()
    manifests: dict[str, list[dict[str, object]]] = {}
    for stage in supervisor.STAGES:
        root = run / f"{stage}-source"
        root.mkdir()
        payload = f"{stage}-v3\n".encode("ascii")
        (root / "actor.py").write_bytes(payload)
        manifests[stage] = [{
            "relative_path": "actor.py",
            "byte_count": len(payload),
            "raw_sha256": _digest(payload),
        }]
    rust_binary = tmp_path / "rust-binary"
    rust_binary.write_bytes(b"rust-v3")
    rust_binary.chmod(0o555)
    admission = tmp_path / "admission.json"
    admission.write_bytes(_canonical({"kind": "admission"}))
    seccomp = tmp_path / "seccomp.json"
    seccomp.write_bytes(_canonical({"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}))
    evidence_paths: dict[str, Path] = {}
    for name in (
        "start-capsule", "start-authorization", "successor-bundle",
        "base-prestart", "failed-v2-capsule", "failed-v2-authorization",
        "failed-v2-consumption", "failed-v2-terminal",
    ):
        path = (run if name == "start-capsule" else tmp_path) / f"{name}.json"
        path.write_bytes(_canonical({"name": name}))
        evidence_paths[name] = path
    digest = lambda token: "sha256:" + token * 64
    limits = {
        "python": supervisor.RuntimeLimitV3("24", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "rust": supervisor.RuntimeLimitV3("25", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "host": supervisor.RuntimeLimitV3("26", 67_108_864, 2_097_152, 1, 2_097_152, 3_600).object(),
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
        "failed_start_v2_capsule": str(evidence_paths["failed-v2-capsule"]),
        "failed_start_v2_capsule_raw_sha256": supervisor.FAILED_V2_CAPSULE_RAW_SHA256,
        "failed_start_v2_capsule_digest": supervisor.FAILED_V2_CAPSULE_DIGEST,
        "failed_start_v2_authorization": str(evidence_paths["failed-v2-authorization"]),
        "failed_start_v2_authorization_raw_sha256": supervisor.FAILED_V2_AUTHORIZATION_RAW_SHA256,
        "failed_start_v2_authorization_digest": supervisor.FAILED_V2_AUTHORIZATION_DIGEST,
        "failed_start_v2_consumption": str(evidence_paths["failed-v2-consumption"]),
        "failed_start_v2_consumption_raw_sha256": supervisor.FAILED_V2_CONSUMPTION_RAW_SHA256,
        "failed_start_v2_consumption_record_digest": supervisor.FAILED_V2_CONSUMPTION_RECORD_DIGEST,
        "failed_start_v2_terminal": str(evidence_paths["failed-v2-terminal"]),
        "failed_start_v2_terminal_raw_sha256": supervisor.FAILED_V2_TERMINAL_RAW_SHA256,
        "failed_start_v2_terminal_record_digest": supervisor.FAILED_V2_TERMINAL_RECORD_DIGEST,
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


def test_v3_run_spec_binds_full_v2_lineage_and_128_mib_endpoint_scratch(tmp_path: Path) -> None:
    payload = _run_spec_payload(tmp_path)
    spec = supervisor.load_run_spec_v3(payload)
    assert spec.start_capsule_digest != spec.failed_start_v2_capsule_digest
    assert spec.runtime_limits["python"].scratch_bytes == 134_217_728
    assert set(spec.authority_binding()) == supervisor._AUTHORITY_BINDING_KEYS
    assert spec.start_capsule.parent == spec.run_root

    value = json.loads(payload)
    value["runtime_limits"]["python"]["scratch_bytes"] = 4_194_304
    with pytest.raises(supervisor.Q1StartSupervisorError, match="frozen resource envelope differs"):
        supervisor.load_run_spec_v3(_canonical(value))


def test_v3_run_spec_rejects_capsule_outside_run_root_and_binds_full_authorization_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = json.loads(_run_spec_payload(tmp_path))
    outside = tmp_path / "outside-capsule.json"
    outside.write_bytes(Path(value["start_capsule"]).read_bytes())
    value["start_capsule"] = str(outside)
    value["start_capsule_raw_sha256"] = _digest(outside.read_bytes())
    with pytest.raises(supervisor.Q1StartSupervisorError, match="direct run-root child"):
        supervisor.load_run_spec_v3(_canonical(value))

    second = tmp_path / "second"
    second.mkdir()
    value = json.loads(_run_spec_payload(second))
    spec = supervisor.load_run_spec_v3(_canonical(value))
    monkeypatch.setattr(supervisor, "_validate_successor_bundle", lambda current, raw: {})
    monkeypatch.setattr(supervisor, "_validate_start_capsule", lambda current, capsule, bundle: None)
    monkeypatch.setattr(supervisor, "_validate_base_authoritative_artifact", lambda value: None)
    monkeypatch.setattr(supervisor, "_validate_failed_v2_lineage", lambda *args: None)
    Path(value["start_authorization"]).write_bytes(_canonical({"tampered": True}))
    with pytest.raises(supervisor.PendingCondition, match="bytes differ"):
        supervisor._replay_successor_authority(spec)


def test_every_actor_has_two_authority_replays_before_one_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v3(_run_spec_payload(tmp_path))
    authority_calls: list[str] = []
    launches: list[str] = []

    def authority(_spec: supervisor.Q1StartRunSpecV3) -> dict[str, object]:
        authority_calls.append("replay")
        return {"authority": "exact"}

    def launcher(stage: str, _command: object, _spec: object) -> supervisor.ActorProcessResultV3:
        launches.append(stage)
        return supervisor.ActorProcessResultV3(0, b"", b"", True, {})

    def validate_endpoint(stage: str, _process: object, _diagnostic: object, current: supervisor.Q1StartRunSpecV3, _replayer: object):
        result = {"stage": stage}
        (current.output_for(stage) / "endpoint-result.json").write_bytes(_canonical(result))
        return result, {"runtime_evidence": {}}, len(_canonical(result))

    def validate_host(_process: object, _diagnostic: object, current: supervisor.Q1StartRunSpecV3):
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
    outcome = supervisor.run_supervisor_v3(
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
        assert intent["failed_start_v2_terminal_record_digest"] == spec.failed_start_v2_terminal_record_digest


def test_second_preintent_lineage_replay_pending_never_publishes_intent_or_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v3(_run_spec_payload(tmp_path))
    calls = 0
    launches: list[str] = []

    def authority(_spec: supervisor.Q1StartRunSpecV3) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise supervisor.PendingCondition("Q1_START_BOUND_FILE_PENDING", "capsule changed")
        return {"authority": "exact"}

    monkeypatch.setattr(supervisor, "_replay_successor_authority", authority)
    monkeypatch.setattr(supervisor, "_replay_stage_bindings", lambda stage, current: {})
    outcome = supervisor.run_supervisor_v3(
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


def test_failed_v2_terminal_must_be_exact_unknown_no_actor_lineage(tmp_path: Path) -> None:
    spec = supervisor.load_run_spec_v3(_run_spec_payload(tmp_path))
    capsule = {
        "schema_version": supervisor.FAILED_V2_CAPSULE_SCHEMA,
        "status": "Q1_START_V2_CAPSULE_READY_NOT_AUTHORIZED",
        "capsule_digest": spec.failed_start_v2_capsule_digest,
    }
    authorization = {
        "schema_version": supervisor.FAILED_V2_AUTHORIZATION_SCHEMA,
        "status": "Q1_START_V2_AUTHORIZATION_RECORDED_NOT_CONSUMED",
        "capsule_digest": spec.failed_start_v2_capsule_digest,
        "authorization_digest": spec.failed_start_v2_authorization_digest,
    }
    consumption = {
        "schema_version": supervisor.FAILED_V2_CONSUMPTION_SCHEMA,
        "status": "Q1_START_V2_AUTHORIZATION_CONSUMED",
        "capsule_digest": spec.failed_start_v2_capsule_digest,
        "authorization_digest": spec.failed_start_v2_authorization_digest,
        "record_digest": spec.failed_start_v2_consumption_record_digest,
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
    }
    terminal = {
        "schema_version": supervisor.FAILED_V2_TERMINAL_SCHEMA,
        "status": supervisor.FAILED_V2_TERMINAL_STATUS,
        "record_digest": spec.failed_start_v2_terminal_record_digest,
        "capsule_digest": spec.failed_start_v2_capsule_digest,
        "authorization_digest": spec.failed_start_v2_authorization_digest,
        "consumption_record_digest": spec.failed_start_v2_consumption_record_digest,
        "logical_campaign_count": 1,
        "authorization_consumption_count": 1,
        "artifact_raw_sha256": None,
        "outcome": {
            "status": supervisor.FAILED_V2_TERMINAL_STATUS,
            "capsule_digest": spec.failed_start_v2_capsule_digest,
            "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
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
    supervisor._validate_failed_v2_lineage(
        spec, capsule, authorization, consumption, terminal
    )
    terminal["outcome"]["actor_start_count_by_actor"]["rust"] = 1
    with pytest.raises(supervisor.Q1StartSupervisorError, match="terminal no-resume state differs"):
        supervisor._validate_failed_v2_lineage(
            spec, capsule, authorization, consumption, terminal
        )


def test_authority_startup_smoke_is_zero_popen_and_zero_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_spec_payload(tmp_path)
    spec = supervisor.load_run_spec_v3(payload)
    before = sorted(str(path.relative_to(spec.run_root)) for path in spec.run_root.rglob("*"))
    calls: list[str] = []
    monkeypatch.setattr(
        supervisor,
        "_replay_successor_authority",
        lambda current: {"capsule": current.start_capsule_digest},
    )
    monkeypatch.setattr(
        supervisor,
        "_replay_stage_bindings",
        lambda stage, current: {"stage": stage, "source": str(current.source_for(stage))},
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *args, **kwargs: calls.append("subprocess") or pytest.fail("subprocess called"),
    )
    result = supervisor.authority_startup_smoke_v3(payload)
    after = sorted(str(path.relative_to(spec.run_root)) for path in spec.run_root.rglob("*"))
    assert result["status"] == supervisor.AUTHORITY_SMOKE_STATUS
    assert result["actor_popen_count"] == 0
    assert result["checkpoint_or_mutable_roots_created"] is False
    assert result["subprocess_or_docker_invoked"] is False
    assert before == after
    assert calls == []
    assert not spec.checkpoint_root.exists()


def test_process_diagnostic_retains_bounded_head_and_tail() -> None:
    payload = b"head:" + b"a" * supervisor.PROCESS_DIAGNOSTIC_CAPTURE_BYTES + b":tail"
    captured = supervisor._captured_process_stream(payload)
    import base64

    head = base64.b64decode(captured["head_base64"])
    tail = base64.b64decode(captured["tail_base64"])
    assert head.startswith(b"head:")
    assert tail.endswith(b":tail")
    assert captured["captured_bytes"] == supervisor.PROCESS_DIAGNOSTIC_CAPTURE_BYTES
    assert captured["truncated"] is True


def test_private_bundle_loader_replays_fixture_without_executing_package_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "hegel_machine"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('package init must never execute')\n",
        encoding="utf-8",
    )
    (package / "strict_cbor_v1.py").write_text(
        "TOKEN = 'strict-cbor-loaded'\n",
        encoding="utf-8",
    )
    (package / "phase3_q1_prestart_successor_candidate_v3.py").write_text(
        "from .strict_cbor_v1 import TOKEN\n"
        "import json\n"
        "def replay_q1_prestart_successor_candidate_v3(payload):\n"
        "    value = json.loads(payload.decode('ascii'))\n"
        "    assert TOKEN == 'strict-cbor-loaded'\n"
        "    return value\n"
        "def successor_bundle_root_v3(value): return value['bundle_root']\n"
        "def actor_source_manifest_root_v3(actor, rows): return rows[0]['raw_sha256']\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "PACKAGE_ROOT", package)
    monkeypatch.delitem(sys.modules, "hegel_machine", raising=False)
    module = supervisor._load_successor_bundle_core_v3()
    bundle = _canonical({"bundle_root": "sha256:" + "a" * 64})
    replayed = module.replay_q1_prestart_successor_candidate_v3(bundle)
    assert module.successor_bundle_root_v3(replayed) == "sha256:" + "a" * 64
    assert "hegel_machine" not in sys.modules
