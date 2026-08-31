from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_start_supervisor_v7",
    ROOT / "tools" / "phase3_q1_start_supervisor_v7.py",
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


def _native_test_root(request: pytest.FixtureRequest, label: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f"q1-v7-{label}-", dir="/tmp"))
    request.addfinalizer(lambda: shutil.rmtree(root, ignore_errors=True))
    return root


def _run_spec_payload(tmp_path: Path) -> bytes:
    run = tmp_path / "fresh-v7-run"
    run.mkdir()
    manifests: dict[str, list[dict[str, object]]] = {}
    for stage in supervisor.STAGES:
        root = run / f"{stage}-source"
        root.mkdir()
        payload = f"{stage}-v7\n".encode("ascii")
        (root / "actor.py").write_bytes(payload)
        manifests[stage] = [{
            "relative_path": "actor.py",
            "byte_count": len(payload),
            "raw_sha256": _digest(payload),
        }]
    rust_binary = tmp_path / "rust-binary"
    rust_binary.write_bytes(b"rust-v7")
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
        "failed-v3-capsule", "failed-v3-authorization",
        "failed-v3-consumption", "failed-v3-terminal",
        "failed-v4-capsule", "failed-v4-authorization",
        "failed-v4-consumption", "failed-v4-terminal",
        "failed-v5-capsule", "failed-v5-authorization",
        "failed-v5-consumption", "failed-v5-terminal",
    ):
        path = (run if name == "start-capsule" else tmp_path) / f"{name}.json"
        path.write_bytes(_canonical({"name": name}))
        evidence_paths[name] = path
    failed_v6_run = tmp_path / "failed-v6-run"
    failed_v6_checkpoints = failed_v6_run / "checkpoints"
    failed_v6_ledger = tmp_path / "failed-v6-ledger"
    failed_v6_checkpoints.mkdir(parents=True)
    failed_v6_ledger.mkdir()
    failed_v6_paths = {
        "failed-v6-capsule": failed_v6_run / "capsule.json",
        "failed-v6-authorization": failed_v6_ledger / "owner.authorization.json",
        "failed-v6-consumption": failed_v6_ledger / "campaign.consumed.json",
        "failed-v6-terminal": failed_v6_ledger / "campaign.terminal.json",
        "failed-v6-supervisor-outcome": failed_v6_checkpoints / "terminal-outcome.json",
        "failed-v6-controller-outcome": failed_v6_run / "campaign.outcome.json",
        "failed-v6-oom-event-receipt": tmp_path / "failed-v6-oom-events.jsonl",
    }
    for name, path in failed_v6_paths.items():
        path.write_bytes(_canonical({"name": name}))
        evidence_paths[name] = path
    digest = lambda token: "sha256:" + token * 64
    limits = {
        "python": supervisor.RuntimeLimitV7("24", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "rust": supervisor.RuntimeLimitV7("25", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "host": supervisor.RuntimeLimitV7("26", 2_147_483_648, 2_097_152, 1, 2_097_152, 3_600).object(),
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
        "failed_start_v3_capsule": str(evidence_paths["failed-v3-capsule"]),
        "failed_start_v3_capsule_raw_sha256": supervisor.FAILED_V3_CAPSULE_RAW_SHA256,
        "failed_start_v3_capsule_digest": supervisor.FAILED_V3_CAPSULE_DIGEST,
        "failed_start_v3_authorization": str(evidence_paths["failed-v3-authorization"]),
        "failed_start_v3_authorization_raw_sha256": supervisor.FAILED_V3_AUTHORIZATION_RAW_SHA256,
        "failed_start_v3_authorization_digest": supervisor.FAILED_V3_AUTHORIZATION_DIGEST,
        "failed_start_v3_consumption": str(evidence_paths["failed-v3-consumption"]),
        "failed_start_v3_consumption_raw_sha256": supervisor.FAILED_V3_CONSUMPTION_RAW_SHA256,
        "failed_start_v3_consumption_record_digest": supervisor.FAILED_V3_CONSUMPTION_RECORD_DIGEST,
        "failed_start_v3_terminal": str(evidence_paths["failed-v3-terminal"]),
        "failed_start_v3_terminal_raw_sha256": supervisor.FAILED_V3_TERMINAL_RAW_SHA256,
        "failed_start_v3_terminal_record_digest": supervisor.FAILED_V3_TERMINAL_RECORD_DIGEST,
        "failed_start_v4_capsule": str(evidence_paths["failed-v4-capsule"]),
        "failed_start_v4_capsule_raw_sha256": supervisor.FAILED_V4_CAPSULE_RAW_SHA256,
        "failed_start_v4_capsule_digest": supervisor.FAILED_V4_CAPSULE_DIGEST,
        "failed_start_v4_authorization": str(evidence_paths["failed-v4-authorization"]),
        "failed_start_v4_authorization_raw_sha256": supervisor.FAILED_V4_AUTHORIZATION_RAW_SHA256,
        "failed_start_v4_authorization_digest": supervisor.FAILED_V4_AUTHORIZATION_DIGEST,
        "failed_start_v4_consumption": str(evidence_paths["failed-v4-consumption"]),
        "failed_start_v4_consumption_raw_sha256": supervisor.FAILED_V4_CONSUMPTION_RAW_SHA256,
        "failed_start_v4_consumption_record_digest": supervisor.FAILED_V4_CONSUMPTION_RECORD_DIGEST,
        "failed_start_v4_terminal": str(evidence_paths["failed-v4-terminal"]),
        "failed_start_v4_terminal_raw_sha256": supervisor.FAILED_V4_TERMINAL_RAW_SHA256,
        "failed_start_v4_terminal_record_digest": supervisor.FAILED_V4_TERMINAL_RECORD_DIGEST,
        "failed_start_v5_capsule": str(evidence_paths["failed-v5-capsule"]),
        "failed_start_v5_capsule_raw_sha256": supervisor.FAILED_V5_CAPSULE_RAW_SHA256,
        "failed_start_v5_capsule_digest": supervisor.FAILED_V5_CAPSULE_DIGEST,
        "failed_start_v5_authorization": str(evidence_paths["failed-v5-authorization"]),
        "failed_start_v5_authorization_raw_sha256": supervisor.FAILED_V5_AUTHORIZATION_RAW_SHA256,
        "failed_start_v5_authorization_digest": supervisor.FAILED_V5_AUTHORIZATION_DIGEST,
        "failed_start_v5_consumption": str(evidence_paths["failed-v5-consumption"]),
        "failed_start_v5_consumption_raw_sha256": supervisor.FAILED_V5_CONSUMPTION_RAW_SHA256,
        "failed_start_v5_consumption_record_digest": supervisor.FAILED_V5_CONSUMPTION_RECORD_DIGEST,
        "failed_start_v5_terminal": str(evidence_paths["failed-v5-terminal"]),
        "failed_start_v5_terminal_raw_sha256": supervisor.FAILED_V5_TERMINAL_RAW_SHA256,
        "failed_start_v5_terminal_record_digest": supervisor.FAILED_V5_TERMINAL_RECORD_DIGEST,
        "failed_start_v6_capsule": str(evidence_paths["failed-v6-capsule"]),
        "failed_start_v6_capsule_raw_sha256": supervisor.FAILED_V6_CAPSULE_RAW_SHA256,
        "failed_start_v6_capsule_digest": supervisor.FAILED_V6_CAPSULE_DIGEST,
        "failed_start_v6_authorization": str(evidence_paths["failed-v6-authorization"]),
        "failed_start_v6_authorization_raw_sha256": supervisor.FAILED_V6_AUTHORIZATION_RAW_SHA256,
        "failed_start_v6_authorization_digest": supervisor.FAILED_V6_AUTHORIZATION_DIGEST,
        "failed_start_v6_consumption": str(evidence_paths["failed-v6-consumption"]),
        "failed_start_v6_consumption_raw_sha256": supervisor.FAILED_V6_CONSUMPTION_RAW_SHA256,
        "failed_start_v6_consumption_record_digest": supervisor.FAILED_V6_CONSUMPTION_RECORD_DIGEST,
        "failed_start_v6_terminal": str(evidence_paths["failed-v6-terminal"]),
        "failed_start_v6_terminal_raw_sha256": supervisor.FAILED_V6_TERMINAL_RAW_SHA256,
        "failed_start_v6_terminal_record_digest": supervisor.FAILED_V6_TERMINAL_RECORD_DIGEST,
        "failed_start_v6_supervisor_outcome": str(evidence_paths["failed-v6-supervisor-outcome"]),
        "failed_start_v6_supervisor_outcome_raw_sha256": supervisor.FAILED_V6_SUPERVISOR_OUTCOME_RAW_SHA256,
        "failed_start_v6_controller_outcome": str(evidence_paths["failed-v6-controller-outcome"]),
        "failed_start_v6_controller_outcome_raw_sha256": supervisor.FAILED_V6_CONTROLLER_OUTCOME_RAW_SHA256,
        "failed_start_v6_oom_event_receipt": str(evidence_paths["failed-v6-oom-event-receipt"]),
        "failed_start_v6_oom_event_receipt_raw_sha256": supervisor.FAILED_V6_OOM_EVENT_RECEIPT_RAW_SHA256,
        "corrected_resource_envelope_root": digest("6"),
        "actor_source_manifest_roots": {"python": digest("7"), "rust": digest("8"), "host": digest("9")},
        "actor_source_closure_root": digest("a"),
        "source_commit": "b" * 40,
        "run_genesis_root": digest("c"),
        "prestart_admission_receipt_root": digest("d"),
        "python_image": "python@" + digest("e"),
        "rust_image": "rust@" + digest("f"),
        "container_user": supervisor._host_effective_user(),
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


def _runtime_evidence(
    spec: supervisor.Q1StartRunSpecV7,
    stage: str,
    *,
    memory_events: dict[str, int],
) -> dict[str, object]:
    limit = spec.runtime_limits[stage]
    return {
        "schema_version": supervisor.RUNTIME_EVIDENCE_SCHEMA,
        "stage": stage,
        "container_id": "a" * 64,
        "image": spec.image_for(stage),
        "cpuset_cpus": limit.cpuset_cpus,
        "memory_limit_bytes": limit.memory_bytes,
        "memory_peak_bytes": min(1_048_576, limit.memory_bytes),
        "memory_events": dict(memory_events),
        "network_mode": "none",
        "ipc_mode": "private",
        "read_only_rootfs": True,
        "mounts": supervisor._expected_mounts(spec, stage),
        "cgroup_path": "/system.slice/docker-test.scope",
        "command": list(supervisor.docker_commands_v7(spec)[stage]),
        "environment_overrides": {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TZ": "UTC",
        },
        "cap_drop": ["ALL"],
        "nofile_limit": supervisor.NOFILE_LIMIT,
        "tmpfs": supervisor.TMPFS_SPEC,
        "seccomp_profile_raw_sha256": spec.seccomp_profile_raw_sha256,
        "measurement_interval_start_monotonic_ns": 10,
        "actor_return_monotonic_ns": 20,
        "container_exit_monotonic_ns": 30,
        "measurement_wall_time_ns": 20,
        "wall_time_limit_ns": limit.wall_time_seconds * 1_000_000_000,
        "maximum_concurrent_actor_containers": 1,
        "actor_terminal_scratch_payload_bytes": 0,
        "supervisor_control_peak_payload_bytes": 0,
    }


def _failed_lineage_fixture(version: int) -> tuple[dict[str, object], ...]:
    if version == 2:
        capsule_digest = supervisor.FAILED_V2_CAPSULE_DIGEST
        authorization_digest = supervisor.FAILED_V2_AUTHORIZATION_DIGEST
        consumption_digest = supervisor.FAILED_V2_CONSUMPTION_RECORD_DIGEST
        terminal_digest = supervisor.FAILED_V2_TERMINAL_RECORD_DIGEST
        status = supervisor.FAILED_V2_TERMINAL_STATUS
        failure: dict[str, object] = {}
        terminal_actor_starts = {"host": 0, "python": 0, "rust": 0}
    elif version == 3:
        capsule_digest = supervisor.FAILED_V3_CAPSULE_DIGEST
        authorization_digest = supervisor.FAILED_V3_AUTHORIZATION_DIGEST
        consumption_digest = supervisor.FAILED_V3_CONSUMPTION_RECORD_DIGEST
        terminal_digest = supervisor.FAILED_V3_TERMINAL_RECORD_DIGEST
        status = supervisor.FAILED_V3_TERMINAL_STATUS
        failure = {
            "failure_code": "REJECT_Q1_START_SUCCESSOR_BUNDLE",
            "failure_detail": (
                "validator import failed: AttributeError: module 'importlib' has no attribute 'util'"
            ),
        }
        terminal_actor_starts = {"host": 0, "python": 0, "rust": 0}
    elif version == 4:
        capsule_digest = supervisor.FAILED_V4_CAPSULE_DIGEST
        authorization_digest = supervisor.FAILED_V4_AUTHORIZATION_DIGEST
        consumption_digest = supervisor.FAILED_V4_CONSUMPTION_RECORD_DIGEST
        terminal_digest = supervisor.FAILED_V4_TERMINAL_RECORD_DIGEST
        status = supervisor.FAILED_V4_TERMINAL_STATUS
        failure = {
            "failure_code": "Q1_START_ENDPOINT_REPLAY_MISMATCH",
            "failure_detail": (
                "python:Q1FormalHostReplayError:REJECT_Q1_FORMAL_ENDPOINT_RESULT: "
                "same_run_record_set_root differs from replay"
            ),
            "supervisor_outcome_raw_sha256": (
                supervisor.FAILED_V4_SUPERVISOR_OUTCOME_RAW_SHA256
            ),
        }
        terminal_actor_starts = {"host": 0, "python": 1, "rust": 0}
    elif version == 5:
        capsule_digest = supervisor.FAILED_V5_CAPSULE_DIGEST
        authorization_digest = supervisor.FAILED_V5_AUTHORIZATION_DIGEST
        consumption_digest = supervisor.FAILED_V5_CONSUMPTION_RECORD_DIGEST
        terminal_digest = supervisor.FAILED_V5_TERMINAL_RECORD_DIGEST
        status = supervisor.FAILED_V5_TERMINAL_STATUS
        failure = {
            "failure_code": "Q1_START_POST_INTENT_EXCEPTION",
            "failure_detail": (
                "PermissionError: [Errno 1] Operation not permitted: "
                "'/home/erzhu419/.local/state/hegel-machine/q1-start-v5-run/"
                "python-output/signature-02/stream-04/chunk-000000.framed'"
            ),
            "supervisor_outcome_raw_sha256": (
                supervisor.FAILED_V5_SUPERVISOR_OUTCOME_RAW_SHA256
            ),
        }
        terminal_actor_starts = {"host": 0, "python": 1, "rust": 0}
    elif version == 6:
        capsule_digest = supervisor.FAILED_V6_CAPSULE_DIGEST
        authorization_digest = supervisor.FAILED_V6_AUTHORIZATION_DIGEST
        consumption_digest = supervisor.FAILED_V6_CONSUMPTION_RECORD_DIGEST
        terminal_digest = supervisor.FAILED_V6_TERMINAL_RECORD_DIGEST
        status = supervisor.FAILED_V6_TERMINAL_STATUS
        failure = {
            "failure_code": "Q1_START_ACTOR_NONZERO",
            "failure_detail": "host:137",
            "supervisor_outcome_raw_sha256": (
                supervisor.FAILED_V6_SUPERVISOR_OUTCOME_RAW_SHA256
            ),
        }
        terminal_actor_starts = {"host": 1, "python": 1, "rust": 1}
    else:
        raise AssertionError(f"unsupported predecessor version {version}")
    capsule = {
        "schema_version": f"hegel-phase3a-q1-start-capsule/{version}",
        "status": f"Q1_START_V{version}_CAPSULE_READY_NOT_AUTHORIZED",
        "capsule_digest": capsule_digest,
    }
    authorization = {
        "schema_version": f"hegel-phase3a-q1-start-authorization/{version}",
        "status": f"Q1_START_V{version}_AUTHORIZATION_RECORDED_NOT_CONSUMED",
        "capsule_digest": capsule_digest,
        "authorization_digest": authorization_digest,
    }
    consumption = {
        "schema_version": f"hegel-phase3a-q1-start-consumption/{version}",
        "status": f"Q1_START_V{version}_AUTHORIZATION_CONSUMED",
        "capsule_digest": capsule_digest,
        "authorization_digest": authorization_digest,
        "record_digest": consumption_digest,
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
    }
    outcome = {
        "status": status,
        "capsule_digest": capsule_digest,
        "actor_start_count_by_actor": terminal_actor_starts,
        **failure,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }
    terminal = {
        "schema_version": f"hegel-phase3a-q1-start-terminal/{version}",
        "status": status,
        "record_digest": terminal_digest,
        "capsule_digest": capsule_digest,
        "authorization_digest": authorization_digest,
        "consumption_record_digest": consumption_digest,
        "logical_campaign_count": 1,
        "authorization_consumption_count": 1,
        "artifact_raw_sha256": None,
        "outcome": outcome,
    }
    return capsule, authorization, consumption, terminal


def _isolated_smoke_fixture(request: pytest.FixtureRequest) -> tuple[Path, Path, Path]:
    fixture = _native_test_root(request, "isolated-smoke")
    tools = fixture / "tools"
    package = fixture / "src" / "hegel_machine"
    run = fixture / "run"
    tools.mkdir(parents=True)
    package.mkdir(parents=True)
    run.mkdir()

    predecessor_paths: dict[tuple[int, str], Path] = {}
    predecessor_raw: dict[tuple[int, str], str] = {}
    labels = ("capsule", "authorization", "consumption", "terminal")
    for version in (2, 3, 4, 5):
        for label, value in zip(labels, _failed_lineage_fixture(version), strict=True):
            path = fixture / f"failed-v{version}-{label}.json"
            path.write_bytes(_canonical(value))
            predecessor_paths[version, label] = path
            predecessor_raw[version, label] = _digest(path.read_bytes())
    failed_v6_run = fixture / "failed-v6-run"
    failed_v6_checkpoints = failed_v6_run / "checkpoints"
    failed_v6_ledger = fixture / "failed-v6-ledger"
    failed_v6_checkpoints.mkdir(parents=True)
    failed_v6_ledger.mkdir()
    v6_capsule, v6_authorization, v6_consumption, v6_terminal = (
        _failed_lineage_fixture(6)
    )
    v6_supervisor_outcome = {
        "schema_version": "hegel-phase3a-q1-start-successor-supervisor-outcome/6",
        "status": supervisor.TERMINAL_STATUS,
        "error_code": "Q1_START_ACTOR_NONZERO",
        "error_detail": "host:137",
        "completed_stages": ["python", "rust"],
        "active_stage": "host",
        "resume_allowed": False,
        "actor_process_diagnostic": {
            "stage": "host",
            "return_code": 137,
            "outcome_known": True,
            "runtime_evidence": {
                "memory_limit_bytes": 67_108_864,
                "memory_peak_bytes": 67_108_864,
            },
        },
    }
    v6_paths = {
        "capsule": failed_v6_run / "capsule.json",
        "authorization": failed_v6_ledger / "owner.authorization.json",
        "consumption": failed_v6_ledger / "campaign.consumed.json",
        "terminal": failed_v6_ledger / "campaign.terminal.json",
        "supervisor_outcome": failed_v6_checkpoints / "terminal-outcome.json",
        "controller_outcome": failed_v6_run / "campaign.outcome.json",
        "oom_event_receipt": fixture / "failed-v6-oom-events.jsonl",
    }
    v6_paths["supervisor_outcome"].write_bytes(_canonical(v6_supervisor_outcome))
    predecessor_raw[6, "supervisor_outcome"] = _digest(
        v6_paths["supervisor_outcome"].read_bytes()
    )
    v6_terminal["outcome"]["supervisor_outcome_raw_sha256"] = predecessor_raw[
        6, "supervisor_outcome"
    ]
    v6_values = {
        "capsule": v6_capsule,
        "authorization": v6_authorization,
        "consumption": v6_consumption,
        "terminal": v6_terminal,
        "controller_outcome": v6_terminal["outcome"],
    }
    for label, value in v6_values.items():
        v6_paths[label].write_bytes(_canonical(value))
        predecessor_raw[6, label] = _digest(v6_paths[label].read_bytes())
    oom_container_id = "a" * 64
    oom_name = "hegel-q1-start-v6-host-7d1a3e618dd8"
    oom_events = []
    for action in ("create", "attach", "start", "oom", "die", "destroy"):
        attributes = {"name": oom_name}
        if action == "die":
            attributes["exitCode"] = "137"
        oom_events.append(
            {
                "Type": "container",
                "Action": action,
                "Actor": {"ID": oom_container_id, "Attributes": attributes},
            }
        )
    v6_paths["oom_event_receipt"].write_bytes(
        b"".join(_canonical(event) for event in oom_events)
    )
    predecessor_raw[6, "oom_event_receipt"] = _digest(
        v6_paths["oom_event_receipt"].read_bytes()
    )
    for label, path in v6_paths.items():
        predecessor_paths[6, label] = path

    fake_docker = fixture / "fake-docker"
    fake_docker.write_text(
        "#!/usr/bin/python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "if sys.argv[1:3] == ['rm', '-f']:\n"
        "    raise SystemExit(1)\n"
        "mount = sys.argv[sys.argv.index('--mount') + 1]\n"
        "fields = dict(item.split('=', 1) for item in mount.split(',') if '=' in item)\n"
        "target = Path(fields['src']) / 'signature-02/stream-04/chunk-000000.framed'\n"
        "target.write_bytes(b'q1-start-v7-owner-probe\\n')\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o555)

    source_text = (ROOT / "tools" / "phase3_q1_start_supervisor_v7.py").read_text(
        encoding="utf-8"
    )
    docker_binding = 'DOCKER: Final = "/usr/bin/docker"'
    assert source_text.count(docker_binding) == 1
    source_text = source_text.replace(
        docker_binding,
        f"DOCKER: Final = {str(fake_docker)!r}",
    )
    raw_constant_names = {
        (2, "capsule"): "FAILED_V2_CAPSULE_RAW_SHA256",
        (2, "authorization"): "FAILED_V2_AUTHORIZATION_RAW_SHA256",
        (2, "consumption"): "FAILED_V2_CONSUMPTION_RAW_SHA256",
        (2, "terminal"): "FAILED_V2_TERMINAL_RAW_SHA256",
        (3, "capsule"): "FAILED_V3_CAPSULE_RAW_SHA256",
        (3, "authorization"): "FAILED_V3_AUTHORIZATION_RAW_SHA256",
        (3, "consumption"): "FAILED_V3_CONSUMPTION_RAW_SHA256",
        (3, "terminal"): "FAILED_V3_TERMINAL_RAW_SHA256",
        (4, "capsule"): "FAILED_V4_CAPSULE_RAW_SHA256",
        (4, "authorization"): "FAILED_V4_AUTHORIZATION_RAW_SHA256",
        (4, "consumption"): "FAILED_V4_CONSUMPTION_RAW_SHA256",
        (4, "terminal"): "FAILED_V4_TERMINAL_RAW_SHA256",
        (5, "capsule"): "FAILED_V5_CAPSULE_RAW_SHA256",
        (5, "authorization"): "FAILED_V5_AUTHORIZATION_RAW_SHA256",
        (5, "consumption"): "FAILED_V5_CONSUMPTION_RAW_SHA256",
        (5, "terminal"): "FAILED_V5_TERMINAL_RAW_SHA256",
        (6, "capsule"): "FAILED_V6_CAPSULE_RAW_SHA256",
        (6, "authorization"): "FAILED_V6_AUTHORIZATION_RAW_SHA256",
        (6, "consumption"): "FAILED_V6_CONSUMPTION_RAW_SHA256",
        (6, "terminal"): "FAILED_V6_TERMINAL_RAW_SHA256",
        (6, "supervisor_outcome"): "FAILED_V6_SUPERVISOR_OUTCOME_RAW_SHA256",
        (6, "controller_outcome"): "FAILED_V6_CONTROLLER_OUTCOME_RAW_SHA256",
        (6, "oom_event_receipt"): "FAILED_V6_OOM_EVENT_RECEIPT_RAW_SHA256",
    }
    for key, constant_name in raw_constant_names.items():
        original = getattr(supervisor, constant_name)
        replacement = predecessor_raw[key]
        assert source_text.count(original) == 1
        source_text = source_text.replace(original, replacement)
    script = tools / "phase3_q1_start_supervisor_v7.py"
    script.write_text(source_text, encoding="utf-8")

    (package / "strict_cbor_v1.py").write_text("TOKEN = 'loaded'\n", encoding="utf-8")
    manifest_roots = {
        "python": "sha256:" + "7" * 64,
        "rust": "sha256:" + "8" * 64,
        "host": "sha256:" + "9" * 64,
    }
    (package / "phase3_q1_prestart_successor_candidate_v7.py").write_text(
        "from .strict_cbor_v1 import TOKEN\n"
        "import json\n"
        f"ROOTS = {manifest_roots!r}\n"
        "def replay_q1_prestart_successor_candidate_v7(payload):\n"
        "    assert TOKEN == 'loaded'\n"
        "    return json.loads(payload.decode('ascii'))\n"
        "def successor_bundle_root_v7(value): return value['bundle_root']\n"
        "def actor_source_manifest_root_v7(actor, rows): return ROOTS[actor]\n",
        encoding="utf-8",
    )

    manifests: dict[str, list[dict[str, object]]] = {}
    for stage in supervisor.STAGES:
        source_root = run / f"{stage}-source"
        source_root.mkdir()
        payload = f"{stage}-isolated-v7\n".encode("ascii")
        (source_root / "actor.py").write_bytes(payload)
        manifests[stage] = [{
            "relative_path": "actor.py",
            "byte_count": len(payload),
            "raw_sha256": _digest(payload),
        }]
    rust_binary = fixture / "rust-binary"
    rust_binary.write_bytes(b"rust-v7-isolated")
    rust_binary.chmod(0o555)
    admission = fixture / "admission.json"
    admission.write_bytes(_canonical({"kind": "admission"}))
    seccomp = fixture / "seccomp.json"
    seccomp.write_bytes(_canonical({"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}))

    digest = lambda token: "sha256:" + token * 64
    source_commit = "b" * 40
    bundle_root = digest("3")
    receipt_root = digest("d")
    run_genesis_root = digest("c")
    implementation_roots = {
        "python": digest("1"), "rust": digest("2"), "host": digest("3")
    }
    corrected_root = digest("6")
    bundle = {
        "schema_version": supervisor.SUCCESSOR_BUNDLE_SCHEMA,
        "status": supervisor.SUCCESSOR_BUNDLE_STATUS,
        "bundle_root": bundle_root,
        "receipt_root": receipt_root,
        "run_genesis_root": run_genesis_root,
        "actor_implementation_roots": implementation_roots,
        "actor_source_manifest_roots": manifest_roots,
        "actor_source_closure_root": digest("a"),
        "corrected_resource_envelope_root": corrected_root,
        "source_identity": {"source_commit": source_commit, "source_tree": digest("b")},
        "corrected_resource_envelope": {
            "endpoint": {
                "memory_bytes": 2_147_483_648,
                "output_bytes": 16_777_216,
                "scratch_bytes": 134_217_728,
                "wall_time_seconds": 3_600,
            },
            "trusted_host_replay": {
                "memory_bytes": 2_147_483_648,
                "output_bytes": 2_097_152,
                "scratch_bytes": 1,
                "wall_time_seconds": 3_600,
            },
        },
        "intended_q1_state": "NOT_RUN",
        "intended_q1_gate_count": 14,
        "intended_q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "candidate_only": True,
        "authorized": False,
        "authorization_consumption_count": 0,
        "campaign_execution_count": 0,
        "popen_count": 0,
        "published": False,
    }
    bundle_path = fixture / "successor-bundle.json"
    bundle_path.write_bytes(_canonical(bundle))
    base = {
        "status": supervisor.BASE_PRESTART_STATUS,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_total": 20,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "q1_execution_started": False,
        "future_q1_start_requires_distinct_authorization": True,
        "authorization_consumption_count": 1,
        "campaign_execution_count": 1,
        "popen_count": 0,
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }
    base_path = fixture / "base-prestart.json"
    base_path.write_bytes(_canonical(base))

    capsule_digest = digest("1")
    v2_descriptor = {
        "capsule_path": str(predecessor_paths[2, "capsule"]),
        "capsule_raw_sha256": predecessor_raw[2, "capsule"],
        "capsule_digest": supervisor.FAILED_V2_CAPSULE_DIGEST,
        "authorization_path": str(predecessor_paths[2, "authorization"]),
        "authorization_raw_sha256": predecessor_raw[2, "authorization"],
        "authorization_digest": supervisor.FAILED_V2_AUTHORIZATION_DIGEST,
        "consumption_path": str(predecessor_paths[2, "consumption"]),
        "consumption_raw_sha256": predecessor_raw[2, "consumption"],
        "consumption_record_digest": supervisor.FAILED_V2_CONSUMPTION_RECORD_DIGEST,
        "terminal_path": str(predecessor_paths[2, "terminal"]),
        "terminal_raw_sha256": predecessor_raw[2, "terminal"],
        "terminal_record_digest": supervisor.FAILED_V2_TERMINAL_RECORD_DIGEST,
    }
    v3_descriptor = {
        "capsule_path": str(predecessor_paths[3, "capsule"]),
        "capsule_raw_sha256": predecessor_raw[3, "capsule"],
        "capsule_digest": supervisor.FAILED_V3_CAPSULE_DIGEST,
        "authorization_path": str(predecessor_paths[3, "authorization"]),
        "authorization_raw_sha256": predecessor_raw[3, "authorization"],
        "authorization_digest": supervisor.FAILED_V3_AUTHORIZATION_DIGEST,
        "consumption_path": str(predecessor_paths[3, "consumption"]),
        "consumption_raw_sha256": predecessor_raw[3, "consumption"],
        "consumption_record_digest": supervisor.FAILED_V3_CONSUMPTION_RECORD_DIGEST,
        "terminal_path": str(predecessor_paths[3, "terminal"]),
        "terminal_raw_sha256": predecessor_raw[3, "terminal"],
        "terminal_record_digest": supervisor.FAILED_V3_TERMINAL_RECORD_DIGEST,
        "terminal_no_resume": True,
        "work_product_reuse_allowed": False,
        "prior_actor_actions_conservatively_counted": False,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
    }
    v4_descriptor = {
        "capsule_path": str(predecessor_paths[4, "capsule"]),
        "capsule_raw_sha256": predecessor_raw[4, "capsule"],
        "capsule_digest": supervisor.FAILED_V4_CAPSULE_DIGEST,
        "authorization_path": str(predecessor_paths[4, "authorization"]),
        "authorization_raw_sha256": predecessor_raw[4, "authorization"],
        "authorization_digest": supervisor.FAILED_V4_AUTHORIZATION_DIGEST,
        "consumption_path": str(predecessor_paths[4, "consumption"]),
        "consumption_raw_sha256": predecessor_raw[4, "consumption"],
        "consumption_record_digest": supervisor.FAILED_V4_CONSUMPTION_RECORD_DIGEST,
        "terminal_path": str(predecessor_paths[4, "terminal"]),
        "terminal_raw_sha256": predecessor_raw[4, "terminal"],
        "terminal_record_digest": supervisor.FAILED_V4_TERMINAL_RECORD_DIGEST,
        "status": supervisor.FAILED_V4_TERMINAL_STATUS,
        "failure_code": "Q1_START_ENDPOINT_REPLAY_MISMATCH",
        "failure_detail": (
            "python:Q1FormalHostReplayError:REJECT_Q1_FORMAL_ENDPOINT_RESULT: "
            "same_run_record_set_root differs from replay"
        ),
        "actor_start_count_by_actor": {"host": 0, "python": 1, "rust": 0},
        "completed_prefix": [],
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "terminal_no_resume": True,
        "prior_actor_actions_conservatively_counted": True,
        "work_product_reuse_allowed": False,
    }
    v5_descriptor = {
        "capsule_path": str(predecessor_paths[5, "capsule"]),
        "capsule_raw_sha256": predecessor_raw[5, "capsule"],
        "capsule_digest": supervisor.FAILED_V5_CAPSULE_DIGEST,
        "authorization_path": str(predecessor_paths[5, "authorization"]),
        "authorization_raw_sha256": predecessor_raw[5, "authorization"],
        "authorization_digest": supervisor.FAILED_V5_AUTHORIZATION_DIGEST,
        "consumption_path": str(predecessor_paths[5, "consumption"]),
        "consumption_raw_sha256": predecessor_raw[5, "consumption"],
        "consumption_record_digest": supervisor.FAILED_V5_CONSUMPTION_RECORD_DIGEST,
        "terminal_path": str(predecessor_paths[5, "terminal"]),
        "terminal_raw_sha256": predecessor_raw[5, "terminal"],
        "terminal_record_digest": supervisor.FAILED_V5_TERMINAL_RECORD_DIGEST,
        "status": supervisor.FAILED_V5_TERMINAL_STATUS,
        "failure_code": "Q1_START_POST_INTENT_EXCEPTION",
        "failure_detail": (
            "PermissionError: [Errno 1] Operation not permitted: "
            "'/home/erzhu419/.local/state/hegel-machine/q1-start-v5-run/"
            "python-output/signature-02/stream-04/chunk-000000.framed'"
        ),
        "actor_start_count_by_actor": {"host": 0, "python": 1, "rust": 0},
        "completed_prefix": [],
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "terminal_no_resume": True,
        "prior_actor_actions_conservatively_counted": True,
        "work_product_reuse_allowed": False,
    }
    v6_descriptor = {
        "run_root": str(predecessor_paths[6, "capsule"].parent),
        "ledger_root": str(predecessor_paths[6, "authorization"].parent),
        "capsule_path": str(predecessor_paths[6, "capsule"]),
        "capsule_raw_sha256": predecessor_raw[6, "capsule"],
        "capsule_digest": supervisor.FAILED_V6_CAPSULE_DIGEST,
        "authorization_path": str(predecessor_paths[6, "authorization"]),
        "authorization_raw_sha256": predecessor_raw[6, "authorization"],
        "authorization_digest": supervisor.FAILED_V6_AUTHORIZATION_DIGEST,
        "consumption_path": str(predecessor_paths[6, "consumption"]),
        "consumption_raw_sha256": predecessor_raw[6, "consumption"],
        "consumption_record_digest": supervisor.FAILED_V6_CONSUMPTION_RECORD_DIGEST,
        "terminal_path": str(predecessor_paths[6, "terminal"]),
        "terminal_raw_sha256": predecessor_raw[6, "terminal"],
        "terminal_record_digest": supervisor.FAILED_V6_TERMINAL_RECORD_DIGEST,
        "supervisor_outcome_path": str(predecessor_paths[6, "supervisor_outcome"]),
        "supervisor_outcome_raw_sha256": predecessor_raw[6, "supervisor_outcome"],
        "controller_outcome_path": str(predecessor_paths[6, "controller_outcome"]),
        "controller_outcome_raw_sha256": predecessor_raw[6, "controller_outcome"],
        "oom_event_receipt_path": str(predecessor_paths[6, "oom_event_receipt"]),
        "oom_event_receipt_raw_sha256": predecessor_raw[6, "oom_event_receipt"],
        "status": supervisor.FAILED_V6_TERMINAL_STATUS,
        "failure_code": "Q1_START_ACTOR_NONZERO",
        "failure_detail": "host:137",
        "actor_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
        "completed_prefix": ["python", "rust"],
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "terminal_no_resume": True,
        "prior_actor_actions_conservatively_counted": True,
        "resume_allowed": False,
        "work_product_reuse_allowed": False,
        "diagnostic_host_resource_peak_reuse_allowed": False,
    }
    capsule = {
        "schema_version": supervisor.START_CAPSULE_SCHEMA,
        "status": "Q1_START_V7_CAPSULE_READY_NOT_AUTHORIZED",
        "capsule_digest": capsule_digest,
        "formal_source_commit": source_commit,
        "formal_source_tree": bundle["source_identity"]["source_tree"],
        "successor_prestart_bundle": bundle,
        "successor_prestart_bundle_descriptor": {
            "path": str(bundle_path),
            "raw_sha256": _digest(bundle_path.read_bytes()),
            "bundle_root": bundle_root,
        },
        "base_authoritative_predecessor": {
            "prestart_artifact_path": str(base_path),
            "prestart_artifact_raw_sha256": _digest(base_path.read_bytes()),
        },
        "failed_start_v2_predecessor": v2_descriptor,
        "failed_start_v3_predecessor": v3_descriptor,
        "failed_start_v4_predecessor": v4_descriptor,
        "failed_start_v5_predecessor": v5_descriptor,
        "failed_start_v6_predecessor": v6_descriptor,
    }
    capsule_path = run / "capsule.json"
    capsule_path.write_bytes(_canonical(capsule))
    authorization_digest = digest("2")
    authorization = {
        "schema_version": supervisor.START_AUTHORIZATION_SCHEMA,
        "status": "Q1_START_V7_AUTHORIZATION_RECORDED_NOT_CONSUMED",
        "capsule_digest": capsule_digest,
        "authorization_digest": authorization_digest,
        "authorized_scope": supervisor.START_AUTHORIZATION_SCOPE,
        "successor_prestart_bundle_raw_sha256": _digest(bundle_path.read_bytes()),
        "successor_prestart_bundle_root": bundle_root,
        "base_authoritative_artifact_raw_sha256": _digest(base_path.read_bytes()),
        "authorization_consumption_limit": 1,
        "maximum_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
        "network_allowed": False,
        "q1_start_authorized": True,
        "q2_start_authorized": False,
        "adoption_promotion_current_active_authorized": False,
        "fresh_v7_actor_actions_authorized": True,
        "failed_start_v2_resume_authorized": False,
        "failed_start_v2_work_product_reuse_authorized": False,
        "failed_start_v3_resume_authorized": False,
        "failed_start_v3_work_product_reuse_authorized": False,
        "failed_start_v4_resume_authorized": False,
        "failed_start_v4_work_product_reuse_authorized": False,
        "failed_start_v5_resume_authorized": False,
        "failed_start_v5_work_product_reuse_authorized": False,
        "failed_start_v6_resume_authorized": False,
        "failed_start_v6_work_product_reuse_authorized": False,
        "preconsumption_startup_smoke_popen_limit_per_entry": 1,
        "startup_smoke_failure_preconsumption_pending": True,
    }
    for version, descriptor in (
        (2, v2_descriptor),
        (3, v3_descriptor),
        (4, v4_descriptor),
        (5, v5_descriptor),
        (6, v6_descriptor),
    ):
        for name in ("capsule", "authorization", "consumption", "terminal"):
            authorization[f"failed_start_v{version}_{name}_raw_sha256"] = descriptor[
                f"{name}_raw_sha256"
            ]
        authorization[f"failed_start_v{version}_capsule_digest"] = descriptor["capsule_digest"]
        authorization[f"failed_start_v{version}_authorization_digest"] = descriptor[
            "authorization_digest"
        ]
        authorization[f"failed_start_v{version}_consumption_record_digest"] = descriptor[
            "consumption_record_digest"
        ]
        authorization[f"failed_start_v{version}_terminal_record_digest"] = descriptor[
            "terminal_record_digest"
        ]
    authorization["failed_start_v6_supervisor_outcome_raw_sha256"] = v6_descriptor[
        "supervisor_outcome_raw_sha256"
    ]
    authorization["failed_start_v6_controller_outcome_raw_sha256"] = v6_descriptor[
        "controller_outcome_raw_sha256"
    ]
    authorization["failed_start_v6_oom_event_receipt_raw_sha256"] = v6_descriptor[
        "oom_event_receipt_raw_sha256"
    ]
    authorization_path = fixture / "authorization.json"
    authorization_path.write_bytes(_canonical(authorization))

    limits = {
        "python": supervisor.RuntimeLimitV7("24", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "rust": supervisor.RuntimeLimitV7("25", 2_147_483_648, 16_777_216, 134_217_728, 16_777_216, 3_600).object(),
        "host": supervisor.RuntimeLimitV7("26", 2_147_483_648, 2_097_152, 1, 2_097_152, 3_600).object(),
    }
    run_spec: dict[str, object] = {
        "schema_version": supervisor.RUN_SPEC_SCHEMA,
        "start_capsule_digest": capsule_digest,
        "start_capsule": str(capsule_path),
        "start_capsule_raw_sha256": _digest(capsule_path.read_bytes()),
        "start_authorization": str(authorization_path),
        "start_authorization_raw_sha256": _digest(authorization_path.read_bytes()),
        "start_authorization_digest": authorization_digest,
        "successor_prestart_bundle": str(bundle_path),
        "successor_prestart_bundle_raw_sha256": _digest(bundle_path.read_bytes()),
        "successor_prestart_bundle_root": bundle_root,
        "base_authoritative_artifact": str(base_path),
        "base_authoritative_artifact_raw_sha256": _digest(base_path.read_bytes()),
        "corrected_resource_envelope_root": corrected_root,
        "actor_source_manifest_roots": manifest_roots,
        "actor_source_closure_root": digest("a"),
        "source_commit": source_commit,
        "run_genesis_root": run_genesis_root,
        "prestart_admission_receipt_root": receipt_root,
        "python_image": "python@" + digest("e"),
        "rust_image": "rust@" + digest("f"),
        "container_user": supervisor._host_effective_user(),
        "rust_binary_raw_sha256": _digest(rust_binary.read_bytes()),
        "seccomp_profile_raw_sha256": _digest(seccomp.read_bytes()),
        "python_admission_raw_sha256": _digest(admission.read_bytes()),
        "actor_implementation_roots": implementation_roots,
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
    }
    for version in (2, 3, 4, 5, 6):
        run_spec[f"failed_start_v{version}_capsule"] = str(predecessor_paths[version, "capsule"])
        run_spec[f"failed_start_v{version}_capsule_raw_sha256"] = predecessor_raw[version, "capsule"]
        run_spec[f"failed_start_v{version}_capsule_digest"] = getattr(
            supervisor, f"FAILED_V{version}_CAPSULE_DIGEST"
        )
        run_spec[f"failed_start_v{version}_authorization"] = str(
            predecessor_paths[version, "authorization"]
        )
        run_spec[f"failed_start_v{version}_authorization_raw_sha256"] = predecessor_raw[
            version, "authorization"
        ]
        run_spec[f"failed_start_v{version}_authorization_digest"] = getattr(
            supervisor, f"FAILED_V{version}_AUTHORIZATION_DIGEST"
        )
        run_spec[f"failed_start_v{version}_consumption"] = str(
            predecessor_paths[version, "consumption"]
        )
        run_spec[f"failed_start_v{version}_consumption_raw_sha256"] = predecessor_raw[
            version, "consumption"
        ]
        run_spec[f"failed_start_v{version}_consumption_record_digest"] = getattr(
            supervisor, f"FAILED_V{version}_CONSUMPTION_RECORD_DIGEST"
        )
        run_spec[f"failed_start_v{version}_terminal"] = str(predecessor_paths[version, "terminal"])
        run_spec[f"failed_start_v{version}_terminal_raw_sha256"] = predecessor_raw[
            version, "terminal"
        ]
        run_spec[f"failed_start_v{version}_terminal_record_digest"] = getattr(
            supervisor, f"FAILED_V{version}_TERMINAL_RECORD_DIGEST"
        )
    run_spec["failed_start_v6_supervisor_outcome"] = str(
        predecessor_paths[6, "supervisor_outcome"]
    )
    run_spec["failed_start_v6_supervisor_outcome_raw_sha256"] = predecessor_raw[
        6, "supervisor_outcome"
    ]
    run_spec["failed_start_v6_controller_outcome"] = str(
        predecessor_paths[6, "controller_outcome"]
    )
    run_spec["failed_start_v6_controller_outcome_raw_sha256"] = predecessor_raw[
        6, "controller_outcome"
    ]
    run_spec["failed_start_v6_oom_event_receipt"] = str(
        predecessor_paths[6, "oom_event_receipt"]
    )
    run_spec["failed_start_v6_oom_event_receipt_raw_sha256"] = predecessor_raw[
        6, "oom_event_receipt"
    ]
    run_spec_path = fixture / "run-spec.json"
    run_spec_path.write_bytes(_canonical(run_spec))
    return script, run_spec_path, run


def test_v7_run_spec_binds_full_v2_through_v6_lineage_and_2_gib_host_memory(
    tmp_path: Path,
) -> None:
    payload = _run_spec_payload(tmp_path)
    spec = supervisor.load_run_spec_v7(payload)
    assert spec.start_capsule_digest != spec.failed_start_v2_capsule_digest
    assert spec.start_capsule_digest != spec.failed_start_v3_capsule_digest
    assert spec.start_capsule_digest != spec.failed_start_v4_capsule_digest
    assert spec.start_capsule_digest != spec.failed_start_v5_capsule_digest
    assert spec.start_capsule_digest != spec.failed_start_v6_capsule_digest
    assert spec.container_user == supervisor._host_effective_user()
    assert spec.runtime_limits["python"].scratch_bytes == 134_217_728
    assert spec.runtime_limits["host"].memory_bytes == 2_147_483_648
    assert set(spec.authority_binding()) == supervisor._AUTHORITY_BINDING_KEYS
    assert spec.start_capsule.parent == spec.run_root

    value = json.loads(payload)
    value["runtime_limits"]["python"]["scratch_bytes"] = 4_194_304
    with pytest.raises(supervisor.Q1StartSupervisorError, match="frozen resource envelope differs"):
        supervisor.load_run_spec_v7(_canonical(value))


def test_v7_run_spec_rejects_capsule_outside_run_root_and_binds_full_authorization_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = json.loads(_run_spec_payload(tmp_path))
    outside = tmp_path / "outside-capsule.json"
    outside.write_bytes(Path(value["start_capsule"]).read_bytes())
    value["start_capsule"] = str(outside)
    value["start_capsule_raw_sha256"] = _digest(outside.read_bytes())
    with pytest.raises(supervisor.Q1StartSupervisorError, match="direct run-root child"):
        supervisor.load_run_spec_v7(_canonical(value))

    second = tmp_path / "second"
    second.mkdir()
    value = json.loads(_run_spec_payload(second))
    spec = supervisor.load_run_spec_v7(_canonical(value))
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
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    authority_calls: list[str] = []
    launches: list[str] = []

    def authority(_spec: supervisor.Q1StartRunSpecV7) -> dict[str, object]:
        authority_calls.append("replay")
        return {"authority": "exact"}

    def launcher(stage: str, _command: object, _spec: object) -> supervisor.ActorProcessResultV7:
        launches.append(stage)
        return supervisor.ActorProcessResultV7(0, b"", b"", True, {})

    def validate_endpoint(stage: str, _process: object, _diagnostic: object, current: supervisor.Q1StartRunSpecV7, _replayer: object):
        result = {"stage": stage}
        (current.output_for(stage) / "endpoint-result.json").write_bytes(_canonical(result))
        return result, {"runtime_evidence": {}}, len(_canonical(result))

    def validate_host(_process: object, _diagnostic: object, current: supervisor.Q1StartRunSpecV7):
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
    outcome = supervisor.run_supervisor_v7(
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
        assert intent["failed_start_v3_terminal_record_digest"] == spec.failed_start_v3_terminal_record_digest
        assert intent["failed_start_v4_terminal_record_digest"] == spec.failed_start_v4_terminal_record_digest
        assert intent["failed_start_v5_terminal_record_digest"] == spec.failed_start_v5_terminal_record_digest
        assert intent["failed_start_v6_terminal_record_digest"] == spec.failed_start_v6_terminal_record_digest
        assert intent["container_user"] == spec.container_user


def test_second_preintent_lineage_replay_pending_never_publishes_intent_or_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    calls = 0
    launches: list[str] = []

    def authority(_spec: supervisor.Q1StartRunSpecV7) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise supervisor.PendingCondition("Q1_START_BOUND_FILE_PENDING", "capsule changed")
        return {"authority": "exact"}

    monkeypatch.setattr(supervisor, "_replay_successor_authority", authority)
    monkeypatch.setattr(supervisor, "_replay_stage_bindings", lambda stage, current: {})
    outcome = supervisor.run_supervisor_v7(
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
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
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


def test_failed_v3_terminal_must_be_exact_import_failure_no_actor_lineage(
    tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    capsule = {
        "schema_version": supervisor.FAILED_V3_CAPSULE_SCHEMA,
        "status": "Q1_START_V3_CAPSULE_READY_NOT_AUTHORIZED",
        "capsule_digest": spec.failed_start_v3_capsule_digest,
    }
    authorization = {
        "schema_version": supervisor.FAILED_V3_AUTHORIZATION_SCHEMA,
        "status": "Q1_START_V3_AUTHORIZATION_RECORDED_NOT_CONSUMED",
        "capsule_digest": spec.failed_start_v3_capsule_digest,
        "authorization_digest": spec.failed_start_v3_authorization_digest,
    }
    consumption = {
        "schema_version": supervisor.FAILED_V3_CONSUMPTION_SCHEMA,
        "status": "Q1_START_V3_AUTHORIZATION_CONSUMED",
        "capsule_digest": spec.failed_start_v3_capsule_digest,
        "authorization_digest": spec.failed_start_v3_authorization_digest,
        "record_digest": spec.failed_start_v3_consumption_record_digest,
        "authorization_consumption_count": 1,
        "logical_campaign_count": 1,
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
    }
    terminal = {
        "schema_version": supervisor.FAILED_V3_TERMINAL_SCHEMA,
        "status": supervisor.FAILED_V3_TERMINAL_STATUS,
        "record_digest": spec.failed_start_v3_terminal_record_digest,
        "capsule_digest": spec.failed_start_v3_capsule_digest,
        "authorization_digest": spec.failed_start_v3_authorization_digest,
        "consumption_record_digest": spec.failed_start_v3_consumption_record_digest,
        "logical_campaign_count": 1,
        "authorization_consumption_count": 1,
        "artifact_raw_sha256": None,
        "outcome": {
            "status": supervisor.FAILED_V3_TERMINAL_STATUS,
            "capsule_digest": spec.failed_start_v3_capsule_digest,
            "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
            "failure_code": "REJECT_Q1_START_SUCCESSOR_BUNDLE",
            "failure_detail": (
                "validator import failed: AttributeError: module 'importlib' has no attribute 'util'"
            ),
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
    supervisor._validate_failed_v3_lineage(
        spec, capsule, authorization, consumption, terminal
    )
    terminal["outcome"]["failure_detail"] = "different failure"
    with pytest.raises(supervisor.Q1StartSupervisorError, match="terminal no-resume state differs"):
        supervisor._validate_failed_v3_lineage(
            spec, capsule, authorization, consumption, terminal
        )


def test_failed_v4_terminal_must_be_exact_python_once_lineage(
    tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    capsule, authorization, consumption, terminal = _failed_lineage_fixture(4)
    supervisor._validate_failed_v4_lineage(
        spec, capsule, authorization, consumption, terminal
    )
    terminal["outcome"]["actor_start_count_by_actor"]["python"] = 0
    with pytest.raises(
        supervisor.Q1StartSupervisorError,
        match="terminal no-resume state differs",
    ):
        supervisor._validate_failed_v4_lineage(
            spec, capsule, authorization, consumption, terminal
        )


def test_failed_v5_terminal_must_be_exact_permission_failure_python_once_lineage(
    tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    capsule, authorization, consumption, terminal = _failed_lineage_fixture(5)
    supervisor._validate_failed_v5_lineage(
        spec, capsule, authorization, consumption, terminal
    )
    terminal["outcome"]["failure_detail"] = "different permission failure"
    with pytest.raises(
        supervisor.Q1StartSupervisorError,
        match="terminal no-resume state differs",
    ):
        supervisor._validate_failed_v5_lineage(
            spec, capsule, authorization, consumption, terminal
        )


def test_failed_v6_terminal_and_oom_receipt_are_exact_host_137_lineage(
    tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    capsule, authorization, consumption, terminal = _failed_lineage_fixture(6)
    supervisor_outcome = {
        "schema_version": "hegel-phase3a-q1-start-successor-supervisor-outcome/6",
        "status": supervisor.TERMINAL_STATUS,
        "error_code": "Q1_START_ACTOR_NONZERO",
        "error_detail": "host:137",
        "completed_stages": ["python", "rust"],
        "active_stage": "host",
        "resume_allowed": False,
        "actor_process_diagnostic": {
            "stage": "host",
            "return_code": 137,
            "outcome_known": True,
            "runtime_evidence": {
                "memory_limit_bytes": 67_108_864,
                "memory_peak_bytes": 67_108_864,
            },
        },
    }
    events = [
        {
            "Type": "container",
            "Action": action,
            "Actor": {
                "ID": "a" * 64,
                "Attributes": {
                    "name": "hegel-q1-start-v6-host-7d1a3e618dd8",
                    **({"exitCode": "137"} if action == "die" else {}),
                },
            },
        }
        for action in ("create", "attach", "start", "oom", "die", "destroy")
    ]
    supervisor._validate_failed_v6_lineage(
        spec,
        capsule,
        authorization,
        consumption,
        terminal,
        supervisor_outcome,
        terminal["outcome"],
        events,
    )
    events[4]["Actor"]["Attributes"]["exitCode"] = "1"
    with pytest.raises(supervisor.Q1StartSupervisorError, match="OOM event identity differs"):
        supervisor._validate_failed_v6_lineage(
            spec,
            capsule,
            authorization,
            consumption,
            terminal,
            supervisor_outcome,
            terminal["outcome"],
            events,
        )


def test_startup_smoke_rejects_v4_descriptor_with_nonempty_completed_prefix(
    request: pytest.FixtureRequest,
) -> None:
    script, run_spec_path, run_root = _isolated_smoke_fixture(request)
    run_spec = json.loads(run_spec_path.read_bytes())
    capsule_path = Path(run_spec["start_capsule"])
    capsule = json.loads(capsule_path.read_bytes())
    capsule["failed_start_v4_predecessor"]["completed_prefix"] = ["python"]
    capsule_path.write_bytes(_canonical(capsule))
    run_spec["start_capsule_raw_sha256"] = _digest(capsule_path.read_bytes())
    run_spec_path.write_bytes(_canonical(run_spec))
    before = sorted(str(path.relative_to(run_root)) for path in run_root.rglob("*"))
    process = subprocess.run(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            str(script),
            "--startup-smoke",
            "--run-spec",
            str(run_spec_path),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    result = json.loads(process.stdout)
    assert process.returncode == 1
    assert result["error_code"] == "REJECT_Q1_START_CAPSULE"
    assert result["campaign_supervisor_popen_count"] == 0
    assert result["actor_popen_count"] == 0
    assert result["checkpoint_or_mutable_roots_created"] is False
    assert sorted(str(path.relative_to(run_root)) for path in run_root.rglob("*")) == before


def test_authority_startup_smoke_runs_one_ownership_probe_and_zero_actor_mutation(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_spec_payload(_native_test_root(request, "authority-smoke"))
    spec = supervisor.load_run_spec_v7(payload)
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

    def ownership_probe(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append("ownership-probe")
        assert command[0] == supervisor.DOCKER
        assert command[1] == "run"
        assert command[command.index("--user") + 1] == spec.container_user
        assert command[command.index("--network") + 1] == "none"
        assert command[command.index("--pull") + 1] == "never"
        assert spec.python_image in command
        mount = command[command.index("--mount") + 1]
        fields = dict(item.split("=", 1) for item in mount.split(",") if "=" in item)
        target = (
            Path(fields["src"])
            / "signature-02/stream-04/chunk-000000.framed"
        )
        target.write_bytes(b"q1-start-v7-owner-probe\n")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        ownership_probe,
    )
    result = supervisor.authority_startup_smoke_v7(payload)
    after = sorted(str(path.relative_to(spec.run_root)) for path in spec.run_root.rglob("*"))
    assert result["status"] == supervisor.AUTHORITY_SMOKE_STATUS
    assert result["actor_popen_count"] == 0
    assert result["campaign_supervisor_popen_count"] == 0
    assert result["checkpoint_or_mutable_roots_created"] is False
    assert result["docker_invoked"] is True
    assert result["ownership_probe_popen_count"] == 1
    assert result["container_user"] == spec.container_user
    assert result["host_effective_user"] == supervisor._host_effective_user()
    assert result["container_user_matches_host_effective_user"] is True
    assert result["ownership_freeze_probe"] == {
        "schema_version": supervisor.OWNERSHIP_FREEZE_PROBE_SCHEMA,
        "status": supervisor.OWNERSHIP_FREEZE_PROBE_STATUS,
        "container_user": spec.container_user,
        "python_image": spec.python_image,
        "network_mode": "none",
        "container_exit_code": 0,
        "observed_uid": supervisor._host_effective_ids()[0],
        "observed_gid": supervisor._host_effective_ids()[1],
        "probe_file_relative_path": "signature-02/stream-04/chunk-000000.framed",
        "probe_file_mode_after_freeze": "0444",
        "probe_directory_mode_after_freeze": "0555",
        "host_freeze_succeeded": True,
        "docker_popen_count": 1,
        "probe_root_removed": True,
    }
    assert result["failed_start_v3_terminal_record_digest"] == spec.failed_start_v3_terminal_record_digest
    assert result["failed_start_v4_terminal_record_digest"] == spec.failed_start_v4_terminal_record_digest
    assert result["failed_start_v5_terminal_record_digest"] == spec.failed_start_v5_terminal_record_digest
    assert result["failed_start_v6_terminal_record_digest"] == spec.failed_start_v6_terminal_record_digest
    assert before == after
    assert calls == ["ownership-probe"]
    assert not spec.checkpoint_root.exists()


def test_post_probe_mutable_root_failure_retains_successful_probe_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    monkeypatch.setattr(
        supervisor,
        "_replay_successor_authority",
        lambda current: {"capsule": current.start_capsule_digest},
    )
    monkeypatch.setattr(
        supervisor,
        "_replay_stage_bindings",
        lambda stage, _current: {"stage": stage},
    )

    def successful_probe_with_mutation(
        current: supervisor.Q1StartRunSpecV7,
    ) -> dict[str, object]:
        current.checkpoint_root.mkdir()
        return {
            "status": supervisor.OWNERSHIP_FREEZE_PROBE_STATUS,
            "docker_popen_count": 1,
            "probe_root_removed": True,
        }

    monkeypatch.setattr(
        supervisor,
        "_ownership_freeze_probe_v7",
        successful_probe_with_mutation,
    )
    with pytest.raises(supervisor.Q1StartSupervisorError) as captured:
        supervisor.authority_startup_smoke_v7(spec)
    error = captured.value
    assert error.code == "REJECT_Q1_START_AUTHORITY_SMOKE_MUTABLE_ROOT_PRESENT"
    assert error.startup_smoke_audit == {
        "ownership_probe_run_popen_count": 1,
        "ownership_probe_cleanup_popen_count": 0,
        "ownership_probe_absence_check_popen_count": 0,
        "ownership_probe_container_absence_proven": True,
        "ownership_probe_temp_root_absence_proven": True,
        "docker_invoked": True,
    }
    failure = supervisor._startup_smoke_failure(
        error.code,
        error.detail,
        error.startup_smoke_audit,
    )
    assert failure["docker_invoked"] is True
    assert failure["ownership_probe_run_popen_count"] == 1
    assert failure["ownership_probe_cleanup_popen_count"] == 0
    assert failure["ownership_probe_absence_check_popen_count"] == 0
    assert failure["ownership_probe_container_absence_proven"] is True
    assert failure["ownership_probe_temp_root_absence_proven"] is True


def test_v7_container_user_is_bound_into_every_actor_command(tmp_path: Path) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    assert spec.container_user == supervisor._host_effective_user()
    commands = supervisor.docker_commands_v7(spec)
    for command in commands.values():
        assert command.count("--user") == 1
        user_index = command.index("--user")
        assert command[user_index + 1] == spec.container_user
    host = commands["host"]
    assert host[host.index("--memory") + 1] == "2147483648"
    assert host[host.index("--memory-swap") + 1] == "2147483648"


def test_memory_events_are_captured_after_return_marker_before_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    spec.checkpoint_root.mkdir()
    spec.host_scratch.mkdir()
    return_receipt = spec.host_scratch / (
        supervisor.ACTOR_RETURN_PREFIX + "137" + supervisor.ACTOR_RETURN_SUFFIX
    )
    return_receipt.write_bytes(b"")
    memory_events = {"max": 9, "oom": 2, "oom_kill": 1, "oom_group_kill": 0}
    order: list[str] = []

    class FakeProcess:
        returncode = 137

        def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

        def poll(self) -> int:
            return 137

        def terminate(self) -> None:
            raise AssertionError("completed fake actor must not be terminated")

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(supervisor, "_read_cid", lambda *args: "b" * 64)

    def held_marker(*_args: object) -> tuple[int, Path]:
        order.append("return-marker")
        return 137, return_receipt

    def inspect(*_args: object) -> tuple[dict[str, object], int, dict[str, int]]:
        order.append("memory-events")
        assert not (spec.host_scratch / supervisor.ACTOR_RELEASE).exists()
        return (
            {
                "cgroup_path": "/system.slice/docker-test.scope",
                "mounts": supervisor._expected_mounts(spec, "host"),
            },
            2_147_483_648,
            memory_events,
        )

    monkeypatch.setattr(supervisor, "_wait_for_actor_receipt", held_marker)
    monkeypatch.setattr(supervisor, "_docker_inspect", inspect)
    result = supervisor.actual_launcher_v7(
        "host",
        supervisor.docker_commands_v7(spec)["host"],
        spec,
    )
    assert order == ["return-marker", "memory-events"]
    assert result.return_code == 137
    assert result.runtime_evidence["memory_events"] == memory_events
    diagnostic = supervisor._actor_process_diagnostic("host", result)
    assert diagnostic["runtime_evidence"]["memory_events"] == memory_events
    assert not (spec.host_scratch / supervisor.ACTOR_RELEASE).exists()


@pytest.mark.parametrize(
    ("oom_kill", "expected_code"),
    [
        (1, "Q1_START_ACTOR_CGROUP_OOM"),
        (0, "Q1_START_ACTOR_NONZERO"),
    ],
)
def test_exit_137_classification_requires_persisted_oom_kill_evidence(
    tmp_path: Path,
    oom_kill: int,
    expected_code: str,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    evidence = _runtime_evidence(
        spec,
        "host",
        memory_events={
            "max": 4,
            "oom": oom_kill,
            "oom_kill": oom_kill,
            "oom_group_kill": 0,
        },
    )
    process = supervisor.ActorProcessResultV7(137, b"", b"", True, evidence)
    diagnostic = supervisor._actor_process_diagnostic("host", process)
    with pytest.raises(supervisor.TerminalCondition) as captured:
        supervisor._validate_host(process, diagnostic, spec)
    assert captured.value.code == expected_code
    assert captured.value.detail == ("host" if oom_kill else "host:137")


def test_memory_events_reader_persists_required_cgroup_v2_counters(tmp_path: Path) -> None:
    events = tmp_path / "memory.events"
    events.write_text(
        "low 0\nhigh 0\nmax 7\noom 3\noom_kill 2\noom_group_kill 1\n",
        encoding="ascii",
    )
    assert supervisor._read_memory_events(events) == {
        "max": 7,
        "oom": 3,
        "oom_kill": 2,
        "oom_group_kill": 1,
    }


def test_probe_timeout_checks_rm_and_proves_container_absence(
    tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    calls: list[tuple[str, ...]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(tuple(command))
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 30)
        if command[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(command, 1, b"", b"already absent")
        assert command[1] == "ps"
        assert command[command.index("--filter") + 1].startswith("name=^/")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    with pytest.raises(supervisor.Q1StartSupervisorError) as captured:
        supervisor._ownership_freeze_probe_v7(spec, runner=runner)
    error = captured.value
    assert error.code == "REJECT_Q1_START_OUTPUT_OWNERSHIP_PROBE"
    assert "rm_exit=1" in error.detail
    assert "ps_exit=0" in error.detail
    assert error.startup_smoke_audit == {
        "ownership_probe_run_popen_count": 1,
        "ownership_probe_cleanup_popen_count": 1,
        "ownership_probe_absence_check_popen_count": 1,
        "ownership_probe_container_absence_proven": True,
        "ownership_probe_temp_root_absence_proven": True,
        "docker_invoked": True,
    }
    failure = supervisor._startup_smoke_failure(
        error.code,
        error.detail,
        error.startup_smoke_audit,
    )
    assert failure["docker_invoked"] is True
    assert failure["ownership_probe_run_popen_count"] == 1
    assert failure["ownership_probe_cleanup_popen_count"] == 1
    assert failure["ownership_probe_absence_check_popen_count"] == 1
    assert failure["ownership_probe_container_absence_proven"] is True
    assert failure["ownership_probe_temp_root_absence_proven"] is True
    assert [command[1] for command in calls] == ["run", "rm", "ps"]
    assert list(tmp_path.glob(".q1-start-v7-ownership-probe-*")) == []


def test_probe_timeout_marks_container_cleanup_unproved_when_ps_is_not_empty(
    tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    calls: list[tuple[str, ...]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(tuple(command))
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 30)
        if command[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(command, 2, b"", b"daemon error")
        assert command[1] == "ps"
        container_name = command[command.index("--filter") + 1].removeprefix("name=^/").removesuffix("$")
        return subprocess.CompletedProcess(
            command,
            0,
            (container_name + "\n").encode("ascii"),
            b"",
        )

    with pytest.raises(supervisor.Q1StartSupervisorError) as captured:
        supervisor._ownership_freeze_probe_v7(spec, runner=runner)
    error = captured.value
    assert error.code == (
        "REJECT_Q1_START_OUTPUT_OWNERSHIP_PROBE_CONTAINER_CLEANUP_UNPROVEN"
    )
    assert "rm_exit=2" in error.detail
    assert "ps_stdout=b'hegel-q1-v7-owner-probe-" in error.detail
    assert error.startup_smoke_audit == {
        "ownership_probe_run_popen_count": 1,
        "ownership_probe_cleanup_popen_count": 1,
        "ownership_probe_absence_check_popen_count": 1,
        "ownership_probe_container_absence_proven": False,
        "ownership_probe_temp_root_absence_proven": True,
        "docker_invoked": True,
    }
    failure = supervisor._startup_smoke_failure(
        error.code,
        error.detail,
        error.startup_smoke_audit,
    )
    assert failure["ownership_probe_container_absence_proven"] is False
    assert failure["ownership_probe_temp_root_absence_proven"] is True
    assert failure["docker_invoked"] is True
    assert [command[1] for command in calls] == ["run", "rm", "ps"]
    assert list(tmp_path.glob(".q1-start-v7-ownership-probe-*")) == []


def test_v7_host_owned_actor_tree_freezes_with_host_owner(
    request: pytest.FixtureRequest,
) -> None:
    root = _native_test_root(request, "freeze-tree") / "actor-output"
    chunk = root / "signature-02/stream-04/chunk-000000.framed"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"actor-output\n")
    expected_owner = supervisor._host_effective_ids()
    assert (chunk.lstat().st_uid, chunk.lstat().st_gid) == expected_owner

    supervisor._freeze_tree(root)

    assert chunk.stat().st_mode & 0o777 == 0o444
    assert root.stat().st_mode & 0o777 == 0o555
    assert chunk.parent.stat().st_mode & 0o777 == 0o555


def test_v7_cross_uid_tree_is_rejected_before_any_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "foreign-output"
    chunk = root / "signature-02/stream-04/chunk-000000.framed"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"foreign-output\n")
    uid, gid = supervisor._host_effective_ids()
    monkeypatch.setattr(supervisor, "_host_effective_ids", lambda: (uid + 1, gid + 1))
    chmod_calls: list[tuple[Path, int]] = []

    def unexpected_chmod(path: Path, mode: int, *args: object, **kwargs: object) -> None:
        chmod_calls.append((path, mode))
        pytest.fail("chmod must not run after owner mismatch")

    monkeypatch.setattr(Path, "chmod", unexpected_chmod)
    with pytest.raises(
        supervisor.Q1StartSupervisorError,
        match="Q1_START_OUTPUT_OWNER_MISMATCH",
    ):
        supervisor._freeze_tree(root)
    assert chmod_calls == []


def test_startup_smoke_rejects_container_user_drift_before_replay_or_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v7(_run_spec_payload(tmp_path))
    monkeypatch.setattr(supervisor, "_host_effective_user", lambda: "999:999")
    monkeypatch.setattr(
        supervisor,
        "_replay_successor_authority",
        lambda _spec: pytest.fail("authority replay reached after owner drift"),
    )
    monkeypatch.setattr(
        supervisor,
        "_ownership_freeze_probe_v7",
        lambda _spec: pytest.fail("ownership probe reached after owner drift"),
    )
    with pytest.raises(
        supervisor.Q1StartSupervisorError,
        match="REJECT_Q1_START_CONTAINER_USER_BINDING",
    ):
        supervisor.authority_startup_smoke_v7(spec)


def test_startup_smoke_cli_runs_in_clean_isolated_python_without_actor_or_mutation(
    request: pytest.FixtureRequest,
) -> None:
    script, run_spec, run_root = _isolated_smoke_fixture(request)
    before = sorted(str(path.relative_to(run_root)) for path in run_root.rglob("*"))
    process = subprocess.run(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            str(script),
            "--startup-smoke",
            "--run-spec",
            str(run_spec),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8", errors="replace")
    result = json.loads(process.stdout)
    assert process.stdout == _canonical(result)
    assert process.stderr == b""
    assert result["schema_version"] == supervisor.AUTHORITY_SMOKE_SCHEMA
    assert result["status"] == supervisor.AUTHORITY_SMOKE_STATUS
    assert result["campaign_supervisor_popen_count"] == 0
    assert result["actor_popen_count"] == 0
    assert result["docker_invoked"] is True
    assert result["ownership_probe_popen_count"] == 1
    assert result["container_user"] == supervisor._host_effective_user()
    assert result["host_effective_user"] == supervisor._host_effective_user()
    assert result["container_user_matches_host_effective_user"] is True
    assert result["ownership_freeze_probe"]["status"] == supervisor.OWNERSHIP_FREEZE_PROBE_STATUS
    assert result["ownership_freeze_probe"]["host_freeze_succeeded"] is True
    assert result["ownership_freeze_probe"]["probe_root_removed"] is True
    assert result["checkpoint_or_mutable_roots_created"] is False
    assert result["failed_start_v3_terminal_record_digest"] == supervisor.FAILED_V3_TERMINAL_RECORD_DIGEST
    assert result["failed_start_v4_terminal_record_digest"] == supervisor.FAILED_V4_TERMINAL_RECORD_DIGEST
    assert result["failed_start_v5_terminal_record_digest"] == supervisor.FAILED_V5_TERMINAL_RECORD_DIGEST
    assert result["failed_start_v6_terminal_record_digest"] == supervisor.FAILED_V6_TERMINAL_RECORD_DIGEST
    after = sorted(str(path.relative_to(run_root)) for path in run_root.rglob("*"))
    assert after == before
    assert not (run_root / "checkpoints").exists()


def test_startup_smoke_cli_failure_is_nonzero_complete_canonical_diagnostic(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-run-spec.json"
    process = subprocess.run(
        [
            "/usr/bin/python3.10",
            "-I",
            "-S",
            "-B",
            str(ROOT / "tools" / "phase3_q1_start_supervisor_v7.py"),
            "--startup-smoke",
            "--run-spec",
            str(missing),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    result = json.loads(process.stdout)
    assert process.returncode == 1
    assert process.stdout == _canonical(result)
    assert process.stderr == b""
    assert result == {
        "schema_version": supervisor.AUTHORITY_SMOKE_SCHEMA,
        "status": "Q1_START_V7_AUTHORITY_STARTUP_SMOKE_FAILED",
        "error_code": "REJECT_Q1_START_RUN_SPEC_IO",
        "error_detail": f"[Errno 2] No such file or directory: '{missing}'",
        "campaign_supervisor_popen_count": 0,
        "actor_popen_count": 0,
        "checkpoint_or_mutable_roots_created": False,
        "ownership_probe_run_popen_count": 0,
        "ownership_probe_cleanup_popen_count": 0,
        "ownership_probe_absence_check_popen_count": 0,
        "ownership_probe_container_absence_proven": True,
        "ownership_probe_temp_root_absence_proven": True,
        "docker_invoked": False,
    }


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
    (package / "phase3_q1_prestart_successor_candidate_v7.py").write_text(
        "from .strict_cbor_v1 import TOKEN\n"
        "import json\n"
        "def replay_q1_prestart_successor_candidate_v7(payload):\n"
        "    value = json.loads(payload.decode('ascii'))\n"
        "    assert TOKEN == 'strict-cbor-loaded'\n"
        "    return value\n"
        "def successor_bundle_root_v7(value): return value['bundle_root']\n"
        "def actor_source_manifest_root_v7(actor, rows): return rows[0]['raw_sha256']\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "PACKAGE_ROOT", package)
    monkeypatch.delitem(sys.modules, "hegel_machine", raising=False)
    module = supervisor._load_successor_bundle_core_v7()
    bundle = _canonical({"bundle_root": "sha256:" + "a" * 64})
    replayed = module.replay_q1_prestart_successor_candidate_v7(bundle)
    assert module.successor_bundle_root_v7(replayed) == "sha256:" + "a" * 64
    assert "hegel_machine" not in sys.modules
