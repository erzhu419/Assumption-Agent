from __future__ import annotations

import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_start_supervisor_v1",
    ROOT / "tools" / "phase3_q1_start_supervisor_v1.py",
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
supervisor = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = supervisor
MODULE_SPEC.loader.exec_module(supervisor)


CAPSULE = "sha256:" + "1" * 64
PRESTART_CAPSULE = "sha256:" + "2" * 64
SOURCE = "3" * 40
GENESIS = "sha256:" + "4" * 64
PRESTART_RECEIPT = "sha256:" + "5" * 64
PYTHON_IMAGE = "python@sha256:" + "6" * 64
RUST_IMAGE = "rust@sha256:" + "7" * 64


@pytest.fixture
def native_tmp_path() -> Path:
    path = Path(tempfile.mkdtemp(prefix="hegel-q1-start-test-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _raw_sha256(payload: bytes) -> str:
    return "sha256:" + sha256(payload).hexdigest()


def _limits(cpuset: str, *, host: bool = False) -> dict[str, object]:
    if host:
        return {
            "cpuset_cpus": cpuset,
            "memory_bytes": supervisor.HOST_MEMORY_BYTES,
            "output_bytes": supervisor.HOST_OUTPUT_BYTES,
            "scratch_bytes": supervisor.HOST_SCRATCH_BYTES,
            "stdout_bytes": supervisor.HOST_OUTPUT_BYTES,
            "wall_time_seconds": supervisor.HOST_WALL_SECONDS,
        }
    return {
        "cpuset_cpus": cpuset,
        "memory_bytes": supervisor.ENDPOINT_MEMORY_BYTES,
        "output_bytes": supervisor.ENDPOINT_OUTPUT_BYTES,
        "scratch_bytes": supervisor.ENDPOINT_SCRATCH_BYTES,
        "stdout_bytes": supervisor.ENDPOINT_OUTPUT_BYTES,
        "wall_time_seconds": supervisor.ENDPOINT_WALL_SECONDS,
    }


def _spec_payload(tmp_path: Path) -> bytes:
    run = tmp_path / "run"
    run.mkdir()
    python_source = run / "python-source"
    rust_source = run / "rust-source"
    host_source = run / "host-source"
    source_manifests: dict[str, list[dict[str, object]]] = {}
    for stage, path in (
        ("python", python_source),
        ("rust", rust_source),
        ("host", host_source),
    ):
        path.mkdir()
        payload = f"{stage}-source\n".encode("ascii")
        (path / "actor.py").write_bytes(payload)
        source_manifests[stage] = [
            {
                "relative_path": "actor.py",
                "byte_count": len(payload),
                "raw_sha256": _raw_sha256(payload),
            }
        ]
    rust_binary = tmp_path / "rust-binary"
    rust_binary_payload = b"binary"
    rust_binary.write_bytes(rust_binary_payload)
    rust_binary.chmod(0o555)
    admission = tmp_path / "admission.json"
    admission_payload = _canonical(
            {
                "action_id": supervisor.MATERIALIZER_ACTION,
                "capsule_digest": CAPSULE,
                "diagnostic_only": False,
                "prestart_admission_receipt_root": PRESTART_RECEIPT,
                "projection_profile_root": "sha256:" + "8" * 64,
                "q1_gate_count": 14,
                "q1_gate_mask": 16_383,
                "q1_gate_total": 20,
                "q1_output_slots": [None] * 8,
                "q1_state": "NOT_RUN",
                "role_evaluation_performed": False,
                "run_genesis_root": GENESIS,
                "schema_version": "hegel-q1-formal-archive-materializer-admission/1",
                "source_commit": SOURCE,
                "split_accessed": False,
                "target_truth_accessed": False,
            }
        )
    admission.write_bytes(admission_payload)
    seccomp = tmp_path / "seccomp.json"
    seccomp_payload = _canonical({"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []})
    seccomp.write_bytes(seccomp_payload)
    prestart = tmp_path / "prestart.json"
    prestart.write_bytes(
        _canonical(
            {
                "capsule_digest": PRESTART_CAPSULE,
                "actor_implementation_roots": {
                    "python": "sha256:" + "a" * 64,
                    "rust": "sha256:" + "b" * 64,
                    "host": "sha256:" + "c" * 64,
                },
                "future_q1_start_requires_distinct_authorization": True,
                "prestart_admission_receipt_root": PRESTART_RECEIPT,
                "q1_execution_started": False,
                "q1_gate_count": 14,
                "q1_gate_mask": 16_383,
                "q1_gate_total": 20,
                "q1_output_slots": [None] * 8,
                "q1_state": "NOT_RUN",
                "run_genesis_root": GENESIS,
                "source_commit": SOURCE,
                "status": supervisor.PRESTART_STATUS,
            }
        )
    )
    return _canonical(
        {
            "schema_version": supervisor.RUN_SPEC_SCHEMA,
            "capsule_digest": CAPSULE,
            "prestart_capsule_digest": PRESTART_CAPSULE,
            "source_commit": SOURCE,
            "run_genesis_root": GENESIS,
            "prestart_admission_receipt_root": PRESTART_RECEIPT,
            "prestart_artifact": str(prestart),
            "python_image": PYTHON_IMAGE,
            "rust_image": RUST_IMAGE,
            "rust_binary_raw_sha256": _raw_sha256(rust_binary_payload),
            "seccomp_profile_raw_sha256": _raw_sha256(seccomp_payload),
            "python_admission_raw_sha256": _raw_sha256(admission_payload),
            "actor_implementation_roots": {
                "python": "sha256:" + "a" * 64,
                "rust": "sha256:" + "b" * 64,
                "host": "sha256:" + "c" * 64,
            },
            "python_source_root": str(python_source),
            "rust_source_root": str(rust_source),
            "host_source_root": str(host_source),
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
            "runtime_limits": {
                "python": _limits("24"),
                "rust": _limits("25"),
                "host": _limits("26", host=True),
            },
            "actor_source_manifests": source_manifests,
        }
    )


def _endpoint_result(stage: str, spec: supervisor.Q1StartRunSpecV1) -> dict[str, object]:
    return {
        "schema_version": supervisor.MATERIALIZER_SCHEMA,
        "actor_id": stage,
        "implementation_id": f"hegel-{stage}-q1-formal-archive-materializer-v1",
        "action_id": supervisor.MATERIALIZER_ACTION,
        "status": supervisor.MATERIALIZER_STATUS,
        **spec.binding(),
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_total": 20,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "diagnostic_only": False,
        "formal_archive_materialized": True,
        "actual_formal_peak_scratch_bytes": 0,
        "actual_formal_terminal_live_scratch_bytes": 0,
        "closure_bundle_root": "sha256:" + "c" * 64,
        "closure_semantics_version": "hegel-quotient-closure-v1.0.1",
        "dsl_version": "hegel-old-dsl-v1.6.0",
        "formal_archive_payload_bytes": 1,
        "formal_capacity_limits_by_signature": list(supervisor._FORMAL_LIMIT_ROWS),
        "formal_output_bytes_limit": supervisor.ENDPOINT_OUTPUT_BYTES,
        "formal_scratch_bytes_limit": supervisor.ENDPOINT_SCRATCH_BYTES,
        "full_v16_structural_limits_applied": True,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "partitions": [{}, {}],
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }


def _host_result(spec: supervisor.Q1StartRunSpecV1) -> dict[str, object]:
    return {
        "schema_version": supervisor.HOST_SCHEMA,
        "actor_id": "host",
        "implementation_id": "hegel-host-q1-formal-archive-replay-v1",
        "action_id": supervisor.HOST_ACTION,
        "status": supervisor.HOST_STATUS,
        **spec.binding(),
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_total": 20,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "formal_capacity_limits_by_signature": list(supervisor._FORMAL_LIMIT_ROWS),
        "formal_output_bytes_limit": supervisor.ENDPOINT_OUTPUT_BYTES,
        "formal_scratch_bytes_limit": supervisor.ENDPOINT_SCRATCH_BYTES,
        "semantic_binding_root": "sha256:" + "9" * 64,
        "projection_profile_root": "sha256:" + "a" * 64,
        "closure_bundle_root": "sha256:" + "b" * 64,
        "archive_payload_bytes_per_endpoint": 1,
        "formal_file_count_per_endpoint": 1,
        "signatures": [
            {
                "input_signature_id": signature_id,
                "universe_root": "sha256:" + f"{50 + signature_id:064x}",
                "fixed_point_record_root": "sha256:" + f"{60 + signature_id:064x}",
                "signature_manifest_root": "sha256:" + f"{70 + signature_id:064x}",
                "saturation_state_root": "sha256:" + f"{80 + signature_id:064x}",
                "record_count_by_stream": [1, 1, 1, 1],
            }
            for signature_id in (1, 2)
        ],
        "python_rust_formal_bytes_exact": True,
        "python_rust_replayed_roots_exact": True,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "target_split_role_absence_attested": True,
        "q1_completion_receipt_issued": False,
        "state_published": False,
    }


def _runtime(stage: str, spec: supervisor.Q1StartRunSpecV1) -> dict[str, object]:
    ordinal = supervisor.STAGES.index(stage) + 1
    return {
        "schema_version": supervisor.RUNTIME_EVIDENCE_SCHEMA,
        "stage": stage,
        "container_id": f"{ordinal:064x}",
        "image": spec.image_for(stage),
        "cpuset_cpus": spec.runtime_limits[stage].cpuset_cpus,
        "memory_limit_bytes": spec.runtime_limits[stage].memory_bytes,
        "memory_peak_bytes": 1_000_000 + ordinal,
        "network_mode": "none",
        "ipc_mode": "private",
        "read_only_rootfs": True,
        "mounts": supervisor._expected_mounts(spec, stage),
        "cgroup_path": f"/test/{stage}",
        "measurement_interval_start_monotonic_ns": ordinal * 10,
        "actor_return_monotonic_ns": ordinal * 10 + 1,
        "container_exit_monotonic_ns": ordinal * 10 + 2,
        "measurement_wall_time_ns": 2,
        "wall_time_limit_ns": spec.runtime_limits[stage].wall_time_seconds * 1_000_000_000,
        "maximum_concurrent_actor_containers": 1,
        "actor_terminal_scratch_payload_bytes": 0,
        "supervisor_control_peak_payload_bytes": 0,
        "command": list(supervisor.docker_commands_v1(spec)[stage]),
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
    }


def _replayer(stage: str, _output: Path, _spec: supervisor.Q1StartRunSpecV1) -> dict[str, object]:
    signature_rows = []
    for signature_id in (1, 2):
        limits = supervisor._FORMAL_GUARD_LIMITS[signature_id]
        signature_rows.append(
            {
                "input_signature_id": signature_id,
                "fixed_point_record_root": "sha256:" + f"{10 + signature_id:064x}",
                "same_run_record_set_root": "sha256:" + f"{20 + signature_id:064x}",
                "signature_manifest_root": "sha256:" + f"{30 + signature_id:064x}",
                "saturation_state_root": "sha256:" + f"{40 + signature_id:064x}",
                "guard_evidence_rows": [
                    {
                        "guard_id": guard_id,
                        "guard_name": supervisor._GUARD_NAMES[guard_id - 1],
                        "evidence_kind": (
                            "REPLAYED_FIXED_POINT_COUNT"
                            if guard_id <= 6
                            else "FROZEN_LIMIT_CONSERVATIVE_UPPER_BOUND"
                        ),
                        "observed_or_conservative_bound": (
                            min(1, limits[guard_id - 1])
                            if guard_id <= 6
                            else limits[guard_id - 1]
                        ),
                        "frozen_limit": limits[guard_id - 1],
                        "guard_hit": False,
                    }
                    for guard_id in range(1, 9)
                ],
            }
        )
    return {
        "actor_id": stage,
        "archive_payload_bytes": 1,
        "closure_bundle_root": "sha256:" + "c" * 64,
        "formal_relative_paths": ["formal.bin"],
        "formal_file_closure_root": "sha256:" + "d" * 64,
        "signature_rows": signature_rows,
    }


def _launcher(log: list[str]) -> Callable:
    def launch(
        stage: str,
        _command: list[str] | tuple[str, ...],
        spec: supervisor.Q1StartRunSpecV1,
    ) -> supervisor.ActorProcessResultV1:
        log.append(stage)
        value = _host_result(spec) if stage == "host" else _endpoint_result(stage, spec)
        payload = _canonical(value)
        if stage != "host":
            (spec.output_for(stage) / "endpoint-result.json").write_bytes(payload)
            (spec.output_for(stage) / "formal.bin").write_bytes(b"same-formal-bytes")
        return supervisor.ActorProcessResultV1(0, payload, b"", True, _runtime(stage, spec))

    return launch


def test_run_spec_is_strict_and_commands_bind_real_formal_actor_interfaces(native_tmp_path: Path) -> None:
    payload = _spec_payload(native_tmp_path)
    spec = supervisor.load_run_spec_v1(payload)
    commands = supervisor.docker_commands_v1(spec)
    assert tuple(commands) == supervisor.STAGES
    for stage, command in commands.items():
        assert command[0] == supervisor.DOCKER
        assert command[1:4] == ("run", "--rm", "--pull")
        assert command[command.index("--network") + 1] == "none"
        assert "--read-only" in command
        assert command[command.index("--memory") + 1] == str(spec.runtime_limits[stage].memory_bytes)
        assert command[command.index("--cpuset-cpus") + 1] == spec.runtime_limits[stage].cpuset_cpus
        writable = sorted(
            item.split(",dst=", 1)[1]
            for item in command
            if item.startswith("type=bind") and not item.endswith(",readonly")
        )
        assert writable == ["/output", "/scratch"]
    assert supervisor.MATERIALIZER_ACTION in commands["python"]
    assert "/admission/admission.json" in commands["python"]
    assert supervisor.RUST_CONTAINER_BINARY in commands["rust"]
    assert supervisor.HOST_ACTION in commands["host"]
    assert "/python-output" in commands["host"]
    assert "/rust-output" in commands["host"]

    with pytest.raises(supervisor.Q1StartSupervisorError):
        supervisor.load_run_spec_v1(payload[:-1] + b" \n")
    value = json.loads(payload)
    value["unexpected"] = True
    with pytest.raises(supervisor.Q1StartSupervisorError):
        supervisor.load_run_spec_v1(_canonical(value))


def test_supervisor_runs_python_rust_host_once_and_seals_clean_checkpoints(native_tmp_path: Path) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    calls: list[str] = []
    outcome = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(calls),
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    assert calls == ["python", "rust", "host"]
    assert outcome["status"] == supervisor.SUCCESS_STATUS
    assert outcome["completed_stages"] == ["python", "rust", "host"]
    assert outcome["gate18_runtime_receipt_inputs"]["actor_order"] == calls
    assert outcome["gate18_runtime_receipt_inputs"]["maximum_concurrent_actor_containers"] == 1
    for stage in supervisor.STAGES:
        intent, checkpoint = supervisor._stage_paths(spec, stage)
        assert intent.is_file()
        assert checkpoint.is_file()
        assert not spec.scratch_for(stage).exists()
        assert (spec.output_for(stage).stat().st_mode & 0o777) == 0o555
        checkpoint_value = json.loads(checkpoint.read_bytes())
        assert checkpoint_value["actor_popen_count"] == 1
        assert checkpoint_value["scratch_absent_after_cleanup"] is True


def test_pending_after_python_resumes_at_rust_without_remeasuring_python(native_tmp_path: Path) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    first_calls: list[str] = []

    def first_preflight(stage: str, _spec: supervisor.Q1StartRunSpecV1) -> None:
        if stage == "rust":
            raise supervisor.PendingCondition("TEST_NONCORE_PENDING", "Rust path temporarily unavailable")

    pending = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(first_calls),
        preflight=first_preflight,
        endpoint_replayer=_replayer,
    )
    assert pending["status"] == supervisor.PENDING_STATUS
    assert pending["next_stage"] == "rust"
    assert pending["next_actor_intent_absent"] is True
    assert pending["actor_popen_count_this_entry"] == 1
    assert first_calls == ["python"]
    assert not supervisor._stage_paths(spec, "rust")[0].exists()

    resumed_calls: list[str] = []
    complete = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(resumed_calls),
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    assert complete["status"] == supervisor.SUCCESS_STATUS
    assert resumed_calls == ["rust", "host"]

    replay_calls: list[str] = []
    replayed = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(replay_calls),
        preflight=lambda _stage, _spec: pytest.fail("completed replay must not preflight"),
        endpoint_replayer=_replayer,
    )
    assert replayed == complete
    assert replay_calls == []


def test_stage_root_oserror_is_clean_pending_before_intent_and_popen(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    calls: list[str] = []
    original_chmod = Path.chmod

    def fail_python_output_chmod(path: Path, mode: int, *args: object, **kwargs: object) -> None:
        if path == spec.python_output:
            raise OSError("synthetic chmod failure")
        original_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", fail_python_output_chmod)
    pending = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(calls),
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    python_intent, python_checkpoint = supervisor._stage_paths(spec, "python")
    assert pending["status"] == supervisor.PENDING_STATUS
    assert pending["completed_stages"] == []
    assert pending["next_stage"] == "python"
    assert pending["pending_code"] == "Q1_START_STAGE_ROOT_PREPARATION_PENDING"
    assert pending["actor_popen_count_this_entry"] == 0
    assert calls == []
    assert not python_intent.exists()
    assert not python_checkpoint.exists()
    assert not spec.python_output.exists()
    assert not spec.python_scratch.exists()


def test_intent_open_failure_is_clean_pending_before_popen(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    calls: list[str] = []
    python_intent, python_checkpoint = supervisor._stage_paths(spec, "python")
    original_publish = supervisor._publish_exclusive

    def fail_before_intent_create(path: Path, value: dict[str, object]) -> None:
        if path == python_intent:
            raise OSError("synthetic intent open failure")
        original_publish(path, value)

    monkeypatch.setattr(supervisor, "_publish_exclusive", fail_before_intent_create)
    pending = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(calls),
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    assert pending["status"] == supervisor.PENDING_STATUS
    assert pending["completed_stages"] == []
    assert pending["next_stage"] == "python"
    assert pending["pending_code"] == "Q1_START_INTENT_PUBLICATION_PENDING"
    assert pending["actor_popen_count_this_entry"] == 0
    assert calls == []
    assert not python_intent.exists()
    assert not python_checkpoint.exists()
    assert not spec.python_output.exists()
    assert not spec.python_scratch.exists()


def test_unknown_or_partial_post_intent_is_terminal_and_never_relaunched(native_tmp_path: Path) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    calls: list[str] = []

    def unknown(
        stage: str,
        _command: list[str] | tuple[str, ...],
        actual: supervisor.Q1StartRunSpecV1,
    ) -> supervisor.ActorProcessResultV1:
        calls.append(stage)
        (actual.output_for(stage) / "partial.bin").write_bytes(b"partial")
        return supervisor.ActorProcessResultV1(None, b"", b"", False, {})

    failed = supervisor.run_supervisor_v1(
        spec,
        launcher=unknown,
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    assert failed["status"] == supervisor.TERMINAL_STATUS
    assert failed["error_code"] == "Q1_START_ACTOR_OUTCOME_UNKNOWN"
    assert failed["resume_allowed"] is False
    assert calls == ["python"]
    assert supervisor._stage_paths(spec, "python")[0].is_file()
    assert not supervisor._stage_paths(spec, "python")[1].exists()
    assert (spec.python_output / "partial.bin").is_file()

    def must_not_launch(*_args: object, **_kwargs: object) -> supervisor.ActorProcessResultV1:
        pytest.fail("terminal replay must not launch an actor")

    replayed = supervisor.run_supervisor_v1(
        spec,
        launcher=must_not_launch,
        preflight=lambda _stage, _spec: pytest.fail("terminal replay must not preflight"),
        endpoint_replayer=_replayer,
    )
    assert replayed == failed


@pytest.mark.parametrize("mutation", ("replace", "extra"))
def test_source_manifest_drift_is_pending_before_intent_and_popen(
    native_tmp_path: Path,
    mutation: str,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    if mutation == "replace":
        (spec.python_source_root / "actor.py").write_bytes(b"changed-source\n")
    else:
        (spec.python_source_root / "extra.py").write_bytes(b"extra\n")
    calls: list[str] = []
    outcome = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(calls),
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    assert outcome["status"] == supervisor.PENDING_STATUS
    assert outcome["pending_code"] == "Q1_START_SOURCE_MANIFEST_PENDING"
    assert outcome["actor_popen_count_this_entry"] == 0
    assert calls == []
    assert not supervisor._stage_paths(spec, "python")[0].exists()
    assert not spec.python_output.exists()
    assert not spec.python_scratch.exists()


@pytest.mark.parametrize("bound_file", ("python_admission_json", "seccomp_profile"))
def test_python_runtime_binding_replacement_is_pending_before_intent(
    native_tmp_path: Path,
    bound_file: str,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    getattr(spec, bound_file).write_bytes(b"replacement\n")
    calls: list[str] = []
    outcome = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(calls),
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    assert outcome["status"] == supervisor.PENDING_STATUS
    assert outcome["pending_code"] == "Q1_START_BOUND_FILE_PENDING"
    assert calls == []
    assert not supervisor._stage_paths(spec, "python")[0].exists()


def test_rust_binary_replacement_pends_after_python_with_truthful_launch_count(
    native_tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    spec.rust_binary.chmod(0o755)
    spec.rust_binary.write_bytes(b"replacement-rust-binary")
    calls: list[str] = []
    outcome = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(calls),
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    assert outcome["status"] == supervisor.PENDING_STATUS
    assert outcome["next_stage"] == "rust"
    assert outcome["actor_popen_count_this_entry"] == 1
    assert calls == ["python"]
    assert not supervisor._stage_paths(spec, "rust")[0].exists()


def test_completed_checkpoint_rehashes_source_without_relaunch(
    native_tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    first_calls: list[str] = []

    def stop_before_rust(stage: str, _spec: supervisor.Q1StartRunSpecV1) -> None:
        if stage == "rust":
            raise supervisor.PendingCondition("TEST_STOP", "stop")

    first = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(first_calls),
        preflight=stop_before_rust,
        endpoint_replayer=_replayer,
    )
    assert first["status"] == supervisor.PENDING_STATUS
    assert first_calls == ["python"]
    (spec.python_source_root / "actor.py").chmod(0o644)
    (spec.python_source_root / "actor.py").write_bytes(b"changed-after-checkpoint\n")
    replay_calls: list[str] = []
    replay = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher(replay_calls),
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    assert replay["status"] == supervisor.PENDING_STATUS
    assert replay["pending_code"] == "Q1_START_SOURCE_MANIFEST_PENDING"
    assert replay["completed_stages"] == ["python"]
    assert replay["next_stage"] == "rust"
    assert replay["next_actor_intent_absent"] is True
    assert supervisor._stage_paths(spec, "python")[0].is_file()
    assert not supervisor._stage_paths(spec, "rust")[0].exists()
    assert replay_calls == []


def test_host_checkpoint_binding_loss_is_terminal_with_truthful_prefix(
    native_tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    complete = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher([]),
        preflight=lambda _stage, _spec: None,
        endpoint_replayer=_replayer,
    )
    assert complete["status"] == supervisor.SUCCESS_STATUS
    (spec.host_source_root / "actor.py").chmod(0o644)
    (spec.host_source_root / "actor.py").write_bytes(b"changed-after-host\n")

    replay = supervisor.run_supervisor_v1(
        spec,
        launcher=lambda *_args, **_kwargs: pytest.fail("must not relaunch"),
        preflight=lambda _stage, _spec: pytest.fail("must not preflight"),
        endpoint_replayer=_replayer,
    )
    assert replay["status"] == supervisor.TERMINAL_STATUS
    assert replay["error_code"] == (
        "Q1_START_POST_HOST_CHECKPOINT_BINDING_UNAVAILABLE"
    )
    assert replay["completed_stages"] == ["python", "rust", "host"]
    assert replay["resume_allowed"] is False


def test_missing_prestart_after_python_is_clean_pending_for_rust(
    native_tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))

    def stop_before_rust(stage: str, _spec: supervisor.Q1StartRunSpecV1) -> None:
        if stage == "rust":
            raise supervisor.PendingCondition("TEST_STOP", "stop")

    first = supervisor.run_supervisor_v1(
        spec,
        launcher=_launcher([]),
        preflight=stop_before_rust,
        endpoint_replayer=_replayer,
    )
    assert first["completed_stages"] == ["python"]
    hidden = spec.prestart_artifact.with_suffix(".hidden")
    spec.prestart_artifact.rename(hidden)
    try:
        pending = supervisor.run_supervisor_v1(
            spec,
            launcher=lambda *_args, **_kwargs: pytest.fail("must not relaunch"),
            preflight=lambda _stage, _spec: pytest.fail("must not preflight"),
            endpoint_replayer=_replayer,
        )
    finally:
        hidden.rename(spec.prestart_artifact)
    assert pending["status"] == supervisor.PENDING_STATUS
    assert pending["pending_code"] == "Q1_START_PREDECESSOR_PENDING"
    assert pending["completed_stages"] == ["python"]
    assert pending["next_stage"] == "rust"
    assert pending["next_actor_intent_absent"] is True


def test_second_binding_replay_cleans_preintent_roots_before_pending(
    native_tmp_path: Path,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))

    def mutate_after_first_replay(stage: str, actual: supervisor.Q1StartRunSpecV1) -> None:
        assert stage == "python"
        (actual.python_source_root / "actor.py").write_bytes(b"changed-in-preflight\n")

    outcome = supervisor.run_supervisor_v1(
        spec,
        launcher=lambda *_args: pytest.fail("binding drift must precede Popen"),
        preflight=mutate_after_first_replay,
        endpoint_replayer=_replayer,
    )
    assert outcome["status"] == supervisor.PENDING_STATUS
    assert outcome["pending_code"] == "Q1_START_SOURCE_MANIFEST_PENDING"
    assert outcome["actor_popen_count_this_entry"] == 0
    assert not supervisor._stage_paths(spec, "python")[0].exists()
    assert not spec.python_output.exists()
    assert not spec.python_scratch.exists()


def test_docker_inspect_replays_exact_runtime_contract_without_actor_result(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    stage = "python"
    command = supervisor.docker_commands_v1(spec)[stage]
    seccomp = json.loads(spec.seccomp_profile.read_bytes())
    row = {
        "State": {"Pid": 1234},
        "HostConfig": {
            "NetworkMode": "none",
            "IpcMode": "private",
            "ReadonlyRootfs": True,
            "CpusetCpus": spec.runtime_limits[stage].cpuset_cpus,
            "Memory": supervisor.ENDPOINT_MEMORY_BYTES,
            "MemorySwap": supervisor.ENDPOINT_MEMORY_BYTES,
            "PidsLimit": supervisor.PIDS_LIMIT,
            "CapDrop": ["ALL"],
            "Ulimits": [{"Name": "nofile", "Hard": 256, "Soft": 256}],
            "Tmpfs": {"/tmp": supervisor.TMPFS_SPEC.split(":", 1)[1]},
            "AutoRemove": True,
            "SecurityOpt": [
                "no-new-privileges",
                "seccomp=" + json.dumps(seccomp, separators=(",", ":"), sort_keys=True),
            ],
        },
        "Config": {
            "Image": spec.python_image,
            "User": supervisor.CONTAINER_USER,
            "WorkingDir": "/source",
            "Cmd": list(command)[list(command).index(spec.python_image) + 1 :],
            "Env": [
                "HOME=/tmp",
                "LANG=C.UTF-8",
                "LC_ALL=C.UTF-8",
                "PYTHONDONTWRITEBYTECODE=1",
                "TZ=UTC",
            ],
        },
        "Mounts": [
            {
                "Type": "bind",
                "Destination": item["destination"],
                "RW": item["read_write"],
                "Source": item["source"],
            }
            for item in supervisor._expected_mounts(spec, stage)
        ],
    }
    monkeypatch.setattr(
        supervisor.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps([row]).encode("ascii"),
            stderr=b"",
        ),
    )
    original_read_text = Path.read_text

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/proc/1234/cgroup":
            return "0::/test-q1-start.scope\n"
        if str(path).endswith("/test-q1-start.scope/memory.peak"):
            return "123456\n"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    evidence, peak = supervisor._docker_inspect(
        "d" * 64,
        spec,
        stage,
        command,
        time.monotonic() + 10,
    )
    assert peak == 123456
    assert evidence["mounts"] == supervisor._expected_mounts(spec, stage)


def test_host_boundary_rejects_malformed_signature_replay(native_tmp_path: Path) -> None:
    spec = supervisor.load_run_spec_v1(_spec_payload(native_tmp_path))
    result = _host_result(spec)
    result["signatures"][0]["record_count_by_stream"] = [1, 1, 1]
    with pytest.raises(supervisor.TerminalCondition):
        supervisor._validate_host_result_object(result, spec)


def test_zero_byte_filename_receipt_respects_host_one_byte_scratch_bound(
    native_tmp_path: Path,
) -> None:
    scratch = native_tmp_path / "scratch"
    scratch.mkdir()
    receipt = scratch / (
        supervisor.ACTOR_RETURN_PREFIX + "0" + supervisor.ACTOR_RETURN_SUFFIX
    )
    receipt.write_bytes(b"")

    class Running:
        @staticmethod
        def poll() -> None:
            return None

    code, observed = supervisor._wait_for_actor_receipt(
        Running(),  # type: ignore[arg-type]
        scratch,
        10**12,
    )
    assert (code, observed, observed.stat().st_size) == (0, receipt, 0)
    assert supervisor.HOST_SCRATCH_BYTES == 1
    assert "printf" not in supervisor._HOLD_SCRIPT


def _snapshot_package(native_tmp_path: Path, marker: str) -> Path:
    host_source = native_tmp_path / "host-source"
    package = host_source / "src" / "hegel_machine"
    shutil.copytree(
        ROOT / "src" / "hegel_machine",
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    replay_module = package / "phase3_q1_formal_host_replay_v1.py"
    with replay_module.open("ab") as output:
        output.write(f"\nSNAPSHOT_TEST_MARKER = {marker!r}\n".encode("ascii"))
    return host_source


def test_isolated_python_can_bootstrap_snapshot_formal_replay_core_without_site_packages(
    native_tmp_path: Path,
) -> None:
    tool = ROOT / "tools" / "phase3_q1_start_supervisor_v1.py"
    host_source = _snapshot_package(native_tmp_path, "isolated-snapshot")
    script = f"""
import importlib.util, sys, types
from pathlib import Path
path = Path({str(tool)!r})
module_spec = importlib.util.spec_from_file_location('isolated_q1_start', path)
module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = module
module_spec.loader.exec_module(module)
run_spec = types.SimpleNamespace(
    host_source_root=Path({str(host_source)!r}),
    capsule_digest={CAPSULE!r},
)
core = module._load_formal_replay_core_v1(run_spec)
print(core.__file__)
print(core.SNAPSHOT_TEST_MARKER)
"""
    completed = subprocess.run(
        ["/usr/bin/python3.10", "-I", "-S", "-B", "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    lines = completed.stdout.decode("utf-8").splitlines()
    assert Path(lines[0]).is_relative_to(host_source)
    assert lines[1] == "isolated-snapshot"


def test_snapshot_loader_ignores_already_loaded_current_hegel_package(
    native_tmp_path: Path,
) -> None:
    import hegel_machine  # noqa: F401

    host_source = _snapshot_package(native_tmp_path, "bound-snapshot")
    run_spec = SimpleNamespace(
        host_source_root=host_source,
        capsule_digest=CAPSULE,
    )
    core = supervisor._load_formal_replay_core_v1(run_spec)
    assert Path(core.__file__).is_relative_to(host_source)
    assert core.SNAPSHOT_TEST_MARKER == "bound-snapshot"
    assert core.__name__.startswith("_hegel_q1_snapshot_")


def test_cli_top_level_preintent_pending_preserves_full_campaign_binding(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    run_spec_path = native_tmp_path / "run-spec.json"
    run_spec_path.write_bytes(_spec_payload(native_tmp_path))

    def top_level_pending(_spec: object) -> dict[str, object]:
        raise supervisor.PendingCondition(
            "Q1_START_CHECKPOINT_ROOT_PENDING",
            "synthetic local checkpoint-root failure",
        )

    monkeypatch.setattr(supervisor, "run_supervisor_v1", top_level_pending)
    assert supervisor.main(["--run-spec", str(run_spec_path)]) == 75
    outcome = json.loads(capfd.readouterr().out)
    assert outcome == {
        "schema_version": supervisor.OUTCOME_SCHEMA,
        "status": supervisor.PENDING_STATUS,
        "capsule_digest": CAPSULE,
        "source_commit": SOURCE,
        "run_genesis_root": GENESIS,
        "prestart_admission_receipt_root": PRESTART_RECEIPT,
        "next_stage": "python",
        "completed_stages": [],
        "pending_code": "Q1_START_CHECKPOINT_ROOT_PENDING",
        "pending_detail": "synthetic local checkpoint-root failure",
        "next_actor_intent_absent": True,
        "actor_popen_count_this_entry": 0,
        "resume_allowed": True,
    }
