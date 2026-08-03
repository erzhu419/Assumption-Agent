from __future__ import annotations

from dataclasses import fields, replace
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace

import pytest

import hegel_machine.phase3_m25_formal_container_executor_v1 as executor
from hegel_machine.phase3_m25_container_ceremony_v1 import (
    GateEvidenceInputsV1,
    SPLIT_RESPONSE_ROWS,
    SplitCalculatorPublicResponseV2,
    SplitRootCommitment,
    encode_split_calculator_public_frame_v2,
)
from hegel_machine.phase3_m25_external_v1 import MarkerSnapshot


def _basis(*, ready: bool):
    return SimpleNamespace(
        basis_commit="12" * 20,
        implementation_inputs={
            "m3_execution_implementation_bindings_ready": ready,
            "m3_execution_implementation_binding_roots": (
                {
                    "python_implementation_binding_root": b"p" * 32,
                    "rust_implementation_binding_root": b"r" * 32,
                }
                if ready
                else None
            ),
        },
        blocking_gaps=() if ready else ("M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY",),
    )


def _install_fake_docker_boundary(
    backend: executor.DockerCeremonyActorsV1, root: Path
) -> None:
    if backend._root is None:
        backend._root = root / "fake-runtime"
        backend._root.mkdir(parents=True, exist_ok=True)
    config = root / "fake-docker-config"
    config.mkdir(parents=True, exist_ok=True)
    backend._docker_control_plane = SimpleNamespace(
        command=lambda *arguments: ["docker", *arguments],
        environment=MappingProxyType({"PATH": "/usr/bin:/bin"}),
        binding=MappingProxyType({"test_only_local_control_plane": True}),
    )
    backend._docker_daemon_binding = b"d" * 32
    backend._docker_root_directory = root / "docker-root"
    backend._runtime_seccomp_path = root / "runtime-seccomp.json"
    backend._build_seccomp_path = root / "build-seccomp.json"
    backend._runtime_seccomp_path.write_text("{}\n", encoding="ascii")
    backend._build_seccomp_path.write_text("{}\n", encoding="ascii")


def _fake_volume_row(
    backend: executor.DockerCeremonyActorsV1,
    purpose: int,
    name: str,
) -> dict[str, object]:
    assert backend._docker_root_directory is not None
    return {
        "Name": name,
        "Driver": "local",
        "Scope": "local",
        "Options": None,
        "Labels": backend._state_volume_labels(purpose),
        "Mountpoint": str(backend._docker_root_directory / "volumes" / name / "_data"),
    }


def _configure_fake_actor_identity(
    backend: executor.DockerCeremonyActorsV1, root: Path
) -> None:
    if not backend._profile:
        backend._profile = {
            "images": {
                "custodian": "python@sha256:" + "1" * 64,
                "python_attester": "python@sha256:" + "1" * 64,
                "rust_attester": "rust@sha256:" + "2" * 64,
                "policy_auditor": "python@sha256:" + "1" * 64,
            }
        }
    if backend._profile_digest is None:
        backend._profile_digest = b"p" * 32
    if backend._transaction_run_id is None:
        backend._transaction_run_id = b"r" * 16
    _install_fake_docker_boundary(backend, root)


def test_readiness_rejects_static_replayers_as_execution_bindings(monkeypatch) -> None:
    monkeypatch.setattr(
        executor,
        "build_qualified_formal_static_basis_v1",
        lambda _commit: _basis(ready=False),
    )
    report = executor.inspect_formal_ceremony_readiness_v1("12" * 20)
    assert report.ready is False
    assert report.formal_gates_after == 14
    assert report.child_state == "NOT_RUN"
    assert report.qualification_side_effects_performed is True
    assert report.ceremony_actor_key_seed_marker_side_effects_performed is False
    assert "M3_EXECUTION_IMPLEMENTATION_BINDINGS_NOT_READY" in report.blockers
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.require_formal_ceremony_ready_v1(_basis(ready=False))
    assert captured.value.code == executor.FAIL_EXECUTION_BINDINGS


def test_readiness_requires_post_commit_rust_bridge_qualification(monkeypatch) -> None:
    monkeypatch.setattr(executor, "UNRESOLVED_AUTHORITATIVE_BLOCKERS", ())
    monkeypatch.setattr(executor, "_PRESTAGE_RECOVERY_IMPLEMENTED", True)
    monkeypatch.setattr(
        executor,
        "load_actor_protocol_archive_qualification_v1",
        lambda _commit: executor.ArchivedActorProtocolQualificationBindingV1(
            "12" * 20,
            b"v" * 32,
            MappingProxyType({
                purpose: bytes([purpose]) * 16
                for purpose in (1, 2, 3, 4)
            }),
            MappingProxyType({"synthetic_test_bundle": True}),
        ),
    )
    monkeypatch.setattr(
        executor,
        "build_qualified_formal_static_basis_v1",
        lambda _commit: _basis(ready=True),
    )
    monkeypatch.setattr(
        executor,
        "load_qualified_rust_bridge_dag_binary_binding_v1",
        lambda **_kwargs: (_ for _ in ()).throw(
            executor.BridgeDagBinaryQualificationError("TEST", "absent")
        ),
    )
    blocked = executor.inspect_formal_ceremony_readiness_v1("12" * 20)
    assert blocked.ready is False
    assert executor.FAIL_BRIDGE_REPLAY_UNRESOLVED in blocked.blockers

    monkeypatch.setattr(
        executor,
        "load_qualified_rust_bridge_dag_binary_binding_v1",
        lambda **_kwargs: ({}, "sha256:" + "11" * 32),
    )
    ready = executor.inspect_formal_ceremony_readiness_v1("12" * 20)
    assert ready.ready is True
    assert ready.blockers == ()


def test_authoritative_backend_no_longer_advertises_post_stage_recovery_blocker(
    tmp_path: Path,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    assert executor.FAIL_POST_STAGE_RECOVERY_UNRESOLVED not in (
        backend.unresolved_formal_blockers()
    )
    assert executor.FAIL_BRIDGE_REPLAY_UNRESOLVED in (
        backend.unresolved_formal_blockers()
    )


def test_formal_actor_launch_probes_actual_clone_path_without_reporting_raw_value(
    tmp_path: Path,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    _configure_fake_actor_identity(backend, tmp_path)
    safe = backend._actor_environment(1)
    launch = backend._actor_launch_environment(1)
    actual = executor.REPOSITORY_ROOT.resolve().as_posix()

    assert "HEGEL_HOST_REPOSITORY_PATH" not in safe
    assert launch["HEGEL_HOST_REPOSITORY_PATH"] == actual
    assert launch["HEGEL_HOST_REPOSITORY_PATH_SHA256"] == hashlib.sha256(
        actual.encode("utf-8")
    ).hexdigest()
    assert set(launch) - set(safe) == executor._ACTOR_PRIVATE_ENVIRONMENT_KEYS

    nonce = b"n" * 16
    request_digest = b"d" * 32
    safe_operation = backend._actor_environment(
        1,
        operation="qualify-only",
        operation_sequence=1,
        operation_nonce=nonce,
        operation_request_digest=request_digest,
    )
    launch_operation = backend._actor_launch_environment(
        1,
        operation="qualify-only",
        operation_sequence=1,
        operation_nonce=nonce,
        operation_request_digest=request_digest,
    )
    assert "HEGEL_HOST_REPOSITORY_PATH" not in safe_operation
    assert launch_operation["HEGEL_HOST_REPOSITORY_PATH"] == actual
    assert set(launch_operation) - set(safe_operation) == (
        executor._ACTOR_PRIVATE_ENVIRONMENT_KEYS
    )


def _load_operation_probe_module():
    project = Path(__file__).resolve().parents[1]
    path = project / "tools/phase3_m25_actor_operation_probe_v1.py"
    spec = importlib.util.spec_from_file_location("hegel_m25_operation_probe_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_operation_probe_removes_raw_path_from_actual_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    _configure_fake_actor_identity(backend, tmp_path)
    launch = backend._actor_launch_environment(
        1,
        operation="qualify-only",
        operation_sequence=1,
        operation_nonce=b"n" * 16,
        operation_request_digest=b"d" * 32,
    )
    operation_probe = _load_operation_probe_module()
    actual_environment = dict(launch)
    monkeypatch.setattr(operation_probe.os, "environ", actual_environment)

    purpose, reported, base, raw_path = operation_probe._validate_environment(
        "qualify-only"
    )

    assert purpose == 1
    assert raw_path == executor.REPOSITORY_ROOT.resolve().as_posix()
    assert "HEGEL_HOST_REPOSITORY_PATH" not in actual_environment
    assert "HEGEL_HOST_REPOSITORY_PATH" not in reported
    assert set(base) == operation_probe.BASE_ENV_KEYS


def test_rust_bridge_qualification_binding_removes_only_dynamic_blocker(
    tmp_path: Path, monkeypatch,
) -> None:
    binary = tmp_path / "hegel-m25-bridge-dag-replay"
    binary.write_bytes(b"qualified-rust-bridge")
    binary.chmod(0o755)
    report = tmp_path / "qualification.json"
    report.write_text("{}\n", encoding="ascii")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    report_id = "ab" * 32
    monkeypatch.setattr(executor, "DEFAULT_RUST_BRIDGE_DAG_BINARY", binary)
    monkeypatch.setattr(
        executor,
        "load_qualified_rust_bridge_dag_binary_binding_v1",
        lambda **kwargs: (
            {"diagnostic_report_sha256": "sha256:" + report_id},
            "sha256:" + digest,
        ),
    )
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        rust_bridge_dag_replay_binary=binary,
        rust_bridge_dag_qualification_report=report,
        timestamp=1,
    )

    assert backend.validate_rust_bridge_dag_binding() == bytes.fromhex(digest)
    assert backend._bound_rust_bridge_dag_report_sha256 == bytes.fromhex(report_id)
    assert executor.FAIL_BRIDGE_REPLAY_UNRESOLVED not in (
        backend.unresolved_formal_blockers()
    )


def test_backend_protocol_blockers_do_not_require_its_not_yet_generated_self_report(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._bound_rust_bridge_dag_digest = b"b" * 32
    backend._bound_rust_bridge_dag_report_sha256 = b"q" * 32
    monkeypatch.setattr(
        executor,
        "load_actor_protocol_archive_qualification_v1",
        lambda _commit: (_ for _ in ()).throw(AssertionError("self-report consulted")),
    )
    assert backend.unresolved_formal_blockers() == ()


def test_static_replay_runtime_preparation_allows_exactly_one_actor_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    runtime = object()

    def prepare_runtime() -> None:
        backend._temporary = runtime  # type: ignore[assignment]
        backend._docker_control_plane = SimpleNamespace(command=lambda *args: list(args))
        backend._docker_daemon_binding = b"d" * 32

    monkeypatch.setattr(backend, "_ensure_local_runtime", prepare_runtime)
    monkeypatch.setattr(backend, "_start_with_local_runtime", lambda: backend)

    control_plane, daemon_binding = backend.static_replay_control_plane_v1()
    assert control_plane is backend._docker_control_plane
    assert daemon_binding == b"d" * 32
    assert backend.start() is backend
    # Cleanup may erase every runtime handle, but it must never reopen the
    # one-shot actor-start boundary on this backend object.
    backend._temporary = None
    backend._docker_control_plane = None
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend.start()
    assert captured.value.code == executor.FAIL_CONTAINER


def test_failed_actor_start_attempt_cannot_be_retried_on_same_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    cleanup_calls = 0

    def prepare_runtime() -> None:
        backend._temporary = object()  # type: ignore[assignment]

    def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        backend._temporary = None

    monkeypatch.setattr(backend, "_ensure_local_runtime", prepare_runtime)
    monkeypatch.setattr(backend, "close", cleanup)
    monkeypatch.setattr(
        backend,
        "_start_with_local_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("start failed")),
    )

    with pytest.raises(RuntimeError, match="start failed"):
        backend.start()
    assert cleanup_calls == 1
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend.start()
    assert captured.value.code == executor.FAIL_CONTAINER
    assert cleanup_calls == 1


def test_second_start_does_not_cleanup_the_live_actor_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    live_containers = {purpose: str(purpose) * 64 for purpose in (1, 2, 3, 4)}
    cleanup_calls = 0

    def start_live_actors() -> executor.DockerCeremonyActorsV1:
        backend._containers = dict(live_containers)
        return backend

    def cleanup() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr(backend, "_ensure_local_runtime", lambda: None)
    monkeypatch.setattr(backend, "_start_with_local_runtime", start_live_actors)
    monkeypatch.setattr(backend, "close", cleanup)

    assert backend.start() is backend
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend.start()
    assert captured.value.code == executor.FAIL_CONTAINER
    assert cleanup_calls == 0
    assert backend._containers == live_containers


def test_operation_request_digest_override_is_narrowly_purpose4_only(
    tmp_path: Path,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._live_actor_set_qualified = True
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend._exec(
            1,
            "purpose1-authorized-sign",
            operation_request_digest_override=b"r" * 32,
        )
    assert captured.value.code == executor.FAIL_CONTAINER


@pytest.mark.parametrize("splice_operation_probe", (False, True))
def test_purpose4_sign_parent_passes_no_host_rows_or_preimage(
    tmp_path: Path, monkeypatch, splice_operation_probe: bool,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=7,
    )
    backend._root = tmp_path / "runtime"
    input_directory = backend._root / "purpose-4/input"
    output_directory = backend._root / "purpose-4/output"
    input_directory.mkdir(parents=True)
    output_directory.mkdir()
    backend._containers[4] = "4" * 64
    backend._profile = {
        "images": {"policy_auditor": "python@sha256:" + "1" * 64}
    }
    backend._public_keys[4] = b"p" * 32
    backend._key_ids[4] = hashlib.sha256(b"p" * 32).digest()[:16]
    backend._purpose4_snapshot_manifest = MappingProxyType({"snapshot": True})
    backend._purpose4_runtime_bundle = MappingProxyType({
        "runtime_inventory": {"inventory": True},
        "runtime_source_bindings": {"sources": True},
    })
    backend._purpose4_snapshot_path = input_directory / "detached-parent-snapshot"
    backend._purpose4_snapshot_path.mkdir()
    expected_root = b"a" * 32
    evidence = SimpleNamespace(audit_bundle_root=b"b" * 32)
    request = {
        "schema": "test-only",
        "request_sha256": (b"r" * 32).hex(),
    }
    operation_probe = {"schema": "test-operation-probe"}
    response = {
        "schema": "test-response",
        "operation_probe_receipt": (
            {"schema": "stale-spliced-probe"}
            if splice_operation_probe
            else operation_probe
        ),
    }
    response_payload = executor.purpose4_canonical_json_v1(response)
    captured_exec: dict[str, object] = {}

    monkeypatch.setattr(
        executor,
        "candidate_content_root",
        lambda name, _fields: expected_root
        if name == "ParentManifestAbsenceAttestationV2"
        else (_ for _ in ()).throw(AssertionError(name)),
    )
    monkeypatch.setattr(
        executor,
        "build_purpose4_keybearing_request_v1",
        lambda **_kwargs: request,
    )

    def fake_exec(purpose: int, operation: str, **kwargs):
        captured_exec.update({
            "purpose": purpose,
            "operation": operation,
            **kwargs,
        })
        (output_directory / "purpose4-keybearing-detached-response.json").write_bytes(
            response_payload
        )
        return MappingProxyType(operation_probe)

    monkeypatch.setattr(backend, "_exec", fake_exec)
    monkeypatch.setattr(
        executor,
        "make_openssl_ed25519_verifier_v1",
        lambda _directory: object(),
    )
    monkeypatch.setattr(
        executor,
        "validate_purpose4_keybearing_response_v1",
        lambda *_args, **_kwargs: SimpleNamespace(
            attestation_root=expected_root,
            audit_bundle_root=evidence.audit_bundle_root,
            signer_public_key=backend._public_keys[4],
            signer_key_id=backend._key_ids[4],
            signature=b"s" * 64,
        ),
    )

    if splice_operation_probe:
        with pytest.raises(executor.FormalContainerExecutorError) as captured:
            backend.sign_parent(evidence, {})
        assert captured.value.code == executor.FAIL_CONTAINER
        assert "different operation probe" in captured.value.detail
        return
    assert backend.sign_parent(evidence, {}) == b"s" * 64
    assert captured_exec == {
        "purpose": 4,
        "operation": "purpose4-parent-sign",
        "operation_request_digest_override": b"r" * 32,
        "timeout_seconds": 1800,
    }
    written_request = json.loads(
        (input_directory / "purpose4-keybearing-request.json").read_bytes()
    )
    assert written_request == request
    assert not (input_directory / "parent-audit-replay.json").exists()
    assert not (input_directory / "signing-preimage.bin").exists()


def test_purpose4_exec_uses_keybearing_runtime_worker_and_request_digest(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "runtime"
    (backend._root / "purpose-4/input").mkdir(parents=True)
    (backend._root / "purpose-4/output").mkdir()
    backend._containers[4] = "4" * 64
    backend._container_names[4] = "actor-4"
    backend._live_actor_set_qualified = True
    backend._transaction_run_id = b"r" * 16
    backend._profile_digest = b"p" * 32
    backend._docker_daemon_binding = b"d" * 32
    backend._profile = {
        "images": {"policy_auditor": "python@sha256:" + "1" * 64}
    }
    backend._docker_control_plane = SimpleNamespace(
        command=lambda *arguments: ["/usr/bin/docker", *arguments],
        environment=MappingProxyType({"PATH": "/usr/bin:/bin"}),
    )
    request_digest = b"q" * 32
    commands: list[list[str]] = []
    identity = {"inspection_sha256": "a" * 64}

    monkeypatch.setattr(backend, "_inspect_live_actor", lambda _purpose: identity)
    monkeypatch.setattr(
        backend,
        "_operation_request_digest",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("generic input-tree digest must not replace request_sha256")
        ),
    )
    monkeypatch.setattr(
        backend,
        "_read_single_json_line",
        lambda _path: ({}, b"{}\n"),
    )
    monkeypatch.setattr(
        backend,
        "_validate_python_operation_receipt",
        lambda *_args, **_kwargs: {"pid": "pid:[1]"},
    )

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    backend._exec(
        4,
        "purpose4-parent-sign",
        operation_request_digest_override=request_digest,
        timeout_seconds=1800,
    )
    assert len(commands) == 1
    command = commands[0]
    worker_index = command.index(
        "/input/runtime/tools/phase3_m25_purpose4_keybearing_detached_worker_v1.py"
    )
    assert command[worker_index - 3 : worker_index] == [
        "/usr/local/bin/python3",
        "-I",
        "-B",
    ]
    assert f"HEGEL_OPERATION_REQUEST_SHA256={request_digest.hex()}" in command
    assert backend.operation_probe_receipts[-1][
        "operation_request_sha256"
    ] == request_digest.hex()


def test_purpose4_preparation_failure_unwinds_adopted_snapshot_and_runtime(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "ceremony"
    (backend._root / "purpose-4/input").mkdir(parents=True)
    source_root = tmp_path / "snapshot-source"
    git_directory = source_root / ".git"
    git_directory.mkdir(parents=True)
    frozen_payload = git_directory / "frozen-object"
    frozen_payload.write_bytes(b"detached-parent-object-bytes")
    frozen_payload.chmod(0o444)
    git_directory.chmod(0o555)
    source_root.chmod(0o555)
    frozen_source_inode = source_root.stat().st_ino

    class TemporaryOwner:
        cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True

    class Snapshot:
        def __init__(self) -> None:
            self.root = source_root
            self.manifest = {"git_runtime_binding": {}}
            self.git_executable = Path("/usr/bin/git")
            self._temporary = TemporaryOwner()

        def close(self) -> None:
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None

    snapshot = Snapshot()
    monkeypatch.setattr(
        executor,
        "prepare_detached_parent_snapshot_v1",
        lambda *_args, **_kwargs: snapshot,
    )

    def validate_adopted_snapshot(
        path: Path,
        manifest,
        **kwargs,
    ):
        assert path == backend._root / "purpose-4/input/detached-parent-snapshot"
        assert not source_root.exists()
        assert path.stat().st_ino == frozen_source_inode
        assert stat.S_IMODE(path.stat().st_mode) == 0o555
        assert stat.S_IMODE((path / ".git").stat().st_mode) == 0o555
        assert stat.S_IMODE((path / ".git/frozen-object").stat().st_mode) == 0o444
        assert (path / ".git/frozen-object").read_bytes() == (
            b"detached-parent-object-bytes"
        )
        assert kwargs == {
            "git_executable": Path("/usr/bin/git"),
            "require_frozen_parent": True,
            "expected_basis_commit": "12" * 20,
        }
        return dict(manifest)

    monkeypatch.setattr(
        executor,
        "validate_detached_parent_snapshot_v1",
        validate_adopted_snapshot,
    )
    monkeypatch.setattr(
        executor,
        "_purpose4_runtime_source_bindings_v1",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend._prepare_purpose4_detached_inputs()
    assert captured.value.code == executor.FAIL_CONTAINER
    assert backend._purpose4_snapshot_path is None
    assert backend._purpose4_runtime_path is None
    assert snapshot._temporary is None
    assert not source_root.exists()
    assert not (backend._root / "purpose-4/input/detached-parent-snapshot").exists()
    assert not (backend._root / "purpose-4/input/runtime").exists()


def test_purpose4_read_only_snapshot_failed_rename_restores_source_without_copy(
    tmp_path: Path, monkeypatch,
) -> None:
    source_parent = tmp_path / "source-parent"
    destination_parent = tmp_path / "destination-parent"
    source_parent.mkdir(mode=0o700)
    destination_parent.mkdir(mode=0o700)
    source = source_parent / "snapshot"
    (source / ".git").mkdir(parents=True)
    payload = source / ".git/frozen-object"
    payload.write_bytes(b"frozen-object-bytes")
    payload.chmod(0o444)
    (source / ".git").chmod(0o555)
    source.chmod(0o555)
    source_inode = source.stat().st_ino
    destination = destination_parent / "adopted"
    snapshot = SimpleNamespace(
        root=source,
        manifest={"schema": "test-only"},
        git_executable=Path("/usr/bin/git"),
    )

    def fail_rename(*_args, **_kwargs) -> None:
        assert stat.S_IMODE(source.stat().st_mode) == 0o755
        assert source.stat().st_ino == source_inode
        assert payload.read_bytes() == b"frozen-object-bytes"
        assert not destination.exists()
        raise PermissionError("injected rename failure")

    monkeypatch.setattr(executor, "_renameat2_noreplace_v1", fail_rename)
    with pytest.raises(PermissionError, match="injected rename failure"):
        executor._adopt_read_only_purpose4_snapshot_v1(
            snapshot,  # type: ignore[arg-type]
            destination,
            expected_basis_commit="12" * 20,
        )
    assert source.exists()
    assert source.stat().st_ino == source_inode
    assert stat.S_IMODE(source.stat().st_mode) == 0o555
    assert stat.S_IMODE((source / ".git").stat().st_mode) == 0o555
    assert stat.S_IMODE(payload.stat().st_mode) == 0o444
    assert payload.read_bytes() == b"frozen-object-bytes"
    assert not destination.exists()


def test_purpose4_exact_cleanup_restores_guard_action_replacement_without_touching_it(
    tmp_path: Path, monkeypatch,
) -> None:
    parent = tmp_path / "private-parent"
    parent.mkdir(mode=0o700)
    owned = parent / "owned-tree"
    (owned / "nested").mkdir(parents=True)
    exact_payload = owned / "nested/exact"
    exact_payload.write_bytes(b"exact-owned-bytes")
    exact_payload.chmod(0o444)
    (owned / "nested").chmod(0o555)
    owned.chmod(0o555)
    exact_identity = owned.stat().st_dev, owned.stat().st_ino
    moved_exact = parent / "moved-exact"
    replacement_identity: tuple[int, int] | None = None
    actual_renameat2 = executor._renameat2_noreplace_v1
    injected = False

    def replace_between_guard_and_quarantine(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal injected, replacement_identity
        if not injected and source_name == owned.name:
            injected = True
            os.rename(owned, moved_exact)
            owned.mkdir(mode=0o700)
            (owned / "replacement-sentinel").write_bytes(b"preserve-replacement")
            replacement = owned.lstat()
            replacement_identity = replacement.st_dev, replacement.st_ino
        actual_renameat2(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )

    monkeypatch.setattr(
        executor,
        "_renameat2_noreplace_v1",
        replace_between_guard_and_quarantine,
    )
    recorded: list[Path] = []
    with pytest.raises(OSError, match="captured a replacement inode"):
        executor._remove_exact_owned_purpose4_tree_v1(
            owned,
            exact_identity,
            record_quarantine_path=recorded.append,
        )
    assert recorded == []
    assert replacement_identity is not None
    assert (owned.stat().st_dev, owned.stat().st_ino) == replacement_identity
    assert (owned / "replacement-sentinel").read_bytes() == b"preserve-replacement"
    assert (moved_exact.stat().st_dev, moved_exact.stat().st_ino) == exact_identity
    assert stat.S_IMODE(moved_exact.stat().st_mode) == 0o555
    assert (moved_exact / "nested/exact").read_bytes() == b"exact-owned-bytes"


def test_purpose4_completed_exact_cleanup_is_not_lost_to_fd_close_diagnostic(
    tmp_path: Path, monkeypatch,
) -> None:
    parent = tmp_path / "private-parent"
    parent.mkdir(mode=0o700)
    owned = parent / "owned-tree"
    owned.mkdir(mode=0o555)
    expected_identity = owned.stat().st_dev, owned.stat().st_ino
    actual_close = executor.os.close
    close_failure_pending = True

    def close_then_report_error(descriptor: int) -> None:
        nonlocal close_failure_pending
        actual_close(descriptor)
        if close_failure_pending:
            close_failure_pending = False
            raise OSError("injected post-removal close error")

    monkeypatch.setattr(executor.os, "close", close_then_report_error)
    recorded: list[Path] = []
    executor._remove_exact_owned_purpose4_tree_v1(
        owned,
        expected_identity,
        record_quarantine_path=recorded.append,
    )
    assert len(recorded) == 1
    assert not owned.exists()
    assert not recorded[0].exists()


def test_purpose4_eexist_survives_restore_and_close_errors_and_blocks_cleanup(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "ceremony"
    input_directory = backend._root / "purpose-4/input"
    input_directory.mkdir(parents=True)
    source_owner = tmp_path / "snapshot-owner"
    source_owner.mkdir(mode=0o700)
    source = source_owner / "snapshot"
    (source / ".git").mkdir(parents=True)
    (source / ".git/frozen").write_bytes(b"source")
    (source / ".git/frozen").chmod(0o444)
    (source / ".git").chmod(0o555)
    source.chmod(0o555)

    class TemporaryOwner:
        calls = 0

        def cleanup(self) -> None:
            self.calls += 1

    temporary_owner = TemporaryOwner()
    snapshot = SimpleNamespace(
        root=source,
        manifest={"schema": "test-only"},
        git_executable=Path("/usr/bin/git"),
        _temporary=temporary_owner,
    )
    monkeypatch.setattr(
        executor,
        "prepare_detached_parent_snapshot_v1",
        lambda *_args, **_kwargs: snapshot,
    )
    destination = input_directory / "detached-parent-snapshot"
    actual_renameat2 = executor._renameat2_noreplace_v1
    raced_identity: tuple[int, int] | None = None

    def inject_eexist(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal raced_identity
        if raced_identity is None:
            os.mkdir(destination_name, mode=0o700, dir_fd=destination_parent_fd)
            raced = os.stat(
                destination_name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
            raced_identity = raced.st_dev, raced.st_ino
        actual_renameat2(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )

    monkeypatch.setattr(executor, "_renameat2_noreplace_v1", inject_eexist)
    actual_fchmod = executor.os.fchmod
    restore_failure_pending = True

    def fail_first_restore(descriptor: int, mode: int) -> None:
        nonlocal restore_failure_pending
        if mode == 0o555 and restore_failure_pending:
            restore_failure_pending = False
            raise OSError("injected restore failure")
        actual_fchmod(descriptor, mode)

    monkeypatch.setattr(executor.os, "fchmod", fail_first_restore)
    actual_close = executor.os.close
    close_failure_pending = True

    def fail_first_close(descriptor: int) -> None:
        nonlocal close_failure_pending
        actual_close(descriptor)
        if close_failure_pending:
            close_failure_pending = False
            raise OSError("injected close failure")

    monkeypatch.setattr(executor.os, "close", fail_first_close)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend._prepare_purpose4_detached_inputs()
    assert captured.value.code == executor.FAIL_CONTAINER
    assert "File exists" in captured.value.detail
    primary = captured.value.__context__
    assert isinstance(primary, FileExistsError)
    assert primary.errno == errno.EEXIST
    cleanup_errors = getattr(primary, "_hegel_cleanup_error_chain", ())
    assert any(row[0] == "snapshot-root-mode-restore" for row in cleanup_errors)
    assert any(row[0] == "snapshot-descriptor-close" for row in cleanup_errors)
    assert raced_identity is not None
    assert (destination.stat().st_dev, destination.stat().st_ino) == raced_identity
    assert backend._purpose4_foreign_entries[destination] == raced_identity
    assert backend._purpose4_snapshot_owner is snapshot
    assert backend._purpose4_snapshot_tree_removed is True
    assert temporary_owner.calls == 0


def test_purpose4_raced_empty_destination_is_preserved_and_source_is_unwound(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "ceremony"
    input_directory = backend._root / "purpose-4/input"
    input_directory.mkdir(parents=True)
    source_owner = tmp_path / "snapshot-owner"
    source_owner.mkdir(mode=0o700)
    source = source_owner / "snapshot"
    (source / ".git").mkdir(parents=True)
    payload = source / ".git/frozen-object"
    payload.write_bytes(b"raced-adoption-source")
    payload.chmod(0o444)
    (source / ".git").chmod(0o555)
    source.chmod(0o555)
    source_inode = source.stat().st_ino
    destination = input_directory / "detached-parent-snapshot"

    class TemporaryOwner:
        cleaned = False

        def cleanup(self) -> None:
            self.cleaned = True
            if source_owner.exists():
                shutil.rmtree(source_owner)

    temporary_owner = TemporaryOwner()

    class Snapshot:
        def __init__(self) -> None:
            self.root = source
            self.manifest = {"schema": "test-only"}
            self.git_executable = Path("/usr/bin/git")
            self._temporary = temporary_owner
            self.close_saw_restored_source = False

        def close(self) -> None:
            assert self.root == source
            assert self.root.stat().st_ino == source_inode
            assert stat.S_IMODE(self.root.stat().st_mode) == 0o555
            assert payload.read_bytes() == b"raced-adoption-source"
            assert not destination.samefile(self.root)
            self.close_saw_restored_source = True
            executor._set_purpose4_snapshot_read_only_v1(self.root, False)
            self._temporary.cleanup()
            self._temporary = None

    snapshot = Snapshot()
    monkeypatch.setattr(
        executor,
        "prepare_detached_parent_snapshot_v1",
        lambda *_args, **_kwargs: snapshot,
    )
    actual_renameat2 = executor._renameat2_noreplace_v1
    raced_identity: tuple[int, int] | None = None

    def inject_empty_destination(
        source_parent_fd: int,
        source_name: str,
        destination_parent_fd: int,
        destination_name: str,
    ) -> None:
        nonlocal raced_identity
        if raced_identity is None:
            os.mkdir(destination_name, mode=0o700, dir_fd=destination_parent_fd)
            raced = os.stat(
                destination_name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            )
            raced_identity = raced.st_dev, raced.st_ino
        actual_renameat2(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )

    monkeypatch.setattr(
        executor,
        "_renameat2_noreplace_v1",
        inject_empty_destination,
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend._prepare_purpose4_detached_inputs()
    assert captured.value.code == executor.FAIL_CONTAINER
    assert "File exists" in captured.value.detail
    assert "snapshot-foreign-entry-retained" in captured.value.detail
    assert snapshot.close_saw_restored_source is False
    assert temporary_owner.cleaned is False
    assert source_owner.exists()
    assert not source.exists()
    assert raced_identity is not None
    raced = destination.lstat()
    assert (raced.st_dev, raced.st_ino) == raced_identity
    assert stat.S_ISDIR(raced.st_mode)
    assert not any(destination.iterdir())
    assert backend._purpose4_snapshot_path is None
    assert backend._purpose4_runtime_path is None
    assert backend._purpose4_foreign_entries[destination] == raced_identity
    assert backend._purpose4_snapshot_owner is snapshot
    assert backend._purpose4_snapshot_tree_removed is True

    with pytest.raises(executor.FormalContainerExecutorError) as cleanup:
        backend._cleanup_local_runtime()
    assert cleanup.value.code == executor.FAIL_CONTAINER
    assert "purpose4-foreign-entry-retained" in cleanup.value.detail
    raced_after_cleanup = destination.lstat()
    assert (raced_after_cleanup.st_dev, raced_after_cleanup.st_ino) == raced_identity
    assert not any(destination.iterdir())
    assert backend._root == tmp_path / "ceremony"


def test_purpose4_postrename_replacement_is_never_touched_and_state_is_retained(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "ceremony"
    input_directory = backend._root / "purpose-4/input"
    input_directory.mkdir(parents=True)
    source_owner = backend._root / "detached-owner"
    source_owner.mkdir(mode=0o700)
    source = source_owner / "snapshot"
    (source / ".git").mkdir(parents=True)
    frozen = source / ".git/frozen-object"
    frozen.write_bytes(b"exact-adopted-inode")
    frozen.chmod(0o444)
    (source / ".git").chmod(0o555)
    source.chmod(0o555)

    class TemporaryOwner:
        calls = 0

        def cleanup(self) -> None:
            self.calls += 1

    temporary_owner = TemporaryOwner()

    class Snapshot:
        def __init__(self) -> None:
            self.root = source
            self.manifest = {"schema": "test-only"}
            self.git_executable = Path("/usr/bin/git")
            self._temporary = temporary_owner
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise AssertionError("replacement path must never reach snapshot.close")

    snapshot = Snapshot()
    monkeypatch.setattr(
        executor,
        "prepare_detached_parent_snapshot_v1",
        lambda *_args, **_kwargs: snapshot,
    )
    destination = input_directory / "detached-parent-snapshot"
    quarantined = input_directory / "exact-adopted-inode-quarantine"
    replacement_identity: tuple[int, int] | None = None

    def replace_during_validation(_path: Path, _manifest, **_kwargs):
        nonlocal replacement_identity
        os.rename(destination, quarantined)
        destination.mkdir(mode=0o700)
        (destination / "replacement-sentinel").write_bytes(b"do-not-touch")
        replacement = destination.lstat()
        replacement_identity = replacement.st_dev, replacement.st_ino
        raise OSError("injected post-rename validation failure")

    monkeypatch.setattr(
        executor,
        "validate_detached_parent_snapshot_v1",
        replace_during_validation,
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend._prepare_purpose4_detached_inputs()
    assert captured.value.code == executor.FAIL_CONTAINER
    assert "snapshot-owner-exact-remove:OSError" in captured.value.detail
    assert replacement_identity is not None
    replacement = destination.lstat()
    assert (replacement.st_dev, replacement.st_ino) == replacement_identity
    assert (destination / "replacement-sentinel").read_bytes() == b"do-not-touch"
    adopted = quarantined.lstat()
    assert (adopted.st_dev, adopted.st_ino) == backend._purpose4_snapshot_identity
    assert stat.S_IMODE(adopted.st_mode) == 0o555
    assert snapshot.close_calls == 0
    assert temporary_owner.calls == 0
    assert backend._purpose4_snapshot_path == destination
    assert backend._purpose4_snapshot_owner is snapshot

    with pytest.raises(executor.FormalContainerExecutorError) as cleanup:
        backend._cleanup_local_runtime()
    assert cleanup.value.code == executor.FAIL_CONTAINER
    assert "purpose4-detached-owner-exact-remove:OSError" in cleanup.value.detail
    assert (destination.lstat().st_dev, destination.lstat().st_ino) == (
        replacement_identity
    )
    assert (destination / "replacement-sentinel").read_bytes() == b"do-not-touch"
    assert backend._root == tmp_path / "ceremony"
    assert backend._purpose4_snapshot_path == destination
    assert backend._purpose4_snapshot_owner is snapshot
    assert snapshot.close_calls == 0


def test_purpose4_preexisting_destination_blocks_broad_runtime_cleanup(
    tmp_path: Path,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "ceremony"
    destination = backend._root / "purpose-4/input/detached-parent-snapshot"
    destination.mkdir(parents=True)
    (destination / "foreign-sentinel").write_bytes(b"preserve-me")
    foreign_identity = destination.stat().st_dev, destination.stat().st_ino

    class BroadTemporary:
        calls = 0

        def cleanup(self) -> None:
            self.calls += 1
            shutil.rmtree(backend._root)

    temporary = BroadTemporary()
    backend._temporary = temporary  # type: ignore[assignment]
    with pytest.raises(executor.FormalContainerExecutorError) as prepared:
        backend._prepare_purpose4_detached_inputs()
    assert prepared.value.code == executor.FAIL_CONTAINER
    assert backend._purpose4_foreign_entries[destination] == foreign_identity

    with pytest.raises(executor.FormalContainerExecutorError) as cleanup:
        backend._cleanup_local_runtime()
    assert cleanup.value.code == executor.FAIL_CONTAINER
    assert "purpose4-foreign-entry-retained" in cleanup.value.detail
    assert temporary.calls == 0
    assert (destination.stat().st_dev, destination.stat().st_ino) == foreign_identity
    assert (destination / "foreign-sentinel").read_bytes() == b"preserve-me"
    assert backend._temporary is temporary
    assert backend._root == tmp_path / "ceremony"


def test_purpose4_snapshot_owner_close_failure_is_reported_and_retained(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "ceremony"
    input_directory = backend._root / "purpose-4/input"
    input_directory.mkdir(parents=True)
    source_owner = backend._root / "detached-owner"
    source_owner.mkdir(mode=0o700)
    source = source_owner / "snapshot"
    (source / ".git").mkdir(parents=True)
    frozen = source / ".git/frozen-object"
    frozen.write_bytes(b"retry-owned-snapshot")
    frozen.chmod(0o444)
    (source / ".git").chmod(0o555)
    source.chmod(0o555)

    class FailingTemporaryOwner:
        calls = 0

        def cleanup(self) -> None:
            self.calls += 1
            raise OSError("injected detached-owner cleanup failure")

    temporary_owner = FailingTemporaryOwner()

    class Snapshot:
        def __init__(self) -> None:
            self.root = source
            self.manifest = {"schema": "test-only"}
            self.git_executable = Path("/usr/bin/git")
            self._temporary = temporary_owner
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            executor._set_purpose4_snapshot_read_only_v1(self.root, False)
            self._temporary.cleanup()
            self._temporary = None

    snapshot = Snapshot()
    monkeypatch.setattr(
        executor,
        "prepare_detached_parent_snapshot_v1",
        lambda *_args, **_kwargs: snapshot,
    )

    def validate_adopted(path: Path, manifest, **_kwargs):
        assert stat.S_IMODE(path.stat().st_mode) == 0o555
        return dict(manifest)

    monkeypatch.setattr(
        executor,
        "validate_detached_parent_snapshot_v1",
        validate_adopted,
    )
    destination = input_directory / "detached-parent-snapshot"
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend._prepare_purpose4_detached_inputs()
    assert captured.value.code == executor.FAIL_CONTAINER
    assert "snapshot-owner-close:OSError" in captured.value.detail
    assert temporary_owner.calls == 2
    assert snapshot.close_calls == 0
    assert backend._purpose4_snapshot_owner is snapshot
    assert backend._purpose4_snapshot_tree_removed is True
    assert backend._purpose4_snapshot_path is not None
    assert not backend._purpose4_snapshot_path.exists()
    assert not destination.exists()

    destination.mkdir(mode=0o700)
    (destination / "foreign-after-owner-remove").write_bytes(b"preserve")
    foreign_identity = destination.stat().st_dev, destination.stat().st_ino
    with pytest.raises(executor.FormalContainerExecutorError) as cleanup:
        backend._cleanup_local_runtime()
    assert cleanup.value.code == executor.FAIL_CONTAINER
    assert "foreign-entry-retained" in cleanup.value.detail
    assert temporary_owner.calls == 2
    assert (destination.stat().st_dev, destination.stat().st_ino) == foreign_identity
    assert (destination / "foreign-after-owner-remove").read_bytes() == b"preserve"
    assert backend._purpose4_snapshot_owner is snapshot


def test_purpose4_purge_failure_tracks_first_vacated_name_across_retry(
    tmp_path: Path,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "ceremony"
    snapshot = backend._root / "purpose-4/input/detached-parent-snapshot"
    snapshot.mkdir(parents=True)
    os.symlink("/nonexistent", snapshot / "forbidden-link")
    snapshot.chmod(0o555)
    backend._purpose4_snapshot_path = snapshot
    backend._purpose4_snapshot_identity = (
        snapshot.stat().st_dev,
        snapshot.stat().st_ino,
    )

    class BroadTemporary:
        calls = 0

        def cleanup(self) -> None:
            self.calls += 1
            shutil.rmtree(backend._root)

    temporary = BroadTemporary()
    backend._temporary = temporary  # type: ignore[assignment]
    with pytest.raises(executor.FormalContainerExecutorError) as first:
        backend._cleanup_local_runtime()
    assert first.value.code == executor.FAIL_CONTAINER
    quarantine = backend._purpose4_snapshot_path
    assert quarantine is not None and quarantine != snapshot
    assert quarantine.exists()
    assert snapshot in backend._purpose4_vacated_paths
    assert temporary.calls == 0

    (quarantine / "forbidden-link").unlink()
    snapshot.mkdir(mode=0o700)
    (snapshot / "foreign-after-purge-failure").write_bytes(b"preserve")
    foreign_identity = snapshot.stat().st_dev, snapshot.stat().st_ino
    with pytest.raises(executor.FormalContainerExecutorError) as second:
        backend._cleanup_local_runtime()
    assert second.value.code == executor.FAIL_CONTAINER
    assert "foreign-entry-retained" in second.value.detail
    assert temporary.calls == 0
    assert backend._purpose4_snapshot_path is None
    assert (snapshot.stat().st_dev, snapshot.stat().st_ino) == foreign_identity
    assert (snapshot / "foreign-after-purge-failure").read_bytes() == b"preserve"


def test_local_runtime_cleanup_failure_keeps_retry_handles_and_skips_broad_remove(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "ceremony"
    snapshot = backend._root / "purpose-4/input/detached-parent-snapshot"
    runtime = backend._root / "purpose-4/input/runtime"
    snapshot.mkdir(parents=True)
    runtime.mkdir()
    backend._purpose4_snapshot_path = snapshot
    backend._purpose4_snapshot_identity = (
        snapshot.stat().st_dev,
        snapshot.stat().st_ino,
    )
    backend._purpose4_runtime_path = runtime
    backend._purpose4_runtime_identity = (
        runtime.stat().st_dev,
        runtime.stat().st_ino,
    )

    class FailingTemporary:
        calls = 0

        def cleanup(self) -> None:
            self.calls += 1
            raise OSError("injected cleanup failure")

    temporary = FailingTemporary()
    backend._temporary = temporary  # type: ignore[assignment]

    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend._cleanup_local_runtime()
    assert captured.value.code == executor.FAIL_CONTAINER
    assert temporary.calls == 1
    assert backend._temporary is temporary
    assert backend._root == tmp_path / "ceremony"
    assert backend._purpose4_snapshot_path is None
    assert backend._purpose4_runtime_path is None


def test_bridge_signer_receives_only_full_dag_package(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "runtime"
    input_directory = backend._root / "purpose-1/input"
    output_directory = backend._root / "purpose-1/output"
    input_directory.mkdir(parents=True)
    output_directory.mkdir()
    backend._containers[1] = "1" * 64
    backend._public_keys[1] = b"p" * 32
    root = b"b" * 32
    result = SimpleNamespace(
        purpose_id=1,
        authoritative=True,
        eligible_to_sign_bridge_statement=True,
        purpose1_signature_verified=False,
        split_membership_recomputed=False,
        bridge_statement_root=root,
    )
    signature = b"s" * 64
    receipt_payload = b'{"test":true}\n'
    verifier_calls: list[tuple[bytes, bytes, bytes]] = []

    monkeypatch.setattr(executor, "candidate_content_root", lambda *_args: root)
    monkeypatch.setattr(
        executor,
        "replay_bridge_dag_package_v1",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr(
        executor,
        "make_openssl_ed25519_verifier_v1",
        lambda _directory: (
            lambda public, candidate_signature, message: verifier_calls.append(
                (public, candidate_signature, message)
            )
        ),
    )
    monkeypatch.setattr(
        executor,
        "validate_bridge_actor_replay_receipt_v1",
        lambda payload, **_kwargs: MappingProxyType({"payload": payload}),
    )

    def fake_exec(_purpose: int, _operation: str):
        (output_directory / "ed25519-signature.bin").write_bytes(signature)
        (output_directory / "bridge-dag-replay-receipt.json").write_bytes(
            receipt_payload
        )
        return MappingProxyType({})

    monkeypatch.setattr(backend, "_exec", fake_exec)
    assert backend.sign_bridge(1, {}, b"full-dag-package") == signature
    assert (input_directory / "bridge-dag-package.cbor").read_bytes() == b"full-dag-package"
    for forbidden in (
        "bridge-statement.cbor",
        "expected-root.bin",
        "signing-preimage.bin",
    ):
        assert not (input_directory / forbidden).exists()
    assert verifier_calls and verifier_calls[-1][:2] == (b"p" * 32, signature)


def test_ready_binding_set_must_name_both_complete_enumerators(monkeypatch) -> None:
    def validate(basis):
        roots = basis.implementation_inputs["m3_execution_implementation_binding_roots"]
        if set(roots) != {
            "python_implementation_binding_root",
            "rust_implementation_binding_root",
        }:
            raise executor.M3ImplementationQualificationError(
                "FAIL_TEST_BINDING", "binding names differ"
            )
        return MappingProxyType(dict(roots))

    monkeypatch.setattr(
        executor, "validate_m3_execution_implementation_bindings_v1", validate
    )
    roots = executor.require_formal_ceremony_ready_v1(_basis(ready=True))
    assert set(roots) == {
        "python_implementation_binding_root",
        "rust_implementation_binding_root",
    }
    bad = _basis(ready=True)
    bad.implementation_inputs["m3_execution_implementation_binding_roots"] = {
        "python_static_replay_implementation_binding_root": b"p" * 32,
        "rust_static_replay_implementation_binding_root": b"r" * 32,
    }
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.require_formal_ceremony_ready_v1(bad)
    assert captured.value.code == executor.FAIL_EXECUTION_BINDINGS


def _dummy_gate_inputs() -> GateEvidenceInputsV1:
    values = {}
    for field in fields(GateEvidenceInputsV1):
        if field.name == "basis_commit":
            values[field.name] = "34" * 20
        elif field.name == "marker_snapshot":
            values[field.name] = MarkerSnapshot(
                "COMPLETE", b"s" * 32, b"m" * 32, b"k" * 16, 1
            )
        elif field.name.endswith("_frame"):
            values[field.name] = b"frame"
        elif field.name == "split_seed_commitment_fields":
            values[field.name] = {"split_seed_commitment_digest": b"c" * 32}
        elif field.name in {
            "actor_key_manifests",
            "parent_top_level_path_rows",
            "parent_history_rows",
            "parent_touched_rows",
            "parent_legacy_rows",
            "external_envelopes",
            "canonical_binding_objects",
            "opaque_registration_intents",
            "opaque_registry_records",
            "opaque_registry_snapshots",
            "bridge_envelopes",
        }:
            values[field.name] = ()
        else:
            values[field.name] = {}
    return GateEvidenceInputsV1(**values)


def test_public_gate_evidence_round_trip_contains_every_replay_input() -> None:
    original = _dummy_gate_inputs()
    payload = executor.serialize_gate_evidence_inputs_v1(original)
    restored = executor.load_gate_evidence_inputs_v1(payload)
    assert restored == original
    assert payload["contains_private_key"] is False
    assert payload["contains_raw_split_seed"] is False
    assert payload["contains_split_assignment_rows"] is False


def test_public_gate_evidence_digest_is_tamper_evident() -> None:
    payload = executor.serialize_gate_evidence_inputs_v1(_dummy_gate_inputs())
    payload["contains_raw_split_seed"] = True
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.load_gate_evidence_inputs_v1(payload)
    assert captured.value.code == executor.FAIL_PUBLICATION


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: payload.update({"unexpected": False}),
        lambda payload: payload.update({"contains_private_key": 0}),
        lambda payload: payload.update({"contains_raw_split_seed": True}),
        lambda payload: payload.update({"artifact_kind": "OTHER"}),
    ),
)
def test_public_gate_evidence_loader_requires_exact_top_level_and_false_flags(
    mutation,
) -> None:
    payload = executor.serialize_gate_evidence_inputs_v1(_dummy_gate_inputs())
    mutation(payload)
    body = dict(payload)
    body.pop("payload_sha256", None)
    payload["payload_sha256"] = hashlib.sha256(executor._canonical_json(body)).hexdigest()
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.load_gate_evidence_inputs_v1(payload)
    assert captured.value.code == executor.FAIL_PUBLICATION


def test_synthetic_actor_cannot_enter_formal_execute(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(executor, "build_qualified_formal_static_basis_v1", lambda _commit: _basis(ready=True))
    monkeypatch.setattr(
        executor,
        "validate_m3_execution_implementation_bindings_v1",
        lambda basis: MappingProxyType(
            dict(basis.implementation_inputs["m3_execution_implementation_binding_roots"])
        ),
    )
    monkeypatch.setattr(executor, "validate_ceremony_admission_v1", lambda **_kwargs: {})
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    actors = SimpleNamespace(authoritative=False)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.execute_formal_container_ceremony_v1(
            basis_commit="12" * 20,
            actor_qualification_report={},
            errata_qualification_report={},
            custody_directory=custody,
            public_evidence_path=tmp_path / "evidence.json",
            public_promotion_path=tmp_path / "promotion.json",
            actors=actors,
        )
    assert captured.value.code == executor.FAIL_SYNTHETIC_PROMOTION
    assert list(custody.iterdir()) == []


def test_committed_actor_profile_requires_exact_centralized_disclosure(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    valid = {"authority_disclosure": dict(executor.TECHNICAL_ACTOR_DISCLOSURE_V1)}
    monkeypatch.setattr(
        backend,
        "_git_blob",
        lambda _path: executor._canonical_json(valid),
    )
    backend._load_committed_profile()
    assert backend._profile["authority_disclosure"] == dict(
        executor.TECHNICAL_ACTOR_DISCLOSURE_V1
    )

    legacy = {
        "authority_disclosure": {
            "same_admin_controller": True,
            "organizational_independence": False,
            "independent_human_actors": False,
            "technical_role_independence_required": True,
            "owner_accepts_profile_for_formal_actor_eligibility": True,
        }
    }
    monkeypatch.setattr(
        backend,
        "_git_blob",
        lambda _path: executor._canonical_json(legacy),
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend._load_committed_profile()
    assert captured.value.code == executor.FAIL_CONTAINER


@pytest.mark.parametrize(
    "mutator",
    (
        lambda disclosure: disclosure.pop("remote_attestation"),
        lambda disclosure: disclosure.__setitem__("extra_claim", False),
        lambda disclosure: disclosure.__setitem__("same_admin_controller", False),
    ),
)
def test_committed_actor_profile_rejects_disclosure_shape_or_value_drift(
    tmp_path: Path, monkeypatch, mutator,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    disclosure = dict(executor.TECHNICAL_ACTOR_DISCLOSURE_V1)
    mutator(disclosure)
    monkeypatch.setattr(
        backend,
        "_git_blob",
        lambda _path: executor._canonical_json({"authority_disclosure": disclosure}),
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend._load_committed_profile()
    assert captured.value.code == executor.FAIL_CONTAINER


def test_formal_daemon_binding_accepts_same_identity_across_backend_instances_and_rejects_swap(
    tmp_path: Path, monkeypatch,
) -> None:
    backends = tuple(
        executor.DockerCeremonyActorsV1(
            basis_commit="12" * 20,
            custody_directory=tmp_path,
            rust_formal_replay_binary=tmp_path / "rust-replay",
            timestamp=1,
        )
        for _ in range(2)
    )
    for backend in backends:
        backend._docker_daemon_binding = b"d" * 32
        monkeypatch.setattr(backend, "_ensure_local_runtime", lambda: None)
        backend.validate_frozen_daemon_receipt_binding_v1(b"d" * 32)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backends[1].validate_frozen_daemon_receipt_binding_v1(b"x" * 32)
    assert captured.value.code == executor.FAIL_PREFLIGHT


def test_actor_protocol_binding_shape_and_formal_key_disjointness_are_fail_closed() -> None:
    key_ids = {purpose: bytes([purpose]) * 16 for purpose in (1, 2, 3, 4)}
    report = {"synthetic_test_bundle": True}
    archive = executor._validate_actor_protocol_binding_object_v1(
        SimpleNamespace(
            basis_commit="12" * 20,
            bundle_content_id=b"b" * 32,
            qualification_key_ids=key_ids,
            report=report,
        ),
        basis_commit="12" * 20,
        live=False,
    )
    assert dict(archive.qualification_key_ids) == key_ids
    live = executor._validate_actor_protocol_binding_object_v1(
        SimpleNamespace(
            basis_commit="12" * 20,
            bundle_content_id=b"b" * 32,
            qualification_key_ids=key_ids,
            daemon_receipt_binding=b"d" * 32,
            canonical_bundle_bytes=executor._canonical_json(report),
        ),
        basis_commit="12" * 20,
        live=True,
    )
    assert live.canonical_bundle_bytes == executor._canonical_json(report)
    for invalid in (
        SimpleNamespace(
            basis_commit="12" * 20,
            bundle_content_id=None,
            qualification_key_ids=key_ids,
            report=report,
        ),
        SimpleNamespace(
            basis_commit="12" * 20,
            bundle_content_id=b"b" * 32,
            qualification_key_ids={purpose: b"z" * 16 for purpose in (1, 2, 3, 4)},
            report=report,
        ),
    ):
        with pytest.raises(ValueError):
            executor._validate_actor_protocol_binding_object_v1(
                invalid, basis_commit="12" * 20, live=False
            )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor._require_formal_key_ids_disjoint_from_qualification_v1(
            {1: b"a" * 16, 2: b"b" * 16, 3: b"c" * 16, 4: b"d" * 16},
            {1: b"x" * 16, 2: b"b" * 16, 3: b"y" * 16, 4: b"z" * 16},
        )
    assert captured.value.code == executor.FAIL_PREFLIGHT


def test_docker_command_keeps_four_role_state_private_and_offline(tmp_path: Path) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "work"
    backend._profile = {
        "images": {
            "custodian": "python@sha256:" + "1" * 64,
            "python_attester": "python@sha256:" + "1" * 64,
            "rust_attester": "rust@sha256:" + "2" * 64,
            "policy_auditor": "python@sha256:" + "1" * 64,
        }
    }
    backend._ceremony_token = "ab" * 8
    backend._transaction_run_id = bytes.fromhex("cd" * 16)
    backend._profile_digest = b"d" * 32
    _install_fake_docker_boundary(backend, tmp_path)
    backend._state_volumes = {
        purpose: f"hegel-m25-state-{'cd' * 16}-p{purpose}"
        for purpose in (1, 2, 3, 4)
    }
    for purpose in (1, 2, 3, 4):
        (backend._root / f"purpose-{purpose}/input").mkdir(parents=True)
        (backend._root / f"purpose-{purpose}/output").mkdir()
        command = backend._base_container_command(purpose, f"actor-{purpose}")
        assert "--pull=never" in command
        assert "--network=none" in command
        assert "--read-only" in command
        assert "--cap-drop=ALL" in command
        assert not any(item.startswith("--tmpfs=/state:") for item in command)
        state_mounts = [item for item in command if "dst=/state" in item]
        assert state_mounts == [
            f"--mount=type=volume,src={backend._state_volumes[purpose]},dst=/state,volume-nocopy"
        ]
        custody_mounts = [item for item in command if "dst=/custody" in item]
        assert len(custody_mounts) == (1 if purpose == 1 else 0)
        assert not any(
            item.startswith("HEGEL_HOST_REPOSITORY_PATH=") for item in command
        )
        assert any(
            item.startswith("HEGEL_HOST_REPOSITORY_PATH_SHA256=")
            for item in command
        )


def test_pending_crash_is_not_redrawn(monkeypatch, tmp_path: Path) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=123,
    )
    backend._root = tmp_path / "work"
    backend._containers = {1: "a" * 64}
    actor = backend._root / "purpose-1"
    (actor / "input").mkdir(parents=True)
    (actor / "output").mkdir()
    backend._key_ids[1] = b"k" * 16
    monkeypatch.setattr(backend, "_handoff_custody_to_actor", lambda: None)

    def crash(_purpose: int, _operation: str) -> None:
        raise executor.FormalContainerExecutorError(executor.FAIL_CONTAINER, "injected")

    monkeypatch.setattr(backend, "_exec", crash)
    with pytest.raises(executor.FormalContainerExecutorError):
        backend.seed_split()
    marker = custody / "split_seed_instantiation.marker"
    assert marker.exists()
    assert b'"state":"PENDING"' in marker.read_bytes()
    with pytest.raises(Exception) as second:
        backend.seed_split()
    assert "PENDING_EXTERNAL_RECOVERY_REQUIRED" in str(second.value)


def test_post_stage_docker_resume_requires_completed_seed_before_worker_entry(
    monkeypatch, tmp_path: Path,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    entered = False

    def forbidden_resume():
        nonlocal entered
        entered = True
        raise AssertionError("seed worker entered without completed seed state")

    monkeypatch.setattr(backend, "resume_pending_seed_split", forbidden_resume)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend.resume_post_stage_seed_split()
    assert captured.value.code == executor.FAIL_CUSTODY
    assert entered is False


def test_complete_recovery_deletes_any_exact_remaining_volume_subset(
    monkeypatch, tmp_path: Path,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    profile = {
        "images": {
            "custodian": "python@sha256:" + "1" * 64,
            "python_attester": "python@sha256:" + "1" * 64,
            "rust_attester": "rust@sha256:" + "2" * 64,
            "policy_auditor": "python@sha256:" + "1" * 64,
        }
    }

    def load_profile() -> None:
        backend._profile = profile
        backend._profile_digest = b"d" * 32

    monkeypatch.setattr(backend, "_load_committed_profile", load_profile)
    _install_fake_docker_boundary(backend, tmp_path)
    monkeypatch.setattr(backend, "_git_blob", lambda _relative: b"{}\n")
    monkeypatch.setattr(
        executor,
        "validate_linux_local_durable_custody_location_v1",
        lambda *_args, **_kwargs: {
            "schema": "test-only-location",
            "owner_uid": os.geteuid(),
        },
    )
    monkeypatch.setattr(
        executor,
        "validate_linux_local_durable_custody_v1",
        lambda *_args, **_kwargs: {"schema": "test-only"},
    )
    monkeypatch.setattr(
        backend,
        "_verify_complete_custody_retained",
        lambda: MappingProxyType({"schema": "test-only"}),
    )
    actor_cleanup_called = False

    def actor_cleanup() -> None:
        nonlocal actor_cleanup_called
        actor_cleanup_called = True

    monkeypatch.setattr(backend, "_recover_remove_exact_actor_containers", actor_cleanup)
    present = {1, 3}

    def fake_run(command, **_kwargs):
        if command[:3] == ["docker", "volume", "inspect"]:
            purpose = int(command[3].rsplit("-p", 1)[1])
            if purpose not in present:
                return SimpleNamespace(returncode=1, stdout=b"", stderr=b"")
            payload = [
                _fake_volume_row(backend, purpose, command[3])
            ]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload).encode("ascii"),
                stderr=b"",
            )
        if command[:3] == ["docker", "volume", "rm"]:
            purpose = int(command[3].rsplit("-p", 1)[1])
            present.remove(purpose)
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if command[:3] == ["docker", "volume", "ls"]:
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        raise AssertionError(command)

    monkeypatch.setattr(executor, "_run", fake_run)
    marker = MarkerSnapshot("COMPLETE", b"s" * 32, b"m" * 32, b"k" * 16, 1)
    backend.recover_complete_private_state_and_verify_absent(b"r" * 16, marker)
    assert actor_cleanup_called is True
    assert present == set()
    assert backend._marker_completed_after_staging is True


def test_recovery_removes_only_exact_run_bound_actor_containers(
    monkeypatch, tmp_path: Path,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._transaction_run_id = b"r" * 16
    backend._profile_digest = b"d" * 32
    backend._profile = {
        "images": {
            "custodian": "python@sha256:" + "1" * 64,
            "python_attester": "python@sha256:" + "1" * 64,
            "rust_attester": "rust@sha256:" + "2" * 64,
            "policy_auditor": "python@sha256:" + "1" * 64,
        }
    }
    _install_fake_docker_boundary(backend, tmp_path)
    image_keys = {
        1: "custodian", 2: "python_attester", 3: "rust_attester", 4: "policy_auditor"
    }
    ids = {purpose: str(purpose) * 64 for purpose in (1, 2, 3, 4)}
    removed: set[str] = set()

    def actor_row(purpose: int) -> dict[str, object]:
        return {
            "Config": {
                "Labels": {
                    "hegel.m25.ceremony": "ab" * 8,
                    "hegel.m25.purpose": str(purpose),
                    "hegel.m25.run": (b"r" * 16).hex(),
                    "hegel.m25.basis": "12" * 20,
                    "hegel.m25.profile_sha256": (b"d" * 32).hex(),
                },
                "User": "65534:65534",
                "Image": backend._profile["images"][image_keys[purpose]],
            },
            "Mounts": [
                {
                    "Type": "volume", "Destination": "/state", "RW": True,
                    "Name": backend._state_volume_name(purpose),
                },
                {"Type": "bind", "Destination": "/input", "RW": False},
                {"Type": "bind", "Destination": "/output", "RW": True},
                *(
                    [{"Type": "bind", "Destination": "/custody", "RW": True}]
                    if purpose == 1 else []
                ),
            ],
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "Privileged": False,
                "CapDrop": ["ALL"],
            },
        }

    def fake_run(command, **_kwargs):
        if command[:3] == ["docker", "ps", "-aq"]:
            filter_value = command[-1]
            if filter_value.startswith("volume="):
                purpose = int(filter_value.rsplit("-p", 1)[1])
                rows = [] if ids[purpose] in removed else [ids[purpose]]
            else:
                rows = [value for value in ids.values() if value not in removed]
            return SimpleNamespace(
                returncode=0,
                stdout=(("\n".join(rows) + "\n") if rows else "").encode("ascii"),
                stderr=b"",
            )
        if command[:2] == ["docker", "inspect"]:
            purposes = [int(container_id[0]) for container_id in command[2:]]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([actor_row(purpose) for purpose in purposes]).encode("ascii"),
                stderr=b"",
            )
        if command[:3] == ["docker", "rm", "--force"]:
            removed.add(command[3])
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        raise AssertionError(command)

    monkeypatch.setattr(executor, "_run", fake_run)
    backend._recover_remove_exact_actor_containers()
    assert removed == set(ids.values())


def test_workers_expose_no_real_generic_sign_operation() -> None:
    project = Path(__file__).resolve().parents[1]
    custodian_worker = (project / "tools/phase3_m25_formal_actor_worker_v1.py").read_text()
    python_worker = (
        project / "tools/phase3_m25_python_bridge_actor_worker_v1.py"
    ).read_text()
    auditor_worker = (
        project / "tools/phase3_m25_parent_auditor_actor_worker_v1.py"
    ).read_text()
    rust_worker = (project / "tools/phase3_m25_formal_rust_actor_worker_v1.sh").read_text()
    assert 'operation == "sign"' not in python_worker
    assert "bridge-replay-sign-rust" in rust_worker
    assert "purpose1-authorized-sign" in custodian_worker
    assert "purpose4-parent-sign" not in custodian_worker
    assert "seed-split-real" not in python_worker
    assert "purpose4-parent-sign" in auditor_worker
    assert "seed-split-real" not in auditor_worker


def test_actor_snapshot_allowlists_are_purpose_private_and_minimal() -> None:
    paths = executor.ACTOR_SNAPSHOT_PATHS_BY_PURPOSE
    assert set(paths) == {1, 2, 3, 4}
    assert all("phase3_dsl_v1.py" not in row for rows in paths.values() for row in rows)
    assert all("phase3_m25_secret_absence_v1.py" not in row for rows in paths.values() for row in rows)
    assert any("split_partition_calculator" in row for row in paths[1])
    assert not any("split_partition_calculator" in row for purpose in (2, 3, 4) for row in paths[purpose])
    assert any("parent_absence_audit" in row for row in paths[4])
    assert not any("parent_absence_audit" in row for purpose in (1, 2, 3) for row in paths[purpose])
    assert len({row for rows in paths.values() for row in rows if "actor_worker" in row}) == 4


def test_purpose_worker_sources_hide_unrelated_operation_tokens() -> None:
    tools = Path(__file__).resolve().parents[1] / "tools"
    custodian = (tools / "phase3_m25_formal_actor_worker_v1.py").read_text()
    bridge = (tools / "phase3_m25_python_bridge_actor_worker_v1.py").read_text()
    rust_bridge = (tools / "phase3_m25_formal_rust_actor_worker_v1.sh").read_text()
    auditor = (tools / "phase3_m25_parent_auditor_actor_worker_v1.py").read_text()
    assert "ParentManifestAbsenceAttestationV2" not in custodian
    assert "split_master_seed" not in bridge
    assert "parent-audit-replay" not in bridge
    assert "split_master_seed" not in rust_bridge
    assert "split_master_seed" not in auditor
    assert "bridge-replay-sign" not in auditor


def _public_replay_stub(_payload):
    return {"qualified": True, "state": "NOT_RUN"}


def _transaction(tmp_path: Path, fault_injector=None):
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    os.chmod(custody, 0o700)
    public = tmp_path / "public"
    public.mkdir()
    actor_trust = _test_actor_trust()
    intent = executor.build_prestage_intent_fields_v1(
        basis_commit="12" * 20,
        run_id=b"r" * 16,
        ledger_id=b"l" * 16,
        created_at_unix_seconds=7,
        trust_genesis_id=b"t" * 16,
        actor_qualification_report={},
        errata_qualification_report={},
        rust_bridge_dag_qualification_report_sha256=b"q" * 32,
        live_actor_protocol_qualification_bundle_content_id=b"v" * 32,
        qualification_only_key_ids={
            purpose: bytes([purpose]) * 16 for purpose in (1, 2, 3, 4)
        },
        live_actor_protocol_qualification_bundle={"synthetic_test_bundle": True},
        live_actor_protocol_qualification_canonical_bundle_bytes=(
            executor._canonical_json({"synthetic_test_bundle": True})
        ),
        live_actor_protocol_daemon_receipt_binding=b"d" * 32,
        runtime_binding_fields=_test_runtime_binding_fields(),
    )
    transaction = executor.FormalCeremonyTransactionV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        public_evidence_path=public / "evidence.json",
        public_promotion_path=public / "promotion.json",
        run_id=b"r" * 16,
        ledger_id=b"l" * 16,
        prestage_intent_fields=intent,
        fault_injector=fault_injector,
    )
    transaction._test_actor_trust = actor_trust
    return transaction


def _test_runtime_binding_fields() -> dict[str, object]:
    return {
        "m3_execution_implementation_binding_roots": {
            "python_implementation_binding_root": b"p" * 32,
            "rust_implementation_binding_root": b"r" * 32,
        },
        "formal_rust_replay_binary_path": "/test/formal-rust-replay",
        "formal_rust_replay_binary_sha256": b"f" * 32,
        "rust_bridge_dag_replay_binary_path": "/test/bridge-rust-replay",
        "rust_bridge_dag_replay_binary_sha256": b"b" * 32,
        "rust_bridge_dag_qualification_report_sha256": b"q" * 32,
        "actor_profile_sha256": b"a" * 32,
    }


@pytest.mark.parametrize("colliding_trust", (b"r" * 16, b"l" * 16))
def test_prestage_intent_rejects_run_ledger_trust_identity_collision(
    colliding_trust: bytes,
) -> None:
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.build_prestage_intent_fields_v1(
            basis_commit="12" * 20,
            run_id=b"r" * 16,
            ledger_id=b"l" * 16,
            created_at_unix_seconds=7,
            trust_genesis_id=colliding_trust,
            actor_qualification_report={},
            errata_qualification_report={},
            rust_bridge_dag_qualification_report_sha256=b"q" * 32,
            live_actor_protocol_qualification_bundle_content_id=b"v" * 32,
            qualification_only_key_ids={
                purpose: bytes([purpose]) * 16 for purpose in (1, 2, 3, 4)
            },
            live_actor_protocol_qualification_bundle={"synthetic_test_bundle": True},
            live_actor_protocol_qualification_canonical_bundle_bytes=(
                executor._canonical_json({"synthetic_test_bundle": True})
            ),
            live_actor_protocol_daemon_receipt_binding=b"d" * 32,
            runtime_binding_fields=_test_runtime_binding_fields(),
        )
    assert captured.value.code == executor.FAIL_PREFLIGHT


def _test_actor_trust() -> executor.ActorPublicKeysV1:
    return executor.build_actor_trust_v1(
        public_keys={
            1: b"p" * 32,
            2: b"2" * 32,
            3: b"3" * 32,
            4: b"4" * 32,
        },
        timestamp=7,
        basis_commit="12" * 20,
        trust_genesis_id=b"t" * 16,
    )


def _reserve_test_transaction(
    transaction: executor.FormalCeremonyTransactionV1,
) -> None:
    transaction.reserve()
    transaction.persist_actor_trust_checkpoint_v1(transaction._test_actor_trust)


_RESERVATION_BOOTSTRAP_FAULT_POINTS = (
    "after_reservation_persistent_lock",
    *tuple(
        point
        for step in executor._RESERVATION_BOOTSTRAP_ORDER
        for point in (f"before_reservation_{step}", f"after_reservation_{step}")
    ),
)

_BOOTSTRAP_FILE_LABELS = (
    "persistent_lock",
    *tuple(
        step
        for step in executor._RESERVATION_BOOTSTRAP_ORDER
        if step != "stage_directory"
    ),
)
_BOOTSTRAP_ATOMIC_FAULT_POINTS = tuple(
    f"{point}_{label}"
    for label in _BOOTSTRAP_FILE_LABELS
    for point in (
        "after_bootstrap_next_inode",
        "after_bootstrap_partial_write",
        "after_bootstrap_next_fsync",
        "after_bootstrap_link",
        "after_bootstrap_next_unlink",
    )
)


@pytest.mark.parametrize("fault_point", _RESERVATION_BOOTSTRAP_FAULT_POINTS)
def test_reservation_bootstrap_exact_prefix_is_fresh_process_resumable(
    tmp_path: Path, fault_point: str,
) -> None:
    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError(point)

    transaction = _transaction(tmp_path, fail)
    with pytest.raises(RuntimeError, match=fault_point):
        transaction.reserve()
    assert transaction._lock_descriptor is None
    existing = {
        path: path.read_bytes()
        for root in (tmp_path / "custody", tmp_path / "public")
        for path in root.rglob("*")
        if path.is_file()
    }
    recovered = executor.FormalCeremonyTransactionV1.rehydrate_reservation_bootstrap_v1(
        custody_directory=tmp_path / "custody",
        public_evidence_path=transaction.public_evidence_path,
        public_promotion_path=transaction.public_promotion_path,
    )
    try:
        assert recovered.state == "RESERVED"
        assert recovered._validate_reservation_exact_prefix_v1() == len(
            executor._RESERVATION_BOOTSTRAP_ORDER
        )
        assert all(path.read_bytes() == payload for path, payload in existing.items())
    finally:
        recovered.close_lock()


def test_reservation_lock_is_complete_immutable_recovery_plan(tmp_path: Path) -> None:
    transaction = _transaction(
        tmp_path,
        lambda point: (_ for _ in ()).throw(RuntimeError(point))
        if point == "after_reservation_persistent_lock"
        else None,
    )
    with pytest.raises(RuntimeError):
        transaction.reserve()
    lock = json.loads(
        (tmp_path / "custody/phase3_m25_ceremony.lock").read_text(encoding="ascii")
    )
    assert lock["reservation_bootstrap_state"] == "RESERVING_EXACT_PREFIX"
    assert lock["prestage_intent_transport_or_null"] is not None
    specs = lock["ordered_reservation_artifact_specs_or_null"]
    assert tuple(row["step"] for row in specs) == executor._RESERVATION_BOOTSTRAP_ORDER
    assert all(
        row["payload_transport"] is not None and len(row["payload_sha256"]) == 64
        for row in specs
        if row["inode_kind"] == "regular_file"
    )


def test_reservation_bootstrap_rejects_nonprefix_gap(tmp_path: Path) -> None:
    transaction = _transaction(
        tmp_path,
        lambda point: (_ for _ in ()).throw(RuntimeError(point))
        if point == "after_reservation_persistent_lock"
        else None,
    )
    with pytest.raises(RuntimeError):
        transaction.reserve()
    _first, later = transaction._reservation_step_materials_v1()[:2]
    _step, _kind, path, mode, payload = later
    assert payload is not None
    path.write_bytes(payload)
    path.chmod(mode)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.FormalCeremonyTransactionV1.rehydrate_reservation_bootstrap_v1(
            custody_directory=tmp_path / "custody",
            public_evidence_path=transaction.public_evidence_path,
            public_promotion_path=transaction.public_promotion_path,
        )
    assert captured.value.code == executor.FAIL_TRANSACTION_LOCK


@pytest.mark.parametrize("fault_point", _BOOTSTRAP_ATOMIC_FAULT_POINTS)
def test_reservation_bootstrap_recovers_every_atomic_install_crash_shape(
    tmp_path: Path, fault_point: str,
) -> None:
    def fail(point: str) -> None:
        if point == fault_point:
            raise RuntimeError(point)

    transaction = _transaction(tmp_path, fail)
    with pytest.raises(RuntimeError, match=fault_point):
        transaction.reserve()
    assert transaction._lock_descriptor is None
    lock = tmp_path / "custody/phase3_m25_ceremony.lock"
    if lock.is_file():
        recovered = (
            executor.FormalCeremonyTransactionV1.rehydrate_reservation_bootstrap_v1(
                custody_directory=tmp_path / "custody",
                public_evidence_path=transaction.public_evidence_path,
                public_promotion_path=transaction.public_promotion_path,
            )
        )
    else:
        # The sole lock.next is explicitly precommit. The same immutable
        # caller plan can repair it without selecting any new identity.
        transaction._fault_injector = None
        transaction.reserve()
        recovered = transaction
    try:
        assert recovered.state == "RESERVED"
        assert not any(
            path.name.endswith(".next")
            for root in (tmp_path / "custody", tmp_path / "public")
            for path in root.rglob("*")
        )
    finally:
        recovered.close_lock()


def test_reservation_bootstrap_repairs_kernel_short_write_precommit_inode(
    tmp_path: Path, monkeypatch,
) -> None:
    transaction = _transaction(tmp_path)
    real_write = os.write
    calls = 0

    def short_write(descriptor: int, payload: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            return 0
        return real_write(descriptor, payload)

    monkeypatch.setattr(os, "write", short_write)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        transaction.reserve()
    assert captured.value.code == executor.FAIL_TRANSACTION_LOCK
    assert not (tmp_path / "custody/phase3_m25_ceremony.lock").exists()
    assert (tmp_path / "custody/phase3_m25_ceremony.lock.next").exists()
    monkeypatch.setattr(os, "write", real_write)
    transaction.reserve()
    try:
        assert transaction.state == "RESERVED"
        assert not (tmp_path / "custody/phase3_m25_ceremony.lock.next").exists()
    finally:
        transaction.close_lock()


def test_reservation_bootstrap_refsyncs_exact_next_after_fsync_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    transaction = _transaction(tmp_path)
    real_fsync = os.fsync
    failed = False

    def fail_first_regular_fsync(descriptor: int) -> None:
        nonlocal failed
        if not failed and stat.S_ISREG(os.fstat(descriptor).st_mode):
            failed = True
            raise OSError("injected fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_regular_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        transaction.reserve()
    monkeypatch.setattr(os, "fsync", real_fsync)
    transaction.reserve()
    try:
        assert transaction.state == "RESERVED"
    finally:
        transaction.close_lock()


def test_transaction_rejects_caller_symlink_alias_before_bootstrap_write(
    tmp_path: Path,
) -> None:
    real_custody = tmp_path / "real-custody"
    real_custody.mkdir(mode=0o700)
    alias = tmp_path / "custody-alias"
    alias.symlink_to(real_custody, target_is_directory=True)
    public = tmp_path / "public"
    public.mkdir()
    intent = executor.build_prestage_intent_fields_v1(
        basis_commit="12" * 20,
        run_id=b"r" * 16,
        ledger_id=b"l" * 16,
        created_at_unix_seconds=7,
        trust_genesis_id=b"t" * 16,
        actor_qualification_report={},
        errata_qualification_report={},
        rust_bridge_dag_qualification_report_sha256=b"q" * 32,
        live_actor_protocol_qualification_bundle_content_id=b"v" * 32,
        qualification_only_key_ids={
            purpose: bytes([purpose]) * 16 for purpose in (1, 2, 3, 4)
        },
        live_actor_protocol_qualification_bundle={"synthetic_test_bundle": True},
        live_actor_protocol_qualification_canonical_bundle_bytes=(
            executor._canonical_json({"synthetic_test_bundle": True})
        ),
        live_actor_protocol_daemon_receipt_binding=b"d" * 32,
        runtime_binding_fields=_test_runtime_binding_fields(),
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.FormalCeremonyTransactionV1(
            basis_commit="12" * 20,
            custody_directory=alias,
            public_evidence_path=public / "evidence.json",
            public_promotion_path=public / "promotion.json",
            run_id=b"r" * 16,
            ledger_id=b"l" * 16,
            prestage_intent_fields=intent,
        )
    assert captured.value.code == executor.FAIL_PREFLIGHT
    assert list(real_custody.iterdir()) == []


def test_transaction_local_qualification_bundle_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    transaction = _transaction(tmp_path)
    transaction.reserve()
    transaction.close_lock()
    bundle_path = (
        transaction.public_evidence_path.parent
        / (".hegel-m25-stage-" + transaction.run_id.hex())
        / executor._LIVE_QUALIFICATION_BUNDLE_FILENAME
    )
    bundle_path.write_bytes(executor._canonical_json({"synthetic_test_bundle": False}))
    bundle_path.chmod(0o600)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.FormalCeremonyTransactionV1.rehydrate_reservation_bootstrap_v1(
            custody_directory=transaction.custody_directory,
            public_evidence_path=transaction.public_evidence_path,
            public_promotion_path=transaction.public_promotion_path,
        )
    assert captured.value.code == executor.FAIL_TRANSACTION_LOCK


def _reserved_prestage_transaction(
    tmp_path: Path, *, fault_injector=None
) -> tuple[executor.FormalCeremonyTransactionV1, executor.ActorPublicKeysV1]:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    os.chmod(custody, 0o700)
    public = tmp_path / "public"
    public.mkdir()
    actor_trust = executor.build_actor_trust_v1(
        public_keys={purpose: bytes([purpose]) * 32 for purpose in (1, 2, 3, 4)},
        timestamp=7,
        basis_commit="12" * 20,
        trust_genesis_id=b"t" * 16,
    )
    intent = executor.build_prestage_intent_fields_v1(
        basis_commit="12" * 20,
        run_id=b"r" * 16,
        ledger_id=b"l" * 16,
        created_at_unix_seconds=7,
        trust_genesis_id=b"t" * 16,
        actor_qualification_report={},
        errata_qualification_report={},
        rust_bridge_dag_qualification_report_sha256=b"q" * 32,
        live_actor_protocol_qualification_bundle_content_id=b"v" * 32,
        qualification_only_key_ids={
            purpose: bytes([purpose]) * 16 for purpose in (1, 2, 3, 4)
        },
        live_actor_protocol_qualification_bundle={"synthetic_test_bundle": True},
        live_actor_protocol_qualification_canonical_bundle_bytes=(
            executor._canonical_json({"synthetic_test_bundle": True})
        ),
        live_actor_protocol_daemon_receipt_binding=b"d" * 32,
        runtime_binding_fields=_test_runtime_binding_fields(),
    )
    transaction = executor.FormalCeremonyTransactionV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        public_evidence_path=public / "evidence.json",
        public_promotion_path=public / "promotion.json",
        run_id=b"r" * 16,
        ledger_id=b"l" * 16,
        prestage_intent_fields=intent,
        fault_injector=fault_injector,
    )
    transaction.reserve()
    transaction.persist_actor_trust_checkpoint_v1(actor_trust)
    return transaction, actor_trust


_RECOVERY_PUBLIC_KEY = b"p" * 32
_RECOVERY_KEY_ID = hashlib.sha256(_RECOVERY_PUBLIC_KEY).digest()[:16]
_RECOVERY_SEED = b"z" * 32
_RECOVERY_SEED_COMMITMENT = hashlib.sha256(
    b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1\x00" + _RECOVERY_SEED
).digest()
_RECOVERY_SPLIT_FRAME = encode_split_calculator_public_frame_v2(
    SplitCalculatorPublicResponseV2(
        _RECOVERY_SEED_COMMITMENT,
        tuple(
            SplitRootCommitment(role, partition, count, bytes([index + 1]) * 32)
            for index, (role, partition, count) in enumerate(SPLIT_RESPONSE_ROWS)
        ),
    )
)


def _seed_custody_verification_receipt(
    commitment: bytes = _RECOVERY_SEED_COMMITMENT,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": executor.SEED_CUSTODY_VERIFICATION_SCHEMA,
        "verified": True,
        "seed_commitment_hex": commitment.hex(),
        "seed_length_bytes": 32,
        "seed_intent_sha256": "1" * 64,
        "completion_receipt_sha256": "2" * 64,
        "raw_seed_read_inside_keyless_verifier": True,
        "raw_seed_exported": False,
        "private_key_mount_present": False,
        "state_mount_present": False,
        "verifier_numeric_uid": os.geteuid(),
        "verifier_numeric_gid": os.getegid(),
        "custody_artifacts_owned_by_verifier_identity": True,
        "inner_receipt_sha256": "3" * 64,
        "verifier_tool_sha256": "4" * 64,
        "docker_command_argv_sha256": "5" * 64,
        "docker_command_policy_sha256": "9" * 64,
        "docker_image_ref": "test-custodian@sha256:" + "6" * 64,
        "docker_seccomp_sha256": "7" * 64,
        "docker_daemon_receipt_sha256": (b"d" * 32).hex(),
        "docker_control_plane_binding_sha256": "8" * 64,
        "docker_network_mode": "none",
        "docker_ipc_mode": "private",
        "docker_read_only_rootfs": True,
        "docker_stdout_limit_bytes": 8192,
        "docker_timeout_seconds": 120,
        "custody_owner_policy_id": "EXACT_UNIFORM_CURRENT_OWNER_HOST_OR_65534_V1",
        "incarnation_fields_nonidentity": True,
    }
    body["receipt_sha256"] = hashlib.sha256(
        executor._canonical_json(body)
    ).hexdigest()
    return body


def _seed_custody_inner_receipt(
    commitment: bytes,
    *,
    uid: int,
    gid: int,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": executor.SEED_CUSTODY_INNER_VERIFICATION_SCHEMA,
        "verified": True,
        "seed_commitment_hex": commitment.hex(),
        "seed_length_bytes": 32,
        "seed_intent_sha256": "1" * 64,
        "completion_receipt_sha256": "2" * 64,
        "raw_seed_read_inside_keyless_verifier": True,
        "raw_seed_exported": False,
        "private_key_mount_present": False,
        "state_mount_present": False,
        "verifier_numeric_uid": uid,
        "verifier_numeric_gid": gid,
        "custody_artifacts_owned_by_verifier_identity": True,
    }
    body["receipt_sha256"] = hashlib.sha256(
        executor._canonical_json(body)
    ).hexdigest()
    return body


def _verify_staged_seed_for_test(
    transaction: executor.FormalCeremonyTransactionV1,
) -> None:
    commitment = transaction._staged_seed_commitment
    assert type(commitment) is bytes and len(commitment) == 32
    transaction.record_seed_custody_verification_v1(
        _seed_custody_verification_receipt(commitment)
    )


def _post_stage_inputs() -> GateEvidenceInputsV1:
    run_id = b"r" * 16
    ledger_id = b"l" * 16
    actor_trust = _test_actor_trust()
    base = _dummy_gate_inputs()
    return replace(
        base,
        basis_commit="12" * 20,
        marker_snapshot=MarkerSnapshot(
            "COMPLETE", executor.SPLIT_VERSION_DIGEST, b"m" * 32,
            _RECOVERY_KEY_ID, 7,
        ),
        split_seed_commitment_fields={
            "split_seed_commitment_digest": _RECOVERY_SEED_COMMITMENT,
        },
        actor_key_manifests=actor_trust.manifests,
        replacement_policy_fields=actor_trust.replacement_policy_fields,
        trust_genesis_fields=actor_trust.trust_genesis_fields,
        ledger_genesis_fields={"ledger_id": ledger_id},
        python_split_frame=_RECOVERY_SPLIT_FRAME,
        rust_split_frame=_RECOVERY_SPLIT_FRAME,
        opaque_registration_intents=(
            {"opaque_id_kind_id": 1, "opaque_id_16_bytes": run_id},
            {"opaque_id_kind_id": 2, "opaque_id_16_bytes": ledger_id},
        ),
        opaque_registry_records=(
            {"opaque_id_kind_id": 1, "opaque_id_16_bytes": run_id},
            {"opaque_id_kind_id": 2, "opaque_id_16_bytes": ledger_id},
        ),
        execution_candidate_fields={"run_id": run_id},
        bridge_statement_fields={"run_id": run_id},
        execution_manifest_fields={"run_id": run_id},
        run_genesis_fields={"run_id": run_id},
    )


def _write_post_stage_seed_state(custody: Path) -> None:
    intent = executor._canonical_json({
        "schema": "hegel-phase3-m25-seed-generation-intent/1",
        "state": "CSPRNG_CALL_COMMITTED_NO_REDRAW",
    })
    completion = executor._canonical_json({
        "attempt": 1,
        "intent_sha256": hashlib.sha256(intent).hexdigest(),
        "schema": "hegel-phase3-m25-seed-generation-complete/1",
        "seed_commitment_hex": _RECOVERY_SEED_COMMITMENT.hex(),
        "seed_length_bytes": 32,
    })
    for name, payload in (
        ("split_seed_generation.intent", intent),
        ("split_master_seed.bin", _RECOVERY_SEED),
        ("split_seed_generation.complete", completion),
    ):
        path = custody / name
        path.write_bytes(payload)
        path.chmod(0o600)


def _load_seed_custody_verifier_module():
    spec = importlib.util.spec_from_file_location(
        "hegel_test_seed_custody_verifier",
        executor.SEED_CUSTODY_VERIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_seed_custody_verifier_module(module, custody: Path, state: Path) -> None:
    module.CUSTODY = custody
    module.STATE_MOUNT = state
    module.SEED_INTENT = custody / "split_seed_generation.intent"
    module.SEED_FILE = custody / "split_master_seed.bin"
    module.SEED_COMPLETE = custody / "split_seed_generation.complete"


def test_keyless_seed_custody_verifier_accepts_exact_local_fixture(
    tmp_path: Path, monkeypatch, capfd,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    _write_post_stage_seed_state(custody)
    module = _load_seed_custody_verifier_module()
    _configure_seed_custody_verifier_module(module, custody, tmp_path / "absent-state")
    monkeypatch.setenv(
        "HEGEL_EXPECTED_SEED_COMMITMENT_HEX", _RECOVERY_SEED_COMMITMENT.hex()
    )
    monkeypatch.setenv("HEGEL_VERIFIER_NUMERIC_UID", str(os.geteuid()))
    monkeypatch.setenv("HEGEL_VERIFIER_NUMERIC_GID", str(os.getegid()))
    assert module.main() == 0
    output = capfd.readouterr()
    receipt = json.loads(output.out)
    assert receipt["verified"] is True
    assert receipt["seed_commitment_hex"] == _RECOVERY_SEED_COMMITMENT.hex()
    assert receipt["raw_seed_exported"] is False
    assert "7a" * 32 not in output.out


@pytest.mark.parametrize(
    "mutation",
    (
        "seed_byte",
        "completion",
        "intent",
        "mode",
        "symlink",
        "wrong_length",
        "state_mount",
    ),
)
def test_keyless_seed_custody_verifier_rejects_every_raw_state_drift(
    tmp_path: Path, monkeypatch, capfd, mutation: str,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    _write_post_stage_seed_state(custody)
    state = tmp_path / "absent-state"
    seed_path = custody / "split_master_seed.bin"
    if mutation == "seed_byte":
        changed = bytearray(seed_path.read_bytes())
        changed[0] ^= 1
        seed_path.write_bytes(bytes(changed))
    elif mutation == "completion":
        path = custody / "split_seed_generation.complete"
        path.write_bytes(path.read_bytes().replace(b'"attempt":1', b'"attempt":2'))
    elif mutation == "intent":
        path = custody / "split_seed_generation.intent"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "mode":
        seed_path.chmod(0o640)
    elif mutation == "symlink":
        target = tmp_path / "seed-target"
        target.write_bytes(_RECOVERY_SEED)
        target.chmod(0o600)
        seed_path.unlink()
        seed_path.symlink_to(target)
    elif mutation == "wrong_length":
        seed_path.write_bytes(_RECOVERY_SEED[:-1])
    else:
        state.mkdir()
    module = _load_seed_custody_verifier_module()
    _configure_seed_custody_verifier_module(module, custody, state)
    monkeypatch.setenv(
        "HEGEL_EXPECTED_SEED_COMMITMENT_HEX", _RECOVERY_SEED_COMMITMENT.hex()
    )
    monkeypatch.setenv("HEGEL_VERIFIER_NUMERIC_UID", str(os.geteuid()))
    monkeypatch.setenv("HEGEL_VERIFIER_NUMERIC_GID", str(os.getegid()))
    with pytest.raises(SystemExit) as captured:
        module.main()
    assert captured.value.code == 70
    output = capfd.readouterr()
    assert output.out == ""
    assert output.err == "FAIL_M25_KEYLESS_SEED_CUSTODY_VERIFICATION\n"


@pytest.mark.parametrize("owner_kind", ("host", "nobody"))
def test_docker_keyless_verifier_runs_as_exact_current_custody_owner(
    tmp_path: Path, monkeypatch, owner_kind: str,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    _write_post_stage_seed_state(custody)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._profile = {
        "images": {"custodian": "test-custodian@sha256:" + "6" * 64}
    }
    _install_fake_docker_boundary(backend, tmp_path)
    verifier_source = executor.SEED_CUSTODY_VERIFIER_PATH.read_bytes()
    monkeypatch.setattr(backend, "_git_blob", lambda _relative: verifier_source)
    uid, gid = (
        (os.geteuid(), os.getegid()) if owner_kind == "host" else (65534, 65534)
    )
    if owner_kind == "nobody":
        original_lstat = Path.lstat
        ownership_paths = {
            custody,
            custody / "split_seed_generation.intent",
            custody / "split_master_seed.bin",
            custody / "split_seed_generation.complete",
        }

        def fake_lstat(path: Path):
            metadata = original_lstat(path)
            if path in ownership_paths:
                values = list(metadata)
                values[4] = uid
                values[5] = gid
                return os.stat_result(values)
            return metadata

        monkeypatch.setattr(Path, "lstat", fake_lstat)
    observed_command: tuple[str, ...] | None = None

    def fake_run(command, **_kwargs):
        nonlocal observed_command
        observed_command = tuple(command)
        inner = _seed_custody_inner_receipt(
            _RECOVERY_SEED_COMMITMENT, uid=uid, gid=gid
        )
        return SimpleNamespace(
            returncode=0,
            stdout=executor._canonical_json(inner),
            stderr=b"",
        )

    monkeypatch.setattr(executor, "_run", fake_run)
    receipt = backend.verify_seed_custody_commitment_v1(
        _RECOVERY_SEED_COMMITMENT
    )
    assert receipt["verifier_numeric_uid"] == uid
    assert receipt["verifier_numeric_gid"] == gid
    assert observed_command is not None
    assert f"--user={uid}:{gid}" in observed_command
    assert "--ipc=private" in observed_command
    assert "--ulimit=nofile=32:32" in observed_command
    assert "--network=none" in observed_command
    assert not any("dst=/state" in value for value in observed_command)


def test_executor_git_blob_uses_absolute_sanitized_offline_read(
    tmp_path: Path, monkeypatch,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    for key, value in {
        "PATH": "/tmp/hostile-path",
        "HOME": "/tmp/hostile-home",
        "GIT_DIR": "/tmp/hostile-git-dir",
        "GIT_WORK_TREE": "/tmp/hostile-work-tree",
        "GIT_OBJECT_DIRECTORY": "/tmp/hostile-objects",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/tmp/hostile-alternates",
        "GIT_REPLACE_REF_BASE": "refs/hostile/",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.pager",
        "GIT_CONFIG_VALUE_0": "/tmp/hostile-pager",
        "GIT_NO_REPLACE_OBJECTS": "0",
        "GIT_NO_LAZY_FETCH": "0",
    }.items():
        monkeypatch.setenv(key, value)
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = tuple(command)
        observed["environment"] = dict(kwargs["environment"])
        observed["working_directory"] = kwargs["working_directory"]
        return SimpleNamespace(returncode=0, stdout=b"committed-blob", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    assert backend._git_blob("Hegel Machine/example.txt") == b"committed-blob"
    assert observed["command"] == (
        "/usr/bin/git",
        "show",
        f"{'12' * 20}:Hegel Machine/example.txt",
    )
    assert observed["environment"] == executor.formal_git_environment_v1()
    assert "GIT_DIR" not in observed["environment"]
    assert "GIT_OBJECT_DIRECTORY" not in observed["environment"]
    assert observed["environment"]["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert observed["environment"]["GIT_NO_LAZY_FETCH"] == "1"
    assert observed["working_directory"] == executor.REPOSITORY_ROOT


def test_executor_git_blob_is_repository_bound_from_hostile_other_repo(
    tmp_path: Path, monkeypatch,
) -> None:
    commit = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=executor.REPOSITORY_ROOT,
        env=executor.formal_git_environment_v1(),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    relative = "Hegel Machine/README.md"
    expected = subprocess.run(
        ["/usr/bin/git", "show", f"{commit}:{relative}"],
        cwd=executor.REPOSITORY_ROOT,
        env=executor.formal_git_environment_v1(),
        check=True,
        capture_output=True,
    ).stdout

    hostile = tmp_path / "hostile-other-repository"
    hostile.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q"], cwd=hostile, check=True)
    subprocess.run(
        ["/usr/bin/git", "config", "user.email", "hostile@example.invalid"],
        cwd=hostile,
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "config", "user.name", "hostile-repository"],
        cwd=hostile,
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "fetch", "--no-tags", str(executor.REPOSITORY_ROOT), commit],
        cwd=hostile,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["/usr/bin/git", "commit", "--allow-empty", "-qm", "empty replacement"],
        cwd=hostile,
        check=True,
    )
    replacement = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=hostile,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    subprocess.run(
        ["/usr/bin/git", "replace", commit, replacement],
        cwd=hostile,
        check=True,
    )
    hostile_read = subprocess.run(
        ["/usr/bin/git", "show", f"{commit}:{relative}"],
        cwd=hostile,
        check=False,
        capture_output=True,
    )
    assert hostile_read.returncode != 0

    monkeypatch.chdir(hostile)
    for key, value in {
        "GIT_DIR": str(hostile / ".git"),
        "GIT_WORK_TREE": str(hostile),
        "GIT_REPLACE_REF_BASE": "refs/replace/",
        "GIT_NO_REPLACE_OBJECTS": "0",
        "GIT_NO_LAZY_FETCH": "0",
        "PATH": str(tmp_path / "hostile-bin"),
    }.items():
        monkeypatch.setenv(key, value)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit=commit,
        custody_directory=tmp_path / "unused-custody",
        rust_formal_replay_binary=tmp_path / "unused-rust-replay",
        timestamp=1,
    )
    assert backend._git_blob(relative) == expected


def test_executor_git_blob_ignores_replace_ref_in_bound_repository(
    tmp_path: Path, monkeypatch,
) -> None:
    repository = tmp_path / "bound-repository-with-replace"
    repository.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["/usr/bin/git", "config", "user.email", "replace@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "config", "user.name", "bound-replace-test"],
        cwd=repository,
        check=True,
    )
    payload = repository / "payload.txt"
    payload.write_bytes(b"honest-original\n")
    subprocess.run(["/usr/bin/git", "add", "payload.txt"], cwd=repository, check=True)
    subprocess.run(["/usr/bin/git", "commit", "-qm", "original"], cwd=repository, check=True)
    original = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    payload.write_bytes(b"hostile-replacement\n")
    subprocess.run(
        ["/usr/bin/git", "commit", "-qam", "replacement"],
        cwd=repository,
        check=True,
    )
    replacement = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    subprocess.run(
        ["/usr/bin/git", "replace", original, replacement],
        cwd=repository,
        check=True,
    )
    ambient = subprocess.run(
        ["/usr/bin/git", "show", f"{original}:payload.txt"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    assert ambient == b"hostile-replacement\n"
    monkeypatch.setattr(executor, "REPOSITORY_ROOT", repository)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit=original,
        custody_directory=tmp_path / "unused-custody",
        rust_formal_replay_binary=tmp_path / "unused-rust-replay",
        timestamp=1,
    )
    assert backend._git_blob("payload.txt") == b"honest-original\n"


def _durable_post_stage_transaction(
    tmp_path: Path, *, fault_injector=None
) -> tuple[executor.FormalCeremonyTransactionV1, GateEvidenceInputsV1]:
    transaction = _transaction(tmp_path, fault_injector)
    _reserve_test_transaction(transaction)
    inputs = _post_stage_inputs()
    executor.create_pending_marker_v1(
        secret_state_directory=transaction.custody_directory,
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=_RECOVERY_KEY_ID,
        created_at_unix_seconds=7,
    )
    _write_post_stage_seed_state(transaction.custody_directory)
    payload = executor.serialize_gate_evidence_inputs_v1(inputs)
    transaction.stage_and_prospectively_replay(
        payload, _public_replay_stub(payload), replay=_public_replay_stub
    )
    _verify_staged_seed_for_test(transaction)
    return transaction, inputs


class _PostStageRecoveryActors(executor.CeremonyActorsV1):
    authoritative = True

    def __init__(self, custody: Path, *, private_state_present: bool) -> None:
        self.custody = custody
        self.private_state_present = private_state_present
        self.prepared_run_id = None
        self.started = False
        self.resume_count = 0
        self.complete_cleanup_count = 0

    def validate_frozen_daemon_receipt_binding_v1(self, expected: bytes) -> None:
        assert expected == b"d" * 32

    def prepare_post_stage_pending_recovery(self, run_id: bytes) -> None:
        self.prepared_run_id = run_id

    def start(self):
        assert self.prepared_run_id == b"r" * 16
        assert self.private_state_present is True
        self.started = True
        return self

    def keygen(self, purpose: int) -> bytes:
        assert purpose == 1 and self.started
        return _RECOVERY_PUBLIC_KEY

    def resume_post_stage_seed_split(self) -> tuple[bytes, bytes]:
        assert self.started
        self.resume_count += 1
        return _RECOVERY_SPLIT_FRAME, _RECOVERY_SPLIT_FRAME

    def verify_seed_custody_commitment_v1(
        self, expected_commitment: bytes
    ) -> dict[str, object]:
        seed = (self.custody / "split_master_seed.bin").read_bytes()
        actual_commitment = hashlib.sha256(
            b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1\x00" + seed
        ).digest()
        if actual_commitment != expected_commitment:
            raise executor.FormalContainerExecutorError(
                executor.FAIL_CUSTODY,
                "synthetic keyless verifier rejected retained seed",
            )
        return _seed_custody_verification_receipt(expected_commitment)

    def complete_marker(self, seed_manifest_root: bytes) -> MarkerSnapshot:
        return executor.complete_marker_v1(
            marker_path=self.custody / "split_seed_instantiation.marker",
            seed_commitment_manifest_root=seed_manifest_root,
        )

    def authorize_private_state_destruction(self, marker: MarkerSnapshot) -> None:
        assert marker.state == "COMPLETE"

    def destroy_private_state_and_verify_absent(self) -> None:
        self.private_state_present = False
        self.started = False

    def stop_for_recovery_and_verify_absent(self) -> None:
        self.started = False

    def recover_complete_private_state_and_verify_absent(
        self, run_id: bytes, marker: MarkerSnapshot
    ) -> None:
        assert run_id == b"r" * 16 and marker.state == "COMPLETE"
        self.complete_cleanup_count += 1
        self.private_state_present = False


def test_transaction_forbids_complete_before_durable_prospective_replay(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    expected = MarkerSnapshot("COMPLETE", b"s" * 32, b"m" * 32, b"k" * 16, 1)
    _reserve_test_transaction(transaction)
    try:
        with pytest.raises(executor.FormalContainerExecutorError) as captured:
            transaction.record_marker_complete(expected, expected)
        assert captured.value.code == executor.FAIL_CUSTODY
        replay_payload = executor.serialize_gate_evidence_inputs_v1(_dummy_gate_inputs())
        transaction.stage_and_prospectively_replay(
            replay_payload, _public_replay_stub(replay_payload), replay=_public_replay_stub
        )
        with pytest.raises(executor.FormalContainerExecutorError) as captured:
            transaction.record_marker_complete(expected, expected)
        assert captured.value.code == executor.FAIL_CUSTODY
        _verify_staged_seed_for_test(transaction)
        transaction.record_marker_complete(expected, expected)
        assert transaction.state == "MARKER_COMPLETE"
        journal = json.loads(transaction.journal_path.read_text(encoding="ascii"))
        assert journal["marker_complete"] is True
        assert journal["public_outputs_complete"] is False
    finally:
        transaction.close_lock()


_SEED_VERIFICATION_FAULT_POINTS = (
    "before_stage_next_write_seed_custody_verification",
    "after_stage_next_fsync_seed_custody_verification",
    "after_stage_rename_before_dir_fsync_seed_custody_verification",
    "after_stage_dir_fsync_seed_custody_verification",
    "after_seed_custody_verification_receipt_durable",
    "before_transition_next_write_publication_seed_verifier_binding",
    "after_transition_next_fsync_publication_seed_verifier_binding",
    "after_transition_rename_publication_seed_verifier_binding",
    "after_transition_dir_fsync_publication_seed_verifier_binding",
    "after_journal_next_fsync",
    "after_seed_custody_verification",
)


@pytest.mark.parametrize("fault_point", _SEED_VERIFICATION_FAULT_POINTS)
def test_seed_verification_receipt_and_publication_binding_transition_is_resumable(
    tmp_path: Path, fault_point: str,
) -> None:
    raised = False
    armed = False

    def inject(point: str) -> None:
        nonlocal raised
        if armed and point == fault_point and not raised:
            raised = True
            raise RuntimeError(point)

    transaction = _transaction(tmp_path, inject)
    _reserve_test_transaction(transaction)
    replay_payload = executor.serialize_gate_evidence_inputs_v1(_dummy_gate_inputs())
    transaction.stage_and_prospectively_replay(
        replay_payload,
        _public_replay_stub(replay_payload),
        replay=_public_replay_stub,
    )
    armed = True
    with pytest.raises(RuntimeError, match=fault_point):
        _verify_staged_seed_for_test(transaction)
    assert raised is True
    _verify_staged_seed_for_test(transaction)
    assert transaction.state == "SEED_CUSTODY_VERIFIED"
    assert transaction.journal_path is not None
    journal = json.loads(transaction.journal_path.read_bytes())
    assert journal["state"] == "SEED_CUSTODY_VERIFIED"
    verification_path = (
        transaction.journal_path.parent
        / executor._SEED_CUSTODY_VERIFICATION_FILENAME
    )
    verification_payload = verification_path.read_bytes()
    publication_receipt = json.loads(
        (transaction.journal_path.parent / "publication-receipt.json").read_bytes()
    )
    assert publication_receipt[
        "seed_custody_verification_receipt_sha256_or_null"
    ] == hashlib.sha256(verification_payload).hexdigest()
    assert not (verification_path.with_name(verification_path.name + ".next")).exists()
    assert not (
        transaction.journal_path.parent / "publication-receipt.json.next"
    ).exists()
    transaction.close_lock()


def test_fault_after_staging_leaves_pending_marker_and_recoverable_public_stage(
    tmp_path: Path,
) -> None:
    def fail(point: str) -> None:
        if point == "after_durable_staging":
            raise RuntimeError("injected-after-stage")

    transaction = _transaction(tmp_path, fail)
    _reserve_test_transaction(transaction)
    replay_payload = executor.serialize_gate_evidence_inputs_v1(_dummy_gate_inputs())
    try:
        with pytest.raises(RuntimeError, match="injected-after-stage"):
            transaction.stage_and_prospectively_replay(
                replay_payload, _public_replay_stub(replay_payload), replay=_public_replay_stub
            )
        journal = json.loads(transaction.journal_path.read_text(encoding="ascii"))
        assert journal["state"] == "STAGED_PROSPECTIVE_REPLAY_PASSED"
        assert journal["marker_complete"] is False
        assert (transaction.journal_path.parent / "public-evidence.json").is_file()
        assert not (tmp_path / "custody/split_seed_instantiation.marker").exists()
    finally:
        transaction.close_lock()


def test_publication_is_guarded_by_marker_and_verified_actor_absence(tmp_path: Path) -> None:
    transaction, actor_trust = _reserved_prestage_transaction(tmp_path)
    replay_payload = executor.serialize_gate_evidence_inputs_v1(_dummy_gate_inputs())
    expected = MarkerSnapshot("COMPLETE", b"s" * 32, b"m" * 32, b"k" * 16, 1)
    try:
        transaction.stage_and_prospectively_replay(
            replay_payload, _public_replay_stub(replay_payload), replay=_public_replay_stub
        )
        with pytest.raises(executor.FormalContainerExecutorError):
            transaction.publish(replay=_public_replay_stub)
        _verify_staged_seed_for_test(transaction)
        transaction.record_marker_complete(expected, expected)
        with pytest.raises(executor.FormalContainerExecutorError):
            transaction.publish(replay=_public_replay_stub)
        transaction.record_actors_absent()
        transaction.publish(replay=_public_replay_stub)
        assert transaction.state == "PUBLISHED"
        assert json.loads(transaction.public_evidence_path.read_text()) == replay_payload
        assert json.loads(transaction.public_promotion_path.read_text()) == _public_replay_stub(replay_payload)
        receipt = json.loads(transaction.publication_receipt_path.read_text())
        assert receipt["actor_cleanup_required_before_publication"] is True
    finally:
        transaction.close_lock()


def test_transaction_lock_and_exact_output_reservations_are_fail_closed(tmp_path: Path) -> None:
    transaction = _transaction(tmp_path)
    _reserve_test_transaction(transaction)
    try:
        assert (tmp_path / "custody/phase3_m25_ceremony.lock").stat().st_mode & 0o777 == 0o600
        assert len(list((tmp_path / "custody").glob("opaque-*-*.reserved"))) == 2
        assert not transaction.public_evidence_path.exists()
        reservation = transaction.public_evidence_path.with_name(
            ".evidence.json.hegel-reserved"
        )
        assert b"RESERVED_NOT_PUBLIC" in reservation.read_bytes()
        second = executor.FormalCeremonyTransactionV1(
            basis_commit="12" * 20,
            custody_directory=tmp_path / "custody",
            public_evidence_path=tmp_path / "public/evidence-2.json",
            public_promotion_path=tmp_path / "public/promotion-2.json",
            run_id=b"x" * 16,
            ledger_id=b"y" * 16,
        )
        with pytest.raises(executor.FormalContainerExecutorError) as captured:
            second.reserve()
        assert captured.value.code == executor.FAIL_TRANSACTION_LOCK
    finally:
        transaction.close_lock()


def test_explicit_pending_recovery_reopens_exact_persistent_transaction(
    tmp_path: Path,
) -> None:
    transaction, actor_trust = _reserved_prestage_transaction(tmp_path)
    executor.create_pending_marker_v1(
        secret_state_directory=tmp_path / "custody",
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=actor_trust.key_ids[1],
        created_at_unix_seconds=7,
    )
    transaction.close_lock()
    with executor.acquire_pending_ceremony_recovery_v1(
        custody_directory=tmp_path / "custody",
        public_evidence_path=transaction.public_evidence_path,
        public_promotion_path=transaction.public_promotion_path,
    ) as recovery:
        assert recovery.basis_commit == "12" * 20
        assert recovery.run_id == b"r" * 16
        assert recovery.ledger_id == b"l" * 16
        assert recovery.marker_snapshot.state == "PENDING"
        assert recovery.journal_state == "RESERVED"
        with pytest.raises(executor.FormalContainerExecutorError) as captured:
            executor.acquire_pending_ceremony_recovery_v1(
                custody_directory=tmp_path / "custody",
                public_evidence_path=transaction.public_evidence_path,
                public_promotion_path=transaction.public_promotion_path,
            )
        assert captured.value.code == executor.FAIL_TRANSACTION_LOCK


def test_live_original_anchor_lock_forbids_concurrent_pending_reclaim(
    tmp_path: Path,
) -> None:
    transaction, actor_trust = _reserved_prestage_transaction(tmp_path)
    executor.create_pending_marker_v1(
        secret_state_directory=tmp_path / "custody",
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=actor_trust.key_ids[1],
        created_at_unix_seconds=7,
    )
    try:
        with pytest.raises(executor.FormalContainerExecutorError) as captured:
            executor.acquire_pending_ceremony_recovery_v1(
                custody_directory=tmp_path / "custody",
                public_evidence_path=transaction.public_evidence_path,
                public_promotion_path=transaction.public_promotion_path,
            )
        assert captured.value.code == executor.FAIL_TRANSACTION_LOCK
        assert "live-locked" in captured.value.detail
    finally:
        transaction.close_lock()


def test_anchor_routes_65534_owned_pending_tree_through_exact_docker_reclaimer(
    tmp_path: Path, monkeypatch,
) -> None:
    transaction, actor_trust = _reserved_prestage_transaction(tmp_path)
    executor.create_pending_marker_v1(
        secret_state_directory=tmp_path / "custody",
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=actor_trust.key_ids[1],
        created_at_unix_seconds=7,
    )
    transaction.close_lock()
    custody = transaction.custody_directory
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=7,
    )
    original_lstat = Path.lstat
    handed_off = True
    reclaim_calls = 0

    def fake_lstat(path: Path):
        metadata = original_lstat(path)
        if handed_off and path == custody:
            values = list(metadata)
            values[4] = 65534
            values[5] = 65534
            return os.stat_result(values)
        return metadata

    def fake_reclaim(anchor_fields):
        nonlocal handed_off, reclaim_calls
        assert anchor_fields["custody_st_ino"] == original_lstat(custody).st_ino
        reclaim_calls += 1
        handed_off = False
        return MappingProxyType({"raw_seed_bytes_read": False})

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    monkeypatch.setattr(backend, "reclaim_pending_custody_from_anchor_v1", fake_reclaim)
    with executor.acquire_pending_ceremony_recovery_v1(
        custody_directory=custody,
        public_evidence_path=transaction.public_evidence_path,
        public_promotion_path=transaction.public_promotion_path,
        actors=backend,
    ) as recovery:
        assert recovery.marker_snapshot.state == "PENDING"
        assert reclaim_calls == 1
        assert handed_off is False


@pytest.mark.parametrize("seed_prefix_length", (0, 1, 2, 3))
def test_anchor_reclaims_every_exact_pending_seed_creation_prefix(
    tmp_path: Path, monkeypatch, seed_prefix_length: int,
) -> None:
    transaction, actor_trust = _reserved_prestage_transaction(tmp_path)
    executor.create_pending_marker_v1(
        secret_state_directory=transaction.custody_directory,
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=actor_trust.key_ids[1],
        created_at_unix_seconds=7,
    )
    _write_post_stage_seed_state(transaction.custody_directory)
    seed_order = (
        "split_seed_generation.intent",
        "split_master_seed.bin",
        "split_seed_generation.complete",
    )
    for name in seed_order[seed_prefix_length:]:
        (transaction.custody_directory / name).unlink()
    transaction.close_lock()
    custody = transaction.custody_directory
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=7,
    )
    original_lstat = Path.lstat
    handed_off = True

    def fake_lstat(path: Path):
        metadata = original_lstat(path)
        if handed_off and path == custody:
            values = list(metadata)
            values[4] = 65534
            values[5] = 65534
            return os.stat_result(values)
        return metadata

    def fake_reclaim(_anchor_fields):
        nonlocal handed_off
        handed_off = False
        return MappingProxyType({"raw_seed_bytes_read": False})

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    monkeypatch.setattr(backend, "reclaim_pending_custody_from_anchor_v1", fake_reclaim)
    with executor.acquire_pending_ceremony_recovery_v1(
        custody_directory=custody,
        public_evidence_path=transaction.public_evidence_path,
        public_promotion_path=transaction.public_promotion_path,
        actors=backend,
    ) as recovery:
        assert recovery.marker_snapshot.state == "PENDING"
        assert handed_off is False
        assert tuple(
            name for name in seed_order if (custody / name).exists()
        ) == seed_order[:seed_prefix_length]


def test_post_stage_rehydration_reclaims_65534_tree_via_host_anchor(
    tmp_path: Path, monkeypatch,
) -> None:
    transaction, _inputs = _durable_post_stage_transaction(tmp_path)
    transaction.close_lock()
    custody = transaction.custody_directory
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=7,
    )
    original_lstat = Path.lstat
    handed_off = True
    reclaim_calls = 0

    def fake_lstat(path: Path):
        metadata = original_lstat(path)
        if handed_off and path == custody:
            values = list(metadata)
            values[4] = 65534
            values[5] = 65534
            return os.stat_result(values)
        return metadata

    def fake_reclaim(_anchor_fields):
        nonlocal handed_off, reclaim_calls
        reclaim_calls += 1
        handed_off = False
        return MappingProxyType({"raw_seed_bytes_read": False})

    monkeypatch.setattr(Path, "lstat", fake_lstat)
    monkeypatch.setattr(backend, "reclaim_pending_custody_from_anchor_v1", fake_reclaim)
    recovered = executor.FormalCeremonyTransactionV1.rehydrate_post_stage_v1(
        custody_directory=custody,
        public_evidence_path=transaction.public_evidence_path,
        public_promotion_path=transaction.public_promotion_path,
        replay=_public_replay_stub,
        actors=backend,
    )
    try:
        assert reclaim_calls == 1
        assert handed_off is False
        assert recovered.state == "SEED_CUSTODY_VERIFIED"
        assert recovered.recovery_phase == "STAGED_PENDING"
        assert recovered._anchor_descriptor is not None
        assert recovered._directory_descriptor is not None
    finally:
        recovered.close_lock()


def test_real_offline_cap_chown_helper_round_trips_0700_nobody_owned_tree(
    tmp_path: Path,
) -> None:
    docker = Path("/usr/bin/docker")
    image = "python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
    if not docker.is_file():
        pytest.skip("approved Docker executable is absent")
    daemon = subprocess.run(
        [docker, "--host=unix:///var/run/docker.sock", "info"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=20,
    )
    image_probe = subprocess.run(
        [
            docker,
            "--host=unix:///var/run/docker.sock",
            "image",
            "inspect",
            image,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=20,
    )
    if daemon.returncode != 0 or image_probe.returncode != 0:
        pytest.skip("offline daemon or pinned custodian image is unavailable")
    custody = tmp_path / "real-cap-chown-custody"
    custody.mkdir(mode=0o700)
    retained_paths = (
        custody / "phase3_m25_ceremony.lock",
        custody / f"opaque-run-{'12' * 16}.reserved",
        custody / f"opaque-ledger-{'34' * 16}.reserved",
    )
    for index, retained in enumerate(retained_paths):
        retained.write_bytes(f"metadata-only-{index}".encode("ascii"))
        retained.chmod(0o600)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    profile = json.loads(executor.PROFILE_PATH.read_bytes())
    backend._profile = profile
    with executor.LinuxLocalTemporaryDirectoryV1(
        prefix="hegel-cap-chown-test-",
        repository_root=executor.REPOSITORY_ROOT,
    ) as runtime:
        backend._docker_control_plane = executor.prepare_local_docker_control_plane_v1(
            Path(runtime), repository_root=executor.REPOSITORY_ROOT
        )
        backend._runtime_seccomp_path = executor.SECCOMP_PATH
        try:
            backend._set_custody_owner(65534, 65534)
            handed = custody.lstat()
            assert (handed.st_uid, handed.st_gid) == (65534, 65534)
            assert stat.S_IMODE(handed.st_mode) == 0o700
            backend._custody_handed_off = True
            backend._reclaim_custody_from_actor()
            reclaimed = custody.lstat()
            assert (reclaimed.st_uid, reclaimed.st_gid) == (
                os.geteuid(),
                os.getegid(),
            )
            assert [path.read_bytes() for path in retained_paths] == [
                b"metadata-only-0",
                b"metadata-only-1",
                b"metadata-only-2",
            ]
        finally:
            if custody.lstat().st_uid == 65534:
                backend._custody_handed_off = True
                backend._reclaim_custody_from_actor()


class _PreseedAbortActors(executor.CeremonyActorsV1):
    authoritative = True

    def __init__(self) -> None:
        self.calls = 0

    def recover_preseed_private_state_and_verify_absent(
        self, run_id: bytes,
    ) -> Mapping[str, object]:
        self.calls += 1
        body: dict[str, object] = {
            "schema": executor._PRESEED_ABORT_ABSENCE_SCHEMA,
            "basis_commit": "12" * 20,
            "run_id_hex": run_id.hex(),
            "exact_run_label_checked": True,
            "actor_containers_absent": True,
            "actor_key_volumes_absent": True,
            "seed_continuity_state_absent": True,
            "docker_daemon_receipt_sha256": (b"d" * 32).hex(),
        }
        body["receipt_sha256"] = hashlib.sha256(
            executor._canonical_json(body)
        ).hexdigest()
        return MappingProxyType(body)


def _preseed_abort_fixture(
    tmp_path: Path, *, fault_injector=None,
) -> executor.FormalCeremonyTransactionV1:
    transaction, _actor_trust = _reserved_prestage_transaction(
        tmp_path, fault_injector=fault_injector
    )
    transaction.close_lock()
    return transaction


def _preseed_abort_before_checkpoint_fixture(
    tmp_path: Path, *, fault_injector=None,
) -> executor.FormalCeremonyTransactionV1:
    transaction = _transaction(tmp_path, fault_injector=fault_injector)
    transaction.reserve()
    transaction.close_lock()
    return transaction


def _run_preseed_abort_fixture(
    transaction: executor.FormalCeremonyTransactionV1,
    actors: _PreseedAbortActors,
    *, fault_injector=None,
) -> None:
    executor._abort_preseed_reserved_transaction_core_v1(
        custody_directory=transaction.custody_directory,
        public_evidence_path=transaction.public_evidence_path,
        public_promotion_path=transaction.public_promotion_path,
        actors=actors,
        fault_injector=fault_injector,
    )


def test_preseed_abort_uses_exact_plan_and_leaves_no_transaction_residue(
    tmp_path: Path,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    actors = _PreseedAbortActors()
    _run_preseed_abort_fixture(transaction, actors)
    assert actors.calls == 1
    assert list(transaction.custody_directory.iterdir()) == []
    assert not (
        transaction.public_evidence_path.parent
        / (".hegel-m25-stage-" + transaction.run_id.hex())
    ).exists()
    for output in (
        transaction.public_evidence_path,
        transaction.public_promotion_path,
        transaction.publication_receipt_path,
    ):
        assert not output.exists()
        assert not output.with_name(f".{output.name}.hegel-reserved").exists()
    # Completed abort is idempotent and cannot recreate an identity.
    _run_preseed_abort_fixture(transaction, actors)
    assert actors.calls == 2
    tombstone = transaction.public_evidence_path.with_name(
        f".{transaction.public_evidence_path.name}.hegel-preseed-abort-terminal.json"
    )
    assert tombstone.is_file()
    assert json.loads(tombstone.read_bytes())["run_id_hex"] == transaction.run_id.hex()
    for output in (
        transaction.public_evidence_path,
        transaction.public_promotion_path,
        transaction.publication_receipt_path,
    ):
        marker = executor._preseed_abort_retirement_marker_path_v1(output)
        assert marker.is_file()
        assert json.loads(marker.read_bytes())["retired_output_path"] == str(output)


def test_preseed_abort_accepts_reserved_transaction_before_trust_checkpoint(
    tmp_path: Path,
) -> None:
    transaction = _preseed_abort_before_checkpoint_fixture(tmp_path)
    stage = (
        transaction.public_evidence_path.parent
        / (".hegel-m25-stage-" + transaction.run_id.hex())
    )
    assert not (stage / executor._ACTOR_TRUST_CHECKPOINT_FILENAME).exists()
    assert not (
        stage / (executor._ACTOR_TRUST_CHECKPOINT_FILENAME + ".next")
    ).exists()

    actors = _PreseedAbortActors()
    _run_preseed_abort_fixture(transaction, actors)

    assert actors.calls == 1
    assert list(transaction.custody_directory.iterdir()) == []
    assert not stage.exists()
    for output in (
        transaction.public_evidence_path,
        transaction.public_promotion_path,
        transaction.publication_receipt_path,
    ):
        assert executor._preseed_abort_retirement_marker_path_v1(output).is_file()


def test_preseed_abort_accepts_checkpoint_next_as_the_only_checkpoint_inode(
    tmp_path: Path,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    stage = (
        transaction.public_evidence_path.parent
        / (".hegel-m25-stage-" + transaction.run_id.hex())
    )
    checkpoint = stage / executor._ACTOR_TRUST_CHECKPOINT_FILENAME
    checkpoint.rename(checkpoint.with_name(checkpoint.name + ".next"))

    _run_preseed_abort_fixture(transaction, _PreseedAbortActors())

    assert list(transaction.custody_directory.iterdir()) == []
    assert not stage.exists()


@pytest.mark.parametrize("unexpected", ("both_checkpoints", "unknown_stage_file"))
def test_preseed_abort_rejects_ambiguous_or_unknown_checkpoint_stage_state(
    tmp_path: Path, unexpected: str,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    stage = (
        transaction.public_evidence_path.parent
        / (".hegel-m25-stage-" + transaction.run_id.hex())
    )
    checkpoint = stage / executor._ACTOR_TRUST_CHECKPOINT_FILENAME
    if unexpected == "both_checkpoints":
        extra = checkpoint.with_name(checkpoint.name + ".next")
        extra.write_bytes(checkpoint.read_bytes())
    else:
        extra = stage / "unexpected-checkpoint-state.json"
        extra.write_bytes(b"{}\n")
    extra.chmod(0o600)
    actors = _PreseedAbortActors()

    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        _run_preseed_abort_fixture(transaction, actors)

    assert captured.value.code == executor.FAIL_TRANSACTION_LOCK
    assert actors.calls == 0


def _preseed_abort_fault_points(row_count: int) -> tuple[str, ...]:
    return (
        "after_preseed_abort_actor_absence",
        "before_stage_next_write_preseed_abort_actor_absence",
        "after_stage_next_fsync_preseed_abort_actor_absence",
        "after_stage_rename_before_dir_fsync_preseed_abort_actor_absence",
        "after_stage_dir_fsync_preseed_abort_actor_absence",
        "after_preseed_abort_actor_absence_receipt_durable",
        "before_stage_next_write_preseed_abort_plan",
        "after_stage_next_fsync_preseed_abort_plan",
        "after_stage_rename_before_dir_fsync_preseed_abort_plan",
        "after_stage_dir_fsync_preseed_abort_plan",
        "after_preseed_abort_plan_durable",
        "before_stage_next_write_preseed_abort_terminal_tombstone",
        "after_stage_next_fsync_preseed_abort_terminal_tombstone",
        "after_stage_rename_before_dir_fsync_preseed_abort_terminal_tombstone",
        "after_stage_dir_fsync_preseed_abort_terminal_tombstone",
        "after_preseed_abort_terminal_tombstone_durable",
        *(
            point
            for role in ("evidence", "promotion", "publication_receipt")
            for point in (
                f"before_stage_next_write_preseed_abort_retirement_{role}",
                f"after_stage_next_fsync_preseed_abort_retirement_{role}",
                f"after_stage_rename_before_dir_fsync_preseed_abort_retirement_{role}",
                f"after_stage_dir_fsync_preseed_abort_retirement_{role}",
                f"after_preseed_abort_retirement_marker_durable_{role}",
            )
        ),
        *(f"before_preseed_abort_delete_{index}" for index in range(row_count)),
        *(
            f"after_preseed_abort_delete_{index}_before_parent_fsync"
            for index in range(row_count)
        ),
        *(
            f"after_preseed_abort_delete_{index}_parent_fsync"
            for index in range(row_count)
        ),
    )


@pytest.mark.parametrize("fault_point", _preseed_abort_fault_points(14))
def test_precheckpoint_preseed_abort_resumes_full_fault_matrix(
    tmp_path: Path, fault_point: str,
) -> None:
    transaction = _preseed_abort_before_checkpoint_fixture(tmp_path)
    actors = _PreseedAbortActors()
    raised = False

    def inject(point: str) -> None:
        nonlocal raised
        if point == fault_point and not raised:
            raised = True
            raise RuntimeError(point)

    with pytest.raises(RuntimeError, match=fault_point):
        _run_preseed_abort_fixture(
            transaction, actors, fault_injector=inject
        )
    assert raised is True
    _run_preseed_abort_fixture(transaction, actors)
    assert list(transaction.custody_directory.iterdir()) == []


_PRESEED_ABORT_FAULT_POINTS = _preseed_abort_fault_points(15)


@pytest.mark.parametrize("fault_point", _PRESEED_ABORT_FAULT_POINTS)
def test_preseed_abort_fault_matrix_resumes_only_exact_deleted_prefix(
    tmp_path: Path, fault_point: str,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    actors = _PreseedAbortActors()
    raised = False

    def inject(point: str) -> None:
        nonlocal raised
        if point == fault_point and not raised:
            raised = True
            raise RuntimeError(point)

    with pytest.raises(RuntimeError, match=fault_point):
        _run_preseed_abort_fixture(
            transaction, actors, fault_injector=inject
        )
    assert raised is True
    _run_preseed_abort_fixture(transaction, actors)
    assert list(transaction.custody_directory.iterdir()) == []
    assert actors.calls >= 1


def test_preseed_abort_terminal_lock_unlink_before_fsync_recovers_from_tombstone(
    tmp_path: Path,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    actors = _PreseedAbortActors()
    first_raised = False

    def stop_after_plan_delete(point: str) -> None:
        nonlocal first_raised
        if point == "after_preseed_abort_delete_13_parent_fsync" and not first_raised:
            first_raised = True
            raise RuntimeError(point)

    with pytest.raises(RuntimeError, match="delete_13"):
        _run_preseed_abort_fixture(
            transaction, actors, fault_injector=stop_after_plan_delete
        )
    second_raised = False

    def stop_after_terminal_unlink(point: str) -> None:
        nonlocal second_raised
        if (
            point == "after_preseed_abort_delete_terminal_lock_before_parent_fsync"
            and not second_raised
        ):
            second_raised = True
            raise RuntimeError(point)

    with pytest.raises(RuntimeError, match="terminal_lock"):
        _run_preseed_abort_fixture(
            transaction, actors, fault_injector=stop_after_terminal_unlink
        )
    _run_preseed_abort_fixture(transaction, actors)
    assert list(transaction.custody_directory.iterdir()) == []
    assert actors.calls == 3


def test_precheckpoint_abort_terminal_lock_recovery_uses_shorter_exact_plan(
    tmp_path: Path,
) -> None:
    transaction = _preseed_abort_before_checkpoint_fixture(tmp_path)
    actors = _PreseedAbortActors()
    plan_deleted = False

    def stop_after_plan_delete(point: str) -> None:
        nonlocal plan_deleted
        if point == "after_preseed_abort_delete_12_parent_fsync" and not plan_deleted:
            plan_deleted = True
            raise RuntimeError(point)

    with pytest.raises(RuntimeError, match="delete_12"):
        _run_preseed_abort_fixture(
            transaction, actors, fault_injector=stop_after_plan_delete
        )
    terminal_unlinked = False

    def stop_after_terminal_unlink(point: str) -> None:
        nonlocal terminal_unlinked
        if (
            point == "after_preseed_abort_delete_terminal_lock_before_parent_fsync"
            and not terminal_unlinked
        ):
            terminal_unlinked = True
            raise RuntimeError(point)

    with pytest.raises(RuntimeError, match="terminal_lock"):
        _run_preseed_abort_fixture(
            transaction, actors, fault_injector=stop_after_terminal_unlink
        )
    _run_preseed_abort_fixture(transaction, actors)
    assert list(transaction.custody_directory.iterdir()) == []
    assert actors.calls == 3


def test_preseed_abort_refuses_live_executor_before_docker_cleanup(
    tmp_path: Path,
) -> None:
    transaction, _actor_trust = _reserved_prestage_transaction(tmp_path)
    actors = _PreseedAbortActors()
    try:
        with pytest.raises(executor.FormalContainerExecutorError) as captured:
            _run_preseed_abort_fixture(transaction, actors)
        assert captured.value.code == executor.FAIL_TRANSACTION_LOCK
        assert actors.calls == 0
    finally:
        transaction.close_lock()


def test_preseed_abort_tombstone_permanently_retires_output_path(
    tmp_path: Path,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    _run_preseed_abort_fixture(transaction, _PreseedAbortActors())
    replacement = executor.FormalCeremonyTransactionV1(
        basis_commit="12" * 20,
        custody_directory=transaction.custody_directory,
        public_evidence_path=transaction.public_evidence_path,
        public_promotion_path=transaction.public_promotion_path,
        run_id=b"x" * 16,
        ledger_id=b"y" * 16,
        prestage_intent_fields=executor.build_prestage_intent_fields_v1(
            basis_commit="12" * 20,
            run_id=b"x" * 16,
            ledger_id=b"y" * 16,
            created_at_unix_seconds=7,
            trust_genesis_id=b"t" * 16,
            actor_qualification_report={},
            errata_qualification_report={},
            rust_bridge_dag_qualification_report_sha256=b"q" * 32,
            live_actor_protocol_qualification_bundle_content_id=b"v" * 32,
            qualification_only_key_ids={
                purpose: bytes([purpose]) * 16 for purpose in (1, 2, 3, 4)
            },
            live_actor_protocol_qualification_bundle={"synthetic_test_bundle": True},
            live_actor_protocol_qualification_canonical_bundle_bytes=executor._canonical_json(
                {"synthetic_test_bundle": True}
            ),
            live_actor_protocol_daemon_receipt_binding=b"d" * 32,
            runtime_binding_fields=_test_runtime_binding_fields(),
        ),
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        replacement.reserve()
    assert captured.value.code == executor.FAIL_TRANSACTION_LOCK
    assert "permanently retired" in captured.value.detail


def _replacement_transaction_for_output_paths(
    transaction: executor.FormalCeremonyTransactionV1,
    *,
    evidence: Path,
    promotion: Path,
) -> executor.FormalCeremonyTransactionV1:
    intent = executor.build_prestage_intent_fields_v1(
        basis_commit="12" * 20,
        run_id=b"x" * 16,
        ledger_id=b"y" * 16,
        created_at_unix_seconds=7,
        trust_genesis_id=b"t" * 16,
        actor_qualification_report={},
        errata_qualification_report={},
        rust_bridge_dag_qualification_report_sha256=b"q" * 32,
        live_actor_protocol_qualification_bundle_content_id=b"v" * 32,
        qualification_only_key_ids={
            purpose: bytes([purpose]) * 16 for purpose in (1, 2, 3, 4)
        },
        live_actor_protocol_qualification_bundle={"synthetic_test_bundle": True},
        live_actor_protocol_qualification_canonical_bundle_bytes=executor._canonical_json(
            {"synthetic_test_bundle": True}
        ),
        live_actor_protocol_daemon_receipt_binding=b"d" * 32,
        runtime_binding_fields=_test_runtime_binding_fields(),
    )
    return executor.FormalCeremonyTransactionV1(
        basis_commit="12" * 20,
        custody_directory=transaction.custody_directory,
        public_evidence_path=evidence,
        public_promotion_path=promotion,
        run_id=b"x" * 16,
        ledger_id=b"y" * 16,
        prestage_intent_fields=intent,
    )


@pytest.mark.parametrize(
    "reuse_case",
    (
        "same_promotion_new_evidence",
        "same_evidence_new_promotion",
        "old_promotion_as_new_evidence",
        "old_receipt_as_new_promotion",
    ),
)
def test_preseed_abort_retires_each_physical_output_across_new_combinations(
    tmp_path: Path, reuse_case: str,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    _run_preseed_abort_fixture(transaction, _PreseedAbortActors())
    fresh_evidence = transaction.public_evidence_path.with_name("fresh-evidence.json")
    fresh_promotion = transaction.public_promotion_path.with_name("fresh-promotion.json")
    if reuse_case == "same_promotion_new_evidence":
        evidence, promotion = fresh_evidence, transaction.public_promotion_path
    elif reuse_case == "same_evidence_new_promotion":
        evidence, promotion = transaction.public_evidence_path, fresh_promotion
    elif reuse_case == "old_promotion_as_new_evidence":
        evidence, promotion = transaction.public_promotion_path, fresh_promotion
    else:
        evidence, promotion = fresh_evidence, transaction.publication_receipt_path
    replacement = _replacement_transaction_for_output_paths(
        transaction, evidence=evidence, promotion=promotion
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        replacement.reserve()
    assert captured.value.code == executor.FAIL_TRANSACTION_LOCK
    assert "permanently retired" in captured.value.detail


def test_preseed_abort_allows_a_fully_fresh_output_triple(
    tmp_path: Path,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    _run_preseed_abort_fixture(transaction, _PreseedAbortActors())
    replacement = _replacement_transaction_for_output_paths(
        transaction,
        evidence=transaction.public_evidence_path.with_name("fresh-evidence.json"),
        promotion=transaction.public_promotion_path.with_name("fresh-promotion.json"),
    )
    try:
        replacement.reserve()
        assert replacement.state == "RESERVED"
    finally:
        replacement.close_lock()


@pytest.mark.parametrize(
    "mutation",
    ("stage_path", "basis_commit", "run_stage_relation", "promotion_path", "parent_inode"),
)
def test_terminal_abort_rejects_canonical_tombstone_relation_tamper_before_docker(
    tmp_path: Path, mutation: str,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    actors = _PreseedAbortActors()
    _run_preseed_abort_fixture(transaction, actors)
    tombstone_path = transaction.public_evidence_path.with_name(
        f".{transaction.public_evidence_path.name}.hegel-preseed-abort-terminal.json"
    )
    fields = json.loads(tombstone_path.read_bytes())
    if mutation == "stage_path":
        fields["stage_absolute_path"] = str(tmp_path / "another-absent-stage")
    elif mutation == "basis_commit":
        fields["basis_commit"] = "g" * 40
    elif mutation == "run_stage_relation":
        fields["run_id_hex"] = "ab" * 16
    elif mutation == "promotion_path":
        fields["public_promotion_path"] = str(
            transaction.public_promotion_path.with_name("another-promotion.json")
        )
    else:
        fields["public_parent_st_ino"] += 1
    tombstone_path.write_bytes(executor._canonical_json(fields))
    tombstone_path.chmod(0o600)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        _run_preseed_abort_fixture(transaction, actors)
    assert captured.value.code == executor.FAIL_TRANSACTION_LOCK
    assert actors.calls == 1


def test_terminal_abort_rejects_caller_symlink_chain_before_docker(
    tmp_path: Path,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    actors = _PreseedAbortActors()
    _run_preseed_abort_fixture(transaction, actors)
    custody_alias = tmp_path / "custody-alias"
    custody_alias.symlink_to(transaction.custody_directory, target_is_directory=True)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor._abort_preseed_reserved_transaction_core_v1(
            custody_directory=custody_alias,
            public_evidence_path=transaction.public_evidence_path,
            public_promotion_path=transaction.public_promotion_path,
            actors=actors,
        )
    assert captured.value.code == executor.FAIL_PREFLIGHT
    assert actors.calls == 1


def test_preseed_abort_rejects_a_deletion_gap_before_plan_commit(
    tmp_path: Path,
) -> None:
    transaction = _preseed_abort_fixture(tmp_path)
    reservation = transaction.public_promotion_path.with_name(
        f".{transaction.public_promotion_path.name}.hegel-reserved"
    )
    reservation.unlink()
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        _run_preseed_abort_fixture(transaction, _PreseedAbortActors())
    assert captured.value.code == executor.FAIL_TRANSACTION_LOCK


def test_explicit_pending_recovery_rejects_tampered_output_reservation(
    tmp_path: Path,
) -> None:
    transaction, actor_trust = _reserved_prestage_transaction(tmp_path)
    executor.create_pending_marker_v1(
        secret_state_directory=tmp_path / "custody",
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=actor_trust.key_ids[1],
        created_at_unix_seconds=7,
    )
    transaction.close_lock()
    reservation = transaction.public_evidence_path.with_name(
        ".evidence.json.hegel-reserved"
    )
    reservation.write_bytes(b"{}\n")
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.acquire_pending_ceremony_recovery_v1(
            custody_directory=tmp_path / "custody",
            public_evidence_path=transaction.public_evidence_path,
            public_promotion_path=transaction.public_promotion_path,
        )
    assert captured.value.code == executor.FAIL_TRANSACTION_LOCK


def _rehydrate_post_stage(
    transaction: executor.FormalCeremonyTransactionV1,
):
    return executor.FormalCeremonyTransactionV1.rehydrate_post_stage_v1(
        custody_directory=transaction.custody_directory,
        public_evidence_path=transaction.public_evidence_path,
        public_promotion_path=transaction.public_promotion_path,
        replay=_public_replay_stub,
    )


def test_fresh_process_staged_pending_continuation_reuses_ids_seed_and_key(
    tmp_path: Path, monkeypatch,
) -> None:
    transaction, _inputs = _durable_post_stage_transaction(tmp_path)
    transaction.close_lock()
    monkeypatch.setattr(
        executor.secrets,
        "token_bytes",
        lambda _length: (_ for _ in ()).throw(AssertionError("recovery drew entropy")),
    )
    recovered = _rehydrate_post_stage(transaction)
    actors = _PostStageRecoveryActors(
        transaction.custody_directory, private_state_present=True
    )
    try:
        assert recovered.recovery_phase == "STAGED_PENDING"
        payload, promotion = executor._continue_post_stage_transaction_recovery_core_v1(
            transaction=recovered, actors=actors, replay=_public_replay_stub
        )
        assert payload["artifact_kind"] == "FORMAL_GATE_EVIDENCE_INPUTS_PUBLIC_REPLAY"
        assert promotion == {"qualified": True, "state": "NOT_RUN"}
        assert actors.resume_count == 1
        assert actors.private_state_present is False
        assert recovered.run_id == b"r" * 16
        assert recovered.ledger_id == b"l" * 16
        assert {
            path.name for path in recovered.custody_directory.glob("opaque-*-*.reserved")
        } == {
            f"opaque-run-{(b'r' * 16).hex()}.reserved",
            f"opaque-ledger-{(b'l' * 16).hex()}.reserved",
        }
        assert recovered.state == "PUBLISHED"
    finally:
        recovered.close_lock()


def test_one_byte_seed_substitution_fails_before_marker_cleanup_or_publication(
    tmp_path: Path,
) -> None:
    transaction, _inputs = _durable_post_stage_transaction(tmp_path)
    seed_path = transaction.custody_directory / "split_master_seed.bin"
    changed = bytearray(seed_path.read_bytes())
    changed[-1] ^= 1
    seed_path.write_bytes(bytes(changed))
    seed_path.chmod(0o600)
    transaction.close_lock()
    recovered = _rehydrate_post_stage(transaction)
    actors = _PostStageRecoveryActors(
        transaction.custody_directory, private_state_present=True
    )
    try:
        with pytest.raises(executor.FormalContainerExecutorError) as captured:
            executor._continue_post_stage_transaction_recovery_core_v1(
                transaction=recovered,
                actors=actors,
                replay=_public_replay_stub,
            )
        assert captured.value.code == executor.FAIL_CUSTODY
        marker = executor.read_marker_snapshot_v1(
            recovered.custody_directory / "split_seed_instantiation.marker"
        )
        assert marker.state == "PENDING"
        assert actors.private_state_present is True
        assert actors.complete_cleanup_count == 0
        assert not recovered.public_evidence_path.exists()
        assert not recovered.public_promotion_path.exists()
        assert not recovered.publication_receipt_path.exists()
        assert recovered.journal_path is not None
        journal = json.loads(recovered.journal_path.read_bytes())
        assert journal["state"] == "SEED_CUSTODY_VERIFIED"
        assert (
            recovered.journal_path.parent
            / executor._SEED_CUSTODY_VERIFICATION_FILENAME
        ).is_file()
    finally:
        recovered.close_lock()


@pytest.mark.parametrize(
    ("journal_state", "private_state_present", "expected_phase"),
    (
        ("STAGED_PROSPECTIVE_REPLAY_PASSED", True, "MARKER_COMPLETE_CLEANUP_STATUS_UNKNOWN"),
        ("MARKER_COMPLETE", False, "MARKER_COMPLETE_CLEANUP_STATUS_UNKNOWN"),
        ("ACTORS_ABSENT", False, "ACTORS_ABSENT"),
    ),
)
def test_complete_marker_recovery_before_or_after_private_volume_deletion(
    tmp_path: Path,
    journal_state: str,
    private_state_present: bool,
    expected_phase: str,
) -> None:
    transaction, inputs = _durable_post_stage_transaction(tmp_path)
    actual = executor.complete_marker_v1(
        marker_path=transaction.custody_directory / "split_seed_instantiation.marker",
        seed_commitment_manifest_root=inputs.marker_snapshot.seed_commitment_manifest_root,
    )
    if journal_state in {"MARKER_COMPLETE", "ACTORS_ABSENT"}:
        transaction.record_marker_complete(actual, inputs.marker_snapshot)
    if journal_state == "ACTORS_ABSENT":
        transaction.record_actors_absent()
    transaction.close_lock()
    recovered = _rehydrate_post_stage(transaction)
    actors = _PostStageRecoveryActors(
        transaction.custody_directory,
        private_state_present=private_state_present,
    )
    try:
        assert recovered.recovery_phase == expected_phase
        executor._continue_post_stage_transaction_recovery_core_v1(
            transaction=recovered, actors=actors, replay=_public_replay_stub
        )
        assert actors.complete_cleanup_count == 1
        assert actors.private_state_present is False
        assert recovered.state == "PUBLISHED"
    finally:
        recovered.close_lock()


@pytest.mark.parametrize(
    ("fault_point", "expected_phase"),
    (
        ("after_evidence_publication", "PARTIAL_PUBLICATION"),
        ("after_promotion_publication", "PARTIAL_PUBLICATION"),
        ("after_receipt_publication", "ALL_PUBLIC_OUTPUTS_UNJOURNALED"),
        ("after_output_reservation_1_cleanup", "ALL_PUBLIC_OUTPUTS_UNJOURNALED"),
        ("after_output_reservation_2_cleanup", "ALL_PUBLIC_OUTPUTS_UNJOURNALED"),
        ("after_output_reservation_cleanup", "ALL_PUBLIC_OUTPUTS_UNJOURNALED"),
    ),
)
def test_partial_three_file_publication_is_exactly_and_idempotently_completed(
    tmp_path: Path, fault_point: str, expected_phase: str,
) -> None:
    def inject(point: str) -> None:
        if point == fault_point:
            raise RuntimeError("injected-publication-crash")

    transaction, inputs = _durable_post_stage_transaction(
        tmp_path, fault_injector=inject
    )
    actual = executor.complete_marker_v1(
        marker_path=transaction.custody_directory / "split_seed_instantiation.marker",
        seed_commitment_manifest_root=inputs.marker_snapshot.seed_commitment_manifest_root,
    )
    transaction.record_marker_complete(actual, inputs.marker_snapshot)
    transaction.record_actors_absent()
    with pytest.raises(RuntimeError, match="injected-publication-crash"):
        transaction.publish(replay=_public_replay_stub)
    staged = dict(transaction._staged_payloads)
    transaction.close_lock()

    recovered = _rehydrate_post_stage(transaction)
    actors = _PostStageRecoveryActors(
        transaction.custody_directory, private_state_present=False
    )
    try:
        assert recovered.recovery_phase == expected_phase
        executor._continue_post_stage_transaction_recovery_core_v1(
            transaction=recovered, actors=actors, replay=_public_replay_stub
        )
        assert recovered.public_evidence_path.read_bytes() == staged["evidence"]
        assert recovered.public_promotion_path.read_bytes() == staged["promotion"]
        assert recovered.publication_receipt_path.read_bytes() == staged["receipt"]
        assert not any(
            path.exists() for path in recovered._output_reservation_paths.values()
        )
        assert recovered.state == "PUBLISHED"
    finally:
        recovered.close_lock()


def test_published_transaction_recovery_is_idempotent(tmp_path: Path) -> None:
    transaction, inputs = _durable_post_stage_transaction(tmp_path)
    actual = executor.complete_marker_v1(
        marker_path=transaction.custody_directory / "split_seed_instantiation.marker",
        seed_commitment_manifest_root=inputs.marker_snapshot.seed_commitment_manifest_root,
    )
    transaction.record_marker_complete(actual, inputs.marker_snapshot)
    transaction.record_actors_absent()
    transaction.publish(replay=_public_replay_stub)
    expected = (
        transaction.public_evidence_path.read_bytes(),
        transaction.public_promotion_path.read_bytes(),
        transaction.publication_receipt_path.read_bytes(),
    )
    transaction.close_lock()
    recovered = _rehydrate_post_stage(transaction)
    actors = _PostStageRecoveryActors(
        transaction.custody_directory, private_state_present=False
    )
    try:
        assert recovered.recovery_phase == "PUBLISHED"
        executor._continue_post_stage_transaction_recovery_core_v1(
            transaction=recovered, actors=actors, replay=_public_replay_stub
        )
        assert expected == (
            recovered.public_evidence_path.read_bytes(),
            recovered.public_promotion_path.read_bytes(),
            recovered.publication_receipt_path.read_bytes(),
        )
    finally:
        recovered.close_lock()


@pytest.mark.parametrize(
    ("crash_transition", "expected_phase"),
    (
        ("STAGED", "STAGED_PENDING"),
        ("MARKER_COMPLETE", "MARKER_COMPLETE_CLEANUP_STATUS_UNKNOWN"),
        ("ACTORS_ABSENT", "ACTORS_ABSENT"),
        ("PUBLISHED", "PUBLISHED"),
    ),
)
def test_fsync_complete_next_journal_is_promoted_exactly_one_step(
    tmp_path: Path, crash_transition: str, expected_phase: str,
) -> None:
    active = False

    def inject(point: str) -> None:
        if active and point == "after_journal_next_fsync":
            raise RuntimeError("crash-after-next-journal-fsync")

    transaction = _transaction(tmp_path, inject)
    _reserve_test_transaction(transaction)
    inputs = _post_stage_inputs()
    executor.create_pending_marker_v1(
        secret_state_directory=transaction.custody_directory,
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=_RECOVERY_KEY_ID,
        created_at_unix_seconds=7,
    )
    _write_post_stage_seed_state(transaction.custody_directory)
    payload = executor.serialize_gate_evidence_inputs_v1(inputs)
    if crash_transition == "STAGED":
        active = True
        with pytest.raises(RuntimeError, match="next-journal"):
            transaction.stage_and_prospectively_replay(
                payload, _public_replay_stub(payload), replay=_public_replay_stub
            )
    else:
        transaction.stage_and_prospectively_replay(
            payload, _public_replay_stub(payload), replay=_public_replay_stub
        )
        _verify_staged_seed_for_test(transaction)
        actual = executor.complete_marker_v1(
            marker_path=transaction.custody_directory / "split_seed_instantiation.marker",
            seed_commitment_manifest_root=inputs.marker_snapshot.seed_commitment_manifest_root,
        )
        if crash_transition == "MARKER_COMPLETE":
            active = True
            with pytest.raises(RuntimeError, match="next-journal"):
                transaction.record_marker_complete(actual, inputs.marker_snapshot)
        else:
            transaction.record_marker_complete(actual, inputs.marker_snapshot)
            if crash_transition == "ACTORS_ABSENT":
                active = True
                with pytest.raises(RuntimeError, match="next-journal"):
                    transaction.record_actors_absent()
            else:
                transaction.record_actors_absent()
                active = True
                with pytest.raises(RuntimeError, match="next-journal"):
                    transaction.publish(replay=_public_replay_stub)
    transaction.close_lock()
    recovered = _rehydrate_post_stage(transaction)
    try:
        assert recovered.recovery_phase == expected_phase
        assert not (recovered.journal_path.parent / "transaction-journal.next").exists()
    finally:
        recovered.close_lock()


def test_exact_stage_replay_recovers_crash_before_staged_journal_inode(
    tmp_path: Path,
) -> None:
    armed = False

    def inject(point: str) -> None:
        if armed and point == "before_journal_next_write":
            raise RuntimeError("crash-before-next-journal")

    transaction = _transaction(tmp_path, inject)
    _reserve_test_transaction(transaction)
    inputs = _post_stage_inputs()
    executor.create_pending_marker_v1(
        secret_state_directory=transaction.custody_directory,
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=_RECOVERY_KEY_ID,
        created_at_unix_seconds=7,
    )
    _write_post_stage_seed_state(transaction.custody_directory)
    payload = executor.serialize_gate_evidence_inputs_v1(inputs)
    armed = True
    with pytest.raises(RuntimeError, match="before-next-journal"):
        transaction.stage_and_prospectively_replay(
            payload, _public_replay_stub(payload), replay=_public_replay_stub
        )
    transaction.close_lock()
    recovered = _rehydrate_post_stage(transaction)
    try:
        assert recovered.recovery_phase == "STAGED_PENDING"
        journal = json.loads(recovered.journal_path.read_bytes())
        assert journal["state"] == "STAGED_PROSPECTIVE_REPLAY_PASSED"
    finally:
        recovered.close_lock()


def test_fsync_complete_marker_temp_is_promoted_after_full_stage_validation(
    tmp_path: Path,
) -> None:
    transaction, inputs = _durable_post_stage_transaction(tmp_path)
    marker_path = transaction.custody_directory / "split_seed_instantiation.marker"
    pending_bytes = marker_path.read_bytes()
    executor.complete_marker_v1(
        marker_path=marker_path,
        seed_commitment_manifest_root=inputs.marker_snapshot.seed_commitment_manifest_root,
    )
    complete_bytes = marker_path.read_bytes()
    marker_path.write_bytes(pending_bytes)
    marker_path.chmod(0o600)
    next_path = marker_path.with_name(marker_path.name + ".complete.tmp")
    next_path.write_bytes(complete_bytes)
    next_path.chmod(0o600)
    transaction.close_lock()

    recovered = _rehydrate_post_stage(transaction)
    try:
        assert recovered.recovery_phase == "MARKER_COMPLETE_CLEANUP_STATUS_UNKNOWN"
        assert executor.read_marker_snapshot_v1(marker_path) == inputs.marker_snapshot
        assert not next_path.exists()
    finally:
        recovered.close_lock()


@pytest.mark.parametrize("tamper", ("stage", "marker", "opaque_id", "public"))
def test_post_stage_rehydration_rejects_every_tampered_identity_or_byte(
    tmp_path: Path, tamper: str,
) -> None:
    transaction, inputs = _durable_post_stage_transaction(tmp_path)
    if tamper == "stage":
        path = transaction.journal_path.parent / "promotion.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif tamper == "marker":
        actual = executor.complete_marker_v1(
            marker_path=transaction.custody_directory / "split_seed_instantiation.marker",
            seed_commitment_manifest_root=b"x" * 32,
        )
        assert actual != inputs.marker_snapshot
    elif tamper == "opaque_id":
        path = next(transaction.custody_directory.glob("opaque-run-*.reserved"))
        value = json.loads(path.read_bytes())
        value["opaque_id_hex"] = (b"x" * 16).hex()
        path.write_bytes(executor._canonical_json(value))
    else:
        actual = executor.complete_marker_v1(
            marker_path=transaction.custody_directory / "split_seed_instantiation.marker",
            seed_commitment_manifest_root=inputs.marker_snapshot.seed_commitment_manifest_root,
        )
        transaction.record_marker_complete(actual, inputs.marker_snapshot)
        transaction.record_actors_absent()
        transaction.publish(replay=_public_replay_stub)
        transaction.public_evidence_path.write_bytes(b"{}\n")
    transaction.close_lock()
    with pytest.raises(executor.FormalContainerExecutorError):
        _rehydrate_post_stage(transaction)


@pytest.mark.parametrize(
    "inconsistent_phase",
    ("pending_with_complete_journal", "published_before_actors_absent", "nonprefix_output"),
)
def test_post_stage_phase_classifier_rejects_inconsistent_artifact_orders(
    tmp_path: Path, inconsistent_phase: str,
) -> None:
    transaction, inputs = _durable_post_stage_transaction(tmp_path)
    if inconsistent_phase == "pending_with_complete_journal":
        transaction.journal_path.write_bytes(
            executor._canonical_json(transaction._journal_fields("MARKER_COMPLETE"))
        )
    else:
        actual = executor.complete_marker_v1(
            marker_path=transaction.custody_directory / "split_seed_instantiation.marker",
            seed_commitment_manifest_root=inputs.marker_snapshot.seed_commitment_manifest_root,
        )
        if inconsistent_phase == "published_before_actors_absent":
            staged = transaction.journal_path.parent / "public-evidence.json"
            transaction.public_evidence_path.write_bytes(staged.read_bytes())
            transaction.public_evidence_path.chmod(0o644)
        else:
            transaction.record_marker_complete(actual, inputs.marker_snapshot)
            transaction.record_actors_absent()
            staged = transaction.journal_path.parent / "promotion.json"
            transaction.public_promotion_path.write_bytes(staged.read_bytes())
            transaction.public_promotion_path.chmod(0o644)
    transaction.close_lock()
    with pytest.raises(executor.FormalContainerExecutorError):
        _rehydrate_post_stage(transaction)


@pytest.mark.parametrize(
    "relative",
    (
        "artifacts/phase3_m25_external/.hegel-m25-stage-001/transaction-journal.json",
        "artifacts/phase3_m25_external/.evidence.json.hegel-reserved",
        "artifacts/phase3_m25_external/nested/.promotion.json.hegel-reserved",
        "artifacts/phase3_m25_external/private/split_seed_generation.intent",
        "artifacts/phase3_m25_external/private/split_seed_generation.complete",
        "artifacts/phase3_m25_external/private/split_master_seed.bin",
    ),
)
def test_commit_b_transient_ceremony_state_remains_gitignored(relative: str) -> None:
    project = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "check-ignore", "-q", relative],
        cwd=project,
        check=False,
    )
    assert completed.returncode == 0


def test_commit_b_output_names_are_exactly_allowlisted(tmp_path: Path) -> None:
    executor.validate_commit_b_output_names_v1(
        tmp_path / executor.COMMIT_B_EVIDENCE_BASENAME,
        tmp_path / executor.COMMIT_B_PROMOTION_BASENAME,
    )
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        executor.validate_commit_b_output_names_v1(
            tmp_path / "evidence.json", tmp_path / executor.COMMIT_B_PROMOTION_BASENAME
        )
    assert captured.value.code == executor.FAIL_PUBLICATION


def test_rust_replay_binary_path_and_digest_are_both_bound(tmp_path: Path) -> None:
    binary = tmp_path / "rust-replay"
    binary.write_bytes(b"exact-binary")
    binary.chmod(0o755)
    basis = SimpleNamespace(
        implementation_inputs={
            "rust_binary_path": str(binary.resolve()),
            "rust_binary_sha256": hashlib.sha256(binary.read_bytes()).digest(),
        }
    )
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=binary,
        timestamp=1,
    )
    assert backend.validate_rust_replay_binding(basis) == hashlib.sha256(b"exact-binary").digest()
    binary.write_bytes(b"drift")
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend.validate_rust_replay_binding(basis)
    assert captured.value.code == executor.FAIL_PREFLIGHT


def test_rust_split_build_directory_is_explicitly_writable_after_umask(
    tmp_path: Path, monkeypatch,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._root = tmp_path / "work"
    source = backend._root / "purpose-1/input/tools/phase3_split_partition_calculator_fd3_v1.rs"
    source.parent.mkdir(parents=True)
    source.write_text("fn main() {}\n")
    backend._profile = {"images": {"rust_attester": "rust@sha256:" + "2" * 64}}
    _install_fake_docker_boundary(backend, tmp_path)
    probe_source = backend._root / "purpose-3/input/tools/phase3_container_actor_probe_v1.rs"
    probe_source.parent.mkdir(parents=True)
    probe_source.write_text("fn main() {}\n")

    def fake_run(command, **_kwargs):
        build = backend._root / "rust-split-build"
        assert stat_mode(build) == 0o777
        binary = build / "rust-split-calculator"
        binary.write_bytes(b"binary")
        binary.chmod(0o755)
        probe_binary = build / "rust-live-probe"
        probe_binary.write_bytes(b"probe")
        probe_binary.chmod(0o755)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    backend._compile_rust_split()
    assert (backend._root / "purpose-1/input/rust-split-calculator").stat().st_mode & 0o777 == 0o555


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_completed_actor_cleanup_removes_and_verifies_all_private_volumes(
    tmp_path: Path, monkeypatch,
) -> None:
    os.chmod(tmp_path, 0o700)
    marker_path, _pending = executor.create_pending_marker_v1(
        secret_state_directory=tmp_path,
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=b"k" * 16,
        created_at_unix_seconds=1,
    )
    complete = executor.complete_marker_v1(
        marker_path=marker_path,
        seed_commitment_manifest_root=b"m" * 32,
    )
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._containers = {purpose: str(purpose) * 64 for purpose in (1, 2, 3, 4)}
    backend._state_volumes = {purpose: f"volume-{purpose}" for purpose in (1, 2, 3, 4)}
    backend._transaction_run_id = b"r" * 16
    backend._ceremony_token = "token"
    _configure_fake_actor_identity(backend, tmp_path)
    monkeypatch.setattr(
        backend,
        "_verify_complete_custody_retained",
        lambda: MappingProxyType({"schema": "test-only"}),
    )
    backend.authorize_private_state_destruction(complete)
    commands = []
    present = set(backend._state_volumes.values())

    def fake_run(command, **_kwargs):
        commands.append(tuple(command))
        if command[1:2] == ["inspect"]:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"absent")
        if command[1:3] == ["volume", "inspect"]:
            name = command[3]
            if name not in present:
                return SimpleNamespace(returncode=1, stdout=b"", stderr=b"absent")
            purpose = int(name.rsplit("-", 1)[1])
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([_fake_volume_row(backend, purpose, name)]).encode(),
                stderr=b"",
            )
        if command[1:3] == ["volume", "rm"]:
            present.remove(command[3])
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    backend.destroy_private_state_and_verify_absent()
    assert backend._containers == {}
    assert backend._state_volumes == {}
    assert sum(command[:3] == ("docker", "volume", "rm") for command in commands) == 4
    assert any(command[:3] == ("docker", "volume", "ls") for command in commands)


def test_complete_marker_without_durable_stage_authorization_retains_volumes(
    tmp_path: Path, monkeypatch,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    os.chmod(custody, 0o700)
    marker_path, _pending = executor.create_pending_marker_v1(
        secret_state_directory=custody,
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=b"k" * 16,
        created_at_unix_seconds=1,
    )
    executor.complete_marker_v1(
        marker_path=marker_path,
        seed_commitment_manifest_root=b"m" * 32,
    )
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._state_volumes = {1: "volume-1"}
    _configure_fake_actor_identity(backend, tmp_path)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(tuple(command))
        if command[1:3] == ["volume", "inspect"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([_fake_volume_row(backend, 1, command[3])]).encode(),
                stderr=b"",
            )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    backend.stop_for_recovery_and_verify_absent()
    assert backend._state_volumes == {1: "volume-1"}
    assert not any(command[:3] == ("docker", "volume", "rm") for command in commands)


def test_container_removal_failure_is_fatal(tmp_path: Path, monkeypatch) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._containers = {1: "1" * 64}
    backend._ceremony_token = "token"

    def fake_run(command, **_kwargs):
        if command[1:3] == ["rm", "--force"]:
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"failure")
        return SimpleNamespace(returncode=1 if command[1] == "inspect" else 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend.stop_for_recovery_and_verify_absent()
    assert captured.value.code == executor.FAIL_CONTAINER


def test_pending_cleanup_retains_and_verifies_private_state_volumes(
    tmp_path: Path, monkeypatch,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    os.chmod(custody, 0o700)
    executor.create_pending_marker_v1(
        secret_state_directory=custody,
        split_version_digest=executor.SPLIT_VERSION_DIGEST,
        custodian_key_id=b"k" * 16,
        created_at_unix_seconds=1,
    )
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._state_volumes = {1: "volume-1", 2: "volume-2"}
    _configure_fake_actor_identity(backend, tmp_path)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(tuple(command))
        if command[1:3] == ["volume", "inspect"]:
            purpose = int(command[3].rsplit("-", 1)[1])
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [_fake_volume_row(backend, purpose, command[3])]
                ).encode(),
                stderr=b"",
            )
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    backend.stop_for_recovery_and_verify_absent()
    assert backend._state_volumes == {1: "volume-1", 2: "volume-2"}
    assert not any(command[:3] == ("docker", "volume", "rm") for command in commands)
    assert sum(command[:3] == ("docker", "volume", "inspect") for command in commands) == 2


def test_marker_absent_cleanup_destroys_new_private_state_volumes(
    tmp_path: Path, monkeypatch,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    os.chmod(custody, 0o700)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._state_volumes = {1: "volume-1", 2: "volume-2"}
    _configure_fake_actor_identity(backend, tmp_path)
    commands = []
    present = set(backend._state_volumes.values())

    def fake_run(command, **_kwargs):
        commands.append(tuple(command))
        if command[1:3] == ["volume", "inspect"]:
            name = command[3]
            if name not in present:
                return SimpleNamespace(returncode=1, stdout=b"", stderr=b"absent")
            purpose = int(name.rsplit("-", 1)[1])
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps([_fake_volume_row(backend, purpose, name)]).encode(),
                stderr=b"",
            )
        if command[1:3] == ["volume", "rm"]:
            present.remove(command[3])
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    backend.stop_for_recovery_and_verify_absent()
    assert backend._state_volumes == {}
    assert sum(command[:3] == ("docker", "volume", "rm") for command in commands) == 2


def test_invalid_marker_retains_private_state_and_makes_cleanup_fatal(
    tmp_path: Path, monkeypatch,
) -> None:
    custody = tmp_path / "custody"
    custody.mkdir(mode=0o700)
    os.chmod(custody, 0o700)
    marker = custody / "split_seed_instantiation.marker"
    marker.write_bytes(b"not-json\n")
    marker.chmod(0o600)
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=custody,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    backend._state_volumes = {1: "volume-1"}
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout=b"[]", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    with pytest.raises(executor.FormalContainerExecutorError) as captured:
        backend.stop_for_recovery_and_verify_absent()
    assert captured.value.code == executor.FAIL_CONTAINER
    assert backend._state_volumes == {1: "volume-1"}
    assert not any(command[:3] == ("docker", "volume", "rm") for command in commands)


def test_private_volume_initializer_is_offline_minimal_and_nonroot_probed(
    tmp_path: Path, monkeypatch,
) -> None:
    backend = executor.DockerCeremonyActorsV1(
        basis_commit="12" * 20,
        custody_directory=tmp_path,
        rust_formal_replay_binary=tmp_path / "rust-replay",
        timestamp=1,
    )
    image = "python@sha256:" + "1" * 64
    backend._profile = {"images": {"custodian": image}}
    backend._profile_digest = b"p" * 32
    backend._transaction_run_id = b"r" * 16
    _install_fake_docker_boundary(backend, tmp_path)
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(tuple(command))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(executor, "_run", fake_run)
    backend._initialize_new_state_volume(1, "private-volume")
    assert len(commands) == 2
    initializer, probe = commands
    assert "--pull=never" in initializer and "--network=none" in initializer
    assert "--cap-drop=ALL" in initializer and "--cap-add=CHOWN" in initializer
    assert "--user=0:0" in initializer
    assert "--pull=never" in probe and "--network=none" in probe
    assert "--cap-drop=ALL" in probe and not any("--cap-add" in row for row in probe)
    assert "--user=65534:65534" in probe
    receipt = backend.volume_initialization_receipts[1]
    assert receipt["nonroot_live_write_stat_probe_passed"] is True
    assert receipt["initializer_capabilities"] == ["CHOWN"]
    assert receipt["profile_sha256"] == (b"p" * 32).hex()


@pytest.mark.skipif(
    os.environ.get("HEGEL_RUN_M25_VOLUME_LIVE_TEST") != "1",
    reason="opt-in local-Docker probe; never pulls and never enables networking",
)
def test_private_volume_initializer_live_docker_offline() -> None:
    profile = json.loads(executor.PROFILE_PATH.read_text(encoding="ascii"))
    volume_name = f"hegel-m25-volume-init-live-probe-{os.getpid()}"
    created = subprocess.run(
        ["docker", "volume", "create", volume_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert created.returncode == 0, created.stderr.decode("utf-8", "replace")
    try:
        backend = executor.DockerCeremonyActorsV1(
            basis_commit="12" * 20,
            custody_directory=Path("/tmp"),
            rust_formal_replay_binary=Path("/tmp/not-used"),
            timestamp=1,
        )
        backend._profile = profile
        backend._profile_digest = hashlib.sha256(
            executor.PROFILE_PATH.read_bytes()
        ).digest()
        backend._transaction_run_id = b"v" * 16
        backend._initialize_new_state_volume(1, volume_name)
        assert backend.volume_initialization_receipts[1][
            "nonroot_live_write_stat_probe_passed"
        ] is True
    finally:
        subprocess.run(
            ["docker", "volume", "rm", "--force", volume_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


@pytest.mark.skipif(
    os.environ.get("HEGEL_RUN_FORMAL_SIGNER_LIVE_PROBE") != "1",
    reason="opt-in four-container qualify-only probe; generates no key or seed",
)
def test_four_long_lived_signers_qualify_offline_without_secret_generation() -> None:
    durable_parent = Path.home() / ".local/state/hegel-machine-test-custody"
    durable_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    durable_parent.chmod(0o700)
    project = Path(__file__).resolve().parents[1]
    repository = project.parent
    with tempfile.TemporaryDirectory(
        prefix="m25-qualify-only-", dir=durable_parent
    ) as raw_custody:
        custody = Path(raw_custody)
        custody.chmod(0o700)
        binary = Path("/usr/bin/true")
        run_id = hashlib.sha256(raw_custody.encode("utf-8")).digest()[:16]
        backend = executor.DockerCeremonyActorsV1(
            basis_commit="12" * 20,
            custody_directory=custody,
            rust_formal_replay_binary=binary,
            timestamp=1,
        )
        backend.bind_transaction_identity(run_id)
        backend._bound_rust_replay_digest = hashlib.sha256(
            binary.read_bytes()
        ).digest()
        backend._git_blob = lambda relative: (repository / relative).read_bytes()
        try:
            backend.start()
            assert set(backend.live_actor_probe_receipts) == {1, 2, 3, 4}
            assert len(backend.operation_probe_receipts) == 4
            first = backend.operation_probe_receipts[0]
            backend._exec(1, "qualify-only")
            second = backend.operation_probe_receipts[-1]
            assert second["operation_sequence"] == first["operation_sequence"] + 1
            assert second["operation_nonce_hex"] != first["operation_nonce_hex"]
            assert (
                second["operation_request_sha256"]
                != first["operation_request_sha256"]
            )
            assert len(backend.operation_probe_receipts) == 5
            assert backend._public_keys == {}
            assert backend._key_ids == {}
            assert list(custody.iterdir()) == []
        finally:
            backend.close()
        assert list(custody.iterdir()) == []
        docker_environment = {
            "DOCKER_CONFIG": "/tmp",
            "DOCKER_HOST": "unix:///var/run/docker.sock",
            "HOME": "/tmp",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        }
        for arguments in (
            (
                "ps", "-aq", "--filter", f"label=hegel.m25.run={run_id.hex()}"
            ),
            (
                "volume", "ls", "-q", "--filter",
                f"label=hegel.m25.run={run_id.hex()}",
            ),
        ):
            completed = subprocess.run(
                [
                    "/usr/bin/docker",
                    "--host=unix:///var/run/docker.sock",
                    *arguments,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
                env=docker_environment,
            )
            assert completed.returncode == 0
            assert completed.stdout.strip() == b""


def _load_worker_module():
    project = Path(__file__).resolve().parents[1]
    path = project / "tools/phase3_m25_formal_actor_worker_v1.py"
    spec = importlib.util.spec_from_file_location("hegel_m25_worker_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_fd_calculator_runs_real_python_endpoint_with_synthetic_seed() -> None:
    worker = _load_worker_module()
    project = Path(__file__).resolve().parents[1]
    calculator = project / "tools/phase3_split_partition_calculator_fd3_v1.py"
    seed = hashlib.sha256(b"HEGEL/M25/TEST/SYNTHETIC/FD3/SEED/V1").digest()

    frame = worker.run_fd_calculator([sys.executable, str(calculator)], seed)

    assert frame
    assert len(frame) <= 2048


def test_fd_calculator_real_python_rust_endpoints_agree_offline() -> None:
    """Compile the dependency-free Rust endpoint locally and exercise FD3/FD5.

    The Docker fallback is strictly local/offline and skips when the pinned
    compiler image is not already present.  It never pulls or builds an image
    and the test seed is a deterministic non-authoritative fixture.
    """

    worker = _load_worker_module()
    project = Path(__file__).resolve().parents[1]
    python_calculator = project / "tools/phase3_split_partition_calculator_fd3_v1.py"
    rust_source = project / "tools/phase3_split_partition_calculator_fd3_v1.rs"
    seed = hashlib.sha256(b"HEGEL/M25/TEST/SYNTHETIC/FD3/SEED/V1").digest()
    with tempfile.TemporaryDirectory(
        prefix="hegel-m25-fd-e2e-", dir="/tmp"
    ) as raw_temporary:
        local_root = Path(raw_temporary)
        rust_binary = local_root / "rust-split-calculator"
        rustc = shutil.which("rustc")
        if rustc is not None:
            compile_command = [
                rustc,
                "--edition=2021",
                "-C",
                "debuginfo=0",
                "-C",
                "strip=symbols",
                str(rust_source),
                "-o",
                str(rust_binary),
            ]
            completed = subprocess.run(
                compile_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=120,
                env={
                    "LC_ALL": "C",
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                },
            )
            assert completed.returncode == 0, completed.stderr.decode(
                "utf-8", "replace"
            )
        else:
            docker = shutil.which("docker")
            socket_path = Path("/var/run/docker.sock")
            if docker is None or not socket_path.exists():
                pytest.skip("neither host rustc nor a local Docker daemon is available")
            profile = json.loads(executor.PROFILE_PATH.read_text(encoding="ascii"))
            image = profile["images"]["rust_attester"]
            build_seccomp = (
                project / "config/phase3_m3_offline_build_seccomp_v1.json"
            ).resolve()
            docker_config = local_root / "docker-config"
            build_dir = local_root / "rust-build"
            docker_config.mkdir(mode=0o700)
            build_dir.mkdir(mode=0o700)
            docker_env = {
                "DOCKER_CONFIG": str(docker_config),
                "DOCKER_HOST": "unix:///var/run/docker.sock",
                "LC_ALL": "C",
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            }
            present = subprocess.run(
                [docker, "image", "inspect", image],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=30,
                env=docker_env,
            )
            if present.returncode != 0:
                pytest.skip("pinned Rust image is not already present locally")
            name = f"hegel-m25-fd-e2e-{os.getpid()}"
            command = [
                docker,
                "run",
                "--rm",
                f"--name={name}",
                "--pull=never",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges=true",
                f"--security-opt=seccomp={build_seccomp}",
                f"--user={os.getuid()}:{os.getgid()}",
                "--pids-limit=64",
                "--memory=512m",
                "--memory-swap=512m",
                "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m,mode=0700",
                "--mount",
                f"type=bind,src={rust_source},dst=/source.rs,readonly,bind-propagation=rprivate",
                "--mount",
                f"type=bind,src={build_dir},dst=/build,bind-propagation=rprivate",
                "--entrypoint=/usr/bin/env",
                image,
                "-i",
                "PATH=/usr/local/cargo/bin:/usr/bin:/bin",
                "RUSTUP_HOME=/usr/local/rustup",
                "CARGO_HOME=/usr/local/cargo",
                "TMPDIR=/build",
                "/usr/local/cargo/bin/rustc",
                "--edition=2021",
                "-C",
                "debuginfo=0",
                "-C",
                "strip=symbols",
                "/source.rs",
                "-o",
                "/build/rust-split-calculator",
            ]
            try:
                completed = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=180,
                    env=docker_env,
                )
                assert completed.returncode == 0, completed.stderr.decode(
                    "utf-8", "replace"
                )
            finally:
                subprocess.run(
                    [docker, "rm", "--force", name],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=30,
                    env=docker_env,
                )
            shutil.copyfile(build_dir / "rust-split-calculator", rust_binary)
            rust_binary.chmod(0o700)

        python_frame = worker.run_fd_calculator(
            [sys.executable, str(python_calculator)], seed
        )
        rust_frame = worker.run_fd_calculator([str(rust_binary)], seed)
    assert python_frame == rust_frame
    assert python_frame


def _prepare_worker_seed_test(worker, tmp_path: Path, monkeypatch) -> None:
    custody = tmp_path / "custody"
    output = tmp_path / "output"
    custody.mkdir()
    output.mkdir()
    marker = custody / "split_seed_instantiation.marker"
    marker.write_text('{"state":"PENDING"}\n', encoding="ascii")
    monkeypatch.setattr(worker, "CUSTODY", custody)
    monkeypatch.setattr(worker, "OUTPUT", output)
    monkeypatch.setattr(worker, "MARKER", marker)
    monkeypatch.setattr(worker, "SEED_FILE", custody / "split_master_seed.bin")
    monkeypatch.setattr(worker, "SEED_INTENT", custody / "split_seed_generation.intent")
    monkeypatch.setattr(worker, "SEED_COMPLETE", custody / "split_seed_generation.complete")
    monkeypatch.setattr(worker, "require_profile", lambda _purpose: None)
    monkeypatch.setattr(worker, "run_fd_calculator", lambda _argv, _seed: b"frame")


def test_seed_intent_allows_one_csprng_then_exact_seed_resume_without_redraw(
    tmp_path: Path, monkeypatch,
) -> None:
    worker = _load_worker_module()
    _prepare_worker_seed_test(worker, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(worker.os, "getrandom", lambda size: calls.append(size) or b"z" * size)
    worker.seed_and_split(synthetic=False)
    assert calls == [32]
    assert worker.SEED_INTENT.is_file()
    assert worker.SEED_FILE.read_bytes() == b"z" * 32
    assert worker.SEED_COMPLETE.is_file()
    for name in ("python-split-frame.bin", "rust-split-frame.bin", "split-mode.txt"):
        (worker.OUTPUT / name).unlink()
    monkeypatch.setattr(worker.os, "getrandom", lambda _size: pytest.fail("resume redrew seed"))
    worker.seed_and_split(synthetic=False, recovery=True)
    assert worker.SEED_FILE.read_bytes() == b"z" * 32


def test_pending_without_seed_intent_allows_exactly_one_first_genesis_on_explicit_recovery(
    tmp_path: Path, monkeypatch,
) -> None:
    worker = _load_worker_module()
    _prepare_worker_seed_test(worker, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(worker.os, "getrandom", lambda size: calls.append(size) or b"g" * size)
    worker.seed_and_split(synthetic=False, recovery=True)
    assert calls == [32]
    assert (worker.OUTPUT / "split-mode.txt").read_text(encoding="ascii") == (
        "REAL_FIRST_GENESIS_AFTER_PENDING_NO_INTENT\n"
    )
    for name in ("python-split-frame.bin", "rust-split-frame.bin", "split-mode.txt"):
        (worker.OUTPUT / name).unlink()
    monkeypatch.setattr(worker.os, "getrandom", lambda _size: pytest.fail("resume redrew seed"))
    worker.seed_and_split(synthetic=False, recovery=True)
    assert (worker.OUTPUT / "split-mode.txt").read_text(encoding="ascii") == (
        "REAL_PENDING_RESUME\n"
    )


def test_ordinary_worker_never_auto_selects_existing_completed_seed_recovery(
    tmp_path: Path, monkeypatch,
) -> None:
    worker = _load_worker_module()
    _prepare_worker_seed_test(worker, tmp_path, monkeypatch)
    monkeypatch.setattr(worker.os, "getrandom", lambda size: b"e" * size)
    worker.seed_and_split(synthetic=False, recovery=False)
    for name in ("python-split-frame.bin", "rust-split-frame.bin", "split-mode.txt"):
        (worker.OUTPUT / name).unlink()
    monkeypatch.setattr(worker.os, "getrandom", lambda _size: pytest.fail("ordinary retry redrew"))
    with pytest.raises(SystemExit):
        worker.seed_and_split(synthetic=False, recovery=False)


@pytest.mark.parametrize("seed_bytes", (None, b"", b"partial", b"x" * 33))
def test_seed_intent_with_missing_or_nonexact_seed_is_terminal_no_redraw(
    tmp_path: Path, monkeypatch, seed_bytes: bytes | None,
) -> None:
    worker = _load_worker_module()
    _prepare_worker_seed_test(worker, tmp_path, monkeypatch)
    worker.SEED_INTENT.write_text("intent\n", encoding="ascii")
    if seed_bytes is not None:
        worker.SEED_FILE.write_bytes(seed_bytes)
    monkeypatch.setattr(worker.os, "getrandom", lambda _size: pytest.fail("terminal path redrew seed"))
    with pytest.raises(SystemExit):
        worker.seed_and_split(synthetic=False, recovery=True)


def test_exact_seed_without_durable_completion_receipt_is_terminal_no_redraw(
    tmp_path: Path, monkeypatch,
) -> None:
    worker = _load_worker_module()
    _prepare_worker_seed_test(worker, tmp_path, monkeypatch)
    worker.SEED_INTENT.write_bytes(worker.SEED_INTENT_PAYLOAD)
    worker.SEED_INTENT.chmod(0o600)
    worker.SEED_FILE.write_bytes(b"q" * 32)
    worker.SEED_FILE.chmod(0o600)
    monkeypatch.setattr(worker.os, "getrandom", lambda _size: pytest.fail("terminal path redrew seed"))
    with pytest.raises(SystemExit):
        worker.seed_and_split(synthetic=False, recovery=True)


def test_crash_before_completion_receipt_makes_exact_seed_terminal(
    tmp_path: Path, monkeypatch,
) -> None:
    worker = _load_worker_module()
    _prepare_worker_seed_test(worker, tmp_path, monkeypatch)
    monkeypatch.setattr(worker.os, "getrandom", lambda size: b"c" * size)
    original = worker.exclusive_bytes

    def crash_on_completion(path, payload, mode):
        if path == worker.SEED_COMPLETE:
            raise OSError("injected-before-completion-receipt")
        return original(path, payload, mode)

    monkeypatch.setattr(worker, "exclusive_bytes", crash_on_completion)
    with pytest.raises(OSError, match="injected-before-completion"):
        worker.seed_and_split(synthetic=False, recovery=False)
    assert worker.SEED_FILE.stat().st_size == 32
    assert not worker.SEED_COMPLETE.exists()
    monkeypatch.setattr(worker, "exclusive_bytes", original)
    monkeypatch.setattr(worker.os, "getrandom", lambda _size: pytest.fail("terminal path redrew seed"))
    with pytest.raises(SystemExit):
        worker.seed_and_split(synthetic=False, recovery=True)
