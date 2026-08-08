from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType, SimpleNamespace

import pytest

from hegel_machine import phase3_m3_offline_docker_runner_v1 as runner_module
from hegel_machine.phase3_m3_dual_enumeration_supervisor_v1 import (
    COMMIT_A,
    FROZEN_IMPLEMENTATIONS,
    EnumerationInvocationV1,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def linux_tmp_path() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(
        prefix="hegel-m3-runner-test-",
        dir="/tmp",
    ) as raw:
        root = Path(raw).resolve(strict=True)
        root.chmod(0o700)
        yield root


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path.resolve(strict=True)


def _runner_and_invocation(
    tmp_path: Path, *, implementation: str = "python"
) -> tuple[
    runner_module.OfflineDockerEnumerationRunnerV1,
    EnumerationInvocationV1,
]:
    attempt = _private_directory(tmp_path / "attempt")
    formal_output = _private_directory(attempt / "formal-enumeration")
    frozen = FROZEN_IMPLEMENTATIONS[implementation]
    runner = runner_module.OfflineDockerEnumerationRunnerV1(
        repository_root=REPOSITORY_ROOT,
        attempt_root=attempt,
        implementation_qualification_receipt={"receipt_root": "a1" * 32},
    )
    invocation = EnumerationInvocationV1(
        implementation=implementation,
        implementation_id=frozen.implementation_id,
        basis_commit=COMMIT_A,
        source_root=frozen.source_root,
        binary_digest=frozen.binary_digest,
        image_ref=frozen.image_ref,
        implementation_binding_root=frozen.implementation_binding_root,
        bound_executable_locator=frozen.bound_executable_locator,
        child_dsl_spec_root=b"\x11" * 32,
        operator_semantics_root=b"\x22" * 32,
        identifier_registry_root=b"\x33" * 32,
        canonical_program_budget=50_000,
        raw_operator_application_cap=5_000_000,
        pull_policy="never",
        network_mode="none",
        output_parent=formal_output / implementation,
    )
    return runner, invocation


def _activate_journal_only(
    runner: runner_module.OfflineDockerEnumerationRunnerV1,
) -> None:
    runner._journal_root = _private_directory(  # noqa: SLF001
        runner.attempt_root / "runner-journal"
    )
    runner._journal_directory_identity = (  # noqa: SLF001
        runner_module._private_directory_identity(  # noqa: SLF001
            runner._journal_root,  # noqa: SLF001
            code=runner_module.FAIL_STABILITY,
            label="test journal root",
        )
    )
    runner._container_names = MappingProxyType(  # noqa: SLF001
        {
            "python": "hegel-m3-test-python",
            "rust": "hegel-m3-test-rust",
        }
    )
    runner._attempt_intent_sha256 = "b2" * 32  # noqa: SLF001


class _FakeDockerControlPlane:
    environment = MappingProxyType({"PATH": "/usr/bin:/bin"})

    @staticmethod
    def command(*arguments: str) -> list[str]:
        return list(arguments)


def _activate_mock_execution_context(
    runner: runner_module.OfflineDockerEnumerationRunnerV1,
) -> None:
    runner._control_plane = _FakeDockerControlPlane()  # noqa: SLF001
    runner._python_snapshot = Path("/unused-python-snapshot")  # noqa: SLF001
    runner._rust_binary = Path("/unused-rust-binary")  # noqa: SLF001


def _docker_state_payload(*, running: bool) -> bytes:
    return runner_module._canonical_json_bytes(  # noqa: SLF001
        {
            "Status": "running" if running else "exited",
            "Running": running,
            "Restarting": False,
            "ExitCode": 0 if not running else 137,
            "OOMKilled": False,
            "Error": "",
            "StartedAt": "2026-08-09T00:00:01Z",
            "FinishedAt": (
                "0001-01-01T00:00:00Z"
                if running
                else "2026-08-09T00:00:09Z"
            ),
        }
    )


def test_stable_regular_read_rejects_named_inode_replacement(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = _private_directory(linux_tmp_path / "private")
    subject = private / "subject.bin"
    subject.write_bytes(b"stable-payload")
    subject.chmod(0o600)
    displaced = private / "displaced.bin"
    original_read = os.read
    raced = False

    def replacing_read(descriptor: int, length: int) -> bytes:
        nonlocal raced
        payload = original_read(descriptor, length)
        if payload and not raced:
            raced = True
            subject.rename(displaced)
            subject.write_bytes(b"hostile-replay")
            subject.chmod(0o600)
        return payload

    monkeypatch.setattr(os, "read", replacing_read)
    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as captured:
        runner_module._stable_regular_read(  # noqa: SLF001
            subject,
            maximum_bytes=1024,
            code=runner_module.FAIL_STABILITY,
        )
    assert captured.value.code == runner_module.FAIL_STABILITY
    assert "changed while read" in captured.value.detail


def test_exclusive_write_is_exact_replay_safe_and_cleans_partial(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = _private_directory(linux_tmp_path / "private")
    target = private / "evidence.json"
    payload = b'{"evidence":true}\n'

    assert (
        runner_module._exclusive_write(target, payload, mode=0o600)  # noqa: SLF001
        == "CREATED_NEW"
    )
    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o777 == 0o600
    assert (
        runner_module._exclusive_write(  # noqa: SLF001
            target,
            payload,
            mode=0o600,
            allow_identical_existing=True,
        )
        == "EXISTING_IDENTICAL"
    )
    with pytest.raises(runner_module.M3OfflineDockerRunnerError):
        runner_module._exclusive_write(  # noqa: SLF001
            target,
            b"different\n",
            mode=0o600,
            allow_identical_existing=True,
        )
    target.chmod(0o644)
    with pytest.raises(runner_module.M3OfflineDockerRunnerError):
        runner_module._exclusive_write(  # noqa: SLF001
            target,
            payload,
            mode=0o600,
            allow_identical_existing=True,
        )
    target.chmod(0o600)

    partial = private / "partial.bin"
    original_fsync = os.fsync
    fsync_calls = 0

    def fail_first_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 1:
            raise OSError("injected durability failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_first_fsync)
    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as captured:
        runner_module._exclusive_write(partial, b"partial")  # noqa: SLF001
    assert captured.value.code == runner_module.FAIL_EXECUTION
    assert not partial.exists()
    assert fsync_calls >= 2


def test_verified_executable_is_copied_and_decoupled_from_live_source(
    linux_tmp_path: Path,
) -> None:
    source_root = _private_directory(linux_tmp_path / "source")
    destination_root = _private_directory(linux_tmp_path / "destination")
    source = source_root / "enumerator"
    payload = b"frozen-rust-executable"
    source.write_bytes(payload)
    source.chmod(0o555)
    destination = destination_root / "enumerator"

    copied_path, copied_payload = runner_module._snapshot_verified_executable(  # noqa: SLF001
        source.resolve(strict=True),
        destination,
        expected_digest=hashlib.sha256(payload).digest(),
        expected_size=len(payload),
    )
    assert copied_path == destination
    assert copied_payload == payload
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o555

    source.rename(source_root / "old-enumerator")
    source.write_bytes(b"changed-live-executable")
    source.chmod(0o555)
    assert destination.read_bytes() == payload


def test_input_tree_digest_binds_modes_and_rejects_writable_metadata(
    linux_tmp_path: Path,
) -> None:
    tree = _private_directory(linux_tmp_path / "tree")
    child = tree / "source.py"
    child.write_bytes(b"VALUE = 1\n")
    child.chmod(0o444)
    frozen_digest = runner_module._tree_digest(tree)  # noqa: SLF001

    child.chmod(0o400)
    assert runner_module._tree_digest(tree) != frozen_digest  # noqa: SLF001
    child.chmod(0o666)
    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as captured:
        runner_module._tree_digest(tree)  # noqa: SLF001
    assert captured.value.code == runner_module.FAIL_STABILITY


def test_invocation_validation_confines_output_and_digest_binds_roots(
    linux_tmp_path: Path,
) -> None:
    runner, invocation = _runner_and_invocation(linux_tmp_path)

    runner._validate_invocation(invocation)  # noqa: SLF001
    original_digest = runner_module._invocation_digest_v1(  # noqa: SLF001
        invocation,
        attempt_root=runner.attempt_root,
    )
    changed_root = replace(invocation, child_dsl_spec_root=b"\x44" * 32)
    assert (
        runner_module._invocation_digest_v1(  # noqa: SLF001
            changed_root,
            attempt_root=runner.attempt_root,
        )
        != original_digest
    )

    escaped = replace(
        invocation,
        output_parent=runner.attempt_root.parent / "escaped-python-output",
    )
    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as captured:
        runner._validate_invocation(escaped)  # noqa: SLF001
    assert captured.value.code == runner_module.FAIL_INVOCATION


def test_docker_command_disables_restart_without_removing_evidence_container(
    linux_tmp_path: Path,
) -> None:
    runner, invocation = _runner_and_invocation(linux_tmp_path)
    _activate_journal_only(runner)
    _activate_mock_execution_context(runner)
    runner._seccomp_snapshot = Path("/immutable/seccomp.json")  # noqa: SLF001

    command = runner._docker_command(  # noqa: SLF001
        invocation,
        options=(),
        command=("true",),
        environment={"PATH": "/usr/bin:/bin"},
    )
    assert "--restart=no" in command
    assert "--rm" not in command
    assert "--name=hegel-m3-test-python" in command


def test_completed_journal_resumes_only_the_exact_invocation(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, invocation = _runner_and_invocation(linux_tmp_path)
    _activate_journal_only(runner)
    started = 101
    finished = 109
    invocation_sha256 = runner._claim_execution_start(  # noqa: SLF001
        invocation,
        started_at_unix_seconds=started,
    )
    _private_directory(invocation.output_parent)
    stdout = b'{"closure_status":"DSL_TOO_LARGE"}\n'
    stderr = b""
    runner_module._exclusive_write(  # noqa: SLF001
        invocation.output_parent / "execution-stdout.json", stdout
    )
    runner_module._exclusive_write(  # noqa: SLF001
        invocation.output_parent / "execution-stderr.bin", stderr
    )
    container_state: dict[str, object] = {
        "Status": "exited",
        "Running": False,
        "ExitCode": 0,
        "OOMKilled": False,
        "Error": "",
        "StartedAt": "2026-08-09T00:00:01Z",
        "FinishedAt": "2026-08-09T00:00:09Z",
    }
    process_completion: dict[str, object] = {
        "schema": runner_module.COMPLETION_MARKER_SCHEMA,
        "implementation": invocation.implementation,
        "implementation_id": invocation.implementation_id,
        "container_name": "hegel-m3-test-python",
        "invocation_sha256": invocation_sha256,
        "attempt_intent_sha256": "b2" * 32,
        "started_at_unix_seconds": started,
        "finished_at_unix_seconds": finished,
        "process_exit_code": 0,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "pull_policy": "never",
        "network_mode": "none",
        "docker_started_at": container_state["StartedAt"],
        "docker_finished_at": container_state["FinishedAt"],
        "docker_oom_killed": False,
        "docker_error": "",
    }
    completion_payload = runner_module._canonical_json_bytes(  # noqa: SLF001
        process_completion
    )
    runner_module._exclusive_write(  # noqa: SLF001
        invocation.output_parent / "process-completion.json",
        completion_payload,
    )
    runner._publish_completion_journal(  # noqa: SLF001
        invocation,
        invocation_sha256=invocation_sha256,
        started_at_unix_seconds=started,
        finished_at_unix_seconds=finished,
        process_completion_payload=completion_payload,
    )
    monkeypatch.setattr(
        runner,
        "_inspect_container_state",
        lambda _invocation: container_state,
    )

    resumed = runner._resume_completed_result(invocation)  # noqa: SLF001
    assert resumed is not None
    assert resumed.started_at_unix_seconds == started
    assert resumed.finished_at_unix_seconds == finished
    assert resumed.report == {"closure_status": "DSL_TOO_LARGE"}

    changed = replace(invocation, child_dsl_spec_root=b"\x99" * 32)
    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as captured:
        runner._resume_completed_result(changed)  # noqa: SLF001
    assert captured.value.code == runner_module.FAIL_RESUME


def test_incomplete_started_journal_blocks_silent_rerun(
    linux_tmp_path: Path,
) -> None:
    runner, invocation = _runner_and_invocation(linux_tmp_path)
    _activate_journal_only(runner)
    runner._claim_execution_start(  # noqa: SLF001
        invocation,
        started_at_unix_seconds=101,
    )

    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as captured:
        runner._resume_completed_result(invocation)  # noqa: SLF001
    assert captured.value.code == runner_module.FAIL_RESUME
    assert "cannot be rerun" in captured.value.detail


def test_timeout_failure_stops_named_container_before_return_and_journals(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, invocation = _runner_and_invocation(linux_tmp_path)
    _activate_journal_only(runner)
    _activate_mock_execution_context(runner)
    container_name = "hegel-m3-test-python"
    state = {"running": True}
    commands: list[tuple[str, ...]] = []

    def fake_control_run(
        command: list[str],
        *,
        code: str,
        timeout: int,
        environment: object,
    ) -> SimpleNamespace:
        del code, timeout, environment
        arguments = tuple(command)
        commands.append(arguments)
        if arguments == ("docker-run",):
            raise runner_module.M3OfflineDockerRunnerError(
                runner_module.FAIL_EXECUTION,
                "docker run timed out",
            )
        if arguments[:2] == ("container", "ls"):
            return SimpleNamespace(
                stdout=(f'"{container_name}"\n').encode("ascii"),
                stderr=b"",
                returncode=0,
            )
        if arguments[:2] == ("container", "inspect"):
            return SimpleNamespace(
                stdout=_docker_state_payload(running=state["running"]),
                stderr=b"",
                returncode=0,
            )
        if arguments[:2] == ("container", "stop"):
            state["running"] = False
            return SimpleNamespace(
                stdout=(container_name + "\n").encode("ascii"),
                stderr=b"",
                returncode=0,
            )
        raise AssertionError(f"unexpected control command: {arguments}")

    monkeypatch.setattr(runner_module._qualification, "_run", fake_control_run)
    monkeypatch.setattr(
        runner,
        "verify_inputs_stable_v1",
        lambda: None,
    )
    monkeypatch.setattr(
        runner,
        "_docker_command",
        lambda _invocation, **_kwargs: ["docker-run"],
    )
    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as captured:
        runner(invocation)
    assert captured.value.code == runner_module.FAIL_EXECUTION
    assert state["running"] is False
    assert any(command[:2] == ("container", "stop") for command in commands)
    assert not any("rm" in command for command in commands)

    failure_paths = sorted(
        runner._journal_root.glob("python-failure-*.json")  # noqa: SLF001
    )
    assert len(failure_paths) == 1
    failure = json.loads(failure_paths[0].read_text(encoding="ascii"))
    assert failure["terminalization_status"] == "SAFE_CONTAINER_NOT_RUNNING"
    assert failure["safe_to_terminalize_execution"] is True
    assert failure["container_removal_attempted"] is False

def test_named_python_probe_timeout_is_stopped_and_journaled(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _invocation = _runner_and_invocation(linux_tmp_path)
    _activate_journal_only(runner)
    _activate_mock_execution_context(runner)
    runner._seccomp_snapshot = Path("/immutable/seccomp.json")  # noqa: SLF001
    runner._probe_container_name = "hegel-m3-test-python-probe"  # noqa: SLF001
    state = {"exists": False, "running": False}
    commands: list[tuple[str, ...]] = []

    def fake_control_run(
        command: list[str],
        *,
        code: str,
        timeout: int,
        environment: object,
    ) -> SimpleNamespace:
        del code, timeout, environment
        arguments = tuple(command)
        commands.append(arguments)
        if arguments[0] == "run":
            state.update(exists=True, running=True)
            raise runner_module.M3OfflineDockerRunnerError(
                runner_module.FAIL_PREFLIGHT,
                "named Python probe timed out",
            )
        if arguments[:2] == ("container", "ls"):
            stdout = (
                b'"hegel-m3-test-python-probe"\n'
                if state["exists"]
                else b""
            )
            return SimpleNamespace(stdout=stdout, stderr=b"", returncode=0)
        if arguments[:2] == ("container", "inspect"):
            return SimpleNamespace(
                stdout=_docker_state_payload(running=state["running"]),
                stderr=b"",
                returncode=0,
            )
        if arguments[:2] == ("container", "stop"):
            state["running"] = False
            return SimpleNamespace(
                stdout=b"hegel-m3-test-python-probe\n",
                stderr=b"",
                returncode=0,
            )
        raise AssertionError(f"unexpected control command: {arguments}")

    monkeypatch.setattr(runner_module._qualification, "_run", fake_control_run)
    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as captured:
        runner._run_or_resume_named_python_probe_v1()  # noqa: SLF001
    assert captured.value.code == runner_module.FAIL_PREFLIGHT
    assert state == {"exists": True, "running": False}
    run_commands = [command for command in commands if command[0] == "run"]
    assert len(run_commands) == 1
    assert "--name=hegel-m3-test-python-probe" in run_commands[0]
    assert "--restart=no" in run_commands[0]
    assert "--rm" not in run_commands[0]
    assert any(command[:2] == ("container", "stop") for command in commands)

    failures = sorted(
        runner._journal_root.glob(  # noqa: SLF001
            "python-probe-failure-*.json"
        )
    )
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="ascii"))
    assert failure["terminalization_status"] == "SAFE_CONTAINER_NOT_RUNNING"
    assert failure["safe_to_terminalize_execution"] is True
    assert failure["container_removal_attempted"] is False

    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as replayed:
        runner._run_or_resume_named_python_probe_v1()  # noqa: SLF001
    assert replayed.value.code == runner_module.FAIL_PREFLIGHT
    assert len([command for command in commands if command[0] == "run"]) == 1
    assert state == {"exists": True, "running": False}
    assert len(
        list(
            runner._journal_root.glob(  # noqa: SLF001
                "python-probe-failure-*.json"
            )
        )
    ) == 2


def test_named_python_probe_completion_replays_without_second_container(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, _invocation = _runner_and_invocation(linux_tmp_path)
    _activate_journal_only(runner)
    _activate_mock_execution_context(runner)
    runner._seccomp_snapshot = Path("/immutable/seccomp.json")  # noqa: SLF001
    runner._probe_container_name = "hegel-m3-test-python-probe"  # noqa: SLF001
    state = {"exists": False, "running": False}
    run_count = 0
    stdout = runner_module._canonical_json_bytes(  # noqa: SLF001
        {
            "binary_path": "/usr/local/bin/python3.11",
            "binary_sha256": FROZEN_IMPLEMENTATIONS["python"].binary_digest.hex(),
            "version": "3.11.0 deterministic test",
        }
    )

    def fake_control_run(
        command: list[str],
        *,
        code: str,
        timeout: int,
        environment: object,
    ) -> SimpleNamespace:
        nonlocal run_count
        del code, timeout, environment
        arguments = tuple(command)
        if arguments[0] == "run":
            run_count += 1
            state.update(exists=True, running=False)
            return SimpleNamespace(stdout=stdout, stderr=b"", returncode=0)
        if arguments[:2] == ("container", "ls"):
            listed = (
                b'"hegel-m3-test-python-probe"\n'
                if state["exists"]
                else b""
            )
            return SimpleNamespace(stdout=listed, stderr=b"", returncode=0)
        if arguments[:2] == ("container", "inspect"):
            return SimpleNamespace(
                stdout=_docker_state_payload(running=False),
                stderr=b"",
                returncode=0,
            )
        raise AssertionError(f"unexpected control command: {arguments}")

    monkeypatch.setattr(runner_module._qualification, "_run", fake_control_run)
    first = runner._run_or_resume_named_python_probe_v1()  # noqa: SLF001
    second = runner._run_or_resume_named_python_probe_v1()  # noqa: SLF001
    assert first == second
    assert first[1] == FROZEN_IMPLEMENTATIONS["python"].binary_digest
    assert run_count == 1
    assert (runner.attempt_root / "python-runtime-probe-stdout.json").is_file()
    assert (
        runner._journal_root / "python-probe-completed.json"  # noqa: SLF001
    ).is_file()


def test_unstoppable_named_container_upgrades_to_unsafe_terminalization(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, invocation = _runner_and_invocation(linux_tmp_path)
    _activate_journal_only(runner)
    _activate_mock_execution_context(runner)
    runner._claim_execution_start(  # noqa: SLF001
        invocation,
        started_at_unix_seconds=101,
    )
    container_name = "hegel-m3-test-python"
    commands: list[tuple[str, ...]] = []

    def fake_control_run(
        command: list[str],
        *,
        code: str,
        timeout: int,
        environment: object,
    ) -> SimpleNamespace:
        del code, timeout, environment
        arguments = tuple(command)
        commands.append(arguments)
        if arguments[:2] == ("container", "ls"):
            return SimpleNamespace(
                stdout=(f'"{container_name}"\n').encode("ascii"),
                stderr=b"",
                returncode=0,
            )
        if arguments[:2] == ("container", "inspect"):
            return SimpleNamespace(
                stdout=_docker_state_payload(running=True),
                stderr=b"",
                returncode=0,
            )
        if arguments[:2] in {
            ("container", "stop"),
            ("container", "kill"),
        }:
            raise runner_module.M3OfflineDockerRunnerError(
                runner_module.FAIL_TERMINALIZE,
                "injected control-plane refusal",
            )
        raise AssertionError(f"unexpected control command: {arguments}")

    def simulated_docker_run_timeout(
        _invocation: EnumerationInvocationV1,
    ) -> None:
        raise runner_module.M3OfflineDockerRunnerError(
            runner_module.FAIL_EXECUTION,
            "docker run timed out",
        )

    monkeypatch.setattr(runner_module._qualification, "_run", fake_control_run)
    monkeypatch.setattr(
        runner,
        "_execute_or_resume_v1",
        simulated_docker_run_timeout,
    )
    with pytest.raises(runner_module.M3OfflineDockerRunnerError) as captured:
        runner(invocation)
    assert captured.value.code == runner_module.FAIL_TERMINALIZE
    assert any(command[:2] == ("container", "stop") for command in commands)
    assert any(command[:2] == ("container", "kill") for command in commands)
    assert not any("rm" in command for command in commands)

    failure_paths = sorted(
        runner._journal_root.glob("python-failure-*.json")  # noqa: SLF001
    )
    assert len(failure_paths) == 1
    failure = json.loads(failure_paths[0].read_text(encoding="ascii"))
    assert failure["terminalization_status"] == "UNSAFE_CONTAINER_STILL_RUNNING"
    assert failure["safe_to_terminalize_execution"] is False
    assert failure["container_removal_attempted"] is False
