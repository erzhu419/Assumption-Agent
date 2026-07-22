from __future__ import annotations

import json
from pathlib import Path
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import tatqa_p21_label_free_runtime_v1 as features
from replication_runtime.tatqa_p21_v1 import formal_runtime as runtime
from replication_runtime.tatqa_p21_v1 import hipporag_contract, typed_plan_contract


def _outer_launcher_environment(uid: int = 1001) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": "1",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus",
        "HF_HUB_OFFLINE": "1",
        "HOME": "/home/erzhu419",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_RUNTIME_DIR": f"/run/user/{uid}",
    }


def _install_launcher_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    environment: dict[str, str] | None = None,
    socket_uid: int = 1001,
    socket_mode: int = stat.S_IFSOCK | 0o600,
) -> dict[str, str]:
    uid = 1001
    values = _outer_launcher_environment(uid) if environment is None else environment
    monkeypatch.setattr(runtime.os, "getuid", lambda: uid)
    monkeypatch.setattr(runtime.os, "environ", values)
    monkeypatch.setattr(
        runtime,
        "_user_systemd_socket_metadata",
        lambda _path: SimpleNamespace(st_mode=socket_mode, st_uid=socket_uid),
    )
    return values


@pytest.fixture
def launcher_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_launcher_environment",
        lambda: {
            "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1001/bus",
            "HOME": "/home/erzhu419",
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "XDG_RUNTIME_DIR": "/run/user/1001",
        },
    )


def _synthetic_launcher_receipt() -> dict[str, object]:
    def phase_receipt(phase: str) -> dict[str, object]:
        post = phase != "entry"
        body: dict[str, object] = {
            "schema": runtime.USER_SYSTEMD_LAUNCHER_PHASE_RECEIPT_SCHEMA,
            "phase": phase,
            "status": "verified_exact_environment_and_user_bus",
            "effective_uid_sha256": "1" * 64,
            "variable_name_allowlist": list(
                runtime.USER_SYSTEMD_POST_RUNTIME_ENVIRONMENT_VARIABLE_ALLOWLIST
                if post
                else runtime.USER_SYSTEMD_ENTRY_ENVIRONMENT_VARIABLE_ALLOWLIST
            ),
            "cuda_visible_devices_state": (
                "empty_after_minilm" if phase == "post_minilm" else "physical_GPU_1"
            ),
            "cuda_module_loading_LAZY_present": post,
            "all_other_frozen_values_exact": True,
            "path_address_and_socket_path_SHA256_values": {
                "dbus_session_bus_address": "2" * 64,
                "session_bus_socket_path": "3" * 64,
                "systemd_private_socket_path": "4" * 64,
                "xdg_runtime_dir": "5" * 64,
            },
            "socket_type_and_effective_uid_ownership_booleans": {
                role: {
                    "is_owned_by_effective_uid": True,
                    "is_unix_socket": True,
                }
                for role in ("session_bus", "systemd_private")
            },
            "raw_environment_values_or_credentials_recorded": False,
            "provider_or_api_credentials_read": False,
        }
        return {**body, "self_sha256": runtime.stable_hash(body)}

    return runtime.user_systemd_launcher_capability_receipt(
        entry=phase_receipt("entry"),
        post_runtime_inventory=phase_receipt("post_runtime_inventory"),
    )


class _Paths(SimpleNamespace):
    def checked(self):
        return self


def _paths(tmp_path: Path) -> _Paths:
    work = tmp_path / "work"
    work.mkdir()
    typed_executable = tmp_path / "typed-runtime" / "bin" / "python"
    typed_executable.parent.mkdir(parents=True)
    typed_executable.write_bytes(b"typed")
    hippo_executable = tmp_path / "hippo-runtime" / "bin" / "python"
    hippo_executable.parent.mkdir(parents=True)
    hippo_executable.write_bytes(b"hippo")
    return _Paths(
        project_root=tmp_path,
        typed_runtime_python=typed_executable,
        hippo_runtime_python=hippo_executable,
        qwen_model=tmp_path / "qwen",
        minilm_asset_manifest=tmp_path / "minilm.json",
        minilm_model=tmp_path / "minilm",
        hippo_llm_model=tmp_path / "hippo-llm",
        hippo_embedding_model=tmp_path / "hippo-embed",
        hipporag_source=tmp_path / "hipporag-source",
        hippo_attestation=tmp_path / "hippo-attestation.json",
        fingerprint_manifest=tmp_path / "fingerprint.json",
        work_root=work,
    )


def _runtime_item() -> features.LabelFreeRuntimeItem:
    return features.LabelFreeRuntimeItem(
        item_id="f" * 64,
        question="Compare Acme revenue for 2023 and 2024.",
        units=(
            features.RuntimeUnit("T:0", "TABLE_HEADER|C0=year||C1=revenue"),
            features.RuntimeUnit("T:1", "TABLE_ROW_1|year=2023||revenue=100"),
            features.RuntimeUnit("T:2", "TABLE_ROW_2|year=2024||revenue=130"),
            features.RuntimeUnit("P:1", "PARAGRAPH_1|Acme reported annual revenue."),
            features.RuntimeUnit("P:2", "PARAGRAPH_2|Costs were also discussed."),
        ),
    )


def _unit_closure(unit_name: str) -> dict[str, object]:
    empty = runtime.hashlib.sha256(b"").hexdigest()
    return {
        "schema": runtime.SYSTEMD_UNIT_CLOSURE_SCHEMA,
        "unit_name_sha256": runtime._unit_name_sha256(unit_name),
        "load_state": "not-found",
        "active_state": "inactive",
        "sub_state": "dead",
        "main_pid": 0,
        "control_group_sha256": empty,
        "control_group_process_count": 0,
        "control_group_thread_count": 0,
        "systemctl_show_returncode": 0,
        "systemctl_show_stdout_sha256": empty,
        "systemctl_show_stderr_sha256": empty,
        "systemctl_reset_failed_returncode": 1,
        "systemctl_reset_failed_stdout_sha256": empty,
        "systemctl_reset_failed_stderr_sha256": empty,
    }


def _start_policy(unit_name: str) -> dict[str, object]:
    empty = runtime.hashlib.sha256(b"").hexdigest()
    return {
        "schema": runtime.SYSTEMD_START_POLICY_SCHEMA,
        "unit_name_sha256": runtime._unit_name_sha256(unit_name),
        "load_state": "loaded",
        "active_state": "active",
        "sub_state": "running",
        "main_pid": 124,
        "control_group_sha256": empty,
        "tasks_max": 3,
        "kill_mode": "control-group",
        "systemctl_show_returncode": 0,
        "systemctl_show_stdout_sha256": empty,
        "systemctl_show_stderr_sha256": empty,
    }


class _FakeSystemdProcess:
    def __init__(self, machine: "_FakeSystemdMachine") -> None:
        self.machine = machine
        self.returncode = None
        self._killed = threading.Event()
        self._communicate_count = 0

    def communicate(self, timeout=None):
        self._communicate_count += 1
        self.machine.communicate_entered.set()
        if self.machine.mode == "timeout" and self._communicate_count == 1:
            raise subprocess.TimeoutExpired(["systemd-run"], timeout)
        if self.machine.mode == "blocking":
            if not self._killed.wait(timeout=5):
                raise AssertionError("fake systemd-run client was not terminated")
        if self._killed.is_set():
            self.returncode = -9
            return b"", b""
        self.machine.exists = False
        self.machine.phase = "absent"
        self.returncode = 0
        return b'worker terminal\n', b""

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9
        self._killed.set()

    def wait(self, timeout=None):
        if self.returncode is None and not self._killed.wait(timeout=timeout):
            raise subprocess.TimeoutExpired(["systemd-run"], timeout)
        return self.returncode


class _FakeSystemdMachine:
    def __init__(
        self,
        *,
        mode: str = "normal",
        stop_leaves_child: bool = True,
        tasks_max: int = 3,
    ) -> None:
        self.mode = mode
        self.stop_leaves_child = stop_leaves_child
        self.tasks_max = tasks_max
        self.exists = False
        self.phase = "absent"
        self.unit_name = None
        self.commands: list[list[str]] = []
        self.process = None
        self.communicate_entered = threading.Event()

    def popen(self, command, **kwargs):
        self.commands.append(list(command))
        units = [row.split("=", 1)[1] for row in command if row.startswith("--unit=")]
        assert len(units) == 1
        assert "TasksMax=3" in command
        assert "KillMode=control-group" in command
        self.unit_name = units[0]
        self.exists = True
        self.phase = "active"
        self.process = _FakeSystemdProcess(self)
        return self.process

    def cgroup_counts(self, control_group):
        if not control_group or not self.exists:
            return 0, 0
        if self.phase == "active":
            return 2, 3
        return 1, 1

    def systemctl(self, command, **_kwargs):
        self.commands.append(list(command))
        action = command[2]
        if action == "show":
            properties = [
                row.split("=", 1)[1]
                for row in command
                if row.startswith("--property=")
            ]
            if self.exists:
                active = "active" if self.phase == "active" else "deactivating"
                sub = "running" if self.phase == "active" else "stop-sigterm"
                values = {
                    "LoadState": "loaded",
                    "ActiveState": active,
                    "SubState": sub,
                    "MainPID": "4242",
                    "ControlGroup": "/user.slice/fake-p18.service",
                    "TasksMax": str(self.tasks_max),
                    "KillMode": "control-group",
                }
            else:
                values = {
                    "LoadState": "not-found",
                    "ActiveState": "inactive",
                    "SubState": "dead",
                    "MainPID": "0",
                    "ControlGroup": "",
                    "TasksMax": "infinity",
                    "KillMode": "control-group",
                }
            stdout = "".join(f"{key}={values[key]}\n" for key in properties).encode()
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        if action == "stop":
            if self.exists:
                if self.stop_leaves_child:
                    self.phase = "lingering"
                else:
                    self.exists = False
                    self.phase = "absent"
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if action == "kill":
            self.exists = False
            self.phase = "absent"
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if action == "reset-failed":
            if self.exists:
                self.exists = False
                self.phase = "absent"
                return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
            unit = command[-1]
            return SimpleNamespace(
                returncode=1,
                stdout=b"",
                stderr=(
                    "Failed to reset failed state of unit "
                    f"{unit}: Unit {unit} not loaded.\n"
                ).encode(),
            )
        raise AssertionError(f"unexpected systemctl action: {action}")


def _fake_supervisor(machine: _FakeSystemdMachine):
    return runtime._SystemdWorkerSupervisor(
        popen_factory=machine.popen,
        systemctl_runner=machine.systemctl,
        sleeper=lambda _seconds: None,
        cgroup_counter=machine.cgroup_counts,
    )


def _fake_supervised_run(machine: _FakeSystemdMachine, *, suffix: str):
    supervisor = _fake_supervisor(machine)
    unit = runtime._worker_unit_name(
        role="HippoRAG", identity={"fake_state_machine": suffix}
    )
    result = runtime._run_worker(
        command=["/bin/true"],
        environment={"LANG": "C.UTF-8"},
        inaccessible_paths=(),
        timeout=1,
        role="HippoRAG",
        unit_name=unit,
        supervisor=supervisor,
    )
    return supervisor, unit, result


def test_systemd_preflight_freezes_network_denial_and_clean_environment(
    monkeypatch,
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(runtime, "SYSTEMD_RUN", Path("/usr/bin/true"))
    monkeypatch.setattr(runtime, "ENV_EXECUTABLE", Path("/usr/bin/env"))
    environment = _outer_launcher_environment()
    environment["CUDA_MODULE_LOADING"] = "LAZY"
    _install_launcher_state(monkeypatch, environment=environment)
    receipt = runtime.systemd_network_preflight(runner=fake_run)
    command, kwargs = calls[0]
    assert "IPAddressDeny=any" in command
    assert "RestrictAddressFamilies=AF_UNIX" in command
    assert "--ignore-environment" in command
    assert "LANG=C.UTF-8" in command
    assert not any("API" in value or "RUOLI" in value for value in command)
    assert receipt["network_properties"] == list(runtime.SYSTEMD_NETWORK_PROPERTIES)
    assert kwargs["timeout"] == 30
    isolated = runtime._systemd_prefix(
        inaccessible_paths=(Path("/tmp/source"), Path("/tmp/acquisition"))
    )
    assert "InaccessiblePaths=-/tmp/source" in isolated
    assert "InaccessiblePaths=-/tmp/acquisition" in isolated
    hippo_unit = runtime._worker_unit_name(role="HippoRAG", identity={"x": "y"})
    assert hippo_unit.startswith("tatqa-p21-hippo-")
    bounded = runtime._systemd_prefix(
        unit_name=hippo_unit, tasks_max=runtime.HIPPORAG_SYSTEMD_TASKS_MAX
    )
    assert f"--unit={hippo_unit}" in bounded
    assert "TasksMax=3" in bounded
    assert "KillMode=control-group" in bounded


def test_user_systemd_launcher_receipt_binds_exact_safe_outer_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_launcher_state(monkeypatch)
    receipt = runtime.user_systemd_launcher_phase_receipt(phase="entry")
    assert runtime.validate_user_systemd_launcher_phase_receipt(
        receipt, expected_phase="entry"
    ) == (
        receipt["self_sha256"]
    )
    assert receipt["schema"] == (
        "tatqa_p21_user_systemd_launcher_phase_receipt_v1"
    )
    assert receipt["variable_name_allowlist"] == list(
        runtime.USER_SYSTEMD_OUTER_ENVIRONMENT_VARIABLE_ALLOWLIST
    )
    assert receipt["effective_uid_sha256"] == runtime.hashlib.sha256(
        b"1001"
    ).hexdigest()
    serialized = runtime.canonical_json_bytes(receipt).decode("ascii")
    assert "/run/user/1001" not in serialized
    assert "unix:path=" not in serialized
    assert "/home/erzhu419" not in serialized
    assert "credential" in serialized


@pytest.mark.parametrize(
    "forbidden_name",
    (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "RUOLI_API_KEY",
        "SERVICE_SECRET",
        "SERVICE_TOKEN",
    ),
)
def test_user_systemd_launcher_rejects_any_unlisted_or_credential_environment_name(
    monkeypatch: pytest.MonkeyPatch, forbidden_name: str
) -> None:
    environment = _outer_launcher_environment()
    environment[forbidden_name] = "must-never-be-read-or-recorded"
    _install_launcher_state(monkeypatch, environment=environment)
    with pytest.raises(
        runtime.TatqaP21FormalRuntimeError, match="allowlist"
    ) as caught:
        runtime.user_systemd_launcher_phase_receipt(phase="entry")
    assert "must-never" not in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("XDG_RUNTIME_DIR", "/run/user/999"),
        ("DBUS_SESSION_BUS_ADDRESS", "unix:path=/run/user/999/bus"),
        ("HOME", "/tmp/not-the-frozen-home"),
        ("PATH", "/tmp/bin"),
        ("HF_HUB_OFFLINE", "0"),
        ("CUDA_VISIBLE_DEVICES", "0"),
    ),
)
def test_user_systemd_launcher_rejects_outer_value_drift(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    environment = _outer_launcher_environment()
    environment[field] = value
    _install_launcher_state(monkeypatch, environment=environment)
    with pytest.raises(runtime.TatqaP21FormalRuntimeError):
        runtime.user_systemd_launcher_phase_receipt(phase="entry")


@pytest.mark.parametrize(
    ("socket_uid", "socket_mode"),
    (
        (1002, stat.S_IFSOCK | 0o600),
        (1001, stat.S_IFREG | 0o600),
    ),
)
def test_user_systemd_launcher_rejects_wrong_socket_owner_or_type(
    monkeypatch: pytest.MonkeyPatch, socket_uid: int, socket_mode: int
) -> None:
    _install_launcher_state(
        monkeypatch, socket_uid=socket_uid, socket_mode=socket_mode
    )
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="socket"):
        runtime.user_systemd_launcher_phase_receipt(phase="entry")


def test_worker_launcher_allows_only_the_frozen_minilm_cuda_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _install_launcher_state(monkeypatch)
    entry = runtime.user_systemd_launcher_phase_receipt(phase="entry")
    environment["CUDA_MODULE_LOADING"] = "LAZY"
    post_inventory = runtime.user_systemd_launcher_phase_receipt(
        phase="post_runtime_inventory"
    )
    capability = runtime.user_systemd_launcher_capability_receipt(
        entry=entry, post_runtime_inventory=post_inventory
    )
    assert runtime.validate_user_systemd_launcher_capability_receipt(capability)
    assert list(capability["phase_receipts"]) == [
        "entry",
        "post_runtime_inventory",
    ]
    inventory_launcher = runtime._launcher_environment()
    assert set(inventory_launcher) == set(
        runtime.USER_SYSTEMD_NESTED_CLIENT_ENVIRONMENT_VARIABLE_ALLOWLIST
    )
    assert "CUDA_MODULE_LOADING" not in inventory_launcher
    environment["CUDA_VISIBLE_DEVICES"] = ""
    post_minilm = runtime.user_systemd_launcher_phase_receipt(phase="post_minilm")
    assert runtime.validate_user_systemd_launcher_phase_receipt(
        post_minilm, expected_phase="post_minilm"
    )
    launcher = runtime._launcher_environment()
    assert launcher["XDG_RUNTIME_DIR"] == "/run/user/1001"
    assert launcher["DBUS_SESSION_BUS_ADDRESS"] == (
        "unix:path=/run/user/1001/bus"
    )
    environment["OPENAI_API_KEY"] = "must-never-be-read"
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="allowlist"):
        runtime._launcher_environment()


@pytest.mark.parametrize(
    ("mutation", "phase"),
    (
        ({"CUDA_MODULE_LOADING": "EAGER"}, "post_runtime_inventory"),
        ({"CUDA_MODULE_LOADING": "LAZY", "EXTRA": "x"}, "post_runtime_inventory"),
        ({"CUDA_MODULE_LOADING": "LAZY", "CUDA_VISIBLE_DEVICES": "0"}, "post_minilm"),
    ),
)
def test_launcher_phase_rejects_wrong_or_additional_mutation(
    monkeypatch: pytest.MonkeyPatch, mutation: dict[str, str], phase: str
) -> None:
    environment = _outer_launcher_environment()
    environment.update(mutation)
    if phase == "post_minilm" and "CUDA_VISIBLE_DEVICES" not in mutation:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    _install_launcher_state(monkeypatch, environment=environment)
    with pytest.raises(runtime.TatqaP21FormalRuntimeError):
        runtime.user_systemd_launcher_phase_receipt(phase=phase)


def test_launcher_receipt_validator_rejects_tampered_safety_claim() -> None:
    receipt = _synthetic_launcher_receipt()
    receipt["raw_environment_values_or_credentials_recorded"] = True
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="drifted"):
        runtime.validate_user_systemd_launcher_capability_receipt(receipt)


def test_worker_environments_are_exactly_offline_and_thread_bounded(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    root = tmp_path / "stage"
    for name in ("home", "hf", "tmp"):
        (root / name).mkdir(parents=True, exist_ok=True)
    qwen = runtime._worker_environment(paths, root, role="Qwen")
    hippo = runtime._worker_environment(paths, root, role="HippoRAG")
    assert qwen["CUDA_VISIBLE_DEVICES"] == "1"
    assert hippo["CUDA_VISIBLE_DEVICES"] == ""
    assert qwen["PATH"].startswith(f"{paths.typed_runtime_python.parent}:")
    assert hippo["PATH"].startswith(f"{paths.hippo_runtime_python.parent}:")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert hippo[key] == "1"
    assert qwen["HF_HUB_OFFLINE"] == hippo["TRANSFORMERS_OFFLINE"] == "1"
    assert not any("API" in key or "RUOLI" in key for key in (*qwen, *hippo))


def test_named_unit_success_attests_kernel_policy_and_live_closure(
    launcher_state: None,
) -> None:
    machine = _FakeSystemdMachine(mode="normal")
    supervisor, unit, (_stdout, _stderr, policy, closure) = _fake_supervised_run(
        machine, suffix="success"
    )
    assert policy["tasks_max"] == 3
    assert policy["kill_mode"] == "control-group"
    assert policy["unit_name_sha256"] == runtime._unit_name_sha256(unit)
    assert closure["load_state"] == "not-found"
    assert closure["main_pid"] == 0
    assert closure["control_group_process_count"] == 0
    assert closure["control_group_thread_count"] == 0
    assert supervisor.verify_all_workers_closed() == (closure,)


def test_systemd_policy_and_closure_reject_boolean_numeric_tamper() -> None:
    unit = runtime._worker_unit_name(
        role="HippoRAG", identity={"fake_state_machine": "bool-tamper"}
    )
    unit_sha = runtime._unit_name_sha256(unit)
    closure = _unit_closure(unit)
    for field in (
        "main_pid",
        "control_group_process_count",
        "control_group_thread_count",
        "systemctl_show_returncode",
        "systemctl_reset_failed_returncode",
    ):
        tampered = {**closure, field: False}
        with pytest.raises(
            runtime.TatqaP21FormalRuntimeError, match="closure receipt"
        ):
            runtime._validate_unit_closure_receipt(
                tampered, expected_unit_name_sha256=unit_sha
            )
    policy = _start_policy(unit)
    for field in ("main_pid", "tasks_max", "systemctl_show_returncode"):
        tampered = {**policy, field: False}
        with pytest.raises(
            runtime.TatqaP21FormalRuntimeError, match="start-policy receipt"
        ):
            runtime._validate_start_policy_receipt(
                tampered, expected_unit_name_sha256=unit_sha
            )


def test_wrapper_timeout_kills_named_unit_control_group_without_leak(
    launcher_state: None,
) -> None:
    machine = _FakeSystemdMachine(mode="timeout", stop_leaves_child=True)
    supervisor = _fake_supervisor(machine)
    unit = runtime._worker_unit_name(
        role="HippoRAG", identity={"fake_state_machine": "timeout"}
    )
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="timed out"):
        runtime._run_worker(
            command=["/bin/false"],
            environment={"LANG": "C.UTF-8"},
            inaccessible_paths=(),
            timeout=1,
            role="HippoRAG",
            unit_name=unit,
            supervisor=supervisor,
        )
    actions = [row[2] for row in machine.commands if row[:2] == [str(runtime.SYSTEMCTL), "--user"]]
    assert "stop" in actions
    assert "kill" in actions
    assert machine.exists is False
    closures = supervisor.verify_all_workers_closed()
    assert closures[0]["main_pid"] == 0


def test_running_worker_abort_stops_kills_and_reaps_actual_client(
    launcher_state: None,
) -> None:
    machine = _FakeSystemdMachine(mode="blocking", stop_leaves_child=True)
    supervisor = _fake_supervisor(machine)
    unit = runtime._worker_unit_name(
        role="HippoRAG", identity={"fake_state_machine": "cancel"}
    )
    failures: list[BaseException] = []

    def launch() -> None:
        try:
            runtime._run_worker(
                command=["/bin/false"],
                environment={"LANG": "C.UTF-8"},
                inaccessible_paths=(),
                timeout=30,
                role="HippoRAG",
                unit_name=unit,
                supervisor=supervisor,
            )
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=launch)
    thread.start()
    assert machine.communicate_entered.wait(timeout=1)
    closures = supervisor.abort_all_workers()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert failures
    assert closures[0]["load_state"] == "not-found"
    actions = [row[2] for row in machine.commands if row[:2] == [str(runtime.SYSTEMCTL), "--user"]]
    assert "stop" in actions
    assert "kill" in actions
    assert machine.process.poll() == -9
    assert machine.exists is False


def test_terminal_abort_seals_empty_supervisor_against_late_reservation() -> None:
    machine = _FakeSystemdMachine(mode="normal")
    supervisor = _fake_supervisor(machine)
    assert supervisor.abort_all_workers() == ()
    unit = runtime._worker_unit_name(
        role="HippoRAG", identity={"fake_state_machine": "late-after-empty-abort"}
    )
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="sealed"):
        runtime._run_worker(
            command=["/bin/false"],
            environment={"LANG": "C.UTF-8"},
            inaccessible_paths=(),
            timeout=1,
            role="HippoRAG",
            unit_name=unit,
            supervisor=supervisor,
        )
    assert machine.process is None
    assert machine.exists is False


def test_terminal_abort_attempts_every_unit_after_earlier_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    machine = _FakeSystemdMachine(mode="blocking")
    supervisor = _fake_supervisor(machine)
    names = [
        runtime._worker_unit_name(
            role="HippoRAG", identity={"aggregate_abort": index}
        )
        for index in range(2)
    ]
    processes = [object(), object()]
    records = [
        runtime._TrackedSystemdUnit(
            name=name,
            name_sha256=runtime._unit_name_sha256(name),
            process=process,
        )
        for name, process in zip(names, processes)
    ]
    supervisor._records = {record.name: record for record in records}
    controlled: list[tuple[str, str]] = []
    killed: list[object] = []
    finalized: list[str] = []

    monkeypatch.setattr(
        supervisor,
        "_best_effort_control",
        lambda arguments: controlled.append((arguments[0], arguments[-1])),
    )

    def kill_client(process):
        killed.append(process)
        if process is processes[0]:
            raise runtime.TatqaP21FormalRuntimeError("injected first client failure")

    def finalize(record):
        finalized.append(record.name)
        if record is records[0]:
            raise runtime.TatqaP21FormalRuntimeError("injected first finalize failure")
        return _unit_closure(record.name)

    monkeypatch.setattr(supervisor, "_kill_client", kill_client)
    monkeypatch.setattr(supervisor, "_finalize", finalize)
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="attempting all"):
        supervisor.abort_all_workers()
    assert controlled == [
        (action, name) for name in names for action in ("stop", "kill")
    ]
    assert killed == processes
    assert finalized == names
    assert all(record.abort_requested for record in records)
    assert supervisor._sealed is True


def test_abort_and_reserve_are_linearized_without_late_unit_escape(
    launcher_state: None,
) -> None:
    machine = _FakeSystemdMachine(mode="normal")
    initial_show_entered = threading.Event()
    release_initial_show = threading.Event()
    original_systemctl = machine.systemctl
    blocked_once = False

    def blocking_systemctl(command, **kwargs):
        nonlocal blocked_once
        if command[2] == "show" and not blocked_once:
            blocked_once = True
            initial_show_entered.set()
            assert release_initial_show.wait(timeout=2)
        return original_systemctl(command, **kwargs)

    machine.systemctl = blocking_systemctl
    supervisor = _fake_supervisor(machine)
    unit = runtime._worker_unit_name(
        role="HippoRAG", identity={"fake_state_machine": "abort-reserve-race"}
    )
    launch_outcome: list[object] = []
    abort_outcome: list[object] = []

    def launch() -> None:
        try:
            launch_outcome.append(
                runtime._run_worker(
                    command=["/bin/true"],
                    environment={"LANG": "C.UTF-8"},
                    inaccessible_paths=(),
                    timeout=1,
                    role="HippoRAG",
                    unit_name=unit,
                    supervisor=supervisor,
                )
            )
        except BaseException as exc:
            launch_outcome.append(exc)

    launch_thread = threading.Thread(target=launch)
    abort_thread = threading.Thread(
        target=lambda: abort_outcome.append(supervisor.abort_all_workers())
    )
    launch_thread.start()
    assert initial_show_entered.wait(timeout=1)
    abort_thread.start()
    release_initial_show.set()
    launch_thread.join(timeout=2)
    abort_thread.join(timeout=2)
    assert not launch_thread.is_alive() and not abort_thread.is_alive()
    assert launch_outcome and abort_outcome
    assert machine.exists is False
    assert supervisor.verify_all_workers_closed()
    late = runtime._worker_unit_name(
        role="HippoRAG", identity={"fake_state_machine": "after-race"}
    )
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="sealed"):
        runtime._run_worker(
            command=["/bin/true"],
            environment={"LANG": "C.UTF-8"},
            inaccessible_paths=(),
            timeout=1,
            role="HippoRAG",
            unit_name=late,
            supervisor=supervisor,
        )


def test_live_policy_mismatch_fails_closed_and_verify_rejects_reopened_unit(
    launcher_state: None,
) -> None:
    drifted = _FakeSystemdMachine(mode="normal", tasks_max=4)
    supervisor = _fake_supervisor(drifted)
    unit = runtime._worker_unit_name(
        role="HippoRAG", identity={"fake_state_machine": "policy-drift"}
    )
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="start-policy"):
        runtime._run_worker(
            command=["/bin/false"],
            environment={"LANG": "C.UTF-8"},
            inaccessible_paths=(),
            timeout=1,
            role="HippoRAG",
            unit_name=unit,
            supervisor=supervisor,
        )
    assert drifted.exists is False

    machine = _FakeSystemdMachine(mode="normal")
    supervisor, _unit, _result = _fake_supervised_run(machine, suffix="reopen")
    machine.exists = True
    machine.phase = "active"
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="reopened"):
        supervisor.verify_all_workers_closed()


def test_qwen_byte_runner_binds_canonical_input_and_output(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    item = typed_plan_contract.project_item(_runtime_item(), 0)
    canonical_input = typed_plan_contract.canonical_json_bytes(
        typed_plan_contract.input_payload((item,))
    )

    def fake_worker(
        *,
        command,
        environment,
        inaccessible_paths,
        timeout,
        role,
        unit_name,
        supervisor,
    ):
        output_path = Path(command[command.index("--output") + 1])
        assert command[0] == str(paths.typed_runtime_python)
        completion = (
            '{"entity_facets":["Acme"],"metric_facets":["revenue"],'
            '"time_facets":["2023","2024"],"operation":"COMPARE",'
            '"relation_query":"Acme revenue comparison"}'
        )
        row = typed_plan_contract.build_output_item(
            item=item,
            completion=completion,
            completion_token_count=30,
            prompt_sha256="a" * 64,
            prompt_token_count=100,
            prompt_projection_sha256="b" * 64,
        )
        output_path.write_bytes(
            typed_plan_contract.canonical_json_bytes(
                typed_plan_contract.output_payload((row,))
            )
        )
        assert environment["CUDA_VISIBLE_DEVICES"] == "1"
        assert timeout == runtime.QWEN_TIMEOUT_SECONDS
        assert len(inaccessible_paths) == 2
        return (
            b'{"generation_valid_count":1,"item_count":1,'
            b'"model_context_tokens":32768,'
            b'"model_execution_started_monotonic_ns":100,'
            b'"model_execution_finished_monotonic_ns":200,'
            b'"status":"passed","worker_pid":123}\n',
            b"",
            None,
            _unit_closure(unit_name),
        )

    monkeypatch.setattr(runtime, "_run_worker", fake_worker)
    runner = runtime.SystemdTypedPlanBatchRunner(paths)
    raw = runner("A_form", canonical_input)
    output = typed_plan_contract.parse_output(raw)
    assert output["items"][0]["plan"]["operation"] == "COMPARE"
    assert runner.receipts["A_form"]["input_sha256"]
    assert runner.receipts["A_form"]["item_count"] == 1
    assert runner.receipts["A_form"]["model_execution_started_monotonic_ns"] == 100
    assert runner.receipts["A_form"]["model_execution_finished_monotonic_ns"] == 200
    assert runner.receipts["A_form"]["systemd_unit_closure"]["main_pid"] == 0
    assert runner.transport_receipt("A_form")["worker_pid"] == 123


def test_hippo_byte_runner_rejects_replay_and_binds_same_item_corpus(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    item = _runtime_item()
    units = [
        {"ordinal": index, "unit_id": row.unit_id, "text": row.text}
        for index, row in enumerate(item.units)
    ]
    payload = hipporag_contract.input_payload(query=item.question, units=units)
    canonical_input = hipporag_contract.canonical_json_bytes(payload)

    def fake_worker(
        *,
        command,
        environment,
        inaccessible_paths,
        timeout,
        role,
        unit_name,
        supervisor,
    ):
        output_path = Path(command[command.index("--output") + 1])
        assert command[0] == str(paths.hippo_runtime_python)
        value = hipporag_contract.output_payload(
            top_unit_ids=[row.unit_id for row in item.units[:5]],
            graph_nodes=7,
            graph_edges=6,
            unit_count=len(item.units),
            input_sha256=payload["input_sha256"],
        )
        output_path.write_bytes(hipporag_contract.canonical_json_bytes(value))
        assert environment["CUDA_VISIBLE_DEVICES"] == ""
        assert environment["OMP_NUM_THREADS"] == "1"
        assert timeout == runtime.HIPPORAG_TIMEOUT_SECONDS
        assert len(inaccessible_paths) == 2
        return (
            b'{"configured_torch_interop_threads":1,'
            b'"configured_torch_intraop_threads":1,'
            b'"graph_edge_count":6,"graph_node_count":7,'
            b'"model_execution_finished_monotonic_ns":250,'
            b'"model_execution_started_monotonic_ns":150,'
            b'"observed_process_thread_peak":2,"status":"passed",'
            b'"unit_count":5,"worker_pid":124}\n',
            b"",
            _start_policy(unit_name),
            _unit_closure(unit_name),
        )

    monkeypatch.setattr(runtime, "_run_worker", fake_worker)
    runner = runtime.SystemdHippoByteRunner(paths)
    raw = runner("A_hold", item.item_id, canonical_input)
    assert hipporag_contract.parse_output(raw)["input_sha256"] == payload["input_sha256"]
    assert runner.receipts[0]["CPU_threads"] == 2
    assert runner.receipts[0]["configured_torch_intraop_threads"] == 1
    assert runner.receipts[0]["configured_torch_interop_threads"] == 1
    assert runner.receipts[0]["observed_process_thread_peak"] == 2
    assert runner.receipts[0]["model_execution_started_monotonic_ns"] == 150
    assert runner.receipts[0]["model_execution_finished_monotonic_ns"] == 250
    assert runner.receipts[0]["systemd_tasks_max"] == 3
    assert runner.receipts[0]["thread_monitor_process_reservation"] == 1
    assert runner.receipts[0]["maximum_worker_process_threads"] == 2
    assert runner.receipts[0]["systemd_start_policy"]["kill_mode"] == "control-group"
    assert runner.receipts[0]["systemd_unit_closure"]["main_pid"] == 0
    assert runner.transport_receipt("A_hold", item.item_id)["worker_pid"] == 124
    try:
        runner("A_hold", item.item_id, canonical_input)
    except runtime.TatqaP21FormalRuntimeError as exc:
        assert "replay" in str(exc)
    else:  # pragma: no cover - fail loudly without importing pytest in worker fixtures
        raise AssertionError("item-local replay was accepted")


def test_terminal_receipts_reject_nonpositive_intervals_and_measured_thread_overrun() -> None:
    qwen = {
        "generation_valid_count": 1,
        "item_count": 1,
        "model_context_tokens": 32768,
        "model_execution_finished_monotonic_ns": 100,
        "model_execution_started_monotonic_ns": 100,
        "status": "passed",
        "worker_pid": 123,
    }
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="interval"):
        runtime._terminal_status(
            runtime.canonical_json_bytes(qwen), role="Qwen", item_count=1
        )

    hippo = {
        "configured_torch_interop_threads": 1,
        "configured_torch_intraop_threads": 1,
        "graph_edge_count": 6,
        "graph_node_count": 7,
        "model_execution_finished_monotonic_ns": 200,
        "model_execution_started_monotonic_ns": 100,
        "observed_process_thread_peak": 3,
        "status": "passed",
        "unit_count": 5,
        "worker_pid": 124,
    }
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="HippoRAG"):
        runtime._terminal_status(
            runtime.canonical_json_bytes(hippo), role="HippoRAG", item_count=5
        )


def test_tree_receipt_is_content_ordered_and_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "b").write_bytes(b"two")
    (root / "a").write_bytes(b"one")
    first = runtime.tree_receipt(root)
    second = runtime.tree_receipt(root)
    assert first == second
    assert first["file_count"] == 2
    (root / "link").symlink_to(root / "a")
    try:
        runtime.tree_receipt(root)
    except runtime.TatqaP21FormalRuntimeError as exc:
        assert "non-file" in str(exc) or "symlink" in str(exc)
    else:
        raise AssertionError("symlinked asset member was accepted")


def test_runtime_inventory_binds_active_interpreter_and_launch_executables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "config.json").write_text(
        json.dumps({"max_position_embeddings": 32768}), encoding="utf-8"
    )
    minilm = tmp_path / "minilm.json"
    hippo = tmp_path / "hippo.json"
    minilm.write_bytes(b"minilm")
    hippo.write_bytes(b"hippo")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: 0),
        version=SimpleNamespace(cuda="test-cuda"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runtime.metadata, "version", lambda name: f"{name}-v")
    typed_pyvenv = tmp_path / "typed.pyvenv.cfg"
    typed_pyvenv.write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.10.0\n",
        encoding="utf-8",
    )
    hippo_pyvenv = tmp_path / "hippo.pyvenv.cfg"
    hippo_pyvenv.write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\nversion = 3.11.0\n",
        encoding="utf-8",
    )
    typed_site_root = tmp_path / "typed-site-packages"
    typed_site_root.mkdir()
    hippo_site_root = tmp_path / "hippo-site-packages"
    hippo_site_root.mkdir()
    typed_rows = [{"name": "torch", "version": "2.8.0"}]
    hippo_rows = [{"name": "hipporag", "version": "2.0.0a4"}]
    monkeypatch.setattr(
        runtime,
        "_typed_dependency_rows",
        lambda **_kwargs: (
            typed_rows,
            typed_pyvenv,
            [("overlay", typed_site_root)],
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_hippo_attested_dependency_rows",
        lambda **_kwargs: {
            "active_module_rows": [
                {
                    "distribution_name": "hipporag",
                    "module_name": "hipporag",
                    "origin_file_sha256": "a" * 64,
                    "origin_path": "/fixed/hipporag/__init__.py",
                }
            ],
            "attestation_receipt_sha256": "b" * 64,
            "dependency_rows": hippo_rows,
            "pth_rows": [
                {"name": "fixed.pth", "sha256": "c" * 64, "size_bytes": 1}
            ],
            "pyvenv": hippo_pyvenv,
            "pyvenv_binding": {
                "file_sha256": runtime.file_sha256(hippo_pyvenv),
                "size_bytes": hippo_pyvenv.stat().st_size,
            },
            "search_roots": [("overlay", hippo_site_root)],
        },
    )
    hippo_python = tmp_path / "hippo-runtime" / "bin" / "python"
    hippo_python.parent.mkdir(parents=True)
    execution_marker = tmp_path / "hippo-was-executed"
    hippo_python.write_text(
        f"#!/bin/sh\ntouch {execution_marker}\n",
        encoding="utf-8",
    )
    hippo_python.chmod(0o700)

    inventory = runtime.runtime_inventory_snapshot(
        typed_runtime_python=Path(sys.executable),
        hippo_runtime_python=hippo_python,
        qwen_model=qwen,
        minilm_manifest=minilm,
        hippo_attestation=hippo,
    )

    support_executables = inventory["support_executable_bindings"]
    assert set(support_executables) == {
        "systemd_run",
        "systemctl",
        "environment_clearer",
        "network_preflight_python",
    }
    subfingerprints = inventory["runtime_python_subfingerprints"]
    assert set(subfingerprints) == {
        "typed_plan_minilm_runtime_python",
        "hipporag_runtime_python",
    }
    typed_subfingerprint = subfingerprints["typed_plan_minilm_runtime_python"]
    hippo_subfingerprint = subfingerprints["hipporag_runtime_python"]
    assert typed_subfingerprint["schema"] == (
        "tatqa_p21_typed_minilm_runtime_python_subfingerprint_v1"
    )
    assert hippo_subfingerprint["schema"] == (
        "tatqa_p21_hipporag_runtime_python_subfingerprint_v1"
    )
    assert typed_subfingerprint["executable_binding"][
        "samefile_with_active_sys_executable"
    ] is True
    assert hippo_subfingerprint["executable_binding"][
        "passive_filesystem_attestation_only"
    ] is True
    assert not execution_marker.exists()
    assert typed_subfingerprint["filesystem_binding"][
        "dependency_metadata_rows"
    ] == typed_rows
    assert hippo_subfingerprint["filesystem_binding"][
        "dependency_metadata_rows"
    ] == hippo_rows
    for subfingerprint in subfingerprints.values():
        body = dict(subfingerprint)
        declared = body.pop("self_sha256")
        assert declared == runtime.stable_hash(body)
    assert inventory["runtime_role_assignment"] == {
        "HippoRAG": "hipporag_runtime_python",
        "MiniLM": "typed_plan_minilm_runtime_python",
        "Qwen": "typed_plan_minilm_runtime_python",
    }
    assert inventory["Qwen_config"]["max_position_embeddings"] == 32768

    other = tmp_path / "other-python"
    other.write_bytes(b"#!/bin/sh\nexit 0\n")
    other.chmod(0o700)
    with pytest.raises(
        runtime.TatqaP21FormalRuntimeError, match="sys.executable"
    ):
        runtime.runtime_inventory_snapshot(
            typed_runtime_python=other,
            hippo_runtime_python=hippo_python,
            qwen_model=qwen,
            minilm_manifest=minilm,
            hippo_attestation=hippo,
        )
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="not distinct"):
        runtime.runtime_inventory_snapshot(
            typed_runtime_python=Path(sys.executable),
            hippo_runtime_python=Path(sys.executable),
            qwen_model=qwen,
            minilm_manifest=minilm,
            hippo_attestation=hippo,
        )


def _fingerprint_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, llm_tree: str | None = None
) -> _Paths:
    paths = _paths(tmp_path)
    tree_hashes = {
        paths.qwen_model: runtime.QWEN_MODEL_TREE_SHA256,
        paths.minilm_model: runtime.MINILM_GENERIC_TREE_SHA256,
        paths.hippo_llm_model: llm_tree or runtime.HIPPORAG_LLM_TREE_SHA256,
        paths.hippo_embedding_model: runtime.HIPPORAG_EMBEDDING_TREE_SHA256,
        paths.hipporag_source: runtime.HIPPORAG_SOURCE_TREE_SHA256,
    }
    receipts = {
        name: {"file_count": 1, "size_bytes": 1, "tree_sha256": tree_hashes[path]}
        for name, path in {
            "Qwen": paths.qwen_model,
            "MiniLM": paths.minilm_model,
            "HippoRAG_LLM": paths.hippo_llm_model,
            "HippoRAG_embedding": paths.hippo_embedding_model,
            "HippoRAG_source": paths.hipporag_source,
        }.items()
    }
    inventory = {"exact": "live-runtime"}
    launcher = _synthetic_launcher_receipt()
    body = {
        "schema": "tatqa_p21_composite_runtime_fingerprint_v1",
        "status": "verified_before_formal_source_open",
        "study_design_self_sha256": runtime.STUDY_DESIGN_SELF_SHA256,
        "asset_bindings": receipts,
        "runtime_inventory": inventory,
        "safe_user_systemd_launch_envelope": launcher,
    }
    value = {**body, "self_sha256": runtime.stable_hash(body)}
    paths.fingerprint_manifest.write_bytes(runtime.canonical_json_bytes(value))
    monkeypatch.setattr(runtime, "tree_receipt", lambda path: receipts[next(
        name for name, bound in {
            "Qwen": paths.qwen_model,
            "MiniLM": paths.minilm_model,
            "HippoRAG_LLM": paths.hippo_llm_model,
            "HippoRAG_embedding": paths.hippo_embedding_model,
            "HippoRAG_source": paths.hipporag_source,
        }.items() if bound == path
    )])
    monkeypatch.setattr(
        runtime,
        "file_sha256",
        lambda path: (
            runtime.MINILM_EXPECTED_ASSET_FILE_SHA256
            if path == paths.minilm_asset_manifest
            else runtime.HIPPORAG_ATTESTATION_FILE_SHA256
        ),
    )
    monkeypatch.setattr(runtime, "_verify_hipporag_attestation_identity", lambda _p: None)
    monkeypatch.setattr(runtime, "runtime_inventory_snapshot", lambda **_k: inventory)
    monkeypatch.setattr(
        runtime,
        "user_systemd_launcher_phase_receipt",
        lambda *, phase: launcher["phase_receipts"][phase],
    )
    return paths


def test_runtime_fingerprint_rechecks_live_inventory_and_fixed_hippo_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fingerprint_fixture(tmp_path, monkeypatch)
    assert runtime.verify_runtime_fingerprint(paths)["runtime_inventory"] == {
        "exact": "live-runtime"
    }
    monkeypatch.setattr(
        runtime, "runtime_inventory_snapshot", lambda **_k: {"exact": "drifted"}
    )
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="inventory"):
        runtime.verify_runtime_fingerprint(paths)

    second = tmp_path / "wrong-llm"
    second.mkdir()
    wrong_paths = _fingerprint_fixture(
        second, monkeypatch, llm_tree="0" * 64
    )
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="asset drifted"):
        runtime.verify_runtime_fingerprint(wrong_paths)


def test_bound_p19_hipporag_attestation_key_identity_parses() -> None:
    path = Path("manifests/tatqa_p19_hipporag_runtime_attestation_v1.json")
    assert runtime.file_sha256(path) == runtime.HIPPORAG_ATTESTATION_FILE_SHA256
    value = runtime._verify_hipporag_attestation_identity(path)
    assert value["schema"] == "tatqa_p19_hipporag_runtime_attestation_v1"
    assert value["receipt_sha256"] == (
        runtime.HIPPORAG_ATTESTATION_RECEIPT_SHA256
    )
    assert set(value["asset_bindings"]) == {"HippoRAG_source", "MiniLM", "SmolLM"}
    assert value["active_distribution_binding"]["rows"]
    assert value["active_module_binding"]["rows"]


def test_attested_dependency_dist_info_version_and_tree_are_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_python = tmp_path / "hippo-venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    execution_marker = tmp_path / "hippo-python-executed"
    runtime_python.write_text(
        f"#!/bin/sh\ntouch {execution_marker}\n", encoding="utf-8"
    )
    runtime_python.chmod(0o700)
    venv_root = runtime_python.parent.parent
    pyvenv = venv_root / "pyvenv.cfg"
    pyvenv.write_text("fixed venv\n", encoding="utf-8")
    root = venv_root / "lib" / "python3.11" / "site-packages"
    dist_info = root / "hipporag-2.0.0a4.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Name: hipporag\nVersion: 2.0.0a4\n", encoding="utf-8"
    )
    (dist_info / "RECORD").write_text("fixed\n", encoding="utf-8")
    pth = root / "fixed.pth"
    pth.write_text("/fixed/path\n", encoding="utf-8")
    pth_rows = [
        {
            "name": pth.name,
            "sha256": runtime.file_sha256(pth),
            "size_bytes": pth.stat().st_size,
        }
    ]
    count, tree_sha = runtime._attestation_tree_binding(dist_info)
    rows = [
        {
            "dist_info_file_count": count,
            "dist_info_name": dist_info.name,
            "dist_info_tree_sha256": tree_sha,
            "name": "hipporag",
            "root_path": str(root),
            "version": "2.0.0a4",
        }
    ]
    module_origin = tmp_path / "hipporag" / "__init__.py"
    module_origin.parent.mkdir()
    module_origin.write_text("# fixed\n", encoding="utf-8")
    module_rows = [
        {
            "distribution_name": "hipporag",
            "module_name": "hipporag",
            "origin_file_sha256": runtime.file_sha256(module_origin),
            "origin_path": str(module_origin),
        }
    ]
    attestation = tmp_path / "attestation.json"
    attestation.write_bytes(b"identity validation is mocked in this unit test\n")
    target = runtime_python.resolve(strict=True)
    value = {
        "active_distribution_binding": {
            "rows": rows,
            "set_sha256": runtime.stable_hash(rows),
        },
        "active_module_binding": {
            "rows": module_rows,
            "set_sha256": runtime.stable_hash(module_rows),
        },
        "receipt_sha256": "a" * 64,
        "runtime_python_binding": {
            "active_sys_executable": str(runtime_python),
            "active_sys_prefix": str(venv_root),
            "existing_site_package_roots": [str(root)],
            "implementation": "CPython",
            "lexical_path": str(runtime_python),
            "lexical_path_sha256": runtime.hashlib.sha256(
                str(runtime_python).encode("utf-8")
            ).hexdigest(),
            "pth_rows": pth_rows,
            "pth_set_sha256": runtime.stable_hash(pth_rows),
            "python_version": "3.11.0",
            "pyvenv_cfg_sha256": runtime.file_sha256(pyvenv),
            "resolved_path": str(target),
            "resolved_target_sha256": runtime.file_sha256(target),
            "resolved_target_size_bytes": target.stat().st_size,
            "samefile_with_active_sys_executable": True,
        },
    }
    monkeypatch.setattr(
        runtime,
        "_verify_hipporag_attestation_identity",
        lambda _path: value,
    )
    monkeypatch.setattr(
        runtime,
        "HIPPORAG_ATTESTATION_RUNTIME_TARGET_SHA256",
        runtime.file_sha256(target),
    )
    monkeypatch.setattr(
        runtime,
        "HIPPORAG_ATTESTATION_PYVENV_CFG_SHA256",
        runtime.file_sha256(pyvenv),
    )

    observed = runtime._hippo_attested_dependency_rows(
        hippo_runtime_python=runtime_python,
        hippo_attestation=attestation,
    )
    assert observed["dependency_rows"] == rows
    assert observed["active_module_rows"] == module_rows
    assert observed["pth_rows"] == pth_rows
    assert observed["pyvenv"] == pyvenv
    assert not execution_marker.exists()
    pth.write_text("/tampered/path\n", encoding="utf-8")
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match=r"\.pth"):
        runtime._hippo_attested_dependency_rows(
            hippo_runtime_python=runtime_python,
            hippo_attestation=attestation,
        )
    pth.write_text("/fixed/path\n", encoding="utf-8")
    module_origin.write_text("# tampered\n", encoding="utf-8")
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="active-module"):
        runtime._hippo_attested_dependency_rows(
            hippo_runtime_python=runtime_python,
            hippo_attestation=attestation,
        )
    module_origin.write_text("# fixed\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="metadata drifted"):
        runtime._hippo_attested_dependency_rows(
            hippo_runtime_python=runtime_python,
            hippo_attestation=attestation,
        )


def test_typed_dependency_dist_info_trees_are_complete_and_unambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "typed-site-packages"
    root.mkdir()
    expected_versions = {}
    for index, name in enumerate(runtime.RUNTIME_DISTRIBUTIONS):
        version = f"1.{index}.0"
        expected_versions[name] = version
        dist_info = root / f"{name.replace('-', '_')}-{version}.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Name: {name}\nVersion: {version}\n", encoding="utf-8"
        )
        (dist_info / "RECORD").write_text("fixed\n", encoding="utf-8")
    pyvenv = tmp_path / "typed-pyvenv.cfg"
    pyvenv.write_text("fixed\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime,
        "_runtime_search_roots",
        lambda _python: (pyvenv, [("overlay", root)]),
    )

    rows, observed_pyvenv, roots = runtime._typed_dependency_rows(
        typed_runtime_python=tmp_path / "typed" / "bin" / "python",
        expected_versions=expected_versions,
    )
    assert observed_pyvenv == pyvenv
    assert roots == [("overlay", root)]
    assert [row["name"] for row in rows] == list(runtime.RUNTIME_DISTRIBUTIONS)
    assert all(row["dist_info_file_count"] == 2 for row in rows)

    duplicate = root / "torch-duplicate.dist-info"
    duplicate.mkdir()
    (duplicate / "METADATA").write_text(
        f"Name: torch\nVersion: {expected_versions['torch']}\n",
        encoding="utf-8",
    )
    (duplicate / "RECORD").write_text("duplicate\n", encoding="utf-8")
    with pytest.raises(runtime.TatqaP21FormalRuntimeError, match="ambiguous"):
        runtime._typed_dependency_rows(
            typed_runtime_python=tmp_path / "typed" / "bin" / "python",
            expected_versions=expected_versions,
        )
