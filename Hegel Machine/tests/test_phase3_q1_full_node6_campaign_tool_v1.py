from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hegel_machine import phase3_q1_full_node6_campaign_v1 as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools/phase3_q1_full_node6_campaign_v1.py"


def _load_tool():
    name = "_test_phase3_q1_full_node6_campaign_tool_v1"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _capsule(tool, contract: dict[str, object]) -> dict[str, object]:
    return tool._core.build_capsule_v1(
        contract,
        source_commit="1" * 40,
        source_tree="2" * 40,
        source_closure_digest="sha256:" + "3" * 64,
        static_gate_receipts={"source": {"qualified": True}},
    )


def _contract(tool) -> dict[str, object]:
    contract = json.loads(tool.CONFIG_PATH.read_bytes())
    contract["execution_policy"]["supervisor_timeout_seconds"] = 1212000
    unsigned = deepcopy(contract)
    unsigned.pop("contract_digest")
    contract["contract_digest"] = tool._core.domain_digest_v1(
        tool._core.DOMAINS["contract"],
        unsigned,
    )
    return tool._core.validate_contract_v1(contract)


def _temporary_contract(tool, tmp_path: Path) -> dict[str, object]:
    contract = deepcopy(_contract(tool))
    evidence = tmp_path / "evidence"
    contract["fixed_paths"].update(
        {
            "campaign_lock_path": str(tmp_path / "campaign.lock"),
            "ledger_root": str(tmp_path / "ledger"),
            "evidence_root": str(evidence),
            "trajectory_root": str(tmp_path / "trajectory"),
            "run_root": str(tmp_path / "run"),
            "probe_root": str(tmp_path / "probe"),
            "artifact_path": str(tmp_path / "artifact.json"),
            "stdout_path": str(evidence / "supervisor.stdout"),
            "stderr_path": str(evidence / "supervisor.stderr"),
        }
    )
    unsigned = deepcopy(contract)
    unsigned.pop("contract_digest")
    contract["contract_digest"] = tool._core.domain_digest_v1(
        tool._core.DOMAINS["contract"],
        unsigned,
    )
    return tool._core.validate_contract_v1(contract)


def _consumed(tool, contract, capsule):
    return tool._core.build_ledger_record_v1(
        contract,
        capsule,
        sequence=1,
        record_type="CONSUMED",
        status=tool._core.CONSUMED,
        previous_record_digest_or_null=None,
        body={
            "capsule": capsule,
            "maximum_supervisor_popen_count": 1,
            "maximum_measurements_per_endpoint_signature": 1,
        },
    )


def _failed_outcome(tool, contract, capsule):
    return tool._outcome_value(
        contract,
        capsule,
        return_code=2,
        terminal_status=tool._core.FAILED_TERMINAL,
        replay_passed=False,
        stdout_sha256="sha256:" + "a" * 64,
        stderr_sha256="sha256:" + "b" * 64,
        artifact_sha256=None,
        decision=tool._core.CampaignDecisionV1(
            tool._core.FAILED_TERMINAL,
            False,
            False,
        ),
    )


def test_live_pending_is_bound_to_the_same_capsule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    contract = _contract(tool)
    monkeypatch.setattr(tool, "load_contract_v1", lambda: contract)
    monkeypatch.setattr(
        tool,
        "_source_gate",
        lambda _contract: {
            "source_commit": "1" * 40,
            "source_tree": "2" * 40,
            "source_closure_digest": "sha256:" + "3" * 64,
            "source_rows": [],
        },
    )
    monkeypatch.setattr(tool, "_predecessor_gate", lambda _contract: {"ok": True})
    monkeypatch.setattr(
        tool,
        "_superseded_campaign_gate",
        lambda _contract: {"measurement_endpoint_start_count": 0},
    )
    monkeypatch.setattr(tool, "_technical_contract_gate", lambda _contract: {"ok": True})
    monkeypatch.setattr(tool, "_dependency_gate", lambda *_args: {"ok": True})
    monkeypatch.setattr(
        tool,
        "_docker_gate",
        lambda: (_ for _ in ()).throw(
            tool.GatePending(
                "PINNED_LOCAL_IMAGES_AND_DAEMON",
                "LOCAL_DOCKER_DAEMON_UNAVAILABLE",
                "daemon is temporarily unavailable",
            )
        ),
    )

    with pytest.raises(tool.GatePending) as caught:
        tool.derive_preflight_v1(include_live_gates=True)
    error = caught.value
    assert error.capsule is not None
    assert error.capsule_digest_hint == error.capsule["capsule_digest"]
    pending = tool._core.build_pending_v1(
        contract,
        error.capsule,
        phase=error.phase,
        condition=error.condition,
        detail=error.detail,
    )
    assert pending["capsule_digest"] == error.capsule_digest_hint
    assert pending["campaign_execution_count"] == 0
    assert pending["authorization_consumption_count"] == 0
    assert pending["supervisor_popen_count"] == 0
    assert pending["q1_state"] == "NOT_RUN"


def test_supervisor_import_permission_failure_is_preconsumption_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()

    class Loader:
        def exec_module(self, _module):
            raise PermissionError("synthetic supervisor read permission")

    spec = SimpleNamespace(loader=Loader())
    monkeypatch.setattr(tool.importlib.util, "spec_from_file_location", lambda *_args: spec)
    monkeypatch.setattr(tool.importlib.util, "module_from_spec", lambda _spec: SimpleNamespace())
    with pytest.raises(tool.GatePending) as caught:
        tool._load_supervisor_module_v1()
    assert caught.value.phase == "PATH_AND_FORMAT"
    assert caught.value.condition == "SUPERVISOR_SOURCE_IMPORT_UNAVAILABLE"


def test_preflight_main_emits_digest_bound_pending(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    contract = _contract(tool)
    capsule = _capsule(tool, contract)
    monkeypatch.setattr(tool, "load_contract_v1", lambda: contract)

    def pending(*, include_live_gates: bool):
        assert include_live_gates is True
        raise tool.GatePending(
            "MEMORY_AND_STORAGE_RESERVATION",
            "REQUIRED_AVAILABLE_MEMORY_UNAVAILABLE",
            "wait for unrelated local jobs to release memory",
            capsule=capsule,
            capsule_digest_hint=capsule["capsule_digest"],
        )

    monkeypatch.setattr(tool, "derive_preflight_v1", pending)
    assert tool.main(["--preflight"]) == tool.PENDING_EXIT_CODE
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == core.PENDING_NOT_CONSUMED
    assert payload["capsule_digest"] == capsule["capsule_digest"]
    assert payload["authorization_consumption_count"] == 0
    assert payload["supervisor_popen_count"] == 0
    assert payload["q1_state"] == "NOT_RUN"


def test_static_pending_under_run_keeps_the_supplied_capsule_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    contract = _contract(tool)
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(tool, "load_contract_v1", lambda: contract)
    monkeypatch.setattr(
        tool,
        "_replay_existing",
        lambda _contract, _digest: None,
    )
    monkeypatch.setattr(
        tool,
        "derive_preflight_v1",
        lambda **_kwargs: (_ for _ in ()).throw(
            tool.GatePending(
                "SOURCE_LINEAGE_AND_OBJECT_CLOSURE",
                "READONLY_COMMAND_TIMEOUT",
                "temporary readonly check timeout",
            )
        ),
    )
    monkeypatch.setattr(tool._CampaignLock, "__enter__", lambda self: self)
    monkeypatch.setattr(tool._CampaignLock, "__exit__", lambda self, *args: None)
    with pytest.raises(tool.GatePending) as caught:
        tool.execute_once_v1(digest)
    assert caught.value.capsule_digest_hint == digest


def test_second_absence_pending_carries_the_complete_replayed_capsule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    contract = _contract(tool)
    capsule = _capsule(tool, contract)
    monkeypatch.setattr(tool, "load_contract_v1", lambda: contract)
    monkeypatch.setattr(tool, "_replay_existing", lambda *_args: None)
    monkeypatch.setattr(
        tool,
        "derive_preflight_v1",
        lambda **_kwargs: {"capsule": capsule},
    )
    monkeypatch.setattr(
        tool,
        "_state_absence_gate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            tool.GatePending(
                "CURRENT_FUTURE_FORMAL_AND_PROBE_ROOTS_ABSENT",
                "TRANSIENT_ROOT_PRESENT",
                "synthetic second absence failure",
            )
        ),
    )
    monkeypatch.setattr(tool._CampaignLock, "__enter__", lambda self: self)
    monkeypatch.setattr(tool._CampaignLock, "__exit__", lambda self, *args: None)
    with pytest.raises(tool.GatePending) as caught:
        tool.execute_once_v1(capsule["capsule_digest"])
    assert caught.value.capsule == capsule
    assert caught.value.capsule_digest_hint == capsule["capsule_digest"]


def test_source_gate_binds_required_parent_descendant_and_diffs_from_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()
    ancestor = "a" * 40
    required_parent = "b" * 40
    head = "c" * 40
    tree = "d" * 40
    relative = "tools/example.py"
    calls: list[tuple[str, ...]] = []

    def git(*arguments: str, timeout: float = 60.0) -> str:
        del timeout
        calls.append(arguments)
        values = {
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("rev-parse", "HEAD"): head,
            ("rev-parse", "HEAD^{tree}"): tree,
            ("rev-parse", "--show-prefix"): "Hegel Machine/",
            ("rev-parse", "HEAD^"): required_parent,
            ("merge-base", ancestor, required_parent): ancestor,
            (
                "diff",
                "--name-only",
                "--diff-filter=A",
                ancestor,
                "HEAD",
            ): f"Hegel Machine/{relative}",
            (
                "diff",
                "--name-only",
                "--diff-filter=DMRTUXB",
                ancestor,
                "HEAD",
            ): "",
        }
        return values[arguments]

    def run(command, **_kwargs):
        if command[1:2] == ["fsck"]:
            return subprocess.CompletedProcess(command, 0, b"", b"")
        assert command[1:6] == ["ls-tree", "-z", "-l", "HEAD", "--"]
        payload = f"100644 blob {'e' * 40} 7\t{relative}\0".encode()
        return subprocess.CompletedProcess(command, 0, payload, b"")

    monkeypatch.setattr(tool, "_git", git)
    monkeypatch.setattr(tool.subprocess, "run", run)
    receipt = tool._source_gate(
        {
            "source_lineage": {
                "project_root": str(tool.PROJECT_ROOT),
                "required_ancestor_commit": ancestor,
                "required_parent_commit": required_parent,
                "additive_paths_in_exact_order": [relative],
            }
        }
    )
    assert receipt["source_commit"] == head
    assert receipt["source_tree"] == tree
    assert (
        "diff",
        "--name-only",
        "--diff-filter=A",
        ancestor,
        "HEAD",
    ) in calls
    assert not any(
        call[:3] == ("diff", "--name-only", "--diff-filter=A")
        and required_parent in call
        for call in calls
    )


def test_superseded_campaign_replays_terminal_and_proves_zero_measurement_start() -> None:
    tool = _load_tool()
    receipt = tool._superseded_campaign_gate(_contract(tool))
    assert receipt["terminal_status"] == tool._core.FAILED_TERMINAL
    assert receipt["measurement_endpoint_start_count"] == 0
    assert receipt["trajectory_event_count"] == 8
    assert receipt["q1_state"] == "NOT_RUN"
    assert receipt["retry_or_resume_allowed"] is False


@pytest.mark.parametrize(
    ("head_parent", "merge_base"),
    (("f" * 40, "a" * 40), ("b" * 40, "f" * 40)),
)
def test_source_gate_rejects_either_broken_lineage_hop(
    monkeypatch: pytest.MonkeyPatch,
    head_parent: str,
    merge_base: str,
) -> None:
    tool = _load_tool()
    ancestor = "a" * 40
    required_parent = "b" * 40

    def git(*arguments: str, timeout: float = 60.0) -> str:
        del timeout
        values = {
            ("status", "--porcelain=v1", "--untracked-files=all"): "",
            ("rev-parse", "HEAD"): "c" * 40,
            ("rev-parse", "HEAD^{tree}"): "d" * 40,
            ("rev-parse", "--show-prefix"): "Hegel Machine/",
            ("rev-parse", "HEAD^"): head_parent,
            ("merge-base", ancestor, required_parent): merge_base,
        }
        return values[arguments]

    monkeypatch.setattr(tool, "_git", git)
    with pytest.raises(tool.CampaignTerminal) as caught:
        tool._source_gate(
            {
                "source_lineage": {
                    "project_root": str(tool.PROJECT_ROOT),
                    "required_ancestor_commit": ancestor,
                    "required_parent_commit": required_parent,
                    "additive_paths_in_exact_order": ["tools/example.py"],
                }
            }
        )
    assert caught.value.code == "SOURCE_LINEAGE_DIFFERS"


def test_run_path_pending_before_capsule_replay_echoes_requested_digest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool()
    digest = "sha256:" + "a" * 64
    monkeypatch.setattr(
        tool,
        "load_contract_v1",
        lambda: (_ for _ in ()).throw(
            tool.GatePending(
                "PATH_AND_FORMAT",
                "READONLY_FILE_UNAVAILABLE",
                "synthetic contract read failure",
            )
        ),
    )
    assert tool.main(["--run", "--capsule-digest", digest]) == tool.PENDING_EXIT_CODE
    payload = json.loads(capsys.readouterr().out)
    assert payload["capsule_digest_or_null"] == digest
    assert payload["authorization_consumption_count"] == 0
    assert payload["supervisor_popen_count"] == 0


def test_dependency_setup_failure_removes_its_transient_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    cargo_cache = tmp_path / "cargo-cache"
    cargo_cache.mkdir()
    (cargo_cache / "registry").mkdir()
    probe = tmp_path / "probe"
    contract = {
        "fixed_paths": {
            "cargo_cache_source": str(cargo_cache),
            "probe_root": str(probe),
        },
        "target": {
            "python_endpoint_relative_path": (
                "tools/phase3_q1_full_node6_capacity_entrypoint_v1.py"
            ),
            "rust_manifest_relative_path": "rust/example/Cargo.toml",
            "rust_binary_name": "example",
        },
    }
    def completed(command, **_kwargs):
        stdout = (
            b"3.11.15 33\n"
            if command[:2] == ["docker", "run"] and tool.PYTHON_IMAGE in command
            else b"ok\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setattr(tool, "_run_readonly_command", completed)

    def fail_copy(*_args, **_kwargs):
        raise OSError("synthetic copy failure")

    monkeypatch.setattr(tool.shutil, "copytree", fail_copy)
    with pytest.raises(tool.GatePending) as caught:
        tool._dependency_gate(contract, tool._git("rev-parse", "HEAD"))
    assert caught.value.condition == "CARGO_OFFLINE_CLOSURE_UNAVAILABLE"
    assert not probe.exists()


def test_readonly_command_pending_includes_bounded_process_output() -> None:
    tool = _load_tool()
    with pytest.raises(tool.GatePending) as caught:
        tool._run_readonly_command(
            [
                "/usr/bin/python3.10",
                "-c",
                (
                    "import sys;sys.stdout.write('recipe-out');"
                    "sys.stderr.write('permission denied');sys.exit(7)"
                ),
            ],
            phase="PYTHON_RUST_OFFLINE_DEPENDENCIES",
            condition="PYTHON_SOURCE_RECIPE_QUALIFICATION_FAILED",
            timeout=15,
        )
    assert "return_code=7" in caught.value.detail
    assert "stdout_tail='recipe-out'" in caught.value.detail
    assert "stderr_tail='permission denied'" in caught.value.detail


def test_exact_commit_materialization_failure_is_pending_and_cleans_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    cargo_cache = tmp_path / "cargo-cache"
    cargo_cache.mkdir()
    probe = tmp_path / "probe"

    class SnapshotFailure(RuntimeError):
        code = "FAIL_FULL_NODE6_SUPERVISOR_SOURCE"
        detail = "synthetic nested repository blob failure"

    def fail_materialization(_project, _commit, target, _paths):
        target.mkdir()
        raise SnapshotFailure

    supervisor = SimpleNamespace(
        FullNode6SupervisorError=SnapshotFailure,
        PYTHON_ENDPOINT_SOURCE_PATHS=(
            "tools/phase3_q1_full_node6_capacity_entrypoint_v1.py",
        ),
        RUST_ENDPOINT_SOURCE_PATHS=("rust/example/Cargo.toml",),
        HOST_REPLAY_SOURCE_PATHS=("src/example.py",),
        EXECUTION_CLOSURE_ROW_COUNT=2,
        _repository_tree_prefix_v1=lambda _project: "Hegel Machine/",
        _materialize_snapshot=fail_materialization,
    )
    monkeypatch.setattr(tool, "_load_supervisor_module_v1", lambda: supervisor)
    contract = {
        "fixed_paths": {
            "cargo_cache_source": str(cargo_cache),
            "probe_root": str(probe),
        },
        "target": {
            "python_endpoint_relative_path": (
                "tools/phase3_q1_full_node6_capacity_entrypoint_v1.py"
            ),
            "rust_manifest_relative_path": "rust/example/Cargo.toml",
            "rust_binary_name": "example",
        },
    }
    with pytest.raises(tool.GatePending) as caught:
        tool._dependency_gate(contract, "1" * 40)
    assert caught.value.phase == "EXACT_COMMIT_SNAPSHOT_MATERIALIZATION"
    assert caught.value.condition == "EXACT_COMMIT_BLOB_MATERIALIZATION_UNAVAILABLE"
    assert not probe.exists()


def test_exact_materialization_pending_precedes_all_state_and_popen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    digest = "sha256:" + "a" * 64
    popen_count = 0

    def unexpected_postconsumption(*_args, **_kwargs):
        nonlocal popen_count
        popen_count += 1
        raise AssertionError("postconsumption path must remain unreachable")

    monkeypatch.setattr(tool, "load_contract_v1", lambda: contract)
    monkeypatch.setattr(tool, "_replay_existing", lambda *_args: None)
    monkeypatch.setattr(
        tool,
        "derive_preflight_v1",
        lambda **_kwargs: (_ for _ in ()).throw(
            tool.GatePending(
                "EXACT_COMMIT_SNAPSHOT_MATERIALIZATION",
                "EXACT_COMMIT_BLOB_MATERIALIZATION_UNAVAILABLE",
                "synthetic exact commit blob failure",
            )
        ),
    )
    monkeypatch.setattr(tool, "_execute_postconsumption_v1", unexpected_postconsumption)
    monkeypatch.setattr(tool._CampaignLock, "__enter__", lambda self: self)
    monkeypatch.setattr(tool._CampaignLock, "__exit__", lambda self, *args: None)
    with pytest.raises(tool.GatePending) as caught:
        tool.execute_once_v1(digest)
    assert caught.value.capsule_digest_hint == digest
    assert popen_count == 0
    for name in (
        "ledger_root",
        "evidence_root",
        "trajectory_root",
        "run_root",
        "probe_root",
        "artifact_path",
    ):
        assert not Path(contract["fixed_paths"][name]).exists()


def test_dependency_receipt_is_source_recipe_bound_and_probe_is_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    cargo_cache = tmp_path / "cargo-cache"
    cargo_cache.mkdir()
    (cargo_cache / "registry").mkdir()
    probe = tmp_path / "probe"
    contract = {
        "fixed_paths": {
            "cargo_cache_source": str(cargo_cache),
            "probe_root": str(probe),
        },
        "target": {
            "python_endpoint_relative_path": (
                "tools/phase3_q1_full_node6_capacity_entrypoint_v1.py"
            ),
            "rust_manifest_relative_path": "rust/example/Cargo.toml",
            "rust_binary_name": "example",
        },
    }

    commands: list[list[str]] = []
    readonly_roots: list[str] = []
    supervisor = tool._load_supervisor_module_v1()
    original_make_tree_readonly = supervisor._make_tree_readonly

    def make_tree_readonly(root: Path) -> None:
        readonly_roots.append(root.name)
        original_make_tree_readonly(root)

    monkeypatch.setattr(supervisor, "_make_tree_readonly", make_tree_readonly)
    monkeypatch.setattr(tool, "_load_supervisor_module_v1", lambda: supervisor)

    def completed(command, **_kwargs):
        commands.append(list(command))
        if command[:4] == ["docker", "image", "inspect", "--format"]:
            stdout = b"sha256:" + b"1" * 64 + b"\n"
        elif command[:2] == ["docker", "run"] and tool.PYTHON_IMAGE in command:
            stdout = b"3.11.15 33\n"
        elif "metadata" in command:
            stdout = b'{"packages":[]}\n'
        elif command[:1] == ["/usr/bin/python3.10"]:
            stdout = b"Python 3.10.12\n"
        else:
            stdout = b""
        return subprocess.CompletedProcess(command, 0, stdout, b"")

    monkeypatch.setattr(tool, "_run_readonly_command", completed)
    monkeypatch.setattr(
        tool,
        "_sha256_file",
        lambda *_args, **_kwargs: ("sha256:" + "2" * 64, 12345),
    )
    receipt = tool._dependency_gate(contract, tool._git("rev-parse", "HEAD"))
    assert receipt["cargo_test_locked_offline_all_targets"] == "PASS"
    assert receipt["rust_release_binary_sha256"] == "sha256:" + "2" * 64
    assert receipt["python_endpoint_runtime_version"] == "3.11.15"
    assert receipt["python_endpoint_loaded_project_module_count"] == 33
    assert receipt["python_endpoint_source_recipe"] == "PASS"
    assert receipt["python_endpoint_source_row_count"] == 34
    assert receipt["rust_release_build_source_row_count"] == 21
    assert receipt["endpoint_execution_union_row_count"] == 55
    assert receipt["exact_execution_source_closure"] == "PASS"
    assert receipt["exact_commit_snapshot_materialization"] == "PASS"
    assert receipt["host_replay_source_row_count"] == 1
    assert receipt["repository_tree_prefix"] == "Hegel Machine/"
    assert readonly_roots == ["python-source"]
    python_recipe = next(
        command
        for command in commands
        if command[:2] == ["docker", "run"] and tool.PYTHON_IMAGE in command
    )
    assert ["--network", "none"] == python_recipe[
        python_recipe.index("--network") : python_recipe.index("--network") + 2
    ]
    assert ["--user", "65534:65534"] == python_recipe[
        python_recipe.index("--user") : python_recipe.index("--user") + 2
    ]
    assert ["--ipc", "private"] == python_recipe[
        python_recipe.index("--ipc") : python_recipe.index("--ipc") + 2
    ]
    assert "nofile=256:256" in python_recipe
    python_mount = python_recipe[python_recipe.index("--mount") + 1]
    assert "/python-source,dst=/snapshot,readonly" in python_mount
    rust_build = next(command for command in commands if "build" in command)
    rust_build_mounts = [
        rust_build[index + 1]
        for index, item in enumerate(rust_build[:-1])
        if item == "--mount"
    ]
    assert any(
        "/rust-release-source,dst=/snapshot,readonly" in item
        for item in rust_build_mounts
    )
    assert rust_build[rust_build.index("--cpuset-cpus") + 1] == "12-23"
    assert rust_build[rust_build.index("--memory") + 1] == "15032385536"
    assert rust_build[rust_build.index("--pids-limit") + 1] == "128"
    assert not any("dst=/test-target" in item for item in rust_build_mounts)
    assert rust_build[-6:] == [
        "build",
        "--release",
        "--offline",
        "--locked",
        "--target-dir",
        "/build/target",
    ]
    rust_test = next(command for command in commands if "test" in command)
    rust_test_mounts = [
        rust_test[index + 1]
        for index, item in enumerate(rust_test[:-1])
        if item == "--mount"
    ]
    assert any(
        f"type=bind,src={tool.PROJECT_ROOT},dst=/snapshot,readonly" == item
        for item in rust_test_mounts
    )
    assert "cargo_cache_bytes" not in receipt
    assert not probe.exists()


def test_outer_supervisor_command_binds_the_fixed_trajectory_root() -> None:
    tool = _load_tool()
    contract = _contract(tool)
    capsule = _capsule(tool, contract)
    command = tool._supervisor_command(contract, capsule)
    index = command.index("--trajectory-root")
    assert command[index + 1] == contract["fixed_paths"]["trajectory_root"]
    assert command.count("--trajectory-root") == 1


def test_technical_gate_replays_frozen_measurement_and_isolation_contracts() -> None:
    tool = _load_tool()
    receipt = tool._technical_contract_gate(_contract(tool))
    assert receipt["frozen_measurement_contract_replayed"] is True
    assert receipt["frozen_runtime_isolation_profile_replayed"] is True
    assert receipt["execution_source_closure_replayed"] is True
    assert receipt["python_endpoint_source_row_count"] == 34
    assert receipt["rust_release_build_source_row_count"] == 21
    assert receipt["endpoint_execution_union_row_count"] == 55
    assert receipt["host_replay_source_row_count"] == 1
    assert receipt["endpoint_execution_order"] == [
        "PYTHON_ENDPOINT",
        "RUST_ENDPOINT",
    ]
    assert receipt["maximum_concurrently_live_measurement_containers"] == 1
    assert receipt["minimum_available_memory_bytes"] == 16 * 1024**3


def test_resource_gate_uses_one_14_gib_actor_plus_2_gib_host_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_tool()

    class Storage:
        f_bavail = 200 * 1024**3
        f_frsize = 1

    monkeypatch.setattr(tool.os, "statvfs", lambda _path: Storage())
    monkeypatch.setattr(tool, "_mem_available_bytes", lambda: 20_000_000_000)
    receipt = tool._resource_gate(_contract(tool))
    assert receipt["active_container_memory_bytes"] == 14 * 1024**3
    assert receipt["host_control_plane_memory_reserve_bytes"] == 2 * 1024**3
    assert receipt["required_available_memory_bytes"] == 16 * 1024**3

    monkeypatch.setattr(
        tool,
        "_mem_available_bytes",
        lambda: 16 * 1024**3 - 1,
    )
    with pytest.raises(tool.GatePending) as caught:
        tool._resource_gate(_contract(tool))
    assert caught.value.phase == "MEMORY_AND_STORAGE_RESERVATION"
    assert caught.value.condition == "REQUIRED_AVAILABLE_MEMORY_UNAVAILABLE"


def test_guard_and_failed_replay_outcomes_preserve_published_artifacts() -> None:
    tool = _load_tool()
    contract = _contract(tool)
    capsule = _capsule(tool, contract)
    stdout = "sha256:" + "a" * 64
    stderr = "sha256:" + "b" * 64
    artifact = "sha256:" + "c" * 64
    guard_decision = tool._core.decide_supervisor_return_v1(
        return_code=0,
        terminal_status_or_null=tool._core.GUARD_HIT,
        replay_passed=True,
    )
    guard = tool._outcome_value(
        contract,
        capsule,
        return_code=0,
        terminal_status=tool._core.GUARD_HIT,
        replay_passed=True,
        stdout_sha256=stdout,
        stderr_sha256=stderr,
        artifact_sha256=artifact,
        decision=guard_decision,
    )
    assert tool._validate_outcome_v1(guard, contract, capsule) == guard

    failed_decision = tool._core.decide_supervisor_return_v1(
        return_code=3,
        terminal_status_or_null=tool._core.RESOURCE_INFEASIBLE,
        replay_passed=False,
    )
    failed = tool._outcome_value(
        contract,
        capsule,
        return_code=3,
        terminal_status=tool._core.RESOURCE_INFEASIBLE,
        replay_passed=False,
        stdout_sha256=stdout,
        stderr_sha256=stderr,
        artifact_sha256=artifact,
        decision=failed_decision,
    )
    assert failed["status"] == tool._core.FAILED_TERMINAL
    assert tool._validate_outcome_v1(failed, contract, capsule) == failed


def test_no_replace_publication_never_exposes_partial_final_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    root = tmp_path / "publication"
    root.mkdir(mode=0o700)

    def partial_then_fail(descriptor: int, _payload: bytes) -> None:
        os.write(descriptor, b"{")
        raise OSError("synthetic partial write")

    monkeypatch.setattr(tool, "_write_all", partial_then_fail)
    with pytest.raises(OSError, match="synthetic partial write"):
        tool._publish_no_replace(root, "record.json", {"complete": True})
    assert not (root / "record.json").exists()
    assert list(root.iterdir()) == []


def test_no_replace_publication_preserves_existing_exact_bytes(tmp_path: Path) -> None:
    tool = _load_tool()
    root = tmp_path / "publication"
    root.mkdir(mode=0o700)
    first = {"sequence": 1}
    expected = tool._core.canonical_json_bytes_v1(first)
    tool._publish_no_replace(root, "record.json", first)

    with pytest.raises(OSError):
        tool._publish_no_replace(root, "record.json", {"sequence": 2})
    assert (root / "record.json").read_bytes() == expected
    assert tool._publish_or_replay_exact(root, "record.json", first) == expected
    assert [path.name for path in root.iterdir()] == ["record.json"]


def test_consumed_publication_failure_before_final_link_rolls_back_to_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    ledger.mkdir(mode=0o700)
    monkeypatch.setattr(
        tool,
        "_publish_no_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic failure before final link")
        ),
    )
    with pytest.raises(tool.GatePending) as caught:
        tool._publish_consumed_or_pending_v1(
            ledger,
            consumed,
            capsule,
            capsule["capsule_digest"],
        )
    assert caught.value.condition == "CONSUMED_LEDGER_PUBLICATION_NOT_COMMITTED"
    assert caught.value.capsule == capsule
    assert not ledger.exists()


def test_consumed_publication_exact_final_after_error_remains_consumed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    ledger.mkdir(mode=0o700)

    def link_then_fail(root: Path, name: str, value: object) -> bytes:
        (root / name).write_bytes(tool._core.canonical_json_bytes_v1(value))
        raise OSError("synthetic error after final link")

    monkeypatch.setattr(tool, "_publish_no_replace", link_then_fail)
    tool._publish_consumed_or_pending_v1(
        ledger,
        consumed,
        capsule,
        capsule["capsule_digest"],
    )
    assert json.loads((ledger / "campaign.consumed.json").read_bytes()) == consumed


def test_consumed_publication_nonexact_final_is_terminal_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    ledger.mkdir(mode=0o700)

    def corrupt_then_fail(root: Path, name: str, _value: object) -> bytes:
        (root / name).write_bytes(b"{}\n")
        raise OSError("synthetic ambiguous final link")

    monkeypatch.setattr(tool, "_publish_no_replace", corrupt_then_fail)
    with pytest.raises(tool.CampaignTerminal) as caught:
        tool._publish_consumed_or_pending_v1(
            ledger,
            consumed,
            capsule,
            capsule["capsule_digest"],
        )
    assert caught.value.code == "CONSUMED_LEDGER_PUBLICATION_UNCERTAIN"


def test_consumed_publication_recovers_real_postlink_directory_fsync_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    ledger.mkdir(mode=0o700)
    real_fsync = tool.os.fsync
    fsync_calls = 0

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("synthetic first ledger directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(tool.os, "fsync", fail_first_directory_fsync)
    tool._publish_consumed_or_pending_v1(
        ledger,
        consumed,
        capsule,
        capsule["capsule_digest"],
    )
    assert fsync_calls == 3
    assert json.loads((ledger / "campaign.consumed.json").read_bytes()) == consumed
    assert [path.name for path in ledger.iterdir()] == ["campaign.consumed.json"]


def test_consumed_exact_final_with_temporary_residue_is_terminal_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    ledger.mkdir(mode=0o700)

    def leave_residue(root: Path, name: str, value: object) -> bytes:
        payload = tool._core.canonical_json_bytes_v1(value)
        (root / name).write_bytes(payload)
        (root / ".campaign.consumed.json.tmp-residue").write_bytes(payload)
        raise OSError("synthetic temporary-link cleanup failure")

    monkeypatch.setattr(tool, "_publish_no_replace", leave_residue)
    with pytest.raises(tool.CampaignTerminal) as caught:
        tool._publish_consumed_or_pending_v1(
            ledger,
            consumed,
            capsule,
            capsule["capsule_digest"],
        )
    assert caught.value.code == "CONSUMED_LEDGER_PUBLICATION_UNCERTAIN"


def test_finalizer_publishes_terminal_when_external_outcome_publication_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    ledger.mkdir(mode=0o700)
    tool._publish_no_replace(ledger, "campaign.consumed.json", consumed)
    outcome = tool._unknown_outcome_value(
        contract,
        capsule,
        supervisor_popen_count=0,
        return_code=None,
        stdout_sha256=None,
        stderr_sha256=None,
        detail="synthetic post-consumption failure",
    )
    real_publish = tool._publish_or_replay_exact

    def fail_external(root: Path, name: str, value: object) -> bytes:
        if name == "campaign.outcome.json":
            raise OSError("synthetic external outcome failure")
        return real_publish(root, name, value)

    monkeypatch.setattr(tool, "_publish_or_replay_exact", fail_external)
    checked, terminal = tool._finalize_consumed_v1(
        contract,
        capsule,
        consumed,
        outcome,
    )
    assert checked == outcome
    assert terminal["status"] == tool._core.OUTCOME_UNKNOWN_TERMINAL
    assert terminal["body"]["outcome"] == outcome
    assert terminal["body"]["external_outcome_state"] == "UNAVAILABLE_OR_INVALID"
    assert "synthetic external outcome failure" in terminal["body"][
        "external_outcome_error_or_null"
    ]
    assert (ledger / "campaign.terminal.json").is_file()
    assert not (
        Path(contract["fixed_paths"]["evidence_root"]) / "campaign.outcome.json"
    ).exists()


def test_reentry_uses_strict_existing_outcome_and_completes_terminal(tmp_path: Path) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    evidence = Path(contract["fixed_paths"]["evidence_root"])
    ledger.mkdir(mode=0o700)
    evidence.mkdir(mode=0o700)
    tool._publish_no_replace(ledger, "campaign.consumed.json", consumed)
    outcome = _failed_outcome(tool, contract, capsule)
    tool._publish_no_replace(evidence, "campaign.outcome.json", outcome)

    result = tool._replay_existing(contract, capsule["capsule_digest"])
    assert result is not None
    assert result["status"] == tool._core.FAILED_TERMINAL
    assert result["replayed_existing_terminal"] is True
    assert result["authorization_consumption_count_this_entry"] == 0
    assert result["supervisor_popen_count_this_entry"] == 0
    terminal = tool._read_record(ledger / "campaign.terminal.json")
    assert terminal is not None
    assert terminal["body"]["outcome"] == outcome
    tool._validate_terminal_record_v1(terminal, contract, capsule, consumed)


def test_reentry_without_outcome_publishes_explicit_unknown_and_terminal(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    evidence = Path(contract["fixed_paths"]["evidence_root"])
    ledger.mkdir(mode=0o700)
    tool._publish_no_replace(ledger, "campaign.consumed.json", consumed)

    result = tool._replay_existing(contract, capsule["capsule_digest"])
    assert result is not None
    assert result["status"] == tool._core.OUTCOME_UNKNOWN_TERMINAL
    assert result["supervisor_popen_count"] is None
    assert result["authorization_consumption_count_this_entry"] == 0
    assert result["supervisor_popen_count_this_entry"] == 0
    outcome = json.loads((evidence / "campaign.outcome.json").read_bytes())
    assert outcome["status"] == tool._core.OUTCOME_UNKNOWN_TERMINAL
    assert outcome["supervisor_popen_count"] is None
    assert outcome["postconsumption_error_or_null"].startswith(
        "CONSUMED_WITHOUT_TERMINAL_REENTRY:"
    )
    terminal = tool._read_record(ledger / "campaign.terminal.json")
    assert terminal is not None
    assert terminal["body"]["outcome"] == outcome
    tool._validate_terminal_record_v1(terminal, contract, capsule, consumed)


def test_unreadable_existing_ledger_is_terminal_not_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    path = tmp_path / "campaign.consumed.json"
    path.write_bytes(b"{}\n")
    monkeypatch.setattr(
        tool,
        "_read_regular",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            tool.GatePending(
                "PATH_AND_FORMAT",
                "READONLY_FILE_UNAVAILABLE",
                "synthetic ledger read failure",
            )
        ),
    )
    with pytest.raises(tool.CampaignTerminal) as caught:
        tool._read_record(path)
    assert caught.value.code == "LEDGER_RECORD_UNREADABLE"


def test_postconsumption_popen_failure_finalizes_unknown_without_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    evidence = Path(contract["fixed_paths"]["evidence_root"])
    ledger.mkdir(mode=0o700)
    tool._publish_no_replace(ledger, "campaign.consumed.json", consumed)

    def fail_popen(*_args, **_kwargs):
        raise OSError("synthetic Popen failure")

    monkeypatch.setattr(tool.subprocess, "Popen", fail_popen)
    result = tool._execute_postconsumption_v1(contract, capsule, consumed)
    assert result["status"] == tool._core.OUTCOME_UNKNOWN_TERMINAL
    assert result["authorization_consumption_count"] == 1
    assert result["supervisor_popen_count"] == 0
    assert result["supervisor_popen_count_this_entry"] == 0
    terminal = tool._read_record(ledger / "campaign.terminal.json")
    assert terminal is not None
    outcome = terminal["body"]["outcome"]
    assert outcome["status"] == tool._core.OUTCOME_UNKNOWN_TERMINAL
    assert outcome["supervisor_popen_count"] == 0
    assert "synthetic Popen failure" in outcome["postconsumption_error_or_null"]
    assert terminal["body"]["retry_or_resume_allowed"] is False
    if (evidence / "campaign.outcome.json").exists():
        assert json.loads((evidence / "campaign.outcome.json").read_bytes()) == outcome


def test_postconsumption_stdout_read_pending_becomes_unknown_after_one_popen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    contract = _temporary_contract(tool, tmp_path)
    capsule = _capsule(tool, contract)
    consumed = _consumed(tool, contract, capsule)
    ledger = Path(contract["fixed_paths"]["ledger_root"])
    ledger.mkdir(mode=0o700)
    tool._publish_no_replace(ledger, "campaign.consumed.json", consumed)

    class CompletedChild:
        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

        def kill(self):
            raise AssertionError("completed child must not be killed")

    real_read = tool._read_regular

    def fail_stdout(path: Path, label: str, maximum=tool.MAX_SMALL_JSON):
        if label == "supervisor stdout":
            raise tool.GatePending(
                "PATH_AND_FORMAT",
                "READONLY_FILE_UNAVAILABLE",
                "synthetic stdout read failure",
            )
        return real_read(path, label, maximum)

    monkeypatch.setattr(tool.subprocess, "Popen", lambda *_args, **_kwargs: CompletedChild())
    monkeypatch.setattr(tool, "_read_regular", fail_stdout)
    result = tool._execute_postconsumption_v1(contract, capsule, consumed)
    assert result["status"] == tool._core.OUTCOME_UNKNOWN_TERMINAL
    assert result["supervisor_popen_count"] == 1
    assert result["supervisor_popen_count_this_entry"] == 1
    terminal = json.loads((ledger / "campaign.terminal.json").read_bytes())
    outcome = terminal["body"]["outcome"]
    assert outcome["supervisor_popen_count"] == 1
    assert "synthetic stdout read failure" in outcome["postconsumption_error_or_null"]
    assert terminal["body"]["retry_or_resume_allowed"] is False
