from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Callable, Iterator, Mapping

import pytest

from replication_runtime.wikisql_uao_runtime_qualification import (
    contract,
    runner,
)


HEX_A = "a" * 64
GPU_UUIDS = {
    "0": "GPU-00000000-0000-0000-0000-000000000000",
    "1": "GPU-11111111-1111-1111-1111-111111111111",
}


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Use native Linux mode bits for the immutable receipt contract."""

    with tempfile.TemporaryDirectory(
        prefix="wikisql-uao-qualification-runner-", dir="/tmp"
    ) as value:
        yield Path(value)


def _file_row(path: Path) -> dict[str, object]:
    return {
        "mode_octal": "0600",
        "path": str(path),
        "sha256": HEX_A,
        "size_bytes": 1,
    }


def _tree_row(path: Path) -> dict[str, object]:
    return {
        "file_count": 1,
        "path": str(path),
        "sha256": HEX_A,
        "total_bytes": 1,
    }


def _config_payload(root: Path) -> dict[str, object]:
    code = root / "reconstruction_v2"
    files = {
        name: _file_row(root / "assets" / name)
        for name in contract.REQUIRED_FILES
    }
    files["service_unit"] = _file_row(
        code / contract.SERVICE_RELATIVE_PATH
    )
    trees = {
        name: _tree_row(root / "assets" / name)
        for name in contract.REQUIRED_TREES
    }
    trees["code_tree"] = _tree_row(code)
    trees["python_runtime_tree"] = _tree_row(
        contract.PYTHONHOME_ROOT
    )
    trees["official_python_runtime_tree"] = _tree_row(
        contract.PYTHONHOME_ROOT
    )
    trees["babel_dependency_tree"] = _tree_row(contract.BABEL_ROOT)
    trees["official_hipporag_tree"] = _tree_row(
        contract.OFFICIAL_HIPPORAG_ROOT
    )
    trees["official_base_dependency_tree"] = _tree_row(
        contract.OFFICIAL_BASE_ROOT
    )
    return contract.addressed(
        {
            "bindings": {"files": files, "trees": trees},
            "capability_boundary": dict(contract.CAPABILITY_BOUNDARY),
            "encoder_model_semantic_sha256": HEX_A,
            "expected_babel_version": contract.EXPECTED_BABEL_VERSION,
            "gpu_uuids": dict(GPU_UUIDS),
            "pythonpath_order": dict(contract.PYTHONPATH_ORDER),
            "qualification_id": contract.QUALIFICATION_ID,
            "qualification_root": str(root),
            "resource_policy": {"schema": "synthetic-shared-policy"},
            "schema": contract.CONFIG_SCHEMA,
            "unit_name": contract.UNIT_NAME,
        }
    )


def _publish_config(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(contract.canonical_json_bytes(payload))
    path.chmod(0o600)


def test_config_contract_has_no_formal_source_or_scoring_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "qualification"
    monkeypatch.setattr(contract, "QUALIFICATION_ROOT", root)
    monkeypatch.setattr(
        contract,
        "PYTHONHOME_ROOT",
        root / "runtime_assets/python310_clean",
    )
    monkeypatch.setattr(
        contract,
        "BABEL_ROOT",
        root / "runtime_assets/babel_2_10_3_clean",
    )
    monkeypatch.setattr(
        contract,
        "OFFICIAL_HIPPORAG_ROOT",
        root / "runtime_assets/hipporag_source_clean",
    )
    monkeypatch.setattr(
        contract,
        "OFFICIAL_BASE_ROOT",
        root / "runtime_assets/official_base_import_clean",
    )
    path = root / contract.CONFIG_RELATIVE_PATH
    payload = _config_payload(root)
    _publish_config(path, payload)

    observed = contract.load_config(path)

    assert observed.path == path
    assert contract.CAPABILITY_BOUNDARY == {
        "api_or_network_evaluation_authorized": False,
        "classification": "non_scoring_iterative_runtime_qualification",
        "effect_study_attempt_count": 0,
        "evaluator_or_score_paths_bound": 0,
        "formal_source_paths_bound": 0,
        "label_or_qrel_paths_bound": 0,
    }
    assert all(
        forbidden not in binding.path.parts
        for forbidden in (
            "dataset",
            "label",
            "qrel",
            "score",
            "source",
        )
        for binding in (*observed.files.values(), *observed.trees.values())
    )

    extra_capability = dict(payload)
    extra_capability["source_path"] = str(root / "source")
    extra_capability.pop("self_sha256")
    _publish_config(path, contract.addressed(extra_capability))
    with pytest.raises(
        contract.QualificationContractError, match="shape drifted"
    ):
        contract.load_config(path)

    path_capability = _config_payload(root)
    bindings = path_capability["bindings"]
    assert isinstance(bindings, dict)
    files = bindings["files"]
    assert isinstance(files, dict)
    files["python_executable"] = _file_row(
        root / "source" / "python"
    )
    path_capability.pop("self_sha256")
    _publish_config(path, contract.addressed(path_capability))
    with pytest.raises(
        contract.QualificationContractError,
        match="not representable",
    ):
        contract.load_config(path)

    scoring_capability = _config_payload(root)
    scoring_capability["capability_boundary"] = {
        **contract.CAPABILITY_BOUNDARY,
        "evaluator_or_score_paths_bound": 1,
    }
    scoring_capability.pop("self_sha256")
    _publish_config(path, contract.addressed(scoring_capability))
    with pytest.raises(
        contract.QualificationContractError,
        match="identity drifted",
    ):
        contract.load_config(path)


@dataclass
class _BindingProbe:
    calls: list[str]
    name: str
    should_fail: bool = False

    def verify(self, _field: str) -> None:
        self.calls.append(self.name)
        if self.should_fail:
            raise RuntimeError(f"{self.name} failed")


class _StaticConfig:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.files = {
            "file_a": _BindingProbe(
                self.calls, "file_a", should_fail=True
            ),
            "file_b": _BindingProbe(self.calls, "file_b"),
        }
        self.trees = {
            "tree_a": _BindingProbe(
                self.calls, "tree_a", should_fail=True
            ),
            "tree_b": _BindingProbe(self.calls, "tree_b"),
        }

    def file(self, name: str) -> _BindingProbe:
        return self.files[name]

    def tree(self, name: str) -> _BindingProbe:
        return self.trees[name]


def test_static_checks_aggregate_all_independent_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _StaticConfig()
    paths = SimpleNamespace(root=tmp_path)
    observed: list[str] = []

    def fail(name: str) -> Callable[..., Mapping[str, object]]:
        def probe(*_args: object, **_kwargs: object) -> Mapping[str, object]:
            observed.append(name)
            raise RuntimeError(f"{name} failed")

        return probe

    monkeypatch.setattr(
        runner, "_verify_encoder_semantic", fail("encoder")
    )
    monkeypatch.setattr(
        runner, "_service_source_check", fail("service")
    )
    monkeypatch.setattr(
        runner, "_systemctl_properties", fail("systemd")
    )
    monkeypatch.setattr(
        runner, "_landlock_check", lambda: {"abi": 6}
    )
    monkeypatch.setattr(
        runner, "_dev_null_check", lambda: {"dev_null_o_rdwr": True}
    )
    monkeypatch.setattr(
        runner, "_alias_path_check", lambda *_: {"name_max_bytes": 255}
    )
    monkeypatch.setattr(
        runner, "_command_check", lambda *_: {"lane_count": 3}
    )
    monkeypatch.setattr(
        runner,
        "_capability_check",
        lambda *_: dict(contract.CAPABILITY_BOUNDARY),
    )

    results = runner.static_checks(config, paths, {})
    failed = {
        result.name
        for result in results
        if result.status == "failed"
    }

    assert config.calls == ["file_a", "file_b", "tree_a", "tree_b"]
    assert observed == ["encoder", "service", "systemd"]
    assert failed == {
        "file_binding.file_a",
        "tree_binding.tree_a",
        "encoder_model_semantic_identity",
        "service_source_profile",
        "systemd_effective_profile",
    }
    assert len(results) == 12


class _Process:
    def __init__(self, lane: str, events: list[str]) -> None:
        self.lane = lane
        self.events = events

    def wait(self, *, timeout: float) -> int:
        assert timeout >= 0
        self.events.append(f"wait:{self.lane}")
        return 0


def test_launcher_attempts_all_lanes_before_wait_even_if_first_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    commands = {
        lane: SimpleNamespace(name=lane) for lane in runner.LANES
    }

    def launch(
        command: object, *, child_landlock: Callable[..., None]
    ) -> _Process:
        del child_landlock
        lane = command.name
        events.append(f"launch:{lane}")
        if lane == "Agent":
            raise runner.QualificationRuntimeError("synthetic launch failure")
        return _Process(lane, events)

    monkeypatch.setattr(runner, "_launch_one", launch)
    results = runner.launch_all_and_collect(
        commands,
        child_landlock=lambda **_kwargs: None,
        timeout_seconds=60,
    )

    assert events[:3] == [
        "launch:Agent",
        "launch:RAW",
        "launch:HippoRAG",
    ]
    assert events[3:] == ["wait:RAW", "wait:HippoRAG"]
    assert [result.lane for result in results] == list(runner.LANES)
    assert results[0].launched is False
    assert results[0].launch_error_class == (
        "QualificationRuntimeError"
    )
    assert [(result.launched, result.returncode) for result in results[1:]] == [
        (True, 0),
        (True, 0),
    ]


def test_controller_discovers_invocation_after_env_i_removed_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "stable-qualification"
    config_path = root / contract.CONFIG_RELATIVE_PATH
    _install_controller_fakes(root, config_path, monkeypatch)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr(
        runner,
        "_discover_systemd_invocation_id",
        lambda: "7" * 32,
    )

    terminal = runner.run_controller(
        config_path,
        lock_factory=lambda _path: _Lock(),
        admission_probe=lambda **_kwargs: _admission(
            "DEFERRED_SHARED_RESOURCE"
        ),
        static_probe=lambda *_args: (),
        outer_landlock=lambda *_args: None,
        launcher=lambda _commands: (),
    )

    assert terminal["status"] == "DEFERRED_SHARED_RESOURCE"
    assert terminal["effect_study_attempt_count"] == 0
    assert not (root / "attempts").exists()


class _Lock:
    def __enter__(self) -> bool:
        return True

    def __exit__(self, *_args: object) -> None:
        return None


class _ControllerConfig:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.self_sha256 = "1" * 64
        self.resource_policy: dict[str, object] = {}
        self.gpu_uuids = dict(GPU_UUIDS)

    def file(self, name: str) -> SimpleNamespace:
        assert name == "nvidia_smi_executable"
        return SimpleNamespace(path=Path("/usr/bin/nvidia-smi"))


def _install_controller_fakes(
    root: Path,
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _ControllerConfig:
    root.mkdir(mode=0o700)
    (root / "control").mkdir(mode=0o700)
    config = _ControllerConfig(config_path)
    monkeypatch.setattr(contract, "QUALIFICATION_ROOT", root)
    monkeypatch.setattr(
        contract, "load_config", lambda _path: config
    )

    def commands(
        _config: object, paths: contract.AttemptPaths
    ) -> dict[str, SimpleNamespace]:
        return {
            "Agent": SimpleNamespace(name="Agent", cwd=paths.agent),
            "RAW": SimpleNamespace(name="RAW", cwd=paths.raw),
            "HippoRAG": SimpleNamespace(
                name="HippoRAG", cwd=paths.hippo
            ),
        }

    monkeypatch.setattr(runner, "build_commands", commands)
    return config


def _admission(status: str) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        receipt={
            "effect_attempt_claimed": False,
            "status": status,
        },
    )


def _run(
    config_path: Path,
    invocation_id: str,
    *,
    admission: str,
    static_probe: Callable[..., object],
    launcher: Callable[..., object] = lambda _commands: (),
) -> Mapping[str, object]:
    return runner.run_controller(
        config_path,
        invocation_id=invocation_id,
        lock_factory=lambda _path: _Lock(),
        admission_probe=lambda **_kwargs: _admission(admission),
        static_probe=static_probe,
        outer_landlock=lambda *_args: None,
        launcher=launcher,
    )


def test_shared_resource_deferral_creates_no_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "stable-qualification"
    config_path = root / contract.CONFIG_RELATIVE_PATH
    _install_controller_fakes(root, config_path, monkeypatch)

    terminal = _run(
        config_path,
        "2" * 32,
        admission="DEFERRED_SHARED_RESOURCE",
        static_probe=lambda *_args: (_ for _ in ()).throw(
            AssertionError("static probes must not run while deferred")
        ),
    )

    assert terminal["status"] == "DEFERRED_SHARED_RESOURCE"
    assert terminal["attempt_id"] is None
    assert terminal["retryable"] is True
    assert terminal["effect_study_attempt_count"] == 0
    assert terminal["runtime_lane_launch_count"] == 0
    assert not (root / "attempts").exists()


def test_multiple_invocations_share_one_root_without_overwrite_and_failures_do_not_claim_effect_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "stable-qualification"
    config_path = root / contract.CONFIG_RELATIVE_PATH
    config = _install_controller_fakes(
        root, config_path, monkeypatch
    )
    static = lambda *_args: (
        runner.CheckResult(
            "synthetic.static.a", "failed", {"error": "a"}
        ),
        runner.CheckResult(
            "synthetic.static.b", "failed", {"error": "b"}
        ),
    )

    first = _run(
        config_path,
        "3" * 32,
        admission="ADMITTED",
        static_probe=static,
    )
    second = _run(
        config_path,
        "4" * 32,
        admission="ADMITTED",
        static_probe=static,
    )

    expected_ids = {
        contract.attempt_id("3" * 32, config.self_sha256),
        contract.attempt_id("4" * 32, config.self_sha256),
    }
    assert first["status"] == second["status"] == "FAILED_INFRASTRUCTURE"
    assert {first["attempt_id"], second["attempt_id"]} == expected_ids
    assert {
        child.name for child in (root / "attempts").iterdir()
    } == expected_ids
    for identifier in expected_ids:
        attempt = root / "attempts" / identifier
        terminal = runner._load(
            attempt / "terminal.safe.json", "test terminal"
        )
        started = runner._load(
            attempt / "attempt.started.safe.json", "test started"
        )
        assert terminal["effect_study_attempt_count"] == 0
        assert started["effect_study_attempt_count"] == 0
        assert terminal["failed_check_names"] == [
            "synthetic.static.a",
            "synthetic.static.b",
        ]
    assert contract.UNIT_NAME == (
        "wikisql-uao-runtime-qualification.service"
    )
    assert "formal" not in root.name


@pytest.mark.parametrize(
    ("missing_lane", "expected_status"),
    [
        ("HippoRAG", "FAILED_INFRASTRUCTURE"),
        (None, "PASSED_FULL_STACK"),
    ],
)
def test_passed_terminal_requires_all_three_lane_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_lane: str | None,
    expected_status: str,
) -> None:
    root = tmp_path / "stable-qualification"
    config_path = root / contract.CONFIG_RELATIVE_PATH
    _install_controller_fakes(root, config_path, monkeypatch)

    def verify(path: Path, lane: str, _config: object) -> dict[str, object]:
        if not path.is_file():
            raise runner.QualificationRuntimeError(
                f"{lane} receipt is missing"
            )
        return {"self_sha256": lane.casefold().ljust(64, "0")}

    monkeypatch.setattr(runner, "_verify_lane", verify)

    def launch(commands: Mapping[str, object]) -> tuple[runner.LaunchResult, ...]:
        for lane, command in commands.items():
            if lane != missing_lane:
                (command.cwd / "lane.safe.json").write_bytes(b"receipt\n")
        return tuple(
            runner.LaunchResult(
                lane=lane,
                launched=True,
                returncode=0,
                timed_out=False,
                launch_ordinal=index,
                launch_error_class=None,
            )
            for index, lane in enumerate(runner.LANES, start=1)
        )

    terminal = _run(
        config_path,
        ("5" if missing_lane else "6") * 32,
        admission="ADMITTED",
        static_probe=lambda *_args: (
            runner.CheckResult(
                "synthetic.static", "passed", {"verified": True}
            ),
        ),
        launcher=launch,
    )

    assert terminal["status"] == expected_status
    assert terminal["effect_study_attempt_count"] == 0
    if missing_lane:
        assert terminal["failure_stage"] == "aggregated_runtime_checks"
        assert terminal["failed_check_names"] == [
            f"runtime_lane.{missing_lane}.receipt"
        ]
    else:
        assert terminal["runtime_lane_launch_count"] == 3
        assert terminal["all_three_submitted_before_wait"] is True
