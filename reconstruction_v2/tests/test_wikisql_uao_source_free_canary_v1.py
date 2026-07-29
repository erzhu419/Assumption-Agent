from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
from typing import Mapping

import pytest

from assumption_agent.benchmarks import (
    wikisql_uao_action_runtime_v1 as action_runtime,
)
from replication_runtime.wikisql_uao_formal_v1 import runner as formal
from replication_runtime.wikisql_uao_source_free_canary_v1 import (
    runner as canary,
)


@pytest.fixture()
def posix_root() -> Path:
    with tempfile.TemporaryDirectory(
        prefix="wikisql-uao-canary-test-", dir="/tmp"
    ) as raw:
        yield Path(raw)


def _write(path: Path, raw: bytes, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(descriptor, mode)
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_binding(path: Path) -> dict[str, object]:
    metadata = path.lstat()
    raw = path.read_bytes()
    return {
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _tree_binding(path: Path) -> dict[str, object]:
    digest, count, size = formal.tree_identity(path)
    return {
        "file_count": count,
        "path": str(path),
        "sha256": digest,
        "total_bytes": size,
    }


def _fixture(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    extra: Mapping[str, object] | None = None,
    source_tree: str | None = None,
) -> tuple[canary.Config | None, Path]:
    monkeypatch.setattr(canary, "CANARY_ROOT", root)
    root.mkdir(mode=0o700)
    control = root / "control"
    work = root / "work"
    control.mkdir(mode=0o700)
    work.mkdir(mode=0o700)
    code = root / "reconstruction_v2"
    service = code / canary.SERVICE_RELATIVE_PATH
    _write(service, b"[Service]\nType=oneshot\n", 0o644)
    file_paths = {
        "python_executable": root / "bin/common-python",
        "official_python_executable": root / "bin/official-python",
        "nvidia_smi_executable": root / "bin/nvidia-smi",
        "service_unit": service,
        "systemctl_executable": root / "bin/systemctl",
    }
    for name, path in file_paths.items():
        if name != "service_unit":
            _write(path, (name + "\n").encode(), 0o755)
    tree_paths: dict[str, Path] = {}
    for name in canary.TREES:
        if name == "code_tree":
            tree_paths[name] = code
            continue
        path = (
            root / "runtime_assets/babel_2_10_3_clean"
            if name == "babel_dependency_tree"
            else root / "assets" / name
        )
        if name == source_tree:
            path = root / "formal/source" / name
        if name == "babel_dependency_tree":
            _write(path / "babel/__init__.py", b'__version__ = "2.10.3"\n', 0o644)
        else:
            _write(path / "bound.txt", (name + "\n").encode(), 0o644)
        tree_paths[name] = path
    files = {name: _file_binding(path) for name, path in file_paths.items()}
    trees = {name: _tree_binding(path) for name, path in tree_paths.items()}
    semantic = action_runtime.directory_tree_sha256(
        tree_paths["encoder_model_tree"]
    )
    body: dict[str, object] = {
        "bindings": {"files": files, "trees": trees},
        "canary_root": str(root),
        "encoder_model_semantic_sha256": semantic,
        "expected_babel_version": "2.10.3",
        "gpu_uuids": {
            "0": "GPU-00000000-0000-0000-0000-000000000000",
            "1": "GPU-11111111-1111-1111-1111-111111111111",
        },
        "pythonpath_order": {
            "common": list(canary.COMMON_ORDER),
            "official": list(canary.OFFICIAL_ORDER),
        },
        "schema": canary.CONFIG_SCHEMA,
        "study_id": canary.STUDY_ID,
        "unit_name": canary.UNIT_NAME,
    }
    if extra:
        body.update(extra)
    value = {**body, "self_sha256": formal.semantic_sha256(body)}
    config_path = control / "canary_config.json"
    _write(config_path, formal.canonical_json_bytes(value), 0o600)
    try:
        return canary.load_config(config_path), config_path
    except canary.WikiSQLUAOCanaryError:
        return None, config_path


def _safe_lane(
    config: canary.Config,
    lane: str,
    fields: Mapping[str, object],
) -> dict[str, object]:
    official = lane == "HippoRAG"
    order = canary.OFFICIAL_ORDER if official else canary.COMMON_ORDER
    executable = config.file(
        "official_python_executable" if official else "python_executable"
    )
    babel = config.tree("babel_dependency_tree").path / "babel/__init__.py"
    base = {
        "API_call_count": 0,
        "babel_origin_file_sha256": hashlib.sha256(babel.read_bytes()).hexdigest(),
        "babel_version": "2.10.3",
        "config_self_sha256": config.self_sha256,
        "interpreter_file_sha256": executable.sha256,
        "lane": lane,
        "network_call_count": 0,
        "online_evaluator_call_count": 0,
        "pythonpath_order_sha256": formal.semantic_sha256(
            [str(config.tree(name).path) for name in order]
        ),
        "replay_count": 0,
        "retry_count": 0,
        "schema": canary.LANE_SCHEMAS[lane],
        "status": "passed",
        **fields,
    }
    return {**base, "self_sha256": formal.semantic_sha256(base)}


def test_config_is_exact_and_rejects_source_capabilities(
    posix_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _fixture(monkeypatch, posix_root / "valid")
    assert config is not None

    invalid, _ = _fixture(
        monkeypatch,
        posix_root / "extra",
        extra={"source_archive": "/never/read/data.tar.bz2"},
    )
    assert invalid is None

    invalid, _ = _fixture(
        monkeypatch,
        posix_root / "source-path",
        source_tree="encoder_model_tree",
    )
    assert invalid is None


def test_commands_freeze_two_minus_s_orders_and_three_physical_lanes(
    posix_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _fixture(monkeypatch, posix_root / "commands")
    assert config is not None
    monkeypatch.setattr(
        formal,
        "_gpu_device_paths",
        lambda index: (Path(f"/dev/nvidia{index}"), Path("/dev/nvidiactl")),
    )
    paths = canary.Paths.fixed()
    commands = canary.build_commands(config, paths)
    assert set(commands) == {"Agent", "RAW", "HippoRAG"}
    assert all("-S" in command.argv for command in commands.values())
    assert commands["Agent"].environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert commands["HippoRAG"].environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert commands["RAW"].environment["CUDA_VISIBLE_DEVICES"] == ""
    assert commands["Agent"].environment["PYTHONPATH"].split(os.pathsep) == [
        str(config.tree(name).path) for name in canary.COMMON_ORDER
    ]
    assert commands["HippoRAG"].environment["PYTHONPATH"].split(os.pathsep) == [
        str(config.tree(name).path) for name in canary.OFFICIAL_ORDER
    ]
    assert FORMAL_SOURCE_ABSENT(commands)


def FORMAL_SOURCE_ABSENT(
    commands: Mapping[str, formal.CommandSpec],
) -> bool:
    return all(
        canary.FORMAL_SOURCE_ROOT != path
        and canary.FORMAL_SOURCE_ROOT not in path.parents
        for command in commands.values()
        for path in (*command.read_paths, *command.write_paths)
    )


def test_dependency_probe_requires_exact_babel_origin_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "python"
    _write(python, b"python\n", 0o755)
    roots = [tmp_path / "code", tmp_path / "deps", tmp_path / "babel-root"]
    for root in roots:
        root.mkdir(mode=0o700)
    origin = roots[-1] / "babel/__init__.py"
    _write(origin, b'__version__ = "2.10.3"\n', 0o644)
    fake_babel = SimpleNamespace(__version__="2.10.3", __file__=str(origin))
    monkeypatch.setitem(__import__("sys").modules, "babel", fake_babel)
    monkeypatch.setattr(canary.sys, "executable", str(python))
    monkeypatch.setattr(
        canary.sys,
        "flags",
        SimpleNamespace(no_site=1, no_user_site=1),
    )
    monkeypatch.setattr(canary.sys, "path", [str(tmp_path / "cwd"), *map(str, roots)])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, roots)))
    arguments = argparse.Namespace(
        expected_python=python,
        expected_python_sha256=hashlib.sha256(python.read_bytes()).hexdigest(),
        config_self_sha256="a" * 64,
        pythonpath_root=list(map(str, roots)),
        babel_root=roots[-1],
    )
    receipt = canary._dependency(arguments)
    assert receipt["babel_version"] == "2.10.3"
    fake_babel.__version__ = "2.11.0"
    with pytest.raises(canary.WikiSQLUAOCanaryError, match="Babel"):
        canary._dependency(arguments)


def test_synthetic_input_runs_real_raw_path() -> None:
    view = canary.synthetic_input()
    items = canary.official_contract.validate_input(view)
    assert len(items) == 1
    assert len(items[0].rows) == 11
    action = action_runtime.run_raw(view_pack=view)
    rows = action_runtime.decode_action_pack(
        action,
        expected_block="A_hold",
        expected_arm="RAW",
        expected_action_view_pack_sha256=str(view["self_sha256"]),
    )
    assert len(rows) == 1
    assert len(rows[0]["top5_row_ids"]) == 5


def test_controller_passes_only_safe_aggregates_and_is_one_shot(
    posix_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, config_path = _fixture(monkeypatch, posix_root / "success")
    assert config is not None

    def launcher(
        commands: Mapping[str, formal.CommandSpec],
        child_landlock: object,
        on_launch: object,
    ) -> Mapping[str, int]:
        del child_landlock
        for _ in commands:
            on_launch()  # type: ignore[operator]
        _write(
            commands["Agent"].cwd / "lane.safe.json",
            formal.canonical_json_bytes(
                _safe_lane(
                    config,
                    "Agent",
                    {
                        "cuda_logical_device_count": 1,
                        "embedding_dimension": 384,
                        "embedding_matrix_sha256": "1" * 64,
                        "model_semantic_sha256": config.encoder_model_semantic_sha256,
                        "request_count": 2,
                    },
                )
            ),
            0o600,
        )
        _write(
            commands["RAW"].cwd / "lane.safe.json",
            formal.canonical_json_bytes(
                _safe_lane(
                    config,
                    "RAW",
                    {
                        "action_pack_sha256": "2" * 64,
                        "cpu_only": True,
                        "input_pack_sha256": "3" * 64,
                        "item_count": 1,
                        "row_count": 11,
                    },
                )
            ),
            0o600,
        )
        _write(
            commands["HippoRAG"].cwd / "lane.safe.json",
            formal.canonical_json_bytes(
                _safe_lane(
                    config,
                    "HippoRAG",
                    {
                        "cuda_logical_device_count": 1,
                        "index_call_count": 1,
                        "item_count": 1,
                        "retrieve_call_count": 1,
                        "row_count": 11,
                    },
                )
            ),
            0o600,
        )
        return {"Agent": 0, "RAW": 0, "HippoRAG": 0}

    terminal = canary.run_controller(
        config_path,
        preflight=lambda _config: ("f" * 32, 8),
        outer=lambda _config, _paths: None,
        launcher=launcher,
        child_landlock=lambda **_kwargs: None,
    )
    assert terminal["status"] == "PASS_WIKISQL_UAO_SOURCE_FREE_PRODUCTION_CANARY"
    assert terminal["formal_source_access_count"] == 0
    assert "question" not in str(terminal)
    with pytest.raises(canary.WikiSQLUAOCanaryError, match="retry is forbidden"):
        canary.run_controller(config_path)


def test_failed_launch_writes_no_retry_failure(
    posix_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config, config_path = _fixture(monkeypatch, posix_root / "failure")

    def launcher(
        commands: Mapping[str, formal.CommandSpec],
        child_landlock: object,
        on_launch: object,
    ) -> Mapping[str, int]:
        del commands, child_landlock
        for _ in range(3):
            on_launch()  # type: ignore[operator]
        return {"Agent": 0, "RAW": 7, "HippoRAG": 0}

    result = canary.run_controller(
        config_path,
        preflight=lambda _config: ("e" * 32, 8),
        outer=lambda _config, _paths: None,
        launcher=launcher,
        child_landlock=lambda **_kwargs: None,
    )
    assert result["status"] == "FAILED_NO_RETRY"
    assert result["failure_stage"] == "three_lane_launch"
    assert canary.Paths.fixed().attempt.exists()
    assert canary.Paths.fixed().failure.exists()


def test_service_is_minimal_user_oneshot_without_capability_pruning() -> None:
    service = (
        Path(__file__).parents[1]
        / "manifests/wikisql-uao-p4-source-free-canary-v1.service"
    ).read_text()
    assert "Type=oneshot" in service
    assert " -S " in service
    assert "Restart=no" in service
    assert "RestrictAddressFamilies=AF_UNIX" in service
    assert "IPAddressDeny=any" in service
    assert "NoNewPrivileges=yes" in service
    assert all(prefix not in service for prefix in formal._FORBIDDEN_SERVICE_PREFIXES)


def test_invocation_identity_comes_from_systemd_not_cleared_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    attestation = SimpleNamespace(invocation_id="a" * 32)
    assert canary._attested_invocation_id(attestation) == "a" * 32
    monkeypatch.setenv("INVOCATION_ID", "b" * 32)
    assert canary._attested_invocation_id(attestation) == "a" * 32
    attestation.invocation_id = "not-an-invocation"
    with pytest.raises(canary.WikiSQLUAOCanaryError, match="InvocationID"):
        canary._attested_invocation_id(attestation)
