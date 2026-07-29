from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from replication_runtime.wikisql_uao_formal_v1 import runner as formal_v1
from replication_runtime.wikisql_uao_formal_v3 import runner as formal_v3
from replication_runtime.wikisql_uao_formal_v4 import runner as formal_v4
from replication_runtime.wikisql_uao_source_free_canary_v3 import (
    runner as canary_v3,
)
from replication_runtime.wikisql_uao_source_free_canary_v4 import (
    runner as canary_v4,
)


def _commands() -> dict[str, formal_v4.CommandSpec]:
    return {
        "Agent": formal_v4.CommandSpec(
            "Agent",
            ("python", "agent"),
            Path("/private/agent"),
            {"CUDA_VISIBLE_DEVICES": "1"},
            (Path("/private/a-form"),),
            (Path("/private/agent"),),
            (Path("/dev/nvidia1"),),
        ),
        "RAW": formal_v4.CommandSpec(
            "RAW",
            ("python", "raw"),
            Path("/private/raw"),
            {"CUDA_VISIBLE_DEVICES": ""},
            (Path("/private/a-hold-view"),),
            (Path("/private/raw"),),
        ),
        "HippoRAG": formal_v4.CommandSpec(
            "HippoRAG",
            ("python", "hippo"),
            Path("/private/hippo"),
            {"CUDA_VISIBLE_DEVICES": "0"},
            (Path("/private/a-hold-view"),),
            (Path("/private/hippo"),),
            (Path("/dev/nvidia0"),),
        ),
    }


def test_v4_uses_fresh_roots_and_preserves_retired_v3() -> None:
    assert formal_v3.FORMAL_ROOT.name == "formal_v3"
    assert canary_v3.CANARY_ROOT.name == "source_free_canary_v3"
    assert formal_v4.FORMAL_ROOT.name == "formal_v4"
    assert canary_v4.CANARY_ROOT.name == "source_free_canary_v4"


def test_v4_services_bind_private_pythonhome_and_exact_modules() -> None:
    manifests = Path(__file__).parents[1] / "manifests"
    cases = {
        "wikisql-uao-p4-formal-v4.service": (
            formal_v4.PYTHONHOME_ROOT,
            formal_v4.MODULE,
        ),
        "wikisql-uao-p4-source-free-canary-v4.service": (
            canary_v4.PYTHONHOME_ROOT,
            canary_v4.MODULE,
        ),
    }
    for name, (pythonhome, module) in cases.items():
        service = (manifests / name).read_text()
        assert f"PYTHONHOME={pythonhome}" in service
        assert f"-m {module}" in service
        assert "Restart=no" in service
        assert "RestrictAddressFamilies=AF_UNIX" in service
        assert "IPAddressDeny=any" in service
        assert all(
            prefix not in service
            for prefix in formal_v1._FORBIDDEN_SERVICE_PREFIXES
        )


def test_formal_v4_service_profile_rejects_retired_module() -> None:
    raw = (
        Path(__file__).parents[1]
        / "manifests/wikisql-uao-p4-formal-v4.service"
    ).read_bytes()
    trees = {
        "babel_dependency_tree": (
            formal_v4.FORMAL_ROOT / "runtime_assets/babel_2_10_3_clean"
        ),
        "code_tree": formal_v4.FORMAL_ROOT / "reconstruction_v2",
        "python_dependency_tree": Path(
            "/home/erzhu419/p19_runtime_assets_20260723/typed_venv/"
            "lib/python3.10/site-packages"
        ),
    }

    class Config:
        def file(self, name: str) -> SimpleNamespace:
            assert name == "python_executable"
            return SimpleNamespace(
                path=Path(
                    "/home/erzhu419/p19_runtime_assets_20260723/"
                    "typed_venv/bin/python"
                )
            )

        def tree(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(path=trees[name])

    formal_v4._verify_service_profile(raw, Config())
    drifted = raw.replace(
        formal_v4.MODULE.encode("ascii"),
        formal_v3.MODULE.encode("ascii"),
    )
    with pytest.raises(formal_v4.WikiSQLUAOFormalError):
        formal_v4._verify_service_profile(drifted, Config())


def test_v4_outer_layers_admit_proc_before_children_narrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def apply(**kwargs) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(formal_v4._base, "apply_landlock", apply)
    monkeypatch.setattr(
        formal_v4,
        "_all_gpu_device_paths",
        lambda: (Path("/dev/nvidia0"), Path("/dev/nvidia1")),
    )
    config = SimpleNamespace(files={}, trees={})
    formal_v4._outer_landlock(
        config, SimpleNamespace(root=Path("/private/formal"))
    )
    assert Path("/proc") in captured[-1]["write_paths"]
    assert Path("/dev/null") in captured[-1]["write_paths"]
    assert tuple(captured[-1]["device_paths"]) == (
        Path("/dev/nvidia0"),
        Path("/dev/nvidia1"),
    )

    captured.clear()
    canary_v4._outer(
        config, SimpleNamespace(root=Path("/private/canary"))
    )
    assert Path("/proc") in captured[-1]["write_paths"]
    assert Path("/dev/null") in captured[-1]["write_paths"]


def test_formal_v4_gpu_children_narrow_proc_and_keep_logical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        formal_v4, "_original_action_commands", lambda *_: _commands()
    )
    monkeypatch.setattr(
        formal_v4,
        "_all_gpu_device_paths",
        lambda: (Path("/dev/nvidia0"), Path("/dev/nvidia1")),
    )
    paths = SimpleNamespace(
        a_hold_labels=Path("/private/a-hold-labels")
    )
    commands = formal_v4._action_commands(None, paths, None)
    for lane in ("Agent", "HippoRAG"):
        assert Path("/proc/self/task") in commands[lane].write_paths
        assert commands[lane].device_paths == (
            Path("/dev/nvidia0"),
            Path("/dev/nvidia1"),
        )
    assert commands["Agent"].environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert commands["HippoRAG"].environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert commands["RAW"] == _commands()["RAW"]


def test_canary_v4_adds_native_thread_environment() -> None:
    environment = canary_v4._environment(
        Path("/private/lane"),
        "0",
        (Path("/private/modules"),),
    )
    assert environment["VECLIB_MAXIMUM_THREADS"] == "1"
    assert environment["PYTHONHOME"] == str(canary_v4.PYTHONHOME_ROOT)


def test_canary_v4_gpu_children_use_qualified_landlock_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        canary_v4, "_original_build_commands", lambda *_: _commands()
    )
    monkeypatch.setattr(
        formal_v4,
        "_all_gpu_device_paths",
        lambda: (Path("/dev/nvidia0"), Path("/dev/nvidia1")),
    )
    commands = canary_v4.build_commands(None, None)
    for lane in ("Agent", "HippoRAG"):
        assert Path("/proc/self/task") in commands[lane].write_paths
        assert commands[lane].device_paths == (
            Path("/dev/nvidia0"),
            Path("/dev/nvidia1"),
        )
        assert commands[lane].environment["VECLIB_MAXIMUM_THREADS"] == "1"
    assert commands["RAW"] == _commands()["RAW"]


def test_canary_v4_controller_injects_v4_preflight_and_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def controller(config_path: Path, **kwargs):
        captured["path"] = config_path
        captured.update(kwargs)
        return {"status": "synthetic"}

    monkeypatch.setattr(canary_v4, "_original_run_controller", controller)
    path = Path("/private/config.json")
    assert canary_v4.run_controller(path) == {"status": "synthetic"}
    assert captured["preflight"] is canary_v4._preflight
    assert captured["outer"] is canary_v4._outer


def test_v4_preserves_effect_contract_and_replaces_dependencies() -> None:
    assert formal_v4.STUDY_ID == formal_v1.STUDY_ID
    assert canary_v4.STUDY_ID == formal_v1.STUDY_ID
    assert formal_v4.CONFIG_SCHEMA == formal_v1.CONFIG_SCHEMA
    assert canary_v4.CONFIG_SCHEMA.endswith(
        "source_free_production_canary_v4_content_addressed_config_v1"
    )
    assert (
        formal_v4._base.PRODUCTION_DEPENDENCIES.outer_landlock
        is formal_v4._outer_landlock
    )
    assert (
        formal_v4._base.PRODUCTION_DEPENDENCIES.action_commands
        is formal_v4._action_commands
    )
