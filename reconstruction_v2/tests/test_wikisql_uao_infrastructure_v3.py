from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from replication_runtime.wikisql_uao_formal_v1 import runner as formal_v1
from replication_runtime.wikisql_uao_formal_v2 import runner as formal_v2
from replication_runtime.wikisql_uao_formal_v3 import runner as formal_v3
from replication_runtime.wikisql_uao_source_free_canary_v2 import (
    runner as canary_v2,
)
from replication_runtime.wikisql_uao_source_free_canary_v3 import (
    runner as canary_v3,
)


def test_v3_isolated_wrappers_preserve_retired_v1_and_v2_modules() -> None:
    assert formal_v1.FORMAL_ROOT.name == "formal_v1"
    assert formal_v2.FORMAL_ROOT.name == "formal_v2"
    assert canary_v2.CANARY_ROOT.name == "source_free_canary_v2"
    assert formal_v3.FORMAL_ROOT.name == "formal_v3"
    assert canary_v3.CANARY_ROOT.name == "source_free_canary_v3"


def test_v3_services_bind_private_pythonhome_and_exact_v3_modules() -> None:
    manifests = Path(__file__).parents[1] / "manifests"
    cases = {
        "wikisql-uao-p4-formal-v3.service": (
            formal_v3.PYTHONHOME_ROOT,
            formal_v3.MODULE,
        ),
        "wikisql-uao-p4-source-free-canary-v3.service": (
            canary_v3.PYTHONHOME_ROOT,
            canary_v3.MODULE,
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


def test_formal_v3_service_profile_rejects_retired_module() -> None:
    raw = (
        Path(__file__).parents[1]
        / "manifests/wikisql-uao-p4-formal-v3.service"
    ).read_bytes()
    trees = {
        "babel_dependency_tree": (
            formal_v3.FORMAL_ROOT / "runtime_assets/babel_2_10_3_clean"
        ),
        "code_tree": formal_v3.FORMAL_ROOT / "reconstruction_v2",
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

    formal_v3._verify_service_profile(raw, Config())
    drifted = raw.replace(
        formal_v3.MODULE.encode("ascii"),
        formal_v2.MODULE.encode("ascii"),
    )
    with pytest.raises(formal_v3.WikiSQLUAOFormalError):
        formal_v3._verify_service_profile(drifted, Config())


def test_v3_outer_layers_allow_parent_devnull_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []

    def apply(**kwargs) -> None:
        captured.append(kwargs)

    monkeypatch.setattr(formal_v3, "apply_landlock", apply)
    monkeypatch.setattr(
        formal_v3,
        "_gpu_device_paths",
        lambda index: (Path(f"/dev/nvidia{index}"),),
    )
    config = SimpleNamespace(files={}, trees={})
    paths = SimpleNamespace(root=Path("/private/canary"))
    canary_v3._outer(config, paths)
    assert Path("/dev/null") in captured[-1]["write_paths"]

    captured.clear()
    monkeypatch.setattr(formal_v3._base, "apply_landlock", apply)
    formal_v3._outer_landlock(
        config,
        SimpleNamespace(root=Path("/private/formal")),
    )
    assert Path("/dev/null") in captured[-1]["write_paths"]
    assert (
        formal_v3._base.PRODUCTION_DEPENDENCIES.outer_landlock
        is formal_v3._outer_landlock
    )


def test_canary_v3_controller_injects_v3_preflight_and_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def controller(config_path: Path, **kwargs):
        captured["path"] = config_path
        captured.update(kwargs)
        return {"status": "synthetic"}

    monkeypatch.setattr(canary_v3, "_original_run_controller", controller)
    path = Path("/private/config.json")
    assert canary_v3.run_controller(path) == {"status": "synthetic"}
    assert captured["preflight"] is canary_v3._preflight
    assert captured["outer"] is canary_v3._outer


def test_v3_preserves_the_same_effect_study_and_config_contract() -> None:
    assert formal_v3.STUDY_ID == formal_v1.STUDY_ID
    assert canary_v3.STUDY_ID == formal_v1.STUDY_ID
    assert formal_v3.CONFIG_SCHEMA == formal_v1.CONFIG_SCHEMA
    assert canary_v3.CONFIG_SCHEMA.endswith(
        "source_free_production_canary_v3_content_addressed_config_v1"
    )
