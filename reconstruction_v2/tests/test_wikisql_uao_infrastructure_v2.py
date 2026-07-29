from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from replication_runtime.wikisql_uao_formal_v1 import runner as formal_v1
from replication_runtime.wikisql_uao_formal_v2 import runner as formal_v2
from replication_runtime.wikisql_uao_source_free_canary_v1 import (
    runner as canary_v1,
)
from replication_runtime.wikisql_uao_source_free_canary_v2 import (
    runner as canary_v2,
)


def test_v2_wrappers_do_not_mutate_frozen_v1_modules() -> None:
    assert formal_v1.FORMAL_ROOT.name == "formal_v1"
    assert formal_v1.UNIT_NAME == "wikisql-uao-p4-formal-v1.service"
    assert canary_v1.CANARY_ROOT.name == "source_free_canary_v1"
    assert canary_v1.UNIT_NAME == (
        "wikisql-uao-p4-source-free-canary-v1.service"
    )
    assert formal_v2.FORMAL_ROOT.name == "formal_v2"
    assert canary_v2.CANARY_ROOT.name == "source_free_canary_v2"


def test_v2_child_environments_bind_private_pythonhome() -> None:
    roots = {
        "code_tree": Path("/private/code"),
        "python_dependency_tree": Path("/private/dependencies"),
        "babel_dependency_tree": Path("/private/babel"),
    }

    class Config:
        def tree(self, name: str) -> SimpleNamespace:
            return SimpleNamespace(path=roots[name])

    formal_environment = formal_v2._lane_environment(
        Config(),
        Path("/private/lane"),
        cuda_visible_devices="1",
    )
    canary_environment = canary_v2._environment(
        Path("/private/lane"),
        "0",
        tuple(roots.values()),
    )
    assert formal_environment["PYTHONHOME"] == str(
        formal_v2.PYTHONHOME_ROOT
    )
    assert canary_environment["PYTHONHOME"] == str(
        canary_v2.PYTHONHOME_ROOT
    )
    assert formal_environment["PYTHONPATH"].split(":") == list(
        map(str, roots.values())
    )


def test_v2_services_are_minimal_and_bind_exact_private_pythonhome() -> None:
    manifest_root = Path(__file__).parents[1] / "manifests"
    cases = {
        "wikisql-uao-p4-formal-v2.service": (
            formal_v2.PYTHONHOME_ROOT,
            formal_v2.MODULE,
        ),
        "wikisql-uao-p4-source-free-canary-v2.service": (
            canary_v2.PYTHONHOME_ROOT,
            canary_v2.MODULE,
        ),
    }
    for filename, (pythonhome, module) in cases.items():
        service = (manifest_root / filename).read_text()
        assert "Type=oneshot" in service
        assert "Restart=no" in service
        assert "RestrictAddressFamilies=AF_UNIX" in service
        assert "IPAddressDeny=any" in service
        assert "NoNewPrivileges=yes" in service
        assert f"PYTHONHOME={pythonhome}" in service
        assert f"-m {module}" in service
        assert all(
            prefix not in service
            for prefix in formal_v1._FORBIDDEN_SERVICE_PREFIXES
        )


def test_formal_v2_service_profile_accepts_only_the_v2_module_and_layout() -> None:
    raw = (
        Path(__file__).parents[1]
        / "manifests/wikisql-uao-p4-formal-v2.service"
    ).read_bytes()
    trees = {
        "babel_dependency_tree": (
            formal_v2.FORMAL_ROOT / "runtime_assets/babel_2_10_3_clean"
        ),
        "code_tree": formal_v2.FORMAL_ROOT / "reconstruction_v2",
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

    formal_v2._verify_service_profile(raw, Config())
    with_module_drift = raw.replace(
        formal_v2.MODULE.encode("ascii"),
        b"replication_runtime.wikisql_uao_formal_v1.runner",
    )
    try:
        formal_v2._verify_service_profile(with_module_drift, Config())
    except formal_v2.WikiSQLUAOFormalError:
        pass
    else:
        raise AssertionError("v2 service accepted the retired v1 module")


def test_v2_config_schemas_preserve_the_same_efficacy_study() -> None:
    assert formal_v2.STUDY_ID == formal_v1.STUDY_ID
    assert canary_v2.STUDY_ID == formal_v1.STUDY_ID
    assert formal_v2.CONFIG_SCHEMA == formal_v1.CONFIG_SCHEMA
    assert canary_v2.CONFIG_SCHEMA.endswith(
        "source_free_production_canary_v2_content_addressed_config_v1"
    )


def test_canary_v2_production_controller_explicitly_uses_v2_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def controller(config_path: Path, **kwargs):
        captured["config_path"] = config_path
        captured.update(kwargs)
        return {"status": "synthetic"}

    monkeypatch.setattr(canary_v2, "_original_run_controller", controller)
    path = Path("/private/canary_config.json")
    result = canary_v2.run_controller(path)
    assert result == {"status": "synthetic"}
    assert captured["config_path"] == path
    assert captured["preflight"] is canary_v2._preflight
