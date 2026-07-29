from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v1 as legacy_source_compiler,
)
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v5_repair as repair_source_compiler,
)
from replication_runtime.wikisql_uao_formal_v5 import (
    runner as original_v5,
)
from replication_runtime.wikisql_uao_formal_v5_repair_r1 import prepare
from replication_runtime.wikisql_uao_formal_v5_repair_r1 import (
    runner as subject,
)


class _PathBinding:
    def __init__(self, path: Path) -> None:
        self.path = path


class _ServiceConfig:
    def file(self, name: str) -> _PathBinding:
        if name != "python_executable":
            raise AssertionError(f"unexpected file binding: {name}")
        return _PathBinding(
            Path(
                "/home/erzhu419/p19_runtime_assets_20260723/"
                "typed_venv/bin/python"
            )
        )

    def tree(self, name: str) -> _PathBinding:
        paths = {
            "code_tree": subject.FORMAL_ROOT / "reconstruction_v2",
            "python_dependency_tree": Path(
                "/home/erzhu419/p19_runtime_assets_20260723/"
                "typed_venv/lib/python3.10/site-packages"
            ),
            "babel_dependency_tree": (
                subject.FORMAL_ROOT
                / "runtime_assets/babel_2_10_3_clean"
            ),
        }
        return _PathBinding(paths[name])


def _service_bytes() -> bytes:
    return (
        Path(__file__).parents[1]
        / "manifests/wikisql-uao-p4-formal-v5-repair-r1.service"
    ).read_bytes()


def test_repair_is_append_only_and_does_not_mutate_original_v5() -> None:
    original_root = Path(
        "/home/erzhu419/wikisql_uao_p4_20260729/formal_v5"
    )
    repaired_root = Path(
        "/home/erzhu419/wikisql_uao_p4_20260729/formal_v5_repair_r1"
    )

    assert subject.ORIGINAL_V5_ROOT == original_root
    assert subject.FORMAL_ROOT == repaired_root
    assert subject._base.FORMAL_ROOT == repaired_root
    assert subject._v5.FORMAL_ROOT == repaired_root
    assert subject.ADMISSION_PATH.is_relative_to(repaired_root)
    assert subject.ADMISSION_FAILURE_PATH.is_relative_to(repaired_root)
    assert subject.DEFERRAL_ROOT.is_relative_to(repaired_root)

    assert original_v5.FORMAL_ROOT == original_root
    assert original_v5._base.FORMAL_ROOT == original_root
    assert original_v5.UNIT_NAME == "wikisql-uao-p4-formal-v5.service"
    assert (
        original_v5._base.source_compiler
        is legacy_source_compiler
    )


def test_repair_preserves_study_contract_and_binds_repaired_compiler() -> None:
    assert subject.STUDY_ID == original_v5.STUDY_ID
    assert subject.CONFIG_SCHEMA == original_v5.CONFIG_SCHEMA
    assert subject._base.ACTION_MODULE == original_v5._base.ACTION_MODULE
    assert subject._base.OFFICIAL_MODULE == original_v5._base.OFFICIAL_MODULE
    assert subject._base.SCORER_MODULE == original_v5._base.SCORER_MODULE
    assert subject._base.FORMAL_ITEM_COUNT == 72

    interface = subject._base.source_compiler
    assert interface.compile_archive is repair_source_compiler.compile_archive
    assert (
        interface.write_compilation
        is repair_source_compiler.write_compilation
    )
    assert interface.REQUIRED_MEMBERS == repair_source_compiler.REQUIRED_MEMBERS
    assert interface.VERSION == repair_source_compiler.VERSION
    assert interface.LABEL_VIEW_FIELDS == (
        legacy_source_compiler.LABEL_VIEW_FIELDS
    )
    assert (
        subject._base.PRODUCTION_DEPENDENCIES.source_compile
        is subject._compile_source_repair
    )
    assert (
        subject._base._verify_source_outputs.__globals__["source_compiler"]
        is interface
    )
    assert (
        subject._base._verify_source_outputs
        is subject._verify_source_outputs_repair
    )
    assert subject._base.load_config is subject.load_config
    assert subject._v5.load_config is subject.load_config


@pytest.mark.parametrize("matches", (True, False))
def test_load_config_binds_pytz_import_tree(
    matches: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (
        subject.OFFICIAL_BASE_IMPORT_ROOT
        if matches
        else Path("/unbound/pytz-parent")
    )
    config = SimpleNamespace(
        tree=lambda name: (
            SimpleNamespace(path=path)
            if name == "official_base_dependency_tree"
            else (_ for _ in ()).throw(AssertionError(name))
        )
    )
    monkeypatch.setattr(
        subject,
        "_original_load_config",
        lambda _path: config,
    )

    if matches:
        assert subject.load_config(Path("/config.json")) is config
    else:
        with pytest.raises(
            subject.WikiSQLUAOFormalError,
            match="dependency tree path drifted",
        ):
            subject.load_config(Path("/config.json"))


def test_repaired_source_compile_is_the_only_bound_compiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "data.tar.bz2"
    compiled = tmp_path / "compiled_source"
    archive.write_bytes(b"public archive")
    calls: list[tuple[object, ...]] = []
    bundle = object()
    output = {"safe/source_compiler_receipt.json": "a" * 64}

    class _CompilerConfig:
        @staticmethod
        def production() -> str:
            calls.append(("config",))
            return "production-repair"

    def compile_archive(
        path: Path,
        *,
        expected_archive_sha256: str,
        config: object,
    ) -> object:
        calls.append(
            ("compile", path, expected_archive_sha256, config)
        )
        return bundle

    def write_compilation(path: Path, observed: object):
        calls.append(("write", path, observed))
        return output

    monkeypatch.setattr(
        subject.source_compiler,
        "CompilerConfig",
        _CompilerConfig,
    )
    monkeypatch.setattr(
        subject.source_compiler,
        "compile_archive",
        compile_archive,
    )
    monkeypatch.setattr(
        subject.source_compiler,
        "write_compilation",
        write_compilation,
    )
    config = SimpleNamespace(
        file=lambda name: SimpleNamespace(
            path=archive,
            sha256="b" * 64,
        )
    )
    paths = SimpleNamespace(compiled=compiled)

    observed = subject._compile_source_repair(config, paths)

    assert observed is output
    assert calls == [
        ("config",),
        ("compile", archive, "b" * 64, "production-repair"),
        ("write", compiled, bundle),
    ]


def test_service_binds_only_repaired_root_module_and_shared_caps() -> None:
    raw = _service_bytes()
    text = raw.decode("utf-8")

    assert (
        "-m replication_runtime."
        "wikisql_uao_formal_v5_repair_r1.runner"
    ) in text
    assert str(subject.FORMAL_ROOT / "control/formal_config.json") in text
    assert (
        str(subject.ORIGINAL_V5_ROOT / "control/formal_config.json")
        not in text
    )
    assert "-m replication_runtime.wikisql_uao_formal_v5.runner" not in text
    assert (
        str(subject.OFFICIAL_BASE_IMPORT_ROOT)
        in text
    )
    for line in (
        "CPUQuota=400%",
        "CPUWeight=25",
        "IOWeight=25",
        "IOSchedulingClass=idle",
        "Nice=10",
        "MemoryHigh=25769803776",
        "MemoryMax=34359738368",
        "MemorySwapMax=0",
        "TasksMax=96",
        "SuccessExitStatus=75",
        "TimeoutStartSec=6h",
    ):
        assert line in text
    subject._verify_service_profile_repair(raw, _ServiceConfig())


def test_service_rejects_missing_pytz_dependency_path() -> None:
    raw = _service_bytes()
    with pytest.raises(
        subject.WikiSQLUAOFormalError,
        match="source dependency path drifted",
    ):
        subject._verify_service_profile_repair(
            raw.replace(
                (
                    b":/home/erzhu419/wikisql_uao_runtime_qualification/"
                    b"runtime_assets/official_base_import_clean"
                ),
                b"",
                1,
            ),
            _ServiceConfig(),
        )


def test_prepare_is_rebound_to_repaired_root_and_existing_custody() -> None:
    assert prepare._base_prepare.formal is subject
    assert (
        prepare._base_prepare.PREPARE_SCHEMA
        == "wikisql_uao_formal_v5_repair_r1_deployment_prepare_v1"
    )
    assert (
        prepare._base_prepare.source_custody.__name__
        == "replication_runtime.wikisql_uao_formal_v5.source_custody"
    )
    assert (
        prepare.build_config.__globals__["formal"].FORMAL_ROOT
        == subject.FORMAL_ROOT
    )
    assert (
        prepare.build_config.__globals__["formal"].SERVICE_RELATIVE_PATH
        == Path(
            "manifests/"
            "wikisql-uao-p4-formal-v5-repair-r1.service"
        )
    )
