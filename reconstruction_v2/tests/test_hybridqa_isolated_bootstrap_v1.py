from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import hybridqa_isolated_bootstrap_v1 as bootstrap


class _ExecCaptured(RuntimeError):
    def __init__(self, arguments: tuple[object, ...]) -> None:
        super().__init__("synthetic exec capture")
        self.arguments = arguments


def test_target_capability_set_includes_exact_controller_entrypoint() -> None:
    assert bootstrap.TARGETS == frozenset(
        {
            "assumption_agent.benchmarks.hybridqa_direct_acquisition_v1",
            "assumption_agent.benchmarks.hybridqa_p6_e2_formal_controller_v1",
            "assumption_agent.benchmarks.hybridqa_query_anchored_formal_runner_v1",
        }
    )


@pytest.mark.parametrize(
    "target",
    (
        "assumption_agent.benchmarks.hybridqa_direct_acquisition_v1",
        "assumption_agent.benchmarks.hybridqa_p6_e2_formal_controller_v1",
        "assumption_agent.benchmarks.hybridqa_query_anchored_formal_runner_v1",
    ),
)
def test_reexec_builds_isolated_empty_pycache_command(
    target: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dependency_site = tmp_path / "dependency-site"
    dependency_site.mkdir(mode=0o700)
    monkeypatch.setattr(bootstrap, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "_dependency_site", lambda: dependency_site)
    monkeypatch.setattr(bootstrap.os, "getpid", lambda: 123)
    monkeypatch.setattr(bootstrap.os, "urandom", lambda _count: b"A" * 8)
    monkeypatch.delenv(bootstrap.TARGET_ENV, raising=False)

    def capture(*arguments: object) -> None:
        raise _ExecCaptured(arguments)

    monkeypatch.setattr(bootstrap.os, "execve", capture)
    with pytest.raises(_ExecCaptured) as raised:
        bootstrap.reexec_isolated(target, ("--project", str(tmp_path)))
    executable, command, environment = raised.value.arguments
    assert executable == bootstrap.sys.executable
    assert isinstance(command, list)
    assert command[1:5] == ["-I", "-B", "-X", command[4]]
    assert command[4].startswith(
        "pycache_prefix=/tmp/hybridqa_formal_empty_pycache_"
    )
    assert command[-2:] == ["--project", str(tmp_path)]
    assert isinstance(environment, dict)
    assert environment[bootstrap.TARGET_ENV] == target
    assert environment[bootstrap.PROJECT_ENV] == str(tmp_path)
    assert environment[bootstrap.PYCACHE_ENV] in command[4]
    assert environment[bootstrap.DEPENDENCY_SITE_ENV] == str(dependency_site)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert str(dependency_site) in command[6]
    assert "PYTHONPATH" not in environment


def test_bootstrap_rejects_ambient_project_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in (
        "assumption_agent/__init__.py",
        "assumption_agent/models.py",
        "assumption_agent/benchmarks/__init__.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n", encoding="ascii")
    evil = tmp_path / "evil.py"
    evil.write_text("VALUE = 1\n", encoding="ascii")
    monkeypatch.setattr(bootstrap, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "assert_isolated", lambda _target: None)
    monkeypatch.setenv(
        bootstrap.TARGET_ENV,
        "assumption_agent.benchmarks.hybridqa_direct_acquisition_v1",
    )
    monkeypatch.setitem(
        bootstrap.sys.modules,
        "synthetic_ambient_project_module",
        SimpleNamespace(__file__=str(evil)),
    )
    with pytest.raises(bootstrap.HybridQaIsolatedBootstrapError, match="preceded"):
        bootstrap.bootstrap_main()
