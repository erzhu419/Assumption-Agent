"""Fresh-interpreter bootstrap for the two irreversible HybridQA entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import stat
import sys
from typing import NoReturn, Sequence


VERSION = "hybridqa_isolated_bootstrap_v1"
TARGETS = frozenset(
    {
        "assumption_agent.benchmarks.hybridqa_direct_acquisition_v1",
        "assumption_agent.benchmarks.hybridqa_query_anchored_formal_runner_v1",
    }
)
TARGET_ENV = "HYBRIDQA_FORMAL_ISOLATED_TARGET_V1"
PROJECT_ENV = "HYBRIDQA_FORMAL_ISOLATED_PROJECT_V1"
PYCACHE_ENV = "HYBRIDQA_FORMAL_ISOLATED_PYCACHE_V1"


class HybridQaIsolatedBootstrapError(RuntimeError):
    """The formal command did not start in the required clean interpreter."""


def _project_root() -> Path:
    root = Path(__file__).resolve(strict=True).parents[2]
    if root.is_symlink() or not root.is_dir():
        raise HybridQaIsolatedBootstrapError("bootstrap project root is unsafe")
    return root


def assert_isolated(target: str) -> None:
    """Verify the isolated child flags and exact bootstrap capability."""

    project = _project_root()
    prefix = os.environ.get(PYCACHE_ENV)
    if (
        target not in TARGETS
        or os.environ.get(TARGET_ENV) != target
        or os.environ.get(PROJECT_ENV) != str(project)
        or sys.flags.isolated != 1
        or not sys.dont_write_bytecode
        or not isinstance(prefix, str)
        or not prefix.startswith("/tmp/hybridqa_formal_empty_pycache_")
        or sys.pycache_prefix != prefix
        or not sys.path
        or Path(sys.path[-1]).resolve(strict=True) != project
    ):
        raise HybridQaIsolatedBootstrapError("formal interpreter isolation drifted")
    path = Path(prefix)
    if os.path.lexists(path):
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or any(path.iterdir())
        ):
            raise HybridQaIsolatedBootstrapError(
                "formal private pycache prefix is not empty"
            )


def reexec_isolated(target: str, argv: Sequence[str]) -> NoReturn | None:
    """Replace an ambient process with an isolated, pycache-empty process."""

    if target not in TARGETS or any(not isinstance(value, str) for value in argv):
        raise HybridQaIsolatedBootstrapError("formal bootstrap target drifted")
    if os.environ.get(TARGET_ENV) == target:
        assert_isolated(target)
        return None
    project = _project_root()
    prefix = Path(
        f"/tmp/hybridqa_formal_empty_pycache_{os.getpid()}_{os.urandom(8).hex()}"
    )
    if os.path.lexists(prefix):
        raise HybridQaIsolatedBootstrapError("private pycache prefix already exists")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"}
    }
    environment.update(
        {
            TARGET_ENV: target,
            PROJECT_ENV: str(project),
            PYCACHE_ENV: str(prefix),
        }
    )
    code = (
        "import sys;"
        f"sys.path.append({str(project)!r});"
        "from assumption_agent.benchmarks.hybridqa_isolated_bootstrap_v1 "
        "import bootstrap_main;"
        "bootstrap_main()"
    )
    os.execve(
        sys.executable,
        [
            sys.executable,
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={prefix}",
            "-c",
            code,
            *tuple(argv),
        ],
        environment,
    )
    raise AssertionError("os.execve returned")


def bootstrap_main() -> None:
    """Reject ambient project modules, then execute the selected module."""

    project = _project_root()
    target = os.environ.get(TARGET_ENV)
    if not isinstance(target, str):
        raise HybridQaIsolatedBootstrapError("formal bootstrap target is absent")
    assert_isolated(target)
    allowed = {
        (project / "assumption_agent/__init__.py").resolve(strict=True),
        (project / "assumption_agent/models.py").resolve(strict=True),
        (project / "assumption_agent/benchmarks/__init__.py").resolve(strict=True),
        Path(__file__).resolve(strict=True),
    }
    for module_name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or origin.startswith("<"):
            continue
        try:
            path = Path(origin).resolve(strict=True)
            path.relative_to(project)
        except (OSError, RuntimeError, ValueError):
            continue
        if path not in allowed:
            raise HybridQaIsolatedBootstrapError(
                f"ambient project module preceded bootstrap: {module_name}"
            )
    runpy.run_module(target, run_name="__main__", alter_sys=True)


__all__ = [
    "HybridQaIsolatedBootstrapError",
    "PROJECT_ENV",
    "PYCACHE_ENV",
    "TARGET_ENV",
    "TARGETS",
    "VERSION",
    "assert_isolated",
    "bootstrap_main",
    "reexec_isolated",
]
