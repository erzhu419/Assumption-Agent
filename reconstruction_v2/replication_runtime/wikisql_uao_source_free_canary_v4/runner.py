"""WikiSQL UAO source-free canary v4 with qualified CUDA Landlock."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from replication_runtime.wikisql_uao_formal_v4 import runner as formal_v4


_SOURCE = (
    Path(__file__).parents[1]
    / "wikisql_uao_source_free_canary_v1/runner.py"
)
_ISOLATED_NAME = (
    "replication_runtime.wikisql_uao_source_free_canary_v4."
    "_isolated_canary_v1"
)
_SPEC = importlib.util.spec_from_file_location(_ISOLATED_NAME, _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("frozen source-free canary v1 cannot be isolated")
_base = importlib.util.module_from_spec(_SPEC)
sys.modules[_ISOLATED_NAME] = _base
_SPEC.loader.exec_module(_base)

VERSION = "wikisql_uao_source_free_production_canary_v4"
CONFIG_SCHEMA = f"{VERSION}_content_addressed_config_v1"
CANARY_ROOT = Path(
    "/home/erzhu419/wikisql_uao_p4_20260729/source_free_canary_v4"
)
UNIT_NAME = "wikisql-uao-p4-source-free-canary-v4.service"
INSTALLED_UNIT_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / UNIT_NAME
)
SERVICE_RELATIVE_PATH = Path(
    "manifests/wikisql-uao-p4-source-free-canary-v4.service"
)
MODULE = "replication_runtime.wikisql_uao_source_free_canary_v4.runner"
PYTHONHOME_ROOT = CANARY_ROOT / "runtime_assets/python310_clean"

_base.formal = formal_v4._base
_base.VERSION = VERSION
_base.CONFIG_SCHEMA = CONFIG_SCHEMA
_base.CANARY_ROOT = CANARY_ROOT
_base.FORMAL_SOURCE_ROOT = formal_v4.FORMAL_ROOT / "source"
_base.UNIT_NAME = UNIT_NAME
_base.INSTALLED_UNIT_PATH = INSTALLED_UNIT_PATH
_base.SERVICE_RELATIVE_PATH = SERVICE_RELATIVE_PATH
_base.MODULE = MODULE
_base.LANE_SCHEMAS = {
    lane: f"{VERSION}_{lane.casefold()}_lane_safe_v1"
    for lane in ("Agent", "RAW", "HippoRAG")
}

_original_environment = _base._environment
_original_load_config = _base.load_config
_original_preflight = _base._preflight
_original_run_controller = _base.run_controller
_original_build_commands = _base.build_commands


def load_config(path: Path):
    config = _original_load_config(path)
    if (
        config.tree("python_runtime_tree").path != PYTHONHOME_ROOT
        or config.tree("official_python_runtime_tree").path
        != PYTHONHOME_ROOT
    ):
        raise _base.WikiSQLUAOCanaryError(
            "private Python home binding drifted"
        )
    return config


def _environment(
    root: Path,
    cuda: str,
    module_roots: Sequence[Path],
) -> dict[str, str]:
    environment = _original_environment(root, cuda, module_roots)
    environment["PYTHONHOME"] = str(PYTHONHOME_ROOT)
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    return environment


def _preflight(config):
    service = config.file("service_unit").path.read_bytes()
    if f"PYTHONHOME={PYTHONHOME_ROOT}".encode("ascii") not in service:
        raise _base.WikiSQLUAOCanaryError(
            "private Python home service binding drifted"
        )
    return _original_preflight(config)


def _outer(config, paths) -> None:
    devices = formal_v4._all_gpu_device_paths()
    if not {"nvidia0", "nvidia1"} <= {path.name for path in devices}:
        raise _base.WikiSQLUAOCanaryError(
            "exact GPU device nodes are unavailable"
        )
    formal_v4._base.apply_landlock(
        read_paths=(
            *formal_v4._base._existing_system_read_paths(),
            *(binding.path for binding in config.files.values()),
            *(binding.path for binding in config.trees.values()),
        ),
        write_paths=(
            paths.root,
            Path("/tmp"),
            Path("/dev/null"),
            Path("/proc"),
        ),
        device_paths=devices,
    )


def build_commands(config, paths):
    commands = dict(_original_build_commands(config, paths))
    devices = formal_v4._all_gpu_device_paths()
    for lane in ("Agent", "HippoRAG"):
        command = commands[lane]
        environment = dict(command.environment)
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        commands[lane] = replace(
            command,
            environment=environment,
            write_paths=(
                *command.write_paths,
                Path("/proc/self/task"),
            ),
            device_paths=devices,
        )
    return commands


_base.load_config = load_config
_base._environment = _environment
_base._preflight = _preflight
_base.build_commands = build_commands

WikiSQLUAOCanaryError = _base.WikiSQLUAOCanaryError
STUDY_ID = _base.STUDY_ID


def run_controller(config_path: Path, **kwargs) -> Mapping[str, object]:
    kwargs.setdefault("preflight", _preflight)
    kwargs.setdefault("outer", _outer)
    return _original_run_controller(config_path, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _base._parser().parse_args(argv)
    if arguments.mode == "controller":
        result = run_controller(arguments.config)
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))
        return int(
            result.get("status")
            != "PASS_WIKISQL_UAO_SOURCE_FREE_PRODUCTION_CANARY"
        )
    {
        "agent": _base._agent,
        "raw": _base._raw,
        "hippo": _base._hippo,
    }[arguments.mode](arguments)
    return 0


def __getattr__(name: str):
    return getattr(_base, name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
