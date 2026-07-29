"""WikiSQL UAO formal v2 with a private, symlink-free Python home.

The efficacy design, source, cohort selector, candidate, models, arms, metric,
and stopping rule are byte-identical to formal v1.  This wrapper loads an
isolated copy of the frozen v1 controller and changes only the deployment
root, unit identity, service path, and Python runtime custody.  It exists
because the shared p17 stdlib contains dangling packaging symlinks and cannot
be accepted by the strict content-addressed tree verifier without mutating
historical runtime evidence.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Mapping, Sequence


_SOURCE = (
    Path(__file__).parents[1] / "wikisql_uao_formal_v1/runner.py"
)
_ISOLATED_NAME = (
    "replication_runtime.wikisql_uao_formal_v2._isolated_formal_v1"
)
_SPEC = importlib.util.spec_from_file_location(_ISOLATED_NAME, _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("frozen formal v1 controller cannot be isolated")
_base = importlib.util.module_from_spec(_SPEC)
sys.modules[_ISOLATED_NAME] = _base
_SPEC.loader.exec_module(_base)

FORMAL_ROOT = Path("/home/erzhu419/wikisql_uao_p4_20260729/formal_v2")
UNIT_NAME = "wikisql-uao-p4-formal-v2.service"
INSTALLED_UNIT_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / UNIT_NAME
)
SERVICE_RELATIVE_PATH = Path(
    "manifests/wikisql-uao-p4-formal-v2.service"
)
PYTHONHOME_ROOT = FORMAL_ROOT / "runtime_assets/python310_clean"
MODULE = "replication_runtime.wikisql_uao_formal_v2.runner"

_base.FORMAL_ROOT = FORMAL_ROOT
_base.UNIT_NAME = UNIT_NAME
_base.INSTALLED_UNIT_PATH = INSTALLED_UNIT_PATH
_base.SERVICE_RELATIVE_PATH = SERVICE_RELATIVE_PATH

_original_load_config = _base.load_config
_original_lane_environment = _base._lane_environment
_original_verify_service_profile = _base._verify_service_profile


def load_config(path: Path):
    config = _original_load_config(path)
    if config.tree("python_runtime_tree").path != PYTHONHOME_ROOT:
        raise _base.WikiSQLUAOFormalError(
            "private Python home binding drifted"
        )
    return config


def _lane_environment(
    config,
    root: Path,
    *,
    cuda_visible_devices: str,
) -> dict[str, str]:
    environment = _original_lane_environment(
        config,
        root,
        cuda_visible_devices=cuda_visible_devices,
    )
    environment["PYTHONHOME"] = str(PYTHONHOME_ROOT)
    return environment


def _verify_service_profile(raw: bytes, config) -> None:
    required = f"PYTHONHOME={PYTHONHOME_ROOT}".encode("ascii")
    module = MODULE.encode("ascii")
    retired_module = (
        b"replication_runtime.wikisql_uao_formal_v1.runner"
    )
    if (
        required not in raw
        or raw.count(module) != 1
        or retired_module in raw
    ):
        raise _base.WikiSQLUAOFormalError(
            "private Python home or v2 service module binding drifted"
        )
    rewritten = raw.replace(module, retired_module)
    _original_verify_service_profile(rewritten, config)


def _service_probe(config):
    return _base._systemctl_attestation(
        config,
        unit_name=UNIT_NAME,
        installed_unit_path=INSTALLED_UNIT_PATH,
    )


_base.load_config = load_config
_base._lane_environment = _lane_environment
_base._verify_service_profile = _verify_service_profile
_base.PRODUCTION_DEPENDENCIES = replace(
    _base.PRODUCTION_DEPENDENCIES,
    service_probe=_service_probe,
)

CONFIG_SCHEMA = _base.CONFIG_SCHEMA
STUDY_ID = _base.STUDY_ID
WikiSQLUAOFormalError = _base.WikiSQLUAOFormalError


def run_formal_production(config_path: Path) -> Mapping[str, object]:
    return _base.run_formal_production(config_path)


def main(argv: Sequence[str] | None = None) -> int:
    return _base.main(argv)


def __getattr__(name: str):
    return getattr(_base, name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
