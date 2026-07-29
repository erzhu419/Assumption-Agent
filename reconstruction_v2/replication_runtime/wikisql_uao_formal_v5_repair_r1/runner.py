"""Append-only repaired-v5 runtime for the WikiSQL UAO P4 study.

This module deliberately reuses the already-qualified v5 shared-node
controller while isolating its module state.  It changes only the deployment
root, service binding, and public-source adapter.  The scientific study ID,
three-arm effect contract, scorer, action runtimes, resource policy, and
shared-node limits remain those of v5.

The failed ``formal_v5`` root is never opened or modified by this module.
Repaired execution is confined to the sibling ``formal_v5_repair_r1`` root.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping, Sequence

from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v1 as legacy_source_compiler,
)
from assumption_agent.benchmarks import (
    wikisql_uao_source_compiler_v5_repair as repair_source_compiler,
)


_V5_SOURCE = (
    Path(__file__).parents[1] / "wikisql_uao_formal_v5/runner.py"
)
_ISOLATED_V5_NAME = (
    "replication_runtime.wikisql_uao_formal_v5_repair_r1."
    "_isolated_formal_v5"
)
_V5_SPEC = importlib.util.spec_from_file_location(
    _ISOLATED_V5_NAME,
    _V5_SOURCE,
)
if _V5_SPEC is None or _V5_SPEC.loader is None:
    raise ImportError("frozen formal v5 controller cannot be isolated")
_v5 = importlib.util.module_from_spec(_V5_SPEC)
sys.modules[_ISOLATED_V5_NAME] = _v5
_V5_SPEC.loader.exec_module(_v5)


ORIGINAL_V5_ROOT = Path(
    "/home/erzhu419/wikisql_uao_p4_20260729/formal_v5"
)
FORMAL_ROOT = Path(
    "/home/erzhu419/wikisql_uao_p4_20260729/formal_v5_repair_r1"
)
UNIT_NAME = "wikisql-uao-p4-formal-v5-repair-r1.service"
INSTALLED_UNIT_PATH = (
    Path("/home/erzhu419/.config/systemd/user") / UNIT_NAME
)
SERVICE_RELATIVE_PATH = Path(
    "manifests/wikisql-uao-p4-formal-v5-repair-r1.service"
)
PYTHONHOME_ROOT = FORMAL_ROOT / "runtime_assets/python310_clean"
OFFICIAL_BASE_IMPORT_ROOT = Path(
    "/home/erzhu419/wikisql_uao_runtime_qualification/"
    "runtime_assets/official_base_import_clean"
)
MODULE = (
    "replication_runtime.wikisql_uao_formal_v5_repair_r1.runner"
)
ADMISSION_PATH = FORMAL_ROOT / "control/resource_admission.safe.json"
ADMISSION_FAILURE_PATH = (
    FORMAL_ROOT / "control/resource_admission_failure.safe.json"
)
DEFERRAL_ROOT = FORMAL_ROOT / "control/resource_deferrals"


# The formal controller needs one private-label field registry in addition to
# the public repaired compiler interface.  It is an unchanged pack-schema
# constant, not a fallback to the legacy compiler implementation.
source_compiler = SimpleNamespace(
    BABEL_LOCALE=repair_source_compiler.BABEL_LOCALE,
    CompilerConfig=repair_source_compiler.CompilerConfig,
    LABEL_VIEW_FIELDS=legacy_source_compiler.LABEL_VIEW_FIELDS,
    MAX_COLUMNS=repair_source_compiler.MAX_COLUMNS,
    MAX_HEADER_OR_CELL_CHARACTERS=(
        repair_source_compiler.MAX_HEADER_OR_CELL_CHARACTERS
    ),
    MAX_QUESTION_CHARACTERS=(
        repair_source_compiler.MAX_QUESTION_CHARACTERS
    ),
    PRODUCTION_ARCHIVE_GIT_BLOB_SHA1=(
        repair_source_compiler.PRODUCTION_ARCHIVE_GIT_BLOB_SHA1
    ),
    PRODUCTION_BABEL_VERSION=(
        repair_source_compiler.PRODUCTION_BABEL_VERSION
    ),
    PRODUCTION_PYTZ_VERSION=(
        repair_source_compiler.PRODUCTION_PYTZ_VERSION
    ),
    REQUIRED_MEMBERS=repair_source_compiler.REQUIRED_MEMBERS,
    VERSION=repair_source_compiler.VERSION,
    compile_archive=repair_source_compiler.compile_archive,
    write_compilation=repair_source_compiler.write_compilation,
)


_base = _v5._base
for module in (_v5, _base):
    module.FORMAL_ROOT = FORMAL_ROOT
    module.UNIT_NAME = UNIT_NAME
    module.INSTALLED_UNIT_PATH = INSTALLED_UNIT_PATH
    module.SERVICE_RELATIVE_PATH = SERVICE_RELATIVE_PATH

_v5.PYTHONHOME_ROOT = PYTHONHOME_ROOT
_v5.MODULE = MODULE
_v5.ADMISSION_PATH = ADMISSION_PATH
_v5.ADMISSION_FAILURE_PATH = ADMISSION_FAILURE_PATH
_v5.DEFERRAL_ROOT = DEFERRAL_ROOT
_base.source_compiler = source_compiler

_original_load_config = _v5.load_config


def load_config(path: Path):
    config = _original_load_config(path)
    if (
        config.tree("official_base_dependency_tree").path
        != OFFICIAL_BASE_IMPORT_ROOT
    ):
        raise _v5.WikiSQLUAOFormalError(
            "repaired source dependency tree path drifted"
        )
    return config


_v5.load_config = load_config
_base.load_config = load_config


def _compile_source_repair(
    config,
    paths,
) -> Mapping[str, str]:
    """Compile once with the fully public-qualified repaired adapter."""

    bundle = source_compiler.compile_archive(
        config.file("source_archive").path,
        expected_archive_sha256=config.file("source_archive").sha256,
        config=source_compiler.CompilerConfig.production(),
    )
    return source_compiler.write_compilation(paths.compiled, bundle)


_base.PRODUCTION_DEPENDENCIES = replace(
    _base.PRODUCTION_DEPENDENCIES,
    source_compile=_compile_source_repair,
)

_original_verify_source_outputs = _base._verify_source_outputs
_original_verify_service_profile = _base._verify_service_profile


def _verify_source_outputs_repair(config, paths, output_hashes):
    artifacts = _original_verify_source_outputs(
        config,
        paths,
        output_hashes,
    )
    receipt = _base._load_canonical_json(
        paths.compiler_receipt,
        mode=0o600,
        field="repaired source compiler safe receipt",
    )
    if (
        receipt.get("pytz_runtime_version")
        != source_compiler.PRODUCTION_PYTZ_VERSION
        or receipt.get("pytz_required_production_version")
        != source_compiler.PRODUCTION_PYTZ_VERSION
    ):
        raise WikiSQLUAOFormalError(
            "repaired source compiler pytz binding drifted"
        )
    return artifacts


def _verify_service_profile_repair(raw: bytes, config) -> None:
    required = (
        f":{OFFICIAL_BASE_IMPORT_ROOT} TEMP="
    ).encode("ascii")
    if raw.count(required) != 1:
        raise WikiSQLUAOFormalError(
            "repaired source dependency path drifted"
        )
    _original_verify_service_profile(raw, config)


_base._verify_source_outputs = _verify_source_outputs_repair
_base._verify_service_profile = _verify_service_profile_repair

CONFIG_SCHEMA = _v5.CONFIG_SCHEMA
STUDY_ID = _v5.STUDY_ID
RESOURCE_POLICY_SHA256 = _v5.RESOURCE_POLICY_SHA256
WikiSQLUAOFormalError = _v5.WikiSQLUAOFormalError
FormalPaths = _base.FormalPaths
canonical_json_bytes = _base.canonical_json_bytes
run_formal_production = _v5.run_formal_production
semantic_sha256 = _base.semantic_sha256
tree_identity = _base.tree_identity
_file_sha256 = _base._file_sha256
_self_hashed = _base._self_hashed
_write_once = _base._write_once


def _parser():
    return _v5._parser()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    terminal = run_formal_production(arguments.config)
    print(
        json.dumps(
            terminal,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    status = terminal.get("status")
    if status == "completed_protocol_valid":
        return 0
    if status == "DEFERRED_SHARED_RESOURCE":
        return _v5.resource_admission.EX_TEMPFAIL
    if status == "FAILED_INFRASTRUCTURE_PRE_ATTEMPT":
        return _v5.resource_admission.EX_SOFTWARE
    return 1


def __getattr__(name: str):
    return getattr(_v5, name)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
