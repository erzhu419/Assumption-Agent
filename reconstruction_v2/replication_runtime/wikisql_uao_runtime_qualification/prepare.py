"""Prepare the stable, source-free runtime-qualification configuration.

The command deliberately accepts only a prior *runtime canary* configuration
as an asset-binding template.  It replaces the code and service bindings with
the currently deployed qualification implementation and carries no benchmark
source, label, qrel, score, evaluator, provider, or API binding forward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

from replication_runtime.wikisql_uao_formal_v1 import runner as formal
from replication_runtime.wikisql_uao_runtime_qualification import contract


RESOURCE_POLICY: dict[str, object] = {
    "gpu_roles": {
        "0": {
            "minimum_free_mib": 6144,
            "role": "HippoRAG",
        },
        "1": {
            "minimum_free_mib": 2048,
            "role": "Agent",
        },
    },
    "maximum_gpu_temperature_celsius": 82,
    "maximum_load1_per_cpu": 0.8,
    "maximum_median_cpu_busy_ratio": 0.70,
    "maximum_median_gpu_utilization_percent": 50,
    "minimum_host_mem_available_mib": 16384,
    "minimum_swap_free_mib": 0,
    "sample_count": 3,
    "sample_interval_seconds": 1.0,
    "schema": "wikisql_uao_shared_resource_policy_v1",
    "telemetry_timeout_seconds": 30,
}


class QualificationPrepareError(RuntimeError):
    """The stable qualification deployment could not be prepared."""


def _load_template(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationPrepareError(
            "runtime template is unavailable or malformed"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or type(value) is not dict
        or contract.canonical_json_bytes(value) != raw
    ):
        raise QualificationPrepareError(
            "runtime template metadata or canonical form drifted"
        )
    bindings = value.get("bindings")
    if (
        type(bindings) is not dict
        or set(bindings) != {"files", "trees"}
        or type(bindings["files"]) is not dict
        or set(bindings["files"]) != contract.REQUIRED_FILES
        or type(bindings["trees"]) is not dict
        or set(bindings["trees"]) != contract.REQUIRED_TREES
    ):
        raise QualificationPrepareError(
            "runtime template binding registry drifted"
        )
    return value


def _file_binding(path: Path) -> dict[str, object]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QualificationPrepareError(
            "deployed service file is unavailable"
        ) from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise QualificationPrepareError(
            "deployed service file is not direct"
        )
    sha256, size = formal._file_sha256(path)
    return {
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "path": str(path),
        "sha256": sha256,
        "size_bytes": size,
    }


def _tree_binding(path: Path) -> dict[str, object]:
    try:
        sha256, count, size = formal.tree_identity(path)
    except formal.WikiSQLUAOFormalError as exc:
        raise QualificationPrepareError(
            "deployed qualification tree cannot be addressed"
        ) from exc
    return {
        "file_count": count,
        "path": str(path),
        "sha256": sha256,
        "total_bytes": size,
    }


def build_config(template: Mapping[str, object]) -> dict[str, object]:
    bindings = template["bindings"]
    if type(bindings) is not dict:
        raise QualificationPrepareError("template bindings drifted")
    raw_files = bindings["files"]
    raw_trees = bindings["trees"]
    if type(raw_files) is not dict or type(raw_trees) is not dict:
        raise QualificationPrepareError("template bindings drifted")
    code_root = contract.QUALIFICATION_ROOT / "reconstruction_v2"
    service_path = code_root / contract.SERVICE_RELATIVE_PATH
    files = {
        name: dict(raw_files[name])
        for name in sorted(contract.REQUIRED_FILES - {"service_unit"})
    }
    files["service_unit"] = _file_binding(service_path)
    localized = {
        "babel_dependency_tree": contract.BABEL_ROOT,
        "official_base_dependency_tree": contract.OFFICIAL_BASE_ROOT,
        "official_hipporag_tree": contract.OFFICIAL_HIPPORAG_ROOT,
        "official_python_runtime_tree": contract.PYTHONHOME_ROOT,
        "python_runtime_tree": contract.PYTHONHOME_ROOT,
    }
    trees = {
        name: dict(raw_trees[name])
        for name in sorted(
            contract.REQUIRED_TREES
            - {"code_tree", *localized}
        )
    }
    trees["code_tree"] = _tree_binding(code_root)
    for name, path in localized.items():
        trees[name] = _tree_binding(path)
    gpu = template.get("gpu_uuids")
    semantic = template.get("encoder_model_semantic_sha256")
    body: dict[str, object] = {
        "bindings": {"files": files, "trees": trees},
        "capability_boundary": dict(contract.CAPABILITY_BOUNDARY),
        "encoder_model_semantic_sha256": semantic,
        "expected_babel_version": contract.EXPECTED_BABEL_VERSION,
        "gpu_uuids": gpu,
        "pythonpath_order": dict(contract.PYTHONPATH_ORDER),
        "qualification_id": contract.QUALIFICATION_ID,
        "qualification_root": str(contract.QUALIFICATION_ROOT),
        "resource_policy": RESOURCE_POLICY,
        "schema": contract.CONFIG_SCHEMA,
        "unit_name": contract.UNIT_NAME,
    }
    return contract.addressed(body)


def _write_replace(path: Path, value: Mapping[str, object]) -> str:
    raw = contract.canonical_json_bytes(value)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.preparing"
    )
    if temporary.exists() or temporary.is_symlink():
        raise QualificationPrepareError(
            "temporary config path is unexpectedly occupied"
        )
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if path.is_symlink():
            raise QualificationPrepareError(
                "config destination may not be a symlink"
            )
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except (OSError, QualificationPrepareError) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        if isinstance(exc, QualificationPrepareError):
            raise
        raise QualificationPrepareError(
            "qualification config cannot be published"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def prepare(template_path: Path) -> dict[str, object]:
    template = _load_template(template_path)
    config = build_config(template)
    config_path = (
        contract.QUALIFICATION_ROOT / contract.CONFIG_RELATIVE_PATH
    )
    file_sha256 = _write_replace(config_path, config)
    observed = contract.load_config(config_path)
    if observed.self_sha256 != config["self_sha256"]:
        raise QualificationPrepareError(
            "published qualification config identity drifted"
        )
    return {
        "config_file_sha256": file_sha256,
        "config_self_sha256": observed.self_sha256,
        "effect_study_attempt_count": 0,
        "formal_source_paths_bound": 0,
        "qualification_root": str(contract.QUALIFICATION_ROOT),
        "status": "stable_non_scoring_runtime_config_prepared",
        "unit_name": contract.UNIT_NAME,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-template",
        required=True,
        type=Path,
    )
    arguments = parser.parse_args(argv)
    receipt = prepare(arguments.runtime_template)
    print(
        json.dumps(
            receipt,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
