"""Deterministically prepare the one-shot WikiSQL UAO formal v5 config.

This deployment builder consumes the already-passed stable qualification
config as its runtime-asset template and a local acquisition custody receipt
as the opaque official WikiSQL 1.1 source commitment.  On 311linux it checks
only source metadata: it does not read a source payload byte.  It also never
opens an archive member, selects an item, creates a formal secret, launches an
action, or computes a score.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence

from replication_runtime.wikisql_uao_formal_v5 import runner as formal
from replication_runtime.wikisql_uao_formal_v5 import source_custody
from replication_runtime.wikisql_uao_runtime_qualification import (
    contract as qualification_contract,
)


EXPECTED_DESIGN_SELF_SHA256 = (
    "c700dad09e8c0b26c5ece44b1520925111fb91004fac395aef214610aa808db4"
)
PREPARE_SCHEMA = "wikisql_uao_formal_v5_deployment_prepare_v1"


class WikiSQLUAOFormalV5PrepareError(RuntimeError):
    """The content-addressed formal deployment could not be prepared."""


def _direct_file(path: Path, field: str) -> os.stat_result:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise WikiSQLUAOFormalV5PrepareError(
            f"{field} is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or resolved != path
    ):
        raise WikiSQLUAOFormalV5PrepareError(
            f"{field} is not a direct canonical file"
        )
    return metadata


def _direct_directory(path: Path, field: str) -> os.stat_result:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise WikiSQLUAOFormalV5PrepareError(
            f"{field} is unavailable"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or resolved != path
    ):
        raise WikiSQLUAOFormalV5PrepareError(
            f"{field} is not a direct canonical directory"
        )
    return metadata


def _file_binding(path: Path, field: str) -> dict[str, object]:
    metadata = _direct_file(path, field)
    sha256, size = formal._file_sha256(path)
    return {
        "mode_octal": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "path": str(path),
        "sha256": sha256,
        "size_bytes": size,
    }


def _tree_binding(path: Path, field: str) -> dict[str, object]:
    _direct_directory(path, field)
    sha256, count, size = formal.tree_identity(path)
    return {
        "file_count": count,
        "path": str(path),
        "sha256": sha256,
        "total_bytes": size,
    }


def _opaque_source_metadata(path: Path) -> int:
    metadata = _direct_file(path, "opaque WikiSQL source archive")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise WikiSQLUAOFormalV5PrepareError(
            "opaque WikiSQL source archive mode drifted"
        )
    if metadata.st_size != source_custody.EXPECTED_SOURCE_BYTES:
        raise WikiSQLUAOFormalV5PrepareError(
            "opaque WikiSQL source archive size drifted"
        )
    return metadata.st_size


def _canonical_design(path: Path) -> Mapping[str, object]:
    _direct_file(path, "study design")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WikiSQLUAOFormalV5PrepareError(
            "study design is unavailable or malformed"
        ) from exc
    if (
        type(value) is not dict
        or formal.canonical_json_bytes(value) != raw
        or value.get("schema") != "wikisql_uao_p4_study_design_v1"
        or value.get("study_id") != formal.STUDY_ID
        or value.get("self_sha256") != EXPECTED_DESIGN_SELF_SHA256
    ):
        raise WikiSQLUAOFormalV5PrepareError(
            "study design identity drifted"
        )
    body = {
        key: child
        for key, child in value.items()
        if key != "self_sha256"
    }
    if formal.semantic_sha256(body) != EXPECTED_DESIGN_SELF_SHA256:
        raise WikiSQLUAOFormalV5PrepareError(
            "study design self hash drifted"
        )
    return value


def _root_layout() -> None:
    for path in (
        formal.FORMAL_ROOT,
        formal.FORMAL_ROOT / "control",
        formal.FORMAL_ROOT / "control/home",
        formal.FORMAL_ROOT / "control/tmp",
        formal.FORMAL_ROOT / "work",
        formal.FORMAL_ROOT / "source",
        formal.FORMAL_ROOT / "runtime_assets",
    ):
        metadata = _direct_directory(path, "formal deployment directory")
        if stat.S_IMODE(metadata.st_mode) != 0o700:
            raise WikiSQLUAOFormalV5PrepareError(
                "formal deployment directory mode drifted"
            )
    paths = formal.FormalPaths.for_root(formal.FORMAL_ROOT)
    forbidden = (
        paths.attempt,
        paths.live,
        paths.intent,
        paths.barrier,
        paths.terminal,
        paths.compiled,
        paths.agent_root,
        paths.raw_root,
        paths.hippo_root,
        paths.scorer_root,
        formal.ADMISSION_PATH,
        formal.ADMISSION_FAILURE_PATH,
        formal.DEFERRAL_ROOT,
    )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise WikiSQLUAOFormalV5PrepareError(
            "formal deployment already has execution evidence"
        )


def build_config(
    runtime_template_path: Path,
    source_archive: Path,
    source_custody_path: Path,
) -> dict[str, object]:
    _root_layout()
    if source_archive != formal.FORMAL_ROOT / formal._base.SOURCE_RELATIVE_PATH:
        raise WikiSQLUAOFormalV5PrepareError(
            "formal source archive path drifted"
        )
    runtime = qualification_contract.load_config(runtime_template_path)
    for name in sorted(runtime.files):
        if name != "service_unit":
            runtime.file(name).verify(f"runtime template file {name}")
    for name in sorted(runtime.trees):
        if name not in {
            "code_tree",
            "python_runtime_tree",
            "official_python_runtime_tree",
            "babel_dependency_tree",
        }:
            runtime.tree(name).verify(f"runtime template tree {name}")

    code_root = formal.FORMAL_ROOT / "reconstruction_v2"
    design_path = code_root / formal._base.DESIGN_RELATIVE_PATH
    service_path = code_root / formal.SERVICE_RELATIVE_PATH
    pythonhome = formal.PYTHONHOME_ROOT
    babel = formal.FORMAL_ROOT / "runtime_assets/babel_2_10_3_clean"
    _canonical_design(design_path)
    custody = source_custody.load_receipt(source_custody_path)
    source_size = _opaque_source_metadata(source_archive)

    files = {
        name: {
            "mode_octal": f"{runtime.file(name).mode:04o}",
            "path": str(runtime.file(name).path),
            "sha256": runtime.file(name).sha256,
            "size_bytes": runtime.file(name).size_bytes,
        }
        for name in (
            "nvidia_smi_executable",
            "official_python_executable",
            "python_executable",
            "systemctl_executable",
        )
    }
    files.update(
        {
            "design": _file_binding(design_path, "study design"),
            "service_unit": _file_binding(
                service_path, "formal v5 service"
            ),
            "source_archive": {
                "mode_octal": "0600",
                "path": str(source_archive),
                "sha256": custody["archive_sha256"],
                "size_bytes": source_size,
            },
        }
    )
    carried_trees = (
        "encoder_model_tree",
        "hippo_llm_model_tree",
        "official_base_dependency_tree",
        "official_hipporag_tree",
        "official_overlay_dependency_tree",
        "official_python_dependency_tree",
        "python_dependency_tree",
    )
    trees = {
        name: {
            "file_count": runtime.tree(name).file_count,
            "path": str(runtime.tree(name).path),
            "sha256": runtime.tree(name).sha256,
            "total_bytes": runtime.tree(name).total_bytes,
        }
        for name in carried_trees
    }
    babel_binding = _tree_binding(babel, "formal Babel tree")
    python_binding = _tree_binding(
        pythonhome, "formal private Python home"
    )
    babel_qualified = runtime.tree("babel_dependency_tree")
    python_qualified = runtime.tree("python_runtime_tree")
    official_python_qualified = runtime.tree(
        "official_python_runtime_tree"
    )
    babel_identity = (
        babel_binding["sha256"],
        babel_binding["file_count"],
        babel_binding["total_bytes"],
    )
    python_identity = (
        python_binding["sha256"],
        python_binding["file_count"],
        python_binding["total_bytes"],
    )
    if (
        babel_identity
        != (
            babel_qualified.sha256,
            babel_qualified.file_count,
            babel_qualified.total_bytes,
        )
        or python_identity
        != (
            python_qualified.sha256,
            python_qualified.file_count,
            python_qualified.total_bytes,
        )
        or python_identity
        != (
            official_python_qualified.sha256,
            official_python_qualified.file_count,
            official_python_qualified.total_bytes,
        )
    ):
        raise WikiSQLUAOFormalV5PrepareError(
            "formal copied qualification runtime tree drifted"
        )
    trees.update(
        {
            "babel_dependency_tree": babel_binding,
            "code_tree": _tree_binding(
                code_root, "formal code tree"
            ),
            "python_runtime_tree": python_binding,
        }
    )
    body: dict[str, object] = {
        "bindings": {"files": files, "trees": trees},
        "design_self_sha256": EXPECTED_DESIGN_SELF_SHA256,
        "encoder_model_semantic_sha256": (
            runtime.encoder_model_semantic_sha256
        ),
        "formal_root": str(formal.FORMAL_ROOT),
        "gpu_uuids": dict(runtime.gpu_uuids),
        "schema": formal.CONFIG_SCHEMA,
        "study_id": formal.STUDY_ID,
        "unit_name": formal.UNIT_NAME,
    }
    return formal._self_hashed(body)


def prepare(
    runtime_template_path: Path,
    source_archive: Path,
    source_custody_path: Path,
) -> Mapping[str, object]:
    custody_file_sha256, _ = formal._file_sha256(source_custody_path)
    custody = source_custody.load_receipt(source_custody_path)
    config = build_config(
        runtime_template_path,
        source_archive,
        source_custody_path,
    )
    config_path = formal.FORMAL_ROOT / "control/formal_config.json"
    config_file_sha256 = formal._write_once(
        config_path, config, mode=0o600
    )
    observed = formal.load_config(config_path)
    if observed.self_sha256 != config["self_sha256"]:
        raise WikiSQLUAOFormalV5PrepareError(
            "published formal config identity drifted"
        )
    receipt = formal._self_hashed(
        {
            "API_or_online_evaluation_count": 0,
            "config_file_sha256": config_file_sha256,
            "config_self_sha256": observed.self_sha256,
            "effect_study_attempt_count": 0,
            "formal_source_archive_git_blob_sha1": (
                source_custody.EXPECTED_SOURCE_GIT_BLOB_SHA1
            ),
            "formal_source_archive_sha256": observed.file(
                "source_archive"
            ).sha256,
            "formal_source_member_open_count": 0,
            "formal_source_payload_byte_read_count": 0,
            "resource_policy_sha256": formal.RESOURCE_POLICY_SHA256,
            "schema": PREPARE_SCHEMA,
            "source_custody_file_sha256": custody_file_sha256,
            "source_custody_self_sha256": custody["self_sha256"],
            "status": "FORMAL_V5_CONTENT_ADDRESSED_DEPLOYMENT_PREPARED",
            "study_id": formal.STUDY_ID,
        }
    )
    formal._write_once(
        formal.FORMAL_ROOT / "control/deployment_prepare.safe.json",
        receipt,
        mode=0o600,
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-template",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-archive",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--source-custody",
        required=True,
        type=Path,
    )
    arguments = parser.parse_args(argv)
    receipt = prepare(
        arguments.runtime_template,
        arguments.source_archive,
        arguments.source_custody,
    )
    print(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
