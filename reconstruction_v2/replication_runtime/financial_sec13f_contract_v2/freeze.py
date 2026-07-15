from __future__ import annotations

"""Final fail-closed execution freeze for the SEC-13F contract-v2 study.

This module intentionally has no execution entry point.  It binds and later
recomputes every input needed by the runner while keeping private packs, gold
artifacts, expected-output payloads, sealed content, and provider secrets out
of the freeze.  The benchmark tree receipt is allowed to hash verifier bytes;
the bytes themselves are never parsed or copied into this manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    V320_PROTOCOL_RELATIVE_PATH,
)
from assumption_agent.benchmarks.financial_sec13f_contract_integration_v2 import (
    SharedFinancialSec13FContractPlannerV2,
)
from assumption_agent.benchmarks.offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
    TUNA_PYPI_INDEX_URL,
    offline_verifier_profile_for_family,
    offline_verifier_runtime_key,
    offline_verifier_volume_name,
)
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.models import stable_hash

from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    read_json,
    sha256_file,
    verify_measurement_view,
    write_json,
)
from replication_runtime.financial_semantic_v2.plan import (
    FixedPeriodOutTreatmentV2,
    MeasurementPlanV2,
    MeasurementTargetV2,
    build_measurement_plan_v2,
)

from .formation import (
    ACQUISITION_RECEIPT_VERSION,
    FROZEN_ORACLE_IDS,
    PACK_FORMATION_RECEIPT_VERSION,
    PREREGISTRATION_VERSION,
    STUDY_ID,
    validate_acquisition_receipt_v1,
    validate_preregistration_v1,
)
from .hygienic_materialize import (
    FAMILY,
    MATERIALIZATION_REPORT_NAME,
    MATERIALIZATION_VERSION,
    TREE_RECEIPT_VERSION,
    measurement_benchmark_tree_receipt_v2,
)
from .hygienic_prewarm import (
    OFFLINE_VERIFIER_PROFILE_ID,
    OFFLINE_VERIFIER_REQUIREMENTS,
    PREWARM_VERSION,
    _load_materialization,
    _validate_item_receipts,
    _validate_period_sources,
)
from .provider import (
    build_execution_provider_binding_v1,
    validate_execution_provider_binding_v1,
)
from .treatment import (
    FixedContractCandidateV2,
    build_evaluation_treatment_v2,
    load_fixed_contract_candidate_v2,
    validate_evaluation_treatment_v2,
)


EXECUTION_FREEZE_VERSION = "financial_sec13f_contract_execution_freeze_v2"
SOURCE_CLOSURE_VERSION = "financial_sec13f_contract_runtime_closure_v2"
TYPED_PLAN_SET_VERSION = "financial_sec13f_contract_typed_plan_set_v2"

__all__ = [
    "ContractFreezeError",
    "EXECUTION_FREEZE_VERSION",
    "SOURCE_CLOSURE_VERSION",
    "TYPED_PLAN_SET_VERSION",
    "build_execution_freeze_v2",
    "build_execution_source_closure_v2",
    "load_execution_freeze_v2",
    "validate_execution_freeze_v2",
    "validate_execution_source_closure_v2",
]

_RUNTIME_SOURCE_ROOTS = (
    "assumption_agent",
    "replication_runtime",
    "candidates/financial_sec13f_contract_operator_v2",
)
_TRANSIENT_DIRS = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
_TRANSIENT_SUFFIXES = frozenset({".pyc", ".pyo"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_SECRET_VALUE = re.compile(r"(?i)(?:^sk-[a-z0-9_-]{8,}|^bearer\s+\S+)")

_PREPARATION_RECEIPT_FIELDS = {
    "activation_blocker",
    "base_image_id",
    "base_image_tag",
    "docker_install_network",
    "online_download_attempted",
    "package_index_origin",
    "platform",
    "policy",
    "probe_passed",
    "probe_workspace_mode",
    "profile_hash",
    "profile_id",
    "python_abi",
    "python_version",
    "raw_content_persisted",
    "receipt_hash",
    "report_version",
    "runtime_key",
    "runtime_reused",
    "runtime_volume_hash",
    "semantic_prelude_id",
    "wheel_count",
    "wheel_total_bytes",
    "wheelhouse_reused",
    "wheels",
}


class ContractFreezeError(PermissionError):
    """A final execution binding is incomplete, unsafe, or has drifted."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if not _is_sha256(declared) or declared != payload_hash(body):
        raise ContractFreezeError(f"{label} self hash drifted")
    return str(declared)


def _git_root(project: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        root = Path(completed.stdout.strip()).resolve(strict=True)
        project.relative_to(root)
        return root
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise ContractFreezeError("project Git worktree is unavailable") from exc


def _head_commit(project: Path) -> str:
    try:
        value = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractFreezeError("project Git commit is unavailable") from exc
    if _GIT_COMMIT.fullmatch(value) is None:
        raise ContractFreezeError("project Git commit is malformed")
    return value


def _repo_relative(project: Path, relative: str) -> str:
    try:
        prefix = project.relative_to(_git_root(project))
    except ValueError as exc:
        raise ContractFreezeError("project escaped its Git worktree") from exc
    return (prefix / relative).as_posix()


def _git_blob(project: Path, commit: str, relative: str) -> bytes:
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ContractFreezeError("committed artifact hash is malformed")
    try:
        return subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "show",
                f"{commit}:{_repo_relative(project, relative)}",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractFreezeError("committed artifact blob is unavailable") from exc


def _validate_relative_text(relative: object, *, label: str) -> str:
    if not isinstance(relative, str) or not relative:
        raise ContractFreezeError(f"{label} relative path is malformed")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or "." in raw.parts:
        raise ContractFreezeError(f"{label} relative path is unsafe")
    return raw.as_posix()


def _project_path(project: Path, relative: object, *, label: str) -> Path:
    normalized = _validate_relative_text(relative, label=label)
    current = project
    for part in Path(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise ContractFreezeError(f"{label} path contains a symbolic link")
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(project)
    except (FileNotFoundError, ValueError) as exc:
        raise ContractFreezeError(f"{label} path escaped or is missing") from exc
    return resolved


def _relative_artifact(
    project: Path, supplied: str | Path, *, label: str
) -> tuple[Path, str]:
    unresolved = Path(supplied).expanduser()
    if not unresolved.is_absolute():
        unresolved = project / unresolved
    if unresolved.is_symlink():
        raise ContractFreezeError(f"{label} is a symbolic link")
    try:
        resolved = unresolved.resolve(strict=True)
        relative = resolved.relative_to(project).as_posix()
    except (FileNotFoundError, ValueError) as exc:
        raise ContractFreezeError(f"{label} must be inside the project") from exc
    checked = _project_path(project, relative, label=label)
    if checked != resolved or not resolved.is_file():
        raise ContractFreezeError(f"{label} is not a regular project file")
    return resolved, relative


def _committed_file_binding(
    project: Path, supplied: str | Path, *, label: str
) -> dict[str, str]:
    path, relative = _relative_artifact(project, supplied, label=label)
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                relative,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        commit = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "log",
                "-1",
                "--format=%H",
                "--",
                relative,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractFreezeError(
            f"{label} Git binding is unavailable"
        ) from exc
    live_hash = sha256_file(path)
    if (
        status.strip()
        or _GIT_COMMIT.fullmatch(commit) is None
        or hashlib.sha256(_git_blob(project, commit, relative)).hexdigest()
        != live_hash
    ):
        raise ContractFreezeError(f"{label} must be committed and clean")
    return {
        "relative_path": relative,
        "file_sha256": live_hash,
        "committed_at_git_commit": commit,
    }


def _validate_committed_binding(
    project: Path,
    value: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    if set(value) != {
        "relative_path",
        "file_sha256",
        "committed_at_git_commit",
    } or not _is_sha256(value.get("file_sha256")):
        raise ContractFreezeError(f"{label} committed binding is malformed")
    commit = str(value.get("committed_at_git_commit") or "")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ContractFreezeError(f"{label} commit is malformed")
    path = _project_path(project, value.get("relative_path"), label=label)
    relative = str(value["relative_path"])
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                relative,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractFreezeError(f"{label} Git status is unavailable") from exc
    expected = str(value["file_sha256"])
    if (
        status.strip()
        or not path.is_file()
        or sha256_file(path) != expected
        or hashlib.sha256(_git_blob(project, commit, relative)).hexdigest()
        != expected
    ):
        raise ContractFreezeError(f"{label} committed binding drifted")
    return path


def _runtime_source_files(project: Path) -> tuple[tuple[str, Path], ...]:
    found: dict[str, Path] = {}
    for relative_root in _RUNTIME_SOURCE_ROOTS:
        root = _project_path(project, relative_root, label="runtime source root")
        if root.is_symlink() or not root.is_dir():
            raise ContractFreezeError("runtime source root is unavailable")
        for path in sorted(
            root.rglob("*"),
            key=lambda value: value.relative_to(project).as_posix(),
        ):
            if path.is_symlink():
                raise ContractFreezeError(
                    "runtime source closure contains a symbolic link"
                )
            local = path.relative_to(root)
            if any(part in _TRANSIENT_DIRS for part in local.parts):
                continue
            if path.is_dir():
                continue
            if not path.is_file():
                raise ContractFreezeError(
                    "runtime source closure contains a special file"
                )
            if path.suffix in _TRANSIENT_SUFFIXES:
                continue
            relative = path.relative_to(project).as_posix()
            if relative in found:
                raise ContractFreezeError("runtime source path is duplicated")
            found[relative] = path
    if not found:
        raise ContractFreezeError("runtime source closure is empty")
    return tuple(sorted(found.items()))


def build_execution_source_closure_v2(
    project_root: str | Path,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    rows = [
        {"relative_path": relative, "file_sha256": sha256_file(path)}
        for relative, path in _runtime_source_files(project)
    ]
    body = {
        "closure_version": SOURCE_CLOSURE_VERSION,
        "scope_policy": "entire_runtime_roots_all_regular_files_v2",
        "runtime_scope_paths": list(_RUNTIME_SOURCE_ROOTS),
        "excluded_directory_names": sorted(_TRANSIENT_DIRS),
        "excluded_file_suffixes": sorted(_TRANSIENT_SUFFIXES),
        "files": rows,
        "file_count": len(rows),
        "file_set_hash": payload_hash(rows),
    }
    closure = {
        **body,
        "source_commit": _head_commit(project),
        "closure_hash": payload_hash(body),
    }
    validate_execution_source_closure_v2(
        closure,
        project_root=project,
    )
    return closure


def validate_execution_source_closure_v2(
    value: Mapping[str, Any], *, project_root: str | Path
) -> str:
    project = Path(project_root).expanduser().resolve(strict=True)
    expected_fields = {
        "closure_version",
        "scope_policy",
        "runtime_scope_paths",
        "excluded_directory_names",
        "excluded_file_suffixes",
        "files",
        "file_count",
        "file_set_hash",
        "source_commit",
        "closure_hash",
    }
    rows = value.get("files")
    if (
        set(value) != expected_fields
        or value.get("closure_version") != SOURCE_CLOSURE_VERSION
        or value.get("scope_policy")
        != "entire_runtime_roots_all_regular_files_v2"
        or value.get("runtime_scope_paths") != list(_RUNTIME_SOURCE_ROOTS)
        or value.get("excluded_directory_names") != sorted(_TRANSIENT_DIRS)
        or value.get("excluded_file_suffixes")
        != sorted(_TRANSIENT_SUFFIXES)
        or not isinstance(rows, list)
        or not rows
    ):
        raise ContractFreezeError("runtime source closure fields drifted")
    normalized: list[dict[str, str]] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"relative_path", "file_sha256"}
            or not _is_sha256(row.get("file_sha256"))
        ):
            raise ContractFreezeError("runtime source row is malformed")
        relative = _validate_relative_text(
            row.get("relative_path"), label="runtime source"
        )
        normalized.append(
            {"relative_path": relative, "file_sha256": str(row["file_sha256"])}
        )
    body = {
        "closure_version": SOURCE_CLOSURE_VERSION,
        "scope_policy": "entire_runtime_roots_all_regular_files_v2",
        "runtime_scope_paths": list(_RUNTIME_SOURCE_ROOTS),
        "excluded_directory_names": sorted(_TRANSIENT_DIRS),
        "excluded_file_suffixes": sorted(_TRANSIENT_SUFFIXES),
        "files": normalized,
        "file_count": len(normalized),
        "file_set_hash": payload_hash(normalized),
    }
    commit = str(value.get("source_commit") or "")
    if (
        rows != sorted(normalized, key=lambda row: row["relative_path"])
        or len({row["relative_path"] for row in normalized}) != len(normalized)
        or any(value.get(key) != expected for key, expected in body.items())
        or _GIT_COMMIT.fullmatch(commit) is None
        or value.get("closure_hash") != payload_hash(body)
    ):
        raise ContractFreezeError("runtime source closure identity drifted")
    current_rows = [
        {"relative_path": relative, "file_sha256": sha256_file(path)}
        for relative, path in _runtime_source_files(project)
    ]
    if current_rows != normalized:
        raise ContractFreezeError("runtime source closure changed after freeze")
    try:
        subprocess.run(
            ["git", "-C", str(project), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
        )
        difference = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "diff",
                "--quiet",
                commit,
                "--",
                *_RUNTIME_SOURCE_ROOTS,
            ],
            check=False,
            capture_output=True,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *_RUNTIME_SOURCE_ROOTS,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractFreezeError(
            "runtime source Git comparison is unavailable"
        ) from exc
    if difference.returncode != 0 or status.stdout.strip():
        raise ContractFreezeError("runtime source scope is not committed and clean")
    return str(value["closure_hash"])


def _validate_safe_formation_receipt(
    value: Mapping[str, Any],
    *,
    preregistration_hash: str,
    acquisition_receipt_hash: str,
    measurement_view: Mapping[str, Any],
    preregistration: Mapping[str, Any],
) -> str:
    expected_fields = {
        "receipt_version",
        "study_id",
        "private_pack_hash",
        "measurement_view_hash",
        "prior_measurement_view_hash",
        "preregistration_hash",
        "acquisition_receipt_hash",
        "input_binding_complete",
        "collision_audit",
        "frozen_pack_implementation",
        "frozen_oracle_ids_reserved_for_later_measurement_gold",
        "oracle_calls",
        "gold_formed",
        "prior_private_pack_accessed",
        "prior_sealed_content_accessed",
        "model_calls",
        "network_calls",
        "online_judge_calls",
        "secret_value_persisted",
        "receipt_hash",
    }
    if set(value) != expected_fields:
        raise ContractFreezeError("pack formation receipt fields drifted")
    declared = _require_self_hash(
        value, field="receipt_hash", label="pack formation receipt"
    )
    collision = value.get("collision_audit")
    collision_fields = {
        "policy",
        "prior_instruction_commitment_count",
        "prior_query_commitment_count",
        "new_instruction_commitment_count",
        "new_query_commitment_count",
        "instruction_collision_count",
        "query_collision_count",
        "prior_private_pack_accessed",
        "prior_sealed_content_accessed",
        "audit_hash",
    }
    if not isinstance(collision, Mapping) or set(collision) != collision_fields:
        raise ContractFreezeError("pack collision receipt fields drifted")
    collision_body = dict(collision)
    collision_hash = collision_body.pop("audit_hash", None)
    new_count = int(measurement_view["measurement_item_count"]) + int(
        measurement_view["sealed_item_count"]
    )
    prior_hash = preregistration.get("prior_commitment_view", {}).get(
        "measurement_view_hash"
    )
    if (
        value.get("receipt_version") != PACK_FORMATION_RECEIPT_VERSION
        or value.get("study_id") != STUDY_ID
        or value.get("private_pack_hash")
        != measurement_view["private_pack_hash"]
        or value.get("measurement_view_hash")
        != measurement_view["measurement_view_hash"]
        or value.get("prior_measurement_view_hash") != prior_hash
        or value.get("preregistration_hash") != preregistration_hash
        or value.get("acquisition_receipt_hash")
        != acquisition_receipt_hash
        or value.get("input_binding_complete") is not True
        or value.get("frozen_pack_implementation")
        != "replication_runtime.financial_semantic_v2.pack"
        or value.get("frozen_oracle_ids_reserved_for_later_measurement_gold")
        != list(FROZEN_ORACLE_IDS)
        or value.get("oracle_calls") != 0
        or value.get("gold_formed") is not False
        or value.get("prior_private_pack_accessed") is not False
        or value.get("prior_sealed_content_accessed") is not False
        or value.get("model_calls") != 0
        or value.get("network_calls") != 0
        or value.get("online_judge_calls") != 0
        or value.get("secret_value_persisted") is not False
        or collision_hash != payload_hash(collision_body)
        or collision.get("policy")
        != "old_new_query_instruction_commitments_disjoint_v1"
        or collision.get("prior_instruction_commitment_count") != 12
        or collision.get("prior_query_commitment_count") != 12
        or collision.get("new_instruction_commitment_count") != new_count
        or collision.get("new_query_commitment_count") != new_count
        or collision.get("instruction_collision_count") != 0
        or collision.get("query_collision_count") != 0
        or collision.get("prior_private_pack_accessed") is not False
        or collision.get("prior_sealed_content_accessed") is not False
    ):
        raise ContractFreezeError("pack formation receipt drifted")
    return declared


def _validate_materialization_and_tree(
    *,
    project: Path,
    benchmark_root: str | Path,
    materialization_report_path: str | Path,
    measurement_view: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    unresolved_benchmark = Path(benchmark_root).expanduser()
    if not unresolved_benchmark.is_absolute():
        unresolved_benchmark = project / unresolved_benchmark
    if unresolved_benchmark.is_symlink():
        raise ContractFreezeError("measurement benchmark root is symlinked")
    try:
        benchmark = unresolved_benchmark.resolve(strict=True)
        benchmark_relative = benchmark.relative_to(project).as_posix()
    except (FileNotFoundError, ValueError) as exc:
        raise ContractFreezeError(
            "measurement benchmark must be inside the project"
        ) from exc
    checked = _project_path(
        project, benchmark_relative, label="measurement benchmark"
    )
    if checked != benchmark or not benchmark.is_dir():
        raise ContractFreezeError("measurement benchmark is not a regular tree")
    report_path, report_relative = _relative_artifact(
        project,
        materialization_report_path,
        label="materialization report",
    )
    if report_path != benchmark / MATERIALIZATION_REPORT_NAME:
        raise ContractFreezeError(
            "materialization report is outside the benchmark root"
        )
    try:
        materialization = _load_materialization(benchmark)
        _validate_item_receipts(
            benchmark=benchmark,
            measurement_view=measurement_view,
            materialization=materialization,
        )
        _validate_period_sources(
            measurement_view=measurement_view,
            materialization=materialization,
        )
    except Exception as exc:
        raise ContractFreezeError(
            "measurement materialization failed its hygienic validation"
        ) from exc
    actual_tree = measurement_benchmark_tree_receipt_v2(benchmark)
    item_ids = [
        str(item["item_id"])
        for item in measurement_view["measurement_items"]
    ]
    receipts = materialization.get("items")
    sources = materialization.get("period_source_receipts")
    if (
        materialization.get("materialization_version")
        != MATERIALIZATION_VERSION
        or materialization.get("tree_receipt_version")
        != TREE_RECEIPT_VERSION
        or materialization.get("measurement_view_hash")
        != measurement_view["measurement_view_hash"]
        or materialization.get("private_pack_hash")
        != measurement_view["private_pack_hash"]
        or materialization.get("benchmark_tree_hash")
        != actual_tree["tree_hash"]
        or actual_tree.get("tree_receipt_version") != TREE_RECEIPT_VERSION
        or not isinstance(receipts, list)
        or not all(isinstance(row, Mapping) for row in receipts)
        or [str(row.get("item_id")) for row in receipts] != item_ids
        or not isinstance(sources, Mapping)
        or set(sources) != {"previous", "current"}
        or not all(
            isinstance(sources.get(role), Mapping)
            for role in ("previous", "current")
        )
        or any(
            sources[role].get("source_fingerprint")
            != measurement_view["sources"][role]["source_fingerprint"]
            for role in ("previous", "current")
        )
    ):
        raise ContractFreezeError("measurement materialization binding drifted")
    binding = {
        "relative_path": report_relative,
        "file_sha256": sha256_file(report_path),
        "materialization_hash": materialization["materialization_hash"],
        "benchmark_root_relative_path": benchmark_relative,
        "tree_receipt_version": TREE_RECEIPT_VERSION,
        "benchmark_tree_hash": actual_tree["tree_hash"],
        "sealed_task_count_materialized": 0,
    }
    return materialization, binding, benchmark, benchmark_relative


def _validate_preparation_sidecar(
    prewarm_path: Path,
    prewarm: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    preparation = prewarm.get("offline_verifier_preparation")
    if not isinstance(preparation, Mapping) or set(preparation) != {
        "relative_path",
        "file_sha256",
        "receipt_hash",
        "network_allowed_only_during_preparation",
        "docker_install_network",
        "probe_passed",
    }:
        raise ContractFreezeError("prewarm preparation binding is malformed")
    relative = _validate_relative_text(
        preparation.get("relative_path"), label="prewarm preparation"
    )
    if relative != "offline-verifier.preparation.json":
        raise ContractFreezeError("prewarm preparation sidecar name drifted")
    sidecar = prewarm_path.parent / relative
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ContractFreezeError("prewarm preparation sidecar is unavailable")
    receipt = read_json(sidecar)
    receipt_hash = _require_self_hash(
        receipt,
        field="receipt_hash",
        label="offline verifier preparation",
    )
    wheels = receipt.get("wheels")
    formal_rows = prewarm.get("formal_cache_rows")
    image_ids = (
        {
            str(row.get("image_id"))
            for row in formal_rows
            if isinstance(row, Mapping)
        }
        if isinstance(formal_rows, list)
        else set()
    )
    cache_keys = (
        {
            str(row.get("cache_key"))
            for row in formal_rows
            if isinstance(row, Mapping)
        }
        if isinstance(formal_rows, list)
        else set()
    )
    profile = offline_verifier_profile_for_family(FAMILY)
    expected_runtime_key = (
        offline_verifier_runtime_key(profile=profile)
        if profile is not None
        else ""
    )
    wheel_rows_are_valid = isinstance(wheels, list) and all(
        isinstance(row, Mapping)
        and set(row) == {"filename", "sha256", "size"}
        and isinstance(row.get("filename"), str)
        and bool(row.get("filename"))
        and Path(str(row["filename"])).name == row["filename"]
        and str(row["filename"]).endswith(".whl")
        and _is_sha256(row.get("sha256"))
        and not isinstance(row.get("size"), bool)
        and isinstance(row.get("size"), int)
        and row["size"] > 0
        for row in wheels or ()
    )
    wheel_names = (
        [str(row["filename"]) for row in wheels]
        if wheel_rows_are_valid
        else []
    )
    runtime_reused = receipt.get("runtime_reused")
    wheelhouse_reused = receipt.get("wheelhouse_reused")
    online_download_attempted = receipt.get("online_download_attempted")
    if (
        set(receipt) != _PREPARATION_RECEIPT_FIELDS
        or profile is None
        or sha256_file(sidecar) != preparation.get("file_sha256")
        or receipt_hash != preparation.get("receipt_hash")
        or preparation.get("network_allowed_only_during_preparation") is not True
        or preparation.get("docker_install_network") != "none"
        or preparation.get("probe_passed") is not True
        or receipt.get("report_version")
        != "offline_verifier_preparation_receipt_v2"
        or receipt.get("policy") != OFFLINE_VERIFIER_POLICY_VERSION
        or receipt.get("profile_id") != OFFLINE_VERIFIER_PROFILE_ID
        or receipt.get("profile_hash") != profile.profile_hash
        or receipt.get("profile_hash")
        != prewarm.get("offline_verifier_profile_hash")
        or receipt.get("runtime_key") != expected_runtime_key
        or receipt.get("runtime_key")
        != prewarm.get("offline_verifier_runtime_key")
        or receipt.get("runtime_volume_hash")
        != stable_hash(
            {"volume": offline_verifier_volume_name(expected_runtime_key)}
        )
        or len(image_ids) != 1
        or image_ids != {str(receipt.get("base_image_id"))}
        or len(cache_keys) != 1
        or receipt.get("base_image_tag")
        != "assumption-v2-item:" + next(iter(cache_keys), "")[:24]
        or receipt.get("python_version") != profile.python_version
        or receipt.get("python_abi") != profile.python_abi
        or receipt.get("platform") != profile.platform
        or receipt.get("semantic_prelude_id") != profile.semantic_prelude_id
        or receipt.get("probe_workspace_mode")
        != profile.probe_workspace_mode
        or receipt.get("activation_blocker") != profile.activation_blocker
        or receipt.get("package_index_origin") != TUNA_PYPI_INDEX_URL
        or receipt.get("docker_install_network") != "none"
        or not isinstance(runtime_reused, bool)
        or not isinstance(wheelhouse_reused, bool)
        or not isinstance(online_download_attempted, bool)
        or (runtime_reused and online_download_attempted)
        or (wheelhouse_reused and not wheels)
        or (online_download_attempted and wheelhouse_reused)
        or (not runtime_reused and not wheels)
        or receipt.get("probe_passed") is not True
        or receipt.get("raw_content_persisted") is not False
        or not wheel_rows_are_valid
        or receipt.get("wheel_count") != len(wheels)
        or receipt.get("wheel_total_bytes")
        != sum(int(row["size"]) for row in wheels or ())
        or wheel_names != sorted(wheel_names)
        or len(wheel_names) != len(set(wheel_names))
    ):
        raise ContractFreezeError("offline verifier preparation drifted")
    return receipt, {
        "relative_path": relative,
        "file_sha256": sha256_file(sidecar),
        "receipt_hash": receipt_hash,
    }


def _validate_prewarm(
    *,
    prewarm_path: Path,
    measurement_view: Mapping[str, Any],
    materialization: Mapping[str, Any],
    benchmark_tree_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prewarm = read_json(prewarm_path)
    prewarm_hash = _require_self_hash(
        prewarm, field="prewarm_hash", label="measurement prewarm"
    )
    expected_fields = {
        "prewarm_version",
        "tree_receipt_version",
        "measurement_view_hash",
        "materialization_hash",
        "benchmark_tree_hash",
        "pre_prewarm_tree_hash",
        "post_prewarm_tree_hash",
        "benchmark_tree_unchanged",
        "python_dont_write_bytecode",
        "python_dont_write_bytecode_env",
        "item_count",
        "preparation_rows",
        "preparation_row_set_hash",
        "formal_cache_rows",
        "formal_cache_row_set_hash",
        "unique_image_id_hash",
        "unique_cache_key_hash",
        "offline_verifier_profile_id",
        "offline_verifier_profile_hash",
        "offline_verifier_requirements",
        "offline_verifier_requirements_hash",
        "offline_verifier_runtime_key",
        "offline_verifier_preparation",
        "formal_execution_cache_only",
        "formal_image_cache_only",
        "formal_offline_verifier_cache_only",
        "preparation_network_allowed",
        "formal_verifier_network",
        "model_calls",
        "online_judge_calls",
        "sealed_task_count",
        "sealed_content_accessed",
        "secret_value_persisted",
        "prewarm_hash",
    }
    preparation_rows = prewarm.get("preparation_rows")
    formal_rows = prewarm.get("formal_cache_rows")
    item_ids = [
        str(item["item_id"])
        for item in measurement_view["measurement_items"]
    ]
    preparation_fields = {
        "item_id",
        "item_id_hash",
        "cache_key",
        "environment_hash",
        "source_environment_hash",
        "image_id",
        "agent_runtime_key",
        "agent_runtime_version",
        "prepared_before_formal_cache_check",
    }
    formal_fields = {
        "item_id",
        "item_id_hash",
        "cache_key",
        "environment_hash",
        "source_environment_hash",
        "image_id",
        "agent_runtime_key",
        "agent_runtime_version",
        "prebuilt_cache_reused",
        "offline_verifier_profile_id",
        "offline_verifier_profile_hash",
        "offline_verifier_runtime_key",
        "offline_verifier_runtime_reused",
        "verifier_runtime_network",
    }
    profile = offline_verifier_profile_for_family(FAMILY)
    expected_profile_hash = profile.profile_hash if profile is not None else ""
    expected_runtime_key = (
        offline_verifier_runtime_key(profile=profile)
        if profile is not None
        else ""
    )
    if (
        set(prewarm) != expected_fields
        or prewarm.get("prewarm_version") != PREWARM_VERSION
        or prewarm.get("tree_receipt_version") != TREE_RECEIPT_VERSION
        or prewarm.get("measurement_view_hash")
        != measurement_view["measurement_view_hash"]
        or prewarm.get("materialization_hash")
        != materialization["materialization_hash"]
        or prewarm.get("benchmark_tree_hash") != benchmark_tree_hash
        or prewarm.get("pre_prewarm_tree_hash") != benchmark_tree_hash
        or prewarm.get("post_prewarm_tree_hash") != benchmark_tree_hash
        or prewarm.get("benchmark_tree_unchanged") is not True
        or prewarm.get("python_dont_write_bytecode") is not True
        or prewarm.get("python_dont_write_bytecode_env") != "1"
        or prewarm.get("item_count") != 8
        or not isinstance(preparation_rows, list)
        or len(preparation_rows) != 8
        or not all(isinstance(row, Mapping) for row in preparation_rows)
        or [str(row.get("item_id")) for row in preparation_rows]
        != item_ids
        or any(
            set(row) != preparation_fields
            or row.get("item_id_hash")
            != payload_hash({"item_id": row.get("item_id")})
            or not all(
                _is_sha256(row.get(field))
                for field in (
                    "cache_key",
                    "environment_hash",
                    "source_environment_hash",
                    "agent_runtime_key",
                )
            )
            or not isinstance(row.get("image_id"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", row["image_id"])
            is None
            or not isinstance(row.get("agent_runtime_version"), str)
            or not row["agent_runtime_version"]
            or row.get("prepared_before_formal_cache_check") is not True
            for row in preparation_rows
        )
        or prewarm.get("preparation_row_set_hash")
        != payload_hash(preparation_rows)
        or not isinstance(formal_rows, list)
        or len(formal_rows) != 8
        or not all(isinstance(row, Mapping) for row in formal_rows)
        or [str(row.get("item_id")) for row in formal_rows] != item_ids
        or any(
            set(row) != formal_fields
            or row.get("item_id_hash")
            != payload_hash({"item_id": row.get("item_id")})
            or not all(
                _is_sha256(row.get(field))
                for field in (
                    "cache_key",
                    "environment_hash",
                    "source_environment_hash",
                    "agent_runtime_key",
                )
            )
            or not isinstance(row.get("image_id"), str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", row["image_id"])
            is None
            or not isinstance(row.get("agent_runtime_version"), str)
            or not row["agent_runtime_version"]
            or row.get("prebuilt_cache_reused") is not True
            or row.get("offline_verifier_profile_id")
            != OFFLINE_VERIFIER_PROFILE_ID
            or row.get("offline_verifier_profile_hash")
            != expected_profile_hash
            or row.get("offline_verifier_runtime_key")
            != expected_runtime_key
            or row.get("offline_verifier_runtime_reused") is not True
            or row.get("verifier_runtime_network") != "none"
            for row in formal_rows
        )
        or prewarm.get("formal_cache_row_set_hash") != payload_hash(formal_rows)
        or profile is None
        or prewarm.get("offline_verifier_profile_id")
        != OFFLINE_VERIFIER_PROFILE_ID
        or prewarm.get("offline_verifier_profile_hash")
        != expected_profile_hash
        or tuple(prewarm.get("offline_verifier_requirements") or ())
        != OFFLINE_VERIFIER_REQUIREMENTS
        or prewarm.get("offline_verifier_requirements_hash")
        != payload_hash(list(OFFLINE_VERIFIER_REQUIREMENTS))
        or prewarm.get("offline_verifier_runtime_key")
        != expected_runtime_key
        or prewarm.get("formal_execution_cache_only") is not True
        or prewarm.get("formal_image_cache_only") is not True
        or prewarm.get("formal_offline_verifier_cache_only") is not True
        or prewarm.get("preparation_network_allowed") is not True
        or prewarm.get("formal_verifier_network") != "none"
        or prewarm.get("model_calls") != 0
        or prewarm.get("online_judge_calls") != 0
        or prewarm.get("sealed_task_count") != 0
        or prewarm.get("sealed_content_accessed") is not False
        or prewarm.get("secret_value_persisted") is not False
    ):
        raise ContractFreezeError("measurement prewarm drifted")
    shared_fields = (
        "item_id",
        "item_id_hash",
        "cache_key",
        "environment_hash",
        "source_environment_hash",
        "image_id",
        "agent_runtime_key",
        "agent_runtime_version",
    )
    if any(
        any(preparation.get(field) != formal.get(field) for field in shared_fields)
        for preparation, formal in zip(preparation_rows, formal_rows)
    ):
        raise ContractFreezeError("prewarm preparation/formal rows diverged")
    image_ids = {str(row["image_id"]) for row in formal_rows}
    cache_keys = {str(row["cache_key"]) for row in formal_rows}
    environment_hashes = {str(row["environment_hash"]) for row in formal_rows}
    source_environment_hashes = {
        str(row["source_environment_hash"]) for row in formal_rows
    }
    agent_runtime_keys = {
        str(row["agent_runtime_key"]) for row in formal_rows
    }
    agent_runtime_versions = {
        str(row["agent_runtime_version"]) for row in formal_rows
    }
    runtime_keys = {
        str(row["offline_verifier_runtime_key"]) for row in formal_rows
    }
    if (
        len(image_ids) != 1
        or len(cache_keys) != 1
        or len(environment_hashes) != 1
        or len(source_environment_hashes) != 1
        or len(agent_runtime_keys) != 1
        or len(agent_runtime_versions) != 1
        or runtime_keys != {expected_runtime_key}
        or prewarm.get("unique_image_id_hash")
        != payload_hash({"image_id": next(iter(image_ids), "")})
        or prewarm.get("unique_cache_key_hash")
        != payload_hash({"cache_key": next(iter(cache_keys), "")})
    ):
        raise ContractFreezeError("prewarm shared runtime binding drifted")
    _, sidecar_binding = _validate_preparation_sidecar(prewarm_path, prewarm)
    binding = {
        "relative_path": "",
        "file_sha256": sha256_file(prewarm_path),
        "prewarm_hash": prewarm_hash,
        "preparation_sidecar": sidecar_binding,
        "formal_execution_cache_only": True,
        "formal_verifier_network": "none",
    }
    return prewarm, binding


def _paper_protocol_binding(
    project: Path, path: str | Path
) -> tuple[PaperProtocol, dict[str, Any]]:
    protocol_path, relative = _relative_artifact(
        project, path, label="paper protocol"
    )
    if relative != V320_PROTOCOL_RELATIVE_PATH:
        raise ContractFreezeError("runner paper protocol path drifted")
    protocol = PaperProtocol.read(protocol_path)
    policy_hash = protocol.codex_agent_execution_policy.policy_hash
    if not _is_sha256(policy_hash):
        raise ContractFreezeError("paper execution policy hash is malformed")
    binding = {
        "relative_path": relative,
        "file_sha256": sha256_file(protocol_path),
        "protocol_id": protocol.id,
        "protocol_hash": protocol.protocol_hash,
        "agent_id": str(protocol.payload["agent_id"]),
        "model": str(protocol.payload["model"]),
        "max_steps": int(protocol.payload["max_steps"]),
        "codex_agent_execution_policy_hash": policy_hash,
    }
    return protocol, binding


def _typed_plan_set(
    *,
    measurement_view: Mapping[str, Any],
    candidate: FixedContractCandidateV2,
) -> tuple[str, dict[str, Any]]:
    planner = SharedFinancialSec13FContractPlannerV2(
        asset_path=candidate.operator_asset_path
    )
    by_item: dict[str, dict[str, Any]] = {}
    safe_rows: list[dict[str, Any]] = []
    for item in sorted(
        measurement_view["measurement_items"],
        key=lambda row: str(row["item_id"]),
    ):
        item_id = str(item["item_id"])
        instruction = str(item["instruction"])
        instruction_sha256 = hashlib.sha256(
            instruction.encode("utf-8")
        ).hexdigest()
        if instruction_sha256 != item.get("instruction_sha256"):
            raise ContractFreezeError("measurement instruction hash drifted")
        plan, extraction = planner.build(instruction)
        if (
            plan.get("instruction_sha256") != instruction_sha256
            or not _is_sha256(plan.get("plan_hash"))
            or extraction.get("plan_hash") != plan.get("plan_hash")
            or not _is_sha256(extraction.get("receipt_hash"))
        ):
            raise ContractFreezeError("typed contract plan is malformed")
        bound_planner_hash = stable_hash(
            {
                "policy": "item_local_precomputed_sec13f_contract_v2",
                "shared_planner_hash": planner.planner_hash,
                "instruction_sha256": instruction_sha256,
                "plan_hash": plan["plan_hash"],
                "extraction_receipt_hash": extraction["receipt_hash"],
            }
        )
        runtime_receipt = {
            "applicable": True,
            "instruction_sha256": instruction_sha256,
            "plan_hash": plan["plan_hash"],
            "extraction_receipt_hash": extraction["receipt_hash"],
            "planner_hash": bound_planner_hash,
            "raw_plan_persisted": False,
            "model_calls": 0,
            "online_calls": 0,
        }
        by_item[item_id] = runtime_receipt
        safe_rows.append(
            {
                "item_id_hash": stable_hash({"item_id": item_id}),
                **runtime_receipt,
            }
        )
    if len(by_item) != 8:
        raise ContractFreezeError("typed plan set must cover eight items")
    plan_set_hash = stable_hash(
        {key: dict(value) for key, value in sorted(by_item.items())}
    )
    section = {
        "plan_set_version": TYPED_PLAN_SET_VERSION,
        "item_count": 8,
        "rows": safe_rows,
        "row_set_hash": payload_hash(safe_rows),
        "plan_set_hash": plan_set_hash,
        "raw_plan_persisted": False,
        "raw_instruction_persisted": False,
        "raw_entity_persisted": False,
        "model_calls": 0,
        "online_calls": 0,
    }
    return plan_set_hash, section


def _build_measurement_plan(
    *,
    measurement_view: Mapping[str, Any],
    candidate: FixedContractCandidateV2,
    evaluation: Mapping[str, Any],
    protocol: PaperProtocol,
    evaluator_epoch: str,
) -> MeasurementPlanV2:
    targets = tuple(
        MeasurementTargetV2(
            item_id=str(item["item_id"]),
            fold_id=f"measurement-fold-{int(item['fold'])}",
        )
        for item in measurement_view["measurement_items"]
    )
    plan = build_measurement_plan_v2(
        targets=targets,
        manifest_hash=str(measurement_view["measurement_view_hash"]),
        evaluator_epoch=evaluator_epoch,
        treatment=FixedPeriodOutTreatmentV2(
            recipe_id=candidate.recipe_id,
            program_set_hash=candidate.program_set_hash,
            period_out_treatment_id=str(evaluation["period_out_treatment_id"]),
            external_skill_source_receipt_hash=(
                candidate.external_skill_source_receipt_hash
            ),
            candidate_skill_source=candidate.candidate_skill_source,
        ),
        agent_id=str(protocol.payload["agent_id"]),
        model=str(protocol.payload["model"]),
        max_steps=int(protocol.payload["max_steps"]),
        codex_agent_execution_policy_hash=(
            protocol.codex_agent_execution_policy.policy_hash
        ),
    )
    plan.verify()
    safe = plan.safe_payload()
    if (
        safe.get("physical_work_unit_count") != 16
        or safe.get("raw_execution_count") != 8
        or safe.get("candidate_execution_count") != 8
        or safe.get("maximum_workers") != 16
        or safe.get("retry_count") != 0
        or len(safe.get("work_units") or ()) != 16
    ):
        raise ContractFreezeError("measurement plan is not the exact 16-unit grid")
    return plan


def _validate_provider_paths_and_binding(
    project: Path,
    provider: Mapping[str, Any],
    *,
    env_file: str | Path | None = None,
) -> dict[str, Any]:
    for field in (
        "identity_sidecar_relative_path",
        "selected_canary_relative_path",
        "selected_event_ledger_relative_path",
        "selection_receipt_relative_path",
    ):
        path = _project_path(project, provider.get(field), label=field)
        if not path.is_file():
            raise ContractFreezeError("provider evidence is not a regular file")
    try:
        return validate_execution_provider_binding_v1(
            provider,
            project_root=project,
            env_file=env_file,
        )
    except Exception as exc:
        raise ContractFreezeError("provider evidence changed after freeze") from exc


def _assert_no_secret_or_raw_payload(value: Any) -> None:
    forbidden_keys = {
        "api_key",
        "authorization",
        "answers",
        "expected_output",
        "gold",
        "gold_payload",
        "private_pack",
        "raw_key",
        "secret_key",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or key.casefold() in forbidden_keys:
                raise ContractFreezeError(
                    "execution freeze contains forbidden raw or secret content"
                )
            _assert_no_secret_or_raw_payload(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_secret_or_raw_payload(nested)
    elif isinstance(value, str) and _SECRET_VALUE.search(value):
        raise ContractFreezeError("execution freeze contains a credential value")


def _execution_policy() -> dict[str, Any]:
    return {
        "physical_calls": 16,
        "outer_workers": 16,
        "model_inference_slots": 16,
        "all_futures_submitted_before_results_read": True,
        "retries": 0,
        "model_replay_authorized": False,
        "offline_evaluation_only": True,
        "online_judge_calls": 0,
        "performance_gate_bound": False,
        "promotion_authorized": False,
        "sealed_execution_authorized": False,
    }


def build_execution_freeze_v2(
    *,
    project_root: str | Path,
    preregistration_path: str | Path,
    acquisition_receipt_path: str | Path,
    formation_receipt_path: str | Path,
    measurement_view_path: str | Path,
    benchmark_root: str | Path,
    materialization_report_path: str | Path,
    prewarm_path: str | Path,
    paper_protocol_path: str | Path,
    provider_identity_sidecar_path: str | Path,
    provider_selection_path: str | Path,
    selected_canary_path: str | Path,
    selected_event_ledger_path: str | Path,
    provider_label: str = "plus",
    provider_env_file: str | Path | None = None,
) -> dict[str, Any]:
    project = Path(project_root).expanduser().resolve(strict=True)
    source_closure = build_execution_source_closure_v2(project)

    prereg_binding = _committed_file_binding(
        project, preregistration_path, label="preregistration"
    )
    prereg = read_json(
        _project_path(
            project, prereg_binding["relative_path"], label="preregistration"
        )
    )
    prereg_hash = validate_preregistration_v1(prereg)
    acquisition_binding = _committed_file_binding(
        project, acquisition_receipt_path, label="acquisition receipt"
    )
    acquisition = read_json(
        _project_path(
            project,
            acquisition_binding["relative_path"],
            label="acquisition receipt",
        )
    )
    acquisition_hash = validate_acquisition_receipt_v1(
        acquisition,
        preregistration=prereg,
    )
    view_binding = _committed_file_binding(
        project, measurement_view_path, label="measurement view"
    )
    view = verify_measurement_view(
        read_json(
            _project_path(
                project, view_binding["relative_path"], label="measurement view"
            )
        )
    )
    formation_binding = _committed_file_binding(
        project, formation_receipt_path, label="formation receipt"
    )
    formation_receipt = read_json(
        _project_path(
            project,
            formation_binding["relative_path"],
            label="formation receipt",
        )
    )
    formation_hash = _validate_safe_formation_receipt(
        formation_receipt,
        preregistration_hash=prereg_hash,
        acquisition_receipt_hash=acquisition_hash,
        measurement_view=view,
        preregistration=prereg,
    )

    materialization, materialization_binding, _, _ = (
        _validate_materialization_and_tree(
            project=project,
            benchmark_root=benchmark_root,
            materialization_report_path=materialization_report_path,
            measurement_view=view,
        )
    )
    prewarm_file, prewarm_relative = _relative_artifact(
        project, prewarm_path, label="prewarm report"
    )
    _, prewarm_binding = _validate_prewarm(
        prewarm_path=prewarm_file,
        measurement_view=view,
        materialization=materialization,
        benchmark_tree_hash=str(materialization_binding["benchmark_tree_hash"]),
    )
    prewarm_binding["relative_path"] = prewarm_relative

    protocol, protocol_binding = _paper_protocol_binding(
        project, paper_protocol_path
    )
    candidate = load_fixed_contract_candidate_v2(project)
    candidate_payload = candidate.safe_payload(project)
    evaluation = build_evaluation_treatment_v2(
        candidate=candidate,
        execution_source_closure_hash=str(source_closure["closure_hash"]),
        measurement_view_hash=str(view["measurement_view_hash"]),
        benchmark_tree_hash=str(materialization_binding["benchmark_tree_hash"]),
    )
    validate_evaluation_treatment_v2(evaluation, candidate=candidate)
    evaluator_epoch = (
        "financial-sec13f-contract-v2-" + str(view["private_pack_hash"])[:12]
    )
    treatment = {
        "evaluation_binding": evaluation,
        "recipe_id": candidate.recipe_id,
        "program_set_hash": candidate.program_set_hash,
        "external_skill_source_receipt_hash": (
            candidate.external_skill_source_receipt_hash
        ),
        "evaluator_epoch": evaluator_epoch,
    }
    provider = build_execution_provider_binding_v1(
        project_root=project,
        provider_label=provider_label,
        identity_sidecar_path=provider_identity_sidecar_path,
        selected_canary_report_path=selected_canary_path,
        selected_event_ledger_path=selected_event_ledger_path,
        selection_receipt_path=provider_selection_path,
        env_file=provider_env_file,
    )
    _validate_provider_paths_and_binding(
        project,
        provider,
        env_file=provider_env_file,
    )
    plan_set_hash, typed_plan_set = _typed_plan_set(
        measurement_view=view,
        candidate=candidate,
    )
    plan = _build_measurement_plan(
        measurement_view=view,
        candidate=candidate,
        evaluation=evaluation,
        protocol=protocol,
        evaluator_epoch=evaluator_epoch,
    )

    preregistration_section = {
        **prereg_binding,
        "manifest_hash": prereg_hash,
        "manifest_version": PREREGISTRATION_VERSION,
    }
    acquisition_section = {
        **acquisition_binding,
        "receipt_hash": acquisition_hash,
        "receipt_version": ACQUISITION_RECEIPT_VERSION,
        "archive_set_hash": acquisition["archive_set_hash"],
    }
    formation_section = {
        **formation_binding,
        "receipt_hash": formation_hash,
        "receipt_version": PACK_FORMATION_RECEIPT_VERSION,
        "private_pack_hash": view["private_pack_hash"],
        "private_pack_accessed_by_freeze": False,
    }
    measurement_view_section = {
        **view_binding,
        "measurement_view_hash": view["measurement_view_hash"],
        "private_pack_hash": view["private_pack_hash"],
        "measurement_count": 8,
        "sealed_commitment_count": int(view["sealed_item_count"]),
        "sealed_content_accessed": False,
    }
    body = {
        "manifest_version": EXECUTION_FREEZE_VERSION,
        "study_id": STUDY_ID,
        "preregistration": preregistration_section,
        "acquisition": acquisition_section,
        "formation": formation_section,
        "measurement_view": measurement_view_section,
        "materialization": materialization_binding,
        "prewarm": prewarm_binding,
        "paper_protocol": protocol_binding,
        "provider": provider,
        "candidate": candidate_payload,
        "treatment": treatment,
        "precomputed_plan_set_hash": plan_set_hash,
        "typed_plan_set": typed_plan_set,
        "plan": {"plan_hash": plan.plan_hash, "safe_payload": plan.safe_payload()},
        "execution_source_closure": source_closure,
        "execution": _execution_policy(),
        "private_pack_accessed": False,
        "gold_artifact_accessed": False,
        "expected_output_content_accessed": False,
        "sealed_content_accessed": False,
        "secret_value_persisted": False,
    }
    _assert_no_secret_or_raw_payload(body)
    return {**body, "manifest_hash": payload_hash(body)}


def _binding_core(
    value: Mapping[str, Any], *, extra_fields: set[str], label: str
) -> dict[str, Any]:
    base = {"relative_path", "file_sha256", "committed_at_git_commit"}
    if set(value) != base | extra_fields:
        raise ContractFreezeError(f"{label} section fields drifted")
    return {
        key: value[key]
        for key in ("relative_path", "file_sha256", "committed_at_git_commit")
    }


def validate_execution_freeze_v2(
    value: Mapping[str, Any], *, project_root: str | Path
) -> FixedContractCandidateV2:
    project = Path(project_root).expanduser().resolve(strict=True)
    expected_fields = {
        "manifest_version",
        "study_id",
        "preregistration",
        "acquisition",
        "formation",
        "measurement_view",
        "materialization",
        "prewarm",
        "paper_protocol",
        "provider",
        "candidate",
        "treatment",
        "precomputed_plan_set_hash",
        "typed_plan_set",
        "plan",
        "execution_source_closure",
        "execution",
        "private_pack_accessed",
        "gold_artifact_accessed",
        "expected_output_content_accessed",
        "sealed_content_accessed",
        "secret_value_persisted",
        "manifest_hash",
    }
    body = dict(value)
    declared = body.pop("manifest_hash", None)
    if (
        set(value) != expected_fields
        or value.get("manifest_version") != EXECUTION_FREEZE_VERSION
        or value.get("study_id") != STUDY_ID
        or not _is_sha256(declared)
        or declared != payload_hash(body)
        or value.get("execution") != _execution_policy()
        or value.get("private_pack_accessed") is not False
        or value.get("gold_artifact_accessed") is not False
        or value.get("expected_output_content_accessed") is not False
        or value.get("sealed_content_accessed") is not False
        or value.get("secret_value_persisted") is not False
    ):
        raise ContractFreezeError("execution freeze fields drifted")
    _assert_no_secret_or_raw_payload(body)

    source = value.get("execution_source_closure")
    if not isinstance(source, Mapping):
        raise ContractFreezeError("execution source closure is missing")
    validate_execution_source_closure_v2(source, project_root=project)

    preregistration = value.get("preregistration")
    if not isinstance(preregistration, Mapping):
        raise ContractFreezeError("preregistration section is malformed")
    prereg_core = _binding_core(
        preregistration,
        extra_fields={"manifest_hash", "manifest_version"},
        label="preregistration",
    )
    prereg_path = _validate_committed_binding(
        project, prereg_core, label="preregistration"
    )
    prereg = read_json(prereg_path)
    prereg_hash = validate_preregistration_v1(prereg)
    if (
        preregistration.get("manifest_hash") != prereg_hash
        or preregistration.get("manifest_version") != PREREGISTRATION_VERSION
    ):
        raise ContractFreezeError("preregistration identity drifted")

    acquisition_section = value.get("acquisition")
    if not isinstance(acquisition_section, Mapping):
        raise ContractFreezeError("acquisition section is malformed")
    acquisition_core = _binding_core(
        acquisition_section,
        extra_fields={"receipt_hash", "receipt_version", "archive_set_hash"},
        label="acquisition",
    )
    acquisition_path = _validate_committed_binding(
        project, acquisition_core, label="acquisition"
    )
    acquisition = read_json(acquisition_path)
    acquisition_hash = validate_acquisition_receipt_v1(
        acquisition, preregistration=prereg
    )
    if (
        acquisition_section.get("receipt_hash") != acquisition_hash
        or acquisition_section.get("receipt_version")
        != ACQUISITION_RECEIPT_VERSION
        or acquisition_section.get("archive_set_hash")
        != acquisition.get("archive_set_hash")
    ):
        raise ContractFreezeError("acquisition identity drifted")

    view_section = value.get("measurement_view")
    if not isinstance(view_section, Mapping):
        raise ContractFreezeError("measurement view section is malformed")
    view_core = _binding_core(
        view_section,
        extra_fields={
            "measurement_view_hash",
            "private_pack_hash",
            "measurement_count",
            "sealed_commitment_count",
            "sealed_content_accessed",
        },
        label="measurement view",
    )
    view_path = _validate_committed_binding(
        project, view_core, label="measurement view"
    )
    view = verify_measurement_view(read_json(view_path))
    if (
        view_section.get("measurement_view_hash")
        != view["measurement_view_hash"]
        or view_section.get("private_pack_hash") != view["private_pack_hash"]
        or view_section.get("measurement_count") != 8
        or view_section.get("sealed_commitment_count")
        != view["sealed_item_count"]
        or view_section.get("sealed_content_accessed") is not False
    ):
        raise ContractFreezeError("measurement view identity drifted")

    formation_section = value.get("formation")
    if not isinstance(formation_section, Mapping):
        raise ContractFreezeError("formation section is malformed")
    formation_core = _binding_core(
        formation_section,
        extra_fields={
            "receipt_hash",
            "receipt_version",
            "private_pack_hash",
            "private_pack_accessed_by_freeze",
        },
        label="formation",
    )
    formation_path = _validate_committed_binding(
        project, formation_core, label="formation"
    )
    formation_receipt = read_json(formation_path)
    formation_hash = _validate_safe_formation_receipt(
        formation_receipt,
        preregistration_hash=prereg_hash,
        acquisition_receipt_hash=acquisition_hash,
        measurement_view=view,
        preregistration=prereg,
    )
    if (
        formation_section.get("receipt_hash") != formation_hash
        or formation_section.get("receipt_version")
        != PACK_FORMATION_RECEIPT_VERSION
        or formation_section.get("private_pack_hash")
        != view["private_pack_hash"]
        or formation_section.get("private_pack_accessed_by_freeze") is not False
    ):
        raise ContractFreezeError("formation identity drifted")

    materialization_section = value.get("materialization")
    materialization_fields = {
        "relative_path",
        "file_sha256",
        "materialization_hash",
        "benchmark_root_relative_path",
        "tree_receipt_version",
        "benchmark_tree_hash",
        "sealed_task_count_materialized",
    }
    if (
        not isinstance(materialization_section, Mapping)
        or set(materialization_section) != materialization_fields
    ):
        raise ContractFreezeError("materialization section fields drifted")
    benchmark = _project_path(
        project,
        materialization_section.get("benchmark_root_relative_path"),
        label="measurement benchmark",
    )
    materialization_path = _project_path(
        project,
        materialization_section.get("relative_path"),
        label="materialization report",
    )
    materialization, expected_materialization, _, _ = (
        _validate_materialization_and_tree(
            project=project,
            benchmark_root=benchmark,
            materialization_report_path=materialization_path,
            measurement_view=view,
        )
    )
    if dict(materialization_section) != expected_materialization:
        raise ContractFreezeError("materialization changed after freeze")

    prewarm_section = value.get("prewarm")
    prewarm_fields = {
        "relative_path",
        "file_sha256",
        "prewarm_hash",
        "preparation_sidecar",
        "formal_execution_cache_only",
        "formal_verifier_network",
    }
    if not isinstance(prewarm_section, Mapping) or set(prewarm_section) != (
        prewarm_fields
    ):
        raise ContractFreezeError("prewarm section fields drifted")
    prewarm_path = _project_path(
        project, prewarm_section.get("relative_path"), label="prewarm report"
    )
    _, expected_prewarm = _validate_prewarm(
        prewarm_path=prewarm_path,
        measurement_view=view,
        materialization=materialization,
        benchmark_tree_hash=str(materialization["benchmark_tree_hash"]),
    )
    expected_prewarm["relative_path"] = str(prewarm_section["relative_path"])
    if dict(prewarm_section) != expected_prewarm:
        raise ContractFreezeError("prewarm changed after freeze")

    protocol_section = value.get("paper_protocol")
    protocol_fields = {
        "relative_path",
        "file_sha256",
        "protocol_id",
        "protocol_hash",
        "agent_id",
        "model",
        "max_steps",
        "codex_agent_execution_policy_hash",
    }
    if (
        not isinstance(protocol_section, Mapping)
        or set(protocol_section) != protocol_fields
    ):
        raise ContractFreezeError("paper protocol section fields drifted")
    protocol_path = _project_path(
        project, protocol_section.get("relative_path"), label="paper protocol"
    )
    protocol, expected_protocol = _paper_protocol_binding(
        project, protocol_path
    )
    if dict(protocol_section) != expected_protocol:
        raise ContractFreezeError("paper protocol changed after freeze")

    candidate = load_fixed_contract_candidate_v2(project)
    if value.get("candidate") != candidate.safe_payload(project):
        raise ContractFreezeError("fixed candidate changed after freeze")
    expected_evaluation = build_evaluation_treatment_v2(
        candidate=candidate,
        execution_source_closure_hash=str(source["closure_hash"]),
        measurement_view_hash=str(view["measurement_view_hash"]),
        benchmark_tree_hash=str(materialization["benchmark_tree_hash"]),
    )
    validate_evaluation_treatment_v2(expected_evaluation, candidate=candidate)
    evaluator_epoch = (
        "financial-sec13f-contract-v2-" + str(view["private_pack_hash"])[:12]
    )
    expected_treatment = {
        "evaluation_binding": expected_evaluation,
        "recipe_id": candidate.recipe_id,
        "program_set_hash": candidate.program_set_hash,
        "external_skill_source_receipt_hash": (
            candidate.external_skill_source_receipt_hash
        ),
        "evaluator_epoch": evaluator_epoch,
    }
    if value.get("treatment") != expected_treatment:
        raise ContractFreezeError("evaluation treatment changed after freeze")

    provider = value.get("provider")
    if not isinstance(provider, Mapping):
        raise ContractFreezeError("provider binding is malformed")
    _validate_provider_paths_and_binding(project, provider)

    expected_plan_set_hash, expected_typed = _typed_plan_set(
        measurement_view=view,
        candidate=candidate,
    )
    if (
        value.get("precomputed_plan_set_hash") != expected_plan_set_hash
        or value.get("typed_plan_set") != expected_typed
    ):
        raise ContractFreezeError("precomputed typed plans changed after freeze")
    plan = _build_measurement_plan(
        measurement_view=view,
        candidate=candidate,
        evaluation=expected_evaluation,
        protocol=protocol,
        evaluator_epoch=evaluator_epoch,
    )
    if value.get("plan") != {
        "plan_hash": plan.plan_hash,
        "safe_payload": plan.safe_payload(),
    }:
        raise ContractFreezeError("exact 16-unit plan changed after freeze")
    return candidate


def load_execution_freeze_v2(
    path: str | Path, *, project_root: str | Path
) -> tuple[dict[str, Any], FixedContractCandidateV2]:
    freeze_path = Path(path).expanduser()
    if freeze_path.is_symlink() or not freeze_path.is_file():
        raise ContractFreezeError("execution freeze file is unavailable")
    payload = read_json(freeze_path)
    candidate = validate_execution_freeze_v2(
        payload,
        project_root=project_root,
    )
    return payload, candidate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("execution-freeze")
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--preregistration", type=Path, required=True)
    build.add_argument("--acquisition-receipt", type=Path, required=True)
    build.add_argument("--formation-receipt", type=Path, required=True)
    build.add_argument("--measurement-view", type=Path, required=True)
    build.add_argument("--benchmark-root", type=Path, required=True)
    build.add_argument("--materialization-report", type=Path, required=True)
    build.add_argument("--prewarm", type=Path, required=True)
    build.add_argument("--paper-protocol", type=Path, required=True)
    build.add_argument("--provider-identity-sidecar", type=Path, required=True)
    build.add_argument("--provider-selection", type=Path, required=True)
    build.add_argument("--selected-canary", type=Path, required=True)
    build.add_argument("--selected-event-ledger", type=Path, required=True)
    build.add_argument(
        "--provider-label", choices=("plus", "pro"), default="plus"
    )
    build.add_argument("--provider-env-file", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-execution-freeze")
    verify.add_argument("--project-root", type=Path, required=True)
    verify.add_argument("--execution-freeze", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "execution-freeze":
        payload = build_execution_freeze_v2(
            project_root=args.project_root,
            preregistration_path=args.preregistration,
            acquisition_receipt_path=args.acquisition_receipt,
            formation_receipt_path=args.formation_receipt,
            measurement_view_path=args.measurement_view,
            benchmark_root=args.benchmark_root,
            materialization_report_path=args.materialization_report,
            prewarm_path=args.prewarm,
            paper_protocol_path=args.paper_protocol,
            provider_identity_sidecar_path=args.provider_identity_sidecar,
            provider_selection_path=args.provider_selection,
            selected_canary_path=args.selected_canary,
            selected_event_ledger_path=args.selected_event_ledger,
            provider_label=args.provider_label,
            provider_env_file=args.provider_env_file,
        )
        write_json(args.output, payload)
        print(
            json.dumps(
                {
                    "manifest_hash": payload["manifest_hash"],
                    "provider_label": payload["provider"]["provider_label"],
                    "physical_calls": payload["execution"]["physical_calls"],
                },
                sort_keys=True,
            )
        )
        return 0
    payload, _candidate = load_execution_freeze_v2(
        args.execution_freeze,
        project_root=args.project_root,
    )
    print(
        json.dumps(
            {
                "manifest_hash": payload["manifest_hash"],
                "study_id": payload["study_id"],
                "verified": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
