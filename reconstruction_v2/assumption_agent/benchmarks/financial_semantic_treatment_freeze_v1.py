from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from .codex_execution_policy import MODEL_ONLY_ACTION_BUDGET_POLICY
from .financial_semantic_operator_v1 import (
    FINANCIAL_QA_RUNTIME_ASSET_VERSION,
    FINANCIAL_SEMANTIC_ASSET_VERSION,
    FINANCIAL_SEMANTIC_OPERATOR_VERSION,
)
from .offline_verifier import (
    OFFLINE_VERIFIER_POLICY_VERSION,
    offline_verifier_profile_for_family,
    offline_verifier_runtime_key,
)
from .semantic_assignment_operator_v1 import RUNTIME_ASSET_VERSION
from .skilllearn_compiler import verify_skill_source_tree


FINANCIAL_SEMANTIC_TREATMENT_FREEZE_VERSION = (
    "financial_semantic_treatment_freeze_v1"
)
FRESH_PROVENANCE_SPLIT_VERSION = "skilllearn_fresh_provenance_split_v1"
FRESH_FINANCIAL_ITEM_ID = "financial-analysis-4"
FINANCIAL_FAMILY = "financial-analysis"
HIPPORAG_STATUS_NOT_APPLICABLE = "not_applicable_nonexecuted"
OPAQUE_SELECTION_ID_POLICY = "sha256_full_digest_v1"
MEASUREMENT_POLICY = "single_fresh_item_paired_offline_measurement_v1"
BENCHMARK_ACCESS_POLICY = "git_metadata_only_no_item_content_v1"
V320_PROTOCOL_RELATIVE_PATH = (
    "manifests/skilllearn_paper_protocol_v3_20_ruoli_gpt54mini.json"
)
PARENT_INSTANCE_MANIFEST_RELATIVE_PATH = (
    "manifests/skilllearnbench_instance_holdout_offline_ready_v1.json"
)
TASK_INPUT_PREPARATION_RECEIPT_RELATIVE_PATH = (
    "manifests/skilllearn_task_input_closure_preparation_receipt_v2.json"
)
V320_PREWARM_RECEIPT_RELATIVE_PATH = (
    "artifacts/paper_primary_v3_20_offline86_ruoli_gpt54mini_"
    "outer38_model48_portable01/development_prewarm.json"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_REQUIRED_SOURCE_ROLES = frozenset(
    {
        "operator_source",
        "integration_source",
        "prospective_runner_source",
        "lifecycle_source",
        "offline_verifier_source",
        "codex_execution_policy_source",
        "codex_action_budget_source",
        "treatment_freeze_source",
        "operator_asset",
        "minilm_runtime_asset",
        "qa_runtime_asset",
        "fresh_split_manifest",
        "formation_diagnostic_report",
        "v320_protocol",
        "parent_instance_manifest",
        "task_input_preparation_receipt",
        "v320_development_prewarm_receipt",
    }
)
_EXECUTION_ASSET_PATHS = {
    "v320_protocol": V320_PROTOCOL_RELATIVE_PATH,
    "parent_instance_manifest": PARENT_INSTANCE_MANIFEST_RELATIVE_PATH,
    "task_input_preparation_receipt": (
        TASK_INPUT_PREPARATION_RECEIPT_RELATIVE_PATH
    ),
    "v320_development_prewarm_receipt": V320_PREWARM_RECEIPT_RELATIVE_PATH,
}
_TOP_LEVEL_FIELDS = frozenset(
    {
        "manifest_version",
        "manifest_hash",
        "recipe_id",
        "program_set_hash",
        "treatment_id",
        "candidate_id",
        "candidate_manifest_hash",
        "candidate_skill_source",
        "external_skill_source_receipt_hash",
        "fresh_item_id",
        "evaluator_epoch",
        "official_hipporag",
        "hipporag_status",
        "benchmark_source",
        "project_source",
        "fresh_split_source",
        "operator_asset",
        "runtime_assets",
        "external_skill_source_receipt",
        "formation_evidence",
        "source_closure",
        "execution",
        "measurement",
    }
)


class FinancialSemanticTreatmentFreezeError(PermissionError):
    """A prospective treatment freeze failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinancialSemanticTreatmentFreezeError(
            f"{label} is not readable JSON"
        ) from exc
    if not isinstance(value, dict):
        raise FinancialSemanticTreatmentFreezeError(
            f"{label} must contain one object"
        )
    return value


def _verify_self_hash(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    declared = value.get(field)
    calculated = stable_hash(
        {key: item for key, item in value.items() if key != field}
    )
    if not _is_sha256(declared) or declared != calculated:
        raise FinancialSemanticTreatmentFreezeError(
            f"{label} self-hash mismatch"
        )
    return str(declared)


def _project_path(
    project_root: Path,
    value: str | Path,
    *,
    directory: bool,
    label: str,
) -> tuple[Path, str]:
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else project_root / raw
    if candidate.is_symlink():
        raise FinancialSemanticTreatmentFreezeError(f"{label} may not be a link")
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(project_root)
    except (FileNotFoundError, ValueError) as exc:
        raise FinancialSemanticTreatmentFreezeError(
            f"{label} is missing or escaped the project root"
        ) from exc
    if directory and not resolved.is_dir():
        raise FinancialSemanticTreatmentFreezeError(
            f"{label} must be a directory"
        )
    if not directory and not resolved.is_file():
        raise FinancialSemanticTreatmentFreezeError(f"{label} must be a file")
    return resolved, relative.as_posix()


def _git_commit(benchmark_root: Path) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(benchmark_root), "rev-parse", "--verify", "HEAD^{commit}"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinancialSemanticTreatmentFreezeError(
            "benchmark Git commit is unavailable"
        ) from exc
    commit = completed.stdout.strip().lower()
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise FinancialSemanticTreatmentFreezeError(
            "benchmark Git commit is malformed"
        )
    return commit


def _project_runtime_scope_pathspecs(project_root: Path) -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ("git", "-C", str(project_root), "rev-parse", "--show-toplevel"),
            check=True,
            capture_output=True,
            text=True,
        )
        git_root = Path(completed.stdout.strip()).resolve(strict=True)
        prefix = project_root.relative_to(git_root)
    except (OSError, subprocess.CalledProcessError, FileNotFoundError, ValueError) as exc:
        raise FinancialSemanticTreatmentFreezeError(
            "project Git worktree is unavailable"
        ) from exc
    return tuple(
        ":(top)" + (prefix / relative).as_posix()
        for relative in (
            Path("assumption_agent"),
            Path("candidates/financial_semantic_operator_v1"),
        )
    )


def _assert_project_runtime_scope_unchanged(
    project_root: Path,
    *,
    source_commit: str,
) -> None:
    pathspecs = _project_runtime_scope_pathspecs(project_root)
    try:
        subprocess.run(
            (
                "git",
                "-C",
                str(project_root),
                "cat-file",
                "-e",
                f"{source_commit}^{{commit}}",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        difference = subprocess.run(
            (
                "git",
                "-C",
                str(project_root),
                "diff",
                "--quiet",
                source_commit,
                "--",
                *pathspecs,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            (
                "git",
                "-C",
                str(project_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *pathspecs,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FinancialSemanticTreatmentFreezeError(
            "project runtime source scope could not be verified"
        ) from exc
    if difference.returncode not in {0, 1}:
        raise FinancialSemanticTreatmentFreezeError(
            "project runtime source comparison failed"
        )
    if difference.returncode != 0 or status.stdout.strip():
        raise FinancialSemanticTreatmentFreezeError(
            "project runtime source scope changed after implementation freeze"
        )


def _source_row(role: str, relative_path: str, path: Path) -> dict[str, str]:
    return {
        "role": role,
        "relative_path": relative_path,
        "file_sha256": _sha256_file(path),
    }


def _validate_operator_asset(
    asset: Mapping[str, Any],
    *,
    operator_source_sha256: str,
    minilm_manifest_hash: str,
    qa_manifest_hash: str,
) -> None:
    manifest_hash = _verify_self_hash(
        asset,
        field="manifest_hash",
        label="financial semantic operator asset",
    )
    if (
        asset.get("asset_version") != FINANCIAL_SEMANTIC_ASSET_VERSION
        or asset.get("operator_version") != FINANCIAL_SEMANTIC_OPERATOR_VERSION
        or asset.get("operator_source_sha256") != operator_source_sha256
        or asset.get("minilm_runtime_asset_manifest_hash")
        != minilm_manifest_hash
        or asset.get("qa_runtime_asset_manifest_hash") != qa_manifest_hash
        or asset.get("online_calls") != 0
        or asset.get("prospective_measurement_performed") is not False
        or asset.get("raw_instruction_logged_by_operator") is not False
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "financial semantic operator asset contract drifted"
        )
    excluded = asset.get("excluded_split_access")
    if not isinstance(excluded, Mapping) or excluded != {
        "fresh_validation_content": False,
        "prior_validation_content": False,
        "residual_sealed_content": False,
    }:
        raise FinancialSemanticTreatmentFreezeError(
            "financial semantic operator split-access claim drifted"
        )
    candidate_material = {
        "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
        "formation_source_set_hash": asset.get("formation_source_set_hash"),
        "train_example_set_hash": asset.get("train_example_set_hash"),
        "configuration_hash": asset.get("configuration_hash"),
        "minilm_runtime_asset_manifest_hash": minilm_manifest_hash,
        "qa_runtime_asset_manifest_hash": qa_manifest_hash,
        "operator_source_sha256": operator_source_sha256,
    }
    if (
        not _is_sha256(asset.get("candidate_id"))
        or asset.get("candidate_id") != stable_hash(candidate_material)
        or not _is_sha256(manifest_hash)
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "financial semantic candidate identity drifted"
        )


def _validate_runtime_asset(
    asset: Mapping[str, Any],
    *,
    expected_version: str,
    label: str,
) -> str:
    manifest_hash = _verify_self_hash(asset, field="manifest_hash", label=label)
    execution = asset.get("execution")
    if (
        asset.get("asset_version") != expected_version
        or not isinstance(execution, Mapping)
        or execution.get("local_files_only") is not True
        or execution.get("network_calls") != 0
        or execution.get("device") != "cpu"
    ):
        raise FinancialSemanticTreatmentFreezeError(
            f"{label} is not an offline CPU runtime asset"
        )
    return manifest_hash


def _validate_fresh_split(split: Mapping[str, Any]) -> str:
    manifest_hash = _verify_self_hash(
        split,
        field="manifest_hash",
        label="fresh provenance split",
    )
    formation = split.get("formation_ids")
    fresh = split.get("fresh_validation_ids")
    sealed = split.get("residual_sealed_ids")
    historical = split.get("historical_trial_contaminated_ids")
    local = split.get("local_content_contaminated_ids")
    groups = (formation, fresh, sealed, historical, local)
    if (
        split.get("manifest_version") != FRESH_PROVENANCE_SPLIT_VERSION
        or split.get("sealed_test") is not True
        or split.get("fresh_validation_content_accessed") is not False
        or split.get("residual_sealed_content_accessed") is not False
        or split.get("raw_content_persisted") is not False
        or any(
            not isinstance(group, list)
            or group != sorted(set(str(item) for item in group))
            for group in groups
        )
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "fresh provenance split contract drifted"
        )
    assert all(isinstance(group, list) for group in groups)
    seen: set[str] = set()
    for group in (formation, fresh, sealed):
        current = set(str(item) for item in group)
        if seen & current:
            raise FinancialSemanticTreatmentFreezeError(
                "fresh provenance split partitions overlap"
            )
        seen.update(current)
    if not set(str(item) for item in historical).issubset(set(formation)) or not set(
        str(item) for item in local
    ).issubset(set(formation)):
        raise FinancialSemanticTreatmentFreezeError(
            "contamination annotations are not subsets of formation"
        )
    if FRESH_FINANCIAL_ITEM_ID not in fresh:
        raise FinancialSemanticTreatmentFreezeError(
            "fresh financial item is absent from the frozen split"
        )
    family_by_id = split.get("family_by_id")
    if (
        not isinstance(family_by_id, Mapping)
        or family_by_id.get(FRESH_FINANCIAL_ITEM_ID) != FINANCIAL_FAMILY
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "fresh financial item family binding drifted"
        )
    counts = split.get("counts")
    if not isinstance(counts, Mapping) or any(
        counts.get(key) != expected
        for key, expected in (
            ("formation", len(formation)),
            ("fresh_validation", len(fresh)),
            ("residual_sealed", len(sealed)),
            ("historical_trial_contaminated", len(historical)),
            ("local_content_contaminated", len(local)),
            ("all", len(family_by_id)),
        )
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "fresh provenance split counts drifted"
        )
    if set(str(item) for item in family_by_id) != seen:
        raise FinancialSemanticTreatmentFreezeError(
            "fresh provenance split does not partition the declared universe"
        )
    return manifest_hash


def _skill_receipt_issues(receipt: object) -> list[str]:
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "source_file_hashes",
        "source_tree_hash",
    }:
        return ["external skill source receipt fields drifted"]
    rows = receipt.get("source_file_hashes")
    if not isinstance(rows, list) or not rows:
        return ["external skill source receipt is empty"]
    normalized: list[dict[str, str]] = []
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "sha256"}
            or not isinstance(row.get("path"), str)
            or not row.get("path")
            or Path(str(row["path"])).is_absolute()
            or ".." in Path(str(row["path"])).parts
            or not _is_sha256(row.get("sha256"))
        ):
            return ["external skill source receipt row is malformed"]
        normalized.append(
            {"path": str(row["path"]), "sha256": str(row["sha256"])}
        )
    if rows != sorted(normalized, key=lambda row: row["path"]):
        return ["external skill source receipt order drifted"]
    if receipt.get("source_tree_hash") != stable_hash({"files": rows}):
        return ["external skill source tree hash drifted"]
    return []


def _recipe_id(payload: Mapping[str, Any]) -> str:
    rows = {
        str(row["role"]): str(row["file_sha256"])
        for row in payload["source_closure"]["files"]
    }
    return stable_hash(
        {
            "candidate_id": payload["candidate_id"],
            "candidate_manifest_hash": payload["candidate_manifest_hash"],
            "operator_source_sha256": rows["operator_source"],
            "integration_source_sha256": rows["integration_source"],
            "operator_asset_sha256": rows["operator_asset"],
            "minilm_runtime_asset_manifest_hash": payload["runtime_assets"][
                "minilm"
            ]["manifest_hash"],
            "qa_runtime_asset_manifest_hash": payload["runtime_assets"]["qa"][
                "manifest_hash"
            ],
            "external_skill_source_receipt_hash": payload[
                "external_skill_source_receipt_hash"
            ],
            "opaque_id_policy": OPAQUE_SELECTION_ID_POLICY,
        }
    )


def _treatment_id(payload: Mapping[str, Any]) -> str:
    file_rows = {
        str(row["role"]): row
        for row in payload["source_closure"]["files"]
    }
    rows = {
        role: str(row["file_sha256"])
        for role, row in file_rows.items()
    }
    return stable_hash(
        {
            "recipe_id": payload["recipe_id"],
            "program_set_hash": payload["program_set_hash"],
            "fresh_item_id": payload["fresh_item_id"],
            "fresh_split_manifest_hash": payload["fresh_split_source"][
                "manifest_hash"
            ],
            "benchmark_git_commit": payload["benchmark_source"]["git_commit"],
            "prospective_runner_source_sha256": rows[
                "prospective_runner_source"
            ],
            "lifecycle_source_sha256": rows["lifecycle_source"],
            "offline_verifier_source_sha256": rows[
                "offline_verifier_source"
            ],
            "formation_diagnostic_report_hash": payload[
                "formation_evidence"
            ]["report_hash"],
            "codex_execution_policy_source_sha256": rows[
                "codex_execution_policy_source"
            ],
            "codex_action_budget_source_sha256": rows[
                "codex_action_budget_source"
            ],
            "treatment_freeze_source_sha256": rows[
                "treatment_freeze_source"
            ],
            "project_source_commit": payload["project_source"]["git_commit"],
            "execution_asset_closure": {
                role: {
                    "relative_path": file_rows[role]["relative_path"],
                    "file_sha256": rows[role],
                }
                for role in sorted(_EXECUTION_ASSET_PATHS)
            },
            "execution_hash": stable_hash(payload["execution"]),
            "measurement_policy": MEASUREMENT_POLICY,
            "official_hipporag": False,
            "hipporag_status": HIPPORAG_STATUS_NOT_APPLICABLE,
            "opaque_id_policy": OPAQUE_SELECTION_ID_POLICY,
        }
    )


def validate_financial_semantic_treatment_freeze_v1(
    payload: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
    benchmark_root: str | Path | None = None,
) -> None:
    """Strictly validate identity, provenance and optional live file closure."""

    if set(payload) != _TOP_LEVEL_FIELDS:
        raise FinancialSemanticTreatmentFreezeError(
            "treatment freeze top-level fields drifted"
        )
    if payload.get("manifest_version") != FINANCIAL_SEMANTIC_TREATMENT_FREEZE_VERSION:
        raise FinancialSemanticTreatmentFreezeError(
            "treatment freeze version drifted"
        )
    if (
        payload.get("fresh_item_id") != FRESH_FINANCIAL_ITEM_ID
        or payload.get("official_hipporag") is not False
        or payload.get("hipporag_status") != HIPPORAG_STATUS_NOT_APPLICABLE
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "treatment activation or HippoRAG status drifted"
        )

    benchmark = payload.get("benchmark_source")
    if not isinstance(benchmark, Mapping) or set(benchmark) != {
        "git_commit",
        "access_policy",
    }:
        raise FinancialSemanticTreatmentFreezeError(
            "benchmark source binding is malformed"
        )
    if (
        _GIT_COMMIT.fullmatch(str(benchmark.get("git_commit") or "")) is None
        or benchmark.get("access_policy") != BENCHMARK_ACCESS_POLICY
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "benchmark source binding drifted"
        )

    project_source = payload.get("project_source")
    if (
        not isinstance(project_source, Mapping)
        or set(project_source)
        != {"git_commit", "runtime_scope_paths", "scope_policy"}
        or project_source.get("runtime_scope_paths")
        != [
            "assumption_agent",
            "candidates/financial_semantic_operator_v1",
        ]
        or project_source.get("scope_policy")
        != "bound_commit_diff_and_no_untracked_v1"
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "project source binding is malformed"
        )
    if _GIT_COMMIT.fullmatch(str(project_source.get("git_commit") or "")) is None:
        raise FinancialSemanticTreatmentFreezeError(
            "project source commit is malformed"
        )

    split = payload.get("fresh_split_source")
    if not isinstance(split, Mapping) or set(split) != {
        "relative_path",
        "file_sha256",
        "manifest_version",
        "manifest_hash",
        "formation_count",
        "fresh_validation_count",
        "residual_sealed_count",
        "fresh_validation_set_hash",
        "residual_sealed_set_hash",
    }:
        raise FinancialSemanticTreatmentFreezeError(
            "fresh split source binding is malformed"
        )
    if (
        split.get("manifest_version") != FRESH_PROVENANCE_SPLIT_VERSION
        or not all(
            _is_sha256(split.get(field))
            for field in (
                "file_sha256",
                "manifest_hash",
                "fresh_validation_set_hash",
                "residual_sealed_set_hash",
            )
        )
        or any(
            isinstance(split.get(field), bool)
            or not isinstance(split.get(field), int)
            or split.get(field) <= 0
            for field in (
                "formation_count",
                "fresh_validation_count",
                "residual_sealed_count",
            )
        )
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "fresh split source binding drifted"
        )

    operator = payload.get("operator_asset")
    if not isinstance(operator, Mapping) or set(operator) != {
        "relative_path",
        "file_sha256",
        "asset_version",
        "operator_version",
        "manifest_hash",
        "candidate_id",
    }:
        raise FinancialSemanticTreatmentFreezeError(
            "operator asset binding is malformed"
        )
    if (
        operator.get("asset_version") != FINANCIAL_SEMANTIC_ASSET_VERSION
        or operator.get("operator_version") != FINANCIAL_SEMANTIC_OPERATOR_VERSION
        or not all(
            _is_sha256(operator.get(field))
            for field in ("file_sha256", "manifest_hash", "candidate_id")
        )
        or operator.get("candidate_id") != payload.get("candidate_id")
        or operator.get("manifest_hash") != payload.get("candidate_manifest_hash")
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "operator asset identity binding drifted"
        )

    formation = payload.get("formation_evidence")
    if not isinstance(formation, Mapping) or set(formation) != {
        "relative_path",
        "file_sha256",
        "report_version",
        "report_hash",
        "evidence_scope",
        "cross_fit",
        "prospective_claim_authorized",
        "performance_gate_bound",
    }:
        raise FinancialSemanticTreatmentFreezeError(
            "formation evidence binding is malformed"
        )
    if (
        not _is_sha256(formation.get("file_sha256"))
        or not _is_sha256(formation.get("report_hash"))
        or formation.get("evidence_scope") != "in_sample_formation_replay"
        or formation.get("cross_fit") is not False
        or formation.get("prospective_claim_authorized") is not False
        or formation.get("performance_gate_bound") is not False
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "formation evidence is not non-gating in-sample replay"
        )

    runtime_assets = payload.get("runtime_assets")
    if not isinstance(runtime_assets, Mapping) or set(runtime_assets) != {
        "minilm",
        "qa",
    }:
        raise FinancialSemanticTreatmentFreezeError(
            "runtime asset bindings are malformed"
        )
    for key, expected_version in (
        ("minilm", RUNTIME_ASSET_VERSION),
        ("qa", FINANCIAL_QA_RUNTIME_ASSET_VERSION),
    ):
        row = runtime_assets.get(key)
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "relative_path",
                "file_sha256",
                "asset_version",
                "manifest_hash",
            }
            or row.get("asset_version") != expected_version
            or not _is_sha256(row.get("file_sha256"))
            or not _is_sha256(row.get("manifest_hash"))
        ):
            raise FinancialSemanticTreatmentFreezeError(
                f"{key} runtime asset binding drifted"
            )

    receipt = payload.get("external_skill_source_receipt")
    receipt_issues = _skill_receipt_issues(receipt)
    if receipt_issues:
        raise FinancialSemanticTreatmentFreezeError(receipt_issues[0])
    assert isinstance(receipt, Mapping)
    if (
        not isinstance(payload.get("candidate_skill_source"), str)
        or Path(str(payload["candidate_skill_source"])).is_absolute()
        or ".." in Path(str(payload["candidate_skill_source"])).parts
        or payload.get("external_skill_source_receipt_hash")
        != stable_hash(dict(receipt))
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "external skill source binding drifted"
        )

    closure = payload.get("source_closure")
    if not isinstance(closure, Mapping) or set(closure) != {
        "files",
        "file_count",
        "file_set_hash",
    }:
        raise FinancialSemanticTreatmentFreezeError(
            "source closure is malformed"
        )
    files = closure.get("files")
    if not isinstance(files, list) or not files:
        raise FinancialSemanticTreatmentFreezeError("source closure is empty")
    normalized_files: list[dict[str, str]] = []
    for row in files:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"role", "relative_path", "file_sha256"}
            or not isinstance(row.get("role"), str)
            or not isinstance(row.get("relative_path"), str)
            or Path(str(row.get("relative_path") or "")).is_absolute()
            or ".." in Path(str(row.get("relative_path") or "")).parts
            or not _is_sha256(row.get("file_sha256"))
        ):
            raise FinancialSemanticTreatmentFreezeError(
                "source closure row is malformed"
            )
        normalized_files.append(
            {
                "role": str(row["role"]),
                "relative_path": str(row["relative_path"]),
                "file_sha256": str(row["file_sha256"]),
            }
        )
    if (
        files != sorted(normalized_files, key=lambda row: row["role"])
        or {row["role"] for row in files} != _REQUIRED_SOURCE_ROLES
        or closure.get("file_count") != len(files)
        or closure.get("file_set_hash") != stable_hash(files)
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "source closure set drifted"
        )
    rows_by_role = {row["role"]: row for row in files}
    for role, expected_path in _EXECUTION_ASSET_PATHS.items():
        if rows_by_role[role]["relative_path"] != expected_path:
            raise FinancialSemanticTreatmentFreezeError(
                f"execution asset {role} path drifted"
            )
    for role, binding in (
        ("operator_asset", operator),
        ("minilm_runtime_asset", runtime_assets["minilm"]),
        ("qa_runtime_asset", runtime_assets["qa"]),
        ("fresh_split_manifest", split),
        ("formation_diagnostic_report", formation),
    ):
        if (
            rows_by_role[role]["relative_path"] != binding["relative_path"]
            or rows_by_role[role]["file_sha256"] != binding["file_sha256"]
        ):
            raise FinancialSemanticTreatmentFreezeError(
                f"source closure {role} cross-binding drifted"
            )

    profile = offline_verifier_profile_for_family(FINANCIAL_FAMILY)
    if profile is None:
        raise FinancialSemanticTreatmentFreezeError(
            "financial offline verifier profile is unavailable"
        )
    expected_execution = {
        "codex_agent_execution_policy": (
            MODEL_ONLY_ACTION_BUDGET_POLICY.to_dict()
        ),
        "codex_agent_execution_policy_hash": (
            MODEL_ONLY_ACTION_BUDGET_POLICY.policy_hash
        ),
        "max_steps": payload.get("execution", {}).get("max_steps")
        if isinstance(payload.get("execution"), Mapping)
        else None,
        "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
        "offline_verifier_profile_id": profile.profile_id,
        "offline_verifier_profile_hash": profile.profile_hash,
        "offline_verifier_runtime_key": offline_verifier_runtime_key(
            profile=profile
        ),
        "offline_verifier_command_hash": stable_hash(
            {"command": profile.verifier_command}
        ),
        "verifier_runtime_network": "none",
        "online_evaluation_allowed": False,
    }
    execution = payload.get("execution")
    if (
        not isinstance(execution, Mapping)
        or dict(execution) != expected_execution
        or isinstance(execution.get("max_steps"), bool)
        or not isinstance(execution.get("max_steps"), int)
        or execution.get("max_steps") <= 0
        or not MODEL_ONLY_ACTION_BUDGET_POLICY.action_budget_enforced
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "execution or offline verifier policy drifted"
        )

    if payload.get("measurement") != {
        "policy": MEASUREMENT_POLICY,
        "prospective_measurement_performed": False,
        "performance_gate_bound": False,
        "performance_thresholds_bound": False,
        "raw_content_persisted": False,
    }:
        raise FinancialSemanticTreatmentFreezeError(
            "measurement-only declaration drifted"
        )
    if payload.get("evaluator_epoch") != (
        "financial-semantic-fresh-" + str(split["manifest_hash"])[:12]
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "evaluator epoch drifted"
        )
    expected_recipe = _recipe_id(payload)
    if payload.get("recipe_id") != expected_recipe:
        raise FinancialSemanticTreatmentFreezeError("opaque recipe ID drifted")
    if payload.get("program_set_hash") != stable_hash(
        {"recipe_ids": [expected_recipe]}
    ):
        raise FinancialSemanticTreatmentFreezeError("program set hash drifted")
    if payload.get("treatment_id") != _treatment_id(payload):
        raise FinancialSemanticTreatmentFreezeError("opaque treatment ID drifted")
    _verify_self_hash(payload, field="manifest_hash", label="treatment freeze")

    if project_root is not None:
        project = Path(project_root).expanduser().resolve(strict=True)
        _assert_project_runtime_scope_unchanged(
            project,
            source_commit=str(project_source["git_commit"]),
        )
        for row in files:
            path, relative = _project_path(
                project,
                str(row["relative_path"]),
                directory=False,
                label=f"source closure role {row['role']}",
            )
            if relative != row["relative_path"] or _sha256_file(path) != row[
                "file_sha256"
            ]:
                raise FinancialSemanticTreatmentFreezeError(
                    f"source closure role {row['role']} changed after freeze"
                )
        skill_root, skill_relative = _project_path(
            project,
            str(payload["candidate_skill_source"]),
            directory=True,
            label="candidate skill source",
        )
        current_skill_receipt = verify_skill_source_tree(skill_root)
        if (
            skill_relative != payload["candidate_skill_source"]
            or current_skill_receipt.to_dict() != receipt
            or current_skill_receipt.receipt_hash
            != payload["external_skill_source_receipt_hash"]
        ):
            raise FinancialSemanticTreatmentFreezeError(
                "candidate skill source changed after freeze"
            )

        operator_payload = _read_json_object(
            project / str(operator["relative_path"]), "operator asset"
        )
        minilm_payload = _read_json_object(
            project / str(runtime_assets["minilm"]["relative_path"]),
            "MiniLM runtime asset",
        )
        qa_payload = _read_json_object(
            project / str(runtime_assets["qa"]["relative_path"]),
            "QA runtime asset",
        )
        split_payload = _read_json_object(
            project / str(split["relative_path"]), "fresh provenance split"
        )
        formation_payload = _read_json_object(
            project / str(formation["relative_path"]),
            "formation diagnostic report",
        )
        minilm_hash = _validate_runtime_asset(
            minilm_payload,
            expected_version=RUNTIME_ASSET_VERSION,
            label="MiniLM runtime asset",
        )
        qa_hash = _validate_runtime_asset(
            qa_payload,
            expected_version=FINANCIAL_QA_RUNTIME_ASSET_VERSION,
            label="QA runtime asset",
        )
        _validate_operator_asset(
            operator_payload,
            operator_source_sha256=rows_by_role["operator_source"][
                "file_sha256"
            ],
            minilm_manifest_hash=minilm_hash,
            qa_manifest_hash=qa_hash,
        )
        split_hash = _validate_fresh_split(split_payload)
        formation_report_hash = _verify_self_hash(
            formation_payload,
            field="report_hash",
            label="formation diagnostic report",
        )
        if (
            formation_payload.get("report_version")
            != formation.get("report_version")
            or formation_report_hash != formation.get("report_hash")
            or formation_payload.get("candidate_id") != payload["candidate_id"]
            or formation_payload.get("candidate_manifest_hash")
            != payload["candidate_manifest_hash"]
            or formation_payload.get("operator_source_sha256")
            != rows_by_role["operator_source"]["file_sha256"]
            or formation_payload.get("cross_fit") is not False
            or formation_payload.get("in_sample_formation_replay") is not True
            or formation_payload.get("retrospective_formation_replay_gain")
            is not True
            or formation_payload.get("prospective_claim_authorized") is not False
            or formation_payload.get("causal_gain_claim_authorized") is not False
            or formation_payload.get("online_calls") != 0
            or formation_payload.get("online_judge_calls") != 0
            or formation_payload.get("offline_verifier_only") is not True
            or formation_payload.get("validation_content_accessed") is not False
            or formation_payload.get("sealed_content_accessed") is not False
        ):
            raise FinancialSemanticTreatmentFreezeError(
                "formation diagnostic report claim boundary drifted"
            )
        if (
            operator_payload["manifest_hash"] != operator["manifest_hash"]
            or operator_payload["candidate_id"] != operator["candidate_id"]
            or minilm_hash != runtime_assets["minilm"]["manifest_hash"]
            or qa_hash != runtime_assets["qa"]["manifest_hash"]
            or split_hash != split["manifest_hash"]
            or split["formation_count"] != len(split_payload["formation_ids"])
            or split["fresh_validation_count"]
            != len(split_payload["fresh_validation_ids"])
            or split["residual_sealed_count"]
            != len(split_payload["residual_sealed_ids"])
            or split["fresh_validation_set_hash"]
            != stable_hash(split_payload["fresh_validation_ids"])
            or split["residual_sealed_set_hash"]
            != stable_hash(split_payload["residual_sealed_ids"])
        ):
            raise FinancialSemanticTreatmentFreezeError(
                "live asset or split identity drifted"
            )

    if benchmark_root is not None:
        benchmark_path = Path(benchmark_root).expanduser().resolve(strict=True)
        if _git_commit(benchmark_path) != benchmark["git_commit"]:
            raise FinancialSemanticTreatmentFreezeError(
                "benchmark Git commit changed after freeze"
            )


def build_financial_semantic_treatment_freeze_v1(
    *,
    project_root: str | Path,
    benchmark_root: str | Path,
    operator_asset_path: str | Path,
    minilm_runtime_asset_path: str | Path,
    qa_runtime_asset_path: str | Path,
    fresh_split_manifest_path: str | Path,
    formation_diagnostic_report_path: str | Path,
    candidate_skill_source: str | Path,
    operator_source_path: str | Path,
    integration_source_path: str | Path,
    prospective_runner_source_path: str | Path,
    lifecycle_source_path: str | Path,
    offline_verifier_source_path: str | Path,
    codex_execution_policy_source_path: str | Path,
    codex_action_budget_source_path: str | Path,
    treatment_freeze_source_path: str | Path,
    v320_protocol_path: str | Path,
    parent_instance_manifest_path: str | Path,
    task_input_preparation_receipt_path: str | Path,
    v320_development_prewarm_receipt_path: str | Path,
    max_steps: int = 100,
) -> dict[str, Any]:
    """Build a no-outcome prospective treatment identity.

    This function reads only project assets/source, split metadata and the
    benchmark Git commit.  It never discovers or opens benchmark item content.
    """

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise FinancialSemanticTreatmentFreezeError(
            "max_steps must be a positive integer"
        )
    project = Path(project_root).expanduser().resolve(strict=True)
    benchmark = Path(benchmark_root).expanduser().resolve(strict=True)
    project_source_commit = _git_commit(project)
    _assert_project_runtime_scope_unchanged(
        project,
        source_commit=project_source_commit,
    )

    file_inputs = {
        "operator_asset": operator_asset_path,
        "minilm_runtime_asset": minilm_runtime_asset_path,
        "qa_runtime_asset": qa_runtime_asset_path,
        "fresh_split_manifest": fresh_split_manifest_path,
        "formation_diagnostic_report": formation_diagnostic_report_path,
        "operator_source": operator_source_path,
        "integration_source": integration_source_path,
        "prospective_runner_source": prospective_runner_source_path,
        "lifecycle_source": lifecycle_source_path,
        "offline_verifier_source": offline_verifier_source_path,
        "codex_execution_policy_source": codex_execution_policy_source_path,
        "codex_action_budget_source": codex_action_budget_source_path,
        "treatment_freeze_source": treatment_freeze_source_path,
        "v320_protocol": v320_protocol_path,
        "parent_instance_manifest": parent_instance_manifest_path,
        "task_input_preparation_receipt": task_input_preparation_receipt_path,
        "v320_development_prewarm_receipt": (
            v320_development_prewarm_receipt_path
        ),
    }
    resolved_files: dict[str, tuple[Path, str]] = {
        role: _project_path(
            project,
            path,
            directory=False,
            label=role,
        )
        for role, path in file_inputs.items()
    }
    skill_root, skill_relative = _project_path(
        project,
        candidate_skill_source,
        directory=True,
        label="candidate skill source",
    )

    operator_asset = _read_json_object(
        resolved_files["operator_asset"][0], "operator asset"
    )
    minilm_asset = _read_json_object(
        resolved_files["minilm_runtime_asset"][0], "MiniLM runtime asset"
    )
    qa_asset = _read_json_object(
        resolved_files["qa_runtime_asset"][0], "QA runtime asset"
    )
    split_payload = _read_json_object(
        resolved_files["fresh_split_manifest"][0], "fresh provenance split"
    )
    formation_payload = _read_json_object(
        resolved_files["formation_diagnostic_report"][0],
        "formation diagnostic report",
    )
    minilm_hash = _validate_runtime_asset(
        minilm_asset,
        expected_version=RUNTIME_ASSET_VERSION,
        label="MiniLM runtime asset",
    )
    qa_hash = _validate_runtime_asset(
        qa_asset,
        expected_version=FINANCIAL_QA_RUNTIME_ASSET_VERSION,
        label="QA runtime asset",
    )
    _validate_operator_asset(
        operator_asset,
        operator_source_sha256=_sha256_file(
            resolved_files["operator_source"][0]
        ),
        minilm_manifest_hash=minilm_hash,
        qa_manifest_hash=qa_hash,
    )
    split_hash = _validate_fresh_split(split_payload)
    formation_report_hash = _verify_self_hash(
        formation_payload,
        field="report_hash",
        label="formation diagnostic report",
    )
    if (
        formation_payload.get("candidate_id") != operator_asset["candidate_id"]
        or formation_payload.get("candidate_manifest_hash")
        != operator_asset["manifest_hash"]
        or formation_payload.get("operator_source_sha256")
        != _sha256_file(resolved_files["operator_source"][0])
        or formation_payload.get("cross_fit") is not False
        or formation_payload.get("in_sample_formation_replay") is not True
        or formation_payload.get("retrospective_formation_replay_gain") is not True
        or formation_payload.get("prospective_claim_authorized") is not False
        or formation_payload.get("causal_gain_claim_authorized") is not False
        or formation_payload.get("online_calls") != 0
        or formation_payload.get("online_judge_calls") != 0
        or formation_payload.get("offline_verifier_only") is not True
        or formation_payload.get("validation_content_accessed") is not False
        or formation_payload.get("sealed_content_accessed") is not False
    ):
        raise FinancialSemanticTreatmentFreezeError(
            "formation diagnostic report is not eligible as non-gating replay evidence"
        )

    source_rows = sorted(
        (
            _source_row(role, relative, path)
            for role, (path, relative) in resolved_files.items()
        ),
        key=lambda row: row["role"],
    )
    skill_receipt = verify_skill_source_tree(skill_root)
    profile = offline_verifier_profile_for_family(FINANCIAL_FAMILY)
    if profile is None:
        raise FinancialSemanticTreatmentFreezeError(
            "financial offline verifier profile is unavailable"
        )
    split_relative = resolved_files["fresh_split_manifest"][1]
    operator_relative = resolved_files["operator_asset"][1]
    minilm_relative = resolved_files["minilm_runtime_asset"][1]
    qa_relative = resolved_files["qa_runtime_asset"][1]
    payload: dict[str, Any] = {
        "manifest_version": FINANCIAL_SEMANTIC_TREATMENT_FREEZE_VERSION,
        "recipe_id": "",
        "program_set_hash": "",
        "treatment_id": "",
        "candidate_id": operator_asset["candidate_id"],
        "candidate_manifest_hash": operator_asset["manifest_hash"],
        "candidate_skill_source": skill_relative,
        "external_skill_source_receipt_hash": skill_receipt.receipt_hash,
        "fresh_item_id": FRESH_FINANCIAL_ITEM_ID,
        "evaluator_epoch": "financial-semantic-fresh-" + split_hash[:12],
        "official_hipporag": False,
        "hipporag_status": HIPPORAG_STATUS_NOT_APPLICABLE,
        "benchmark_source": {
            "git_commit": _git_commit(benchmark),
            "access_policy": BENCHMARK_ACCESS_POLICY,
        },
        "project_source": {
            "git_commit": project_source_commit,
            "runtime_scope_paths": [
                "assumption_agent",
                "candidates/financial_semantic_operator_v1",
            ],
            "scope_policy": "bound_commit_diff_and_no_untracked_v1",
        },
        "fresh_split_source": {
            "relative_path": split_relative,
            "file_sha256": _sha256_file(
                resolved_files["fresh_split_manifest"][0]
            ),
            "manifest_version": FRESH_PROVENANCE_SPLIT_VERSION,
            "manifest_hash": split_hash,
            "formation_count": len(split_payload["formation_ids"]),
            "fresh_validation_count": len(
                split_payload["fresh_validation_ids"]
            ),
            "residual_sealed_count": len(split_payload["residual_sealed_ids"]),
            "fresh_validation_set_hash": stable_hash(
                split_payload["fresh_validation_ids"]
            ),
            "residual_sealed_set_hash": stable_hash(
                split_payload["residual_sealed_ids"]
            ),
        },
        "operator_asset": {
            "relative_path": operator_relative,
            "file_sha256": _sha256_file(resolved_files["operator_asset"][0]),
            "asset_version": FINANCIAL_SEMANTIC_ASSET_VERSION,
            "operator_version": FINANCIAL_SEMANTIC_OPERATOR_VERSION,
            "manifest_hash": operator_asset["manifest_hash"],
            "candidate_id": operator_asset["candidate_id"],
        },
        "runtime_assets": {
            "minilm": {
                "relative_path": minilm_relative,
                "file_sha256": _sha256_file(
                    resolved_files["minilm_runtime_asset"][0]
                ),
                "asset_version": RUNTIME_ASSET_VERSION,
                "manifest_hash": minilm_hash,
            },
            "qa": {
                "relative_path": qa_relative,
                "file_sha256": _sha256_file(
                    resolved_files["qa_runtime_asset"][0]
                ),
                "asset_version": FINANCIAL_QA_RUNTIME_ASSET_VERSION,
                "manifest_hash": qa_hash,
            },
        },
        "external_skill_source_receipt": skill_receipt.to_dict(),
        "formation_evidence": {
            "relative_path": resolved_files["formation_diagnostic_report"][1],
            "file_sha256": _sha256_file(
                resolved_files["formation_diagnostic_report"][0]
            ),
            "report_version": formation_payload.get("report_version"),
            "report_hash": formation_report_hash,
            "evidence_scope": "in_sample_formation_replay",
            "cross_fit": False,
            "prospective_claim_authorized": False,
            "performance_gate_bound": False,
        },
        "source_closure": {
            "files": source_rows,
            "file_count": len(source_rows),
            "file_set_hash": stable_hash(source_rows),
        },
        "execution": {
            "codex_agent_execution_policy": (
                MODEL_ONLY_ACTION_BUDGET_POLICY.to_dict()
            ),
            "codex_agent_execution_policy_hash": (
                MODEL_ONLY_ACTION_BUDGET_POLICY.policy_hash
            ),
            "max_steps": max_steps,
            "offline_verifier_policy": OFFLINE_VERIFIER_POLICY_VERSION,
            "offline_verifier_profile_id": profile.profile_id,
            "offline_verifier_profile_hash": profile.profile_hash,
            "offline_verifier_runtime_key": offline_verifier_runtime_key(
                profile=profile
            ),
            "offline_verifier_command_hash": stable_hash(
                {"command": profile.verifier_command}
            ),
            "verifier_runtime_network": "none",
            "online_evaluation_allowed": False,
        },
        "measurement": {
            "policy": MEASUREMENT_POLICY,
            "prospective_measurement_performed": False,
            "performance_gate_bound": False,
            "performance_thresholds_bound": False,
            "raw_content_persisted": False,
        },
    }
    payload["recipe_id"] = _recipe_id(payload)
    payload["program_set_hash"] = stable_hash(
        {"recipe_ids": [payload["recipe_id"]]}
    )
    payload["treatment_id"] = _treatment_id(payload)
    payload["manifest_hash"] = stable_hash(payload)
    validate_financial_semantic_treatment_freeze_v1(
        payload,
        project_root=project,
        benchmark_root=benchmark,
    )
    return payload


def load_financial_semantic_treatment_freeze_v1(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    benchmark_root: str | Path | None = None,
) -> dict[str, Any]:
    payload = _read_json_object(Path(path), "financial semantic treatment freeze")
    validate_financial_semantic_treatment_freeze_v1(
        payload,
        project_root=project_root,
        benchmark_root=benchmark_root,
    )
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/validate the financial semantic treatment freeze v1"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    for name in (
        "project-root",
        "benchmark-root",
        "operator-asset",
        "minilm-runtime-asset",
        "qa-runtime-asset",
        "fresh-split-manifest",
        "formation-diagnostic-report",
        "candidate-skill-source",
        "operator-source",
        "integration-source",
        "prospective-runner-source",
        "lifecycle-source",
        "offline-verifier-source",
        "codex-execution-policy-source",
        "codex-action-budget-source",
        "treatment-freeze-source",
        "v320-protocol",
        "parent-instance-manifest",
        "task-input-preparation-receipt",
        "v320-development-prewarm-receipt",
        "output",
    ):
        build.add_argument(f"--{name}", required=True)
    build.add_argument("--max-steps", type=int, default=100)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--project-root")
    validate.add_argument("--benchmark-root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        payload = load_financial_semantic_treatment_freeze_v1(
            args.manifest,
            project_root=args.project_root,
            benchmark_root=args.benchmark_root,
        )
        print(
            json.dumps(
                {
                    "manifest_hash": payload["manifest_hash"],
                    "recipe_id": payload["recipe_id"],
                    "treatment_id": payload["treatment_id"],
                    "valid": True,
                },
                sort_keys=True,
            )
        )
        return 0
    payload = build_financial_semantic_treatment_freeze_v1(
        project_root=args.project_root,
        benchmark_root=args.benchmark_root,
        operator_asset_path=args.operator_asset,
        minilm_runtime_asset_path=args.minilm_runtime_asset,
        qa_runtime_asset_path=args.qa_runtime_asset,
        fresh_split_manifest_path=args.fresh_split_manifest,
        formation_diagnostic_report_path=args.formation_diagnostic_report,
        candidate_skill_source=args.candidate_skill_source,
        operator_source_path=args.operator_source,
        integration_source_path=args.integration_source,
        prospective_runner_source_path=args.prospective_runner_source,
        lifecycle_source_path=args.lifecycle_source,
        offline_verifier_source_path=args.offline_verifier_source,
        codex_execution_policy_source_path=args.codex_execution_policy_source,
        codex_action_budget_source_path=args.codex_action_budget_source,
        treatment_freeze_source_path=args.treatment_freeze_source,
        v320_protocol_path=args.v320_protocol,
        parent_instance_manifest_path=args.parent_instance_manifest,
        task_input_preparation_receipt_path=(
            args.task_input_preparation_receipt
        ),
        v320_development_prewarm_receipt_path=(
            args.v320_development_prewarm_receipt
        ),
        max_steps=args.max_steps,
    )
    _write_json(Path(args.output), payload)
    print(
        json.dumps(
            {
                "manifest_hash": payload["manifest_hash"],
                "recipe_id": payload["recipe_id"],
                "treatment_id": payload["treatment_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
